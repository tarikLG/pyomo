# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy import MindtPy
from pyomo.contrib.mindtpy import algorithm_base_class
from pyomo.contrib.mindtpy import cut_generation
from pyomo.contrib.mindtpy import extended_cutting_plane
from pyomo.contrib.mindtpy import global_outer_approximation
from pyomo.contrib.mindtpy import outer_approximation
from pyomo.contrib.mindtpy import util
from pyomo.contrib.mindtpy.tests._helpers import FakeCallbackSolverModel
from pyomo.contrib.mindtpy.tests._helpers import FakePersistentBase
from pyomo.contrib.mindtpy.tests._helpers import FakePersistentSolver
from pyomo.contrib.mindtpy.tests._helpers import FakeSolver
from pyomo.contrib.mindtpy.tests._helpers import make_algorithm
from pyomo.contrib.mindtpy.tests._helpers import make_config
from pyomo.contrib.mindtpy.tests._helpers import make_core_model
from pyomo.contrib.mindtpy.tests._helpers import make_cut_model
from pyomo.contrib.mindtpy.tests._helpers import make_goa_config
from pyomo.contrib.mindtpy.tests._helpers import make_logger
from pyomo.contrib.mindtpy.tests._helpers import make_results
from pyomo.core import Binary
from pyomo.core import Block
from pyomo.core import ConcreteModel
from pyomo.core import Constraint
from pyomo.core import ConstraintList
from pyomo.core import Objective
from pyomo.core import Reals
from pyomo.core import Var
from pyomo.core import VarList
from pyomo.core import maximize
from pyomo.core import minimize
from pyomo.core import value
from pyomo.opt import SolverResults
from pyomo.opt import SolverStatus
from pyomo.opt import TerminationCondition as tc
from pyomo.opt.results.solution import SolutionStatus


class TestMindtPyTopLevel(unittest.TestCase):
    def test_mindtpy_solver_dispatches_to_selected_strategy(self):
        solver = MindtPy.MindtPySolver()
        model = make_core_model()
        dispatched = FakeSolver()
        created_solver_names = []

        def fake_solver_factory(name):
            created_solver_names.append(name)
            return dispatched

        with patch.object(MindtPy, 'SolverFactory', side_effect=fake_solver_factory):
            solver.solve(
                model,
                strategy='OA',
                mip_solver='glpk',
                nlp_solver='ipopt',
                options={'iteration_limit': 3},
            )

        self.assertEqual(created_solver_names, ['mindtpy.oa'])
        self.assertEqual(len(dispatched.solve_calls), 1)
        _, forwarded_kwargs = dispatched.solve_calls[0]
        self.assertNotIn('options', forwarded_kwargs)
        self.assertEqual(forwarded_kwargs['mip_solver'], 'glpk')
        self.assertEqual(forwarded_kwargs['nlp_solver'], 'ipopt')
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())
        self.assertEqual(solver.version(), MindtPy.__version__)


class TestAlgorithmBaseClass(unittest.TestCase):
    def _make_rnlp_algorithm(self, termination, message):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.working_model = make_cut_model()
        algorithm.mip = make_cut_model()
        algorithm.results = SolverResults()
        algorithm.log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.rel_gap = 0.6
        algorithm.nlp_opt = FakeSolver(
            solve_result=SimpleNamespace(
                solution=[],
                solver=SimpleNamespace(
                    termination_condition=termination, message=message
                ),
            )
        )
        return algorithm

    def _build_regularization_setup_algorithm(
        self, add_regularization, single_tree=False, sense=minimize
    ):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            add_regularization=add_regularization,
            add_no_good_cuts=True,
            single_tree=single_tree,
        )
        algorithm.objective_sense = sense
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.best_solution_found = make_cut_model(sense=sense)
        algorithm.mip = make_cut_model(sense=sense)
        algorithm.mip_constraint_polynomial_degree = {0, 1, 2}
        algorithm.mip_objective_polynomial_degree = {2}
        mindtpy = algorithm.mip.MindtPy_utils
        mindtpy.mip_obj = Objective(expr=algorithm.mip.x)
        mindtpy.objective_constr = Constraint(expr=algorithm.mip.x >= -2)
        mindtpy.objective_constr.deactivate()
        mindtpy.objective_value = VarList()
        mindtpy.objective_value.add()
        mindtpy.objective_value[1].set_value(1.0)
        if single_tree:
            mindtpy.roa_proj_mip_obj = Objective(expr=algorithm.mip.x)
            mindtpy.cuts.obj_reg_estimate = Constraint(expr=algorithm.mip.x >= -2)
        return algorithm

    def test_set_up_solve_data_covers_constant_objective_and_quadratic_modes(self):
        constant_model = ConcreteModel()
        constant_model.x = Var(bounds=(-1, 1), initialize=0.0)
        constant_model.obj = Objective(expr=5.0)
        constant_model.c = Constraint(expr=constant_model.x >= -0.5)

        algorithm = make_algorithm()
        algorithm.config = make_config(use_fbbt=True, quadratic_strategy=1)
        with patch.object(algorithm_base_class, 'fbbt') as fbbt:
            algorithm.set_up_solve_data(constant_model)

        self.assertFalse(algorithm.config.use_dual_bound)
        fbbt.assert_called_once_with(constant_model)
        self.assertEqual(algorithm.mip_objective_polynomial_degree, {0, 1, 2})
        self.assertEqual(algorithm.mip_constraint_polynomial_degree, {0, 1})
        self.assertTrue(hasattr(algorithm.working_model, 'ipopt_zL_out'))
        self.assertTrue(hasattr(algorithm.working_model, 'ipopt_zU_out'))

        maximize_model = ConcreteModel()
        maximize_model.x = Var(bounds=(-1, 2), initialize=0.5)
        maximize_model.obj = Objective(expr=maximize_model.x**2, sense=maximize)
        maximize_model.c = Constraint(expr=maximize_model.x <= 1.5)

        quadratic_algorithm = make_algorithm()
        quadratic_algorithm.config = make_config(quadratic_strategy=2)
        quadratic_algorithm.set_up_solve_data(maximize_model)
        self.assertEqual(
            quadratic_algorithm.mip_constraint_polynomial_degree, {0, 1, 2}
        )
        self.assertEqual(quadratic_algorithm.primal_bound, float('-inf'))
        self.assertEqual(quadratic_algorithm.dual_bound, float('inf'))

    def test_model_is_valid_short_circuits_linear_program_with_persistent_solver(self):
        model = make_core_model(with_binary=False, nonlinear=False)
        algorithm = make_algorithm(model=model)
        algorithm.results = SolverResults()
        algorithm.mip_opt = FakePersistentSolver(
            solve_result=make_results(lower_bound=1.0, upper_bound=1.0)
        )
        algorithm.nlp_opt = FakeSolver()
        with patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase):
            self.assertFalse(algorithm.model_is_valid())
        self.assertIs(algorithm.mip_opt.instance, algorithm.original_model)
        self.assertEqual(algorithm.results.problem.lower_bound, 1.0)
        self.assertEqual(algorithm.results.problem.upper_bound, 1.0)

    def test_model_is_valid_short_circuits_nlp_and_uses_primal_bound_fallback(self):
        model = make_core_model(with_binary=False, nonlinear=True)
        algorithm = make_algorithm(model=model)
        algorithm.results = SolverResults()
        nlp_results = make_results()
        nlp_results.problem.lower_bound = None
        nlp_results.problem.upper_bound = None
        algorithm.nlp_opt = FakeSolver(solve_result=nlp_results)
        algorithm.mip_opt = FakeSolver()
        self.assertFalse(algorithm.model_is_valid())
        self.assertEqual(
            algorithm.results.problem.upper_bound,
            value(next(model.component_data_objects(Objective, active=True)).expr),
        )
        self.assertEqual(algorithm.results.problem.lower_bound, float('-inf'))

    def test_model_is_valid_sets_dual_suffix_and_rejects_non_suffix_dual(self):
        model = make_core_model()
        algorithm = make_algorithm(model=model, calculate_dual_at_solution=True)
        algorithm.results = SolverResults()
        algorithm.mip_opt = FakeSolver()
        algorithm.nlp_opt = FakeSolver()
        self.assertTrue(algorithm.model_is_valid())
        self.assertTrue(hasattr(algorithm.working_model, 'dual'))

        bad_algorithm = make_algorithm(
            model=make_core_model(), calculate_dual_at_solution=True
        )
        bad_algorithm.results = SolverResults()
        bad_algorithm.working_model.dual = 1
        bad_algorithm.mip_opt = FakeSolver()
        bad_algorithm.nlp_opt = FakeSolver()
        with self.assertRaisesRegex(ValueError, 'dual is not defined as a Suffix'):
            bad_algorithm.model_is_valid()

    def test_bound_updates_and_termination_helpers(self):
        algorithm = make_algorithm(model=make_core_model())
        algorithm.results = SolverResults()
        algorithm.update_gap()
        algorithm.update_dual_bound(-1.0)
        algorithm.update_primal_bound(3.0)
        self.assertTrue(algorithm.dual_bound_improved)
        self.assertTrue(algorithm.primal_bound_improved)

        results = make_results(lower_bound=2.0, upper_bound=5.0)
        algorithm.update_suboptimal_dual_bound(results)
        self.assertEqual(algorithm.dual_bound, 2.0)

        algorithm.best_solution_found = algorithm.working_model.clone()
        algorithm.abs_gap = 0.0
        self.assertTrue(algorithm.bounds_converged())

        algorithm.config.single_tree = False
        algorithm.mip_iter = algorithm.config.iteration_limit
        self.assertTrue(algorithm.reached_iteration_limit())
        self.assertIs(algorithm.results.solver.termination_condition, tc.maxIterations)

        algorithm.results = SolverResults()
        algorithm.config.absolute_bound_tolerance = 1e-6
        algorithm.primal_bound_progress = [10.0, 10.0]
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 9.0
        algorithm.best_solution_found = None
        algorithm.config.stalling_limit = 1
        self.assertTrue(algorithm.reached_stalling_limit())
        self.assertIs(algorithm.results.solver.termination_condition, tc.noSolution)

        algorithm.results = SolverResults()
        with patch.object(
            algorithm_base_class,
            'get_main_elapsed_time',
            return_value=algorithm.config.time_limit,
        ):
            self.assertTrue(algorithm.reached_time_limit())
        self.assertIs(algorithm.results.solver.termination_condition, tc.maxTimeLimit)

    def test_iteration_cycling_and_update_result(self):
        algorithm = make_algorithm(model=make_core_model())
        algorithm.results = SolverResults()
        algorithm.mip = make_cut_model()
        algorithm.integer_list = [(1,)]
        algorithm.config.cycling_check = True
        algorithm.mip_iter = 1
        algorithm.primal_bound = 5
        algorithm.dual_bound = 4
        with patch.object(
            algorithm_base_class, 'get_integer_solution', return_value=(1,)
        ):
            self.assertTrue(algorithm.iteration_cycling())
        self.assertIs(algorithm.results.solver.termination_condition, tc.feasible)

        algorithm.results = SolverResults()
        algorithm.timing = SimpleNamespace(total=7.0)
        algorithm.best_solution_found_time = 3.0
        algorithm.primal_integral = 1.0
        algorithm.dual_integral = 2.0
        algorithm.primal_dual_gap_integral = 3.0
        algorithm.nlp_infeasible_counter = 4
        algorithm.mip_iter = 5
        algorithm.objective_sense = minimize
        algorithm.update_result()
        self.assertEqual(algorithm.results.problem.lower_bound, algorithm.dual_bound)
        self.assertEqual(algorithm.results.problem.upper_bound, algorithm.primal_bound)
        self.assertEqual(algorithm.results.solver.iterations, 5)

    def test_check_subsolver_validity_and_config_adjustments(self):
        algorithm = make_algorithm()
        algorithm.mip_opt = FakeSolver()
        algorithm.nlp_opt = FakeSolver()
        algorithm.check_subsolver_validity()

        algorithm.config.mip_solver = 'appsi_highs'
        algorithm.mip_opt = FakeSolver(version=(1, 6, 0))
        with self.assertRaisesRegex(ValueError, 'HIGHS version 1.7.0'):
            algorithm.check_subsolver_validity()

        algorithm.config = make_config(
            init_strategy='FP',
            add_no_good_cuts=False,
            use_tabu_list=True,
            threads=3,
            solver_tee=True,
            mip_solver='appsi_cplex',
            nlp_solver='appsi_ipopt',
            mip_regularization_solver='appsi_gurobi',
            solution_pool=True,
        )
        algorithm.mip_load_solutions = True
        algorithm.nlp_load_solutions = True
        algorithm.regularization_mip_load_solutions = True
        algorithm.check_config()
        self.assertTrue(algorithm.config.add_no_good_cuts)
        self.assertFalse(algorithm.config.use_tabu_list)
        self.assertTrue(algorithm.config.integer_to_binary)
        self.assertTrue(algorithm.config.mip_solver_tee)
        self.assertTrue(algorithm.config.nlp_solver_tee)

        appsi_algorithm = make_algorithm()
        appsi_algorithm.config = make_config(
            mip_solver='appsi_highs',
            nlp_solver='appsi_ipopt',
            mip_regularization_solver='appsi_gurobi',
        )
        appsi_algorithm.mip_load_solutions = True
        appsi_algorithm.nlp_load_solutions = True
        appsi_algorithm.regularization_mip_load_solutions = True
        appsi_algorithm.check_config()
        self.assertFalse(appsi_algorithm.mip_load_solutions)
        self.assertFalse(appsi_algorithm.nlp_load_solutions)
        self.assertFalse(appsi_algorithm.regularization_mip_load_solutions)

    def test_check_subsolver_validity_reports_availability_and_license_failures(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            add_regularization='level_L1',
            mip_regularization_solver='cplex_persistent',
        )

        error_cases = [
            (
                'glpk is not available',
                FakeSolver(available=False),
                FakeSolver(),
                FakeSolver(),
            ),
            (
                'glpk is not licensed',
                FakeSolver(licensed=False),
                FakeSolver(),
                FakeSolver(),
            ),
            (
                'ipopt is not available',
                FakeSolver(),
                FakeSolver(available=False),
                FakeSolver(),
            ),
            (
                'ipopt is not licensed',
                FakeSolver(),
                FakeSolver(licensed=False),
                FakeSolver(),
            ),
            (
                'cplex_persistent is not available',
                FakeSolver(),
                FakeSolver(),
                FakeSolver(available=False),
            ),
            (
                'cplex_persistent is not licensed',
                FakeSolver(),
                FakeSolver(),
                FakeSolver(licensed=False),
            ),
        ]

        for message, mip_opt, nlp_opt, reg_opt in error_cases:
            algorithm.mip_opt = mip_opt
            algorithm.nlp_opt = nlp_opt
            algorithm.regularization_mip_opt = reg_opt
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    algorithm.check_subsolver_validity()

    def test_check_config_covers_baron_gams_and_tabu_adjustments(self):
        baron_algorithm = make_algorithm()
        baron_algorithm.config = make_config(
            nlp_solver='baron', equality_relaxation=True
        )
        baron_algorithm.check_config()
        self.assertFalse(baron_algorithm.config.equality_relaxation)

        gams_algorithm = make_algorithm()
        gams_algorithm.config = make_config(
            nlp_solver='gams', equality_relaxation=True
        )
        gams_algorithm.config.nlp_solver_args['solver'] = 'baron'
        gams_algorithm.check_config()
        self.assertFalse(gams_algorithm.config.equality_relaxation)

        tabu_algorithm = make_algorithm()
        tabu_algorithm.config = make_config(
            use_tabu_list=True, mip_solver='glpk', threads=4
        )
        tabu_algorithm.check_config()
        self.assertEqual(tabu_algorithm.config.mip_solver, 'cplex_persistent')
        self.assertEqual(tabu_algorithm.config.threads, 1)

    def test_set_up_callbacks_and_solution_pool_names(self):
        algorithm = make_algorithm()
        algorithm.config.mip_solver = 'cplex_persistent'
        algorithm.mip_opt = FakeSolver()
        algorithm.mip_opt._solver_model = FakeCallbackSolverModel()
        algorithm.set_up_tabulist_callback()
        self.assertEqual(algorithm.mip_opt.options['preprocessing_reduce'], 1)
        self.assertIsNotNone(algorithm.mip_opt._solver_model.callback)

        algorithm.set_up_lazy_OA_callback()
        self.assertIsNotNone(algorithm.mip_opt._solver_model.callback)

        algorithm.config.mip_solver = 'gurobi_persistent'
        algorithm.mip_opt = FakeSolver()
        algorithm.set_up_lazy_OA_callback()
        self.assertIsNotNone(algorithm.mip_opt.callback)

        algorithm.objective_sense = minimize
        algorithm.config.num_solution_iteration = 2
        algorithm.config.mip_solver = 'cplex_persistent'
        cplex_pool = SimpleNamespace(
            get_names=lambda: ['b', 'a'],
            get_objective_value=lambda name: {'a': 2.0, 'b': 5.0}[name],
        )
        cplex_model = SimpleNamespace(solution=SimpleNamespace(pool=cplex_pool))
        main_results = SimpleNamespace(_solver_model=cplex_model)
        self.assertEqual(
            algorithm.get_solution_name_obj(main_results), [['a', 2.0], ['b', 5.0]]
        )

    def test_set_appsi_solver_update_config_and_initialize_subsolvers(self):
        algorithm = make_algorithm(model=make_core_model())
        algorithm.results = SolverResults()
        algorithm.config.mip_solver = 'appsi_highs'
        algorithm.config.nlp_solver = 'appsi_ipopt'
        algorithm.mip_opt = FakeSolver()
        algorithm.nlp_opt = FakeSolver()
        algorithm.feasibility_nlp_opt = FakeSolver()
        algorithm.set_appsi_solver_update_config()
        self.assertTrue(
            algorithm.mip_opt.update_config.check_for_new_or_removed_constraints
        )
        self.assertFalse(algorithm.nlp_opt.update_config.check_for_new_or_removed_vars)
        self.assertFalse(
            algorithm.feasibility_nlp_opt.update_config.check_for_new_objective
        )

        algorithm = make_algorithm(model=make_core_model())
        algorithm.config.mip_solver = 'gurobi_persistent'
        algorithm.config.single_tree = True
        algorithm.config.nlp_solver = 'gams'
        algorithm.config.mip_regularization_solver = 'cplex_persistent'
        algorithm.config.add_regularization = 'hess_lag'
        algorithm.config.solution_limit = 3
        fake_mip = FakeSolver()
        fake_nlp = FakeSolver()
        fake_feas = FakeSolver()
        fake_reg = FakeSolver()
        with (
            patch.object(
                algorithm_base_class, 'GurobiPersistent4MindtPy', return_value=fake_mip
            ),
            patch.object(
                algorithm_base_class,
                'SolverFactory',
                side_effect=[fake_nlp, fake_feas, fake_reg],
            ),
            patch.object(algorithm, 'check_subsolver_validity'),
            patch.object(algorithm, 'set_appsi_solver_update_config'),
            patch.object(algorithm_base_class, 'set_solver_mipgap'),
            patch.object(
                algorithm_base_class, 'set_solver_constraint_violation_tolerance'
            ),
        ):
            algorithm.initialize_subsolvers()
        self.assertEqual(fake_mip.options['PreCrush'], 1)
        self.assertEqual(fake_mip.options['LazyConstraints'], 1)
        self.assertEqual(fake_nlp.options['add_options'], [])
        self.assertEqual(fake_reg.options['mip_limits_solutions'], 3)
        self.assertEqual(fake_reg.options['mip_strategy_presolvenode'], 3)
        self.assertEqual(fake_reg.options['optimalitytarget'], 3)

    def test_load_solution_and_handle_fp_main_tc(self):
        algorithm = make_algorithm(model=make_core_model())
        algorithm.results = SolverResults()
        algorithm.best_solution_found = algorithm.working_model.clone()
        algorithm.best_solution_found.x.set_value(2.5)
        algorithm.best_solution_found.y.set_value(0)
        algorithm.load_solution()
        self.assertAlmostEqual(algorithm.original_model.x.value, 2.5)
        self.assertEqual(algorithm.original_model.y.value, 0)

        fp_algorithm = make_algorithm(model=make_core_model())
        fp_algorithm.results = SolverResults()
        fp_algorithm.mip = make_cut_model()
        fp_algorithm.mip.MindtPy_utils.fp_mip_obj = Objective(expr=fp_algorithm.mip.x)
        fp_algorithm.mip.MindtPy_utils.cuts.no_good_cuts.add(
            expr=fp_algorithm.mip.y >= 0
        )
        optimal_results = make_results(termination=tc.optimal)
        self.assertFalse(fp_algorithm.handle_fp_main_tc(optimal_results))

        infeasible_results = make_results(termination=tc.infeasible)
        self.assertTrue(fp_algorithm.handle_fp_main_tc(infeasible_results))

        other_results = make_results(termination=tc.other)
        other_results.solution.status = SolutionStatus.feasible
        self.assertFalse(fp_algorithm.handle_fp_main_tc(other_results))

    def test_initialization_raises_for_missing_initial_binary_values(self):
        init_algorithm = make_algorithm()
        init_algorithm.config = make_config(init_strategy='initial_binary')
        init_algorithm.working_model = make_cut_model()
        init_algorithm.mip = make_cut_model()
        init_algorithm.integer_solution_to_cuts_index = {}
        with patch.object(
            algorithm_base_class,
            'get_integer_solution',
            side_effect=TypeError('missing integer values'),
        ):
            with self.assertRaisesRegex(
                ValueError, 'initial integer combination is not provided'
            ):
                init_algorithm.MindtPy_initialization()

    def test_init_rnlp_updates_suboptimal_bound_for_feasible_termination(self):
        feasible_algorithm = self._make_rnlp_algorithm(
            tc.feasible, 'suboptimal rNLP'
        )
        fake_transform = SimpleNamespace(apply_to=MagicMock())
        with (
            patch.object(
                algorithm_base_class,
                'TransformationFactory',
                return_value=fake_transform,
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
            patch.object(
                feasible_algorithm, 'update_suboptimal_dual_bound'
            ) as update_dual,
        ):
            feasible_algorithm.init_rNLP(add_oa_cuts=False)
        update_dual.assert_called_once()

    def test_init_rnlp_sets_max_time_limit_result(self):
        max_time_algorithm = self._make_rnlp_algorithm(tc.maxTimeLimit, 'timeout')
        with (
            patch.object(
                algorithm_base_class,
                'TransformationFactory',
                return_value=SimpleNamespace(apply_to=MagicMock()),
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
        ):
            max_time_algorithm.init_rNLP(add_oa_cuts=False)
        self.assertIs(
            max_time_algorithm.results.solver.termination_condition, tc.maxTimeLimit
        )

    def test_init_rnlp_leaves_unknown_result_for_iteration_limit(self):
        max_iter_algorithm = self._make_rnlp_algorithm(
            tc.maxIterations, 'iteration limit reached'
        )
        with (
            patch.object(
                algorithm_base_class,
                'TransformationFactory',
                return_value=SimpleNamespace(apply_to=MagicMock()),
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
        ):
            max_iter_algorithm.init_rNLP(add_oa_cuts=False)
        self.assertIs(
            max_iter_algorithm.results.solver.termination_condition, tc.unknown
        )

    def test_init_rnlp_raises_on_unhandled_termination(self):
        error_algorithm = self._make_rnlp_algorithm(tc.error, 'unhandled rNLP error')
        with (
            patch.object(
                algorithm_base_class,
                'TransformationFactory',
                return_value=SimpleNamespace(apply_to=MagicMock()),
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
        ):
            with self.assertRaisesRegex(
                ValueError, 'unable to handle relaxed NLP termination condition'
            ):
                error_algorithm.init_rNLP(add_oa_cuts=False)

    def test_init_max_binaries_covers_persistent_and_termination_branches(self):
        optimal_algorithm = make_algorithm()
        optimal_algorithm.config = make_config()
        optimal_algorithm.working_model = make_cut_model()
        optimal_algorithm.results = SolverResults()
        optimal_algorithm.primal_bound = 10.0
        optimal_algorithm.dual_bound = 4.0
        optimal_algorithm.rel_gap = 0.6
        optimal_algorithm.log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        optimal_algorithm.timing = {}
        optimal_algorithm.mip_load_solutions = True
        optimal_results = SimpleNamespace(
            solution=[object()],
            solver=SimpleNamespace(
                termination_condition=tc.optimal, message='optimal max-binary'
            ),
        )
        optimal_algorithm.mip_opt = FakePersistentSolver(solve_result=optimal_results)
        with (
            patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
            patch.object(algorithm_base_class, 'copy_var_list_values') as copy_values,
            patch(
                'pyomo.core.base.PyomoModel.ModelSolutions.load_from',
                return_value=None,
            ),
        ):
            optimal_algorithm.init_max_binaries()
        self.assertFalse(hasattr(optimal_algorithm.mip_opt.instance, 'dual'))
        copy_values.assert_called_once()

        termination_cases = [
            (tc.infeasible, 'MIP main problem is infeasible'),
            (tc.error, 'unable to handle MILP main termination condition'),
        ]
        for termination, message in termination_cases:
            algorithm = make_algorithm()
            algorithm.config = make_config()
            algorithm.working_model = make_cut_model()
            algorithm.results = SolverResults()
            algorithm.mip_opt = FakeSolver(
                solve_result=SimpleNamespace(
                    solution=[],
                    solver=SimpleNamespace(
                        termination_condition=termination, message=message
                    ),
                )
            )
            with (
                patch.object(algorithm_base_class, 'update_solver_timelimit'),
            ):
                with self.subTest(termination=termination):
                    with self.assertRaisesRegex(ValueError, message):
                        algorithm.init_max_binaries()

        time_limit_algorithm = make_algorithm()
        time_limit_algorithm.config = make_config()
        time_limit_algorithm.working_model = make_cut_model()
        time_limit_algorithm.results = SolverResults()
        time_limit_algorithm.mip_opt = FakeSolver(
            solve_result=SimpleNamespace(
                solution=[],
                solver=SimpleNamespace(
                    termination_condition=tc.maxTimeLimit, message='max time'
                ),
            )
        )
        with patch.object(algorithm_base_class, 'update_solver_timelimit'):
            time_limit_algorithm.init_max_binaries()
        self.assertIs(
            time_limit_algorithm.results.solver.termination_condition, tc.maxTimeLimit
        )

        max_iter_algorithm = make_algorithm()
        max_iter_algorithm.config = make_config()
        max_iter_algorithm.working_model = make_cut_model()
        max_iter_algorithm.results = SolverResults()
        max_iter_algorithm.mip_opt = FakeSolver(
            solve_result=SimpleNamespace(
                solution=[],
                solver=SimpleNamespace(
                    termination_condition=tc.maxIterations, message='max iterations'
                ),
            )
        )
        with patch.object(algorithm_base_class, 'update_solver_timelimit'):
            max_iter_algorithm.init_max_binaries()
        self.assertIs(
            max_iter_algorithm.results.solver.termination_condition, tc.unknown
        )

    def test_handle_subproblem_optimal_updates_best_solution_and_cuts(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            calculate_dual_at_solution=True, add_no_good_cuts=True, nlp_solver='cyipopt'
        )
        algorithm.working_model = make_cut_model()
        algorithm.mip = make_cut_model()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.fixed_nlp_log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.primal_bound_progress = [10.0]
        algorithm.dual_bound_progress = [4.0]
        algorithm.rel_gap = 0.6
        algorithm.mip_iter = 1
        algorithm.nlp_iter = 1
        algorithm.objective_sense = minimize
        algorithm.timing = {}
        fixed_nlp = make_cut_model()
        fixed_nlp.MindtPy_utils.objective_list = [fixed_nlp.obj]
        fixed_nlp.tmp_duals = ComponentMap([(fixed_nlp.c, 1.0)])
        fixed_nlp.dual[fixed_nlp.c] = 2.0

        def mark_primal_bound(_bound):
            algorithm.primal_bound_improved = True

        algorithm.update_primal_bound = mark_primal_bound
        algorithm.add_cuts = MagicMock()
        with (
            patch.object(algorithm_base_class, 'copy_var_list_values'),
            patch.object(algorithm_base_class, 'add_no_good_cuts') as add_nogood,
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            algorithm.handle_subproblem_optimal(fixed_nlp)
        algorithm.add_cuts.assert_called_once()
        add_nogood.assert_called_once()
        self.assertIsNotNone(algorithm.best_solution_found)

    def test_handle_subproblem_infeasible_adds_cuts_and_no_good_cut(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            calculate_dual_at_solution=True, add_no_good_cuts=True
        )
        algorithm.mip = make_cut_model()
        algorithm.infeasible_fixed_nlp_log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.primal_bound_progress = [10.0]
        algorithm.dual_bound_progress = [4.0]
        algorithm.rel_gap = 0.6
        algorithm.mip_iter = 1
        algorithm.nlp_iter = 1
        algorithm.objective_sense = minimize
        algorithm.timing = {}
        fixed_nlp = make_cut_model()
        fixed_nlp.MindtPy_utils.objective_list = [fixed_nlp.obj]
        algorithm.solve_feasibility_subproblem = lambda: (
            make_cut_model(),
            make_results(),
        )
        algorithm.add_cuts = MagicMock()
        with (
            patch.object(algorithm_base_class, 'copy_var_list_values'),
            patch.object(algorithm_base_class, 'add_no_good_cuts') as add_nogood,
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            algorithm.handle_subproblem_infeasible(fixed_nlp)
        algorithm.add_cuts.assert_called_once()
        add_nogood.assert_called_once()

    def test_handle_subproblem_other_termination_and_dispatch(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(add_no_good_cuts=True)
        algorithm.results = SolverResults()
        fixed_nlp = make_cut_model()

        with patch.object(algorithm_base_class, 'add_no_good_cuts') as add_nogood:
            algorithm.handle_subproblem_other_termination(fixed_nlp, tc.maxIterations)
        add_nogood.assert_called_once()
        with self.assertRaisesRegex(
            ValueError, 'unable to handle NLP subproblem termination'
        ):
            algorithm.handle_subproblem_other_termination(fixed_nlp, tc.error)

        with (
            patch.object(algorithm, 'handle_subproblem_optimal') as optimal_handler,
            patch.object(
                algorithm, 'handle_subproblem_infeasible'
            ) as infeasible_handler,
        ):
            algorithm.handle_nlp_subproblem_tc(fixed_nlp, make_results())
            algorithm.handle_nlp_subproblem_tc(
                fixed_nlp, make_results(termination=tc.infeasible)
            )
        optimal_handler.assert_called_once()
        infeasible_handler.assert_called_once()

    def test_handle_main_problem_handlers_cover_optimal_timelimit_and_unbounded(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.mip_opt = FakePersistentSolver(solve_result=make_results())
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.primal_bound_progress = [10.0]
        algorithm.dual_bound_progress = [4.0]
        algorithm.rel_gap = 0.6
        algorithm.mip_iter = 1
        algorithm.timing = {}
        main_mip = make_cut_model()
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)
        with (
            patch.object(algorithm_base_class, 'copy_var_list_values'),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
        ):
            algorithm.handle_main_optimal(main_mip)
            algorithm.handle_main_max_timelimit(main_mip, make_results(lower_bound=1.0))
            algorithm.handle_main_unbounded(main_mip)

    def test_handle_main_infeasible_and_regularization_dispatch(self):
        algorithm = make_algorithm()
        algorithm.results = SolverResults()
        algorithm.results.solver.termination_condition = None
        algorithm.primal_bound = float('inf')
        algorithm.objective_sense = minimize
        algorithm.mip_iter = 0
        algorithm.handle_main_infeasible()
        self.assertIs(algorithm.results.solver.termination_condition, tc.infeasible)

        main_mip = make_cut_model()
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)
        with patch.object(algorithm, 'handle_main_optimal') as handle_main_optimal:
            algorithm.handle_regularization_main_tc(main_mip, None)
            algorithm.handle_regularization_main_tc(main_mip, make_results())
            unknown = make_results(termination=tc.unknown, lower_bound=0.0)
            algorithm.handle_regularization_main_tc(main_mip, unknown)
        self.assertEqual(handle_main_optimal.call_count, 2)

    def test_handle_main_mip_termination_dispatches_status_handlers(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.results = SolverResults()
        algorithm.fixed_nlp = make_cut_model()
        main_mip = make_cut_model()
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)
        with (
            patch.object(algorithm, 'handle_main_optimal') as handle_main_optimal,
            patch.object(algorithm, 'handle_main_infeasible') as handle_main_infeasible,
            patch.object(
                algorithm, 'handle_main_unbounded', return_value=make_results()
            ) as handle_main_unbounded,
            patch.object(
                algorithm, 'handle_main_max_timelimit'
            ) as handle_main_max_timelimit,
            patch.object(algorithm_base_class, 'copy_var_list_values'),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            self.assertFalse(
                algorithm.handle_main_mip_termination(main_mip, make_results())
            )
            self.assertTrue(
                algorithm.handle_main_mip_termination(
                    main_mip, make_results(termination=tc.infeasible)
                )
            )
            algorithm.handle_main_mip_termination(
                main_mip, make_results(termination=tc.unbounded)
            )
            algorithm.handle_main_mip_termination(
                main_mip, make_results(termination=tc.maxTimeLimit, lower_bound=1.0)
            )
        handle_main_optimal.assert_called()
        handle_main_infeasible.assert_called()
        handle_main_unbounded.assert_called()
        handle_main_max_timelimit.assert_called()

    def test_handle_main_mip_termination_covers_feasible_and_none_paths(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.results = SolverResults()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 4.0
        algorithm.rel_gap = 0.6
        algorithm.mip_iter = 1
        algorithm.timing = {}
        algorithm.log_formatter = '{0}{1}{2}{3}{4}{5}{6}'

        main_mip = make_cut_model()
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)

        with (
            patch.object(algorithm_base_class, 'copy_var_list_values') as copy_values,
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
            patch.object(
                algorithm, 'update_suboptimal_dual_bound'
            ) as update_suboptimal_dual_bound,
        ):
            should_terminate = algorithm.handle_main_mip_termination(
                main_mip, make_results(termination=tc.feasible)
            )
        self.assertFalse(should_terminate)
        copy_values.assert_called_once()
        update_suboptimal_dual_bound.assert_called_once()

        other_results = make_results(termination=tc.other)
        other_results.solution.status = SolutionStatus.feasible
        with (
            patch.object(algorithm_base_class, 'copy_var_list_values') as copy_values,
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
            patch.object(
                algorithm, 'update_suboptimal_dual_bound'
            ) as update_suboptimal_dual_bound,
        ):
            should_terminate = algorithm.handle_main_mip_termination(
                main_mip, other_results
            )
        self.assertFalse(should_terminate)
        copy_values.assert_called_once()
        update_suboptimal_dual_bound.assert_called_once()

        self.assertTrue(algorithm.handle_main_mip_termination(main_mip, None))

    def test_solve_subproblem_covers_restore_load_and_infeasible_paths(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(calculate_dual_at_solution=True)
        algorithm.timing = {}
        algorithm.fixed_nlp = make_cut_model()
        algorithm.initial_var_values = [7.0, 1.0]
        result_with_solution = SimpleNamespace(
            solution=[object()],
            solver=SimpleNamespace(termination_condition=tc.optimal),
        )
        algorithm.nlp_opt = FakeSolver(solve_result=result_with_solution)
        fake_transform = SimpleNamespace(apply_to=MagicMock(), revert=MagicMock())

        def fake_value(expr, *args, **kwargs):
            if expr is algorithm.fixed_nlp.c.body:
                raise ValueError('bad body evaluation')
            return value(expr, *args, **kwargs)

        with (
            patch.object(
                algorithm_base_class,
                'TransformationFactory',
                return_value=fake_transform,
            ),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(algorithm.fixed_nlp.solutions, 'load_from') as load_from,
            patch.object(algorithm_base_class, 'value', side_effect=fake_value),
        ):
            fixed_nlp, results = algorithm.solve_subproblem()
        self.assertIs(fixed_nlp, algorithm.fixed_nlp)
        self.assertIs(results, result_with_solution)
        self.assertEqual(algorithm.fixed_nlp.x.value, 7.0)
        load_from.assert_called_once_with(result_with_solution)
        fake_transform.revert.assert_called_once_with(algorithm.fixed_nlp)

        infeasible_algorithm = make_algorithm()
        infeasible_algorithm.config = make_config()
        infeasible_algorithm.fixed_nlp = make_cut_model()
        infeasible_algorithm.initial_var_values = [1.0, 1.0]
        infeasible_transform = SimpleNamespace(
            apply_to=MagicMock(
                side_effect=InfeasibleConstraintException('trivial infeasibility')
            ),
            revert=MagicMock(),
        )
        with patch.object(
            algorithm_base_class,
            'TransformationFactory',
            return_value=infeasible_transform,
        ):
            _, infeasible_results = infeasible_algorithm.solve_subproblem()
        self.assertIs(infeasible_results.solver.termination_condition, tc.infeasible)

    def test_nlp_and_feasibility_termination_edge_cases(self):
        algorithm = make_algorithm()
        algorithm.results = SolverResults()
        fixed_nlp = make_cut_model()
        algorithm.handle_nlp_subproblem_tc(
            fixed_nlp, make_results(termination=tc.maxTimeLimit)
        )
        self.assertTrue(algorithm.should_terminate)
        self.assertIs(algorithm.results.solver.termination_condition, tc.maxTimeLimit)

        algorithm.results = SolverResults()
        algorithm.should_terminate = False
        algorithm.handle_nlp_subproblem_tc(
            fixed_nlp, make_results(termination=tc.maxEvaluations)
        )
        self.assertTrue(algorithm.should_terminate)
        self.assertIs(algorithm.results.solver.termination_condition, tc.maxEvaluations)

        algorithm.results = SolverResults()
        algorithm.should_terminate = False
        mindtpy = make_cut_model().MindtPy_utils
        mindtpy.feas_obj = Objective(expr=0)
        mindtpy.feas_obj.construct()
        algorithm.handle_feasibility_subproblem_tc(tc.maxIterations, mindtpy)
        self.assertTrue(algorithm.should_terminate)
        self.assertIs(algorithm.results.solver.status, SolverStatus.error)

        algorithm.results = SolverResults()
        algorithm.should_terminate = False
        algorithm.handle_feasibility_subproblem_tc(tc.error, mindtpy)
        self.assertTrue(algorithm.should_terminate)
        self.assertIs(algorithm.results.solver.status, SolverStatus.error)

        algorithm.results = SolverResults()
        algorithm.should_terminate = True
        algorithm.primal_bound = float('inf')
        algorithm.primal_bound_progress = [float('inf')]
        self.assertTrue(algorithm.algorithm_should_terminate(check_cycling=False))
        self.assertIs(algorithm.results.solver.termination_condition, tc.noSolution)

        algorithm.results = SolverResults()
        algorithm.should_terminate = True
        algorithm.primal_bound = 0.0
        algorithm.primal_bound_progress = [float('inf')]
        self.assertTrue(algorithm.algorithm_should_terminate(check_cycling=False))
        self.assertIs(algorithm.results.solver.termination_condition, tc.feasible)

    def test_fix_dual_bound_and_set_up_mip_solver(self):
        algorithm = make_algorithm()
        algorithm.results = SolverResults()
        algorithm.config = make_config(single_tree=True)
        algorithm.primal_bound = 3.0
        algorithm.stored_bound = {3.0: 1.0}
        algorithm.fix_dual_bound(last_iter_cuts=True)
        self.assertEqual(algorithm.dual_bound, 1.0)

        algorithm.stored_bound = {}
        algorithm.fix_dual_bound(last_iter_cuts=True)

        non_single_tree = make_algorithm()
        non_single_tree.results = SolverResults()
        non_single_tree.config = make_config(add_regularization='grad_lag')
        non_single_tree.primal_bound = 1.0
        non_single_tree.dual_bound = 1.0
        non_single_tree.timing = {}
        non_single_tree.mip = make_cut_model()
        non_single_tree.mip.MindtPy_utils.objective_list[-1].deactivate()
        non_single_tree.fixed_nlp = make_cut_model()
        mip_results = SimpleNamespace(
            solution=[object()],
            solver=SimpleNamespace(termination_condition=tc.optimal),
            problem=SimpleNamespace(lower_bound=1.0, upper_bound=1.0),
        )
        non_single_tree.mip_opt = FakePersistentSolver(solve_result=mip_results)
        with (
            patch.object(
                non_single_tree,
                'solve_subproblem',
                return_value=(make_cut_model(), make_results()),
            ),
            patch.object(non_single_tree, 'handle_nlp_subproblem_tc'),
            patch.object(
                non_single_tree,
                'deactivate_no_good_cuts_when_fixing_bound',
                create=True,
            ),
            patch.object(
                non_single_tree, 'update_suboptimal_dual_bound'
            ) as update_dual,
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(non_single_tree.mip.solutions, 'load_from') as load_from,
            patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase),
        ):
            non_single_tree.fix_dual_bound(last_iter_cuts=False)
        self.assertTrue(non_single_tree.mip.MindtPy_utils.objective_list[-1].active)
        self.assertIs(non_single_tree.results.solver.termination_condition, tc.optimal)
        update_dual.assert_called_once_with(mip_results)
        load_from.assert_called_once_with(mip_results)

        callback_algorithm = make_algorithm()
        callback_algorithm.config = make_config(
            single_tree=True, use_tabu_list=True, mip_solver='cplex_persistent'
        )
        callback_algorithm.mip = make_cut_model()
        callback_algorithm.mip_opt = FakePersistentSolver()
        callback_algorithm.set_up_lazy_OA_callback = MagicMock()
        callback_algorithm.set_up_tabulist_callback = MagicMock()
        with patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase):
            mip_args = callback_algorithm.set_up_mip_solver()
        self.assertTrue(mip_args['warmstart'])
        callback_algorithm.set_up_lazy_OA_callback.assert_called_once()
        callback_algorithm.set_up_tabulist_callback.assert_called_once()

    def test_solve_main_returns_timeout_after_solver_exception(self):
        algorithm = make_algorithm()
        algorithm.results = SolverResults()
        algorithm.config = make_config(
            single_tree=True, strategy='GOA', add_no_good_cuts=True
        )
        algorithm.mip = make_cut_model()
        algorithm.setup_main = MagicMock()
        algorithm.set_up_mip_solver = MagicMock(return_value={})
        algorithm.mip_opt = FakeSolver()
        algorithm.mip_opt.solve = MagicMock(side_effect=ValueError('bad solve'))
        with (
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class,
                'get_main_elapsed_time',
                return_value=algorithm.config.time_limit,
            ),
        ):
            main_mip, main_results = algorithm.solve_main()
        self.assertIsNone(main_mip)
        self.assertIsNone(main_results)
        self.assertIs(
            algorithm.results.solver.termination_condition, tc.maxTimeLimit
        )

    def test_solve_main_solution_pool_preserves_solver_artifacts(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            single_tree=True, add_no_good_cuts=False, solution_pool=True
        )
        algorithm.mip = make_cut_model()
        algorithm.setup_main = MagicMock()
        algorithm.set_up_mip_solver = MagicMock(return_value={})
        optimal_results = SimpleNamespace(
            solution=[object()],
            solver=SimpleNamespace(termination_condition=tc.optimal),
            problem=SimpleNamespace(),
        )
        algorithm.mip_opt = FakePersistentSolver(solve_result=optimal_results)
        algorithm.mip_opt._solver_model = 'solver-model'
        algorithm.mip_opt._pyomo_var_to_solver_var_map = 'var-map'
        with (
            patch.object(algorithm, 'update_suboptimal_dual_bound') as update_dual,
            patch.object(algorithm.mip.solutions, 'load_from') as load_from,
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase),
        ):
            _, pool_results = algorithm.solve_main()
        self.assertEqual(pool_results._solver_model, 'solver-model')
        self.assertEqual(pool_results._pyomo_var_to_solver_var_map, 'var-map')
        update_dual.assert_called_once_with(optimal_results)
        load_from.assert_called_once_with(optimal_results)

    def test_solve_main_and_fp_main_distinguish_infeasible_or_unbounded(self):
        distinguish_algorithm = make_algorithm()
        distinguish_algorithm.config = make_config()
        distinguish_algorithm.mip = make_cut_model()
        distinguish_algorithm.setup_main = MagicMock()
        distinguish_algorithm.set_up_mip_solver = MagicMock(return_value={})
        distinguish_algorithm.mip_opt = FakeSolver(
            solve_result=make_results(termination=tc.infeasibleOrUnbounded)
        )
        with (
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class,
                'distinguish_mip_infeasible_or_unbounded',
                return_value=(make_results(termination=tc.unbounded), None),
            ) as distinguish,
        ):
            distinguish_algorithm.solve_main()
        distinguish.assert_called_once()

        fp_algorithm = make_algorithm()
        fp_algorithm.config = make_config()
        fp_algorithm.mip = make_cut_model()
        fp_algorithm.setup_fp_main = MagicMock()
        fp_algorithm.set_up_mip_solver = MagicMock(return_value={})
        fp_algorithm.mip_opt = FakeSolver(
            solve_result=make_results(termination=tc.infeasibleOrUnbounded)
        )
        with (
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class,
                'distinguish_mip_infeasible_or_unbounded',
                return_value=(make_results(termination=tc.unbounded), None),
            ) as distinguish,
        ):
            fp_algorithm.solve_fp_main()
        distinguish.assert_called_once()

    def test_handle_main_optimal_sets_missing_binary_values(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.fixed_nlp = make_cut_model()
        main_mip = make_cut_model()
        main_mip.y.set_value(None, skip_validation=True)
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)
        with (
            patch.object(algorithm_base_class, 'copy_var_list_values'),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            algorithm.handle_main_optimal(main_mip, update_bound=False)
        self.assertEqual(main_mip.y.value, 0)

    def test_handle_main_unbounded_loads_solution_with_persistent_solver(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.mip = make_cut_model()
        algorithm.primal_bound = 10.0
        algorithm.dual_bound = 5.0
        algorithm.rel_gap = 0.5
        algorithm.mip_iter = 1
        algorithm.termination_condition_log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        algorithm.mip_opt = FakePersistentSolver(
            solve_result=SimpleNamespace(
                solution=[object()],
                solver=SimpleNamespace(termination_condition=tc.optimal),
            )
        )
        algorithm.timing = {}
        main_mip = make_cut_model()
        main_mip.MindtPy_utils.mip_obj = Objective(expr=main_mip.x)
        with (
            patch.object(algorithm.mip.solutions, 'load_from') as load_from,
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            algorithm.handle_main_unbounded(main_mip)
        load_from.assert_called_once()

    def test_handle_regularization_main_tc_covers_supported_and_error_paths(self):
        algorithm = make_algorithm()
        algorithm.results = SolverResults()
        algorithm.config = make_config()
        main_mip = make_cut_model()
        algorithm.handle_regularization_main_tc(
            main_mip, make_results(termination=tc.maxTimeLimit)
        )
        self.assertIs(algorithm.results.solver.termination_condition, tc.maxTimeLimit)
        algorithm.handle_regularization_main_tc(
            main_mip, make_results(termination=tc.infeasible)
        )
        algorithm.handle_regularization_main_tc(
            main_mip, make_results(termination=tc.unbounded)
        )
        algorithm.handle_regularization_main_tc(
            main_mip, make_results(termination=tc.infeasibleOrUnbounded)
        )
        unknown = make_results(termination=tc.unknown)
        unknown.problem.lower_bound = float('-inf')
        algorithm.handle_regularization_main_tc(main_mip, unknown)
        with self.assertRaisesRegex(ValueError, 'regularization problem termination'):
            algorithm.handle_regularization_main_tc(
                main_mip,
                make_results(termination=tc.error, message='bad regularization solve'),
            )

    def test_solve_regularization_main_cleans_up_projection_components(self):
        regularization_algorithm = make_algorithm()
        regularization_algorithm.config = make_config(
            mip_regularization_solver='cplex_persistent', add_regularization='level_L1'
        )
        regularization_algorithm.regularization_mip_type = 'MILP'
        regularization_algorithm.primal_bound = 10.0
        regularization_algorithm.dual_bound = 5.0
        regularization_algorithm.rel_gap = 0.5
        regularization_algorithm.mip_iter = 1
        regularization_algorithm.timing = {}
        regularization_algorithm.log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        regularization_algorithm.mip = make_cut_model()
        regularization_algorithm.mip.MindtPy_utils.roa_proj_mip_obj = Objective(
            expr=regularization_algorithm.mip.x
        )
        regularization_algorithm.mip.MindtPy_utils.objective_constr = Constraint(
            expr=regularization_algorithm.mip.x >= -2
        )
        regularization_algorithm.mip.MindtPy_utils.cuts.obj_reg_estimate = Constraint(
            expr=regularization_algorithm.mip.x >= -2
        )
        regularization_algorithm.mip.MindtPy_utils.L1_obj = Objective(
            expr=regularization_algorithm.mip.x
        )
        reg_results = SimpleNamespace(
            solution=[object()],
            solver=SimpleNamespace(termination_condition=tc.optimal),
        )
        regularization_algorithm.regularization_mip_opt = FakePersistentSolver(
            solve_result=reg_results
        )
        with (
            patch.object(regularization_algorithm, 'setup_regularization_main'),
            patch.object(
                regularization_algorithm.mip.solutions, 'load_from'
            ) as load_from,
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(algorithm_base_class, 'PersistentSolver', FakePersistentBase),
            patch.object(
                algorithm_base_class, 'get_main_elapsed_time', return_value=1.0
            ),
        ):
            regularization_algorithm.solve_regularization_main()
        load_from.assert_called_once_with(reg_results)
        self.assertIsNone(
            regularization_algorithm.mip.MindtPy_utils.component('roa_proj_mip_obj')
        )
        self.assertIsNone(
            regularization_algorithm.mip.MindtPy_utils.cuts.component(
                'obj_reg_estimate'
            )
        )
        self.assertIsNone(
            regularization_algorithm.mip.MindtPy_utils.component('L1_obj')
        )

    def test_solve_regularization_main_distinguishes_infeasible_or_unbounded(self):
        distinguish_regularization = make_algorithm()
        distinguish_regularization.config = make_config(
            mip_regularization_solver='gurobi', add_regularization='level_L2'
        )
        distinguish_regularization.regularization_mip_type = 'MIQP'
        distinguish_regularization.primal_bound = 10.0
        distinguish_regularization.dual_bound = 5.0
        distinguish_regularization.rel_gap = 0.5
        distinguish_regularization.mip_iter = 1
        distinguish_regularization.timing = {}
        distinguish_regularization.log_formatter = '{0}{1}{2}{3}{4}{5}{6}'
        distinguish_regularization.mip = make_cut_model()
        distinguish_regularization.mip.MindtPy_utils.roa_proj_mip_obj = Objective(
            expr=distinguish_regularization.mip.x
        )
        distinguish_regularization.mip.MindtPy_utils.objective_constr = Constraint(
            expr=distinguish_regularization.mip.x >= -2
        )
        distinguish_regularization.mip.MindtPy_utils.cuts.obj_reg_estimate = Constraint(
            expr=distinguish_regularization.mip.x >= -2
        )
        distinguish_regularization.regularization_mip_opt = FakeSolver(
            solve_result=make_results(termination=tc.infeasibleOrUnbounded)
        )
        with (
            patch.object(distinguish_regularization, 'setup_regularization_main'),
            patch.object(algorithm_base_class, 'update_solver_timelimit'),
            patch.object(
                algorithm_base_class,
                'distinguish_mip_infeasible_or_unbounded',
                return_value=(make_results(termination=tc.infeasible), None),
            ) as distinguish,
        ):
            distinguish_regularization.solve_regularization_main()
        distinguish.assert_called_once()

    def test_setup_fp_main_supports_l2_and_l_infinity_norms(self):
        fp_algorithm = make_algorithm()
        fp_algorithm.mip = make_cut_model()
        fp_algorithm.working_model = make_cut_model()
        fp_algorithm.mip_constraint_polynomial_degree = {0, 1, 2}
        fp_algorithm.mip.MindtPy_utils.mip_obj = Objective(expr=fp_algorithm.mip.x)
        fp_algorithm.mip.MindtPy_utils.fp_mip_obj = Objective(expr=fp_algorithm.mip.x)
        fp_algorithm.config = make_config(fp_main_norm='L2')
        with patch.object(
            algorithm_base_class,
            'generate_norm2sq_objective_function',
            return_value=Objective(expr=fp_algorithm.mip.x),
        ):
            fp_algorithm.setup_fp_main()
        self.assertIsNotNone(fp_algorithm.mip.MindtPy_utils.fp_mip_obj)

        fp_algorithm = make_algorithm()
        fp_algorithm.mip = make_cut_model()
        fp_algorithm.working_model = make_cut_model()
        fp_algorithm.mip_constraint_polynomial_degree = {0, 1, 2}
        fp_algorithm.mip.MindtPy_utils.mip_obj = Objective(expr=fp_algorithm.mip.x)
        fp_algorithm.mip.MindtPy_utils.fp_mip_obj = Objective(expr=fp_algorithm.mip.x)
        fp_algorithm.config = make_config(fp_main_norm='L_infinity')
        with patch.object(
            algorithm_base_class,
            'generate_norm_inf_objective_function',
            return_value=Objective(expr=fp_algorithm.mip.x),
        ):
            fp_algorithm.setup_fp_main()
        self.assertIsNotNone(fp_algorithm.mip.MindtPy_utils.fp_mip_obj)

    def test_setup_regularization_main_builds_expected_projection_components(self):
        level_l1 = self._build_regularization_setup_algorithm(
            'level_L1', single_tree=True
        )
        with patch.object(
            algorithm_base_class,
            'generate_norm1_objective_function',
            return_value=Objective(expr=level_l1.mip.x),
        ):
            level_l1.setup_regularization_main()
        self.assertTrue(level_l1.mip.MindtPy_utils.objective_constr.active)
        self.assertTrue(level_l1.mip.MindtPy_utils.cuts.no_good_cuts.active)
        self.assertIsNotNone(level_l1.mip.MindtPy_utils.roa_proj_mip_obj)

        level_l2 = self._build_regularization_setup_algorithm('level_L2')
        with patch.object(
            algorithm_base_class,
            'generate_norm2sq_objective_function',
            return_value=Objective(expr=level_l2.mip.x),
        ):
            level_l2.setup_regularization_main()
        self.assertIsNotNone(level_l2.mip.MindtPy_utils.roa_proj_mip_obj)

        level_linf = self._build_regularization_setup_algorithm(
            'level_L_infinity', sense=maximize
        )
        with patch.object(
            algorithm_base_class,
            'generate_norm_inf_objective_function',
            return_value=Objective(expr=level_linf.mip.x),
        ):
            level_linf.setup_regularization_main()
        self.assertIsNotNone(level_linf.mip.MindtPy_utils.roa_proj_mip_obj)
        self.assertTrue(level_linf.mip.MindtPy_utils.cuts.obj_reg_estimate.active)

        lag_algorithm = self._build_regularization_setup_algorithm('grad_lag')
        with patch.object(
            algorithm_base_class,
            'generate_lag_objective_function',
            return_value=Objective(expr=lag_algorithm.mip.x),
        ):
            lag_algorithm.setup_regularization_main()
        self.assertIsNotNone(lag_algorithm.mip.MindtPy_utils.roa_proj_mip_obj)

    def test_iteration_loop_and_feasibility_handler(self):
        algorithm = make_algorithm()
        algorithm.config = make_config()
        algorithm.config.iteration_limit = 2
        algorithm.timing = {}
        algorithm.mip = make_cut_model()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.integer_list = []
        with (
            patch.object(
                algorithm, 'solve_main', return_value=(make_cut_model(), make_results())
            ),
            patch.object(algorithm, 'handle_main_mip_termination', return_value=False),
            patch.object(
                algorithm,
                'solve_subproblem',
                return_value=(make_cut_model(), make_results()),
            ),
            patch.object(algorithm, 'handle_nlp_subproblem_tc'),
            patch.object(
                algorithm, 'algorithm_should_terminate', side_effect=[False, True]
            ),
        ):
            algorithm.MindtPy_iteration_loop()
        self.assertTrue(algorithm.last_iter_cuts)

        algorithm.results = SolverResults()
        algorithm.working_model = make_cut_model()
        algorithm.config = make_config()
        algorithm.timing = {}
        mindtpy = make_cut_model().MindtPy_utils
        mindtpy.feas_obj = Objective(expr=0)
        mindtpy.feas_obj.construct()
        with patch.object(algorithm_base_class, 'copy_var_list_values'):
            algorithm.handle_feasibility_subproblem_tc(tc.optimal, mindtpy)
        algorithm.handle_feasibility_subproblem_tc(tc.infeasible, mindtpy)
        self.assertTrue(algorithm.should_terminate)

    def test_iteration_loop_covers_single_tree_regularization_path(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(
            single_tree=True, add_regularization='level_L1'
        )
        algorithm.config.iteration_limit = 1
        algorithm.timing = {}
        algorithm.mip = make_cut_model()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.integer_list = []
        algorithm.solve_main = MagicMock(return_value=(make_cut_model(), make_results()))
        algorithm.handle_main_mip_termination = MagicMock(side_effect=[False, True])
        algorithm.add_regularization = MagicMock()
        algorithm.solve_subproblem = MagicMock(
            return_value=(algorithm.fixed_nlp, make_results())
        )
        algorithm.handle_nlp_subproblem_tc = MagicMock()
        algorithm.algorithm_should_terminate = MagicMock(return_value=True)
        algorithm.config.call_before_subproblem_solve = MagicMock()

        with (
            patch.object(
                algorithm_base_class, 'get_integer_solution', return_value=(1,)
            ),
            patch.object(algorithm_base_class, 'copy_var_list_values') as copy_values,
        ):
            algorithm.MindtPy_iteration_loop()

        algorithm.add_regularization.assert_not_called()
        copy_values.assert_called_once()
        algorithm.config.call_before_subproblem_solve.assert_called_once_with(
            algorithm.fixed_nlp
        )
        algorithm.solve_subproblem.assert_called_once()
        algorithm.handle_nlp_subproblem_tc.assert_called_once()
        self.assertFalse(algorithm.last_iter_cuts)

    def test_iteration_loop_covers_solution_pool_replay_and_skip(self):
        algorithm = make_algorithm()
        algorithm.config = make_config(solution_pool=True)
        algorithm.config.iteration_limit = 1
        algorithm.timing = {}
        algorithm.mip = make_cut_model()
        algorithm.fixed_nlp = make_cut_model()
        algorithm.integer_list = [(1,)]
        algorithm.solve_main = MagicMock(
            return_value=(
                make_cut_model(),
                SimpleNamespace(
                    _solver_model='solver-model',
                    _pyomo_var_to_solver_var_map='var-map',
                ),
            )
        )
        algorithm.handle_main_mip_termination = MagicMock(side_effect=[False, True])
        algorithm.get_solution_name_obj = MagicMock(
            return_value=[('first', 1.0), ('repeat', 2.0), ('new', 3.0)]
        )
        algorithm.solve_subproblem = MagicMock(
            return_value=(algorithm.fixed_nlp, make_results())
        )
        algorithm.handle_nlp_subproblem_tc = MagicMock()
        algorithm.algorithm_should_terminate = MagicMock(
            side_effect=[False, False, True]
        )
        algorithm.config.call_before_subproblem_solve = MagicMock()
        algorithm.config.call_after_subproblem_solve = MagicMock()
        algorithm.config.call_after_main_solve = MagicMock()

        with (
            patch.object(
                algorithm_base_class, 'copy_var_list_values_from_solution_pool'
            ) as copy_from_solution_pool,
            patch.object(
                algorithm_base_class,
                'get_integer_solution',
                side_effect=[(1,), (2,)],
            ),
        ):
            algorithm.MindtPy_iteration_loop()

        self.assertEqual(copy_from_solution_pool.call_count, 2)
        self.assertEqual(algorithm.solve_subproblem.call_count, 2)
        self.assertEqual(algorithm.handle_nlp_subproblem_tc.call_count, 2)
        self.assertEqual(
            algorithm.config.call_before_subproblem_solve.call_count, 2
        )
        self.assertEqual(
            algorithm.config.call_after_subproblem_solve.call_count, 2
        )
        self.assertEqual(algorithm.integer_list, [(1,), (2,)])
        self.assertTrue(algorithm.last_iter_cuts)


class TestOaAndGoaSolvers(unittest.TestCase):
    def test_goa_configuration_and_no_good_cut_bookkeeping(self):
        solver = global_outer_approximation.MindtPy_GOA_Solver()
        solver.config = make_goa_config(
            add_no_good_cuts=False, use_tabu_list=False, single_tree=False
        )
        with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
            solver.check_config()
        self.assertTrue(solver.config.add_no_good_cuts)

        solver.config = make_goa_config(single_tree=True, mip_solver='glpk')
        with self.assertRaisesRegex(
            ValueError, 'Only cplex_persistent and gurobi_persistent'
        ):
            with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
                solver.check_config()

        solver.config = make_goa_config(
            add_no_good_cuts=True, use_tabu_list=True, single_tree=False
        )
        solver.mip = make_cut_model()
        solver.mip.MindtPy_utils.cuts.no_good_cuts.add(expr=solver.mip.y >= 0)
        solver.num_no_good_cuts_added = {}
        solver.primal_bound_progress_time = []
        solver.primal_bound = 9.0
        with (
            patch.object(
                algorithm_base_class._MindtPyAlgorithm, 'update_primal_bound'
            ) as update_primal_bound,
            patch.object(
                global_outer_approximation, 'get_main_elapsed_time', return_value=4.0
            ),
        ):
            update_primal_bound.side_effect = lambda bound: setattr(
                solver, 'primal_bound_improved', True
            )
            solver.update_primal_bound(9.0)
        self.assertEqual(solver.num_no_good_cuts_added[9.0], 1)

        solver.primal_bound = 9.0
        solver.integer_list = [(1,), (0,)]
        solver.mip.MindtPy_utils.cuts.no_good_cuts.add(expr=solver.mip.y <= 1)
        solver.num_no_good_cuts_added = {9.0: 1}
        solver.deactivate_no_good_cuts_when_fixing_bound(
            solver.mip.MindtPy_utils.cuts.no_good_cuts
        )
        self.assertFalse(solver.mip.MindtPy_utils.cuts.no_good_cuts[2].active)
        self.assertEqual(solver.integer_list, [(1,)])

    def test_goa_single_tree_threads_and_cut_helpers(self):
        solver = global_outer_approximation.MindtPy_GOA_Solver()
        solver.config = make_goa_config(
            single_tree=True, mip_solver='cplex_persistent', threads=4
        )
        with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
            solver.check_config()
        self.assertEqual(solver.config.threads, 1)

        solver = global_outer_approximation.MindtPy_GOA_Solver()
        solver.config = make_goa_config()
        with patch.object(
            algorithm_base_class._MindtPyAlgorithm,
            'initialize_mip_problem',
            side_effect=lambda: setattr(solver, 'mip', make_cut_model()),
        ):
            solver.initialize_mip_problem()
        self.assertIsNotNone(solver.mip.MindtPy_utils.cuts.aff_cuts)

        solver.mip = make_cut_model()
        with patch.object(global_outer_approximation, 'add_affine_cuts') as add_affine:
            solver.add_cuts()
        add_affine.assert_called_once_with(solver.mip, solver.config, solver.timing)

        no_good_cuts = solver.mip.MindtPy_utils.cuts.no_good_cuts
        no_good_cuts.add(expr=solver.mip.y >= 0)
        solver.num_no_good_cuts_added = {}
        solver.primal_bound = 1.0
        solver.config = make_goa_config()
        solver.integer_list = [(1,)]
        solver.deactivate_no_good_cuts_when_fixing_bound(no_good_cuts)
        self.assertTrue(no_good_cuts[1].active)
        self.assertEqual(solver.integer_list, [(1,)])

    def test_outer_approximation_configuration_and_objective_reformulation(self):
        solver = outer_approximation.MindtPy_OA_Solver()
        solver.config = make_config(
            add_regularization='grad_lag',
            regularization_mip_threads=0,
            threads=2,
            mip_solver='glpk',
            heuristic_nonconvex=True,
            init_strategy='FP',
        )
        with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
            solver.check_config()
        self.assertTrue(solver.config.calculate_dual_at_solution)
        self.assertTrue(solver.config.move_objective)
        self.assertEqual(solver.regularization_mip_type, 'MILP')

        solver.config = make_config(
            add_regularization='grad_lag',
            mip_solver='cplex_persistent',
            single_tree=True,
            threads=3,
        )
        with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
            solver.check_config()
        self.assertEqual(solver.config.iteration_limit, 1)
        self.assertEqual(solver.config.threads, 1)

        solver.working_model = ConcreteModel()
        solver.working_model.z = Var()
        solver.working_model.MindtPy_utils = Block()
        solver.working_model.obj0 = Objective(expr=1, sense=minimize)
        solver.working_model.MindtPy_utils.objective_list = [solver.working_model.obj0]
        solver.working_model.MindtPy_utils.objective_constr = Constraint(
            expr=solver.working_model.z >= 0
        )
        solver.working_model.MindtPy_utils.objective = Objective(
            expr=solver.working_model.z, sense=minimize
        )
        solver.config = make_config(add_regularization='grad_lag')
        solver.mip_objective_polynomial_degree = {0, 1}
        with patch.object(solver, 'process_objective'):
            solver.objective_reformulation()
        self.assertTrue(solver.working_model.MindtPy_utils.objective_list[0].active)
        self.assertFalse(solver.working_model.MindtPy_utils.objective_constr.active)
        self.assertFalse(solver.working_model.MindtPy_utils.objective.active)

    def test_outer_approximation_initialize_and_add_cuts(self):
        solver = outer_approximation.MindtPy_OA_Solver()
        solver.config = make_config()
        solver.objective_sense = minimize
        solver.mip_iter = 2
        solver.mip_constraint_polynomial_degree = {0, 1}
        solver.timing = SimpleNamespace()
        solver.mip = make_cut_model()
        solver.jacobians = ComponentMap()
        solver.mip.MindtPy_utils.grey_box_list = [object()]
        with (
            patch.object(
                algorithm_base_class._MindtPyAlgorithm,
                'initialize_mip_problem',
                side_effect=lambda: None,
            ),
            patch.object(outer_approximation, 'calc_jacobians', return_value='jac'),
        ):
            solver.initialize_mip_problem()
        self.assertEqual(solver.jacobians, 'jac')
        self.assertTrue(hasattr(solver.mip.MindtPy_utils.cuts, 'oa_cuts'))

        with (
            patch.object(outer_approximation, 'add_oa_cuts') as add_oa_cuts,
            patch.object(
                outer_approximation, 'add_oa_cuts_for_grey_box'
            ) as add_grey_box,
        ):
            solver.add_cuts([1.0], nlp='grey')
        add_oa_cuts.assert_called_once()
        add_grey_box.assert_called_once()

        solver.config.add_no_good_cuts = True
        solver.config.use_tabu_list = True
        solver.integer_list = [(1,), (0,)]
        solver.mip.MindtPy_utils.cuts.no_good_cuts.add(expr=solver.mip.y >= 0)
        solver.deactivate_no_good_cuts_when_fixing_bound(
            solver.mip.MindtPy_utils.cuts.no_good_cuts
        )
        self.assertFalse(solver.mip.MindtPy_utils.cuts.no_good_cuts[1].active)
        self.assertEqual(solver.integer_list, [(1,)])


class TestCutGeneration(unittest.TestCase):
    def test_add_oa_cuts_covers_equality_and_inequality_paths(self):
        eq_model = make_cut_model(equality=True, upper=2.0)
        eq_config = make_config(
            equality_relaxation=True,
            add_slack=True,
            single_tree=True,
            mip_solver='gurobi_persistent',
        )
        eq_jacs = ComponentMap(
            [(eq_model.c, ComponentMap([(eq_model.x, 2.0), (eq_model.y, 1.0)]))]
        )
        eq_cb_opt = SimpleNamespace(cbLazy=MagicMock())
        cut_generation.add_oa_cuts(
            eq_model,
            [1.0],
            eq_jacs,
            minimize,
            {0, 1},
            1,
            eq_config,
            {},
            cb_opt=eq_cb_opt,
        )
        self.assertEqual(len(eq_model.MindtPy_utils.cuts.oa_cuts), 1)
        eq_cb_opt.cbLazy.assert_called_once()

        ineq_model = make_cut_model(upper=1.5)
        ineq_config = make_config(add_slack=False, linearize_inactive=False)
        ineq_model.x.set_value(1.0)
        ineq_model.y.set_value(1)
        ineq_jacs = ComponentMap(
            [(ineq_model.c, ComponentMap([(ineq_model.x, 2.0), (ineq_model.y, 1.0)]))]
        )
        cut_generation.add_oa_cuts(
            ineq_model,
            [1.0],
            ineq_jacs,
            minimize,
            {0, 1},
            0,
            ineq_config,
            {},
            linearize_violated=True,
        )
        self.assertGreaterEqual(len(ineq_model.MindtPy_utils.cuts.oa_cuts), 1)

    def test_add_oa_cuts_covers_epigraph_lazy_cut_branches(self):
        upper_model = make_cut_model()
        upper_model.MindtPy_utils.objective_constr = Constraint(
            expr=upper_model.x**2 + upper_model.y <= 2.0
        )
        upper_constr = upper_model.MindtPy_utils.objective_constr
        upper_model.MindtPy_utils.constraint_list = [upper_constr]
        upper_model.MindtPy_utils.nonlinear_constraint_list = [upper_constr]
        upper_jacs = ComponentMap(
            [(upper_constr, ComponentMap([(upper_model.x, 2.0), (upper_model.y, 1.0)]))]
        )
        upper_config = make_config(
            add_slack=True, single_tree=True, mip_solver='gurobi_persistent'
        )
        upper_cb = SimpleNamespace(cbLazy=MagicMock())
        cut_generation.add_oa_cuts(
            upper_model,
            [1.0],
            upper_jacs,
            minimize,
            {0, 1},
            1,
            upper_config,
            {},
            cb_opt=upper_cb,
            linearize_active=False,
            linearize_violated=False,
        )
        self.assertEqual(len(upper_model.MindtPy_utils.cuts.oa_cuts), 1)
        self.assertEqual(len(upper_model.MindtPy_utils.cuts.slack_vars), 1)
        upper_cb.cbLazy.assert_called_once()

        lower_model = make_cut_model()
        lower_model.MindtPy_utils.objective_constr = Constraint(
            expr=lower_model.x**2 + lower_model.y >= 2.0
        )
        lower_constr = lower_model.MindtPy_utils.objective_constr
        lower_model.MindtPy_utils.constraint_list = [lower_constr]
        lower_model.MindtPy_utils.nonlinear_constraint_list = [lower_constr]
        lower_jacs = ComponentMap(
            [(lower_constr, ComponentMap([(lower_model.x, 2.0), (lower_model.y, 1.0)]))]
        )
        lower_config = make_config(
            add_slack=True, single_tree=True, mip_solver='gurobi_persistent'
        )
        lower_cb = SimpleNamespace(cbLazy=MagicMock())
        cut_generation.add_oa_cuts(
            lower_model,
            [1.0],
            lower_jacs,
            minimize,
            {0, 1},
            1,
            lower_config,
            {},
            cb_opt=lower_cb,
            linearize_active=False,
            linearize_violated=False,
        )
        self.assertEqual(len(lower_model.MindtPy_utils.cuts.oa_cuts), 1)
        self.assertEqual(len(lower_model.MindtPy_utils.cuts.slack_vars), 1)
        lower_cb.cbLazy.assert_called_once()

    def test_add_oa_cuts_for_grey_box_adds_slack_when_requested(self):
        target_model = make_cut_model()
        target_model.grey = Block()
        target_model.grey.inputs = VarList()
        grey_input = target_model.grey.inputs.add()
        grey_input.set_value(1.0)
        target_model.grey.outputs = VarList()
        grey_output = target_model.grey.outputs.add()
        grey_output.set_value(2.0)
        target_model.MindtPy_utils.grey_box_list = [target_model.grey]

        class FakeJacobianModel:
            def get_external_model(self):
                return SimpleNamespace(
                    evaluate_jacobian_outputs=lambda: SimpleNamespace(
                        toarray=lambda: [[3.0]]
                    )
                )

        jacobian_model = FakeJacobianModel()
        jacobians_model = SimpleNamespace(
            MindtPy_utils=SimpleNamespace(grey_box_list=[jacobian_model]),
            dual={
                jacobian_model: {grey_output.name.replace('outputs', 'output_constraints'): 2.0}
            },
        )

        cut_generation.add_oa_cuts_for_grey_box(
            target_model,
            jacobians_model,
            make_config(add_slack=True),
            minimize,
            0,
        )
        self.assertEqual(len(target_model.MindtPy_utils.cuts.oa_cuts), 1)
        self.assertEqual(len(target_model.MindtPy_utils.cuts.slack_vars), 1)

    def test_add_ecp_cuts_and_no_good_cuts(self):
        model = ConcreteModel()
        model.x = Var(bounds=(0, 2), initialize=1.0)
        model.c1 = Constraint(expr=model.x**2 <= 0.5)
        model.c2 = Constraint(expr=1.5 <= model.x**2)
        model.c3 = Constraint(expr=(0.1, model.x**2, 4.0))
        model.MindtPy_utils = Block()
        model.MindtPy_utils.nonlinear_constraint_list = [model.c1, model.c2, model.c3]
        model.MindtPy_utils.cuts = Block()
        model.MindtPy_utils.cuts.ecp_cuts = ConstraintList()
        model.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
        model.MindtPy_utils.cuts.slack_vars = VarList(domain=Reals)
        config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config, ecp_tolerance=1e-6
        )
        jacobians = ComponentMap(
            [
                (model.c1, ComponentMap([(model.x, 2.0)])),
                (model.c2, ComponentMap([(model.x, 2.0)])),
                (model.c3, ComponentMap([(model.x, 2.0)])),
            ]
        )
        cut_generation.add_ecp_cuts(model, jacobians, config, {})
        self.assertGreaterEqual(len(model.MindtPy_utils.cuts.ecp_cuts), 2)

        nogood_model = make_cut_model()
        nogood_cb = SimpleNamespace(cbLazy=MagicMock())
        nogood_config = make_config(
            add_no_good_cuts=True, single_tree=True, mip_solver='gurobi_persistent'
        )
        cut_generation.add_no_good_cuts(
            nogood_model, [1.0, 1.0], nogood_config, {}, mip_iter=1, cb_opt=nogood_cb
        )
        self.assertEqual(len(nogood_model.MindtPy_utils.cuts.no_good_cuts), 1)
        nogood_cb.cbLazy.assert_called_once()

        with self.assertRaisesRegex(ValueError, 'is not 0 or 1'):
            cut_generation.add_no_good_cuts(nogood_model, [1.0, 0.3], nogood_config, {})

    def test_add_ecp_and_no_good_cuts_handle_skip_conditions(self):
        model = ConcreteModel()
        model.x = Var(bounds=(0, 2), initialize=1.0)
        model.c1 = Constraint(expr=model.x**2 <= 0.5)
        model.c2 = Constraint(expr=1.5 <= model.x**2)
        model.MindtPy_utils = Block()
        model.MindtPy_utils.nonlinear_constraint_list = [model.c1, model.c2]
        model.MindtPy_utils.cuts = Block()
        model.MindtPy_utils.cuts.ecp_cuts = ConstraintList()
        model.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
        model.MindtPy_utils.cuts.slack_vars = VarList(domain=Reals)
        config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config, ecp_tolerance=1e-6
        )
        jacobians = ComponentMap(
            [
                (model.c1, ComponentMap([(model.x, 2.0)])),
                (model.c2, ComponentMap([(model.x, 2.0)])),
            ]
        )
        with (
            patch.object(model.c1, 'uslack', side_effect=OverflowError('bad uslack')),
            patch.object(model.c2, 'lslack', side_effect=ValueError('bad lslack')),
        ):
            cut_generation.add_ecp_cuts(model, jacobians, config, {})
        self.assertEqual(len(model.MindtPy_utils.cuts.ecp_cuts), 0)

        disabled_model = make_cut_model()
        cut_generation.add_no_good_cuts(
            disabled_model, [1.0, 1.0], make_config(add_no_good_cuts=False), {}
        )
        self.assertEqual(len(disabled_model.MindtPy_utils.cuts.no_good_cuts), 0)

        no_binary_model = make_cut_model(include_binary=False)
        cut_generation.add_no_good_cuts(
            no_binary_model, [1.0, 0.25], make_config(add_no_good_cuts=True), {}
        )
        self.assertEqual(len(no_binary_model.MindtPy_utils.cuts.no_good_cuts), 0)

    def test_add_affine_cuts_handles_valid_and_error_cases(self):
        good_model = make_cut_model(include_binary=False, upper=2.0)
        good_config = make_config()

        class FakeMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(good_model.x, 1.5), (good_model.y, 0.5)])

            def subcv(self):
                return ComponentMap([(good_model.x, 1.0), (good_model.y, 0.25)])

            def concave(self):
                return 0.5

            def convex(self):
                return 0.75

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(cut_generation, 'mc', FakeMcCormick):
            cut_generation.add_affine_cuts(good_model, good_config, {})
        self.assertEqual(len(good_model.MindtPy_utils.cuts.aff_cuts), 2)

        error_model = make_cut_model(include_binary=False, upper=2.0)
        with patch.object(
            cut_generation, 'mc', side_effect=cut_generation.MCPP_Error('bad mcpp')
        ):
            cut_generation.add_affine_cuts(error_model, good_config, {})
        self.assertEqual(len(error_model.MindtPy_utils.cuts.aff_cuts), 0)

    def test_add_affine_cuts_skips_constraints_with_missing_values(self):
        missing_value_model = make_cut_model(include_binary=False, upper=2.0)
        missing_value_model.x.set_value(None, skip_validation=True)
        cut_generation.add_affine_cuts(missing_value_model, make_config(), {})
        self.assertEqual(len(missing_value_model.MindtPy_utils.cuts.aff_cuts), 0)

    def test_add_affine_cuts_keeps_valid_side_when_other_slope_is_nan(self):
        nan_model = make_cut_model(include_binary=False, upper=2.0)

        class NaNMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(nan_model.x, float('nan')), (nan_model.y, 0.0)])

            def subcv(self):
                return ComponentMap([(nan_model.x, 1.0), (nan_model.y, 0.0)])

            def concave(self):
                return 0.5

            def convex(self):
                return 0.75

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(cut_generation, 'mc', NaNMcCormick):
            cut_generation.add_affine_cuts(nan_model, make_config(), {})
        self.assertEqual(len(nan_model.MindtPy_utils.cuts.aff_cuts), 1)

        nan_convex_model = make_cut_model(include_binary=False, upper=2.0)

        class NaNConvexMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap(
                    [(nan_convex_model.x, 1.0), (nan_convex_model.y, 0.0)]
                )

            def subcv(self):
                return ComponentMap(
                    [(nan_convex_model.x, float('nan')), (nan_convex_model.y, 0.0)]
                )

            def concave(self):
                return 0.5

            def convex(self):
                return 0.75

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(cut_generation, 'mc', NaNConvexMcCormick):
            cut_generation.add_affine_cuts(nan_convex_model, make_config(), {})
        self.assertEqual(len(nan_convex_model.MindtPy_utils.cuts.aff_cuts), 1)

    def test_add_affine_cuts_skips_nonfinite_or_zero_mccormick_data(self):
        invalid_model = make_cut_model(include_binary=False, upper=2.0)

        class InvalidMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(invalid_model.x, 0.0), (invalid_model.y, 0.0)])

            def subcv(self):
                return ComponentMap([(invalid_model.x, 0.0), (invalid_model.y, 0.0)])

            def concave(self):
                return float('inf')

            def convex(self):
                return float('inf')

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(cut_generation, 'mc', InvalidMcCormick):
            cut_generation.add_affine_cuts(invalid_model, make_config(), {})
        self.assertEqual(len(invalid_model.MindtPy_utils.cuts.aff_cuts), 0)


class TestUtilities(unittest.TestCase):
    def test_solver_option_helpers_and_integer_solution(self):
        config = make_config()
        config.time_limit = 30
        config.mip_solver_args['solver'] = 'ipopt'
        opt = FakeSolver()
        with patch.object(util, 'get_main_elapsed_time', return_value=5.0):
            util.update_solver_timelimit(opt, 'glpk', SimpleNamespace(), config)
        self.assertEqual(opt.options['tmlim'], 25)
        util.set_solver_mipgap(opt, 'glpk', config)
        self.assertEqual(opt.options['mipgap'], config.mip_solver_mipgap)
        util.set_solver_constraint_violation_tolerance(opt, 'ipopt', config)
        self.assertEqual(opt.options['constr_viol_tol'], config.zero_tolerance)

        cyipopt_opt = FakeSolver()
        util.set_solver_constraint_violation_tolerance(cyipopt_opt, 'cyipopt', config)
        self.assertEqual(
            cyipopt_opt.config.options['constr_viol_tol'], config.zero_tolerance
        )

        gams_opt = FakeSolver()
        gams_opt.options['add_options'] = []
        config.nlp_solver_args['solver'] = 'ipopt'
        util.set_solver_constraint_violation_tolerance(gams_opt, 'gams', config)
        self.assertIn('GAMS_MODEL.optfile=1', gams_opt.options['add_options'][0])

        model = make_cut_model()
        model.y.set_value(-0.0)
        self.assertEqual(util.get_integer_solution(model, string_zero=True), ('-0.0',))

    def test_norm_and_epigraph_helpers(self):
        model = make_cut_model()
        setpoint = make_cut_model()
        model.y.set_value(1)
        setpoint.y.set_value(0)
        norm2 = util.generate_norm2sq_objective_function(
            model, setpoint, discrete_only=True
        )
        norm2.construct()
        self.assertEqual(value(norm2.expr), 1)

        norm1 = util.generate_norm1_objective_function(
            model, setpoint, discrete_only=False
        )
        norm1.construct()
        self.assertEqual(len(model.MindtPy_utils.L1_obj.abs_reform), 4)
        self.assertIsNotNone(norm1)

        norm_inf = util.generate_norm_inf_objective_function(model, setpoint)
        norm_inf.construct()
        self.assertEqual(len(model.MindtPy_utils.L_infinity_obj.abs_reform), 4)
        self.assertIsNotNone(norm_inf)

        config = make_config()
        config.fp_norm_constraint_coef = 1.5
        util.generate_norm_constraint(model, setpoint, config)
        self.assertTrue(
            hasattr(model, 'norm_constraint')
            or hasattr(model.MindtPy_utils, 'L1_norm_constraint')
        )

        slack_vars = VarList()
        slack_vars.construct()
        constraints = ConstraintList()
        constraints.construct()
        util.epigraph_reformulation(
            model.x**2, slack_vars, constraints, False, minimize
        )
        self.assertEqual(len(constraints), 1)

    def test_copy_helpers_and_orthogonality(self):
        config = make_config()
        from_model = make_cut_model()
        to_model = make_cut_model()
        from_model.x.set_value(2.0)
        from_model.y.set_value(0.0)
        to_model.y.fix(1)
        util.copy_var_list_values(
            from_model.MindtPy_utils.variable_list,
            to_model.MindtPy_utils.variable_list,
            config,
        )
        self.assertAlmostEqual(to_model.x.value, 2.0)
        self.assertEqual(to_model.y.value, 1)

        solver_model = SimpleNamespace(
            solution=SimpleNamespace(
                pool=SimpleNamespace(
                    get_values=lambda name, var: {'x': 1.5, 'y': 0.0}[var]
                )
            )
        )
        config.mip_solver = 'cplex_persistent'
        util.copy_var_list_values_from_solution_pool(
            from_model.MindtPy_utils.variable_list,
            to_model.MindtPy_utils.variable_list,
            config,
            solver_model,
            ComponentMap([(from_model.x, 'x'), (from_model.y, 'y')]),
            'sol',
        )
        self.assertAlmostEqual(to_model.x.value, 1.5)

        util.add_orthogonality_cuts(from_model, to_model, config)
        self.assertEqual(len(to_model.MindtPy_utils.cuts.fp_orthogonality_cuts), 1)

    def test_results_object_and_feasibility_helpers(self):
        model = make_cut_model()
        results = SolverResults()
        config = make_config(strategy='OA', init_strategy='rNLP')
        util.setup_results_object(results, model, config)
        self.assertEqual(results.problem.name, model.name)
        self.assertEqual(results.solver.name, 'MindtPyOA')

        model.MindtPy_utils.feas_opt = Block()
        model.MindtPy_utils.feas_opt.feas_constraints = ConstraintList()
        model.MindtPy_utils.feas_opt.slack_var = Var(domain=Reals)
        util.initialize_feas_subproblem(model, 'L_infinity')
        self.assertTrue(hasattr(model.MindtPy_utils, 'feas_obj'))

        working_model = make_cut_model()
        mip_model = make_cut_model()
        working_model.y.set_value(1)
        mip_model.y.set_value(1)
        self.assertTrue(util.fp_converged(working_model, mip_model, 0.0))

    def test_lagrangian_regularization_and_misc_helpers(self):
        model = make_cut_model()
        setpoint = make_cut_model()
        setpoint.dual[setpoint.c] = 1.0
        model.MindtPy_utils.continuous_variable_list = [model.x]
        setpoint.MindtPy_utils.continuous_variable_list = [setpoint.x]

        class FakeMatrix:
            def __init__(self, data):
                self._data = util.numpy.array(data, dtype=float)

            def toarray(self):
                return self._data

        class FakePyomoNLP:
            def __init__(self, nlp_model):
                self.vars = list(nlp_model.MindtPy_utils.variable_list)
                self.cons = list(nlp_model.MindtPy_utils.constraint_list)

            def get_pyomo_constraints(self):
                return self.cons

            def set_duals(self, lam):
                self.lam = lam

            def evaluate_grad_objective(self):
                return util.numpy.array([1.0, 2.0])

            def evaluate_jacobian(self):
                return FakeMatrix([[3.0, 4.0]])

            def get_pyomo_variables(self):
                return self.vars

            def get_primal_indices(self, variables):
                target = variables[0]
                return [[i for i, var in enumerate(self.vars) if var is target][0]]

            def evaluate_hessian_lag(self):
                return FakeMatrix([[2.0, 0.0], [0.0, 2.0]])

        config = make_config(add_regularization='grad_lag')
        with (
            patch.object(
                util,
                'TransformationFactory',
                return_value=SimpleNamespace(apply_to=lambda m: None),
            ),
            patch.object(util, 'pyomo_nlp', SimpleNamespace(PyomoNLP=FakePyomoNLP)),
        ):
            grad_obj = util.generate_lag_objective_function(
                model, setpoint, config, {}, discrete_only=False
            )
            grad_obj.construct()
        self.assertIsNotNone(grad_obj)

        for mode in ('hess_lag', 'hess_only_lag'):
            config = make_config(add_regularization=mode)
            with (
                patch.object(
                    util,
                    'TransformationFactory',
                    return_value=SimpleNamespace(apply_to=lambda m: None),
                ),
                patch.object(util, 'pyomo_nlp', SimpleNamespace(PyomoNLP=FakePyomoNLP)),
            ):
                obj = util.generate_lag_objective_function(
                    model, setpoint, config, {}, discrete_only=False
                )
                obj.construct()
            self.assertIsNotNone(obj)

        config = make_config(add_regularization='sqp_lag')
        config.sqp_lag_scaling_coef = 'fixed'
        with (
            patch.object(
                util,
                'TransformationFactory',
                return_value=SimpleNamespace(apply_to=lambda m: None),
            ),
            patch.object(util, 'pyomo_nlp', SimpleNamespace(PyomoNLP=FakePyomoNLP)),
        ):
            sqp_obj = util.generate_lag_objective_function(
                model, setpoint, config, {}, discrete_only=True
            )
            sqp_obj.construct()
        self.assertIsNotNone(sqp_obj)

        config.fp_main_norm = 'L2'
        util.generate_norm_constraint(model, setpoint, config)
        config.fp_main_norm = 'L_infinity'
        util.generate_norm_constraint(model, setpoint, config)
        with self.assertRaisesRegex(ValueError, 'set_var_valid_value failed'):
            util.set_var_valid_value(model.y, 0.4, 1e-9, 1e-9, False)

        gurobi = util.GurobiPersistent4MindtPy()
        gurobi._pyomo_model = model
        gurobi._callback_func = lambda m, opt, where, solver, cfg: ('seen', where)
        gurobi.mindtpy_solver = object()
        gurobi.config = object()
        self.assertTrue(callable(gurobi._intermediate_callback()))


class TestExtendedCuttingPlane(unittest.TestCase):
    def test_ecp_configuration_and_constraint_satisfaction(self):
        solver = extended_cutting_plane.MindtPy_ECP_Solver()
        solver.config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config, ecp_tolerance=None
        )
        with patch.object(algorithm_base_class._MindtPyAlgorithm, 'check_config'):
            solver.check_config()
        self.assertEqual(
            solver.config.ecp_tolerance, solver.config.absolute_bound_tolerance
        )

        solver.mip = make_cut_model(include_binary=False, upper=2.0)
        solver.dual_bound = 1.0
        solver.results = SolverResults()
        solver.config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config, ecp_tolerance=1e-6
        )
        self.assertTrue(solver.all_nonlinear_constraint_satisfied())
        self.assertEqual(solver.primal_bound, solver.dual_bound)
        self.assertIs(solver.results.solver.termination_condition, tc.optimal)

    def test_ecp_iteration_and_error_paths(self):
        solver = extended_cutting_plane.MindtPy_ECP_Solver()
        solver.config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config
        )
        solver.results = SolverResults()
        solver.timing = {}
        solver.config.iteration_limit = 1
        with (
            patch.object(
                solver, 'solve_main', return_value=(make_cut_model(), make_results())
            ),
            patch.object(solver, 'handle_main_mip_termination', return_value=True),
        ):
            solver.MindtPy_iteration_loop()

        solver = extended_cutting_plane.MindtPy_ECP_Solver()
        solver.config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config
        )
        solver.results = SolverResults()
        solver.should_terminate = True
        solver.primal_bound = float('inf')
        solver.primal_bound_progress = [float('inf')]
        self.assertTrue(solver.algorithm_should_terminate())
        self.assertIs(solver.results.solver.termination_condition, tc.noSolution)

        solver.results = SolverResults()
        solver.primal_bound = 0.0
        solver.primal_bound_progress = [float('inf')]
        self.assertTrue(solver.algorithm_should_terminate())
        self.assertIs(solver.results.solver.termination_condition, tc.feasible)

        solver = extended_cutting_plane.MindtPy_ECP_Solver()
        solver.config = make_config(
            factory=extended_cutting_plane._get_MindtPy_ECP_config, ecp_tolerance=1e-6
        )
        solver.results = SolverResults()
        solver.dual_bound = 3.0
        solver.mip = make_cut_model(lower=0.0, upper=2.0)
        with patch.object(
            solver.mip.c, 'lslack', side_effect=ValueError('bad lower slack')
        ):
            self.assertFalse(solver.all_nonlinear_constraint_satisfied())
        with patch.object(
            solver.mip.c, 'uslack', side_effect=ValueError('bad upper slack')
        ):
            self.assertFalse(solver.all_nonlinear_constraint_satisfied())
