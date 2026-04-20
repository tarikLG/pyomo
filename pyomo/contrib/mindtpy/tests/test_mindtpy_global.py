# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""

import math
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.opt import TerminationCondition

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True

mcpp_available = pyomo.contrib.mcpp.pyomo_mcpp.mcpp_available()
goa_test_requirements_available = subsolvers_available and mcpp_available

model_list = [
    EightProcessFlowsheet(convex=False),
    Nonconvex1(),
    Nonconvex2(),
    Nonconvex3(),
    Nonconvex4(),
]


@unittest.skipIf(
    not goa_test_requirements_available,
    'GOA integration test requirements are not available',
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_GOA(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

    def test_GOA_tabu_list(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    use_tabu_list=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

    def test_GOA_check_config_enforce_no_good_cuts(self):
        """Test the global outer approximation check config enforces add_no_good_cuts correctly when it and use_tabu_list are False."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                opt.solve(model, strategy='GOA', mip_solver=required_solvers[1], nlp_solver=required_solvers[0], add_no_good_cuts=False, use_tabu_list=False)
                self.assertIn(model.MindtPy_utils.results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertTrue(model.MindtPy_utils.config.add_no_good_cuts)
                self.assertFalse(model.MindtPy_utils.config.use_tabu_list)
                self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=2)
                self.check_optimal_solution(model)

    def test_GOA_deactivate_no_good_cuts_when_fixing_bound(self):
        """Test deactivate_no_good_cuts_when_fixing_bound on a model by invoking the method logic to verify cut enhancements."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    use_tabu_list=True,
                )
                self.assertIn(
                    model.MindtPy_utils.results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                no_good_cuts = model.MindtPy_utils.cuts.no_good_cuts
                self.assertGreater(
                    len(no_good_cuts),
                    0,
                    'Expected GOA run to generate no-good cuts before deactivation checks.',
                )

                for i in no_good_cuts:
                    no_good_cuts[i].activate()

                # Keep only the first no-good cut as valid for this bound.
                valid_no_good_cuts_num = 1
                opt.primal_bound = 0
                opt.num_no_good_cuts_added = {opt.primal_bound: valid_no_good_cuts_num}
                opt.config.add_no_good_cuts = True
                opt.config.use_tabu_list = True
                opt.integer_list = list(range(len(no_good_cuts)))

                opt.deactivate_no_good_cuts_when_fixing_bound(no_good_cuts)

                for i in no_good_cuts:
                    if i <= valid_no_good_cuts_num:
                        self.assertTrue(no_good_cuts[i].active)
                    else:
                        self.assertFalse(no_good_cuts[i].active)
                self.assertEqual(len(opt.integer_list), valid_no_good_cuts_num)

                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

    def test_GOA_initialize_mip_problem_affine_cut_structure(self):
        """Assert GOA initializes affine-cut structure using one normal solve per model."""
        for model in model_list:
            model = model.clone()
            with SolverFactory('mindtpy') as opt:
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )
                mip_constraint_polynomial_degree = opt.mip_constraint_polynomial_degree

            self.assertIn(
                results.solver.termination_condition,
                [TerminationCondition.optimal, TerminationCondition.feasible],
            )
            self.assertIsInstance(model.MindtPy_utils.cuts.aff_cuts, ConstraintList)

            cuts = model.MindtPy_utils.cuts.aff_cuts
            cut_count = len(cuts)
            self.assertGreater(cut_count, 0)

            nonlinear_con_count = len(model.MindtPy_utils.nonlinear_constraint_list)
            # add_affine_cuts adds at most one concave and one convex cut per nonlinear constraint.
            self.assertLessEqual(cut_count, 2 * nonlinear_con_count)

            for i in cuts:
                cut = cuts[i]
                self.assertTrue(cut.active)
                self.assertIn(cut.body.polynomial_degree(), mip_constraint_polynomial_degree)
                # Each affine cut should be a one-sided inequality (either <= or >=).
                self.assertTrue(cut.has_ub() or cut.has_lb())
                self.assertFalse(cut.has_ub() and cut.has_lb())

            self.assertAlmostEqual(
                value(model.objective.expr), model.optimal_value, places=2
            )
            self.check_optimal_solution(model)

    def test_GOA_update_primal_bound_progress_tracking(self):
        """Assert GOA records primal-bound progress and consistent timing traces."""
        for model in model_list:
            model = model.clone()
            with SolverFactory('mindtpy') as opt:
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

            self.assertIn(
                results.solver.termination_condition,
                [TerminationCondition.optimal, TerminationCondition.feasible],
            )

            primal_progress = opt.primal_bound_progress
            primal_progress_time = opt.primal_bound_progress_time
            self.assertGreaterEqual(len(primal_progress), 2)
            self.assertGreaterEqual(len(primal_progress_time), len(primal_progress))
            self.assertEqual(opt.primal_bound, primal_progress[-1])

            for i in range(1, len(primal_progress_time)):
                self.assertGreaterEqual(primal_progress_time[i], primal_progress_time[i - 1])

            if opt.objective_sense == minimize:
                for i in range(1, len(primal_progress)):
                    self.assertLessEqual(primal_progress[i], primal_progress[i - 1])
            else:
                for i in range(1, len(primal_progress)):
                    self.assertGreaterEqual(primal_progress[i], primal_progress[i - 1])

            self.assertFalse(math.isnan(primal_progress[-1]))
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=2)
            self.check_optimal_solution(model)

    def test_GOA_update_primal_bound_no_improvement_behavior(self):
        """Assert GOA no-improvement update does not alter no-good-cut bookkeeping."""
        for model in model_list:
            model = model.clone()
            with SolverFactory('mindtpy') as opt:
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

            self.assertIn(
                results.solver.termination_condition,
                [TerminationCondition.optimal, TerminationCondition.feasible],
            )

            old_primal_bound = opt.primal_bound
            old_no_good_cut_map = dict(opt.num_no_good_cuts_added)
            old_progress_len = len(opt.primal_bound_progress)
            old_time_len = len(opt.primal_bound_progress_time)

            # Re-applying the same primal bound must not be marked as an improvement.
            opt.update_primal_bound(old_primal_bound)

            self.assertFalse(opt.primal_bound_improved)
            self.assertEqual(opt.primal_bound, old_primal_bound)
            self.assertEqual(opt.num_no_good_cuts_added, old_no_good_cut_map)
            self.assertEqual(len(opt.primal_bound_progress), old_progress_len + 1)
            # Base update appends one timestamp and GOA appends one additional timestamp.
            self.assertEqual(len(opt.primal_bound_progress_time), old_time_len + 2)

            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=2)
            self.check_optimal_solution(model)

    def test_GOA_deactivate_no_good_cuts_keyerror_path(self):
        """Assert missing primal-bound key is handled without mutating cut/list state."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    use_tabu_list=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )

                no_good_cuts = model.MindtPy_utils.cuts.no_good_cuts
                self.assertGreater(
                    len(no_good_cuts),
                    0,
                    'Expected GOA run to generate no-good cuts before KeyError-path check.',
                )

                for i in no_good_cuts:
                    no_good_cuts[i].activate()
                active_state_before = {i: no_good_cuts[i].active for i in no_good_cuts}

                opt.config.add_no_good_cuts = True
                opt.config.use_tabu_list = True
                opt.integer_list = list(range(len(no_good_cuts)))
                integer_list_before = list(opt.integer_list)

                # Force dictionary miss and exercise the KeyError handling branch.
                opt.primal_bound = '__missing_primal_bound_key__'
                self.assertNotIn(opt.primal_bound, opt.num_no_good_cuts_added)

                # Should not raise: method catches KeyError and logs it.
                opt.deactivate_no_good_cuts_when_fixing_bound(no_good_cuts)

                for i in no_good_cuts:
                    self.assertEqual(no_good_cuts[i].active, active_state_before[i])
                self.assertEqual(opt.integer_list, integer_list_before)

                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

if __name__ == '__main__':
    unittest.main()
