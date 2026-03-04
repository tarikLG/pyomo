# __________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# __________________________________________________________________________________

"""Unit tests for GOA-specific MindtPy behavior.

This module validates the GOA strategy hooks implemented in
``pyomo.contrib.mindtpy.global_outer_approximation`` without requiring external
MIP/NLP solver binaries.

Notes
-----
The tests in this module focus on GOA-specific behavior contracts such as
configuration normalization, affine-cut dispatch, and no-good-cut rollback
logic used during final bound correction.
"""

from unittest.mock import patch

import pyomo.common.unittest as unittest
from pyomo.core import Block, ConcreteModel, ConstraintList, Var

from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.global_outer_approximation import MindtPy_GOA_Solver


class TestMindtPyGOAUnit(unittest.TestCase):
    """Unit tests for GOA hooks and configuration behavior.

    Notes
    -----
    These are targeted unit tests and are not intended to verify end-to-end
    global-optimality performance of GOA on benchmark MINLP instances.
    """

    def _make_solver(self):
        """Create a fresh GOA solver with initialized config.

        Returns
        -------
        MindtPy_GOA_Solver
            Solver instance with ``solver.config`` initialized from
            ``solver.CONFIG()``.
        """
        solver = MindtPy_GOA_Solver()
        solver.config = solver.CONFIG()
        return solver

    def _build_no_good_cut_container(self, n_cuts):
        """Create a minimal model with a no-good cut list.

        Parameters
        ----------
        n_cuts : int
            Number of placeholder no-good cuts to add.

        Returns
        -------
        ConcreteModel
            Model containing ``x`` and ``no_good_cuts``.
        """
        model = ConcreteModel()
        model.x = Var(bounds=(0, 1))
        model.no_good_cuts = ConstraintList()
        for _ in range(n_cuts):
            model.no_good_cuts.add(expr=model.x >= 0)
        return model

    def test_check_config_sets_goa_defaults(self):
        """Verify GOA default normalization in non-single-tree mode.

        The test checks that GOA enforces strategy defaults and enables
        no-good cuts when neither no-good nor tabu mechanisms are configured.
        """
        solver = self._make_solver()
        solver.config.add_no_good_cuts = False
        solver.config.use_tabu_list = False
        solver.config.single_tree = False

        with patch.object(_MindtPyAlgorithm, 'check_config', autospec=True) as base:
            solver.check_config()

        self.assertFalse(solver.config.add_slack)
        self.assertTrue(solver.config.use_mcpp)
        self.assertFalse(solver.config.equality_relaxation)
        self.assertTrue(solver.config.use_fbbt)
        self.assertTrue(solver.config.add_no_good_cuts)
        self.assertFalse(solver.config.use_tabu_list)
        base.assert_called_once_with(solver)

    def test_check_config_single_tree_solver_validation(self):
        """Verify single-tree GOA rejects unsupported master solvers."""
        solver = self._make_solver()
        solver.config.single_tree = True
        solver.config.mip_solver = 'glpk'

        with self.assertRaisesRegex(
            ValueError, 'Only cplex_persistent and gurobi_persistent'
        ):
            solver.check_config()

    def test_check_config_single_tree_threads_correction(self):
        """Verify single-tree GOA enforces callback-safe thread settings."""
        solver = self._make_solver()
        solver.config.single_tree = True
        solver.config.mip_solver = 'cplex_persistent'
        solver.config.threads = 4

        with patch.object(_MindtPyAlgorithm, 'check_config', autospec=True) as base:
            solver.check_config()

        self.assertEqual(solver.config.iteration_limit, 1)
        self.assertEqual(solver.config.threads, 1)
        self.assertFalse(solver.config.add_slack)
        base.assert_called_once_with(solver)

    def test_check_config_does_not_force_no_good_when_tabu_enabled(self):
        """Verify GOA does not force no-good cuts when tabu mode is enabled."""
        solver = self._make_solver()
        solver.config.add_no_good_cuts = False
        solver.config.use_tabu_list = True
        solver.config.single_tree = False

        with patch.object(_MindtPyAlgorithm, 'check_config', autospec=True) as base:
            solver.check_config()

        self.assertFalse(solver.config.add_no_good_cuts)
        self.assertTrue(solver.config.use_tabu_list)
        base.assert_called_once_with(solver)

    def test_check_config_local_nlp_heuristic_disables_no_good_and_tabu(self):
        """Verify local NLP heuristic mode disables no-good and tabu exclusions."""
        solver = self._make_solver()
        solver.config.local_nlp_heuristic = True
        solver.config.nlp_solver = 'ipopt'
        solver.config.add_no_good_cuts = True
        solver.config.use_tabu_list = True

        with patch.object(_MindtPyAlgorithm, 'check_config', autospec=True) as base:
            solver.check_config()

        self.assertFalse(solver.config.add_no_good_cuts)
        self.assertFalse(solver.config.use_tabu_list)
        base.assert_called_once_with(solver)

    def test_initialize_mip_problem_adds_affine_cut_list(self):
        """Verify GOA MIP initialization attaches the affine-cut container."""
        solver = self._make_solver()

        def _fake_base_initialize(self):
            self.mip = ConcreteModel()
            self.mip.MindtPy_utils = Block()
            self.mip.MindtPy_utils.cuts = Block()

        with patch.object(
            _MindtPyAlgorithm,
            'initialize_mip_problem',
            autospec=True,
            side_effect=_fake_base_initialize,
        ) as base:
            solver.initialize_mip_problem()

        base.assert_called_once_with(solver)
        self.assertIsInstance(solver.mip.MindtPy_utils.cuts.aff_cuts, ConstraintList)

    def test_add_cuts_calls_affine_cut_generator(self):
        """Verify GOA cut hook dispatches to ``add_affine_cuts``."""
        solver = self._make_solver()
        solver.mip = object()
        solver.timing = object()

        with patch(
            'pyomo.contrib.mindtpy.global_outer_approximation.add_affine_cuts'
        ) as add_aff:
            solver.add_cuts()

        add_aff.assert_called_once_with(solver.mip, solver.config, solver.timing)

    def test_update_primal_bound_records_cut_count(self):
        """Verify incumbent improvement records no-good-cut checkpoint metadata."""
        solver = self._make_solver()
        solver.timing = object()
        solver.primal_bound_progress_time = []
        solver.mip = ConcreteModel()
        solver.mip.MindtPy_utils = Block()
        solver.mip.MindtPy_utils.cuts = Block()
        solver.mip.x = Var(bounds=(0, 1))
        solver.mip.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
        solver.mip.MindtPy_utils.cuts.no_good_cuts.add(expr=solver.mip.x >= 0)
        solver.primal_bound = 3.14

        def _fake_update(_, __):
            solver.primal_bound_improved = True

        with patch.object(
            _MindtPyAlgorithm,
            'update_primal_bound',
            autospec=True,
            side_effect=_fake_update,
        ):
            with patch(
                'pyomo.contrib.mindtpy.global_outer_approximation.get_main_elapsed_time',
                return_value=1.23,
            ):
                solver.update_primal_bound(0.0)

        self.assertEqual(solver.primal_bound_progress_time[-1], 1.23)
        self.assertEqual(solver.num_no_good_cuts_added[solver.primal_bound], 1)

    def test_update_primal_bound_no_record_when_not_improved(self):
        """Verify no checkpoint metadata is added when incumbent is unchanged."""
        solver = self._make_solver()
        solver.timing = object()
        solver.primal_bound_progress_time = []
        solver.primal_bound = 7.0

        def _fake_update(_, __):
            solver.primal_bound_improved = False

        with patch.object(
            _MindtPyAlgorithm,
            'update_primal_bound',
            autospec=True,
            side_effect=_fake_update,
        ):
            with patch(
                'pyomo.contrib.mindtpy.global_outer_approximation.get_main_elapsed_time',
                return_value=2.34,
            ):
                solver.update_primal_bound(0.0)

        self.assertEqual(solver.primal_bound_progress_time[-1], 2.34)
        self.assertEqual(solver.num_no_good_cuts_added, {})

    def test_deactivate_no_good_cuts_when_fixing_bound(self):
        """Verify GOA deactivates post-incumbent cuts during bound fixing."""
        solver = self._make_solver()
        solver.config.add_no_good_cuts = True
        solver.config.use_tabu_list = True
        solver.primal_bound = 42.0
        solver.num_no_good_cuts_added = {42.0: 1}
        solver.integer_list = [(1,), (0,), (1,)]

        m = self._build_no_good_cut_container(3)

        solver.deactivate_no_good_cuts_when_fixing_bound(m.no_good_cuts)

        self.assertTrue(m.no_good_cuts[1].active)
        self.assertFalse(m.no_good_cuts[2].active)
        self.assertFalse(m.no_good_cuts[3].active)
        self.assertEqual(solver.integer_list, [(1,)])

    def test_deactivate_no_good_cuts_without_flags(self):
        """Verify rollback routine is a no-op when no-good and tabu are disabled."""
        solver = self._make_solver()
        solver.config.add_no_good_cuts = False
        solver.config.use_tabu_list = False
        solver.primal_bound = 1.0
        solver.num_no_good_cuts_added = {1.0: 1}
        solver.integer_list = [(1,), (0,)]

        m = self._build_no_good_cut_container(2)

        solver.deactivate_no_good_cuts_when_fixing_bound(m.no_good_cuts)

        self.assertTrue(m.no_good_cuts[1].active)
        self.assertTrue(m.no_good_cuts[2].active)
        self.assertEqual(solver.integer_list, [(1,), (0,)])

    def test_deactivate_no_good_cuts_key_error_path(self):
        """Verify graceful logging behavior when cut-count checkpoint is missing."""
        solver = self._make_solver()
        solver.config.add_no_good_cuts = True
        solver.config.use_tabu_list = True
        solver.primal_bound = 999.0
        solver.num_no_good_cuts_added = {}

        m = self._build_no_good_cut_container(1)

        with patch.object(solver.config.logger, 'error') as err:
            solver.deactivate_no_good_cuts_when_fixing_bound(m.no_good_cuts)

        self.assertGreaterEqual(err.call_count, 1)
        first_call = err.call_args_list[0]
        self.assertIsInstance(first_call.args[0], KeyError)
        self.assertTrue(first_call.kwargs.get('exc_info', False))
        self.assertEqual(
            err.call_args_list[-1].args[0], 'Deactivating no-good cuts failed.'
        )


if __name__ == '__main__':
    unittest.main()
