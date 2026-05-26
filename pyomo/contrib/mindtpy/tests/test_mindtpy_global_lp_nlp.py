# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Tests for global LP/NLP in the MindtPy solver."""

import os
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.mcpp import pyomo_mcpp

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True

run_full_goa_sweep = os.environ.get('PYOMO_MINDTPY_FULL_GOA_SWEEP') == '1'

representative_goa_model = Nonconvex1()
goa_sweep_models = [
    EightProcessFlowsheet(convex=False),
    Nonconvex1(),
    Nonconvex2(),
    Nonconvex3(),
    Nonconvex4(),
]


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
@unittest.skipIf(not pyomo_mcpp.mcpp_available(), 'MC++ is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_GOA(self):
        """Test the global outer approximation decomposition algorithm."""
        model = representative_goa_model.clone()
        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                model,
                strategy='GOA',
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
                single_tree=True,
            )

        self.assertIn(
            results.solver.termination_condition,
            [TerminationCondition.optimal, TerminationCondition.feasible],
        )
        self.assertAlmostEqual(
            value(model.objective.expr), model.optimal_value, places=2
        )
        self.check_optimal_solution(model)


@unittest.skipIf(
    not (subsolvers_available and pyomo_mcpp.mcpp_available() and run_full_goa_sweep),
    'Set PYOMO_MINDTPY_FULL_GOA_SWEEP=1 to run the full GOA single-tree sweep',
)
@unittest.pytest.mark.expensive
class TestMindtPySingleTreeSweep(unittest.TestCase):
    """Optional broad single-tree GOA sweep for nightly or solver-focused testing."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_GOA_single_tree_sweep(self):
        """Run single-tree GOA across all nonconvex regression models."""
        with SolverFactory('mindtpy') as opt:
            for model in goa_sweep_models:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='GOA',
                    single_tree=True,
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

    @unittest.skipUnless(
        SolverFactory('gurobi_persistent').available(exception_flag=False)
        and SolverFactory('gurobi_direct').available(),
        'gurobi_persistent and gurobi_direct solver is not available',
    )
    def test_GOA_Gurobi(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in goa_sweep_models:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='GOA',
                    mip_solver='gurobi_persistent',
                    nlp_solver=required_solvers[0],
                    single_tree=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)


if __name__ == '__main__':
    unittest.main()
