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

import os
import pyomo.common.unittest as unittest
from pyomo.contrib.mcpp import pyomo_mcpp
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import Binary, ConcreteModel, Constraint, Objective
from pyomo.environ import SolverFactory, Var, value
from pyomo.opt import TerminationCondition

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True

mcpp_available = pyomo_mcpp.mcpp_available()
goa_test_requirements_available = subsolvers_available and mcpp_available
baron_demo_goa_test_requirements_available = (
    SolverFactory('baron').available(exception_flag=False)
    and SolverFactory('cplex_persistent').available(exception_flag=False)
    and mcpp_available
)
run_full_goa_sweep = os.environ.get('PYOMO_MINDTPY_FULL_GOA_SWEEP') == '1'

representative_goa_model = Nonconvex1()
goa_sweep_models = [
    EightProcessFlowsheet(convex=False),
    Nonconvex1(),
    Nonconvex2(),
    Nonconvex3(),
    Nonconvex4(),
]


def make_baron_demo_size_model():
    model = ConcreteModel()
    model.x = Var(bounds=(0, 2), initialize=0.5)
    model.y = Var(domain=Binary, initialize=0)
    model.objective = Objective(expr=(model.x - 1) ** 2 + model.y)
    model.c = Constraint(expr=model.x >= 0.25 + 0.5 * model.y)
    return model


@unittest.skipIf(
    not baron_demo_goa_test_requirements_available,
    'BARON demo GOA integration test requirements are not available',
)
class TestMindtPyFreeBaron(unittest.TestCase):
    """Small GOA smoke test that fits within BARON's free solver limits."""

    def test_GOA_with_baron_free_size_model(self):
        model = make_baron_demo_size_model()

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                model,
                strategy='GOA',
                mip_solver='cplex_persistent',
                nlp_solver='baron',
                allow_baron_demo_license=True,
            )

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertAlmostEqual(value(model.x), 1.0, places=5)
        self.assertAlmostEqual(value(model.y), 0.0, places=5)
        self.assertAlmostEqual(value(model.objective.expr), 0.0, places=5)


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
        model = representative_goa_model.clone()
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
        self.assertAlmostEqual(
            value(model.objective.expr), model.optimal_value, places=2
        )
        self.check_optimal_solution(model)

    def test_GOA_tabu_list(self):
        """Test the global outer approximation decomposition algorithm."""
        model = representative_goa_model.clone()
        with SolverFactory('mindtpy.goa') as opt:
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
        self.assertTrue(model.MindtPy_utils.config.use_tabu_list)
        self.assertAlmostEqual(
            value(model.objective.expr), model.optimal_value, places=2
        )
        self.check_optimal_solution(model)


@unittest.skipIf(
    not (goa_test_requirements_available and run_full_goa_sweep),
    'Set PYOMO_MINDTPY_FULL_GOA_SWEEP=1 to run the full GOA model sweep',
)
@unittest.pytest.mark.expensive
class TestMindtPyGOASweep(unittest.TestCase):
    """Optional broad GOA sweep for nightly or solver-focused testing."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_GOA_sweep(self):
        """Run GOA across all nonconvex regression models."""
        with SolverFactory('mindtpy') as opt:
            for model in goa_sweep_models:
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


if __name__ == '__main__':
    unittest.main()
