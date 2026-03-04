# __________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# __________________________________________________________________________________

"""Integration tests for GOA with local NLP heuristic mode."""

import pyomo.contrib.mcpp.pyomo_mcpp
import pyomo.common.unittest as unittest
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    required_solvers = ('ipopt', 'appsi_highs')
else:
    required_solvers = ('ipopt', 'glpk')

subsolvers_available = all(
    SolverFactory(s).available(exception_flag=False) for s in required_solvers
)


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
@unittest.skipIf(
    not differentiate_available, 'Symbolic differentiation is not available'
)
@unittest.skipIf(
    not pyomo.contrib.mcpp.pyomo_mcpp.mcpp_available(), 'MC++ is not available'
)
class TestMindtPyGOAHeuristic(unittest.TestCase):
    """Integration tests for GOA in local-NLP heuristic mode."""

    def test_goa_local_nlp_heuristic_ipopt(self):
        """Verify GOA runs and returns a valid solution with local NLP heuristic mode."""
        model = SimpleMINLP2().clone()
        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                model,
                strategy='GOA',
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
                local_nlp_heuristic=True,
                add_no_good_cuts=True,
                use_tabu_list=True,
            )

        self.assertIn(
            results.solver.termination_condition,
            [
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.locallyOptimal,
            ],
        )
        self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=1)


if __name__ == '__main__':
    unittest.main()
