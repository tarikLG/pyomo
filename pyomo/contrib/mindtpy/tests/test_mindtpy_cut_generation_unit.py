# __________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# __________________________________________________________________________________

"""Unit tests for MindtPy cut-generation helpers.

Notes
-----
These tests are solver-independent and target branch behavior in
``pyomo.contrib.mindtpy.cut_generation`` using lightweight Pyomo objects and
mocked external dependencies.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.core import (
    Block,
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Var,
    VarList,
    value,
    minimize,
    maximize,
)

from pyomo.contrib.mindtpy.cut_generation import (
    MCPP_Error,
    add_affine_cuts,
    add_ecp_cuts,
    add_no_good_cuts,
    add_oa_cuts,
    add_oa_cuts_for_grey_box,
)


class _CbOpt:
    """Simple callback collector used for lazy-cut path checks."""

    def __init__(self):
        self.calls = 0

    def cbLazy(self, _):
        self.calls += 1


class TestCutGenerationUnit(unittest.TestCase):
    """Unit tests for OA/ECP/no-good/affine cut-generation functions."""

    def _cfg(self, **kw):
        """Build a minimal config object for cut-generation routines.

        Parameters
        ----------
        **kw : dict
            Overrides for default config attributes.

        Returns
        -------
        types.SimpleNamespace
            Config-like object with attributes expected by cut-generation
            helpers.
        """
        data = dict(
            equality_relaxation=False,
            add_slack=False,
            single_tree=False,
            mip_solver='appsi_highs',
            zero_tolerance=1e-7,
            linearize_inactive=False,
            ecp_tolerance=1e-6,
            integer_tolerance=1e-5,
            add_no_good_cuts=True,
            logger=SimpleNamespace(debug=Mock(), warning=Mock(), error=Mock()),
        )
        data.update(kw)
        return SimpleNamespace(**data)

    def _base_model_with_cut_blocks(self):
        """Create a minimal model containing MindtPy cut containers."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=0.25)
        m.y = Var(bounds=(-1, 2), initialize=0.5)

        m.MindtPy_utils = Block()
        m.MindtPy_utils.cuts = Block()
        m.MindtPy_utils.cuts.oa_cuts = ConstraintList()
        m.MindtPy_utils.cuts.ecp_cuts = ConstraintList()
        m.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
        m.MindtPy_utils.cuts.aff_cuts = ConstraintList()
        m.MindtPy_utils.cuts.slack_vars = VarList(initialize=0)
        return m

    def test_add_oa_cuts_equality_and_lazy_callback(self):
        """Covers OA equality-relaxation branch and lazy-cut callback."""
        m = self._base_model_with_cut_blocks()
        m.c_eq = Constraint(expr=m.x * m.x == 0.0625)
        m.MindtPy_utils.constraint_list = [m.c_eq]

        row = ComponentMap()
        row[m.x] = 1.0
        jacs = ComponentMap()
        jacs[m.c_eq] = row
        cfg = self._cfg(
            equality_relaxation=True,
            add_slack=True,
            single_tree=True,
            mip_solver='gurobi_persistent',
        )
        cb = _CbOpt()

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts(
                m,
                dual_values=[1.0],
                jacobians=jacs,
                objective_sense=minimize,
                mip_constraint_polynomial_degree={0, 1},
                mip_iter=1,
                config=cfg,
                timing=None,
                cb_opt=cb,
            )

        self.assertEqual(len(m.MindtPy_utils.cuts.oa_cuts), 1)
        self.assertEqual(cb.calls, 1)

    def test_add_oa_cuts_linear_skip_and_equality_no_slack(self):
        """Covers linear-degree skip and equality OA without slack/callback."""
        m = self._base_model_with_cut_blocks()
        m.c_lin = Constraint(expr=m.x + m.y <= 1.0)
        m.c_eq = Constraint(expr=m.x * m.x == 0.0625)
        m.MindtPy_utils.constraint_list = [m.c_lin, m.c_eq]

        row = ComponentMap()
        row[m.x] = 1.0
        jacs = ComponentMap()
        jacs[m.c_eq] = row

        cfg = self._cfg(equality_relaxation=True, add_slack=False)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts(
                m,
                dual_values=[1.0, -1.0],
                jacobians=jacs,
                objective_sense=minimize,
                mip_constraint_polynomial_degree={1},
                mip_iter=0,
                config=cfg,
                timing=None,
            )

        self.assertEqual(len(m.MindtPy_utils.cuts.oa_cuts), 1)

    def test_add_oa_cuts_inequality_slack_and_callbacks(self):
        """Covers inequality OA upper/lower callback paths with slack vars."""
        m = self._base_model_with_cut_blocks()
        m.c_up = Constraint(expr=(-1.0, m.y * m.y, 0.25))
        m.c_low = Constraint(expr=(0.4, m.y * m.y, 1.0))
        m.MindtPy_utils.constraint_list = [m.c_up, m.c_low]
        m.y.set_value(0.5)

        row_up = ComponentMap()
        row_up[m.y] = 1.0
        row_low = ComponentMap()
        row_low[m.y] = 1.0
        jacs = ComponentMap()
        jacs[m.c_up] = row_up
        jacs[m.c_low] = row_low

        cfg = self._cfg(
            add_slack=True,
            single_tree=True,
            mip_solver='gurobi_persistent',
        )
        cb = _CbOpt()
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts(
                m,
                dual_values=[1.0, 1.0],
                jacobians=jacs,
                objective_sense=minimize,
                mip_constraint_polynomial_degree={0, 1},
                mip_iter=1,
                config=cfg,
                timing=None,
                cb_opt=cb,
            )

        self.assertEqual(len(m.MindtPy_utils.cuts.oa_cuts), 2)
        self.assertEqual(cb.calls, 2)

    def test_add_oa_cuts_inequality_paths(self):
        """Covers OA inequality upper/lower and objective-constraint naming path."""
        m = self._base_model_with_cut_blocks()
        m.c_in = Constraint(expr=(0.1, m.y * m.y, 0.3))
        m.MindtPy_utils.objective_constr = Constraint(expr=(-1.0, m.y * m.y, 1.0))
        m.MindtPy_utils.constraint_list = [m.c_in, m.MindtPy_utils.objective_constr]
        m.y.set_value(0.5)

        row_in = ComponentMap()
        row_in[m.y] = 1.0
        row_obj = ComponentMap()
        row_obj[m.y] = 1.0
        jacs = ComponentMap()
        jacs[m.c_in] = row_in
        jacs[m.MindtPy_utils.objective_constr] = row_obj
        cfg = self._cfg(add_slack=False, linearize_inactive=True)

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts(
                m,
                dual_values=[1.0, 1.0],
                jacobians=jacs,
                objective_sense=maximize,
                mip_constraint_polynomial_degree={0, 1},
                mip_iter=0,
                config=cfg,
                timing=None,
            )

        self.assertGreaterEqual(len(m.MindtPy_utils.cuts.oa_cuts), 2)

    def test_add_oa_cuts_for_grey_box(self):
        """Covers grey-box OA cut generation loop."""

        class _Jac:
            def toarray(self):
                return [[2.0, -1.0]]

        class _ExternalModel:
            def evaluate_jacobian_outputs(self):
                return _Jac()

        class _GB:
            def __init__(self):
                blk = ConcreteModel()
                blk.inputs = Var(['i1', 'i2'])
                blk.outputs = Var(['o1'])
                blk.inputs['i1'].set_value(1.0)
                blk.inputs['i2'].set_value(2.0)
                blk.outputs['o1'].set_value(3.0)
                self.inputs = blk.inputs
                self.outputs = blk.outputs

            def get_external_model(self):
                return _ExternalModel()

        m = self._base_model_with_cut_blocks()
        m.MindtPy_utils.grey_box_list = [_GB()]

        jac_model = self._base_model_with_cut_blocks()
        jac_model.MindtPy_utils.grey_box_list = [_GB()]
        jac_model.dual = ComponentMap(
            {
                jac_model.MindtPy_utils.grey_box_list[0]: {
                    'output_constraints[o1]': 1.0
                }
            }
        )

        cfg = self._cfg(add_slack=True)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts_for_grey_box(
                m, jac_model, cfg, objective_sense=minimize, mip_iter=1, cb_opt=_CbOpt()
            )

        self.assertEqual(len(m.MindtPy_utils.cuts.oa_cuts), 1)

    def test_add_oa_cuts_for_grey_box_without_slack(self):
        """Covers grey-box OA branch when slack variables are disabled."""

        class _Jac:
            def toarray(self):
                return [[1.0, 1.0]]

        class _ExternalModel:
            def evaluate_jacobian_outputs(self):
                return _Jac()

        class _GB:
            def __init__(self):
                blk = ConcreteModel()
                blk.inputs = Var(['i1', 'i2'])
                blk.outputs = Var(['o1'])
                blk.inputs['i1'].set_value(0.0)
                blk.inputs['i2'].set_value(0.0)
                blk.outputs['o1'].set_value(0.0)
                self.inputs = blk.inputs
                self.outputs = blk.outputs

            def get_external_model(self):
                return _ExternalModel()

        m = self._base_model_with_cut_blocks()
        m.MindtPy_utils.grey_box_list = [_GB()]
        jac_model = self._base_model_with_cut_blocks()
        jac_model.MindtPy_utils.grey_box_list = [_GB()]
        jac_model.dual = ComponentMap(
            {
                jac_model.MindtPy_utils.grey_box_list[0]: {
                    'output_constraints[o1]': 1.0
                }
            }
        )

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_oa_cuts_for_grey_box(
                m,
                jac_model,
                self._cfg(add_slack=False),
                objective_sense=maximize,
                mip_iter=0,
                cb_opt=None,
            )

        self.assertEqual(len(m.MindtPy_utils.cuts.oa_cuts), 1)

    def test_add_ecp_cuts_warning_and_error_paths(self):
        """Covers ECP warning and exception-handling branches."""
        m = self._base_model_with_cut_blocks()
        m.c_both = Constraint(expr=(0.0, m.y * m.y, 1.0))
        m.c_up = Constraint(expr=m.y * m.y <= 0.2)
        m.c_low = Constraint(expr=m.y * m.y >= 0.1)
        m.MindtPy_utils.nonlinear_constraint_list = [m.c_both, m.c_up, m.c_low]
        m.y.set_value(0.3)
        row_both = ComponentMap()
        row_both[m.y] = 1.0
        row_up = ComponentMap()
        row_up[m.y] = 1.0
        row_low = ComponentMap()
        row_low[m.y] = 1.0
        jacs = ComponentMap()
        jacs[m.c_both] = row_both
        jacs[m.c_up] = row_up
        jacs[m.c_low] = row_low
        cfg = self._cfg(add_slack=True, ecp_tolerance=1.0)

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_ecp_cuts(m, jacs, cfg, timing=None)

        self.assertGreaterEqual(cfg.logger.warning.call_count, 1)
        self.assertGreaterEqual(len(m.MindtPy_utils.cuts.ecp_cuts), 1)

    def test_add_ecp_cuts_no_cut_and_no_slack_branches(self):
        """Covers ECP no-cut conditions and cut generation without slack vars."""
        m = self._base_model_with_cut_blocks()
        m.c_up_nocut = Constraint(expr=m.y * m.y <= 2.0)
        m.c_low_nocut = Constraint(expr=m.y * m.y >= -1.0)
        m.c_up_cut = Constraint(expr=m.y * m.y <= 0.25)
        m.c_low_cut = Constraint(expr=m.y * m.y >= 0.4)
        m.MindtPy_utils.nonlinear_constraint_list = [
            m.c_up_nocut,
            m.c_low_nocut,
            m.c_up_cut,
            m.c_low_cut,
        ]
        m.y.set_value(0.5)

        row = ComponentMap()
        row[m.y] = 1.0
        jacs = ComponentMap()
        jacs[m.c_up_nocut] = row
        jacs[m.c_low_nocut] = row
        jacs[m.c_up_cut] = row
        jacs[m.c_low_cut] = row

        cfg = self._cfg(add_slack=False, linearize_inactive=False, ecp_tolerance=1e-8)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_ecp_cuts(m, jacs, cfg, timing=None)

        self.assertEqual(len(m.MindtPy_utils.cuts.ecp_cuts), 2)
        self.assertEqual(len(m.MindtPy_utils.cuts.slack_vars), 0)

    def test_add_ecp_cuts_overflow_exception_branch(self):
        """Covers ECP overflow/ValueError path in slack evaluation."""
        m = self._base_model_with_cut_blocks()

        class _FakeConstraint:
            body = SimpleNamespace()

            def has_lb(self):
                return False

            def has_ub(self):
                return True

            def uslack(self):
                raise OverflowError('boom')

        fake_c = _FakeConstraint()
        m.MindtPy_utils.nonlinear_constraint_list = [fake_c]
        cfg = self._cfg()
        with patch('pyomo.contrib.mindtpy.cut_generation.EXPR.identify_variables', return_value=[]):
            with patch(
                'pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()
            ):
                jacs = ComponentMap()
                jacs[fake_c] = ComponentMap()
                add_ecp_cuts(m, jacs, cfg, timing=None)

        self.assertGreaterEqual(cfg.logger.error.call_count, 1)

    def test_add_ecp_cuts_lower_exception_branch(self):
        """Covers ECP lower-slack ValueError handling."""
        m = self._base_model_with_cut_blocks()

        class _FakeConstraint:
            body = SimpleNamespace()

            def has_lb(self):
                return True

            def has_ub(self):
                return False

            def lslack(self):
                raise ValueError('bad lower')

        fake_c = _FakeConstraint()
        m.MindtPy_utils.nonlinear_constraint_list = [fake_c]
        cfg = self._cfg()
        with patch('pyomo.contrib.mindtpy.cut_generation.EXPR.identify_variables', return_value=[]):
            with patch(
                'pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()
            ):
                jacs = ComponentMap()
                jacs[fake_c] = ComponentMap()
                add_ecp_cuts(m, jacs, cfg, timing=None)

        self.assertGreaterEqual(cfg.logger.error.call_count, 1)

    def test_add_no_good_cuts_paths(self):
        """Covers no-good return, error, normal, and callback branches."""
        m = ConcreteModel()
        m.b1 = Var(within=Binary, initialize=1)
        m.b2 = Var(within=Binary, initialize=0)
        m.c1 = Constraint(expr=m.b1 + m.b2 >= 0)
        m.MindtPy_utils = Block()
        m.MindtPy_utils.cuts = Block()
        m.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
        m.MindtPy_utils.variable_list = [m.b1, m.b2]

        cfg = self._cfg(add_no_good_cuts=False)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_no_good_cuts(m, [1, 0], cfg, timing=None)
        self.assertEqual(len(m.MindtPy_utils.cuts.no_good_cuts), 0)

        cfg = self._cfg(add_no_good_cuts=True)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, 'is not 0 or 1'):
                add_no_good_cuts(m, [0.5, 0], cfg, timing=None)

        cfg = self._cfg(add_no_good_cuts=True, single_tree=False)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_no_good_cuts(m, [1, 0], cfg, timing=None)
        self.assertEqual(len(m.MindtPy_utils.cuts.no_good_cuts), 1)

        cb = _CbOpt()
        cfg = self._cfg(add_no_good_cuts=True, single_tree=True, mip_solver='gurobi_persistent')
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_no_good_cuts(m, [1, 0], cfg, timing=None, mip_iter=1, cb_opt=cb)
        self.assertEqual(len(m.MindtPy_utils.cuts.no_good_cuts), 2)
        self.assertEqual(cb.calls, 1)

    def test_add_no_good_cuts_no_binary_branch(self):
        """Covers no-good path where model has no binary variables."""
        m = self._base_model_with_cut_blocks()
        m.MindtPy_utils.variable_list = [m.x, m.y]
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            add_no_good_cuts(m, [0.2, 0.3], self._cfg(add_no_good_cuts=True), timing=None)
        self.assertEqual(len(m.MindtPy_utils.cuts.no_good_cuts), 0)

    def test_add_affine_cuts_branches(self):
        """Covers affine-cut skip, error, invalid, and valid branches."""

        class _FakeMc:
            def __init__(self, cc, cv, c0, v0, lb, ub):
                self._cc = cc
                self._cv = cv
                self._c0 = c0
                self._v0 = v0
                self._lb = lb
                self._ub = ub

            def subcc(self):
                return self._cc

            def subcv(self):
                return self._cv

            def concave(self):
                return self._c0

            def convex(self):
                return self._v0

            def lower(self):
                return self._lb

            def upper(self):
                return self._ub

        m = self._base_model_with_cut_blocks()
        m.x.set_value(0.25)
        m.y.set_value(0.5)
        m.z = Var(initialize=0.0)

        m.c_none = Constraint(expr=m.z * m.z <= 1.0)
        m.c_err = Constraint(expr=m.x * m.y <= 1.0)
        m.c_bad = Constraint(expr=m.x * m.y <= 1.0)
        m.c_ok = Constraint(expr=(0.0, m.x * m.y, 2.0))
        m.MindtPy_utils.nonlinear_constraint_list = [m.c_none, m.c_err, m.c_bad, m.c_ok]

        cfg = self._cfg()

        def _mc_side_effect(expr):
            if expr is m.c_err.body:
                raise MCPP_Error('bad')
            if expr is m.c_bad.body:
                cc = ComponentMap()
                cc[m.x] = 0.0
                cc[m.y] = 0.0
                cv = ComponentMap()
                cv[m.x] = 0.0
                cv[m.y] = 0.0
                return _FakeMc(
                    cc=cc,
                    cv=cv,
                    c0=float('nan'),
                    v0=float('nan'),
                    lb=-1.0,
                    ub=1.0,
                )
            cc = ComponentMap()
            cc[m.x] = 1.0
            cc[m.y] = 0.0
            cv = ComponentMap()
            cv[m.x] = 0.0
            cv[m.y] = 1.0
            return _FakeMc(
                cc=cc,
                cv=cv,
                c0=0.1,
                v0=0.2,
                lb=-1.0,
                ub=2.0,
            )

        # Force variable-with-None path for first constraint
        m.z.set_value(None)
        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            with patch('pyomo.contrib.mindtpy.cut_generation.mc', side_effect=_mc_side_effect):
                add_affine_cuts(m, cfg, timing=None)

        # first skipped by None value, second logs MCPP error, third invalid,
        # fourth generates concave+convex cuts
        self.assertEqual(len(m.MindtPy_utils.cuts.aff_cuts), 2)
        self.assertGreaterEqual(cfg.logger.error.call_count, 1)

    def test_add_affine_cuts_partial_validity_paths(self):
        """Covers affine concave-only and convex-only cut branches."""

        class _FakeMc:
            def __init__(self, cc, cv, c0, v0, lb, ub):
                self._cc = cc
                self._cv = cv
                self._c0 = c0
                self._v0 = v0
                self._lb = lb
                self._ub = ub

            def subcc(self):
                return self._cc

            def subcv(self):
                return self._cv

            def concave(self):
                return self._c0

            def convex(self):
                return self._v0

            def lower(self):
                return self._lb

            def upper(self):
                return self._ub

        m = self._base_model_with_cut_blocks()
        m.c_concave = Constraint(expr=m.x * m.y <= 1.0)
        m.c_convex = Constraint(expr=m.x * m.y <= 1.0)
        m.MindtPy_utils.nonlinear_constraint_list = [m.c_concave, m.c_convex]

        def _mc_side_effect(expr):
            cc = ComponentMap()
            cc[m.x] = 1.0
            cc[m.y] = 0.0
            cv = ComponentMap()
            cv[m.x] = 0.0
            cv[m.y] = 1.0
            if expr is m.c_concave.body:
                cv[m.x] = 0.0
                cv[m.y] = 0.0
            else:
                cc[m.x] = 0.0
                cc[m.y] = 0.0
            return _FakeMc(cc=cc, cv=cv, c0=0.1, v0=0.2, lb=-1.0, ub=1.0)

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            with patch('pyomo.contrib.mindtpy.cut_generation.mc', side_effect=_mc_side_effect):
                add_affine_cuts(m, self._cfg(), timing=None)

        self.assertEqual(len(m.MindtPy_utils.cuts.aff_cuts), 2)

    def test_add_affine_cuts_nan_inf_comparison_paths(self):
        """Covers affine NaN/inf comparison lines and fixed-variable branch."""

        class _EqNan:
            def __eq__(self, other):
                return isinstance(other, float) and other != other

        class _EqInf:
            def __eq__(self, other):
                return isinstance(other, float) and other == float('inf')

        class _FakeMc:
            def __init__(self, cc, cv, c0, v0, lb, ub):
                self._cc = cc
                self._cv = cv
                self._c0 = c0
                self._v0 = v0
                self._lb = lb
                self._ub = ub

            def subcc(self):
                return self._cc

            def subcv(self):
                return self._cv

            def concave(self):
                return self._c0

            def convex(self):
                return self._v0

            def lower(self):
                return self._lb

            def upper(self):
                return self._ub

        m = self._base_model_with_cut_blocks()
        m.x.fix(0.25)
        m.c_bad = Constraint(expr=m.x * m.y <= 1.0)
        m.MindtPy_utils.nonlinear_constraint_list = [m.c_bad]

        cc = ComponentMap()
        cc[m.x] = _EqNan()
        cc[m.y] = _EqNan()
        cv = ComponentMap()
        cv[m.x] = _EqInf()
        cv[m.y] = _EqInf()
        fake_mc = _FakeMc(cc=cc, cv=cv, c0=_EqNan(), v0=_EqInf(), lb=-1.0, ub=1.0)

        with patch('pyomo.contrib.mindtpy.cut_generation.time_code', return_value=nullcontext()):
            with patch('pyomo.contrib.mindtpy.cut_generation.mc', return_value=fake_mc):
                add_affine_cuts(m, self._cfg(), timing=None)

        self.assertEqual(len(m.MindtPy_utils.cuts.aff_cuts), 0)


if __name__ == '__main__':
    unittest.main()
