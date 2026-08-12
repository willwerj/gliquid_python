"""Tests for the centralized parameter-format registry in gliquid.solution.

A ParamFormat declares, per Redlich-Kister order L0..L3: the subterm roles
(a, b, ...), the functional form L_k(T), which parameters are pinned to zero
during fitting, and the fitting topology (guess symbols, invariant-constraint
count, penalty policy). The registry is the SINGLE home for what used to be
~16 scattered param_format string branches across binary.py/ternary.py.

Run with: python -m pytest tests/test_param_formats.py -v
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliquid.solution import (  # noqa: E402
    DEFAULT_TAU,
    PARAM_FORMATS,
    ParamFormat,
    constant_term,
    get_param_format,
    l_sym,
    linear_term,
    t_sym,
)


class TestRegistry:
    def test_registry_carries_the_four_shipped_formats(self):
        assert set(PARAM_FORMATS) == {"linear", "combined", "comb-exp", "regular"}

    @pytest.mark.parametrize("name", ["linear", "combined", "comb-exp", "regular"])
    def test_get_param_format_returns_frozen_definitions(self, name):
        fmt = get_param_format(name)
        assert fmt is PARAM_FORMATS[name]
        assert fmt.name == name
        with pytest.raises(dataclasses.FrozenInstanceError):
            fmt.name = "mutated"  # frozen dataclass

    @pytest.mark.parametrize("bad", ["exponential", "pseudo", "foo", "", None])
    def test_unknown_format_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            get_param_format(bad)


class TestShippedFormatMetadata:
    def test_linear(self):
        fmt = get_param_format("linear")
        assert fmt.orders == (0, 1)
        assert fmt.param_names == ("L0_a", "L0_b", "L1_a", "L1_b")
        assert fmt.pinned_params == ()
        assert fmt.guess_params == ("L0_b", "L1_b")
        assert fmt.solve_params == ("L0_a", "L1_a")
        assert fmt.n_invariant_constraints == 2
        assert fmt.identity_constraints() == []
        assert fmt.lupis_orders == (0, 1)
        assert fmt.lxb_default is False

    def test_combined_shares_linear_fitting_topology(self):
        lin, com = get_param_format("linear"), get_param_format("combined")
        assert com.param_names == lin.param_names
        assert com.guess_params == lin.guess_params
        assert com.solve_params == lin.solve_params
        assert com.n_invariant_constraints == 2

    def test_comb_exp(self):
        fmt = get_param_format("comb-exp")
        assert fmt.param_names == ("L0_a", "L0_b", "L1_a", "L1_b")
        assert fmt.pinned_params == ("L1_b",)
        assert fmt.guess_params == ("L0_b", "L1_a")
        assert fmt.solve_params == ("L0_a", "L1_b")
        assert fmt.n_invariant_constraints == 1
        constraints = fmt.identity_constraints()
        assert constraints == [sp.Eq(l_sym("L1_b"), 0)]
        assert fmt.lupis_orders == (0,)
        assert fmt.lxb_default is True

    def test_regular(self):
        fmt = get_param_format("regular")
        assert fmt.orders == (0,)
        assert fmt.param_names == ("L0_a",)
        assert fmt.guess_params == ()
        assert fmt.n_invariant_constraints == 0


class TestExpressions:
    def test_linear_order_exprs(self):
        fmt = get_param_format("linear")
        assert fmt.order_expr(0) == l_sym("L0_a") + l_sym("L0_b") * t_sym
        assert fmt.order_expr(1) == l_sym("L1_a") + l_sym("L1_b") * t_sym

    def test_combined_order_expr_carries_integer_tau_envelope(self):
        fmt = get_param_format("combined")
        expected = (l_sym("L0_a") + l_sym("L0_b") * t_sym) * sp.exp(-t_sym / sp.Integer(3000))
        assert sp.simplify(fmt.order_expr(0, tau=3000) - expected) == 0

    def test_comb_exp_and_combined_share_the_functional_form(self):
        """The two formats differ ONLY in fitting topology, never in L(T) form."""
        com, cxp = get_param_format("combined"), get_param_format("comb-exp")
        for order in (0, 1):
            assert com.order_expr(order, tau=DEFAULT_TAU) == cxp.order_expr(order, tau=DEFAULT_TAU)

    def test_regular_order_expr_is_a_bare_constant(self):
        fmt = get_param_format("regular")
        assert fmt.order_expr(0) == l_sym("L0_a")

    def test_symbols_align_with_param_names(self):
        fmt = get_param_format("linear")
        assert fmt.symbols() == tuple(l_sym(n) for n in fmt.param_names)


class TestExtensibility:
    def test_linear3p_shape_is_constructible_without_registering(self):
        """The linear-3param branch's format must be expressible (NOT registered here)."""
        fmt = ParamFormat(
            name="linear3p",
            orders=(0, 1),
            order_roles=(("a", "b"), ("a", "b")),
            order_exprs=(linear_term, linear_term),
            pinned_params=("L1_b",),
            guess_params=("L0_b", "L1_a"),
            n_invariant_constraints=1,
            lupis_orders=(0,),
            lxb_default=True,
        )
        assert fmt.param_names == ("L0_a", "L0_b", "L1_a", "L1_b")
        assert fmt.solve_params == ("L0_a", "L1_b")
        assert fmt.order_expr(0) == l_sym("L0_a") + l_sym("L0_b") * t_sym
        assert "linear3p" not in PARAM_FORMATS

    def test_l2_l3_orders_are_expressible(self):
        """Up to four RK orders (L0..L3), roles per order as the format defines."""
        fmt = ParamFormat(
            name="quartic-demo",
            orders=(0, 1, 2, 3),
            order_roles=(("a", "b"), ("a", "b"), ("a",), ("a",)),
            order_exprs=(linear_term, linear_term, constant_term, constant_term),
        )
        assert fmt.param_names == ("L0_a", "L0_b", "L1_a", "L1_b", "L2_a", "L3_a")
        assert fmt.order_expr(2) == l_sym("L2_a")
        assert fmt.order_expr(3) == l_sym("L3_a")

    def test_orders_beyond_l3_are_rejected(self):
        with pytest.raises(ValueError):
            ParamFormat(
                name="too-high",
                orders=(0, 4),
                order_roles=(("a",), ("a",)),
                order_exprs=(constant_term, constant_term),
            )
