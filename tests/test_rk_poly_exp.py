"""Tests for gliquid.solution.RKPolyExp — one pair's excess-interaction model.

Covers: construction/validation (inherits validate_mixing_parameters' raising
contract), symbolic per-order L_k(T) expressions and their H/S decomposition,
component-order swap (odd orders negate — the generalization of flip_binary_l1),
fitting-metadata introspection for the binary Nelder-Mead loop, and pickling.

Run with: python -m pytest tests/test_redlich_kister.py -v
"""

import pickle
import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliquid.solution import (  # noqa: E402
    DEFAULT_TAU,
    ParamFormat,
    RKPolyExp,
    constant_term,
    l_sym,
    linear_term,
    t_sym,
)


class TestConstruction:
    def test_default_is_linear_zeros(self):
        rk = RKPolyExp()
        assert rk.format.name == "linear"
        assert rk.values == [0.0, 0.0, 0.0, 0.0]
        assert rk.tau == DEFAULT_TAU

    def test_explicit_params_and_tau(self):
        rk = RKPolyExp("comb-exp", [-52000.0, 7.0, 10000.0, 0.0], tau=3000)
        assert rk.values == [-52000.0, 7.0, 10000.0, 0.0]
        assert rk.tau == 3000

    def test_accepts_param_format_instance(self):
        """Unregistered formats (e.g. the linear-3param branch's) must be usable."""
        fmt = ParamFormat(
            name="linear3p",
            orders=(0, 1),
            order_roles=(("a", "b"), ("a", "b")),
            order_exprs=(linear_term, linear_term),
            pinned_params=("L1_b",),
            guess_params=("L0_b", "L1_a"),
            n_invariant_constraints=1,
        )
        rk = RKPolyExp(fmt, [1.0, 2.0, 3.0, 0.0])
        assert rk.format is fmt

    def test_regular_takes_a_single_omega(self):
        rk = RKPolyExp("regular", [12500.0])
        assert rk.values == [12500.0]

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            RKPolyExp("pseudo")


class TestValidation:
    """The raising contract inherited from validate_mixing_parameters."""

    @pytest.mark.parametrize("params", ([0, 0, 0, 0], (1, 2, 3, 4), [1.2, -0.5, 3.1, 0]))
    def test_accepts_four_numeric_values(self, params):
        rk = RKPolyExp("linear", params)
        assert rk.values == [float(x) for x in params]

    def test_empty_means_zeros(self):
        assert RKPolyExp("linear", []).values == [0.0, 0.0, 0.0, 0.0]
        assert RKPolyExp("regular", []).values == [0.0]

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "1,2,3,4",
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, "3", 4],
            [True, 0, 0, 0],
            {"L0_a": 1, "L0_b": 2, "L1_a": 3, "L1_b": 4},
        ],
    )
    def test_rejects_invalid_shapes(self, bad):
        with pytest.raises(ValueError):
            RKPolyExp("linear", bad)

    def test_regular_rejects_four_values(self):
        with pytest.raises(ValueError):
            RKPolyExp("regular", [1, 2, 3, 4])


class TestExpressions:
    def test_linear_symbolic_exprs(self):
        rk = RKPolyExp("linear", [100.0, -1.0, 50.0, 0.5])
        l0, l1 = rk.exprs()
        assert l0 == l_sym("L0_a") + l_sym("L0_b") * t_sym
        assert l1 == l_sym("L1_a") + l_sym("L1_b") * t_sym

    def test_subs_map_covers_all_params(self):
        rk = RKPolyExp("linear", [100.0, -1.0, 50.0, 0.5])
        assert rk.subs_map() == {
            l_sym("L0_a"): 100.0,
            l_sym("L0_b"): -1.0,
            l_sym("L1_a"): 50.0,
            l_sym("L1_b"): 0.5,
        }

    def test_numeric_exprs_substitute_values(self):
        rk = RKPolyExp("linear", [100.0, -1.0, 50.0, 0.5])
        l0, l1 = rk.numeric_exprs()
        assert sp.simplify(l0 - (100.0 - 1.0 * t_sym)) == 0
        assert sp.simplify(l1 - (50.0 + 0.5 * t_sym)) == 0

    def test_comb_exp_envelope_uses_instance_tau(self):
        rk = RKPolyExp("comb-exp", [100.0, -1.0, 50.0, 0.0], tau=3000)
        l0 = rk.exprs()[0]
        expected = (l_sym("L0_a") + l_sym("L0_b") * t_sym) * sp.exp(-t_sym / sp.Integer(3000))
        assert sp.simplify(l0 - expected) == 0

    @pytest.mark.parametrize("fmt", ["linear", "combined", "comb-exp"])
    def test_order_h_s_decomposition_is_thermodynamically_consistent(self, fmt):
        rk = RKPolyExp(
            fmt, [100.0, -1.0, 50.0, 0.5] if fmt != "comb-exp" else [100.0, -1.0, 50.0, 0.0]
        )
        for order, l_expr in zip(rk.format.orders, rk.exprs()):
            h_expr, s_expr = rk.order_h_s(order)
            assert sp.simplify(s_expr - (-sp.diff(l_expr, t_sym))) == 0
            assert sp.simplify(h_expr - (l_expr + t_sym * s_expr)) == 0


class TestSwap:
    def test_linear_swap_negates_l1_only(self):
        """Matches flip_binary_l1: params[2:] negate on component-order swap."""
        rk = RKPolyExp("linear", [1.0, 2.0, 3.0, 4.0])
        assert rk.swapped().values == [1.0, 2.0, -3.0, -4.0]

    def test_swap_is_an_involution(self):
        rk = RKPolyExp("comb-exp", [-52000.0, 7.0, 10000.0, 0.0], tau=5000)
        assert rk.swapped().swapped() == rk

    def test_regular_swap_is_identity(self):
        rk = RKPolyExp("regular", [12500.0])
        assert rk.swapped() == rk

    def test_odd_orders_negate_up_to_l3(self):
        fmt = ParamFormat(
            name="quartic-demo",
            orders=(0, 1, 2, 3),
            order_roles=(("a",), ("a",), ("a",), ("a",)),
            order_exprs=(constant_term,) * 4,
        )
        rk = RKPolyExp(fmt, [1.0, 2.0, 3.0, 4.0])
        assert rk.swapped().values == [1.0, -2.0, 3.0, -4.0]


class TestFittingMetadata:
    def test_linear_guess_and_solve_symbols(self):
        rk = RKPolyExp("linear")
        assert rk.guess_symbols == [l_sym("L0_b"), l_sym("L1_b")]
        assert rk.solve_symbols == [l_sym("L0_a"), l_sym("L1_a")]
        assert rk.identity_constraints() == []
        assert rk.guess_param_indices == (1, 3)

    def test_comb_exp_guess_solve_identity(self):
        rk = RKPolyExp("comb-exp")
        assert rk.guess_symbols == [l_sym("L0_b"), l_sym("L1_a")]
        assert rk.solve_symbols == [l_sym("L0_a"), l_sym("L1_b")]
        assert rk.identity_constraints() == [sp.Eq(l_sym("L1_b"), 0)]
        assert rk.guess_param_indices == (1, 2)

    def test_n_params(self):
        assert RKPolyExp("linear").n_params == 4
        assert RKPolyExp("regular").n_params == 1


class TestMutation:
    def test_getitem_by_name_and_index(self):
        rk = RKPolyExp("linear", [1.0, 2.0, 3.0, 4.0])
        assert rk["L0_a"] == 1.0 and rk["L1_b"] == 4.0
        assert rk[1] == 2.0

    def test_setitem_by_name(self):
        rk = RKPolyExp("linear")
        rk["L1_a"] = -750.0
        assert rk.values == [0.0, 0.0, -750.0, 0.0]

    def test_update_replaces_full_vector_with_validation(self):
        rk = RKPolyExp("linear")
        rk.update([1, 2, 3, 4])
        assert rk.values == [1.0, 2.0, 3.0, 4.0]
        with pytest.raises(ValueError):
            rk.update([1, 2, 3])

    def test_values_returns_a_copy(self):
        rk = RKPolyExp("linear", [1.0, 2.0, 3.0, 4.0])
        rk.values.append(99)
        assert rk.n_params == 4 and rk.values == [1.0, 2.0, 3.0, 4.0]


def test_pickle_round_trip():
    rk = RKPolyExp("comb-exp", [-52000.0, 7.0, 10000.0, 0.0], tau=5000)
    clone = pickle.loads(pickle.dumps(rk))
    assert clone == rk
    assert clone.format.name == "comb-exp"
    assert clone.exprs() == rk.exprs()
