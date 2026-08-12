"""Tests for gliquid.solution.SolutionModel — the one N-component solution-phase model.

A SolutionModel bundles per-component reference G(T) expressions, per-pair
RKPolyExp excess models, and the interpolation geometry, and is the single
implementation behind: the binary liquid eqs dict (fitting), the ternary liquid
surface, and every solid-solution phase (binary AND ternary) — replacing the
former numeric/symbolic duplication.

Run with: python -m pytest tests/test_solution_model.py -v
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliquid.solution import (  # noqa: E402
    R,
    RKPolyExp,
    SolutionModel,
    build_solution_expressions,
    comp_symbols,
    t_sym,
)

XB = comp_symbols(2)[0]
X1, X2 = comp_symbols(3)


class TestCoreBuilderRKOrders:
    """build_solution_expressions now takes per-pair SEQUENCES of L_k expressions."""

    def test_two_order_sequence_matches_the_legacy_pair_tuple_shape(self):
        l0, l1 = sp.Symbol("p"), sp.Symbol("q")
        legacy = build_solution_expressions(
            [0 * t_sym, 0 * t_sym], {(0, 1): (l0, l1)}, x_syms=(XB,), ideal="plain"
        )
        diff = (1 - XB) - XB
        expected_gxs = (1 - XB) * XB * (l0 + l1 * diff)
        assert sp.simplify(legacy["g_xs"] - expected_gxs) == 0

    def test_l2_l3_orders_enter_with_diff_powers(self):
        l0, l1, l2, l3 = sp.symbols("p q r s")
        out = build_solution_expressions(
            [0 * t_sym, 0 * t_sym], {(0, 1): (l0, l1, l2, l3)}, x_syms=(XB,), ideal="plain"
        )
        diff = (1 - XB) - XB
        expected = (1 - XB) * XB * (l0 + l1 * diff + l2 * diff**2 + l3 * diff**3)
        assert sp.simplify(out["g_xs"] - expected) == 0

    def test_zero_orders_vanish_symbolically(self):
        omega = sp.Float(12500.0)
        out = build_solution_expressions(
            [0 * t_sym, 0 * t_sym], {(0, 1): (omega, 0)}, x_syms=(XB,), ideal="plain"
        )
        assert sp.simplify(out["g_xs"] - (1 - XB) * XB * omega) == 0


class TestBinaryEqs:
    @pytest.fixture
    def model(self):
        ga = 10710.0 - t_sym * 11.473  # Al liquid reference
        gb = 13260.0 - t_sym * 9.766  # Cu liquid reference
        rk = RKPolyExp("linear", [100.0, -1.0, 50.0, 0.5])
        return SolutionModel(("Al", "Cu"), (ga, gb), {(0, 1): rk})

    def test_eqs_dict_carries_the_pinned_keys_and_shapes(self, model):
        eqs = model.binary_eqs()
        for key in (
            "ga",
            "gb",
            "l0",
            "h_l0",
            "s_l0",
            "h_l0_lambdified",
            "s_l0_lambdified",
            "l1",
            "h_l1",
            "s_l1",
            "h_l1_lambdified",
            "s_l1_lambdified",
            "g_ideal",
            "g_xs",
            "g_liquid",
            "h_liquid",
            "s_liquid",
            "h_liq_lambdified",
            "s_liq_lambdified",
            "g_prime",
            "g_double_prime",
        ):
            assert key in eqs, key

    def test_lambdified_arg_orders_match_the_fitting_loop(self, model):
        """h_l0(t, L0_a, L0_b); h_liq(x, t, *flat params) — positional contracts."""
        eqs = model.binary_eqs()
        assert eqs["h_l0_lambdified"](0, 100.0, -1.0) == pytest.approx(100.0)
        assert eqs["s_l0_lambdified"](0, 100.0, -1.0) == pytest.approx(1.0)
        h = eqs["h_liq_lambdified"](0.5, 1000.0, 100.0, -1.0, 50.0, 0.5)
        s = eqs["s_liq_lambdified"](0.5, 1000.0, 100.0, -1.0, 50.0, 0.5)
        # By hand: g_ref = .5*ga+.5*gb; g_ideal = RT(ln .5); g_xs = .25*(100 - t)
        # h = g + t*s with s = -dg/dt
        g_ref_h = 0.5 * 10710.0 + 0.5 * 13260.0
        g_ref_s = 0.5 * 11.473 + 0.5 * 9.766
        h_expected = g_ref_h + 0.25 * 100.0  # L0_b*t and L1 terms carry into S, not H
        s_expected = g_ref_s - R * np.log(0.5) + 0.25 * 1.0
        assert h == pytest.approx(h_expected, rel=1e-12)
        assert s == pytest.approx(s_expected, rel=1e-12)

    def test_g_prime_is_the_composition_derivative(self, model):
        eqs = model.binary_eqs()
        assert sp.simplify(eqs["g_prime"] - sp.diff(eqs["g_liquid"], XB)) == 0
        assert sp.simplify(eqs["g_double_prime"] - sp.diff(eqs["g_liquid"], XB, 2)) == 0

    def test_binary_eqs_requires_two_components(self):
        rk = RKPolyExp("regular", [1000.0])
        m = SolutionModel(("A", "B", "C"), (0 * t_sym,) * 3, {(0, 1): rk, (1, 2): rk, (2, 0): rk})
        with pytest.raises(ValueError):
            m.binary_eqs()


class TestPairKeyNormalization:
    def test_name_keys_resolve_against_component_order(self):
        rk = RKPolyExp("regular", [1000.0])
        m = SolutionModel(
            ("Al", "Ca", "Ni"), (0 * t_sym,) * 3, {"Al-Ca": rk, "Ca-Ni": rk, "Ni-Al": rk}
        )
        assert list(m.pair_models) == [(0, 1), (1, 2), (2, 0)]

    def test_unknown_name_key_raises(self):
        rk = RKPolyExp("regular", [1000.0])
        with pytest.raises(ValueError):
            SolutionModel(("Al", "Ca"), (0 * t_sym,) * 2, {"Al-Zn": rk})


class TestSolidSolutionFactory:
    SS_MODEL = {
        "omega": {"Hf-Zr": 8000.0},
        "delta_h": {"Hf": 1200.0, "Zr": 900.0},
        "delta_s": {"Hf": 0.4, "Zr": 0.3},
    }

    def test_from_ss_model_reproduces_the_regular_solution_analytics(self):
        # the model follows the GIVEN component order (x = fraction of components[1]);
        # the schema's omega keys stay alphabetized regardless
        m = SolutionModel.from_ss_model(["Zr", "Hf"], self.SS_MODEL)
        assert m.components == ("Zr", "Hf")
        x = np.array([0.0, 0.25, 0.5, 1.0])
        h, s = m.h_s_grid((x,), 1000.0)
        xa = 1 - x
        h_expected = 900.0 * xa + 1200.0 * x + 8000.0 * xa * x
        conf = np.zeros_like(x)
        inner = (x > 0) & (x < 1)
        conf[inner] = xa[inner] * np.log(xa[inner]) + x[inner] * np.log(x[inner])
        s_expected = 0.3 * xa + 0.4 * x - R * conf
        np.testing.assert_allclose(h, h_expected, rtol=1e-12)
        np.testing.assert_allclose(s, s_expected, rtol=1e-12)

    def test_h_s_grid_endpoints_carry_no_ideal_term(self):
        """The 'safe' Piecewise zeroes x*ln(x) at the endpoints — no NaN, no -inf.

        Values agree with the pure-element offsets to float-Add-reordering noise
        (the symbolic path's term order differs from hand-numpy at ~1e-16 rel).
        """
        m = SolutionModel.from_ss_model(["Hf", "Zr"], self.SS_MODEL)
        h, s = m.h_s_grid((np.array([0.0, 1.0]),), 500.0)
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(s))
        np.testing.assert_allclose(h, [1200.0, 900.0], rtol=1e-14)
        np.testing.assert_allclose(s, [0.4, 0.3], rtol=1e-14)


class TestTernarySurface:
    def test_ternary_h_s_matches_hand_built_expressions(self):
        ga, gb, gc = (10710.0 - t_sym * 11.473, 9501.0 - t_sym * 9.001, 17480.0 - t_sym * 10.116)
        rks = {
            "Al-Ca": RKPolyExp("linear", [-45000.0, 6.0, -8000.0, 1.5]),
            "Ca-Ni": RKPolyExp("linear", [-60000.0, 8.0, -12000.0, 2.0]),
            "Ni-Al": RKPolyExp("linear", [-52000.0, 7.0, 10000.0, -1.0]),
        }
        m = SolutionModel(("Al", "Ca", "Ni"), (ga, gb, gc), rks, ideal="safe")
        x1 = np.array([0.25, 0.5])
        x2 = np.array([0.25, 0.25])
        h, s = m.h_s_grid((x1, x2), 1200.0)

        # Hand-build the same surface through the raw core builder.
        exprs = build_solution_expressions(
            [ga, gb, gc],
            {
                (0, 1): rks["Al-Ca"].numeric_exprs(),
                (1, 2): rks["Ca-Ni"].numeric_exprs(),
                (2, 0): rks["Ni-Al"].numeric_exprs(),
            },
            x_syms=(X1, X2),
            ideal="safe",
        )
        h_fn = sp.lambdify([X1, X2, t_sym], exprs["h_liquid"], "numpy")
        s_fn = sp.lambdify([X1, X2, t_sym], exprs["s_liquid"], "numpy")
        np.testing.assert_allclose(h, h_fn(x1, x2, 1200.0), rtol=1e-12)
        np.testing.assert_allclose(s, s_fn(x1, x2, 1200.0), rtol=1e-12)


def test_pickle_round_trip_drops_and_rebuilds_lambdifieds():
    rk = RKPolyExp("linear", [100.0, -1.0, 50.0, 0.5])
    m = SolutionModel(
        ("Al", "Cu"), (10710.0 - t_sym * 11.473, 13260.0 - t_sym * 9.766), {(0, 1): rk}
    )
    x = np.array([0.3, 0.7])
    before = m.h_s_grid((x,), 900.0)
    clone = pickle.loads(pickle.dumps(m))
    after = clone.h_s_grid((x,), 900.0)
    np.testing.assert_allclose(before[0], after[0], rtol=0)
    np.testing.assert_allclose(before[1], after[1], rtol=0)
    assert clone.components == m.components
