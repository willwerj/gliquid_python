"""The liquid's linearization temperature T*(x): a B pass at the x-weighted melting point, then
D passes at the previous hull's liquidus until the per-row tangent error is within tolerance.

Runs offline: cached Pb-Tb and Al-Mg-Si / Hf-Ti-Zr entries, MPDS reads made hermetic.
"""

import numpy as np
import pytest

import gliquid.binary as bmod
import gliquid.ternary as tmod
from gliquid.binary import BinaryLiquid, _grid_liquidus
from gliquid.liquidus_sheets import TstarField
from gliquid.solution import DEFAULT_TSTAR_TOL, x_vals

from ._mpds_builders import hermetic_mpds

PB_TB = [-144129.625, -19.75286865234375, 57807.396484375, 0.0]
HTZ = {
    "Hf-Ti": [-30000.0, -8.0, 6000.0, 0.0],
    "Hf-Zr": [5000.0, -3.0, -2500.0, 0.0],
    "Ti-Zr": [-12000.0, 4.0, -1500.0, 0.0],
}


@pytest.fixture(scope="module")
def pb_tb():
    with hermetic_mpds():
        bl = BinaryLiquid.from_cache("Pb-Tb", param_format="comb-exp")
    bl.update_params(PB_TB)
    return bl


def _b_pass_points(bl):
    """``bl.update_phase_points()`` with the hull cap at 1."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bmod, "DEFAULT_TSTAR_MAX_HULLS", 1)
        bl.update_phase_points()


class TestBinary:
    def test_the_b_pass_keeps_the_x_weighted_melting_point(self, pb_tb):
        try:
            _b_pass_points(pb_tb)
            assert pb_tb.liquid_tstar == pytest.approx(pb_tb.linear_tstar(x_vals[1:-1]))
        finally:
            pb_tb.update_phase_points()

    def test_d_passes_stop_within_the_tolerance(self, pb_tb):
        _b_pass_points(pb_tb)
        b_error = pb_tb.liquid_tstar_error(_grid_liquidus(pb_tb.phases[-1].points, x_vals[1:-1]))
        pb_tb.update_phase_points()
        error = pb_tb.liquid_tstar_error(_grid_liquidus(pb_tb.phases[-1].points, x_vals[1:-1]))
        assert b_error > DEFAULT_TSTAR_TOL
        assert pb_tb.liquid_tstar != pytest.approx(pb_tb.linear_tstar(x_vals[1:-1]))
        assert error <= DEFAULT_TSTAR_TOL

    def test_a_single_missing_row_keeps_the_liquidus_continuous(self, pb_tb):
        saved = pb_tb.phases[-1].points
        try:
            pb_tb.phases[-1].points = [[0.53, 1500.0], [0.55, 1500.0]]
            assert pb_tb.liquidus_is_continuous()
            pb_tb.phases[-1].points = [[0.53, 1500.0], [0.56, 1500.0]]
            assert not pb_tb.liquidus_is_continuous()
        finally:
            pb_tb.phases[-1].points = saved

    @pytest.mark.parametrize(
        "system, params",
        [
            ("Cu-Mg", [21434.39, 19.316912, -3168.2158, 0.0]),
            ("Al-Cu", [149039.0, -26.532398, -37302.914, 0.0]),
        ],
    )
    def test_the_gap_top_row_converges_without_a_spike(self, system, params):
        with hermetic_mpds():
            bl = BinaryLiquid.from_cache(system, param_format="comb-exp")

        def top_excess_and_shift():
            grid = x_vals[1:-1]
            liq = _grid_liquidus(bl.phases[-1].points, grid)
            excess = np.nanmax(liq[1:-1] - np.maximum(liq[:-2], liq[2:]))
            errors = bl.liquid_tstar_row_errors(liq)
            return excess, bl.liquidus_shift_estimate(bl.to_hsx(), liq, errors).max()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(bmod, "DEFAULT_TSTAR_LIQ_TOL", float("inf"))
            mp.setattr(bmod, "DEFAULT_TSTAR_MAX_HULLS", 30)
            bl.update_params(params)
            g_only_excess, g_only_shift = top_excess_and_shift()
        bl.update_params(params)
        excess, shift = top_excess_and_shift()
        assert g_only_excess > 5.0
        assert g_only_shift > bmod.DEFAULT_TSTAR_LIQ_TOL
        assert excess < 2.0
        assert shift <= bmod.DEFAULT_TSTAR_LIQ_TOL

    def test_exact_ties_match_the_hull_for_a_liquid_linear_in_t(self):
        with hermetic_mpds():
            bl = BinaryLiquid.from_cache("Cu-Mg", param_format="linear")
        bl.update_params([40000.0, 0.0, 0.0, 0.0])
        hull_ties = bl.hsx.two_liquid_ties()
        assert hull_ties
        exact = bl._exact_two_liquid_ties(hull_ties, [])
        assert exact
        for lo, hi, temp in exact:
            assert any(
                abs(lo - a) <= 0.02 and abs(hi - b) <= 0.02 and abs(temp - t) <= 0.5
                for a, b, t in hull_ties
            )


@pytest.fixture(scope="module")
def htz():
    """Hf-Ti-Zr processed after the B pass only (``"b"``) and at the default cap, which this
    system converges within (``"default"``)."""

    def build():
        ti = tmod.TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"], xs_mix=HTZ, param_format="comb-exp", delta=0.05, hull_mode="hsx"
        )
        ti.process_data()
        return ti

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tmod, "DEFAULT_TERNARY_TSTAR_MAX_HULLS", 1)
        b_pass = build()
    return {"b": b_pass, "default": build()}


def _node_errors(ti):
    """|G_lin(T_liq; T*) - G(T_liq)| at every liquid node with a liquidus on the processed hull."""
    df = ti.equil_df_list[0]
    liq = df[df["Phase"] == "L"]
    lowest = liq.groupby([liq["x0_orig"].round(4), liq["x1_orig"].round(4)])["T"].min() + 273.15
    nodes = ti.liquid_tstar
    keys = list(zip(nodes["x0"].round(4), nodes["x1"].round(4)))
    t_liq = np.array([lowest.get(k, np.nan) for k in keys])
    ok = np.isfinite(t_liq)
    x = (nodes["x0"].to_numpy()[ok], nodes["x1"].to_numpy()[ok])
    model = ti.liquid_model()
    h_lin, s_lin = model.h_s_grid(x, nodes["tstar"].to_numpy()[ok])
    h_ex, s_ex = model.h_s_grid(x, t_liq[ok])
    return np.abs((h_lin - t_liq[ok] * s_lin) - (h_ex - t_liq[ok] * s_ex))


class TestTernary:
    def test_the_b_pass_keeps_the_x_weighted_melting_point(self, htz):
        ti = htz["b"]
        nodes = ti.liquid_tstar
        assert nodes["tstar"].to_numpy() == pytest.approx(
            ti.linear_tstar(nodes["x0"].to_numpy(), nodes["x1"].to_numpy())
        )

    def test_d_passes_stop_within_the_tolerance(self, htz):
        ti = htz["default"]
        nodes = ti.liquid_tstar
        assert _node_errors(htz["b"]).max() > DEFAULT_TSTAR_TOL
        assert nodes["tstar"].to_numpy() != pytest.approx(
            ti.linear_tstar(nodes["x0"].to_numpy(), nodes["x1"].to_numpy())
        )
        assert _node_errors(ti).max() <= DEFAULT_TSTAR_TOL

    def test_the_processed_hull_is_the_last_relinearized_one(self, htz):
        ti = htz["default"]
        rows = (ti.proc_df["Phase"] == "L").to_numpy()
        h, _ = ti.liquid_model().h_s_grid(
            (ti.liquid_tstar["x0"].to_numpy(), ti.liquid_tstar["x1"].to_numpy()),
            ti.liquid_tstar["tstar"].to_numpy(),
        )
        assert ti.proc_df["H"].to_numpy()[rows] == pytest.approx(np.where(np.isfinite(h), h, 0))


class TestTstarField:
    TM = [1000.0, 1500.0, 2500.0]

    def test_it_takes_the_node_values_and_the_linear_field_between_unmoved_nodes(self):
        nodes = np.array([(i / 4, j / 4) for i in range(5) for j in range(5 - i)])
        linear = TstarField(self.TM)
        values = linear(nodes)
        moved = np.flatnonzero((nodes == [0.25, 0.25]).all(axis=1))
        values[moved] += 40.0
        field = TstarField(self.TM, nodes, values)
        assert field(nodes) == pytest.approx(values)
        far = np.array([[0.7, 0.2], [0.1, 0.8]])
        assert field(far) == pytest.approx(linear(far))

    def test_gradients_follow_t_star(self):
        model = tmod.TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"], xs_mix=HTZ, param_format="comb-exp"
        ).liquid_model()
        field = TstarField(self.TM)
        y = np.array([[0.2, 0.3], [0.5, 0.1], [0.1, 0.7]])
        _, _, dh, ds = field.h_s_gradients(model, y)
        step = 1e-6
        for k in range(2):
            shift = np.zeros(2)
            shift[k] = step
            hi = field.h_s_gradients(model, y + shift)
            lo = field.h_s_gradients(model, y - shift)
            assert dh[k] == pytest.approx((hi[0] - lo[0]) / (2 * step), rel=1e-5, abs=1e-3)
            assert ds[k] == pytest.approx((hi[1] - lo[1]) / (2 * step), rel=1e-5, abs=1e-6)
