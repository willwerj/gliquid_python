"""Plotter presentation ordering (S9b): resolve_component_order + order= APIs.

Plotting classes default to ALPHABETICAL presentation regardless of construction order
(pass order='given' or a permutation spec to override); computation classes preserve
construction order. BLPlotter re-frames through BinaryLiquid.with_component_order —
xs_mix swaps odd RK orders, liquidus mirrors, SS models re-derive, hull rebuilds.
Runs offline on the Hf-Zr fixtures.
"""

import numpy as np
import pytest
from pymatgen.core import Composition

import gliquid.api as api
from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.phase import UNARY, Phase, resolve_component_order
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

SS_MODELS = {
    "BCC": {
        "omega": {"Hf-Zr": 10000.0},
        "delta_h": {"Hf": 2000.0, "Zr": 4000.0},
        "delta_s": {"Hf": 0.5, "Zr": 1.0},
    },
}
X_TEST = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


class TestResolveComponentOrder:
    COMPS = ["Hf", "Ti", "Zr"]

    def test_default_and_alphabetical(self):
        assert resolve_component_order(None, ["Zr", "Hf"]) == ["Hf", "Zr"]
        assert resolve_component_order("alphabetical", self.COMPS) == self.COMPS

    def test_given_preserves(self):
        assert resolve_component_order("given", ["Zr", "Hf"]) == ["Zr", "Hf"]

    def test_string_and_list_specs(self):
        assert resolve_component_order("Zr-Hf-Ti", self.COMPS) == ["Zr", "Hf", "Ti"]
        assert resolve_component_order(["Ti", "Zr", "Hf"], self.COMPS) == ["Ti", "Zr", "Hf"]

    def test_composition_flexible_matching(self):
        # any formula spelling identifies its component — the compound-ready contract
        assert resolve_component_order(["MgCu", "Mg"], ["CuMg", "Mg"]) == ["CuMg", "Mg"]
        assert resolve_component_order([Composition("Zr"), "Hf"], ["Hf", "Zr"]) == ["Zr", "Hf"]

    def test_errors(self):
        with pytest.raises(ValueError, match="exactly once"):
            resolve_component_order(["Hf"], ["Hf", "Zr"])
        with pytest.raises(ValueError, match="unmatched or duplicated"):
            resolve_component_order(["Hf", "Hf"], ["Hf", "Zr"])
        with pytest.raises(ValueError, match="unmatched or duplicated"):
            resolve_component_order(["Hf", "Nb"], ["Hf", "Zr"])


@pytest.fixture(scope="module")
def bl_alpha():
    return BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)


@pytest.fixture(scope="module")
def bl_rev():
    return BinaryLiquid.from_cache("Zr-Hf", pd_ind=0)


class TestWithComponentOrder:
    def test_same_order_returns_self(self, bl_alpha):
        assert bl_alpha.with_component_order(["Hf", "Zr"]) is bl_alpha
        assert bl_alpha.with_component_order("given") is bl_alpha
        assert bl_alpha.with_component_order("alphabetical") is bl_alpha

    def test_reframed_copy_mirrors_everything(self, bl_alpha):
        clone = bl_alpha.with_component_order(["Zr", "Hf"])
        assert clone.components == ["Zr", "Hf"]
        assert clone.sys_name == "Zr-Hf"
        assert clone.xs_mix.values == bl_alpha.xs_mix.swapped().values
        assert [str(el) for el in clone.dft_ch.elements] == ["Zr", "Hf"]
        expected_liq = [[1 - x, t] for x, t in reversed(bl_alpha.digitized_liq)]
        for (xc, tc), (xe, te) in zip(clone.digitized_liq, expected_liq):
            assert xc == pytest.approx(xe, abs=1e-12) and tc == pytest.approx(te)
        # eqs refs follow the new frame (rebuilt via the pickle-restore path)
        import sympy as sp

        from gliquid.solution import t_sym

        assert sp.simplify(clone.eqs["ga"] - UNARY["Zr"].gibbs_ref_expr(t_sym)) == 0
        assert sp.simplify(clone.eqs["gb"] - UNARY["Hf"].gibbs_ref_expr(t_sym)) == 0

    def test_round_trip_restores(self, bl_alpha):
        back = bl_alpha.with_component_order(["Zr", "Hf"]).with_component_order(["Hf", "Zr"])
        assert back.components == ["Hf", "Zr"]
        for (xb, tb), (xa, ta) in zip(back.digitized_liq, bl_alpha.digitized_liq):
            assert xb == pytest.approx(xa, abs=1e-12) and tb == pytest.approx(ta)
        assert back.xs_mix.values == bl_alpha.xs_mix.values

    def test_ss_models_reframe(self):
        ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
        bl = BinaryLiquid(
            "Hf-Zr",
            ["Hf", "Zr"],
            ss_models=SS_MODELS,
            component_data=UNARY.component_data(["Hf", "Zr"]),
            dft_ch=ch,
            phases=[Phase(phase_type="liquid", name="L")],
            temp_range=[300, 3000],
        )
        clone = bl.with_component_order(["Zr", "Hf"])
        h_a, s_a = bl.solid_solution_h_s("BCC", x_vals=X_TEST)
        h_r, s_r = clone.solid_solution_h_s("BCC", x_vals=X_TEST)
        np.testing.assert_allclose(h_r, h_a[::-1], rtol=1e-9)
        np.testing.assert_allclose(s_r, s_a[::-1], rtol=1e-9)


class TestBLPlotterOrder:
    def test_default_is_noop_for_alphabetical_bl(self, bl_alpha):
        plotter = BLPlotter(bl_alpha)
        assert plotter._bl is bl_alpha

    def test_reversed_bl_renders_alphabetical_by_default(self, bl_alpha, bl_rev):
        plotter = BLPlotter(bl_rev)
        assert plotter._bl.components == ["Hf", "Zr"]
        for (xp, tp), (xa, ta) in zip(plotter._bl.digitized_liq, bl_alpha.digitized_liq):
            assert xp == pytest.approx(xa, abs=1e-12) and tp == pytest.approx(ta)
        import plotly.graph_objects as go

        fig = plotter.get_plot("fit+liq")
        assert isinstance(fig, go.Figure)
        base = BLPlotter(bl_alpha).get_plot("fit+liq")
        assert len(fig.data) == len(base.data)

    def test_explicit_order_reframes(self, bl_alpha):
        plotter = BLPlotter(bl_alpha, order="Zr-Hf")
        assert plotter._bl.components == ["Zr", "Hf"]
        import plotly.graph_objects as go

        assert isinstance(plotter.get_plot("scatter"), go.Figure)

    def test_nmp_requires_construction_frame(self, bl_rev):
        with pytest.raises(ValueError, match="construction"):
            BLPlotter(bl_rev).get_plot("nmp")
        # order='given' path reaches the generator's own nmpath check instead
        with pytest.raises(Exception, match="nmpath|Nelder|fit"):
            BLPlotter(bl_rev, order="given").get_plot("nmp")


class TestTernaryPlotterOrder:
    ZERO = {"Hf-Ti": [0.0] * 4, "Ti-Zr": [0.0] * 4, "Zr-Hf": [0.0] * 4}

    def test_gtx_plotter_defaults_alphabetical(self):
        tgp = TLIPlotter.from_components(
            ["Ti", "Hf", "Zr"], xs_mix={k: list(v) for k, v in self.ZERO.items()}
        )
        assert tgp.components == ["Hf", "Ti", "Zr"]
        assert tgp.binary_systems == ["Hf-Ti", "Ti-Zr", "Zr-Hf"]

    def test_gtx_plotter_order_given_preserves(self):
        tgp = TLIPlotter.from_components(
            ["Ti", "Hf", "Zr"],
            order="given",
            xs_mix={"Ti-Hf": [0.0] * 4, "Hf-Zr": [0.0] * 4, "Zr-Ti": [0.0] * 4},
        )
        assert tgp.components == ["Ti", "Hf", "Zr"]

    def test_interpolation_base_preserves_input(self):
        tli = TernaryLiquidInterpolation(["Ti", "Hf", "Zr"])
        assert tli.components == ["Ti", "Hf", "Zr"]
        tli_ordered = TernaryLiquidInterpolation(["Ti", "Hf", "Zr"], order="alphabetical")
        assert tli_ordered.components == ["Hf", "Ti", "Zr"]
