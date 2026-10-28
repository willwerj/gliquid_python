"""hull_mode='hsx' — the ternary pipeline through the generalized HSX class.

The default hull mode. The gates are PHYSICAL (agreement with the gtx slicing within its own
temperature quantization), not 1e-9 pins: gtx temperatures are T-grid multiples, the hsx facet
temperatures are continuous, so per-composition liquidus differences are bounded by one T_incr.
Also pins TLIPlotter's forwarding of solid_solutions/ss_kwargs and of the hull-evaluation
settings to the interpolation. Runs offline (cached Al-Mg-Si MP entries).
"""

import numpy as np
import pandas as pd
import pytest

import gliquid.ternary as tmod

AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}
GTX_T_INCR = 10.0


def _build_plotter(**kwargs):
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        param_format="linear",
        **kwargs,
    )
    plotter.process_data()
    return plotter


@pytest.fixture(scope="module")
def gtx():
    """The gtx plotter at ``GTX_T_INCR`` with its tx figure built, once per module."""
    plotter = _build_plotter(hull_mode="gtx", T_incr=GTX_T_INCR)
    plotter.get_plot("tx")
    return plotter


@pytest.fixture(scope="module")
def hsx():
    """The hsx plotter with its tx figure built, once per module."""
    plotter = _build_plotter(hull_mode="hsx")
    plotter.get_plot("tx")
    return plotter


def _liquidus_by_comp(equil_dfs):
    """Per-composition liquidus: max T of the L vertices of solid-touching facets."""
    df = pd.concat(equil_dfs, ignore_index=True)
    has_solid = df.groupby("simplex_id")["Phase"].transform(lambda s: (s != "L").any())
    liq_rows = df[(df["Phase"] == "L") & has_solid]
    return liq_rows.groupby(["x0_orig", "x1_orig"])["T"].max()


def test_invalid_hull_mode_rejected():
    with pytest.raises(ValueError, match="hull_mode"):
        tmod.TLIPlotter.from_components(["Al", "Mg", "Si"], hull_mode="bogus")


def test_ss_kwargs_forwarding_fixed():
    # Pins that solid_solutions/ss_kwargs are forwarded to ternary_interpolation.
    plotter = tmod.TLIPlotter.from_components(
        ["Hf", "Ti", "Zr"], solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    assert plotter.solid_solutions is True
    assert plotter.ss_kwargs == {"ref_mode": "from_unary_db"}


class TestHsxModeAgreement:
    def test_hsx_output_schema_matches_gtx(self, hsx):
        assert hsx._n_simplex_vertices == 4
        assert len(hsx.equil_df_list) == 1
        df = hsx.equil_df_list[0]
        assert list(df.columns) == [
            "x0",
            "x1",
            "T",
            "Phase",
            "Colors",
            "simplex_id",
            "x0_orig",
            "x1_orig",
        ]
        counts = df["simplex_id"].value_counts()
        assert (counts == 4).all()

    def test_liquidus_agreement_within_gtx_quantization(self, gtx, hsx):
        liq_gtx = _liquidus_by_comp(gtx.equil_df_list)
        liq_hsx = _liquidus_by_comp(hsx.equil_df_list)
        shared = liq_gtx.index.intersection(liq_hsx.index)
        assert len(shared) > 20, "the two modes should share most grid compositions"
        diff = (liq_gtx.loc[shared] - liq_hsx.loc[shared]).abs()
        # Robust-statistic gate: the coarse-grid GTX reference can emit a sliver
        # tie-simplex at an isolated boundary composition, so gate the distribution, not
        # the max.
        assert np.quantile(diff, 0.95) <= GTX_T_INCR + 1e-6
        assert (diff <= GTX_T_INCR + 1e-6).mean() >= 0.95

    def test_same_stable_phase_set(self, gtx, hsx):
        phases_gtx = set(pd.concat(gtx.equil_df_list, ignore_index=True)["Phase"].unique())
        phases_hsx = set(pd.concat(hsx.equil_df_list, ignore_index=True)["Phase"].unique())
        assert phases_hsx == phases_gtx

    def test_eutectic_temperature_agreement(self, gtx, hsx):
        liq_gtx = _liquidus_by_comp(gtx.equil_df_list)
        liq_hsx = _liquidus_by_comp(hsx.equil_df_list)
        assert abs(liq_gtx.min() - liq_hsx.min()) <= GTX_T_INCR + 1e-6

    def test_gtx_mode_keeps_triangles_and_per_temperature_frames(self, gtx):
        assert gtx.hull_mode == "gtx"
        assert gtx._n_simplex_vertices == 3
        assert len(gtx.equil_df_list) > 1

    def test_hsx_is_the_default(self, hsx):
        assert tmod.TLIPlotter.from_components(["Al", "Mg", "Si"]).hull_mode == "hsx"
        assert tmod.TernaryLiquidInterpolation(["Al", "Mg", "Si"]).hull_mode == "hsx"

    def test_get_convex_hull_in_hsx_mode_matches_gtx(self, gtx, hsx):
        # Both grids are multiples of T_incr, so a temperature on both snaps identically.
        shared = sorted(set(gtx.T_grid) & set(hsx.T_grid))
        assert shared, "the two windows should overlap"
        T_celsius = float(shared[len(shared) // 2]) - 273.15
        from_gtx = gtx.get_convex_hull(T_celsius)
        from_hsx = hsx.get_convex_hull(T_celsius)
        assert from_hsx["temperature_k"] == from_gtx["temperature_k"]
        pd.testing.assert_frame_equal(from_hsx["raw_slice_df"], from_gtx["raw_slice_df"])
        assert {tuple(sorted(s)) for s in from_hsx["hull_simplices"]} == {
            tuple(sorted(s)) for s in from_gtx["hull_simplices"]
        }

    def test_the_ch_plot_renders_in_hsx_mode(self, hsx):
        T_celsius = float(hsx.T_grid[len(hsx.T_grid) // 2]) - 273.15
        assert len(hsx.get_plot("ch", T_celsius=T_celsius).data) > 0

    def test_the_ch_plot_hovers_the_at_pct_of_each_element_in_sys_name_order(self, hsx):
        T_celsius = float(hsx.T_grid[len(hsx.T_grid) // 2]) - 273.15
        hovered = [t for t in hsx.get_plot("ch", T_celsius=T_celsius).data if t.hovertemplate]
        assert hovered
        for trace in hovered:
            lines = trace.hovertemplate.split("<br>")
            elements = [line.split(":")[0] for line in lines if line.endswith(" at. %")]
            assert elements == hsx.sys_name.split("-")
            pct = trace.customdata[:, :3].astype(float)
            assert pct.sum(axis=1) == pytest.approx(100.0, abs=0.15)


class TestPlotterSettings:
    def _interp(self, **kwargs):
        return tmod.TernaryLiquidInterpolation(
            ["Al", "Mg", "Si"],
            xs_mix={k: list(v) for k, v in AMS_LD.items()},
            param_format="linear",
            **kwargs,
        )

    def test_the_plotter_keeps_the_models_settings(self):
        ti = self._interp(hull_mode="gtx", T_incr=5, temp_slider=(10, 20))
        plotter = tmod.TLIPlotter(ti)
        assert (plotter.hull_mode, plotter.T_incr, plotter.temp_slider) == ("gtx", 5, [10, 20])

    def test_from_components_forwards_only_the_settings_it_is_given(self):
        plotter = tmod.TLIPlotter.from_components(["Al", "Mg", "Si"], T_incr=5)
        assert (plotter.hull_mode, plotter.T_incr, plotter.temp_slider) == ("hsx", 5, [0, 0])

    @pytest.mark.parametrize(
        "name, value", [("temp_slider", (0, 50)), ("T_incr", 20), ("hull_mode", "hsx")]
    )
    def test_a_removed_plotter_setting_names_its_replacement(self, name, value):
        with pytest.raises(TypeError, match="use TernaryLiquidInterpolation"):
            tmod.TLIPlotter(self._interp(), **{name: value})


class TestLiquidusCoexistentPhases:
    """``liq_plotting_df['coexistent_phases']`` -- the solid assemblage under each liquidus node.

    Behavioural gates, not pins: the numbers below are floors with headroom.
    """

    @staticmethod
    def _facets_touching_a_solid(plotter):
        raw = pd.concat(plotter.equil_df_list, ignore_index=True)
        return raw, raw.groupby("simplex_id")["Phase"].apply(lambda col: (col != "L").any())

    @pytest.mark.parametrize("mode", ["gtx", "hsx"])
    def test_most_liquidus_nodes_name_an_assemblage(self, request, mode):
        col = request.getfixturevalue(mode).liq_plotting_df["coexistent_phases"]
        assert len(col) > 20, "the fixture should cover most grid compositions"
        assert (col.str.len() > 0).mean() >= 0.9

    @pytest.mark.parametrize("mode", ["gtx", "hsx"])
    def test_empty_only_where_no_facet_at_that_node_touches_a_solid(self, request, mode):
        # A node reports no assemblage only when every facet at its liquidus vertex is all-liquid;
        # the facets sharing a vertex differ only in simplex_id, so any solid-touching one must be
        # named.
        plotter = request.getfixturevalue(mode)
        raw, touches_solid = self._facets_touching_a_solid(plotter)
        liq = raw[raw["Phase"] == "L"]
        coldest = liq.groupby(["x0", "x1"])["T"].transform("min")
        rescuable = (
            liq[liq["T"] <= coldest + 1e-9]
            .groupby(["x0", "x1"])["simplex_id"]
            .apply(lambda ids: bool(touches_solid.reindex(ids).any()))
        )
        empty = plotter.liq_plotting_df.set_index(["x0", "x1"])["coexistent_phases"].eq("")
        assert not (empty & rescuable.reindex(empty.index)).any()

    @pytest.mark.parametrize("mode", ["gtx", "hsx"])
    def test_named_phases_are_stable_phases_of_this_system(self, request, mode):
        plotter = request.getfixturevalue(mode)
        raw, _ = self._facets_touching_a_solid(plotter)
        stable_solids = {str(p) for p in raw["Phase"].unique()} - {"L"}
        named = {
            phase
            for entry in plotter.liq_plotting_df["coexistent_phases"]
            if entry
            for phase in entry.split(", ")
        }
        assert named, "no node named an assemblage at all"
        assert named <= stable_solids

    def test_gtx_assemblage_is_contained_in_the_hsx_one(self, gtx, hsx):
        # A gtx facet is a triangle and an hsx facet a tetrahedron, so gtx can name at most
        # two solids where hsx names three. Where both name an assemblage the gtx set is the
        # hsx set or a subset of it -- the two backends read the same equilibrium.
        gtx = gtx.liq_plotting_df.set_index(["x0_orig", "x1_orig"])["coexistent_phases"]
        hsx = hsx.liq_plotting_df.set_index(["x0_orig", "x1_orig"])["coexistent_phases"]
        shared = gtx.index.intersection(hsx.index)
        assert len(shared) > 20
        pairs = [(a, b) for a, b in zip(gtx.loc[shared], hsx.loc[shared]) if a and b]
        assert len(pairs) >= 0.9 * len(shared), (
            "both modes should name an assemblage almost everywhere"
        )
        assert all(set(a.split(", ")) <= set(b.split(", ")) for a, b in pairs)
