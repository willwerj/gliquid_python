"""Figure-level checks for solid-solution phases on the BLPlotter surface.

Split out of ``tests/test_binary_solution.py``, which keeps the model math, the from_cache
reconciliation and the pickle round-trip. Everything here builds a plotly Figure and asserts
on its traces, colors and legend entries -- presentation, not thermodynamics.

The ``ss_from_cache`` fixture is duplicated from that file rather than shared: the two
halves of the split must not import each other.
"""

import pytest

from gliquid.binary import BinaryLiquid


@pytest.fixture(scope="module")
def ss_from_cache():
    return BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )


class TestSSPlotter:
    def test_ss_plot_types_return_figures(self, ss_from_cache):
        import plotly.graph_objects as go

        from gliquid.binary import BLPlotter

        plotter = BLPlotter(ss_from_cache)
        for plot_type in ("fit+liq", "hsx", "scatter", "ch+g"):
            fig = plotter.get_plot(plot_type)
            assert isinstance(fig, go.Figure), f"{plot_type} did not return a Figure"

    def test_ss_phases_get_fixed_colors_and_display_names(self, ss_from_cache):
        from gliquid.binary import SS_FIXED_COLORS, BLPlotter

        plotter = BLPlotter(ss_from_cache)
        ss_from_cache.update_phase_points()
        cmap = plotter._phase_color_map()
        assert cmap["BCC"] == SS_FIXED_COLORS["BCC"]
        assert cmap["HCP"] == SS_FIXED_COLORS["HCP"]
        assert plotter._phase_display_name("BCC") == "BCC (Hf, Zr)"
        assert plotter._phase_display_name("L") == "L"
        # color cache lives on the plotter, so BinaryLiquid pickles stay clean
        assert not hasattr(ss_from_cache, "_ss_phase_color_cache")

    def test_ss_fit_liq_renders_solution_bands(self, ss_from_cache):
        """Unified fit+liq: SS phases render as envelope branches through plot_tx with
        reserved colors, display-name legend entries, and no line-compound verticals."""
        from gliquid.binary import SS_FIXED_COLORS, BLPlotter

        fig = BLPlotter(ss_from_cache).get_plot("fit+liq")
        # Only phases that are actually stable somewhere draw a band. Hf-Zr loads an FCC model
        # via the omegas fallback (neither element has an FCC polymorph), but FCC never reaches
        # the hull here -- a loaded-but-metastable phase drawing nothing is correct.
        stable = {p for p in ss_from_cache.ss_models if p in set(ss_from_cache.hsx.df_tx["label"])}
        assert {"BCC", "HCP"} <= stable
        for name in stable:
            colored = [
                tr for tr in fig.data if getattr(tr.line, "color", None) == SS_FIXED_COLORS[name]
            ]
            assert colored, f"no traces drawn in {name}'s reserved color"
            for tr in colored:
                xs = [] if tr.x is None else [x for x in tr.x if x is not None]
                assert not (len(xs) == 2 and xs[0] == xs[1]), (
                    f"{name} drawn as a vertical line compound"
                )
        legend_names = {tr.name for tr in fig.data if tr.showlegend}
        assert "BCC (Hf, Zr)" in legend_names and "HCP (Hf, Zr)" in legend_names

    def test_ss_ch_g_overlays_dashed_ss_gibbs_curves(self, ss_from_cache):
        """Unified ch+g: dashed per-SS-phase Gibbs curves ride on the non-SS renderer."""
        from gliquid.binary import BLPlotter

        fig = BLPlotter(ss_from_cache).get_plot("ch+g")
        ss_curves = [
            tr
            for tr in fig.data
            if tr.name
            and tr.name.split()[0] in ("BCC", "HCP")
            and getattr(tr.line, "dash", None) == "dash"
        ]
        assert ss_curves, "expected dashed SS Gibbs curve(s) in unified ch+g"
        liquid_curves = [tr for tr in fig.data if tr.name and "Liquid" in tr.name]
        assert liquid_curves, "liquid Gibbs curve(s) missing from unified ch+g"

    def test_no_ss_fit_liq_trace_count_unchanged(self):
        import plotly.graph_objects as go

        from gliquid.binary import BinaryLiquid, BLPlotter

        a = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        base_fig = BLPlotter(a).get_plot("fit+liq")
        assert isinstance(base_fig, go.Figure)
        b = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=False)
        again_fig = BLPlotter(b).get_plot("fit+liq")
        assert len(base_fig.data) == len(again_fig.data)
