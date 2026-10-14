"""An imputed vertex never reads as a DFT-computed one: every figure drawing the two together
marks the difference, including the hull and phase-comparison figures, which draw from
``bl.dft_ch`` once imputation has added the imputed entries.

Every case carries its un-imputed control.
"""

import colorsys
import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import sympy as sp

# The panels here are rasterised off-screen, never shown. Set before pyplot is imported,
# which is when the backend is resolved.
matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.string import htmlify

from gliquid.plotting import binary_figs, binary_tx, style
from gliquid.plotting.binary_figs import imputed_hull_entries
from gliquid.plotting.style import (
    IMPUTED_EDGE_COLOR,
    IMPUTED_LEGEND_LABEL,
    IMPUTED_MARKER_COLOR,
)

from ._mpds_builders import band_bounds as _bounds
from ._mpds_builders import close_figures  # noqa: F401
from ._mpds_builders import imputed_entry as _imputed_entry

COMPONENTS = ["Cu", "Mg"]


def _element_entries():
    return [
        ComputedEntry(Composition({"Cu": 1}), 0.0, entry_id="mp-cu"),
        ComputedEntry(Composition({"Mg": 1}), 0.0, entry_id="mp-mg"),
    ]


def _real_entry(x_b=0.34, energy=-0.2):
    """A stable DFT-style compound, the control every assertion below needs."""
    comp = Composition({"Cu": 1 - x_b, "Mg": x_b})
    return ComputedEntry(comp, energy * comp.num_atoms, entry_id="mp-9999")


def _untagged_twin(x_b=0.66, energy=-0.3):
    """The same composition as ``_imputed_entry`` without the tag.

    Same hull position and therefore the same ``entry.name``, so an MP band keyed on that
    name can be rendered imputed or not with its geometry held fixed.
    """
    comp = Composition({"Cu": 1 - x_b, "Mg": x_b})
    return ComputedEntry(comp, energy * comp.num_atoms, entry_id="mp-9997")


def _dft_entry(energy=-0.2):
    """A stable DFT-style compound at an integer stoichiometry: its label carries no decimal,
    so a relabelling that touched it would show."""
    comp = Composition({"Cu": 2, "Mg": 1})
    return ComputedEntry(comp, energy * comp.num_atoms, entry_id="mp-9998")


def _hull(*entries):
    """Element order pinned to the component order, as every package hull is built."""
    return PhaseDiagram(
        list(_element_entries()) + list(entries), elements=[Element(c) for c in COMPONENTS]
    )


class _Model:
    """The surface ``render_dft_hull`` reads for a plain ``ch`` plot."""

    def __init__(self, chull):
        self.dft_ch = chull
        self.components = COMPONENTS
        self.component_data = None  # forces the 'ch' branch, no liquid energy needed
        self.mpds_json = {}
        self.sys_name = "Cu-Mg"
        self.dft_type = "GGA"
        self.phases = []
        self.eqs = {}


class _GibbsModel(_Model):
    """What the 'ch+g' branch reads on top of ``_Model``: an ideal liquid at G = 0."""

    def __init__(self, chull):
        super().__init__(chull)
        self.component_data = {"Cu": object(), "Mg": object()}
        self.eqs = {"g_liquid": sp.Integer(0), "ga": sp.Integer(0), "gb": sp.Integer(0)}
        self.xs_mix = SimpleNamespace(
            format=SimpleNamespace(symbols=lambda: sp.symbols("L0_a L0_b L1_a L1_b"))
        )
        self.ss_models = {}

    @staticmethod
    def get_params():
        return [0.0, 0.0, 0.0, 0.0]


class TestWhichEntriesCount:
    def test_a_tagged_entry_is_found_and_a_real_one_is_not(self):
        real, imputed = _real_entry(), _imputed_entry()
        found = imputed_hull_entries(_hull(real, imputed))
        assert [e.entry_id for e in found] == ["imputed:Mg2Cu"]

    def test_a_hull_with_none_returns_empty(self):
        assert imputed_hull_entries(_hull(_real_entry())) == []

    def test_it_matches_the_tag_not_the_label(self):
        """The label and the reduced formula differ, and the KEY is the formula:
        ``get_low_temp_phase_data`` keys its MP panel on ``entry.name``."""
        entry = _imputed_entry(label="SomeMpdsLabel")
        (found,) = imputed_hull_entries(_hull(entry))
        assert found.data["label"] == "SomeMpdsLabel"
        assert found.name != "SomeMpdsLabel"


class TestTheHullPlotMarksThem:
    @staticmethod
    def _imputed_traces(fig):
        return [tr for tr in fig.data if getattr(tr, "name", None) == IMPUTED_LEGEND_LABEL]

    def test_an_imputed_vertex_gets_its_own_marked_trace(self):
        fig = binary_figs.render_dft_hull(_Model(_hull(_real_entry(), _imputed_entry())), "ch")
        (trace,) = self._imputed_traces(fig)
        assert list(trace.x) == pytest.approx([0.66])
        assert np.all(np.isfinite(np.asarray(trace.y, dtype=float)))

    def test_a_hull_with_no_imputed_vertex_gets_no_such_trace(self):
        """The control: without it, a renderer that always added the trace would pass above."""
        fig = binary_figs.render_dft_hull(_Model(_hull(_real_entry())), "ch")
        assert self._imputed_traces(fig) == []

    def test_the_marked_trace_carries_the_mpds_label_for_hover(self):
        fig = binary_figs.render_dft_hull(_Model(_hull(_imputed_entry(label="Mg2Cu"))), "ch")
        (trace,) = self._imputed_traces(fig)
        assert list(trace.hovertext) == ["Mg2Cu"]


HULL_CASES = [("ch", {}), ("ch+g", {"t_vals": [1000.0], "t_units": "K"})]


def _hull_figure(plot_type, kwargs, *entries):
    """A rendered hull figure over a pinned Cu-Mg hull built from ``entries``."""
    model = _GibbsModel if plot_type == "ch+g" else _Model
    return binary_figs.render_dft_hull(model(_hull(*entries)), plot_type, **kwargs)


class TestTheHullPlotStandardizesEveryVertex:
    """pymatgen labels a stable vertex with the reduced formula of its composition, which
    orders elements by electronegativity and which for an imputed entry is a fractional
    formula. Every vertex is rewritten through the one compound rule instead -- component
    order, whole-number stoichiometry -- in both the 'ch' and the 'ch+g' branch.
    """

    @staticmethod
    def _trace(fig, name):
        (trace,) = [tr for tr in fig.data if getattr(tr, "name", None) == name]
        return trace

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_imputed_vertex_reads_its_standardized_mpds_label(self, plot_type, kwargs):
        imputed = _imputed_entry(label="Mg2Cu")
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), imputed)
        texts = list(self._trace(fig, "Labels (stable)").text)
        assert htmlify("CuMg2") in texts
        assert htmlify(imputed.composition.reduced_formula) not in texts
        assert not any("." in text for text in texts)

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_hover_reads_it_too(self, plot_type, kwargs):
        imputed = _imputed_entry(label="Mg2Cu")
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), imputed)
        hover = list(self._trace(fig, "Stable").hovertext)
        assert any(htmlify("CuMg2") in text for text in hover)
        assert not any(htmlify(imputed.composition.reduced_formula) in text for text in hover)

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_a_dft_vertex_reads_in_component_order(self, plot_type, kwargs):
        """The control: a DFT vertex is standardized too, with or without an imputed
        neighbour. Its reduced formula orders by electronegativity, the drawn label does not.
        """
        dft = _dft_entry()
        assert dft.composition.reduced_formula == "MgCu2"
        with_imputed = _hull_figure(plot_type, kwargs, dft, _imputed_entry(label="Mg2Cu"))
        without = _hull_figure(plot_type, kwargs, dft)
        for fig in (with_imputed, without):
            texts = list(self._trace(fig, "Labels (stable)").text)
            assert htmlify("Cu2Mg") in texts
            assert htmlify("MgCu2") not in texts

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_marker_and_the_hull_vertex_share_a_frame(self, plot_type, kwargs):
        """Both sit at ``x_s``: the marker is placed from ``components[1]``, the hull vertex
        from the hull's own element order, and they agree only on a pinned hull."""
        imputed = _imputed_entry(x_b=0.66, label="Mg2Cu")
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), imputed)
        marker = self._trace(fig, IMPUTED_LEGEND_LABEL)
        stable = self._trace(fig, "Stable")
        (index,) = [i for i, text in enumerate(stable.hovertext) if htmlify("CuMg2") in text]
        assert list(marker.x) == pytest.approx([0.66])
        assert stable.x[index] == pytest.approx(0.66)


class TestOneLabelEverywhere:
    def test_the_tx_plot_takes_the_label_from_style(self):
        """The tx plot takes its imputed legend label from ``style``, the label's one
        definition."""
        source = binary_tx.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert f'"{IMPUTED_LEGEND_LABEL}"' not in text
        assert "IMPUTED_LEGEND_LABEL" in text


class TestTheSuffixedMpdsNameNeverReachesTheFigure:
    """``Cu33Al17 rt`` on an Al-Cu axis carries a polymorph suffix AND the wrong element order
    for its system; the drawn label shows neither."""

    @staticmethod
    def _al_cu_hull(imputed_label):
        components = ["Al", "Cu"]
        elements = [
            ComputedEntry(Composition({"Al": 1}), 0.0, entry_id="mp-al"),
            ComputedEntry(Composition({"Cu": 1}), 0.0, entry_id="mp-cu"),
        ]
        x_b = 33.0 / 50.0
        comp = Composition({"Al": 1 - x_b, "Cu": x_b})
        imputed = ComputedEntry(
            comp,
            -0.3 * comp.num_atoms,
            entry_id=f"imputed:{imputed_label}",
            data={"imputed": True, "label": imputed_label, "x_b": x_b},
        )
        chull = PhaseDiagram(elements + [imputed], elements=[Element(c) for c in components])
        model = _Model(chull)
        model.components = components
        model.sys_name = "Al-Cu"
        return model

    def test_the_label_is_written_in_component_order_with_no_suffix(self):
        fig = binary_figs.render_dft_hull(self._al_cu_hull("Cu33Al17 rt"), "ch")
        (labels,) = [tr for tr in fig.data if getattr(tr, "name", None) == "Labels (stable)"]
        texts = list(labels.text)
        assert htmlify("Al17Cu33") in texts
        plain = [re.sub(r"</?sub>", "", text) for text in texts]
        assert not any(re.search(r"\b(rt|ht|lt|stab|m-orth)\b|/", text) for text in plain)


def _x_extent(collection):
    """Width, in data units, of a ``fill_betweenx`` band's path."""
    vertices = collection.get_paths()[0].vertices
    return float(vertices[:, 0].max() - vertices[:, 0].min())


def _band_at(fig, x0):
    """The MP-panel band whose left edge sits at ``x0``."""
    (band,) = [c for c in fig.axes[1].collections if _bounds(c)[0] == pytest.approx(x0)]
    return band


def _drawn(fig):
    """``fig`` rendered, so window extents can be read off its artists."""
    fig.canvas.draw()
    return fig


def _px_per_x(fig):
    """Canvas pixels per unit of composition on the MP panel."""
    return float(fig.axes[1].get_window_extent().width)


def _drawn_px_width(fig, x0, x1):
    """Canvas columns inked between ``x0`` and ``x1`` on the MP panel, at the figure's dpi.

    Reads the rasterised RGBA buffer rather than the artists, so a stroke that spills past a
    band's nominal bounds is counted where a bounds check would miss it. The window is the
    band plus a margin on each side, and the rows are the panel's interior.
    """
    _drawn(fig)
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    panel = fig.axes[1].get_window_extent()
    top = int(round(pixels.shape[0] - panel.y1)) + 2
    bottom = int(round(pixels.shape[0] - panel.y0)) - 2
    left, right = (fig.axes[1].transData.transform((x, 0.0))[0] for x in (x0, x1))
    window = pixels[top:bottom, int(left) - 8 : int(right) + 9]
    return int(np.count_nonzero(np.any(window != 255, axis=(0, 2))))


def _dotted_control():
    """A patch carrying the imputed border style, for comparison against a drawn one."""
    return mpatches.Patch(
        linestyle=style.IMPUTED_BORDER_DASHES, linewidth=style.IMPUTED_BORDER_WIDTH
    )


def _over_white(under, over):
    """The MP panel's two fills -- orange under, red over -- composited on a white ground."""
    ground = np.ones(3)
    lower = ground * (1 - under) + np.asarray(to_rgb("orange")) * under
    return lower * (1 - over) + np.asarray(to_rgb("red")) * over


def _pc_figure(monkeypatch, mp_phases, floor, *entries, ebelow=None):
    """The 'pc' figure over a pinned Cu-Mg hull of ``entries``, with the MP bands given.

    ``mp_phases`` and ``ebelow`` map a phase name to ``((lb, ub), value)``; ``floor`` is the
    lowest formation energy. The MPDS panel is empty.
    """
    monkeypatch.setattr(
        binary_figs.mpds,
        "get_low_temp_phase_data",
        lambda mpds_json, dft_ch: (({}, {}, 0, {}), (mp_phases, ebelow or {}, floor)),
    )
    return binary_figs.render_phase_comparison(_Model(_hull(*entries)))


@pytest.fixture
def phase_comparison(monkeypatch):
    """Render the 'pc' figure over two MP bands of IDENTICAL nominal width, one imputed.

    The MPDS/MP tables are supplied directly, so the two bands differ only in the marking.
    """

    def render():
        imputed, real = _imputed_entry(), _dft_entry()
        assert imputed.name != real.name
        mp_phases = {imputed.name: ((0.60, 0.72), -0.30), real.name: ((0.28, 0.40), -0.20)}
        return _pc_figure(monkeypatch, mp_phases, -0.30, real, imputed)

    return render


BAND = (0.60, 0.72)


@pytest.fixture
def phase_comparison_pair(monkeypatch):
    """Render one MP band at FIXED bounds, tagged imputed or not.

    ``render(imputed=..., bounds=...)`` returns the figure. Both renders put the band at the
    same x, so a width read off one is directly comparable with the other and any difference
    is the marking rather than the geometry.
    """

    def render(imputed, bounds=BAND):
        entry = _imputed_entry() if imputed else _untagged_twin()
        return _pc_figure(monkeypatch, {entry.name: (bounds, -0.30)}, -0.30, entry)

    return render


@pytest.fixture
def phase_comparison_both_layers(monkeypatch):
    """Render the MP panel with BOTH fills keyed on the same phase names, as production is.

    Formation energy under, energy-above-hull over, one imputed band and one control.
    """

    def render():
        imputed, real = _imputed_entry(), _dft_entry()
        mp_phases = {imputed.name: (BAND, -0.30), real.name: ((0.28, 0.40), -0.20)}
        ebelow = {imputed.name: (BAND, -0.10), real.name: ((0.28, 0.40), -0.05)}
        return _pc_figure(monkeypatch, mp_phases, -0.30, real, imputed, ebelow=ebelow)

    return render


class TestTheLegendEntryReadsImputed:
    """One word, spelled the same in every figure that draws an imputed phase."""

    def test_the_shared_constant_is_the_bare_word(self):
        assert IMPUTED_LEGEND_LABEL == "Imputed"

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_hull_marker_trace_is_named_it(self, plot_type, kwargs):
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        assert [tr.name for tr in fig.data if tr.name == "Imputed"] == ["Imputed"]

    def test_the_phase_comparison_legend_handle_is_named_it(self, phase_comparison):
        (text,) = phase_comparison().axes[1].get_legend().get_texts()
        assert text.get_text() == "Imputed"


class TestTheImputedHullMarkerIsACircle:
    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_symbol_is_an_open_circle(self, plot_type, kwargs):
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        (trace,) = [tr for tr in fig.data if tr.name == IMPUTED_LEGEND_LABEL]
        assert trace.marker.symbol == "circle-open"

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_it_keeps_the_imputed_marker_color(self, plot_type, kwargs):
        """The ring is stroked in ``marker.color`` at ``marker.line.width``; plotly draws an
        ``-open`` symbol from those two and ignores ``marker.line.color``."""
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        (trace,) = [tr for tr in fig.data if tr.name == IMPUTED_LEGEND_LABEL]
        assert trace.marker.color == IMPUTED_MARKER_COLOR
        assert trace.marker.line.width == 3


class TestThePymatgenTextLayerIsNotALegendEntry:
    """``PDPlotter`` emits its stable-vertex text layer as a legend entry named
    ``"Labels (stable)"``, which names a text layer rather than a phase. The trace itself
    stays -- it is what draws the vertex labels -- and leaves the legend only.
    """

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_text_layer_shows_no_legend_entry(self, plot_type, kwargs):
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        assert [
            tr for tr in fig.data if tr.name == "Labels (stable)" and tr.showlegend is not False
        ] == []

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_the_labels_themselves_survive(self, plot_type, kwargs):
        """The control: suppressing the entry must not delete the drawn labels."""
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        (labels,) = [tr for tr in fig.data if tr.name == "Labels (stable)"]
        assert list(labels.text)

    @pytest.mark.parametrize("plot_type,kwargs", HULL_CASES)
    def test_a_real_phase_entry_still_shows(self, plot_type, kwargs):
        """The control for the control: a blanket suppression would pass the two above."""
        fig = _hull_figure(plot_type, kwargs, _dft_entry(), _imputed_entry())
        shown = {tr.name for tr in fig.data if tr.showlegend is not False}
        assert IMPUTED_LEGEND_LABEL in shown


class TestTheImputedTintKeepsTheOverlapHue:
    """The MP panel fills TWICE from the same phase names, so the colour a reader sees on a
    band is the overlap of the two layers. Scaling both alphas by one factor moves the
    layers' weight ratio and with it the hue; ``imputed_band_alphas`` scales the composite
    itself toward white instead, which is a hue-preserving move by construction.
    """

    def test_the_pair_at_the_panel_alpha(self):
        assert style.imputed_band_alphas(0.5) == pytest.approx((0.26923076923076916, 0.35))

    def test_the_composite_is_a_uniform_tint_of_the_un_imputed_one(self):
        baseline = _over_white(0.5, 0.5)
        imputed = _over_white(*style.imputed_band_alphas(0.5))
        assert imputed == pytest.approx(1.0 + style.IMPUTED_BAND_TINT * (baseline - 1.0), abs=1e-12)

    @pytest.mark.parametrize("alpha", [0.2, 0.35, 0.5, 0.8])
    def test_the_identity_holds_away_from_the_panel_alpha(self, alpha):
        baseline = _over_white(alpha, alpha)
        imputed = _over_white(*style.imputed_band_alphas(alpha))
        assert imputed == pytest.approx(1.0 + style.IMPUTED_BAND_TINT * (baseline - 1.0), abs=1e-12)

    def test_the_hue_is_unchanged(self):
        baseline = _over_white(0.5, 0.5)
        imputed = _over_white(*style.imputed_band_alphas(0.5))
        assert colorsys.rgb_to_hsv(*imputed)[0] == pytest.approx(
            colorsys.rgb_to_hsv(*baseline)[0], abs=1e-12
        )

    def test_the_ink_falls_to_the_tint(self):
        """The control: an unchanged hue on its own is satisfied by changing nothing."""
        baseline = _over_white(0.5, 0.5)
        imputed = _over_white(*style.imputed_band_alphas(0.5))
        assert 1 - imputed.min() == pytest.approx(style.IMPUTED_BAND_TINT * (1 - baseline.min()))

    def test_the_two_composites_in_8_bit_channels(self):
        assert [round(255 * c) for c in _over_white(0.5, 0.5)] == [255, 105, 64]
        assert [round(255 * c) for c in _over_white(*style.imputed_band_alphas(0.5))] == [
            255,
            150,
            121,
        ]

    def test_keeping_all_the_ink_returns_the_un_imputed_pair(self):
        """The control on the rule itself: a tint of 1 must be the identity."""
        assert style.imputed_band_alphas(0.5, tint=1.0) == pytest.approx((0.5, 0.5))


class TestNothingIsHatchedAnyMore:
    def test_no_artist_on_the_pc_figure_carries_a_hatch(self, phase_comparison):
        fig = phase_comparison()
        for ax in fig.axes:
            assert [c for c in ax.collections if c.get_hatch()] == []
            assert [p for p in ax.patches if p.get_hatch()] == []

    def test_the_legend_handle_carries_the_dotted_edge_instead(self, phase_comparison):
        (handle,) = phase_comparison().axes[1].get_legend().get_patches()
        # A legend's copy of a handle keeps the scaled dash pattern but not the linestyle spec
        # it came from, so the pattern is what the control patch below is compared against.
        assert handle.get_hatch() is None
        assert handle._dash_pattern == _dotted_control()._dash_pattern
        assert handle.get_linewidth() == pytest.approx(style.IMPUTED_BORDER_WIDTH)
        assert tuple(handle.get_edgecolor()[:3]) == pytest.approx(to_rgb(IMPUTED_EDGE_COLOR))

    def test_a_solid_handle_would_not_satisfy_that(self):
        """The control on the comparison: a plain patch of the same width differs."""
        plain = mpatches.Patch(linewidth=style.IMPUTED_BORDER_WIDTH)
        assert plain._dash_pattern != _dotted_control()._dash_pattern


class TestTheImputedBandIsOutlinedInsideItsOwnBounds:
    """matplotlib centres a patch edge ON the path, so an outline given a band's bounds
    covers half a linewidth more on each side -- an imputed 'pc' bar would read wider than
    the identical un-imputed one beside it. Clipping the outline to its own rectangle cuts
    the outer half, which is what lets the band be marked without being widened.
    """

    def test_the_two_bands_are_given_the_same_nominal_width(self, phase_comparison):
        """Without this the equal-width assertions below would be vacuous."""
        fig = phase_comparison()
        assert len(fig.axes[1].collections) == 2
        assert _x_extent(_band_at(fig, 0.60)) == pytest.approx(_x_extent(_band_at(fig, 0.28)))

    def test_neither_fill_carries_an_outline(self, phase_comparison):
        for band in phase_comparison().axes[1].collections:
            assert float(np.max(band.get_linewidth())) == 0.0

    def test_one_dotted_rectangle_sits_on_the_imputed_band(self, phase_comparison):
        fig = phase_comparison()
        (rect,) = fig.axes[1].patches
        x0, x1, y0, y1 = _bounds(_band_at(fig, 0.60))
        assert (rect.get_x(), rect.get_width()) == pytest.approx((x0, x1 - x0))
        assert (rect.get_y(), rect.get_height()) == pytest.approx((y0, y1 - y0))
        assert rect.get_facecolor() == pytest.approx((0.0, 0.0, 0.0, 0.0))
        assert rect.get_linestyle() == style.IMPUTED_BORDER_DASHES
        assert rect.get_linewidth() == pytest.approx(style.IMPUTED_BORDER_WIDTH)
        assert tuple(rect.get_edgecolor()[:3]) == pytest.approx(to_rgb(IMPUTED_EDGE_COLOR))

    def test_the_rectangle_is_clipped_to_its_own_bounds(self, phase_comparison):
        """A Rectangle handed to ``set_clip_path`` becomes the artist's clip BOX, so the
        stroke is cut at the band's own edges rather than at the panel's."""
        fig = _drawn(phase_comparison())
        (rect,) = fig.axes[1].patches
        assert rect.get_clip_box().bounds == pytest.approx(rect.get_window_extent().bounds)
        assert rect.get_clip_box().bounds != pytest.approx(fig.axes[1].bbox.bounds)

    def test_an_un_imputed_band_gets_no_rectangle(self, phase_comparison_pair):
        """The control: a renderer outlining every band would satisfy the tests above."""
        assert list(phase_comparison_pair(imputed=False).axes[1].patches) == []

    def test_the_imputed_fill_is_tinted_and_the_control_is_not(self, phase_comparison):
        fig = phase_comparison()
        under, _ = style.imputed_band_alphas(0.5)
        assert _band_at(fig, 0.60).get_alpha() == pytest.approx(under)
        assert _band_at(fig, 0.28).get_alpha() == pytest.approx(0.5)

    def test_each_layer_takes_its_own_alpha_and_only_one_outline(
        self, phase_comparison_both_layers
    ):
        fig = phase_comparison_both_layers()
        alphas = style.imputed_band_alphas(0.5)
        imputed = [c for c in fig.axes[1].collections if _bounds(c)[0] == pytest.approx(0.60)]
        plain = [c for c in fig.axes[1].collections if _bounds(c)[0] == pytest.approx(0.28)]
        assert [c.get_alpha() for c in imputed] == pytest.approx(list(alphas))
        assert [c.get_alpha() for c in plain] == pytest.approx([0.5, 0.5])
        assert len(fig.axes[1].patches) == 1

    def test_the_minimum_width_widening_still_applies_to_both(self, monkeypatch):
        """A band narrower than the labelling minimum is widened whether or not it is
        imputed, so the two stay comparable at the narrow end too."""
        imputed, real = _imputed_entry(), _dft_entry()
        mp_phases = {imputed.name: ((0.66, 0.66), -0.30), real.name: ((0.34, 0.34), -0.20)}
        fig = _pc_figure(monkeypatch, mp_phases, -0.30, real, imputed)
        assert len(fig.axes[1].collections) == 2
        for band in fig.axes[1].collections:
            assert _x_extent(band) == pytest.approx(0.026)


class TestTheMarkedBandIsNotDrawnWider:
    """The bounds check above passes on an outline that spills; only the raster says whether
    the marked bar covers more columns than its control.
    """

    def test_the_panel_geometry_the_widths_are_read_against(self, phase_comparison_pair):
        """The figure is 8 in wide at its own dpi, its panel spanning x 0.117 to 0.909."""
        fig = phase_comparison_pair(imputed=True)
        assert _px_per_x(fig) == pytest.approx(633.6, abs=0.5)

    def test_the_marked_band_inks_exactly_the_columns_its_control_does(self, phase_comparison_pair):
        marked = _drawn_px_width(phase_comparison_pair(imputed=True), *BAND)
        plain = _drawn_px_width(phase_comparison_pair(imputed=False), *BAND)
        assert marked == plain

    def test_that_width_is_the_nominal_one(self, phase_comparison_pair):
        """The control: the equality above would hold on two blank panels."""
        fig = phase_comparison_pair(imputed=True)
        assert _drawn_px_width(fig, *BAND) == pytest.approx(
            (BAND[1] - BAND[0]) * _px_per_x(fig), abs=1.0
        )

    def test_a_band_at_the_labelling_minimum_is_equally_wide_marked(self, phase_comparison_pair):
        """The narrow end, where half a linewidth on each side is a tenth of the bar."""
        centre, half = 0.66, 0.013
        widths = [
            _drawn_px_width(
                phase_comparison_pair(imputed=flag, bounds=(centre, centre)),
                centre - half,
                centre + half,
            )
            for flag in (True, False)
        ]
        assert widths[0] == widths[1]
        assert widths[0] == pytest.approx(2 * half * 633.6, abs=1.0)


class TestThePhaseComparisonLegendIsSetAtTheFigureSize:
    """The 'pc' legend sits 4 pt under the panel labels of the same figure. Both read one
    constant, so they cannot drift apart.
    """

    def test_the_legend_reads_the_constant(self, phase_comparison):
        (text,) = phase_comparison().axes[1].get_legend().get_texts()
        assert text.get_fontsize() == binary_figs.PHASE_COMPARISON_FONTSIZE

    def test_the_panel_labels_read_the_same_constant(self, phase_comparison):
        ax1, ax2 = phase_comparison().axes
        assert ax1.yaxis.get_label().get_text() == "MPDS"
        assert ax2.yaxis.get_label().get_text() == "MP"
        for axis in (ax1.yaxis, ax2.yaxis):
            assert axis.get_label().get_fontsize() == binary_figs.PHASE_COMPARISON_FONTSIZE

    def test_the_legend_is_no_smaller_than_the_panel_label(self, phase_comparison):
        fig = phase_comparison()
        (text,) = fig.axes[1].get_legend().get_texts()
        assert text.get_fontsize() >= fig.axes[0].yaxis.get_label().get_fontsize()

    def test_the_enlarged_entry_still_sits_inside_its_panel(self, phase_comparison):
        """The figure is only 2 inches tall; a larger entry could overflow the panel."""
        fig = _drawn(phase_comparison())
        legend = fig.axes[1].get_legend().get_window_extent()
        panel = fig.axes[1].get_window_extent()
        assert panel.x0 <= legend.x0 and legend.x1 <= panel.x1
        assert panel.y0 <= legend.y0 and legend.y1 <= panel.y1


LOWER_LEFT, LOWER_RIGHT = "lower left", "lower right"
_CORNER_NAMES = {code: name for name, code in Legend.codes.items()}


@pytest.fixture
def phase_comparison_bands(monkeypatch):
    """Render the 'pc' figure over MP bands placed exactly where a case needs them.

    ``render(*bands)`` takes ``(lb, ub, mag)`` triples and keys the first on the imputed
    entry, so the key is drawn. The figure comes back UNDRAWN, as production hands it on.
    """

    def render(*bands):
        imputed = _imputed_entry()
        names = [imputed.name] + [f"Filler{i}" for i in range(1, len(bands))]
        mp_phases = {n: ((lb, ub), mag) for n, (lb, ub, mag) in zip(names, bands)}
        assert len(mp_phases) == len(bands)
        return _pc_figure(monkeypatch, mp_phases, min(mag for _, _, mag in bands), imputed)

    return render


def _legend_corner(fig):
    """The corner ``fig``'s MP-panel legend is pinned to, as a matplotlib ``loc`` string."""
    return _CORNER_NAMES[fig.axes[1].get_legend()._get_loc()]


def _band_boxes(fig):
    """Pixel boxes of the MP panel's bands, read back off the drawn fills."""
    ax = fig.axes[1]
    boxes = []
    for collection in ax.collections:
        x0, x1, y0, y1 = _bounds(collection)
        (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
        boxes.append(Bbox.from_extents(min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1)))
    return boxes


def _legend_overlap(fig):
    """Total pixel area by which the MP-panel legend covers a band, at its current corner."""
    box = _drawn(fig).axes[1].get_legend().get_window_extent()
    total = 0.0
    for other in _band_boxes(fig):
        patch = Bbox.intersection(box, other)
        if patch is not None:
            total += patch.width * patch.height
    return total


def _overlap_at(fig, corner):
    """The legend's overlap area were it pinned to ``corner``; puts it back afterwards."""
    legend = fig.axes[1].get_legend()
    was = _legend_corner(fig)
    legend.set_loc(corner)
    area = _legend_overlap(fig)
    legend.set_loc(was)
    _drawn(fig)
    return area


def _handle_text_gap(legend):
    """Pixels between the right edge of a legend's proxy patch and its label."""
    (patch,) = legend.get_patches()
    (text,) = legend.get_texts()
    return float(text.get_window_extent().x0 - patch.get_window_extent().x1)


def _default_pad_gap():
    """The same key drawn on matplotlib's default handle pads, as a control."""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor="white", edgecolor=IMPUTED_EDGE_COLOR, label=IMPUTED_LEGEND_LABEL
            )
        ],
        loc=LOWER_LEFT,
        fontsize=binary_figs.PHASE_COMPARISON_FONTSIZE,
        frameon=False,
    )
    gap = _handle_text_gap(_drawn(fig).axes[0].get_legend())
    plt.close(fig)
    return gap


class TestTheImputedKeyMovesOffABand:
    """The MP panel's bands hang DOWN from y=0 and the axis floor sits just under the deepest
    one, so a deep band at low composition lies right where a 'lower left' key is drawn. The
    key takes the bottom corner at which it covers a band least.
    """

    def test_a_deep_band_at_low_composition_pushes_it_right(self, phase_comparison_bands):
        assert _legend_corner(phase_comparison_bands((0.02, 0.30, -0.30))) == LOWER_RIGHT

    def test_and_there_it_covers_no_band_at_all(self, phase_comparison_bands):
        assert _legend_overlap(phase_comparison_bands((0.02, 0.30, -0.30))) == 0.0

    def test_the_left_corner_it_left_was_really_covered(self, phase_comparison_bands):
        """The control: without this the move could be unconditional."""
        fig = phase_comparison_bands((0.02, 0.30, -0.30))
        assert _overlap_at(fig, LOWER_LEFT) > 0.0

    def test_bands_only_at_high_composition_leave_it_on_the_left(self, phase_comparison_bands):
        assert _legend_corner(phase_comparison_bands((0.70, 0.98, -0.30))) == LOWER_LEFT

    def test_and_there_it_covers_no_band_either(self, phase_comparison_bands):
        assert _legend_overlap(phase_comparison_bands((0.70, 0.98, -0.30))) == 0.0

    def test_a_system_with_no_imputed_phase_draws_no_key(self, phase_comparison_pair):
        assert phase_comparison_pair(imputed=False).axes[1].get_legend() is None


class TestBothCornersCoveredTakesTheSmallerOverlap:
    """When both bottom corners are covered, the key goes to whichever side hides less of a
    band."""

    def test_it_moves_right_when_the_left_hides_more(self, phase_comparison_bands):
        fig = phase_comparison_bands((0.0, 0.35, -0.30), (0.93, 1.0, -0.30))
        chosen = _legend_overlap(fig)
        assert _legend_corner(fig) == LOWER_RIGHT
        assert 0.0 < chosen < _overlap_at(fig, LOWER_LEFT)

    def test_it_stays_left_when_the_right_hides_more(self, phase_comparison_bands):
        fig = phase_comparison_bands((0.0, 0.05, -0.30), (0.65, 1.0, -0.30))
        chosen = _legend_overlap(fig)
        assert _legend_corner(fig) == LOWER_LEFT
        assert 0.0 < chosen < _overlap_at(fig, LOWER_RIGHT)


class TestThePlacementNeedsNoSavedFigure:
    """The matrix build renders these panels under Agg and reads the returned figure without
    saving it first, so the corner must already be decided when it is returned.
    """

    def test_the_backend_under_test_is_the_one_the_build_uses(self):
        assert matplotlib.get_backend().lower() == "agg"

    def test_the_corner_is_decided_before_any_draw(self, phase_comparison_bands):
        fig = phase_comparison_bands((0.02, 0.30, -0.30))
        assert _legend_corner(fig) == LOWER_RIGHT


class TestTheImputedKeyHandleIsSquare:
    def test_one_constant_sizes_both_handle_axes(self, phase_comparison):
        legend = phase_comparison().axes[1].get_legend()
        assert legend.handlelength == style.IMPUTED_LEGEND_HANDLE_SIZE
        assert legend.handleheight == style.IMPUTED_LEGEND_HANDLE_SIZE

    def test_it_shrinks_the_width_to_the_default_height(self):
        assert style.IMPUTED_LEGEND_HANDLE_SIZE == matplotlib.rcParams["legend.handleheight"]
        assert style.IMPUTED_LEGEND_HANDLE_SIZE < matplotlib.rcParams["legend.handlelength"]

    def test_the_drawn_patch_is_square(self, phase_comparison):
        (handle,) = _drawn(phase_comparison()).axes[1].get_legend().get_patches()
        box = handle.get_window_extent()
        assert box.width == pytest.approx(box.height, abs=0.5)

    def test_a_default_handle_would_not_be(self):
        """The control: on matplotlib's defaults the same patch is drawn nearly 3:1."""
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.legend(handles=[mpatches.Patch(label=IMPUTED_LEGEND_LABEL)], loc=LOWER_LEFT)
        (handle,) = _drawn(fig).axes[0].get_legend().get_patches()
        box = handle.get_window_extent()
        plt.close(fig)
        assert abs(box.width - box.height) > 0.5


class TestTheImputedKeyLabelSitsCloser:
    def test_the_shipped_pad_is_under_the_matplotlib_default(self):
        assert style.IMPUTED_LEGEND_HANDLE_TEXT_PAD < 0.8
        assert style.IMPUTED_LEGEND_HANDLE_TEXT_PAD < matplotlib.rcParams["legend.handletextpad"]

    def test_the_key_is_built_with_it(self, phase_comparison):
        legend = phase_comparison().axes[1].get_legend()
        assert legend.handletextpad == style.IMPUTED_LEGEND_HANDLE_TEXT_PAD

    def test_the_drawn_gap_shrank_against_a_default_legend(self, phase_comparison):
        legend = _drawn(phase_comparison()).axes[1].get_legend()
        assert _handle_text_gap(legend) < _default_pad_gap()
