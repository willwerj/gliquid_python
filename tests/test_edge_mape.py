"""The MAPE chip on fitted binary edge figures: its color scale, its placement, and its value.

``mape_color`` must reproduce the interactive matrix's MAPE coloring (matplotlib ``RdYlGn``
reversed over 0-40 % through the matrix's MAPE knots, channels truncated to integers), so the
reference mapper below is a verbatim copy of the matrix's ``CMapper`` rather than a
re-derivation.
"""

import math

import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib import colormaps

from gliquid.binary import BinaryLiquid
from gliquid.plotting import binary_tx
from gliquid.plotting.binary_tx import CORNER_NOTE_NAME, add_corner_note
from gliquid.plotting.style import (
    ASSESSED_LIQUIDUS_COLOR,
    MAPE_COLOR_KNOTS,
    MAPE_COLOR_MAX,
    contrast_text,
    mape_color,
)
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter, build_binary_edge_figures

FIT_PARAMS = [-15000.0, 0.0, 0.0, 0.0]
# The interactive matrix's MAPE knots, copied verbatim.
MATRIX_MAPE_KNOTS = ((0, 0.0), (5, 0.08), (20, 0.70), (40, 1.0))
XS_MIX = {"Hf-Ti": [0.0] * 4, "Ti-Zr": [0.0] * 4, "Zr-Hf": FIT_PARAMS}
FIT_OR_PRED = {"Zr-Hf": "fit"}
ALL_PREDICTED = {"Zr-Hf": "pred"}


class _MatrixCMapper:
    """The interactive matrix's color mapper, copied verbatim."""

    def __init__(self, cmap, low=0.0, high=1.0, knots=None):
        self.cmap = cmap
        self.low = low
        self.high = high
        self.knots = knots

    def __call__(self, val):
        if self.knots is None:
            normalized_value = (val - self.low) / (self.high - self.low)
        else:
            values, fractions = zip(*self.knots)
            normalized_value = float(np.interp(val, values, fractions))
        norm_rgba = list(self.cmap(1 - normalized_value))
        for c in range(3):
            norm_rgba[c] = int(norm_rgba[c] * 255)
        return tuple(norm_rgba)


def _rgb(text: str) -> tuple[int, int, int]:
    return tuple(int(c) for c in text[text.index("(") + 1 : text.index(")")].split(","))


def _notes(fig) -> list:
    return [a for a in fig.layout.annotations if a.name == CORNER_NOTE_NAME]


class TestMapeColor:
    @pytest.mark.parametrize("mape", [i * 0.5 for i in range(121)])
    def test_it_matches_the_matrix_mapper(self, mape):
        expected = _MatrixCMapper(
            colormaps["RdYlGn"], high=MAPE_COLOR_MAX, knots=MATRIX_MAPE_KNOTS
        )(mape)[:3]
        assert _rgb(mape_color(mape)) == expected

    def test_zero_is_the_green_end_and_the_cap_the_red_end(self):
        green, red = _rgb(mape_color(0.0)), _rgb(mape_color(MAPE_COLOR_MAX))
        assert green[1] > green[0] and red[0] > red[1]

    def test_the_knots_are_the_matrix_knots(self):
        assert MAPE_COLOR_KNOTS == MATRIX_MAPE_KNOTS

    def test_values_past_the_cap_clip_to_the_red_end(self):
        assert mape_color(250.0) == mape_color(MAPE_COLOR_MAX)


class TestContrastText:
    def test_the_pale_middle_of_the_scale_gets_black(self):
        assert contrast_text(mape_color(MAPE_COLOR_MAX / 2)) == "black"

    def test_the_dark_ends_get_white(self):
        assert contrast_text(mape_color(0.0)) == "white"
        assert contrast_text(mape_color(MAPE_COLOR_MAX)) == "white"

    def test_pure_white_and_black(self):
        assert contrast_text("rgb(255,255,255)") == "black"
        assert contrast_text("rgb(0,0,0)") == "white"


class TestAddCornerNote:
    def _plain(self):
        fig = go.Figure(go.Scatter(x=[0, 100], y=[500, 1500]))
        fig.add_annotation(x=50, y=1000, text="L", showarrow=False)
        return fig

    def test_it_adds_exactly_one_named_annotation_and_nothing_else(self):
        plain, noted = self._plain(), add_corner_note(self._plain(), "MAPE 4.4%", mape_color(4.4))
        assert noted.layout.annotations[:-1] == plain.layout.annotations
        assert noted.data == plain.data
        (note,) = _notes(noted)
        assert note is noted.layout.annotations[-1]
        assert note.text == "MAPE 4.4%"

    def test_it_sits_above_the_plot_frame_at_the_left(self):
        (note,) = _notes(add_corner_note(self._plain(), "MAPE 4.4%", mape_color(4.4)))
        assert (note.xref, note.yref, note.x, note.y) == ("paper", "paper", 0, 1)
        assert (note.xanchor, note.yanchor) == ("left", "bottom")
        assert note.showarrow is False

    def test_the_chip_is_the_fill_and_the_text_contrasts(self):
        fill = mape_color(30.0)
        (note,) = _notes(add_corner_note(self._plain(), "MAPE 30.0%", fill))
        assert note.bgcolor == fill
        assert note.font.color == contrast_text(fill)

    def test_it_clears_the_centred_title_of_a_standalone_figure(self):
        """Estimated in px with the plot stack's own glyph metrics, for the widest note."""
        title = "Xx-Yy DFT-Referenced Phase Diagram"
        title_left = binary_tx._FIG_W_PX / 2 - len(title) * binary_tx._CHAR_W_FACTOR * 18 / 2
        note_w = len("MAPE 100.0%") * binary_tx._CHAR_W_FACTOR * 13 + 2 * 3
        note_right = binary_tx._MARGIN_L_PX + note_w
        assert note_right < title_left


@pytest.fixture(scope="module")
def hf_ti_zr():
    return TernaryLiquidInterpolation(
        ["Hf", "Ti", "Zr"],
        delta=0.25,
        xs_mix=XS_MIX,
        T_incr=100,
        hull_mode="hsx",
        order="given",
        fit_or_pred=FIT_OR_PRED,
        param_format="linear",
    )


@pytest.fixture(scope="module")
def noted_figures(hf_ti_zr):
    figs = build_binary_edge_figures(hf_ti_zr, edge_mape=True)
    return dict(zip(hf_ti_zr.binary_systems, figs))


class TestEdgeFigures:
    def test_only_the_fitted_edge_carries_a_chip(self, noted_figures):
        assert {edge: len(_notes(fig)) for edge, fig in noted_figures.items()} == {
            "Hf-Ti": 0,
            "Ti-Zr": 0,
            "Zr-Hf": 1,
        }

    def test_the_chip_shows_the_drawn_edges_mape(self, hf_ti_zr, noted_figures):
        bl = BinaryLiquid.from_cache(
            "Hf-Zr",
            params=FIT_PARAMS,
            param_format=hf_ti_zr.param_format,
            pd_ind=0,
            hull=hf_ti_zr.edge_hull("Hf-Zr"),
        )
        bl.update_phase_points()
        bl.init_error = False
        mape = bl.calculate_deviation_metrics(
            allow_sparse_data=True, unsupported_range_mode="score"
        )[2]
        assert math.isfinite(mape), "a non-finite MAPE would make this vacuous"
        (note,) = _notes(noted_figures["Zr-Hf"])
        assert note.text == f"MAPE {mape:.1f}%"
        assert note.bgcolor == mape_color(mape)

    def test_without_the_flag_no_figure_carries_a_chip(self, hf_ti_zr, noted_figures):
        plain = dict(zip(hf_ti_zr.binary_systems, build_binary_edge_figures(hf_ti_zr)))
        assert all(not _notes(fig) for fig in plain.values())
        for edge, fig in plain.items():
            noted = noted_figures[edge]
            assert noted.data == fig.data
            assert [a for a in noted.layout.annotations if a.name != CORNER_NOTE_NAME] == list(
                fig.layout.annotations
            )

    def test_the_plotter_threads_the_flag(self):
        plotter = TLIPlotter.from_components(
            ["Hf", "Ti", "Zr"],
            order="given",
            edge_mape=True,
            delta=0.25,
            xs_mix=XS_MIX,
            T_incr=100,
            hull_mode="hsx",
            fit_or_pred=FIT_OR_PRED,
            param_format="linear",
        )
        assert plotter.edge_mape is True
        plotter._init_sys()
        assert [len(_notes(fig)) for fig in plotter.bin_fig_list] == [0, 0, 1]

    def test_the_plotter_defaults_to_no_chip(self, hf_ti_zr):
        assert TLIPlotter(hf_ti_zr, order="given").edge_mape is False


def _assessed(fig) -> list:
    return [t for t in fig.data if getattr(t.line, "color", None) == ASSESSED_LIQUIDUS_COLOR]


@pytest.fixture(scope="module")
def all_predicted():
    return TernaryLiquidInterpolation(
        ["Hf", "Ti", "Zr"],
        delta=0.25,
        xs_mix=XS_MIX,
        T_incr=100,
        hull_mode="hsx",
        order="given",
        fit_or_pred=ALL_PREDICTED,
        param_format="linear",
    )


class TestPredictedLiquidus:
    def test_a_predicted_edge_draws_no_liquidus_by_default(self, all_predicted):
        figs = build_binary_edge_figures(all_predicted, edge_mape=True)
        assert all(not _assessed(fig) and not _notes(fig) for fig in figs)

    def test_a_predicted_edge_with_a_digitized_liquidus_draws_it(self, all_predicted):
        figs = dict(
            zip(
                all_predicted.binary_systems,
                build_binary_edge_figures(all_predicted, edge_mape=True, pred_liquidus=True),
            )
        )
        assert {edge: bool(_assessed(fig)) for edge, fig in figs.items()} == {
            "Hf-Ti": False,
            "Ti-Zr": False,
            "Zr-Hf": True,
        }
        assert {edge: len(_notes(fig)) for edge, fig in figs.items()} == {
            "Hf-Ti": 0,
            "Ti-Zr": 0,
            "Zr-Hf": 1,
        }

    def test_its_chip_shows_the_predicted_edges_mape(self, all_predicted):
        bl = BinaryLiquid.from_cache(
            "Hf-Zr",
            params=FIT_PARAMS,
            param_format=all_predicted.param_format,
            pd_ind=0,
            hull=all_predicted.edge_hull("Hf-Zr"),
        )
        bl.update_phase_points()
        bl.init_error = False
        mape = bl.calculate_deviation_metrics(
            allow_sparse_data=True, unsupported_range_mode="score"
        )[2]
        assert math.isfinite(mape)
        figs = build_binary_edge_figures(all_predicted, edge_mape=True, pred_liquidus=True)
        (note,) = _notes(figs[all_predicted.binary_systems.index("Zr-Hf")])
        assert note.text == f"MAPE {mape:.1f}%"

    def test_the_predicted_liquidus_keeps_the_predicted_colour(self, all_predicted):
        plain = build_binary_edge_figures(all_predicted)
        drawn = build_binary_edge_figures(all_predicted, pred_liquidus=True)
        i = all_predicted.binary_systems.index("Zr-Hf")
        assert [t for t in drawn[i].data if t not in _assessed(drawn[i])] == list(plain[i].data)

    def test_the_plotter_threads_the_flag(self):
        plotter = TLIPlotter.from_components(
            ["Hf", "Ti", "Zr"],
            order="given",
            pred_liquidus=True,
            delta=0.25,
            xs_mix=XS_MIX,
            T_incr=100,
            hull_mode="hsx",
            fit_or_pred=ALL_PREDICTED,
            param_format="linear",
        )
        assert plotter.pred_liquidus is True
        plotter._init_sys()
        assert [bool(_assessed(fig)) for fig in plotter.bin_fig_list] == [False, False, True]

    def test_the_plotter_defaults_to_pred_only(self, all_predicted):
        assert TLIPlotter(all_predicted, order="given").pred_liquidus is False


class TestProgress:
    def test_each_edge_reports_once_in_order(self, hf_ti_zr):
        calls = []
        build_binary_edge_figures(hf_ti_zr, progress=lambda *args: calls.append(args))
        assert calls == [("edges", 1, 3), ("edges", 2, 3), ("edges", 3, 3)]

    def test_process_data_reports_model_edges_then_hull(self):
        plotter = TLIPlotter.from_components(
            ["Hf", "Ti", "Zr"],
            order="given",
            delta=0.25,
            xs_mix=XS_MIX,
            T_incr=100,
            hull_mode="hsx",
            fit_or_pred=FIT_OR_PRED,
            param_format="linear",
        )
        calls = []
        plotter.process_data(progress=lambda *args: calls.append(args))
        assert calls == [
            ("model", 1, 1),
            ("edges", 1, 3),
            ("edges", 2, 3),
            ("edges", 3, 3),
            ("hull", 1, 1),
        ]
