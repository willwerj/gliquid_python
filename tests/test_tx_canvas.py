"""plot_tx places its labels for the TxCanvas it is drawn at."""

import copy
import itertools

import pytest

from gliquid.binary import BinaryLiquid
from gliquid.plotting.binary_tx import (
    _PLOT_PX,
    TxCanvas,
    _estimate_label_box,
    _plot_tx_inputs,
    plot_tx,
)

SMALL = TxCanvas(width=480, height=360, margin=(52, 12, 24, 50), label_font=9, liquid_font=11)


@pytest.fixture(scope="module")
def hf_zr_figures():
    """Hf-Zr drawn at the default canvas and at SMALL, each from its own copy of the hsx."""
    bl = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
    )
    bl.update_phase_points()
    hsx, shared = _plot_tx_inputs(bl)
    return {
        canvas: plot_tx(copy.deepcopy(hsx), components=bl.components, canvas=canvas, **shared)
        for canvas in (None, SMALL)
    }


def label_boxes(fig, canvas):
    """``(text, (cx, cy, half_w, half_h))`` of each phase label of ``fig`` drawn at ``canvas``."""
    width, height = canvas.plot_px
    xlim, ylim = tuple(fig.layout.xaxis.range), tuple(fig.layout.yaxis.range)
    out = []
    for a in fig.layout.annotations:
        if a.xref == "paper" or a.text == "L":
            continue
        x, y = a.x, a.y
        if a.showarrow:
            x += (a.ax or 0) * (xlim[1] - xlim[0]) / width
            y -= (a.ay or 0) * (ylim[1] - ylim[0]) / height
        label = {
            "x": x,
            "y": y,
            "text": a.text,
            "textangle": a.textangle or 0,
            "font_size": canvas.label_font,
            "xanchor": a.xanchor or "center",
            "yanchor": a.yanchor or "middle",
        }
        out.append((a.text, _estimate_label_box(label, xlim, ylim, canvas.plot_px)))
    return out, ylim


def misfits(fig, canvas):
    """Labels of ``fig`` that overlap another or reach below the plot floor at ``canvas``."""
    boxes, ylim = label_boxes(fig, canvas)
    found = [
        (a, b)
        for (a, box_a), (b, box_b) in itertools.combinations(boxes, 2)
        if abs(box_a[0] - box_b[0]) < box_a[2] + box_b[2]
        and abs(box_a[1] - box_b[1]) < box_a[3] + box_b[3]
    ]
    return found + [(text,) for text, box in boxes if box[1] - box[3] < ylim[0]]


def test_label_box_scales_with_the_plot_area():
    label = {"x": 50.0, "y": 750.0, "text": "Xx", "textangle": -90}
    half = (_PLOT_PX[0] / 2, _PLOT_PX[1] / 2)
    *_, half_w, half_h = _estimate_label_box(label, (0.0, 100.0), (0.0, 1500.0))
    *_, small_w, small_h = _estimate_label_box(label, (0.0, 100.0), (0.0, 1500.0), half)
    assert (small_w, small_h) == pytest.approx((2 * half_w, 2 * half_h))


def test_the_figure_takes_the_canvas_size_and_fonts(hf_zr_figures):
    fig = hf_zr_figures[SMALL]
    margin = fig.layout.margin
    assert (fig.layout.width, fig.layout.height) == (480, 360)
    assert (margin.l, margin.r, margin.t, margin.b) == (52, 12, 24, 50)
    sizes = {a.text: a.font.size for a in fig.layout.annotations if a.xref != "paper"}
    assert sizes.pop("L") == 11
    assert sizes and set(sizes.values()) == {9}


def test_labels_fit_the_canvas_they_are_placed_for(hf_zr_figures):
    assert misfits(hf_zr_figures[None], SMALL)
    assert not misfits(hf_zr_figures[SMALL], SMALL)
