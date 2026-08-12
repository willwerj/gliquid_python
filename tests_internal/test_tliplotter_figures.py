"""Characterization pins for the TLIPlotter figure / hull-slice API.

Freezes the figure structure of ``plot_ternary`` and the hull-slice data + figure of
``extract_single_hull_at_T`` for the Al-Mg-Si fixture, so the composition refactor
(``get_plot('tx')`` / ``get_plot('ch')`` / ``get_convex_hull``) provably reproduces them.
The pinned VALUES are frozen by scratchpad/freeze_tliplotter_figures.py against the
pre-refactor class; only the construction/method names adapt across the refactor.

Runs offline (shipped Al-Mg-Si MP entries).
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PIN = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tliplotter_figure_pins.json"
    ).read_text()
)
C = PIN["construction"]


@pytest.fixture(scope="module")
def plotter():
    tmod = pu.get_ternary_mod()
    p = tmod.TLIPlotter.from_components(
        list(C["components"]),
        delta=C["delta"],
        T_incr=C["T_incr"],
        xs_mix={k: list(v) for k, v in C["xs_mix"].items()},
    )
    p.process_data()
    return p


def test_tx_figure_structure(plotter):
    fig = plotter.get_plot("tx")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == PIN["tx_n_traces"]
    assert sorted(t.type for t in fig.data) == PIN["tx_trace_types"]


def test_convex_hull_slice_data(plotter):
    hull = plotter.get_convex_hull(PIN["ch_request_c"])
    assert "figure" not in hull  # figure moved to get_plot('ch')
    assert hull["temperature_c"] == pytest.approx(PIN["ch_temperature_c"])
    assert hull["temperature_k"] == pytest.approx(PIN["ch_temperature_k"])
    assert len(hull["hull_points"]) == PIN["ch_n_points"]
    assert pu.canonical_simplices(hull["hull_simplices"]) == PIN["ch_canonical_simplices"]


def test_convex_hull_slice_figure(plotter):
    fig = plotter.get_plot("ch", T_celsius=PIN["ch_request_c"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == PIN["ch_figure_n_traces"]
    assert sorted(t.type for t in fig.data) == PIN["ch_figure_trace_types"]
