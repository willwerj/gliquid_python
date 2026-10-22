"""The tx figure draws each solid phase as one vertical segment per composition it occupies."""

import numpy as np
import pandas as pd
import pytest

from gliquid.plotting.ternary_surface import render_tx_surface

FLOOR = -200.0
COLOR_MAP = {"L": "#000000", "AB": "#ff0000", "FCC": "#00aaaa"}

# Rows in the order TLIPlotter._plot_tx hands them over: every ceiling sorted by T, then the
# floor rows. FCC is a solution phase stable at three compositions; AB is a line compound.
SOLID_ROWS = [
    (0.20, 0.10, 650.0, "FCC"),
    (0.50, 0.30, 700.0, "AB"),
    (0.80, 0.05, 900.0, "FCC"),
    (0.40, 0.20, 1100.0, "FCC"),
    (0.20, 0.10, FLOOR, "FCC"),
    (0.50, 0.30, FLOOR, "AB"),
    (0.80, 0.05, FLOOR, "FCC"),
    (0.40, 0.20, FLOOR, "FCC"),
]


@pytest.fixture(scope="module")
def fig():
    solid = pd.DataFrame(SOLID_ROWS, columns=["x0", "x1", "T", "Phase"])
    solid["Colors"] = solid["Phase"].map(COLOR_MAP)
    liq = pd.DataFrame(
        {
            "x0": [0.0, 0.5, 1.0],
            "x1": [0.0, np.sqrt(3) / 2, 0.0],
            "T": [1200.0, 1300.0, 1400.0],
            "x0_orig": [0.0, 0.0, 1.0],
            "x1_orig": [0.0, 1.0, 0.0],
            "coexistent_phases": ["FCC", "AB", "FCC"],
        }
    )
    return render_tx_surface(
        solid, liq, np.array([[0, 1, 2]]), COLOR_MAP, ["A", "B", "C"], [FLOOR, 1500.0]
    )


def _runs(trace):
    """The trace's polylines, split where it breaks on ``None``."""
    runs, run = [], []
    for point in zip(trace.x, trace.y, trace.z):
        if point[0] is None:
            runs.append(run)
            run = []
        else:
            run.append(point)
    return [*runs, run]


def _solid_trace(fig, phase):
    (trace,) = [t for t in fig.data if t.hovertemplate and f"Phase: {phase}<" in t.hovertemplate]
    return trace


@pytest.mark.parametrize("phase", ["FCC", "AB"])
def test_each_composition_is_its_own_vertical_segment(fig, phase):
    expected = {(x0, x1): T for x0, x1, T, p in SOLID_ROWS if p == phase and T != FLOOR}
    segments = {}
    for run in _runs(_solid_trace(fig, phase)):
        assert len({(x, y) for x, y, _ in run}) == 1, f"{phase} line crosses compositions: {run}"
        segments[run[0][:2]] = sorted(z for _, _, z in run)
    assert segments == {xy: [FLOOR, T] for xy, T in expected.items()}
