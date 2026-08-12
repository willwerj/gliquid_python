"""Characterization pins for the non-SS binary FIGURE surface (BLPlotter + plot_tx).

Frozen by dev/scripts/_scratch/freeze_binary_figure_pins.py into
fixtures/binary_figure_pins.json ahead of the plot-stack export (now gliquid/plotting/binary_tx.py),
the HSX color export, and the SS/non-SS plot unification. These pins are the
behavior-identity gate for those steps: figure trace structure, colors, annotation
texts, axis ranges, and the HSX color_map must reproduce EXACTLY.

Capture order matters (plot_tx mutates hsx.phase_color_remap across calls):
fit+liq -> pred -> ch+g -> color_map on one instance; the ternary_color_map
override on a FRESH instance. The capture helpers here MUST stay byte-equivalent
to the freezer's.
"""

import json
import sys
from pathlib import Path

import pytest

from gliquid.binary import BinaryLiquid, BLPlotter, build_phase_color_map, plot_tx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PINS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "binary_figure_pins.json"
    ).read_text()
)

SYSTEMS = {
    "CuMg": lambda: BinaryLiquid.from_cache(
        "Cu-Mg", params=[-30000.0, 5.0, -8000.0, 2.0], param_format="linear"
    ),
    "HfZr": lambda: BinaryLiquid.from_cache("Hf-Zr", pd_ind=0),
}


def _trace_row(tr):
    line = getattr(tr, "line", None)
    x = getattr(tr, "x", None)
    return [
        tr.type,
        getattr(tr, "mode", None),
        getattr(tr, "name", None),
        getattr(line, "color", None) if line is not None else None,
        getattr(line, "dash", None) if line is not None else None,
        (len(x) if x is not None else None),
    ]


def capture_figure(fig):
    anns = sorted(a.text for a in (fig.layout.annotations or ()) if a.text is not None)
    yr = fig.layout.yaxis.range
    return {
        "n_traces": len(fig.data),
        "traces": [_trace_row(tr) for tr in fig.data],
        "annotations": anns,
        "yaxis_range": list(yr) if yr is not None else None,
    }


def capture_chg(fig):
    return {
        "n_traces": len(fig.data),
        "trace_name_types": [[getattr(tr, "name", None), tr.type] for tr in fig.data],
        "width": fig.layout.width,
        "height": fig.layout.height,
    }


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def sys_capture(request):
    key = request.param
    ctor = SYSTEMS[key]
    plotter = BLPlotter(ctor())
    # 'color_map' was frozen from the retired HSX.__init__ color_map; the live side is
    # its exported successor, build_phase_color_map — the pin IS the parity oracle.
    cap = {
        "fit_liq": capture_figure(plotter.get_plot("fit+liq")),
        "pred": capture_figure(plotter.get_plot("pred")),
        "ch_g": capture_chg(plotter.get_plot("ch+g")),
        "color_map": build_phase_color_map(plotter._bl.hsx.phases),
    }

    bl2 = ctor()
    bl2.update_phase_points()
    first_solid = next(p for p in bl2.hsx.phases if p != "L")
    override = {"L": "black", first_solid: "red"}
    cap["tx_override"] = {
        "map": override,
        "figure": capture_figure(plot_tx(bl2.hsx, ternary_color_map=override)),
    }
    return key, cap


def test_fit_liq_figure_pinned(sys_capture):
    key, cap = sys_capture
    pu.assert_deep_approx(PINS[key]["fit_liq"], cap["fit_liq"])


def test_pred_figure_pinned(sys_capture):
    key, cap = sys_capture
    pu.assert_deep_approx(PINS[key]["pred"], cap["pred"])


def test_ch_g_figure_pinned(sys_capture):
    key, cap = sys_capture
    pu.assert_deep_approx(PINS[key]["ch_g"], cap["ch_g"])


def test_hsx_color_map_pinned(sys_capture):
    key, cap = sys_capture
    assert cap["color_map"] == PINS[key]["color_map"]


def test_ternary_color_override_pinned(sys_capture):
    key, cap = sys_capture
    assert cap["tx_override"]["map"] == PINS[key]["tx_override"]["map"]
    pu.assert_deep_approx(PINS[key]["tx_override"]["figure"], cap["tx_override"]["figure"])
