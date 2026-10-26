"""Bit-exact pins of the decomposition seams on the cached systems.

Frozen by freeze_decomposition_pins.py into fixtures/decomposition_pins.json, which calls
:func:`compute_pins` below. Every value is compared in the ``canon`` form of
``tests/test_decomposition_seams.py``. Pinned:

  * ``invariants``: ``mpds.identify_invariant_points`` on deep copies of each system's
    inputs -- the invariants, the low-temperature phases in order with every key, the
    digitized liquidus after the call, and ``full_comp_ss``;
  * ``fit_probe``: the constraint equations (``sympy.srepr``), ``unsupported_comp_ranges``
    and ``skip_reason`` ``fit_parameters`` reaches before optimisation, with
    ``check_solid_coverage`` True and False;
  * ``coverage``: ``BinaryLiquid.assess_solid_coverage()`` after ``find_invariant_points``;
  * ``figures``: every trace (type, name, mode, line colour / dash / width, fill, x, y),
    every annotation, the y range, the legend, ``hsx.conds`` after the call and the tie
    lines ``record_tie_lines`` collects, for Cu-Mg ``fit+liq`` and ``pred``, Hf-Zr through
    ``plot_tx``, and Hf-Zr with solid solutions ``fit+liq``;
  * ``figure_summaries``: trace rows, annotation texts and y range (``capture_figure`` of
    ``_mpds_builders.py``) of the default Hf-Zr ``fit+liq`` and ``pred``; then, on the same
    plotter as each system's ``fit+liq`` and ``pred``, the ``ch+g`` trace names, types and size
    and ``build_phase_color_map`` of the hsx phases, for Cu-Mg and Hf-Zr; and ``plot_tx`` with a
    ternary colour override on a fresh instance of each.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

import gliquid.mpds as mpds
from gliquid.binary import (
    BinaryLiquid,
    BLPlotter,
    build_phase_color_map,
    build_polymorph_transitions,
    plot_tx,
    record_tie_lines,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests._mpds_builders import capture_figure as figure_summary  # noqa: E402
from tests._mpds_builders import matches_pin  # noqa: E402
from tests.test_decomposition_seams import canon, fit_probe  # noqa: E402

pytestmark = pytest.mark.pins

PINS_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decomposition_pins.json"

SYSTEMS = {
    "Ag-V": {},
    "Al-Cu": {},
    "Bi-La": {},
    "C-Nb": {},
    "Cr-Eu": {},
    "Cu-Mg": {},
    "Hf-Zr": {"pd_ind": 0},
    "Pb-Rb": {},
}

INVARIANT_CASES = {
    "Ag-V": ("Ag-V", {}),
    "Al-Cu": ("Al-Cu", {}),
    "Bi-La": ("Bi-La", {}),
    "C-Nb": ("C-Nb", {}),
    "Cr-Eu": ("Cr-Eu", {}),
    "Cu-Mg@t_tol=15": ("Cu-Mg", {"t_tol": 15}),
    "Hf-Zr@check_full_ss=True": ("Hf-Zr", {"check_full_ss": True}),
    "Hf-Zr@check_full_ss=False": ("Hf-Zr", {"check_full_ss": False}),
    "Pb-Rb": ("Pb-Rb", {}),
}

FIGURES = ("Cu-Mg:fit+liq", "Cu-Mg:pred", "Hf-Zr:plot_tx", "Hf-Zr:ss:fit+liq")
FIGURE_SUMMARIES = (
    "Cu-Mg:ch+g",
    "Cu-Mg:color_map",
    "Cu-Mg:tx_override",
    "Hf-Zr:fit+liq",
    "Hf-Zr:pred",
    "Hf-Zr:ch+g",
    "Hf-Zr:color_map",
    "Hf-Zr:tx_override",
)


def _system(name):
    return BinaryLiquid.from_cache(name, param_format="linear", **SYSTEMS[name])


def run_invariants(system, **call_kwargs):
    """``identify_invariant_points`` on deep copies of a fresh instance's inputs.

    Returns:
        tuple: the three return values and the digitized liquidus after the call.
    """
    bl = _system(system)
    liquidus = copy.deepcopy(bl.digitized_liq)
    kwargs = {"t_tol": 15, "check_full_ss": True}
    kwargs.update(call_kwargs)
    invariants, low_t, full_comp_ss = mpds.identify_invariant_points(
        copy.deepcopy(bl.mpds_json),
        list(bl.components),
        liquidus,
        bl.component_data,
        copy.deepcopy(bl.temp_range),
        liquidus_holes=copy.deepcopy(bl.liquidus_holes),
        **kwargs,
    )
    return invariants, low_t, full_comp_ss, liquidus


def capture_invariants(system, **call_kwargs):
    invariants, low_t, full_comp_ss, liquidus = run_invariants(system, **call_kwargs)
    return canon(
        {
            "invariants": invariants,
            "low_t_exp_phases": low_t,
            "digitized_liq_after": liquidus,
            "full_comp_ss": full_comp_ss,
        }
    )


def capture_fit_probe(system, check_solid_coverage):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return fit_probe(
            monkeypatch,
            _system(system),
            full_srepr=True,
            check_solid_coverage=check_solid_coverage,
        )


def capture_coverage(system):
    bl = _system(system)
    bl.find_invariant_points()
    return canon(bl.assess_solid_coverage())


def capture_figure(fig, hsx, ties):
    traces = []
    for tr in fig.data:
        line = getattr(tr, "line", None)
        traces.append(
            {
                "type": tr.type,
                "name": getattr(tr, "name", None),
                "mode": getattr(tr, "mode", None),
                "line_color": getattr(line, "color", None) if line is not None else None,
                "line_dash": getattr(line, "dash", None) if line is not None else None,
                "line_width": getattr(line, "width", None) if line is not None else None,
                "fill": getattr(tr, "fill", None),
                "x": getattr(tr, "x", None),
                "y": getattr(tr, "y", None),
            }
        )
    annotations = [
        {
            "text": a.text,
            "x": a.x,
            "y": a.y,
            "ax": a.ax,
            "ay": a.ay,
            "xanchor": a.xanchor,
            "yanchor": a.yanchor,
            "textangle": a.textangle,
            "showarrow": a.showarrow,
        }
        for a in fig.layout.annotations
    ]
    yrange = fig.layout.yaxis.range
    return canon(
        {
            "traces": traces,
            "annotations": annotations,
            "yaxis_range": list(yrange) if yrange is not None else None,
            "legend": fig.layout.legend.to_plotly_json(),
            "conds": list(hsx.conds),
            "ties": ties,
        }
    )


def capture_chg(fig):
    """Trace count, ``[name, type]`` per trace, width and height of a ``ch+g`` figure."""
    return {
        "n_traces": len(fig.data),
        "trace_name_types": [[getattr(tr, "name", None), tr.type] for tr in fig.data],
        "width": fig.layout.width,
        "height": fig.layout.height,
    }


def capture_summaries(system, plotter, build, plot_types):
    """``figure_summaries`` entries for ``system``, keyed ``'<system>:<entry>'``.

    Summaries of ``plot_types``, then the ``ch+g`` figure and the colour map, on ``plotter``;
    then ``plot_tx`` with a ternary colour override on a fresh ``build()``.
    """
    out = {plot_type: figure_summary(plotter.get_plot(plot_type)) for plot_type in plot_types}
    out["ch+g"] = capture_chg(plotter.get_plot("ch+g"))
    out["color_map"] = build_phase_color_map(plotter._bl.hsx.phases)
    bl = build()
    bl.update_phase_points()
    first_solid = next(p for p in bl.hsx.phases if p != "L")
    override = {"L": "black", first_solid: "red"}
    out["tx_override"] = {
        "map": override,
        "figure": figure_summary(plot_tx(bl.hsx, ternary_color_map=override)),
    }
    return {f"{system}:{key}": canon(value) for key, value in out.items()}


def _cu_mg():
    return BinaryLiquid.from_cache(
        "Cu-Mg", params=[-30000.0, 5.0, -8000.0, 2.0], param_format="linear"
    )


def _hf_zr():
    return BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)


def capture_figures():
    """``figures`` and ``figure_summaries``, in the order plot_tx's mutations require."""
    out = {}
    plotter = BLPlotter(_cu_mg())
    for plot_type in ("fit+liq", "pred"):
        with record_tie_lines() as ties:
            fig = plotter.get_plot(plot_type)
        out[f"Cu-Mg:{plot_type}"] = capture_figure(fig, plotter._bl.hsx, ties)
    summaries = capture_summaries("Cu-Mg", plotter, _cu_mg, ())

    hf_zr = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
    )
    hf_zr.update_phase_points()
    with record_tie_lines() as ties:
        fig = plot_tx(
            hf_zr.hsx,
            polymorph_transitions=build_polymorph_transitions(hf_zr),
            components=hf_zr.components,
        )
    out["Hf-Zr:plot_tx"] = capture_figure(fig, hf_zr.hsx, ties)

    hf_zr_ss = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    plotter = BLPlotter(hf_zr_ss)
    with record_tie_lines() as ties:
        fig = plotter.get_plot("fit+liq")
    out["Hf-Zr:ss:fit+liq"] = capture_figure(fig, plotter._bl.hsx, ties)

    summaries.update(capture_summaries("Hf-Zr", BLPlotter(_hf_zr()), _hf_zr, ("fit+liq", "pred")))
    return {"figures": out, "figure_summaries": summaries}


def compute_pins():
    """Every capture, keyed by group then case."""
    return {
        "invariants": {
            name: capture_invariants(system, **kwargs)
            for name, (system, kwargs) in sorted(INVARIANT_CASES.items())
        },
        "fit_probe": {
            f"{system}:check_solid_coverage={flag}": capture_fit_probe(system, flag)
            for system in sorted(SYSTEMS)
            for flag in (True, False)
        },
        "coverage": {system: capture_coverage(system) for system in sorted(SYSTEMS)},
        **capture_figures(),
    }


@pytest.fixture(scope="module")
def pins():
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def figures():
    return capture_figures()


@pytest.mark.parametrize("name", sorted(INVARIANT_CASES))
def test_invariants_pinned(pins, name):
    system, kwargs = INVARIANT_CASES[name]
    assert capture_invariants(system, **kwargs) == pins["invariants"][name]


@pytest.mark.parametrize("name", sorted(INVARIANT_CASES))
def test_latent_l5_cmp_pass_retypes_the_returned_phase(name):
    """Each cmp invariant's phase appears in the returned low-temperature list with
    type 'cmp'."""
    system, kwargs = INVARIANT_CASES[name]
    invariants, low_t, _, _ = run_invariants(system, **kwargs)
    for inv in invariants:
        if inv["type"] != "cmp":
            continue
        matches = [p for p in low_t if p["name"] == inv["phases"][0] and p["comp"] == inv["comp"]]
        assert matches, inv
        assert all(p["type"] == "cmp" for p in matches)


@pytest.mark.parametrize("system", sorted(SYSTEMS))
@pytest.mark.parametrize("check_solid_coverage", [True, False])
def test_fit_probe_pinned(pins, system, check_solid_coverage):
    key = f"{system}:check_solid_coverage={check_solid_coverage}"
    assert capture_fit_probe(system, check_solid_coverage) == pins["fit_probe"][key]


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_coverage_pinned(pins, system):
    assert capture_coverage(system) == pins["coverage"][system]


@pytest.mark.parametrize("name", FIGURES)
def test_figure_pinned(pins, figures, name):
    assert matches_pin(figures["figures"][name], pins["figures"][name]), name


@pytest.mark.parametrize("name", FIGURE_SUMMARIES)
def test_figure_summary_pinned(pins, figures, name):
    assert figures["figure_summaries"][name] == pins["figure_summaries"][name]
