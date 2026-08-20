"""Binary T-X phase-diagram rendering: the module-level plot stack over an HSX instance.

Exported from gliquid.binary: label layout + legend placement helpers, the plot_tx
renderer, and polymorph-transition annotations. Colors and display-name rules live in
gliquid.plotting.style. Pure presentation over an HSX instance + dataframes -- no
model imports (import order:
hsx <- plotting.binary_tx <- binary). plot_tx and _place_legend mutate hsx.conds in
place (pinned behavior).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gliquid.plotting.style import (
    ASSESSED_LIQUIDUS_COLOR,
    LIQUID_COLOR,
    PREDICTED_LIQUIDUS_COLOR,
    build_phase_color_map,
    format_phase_display_name,
    subscript_formula,
)

logger = logging.getLogger(__name__)


# --- Solid-solution envelope geometry -------------------------------------------------
# A continuous SS field is drawn as a closed, hatch-filled polygon. Both of its boundaries
# come from the same branch split, so an upper and a lower boundary can never disagree
# about where a composition gap falls (the Hf-Y / Ru-Y / Hf-W / Cr-W disconnection class).
_SS_PINCH_FRAC = 0.05  # terminal spread <= this fraction of the widest -> single apex
_SS_EDGE_STEPS = 1.5  # a terminus this close (in grid steps) to 0/100 is an axis end
_SS_APEX_STEPS = 3.0  # cap on how far past the terminus an extrapolated apex may land
_SS_EXTREME_STEPS = 0.75  # composition tolerance when matching a tie to an extremum
_SS_LINE_W = 2.4  # outline width, uniform around the whole ring
_SS_HATCH_SIZE_PX = 9.0
_SS_HATCH_SOLIDITY = 0.22
# _SS_TIE_T_FRAC is retained alongside the region anchors it belongs to; the extremum
# admission it sized was removed from _ss_tie_allowed and nothing reads it now.
_SS_TIE_T_FRAC = 0.005  # temperature tolerance when matching a tie to a field vertex
# _SS_TIE_FRAME_FRAC is load-bearing: it is admission rule 2 of
# _ss_minimum_tie_allowed -- a field clipped by the bottom of the plotted range has an
# artificial minimum sitting there, not a physical one.
_SS_TIE_FRAME_FRAC = 0.025  # how close to the plotted T range counts as "at its edge"
# Half-width of the "at the minimum" temperature window used when reading the assemblage
# on either side of it, as a fraction of the plotted span. The facets bracketing a
# eutectoid can be a few hundredths of a kelvin away (Y-Zr: 768.0015 -> 768.0507), so this
# only has to exclude the invariant's OWN facets, never a neighbouring equilibrium.
_SS_MIN_T_EPS_FRAC = 1e-6
# Three-phase invariants put a participating solution phase at its solubility limit by
# definition, so they bypass the SS admission filter entirely.
_SS_TIE_ALWAYS_KEYS = ("Eutectics", "Peritectics")
# The inv_points keys plot_tx draws ties from, in the order hsx.py emits them.
_TIE_INV_KEYS = ("Eutectics", "Peritectics", "Misc Gaps", "Solid Ties")
# The label hsx.py gives the liquid phase. An invariant carrying two of these is an
# L1+L2 (monotectic) horizontal, which is admitted regardless of any SS field it touches.
_LIQUID_LABEL = "L"
# Minimum separation (at.%) between the two solid vertices of an L + S1 + S2 invariant for
# them to be two distinct compositions of the field rather than adjacent samples of one
# continuous boundary. Mirrors the width gate hsx.liquidus_invariants already applies to a
# 'Misc Gaps' entry (comp_diff > 0.012 mole fraction), which for this topology measures the
# LIQUID-to-solid distance and so never rejects a collapsed solid pair.
_SS_TIE_SOLID_GAP_PCT = 1.2
# Two sources can emit the SAME physical horizontal with slightly different extents (an
# invariant tie and the polymorph/safety-net tie at the same temperature -- Cu-Ir). Ties
# within this fraction of the plotted temperature span whose composition spans touch or
# overlap are merged into one trace spanning their union. Kept small: merging genuinely
# distinct invariants that happen to sit close in temperature is the failure mode.
_TIE_MERGE_T_FRAC = 0.005  # ~10 K on a 2000 K diagram


# --- Tie-line instrumentation ---------------------------------------------------------
# Read-only diagnostics tooling needs to know WHICH source emitted each drawn tie.
# While a sink is installed, plot_tx appends one mutable record per DRAWN tie --
# ``{'temp', 'x0', 'x1', 'sources'}`` -- and a merge updates the
# record already there rather than adding another, so the sink always mirrors fig.data.
# Production never installs one; the hook is inert when it is None.
_TIE_SINK: list | None = None


@contextmanager
def record_tie_lines(sink: list | None = None):
    """Collect every tie line ``plot_tx`` draws while the block is active.

    Yields the list of records. Re-entrant (the previous sink is restored on exit) and
    exception-safe, so an instrumented render can never leak the hook into later plots.
    """
    global _TIE_SINK
    sink = [] if sink is None else sink
    previous = _TIE_SINK
    _TIE_SINK = sink
    try:
        yield sink
    finally:
        _TIE_SINK = previous


def _grid_step(x_vals: np.ndarray) -> float:
    """Median positive composition spacing of a sampled boundary (1 at.% fallback).

    A LAST-RESORT inference. The composition grid is a property of the hull sampling, not
    of one phase's stability range, so prefer the caller-supplied ``grid_step`` that
    :func:`_ss_regions` and :func:`_split_indices` both accept.

    Two or fewer samples cannot evidence a grid at all: the only spacing present is the
    candidate GAP itself. Mo-Y's and W-Y's BCC fields are sampled at exactly 0.2 and 100
    at.%, and taking 99.8 as the "step" made :func:`_split_indices`' 1.5x rule
    unsatisfiable, welding two terminal branches into one diagram-spanning quadrilateral.
    So a two-sample array is capped at the 1 at.% base composition grid -- it may report a
    FINER spacing (Hf-W's HCP field really is sampled at 0.2 and 0.4), never a coarser one.
    """
    if x_vals.size <= 1:
        return 1.0
    dx = np.diff(x_vals)
    positive = dx[dx > 0]
    if not positive.size:
        return 1.0
    step = float(np.median(positive))
    return min(step, 1.0) if x_vals.size <= 2 else step


def _hull_grid_step(x_pct: np.ndarray) -> float:
    """The composition grid step of a hull, read off its full sampled composition axis.

    The FINEST spacing present, not the median. A presentation hull refines only its
    SOLUTION phases (``BinaryLiquid.refined_hsx``), so wherever no solution phase is stable
    the axis falls back to the coarse 1 at.% liquid grid -- and a median then reports that
    instead of the grid the solution field was actually sampled on. Both Mo-Y (median 1.0)
    and Hf-Y (median 0.2) come off the same 0.2 at.% presentation grid.

    Rounded before differencing: the coarse and refined grids are built by different
    arithmetic (``np.arange(0, 1.01, 0.01)`` against ``np.arange(n + 1) / n``), so a shared
    node can appear twice a few ulps apart and an unrounded minimum would return ~1e-16.
    """
    unique = np.unique(np.round(np.asarray(x_pct, dtype=float), 6))
    if unique.size < 2:
        return 1.0
    dx = np.diff(unique)
    positive = dx[dx > 0]
    return float(positive.min()) if positive.size else 1.0


def _split_indices(
    x_vals: np.ndarray, gap_threshold: float | None = None, *, grid_step: float | None = None
) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` index spans of the contiguous runs in a sorted x array.

    A run breaks wherever the composition spacing jumps past 1.5 grid steps -- the single
    definition of "phase-boundary gap" used by every SS boundary, so the upper and lower
    boundaries of one field are always cut at the same indices.

    ``grid_step`` (at.%) supplies that grid spacing from the caller, which measured it once
    over the whole hull. Omitted, it is inferred from ``x_vals`` by :func:`_grid_step` --
    correct for a densely sampled boundary, degenerate for a sparsely sampled one.

    ``gap_threshold`` (at.%) overrides the grid-relative rule outright, and WINS over
    ``grid_step``, for curves whose sampling is not on the fitting grid. The digitized
    liquidus is the one such curve: it is sampled at whatever spacing the diagram was
    traced at, so a grid-relative threshold would cut it at every mildly uneven patch. It
    passes :data:`_ASSESSED_GAP_PCT` instead.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    if x_vals.size == 0:
        return []
    if x_vals.size == 1:
        return [(0, 1)]
    if gap_threshold is None:
        step = _grid_step(x_vals) if grid_step is None else float(grid_step)
        gap_threshold = max(1.5 * step, 0.8)
    spans, start = [], 0
    for idx in range(1, x_vals.size):
        if x_vals[idx] - x_vals[idx - 1] > gap_threshold:
            spans.append((start, idx))
            start = idx
    spans.append((start, x_vals.size))
    return spans


def _extrapolate_to(
    x_vals: np.ndarray, y_vals: np.ndarray, x_target: float, from_left: bool
) -> float:
    """Linear extension of a boundary to ``x_target`` from its two outermost points."""
    if x_vals.size < 2:
        return float(y_vals[0])
    if from_left:
        x1, y1, x2, y2 = x_vals[0], y_vals[0], x_vals[1], y_vals[1]
    else:
        x1, y1, x2, y2 = x_vals[-1], y_vals[-1], x_vals[-2], y_vals[-2]
    run = float(x2 - x1)
    if abs(run) < 1e-12:
        return float(y1)
    slope = (float(y2) - float(y1)) / run
    return float(y1) + slope * (float(x_target) - float(x1))


def _match_edge_interval(
    edge_intervals: dict | None, edge: float, t_lo_end: float, t_hi_end: float
) -> tuple[float, float] | None:
    """The component polymorph a solid-solution branch degenerates into at ``edge``.

    ``edge_intervals`` maps a composition-axis edge to the ``(t_lo, t_hi)`` stability
    intervals of the non-solution solid phases sitting there -- for Hf-Y's x=0 that is
    alpha-Hf [733.4, 1742.9] and beta-Hf [1742.9, 2232.8]. The branch's own temperature
    extent at its outermost sampled composition picks between them by overlap, so no phase
    name or spacegroup has to be parsed and a bcc branch cannot be snapped onto an hcp
    ground state. Returns None when nothing overlaps, leaving extrapolation in charge.
    """
    candidates = (edge_intervals or {}).get(edge) or ()
    lo_end, hi_end = min(t_lo_end, t_hi_end), max(t_lo_end, t_hi_end)
    best, best_overlap = None, 0.0
    for c_lo, c_hi in candidates:
        overlap = min(hi_end, c_hi) - max(lo_end, c_lo)
        if overlap > best_overlap:
            best, best_overlap = (float(c_lo), float(c_hi)), overlap
    return best


def _x_at_temperature(
    xs: np.ndarray, vals: np.ndarray, target: float, from_left: bool
) -> float | None:
    """Composition at which a boundary reaches ``target``, extended from its two ends."""
    if xs.size < 2:
        return None
    (i0, i1) = (0, 1) if from_left else (-1, -2)
    x0, y0, x1, y1 = float(xs[i0]), float(vals[i0]), float(xs[i1]), float(vals[i1])
    if abs(x0 - x1) < 1e-12 or abs(y0 - y1) < 1e-12:
        return None
    return x0 + (target - y0) * (x0 - x1) / (y0 - y1)


def _apex_from_invariant(
    invariants,
    x_end: float,
    t_lo_end: float,
    t_hi_end: float,
    xs: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    from_left: bool,
    step: float,
) -> tuple[float, float] | None:
    """Close a blunt terminus on the three-phase invariant that terminates the field.

    Maximum solubility occurs AT the eutectic/peritectic, so when one of those sits on the
    blunt face the field's corner is that invariant's temperature -- the composition grid
    just cannot reach the composition. Hf-Y's BCC face spans 1326.3-1488.4 C at x=21.4 and
    its eutectic is at 1330.7 C, so the drawn field asserts a maximum solubility it never
    locates.

    The apex temperature is the invariant's. Its composition comes from extending whichever
    boundary ENDS NEAREST that temperature: at Hf-Y's terminus the solvus is 4.4 K away and
    near-linear (~35 K/at.%) while the solidus is 158 K away and plunging (-236 K/at.% over
    the last step), so extrapolating the near one is both the shortest reach and by far the
    better conditioned. Returns None when no invariant sits on the face or the reach is
    implausible, leaving the blunt terminus alone.
    """
    if not invariants:
        return None
    lo_t, hi_t = min(t_lo_end, t_hi_end), max(t_lo_end, t_hi_end)
    on_face = [
        t
        for cx, t in invariants
        if abs(cx - x_end) <= _SS_EDGE_STEPS * step and lo_t - 1e-9 <= t <= hi_t + 1e-9
    ]
    if not on_face:
        return None
    apex_t = min(on_face, key=lambda t: min(abs(t - lo_t), abs(t - hi_t)))

    near_lo = abs(apex_t - t_lo_end) <= abs(apex_t - t_hi_end)
    apex_x = _x_at_temperature(xs, lo if near_lo else hi, apex_t, from_left)
    if apex_x is None or not np.isfinite(apex_x):
        return None
    outward = -1.0 if from_left else 1.0
    reach = (apex_x - x_end) * outward
    if reach < -1e-9 or reach > _SS_APEX_STEPS * step:
        return None
    return float(apex_x), float(apex_t)


def _ss_terminus(
    xs: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    from_left: bool,
    edge: float,
    step: float,
    widest: float,
    edge_intervals: dict | None = None,
    apex_invariants: list | None = None,
) -> dict:
    """Close one end of an SS branch.

    Three outcomes, in the order the phase diagram makes them: an end within
    ``_SS_EDGE_STEPS`` of a composition-axis edge is extended to the axis and contributes
    two on-edge vertices; an end whose band has pinched shut contributes a single apex
    vertex at the extrapolated crossing of the two boundaries; anything else contributes
    two blunt vertices at the last sampled composition. Returns the vertices (bottom-first)
    plus the anchor record the tie filter matches against.
    """
    idx = 0 if from_left else -1
    x_end = float(xs[idx])
    t_lo_end, t_hi_end = float(lo[idx]), float(hi[idx])

    if abs(x_end - edge) <= _SS_EDGE_STEPS * step:
        snapped = None
        if abs(x_end - edge) > 1e-9:
            # The branch stops short of the axis, so the corner would be EXTRAPOLATED --
            # and a steep boundary overshoots the component's transition temperature
            # (Hf-Y's HCP reaches 1861 C at x=0, 118 K above the alpha->beta transition at
            # 1742.9 C, i.e. inside the beta field). At the axis a solid solution IS the
            # pure component in that structure, so snap to the stability interval of the
            # matching polymorph instead. Match by overlap rather than by name: the branch
            # rides the polymorph it degenerates into. Corners that are actually SAMPLED at
            # the axis are left exactly as the hull produced them.
            snapped = _match_edge_interval(edge_intervals, edge, t_lo_end, t_hi_end)
        if snapped is not None:
            t_lo_edge, t_hi_edge = snapped
        else:
            t_lo_edge = _extrapolate_to(xs, lo, edge, from_left)
            t_hi_edge = _extrapolate_to(xs, hi, edge, from_left)
        if t_hi_edge < t_lo_edge:  # boundaries crossed past the last sample -> pinch there
            t_lo_edge = t_hi_edge = 0.5 * (t_lo_edge + t_hi_edge)
        return {
            "vertices": [
                {"x": edge, "t": t_lo_edge, "on_edge": True, "kind": "edge"},
                {"x": edge, "t": t_hi_edge, "on_edge": True, "kind": "edge"},
            ],
            "edge_anchor": {
                "x": edge,
                "t_bottom": min(t_lo_edge, t_hi_edge),
                "t_top": max(t_lo_edge, t_hi_edge),
            },
            "vertex_anchors": [],
        }

    span = (x_end - _SS_EXTREME_STEPS * step, x_end + _SS_EXTREME_STEPS * step)
    if (t_hi_end - t_lo_end) <= max(_SS_PINCH_FRAC * widest, 1e-9):
        apex_x, apex_t = _ss_apex(xs, lo, hi, from_left=from_left, step=step)
        span = (
            min(span[0], apex_x - _SS_EXTREME_STEPS * step),
            max(span[1], apex_x + _SS_EXTREME_STEPS * step),
        )
        return {
            "vertices": [{"x": apex_x, "t": apex_t, "on_edge": False, "kind": "apex"}],
            "edge_anchor": None,
            "vertex_anchors": [
                {"x_lo": span[0], "x_hi": span[1], "t": apex_t, "x": apex_x, "kind": "apex"}
            ],
        }

    # Still blunt by shape -- but if a three-phase invariant sits on that face, IT is the
    # maximum-solubility point and the face is only a grid artifact. Close on it.
    snapped_apex = _apex_from_invariant(
        apex_invariants, x_end, t_lo_end, t_hi_end, xs, lo, hi, from_left=from_left, step=step
    )
    if snapped_apex is not None:
        apex_x, apex_t = snapped_apex
        span = (
            min(span[0], apex_x - _SS_EXTREME_STEPS * step),
            max(span[1], apex_x + _SS_EXTREME_STEPS * step),
        )
        return {
            "vertices": [{"x": apex_x, "t": apex_t, "on_edge": False, "kind": "apex"}],
            "edge_anchor": None,
            "vertex_anchors": [
                {"x_lo": span[0], "x_hi": span[1], "t": apex_t, "x": apex_x, "kind": "apex"}
            ],
        }

    # A blunt terminus is a vertical face: every temperature on it is at the maximum
    # composition, so composition alone would re-admit one tie per grid step (Hf-Y's BCC
    # field). Only its two corners count as extrema.
    return {
        "vertices": [
            {"x": x_end, "t": t_lo_end, "on_edge": False, "kind": "blunt"},
            {"x": x_end, "t": t_hi_end, "on_edge": False, "kind": "blunt"},
        ],
        "edge_anchor": None,
        "vertex_anchors": [
            {"x_lo": span[0], "x_hi": span[1], "t": t_lo_end, "x": x_end, "kind": "blunt"},
            {"x_lo": span[0], "x_hi": span[1], "t": t_hi_end, "x": x_end, "kind": "blunt"},
        ],
    }


def _ss_apex(
    xs: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, from_left: bool, step: float
) -> tuple[float, float]:
    """Where the two boundaries of a pinching branch cross, just past its last sample."""
    idx = 0 if from_left else -1
    x_end = float(xs[idx])
    mid = 0.5 * (float(lo[idx]) + float(hi[idx]))
    if xs.size < 2:
        return x_end, mid
    t_lo_probe = _extrapolate_to(xs, lo, x_end + step, from_left)
    t_hi_probe = _extrapolate_to(xs, hi, x_end + step, from_left)
    d_lo = (t_lo_probe - float(lo[idx])) / step
    d_hi = (t_hi_probe - float(hi[idx])) / step
    if abs(d_hi - d_lo) < 1e-12:
        return x_end, mid
    # solve hi[idx] + d_hi*u == lo[idx] + d_lo*u for the offset u past the terminus
    offset = (float(lo[idx]) - float(hi[idx])) / (d_hi - d_lo)
    outward = -1.0 if from_left else 1.0
    if (
        not np.isfinite(offset)
        or offset * outward < -_SS_APEX_STEPS * step
        or abs(offset) > _SS_APEX_STEPS * step
    ):
        return x_end, mid
    return x_end + offset, float(hi[idx]) + d_hi * offset


def _ss_regions(
    x_pct,
    t_lo,
    t_hi,
    xlim: tuple[float, float] = (0.0, 100.0),
    edge_intervals: dict | None = None,
    apex_invariants: list | None = None,
    grid_step: float | None = None,
) -> list[dict]:
    """Closed polygons for a solid-solution field, one per contiguous composition branch.

    ``x_pct``/``t_lo``/``t_hi`` are the per-composition lower and upper temperature extents
    of the phase's raw ``df_tx`` points (at.% and °C). Each returned region carries the
    ring (``x``/``t``, first point repeated last), its corner ``vertices`` -- 3 when one end
    pinches to an apex, 4 otherwise -- and the anchors the tie filter reads.

    ``grid_step`` (at.%) is the composition spacing the hull was sampled at, measured once
    by the caller over the whole ``df_tx`` axis (:func:`_hull_grid_step`). It sets the
    branch-splitting gap threshold AND the terminus/anchor windows, both of which are
    grid-relative. Omitted, each branch infers its own from its samples -- which cannot work
    for a field supported at two compositions, where the only spacing present is the gap.
    """
    x_pct = np.asarray(x_pct, dtype=float)
    t_lo = np.asarray(t_lo, dtype=float)
    t_hi = np.asarray(t_hi, dtype=float)
    order = np.argsort(x_pct, kind="stable")
    x_pct, t_lo, t_hi = x_pct[order], t_lo[order], t_hi[order]

    regions = []
    # ONE splitter call for the whole field, so the upper and lower boundaries can never
    # disagree about where a composition gap falls (the Hf-Y / Ru-Y / Hf-W / Cr-W
    # disconnection class). Both branches then share the same grid step.
    for start, stop in _split_indices(x_pct, grid_step=grid_step):
        xs, lo, hi = x_pct[start:stop], t_lo[start:stop], t_hi[start:stop]
        step = _grid_step(xs) if grid_step is None else float(grid_step)
        widest = float(np.max(hi - lo)) if xs.size else 0.0
        left = _ss_terminus(
            xs,
            lo,
            hi,
            from_left=True,
            edge=xlim[0],
            step=step,
            widest=widest,
            edge_intervals=edge_intervals,
            apex_invariants=apex_invariants,
        )
        right = _ss_terminus(
            xs,
            lo,
            hi,
            from_left=False,
            edge=xlim[1],
            step=step,
            widest=widest,
            edge_intervals=edge_intervals,
            apex_invariants=apex_invariants,
        )

        ring: list[tuple[float, float]] = []
        ring += [(v["x"], v["t"]) for v in left["vertices"]]  # bottom -> top
        ring += [(float(x), float(t)) for x, t in zip(xs, hi)]  # upper, left -> right
        ring += [(v["x"], v["t"]) for v in reversed(right["vertices"])]  # top -> bottom
        ring += [(float(x), float(t)) for x, t in zip(xs[::-1], lo[::-1])]  # lower, right -> left
        ring.append(ring[0])
        ring = [
            p
            for i, p in enumerate(ring)
            if i == 0 or abs(p[0] - ring[i - 1][0]) > 1e-9 or abs(p[1] - ring[i - 1][1]) > 1e-9
        ]

        # Interior temperature extrema are critical points too: the coldest point of a
        # field is where it meets the invariant that terminates it (Hf-W BCC bottoms out
        # at 1228.3 C @ x=11.8, exactly the alpha-Hf + BCC + HfW2 peritectoid; Y-Zr BCC at
        # 768.0 C @ x=91.0 and Hf-Y BCC at 733.2 C @ x=61.4, both eutectoids). Anchoring
        # only on composition extrema drops those ties.
        #
        # 'kind' names WHICH extremum an anchor marks, because they are not
        # interchangeable: 'lower_min' is the branch's coldest sampled point and the only
        # one _ss_minimum_tie_allowed will draw a horizontal at. A terminus anchor keeps
        # the terminus' own kind ('apex'/'blunt'), so a minimum that merely sits at the end
        # of the branch is never mistaken for an interior one.
        vertex_anchors = left["vertex_anchors"] + right["vertex_anchors"]
        for kind, arr, idx in (
            ("lower_min", lo, int(np.argmin(lo))),
            ("upper_max", hi, int(np.argmax(hi))),
        ):
            x_at = float(xs[idx])
            if any(a["x_lo"] <= x_at <= a["x_hi"] for a in vertex_anchors):
                continue  # already covered by a terminus anchor
            vertex_anchors.append(
                {
                    "x_lo": x_at - _SS_EXTREME_STEPS * step,
                    "x_hi": x_at + _SS_EXTREME_STEPS * step,
                    "t": float(arr[idx]),
                    "x": x_at,
                    "kind": kind,
                }
            )

        regions.append(
            {
                "x": np.array([p[0] for p in ring], dtype=float),
                "t": np.array([p[1] for p in ring], dtype=float),
                "vertices": left["vertices"] + right["vertices"],
                # 'vertex_anchors'/'edge_tol' are load-bearing: plot_tx reads the
                # 'lower_min' anchor (and 'edge_tol' as its interiority window) to place the
                # eutectoid horizontal at the branch's temperature minimum. 'edge_anchors'
                # stays fixture-only -- no production code consults it, since _ss_tie_allowed
                # still rejects every 'Misc Gaps'/'Solid Ties' entry at a composition extremum.
                "edge_anchors": [a for a in (left["edge_anchor"], right["edge_anchor"]) if a],
                "vertex_anchors": vertex_anchors,
                "edge_tol": _SS_EDGE_STEPS * step,
            }
        )
    return regions


def _ss_solid_pair_phase(comps_pct, phases, ss_regions: dict[str, list[dict]]) -> str | None:
    """The solution phase of an ``L + S1 + S2`` invariant, or ``None`` if it is not one.

    Qualifies only a three-vertex entry with exactly one liquid vertex whose other two
    vertices are the SAME solution phase at compositions that are genuinely APART -- a
    solid miscibility gap meeting the liquidus (Cr-W's ``('BCC','BCC','L')`` at ~1932 C,
    liquid at 14 at.% with the BCC pair at 28.8 and 71.2).

    The separation test is what makes the clause usable. ``hsx.liquidus_invariants`` gates
    a 'Misc Gaps' entry on ``comp_diff > 0.012``, but for this topology ``comp_diff`` is
    the LIQUID-to-nearest-solid distance -- the solid pair is never checked. So every
    slice of the ordinary two-phase L + SS band arrives here as an ``L + S1 + S2`` whose
    two "solid" vertices are adjacent samples of one continuous boundary: a collapsed
    facet, not a gap. Measured over the acceptance corpus, every such artifact sits at
    exactly one sampling step (0.20 at.%) while the one real reaction sits at 42.4 at.%,
    so the emitter's own width gate re-applied to the SOLID pair separates them with two
    orders of magnitude to spare.
    """
    comps = [float(c) for c in comps_pct]
    phases = list(phases)
    if len(phases) != 3 or len(comps) != 3:
        return None
    solids = [(p, c) for p, c in zip(phases, comps) if p != _LIQUID_LABEL]
    if len(solids) != 2 or solids[0][0] != solids[1][0]:
        return None
    if solids[0][0] not in ss_regions:
        return None
    if abs(solids[1][1] - solids[0][1]) <= _SS_TIE_SOLID_GAP_PCT:
        return None
    return solids[0][0]


def _ss_family_maxima(entries, ss_regions: dict[str, list[dict]]) -> dict[str, float]:
    """``{ss_phase: highest temperature}`` over each phase's ``L + S1 + S2`` family.

    ``entries`` is an iterable of ``(inv_key, comps_pct, phases, temp)``.
    ``hsx.liquidus_invariants`` emits one 'Misc Gaps' entry per grid slice that clears its
    width gate, and ``_collapse_gap_runs`` only thins runs, so one reaction can arrive as a
    FAMILY of near-identical horizontals. The Gibbs phase rule gives a binary three-phase
    equilibrium zero degrees of freedom, so exactly one of them is the reaction: the
    hottest, since in a binary the liquid field lies above the solid ones and the two-solid
    region terminates against it at the top.

    Entries under ``_SS_TIE_ALWAYS_KEYS`` are excluded -- those draw unconditionally, so
    letting one set the bar could suppress a genuine solvus-family maximum below it.
    """
    maxima: dict[str, float] = {}
    for inv_key, comps_pct, phases, temp in entries:
        if inv_key in _SS_TIE_ALWAYS_KEYS:
            continue
        phase = _ss_solid_pair_phase(comps_pct, phases, ss_regions)
        if phase is None:
            continue
        t = float(temp)
        if t > maxima.get(phase, -np.inf):
            maxima[phase] = t
    return maxima


def _ss_tie_allowed(
    comps_pct,
    phases,
    temp: float,
    ss_regions: dict[str, list[dict]],
    t_range: tuple[float, float],
    inv_key: str | None = None,
    ss_family_max: dict[str, float] | None = None,
) -> bool:
    """Whether an invariant that touches a solid-solution field earns a tie line.

    Four admissions, in the order they matter:

      1. an invariant touching no SS field at all is passed through untouched -- this
         filter only ever speaks about solution phases;
      2. a eutectic or peritectic is a three-phase invariant, so a participating SS phase
         sits AT its maximum solubility there by definition -- always drawn;
      3. an invariant carrying two liquid vertices is an L1+L2 monotectic. Its remaining
         vertex is a SOLID, which in a system like Sc-V is the solution phase itself
         (['L', 'L', 'BCC'] at 1459 C), so any test that asks the SS field's geometry to
         justify the tie rejects the one horizontal that marks the miscibility gap. The
         liquid pair is what makes it an invariant, not the solid it terminates on --
         always drawn;
      4. an invariant with ONE liquid vertex and two SEPARATED compositions of the same
         solution phase is an L + S1 + S2 -- a solid miscibility gap meeting the liquidus
         (Cr-W's ~1932 C horizontal). Also invariant by the phase rule, so it is drawn,
         but only the EXTREMAL (hottest) member of its family: the emitter hands one
         reaction over as many grid slices, and admitting all of them restores the fan
         this filter exists to stop. ``ss_family_max`` carries the per-phase maximum
         computed once by the caller (``_ss_family_maxima``); without it the clause cannot
         fire and behavior is exactly as it was before the clause existed.

    Everything else -- a 'Misc Gaps' solvus or a 'Solid Ties' pair touching an SS field --
    is dropped. Those mark no invariant: they are one grid-pair slice per temperature step
    along a CONTINUOUS boundary (Cr-W emits 75, Hf-W 64), and the field's own hatched
    polygon already draws where that boundary runs. Admitting them at the field's
    composition extrema, as this function used to, put a tie line where no two-phase
    equilibrium ends.

    ``comps_pct``/``temp``/``t_range`` are read by clause 4 only: ``comps_pct`` measures
    the solid-pair separation (see ``_ss_solid_pair_phase``), and the family maximum is
    matched within ``_TIE_MERGE_T_FRAC`` of the plotted temperature span -- the same window
    below which ``_add_tie`` would merge two ties into one trace anyway, so being exact to
    the grid step buys nothing.
    """
    if not any(p in ss_regions for p in phases):
        return True
    if inv_key in _SS_TIE_ALWAYS_KEYS:
        return True
    if sum(1 for p in phases if p == _LIQUID_LABEL) >= 2:
        return True
    if not ss_family_max:
        return False
    phase = _ss_solid_pair_phase(comps_pct, phases, ss_regions)
    if phase is None or phase not in ss_family_max:
        return False
    tol = _TIE_MERGE_T_FRAC * (float(t_range[1]) - float(t_range[0]))
    return float(temp) >= float(ss_family_max[phase]) - tol


def _facet_assemblages(df_tx) -> list[tuple[float, float, float, frozenset]]:
    """``(temp, x_lo, x_hi, phase set)`` per hull facet, read off ``hsx.df_tx``.

    Each lower-hull facet of a binary is a three-vertex simplex whose coexistence
    temperature ``compute_tx`` writes onto all three of its rows in one go, so ``df_tx``
    is exactly one facet per CONSECUTIVE TRIPLE of rows -- and a facet is precisely one
    tie line: the phases in equilibrium between ``x_lo`` and ``x_hi`` at ``temp``. That is
    the only reading of "what coexists here" the hull itself supports; the drawn geometry
    knows where boundaries run but not what lies between them.

    The triple grouping is VERIFIED, not assumed: a row count that is not a multiple of
    three, or any triple whose three temperatures disagree, means the frame is not the
    binary facet table this reads (an n-component hull, a filtered copy) and an empty list
    is returned. Every caller treats that as "the assemblage is unknown" and draws nothing,
    so a mis-shaped frame can never invent an invariant.

    Phases come back as a SET: a facet carrying the same label at two compositions is one
    phase at both ends of its own two-phase field (a solvus slice), not two phases.
    """
    try:
        n = len(df_tx)
        if not n:
            return []  # no facets is not a mis-shaped frame
        if n % 3 or "x" not in df_tx.columns:
            logger.debug(
                "Facet table is not a binary triple frame (%d rows, columns %s); "
                "no assemblages read, so no derived ties are drawn.",
                n,
                list(getattr(df_tx, "columns", ())),
            )
            return []
        x = df_tx["x"].to_numpy(dtype=float).reshape(-1, 3) * 100.0
        t = df_tx["t"].to_numpy(dtype=float).reshape(-1, 3)
        labels = df_tx["label"].to_numpy().reshape(-1, 3)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Facet table could not be read as a binary triple frame (%s); "
            "no assemblages read, so no derived ties are drawn.",
            exc,
        )
        return []
    if np.any(np.abs(t - t[:, :1]) > 1e-9):
        logger.debug(
            "Facet triples disagree on temperature; the frame is not the binary "
            "facet table, so no derived ties are drawn."
        )
        return []
    return [
        (float(t[i, 0]), float(x[i].min()), float(x[i].max()), frozenset(str(p) for p in labels[i]))
        for i in range(t.shape[0])
    ]


def _assemblage_across(
    assemblages, x_pct: float, temp: float, *, above: bool, temp_eps: float
) -> frozenset | None:
    """The phases coexisting at ``x_pct`` on one side of ``temp``, or ``None``.

    The nearest facet in that direction whose composition span brackets ``x_pct``: in a
    binary the facets are dense inside a two-phase field (one per composition step), so the
    closest one above/below a horizontal names the field on that side of it. Facets within
    ``temp_eps`` of ``temp`` are the horizontal's own and belong to neither side.

    ``None`` means the hull has nothing on that side at that composition -- the assemblage
    is unknown there, which is never evidence that it changed.
    """
    best, best_t = None, None
    for t, x_lo, x_hi, phases in assemblages:
        if not (x_lo - 1e-9 <= x_pct <= x_hi + 1e-9):
            continue
        if above:
            if t <= temp + temp_eps or (best_t is not None and t >= best_t):
                continue
        else:
            if t >= temp - temp_eps or (best_t is not None and t <= best_t):
                continue
        best, best_t = phases, t
    return best


def _ss_minimum_anchor(region: dict) -> dict | None:
    """A branch's interior lower-boundary minimum anchor, or ``None`` if it has none.

    ``_ss_regions`` records at most one per branch, and skips it entirely when the coldest
    sample falls inside a terminus anchor's window -- a minimum AT the end of the branch is
    the terminus, and the invariant that closes a terminus is already drawn by
    ``_SS_TIE_ALWAYS_KEYS``. So "at most one tie per field" is structural here, not a cap
    applied afterwards: however many grid slices the invariant emitter offers along the
    same boundary, this reads one anchor.
    """
    for anchor in region.get("vertex_anchors") or ():
        if anchor.get("kind") == "lower_min":
            return anchor
    return None


def _ss_minimum_tie_allowed(
    anchor: dict | None,
    region: dict,
    assemblages,
    xlim: tuple[float, float],
    t_range: tuple[float, float],
) -> bool:
    """Whether a solid-solution branch's temperature minimum is a real invariant.

    A eutectoid IS the minimum of the parent field's lower boundary, and it is the one
    member of the ``Misc Gaps`` / ``Solid Ties`` fan that marks an actual reaction. Three
    admissions, the first of them the deciding one:

      1. **The coexisting assemblage changes across it.** Read the hull's own facets just
         above and just below the minimum temperature at the minimum composition and
         require the two phase sets to differ. Y-Zr's BCC bottoms out at 768.0 C @ 91.0
         at.% with HCP+HCP below and BCC+HCP above -- a reaction. Sc-V's Sc-side BCC
         sliver bottoms out at 1330.6 C @ 0.4 at.% with BCC+HCP on BOTH sides -- a field
         pinching to an apex inside one two-phase region, which is no invariant at all and
         earns no horizontal. This is the whole physical content of the rule; 2 and 3 only
         throw out minima that are artifacts of WHERE THE FIELD WAS CUT, not of what it does.
      2. **The minimum is interior.** Within ``edge_tol`` (the branch's own
         ``_SS_EDGE_STEPS`` window) of either composition axis it is not a minimum of the
         field at all -- the boundary is still running off the edge, where the solution
         degenerates into the pure component and its unary polymorphic transition. Ti-V's
         BCC lower boundary is monotonic from the Ti-side HCP->BCC transition down to its
         V-side end at 99.8 at.%, one grid step from the axis: rejected here.
      3. **It is not at the plotted temperature floor.** A field the bottom of the frame
         cuts off has its coldest SAMPLE there rather than its coldest point.
    """
    if not anchor:
        return False
    x_at, t_at = float(anchor["x"]), float(anchor["t"])
    span = float(t_range[1]) - float(t_range[0])

    eps = _SS_MIN_T_EPS_FRAC * span if span > 0 else 1e-9
    below = _assemblage_across(assemblages, x_at, t_at, above=False, temp_eps=eps)
    above = _assemblage_across(assemblages, x_at, t_at, above=True, temp_eps=eps)
    if below is None or above is None or below == above:
        return False

    edge_tol = float(region.get("edge_tol") or _SS_EDGE_STEPS)
    if min(x_at - float(xlim[0]), float(xlim[1]) - x_at) <= edge_tol:
        return False

    return t_at > float(t_range[0]) + _SS_TIE_FRAME_FRAC * span


def _split_segments(
    x_vals: np.ndarray, y_vals: np.ndarray, gap_threshold: float | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a polyline into segments wherever the x spacing jumps (phase-boundary gaps)."""
    segments = [(x_vals[a:b], y_vals[a:b]) for a, b in _split_indices(x_vals, gap_threshold)]
    if not segments:
        return [(x_vals, y_vals)]
    return segments


# Assessed (digitized) liquidus break, in at.%. mpds.extract_digitized_liquidus densifies
# every gap INSIDE a digitized liquid region down to <= 3 at.% spacing and refuses to fill
# across the hole between two disjoint 'L' regions, so any surviving gap wider than
# mpds._FILL_GAP_X is an undigitized hole and must not be drawn through. Kept as a literal
# because this module takes no model imports; test_multi_l_liquidus pins the two in sync.
_ASSESSED_GAP_PCT = 6.0
# A gap sitting exactly ON the fill threshold is legitimate in-region sampling that the
# extractor declined to densify (Hf-Zr's widest is 5.96 at.%). x*100 - x'*100 can land a
# few ulps above 6.0 for such a pair, so compare with slack -- real holes are 10+ at.%.
_ASSESSED_GAP_EPS = 1e-6


def _assessed_liquidus_segments(points) -> list[list[list[float]]]:
    """Digitized liquidus (at.%, degC) split into the runs that are actually digitized.

    One segment per contiguous liquid region, so a connect-the-dots trace never draws a
    phantom liquidus across a hole no 'L' shape covers (Bi-Si, Ag-Fe, Er-Ta, Ir-Th).
    """
    if not points:
        return []
    xs = np.asarray([p[0] for p in points], dtype=float)
    ts = np.asarray([p[1] for p in points], dtype=float)
    return [
        [[float(x), float(t)] for x, t in zip(seg_x, seg_t)]
        for seg_x, seg_t in _split_segments(xs, ts, _ASSESSED_GAP_PCT + _ASSESSED_GAP_EPS)
        if seg_x.size
    ]


# ======================================================================================
# Binary TX plot stack (migrated from gliquid/hsx.py: plot-geometry constants, label
# helpers, and plot_tx as a module function over an HSX instance). hsx.py keeps the
# dimension-agnostic hull/compute machinery only.
# ======================================================================================

# ---------------------------------------------------------------------------
# Plotting geometry constants (must match the layout used in HSX.plot_tx).
# These are the single source of truth for the pixel<->data conversions used
# by label-collision detection; the tests import them from this module.
# ---------------------------------------------------------------------------
_FIG_W_PX = 750
_FIG_H_PX = 600
_MARGIN_L_PX = 80  # plotly default left margin
_MARGIN_R_PX = 55
_MARGIN_T_PX = 72
_MARGIN_B_PX = 72
_PLOT_W_PX = _FIG_W_PX - _MARGIN_L_PX - _MARGIN_R_PX  # 615
_PLOT_H_PX = _FIG_H_PX - _MARGIN_T_PX - _MARGIN_B_PX  # 456

# Text-metric estimates (fractions of the font size, in px).
_CHAR_W_FACTOR = 0.6  # mean glyph advance width
_LINE_H_FACTOR = 1.2  # line height
_SUB_W_FACTOR = 0.7  # subscript glyph relative width

# Consistent gap between a bottom-anchored label's box and the lower plot boundary,
# as a fraction of the temperature span. Keeps compound labels off the axis line.
_BOTTOM_PAD_FRAC = 0.015

# Greek polymorph prefixes -> unicode letters.
_GREEK_MAP = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
}

# Long crystal-structure names -> short forms.
_STRUCT_ABBREV = {
    "orthorhombic": "ortho",
    "rhombohedral": "rhomb",
    "monoclinic": "mono",
    "tetragonal": "tetra",
    "hexagonal": "hex",
    "trigonal": "trig",
    "triclinic": "tric",
    "body centered cubic": "bcc",
    "face centered cubic": "fcc",
    "hexagonal close packed": "hcp",
    "complex cubic a12": "A12",
    "complex cubic a13": "A13",
    "diamond cubic": "dc",
    "graphite": "gra",
}

_SUB_TAG_RE = re.compile(r"<sub>(.*?)</sub>")
_TAG_PHASE_TAG_RE = r"(?:rt|lt|ht\d*|r\d*)"  # room/low/high-temperature ordinal tags


def _text_glyph_width(text: str) -> float:
    """Estimate visible text width in glyph units (chars), discounting subscripts and HTML tags."""
    # Mark subscript bodies, then strip remaining tags, then weight characters.
    marked = _SUB_TAG_RE.sub(lambda m: "\x00" * len(m.group(1)), text)
    marked = re.sub(r"<[^>]+>", "", marked)
    return sum(_SUB_W_FACTOR if ch == "\x00" else 1.0 for ch in marked)


def _estimate_label_box(
    label: dict, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Estimate a label's bounding box in data coordinates.

    Returns ``(cx, cy, half_w, half_h)`` where (cx, cy) is the box centre. This is the
    single source of truth for the fragile font/pixel constants; tests import it directly.
    """
    font_size = label.get("font_size", 12)
    glyphs = _text_glyph_width(str(label.get("text", "")))
    w_px = max(glyphs * _CHAR_W_FACTOR * font_size, 0.5 * font_size)
    h_px = _LINE_H_FACTOR * font_size
    if abs(label.get("textangle", 0)) == 90:
        w_px, h_px = h_px, w_px

    span_x = (xlim[1] - xlim[0]) or 1.0
    span_y = (ylim[1] - ylim[0]) or 1.0
    half_w = (w_px / (_PLOT_W_PX / span_x)) / 2.0
    half_h = (h_px / (_PLOT_H_PX / span_y)) / 2.0

    cx, cy = float(label["x"]), float(label["y"])
    xanchor = label.get("xanchor", "center")
    if xanchor == "left":
        cx += half_w
    elif xanchor == "right":
        cx -= half_w
    yanchor = label.get("yanchor", "middle")
    if yanchor == "bottom":
        cy += half_h
    elif yanchor == "top":
        cy -= half_h
    return cx, cy, half_w, half_h


def _box_overlap(b1: tuple, b2: tuple) -> tuple[float, float]:
    """Return positive (x, y) penetration depths if AABBs ``b1``/``b2`` overlap, else <=0."""
    return (b1[2] + b2[2] - abs(b1[0] - b2[0]), b1[3] + b2[3] - abs(b1[1] - b2[1]))


def _parse_elemental_phase(name: str):
    """Parse an elemental solid-solution label into ``(greek_word, element, struct, tag)``.

    Accepts both space- and hyphen-separated greek prefixes (e.g. ``"alpha Mn (bcc)"`` and
    ``"alpha-Ga (orthorhombic)"``). Returns ``None`` for compounds (e.g. ``"ZrMn2"``) so the
    caller falls through to formula subscripting.
    """
    s = str(name).strip()
    greek_word = None
    struct_prefix = None
    # "<Structure words> <Element>" form (e.g. "Diamond cubic Si", "Face centered cubic Al"):
    # a capitalised structure name preceding a trailing element symbol. The element is the last
    # token; everything before it is the structure. Only fires when the leading token is a
    # capitalised, non-greek word (so "alpha Mn" stays a greek polymorph and a hypothetical
    # "Na Cl" — an element-symbol prefix — is not misread as element Cl).
    toks = s.split()
    if (
        len(toks) >= 2
        and re.fullmatch(r"[A-Z][a-z]?", toks[-1])
        and toks[0][:1].isupper()
        and toks[0].lower() not in _GREEK_MAP
        and not re.fullmatch(r"[A-Z][a-z]?", " ".join(toks[:-1]))
    ):
        return None, toks[-1], " ".join(toks[:-1]), None
    # A leading token before a space/hyphen is either a greek polymorph prefix
    # ("alpha-Ga (orthorhombic)") or a lowercase structure tag ("fcc-Al", "bcc-Cr"),
    # provided what follows starts with an element symbol.
    m = re.match(r"^([A-Za-z0-9]+)[ \-](.+)$", s)
    if m and re.match(r"^[A-Z][a-z]?(?:$|[\s(])", m.group(2).strip()):
        prefix = m.group(1)
        if prefix.lower() in _GREEK_MAP:
            greek_word = prefix.lower()
            s = m.group(2).strip()
        elif prefix[0].islower():
            struct_prefix = prefix
            s = m.group(2).strip()

    m2 = re.match(r"^([A-Z][a-z]?)\s*(?:\(([^)]*)\))?\s*(.*)$", s)
    if not m2:
        return None
    element = m2.group(1)
    struct = (m2.group(2) or "").strip() or struct_prefix
    tag = (m2.group(3) or "").strip() or None
    # Reject compounds: ANY trailing remainder that is not a temperature tag means this is
    # a formula, not an elemental phase -- "ZrMn2" parses as element "Zr" + remainder
    # "Mn2", and "Ce(FeSi)2" as element "Ce" + "structure" FeSi + remainder "2".
    #
    # The remainder used to be rejected only when it contained LETTERS, which let the bare
    # stoichiometric "2" of a parenthesised ternary compound through and rendered
    # Ce(FeSi)2 as "Ce (FeSi) 2". No binary system has a parenthesised formula, so this
    # only ever surfaced once the ternary legend started using this formatter.
    if tag and not re.fullmatch(_TAG_PHASE_TAG_RE, tag, re.IGNORECASE):
        return None
    return greek_word, element, struct, tag


def _abbrev_structure(struct: str) -> str:
    """Abbreviate a crystal-structure name; pass short names through, drop unrecognised long ones.

    Returns ``''`` for long, unmapped descriptors (e.g. space-group symbols like ``P6_3/mmc``)
    so the label falls back to just ``(El)``/``(αEl)`` rather than showing truncated noise.
    """
    s = struct.strip()
    key = s.lower()
    if key in _STRUCT_ABBREV:
        return _STRUCT_ABBREV[key]
    m = re.search(r"\b([A-D]\d{1,2})\b", s)  # Strukturbericht token, e.g. A13, B2
    if m:
        return m.group(1)
    if re.search(r"\d", s) and ("-" in s or "/" in s):
        return ""  # space-group symbol (Fm-3m, R-3m, P6_3/mmc) -> omit
    if len(s) <= 6:
        return s
    return ""  # unrecognised long descriptor -> omit


#: Shared with the ternary stack via plotting.style; kept under the old private name
#: so binary.py's re-export and the call site below need no edit.
_subscript_formula = subscript_formula


def _abbreviate_phase_name(name: str, all_names: list[str]) -> str:
    """Format a phase label: greek polymorph prefixes, subscripted stoichiometries,
    abbreviated crystal structures.

    Elemental phases render as ``αMn`` with the structure parenthesised after it —
    ``α-Mn (bcc)`` — matching ``phase_transitions.json``'s ``<name>-<El> (<structure>)``
    convention, so a corner label in the ternary figure and a field label here read the
    same. The greek prefix is dropped when the element has only one phase present in
    ``all_names`` (``Fe (bcc)``), since there is then nothing to tell apart.

    This replaced a ``(αMn) bcc`` form. The parentheses around the ELEMENT were the
    conventional phase-diagram mark for a terminal solid solution, which this notation no
    longer distinguishes from a line compound; that was an explicit call, taken so the two
    figures agree.

    Compounds render with subscripts (``ZrMn2`` -> ``ZrMn<sub>2</sub>``).
    ``"Liquid"``/``"L"`` -> ``"L"``.
    """
    name = str(name).strip()
    if name in ("L", "Liquid"):
        return "L"

    parsed = _parse_elemental_phase(name)
    if parsed is not None:
        greek_word, element, struct, tag = parsed
        same_element = sum(
            1
            for other in all_names
            if (_parse_elemental_phase(str(other).strip()) or (None, None, None, None))[1]
            == element
        )
        prefix = _GREEK_MAP.get(greek_word, "") if (greek_word and same_element > 1) else ""
        label = f"{prefix}-{element}" if prefix else element
        abbr = _abbrev_structure(struct) if struct else ""
        if abbr:
            label += f" ({abbr})"
        if tag:
            label += f" {tag}"
        return label

    return _subscript_formula(name)


def _merge_close_values(values: list[float], tol: float) -> list[float]:
    """Cluster values whose sorted consecutive differences are <= ``tol``; return cluster means."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _curve_crossings_at_temp(
    pts: list[tuple[float, float]], temp: float, temp_tol: float
) -> list[float]:
    """Return interpolated x-values where a polyline crosses ``temp`` (within ``temp_tol``)."""
    crossings = []
    for (x0, t0), (x1, t1) in zip(pts, pts[1:]):
        t_lo, t_hi = (t0, t1) if t0 <= t1 else (t1, t0)
        if t_lo - temp_tol <= temp <= t_hi + temp_tol:
            if abs(t1 - t0) < 1e-9:
                crossings.append((x0 + x1) / 2.0)
            else:
                frac = max(0.0, min(1.0, (temp - t0) / (t1 - t0)))
                crossings.append(x0 + frac * (x1 - x0))
    if len(pts) == 1 and abs(pts[0][1] - temp) <= temp_tol:
        crossings.append(pts[0][0])
    return crossings


def _ss_boundary_crossings(
    ss_regions: dict[str, list[dict]], temp: float, temp_tol: float, skip: dict | None = None
) -> list[float]:
    """Compositions where any solid-solution field's boundary meets the horizontal ``temp``.

    A polymorph tie stops at the first phase boundary inward from the element edge, and a
    solid-solution field is such a boundary -- it just never reached ``compound_bounds``,
    which is built from the line-compound rows only. Each region's ring (``x``/``t``,
    closed) is walked with the same crossing routine the liquidus lookup uses.

    ``skip`` (compared by identity) drops one region from the walk. A horizontal drawn AT a
    branch's own temperature minimum touches that branch's ring by construction, so leaving
    it in would let the field stop its own tie a fraction of a step from where it started;
    every other field, including the branch's siblings, still bounds it.
    """
    crossings: list[float] = []
    for regions in ss_regions.values():
        for region in regions:
            if skip is not None and region is skip:
                continue
            ring = list(zip((float(v) for v in region["x"]), (float(v) for v in region["t"])))
            crossings.extend(_curve_crossings_at_temp(ring, temp, temp_tol))
    return crossings


def _edge_inside_ss_field(
    ss_regions: dict[str, list[dict]], temp: float, side: float, temp_tol: float
) -> bool:
    """Whether the pure element at ``side`` is covered by a solid-solution field at ``temp``.

    When it is, the polymorph whose trace tops out at ``temp`` is subsumed by the solution
    right at the axis: the nearest field boundary is at zero distance, so a horizontal from
    the edge would be drawn straight through the field's interior rather than stopping on
    it. Mn-Si's beta-Mn tops out at 1132.8 C, exactly the on-edge corner of the (Mn) BCC
    field, and the tie ran 13 at.% into the hatched region. Emit nothing there.

    Note this is a COMPOSITION-span test at one temperature, so a field anchored on the
    opposite axis edge can never trigger it.
    """
    for regions in ss_regions.values():
        for region in regions:
            ring = list(zip((float(v) for v in region["x"]), (float(v) for v in region["t"])))
            xs = _curve_crossings_at_temp(ring, temp, temp_tol)
            if xs and min(xs) - 1e-9 <= side <= max(xs) + 1e-9:
                return True
    return False


def _detect_tie_lines(
    invariant_temps: list[float],
    boundary_curves: list[list[tuple]],
    plot_xlim: tuple[float, float],
    temp_tol: float = 1.0,
    x_tol: float = 0.5,
) -> list[dict]:
    """Detect horizontal tie lines at each invariant temperature.

    A boundary curve is treated as a *solid* phase boundary when its x-extent is within
    ``x_tol`` (a near-vertical line). For each invariant temperature the tie line spans the
    outermost solid-boundary crossings; if only one solid boundary crosses, it is paired
    with the nearest liquidus (non-vertical) crossing. Returns ``{temp, x_start, x_end}`` dicts.
    """
    classified = []
    for curve in boundary_curves:
        pts = [(float(x), float(t)) for x, t in curve]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        classified.append((pts, (max(xs) - min(xs)) <= x_tol))

    tie_lines = []
    for temp in invariant_temps:
        solid_x, liq_x = [], []
        for pts, is_solid in classified:
            hits = _curve_crossings_at_temp(pts, temp, temp_tol)
            (solid_x if is_solid else liq_x).extend(hits)

        solid_merged = _merge_close_values(solid_x, x_tol) if solid_x else []
        if len(solid_merged) >= 2:
            tie_lines.append(
                {"temp": float(temp), "x_start": min(solid_merged), "x_end": max(solid_merged)}
            )
        elif len(solid_merged) == 1 and liq_x:
            sx = solid_merged[0]
            nearest = min(liq_x, key=lambda lx: abs(lx - sx))
            lo, hi = sorted((sx, nearest))
            if hi - lo > x_tol:
                tie_lines.append({"temp": float(temp), "x_start": lo, "x_end": hi})
    return tie_lines


def _resolve_label_collisions(
    labels: list[dict],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_iterations: int = 50,
    ceiling=None,
    tie_segments=None,
) -> list[dict]:
    """Iteratively nudge overlapping labels apart (primarily in y for vertical text).

    ``ceiling`` is an optional ``top(x) -> T`` callable (the liquidus envelope); when given,
    each label is kept below it so labels never cross into the liquid field. ``tie_segments``
    (list of ``(x0, x1, T)``) lets compound labels shift up out of any tie line they straddle.
    Returns copies with adjusted positions; labels displaced from their (post-ceiling) home gain
    ``showarrow=True`` and retain ``home_x``/``home_y`` so the caller can draw a leader arrow.
    """
    out = [dict(lbl) for lbl in labels]
    for lbl in out:
        lbl["x"] = float(lbl["x"])
        lbl["y"] = float(lbl["y"])

    span_y = (ylim[1] - ylim[0]) or 1.0
    one_px_y = span_y / _PLOT_H_PX
    ceil_margin = 0.6 * one_px_y * _LINE_H_FACTOR * 12  # ~ small gap below the liquidus
    bottom_pad = _BOTTOM_PAD_FRAC * span_y  # consistent gap above the lower boundary
    tie_segments = tie_segments or []

    def apply_tie_clearance(lbl):
        # Shift an in-band (non-pinned, non-float) label up out of any tie line crossing its box.
        if lbl.get("pin") or lbl.get("above_liquidus") or not tie_segments:
            return
        cx, cy, half_w, half_h = _estimate_label_box(lbl, xlim, ylim)
        # Only a tie that genuinely bisects the box counts: a tie sitting *above* the box top
        # does not cross the label, and lifting toward it (then having ``apply_ceiling`` pull the
        # box back down) lands the label straddling that tie -- the deep-eutectic failure on
        # Au-Sm/SmAu6 and Er-Rh/Er3Rh, where the liquidus ceiling sits just above the eutectic
        # tie. The strict ``< cy + half_h`` upper bound keeps such a placement intact.
        blocking = [
            T
            for (x0, x1, T) in tie_segments
            if x1 >= cx - half_w and x0 <= cx + half_w and cy - half_h < T < cy + half_h
        ]
        if blocking:
            shift = (max(blocking) + ceil_margin) - (cy - half_h)
            if shift > 0:
                lbl["y"] += shift

    def apply_ceiling(lbl):
        if ceiling is None or lbl.get("above_liquidus"):
            return  # labels deliberately floated above the liquidus are exempt
        cx, cy, half_w, half_h = _estimate_label_box(lbl, xlim, ylim)
        # The box must clear the liquidus across its full width, so use the lowest liquidus
        # temperature spanned by the box (the curve descends across a vertical label).
        samples = [ceiling(x) for x in (cx - half_w, cx, cx + half_w)]
        finite = [c for c in samples if np.isfinite(c)]
        if not finite:
            return
        c = min(finite)
        if cy + half_h > c - ceil_margin:
            lbl["y"] -= (cy + half_h) - (c - ceil_margin)

    def clamp_bounds(lbl):
        _, cy, _, half_h = _estimate_label_box(lbl, xlim, ylim)
        # Bottom-anchored labels (compounds) keep a consistent pad above the lower boundary;
        # labels deliberately floated above the liquidus are exempt from the pad.
        floor = ylim[0] + (0.0 if lbl.get("above_liquidus") else bottom_pad)
        if cy - half_h < floor:
            lbl["y"] += floor - (cy - half_h)
        elif cy + half_h > ylim[1] and not lbl.get("above_liquidus"):
            # Floated labels are meant to sit above the liquidus; never pull them back down.
            lbl["y"] += ylim[1] - (cy + half_h)

    # Pull labels below the liquidus *before* recording home, so the ceiling adjustment alone
    # does not trigger leader arrows (only collision displacement should). Pinned labels were
    # already placed in a valid slot by the packer and are never moved.
    for lbl in out:
        if not lbl.get("pin"):
            apply_tie_clearance(lbl)
            apply_ceiling(lbl)
            clamp_bounds(lbl)
        lbl.setdefault("home_x", lbl["x"])
        lbl.setdefault("home_y", lbl["y"])

    if len(out) >= 2:
        for _ in range(max_iterations):
            boxes = [_estimate_label_box(lbl, xlim, ylim) for lbl in out]
            moved = False
            for i in range(len(out)):
                for j in range(i + 1, len(out)):
                    pen_x, pen_y = _box_overlap(boxes[i], boxes[j])
                    if pen_x <= 0 or pen_y <= 0:
                        continue
                    pin_i, pin_j = out[i].get("pin"), out[j].get("pin")
                    if pin_i and pin_j:
                        continue  # both fixed (packer guarantees these do not overlap)
                    shift = pen_y / 2.0 + one_px_y
                    if pin_i:  # only j may move
                        out[j]["y"] += (pen_y + one_px_y) * (
                            1 if out[j]["y"] >= out[i]["y"] else -1
                        )
                    elif pin_j:  # only i may move
                        out[i]["y"] += (pen_y + one_px_y) * (
                            1 if out[i]["y"] >= out[j]["y"] else -1
                        )
                    elif out[i]["y"] <= out[j]["y"]:
                        out[i]["y"] -= shift
                        out[j]["y"] += shift
                    else:
                        out[i]["y"] += shift
                        out[j]["y"] -= shift
                    moved = True
            for lbl in out:
                if not lbl.get("pin"):
                    apply_tie_clearance(lbl)
                    apply_ceiling(lbl)
                    clamp_bounds(lbl)
            if not moved:
                break

    for lbl in out:
        if abs(lbl["y"] - lbl["home_y"]) > one_px_y or abs(lbl["x"] - lbl["home_x"]) > 1e-6:
            lbl["showarrow"] = True
    return out


def _liquidus_top_fn(liq_df: pd.DataFrame, assessed_pts: list | None, combine: str = "max"):
    """Return ``top(x)`` combining the generated (and assessed) liquidus curves.

    ``combine='max'`` gives the upper envelope (for placing things *above* the liquidus, e.g.
    the 'L' label and legend); ``combine='min'`` gives the lower envelope (for keeping solid
    phase labels *below* both curves). Returns ``-inf`` where neither curve is defined.
    """
    lx = liq_df["x"].to_numpy(dtype=float)
    lt = liq_df["t"].to_numpy(dtype=float)
    order = np.argsort(lx)
    lx, lt = lx[order], lt[order]

    ax = at = None
    if assessed_pts:
        ax = np.array([p[0] for p in assessed_pts], dtype=float)
        at = np.array([p[1] for p in assessed_pts], dtype=float)
        aorder = np.argsort(ax)
        ax, at = ax[aorder], at[aorder]

    reduce_fn = max if combine == "max" else min

    def top(x: float) -> float:
        vals = []
        if lx.size and lx.min() <= x <= lx.max():
            vals.append(float(np.interp(x, lx, lt)))
        if ax is not None and ax.size and ax.min() <= x <= ax.max():
            vals.append(float(np.interp(x, ax, at)))
        return reduce_fn(vals) if vals else -np.inf

    return top


def _place_liquid_label(
    liq_df: pd.DataFrame,
    assessed_pts: list | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    font_size: float = 14,
) -> tuple[float, float]:
    """Place the ``L`` label in the widest sufficiently-tall empty band above the liquidus."""
    top = _liquidus_top_fn(liq_df, assessed_pts)
    xs = np.linspace(xlim[0] + 2, xlim[1] - 2, 49)
    tops = np.array([(top(x) if np.isfinite(top(x)) else ylim[0]) for x in xs])
    gaps = ylim[1] - tops
    min_gap = _LINE_H_FACTOR * font_size * (ylim[1] - ylim[0]) / _PLOT_H_PX

    # Find the widest contiguous band tall enough for the label.
    ok = gaps >= min_gap
    best = None  # (run_length, mean_gap, i, j)
    i, n = 0, len(xs)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            cand = (j - i, float(gaps[i : j + 1].mean()), i, j)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
            i = j + 1
        else:
            i += 1

    if best is None:  # nothing tall enough: fall back to the single tallest band
        k = int(np.argmax(gaps))
        return float(xs[k]), float(tops[k] + gaps[k] / 2.0)

    # Within the widest band, place at the gap-weighted centroid so the label tracks the
    # open liquid field (centre for symmetric liquidi; shifted toward the deeper side).
    i, j = best[2], best[3]
    seg_x, seg_gap = xs[i : j + 1], gaps[i : j + 1]
    cx = float(np.sum(seg_x * seg_gap) / np.sum(seg_gap))
    return cx, float(top(cx) + (ylim[1] - top(cx)) / 2.0)


def _pack_labels(
    regions: list[tuple],
    half_hs: list[float],
    ceiling: float,
    tie_temps: list[float],
    floor: float,
    margin: float,
    gap: float = 0.0,
) -> list:
    """Place each vertical label inside its OWN stability band (``regions[i] = (t_bottom,
    t_top)``), in a tie-free sub-interval of ``[floor, ceiling]``.

    A label is only placed in an interval that overlaps its own region, so it is never shifted
    onto a neighbouring polymorph; labels that do not fit in their band return ``None`` (the
    caller floats those). Returns centre y values aligned with ``regions``.
    """
    n = len(regions)
    results = [None] * n
    if not np.isfinite(ceiling) or ceiling <= floor:
        return results

    cuts = sorted(t for t in tie_temps if floor < t < ceiling)
    intervals, lo = [], floor
    for t in cuts:
        if t - margin > lo:
            intervals.append((lo, t - margin))
        lo = max(lo, t + margin)
    if ceiling > lo:
        intervals.append((lo, ceiling))

    next_bottom = floor
    for idx in sorted(range(n), key=lambda i: regions[i][0]):
        rb, rt = regions[idx]
        rt = min(rt, ceiling)
        hh = half_hs[idx]
        mid = 0.5 * (rb + rt)
        for ilo, ihi in intervals:
            if ihi <= rb or ilo >= rt:
                continue  # interval does not overlap this label's own band
            bottom = max(ilo, rb, next_bottom)
            top = min(ihi, rt)
            if bottom + 2 * hh <= top:
                y = min(max(mid, bottom + hh), top - hh)
                results[idx] = y
                next_bottom = y + hh + gap
                break
    return results


def _assign_compound_sides(
    comps: list[float],
    half_ws: list[float],
    xlim: tuple[float, float] = (0.0, 100.0),
    standoff: float = 1.5,
    edge_standoff: float = 2.5,
) -> list[float]:
    """Choose an x for each compound label so adjacent labels fall in DISTINCT two-phase gaps.

    Each compound at composition ``c`` may place its (vertical) label in the gap to its left
    ``(c_prev, c)`` or right ``(c, c_next)``. Processing left-to-right, a label takes its left
    gap by default but switches to the right gap when the left gap was already claimed by the
    previous label or is too narrow for the label box; compounds within 10 at.% of an element
    edge are forced inward to clear the element solid-solution label. Returns one x per input
    compound (in input order). Where no conflict-free side exists the label keeps the roomier
    side and the y-resolver separates the resulting overlap by height.
    """
    n = len(comps)
    order = sorted(range(n), key=lambda i: comps[i])
    result = [0.0] * n
    tiny = 1e-6
    prev_used_right = False  # did the previous label claim the gap shared with this one's left?
    prev_box_right = xlim[0]
    for pos, i in enumerate(order):
        c, hw = float(comps[i]), float(half_ws[i])
        left_nb = comps[order[pos - 1]] if pos > 0 else xlim[0]
        right_nb = comps[order[pos + 1]] if pos < n - 1 else xlim[1]
        near_left_edge, near_right_edge = c - xlim[0] < 10.0, xlim[1] - c < 10.0
        so = edge_standoff if (near_left_edge or near_right_edge) else standoff
        left_x, right_x = c - so, c + so
        left_ok = (
            left_x - hw > left_nb + tiny
            and not prev_used_right
            and left_x - hw > prev_box_right + tiny
        )
        right_ok = right_x + hw < right_nb - tiny
        if near_left_edge:
            chosen, used_right = right_x, True
        elif near_right_edge:
            chosen, used_right = left_x, False
        elif left_ok:
            chosen, used_right = left_x, False
        elif right_ok:
            chosen, used_right = right_x, True
        else:  # forced overlap -> keep the roomier side, let the y-resolver stack
            if (c - left_nb) >= (right_nb - c):
                chosen, used_right = left_x, False
            else:
                chosen, used_right = right_x, True
        result[i] = chosen
        prev_used_right = used_right
        prev_box_right = chosen + hw
    return result


def _place_compound_y(
    c0: float, ceiling: float, tie_temps: list[float], half_h: float, pad: float, margin: float
) -> float | None:
    """Bottom-align a compound label in the LOWEST tie-free sub-band of ``[c0, ceiling]`` that
    is tall enough. Returns the label centre y, or ``None`` if no sub-band fits.

    The bottom-most band keeps ``pad`` above ``c0``; bands above a tie keep ``margin`` above it.
    This realises "drop the label to the bottom, but skip up to the next two-phase region when a
    tie line would cross it" (bugs 1 & 2).
    """
    if not np.isfinite(ceiling) or ceiling <= c0:
        return None
    cuts = sorted(t for t in tie_temps if c0 < t < ceiling)
    intervals, lo = [], c0
    for t in cuts:
        if t - margin > lo:
            intervals.append((lo, t - margin))
        lo = max(lo, t + margin)
    if ceiling > lo:
        intervals.append((lo, ceiling))
    for ilo, ihi in intervals:
        # The floor region keeps ``pad`` above c0; regions above a tie already start at
        # ``tie + margin``, so no extra offset is needed there.
        bottom = ilo + (pad if abs(ilo - c0) < 1e-9 else 0.0)
        if bottom + 2 * half_h <= ihi:
            return bottom + half_h
    return None


def _place_legend(
    hsx,
    liq_df: pd.DataFrame,
    assessed_pts: list | None,
    xlim: tuple[float, float],
    n_entries: int,
    max_label_chars: int,
    float_top_by_side: dict | None = None,
) -> dict:
    """Pick the top corner with the most clearance and expand ``hsx.conds[1]`` (the upper
    temperature limit) so the legend sits inside the plot area without overlapping either
    liquidus or the floated polymorph labels on that side. Never placed above the plot.
    """
    float_top_by_side = float_top_by_side or {0: -np.inf, 100: -np.inf}
    span_x = (xlim[1] - xlim[0]) or 1.0
    w_px = max_label_chars * _CHAR_W_FACTOR * 15 + 40  # text + colour swatch
    h_px = n_entries * 21 + 12
    w_data = w_px / (_PLOT_W_PX / span_x)
    inset_x = 0.01 * span_x
    top = _liquidus_top_fn(liq_df, assessed_pts, combine="max")

    def obstacle(x0, x1, side):
        vals = [top(x) for x in np.linspace(x0, x1, 15)]
        vals = [v for v in vals if np.isfinite(v)]
        liq = max(vals) if vals else -np.inf
        return max(liq, float_top_by_side.get(side, -np.inf))

    # (xanchor, x_paper, x0, x1, side)
    corners = [
        ("right", 0.99, xlim[1] - inset_x - w_data, xlim[1] - inset_x, 100),
        ("left", 0.01, xlim[0] + inset_x, xlim[0] + inset_x + w_data, 0),
    ]
    # Prefer the corner with the lowest obstacle (least y-range expansion, avoids floats).
    corners.sort(key=lambda c: obstacle(c[2], c[3], c[4]))
    xanchor, xp, x0, x1, side = corners[0]
    obs = obstacle(x0, x1, side)

    # Expand conds[1] so the legend band (h_px tall, at the top) clears that obstacle.
    f = (h_px + 6) / _PLOT_H_PX
    if np.isfinite(obs) and f < 1.0:
        needed_c1 = (obs - f * hsx.conds[0]) / (1.0 - f)
        if needed_c1 > hsx.conds[1]:
            hsx.conds[1] = needed_c1

    return {"xanchor": xanchor, "yanchor": "top", "x": xp, "y": 0.99, "font": dict(size=15)}


def plot_tx(
    hsx,
    pred: bool = False,
    digitized_liquidus: list = None,
    polymorph_transitions: list[dict] | None = None,
    imputed_phases: set | None = None,
    ternary_color_map: dict | None = None,
    ss_phases: set | None = None,
) -> go.Figure:
    """Plots the binary phase diagram from computed phase boundaries and invariant points.

    Args:
        hsx: The :class:`~gliquid.hsx.HSX` instance to render — its ``df_tx`` boundaries,
            ``phases``, and ``liquidus_invariants()`` supply everything drawn. Its
            ``conds`` (the plotted temperature range) are mutated in place.
        pred (bool): If True, use prediction color scheme for the liquidus.
        digitized_liquidus (list): Digitized experimental liquidus data points.
        polymorph_transitions (list[dict]): List of elemental polymorph transitions, each dict with keys:
            'name' (str), 'comp_x_pct' (float, 0 or 100), 'transition_temp_C' (float),
            'ground_state_name' (str) for the phase below the transition.
        imputed_phases (set): Names of phases imputed by phase-energy imputation; their
            solid boundary lines are drawn dashed and given a single legend entry.
        ternary_color_map (dict): Optional phase-to-color overrides applied on top of
            the locally built phase color map (used by ternary edge sub-plots).
        ss_phases (set): Names of continuous solid-solution phases. These are excluded
            from the line-compound machinery and rendered as closed, hatch-filled
            polygons (one per composition branch) in reserved colors, with one legend
            entry each. Tie lines touching them are kept only where an invariant reaction
            puts them: at a three-phase invariant, and at a branch's own interior
            temperature minimum when the coexistence changes there (the eutectoid).
            Empty/None leaves the non-SS control flow untouched.
    """
    imputed_phases = imputed_phases or set()
    ss_phases = set(ss_phases or ())
    liq_inv = hsx.liquidus_invariants()
    inv_points = liq_inv[0]
    # Colors are plot-local: built from the phase list here, never stored on the HSX.
    phase_colors = build_phase_color_map(hsx.phases, ss_names=sorted(ss_phases))
    if ternary_color_map:
        phase_colors.update(ternary_color_map)

    liq_df = hsx.df_tx[hsx.df_tx["label"] == "L"].copy()
    liq_df["x"] *= 100
    liq_df.sort_values(by=["x", "t"], inplace=True)
    liq_df.drop_duplicates(subset="x", keep="first", inplace=True)

    # Use raw compute_tx scatter points for solids so polymorphs at identical
    # composition are preserved (invariant-derived groupings can collapse them).
    # Solid-solution phases never enter the line-compound machinery below — they are
    # rendered separately as envelope branches (see the ss_phases block near the end).
    solid_df = hsx.df_tx[hsx.df_tx["label"] != "L"].copy()
    if ss_phases:
        solid_df = solid_df[~solid_df["label"].isin(ss_phases)].copy()
    solid_df["x"] *= 100
    solid_df.drop_duplicates(subset=["x", "t", "label"], keep="first", inplace=True)

    # Closed SS envelope polygons, built once here: the tie filter below reads their
    # extrema and the render block near the end draws them. Geometry depends only on the
    # phase's df_tx extents, so the conds adjustments that follow cannot invalidate it.
    ss_regions: dict[str, list[dict]] = {}
    if ss_phases:
        # Stability intervals of the component polymorphs sitting on each composition axis
        # (Hf-Y x=0: alpha-Hf 733-1743, beta-Hf 1743-2233). An SS branch that stops short
        # of an axis snaps its corner onto whichever of these it overlaps, instead of
        # extrapolating past the component's transition temperature.
        edge_intervals: dict[float, list[tuple[float, float]]] = {}
        for edge in (0.0, 100.0):
            at_edge = solid_df[np.isclose(solid_df["x"], edge)]
            edge_intervals[edge] = sorted(
                (float(grp["t"].min()), float(grp["t"].max()))
                for _label, grp in at_edge.groupby("label")
            )

        # Three-phase invariants, per SS phase, as (composition %, temperature). A field
        # taking part in one is at its maximum solubility there, so a blunt terminus that
        # straddles one closes onto it.
        ss_invariants: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for key in ("Eutectics", "Peritectics"):
            for temp, _mid, comps, inv_phases in inv_points.get(key, []):
                for comp, phase in zip(comps, inv_phases):
                    if phase in ss_phases:
                        ss_invariants[phase].append((float(comp) * 100, float(temp)))

        # The composition grid is a property of the HULL SAMPLING, not of any one phase's
        # stability range, so measure it ONCE here over the axis every phase was sampled
        # on. Inferred per field it degenerates: a field supported at exactly two
        # compositions has only one spacing -- the gap itself -- so no gap can exceed
        # 1.5x it, the branch splitter never fires, and the two terminal branches weld
        # into a single quadrilateral spanning the diagram. Mo-Y and W-Y are that case:
        # BCC at 0.2 at.% (BCC-Mo/W) and at 100 at.% (beta-Y), nothing in between.
        grid_step = _hull_grid_step(hsx.df_tx["x"].to_numpy(dtype=float) * 100.0)

        ss_df = hsx.df_tx[hsx.df_tx["label"].isin(ss_phases)].copy()
        ss_df["x"] *= 100
        for ss_name in ss_phases:
            sub = ss_df[ss_df["label"] == ss_name]
            if sub.empty:
                continue
            extents = sub.groupby("x")["t"].agg(["min", "max"]).sort_index()
            regions = _ss_regions(
                extents.index.to_numpy(dtype=float),
                extents["min"].to_numpy(dtype=float),
                extents["max"].to_numpy(dtype=float),
                edge_intervals=edge_intervals,
                apex_invariants=ss_invariants.get(ss_name),
                grid_step=grid_step,
            )
            if regions:
                ss_regions[ss_name] = regions

    lhs_tm, rhs_tm = liq_df.iloc[0]["t"], liq_df.iloc[-1]["t"]
    max_liq, min_liq = liq_df["t"].max(), liq_df["t"].min()

    fig = go.Figure()

    # Assessed (digitized) liquidus, in plot units (at.% and °C); None when absent.
    assessed_pts = (
        [[p[0] * 100, p[1] - 273.15] for p in digitized_liquidus] if digitized_liquidus else None
    )
    if assessed_pts:  # update liquidus temperature range based on digitized liquidus
        max_liq = max(max_liq, max(p[1] for p in assessed_pts))
        min_liq = min(min_liq, min(p[1] for p in assessed_pts))
        # One trace per digitized region: no dashed line across an undigitized hole.
        for segment in _assessed_liquidus_segments(assessed_pts):
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in segment],
                    y=[p[1] for p in segment],
                    mode="lines",
                    line=dict(color=ASSESSED_LIQUIDUS_COLOR, dash="dash"),
                )
            )

    # expand temperature range based on liquidus extremes
    if max_liq > hsx.conds[1]:
        hsx.conds[1] = max_liq + 0.1 * (hsx.conds[1] - hsx.conds[0])
    if min_liq < hsx.conds[0]:
        hsx.conds[0] = max(-273.15, min_liq - 0.1 * (hsx.conds[1] - hsx.conds[0]))

    # Cap excessive headroom above the liquidus. The 'L' label and legend are auto-placed,
    # so no artificial y-extension is needed to accommodate them.
    headroom_cap = max_liq + 0.18 * (max_liq - hsx.conds[0])
    if hsx.conds[1] > headroom_cap:
        hsx.conds[1] = headroom_cap

    solid_phases = [
        p for p in hsx.phases if p not in [hsx.comps[0], hsx.comps[1], "L"] and p not in ss_phases
    ]
    for phase in solid_phases:
        phase_df = solid_df[solid_df["label"] == phase]
        if phase_df.empty:
            continue

        # expand temperature range based on minimum decomposition temperatures of solid phases
        phase_decomp_temp = phase_df["t"].max()
        if phase_decomp_temp - 0.1 * (hsx.conds[1] - hsx.conds[0]) < hsx.conds[0]:
            hsx.conds[0] = max(-273.15, phase_decomp_temp - 0.1 * (hsx.conds[1] - hsx.conds[0]))

    # Build per-phase lower-extension limits. For polymorphs at the same composition,
    # only the lowest-temperature phase is extended to plot bottom; upper polymorphs
    # are extended down only to the top of the lower polymorph to avoid full overlap.
    phase_rows = []
    comp_groups = defaultdict(list)
    for phase in solid_phases:
        phase_df = solid_df[solid_df["label"] == phase]
        if phase_df.empty:
            continue
        solid_comp = float(phase_df["x"].iloc[0])
        t_min = float(phase_df["t"].min())
        t_max = float(phase_df["t"].max())
        phase_rows.append((phase, phase_df, solid_comp))
        comp_groups[round(solid_comp, 6)].append(
            {
                "phase": phase,
                "t_min": t_min,
                "t_max": t_max,
            }
        )

    phase_low_ext = {}
    for group in comp_groups.values():
        ordered = sorted(group, key=lambda d: (d["t_min"], d["t_max"]))
        prev_top = None
        for idx, entry in enumerate(ordered):
            if idx == 0:
                low_ext = -273.15
            else:
                low_ext = prev_top if prev_top is not None else -273.15
                # Never extend above where this phase already starts.
                low_ext = min(low_ext, entry["t_min"])
            phase_low_ext[entry["phase"]] = low_ext
            prev_top = entry["t_max"]

    # Collect polymorph phase names for separate labeling
    polymorph_names = set()
    if polymorph_transitions:
        polymorph_names = {pt["name"] for pt in polymorph_transitions}
        polymorph_names |= {pt.get("ground_state_name", "") for pt in polymorph_transitions}

    # All phase names (used by _abbreviate_phase_name to decide whether to keep greek prefixes).
    all_phase_names = set(solid_df["label"].unique()) | polymorph_names
    # Labels are collected here, then de-collided in one pass before being drawn.
    label_dicts: list[dict] = []
    # One bookkeeping record per DRAWN tie, in draw order: the span, the fig.data index of
    # its trace, its slot in tie_segments, and its instrumentation record. A merge rewrites
    # a record in place so all three stay in sync with the geometry actually on the figure.
    drawn_ties: list[dict] = []
    tie_segments: list[tuple] = []
    # conds is settled for the tie stage by here (the liquidus/decomposition expansions ran
    # above); the later label-driven lowering of conds[0] cannot reach back into this.
    tie_merge_t_tol = _TIE_MERGE_T_FRAC * (hsx.conds[1] - hsx.conds[0])

    def _add_tie(x0, x1, temp, dedup=True, source=None):
        """Draw one horizontal tie, merging it into a coincident one already drawn.

        Two sources can emit the same physical horizontal with slightly different extents
        (Cu-Ir's peritectic at 1224 C and the two 'Misc Gaps' halves of it at 1219/1223).
        A candidate within ``tie_merge_t_tol`` of a drawn tie whose composition span TOUCHES
        or overlaps it extends that tie to the union span instead of adding a second trace,
        so the drawn count is monotone in the number of physically distinct ties and trace
        order is never disturbed. Disjoint spans at the same temperature stay separate --
        they are the two halves of a genuine three-phase horizontal around a compound.
        """
        lo, hi, t = min(float(x0), float(x1)), max(float(x0), float(x1)), float(temp)
        if dedup:
            for rec in drawn_ties:
                if abs(t - rec["t"]) > tie_merge_t_tol:
                    continue
                if min(hi, rec["x_hi"]) < max(lo, rec["x_lo"]):
                    continue  # spans do not even touch -> a different tie
                new_lo, new_hi = min(lo, rec["x_lo"]), max(hi, rec["x_hi"])
                if (new_lo, new_hi) != (rec["x_lo"], rec["x_hi"]):
                    fig.data[rec["trace"]].x = (new_lo, new_hi)  # y is already constant
                    rec["x_lo"], rec["x_hi"] = new_lo, new_hi
                    tie_segments[rec["segment"]] = (new_lo, new_hi, rec["t"])
                if rec["record"] is not None:
                    rec["record"]["x0"], rec["record"]["x1"] = new_lo, new_hi
                    rec["record"]["sources"].append(source)
                return
        tie_segments.append((lo, hi, t))
        tie = px.line(x=[x0, x1], y=[temp, temp])
        tie.update_traces(line=dict(color="Silver"))
        fig.add_trace(tie.data[0])
        record = None
        if _TIE_SINK is not None:
            record = {"temp": t, "x0": lo, "x1": hi, "sources": [source]}
            _TIE_SINK.append(record)
        drawn_ties.append(
            {
                "t": t,
                "x_lo": lo,
                "x_hi": hi,
                "trace": len(fig.data) - 1,
                "segment": len(tie_segments) - 1,
                "record": record,
            }
        )

    # Compound (non-polymorph) labels are placed after the conds-convergence loop so they can
    # use the final temperature range and the drawn tie lines (region-fit + escalation below).
    compound_phase_list = []
    for phase, phase_df, solid_comp in phase_rows:
        low_ext_temp = phase_low_ext.get(phase, -273.15)
        new_row_df = pd.DataFrame(
            [{"x": solid_comp, "t": low_ext_temp, "label": phase}], columns=phase_df.columns
        )
        phase_df = pd.concat([phase_df, new_row_df], ignore_index=True)
        line = px.line(phase_df, x="x", y="t", color="label", color_discrete_map=phase_colors)

        trace = line.data[0]
        if phase in imputed_phases:
            existing_line = trace.line.to_plotly_json() if trace.line is not None else {}
            existing_line["dash"] = "dash"
            trace.line = existing_line
        fig.add_trace(trace)

        # Skip label here for polymorphs — they are labeled separately below
        if phase not in polymorph_names:
            comp_key = round(float(solid_comp), 6)
            compound_phase_list.append(
                {
                    "phase": phase,
                    "comp": float(solid_comp),
                    "t_min": float(low_ext_temp),
                    "t_max": float(phase_df["t"].max()),
                    "is_stacked": len(comp_groups.get(comp_key, [])) > 1,
                    "text": _abbreviate_phase_name(phase, all_phase_names),
                }
            )

    ss_t_range = (hsx.conds[0], hsx.conds[1])
    # The L + S1 + S2 reaction (a solid miscibility gap meeting the liquidus) arrives as a
    # whole family of grid slices; the phase rule says it is ONE invariant. Resolve the
    # per-phase extremum once, before the loop, so admission stays a per-invariant test.
    ss_family_max = (
        _ss_family_maxima(
            (
                (key, [x * 100 for x in inv[2]], inv[3], inv[0])
                for key in inv_points
                if key in _TIE_INV_KEYS
                for inv in inv_points[key]
            ),
            ss_regions,
        )
        if ss_regions
        else {}
    )
    for key in inv_points.keys():
        if key in _TIE_INV_KEYS:
            for temp, _, comps, inv_phases in inv_points[key]:
                comps = [x * 100 for x in comps]
                if ss_regions and not _ss_tie_allowed(
                    comps, inv_phases, temp, ss_regions, ss_t_range, key, ss_family_max
                ):
                    continue  # SS ties are drawn only where they are physically meaningful
                _add_tie(min(comps), max(comps), temp, source=f"invariant:{key}")

    # --- Polymorph (element solid-solution) tie lines and label regions, derived from the
    # ACTUAL DFT traces in solid_df (their on-hull stability ranges), not the experimental
    # transition temperatures (which do not coincide with the plotted trace boundaries). ---
    liq_pts = list(zip(liq_df["x"].tolist(), liq_df["t"].tolist())) if not liq_df.empty else []
    poly_traces = {0: [], 100: []}
    compound_bounds = []  # (comp, t_min, t_max) for non-polymorph solids
    for phase, _pdf, solid_comp in phase_rows:
        sub = solid_df[solid_df["label"] == phase]
        if sub.empty:
            continue
        if phase in polymorph_names:
            side = 0 if solid_comp < 50 else 100
            poly_traces[side].append(
                {
                    "name": phase,
                    # RAW lower extent of the DFT trace (e.g. ~-273 for a ground state) so a later
                    # lowering of conds[0] can extend this region downward in-band. Clamped to the
                    # *current* conds[0] only at layout time, never frozen here.
                    "t_bottom": phase_low_ext.get(phase, hsx.conds[0]),
                    "t_max": float(sub["t"].max()),
                }
            )
        else:
            compound_bounds.append((solid_comp, float(sub["t"].min()), float(sub["t"].max())))

    boundary_temp_tol = 0.01 * (hsx.conds[1] - hsx.conds[0])

    def _boundary_xs(temp, skip_region=None):
        """Every phase boundary the horizontal at ``temp`` crosses, as compositions.

        SS fields have to be asked for explicitly: ``compound_bounds`` is built from
        ``phase_rows``, which excludes them, so without this a polymorph tie runs straight
        through the hatched field (Mn-Si's alpha->beta horizontal crossed the (Mn) BCC
        field from 6 to 25 at.% instead of terminating on it at 6)."""
        xs = (
            list(_curve_crossings_at_temp(liq_pts, temp, temp_tol=boundary_temp_tol))
            if liq_pts
            else []
        )
        xs += [c for c, t0, t1 in compound_bounds if t0 - 1 <= temp <= t1 + 1]
        xs += _ss_boundary_crossings(ss_regions, temp, boundary_temp_tol, skip=skip_region)
        return xs

    def _adjacent_boundary_x(temp, side):
        """Nearest phase boundary inward from the element edge at temp: a liquidus crossing, a
        compound, a solid-solution field boundary, or — when nothing lies between — the
        opposite element's solid (so a polymorph tie spans to the opposite polymorph)."""
        xs = _boundary_xs(temp)
        opp = 100 if side == 0 else 0
        if temp < (rhs_tm if opp == 100 else lhs_tm):  # opposite element is solid at temp
            xs.append(float(opp))
        inward = [x for x in xs if (x > side if side == 0 else x < side)]
        return min(inward, key=lambda x: abs(x - side)) if inward else None

    def _ss_minimum_span(x_at, temp, region):
        """Grow the horizontal at a branch's temperature minimum out to its bounding fields.

        The span runs outward from the minimum COMPOSITION to the nearest phase boundary
        on each side, which is the span the reaction itself has: Y-Zr's 768.0 C eutectoid
        runs from the Y-side HCP boundary at ~3.6 at.% to the
        Zr-side one at ~96.4, exactly the two HCP vertices of the facet at that temperature.
        The witnesses' own compositions are not used. Returns None when no boundary brackets
        the minimum, since then the extent is not determined.
        """
        xs = _boundary_xs(temp, skip_region=region)
        left = max((x for x in xs if x < x_at - 1e-9), default=None)
        right = min((x for x in xs if x > x_at + 1e-9), default=None)
        # Nothing between the minimum and an axis means the two-phase field reaches the pure
        # element, which bounds it there -- the same fallback _adjacent_boundary_x makes.
        if left is None and temp < lhs_tm:
            left = 0.0
        if right is None and temp < rhs_tm:
            right = 100.0
        if left is None or right is None or right - left <= 0.0:
            return None
        return float(left), float(right)

    # --- Solid-solution eutectoid horizontals. The invariant emitter hands a solvus over as
    # one entry per grid slice and _ss_tie_allowed rejects the lot; the genuine reaction
    # hiding in that fan is the ONE at the field's interior temperature minimum, which the
    # branch's own geometry locates and the hull's facets confirm. At most one per branch. ---
    if ss_regions:
        ss_assemblages = _facet_assemblages(hsx.df_tx)
        for ss_name in sorted(ss_regions):
            for region in ss_regions[ss_name]:
                anchor = _ss_minimum_anchor(region)
                if not _ss_minimum_tie_allowed(
                    anchor, region, ss_assemblages, (0.0, 100.0), ss_t_range
                ):
                    continue
                extent = _ss_minimum_span(float(anchor["x"]), float(anchor["t"]), region)
                if extent is None:
                    continue
                # Through _add_tie, so a minimum that restates an already-drawn eutectic or
                # peritectic extends that trace instead of doubling it (Hf-W's 1228.3 C).
                _add_tie(extent[0], extent[1], float(anchor["t"]), source="ss_minimum")

    polymorph_regions = []  # (region_dict, comp_pct), placed after the safety-net ties
    for side, traces in poly_traces.items():
        elt_melt = lhs_tm if side == 0 else rhs_tm
        for tr in sorted(traces, key=lambda d: d["t_max"]):
            # Tie line at the trace top (polymorph transition or melting), drawn all the way to
            # the adjacent phase boundary so it fully intersects it. Skip the top polymorph's
            # congruent melt at the pure element (a point on the liquidus, not a tie line).
            # A polymorph the solid solution already covers AT the element edge has no tie:
            # the field starts at the axis there, so the horizontal would run through its
            # interior rather than terminate on it.
            T = tr["t_max"]
            congruent_melt = abs(T - elt_melt) < 0.02 * (hsx.conds[1] - hsx.conds[0])
            subsumed = _edge_inside_ss_field(ss_regions, T, float(side), boundary_temp_tol)
            if hsx.conds[0] < T < hsx.conds[1] and not congruent_melt and not subsumed:
                bx = _adjacent_boundary_x(T, side)
                if bx is not None and abs(bx - side) >= 0.6:
                    _add_tie(side, bx, T, source="polymorph")
            # Label region carries the RAW trace extent; visibility is decided at layout time
            # against the (possibly lowered) conds[0]. A region whose entire trace sits below
            # the current floor is dropped.
            if tr["t_max"] > hsx.conds[0]:
                polymorph_regions.append(
                    ({"name": tr["name"], "t_bottom": tr["t_bottom"], "t_max": tr["t_max"]}, side)
                )

    # --- Safety-net tie lines: connect each incongruently-melting solid phase to the
    # liquidus at the top of its boundary. The eutectic/peritectic invariants above already
    # cover most cases; this recovers any solid<->liquidus tie that invariant detection
    # missed, while staying local (no full-width spans). ---
    if not liq_df.empty and phase_rows:
        liq_top = _liquidus_top_fn(liq_df, None)
        span = hsx.conds[1] - hsx.conds[0]
        congruent_tol = 0.02 * span  # phase top within this of the liquidus -> congruent
        max_tie_span = 40.0  # at.%; tie lines are local, never near-full-width
        for phase, _pdf, comp_s in phase_rows:
            if phase in polymorph_names:
                continue  # polymorph ties handled above from their traces
            sub = solid_df[solid_df["label"] == phase]
            if sub.empty:
                continue
            t_max = float(sub["t"].max())
            liq_here = liq_top(comp_s)
            if np.isfinite(liq_here) and abs(liq_here - t_max) <= congruent_tol:
                continue  # congruent melter -> meets liquidus at a point, no horizontal tie
            crossings = _curve_crossings_at_temp(liq_pts, t_max, temp_tol=0.02 * span)
            if not crossings:
                continue
            nearest = min(crossings, key=lambda xc: abs(xc - comp_s))
            if not (0.6 < abs(nearest - comp_s) <= max_tie_span):
                continue
            _add_tie(comp_s, nearest, t_max, source="safety-net")

    # --- Place deferred polymorph/element labels. Preference per label: (1) in-band in its own
    # narrow field; (2) for the lowest-T polymorph, lower conds[0] so it fits (>= absolute zero);
    # (3) in the wider adjacent two-phase region just inside the element, below the liquidus;
    # (4) floated above the liquidus with a leader arrow. ---
    xlim = (0.0, 100.0)
    liq_floor = _liquidus_top_fn(liq_df, assessed_pts, combine="min")  # below BOTH curves
    liq_high = _liquidus_top_fn(liq_df, assessed_pts, combine="max")  # above BOTH curves
    _ABS_ZERO = -273.15
    _CLEAR_PX = 10.0  # clearance (px) between a floated label and the liquidus below it
    clear_frac = _CLEAR_PX / _PLOT_H_PX
    gap_frac = 0.004  # stack gap as a fraction of the y-span
    entry_labels = ["Assessed Liquidus"] if digitized_liquidus else []
    entry_labels.append("Predicted Liquidus" if pred else "Fitted Liquidus")
    if ss_phases:  # SS bands add one legend entry each; size the legend for them
        entry_labels += [
            format_phase_display_name(name, ss_phases, hsx.comps) for name in sorted(ss_phases)
        ]

    def _env_min(fn, cx, half_w):
        vals = [
            v for v in (fn(x) for x in np.linspace(cx - half_w, cx + half_w, 11)) if np.isfinite(v)
        ]
        return min(vals) if vals else None

    def _env_max(fn, cx, half_w):
        vals = [
            v for v in (fn(x) for x in np.linspace(cx - half_w, cx + half_w, 11)) if np.isfinite(v)
        ]
        return max(vals) if vals else None

    # Precompute static per-side data (half-heights as span-invariant fractions, ceilings, ties).
    sides_data = {}
    for side_comp in (0, 100):
        side = [r for (r, c) in polymorph_regions if c == side_comp]
        if not side:
            continue
        label_x = 1.5 if side_comp == 0 else 98.5  # small standoff, like the compound labels
        fx = 8.0 if side_comp == 0 else 92.0  # inward x for two-phase / floated labels
        texts = [_abbreviate_phase_name(r["name"], all_phase_names) for r in side]
        half_hs_frac, half_w = [], 1.2
        for text in texts:
            probe = {
                "x": label_x,
                "y": 0.0,
                "text": text,
                "textangle": -90,
                "font_size": 12,
                "xanchor": "center",
                "yanchor": "middle",
            }
            _, _, half_w, hh = _estimate_label_box(probe, xlim, (hsx.conds[0], hsx.conds[1]))
            half_hs_frac.append(hh / (hsx.conds[1] - hsx.conds[0]))
        # Dodge compound columns: a floated/relocated polymorph label sits at ``fx`` with a leader
        # arrow back to the element edge. If a compound shares that composition (e.g. Be12V at
        # 8 at.% vs the side-0 default fx=8) the float lands on top of the compound label. Shift
        # fx to the nearest compound-free x (keeping clear of the in-band element label too) so
        # neither the box nor the arrow collides; an empty same-side gap leaves fx at default.
        obstacles = [label_x] + [
            cp["comp"] for cp in compound_phase_list if (cp["comp"] < 50) == (side_comp == 0)
        ]
        clearance = 2 * half_w + 0.8
        if any(abs(o - fx) < clearance for o in obstacles):
            lo, hi = (2.0, 48.0) if side_comp == 0 else (52.0, 98.0)
            cands = [
                x
                for x in np.arange(lo, hi + 1e-6, 0.5)
                if all(abs(o - x) >= clearance for o in obstacles)
            ]
            if cands:
                fx = min(cands, key=lambda x: abs(x - fx))
        sides_data[side_comp] = dict(
            side=side,
            label_x=label_x,
            fx=fx,
            texts=texts,
            half_hs_frac=half_hs_frac,
            ceil_raw=_env_min(liq_floor, label_x, half_w),
            fx_ceil_raw=_env_min(liq_floor, fx, half_w),
            float_base=_env_max(liq_high, fx, half_w),
            tie_temps=[
                T for x0, x1, T in tie_segments if x1 >= label_x - half_w and x0 <= label_x + half_w
            ],
            fx_tie_temps=[
                T for x0, x1, T in tie_segments if x1 >= fx - half_w and x0 <= fx + half_w
            ],
            bottom_idx=min(range(len(side)), key=lambda i: side[i]["t_bottom"]),
        )

    def _layout_side(side_comp, c0, c1):
        """Classify each label: ('inband', y) | ('lower', needed_c0) | ('twophase', y) |
        ('float', half_h_frac), evaluated at the given conds.

        Relocated (``twophase``) labels are packed JOINTLY in the inward two-phase column so two
        never share a slot, and ``_pack_labels`` fills them bottom-to-top in their own
        t_bottom order (monotonic => non-crossing leader arrows / shortest longest arrow)."""
        d = sides_data[side_comp]
        span = c1 - c0
        margin, gap = 0.012 * span, 0.004 * span
        half_hs = [hf * span for hf in d["half_hs_frac"]]
        ceil = (d["ceil_raw"] - margin) if d["ceil_raw"] is not None else c1
        fx_ceil = (d["fx_ceil_raw"] - margin) if d["fx_ceil_raw"] is not None else c1
        regions_ty = [(max(r["t_bottom"], c0), min(r["t_max"], c1)) for r in d["side"]]
        ys = _pack_labels(regions_ty, half_hs, ceil, d["tie_temps"], c0, margin, gap)
        out = [None] * len(d["side"])
        for i in range(len(d["side"])):
            if ys[i] is not None:
                out[i] = ("inband", ys[i])
        # Lowest-T polymorph: lower conds[0] so it fits in its bottom gap.
        bi = d["bottom_idx"]
        if out[bi] is None:
            r = d["side"][bi]
            r_top = min(r["t_max"], c1)
            obstacles = sorted(T for T in d["tie_temps"] if c0 < T < min(r_top, ceil))
            gap_top = obstacles[0] if obstacles else min(r_top, ceil)
            needed_c0 = gap_top - 2 * half_hs[bi] - 1.5 * margin
            if needed_c0 >= _ABS_ZERO:
                out[bi] = ("lower", needed_c0)
        # Remaining labels relocate to the inward two-phase region, packed jointly (shared
        # cursor) so they cannot coincide; any that still do not fit float above the liquidus.
        reloc = [i for i in range(len(d["side"])) if out[i] is None]
        if reloc:
            regions_fx = [(max(d["side"][i]["t_bottom"], c0), fx_ceil) for i in reloc]
            fx_ys = _pack_labels(
                regions_fx, [half_hs[i] for i in reloc], fx_ceil, d["fx_tie_temps"], c0, margin, gap
            )
            for k, i in enumerate(reloc):
                out[i] = (
                    ("twophase", fx_ys[k])
                    if fx_ys[k] is not None
                    else ("float", d["half_hs_frac"][i])
                )
        return out

    # Precompute compound-label columns: each compound's x is assigned to a distinct two-phase
    # gap (left/right of the compound); per-column ceiling/ties drive region-fit + escalation.
    compound_cols = []
    if compound_phase_list:
        comps_x = [cp["comp"] for cp in compound_phase_list]
        init_ylim = (hsx.conds[0], hsx.conds[1])
        half_ws = []
        for cp in compound_phase_list:
            probe = {
                "x": cp["comp"],
                "y": 0.0,
                "text": cp["text"],
                "textangle": -90,
                "font_size": 12,
                "xanchor": "center",
                "yanchor": "middle",
            }
            half_ws.append(_estimate_label_box(probe, xlim, init_ylim)[2])
        assigned_x = _assign_compound_sides(comps_x, half_ws, xlim)
        for cp, ax in zip(compound_phase_list, assigned_x):
            # Stacked compounds (>1 phase at the same composition) sit just inside the diagram on
            # one side, separated by their own stability bands rather than left/right gaps.
            label_x = (cp["comp"] + (1.5 if cp["comp"] <= 50 else -1.5)) if cp["is_stacked"] else ax
            probe = {
                "x": label_x,
                "y": 0.0,
                "text": cp["text"],
                "textangle": -90,
                "font_size": 12,
                "xanchor": "center",
                "yanchor": "middle",
            }
            _, _, hw, hh = _estimate_label_box(probe, xlim, init_ylim)
            compound_cols.append(
                dict(
                    comp=cp["comp"],
                    label_x=label_x,
                    text=cp["text"],
                    is_stacked=cp["is_stacked"],
                    t_min=cp["t_min"],
                    t_max=cp["t_max"],
                    half_w=hw,
                    half_h_frac=hh / (init_ylim[1] - init_ylim[0]),
                    ceil_raw=_env_min(liq_floor, label_x, hw),
                    float_base=_env_max(liq_high, label_x, hw),
                    tie_temps=[
                        T for x0, x1, T in tie_segments if x1 >= label_x - hw and x0 <= label_x + hw
                    ],
                )
            )

    def _layout_compounds(c0, c1):
        """Per compound column: ('inband', y) | ('lower', needed_c0) | ('float', half_h_frac).
        ``inband`` puts the label at the bottom of the lowest two-phase region that fits (with a
        consistent bottom pad), skipping up past any tie line that would cross it (bugs 1 & 2)."""
        span = c1 - c0
        margin, gap, pad = 0.012 * span, 0.004 * span, _BOTTOM_PAD_FRAC * span
        out = []
        for col in compound_cols:
            hh = col["half_h_frac"] * span
            ceil = (col["ceil_raw"] - margin) if col["ceil_raw"] is not None else c1
            if col["is_stacked"]:
                region = (max(col["t_min"], c0), min(col["t_max"], c1))
                y = _pack_labels([region], [hh], ceil, col["tie_temps"], c0, margin, gap)[0]
            else:
                y = _place_compound_y(c0, ceil, col["tie_temps"], hh, pad, margin)
            if y is not None:
                out.append(("inband", y))
                continue
            # Escalate: lower conds[0] so the bottom region grows enough; else float.
            cuts = sorted(T for T in col["tie_temps"] if c0 < T < ceil)
            gap_top = cuts[0] if cuts else ceil
            needed_c0 = gap_top - 2 * hh - pad - margin
            if np.isfinite(gap_top) and needed_c0 >= _ABS_ZERO:
                out.append(("lower", needed_c0))
            else:
                out.append(("float", col["half_h_frac"]))
        return out

    # Fixed point: 'lower' lowers conds[0]; floats + the legend raise conds[1]. Both change the
    # y-span (hence label heights), so iterate until conds (and placement) are stable.
    legend_params, float_top_by_side = {}, {0: -np.inf, 100: -np.inf}
    for _ in range(8):
        c0_before, c1_before = hsx.conds[0], hsx.conds[1]
        float_top_by_side = {0: -np.inf, 100: -np.inf}
        for side_comp, d in sides_data.items():
            placements = _layout_side(side_comp, hsx.conds[0], hsx.conds[1])
            for p in placements:
                if p[0] == "lower":
                    hsx.conds[0] = max(min(hsx.conds[0], p[1]), _ABS_ZERO)
            float_fracs = [p[1] for p in placements if p[0] == "float"]
            if float_fracs and d["float_base"] is not None:
                m = clear_frac + sum(2 * hf for hf in float_fracs) + gap_frac * len(float_fracs)
                if m < 0.95:
                    side_c1 = (d["float_base"] - m * hsx.conds[0]) / (1.0 - m)
                    hsx.conds[1] = max(hsx.conds[1], side_c1)
                    float_top_by_side[side_comp] = side_c1
        # Compound columns can also lower conds[0] (region-fit) or raise conds[1] (float).
        for col, p in zip(compound_cols, _layout_compounds(hsx.conds[0], hsx.conds[1])):
            if p[0] == "lower":
                hsx.conds[0] = max(min(hsx.conds[0], p[1]), _ABS_ZERO)
            elif p[0] == "float" and col["float_base"] is not None:
                m = clear_frac + 2 * p[1] + gap_frac
                if m < 0.95:
                    hsx.conds[1] = max(
                        hsx.conds[1], (col["float_base"] - m * hsx.conds[0]) / (1.0 - m)
                    )
        legend_params = _place_legend(
            hsx,
            liq_df,
            assessed_pts,
            xlim,
            n_entries=len(entry_labels),
            max_label_chars=max(len(s) for s in entry_labels),
            float_top_by_side=float_top_by_side,
        )
        if abs(hsx.conds[0] - c0_before) < 1e-6 and abs(hsx.conds[1] - c1_before) < 1e-6:
            break

    # Final emission with the converged conds.
    span = hsx.conds[1] - hsx.conds[0]
    clear_d, gap_d = clear_frac * span, gap_frac * span
    for side_comp, d in sides_data.items():
        placements = _layout_side(side_comp, hsx.conds[0], hsx.conds[1])
        floated = []
        for r, text, hf, p in zip(d["side"], d["texts"], d["half_hs_frac"], placements):
            kind = p[0]
            label = {
                "text": text,
                "xanchor": "center",
                "yanchor": "middle",
                "textangle": -90,
                "font_size": 12,
                "font_color": "black",
                "pin": True,
            }
            # Visible extent of this region under the converged conds (raw t_bottom may sit
            # below conds[0] for a ground-state trace); the home/midpoint use the visible band.
            vis_bottom = max(r["t_bottom"], hsx.conds[0])
            vis_top = min(r["t_max"], hsx.conds[1])
            mid = 0.5 * (vis_bottom + vis_top)
            home_y = min(max(mid, hsx.conds[0]), hsx.conds[1])
            if kind == "inband":
                label["x"], label["y"] = d["label_x"], p[1]
            elif kind == "twophase":
                # In the two-phase region just inside the element, with an arrow to the phase.
                label["x"], label["y"] = d["fx"], p[1]
                label["home_x"] = 99.5 if side_comp == 100 else 0.5
                label["home_y"] = home_y
            else:  # 'float' (or a 'lower' that could not reach absolute zero)
                label["x"] = d["fx"]
                label["home_x"] = 99.5 if side_comp == 100 else 0.5
                label["home_y"] = home_y
                label["above_liquidus"] = True
                base = d["float_base"] if d["float_base"] is not None else vis_top
                floated.append((label, base, hf, mid))
            label_dicts.append(label)
        # Stack this side's floated labels just above the liquidus (bottom-to-top by temperature).
        cursor = (max(b for _, b, _, _ in floated) + clear_d) if floated else 0.0
        for label, _base, hf, _mid in sorted(floated, key=lambda t: t[3]):
            hh_d = hf * span
            label["y"] = cursor + hh_d
            cursor = label["y"] + hh_d + gap_d

    # --- Emit compound labels with the converged conds (region-fit + bottom pad + escalation). ---
    margin_d, pad_d = 0.012 * span, _BOTTOM_PAD_FRAC * span
    for col, p in zip(compound_cols, _layout_compounds(hsx.conds[0], hsx.conds[1])):
        kind = p[0]
        label = {
            "text": col["text"],
            "xanchor": "center",
            "yanchor": "middle",
            "textangle": -90,
            "font_size": 12,
            "font_color": "black",
        }
        vis_bottom, vis_top = max(col["t_min"], hsx.conds[0]), min(col["t_max"], hsx.conds[1])
        mid = 0.5 * (vis_bottom + vis_top)
        if kind == "float":
            # Float above the local liquidus with a leader arrow pointing to the compound.
            label["x"] = col["label_x"]
            label["home_x"], label["home_y"] = (
                col["comp"],
                min(max(mid, hsx.conds[0]), hsx.conds[1]),
            )
            label["above_liquidus"] = True
            base = col["float_base"] if col["float_base"] is not None else vis_top
            label["y"] = base + clear_d + col["half_h_frac"] * span
        else:  # 'inband' / 'lower' -> a bottom-region slot under the final conds
            ceil = (col["ceil_raw"] - margin_d) if col["ceil_raw"] is not None else hsx.conds[1]
            y = (
                p[1]
                if kind == "inband"
                else _place_compound_y(
                    hsx.conds[0], ceil, col["tie_temps"], col["half_h_frac"] * span, pad_d, margin_d
                )
            )
            if y is None:
                y = hsx.conds[0] + pad_d + col["half_h_frac"] * span
            label["x"], label["y"] = col["label_x"], y
            # Home at the placed position: a pure vertical de-collision nudge (the bug-3 forced
            # stacking fallback) must NOT spawn a leader arrow back to an arbitrary point.
            label["home_x"], label["home_y"] = col["label_x"], y
            label["no_arrow"] = True
        label_dicts.append(label)

    # --- Resolve label collisions and draw all collected labels ---
    ylim = (hsx.conds[0], hsx.conds[1])
    # Keep in-band labels below the lower of the two liquidus curves.
    liquidus_ceiling = (
        _liquidus_top_fn(liq_df, assessed_pts, combine="min") if not liq_df.empty else None
    )
    resolved_labels = _resolve_label_collisions(
        label_dicts,
        xlim,
        ylim,
        max_iterations=80,
        ceiling=liquidus_ceiling,
        tie_segments=tie_segments,
    )
    px_per_x = _PLOT_W_PX / (xlim[1] - xlim[0])
    px_per_y = _PLOT_H_PX / (ylim[1] - ylim[0])
    for lbl in resolved_labels:
        common = dict(
            text=lbl["text"],
            textangle=lbl.get("textangle", -90),
            font=dict(size=lbl.get("font_size", 12), color=lbl.get("font_color", "black")),
        )
        if lbl.get("showarrow") and not lbl.get("no_arrow"):
            ax_px = (lbl["x"] - lbl["home_x"]) * px_per_x
            ay_px = -(lbl["y"] - lbl["home_y"]) * px_per_y
            fig.add_annotation(
                x=lbl["home_x"],
                y=lbl["home_y"],
                ax=ax_px,
                ay=ay_px,
                showarrow=True,
                arrowhead=2,
                arrowwidth=1,
                arrowcolor="gray",
                xanchor=lbl.get("xanchor", "center"),
                yanchor=lbl.get("yanchor", "middle"),
                **common,
            )
        else:
            fig.add_annotation(
                x=lbl["x"],
                y=lbl["y"],
                showarrow=False,
                xanchor=lbl.get("xanchor", "center"),
                yanchor=lbl.get("yanchor", "middle"),
                **common,
            )

    if pred:
        phase_colors["L"] = PREDICTED_LIQUIDUS_COLOR
    fig.add_trace(
        px.line(liq_df, x="x", y="t", color="label", color_discrete_map=phase_colors).data[0]
    )
    fig.update_traces(line=dict(width=4), showlegend=False)

    # --- Solid-solution fields: one closed, '/'-hatched polygon per composition branch,
    # outlined uniformly in the phase color over a clear fill. The line-compound machinery
    # above never saw these labels; they render here (after the global width/legend reset
    # so their styling survives) with reserved colors and one legend entry each. ---
    for ss_name in [p for p in hsx.phases if p in ss_regions]:
        color = phase_colors.get(ss_name, "#555555")
        display = format_phase_display_name(ss_name, ss_phases, hsx.comps)
        for shown, region in enumerate(ss_regions[ss_name]):
            fig.add_trace(
                go.Scatter(
                    x=region["x"],
                    y=region["t"],
                    mode="lines",
                    line={"color": color, "width": _SS_LINE_W},
                    fill="toself",
                    fillcolor="rgba(0, 0, 0, 0)",
                    fillpattern={
                        "shape": "/",
                        "fgcolor": color,
                        "bgcolor": "rgba(0, 0, 0, 0)",
                        "size": _SS_HATCH_SIZE_PX,
                        "solidity": _SS_HATCH_SOLIDITY,
                    },
                    name=display,
                    showlegend=not shown,
                    hoverinfo="skip",
                )
            )
    if digitized_liquidus:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=ASSESSED_LIQUIDUS_COLOR, dash="dash"),
                name="Assessed Liquidus",
                showlegend=True,
            )
        )
    if imputed_phases:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Imputed Phase",
                showlegend=True,
            )
        )
    if pred:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                marker=dict(color=PREDICTED_LIQUIDUS_COLOR),
                name="Predicted Liquidus",
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                marker=dict(color=LIQUID_COLOR),
                name="Fitted Liquidus",
                showlegend=True,
            )
        )
    # Legend corner/expansion was already determined above (conds[1] expanded as needed).
    fig.update_layout(
        title=dict(
            text=f"<b>{hsx.comps[0]}-{hsx.comps[1]} DFT-Referenced Phase Diagram</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18, color="black"),
            yanchor="bottom",
        ),
        xaxis=dict(range=[0, 100], title="Composition (at. %)"),
        yaxis=dict(range=[hsx.conds[0], hsx.conds[1]], title="Temperature (°C)", ticksuffix=" "),
        width=_FIG_W_PX,  # 960 for show()
        height=_FIG_H_PX,  # 700 for show()
        plot_bgcolor="white",
        font=dict(size=13, color="black"),
        showlegend=True,
        legend=legend_params,
        margin=dict(t=_MARGIN_T_PX, b=_MARGIN_B_PX, r=_MARGIN_R_PX),
    )
    axes_params_dict = dict(
        title_font=dict(size=16),
        title_standoff=8,  # Space between title and axis line
        mirror=True,  # Draws lines on all four sides
        showline=True,  # Shows the primary axis lines (bottom, left)
        linecolor="black",
        linewidth=1.5,
        ticks="outside",  # Places ticks outside the plot area, starting at the axis line
        tickcolor="black",
        ticklen=5,
        tickwidth=1,
        minor_ticks="outside",  # Places minor ticks outside
        minor=dict(tickcolor="black", ticklen=2, tickwidth=1, nticks=5),
    )
    fig.update_xaxes(tickformat=".0f", **axes_params_dict)
    fig.update_yaxes(**axes_params_dict)
    l_x, l_y = _place_liquid_label(
        liq_df, assessed_pts, (0.0, 100.0), (hsx.conds[0], hsx.conds[1]), font_size=14
    )
    fig.add_annotation(x=l_x, y=l_y, text="L", showarrow=False, font=dict(size=14, color="black"))
    fig.add_annotation(
        x=-0.05,
        y=-0.086,  # Position below the x-axis in paper coordinates
        xref="paper",
        yref="paper",
        text=hsx.comps[0],  # Use the component name from the data
        showarrow=False,
        font=dict(color="black", size=13.5),
        xanchor="left",
        yanchor="middle",
    )
    fig.add_annotation(
        x=1.05,
        y=-0.086,  # Position below the x-axis in paper coordinates
        xref="paper",
        yref="paper",
        text=hsx.comps[1],  # Use the component name from the data
        showarrow=False,
        font=dict(color="black", size=13.5),
        xanchor="right",
        yanchor="middle",
    )

    return fig


def build_polymorph_transitions(sys_obj) -> list[dict]:
    """Endpoint polymorph transition annotations for a binary system's TX plot.

    ``component_data`` values are ``ComponentRef`` objects (attribute access); polymorphs with
    an unknown transition temperature are excluded by the ``polymorphs`` property itself.
    One home for the annotation builder — BLPlotter and the ternary edge sub-plots share it.
    """
    transitions = []
    for i, comp in enumerate(sys_obj.components):
        ref = sys_obj.component_data.get(comp)
        polymorphs = ref.polymorphs if ref is not None else []
        if not polymorphs:
            continue
        ground_state_name = comp
        for phase in sys_obj.phases:
            if (
                phase.name != "L"
                and phase.composition is not None
                and phase.fraction_in(sys_obj.components)[0] == float(i)
            ):
                if phase.enthalpy == 0:
                    ground_state_name = phase.name
                    break
        for poly in polymorphs:
            transitions.append(
                {
                    "name": poly.name,
                    "comp_x_pct": float(i) * 100,
                    "transition_temp_C": poly.t_transition - 273.15,
                    "ground_state_name": ground_state_name,
                }
            )
    return transitions
