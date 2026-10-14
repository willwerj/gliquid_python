"""Ternary liquidus projections drawn with matplotlib.

A projection draws one liquidus surface on a flat composition triangle: filled isotherms,
labelled isotherm lines, the smoothed monovariant lines between primary-crystallization fields,
one non-overlapping label per field, the corner names and outer composition ticks.

The entry points are :func:`draw_liquidus_panel` for the whole triangle and
:func:`draw_closeup_panel` for a rectangular window of it. Both take a record
``{"nodes": {"x0", "x1", "T_K", "primary"}, "delta": delta}``, optionally with
``"imputed_phase_names"``: ``x0`` and ``x1`` are the mole fractions of
``sorted_components[1]`` and ``[2]`` on a grid of step ``delta``, ``T_K`` the liquidus in kelvin
and ``primary`` the comma-joined solid assemblage at each node. ``order`` names the
(lower left, lower right, apex) corners the triangle is drawn in.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

KELVIN = 273.15
SQRT3_OVER_2 = math.sqrt(3.0) / 2.0

# Denominator of the integer lattice monovariant segment endpoints are keyed on (holds halves
# and thirds exactly).
LATTICE_REFINEMENT = 6
# Vertices in the centred moving average along a monovariant chain; chain endpoints stay fixed.
MA_WINDOW = 7
# Chaikin corner-cutting rounds after the moving average.
SMOOTH_ROUNDS = 4
# Smallest primary field, in lattice nodes, that gets a label.
MIN_LABELLED_NODES = 12
# Field-label box in triangle units: width per character, width padding and height.
LABEL_CHAR_W, LABEL_PAD_W, LABEL_H = 0.027, 0.026, 0.064

# Font sizes (pt) and colours of the panels.
FONTS = {"corner": 17, "field": 12, "isotherm": 10, "tick": 10}
COLOURS = {"text": "#0b0b0b", "isotherm": "#3d3c39", "monovariant": "#b3261e", "imputed": "#1baf7a"}
# Sequential ramp of the filled liquidus surface, light (cold) to dark (hot).
LIQUIDUS_RAMP = ("#f7fafe", "#e3eefc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec")

# The six lattice steps to the neighbours of a triangular-grid node.
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


# --- composition geometry -------------------------------------------------------------


def barycentric(record: dict, sorted_components: list[str]) -> dict:
    """``{element: [mole fraction, ...]}`` for a record's liquidus nodes."""
    nodes = record["nodes"]
    x0, x1 = nodes["x0"], nodes["x1"]
    return {
        sorted_components[0]: [1.0 - a - b for a, b in zip(x0, x1)],
        sorted_components[1]: list(x0),
        sorted_components[2]: list(x1),
    }


def project(bary: dict, order: list[str]) -> tuple[list[float], list[float]]:
    """Cartesian ``(x, y)`` with ``order`` at lower left, lower right and apex."""
    left, right, apex = (bary[e] for e in order)
    xs = [r + 0.5 * a for r, a in zip(right, apex)]
    ys = [SQRT3_OVER_2 * a for a in apex]
    return xs, ys


def to_cartesian(points, sorted_components: list[str], order: list[str]) -> list:
    """``(x0, x1)`` grid points into the ``order`` corner frame."""
    if not points:
        return []
    record = {"nodes": {"x0": [p[0] for p in points], "x1": [p[1] for p in points]}}
    xs, ys = project(barycentric(record, sorted_components), order)
    return list(zip(xs, ys))


# --- primary fields and monovariant lines ---------------------------------------------


def assemblage_lattice(nodes: dict, delta: float) -> dict:
    """``{(i, j): {phase, ...}}`` over the composition lattice, empty assemblages dropped."""
    out = {}
    for x0, x1, assemblage in zip(nodes["x0"], nodes["x1"], nodes["primary"]):
        if assemblage:
            out[(round(x0 / delta), round(x1 / delta))] = frozenset(
                part.strip() for part in assemblage.split(",") if part.strip()
            )
    return out


def resolve_primary(assemblages: dict) -> dict:
    """One primary phase per node, chosen from the solids that node coexists with.

    Nodes naming a single solid settle immediately. Each remaining node then takes whichever
    of its own solids the most already-settled neighbours carry, repeatedly, so a coexistence
    band is split up its middle. Ties and unreachable nodes take the alphabetically first
    solid.
    """
    settled = {node: next(iter(phases)) for node, phases in assemblages.items() if len(phases) == 1}
    pending = {node: phases for node, phases in assemblages.items() if len(phases) > 1}
    while pending:
        progress = False
        for node in sorted(pending):
            phases = pending[node]
            counts = Counter()
            for di, dj in _NEIGHBOURS:
                label = settled.get((node[0] + di, node[1] + dj))
                if label in phases:
                    counts[label] += 1
            if counts:
                settled[node] = max(sorted(counts), key=counts.__getitem__)
                del pending[node]
                progress = True
        if not progress:
            settled.update({node: min(phases) for node, phases in pending.items()})
            break
    return settled


def primary_lattice(nodes: dict, delta: float) -> dict:
    """``{(i, j): primary phase}`` over the composition lattice, one solid per node."""
    return resolve_primary(assemblage_lattice(nodes, delta))


def _triangles(lattice: dict):
    """Every upward and downward lattice triangle whose three nodes are all present."""
    for i, j in lattice:
        up = ((i, j), (i + 1, j), (i, j + 1))
        if all(node in lattice for node in up):
            yield up
        down = ((i + 1, j), (i, j + 1), (i + 1, j + 1))
        if all(node in lattice for node in down):
            yield down


def boundary_segments(lattice: dict) -> list:
    """Marching triangles over the primary-phase field, on the sixfold integer lattice.

    A triangle carrying two phases contributes the segment joining its two separating edge
    midpoints; a triangle carrying three (an invariant point) contributes three segments from
    its edge midpoints to its centroid. Returns ``[((a0, a1), (b0, b1)), ...]`` in units of
    ``delta / LATTICE_REFINEMENT``.
    """
    half = LATTICE_REFINEMENT // 2
    third = LATTICE_REFINEMENT // 3
    segments = []
    for triangle in _triangles(lattice):
        labels = [lattice[node] for node in triangle]
        distinct = set(labels)
        if len(distinct) == 1:
            continue
        midpoints = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if labels[a] == labels[b]:
                continue
            midpoints.append(
                (half * (triangle[a][0] + triangle[b][0]), half * (triangle[a][1] + triangle[b][1]))
            )
        if len(distinct) == 2:
            segments.append((midpoints[0], midpoints[1]))
            continue
        centroid = (
            third * sum(node[0] for node in triangle),
            third * sum(node[1] for node in triangle),
        )
        segments.extend((midpoint, centroid) for midpoint in midpoints)
    return segments


def chain_segments(segments: list) -> list:
    """Segments joined at shared endpoints into polylines that break at every junction, so
    each polyline runs between invariant points or domain edges. Closed loops carrying no
    junction are emitted whole."""
    incident = defaultdict(list)
    for index, (a, b) in enumerate(segments):
        incident[a].append(index)
        incident[b].append(index)
    used = [False] * len(segments)

    def walk(start):
        path = [start]
        point = start
        while True:
            if len(path) > 1 and len(incident[point]) != 2:
                break
            step = next((i for i in incident[point] if not used[i]), None)
            if step is None:
                break
            used[step] = True
            a, b = segments[step]
            point = b if a == point else a
            path.append(point)
        return path

    chains = []
    for point in [p for p, links in incident.items() if len(links) != 2]:
        while any(not used[i] for i in incident[point]):
            chains.append(walk(point))
    for index, (a, _b) in enumerate(segments):
        if not used[index]:
            chains.append(walk(a))
    return [chain for chain in chains if len(chain) > 1]


def chaikin(points: list, rounds: int = SMOOTH_ROUNDS) -> list:
    """``points`` with its corners cut ``rounds`` times, endpoints held fixed."""
    closed = len(points) > 2 and points[0] == points[-1]
    for _ in range(rounds):
        if len(points) < 3:
            return points
        out = [points[0]]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        out.append(points[-1])
        points = out
        if closed:
            points[0] = points[-1]
    return points


def moving_average(points, window: int = MA_WINDOW):
    """``points`` with each interior vertex replaced by the centred mean of up to ``window``
    neighbours along the chain; the two endpoints are kept exactly."""
    n = len(points)
    if n < 3 or window < 2:
        return list(points)
    half = window // 2
    out = [points[0]]
    for i in range(1, n - 1):
        k = min(half, i, n - 1 - i)
        seg = points[i - k : i + k + 1]
        out.append((sum(p[0] for p in seg) / len(seg), sum(p[1] for p in seg) / len(seg)))
    out.append(points[-1])
    return out


def smooth_polyline(points, window: int = MA_WINDOW, rounds: int = SMOOTH_ROUNDS) -> list:
    """A moving average of ``window`` vertices, then ``rounds`` of corner cutting."""
    return chaikin(moving_average(points, window), rounds)


def monovariant_polylines(
    nodes: dict,
    delta: float,
    sorted_components,
    order,
    window: int = MA_WINDOW,
    rounds: int = SMOOTH_ROUNDS,
) -> list:
    """The monovariant lines of one liquidus surface, in the ``order`` corner frame.

    ``window=0, rounds=0`` returns the raw lattice chains.
    """
    scale = delta / LATTICE_REFINEMENT
    lines = []
    for chain in chain_segments(boundary_segments(primary_lattice(nodes, delta))):
        grid = [(i * scale, j * scale) for i, j in chain]
        lines.append(smooth_polyline(to_cartesian(grid, sorted_components, order), window, rounds))
    return lines


# --- field labels ---------------------------------------------------------------------


def label_box(x, y, text):
    """``(x0, y0, x1, y1)`` of a centred label, in triangle units."""
    half_w = (LABEL_CHAR_W * len(text) + LABEL_PAD_W) / 2.0
    return x - half_w, y - LABEL_H / 2.0, x + half_w, y + LABEL_H / 2.0


def boxes_overlap(a, b) -> bool:
    """Whether boxes ``a`` and ``b``, each ``(x0, y0, x1, y1)``, overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_field_labels(
    lattice: dict,
    delta: float,
    sorted_components,
    order,
    imputed=frozenset(),
    min_nodes: int = MIN_LABELLED_NODES,
    priority=frozenset(),
    window=None,
    avoid=(),
    box=None,
    lines=None,
) -> list:
    """``[(x, y, phase)]``: one label per primary field, none overlapping another.

    Imputed phases are placed first, then ``priority`` phases, then larger fields. Each field's
    nodes are ranked by distance to the nearest other field and to the triangle's sides, and
    the label goes on the best node whose box clears every label already placed. With no
    clear node, an imputed phase takes its best node anyway and any other field goes
    unlabelled. Fields smaller than ``min_nodes`` are not labelled.

    ``window`` ``(x0, x1, y0, y1)`` limits labels to boxes lying wholly inside it (an imputed phase
    with no such spot still takes its best node) and counts only the nodes inside it towards
    ``min_nodes``; ``avoid`` is a list of ``(x0, y0, x1, y1)`` boxes no label may overlap;
    ``box(x, y, text)`` replaces ``label_box`` as the label's extent; ``lines``, polylines in the
    same frame (the monovariant lines), are crossed by a label only when no other spot is free.
    """
    box = box or label_box
    segments = _segment_array(lines) if lines else None
    positions = dict(
        zip(
            lattice,
            to_cartesian([(i * delta, j * delta) for i, j in lattice], sorted_components, order),
        )
    )
    fields = {}
    for node, phase in lattice.items():
        if window is None or (
            window[0] <= positions[node][0] <= window[1]
            and window[2] <= positions[node][1] <= window[3]
        ):
            fields.setdefault(phase, []).append(node)
    placed, boxes = [], list(avoid)
    for phase in sorted(
        fields, key=lambda p: (p not in imputed, p not in priority, -len(fields[p]), p)
    ):
        inside = fields[phase]
        if len(inside) < min_nodes:
            continue
        outside = [positions[n] for n in lattice if lattice[n] != phase]

        def score(node, outside=outside):
            x, y = positions[node]
            neighbours = (
                min(math.hypot(x - u, y - v) for u, v in outside) if outside else float("inf")
            )
            sides = min(y, (math.sqrt(3.0) * x - y) / 2.0, (math.sqrt(3.0) * (1.0 - x) - y) / 2.0)
            return min(neighbours, sides), neighbours

        text = phase + ("*" if phase in imputed else "")
        ranked = sorted(inside, key=score, reverse=True)
        if window is not None:
            fits = [n for n in ranked if _box_inside(box(*positions[n], text), window)]
            if not fits and phase not in imputed:
                continue
            ranked = fits or ranked

        def clear(n, text=text):
            return not any(boxes_overlap(box(*positions[n], text), b) for b in boxes)

        chosen = None
        if segments is not None:
            chosen = next(
                (
                    n
                    for n in ranked
                    if clear(n) and not _hits_segments(segments, box(*positions[n], text))
                ),
                None,
            )
        if chosen is None:
            chosen = next((n for n in ranked if clear(n)), None)
        if chosen is None:
            if phase not in imputed:
                continue
            chosen = ranked[0]
        x, y = positions[chosen]
        placed.append((x, y, phase))
        boxes.append(box(x, y, text))
    return placed


def clip_segment(p0, p1, window):
    """The part of segment ``p0``-``p1`` inside ``window`` ``(x0, x1, y0, y1)``, or None."""
    (x0, y0), (x1, y1) = p0, p1
    lo, hi = 0.0, 1.0
    for delta, low, high, start in (
        (x1 - x0, window[0], window[1], x0),
        (y1 - y0, window[2], window[3], y0),
    ):
        if delta == 0.0:
            if not low <= start <= high:
                return None
            continue
        a, b = (low - start) / delta, (high - start) / delta
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        if lo >= hi:
            return None
    return ((x0 + lo * (x1 - x0), y0 + lo * (y1 - y0)), (x0 + hi * (x1 - x0), y0 + hi * (y1 - y0)))


def _segment_array(polylines):
    """``(n, 4)`` array of ``x0, y0, x1, y1`` for every segment of ``polylines``."""
    import numpy as np

    rows = [(a[0], a[1], b[0], b[1]) for line in polylines for a, b in zip(line, line[1:])]
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def _hits_segments(segments, b) -> bool:
    """Whether any of ``segments`` (see ``_segment_array``) crosses box ``b``
    ``(x0, y0, x1, y1)``."""
    import numpy as np

    near = segments[
        (np.minimum(segments[:, 0], segments[:, 2]) <= b[2])
        & (np.maximum(segments[:, 0], segments[:, 2]) >= b[0])
        & (np.minimum(segments[:, 1], segments[:, 3]) <= b[3])
        & (np.maximum(segments[:, 1], segments[:, 3]) >= b[1])
    ]
    window = (b[0], b[2], b[1], b[3])
    return any(clip_segment((s[0], s[1]), (s[2], s[3]), window) is not None for s in near)


def _box_inside(b, window) -> bool:
    """Whether box ``b`` ``(x0, y0, x1, y1)`` lies wholly inside ``window`` ``(x0, x1, y0, y1)``."""
    return b[0] >= window[0] and b[2] <= window[1] and b[1] >= window[2] and b[3] <= window[3]


# --- drawing --------------------------------------------------------------------------


def nice_levels(
    vmin_c: float, vmax_c: float, target: int = 10, steps: tuple = (25, 50, 100, 200, 250, 500)
) -> list[float]:
    """Round isotherm levels in degC: the entry of ``steps`` giving ~``target`` lines."""
    span = max(vmax_c - vmin_c, 1.0)
    step = min(steps, key=lambda s: abs(span / s - target))
    lo = math.ceil(vmin_c / step) * step
    return [float(v) for v in range(int(lo), int(vmax_c) + 1, step)]


def levels_for(records, n_fill: int = 25) -> dict:
    """Shared colour scale for several records: ``{"fill", "lines", "cmap", "norm"}``, degC."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    temps_c = np.concatenate([np.asarray(r["nodes"]["T_K"]) - KELVIN for r in records])
    vmin, vmax = float(np.min(temps_c)), float(np.max(temps_c))
    fill = np.linspace(vmin, vmax, n_fill)
    cmap = LinearSegmentedColormap.from_list("liq", list(LIQUIDUS_RAMP))
    return {
        "fill": fill,
        "lines": nice_levels(vmin, vmax),
        "cmap": cmap,
        "norm": BoundaryNorm(fill, cmap.N),
    }


def triangle_frame(
    ax,
    order,
    fontsize: float = FONTS["corner"],
    ticks: bool = True,
    tick_fontsize: float = FONTS["tick"],
):
    """The triangle outline and corner element labels, ``order`` = (lower left, lower right,
    apex). With ``ticks``, composition ticks at ``TICK_STEP`` on the edges, labelled and titled
    (``draw_triangle_ticks``), and room below the triangle for them."""
    h = SQRT3_OVER_2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", linewidth=1.0)
    if ticks:
        draw_triangle_ticks(ax, order, tick_fontsize)
    ax.text(-0.02, -0.03, order[0], ha="right", va="top", fontsize=fontsize, weight="bold")
    ax.text(1.02, -0.03, order[1], ha="left", va="top", fontsize=fontsize, weight="bold")
    ax.text(0.5, h + 0.03, order[2], ha="center", va="bottom", fontsize=fontsize, weight="bold")
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.17 if ticks else -0.12, h + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")


def edge_points(order, edge: str, lo: float, hi: float, offset: float):
    """Cartesian ``[(x, y), (x, y)]`` of the composition interval ``[lo, hi]`` (fraction of
    ``edge``'s second element) on ``edge``, pushed ``offset`` outward from the triangle."""
    import numpy as np

    a, b = edge.split("-")
    corners = {
        order[0]: np.array([0.0, 0.0]),
        order[1]: np.array([1.0, 0.0]),
        order[2]: np.array([0.5, SQRT3_OVER_2]),
    }
    pa, pb = corners[a], corners[b]
    centroid = sum(corners.values()) / 3.0
    mid = (pa + pb) / 2.0
    normal = mid - centroid
    normal = normal / np.linalg.norm(normal)
    p0 = pa + lo * (pb - pa) + offset * normal
    p1 = pa + hi * (pb - pa) + offset * normal
    return [tuple(p0), tuple(p1)]


def draw_liquidus_panel(
    ax,
    record: dict,
    sorted_components,
    order,
    levels: dict,
    imputed=None,
    fonts: dict | None = None,
    colours: dict | None = None,
    frame: bool = True,
    isotherm_labels: bool = True,
    highlight: dict | None = None,
    ticks: bool = True,
):
    """Draw one liquidus projection over the whole composition triangle.

    Draws filled isotherms, labelled isotherm lines, the smoothed monovariant lines, one label
    per primary field and, with ``frame``, the triangle with its corner names.

    Args:
        ax: the matplotlib axes to draw on.
        record: ``{"nodes": {"x0", "x1", "T_K", "primary"}, "delta": delta}``, optionally with
            ``"imputed_phase_names"``.
        sorted_components: the three elements in sorted order; ``x0`` and ``x1`` are the mole
            fractions of ``sorted_components[1]`` and ``[2]``.
        order: the (lower left, lower right, apex) corner elements.
        levels: the colour scale from ``levels_for``.
        imputed: phases whose fields are marked ``*`` in a dashed box; None reads the record's
            ``imputed_phase_names``.
        fonts: font sizes (pt) overriding ``FONTS``.
        colours: colours overriding ``COLOURS``.
        frame: draw the triangle outline and corner names (``triangle_frame``).
        isotherm_labels: label every second isotherm line.
        highlight: ``{phase: colour}``; these fields are labelled first, in a solid box of that
            colour.
        ticks: with ``frame``, draw the outer composition ticks (``draw_triangle_ticks``).

    Returns:
        The filled contour set.
    """
    import matplotlib.tri as mtri
    import numpy as np

    fonts = {**FONTS, **(fonts or {})}
    colours = {**COLOURS, **(colours or {})}
    nodes = record["nodes"]
    delta = record["delta"]
    xs, ys = project(barycentric(record, sorted_components), order)
    tri = mtri.Triangulation(xs, ys)
    t_c = np.asarray(nodes["T_K"]) - KELVIN
    contour = ax.tricontourf(
        tri, t_c, levels=levels["fill"], cmap=levels["cmap"], norm=levels["norm"]
    )
    iso = ax.tricontour(
        tri, t_c, levels=levels["lines"], colors=colours["isotherm"], linewidths=0.55, alpha=0.75
    )
    if isotherm_labels:
        ax.clabel(
            iso,
            iso.levels[::2],
            fmt="%d",
            fontsize=fonts["isotherm"],
            inline=True,
            inline_spacing=2,
            colors=colours["isotherm"],
        )
    for poly in monovariant_polylines(nodes, delta, sorted_components, order):
        ax.plot(
            [p[0] for p in poly],
            [p[1] for p in poly],
            color=colours["monovariant"],
            linewidth=1.7,
            solid_capstyle="round",
            zorder=4,
        )
    lattice = primary_lattice(nodes, delta)
    imputed = set(record.get("imputed_phase_names") or []) if imputed is None else set(imputed)
    highlight = dict(highlight or {})
    for x, y, phase in place_field_labels(
        lattice, delta, sorted_components, order, imputed=imputed, priority=set(highlight)
    ):
        is_imp = phase in imputed
        edge = highlight.get(phase) or (colours["imputed"] if is_imp else "none")
        ax.text(
            x,
            y,
            phase + ("*" if is_imp else ""),
            fontsize=fonts["field"],
            ha="center",
            va="center",
            color=colours["text"],
            zorder=7,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": edge,
                "linestyle": (0, (2, 1.5)) if is_imp else "-",
                "linewidth": 1.6 if phase in highlight else 1.1,
            },
        )
    if frame:
        triangle_frame(ax, order, fonts["corner"], ticks=ticks, tick_fontsize=fonts["tick"])
    return contour


# --- closeups -------------------------------------------------------------------------

# Isotherm steps (degC) a zoomed window may use.
CLOSEUP_STEPS = (2, 5, 10, 20, 25, 50, 100, 200)
# Closeup window width / height.
CLOSEUP_ASPECT = 1.25


def nodes_in_window(record: dict, sorted_components, order, window, pad: float = 0.0):
    """Boolean mask of ``record``'s nodes inside ``window`` grown by ``pad``."""
    import numpy as np

    xs, ys = project(barycentric(record, sorted_components), order)
    xs, ys = np.asarray(xs), np.asarray(ys)
    return (
        (xs >= window[0] - pad)
        & (xs <= window[1] + pad)
        & (ys >= window[2] - pad)
        & (ys <= window[3] + pad)
    )


def window_levels(
    records,
    sorted_components,
    order,
    window,
    n_fill: int = 25,
    target: int = 12,
    steps: tuple = CLOSEUP_STEPS,
    pad: float = 0.03,
) -> dict:
    """``levels_for`` over the nodes of ``records`` inside ``window``: the fill spans the window's
    temperature range (grown to the nodes within ``pad`` of it, which the contour triangles
    reach) and the isotherms sit on a round step of ``steps`` giving ~``target`` lines."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    inner, outer = [], []
    for r in records:
        t_c = np.asarray(r["nodes"]["T_K"]) - KELVIN
        inner.append(t_c[nodes_in_window(r, sorted_components, order, window)])
        outer.append(t_c[nodes_in_window(r, sorted_components, order, window, pad)])
    inner, outer = np.concatenate(inner), np.concatenate(outer)
    fill = np.linspace(float(outer.min()), float(outer.max()), n_fill)
    cmap = LinearSegmentedColormap.from_list("liq", list(LIQUIDUS_RAMP))
    return {
        "fill": fill,
        "cmap": cmap,
        "norm": BoundaryNorm(fill, cmap.N),
        "lines": nice_levels(float(inner.min()), float(inner.max()), target, steps),
    }


def text_box_factory(ax, fontsize: float, pad_points: float = 3.0, fontweight: str = "bold"):
    """``box(x, y, text)``: the ``(x0, y0, x1, y1)`` extent, in data units, of ``text`` centred on
    ``(x, y)`` and padded by ``pad_points``. Reads the axes' current transform, so its limits and
    aspect must already be set."""
    import numpy as np

    fig = ax.figure
    ax.apply_aspect()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    origin = np.asarray(to_data.transform((0.0, 0.0)))
    half = {}

    def size(text):
        if text not in half:
            probe = ax.text(0.0, 0.0, text, fontsize=fontsize, fontweight=fontweight)
            bb = probe.get_window_extent(renderer)
            probe.remove()
            pad = pad_points * fig.dpi / 72.0
            span = np.asarray(to_data.transform((bb.width + 2 * pad, bb.height + 2 * pad)))
            dx, dy = span - origin
            half[text] = (abs(float(dx)) / 2.0, abs(float(dy)) / 2.0)
        return half[text]

    def box(x, y, text):
        hw, hh = size(text)
        return x - hw, y - hh, x + hw, y + hh

    return box


def artist_boxes(ax, artists, pad_points: float = 2.0) -> list:
    """``(x0, y0, x1, y1)`` data-unit extents of text ``artists``, each padded by ``pad_points``."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    pad = pad_points * fig.dpi / 72.0
    out = []
    for artist in artists:
        bb = artist.get_window_extent(renderer)
        (x0, y0), (x1, y1) = to_data.transform(
            [(bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)]
        )
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return out


def _straightest(points, cumulative, centre: float, half: float, span: float):
    """Candidate label anchors on a polyline: ``[(deviation, index, angle)]`` for the vertices
    within ``half`` of arclength ``centre``, best (straightest over ``span``) first.
    ``deviation`` is the largest distance of the line from its chord over the label's length."""
    import numpy as np

    out = []
    for i in np.flatnonzero(abs(cumulative - centre) <= half):
        lo = int(np.searchsorted(cumulative, cumulative[i] - span / 2.0))
        hi = int(np.searchsorted(cumulative, cumulative[i] + span / 2.0))
        hi = min(hi, len(points) - 1)
        if hi <= lo:
            continue
        chord = points[hi] - points[lo]
        norm = float(np.hypot(*chord))
        if norm == 0.0:
            continue
        offsets = points[lo : hi + 1] - points[lo]
        deviation = float(
            np.max(np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])) / norm
        )
        out.append((deviation, int(i), math.atan2(chord[1], chord[0])))
    return sorted(out)


def label_isotherms(
    ax,
    contour,
    window,
    fmt: str = "%d",
    fontsize: float = 10,
    colour: str = COLOURS["isotherm"],
    spacing: float = 0.5,
    avoid=(),
    levels=None,
) -> list:
    """Label the isotherms of ``contour`` where they cross ``window``; return the label texts.

    Each stretch of an isotherm inside the window gets a label near its middle, and another
    every ``spacing`` of the window's width along a long one, put where the line is straightest.
    No label overlaps another or a box of ``avoid``; stretches too short to carry one stay
    unlabelled. ``levels`` limits which levels are labelled (default all). The line under each
    label is cut.
    """
    import numpy as np

    width = window[1] - window[0]
    margin = 0.05 * width
    inner = (window[0] + margin, window[1] - margin, window[2] + margin, window[3] - margin)
    box = text_box_factory(ax, fontsize, pad_points=2.0, fontweight="normal")
    wanted = set(float(v) for v in (contour.levels if levels is None else levels))
    runs = []
    for level, segments in zip(contour.levels, contour.allsegs):
        if float(level) not in wanted:
            continue
        for seg in segments:
            seg = np.asarray(seg)
            inside = (
                (seg[:, 0] >= inner[0])
                & (seg[:, 0] <= inner[1])
                & (seg[:, 1] >= inner[2])
                & (seg[:, 1] <= inner[3])
            )
            start = None
            for k in range(len(seg) + 1):
                if k < len(seg) and inside[k]:
                    start = k if start is None else start
                elif start is not None:
                    if k - start >= 3:
                        runs.append((float(level), seg[start:k]))
                    start = None
    placed_boxes, positions = list(avoid), []
    prepared = []
    for level, pts in runs:
        cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        prepared.append((cumulative[-1], level, pts, cumulative))
    for length, level, pts, cumulative in sorted(prepared, key=lambda r: -r[0]):
        text = fmt % level
        x0, y0, x1, y1 = box(0.0, 0.0, text)
        half_w, half_h = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        if length < 1.6 * (x1 - x0):
            continue
        count = min(3, 1 + int(length / (spacing * width)))
        for slot in range(count):
            centre = length * (slot + 0.5) / count
            reach = min(0.12 * length, length / (2.0 * count))
            for _dev, index, angle in _straightest(pts, cumulative, centre, reach, x1 - x0):
                if not (half_w <= cumulative[index] <= length - half_w):
                    continue
                cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
                hw, hh = cos * half_w + sin * half_h, sin * half_w + cos * half_h
                x, y = pts[index]
                candidate = (x - hw, y - hh, x + hw, y + hh)
                if any(boxes_overlap(candidate, b) for b in placed_boxes):
                    continue
                placed_boxes.append(candidate)
                positions.append((float(x), float(y)))
                break
    if not positions:
        return []
    ax.clabel(
        contour,
        levels=list(contour.levels),
        manual=positions,
        fmt=fmt,
        inline=True,
        inline_spacing=2,
        fontsize=fontsize,
        colors=colour,
    )
    return list(contour.labelTexts)


def draw_closeup_panel(
    ax,
    record: dict,
    sorted_components,
    order,
    levels: dict,
    window,
    imputed=None,
    fonts: dict | None = None,
    colours: dict | None = None,
    label_style: str = "halo",
    isotherm_spacing: float = 0.5,
    ticks: bool = True,
):
    """Draw a liquidus projection zoomed onto a rectangular window of the triangle.

    Every isotherm in view is labelled (``label_isotherms``), and the field labels are placed
    clear of those labels, of each other, of the corner names and, where a spot allows, of the
    monovariant lines. Corner names appear only where a corner lies in the window.

    Args:
        ax: the matplotlib axes to draw on; its limits are set to ``window``.
        record: as for ``draw_liquidus_panel``.
        sorted_components: as for ``draw_liquidus_panel``.
        order: as for ``draw_liquidus_panel``.
        levels: the colour scale from ``window_levels``.
        window: the visible region ``(x0, x1, y0, y1)`` in triangle units.
        imputed: phases whose fields are marked ``*`` with a dashed outline; None reads the
            record's ``imputed_phase_names``.
        fonts: font sizes (pt) overriding ``FONTS``.
        colours: colours overriding ``COLOURS``.
        label_style: ``"halo"`` sets field names in text with a white outline, ``"box"`` in the
            white boxes of ``draw_liquidus_panel``.
        isotherm_spacing: one isotherm label per this fraction of the window's width along a
            long line.
        ticks: frame the window with outer composition ticks (``draw_window_frame``).

    Returns:
        ``(filled contour set, labels)``, ``labels`` being the ``[(x, y, phase)]`` of
        ``place_field_labels``.
    """
    import matplotlib.patheffects as pe
    import matplotlib.tri as mtri
    import numpy as np

    fonts = {**FONTS, **(fonts or {})}
    colours = {**COLOURS, **(colours or {})}
    nodes, delta = record["nodes"], record["delta"]
    xs, ys = project(barycentric(record, sorted_components), order)
    tri = mtri.Triangulation(xs, ys)
    t_c = np.asarray(nodes["T_K"]) - KELVIN
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.apply_aspect()

    contour = ax.tricontourf(
        tri, t_c, levels=levels["fill"], cmap=levels["cmap"], norm=levels["norm"]
    )
    iso = ax.tricontour(
        tri, t_c, levels=levels["lines"], colors=colours["isotherm"], linewidths=0.6, alpha=0.8
    )
    corners = corner_name_boxes(ax, order, window, fonts["corner"])
    if ticks:
        draw_window_frame(ax, order, window, fonts["tick"] - 1.0)
    texts = label_isotherms(
        ax,
        iso,
        window,
        fmt="%d",
        fontsize=fonts["isotherm"],
        colour=colours["isotherm"],
        spacing=isotherm_spacing,
        avoid=corners,
    )
    keep_out = artist_boxes(ax, texts) + corners
    polylines = monovariant_polylines(nodes, delta, sorted_components, order)
    for poly in polylines:
        ax.plot(
            [p[0] for p in poly],
            [p[1] for p in poly],
            color=colours["monovariant"],
            linewidth=1.7,
            solid_capstyle="round",
            zorder=4,
        )
    h = SQRT3_OVER_2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", linewidth=1.0, zorder=6)

    imputed = set(record.get("imputed_phase_names") or []) if imputed is None else set(imputed)
    box = text_box_factory(ax, fonts["field"])
    lattice = primary_lattice(nodes, delta)
    labels = place_field_labels(
        lattice,
        delta,
        sorted_components,
        order,
        imputed=imputed,
        min_nodes=1,
        window=window,
        avoid=keep_out,
        box=box,
        lines=polylines,
    )
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    for x, y, phase in labels:
        is_imp = phase in imputed
        kwargs = {
            "fontsize": fonts["field"],
            "ha": "center",
            "va": "center",
            "zorder": 7,
            "color": colours["text"],
            "fontweight": "bold",
        }
        if label_style == "halo":
            kwargs["path_effects"] = halo
            if is_imp:
                kwargs["bbox"] = {
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "none",
                    "edgecolor": colours["imputed"],
                    "linestyle": (0, (2, 1.5)),
                    "linewidth": 1.2,
                }
        else:
            kwargs["bbox"] = {
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": colours["imputed"] if is_imp else "none",
                "linestyle": (0, (2, 1.5)) if is_imp else "-",
                "linewidth": 1.1,
            }
        ax.text(x, y, phase + ("*" if is_imp else ""), **kwargs)

    draw_corner_names(ax, order, window, fonts["corner"])
    return contour, labels


def _corner_anchors(order, window) -> list:
    """``[(name, x, y, ha, va)]`` for each triangle corner in ``window``: where its name is set,
    inside the window and in from the corner."""
    h = SQRT3_OVER_2
    inset = 0.03 * (window[1] - window[0])
    corners = (
        (order[0], 0.0, 0.0, inset, inset, "left", "bottom"),
        (order[1], 1.0, 0.0, -inset, inset, "right", "bottom"),
        (order[2], 0.5, h, 0.0, -inset, "center", "top"),
    )
    return [
        (name, cx + dx, cy + dy, ha, va)
        for name, cx, cy, dx, dy, ha, va in corners
        if window[0] <= cx <= window[1] and window[2] <= cy <= window[3]
    ]


def draw_corner_names(ax, order, window, fontsize: float) -> None:
    """Name each triangle corner that lies in ``window``, inside it, set in from the corner."""
    import matplotlib.patheffects as pe

    for name, x, y, ha, va in _corner_anchors(order, window):
        ax.text(
            x,
            y,
            name,
            fontsize=fontsize,
            weight="bold",
            ha=ha,
            va=va,
            zorder=8,
            path_effects=[pe.withStroke(linewidth=3.2, foreground="white")],
        )


def corner_name_boxes(ax, order, window, fontsize: float) -> list:
    """``(x0, y0, x1, y1)`` boxes of the names ``draw_corner_names`` sets, for labels to avoid."""
    box = text_box_factory(ax, fontsize, pad_points=2.0)
    out = []
    for name, x, y, ha, va in _corner_anchors(order, window):
        x0, y0, x1, y1 = box(0.0, 0.0, name)
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        cx = x + {"left": hw, "right": -hw, "center": 0.0}[ha]
        cy = y + {"bottom": hh, "top": -hh}[va]
        out.append((cx - hw, cy - hh, cx + hw, cy + hh))
    return out


def draw_locator(ax, order, window, fontsize: float = 9, colour: str = "#b3261e") -> None:
    """A small triangle outline on ``ax`` with ``window`` marked on it and the corner names."""
    from matplotlib.patches import Rectangle

    h = SQRT3_OVER_2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", linewidth=0.9)
    ax.add_patch(
        Rectangle(
            (window[0], window[2]),
            window[1] - window[0],
            window[3] - window[2],
            facecolor=colour,
            alpha=0.25,
            edgecolor=colour,
            linewidth=1.2,
        )
    )
    ax.text(-0.04, -0.04, order[0], ha="right", va="top", fontsize=fontsize, weight="bold")
    ax.text(1.04, -0.04, order[1], ha="left", va="top", fontsize=fontsize, weight="bold")
    ax.text(0.5, h + 0.04, order[2], ha="center", va="bottom", fontsize=fontsize, weight="bold")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.25, h + 0.22)
    ax.set_aspect("equal")
    ax.axis("off")


# --- composition ticks ----------------------------------------------------------------

# Composition step (mole fraction) between ticks.
TICK_STEP = 0.1
# Tick length in triangle units, outside the edge, along the tick's direction.
TICK_LENGTH = 0.016
# Distance from the edge to a tick label, and to the axis title, in triangle units.
TICK_LABEL_OFFSET, TICK_TITLE_OFFSET = 0.045, 0.135
# Colour of tick marks and labels.
TICK_COLOUR = "#3d3c39"


def edge_ticks(order, step: float = TICK_STEP) -> list:
    """The composition ticks of a triangle drawn in the ``order`` corner frame.

    Each edge carries the composition of the corner element it runs towards, counterclockwise:
    the bottom edge ``order[1]`` (rising to the right), the right edge ``order[2]`` (rising
    upward), the left edge ``order[0]`` (rising downward). A tick lies along the line of constant
    composition through its point, so its direction is that of the triangle side opposite the
    element. Returns ``[{"edge", "element", "value", "point", "direction", "normal"}]``, with
    ``direction`` the unit vector pointing into the triangle and ``normal`` the edge's outward
    unit normal.
    """
    h = SQRT3_OVER_2
    out = []
    for k in range(1, int(round(1.0 / step))):
        v = round(k * step, 6)
        out.append(
            {
                "edge": "bottom",
                "element": order[1],
                "value": v,
                "point": (v, 0.0),
                "direction": (0.5, h),
                "normal": (0.0, -1.0),
            }
        )
        out.append(
            {
                "edge": "right",
                "element": order[2],
                "value": v,
                "point": (1.0 - 0.5 * v, h * v),
                "direction": (-1.0, 0.0),
                "normal": (h, 0.5),
            }
        )
        out.append(
            {
                "edge": "left",
                "element": order[0],
                "value": v,
                "point": (0.5 * (1.0 - v), h * (1.0 - v)),
                "direction": (0.5, -h),
                "normal": (-h, 0.5),
            }
        )
    return out


_EDGE_ALIGN = {
    "bottom": ("center", "top"),
    "right": ("left", "center"),
    "left": ("right", "center"),
}
_EDGE_TITLE_ROTATION = {"bottom": 0.0, "right": -60.0, "left": 60.0}


def draw_triangle_ticks(
    ax, order, fontsize: float = 9.0, step: float = TICK_STEP, colour: str = TICK_COLOUR
) -> None:
    """Composition ticks on the three edges (see ``edge_ticks``), each labelled in at.% of its
    edge's element (the number only), with an "at.% X" title centred beside each edge. Ticks lie
    outside the edge."""
    h = SQRT3_OVER_2
    for tick in edge_ticks(order, step):
        (px, py), (dx, dy), (nx, ny) = tick["point"], tick["direction"], tick["normal"]
        ax.plot(
            [px, px - TICK_LENGTH * dx],
            [py, py - TICK_LENGTH * dy],
            color=colour,
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=6,
        )
        ha, va = _EDGE_ALIGN[tick["edge"]]
        ax.text(
            px + TICK_LABEL_OFFSET * nx,
            py + TICK_LABEL_OFFSET * ny,
            f"{round(100.0 * tick['value']):d}",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=colour,
        )
    mids = {
        "bottom": ((0.5, 0.0), (0.0, -1.0), order[1]),
        "right": ((0.75, 0.5 * h), (h, 0.5), order[2]),
        "left": ((0.25, 0.5 * h), (-h, 0.5), order[0]),
    }
    for edge, ((mx, my), (nx, ny), element) in mids.items():
        ax.text(
            mx + TICK_TITLE_OFFSET * nx,
            my + TICK_TITLE_OFFSET * ny,
            f"at.% {element}",
            ha="center",
            va="center",
            rotation=_EDGE_TITLE_ROTATION[edge],
            rotation_mode="anchor",
            fontsize=fontsize + 1,
            color=colour,
        )


def frame_ticks(order, window, step: float = TICK_STEP) -> list:
    """Composition ticks on the border of a closeup ``window`` ``(x0, x1, y0, y1)``, one element
    to a side as on the whole triangle: ``order[1]`` along the bottom, ``order[2]`` up the right
    and ``order[0]`` down the left. A tick lies along the line of constant composition through
    its point (so the bottom ones slant); ``direction`` is its unit vector pointing out of the
    window. Returns ``[{"side", "element", "value", "point", "direction"}]``.
    """
    x0, x1, y0, y1 = window
    s3, h = math.sqrt(3.0), SQRT3_OVER_2
    out = []
    for k in range(1, int(round(1.0 / step))):
        v = round(k * step, 6)
        if x0 <= v + y0 / s3 <= x1:
            out.append(
                {
                    "side": "bottom",
                    "element": order[1],
                    "value": v,
                    "point": (v + y0 / s3, y0),
                    "direction": (-0.5, -h),
                }
            )
        if y0 <= h * v <= y1:
            out.append(
                {
                    "side": "right",
                    "element": order[2],
                    "value": v,
                    "point": (x1, h * v),
                    "direction": (1.0, 0.0),
                }
            )
        y = s3 * (1.0 - v - x0)
        if y0 <= y <= y1:
            out.append(
                {
                    "side": "left",
                    "element": order[0],
                    "value": v,
                    "point": (x0, y),
                    "direction": (-0.5, h),
                }
            )
    return out


_SIDE_NUMBER = {
    "bottom": ((0, -2), "center", "top"),
    "right": ((2, 0), "left", "center"),
    "left": ((-2, 0), "right", "center"),
}
_SIDE_TITLE = {
    "bottom": ((0.5, 0.0), 0.0, "center", "top"),
    "right": ((1.0, 0.5), 270.0, "left", "center"),
    "left": ((0.0, 0.5), 90.0, "right", "center"),
}


def draw_window_frame(
    ax,
    order,
    window,
    fontsize: float = 9.0,
    step: float = TICK_STEP,
    number_sides=("bottom", "left", "right"),
    title_sides=("bottom", "left", "right"),
    length_pt: float = 6.0,
    colour: str = TICK_COLOUR,
) -> None:
    """Frame a closeup ``window`` with outer composition ticks (``frame_ticks``).

    Ticks are drawn on every side; the tick number appears only on ``number_sides`` and the
    "at.% X" title, centred on the side, only on ``title_sides``. Ticks are ``length_pt`` long.
    """
    from matplotlib.patches import Rectangle

    x0, x1, y0, y1 = window
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=colour,
            linewidth=0.8,
            zorder=9,
            clip_on=False,
        )
    )
    width_pt = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    length = length_pt * (x1 - x0) / width_pt
    digits = 2 * fontsize * 0.6
    element = {"bottom": order[1], "right": order[2], "left": order[0]}
    for tick in frame_ticks(order, window, step):
        (px, py), (dx, dy) = tick["point"], tick["direction"]
        tip = (px + length * dx, py + length * dy)
        ax.plot(
            [px, tip[0]],
            [py, tip[1]],
            color=colour,
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=9,
            clip_on=False,
        )
        if tick["side"] in number_sides:
            offset, ha, va = _SIDE_NUMBER[tick["side"]]
            ax.annotate(
                f"{round(100.0 * tick['value']):d}",
                xy=tip,
                xytext=offset,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=colour,
                annotation_clip=False,
            )
    for side in title_sides:
        xy, rotation, ha, va = _SIDE_TITLE[side]
        distance = length_pt + 2.0 + (fontsize * 1.2 if side == "bottom" else digits) + 5.0
        offset = {"bottom": (0, -distance), "right": (distance, 0), "left": (-distance, 0)}[side]
        ax.annotate(
            f"at.% {element[side]}",
            xy=xy,
            xycoords="axes fraction",
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            rotation=rotation,
            fontsize=fontsize + 1,
            color=colour,
            annotation_clip=False,
        )
