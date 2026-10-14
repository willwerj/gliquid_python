"""Primary-crystallisation fields on a triangulated ternary liquidus.

:func:`partition_fields` splits the mesh into one field per phase from per-phase liquidus
sheets ``T_P(x)``: a point belongs to the phase whose sheet is hottest, boundaries run where two
sheets are equal, and the fields tile the surface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Longest mesh edge, in mole fraction, below which :func:`partition_fields` stops refining at
#: field junctions.
JUNCTION_EDGE = 1.5e-3
#: Largest miss, K, of the surface from the liquidus along an edge leaving a side of the
#: composition triangle that :func:`partition_fields` leaves unsplit.
SURFACE_TOLERANCE = 5.0
#: Share of such an edge, from its end on the side, at which it is split.
GRADING_SHARE = 1 / 4
#: Shortest such edge, in mole fraction, that is split.
GRADING_EDGE = 1e-11
#: Largest mole fraction counted as zero when placing a node on a side.
SIDE_FRACTION = 1e-14
#: Largest detached piece of a field that :func:`partition_fields` merges into a neighbour, as a
#: share of the squared median mesh-edge length.
SPLINTER_AREA = 1e-2
#: Closest a crossing or triple point comes to the end of its edge or the side of its triangle,
#: as a share of the edge or barycentric weight.
PARTITION_CLAMP = 1e-9
#: Most steps that place a :func:`partition_fields` crossing on its edge.
CROSSING_STEPS = 40
#: A crossing is placed once its two sheets differ by at most this, K, ...
CROSSING_TOLERANCE = 1e-9
#: ... or once it is bracketed to this share of its edge.
CROSSING_WIDTH = 1e-12
#: Newton steps that place a :func:`partition_fields` triple point.
TRIPLE_POINT_STEPS = 20
#: Step halvings each Newton step may try before it gives up on improving a triple point.
TRIPLE_POINT_HALVINGS = 12


@dataclass(frozen=True)
class SurfacePiece:
    """A triangulated piece of the liquidus surface.

    Attributes:
        xy: (m, 2) plot coordinates of the vertices.
        T: (m,) temperatures of the vertices.
        comp: (m, 2) mole fractions of the vertices.
        parent: (m,) index of the input node nearest each vertex.
        faces: (f, 3) vertex indices of the triangles.
    """

    xy: np.ndarray
    T: np.ndarray
    comp: np.ndarray
    parent: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class PhaseField:
    """A phase's field and the rest of the liquidus, split along the field boundary.

    Attributes:
        inside: the field.
        outside: the rest of the surface; ``None`` when the field covers all of it.
        outline: the boundary of the field, including its stretches along the edges of the
            composition triangle, as polylines of (k, 3) arrays of x, y, T.
    """

    inside: SurfacePiece | None
    outside: SurfacePiece | None
    outline: tuple[np.ndarray, ...]


# Edge ids: lower node * _EDGE_ID + upper node.
_EDGE_ID = np.int64(1) << 32


def _triangle_areas(xy, triangles) -> np.ndarray:
    a, b, c = xy[triangles[:, 0]], xy[triangles[:, 1]], xy[triangles[:, 2]]
    u, w = b - a, c - a
    return 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0])


def _face_edges(faces) -> np.ndarray:
    """(3f, 2) node pairs, lower node first, of the sides of every face: every face's first side,
    then every second side, then every third side."""
    faces = np.asarray(faces)
    return np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)


def _mesh_edges(triangles) -> np.ndarray:
    return np.unique(_face_edges(triangles), axis=0)


def chain_segments(edges) -> list[list[int]]:
    """Join segments, given as pairs of node ids, into polylines of node ids.

    Lines start at nodes of odd degree; a closed loop ends on the node it starts from.
    """
    adjacency = defaultdict(set)
    for u, w in edges:
        adjacency[u].add(w)
        adjacency[w].add(u)
    starts = sorted(adjacency, key=lambda v: (len(adjacency[v]) % 2 == 0, v))
    lines = []
    for start in starts:
        while adjacency[start]:
            line, v = [start], start
            while adjacency[v]:
                w = min(adjacency[v])
                adjacency[v].discard(w)
                adjacency[w].discard(v)
                line.append(w)
                v = w
            lines.append(line)
    return lines


def _piece(faces, xy, T, comp, parent) -> SurfacePiece | None:
    if not len(faces):
        return None
    used, local = np.unique(faces, return_inverse=True)
    return SurfacePiece(
        xy=xy[used], T=T[used], comp=comp[used], parent=parent[used], faces=local.reshape(-1, 3)
    )


def _outline(faces, xy, T) -> tuple[np.ndarray, ...]:
    """The boundary of a set of faces (the edges only one of them uses), as x, y, T polylines."""
    if not len(faces):
        return ()
    unique_edges, counts = np.unique(_face_edges(faces), axis=0, return_counts=True)
    return tuple(
        np.column_stack([xy[line], T[line]])
        for line in chain_segments(map(tuple, unique_edges[counts == 1]))
    )


@dataclass(frozen=True)
class Partition:
    """A liquidus surface split into one field per primary phase.

    Attributes:
        surface: the whole surface; its ``parent`` is ``arange`` over its vertices.
        owner: (f,) column of the phase owning each face of ``surface``.
        phases: the phase names, in sheet-column order.
        fields: phase -> :class:`PhaseField`, for every phase owning a face; the ``parent`` of
            each piece indexes ``surface`` vertices.
        vertex_phases: per ``surface`` vertex, the sorted names of the phases owning a face there.
    """

    surface: SurfacePiece
    owner: np.ndarray
    phases: tuple[str, ...]
    fields: dict[str, PhaseField]
    vertex_phases: tuple[tuple[str, ...], ...]


def field_boundaries(partition, phase) -> dict[str | None, tuple[np.ndarray, ...]]:
    """The outline of ``phase``'s field in ``partition``, split by the field across it.

    Returns:
        dict: neighbouring phase -> the x, y, T polylines of the boundary the two fields share,
        and ``None`` -> those on the sides of the composition triangle.
    """
    k = partition.phases.index(phase)
    edges = _face_edges(partition.surface.faces)
    owner = np.tile(partition.owner, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    _, start, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    first = owner[start]
    second = np.where(counts == 2, owner[np.minimum(start + 1, len(owner) - 1)], -1)
    on_boundary = ((first == k) | (second == k)) & (first != second)
    across = np.where(first == k, second, first)
    surface = partition.surface
    out = {}
    for other in np.unique(across[on_boundary]):
        shared = edges[start[on_boundary & (across == other)]]
        out[partition.phases[other] if other >= 0 else None] = tuple(
            np.column_stack([surface.xy[line], surface.T[line]])
            for line in chain_segments(map(tuple, shared))
        )
    return out


def _label_count(labels) -> np.ndarray:
    """(t,) number of distinct labels on each row of a (t, 3) label array."""
    return (
        1
        + (labels[:, 0] != labels[:, 1])
        + ((labels[:, 2] != labels[:, 0]) & (labels[:, 2] != labels[:, 1]))
    )


def _edge_span(comp, a, b) -> np.ndarray:
    """Largest change of any mole fraction along each segment from ``comp[a]`` to ``comp[b]``."""
    step = comp[b] - comp[a]
    return np.maximum(np.abs(step).max(axis=1), np.abs(step.sum(axis=1)))


def _refine_junctions(comp, xy, triangles, labels, temps, sheets, columns, min_edge):
    """Split the triangles meeting three phases or holding a phase none of their nodes carries
    until their edges are no longer than ``min_edge``, leaving those already narrower than an
    eighth of ``min_edge`` across.

    A phase none of the nodes carries is looked for at the edge midpoints, at the centroid, and
    where the sheets of an edge's two phases cross. A marked triangle is split into four at its
    edge midpoints, as is any triangle with two or more split edges; a triangle with one split
    edge is halved. New nodes are labelled from ``sheets``, and ``temps``, the sheets at the
    nodes, is extended to them.

    Returns:
        tuple: the refined ``comp``, ``xy``, ``triangles``, ``labels`` and ``temps``, and the
        crossings found: sorted edge ids (lower node * ``_EDGE_ID`` + upper node) and their
        shares.
    """
    foreign = np.zeros(len(triangles), dtype=bool)
    fresh = np.ones(len(triangles), dtype=bool)
    tested = np.zeros(0, dtype=np.int64)
    shares = np.zeros(0)
    intruded = np.zeros(0, dtype=np.int64)
    while True:
        n = len(comp)
        node_labels = labels[triangles]
        if fresh.any():
            corners = comp[triangles[fresh]]
            samples = np.concatenate(
                [(corners + np.roll(corners, -1, axis=1)) / 2, corners.mean(axis=1, keepdims=True)],
                axis=1,
            )
            winners = sheets(samples.reshape(-1, 2)).argmax(axis=1).reshape(-1, 4)
            foreign[fresh] = (
                (winners[:, :, None] != node_labels[fresh][:, None, :]).all(axis=2).any(axis=1)
            )
        # Edge j of a triangle joins its vertices j and j + 1.
        ends = np.sort(np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2), axis=2)
        keys = ends[..., 0] * n + ends[..., 1]
        ids = ends[..., 0].astype(np.int64) * _EDGE_ID + ends[..., 1]
        mixed = labels[ends[..., 0]] != labels[ends[..., 1]]
        untested = np.setdiff1d(ids[mixed & fresh[:, None]], tested)
        if len(untested):
            u, w = untested // _EDGE_ID, untested % _EDGE_ID
            s = _sheet_crossings(
                columns, comp[u], comp[w], labels[u], labels[w], temps[u], temps[w]
            )
            at = sheets(comp[u] + s[:, None] * (comp[w] - comp[u])).argmax(axis=1)
            order = np.argsort(np.concatenate([tested, untested]), kind="stable")
            tested = np.concatenate([tested, untested])[order]
            shares = np.concatenate([shares, s])[order]
            intruded = np.union1d(intruded, untested[(at != labels[u]) & (at != labels[w])])
        foreign |= np.isin(ids, intruded).any(axis=1)
        long = _edge_span(comp, ends[..., 0].ravel(), ends[..., 1].ravel()).reshape(-1, 3)
        corners_xy = xy[triangles]
        sides = np.linalg.norm(corners_xy - np.roll(corners_xy, -1, axis=1), axis=2)
        height = 2 * _triangle_areas(xy, triangles) / np.maximum(sides.max(axis=1), 1e-300)
        red = (
            ((_label_count(node_labels) == 3) | foreign)
            & (long.max(axis=1) > min_edge)
            & (height > min_edge / 8)
        )
        if not red.any():
            break
        while True:
            closed = red | (np.isin(keys, keys[red]).sum(axis=1) >= 2)
            if (closed == red).all():
                break
            red = closed
        split = np.unique(keys[red])
        a, b = split // n, split % n
        comp = np.vstack([comp, (comp[a] + comp[b]) / 2])
        xy = np.vstack([xy, (xy[a] + xy[b]) / 2])
        added = sheets(comp[n:])
        labels = np.concatenate([labels, added.argmax(axis=1)])
        temps = np.vstack([temps, added])
        position = np.clip(np.searchsorted(split, keys), 0, len(split) - 1)
        mids = np.where(split[position] == keys, n + position, -1)

        t, m = triangles[red], mids[red]
        red_children = np.concatenate(
            [
                np.column_stack([t[:, 0], m[:, 0], m[:, 2]]),
                np.column_stack([m[:, 0], t[:, 1], m[:, 1]]),
                np.column_stack([m[:, 2], m[:, 1], t[:, 2]]),
                m,
            ]
        )
        green = ~red & (mids >= 0).any(axis=1)
        t, m = triangles[green], mids[green]
        rows = np.arange(len(t))
        j = (m >= 0).argmax(axis=1)
        start, end, apex, mid = t[rows, j], t[rows, (j + 1) % 3], t[rows, (j + 2) % 3], m[rows, j]
        green_children = np.concatenate(
            [np.column_stack([start, mid, apex]), np.column_stack([mid, end, apex])]
        )
        kept = ~red & ~green
        triangles = np.concatenate([triangles[kept], red_children, green_children])
        children = len(red_children) + len(green_children)
        foreign = np.concatenate([foreign[kept], np.zeros(children, dtype=bool)])
        fresh = np.concatenate([np.zeros(kept.sum(), dtype=bool), np.ones(children, dtype=bool)])
    return comp, xy, triangles, labels, temps, tested, shares


def _on_sides(comp) -> np.ndarray:
    """(m, 3) whether each point of ``comp`` lies on the side of the composition triangle where
    each of its three mole fractions is zero."""
    full = np.column_stack([1 - comp.sum(axis=1), comp])
    return full <= SIDE_FRACTION


def _grade_sides(
    comp, xy, triangles, labels, temps, known, shares, sheets, columns, tolerance, scale
):
    """Split the mesh edges that leave a side of the composition triangle, toward that side,
    until the surface along them follows the liquidus, and re-triangulate the triangles they
    bound.

    An edge leaves a side where one end lies on it and the other does not; an edge leaving a
    side at each end is first split at its midpoint. A piece is split at :data:`GRADING_SHARE`
    of its length from its end on the side while it is longer than :data:`GRADING_EDGE` and
    either a phase neither end carries is hottest at the split point or where the sheets of the
    two ends cross, or the surface there misses the liquidus by more than ``tolerance``,
    measured across the surface with ``scale`` K counting as one unit of plot length, and is
    longer than ``tolerance`` the same way. The surface along a piece between two phases runs
    through the crossing of their sheets. When one piece leaving a side is split, every piece
    leaving that side is split with it.

    A triangle with new nodes on one edge is fanned from the opposite corner, one with new nodes
    on two edges is laddered between them from their shared corner, and one with new nodes on all
    three is fanned from a new node at its centroid.

    Returns:
        tuple: the refined ``comp``, ``xy``, ``triangles``, ``labels`` and ``temps``, and
        ``known`` and ``shares`` with the crossings found on the new edges added.
    """
    edges = _mesh_edges(triangles)
    on = _on_sides(comp)
    leaves_a = on[edges[:, 0]] & ~on[edges[:, 1]]
    leaves_b = on[edges[:, 1]] & ~on[edges[:, 0]]
    origin = np.flatnonzero(leaves_a.any(axis=1) | leaves_b.any(axis=1))
    count = len(origin)
    piece = {
        "a": edges[origin, 0],
        "b": edges[origin, 1],
        "t_a": np.zeros(count),
        "t_b": np.ones(count),
        "end_a": np.where(leaves_a[origin].any(axis=1), leaves_a[origin].argmax(axis=1), -1),
        "end_b": np.where(leaves_b[origin].any(axis=1), leaves_b[origin].argmax(axis=1), -1),
        "origin": origin,
        "cross": np.full(count, np.nan),
        "cross_T": np.zeros(count),
        "intruded": np.zeros(count, dtype=bool),
    }
    found = {"origin": [], "t": [], "node": []}
    settled = {"a": [], "b": [], "cross": []}

    def settle(rows):
        for key in settled:
            settled[key].append(piece[key][rows])

    while len(piece["a"]):
        live = _edge_span(comp, piece["a"], piece["b"]) > GRADING_EDGE
        settle(~live)
        piece = {key: value[live] for key, value in piece.items()}
        a, b = piece["a"], piece["b"]
        if not len(a):
            break
        la, lb = labels[a], labels[b]
        mixed = la != lb
        unknown = np.flatnonzero(mixed & np.isnan(piece["cross"]))
        if len(unknown):
            s = _sheet_crossings(
                columns,
                comp[a[unknown]],
                comp[b[unknown]],
                la[unknown],
                lb[unknown],
                temps[a[unknown]],
                temps[b[unknown]],
            )
            at = sheets(comp[a[unknown]] + s[:, None] * (comp[b[unknown]] - comp[a[unknown]]))
            hottest = at.argmax(axis=1)
            piece["cross"][unknown] = s
            piece["cross_T"][unknown] = at.max(axis=1)
            piece["intruded"][unknown] = (hottest != la[unknown]) & (hottest != lb[unknown])
        side_a, side_b = piece["end_a"] >= 0, piece["end_b"] >= 0
        share = np.where(side_a & side_b, 0.5, np.where(side_a, GRADING_SHARE, 1 - GRADING_SHARE))
        point = comp[a] + share[:, None] * (comp[b] - comp[a])
        at_point = sheets(point)
        hot, winner = at_point.max(axis=1), at_point.argmax(axis=1)
        foreign = piece["intruded"] | ((winner != la) & (winner != lb))

        top_a, top_b = temps[a].max(axis=1), temps[b].max(axis=1)
        cross = np.where(mixed, piece["cross"], 1.0)
        cross_T = np.where(mixed, piece["cross_T"], top_b)
        first = share < cross
        lo, hi = np.where(first, 0.0, cross), np.where(first, cross, 1.0)
        rise = (np.where(first, cross_T, top_b) - np.where(first, top_a, cross_T)) / np.maximum(
            hi - lo, PARTITION_CLAMP
        )
        flat = np.where(first, top_a, cross_T) + (share - lo) * rise
        length = np.maximum(np.linalg.norm(xy[b] - xy[a], axis=1), GRADING_EDGE)
        steep = np.abs(rise) / scale / length
        span = np.ptp(np.column_stack([top_a, hot, top_b]), axis=1)
        visible = np.hypot(length * scale, span) > tolerance
        split = visible & (foreign | (np.abs(hot - flat) / np.sqrt(1 + steep**2) > tolerance))
        # a split spreads to every piece leaving the same side
        group = np.where(side_a & side_b, -1, np.where(side_a, piece["end_a"], piece["end_b"]))
        for k in np.unique(group[split & (group >= 0)]):
            split |= group == k
        settle(~split)
        if not split.any():
            break

        new = len(comp) + np.arange(split.sum())
        comp = np.vstack([comp, point[split]])
        xy = np.vstack([xy, xy[a[split]] + share[split, None] * (xy[b[split]] - xy[a[split]])])
        labels = np.concatenate([labels, winner[split]])
        temps = np.vstack([temps, at_point[split]])
        parent = {key: value[split] for key, value in piece.items()}
        cut, here = share[split], winner[split]
        t_new = parent["t_a"] + cut * (parent["t_b"] - parent["t_a"])
        found["origin"].append(parent["origin"])
        found["t"].append(t_new)
        found["node"].append(new)
        pa, pb, s = labels[parent["a"]], labels[parent["b"]], parent["cross"]
        # a crossing carries over to the half holding it when that half keeps the same two phases
        keep_a = (pa != pb) & (s < cut) & (here == pb)
        keep_b = (pa != pb) & (s > cut) & (here == pa)
        off = np.full(len(new), -1)
        halves = {
            "a": np.concatenate([parent["a"], new]),
            "b": np.concatenate([new, parent["b"]]),
            "t_a": np.concatenate([parent["t_a"], t_new]),
            "t_b": np.concatenate([t_new, parent["t_b"]]),
            "end_a": np.concatenate([parent["end_a"], off]),
            "end_b": np.concatenate([off, parent["end_b"]]),
            "origin": np.concatenate([parent["origin"], parent["origin"]]),
            "cross": np.concatenate(
                [np.where(keep_a, s / cut, np.nan), np.where(keep_b, (s - cut) / (1 - cut), np.nan)]
            ),
            "cross_T": np.concatenate([parent["cross_T"], parent["cross_T"]]),
            "intruded": np.concatenate([keep_a & parent["intruded"], keep_b & parent["intruded"]]),
        }
        graded = (halves["end_a"] >= 0) | (halves["end_b"] >= 0)
        piece = halves
        settle(~graded)
        piece = {key: value[graded] for key, value in halves.items()}

    a, b, cross = (np.concatenate(settled[key]) for key in ("a", "b", "cross"))
    placed = ~np.isnan(cross)
    lower = np.minimum(a, b)[placed]
    ids = lower.astype(np.int64) * _EDGE_ID + np.maximum(a, b)[placed]
    from_lower = np.where(a[placed] == lower, cross[placed], 1 - cross[placed])
    ids, first_seen = np.unique(np.concatenate([known, ids]), return_index=True)
    shares = np.concatenate([shares, from_lower])[first_seen]
    known = ids
    if not found["node"]:
        return comp, xy, triangles, labels, temps, known, shares

    found_origin = np.concatenate(found["origin"])
    found_t = np.concatenate(found["t"])
    found_node = np.concatenate(found["node"])
    order = np.lexsort((found_t, found_origin))
    found_origin, found_t, found_node = found_origin[order], found_t[order], found_node[order]
    starts = np.flatnonzero(np.r_[True, found_origin[1:] != found_origin[:-1]])
    stops = np.r_[starts[1:], len(found_origin)]
    on_edge = {int(found_origin[i]): (found_node[i:j], found_t[i:j]) for i, j in zip(starts, stops)}
    n_old = int(edges.max()) + 1
    edge_keys = edges[:, 0].astype(np.int64) * n_old + edges[:, 1]

    def along(u, v):
        position = int(np.searchsorted(edge_keys, min(u, v) * n_old + max(u, v)))
        nodes, t = on_edge.get(position, ((), ()))
        if u < v:
            return list(nodes), list(t)
        return list(nodes[::-1]), [1 - x for x in t[::-1]]

    tri_keys = np.sort(np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2), axis=2)
    positions = np.searchsorted(
        edge_keys, tri_keys[..., 0].astype(np.int64) * n_old + tri_keys[..., 1]
    )
    touched = np.isin(positions, list(on_edge)).any(axis=1)
    faces, centres = [], []
    for corners in triangles[touched]:
        sides = [along(corners[j], corners[(j + 1) % 3]) for j in range(3)]
        filled = [j for j in range(3) if sides[j][0]]
        if len(filled) == 1:
            j = filled[0]
            chain = [corners[j], *sides[j][0], corners[(j + 1) % 3]]
            apex = corners[(j + 2) % 3]
            faces += [(chain[i], chain[i + 1], apex) for i in range(len(chain) - 1)]
        elif len(filled) == 2:
            bare = ({0, 1, 2} - set(filled)).pop()
            apex, x_end, y_end = corners[(bare + 2) % 3], corners[bare], corners[(bare + 1) % 3]
            p_nodes, p_t = along(apex, x_end)
            q_nodes, q_t = along(apex, y_end)
            p_nodes, p_t = [*p_nodes, x_end], [*p_t, 1.0]
            q_nodes, q_t = [*q_nodes, y_end], [*q_t, 1.0]
            faces.append((apex, p_nodes[0], q_nodes[0]))
            i = j = 0
            while i < len(p_nodes) - 1 or j < len(q_nodes) - 1:
                if j == len(q_nodes) - 1 or (i < len(p_nodes) - 1 and p_t[i + 1] <= q_t[j + 1]):
                    faces.append((p_nodes[i], p_nodes[i + 1], q_nodes[j]))
                    i += 1
                else:
                    faces.append((p_nodes[i], q_nodes[j + 1], q_nodes[j]))
                    j += 1
        else:
            ring = [node for j in range(3) for node in (corners[j], *sides[j][0])]
            centre = len(comp) + len(centres)
            centres.append(corners)
            faces += [(centre, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    if centres:
        middle = comp[np.array(centres)].mean(axis=1)
        added = sheets(middle)
        comp = np.vstack([comp, middle])
        xy = np.vstack([xy, xy[np.array(centres)].mean(axis=1)])
        labels = np.concatenate([labels, added.argmax(axis=1)])
        temps = np.vstack([temps, added])
    triangles = np.concatenate([triangles[~touched], np.array(faces, dtype=triangles.dtype)])
    return comp, xy, triangles, labels, temps, known, shares


def _sheet_crossings(columns, a, b, p, q, at_a=None, at_b=None) -> np.ndarray:
    """Where ``T_p = T_q`` on each segment from ``a`` (where ``p`` leads) to ``b``, as a share of
    the segment, by the Illinois method, with bisection steps where a secant step is not
    available. ``columns`` evaluates chosen sheets (see :func:`partition_fields`); ``at_a`` and
    ``at_b``, all the sheets at the two ends where known, seed the secant steps. Where the two
    sheets are equal, the lower column leads, as in a node's label."""
    if not len(a):
        return np.zeros(0)
    rows = np.arange(len(a))
    low, high = np.zeros(len(a)), np.ones(len(a))
    gap_low, gap_high = np.full(len(a), np.nan), np.full(len(a), np.nan)
    if at_a is not None:
        start = at_a[rows, p] - at_a[rows, q]
        gap_low = np.where(start > 0, start, np.nan)
    if at_b is not None:
        end = at_b[rows, p] - at_b[rows, q]
        gap_high = np.where(end < 0, end, np.nan)
    side = np.zeros(len(a), dtype=int)
    found = np.full(len(a), np.nan)
    live = np.ones(len(a), dtype=bool)
    for _ in range(CROSSING_STEPS):
        if not live.any():
            break
        secant = high - gap_high * (high - low) / (gap_high - gap_low)
        usable = (gap_low > 0) & (gap_high < 0) & (secant > low) & (secant < high)
        at = np.where(usable, secant, (low + high) / 2)[live]
        pair = columns(
            a[live] + at[:, None] * (b[live] - a[live]), np.column_stack([p[live], q[live]])
        )
        live_rows = rows[live]
        gap = pair[:, 0] - pair[:, 1]
        ahead = (gap > 0) | ((gap == 0) & (p[live] < q[live]))
        moved_low = live_rows[ahead]
        moved_high = live_rows[~ahead]
        low[moved_low], gap_low[moved_low] = at[ahead], gap[ahead]
        high[moved_high], gap_high[moved_high] = at[~ahead], gap[~ahead]
        # Illinois: halve the stale end's gap when the same end moves twice running.
        gap_high[moved_low[side[moved_low] == 1]] /= 2
        gap_low[moved_high[side[moved_high] == -1]] /= 2
        side[moved_low], side[moved_high] = 1, -1
        exact = (np.abs(gap) <= CROSSING_TOLERANCE) & (gap != 0)
        found[live_rows[exact]] = at[exact]
        live &= np.isnan(found) & (high - low > CROSSING_WIDTH)
    share = np.where(np.isnan(found), (low + high) / 2, found)
    return np.clip(share, PARTITION_CLAMP, 1 - PARTITION_CLAMP)


def _project(weights) -> np.ndarray:
    """``weights`` pulled inside their triangle: each at least :data:`PARTITION_CLAMP`, summing
    to 1."""
    weights = np.clip(np.nan_to_num(weights, nan=1 / 3), PARTITION_CLAMP, None)
    return weights / weights.sum(axis=1, keepdims=True)


def _triple_points(columns, corners, labels, start) -> np.ndarray:
    """(k, 3) barycentric coordinates of the point where the three labelled sheets meet, in each
    triangle of ``corners`` (k, 3, 2), with ``columns`` evaluating chosen sheets (see
    :func:`partition_fields`).

    Damped Newton's method from ``start`` (k, 3): every iterate stays inside the triangle, and a
    step is taken only where it narrows the largest gap between the three sheets, so the result
    is never worse than ``start``.
    """
    if not len(corners):
        return np.zeros((0, 3))

    def gaps(weights):
        temps = columns(np.einsum("ki,kij->kj", weights, corners), labels)
        return temps[:, :1] - temps[:, 1:]

    def spread(here):
        return np.where(np.isfinite(here).all(axis=1), np.abs(here).max(axis=1), np.inf)

    weights = _project(start)
    here = gaps(weights)
    shifts = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    step = 1e-6
    for _ in range(TRIPLE_POINT_STEPS):
        jacobian = np.stack([(gaps(weights + step * s) - here) / step for s in shifts], axis=2)
        det = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        safe = np.isfinite(det) & (np.abs(det) > 1e-12) & np.isfinite(here).all(axis=1)
        det = np.where(safe, det, 1.0)
        du = -(jacobian[:, 1, 1] * here[:, 0] - jacobian[:, 0, 1] * here[:, 1]) / det
        dv = -(-jacobian[:, 1, 0] * here[:, 0] + jacobian[:, 0, 0] * here[:, 1]) / det
        newton = (
            np.where(safe, du, 0.0)[:, None] * shifts[0]
            + np.where(safe, dv, 0.0)[:, None] * shifts[1]
        )
        scale = np.ones(len(corners))
        pending = safe.copy()
        before = spread(here)
        for _ in range(TRIPLE_POINT_HALVINGS):
            if not pending.any():
                break
            trial = _project(weights + scale[:, None] * newton)
            trial_gaps = gaps(trial)
            better = pending & (spread(trial_gaps) < spread(here))
            weights = np.where(better[:, None], trial, weights)
            here = np.where(better[:, None], trial_gaps, here)
            pending &= ~better
            scale = np.where(pending, scale / 2, scale)
        if not (spread(here) < before).any():
            break
    return weights


def _merge_splinters(faces, owner, xy, max_area) -> np.ndarray:
    """``owner`` with every detached piece of a field smaller than ``max_area`` (other than the
    field's largest piece) handed to the neighbouring field it shares the longest boundary with,
    counting only neighbours that are not such pieces themselves."""
    owner = owner.copy()
    n_faces = len(faces)
    edges = _face_edges(faces)
    face_of = np.tile(np.arange(n_faces), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face_of = edges[order], face_of[order]
    shared = (edges[1:] == edges[:-1]).all(axis=1)
    first, second = face_of[:-1][shared], face_of[1:][shared]
    length = np.linalg.norm(xy[edges[:-1][shared][:, 0]] - xy[edges[:-1][shared][:, 1]], axis=1)
    areas = _triangle_areas(xy, faces)
    while True:
        joined = owner[first] == owner[second]
        graph = coo_matrix(
            (np.ones(joined.sum()), (first[joined], second[joined])), shape=(n_faces, n_faces)
        )
        n_pieces, piece = connected_components(graph, directed=False)
        piece_area = np.bincount(piece, weights=areas, minlength=n_pieces)
        piece_owner = np.zeros(n_pieces, dtype=int)
        piece_owner[piece] = owner
        largest = np.zeros(n_pieces, dtype=bool)
        for phase in np.unique(piece_owner):
            members = np.flatnonzero(piece_owner == phase)
            largest[members[np.argmax(piece_area[members])]] = True
        small = np.flatnonzero(~largest & (piece_area < max_area))
        if not len(small):
            return owner
        border = ~joined
        is_small = np.zeros(n_pieces, dtype=bool)
        is_small[small] = True
        sides = np.concatenate([piece[first[border]], piece[second[border]]])
        others = np.concatenate([piece[second[border]], piece[first[border]]])
        neighbours = np.concatenate([owner[second[border]], owner[first[border]]])
        weights = np.concatenate([length[border], length[border]])
        keep = ~is_small[others]
        sides, neighbours, weights = sides[keep], neighbours[keep], weights[keep]
        merged = False
        for p in small:
            here = sides == p
            if here.any():
                totals = np.bincount(neighbours[here], weights=weights[here])
                owner[piece == p] = int(np.argmax(totals))
                merged = True
        if not merged:
            return owner


def partition_fields(
    xy,
    comp,
    triangles,
    sheets,
    phases,
    *,
    min_edge=JUNCTION_EDGE,
    sheet_columns=None,
) -> Partition:
    """Split a liquidus mesh into the primary-phase fields of per-phase liquidus sheets.

    Every point belongs to the phase whose sheet is hottest there. Triangles meeting three phases,
    or holding a phase none of their nodes carries, are first split at their edge midpoints until
    their edges are no longer than ``min_edge``, and edges leaving a side of the composition
    triangle are then split toward it until the surface along them is within
    :data:`SURFACE_TOLERANCE` of the liquidus (:func:`_grade_sides`). Every edge between two
    phases is then cut where
    their sheets are equal, and every triangle meeting three phases at the point inside it where
    the three come closest to equal. Each vertex takes the hottest sheet as its temperature, so
    the surface is the liquidus of the sheets. A detached piece of a field smaller than
    :data:`SPLINTER_AREA` joins the neighbouring field it shares the longest boundary with.

    Args:
        xy: (n, 2) plot coordinates of the nodes, an affine image of ``comp``.
        comp: (n, 2) mole fractions of the nodes.
        triangles: (t, 3) node indices of the mesh triangles.
        sheets: callable, (m, 2) mole fractions -> (m, len(phases)) temperature of each phase.
        phases: the phase names, the columns of ``sheets``.
        min_edge: longest edge, in mole fraction, below which triangles are not split.
        sheet_columns: callable, (m, 2) mole fractions and (m, k) column indices -> (m, k)
            temperature of those columns; taken from ``sheets`` when ``None``.

    Returns:
        Partition: the surface, the owner of each face and every phase's field.
    """
    xy, comp = np.asarray(xy, float), np.asarray(comp, float)
    triangles = np.asarray(triangles, dtype=int)
    phases = tuple(str(phase) for phase in phases)
    spacing = float(
        np.median(np.linalg.norm(np.diff(xy[_mesh_edges(triangles)], axis=1)[:, 0], axis=1))
    )
    if sheet_columns is None:

        def sheet_columns(points, which):
            return np.take_along_axis(sheets(points), which, axis=1)

    temps = sheets(comp)
    labels = temps.argmax(axis=1)
    scale = max(float(np.ptp(temps.max(axis=1))), 1.0)
    comp, xy, triangles, labels, temps, graded, graded_shares = _grade_sides(
        comp,
        xy,
        triangles,
        labels,
        temps,
        np.zeros(0, dtype=np.int64),
        np.zeros(0),
        sheets,
        sheet_columns,
        SURFACE_TOLERANCE,
        scale,
    )
    comp, xy, triangles, labels, temps, known, known_shares = _refine_junctions(
        comp, xy, triangles, labels, temps, sheets, sheet_columns, min_edge
    )
    known, first_seen = np.unique(np.concatenate([known, graded]), return_index=True)
    known_shares = np.concatenate([known_shares, graded_shares])[first_seen]
    n = len(comp)

    edges = _mesh_edges(triangles)
    edges = edges[labels[edges[:, 0]] != labels[edges[:, 1]]]
    u, w = edges[:, 0], edges[:, 1]
    ids = u.astype(np.int64) * _EDGE_ID + w
    position = np.clip(np.searchsorted(known, ids), 0, max(len(known) - 1, 0))
    reused = (known[position] == ids) if len(known) else np.zeros(len(ids), dtype=bool)
    s = np.empty(len(ids))
    s[reused] = known_shares[position[reused]]
    new = ~reused
    s[new] = _sheet_crossings(
        sheet_columns,
        comp[u[new]],
        comp[w[new]],
        labels[u[new]],
        labels[w[new]],
        temps[u[new]],
        temps[w[new]],
    )
    edge_keys = u * n + w

    def crossing(a, b):
        return n + np.searchsorted(edge_keys, np.minimum(a, b) * n + np.maximum(a, b))

    node_labels = labels[triangles]
    count = _label_count(node_labels)
    faces, owner = [triangles[count == 1]], [node_labels[count == 1, 0]]

    t, lab = triangles[count == 2], node_labels[count == 2]
    rows = np.arange(len(t))
    odd = np.where(lab[:, 0] == lab[:, 1], 2, np.where(lab[:, 0] == lab[:, 2], 1, 0))
    o, p, q = t[rows, odd], t[rows, (odd + 1) % 3], t[rows, (odd + 2) % 3]
    x_op, x_oq = crossing(o, p), crossing(o, q)
    faces += [
        np.column_stack([o, x_op, x_oq]),
        np.column_stack([x_op, p, q]),
        np.column_stack([x_op, q, x_oq]),
    ]
    owner += [labels[o], labels[p], labels[p]]

    t = triangles[count == 3]
    start = np.zeros((len(t), 3))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        share = s[crossing(t[:, i], t[:, j]) - n]
        near_i = np.where(t[:, i] < t[:, j], 1 - share, share)
        start[:, i] += near_i / 3
        start[:, j] += (1 - near_i) / 3
    weights = _triple_points(sheet_columns, comp[t], labels[t], start)
    centre = n + len(edges) + np.arange(len(t))
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    x_ab, x_bc, x_ca = crossing(a, b), crossing(b, c), crossing(c, a)
    faces += [
        np.column_stack([a, x_ab, centre]),
        np.column_stack([a, centre, x_ca]),
        np.column_stack([b, x_bc, centre]),
        np.column_stack([b, centre, x_ab]),
        np.column_stack([c, x_ca, centre]),
        np.column_stack([c, centre, x_bc]),
    ]
    owner += [labels[a], labels[a], labels[b], labels[b], labels[c], labels[c]]

    all_comp = np.vstack(
        [
            comp,
            (1 - s)[:, None] * comp[u] + s[:, None] * comp[w],
            np.einsum("ki,kij->kj", weights, comp[t]),
        ]
    )
    all_xy = np.vstack(
        [
            xy,
            (1 - s)[:, None] * xy[u] + s[:, None] * xy[w],
            np.einsum("ki,kij->kj", weights, xy[t]),
        ]
    )
    all_T = np.concatenate([temps.max(axis=1), sheets(all_comp[n:]).max(axis=1)])
    faces, owner = np.concatenate(faces), np.concatenate(owner)
    owner = _merge_splinters(faces, owner, all_xy, SPLINTER_AREA * spacing**2)

    parent = np.arange(len(all_comp))
    surface = SurfacePiece(xy=all_xy, T=all_T, comp=all_comp, parent=parent, faces=faces)
    fields = {}
    for k, phase in enumerate(phases):
        inside = faces[owner == k]
        if len(inside):
            fields[phase] = PhaseField(
                inside=_piece(inside, all_xy, all_T, all_comp, parent),
                outside=_piece(faces[owner != k], all_xy, all_T, all_comp, parent),
                outline=_outline(inside, all_xy, all_T),
            )
    vertex_phases = [set() for _ in range(len(all_comp))]
    for vertex, k in zip(faces.ravel(), np.repeat(owner, 3)):
        vertex_phases[vertex].add(phases[k])
    return Partition(
        surface=surface,
        owner=owner,
        phases=phases,
        fields=fields,
        vertex_phases=tuple(tuple(sorted(found)) for found in vertex_phases),
    )
