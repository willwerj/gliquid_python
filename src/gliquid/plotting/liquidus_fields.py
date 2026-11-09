"""Primary-crystallisation fields on a triangulated ternary liquidus.

:func:`partition_fields` splits the mesh into one field per phase from per-phase liquidus
sheets ``T_P(x)``: a point belongs to the phase whose sheet is hottest, boundaries run where two
sheets are equal, and the fields tile the surface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

from gliquid.liquidus_sheets import TWO_LIQUIDS, _on_sides

#: Longest mesh edge, in mole fraction, below which :func:`partition_fields` stops refining at
#: field junctions.
JUNCTION_EDGE = 1.5e-3
#: Largest miss, K, of the surface from the liquidus along an edge leaving a side of the
#: composition triangle that :func:`partition_fields` leaves unsplit.
SURFACE_TOLERANCE = 5.0
#: Closest a crossing or triple point comes to the end of its edge or the side of its triangle,
#: as a share of the edge or barycentric weight.
PARTITION_CLAMP = 1e-9


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


@dataclass(frozen=True)
class _Grading:
    """Splitting of edges toward a side (:func:`_grade_sides`) or a wall (:func:`_grade_walls`)."""

    #: Share of an edge, from its end on the side or wall, at which it is split.
    share: float = 1 / 4
    #: Shortest edge, in mole fraction, that is split.
    edge: float = 1e-11


GRADING = _Grading()


def _grade_sides(comp, xy, triangles, labels, temps, sheets, columns, scale):
    """Split the mesh edges that leave a side of the composition triangle, toward that side,
    until the surface along them follows the liquidus, and re-triangulate the triangles they
    bound.

    An edge leaves a side where one end lies on it and the other does not; an edge leaving a
    side at each end is first split at its midpoint. A piece is split at ``GRADING.share``
    of its length from its end on the side while it is longer than ``GRADING.edge`` and
    either a phase neither end carries is hottest at the split point or where the sheets of the
    two ends cross, or the surface there misses the liquidus by more than :data:`SURFACE_TOLERANCE`,
    measured across the surface with ``scale`` K counting as one unit of plot length, and is
    longer than that the same way. The surface along a piece between two phases runs
    through the crossing of their sheets. When one piece leaving a side is split, every piece
    leaving that side is split with it.

    A triangle with new nodes on one edge is fanned from the opposite corner, one with new nodes
    on two edges is laddered between them from their shared corner, and one with new nodes on all
    three is fanned from a new node at its centroid.

    Returns:
        tuple: the refined ``comp``, ``xy``, ``triangles``, ``labels`` and ``temps``, and
        the ids (:data:`_EDGE_ID`) and shares from the lower node of the crossings found on the
        new edges.
    """
    edges = _mesh_edges(triangles)
    on = _on_sides(comp)
    leaves_a = on[edges[:, 0]] & ~on[edges[:, 1]]
    leaves_b = on[edges[:, 1]] & ~on[edges[:, 0]]
    origin = np.flatnonzero(leaves_a.any(axis=1) | leaves_b.any(axis=1))
    count = len(origin)
    if not count:
        return comp, xy, triangles, labels, temps, np.zeros(0, dtype=np.int64), np.zeros(0)
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
        live = _edge_span(comp, piece["a"], piece["b"]) > GRADING.edge
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
        share = np.where(side_a & side_b, 0.5, np.where(side_a, GRADING.share, 1 - GRADING.share))
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
        length = np.maximum(np.linalg.norm(xy[b] - xy[a], axis=1), GRADING.edge)
        steep = np.abs(rise) / scale / length
        span = np.ptp(np.column_stack([top_a, hot, top_b]), axis=1)
        visible = np.hypot(length * scale, span) > SURFACE_TOLERANCE
        split = visible & (
            foreign | (np.abs(hot - flat) / np.sqrt(1 + steep**2) > SURFACE_TOLERANCE)
        )
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
    known, first_seen = np.unique(ids, return_index=True)
    shares = from_lower[first_seen]
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
            faces += _ladder(p_nodes, p_t, q_nodes, q_t)
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


def _ladder(p_nodes, p_t, q_nodes, q_t) -> list[tuple[int, int, int]]:
    """Faces between two chains of vertices, ``p_nodes`` and ``q_nodes`` at increasing shares
    ``p_t`` and ``q_t`` along them, from the rung joining their first vertices to the rung
    joining their last."""
    faces = []
    i = j = 0
    while i < len(p_nodes) - 1 or j < len(q_nodes) - 1:
        if j == len(q_nodes) - 1 or (i < len(p_nodes) - 1 and p_t[i + 1] <= q_t[j + 1]):
            faces.append((p_nodes[i], p_nodes[i + 1], q_nodes[j]))
            i += 1
        else:
            faces.append((p_nodes[i], q_nodes[j + 1], q_nodes[j]))
            j += 1
    return faces


@dataclass(frozen=True)
class _Wall:
    """Two-liquid walls of :func:`partition_fields`."""

    #: Share of an edge on either side of its crossing at which each end's sheet is read.
    share: float = 1e-6
    #: Least jump, K, between the two sheets of a crossing next to the
    #: :data:`~gliquid.liquidus_sheets.TWO_LIQUIDS` field that is drawn as a wall.
    jump: float = 25.0
    #: Most points added to the stretch of each wall edge on its two-liquid side.
    grading_rounds: int = 12


WALL = _Wall()


def _grade_walls(starts, ends, comp, xy, T, sheets):
    """Split each segment from vertex ``starts[i]`` to vertex ``ends[i]`` toward its start: while
    the hottest sheet at ``GRADING.share`` of the piece that has the start as an end misses
    the straight line along that piece by more than :data:`SURFACE_TOLERANCE`, the piece is
    longer than ``GRADING.edge``, and fewer than ``WALL.grading_rounds`` such points have
    been added, the piece is split there, and the rest of it is split at its middle where the
    hottest sheet misses the line along the rest by as much.

    Returns:
        tuple: ``comp``, ``xy`` and ``T`` with the split points appended, and per segment its
        vertices from start to end and their shares along it.
    """
    count = len(starts)
    chains = [[(0.0, int(starts[i])), (1.0, int(ends[i]))] for i in range(count)]
    seg, hi, t_hi = np.arange(count), np.ones(count), T[ends]
    span = _edge_span(comp, starts, ends)

    def add(seg, at, point, hot):
        nonlocal comp, xy, T
        for k, i in enumerate(seg):
            chains[i].append((float(at[k]), len(comp) + k))
        lead = xy[starts[seg]]
        xy = np.vstack([xy, lead + at[:, None] * (xy[ends[seg]] - lead)])
        comp, T = np.vstack([comp, point]), np.concatenate([T, hot])

    def miss(seg, lo, t_lo, hi, t_hi, share):
        at = lo + share * (hi - lo)
        a = comp[starts[seg]]
        point = a + at[:, None] * (comp[ends[seg]] - a)
        hot = sheets(point).max(axis=1)
        return at, point, hot, np.abs(hot - t_lo - share * (t_hi - t_lo)) > SURFACE_TOLERANCE

    for _ in range(WALL.grading_rounds):
        if not len(seg):
            break
        at, point, hot, split = miss(seg, 0.0, T[starts[seg]], hi, t_hi, GRADING.share)
        split &= hi * span[seg] > GRADING.edge
        seg, at, point, hot, hi, t_hi = (v[split] for v in (seg, at, point, hot, hi, t_hi))
        add(seg, at, point, hot)
        mid, mid_point, mid_hot, rest = miss(seg, at, hot, hi, t_hi, 0.5)
        add(seg[rest], mid[rest], mid_point[rest], mid_hot[rest])
        hi, t_hi = at, hot
    chains = [sorted(chain) for chain in chains]
    return comp, xy, T, [([v for _, v in c], [t for t, _ in c]) for c in chains]


@dataclass(frozen=True)
class _Crossing:
    """Placement of a crossing on its edge (:func:`_sheet_crossings`)."""

    #: Most steps; the last ones bisect wherever only bisection still closes the bracket to
    #: ``width`` in time.
    steps: int = 40
    #: The crossing is placed once its two sheets differ by at most this, K, ...
    tolerance: float = 1e-9
    #: ... or once it is bracketed to this share of its edge.
    width: float = 1e-9


CROSSING = _Crossing()


def _sheet_crossings(columns, a, b, p, q, at_a, at_b) -> np.ndarray:
    """Where ``T_p = T_q`` on each segment from ``a`` (where ``p`` leads) to ``b``, as a share of
    the segment, by the Illinois method, with bisection steps where a secant step is not
    available, where the sheets differ by more than ``WALL.jump`` at both ends of the
    bracket, or where only bisection still closes the bracket in the steps left. A bracket
    with such a jump at both ends is closed to twice ``WALL.share``. ``columns``
    evaluates chosen sheets (see :func:`partition_fields`); ``at_a`` and ``at_b``, all the
    sheets at the two ends, seed the secant steps. Where the two sheets are equal,
    the lower column leads, as in a node's label."""
    if not len(a):
        return np.zeros(0)
    rows = np.arange(len(a))
    low, high = np.zeros(len(a)), np.ones(len(a))
    start = at_a[rows, p] - at_a[rows, q]
    gap_low = np.where(start > 0, start, np.nan)
    end = at_b[rows, p] - at_b[rows, q]
    gap_high = np.where(end < 0, end, np.nan)
    raw_low, raw_high = gap_low.copy(), gap_high.copy()
    side = np.zeros(len(a), dtype=int)
    found = np.full(len(a), np.nan)
    live = np.ones(len(a), dtype=bool)
    for step in range(CROSSING.steps):
        if not live.any():
            break
        secant = high - gap_high * (high - low) / (gap_high - gap_low)
        usable = (gap_low > 0) & (gap_high < 0) & (secant > low) & (secant < high)
        usable &= np.log2((high - low) / CROSSING.width) < CROSSING.steps - step - 1
        usable &= np.minimum(raw_low, -raw_high) <= WALL.jump
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
        raw_low[moved_low], raw_high[moved_high] = gap[ahead], gap[~ahead]
        # Illinois: halve the stale end's gap when the same end moves twice running.
        gap_high[moved_low[side[moved_low] == 1]] /= 2
        gap_low[moved_high[side[moved_high] == -1]] /= 2
        side[moved_low], side[moved_high] = 1, -1
        exact = (np.abs(gap) <= CROSSING.tolerance) & (gap != 0)
        found[live_rows[exact]] = at[exact]
        width = np.where(np.minimum(raw_low, -raw_high) > WALL.jump, 2 * WALL.share, CROSSING.width)
        live &= np.isnan(found) & (high - low > width)
    share = np.where(np.isnan(found), (low + high) / 2, found)
    return np.clip(share, PARTITION_CLAMP, 1 - PARTITION_CLAMP)


def _project(weights) -> np.ndarray:
    """``weights`` pulled inside their triangle: each at least :data:`PARTITION_CLAMP`, summing
    to 1."""
    weights = np.clip(np.nan_to_num(weights, nan=1 / 3), PARTITION_CLAMP, None)
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class _TriplePoint:
    """Newton placement of a triple point (:func:`_triple_points`)."""

    #: Newton steps.
    steps: int = 20
    #: Step halvings each Newton step may try before it gives up on improving the point.
    halvings: int = 12


TRIPLE_POINT = _TriplePoint()


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

    def gaps(weights, rows=None):
        rows = np.arange(len(corners)) if rows is None else rows
        temps = columns(np.einsum("ki,kij->kj", weights, corners[rows]), labels[rows])
        return temps[:, :1] - temps[:, 1:]

    def spread(here):
        return np.where(np.isfinite(here).all(axis=1), np.abs(here).max(axis=1), np.inf)

    weights = _project(start)
    here = gaps(weights)
    shifts = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    step = 1e-6
    active = np.flatnonzero(spread(here) > 0.0)
    for _ in range(TRIPLE_POINT.steps):
        if not len(active):
            break
        w, h = weights[active], here[active]
        jacobian = np.stack([(gaps(w + step * s, active) - h) / step for s in shifts], axis=2)
        det = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        safe = np.isfinite(det) & (np.abs(det) > 1e-12) & np.isfinite(h).all(axis=1)
        det = np.where(safe, det, 1.0)
        du = -(jacobian[:, 1, 1] * h[:, 0] - jacobian[:, 0, 1] * h[:, 1]) / det
        dv = -(-jacobian[:, 1, 0] * h[:, 0] + jacobian[:, 0, 0] * h[:, 1]) / det
        newton = (
            np.where(safe, du, 0.0)[:, None] * shifts[0]
            + np.where(safe, dv, 0.0)[:, None] * shifts[1]
        )
        scale = np.ones(len(active))
        pending = safe.copy()
        before = spread(h)
        for _ in range(TRIPLE_POINT.halvings):
            if not pending.any():
                break
            idx = np.flatnonzero(pending)
            trial = _project(w[idx] + scale[idx, None] * newton[idx])
            trial_gaps = gaps(trial, active[idx])
            better = spread(trial_gaps) < spread(h[idx])
            took = idx[better]
            w[took], h[took] = trial[better], trial_gaps[better]
            pending[took] = False
            scale = np.where(pending, scale / 2, scale)
        weights[active], here[active] = w, h
        improved = spread(h) < before
        if not improved.any():
            break
        active = active[improved & (spread(h) > 0.0)]
    return weights


#: Largest detached piece of a field that :func:`partition_fields` merges into a neighbour, as a
#: share of the squared median mesh-edge length.
SPLINTER_AREA = 1e-2


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


#: Most, K, a vertex the partition adds may lie above the hottest vertex next to it.
ADDED_VERTEX_RISE = 25.0


def _cap_added(T, faces, first) -> np.ndarray:
    """``T`` with every vertex from index ``first`` on lowered to at most
    :data:`ADDED_VERTEX_RISE` above the hottest vertex next to it in ``faces``, repeated until
    none lies higher."""
    T = np.array(T, dtype=float)
    for _ in range(len(T)):
        ring = np.full(len(T), -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
            np.maximum.at(ring, faces[:, a], T[faces[:, b]])
        over = np.isfinite(ring) & (T > ring + ADDED_VERTEX_RISE)
        over[:first] = False
        if not over.any():
            break
        T[over] = ring[over] + ADDED_VERTEX_RISE
    return T


@dataclass(frozen=True)
class _EdgePoint:
    """Points :func:`_add_edge_points` adds near the sides of the composition triangle."""

    #: Rows of candidates along each side.
    rows: int = 6
    #: Row k lies ``spacing / ratio**k`` from its side.
    ratio: float = 3.0**0.5
    #: Mesh spacing, mole fraction, of the candidates (twice their spacing along a row); they
    #: are added only to a mesh whose median edge is at most twice this.
    spacing: float = 0.025
    #: A candidate is added where the hottest sheet misses the mesh's linear liquidus by more
    #: than this, K, ...
    tolerance: float = 15.0
    #: ... testing the candidates left again after each addition, at most this many times.
    rounds: int = 3


EDGE_POINT = _EdgePoint()


def _edge_candidates() -> np.ndarray:
    """(m, 2) mole fractions of the candidate points of :func:`_add_edge_points`: row k
    (1..EDGE_POINT.rows) at ``d = EDGE_POINT.spacing / EDGE_POINT.ratio**k`` from each side, with
    points ``(1 - d) * EDGE_POINT.spacing / 2`` apart, none within two of those steps
    of a corner."""
    rows = []
    for k in range(1, EDGE_POINT.rows + 1):
        gap = EDGE_POINT.spacing / EDGE_POINT.ratio**k
        count = int(round(2 / EDGE_POINT.spacing))
        along = (1 - gap) * np.arange(2, count - 1) / count
        for i in range(3):
            j, m = (i + 1) % 3, (i + 2) % 3
            row = np.zeros((len(along), 3))
            row[:, i] = gap
            row[:, j] = along
            row[:, m] = 1 - gap - along
            rows.append(row)
    points = np.unique(np.round(np.vstack(rows), 13), axis=0)
    return points[:, 1:]


def _add_edge_points(xy, comp, triangles, sheets):
    """``xy``, ``comp`` and ``triangles`` with the :func:`_edge_candidates` added where the hottest
    sheet misses the linear liquidus of the mesh by more than ``EDGE_POINT.tolerance``, and
    the midpoints :func:`_add_midpoints` adds; the mesh is the Delaunay triangulation of ``xy``,
    its diagonals fitted (:func:`_fit_diagonals`), once any point is added."""
    edges = _mesh_edges(triangles)
    if (
        not len(edges)
        or np.median(_edge_span(comp, edges[:, 0], edges[:, 1])) > 2 * EDGE_POINT.spacing
    ):
        return xy, comp, triangles
    candidates = _edge_candidates()
    affine, *_ = np.linalg.lstsq(np.column_stack([comp, np.ones(len(comp))]), xy, rcond=None)
    candidate_xy = np.column_stack([candidates, np.ones(len(candidates))]) @ affine
    candidate_temps = np.asarray(sheets(candidates), float)
    candidate_t = candidate_temps.max(axis=1)
    temps = np.asarray(sheets(comp), float)
    node_t = temps.max(axis=1)
    live = np.isfinite(candidate_t)
    added = 0
    for _ in range(EDGE_POINT.rounds):
        mesh = LinearNDInterpolator(Delaunay(xy), node_t)
        miss = np.abs(mesh(candidate_xy[live]) - candidate_t[live])
        take = np.flatnonzero(live)[np.nan_to_num(miss, nan=0.0) > EDGE_POINT.tolerance]
        if not len(take):
            break
        xy = np.vstack([xy, candidate_xy[take]])
        comp = np.vstack([comp, candidates[take]])
        node_t = np.concatenate([node_t, candidate_t[take]])
        temps = np.vstack([temps, candidate_temps[take]])
        live[take] = False
        added += len(take)
    if added:
        triangles = Delaunay(xy).simplices
    xy, comp, triangles, node_t, more = _add_midpoints(xy, comp, triangles, temps, sheets)
    if added or more:
        triangles = _fit_diagonals(xy, comp, triangles, node_t, sheets)
    return xy, comp, triangles


@dataclass(frozen=True)
class _Midpoint:
    """Midpoints :func:`_add_midpoints` adds to mesh edges that sag off the sheets."""

    #: Most passes.
    rounds: int = 4
    #: Shortest mesh edge, mole fraction, whose midpoint is added.
    edge: float = 2e-4


MIDPOINT = _Midpoint()


def _add_midpoints(xy, comp, triangles, temps, sheets):
    """``xy``, ``comp`` and ``triangles`` with the midpoint of mesh edges added where the
    hottest sheet there misses the straight line along the edge by more than
    ``EDGE_POINT.tolerance``, and by more than :data:`SURFACE_TOLERANCE` measured across the
    surface with the range of the nodes' liquidus counting as one unit of plot length. ``temps``
    holds every sheet at each node.

    Only edges longer than ``MIDPOINT.edge`` whose two ends and midpoint share their hottest
    sheet, with an end lying more than ``EDGE_POINT.tolerance`` from the mean of its
    neighbours, are tested, each once; the mesh is re-triangulated (Delaunay) after each of at
    most ``MIDPOINT.rounds`` passes.

    Returns:
        tuple: ``xy``, ``comp``, ``triangles``, the hottest sheet at each node and the number
        of points added.
    """
    node_t = temps.max(axis=1)
    top = temps.argmax(axis=1)
    finite_t = node_t[np.isfinite(node_t)]
    scale = max(float(np.ptp(finite_t)) if len(finite_t) else 1.0, 1.0)
    affine, *_ = np.linalg.lstsq(np.column_stack([comp, np.ones(len(comp))]), xy, rcond=None)
    tested = set()
    added = 0
    for _ in range(MIDPOINT.rounds):
        edges = _mesh_edges(triangles)
        a, b = edges[:, 0], edges[:, 1]
        finite = np.isfinite(node_t[a]) & np.isfinite(node_t[b])
        total, count = np.zeros(len(comp)), np.zeros(len(comp))
        for u, v in ((a, b), (b, a)):
            np.add.at(total, u[finite], node_t[v[finite]])
            np.add.at(count, u[finite], 1)
        bent = np.abs(node_t - total / np.maximum(count, 1)) > EDGE_POINT.tolerance
        keys = a.astype(np.int64) * _EDGE_ID + b
        fresh = np.fromiter((k not in tested for k in keys.tolist()), bool, len(keys))
        test = np.flatnonzero(
            finite
            & fresh
            & (bent[a] | bent[b])
            & (top[a] == top[b])
            & (_edge_span(comp, a, b) > MIDPOINT.edge)
        )
        tested.update(keys[test].tolist())
        if not len(test):
            break
        a, b = a[test], b[test]
        mid = 0.5 * (comp[a] + comp[b])
        at = np.asarray(sheets(mid), float)
        hot = at.max(axis=1)
        miss = np.abs(0.5 * (node_t[a] + node_t[b]) - hot)
        length = np.maximum(np.linalg.norm(xy[b] - xy[a], axis=1), MIDPOINT.edge)
        steep = np.abs(node_t[b] - node_t[a]) / scale / length
        take = (
            np.isfinite(hot)
            & (at.argmax(axis=1) == top[a])
            & (miss > EDGE_POINT.tolerance)
            & (miss / np.sqrt(1 + steep**2) > SURFACE_TOLERANCE)
        )
        if not take.any():
            break
        comp = np.vstack([comp, mid[take]])
        xy = np.vstack([xy, np.column_stack([mid[take], np.ones(int(take.sum()))]) @ affine])
        node_t = np.concatenate([node_t, hot[take]])
        top = np.concatenate([top, at.argmax(axis=1)[take]])
        added += int(take.sum())
        triangles = Delaunay(xy).simplices
    return xy, comp, triangles, node_t, added


def _fit_diagonals(xy, comp, triangles, node_t, sheets) -> np.ndarray:
    """``triangles`` with the shared side of two faces forming a convex quad swapped for the
    quad's other diagonal where the linear liquidus ``node_t`` misses the hottest sheet at the
    side's midpoint by more than ``EDGE_POINT.tolerance`` and at the other diagonal's
    midpoint by less; quads within ``EDGE_POINT.spacing`` of a side and without a corner,
    at most ``EDGE_POINT.rounds`` passes."""
    triangles = np.array(triangles)
    t = np.nan_to_num(np.asarray(node_t, dtype=float), nan=0.0)
    full = np.column_stack([1 - comp.sum(axis=1), comp])
    side = np.where((full <= 0).sum(axis=1) >= 2, np.inf, full.min(axis=1))

    def turn(p, q, r):
        u, v = xy[q] - xy[p], xy[r] - xy[p]
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    def miss(p, q):
        mid = 0.5 * (comp[p] + comp[q])
        return np.abs(0.5 * (t[p] + t[q]) - np.asarray(sheets(mid), float).max(axis=1))

    for _ in range(EDGE_POINT.rounds):
        n = len(triangles)
        edges = _face_edges(triangles)
        owner = np.tile(np.arange(n), 3)
        opposite = np.concatenate([triangles[:, 2], triangles[:, 0], triangles[:, 1]])
        key = edges[:, 0] * len(comp) + edges[:, 1]
        order = np.argsort(key, kind="stable")
        pair = np.flatnonzero(key[order][1:] == key[order][:-1])
        i, j = order[pair], order[pair + 1]
        a, b, c, d = edges[i, 0], edges[i, 1], opposite[i], opposite[j]
        quad = np.stack([a, b, c, d], axis=1)
        keep = (
            (side[quad].max(axis=1) < np.inf)
            & (side[quad].min(axis=1) <= EDGE_POINT.spacing)
            & (turn(a, b, c) * turn(a, b, d) < 0)
            & (turn(c, d, a) * turn(c, d, b) < 0)
        )
        r = np.flatnonzero(keep)
        if not len(r):
            break
        now = miss(a[r], b[r])
        r, now = r[now > EDGE_POINT.tolerance], now[now > EDGE_POINT.tolerance]
        if not len(r):
            break
        gain = now - np.nan_to_num(miss(c[r], d[r]), nan=np.inf)
        used = np.zeros(n, dtype=bool)
        flipped = 0
        for k in r[np.argsort(-gain)][np.sort(-gain) < 0]:
            f, g = owner[i[k]], owner[j[k]]
            if used[f] or used[g]:
                continue
            used[f] = used[g] = True
            sign = np.sign(turn(*triangles[f, :, None]))
            for face, (p, q, s) in ((f, (a[k], c[k], d[k])), (g, (b[k], c[k], d[k]))):
                same = np.sign(turn(np.array([p]), np.array([q]), np.array([s]))) == sign
                triangles[face] = (p, q, s) if same[0] else (p, s, q)
            flipped += 1
        if not flipped:
            break
    return triangles


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

    Every point belongs to the phase whose sheet is hottest there. Points near the sides of the
    composition triangle, and midpoints of edges that sag off the sheets, are first added to the
    mesh where its linear liquidus misses them (:func:`_add_edge_points`). Triangles meeting three
    phases, or holding a phase none of their nodes carries, are then split at their edge midpoints
    until their edges are no longer than ``min_edge``, and edges leaving a side are split toward it
    until the surface along them is within :data:`SURFACE_TOLERANCE` of the liquidus
    (:func:`_grade_sides`). Every edge between two phases is then cut where their sheets are equal,
    and every triangle meeting three phases at the point inside it where the three come closest to
    equal.

    Where one end of an edge is in the :data:`~gliquid.liquidus_sheets.TWO_LIQUIDS` field and the
    two ends' sheets, read ``WALL.share`` of the edge before and after the crossing, differ by
    more than ``WALL.jump``, the crossing is a wall: it has one vertex on each side, each
    side's faces use their own, and the faces between the two belong to the hotter side. On its
    two-liquid side, each wall edge is split toward the wall where the surface along it misses
    the hottest sheet (:func:`_grade_walls`).

    Each vertex takes the hottest sheet as its temperature, so the surface is the liquidus of the
    sheets, except that a wall vertex takes its own side's sheet where no other solid's sheet is
    hotter, and a vertex added to the mesh lies at most :data:`ADDED_VERTEX_RISE` above
    the hottest vertex next to it (:func:`_cap_added`). A detached piece of a field smaller than
    :data:`SPLINTER_AREA` joins the neighbouring field it shares the longest boundary with.

    Args:
        xy: (n, 2) plot coordinates of the nodes, an affine image of ``comp``; the returned
            surface also holds the points :func:`_add_edge_points` adds.
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
    xy, comp, triangles = _add_edge_points(xy, comp, triangles, sheets)
    n_nodes = len(comp)
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
        sheets,
        sheet_columns,
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
    lo, hi = np.clip(s - WALL.share, 0.0, 1.0), np.clip(s + WALL.share, 0.0, 1.0)
    wall = np.zeros(len(s), dtype=bool)
    wall_lo_T, wall_hi_T = np.full(len(s), np.nan), np.full(len(s), np.nan)
    if TWO_LIQUIDS in phases:
        gap = phases.index(TWO_LIQUIDS)
        two = np.flatnonzero((labels[u] == gap) | (labels[w] == gap))
        if len(two):
            which = np.column_stack([labels[u[two]], labels[w[two]]])
            step = comp[w[two]] - comp[u[two]]
            before = sheet_columns(comp[u[two]] + lo[two, None] * step, which)
            after = sheet_columns(comp[u[two]] + hi[two, None] * step, which)
            wall[two] = np.abs(before[:, 0] - after[:, 1]) > WALL.jump
            wall_lo_T[two], wall_hi_T[two] = before[:, 0], after[:, 1]
    edge_keys = u * n + w
    node_labels = labels[triangles]
    count = _label_count(node_labels)
    t3 = triangles[count == 3]
    wall_vertex = np.full(len(edges), -1)
    wall_vertex[wall] = n + len(edges) + len(t3) + np.arange(int(wall.sum()))

    def crossing(a, b, side):
        """The crossing vertex of edge (a, b) on the side of its end ``side``."""
        e = np.searchsorted(edge_keys, np.minimum(a, b) * n + np.maximum(a, b))
        return np.where((side == np.maximum(a, b)) & wall[e], wall_vertex[e], n + e)

    faces, owner = [triangles[count == 1]], [node_labels[count == 1, 0]]

    t, lab = triangles[count == 2], node_labels[count == 2]
    rows = np.arange(len(t))
    odd = np.where(lab[:, 0] == lab[:, 1], 2, np.where(lab[:, 0] == lab[:, 2], 1, 0))
    o, p, q = t[rows, odd], t[rows, (odd + 1) % 3], t[rows, (odd + 2) % 3]
    op_o, oq_o = crossing(o, p, o), crossing(o, q, o)
    op_p, oq_q = crossing(o, p, p), crossing(o, q, q)
    faces += [
        np.column_stack([o, op_o, oq_o]),
        np.column_stack([op_p, p, q]),
        np.column_stack([op_p, q, oq_q]),
    ]
    owner += [labels[o], labels[p], labels[p]]
    # faces across walls: (face, vertex on one side, vertex on the other, their labels)
    closing = [
        (np.column_stack([op_o, op_p, oq_q]), op_o, op_p, labels[o], labels[p]),
        (np.column_stack([op_o, oq_q, oq_o]), oq_o, oq_q, labels[o], labels[p]),
    ]

    start = np.zeros((len(t3), 3))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        share = s[crossing(t3[:, i], t3[:, j], -1) - n]
        near_i = np.where(t3[:, i] < t3[:, j], 1 - share, share)
        start[:, i] += near_i / 3
        start[:, j] += (1 - near_i) / 3
    weights = _triple_points(sheet_columns, comp[t3], labels[t3], start)
    centre = n + len(edges) + np.arange(len(t3))
    a, b, c = t3[:, 0], t3[:, 1], t3[:, 2]
    faces += [
        np.column_stack([a, crossing(a, b, a), centre]),
        np.column_stack([a, centre, crossing(c, a, a)]),
        np.column_stack([b, crossing(b, c, b), centre]),
        np.column_stack([b, centre, crossing(a, b, b)]),
        np.column_stack([c, crossing(c, a, c), centre]),
        np.column_stack([c, centre, crossing(b, c, c)]),
    ]
    owner += [labels[a], labels[a], labels[b], labels[b], labels[c], labels[c]]
    for x, y in ((a, b), (b, c), (c, a)):
        vx, vy = crossing(x, y, x), crossing(x, y, y)
        closing.append((np.column_stack([vx, vy, centre]), vx, vy, labels[x], labels[y]))

    s_first = np.where(wall, lo, s)
    all_comp = np.vstack(
        [
            comp,
            (1 - s_first)[:, None] * comp[u] + s_first[:, None] * comp[w],
            np.einsum("ki,kij->kj", weights, comp[t3]),
            (1 - hi[wall])[:, None] * comp[u[wall]] + hi[wall][:, None] * comp[w[wall]],
        ]
    )
    all_xy = np.vstack(
        [
            xy,
            (1 - s_first)[:, None] * xy[u] + s_first[:, None] * xy[w],
            np.einsum("ki,kij->kj", weights, xy[t3]),
            (1 - hi[wall])[:, None] * xy[u[wall]] + hi[wall][:, None] * xy[w[wall]],
        ]
    )
    added = sheets(all_comp[n:])
    raw_T = np.concatenate([temps.max(axis=1), added.max(axis=1)])
    if wall.any():
        solid = np.delete(added, gap, axis=1).max(axis=1, initial=-np.inf)
        # each side of a wall takes its own field's sheet where no other solid's is hotter
        for at, own in (
            (n + np.flatnonzero(wall), wall_lo_T[wall]),
            (wall_vertex[wall], wall_hi_T[wall]),
        ):
            raw_T[at] = np.where(own >= solid[at - n], own, raw_T[at])
    for face, vx, vy, lx, ly in closing:
        faces.append(face)
        owner.append(np.where(raw_T[vx] > raw_T[vy], lx, ly))
    faces, owner = np.concatenate(faces), np.concatenate(owner)
    if wall.any():
        e = np.flatnonzero(wall)
        gap_end = np.where(labels[u[e]] == gap, u[e], w[e])
        start = np.where(gap_end == w[e], wall_vertex[e], n + e)
        all_comp, all_xy, raw_T, chains = _grade_walls(
            start, gap_end, all_comp, all_xy, raw_T, sheets
        )
        chain = {(c[0][0], c[0][-1]): c for c in chains if len(c[0]) > 2}
        dropped, added, added_owner = np.zeros(len(faces), dtype=bool), [], []
        first = int((count == 1).sum())
        for r in range(len(t)):
            if labels[o[r]] == gap:
                ends, blocks = ((op_o[r], o[r]), (oq_o[r], o[r])), (0,)
            elif labels[p[r]] == gap:
                ends, blocks = ((op_p[r], p[r]), (oq_q[r], q[r])), (1, 2)
            else:
                continue
            sides = [chain.get((int(x), int(y))) for x, y in ends]
            if sides[0] is None and sides[1] is None:
                continue
            sides = [s or ([int(x), int(y)], [0.0, 1.0]) for s, (x, y) in zip(sides, ends)]
            added += _ladder(sides[0][0], sides[0][1], sides[1][0], sides[1][1])
            added_owner += [labels[ends[0][1]]] * (len(sides[0][0]) + len(sides[1][0]) - 2)
            dropped[[first + k * len(t) + r for k in blocks]] = True
        size = np.int64(len(all_comp))
        keys = [min(x, y) * size + max(x, y) for x, y in chain]
        ring = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).astype(np.int64)
        on_chain = np.isin(ring.min(axis=2) * size + ring.max(axis=2), keys).any(axis=1)
        for f in np.flatnonzero(on_chain & ~dropped):
            for k in range(3):
                x, y, z = (int(v) for v in np.roll(faces[f], -k))
                c = chain.get((x, y)) or chain.get((y, x))
                if c is not None:
                    nodes = c[0] if c[0][0] == x else c[0][::-1]
                    added += [(nodes[i], nodes[i + 1], z) for i in range(len(nodes) - 1)]
                    added_owner += [owner[f]] * (len(nodes) - 1)
                    dropped[f] = True
                    break
        if added:
            faces = np.concatenate([faces[~dropped], np.array(added, dtype=faces.dtype)])
            owner = np.concatenate([owner[~dropped], np.array(added_owner, dtype=owner.dtype)])
    kept = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    )
    faces, owner = faces[kept], owner[kept]
    all_T = _cap_added(raw_T, faces, n_nodes)
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
