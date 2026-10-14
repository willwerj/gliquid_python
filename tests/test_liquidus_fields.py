"""gliquid.plotting.liquidus_fields -- splitting a liquidus mesh into phase fields.

Synthetic ternary grids, no data. A partition must tile the triangle, put its boundaries where
the sheets meet, keep each field in one piece, and find a phase that wins only between nodes.
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gliquid.plotting import liquidus_fields as lf

SQRT3_2 = np.sqrt(3) / 2


def _grid(delta):
    n = round(1 / delta)
    comp = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    xy = np.column_stack([comp[:, 0] + 0.5 * comp[:, 1], comp[:, 1] * SQRT3_2])
    return comp, xy, Delaunay(xy).simplices


def _area(piece):
    return float(lf._triangle_areas(piece.xy, piece.faces).sum()) if piece else 0.0


# ---- partition_fields: primary-phase fields of per-phase liquidus sheets

TRIANGLE_AREA = np.sqrt(3) / 4
OFFSETS = np.array([7.0, -3.0, 0.0])


def _corner_sheets(curve=0.0):
    """Three phases, each hottest at its own corner; ``curve`` bends the boundaries."""

    def sheets(comp):
        comp = np.atleast_2d(comp)
        full = np.column_stack([1 - comp.sum(axis=1), comp])
        temps = 1000.0 + 900.0 * (full - 1 / 3) + OFFSETS
        temps[:, 0] += curve * comp[:, 1] ** 2
        return temps

    return sheets


def _partition(delta, curve=0.0, **kwargs):
    comp, xy, triangles = _grid(delta)
    return lf.partition_fields(xy, comp, triangles, _corner_sheets(curve), "ABC", **kwargs)


@pytest.mark.parametrize("delta", [0.1, 0.05])
@pytest.mark.parametrize("curve", [0.0, 400.0])
def test_partition_tiles_the_triangle(delta, curve):
    part = _partition(delta, curve)
    assert set(part.fields) == {"A", "B", "C"}
    assert _area(part.surface) == pytest.approx(TRIANGLE_AREA, rel=1e-12)
    fields = [part.fields[phase] for phase in "ABC"]
    assert sum(_area(field.inside) for field in fields) == pytest.approx(TRIANGLE_AREA, rel=1e-12)
    for field in fields:
        assert _area(field.inside) + _area(field.outside) == pytest.approx(TRIANGLE_AREA, rel=1e-12)


@pytest.mark.parametrize("curve", [0.0, 400.0])
def test_partition_boundaries_lie_where_the_sheets_meet(curve):
    part = _partition(0.1, curve)
    sheets = _corner_sheets(curve)(part.surface.comp)
    shared = [v for v, phases in enumerate(part.vertex_phases) if len(phases) > 1]
    assert shared
    for v in shared:
        temps = sheets[v, ["ABC".index(phase) for phase in part.vertex_phases[v]]]
        assert np.ptp(temps) < (1e-6 if len(temps) == 2 else 1.0)
    assert part.surface.T == pytest.approx(sheets.max(axis=1), abs=1e-12)


def test_partition_puts_the_triple_point_where_three_sheets_meet():
    part = _partition(0.1)
    (triple,) = [v for v, phases in enumerate(part.vertex_phases) if len(phases) == 3]
    # 900 (f_A - f_B) = -10 and 900 (f_A - f_C) = -7 with f_A + f_B + f_C = 1
    f_a = (1 - 10 / 900 - 7 / 900) / 3
    expected = np.array([f_a + 10 / 900, f_a + 7 / 900])
    assert part.surface.comp[triple] == pytest.approx(expected, abs=lf.JUNCTION_EDGE)


@pytest.mark.parametrize("curve", [0.0, 400.0])
def test_each_partition_field_is_one_piece_without_holes(curve):
    part = _partition(0.05, curve)
    for field in part.fields.values():
        (loop,) = field.outline
        assert loop[0] == pytest.approx(loop[-1])


def test_refinement_finds_a_phase_no_node_carries():
    comp, xy, triangles = _grid(0.1)
    centre = xy[triangles[40]].mean(axis=0)
    radius = 0.03
    assert np.linalg.norm(xy - centre, axis=1).min() > radius

    def sheets(points):
        points = np.atleast_2d(points)
        cart = np.column_stack([points[:, 0] + 0.5 * points[:, 1], points[:, 1] * SQRT3_2])
        bump = 1050.0 - 50.0 * (np.linalg.norm(cart - centre, axis=1) / radius) ** 2
        return np.column_stack([np.full(len(points), 1000.0), bump])

    part = lf.partition_fields(xy, comp, triangles, sheets, "AD")
    disc = part.fields["D"]
    (loop,) = disc.outline
    assert 0.6 * np.pi * radius**2 < _area(disc.inside) <= np.pi * radius**2
    assert np.linalg.norm(loop[:, :2] - centre, axis=1) == pytest.approx(radius, rel=1e-6)

    unrefined = lf.partition_fields(xy, comp, triangles, sheets, "AD", min_edge=np.inf)
    assert "D" not in unrefined.fields


def test_a_detached_splinter_joins_its_neighbour():
    comp, xy, triangles = _grid(0.1)
    centroids = comp[triangles].mean(axis=1)
    owner = (centroids[:, 0] > 0.5).astype(int)
    island = int(np.argmin(np.linalg.norm(centroids - [0.2, 0.2], axis=1)))
    owner[island] = 1
    small = lf._triangle_areas(xy, triangles[[island]])[0]
    merged = lf._merge_splinters(triangles, owner, xy, 2 * small)
    assert merged[island] == 0
    assert (np.delete(merged, island) == np.delete(owner, island)).all()


def test_mutually_adjacent_splinters_still_merge():
    xy = np.array(
        [[0, 0], [1, 0], [0.5, 0.01], [0.5, -0.01], [2, 2], [-1, -2]]
        + [[10, 0], [12, 0], [11, 2], [20, 0], [22, 0], [21, 2]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 4, 2], [0, 5, 3], [6, 7, 8], [9, 10, 11]])
    owner = np.array([0, 1, 2, 2, 0, 1])
    merged = lf._merge_splinters(faces, owner, xy, max_area=0.01)
    assert merged[:4].tolist() == [2, 2, 2, 2]


def _thin_band_sheets(width, centre):
    """A and B meet on x0 = ``centre``; C leads in a band of ``width`` mole fraction around it."""

    def sheets(points):
        points = np.atleast_2d(points)
        a = 1000.0 - 500.0 * (points[:, 0] - centre)
        b = 1000.0 + 500.0 * (points[:, 0] - centre)
        c = 1000.0 + 250.0 * width - 500.0 * np.abs(points[:, 0] - centre)
        return np.column_stack([a, b, c])

    return sheets


def _owner_at(part, points):
    """Owner of the surface face holding each of ``points`` (plot coordinates)."""
    tri = part.surface.xy[part.surface.faces]
    v0, v1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    out = np.full(len(points), -1)
    for k, point in enumerate(points):
        d = point - tri[:, 0]
        u = (d[:, 0] * v1[:, 1] - d[:, 1] * v1[:, 0]) / det
        v = (v0[:, 0] * d[:, 1] - v0[:, 1] * d[:, 0]) / det
        inside = np.flatnonzero((u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12))
        out[k] = part.owner[inside[0]]
    return out


def test_a_band_thinner_than_the_mesh_is_one_field():
    """The band misses every node, edge midpoint and centroid of the grid."""
    width, centre = 0.003, 0.41
    comp, xy, triangles = _grid(0.025)
    sheets = _thin_band_sheets(width, centre)
    part = lf.partition_fields(xy, comp, triangles, sheets, "ABC")
    (loop,) = part.fields["C"].outline
    assert loop[0] == pytest.approx(loop[-1])
    probe = np.random.default_rng(0).dirichlet([1, 1, 1], 20000)[:, 1:]
    probe = probe[np.abs(probe[:, 0] - centre) < 2 * width]
    truth = sheets(probe).argmax(axis=1)
    found = _owner_at(
        part, np.column_stack([probe[:, 0] + 0.5 * probe[:, 1], probe[:, 1] * SQRT3_2])
    )
    band = truth == 2
    assert (found[band] == 2).mean() > 0.99
    assert (truth[found == 2] == 2).mean() > 0.99


def test_a_tie_between_two_sheets_is_cut_the_same_in_any_node_order():
    def sheets(points):
        points = np.atleast_2d(points)
        flat = np.full(len(points), 1000.0)
        return np.column_stack([flat, flat + np.maximum(0.0, 2000.0 * (points[:, 0] - 0.63))])

    comp, xy, triangles = _grid(0.1)
    areas = []
    for order in (np.arange(len(comp)), np.arange(len(comp))[::-1]):
        inverse = np.argsort(order)
        part = lf.partition_fields(xy[order], comp[order], inverse[triangles], sheets, "AB")
        areas.append(_area(part.fields["B"].inside))
    exact = 0.37**2 * TRIANGLE_AREA
    assert areas == pytest.approx([exact, exact], rel=1e-9)


def test_a_triple_point_beside_a_steep_edge_sits_where_the_sheets_meet():
    """Sheet C rises from 0 like 1 / (2 - ln x1) off the x1 = 0 edge, as the sheet of a phase
    containing the third element does; A, B and C meet about 1e-8 inside the edge."""

    def sheets(points):
        points = np.atleast_2d(points)
        with np.errstate(divide="ignore"):
            c = 3000.0 / (2.0 - np.log(points[:, 1]))
        a = 200.0 - 100.0 * points[:, 0]
        b = 100.0 + 100.0 * points[:, 0]
        return np.column_stack([a, b, c])

    comp, xy, triangles = _grid(0.025)
    part = lf.partition_fields(xy, comp, triangles, sheets, "ABC")
    triples = [v for v, phases in enumerate(part.vertex_phases) if len(phases) == 3]
    assert triples
    temps = sheets(part.surface.comp)
    for v in triples:
        assert np.ptp(temps[v]) < 1.0
    for k in range(3):
        vertices = np.unique(part.surface.faces[part.owner == k])
        assert (part.surface.T[vertices] - temps[vertices, k]).max() < 1.0


def _wall_sheets(points):
    """Toward the side x1 = 0, B leads down to x1 = 1e-5, C from there to near 1e-10, then A."""
    points = np.atleast_2d(points)
    decades = np.minimum(-np.log10(np.maximum(points[:, 1], 1e-300)), 30.0)
    a = 400.0 + 20.0 * points[:, 0]
    b = 700.0 - 40.0 * decades
    c = 600.0 - 20.0 * decades
    return np.column_stack([a, b, c])


def test_a_stripe_along_a_side_runs_the_length_of_the_side():
    comp, xy, triangles = _grid(0.025)
    part = lf.partition_fields(xy, comp, triangles, _wall_sheets, "ABC")
    assert set(part.fields) == {"A", "B", "C"}
    (loop,) = part.fields["C"].outline
    along = part.fields["C"].inside.comp[:, 0]
    assert along.min() < 0.01 and along.max() > 0.99
    meet = [v for v, phases in enumerate(part.vertex_phases) if phases == ("B", "C")]
    assert np.log10(part.surface.comp[meet, 1]) == pytest.approx(-5.0, abs=1e-6)


def _half_wall_sheets(points):
    """:func:`_wall_sheets` along the side for x0 < 0.4, fading to plain B by x0 = 0.6."""
    points = np.atleast_2d(points)
    reach = np.clip((0.6 - points[:, 0]) / 0.2, 0.0, 1.0)
    decades = reach * np.minimum(-np.log10(np.maximum(points[:, 1], 1e-300)), 30.0)
    a = 400.0 + 20.0 * points[:, 0]
    return np.column_stack([a, 700.0 - 40.0 * decades, 600.0 - 20.0 * decades])


def test_every_edge_leaving_a_side_is_split_at_the_same_depths():
    comp, xy, triangles = _grid(0.025)
    temps = _half_wall_sheets(comp)

    def columns(points, which):
        return np.take_along_axis(_half_wall_sheets(points), which, axis=1)

    graded = lf._grade_sides(
        comp,
        xy,
        triangles,
        temps.argmax(axis=1),
        temps,
        np.zeros(0, dtype=np.int64),
        np.zeros(0),
        _half_wall_sheets,
        columns,
        lf.SURFACE_TOLERANCE,
        float(np.ptp(temps.max(axis=1))),
    )[0][len(comp) :]
    edges = lf._mesh_edges(triangles)
    start, end = comp[edges[:, 0]], comp[edges[:, 1]]
    leaving = (start[:, 1] == 0) & (end[:, 1] > 0) & (start[:, 0] > 0.05) & (start[:, 0] < 0.95)
    start, end = start[leaving], end[leaving]
    step = end - start
    offset = graded[None, :, :] - start[:, None, :]
    cross = step[:, None, 0] * offset[..., 1] - step[:, None, 1] * offset[..., 0]
    t = (offset * step[:, None, :]).sum(axis=2) / (step**2).sum(axis=1)[:, None]
    on_edge = (np.abs(cross) < 1e-15) & (t > 0) & (t < 1)
    counts = on_edge.sum(axis=1)
    assert counts.min() > 5
    assert counts.min() == counts.max()


def _segments(lines):
    return {
        frozenset(map(tuple, pair))
        for line in lines
        for pair in zip(np.round(line[:-1], 9), np.round(line[1:], 9))
    }


@pytest.mark.parametrize("curve", [0.0, 400.0])
def test_field_boundaries_split_the_outline_by_neighbour(curve):
    part = _partition(0.05, curve)
    for phase, field in part.fields.items():
        groups = lf.field_boundaries(part, phase)
        assert set(groups) == (set(part.fields) - {phase}) | {None}
        pieces = [_segments(lines) for lines in groups.values()]
        assert sum(len(piece) for piece in pieces) == len(set().union(*pieces))
        assert set().union(*pieces) == _segments(field.outline)
