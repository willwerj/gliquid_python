"""Edge rows of the ternary composition grid and edge points of the liquidus partition."""

import itertools

import numpy as np

import gliquid.plotting.liquidus_fields as lf
import gliquid.ternary as tmod


def _points(grid):
    return np.column_stack([grid["A"], grid["B"], grid["C"]])


def _key(points):
    return set(map(tuple, np.round(points, 10)))


def test_edge_rows_lie_at_their_distances_from_every_side_and_keep_the_grid():
    plain = _points(tmod.generate_comp_grid(0.025))
    rows = _points(tmod.generate_comp_grid(0.025, edge_rows=3))
    assert _key(plain) <= _key(rows)
    assert np.allclose(rows.sum(axis=1), 1.0) and rows.min() >= 0.0
    for side in range(3):
        for k in (1, 2, 3):
            assert np.isclose(rows[:, side], 0.025 / 3**k).sum() > 40


def test_edge_rows_are_the_same_under_every_order_of_the_components():
    rows = _points(tmod.generate_comp_grid(0.025, edge_rows=3))
    for order in itertools.permutations(range(3)):
        assert _key(rows[:, list(order)]) == _key(rows)


def test_edge_rows_apply_only_on_fine_grids():
    def n_nodes(delta):
        ti = tmod.TernaryLiquidInterpolation(
            ["Al", "Mg", "Si"],
            xs_mix={"Al-Mg": [0] * 4, "Mg-Si": [0] * 4, "Si-Al": [0] * 4},
            delta=delta,
            interp_scheme="linear",
        )
        return len(ti.comp_grid["A"])

    assert n_nodes(0.1) == len(tmod.generate_comp_grid(0.1)["A"])
    assert n_nodes(0.025) == len(tmod.generate_comp_grid(0.025, edge_rows=tmod.EDGE_ROW.rows)["A"])


def test_edge_point_candidates_reach_to_two_steps_from_the_corners():
    cand = lf._edge_candidates()
    full = np.sort(np.column_stack([1 - cand.sum(axis=1), cand]), axis=1)
    gap = lf.EDGE_POINT.spacing / lf.EDGE_POINT.ratio
    nearest = lf.EDGE_POINT.spacing * (1 - gap)
    assert full[:, 1].min() >= nearest - 1e-12
    assert (np.isclose(full[:, 0], gap) & np.isclose(full[:, 1], nearest)).any()


def test_edge_points_follow_a_liquidus_that_drops_at_a_side():
    n = 40
    comp = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    xy = np.column_stack([comp[:, 0] + 0.5 * comp[:, 1], comp[:, 1] * np.sqrt(3) / 2])
    from scipy.spatial import Delaunay

    def sheets(points):
        points = np.atleast_2d(points)
        return (1000.0 - 600.0 * np.exp(-points[:, 1] / 0.004))[:, None]

    _, more, _ = lf._add_edge_points(xy, comp, Delaunay(xy).simplices, sheets)
    added = more[len(comp) :]
    assert len(added) and (added[:, 1] < 0.025).all()
    assert np.isclose(added[:, 1], 0.025 / 3).any()


def test_a_diagonal_across_a_steep_side_wall_is_swapped_for_the_one_along_it():
    comp = np.array([[0.002, 0.29], [0.004, 0.30], [0.002, 0.31], [0.0, 0.30]])

    def sheets(points):
        points = np.atleast_2d(points)
        return (1000.0 * np.sqrt(np.clip(points[:, 0], 0.0, None) / 0.004))[:, None]

    node_t = sheets(comp)[:, 0]
    across = np.array([[3, 1, 0], [3, 2, 1]])
    out = lf._fit_diagonals(comp, comp, across, node_t, sheets)
    shared = set(out[0]) & set(out[1])
    assert shared == {0, 2}
    assert lf._fit_diagonals(comp, comp, out, node_t, sheets).tolist() == out.tolist()


def test_edge_row_nodes_stay_a_half_step_apart_away_from_the_corners():
    rows = _points(tmod.generate_comp_grid(0.025, edge_rows=3))
    for gap in 0.025 / 3 ** np.arange(1, 4):
        along = np.sort(rows[np.isclose(rows[:, 0], gap), 1])
        inner = along[(along > 0.025) & (along < 1 - gap - 0.025)]
        assert np.diff(inner).min() > 0.0125
