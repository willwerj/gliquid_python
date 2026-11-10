"""lower_convex_hull drops vertical simplices on every edge of a ternary composition triangle."""

import numpy as np
import pytest

from gliquid.hsx import _base_hull, lower_convex_hull


def _grid(n):
    x = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    return np.column_stack([x, (x**2).sum(axis=1) - x[:, 0] * x[:, 1]])


def _areas(points, simplices):
    c = points[np.asarray(simplices)][:, :, :2]
    return 0.5 * np.abs(np.linalg.det(c[:, 1:] - c[:, :1]))


@pytest.mark.parametrize("n", [5, 10])
def test_no_vertical_simplex_on_any_edge(n):
    points = _grid(n)
    with_vertical = np.asarray(lower_convex_hull(points.copy(), vertical_simplices=True))
    flat = with_vertical[_areas(points, with_vertical) < 1e-12]
    on_hypotenuse = np.isclose(points[flat][:, :, :2].sum(axis=2), 1).all(axis=1)
    assert on_hypotenuse.any()

    hull = lower_convex_hull(points.copy(), vertical_simplices=False)
    areas = _areas(points, hull)
    assert (areas > 1e-12).all()
    assert areas.sum() == pytest.approx(0.5)


def test_hulls_over_the_same_compositions_share_one_composition_hull():
    """Two energy surfaces over one grid: the second call builds no composition hull, and both
    return the simplices a call that builds its own returns."""
    points = _grid(8)
    tilted = np.column_stack([points[:, :2], points[:, 2] + points[:, 0]])
    _base_hull.cache_clear()
    shared = [np.asarray(lower_convex_hull(surface)) for surface in (points, tilted)]
    info = _base_hull.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    for surface, simplices in zip((points, tilted), shared):
        _base_hull.cache_clear()
        assert np.array_equal(simplices, lower_convex_hull(surface))
