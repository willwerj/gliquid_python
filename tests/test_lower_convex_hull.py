"""lower_convex_hull drops vertical simplices on every edge of a ternary composition triangle."""

import numpy as np
import pytest

from gliquid.hsx import lower_convex_hull


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
