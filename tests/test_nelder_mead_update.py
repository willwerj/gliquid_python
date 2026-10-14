"""The simplex update and the initial simplex it starts from.

``BinaryLiquid.nelder_mead`` is exercised through a stand-in that supplies only what the method
reads -- an objective, a parameter vector and a metrics tuple -- so the update rules can be
checked one step at a time against arithmetic done by hand.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gliquid.binary import (
    DEFAULT_B_SPREAD_MIN,
    BinaryLiquid,
    _offset_vertex_value,
    _restore_infeasible_vertices,
    build_init_triangle,
)
from gliquid.solution import DEFAULT_TAU, PARAM_FORMATS

INF = float("inf")


class _SimplexStub:
    """What ``nelder_mead`` reads off a BinaryLiquid, and a record of every point tried."""

    def __init__(self, objective, n_params=2):
        self.objective = objective
        self.xs_mix = SimpleNamespace(n_params=n_params)
        self.guess_symbols = None
        self.init_triangle = None
        self.nmpath = None
        self.tried = []
        self._x = None

    def f(self, x, **kwargs):
        self._x = np.asarray(x, dtype=float).copy()
        value = float(self.objective(self._x))
        self.tried.append((tuple(self._x), value))
        return value

    def get_params(self):
        return np.asarray(self._x, dtype=float)

    def calculate_deviation_metrics(self, **kwargs):
        return (1.0, 2.0, 3.0, 4.0)


def _bowl(x):
    return float(x[0] ** 2 + x[1] ** 2)


def _tabulated(values, fallback=_bowl):
    """An objective reading ``values`` at the listed points and ``fallback`` everywhere else."""

    def objective(x):
        key = (round(float(x[0]), 6), round(float(x[1]), 6))
        return values[key] if key in values else fallback(x)

    return objective


def _run(stub, simplex, max_iter, tol=1e-9):
    """Drive the update to ``max_iter`` iterations, returning None where it does not converge."""
    try:
        return BinaryLiquid.nelder_mead(stub, max_iter=max_iter, tol=tol, initial_guesses=simplex)
    except RuntimeError as exc:
        if "did not converge" in str(exc):
            return None
        raise


class TestReflection:
    def test_a_reflection_between_best_and_second_worst_is_accepted(self):
        """One iteration costs exactly one trial point when the reflection is taken: an
        accepted reflection neither expands nor contracts."""
        simplex = [[1.0, 0.0], [0.0, 2.0], [2.5, 2.0]]
        stub = _SimplexStub(_bowl)
        _run(stub, simplex, max_iter=1)
        assert len(stub.tried) == 4
        reflected, f_reflected = stub.tried[3]
        assert reflected == pytest.approx((-1.5, 0.0))
        # strictly between the best vertex and the second-worst, which is what admits it
        assert 1.0 <= f_reflected < 4.0

    def test_a_reflection_better_than_the_best_expands(self):
        simplex = [[1.0, 0.0], [0.0, 2.0], [2.5, 2.0]]
        stub = _SimplexStub(_tabulated({(-1.5, 0.0): 0.5}))
        _run(stub, simplex, max_iter=1)
        assert len(stub.tried) == 5
        expanded, _ = stub.tried[4]
        assert expanded == pytest.approx((-3.5, -1.0))


class TestShrink:
    def test_shrink_moves_every_vertex_but_the_best_halfway_to_it(self):
        """A failed contraction shrinks the simplex onto its best vertex: the two other
        vertices, not one of them twice."""
        simplex = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        stub = _SimplexStub(
            _tabulated(
                {
                    (0.0, 0.0): 0.0,
                    (2.0, 0.0): 1.0,
                    (0.0, 2.0): 2.0,
                    (2.0, -2.0): 100.0,
                    (0.5, 1.0): 50.0,
                }
            )
        )
        _run(stub, simplex, max_iter=1)
        assert len(stub.tried) == 7
        moved = [point for point, _ in stub.tried[5:]]
        assert moved[0] == pytest.approx((1.0, 0.0))
        assert moved[1] == pytest.approx((0.0, 1.0))


class TestConstraintWall:
    """A bowl whose optimum sits against an infinite wall, the geometry of a fit whose
    reference lies on the T=0 K hull boundary."""

    @staticmethod
    def objective(x):
        return INF if x[0] < 1.0 else float((x[0] - 1.0) ** 2 + x[1] ** 2)

    def test_the_search_leaves_a_start_pinned_against_the_wall(self):
        simplex = [[1.2, 0.6], [0.9, 0.6], [1.2, 0.9]]
        stub = _SimplexStub(self.objective)
        result = _run(stub, simplex, max_iter=200, tol=1e-4)
        assert result is not None
        f_final, metrics, path = result
        assert f_final < 1e-3
        assert metrics["mae"] == 1.0
        assert path.shape[0] == 3 and path.shape[1] == 3

    def test_a_wholly_infeasible_start_is_rejected(self):
        simplex = [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
        stub = _SimplexStub(self.objective)
        with pytest.raises(RuntimeError, match="initialization has produced"):
            BinaryLiquid.nelder_mead(stub, max_iter=8, tol=1e-6, initial_guesses=simplex)


class TestRestoreInfeasibleVertices:
    def test_an_infinite_vertex_is_placed_on_the_feasible_side(self):
        objective = TestConstraintWall.objective
        stub = _SimplexStub(objective)

        def evaluate(x):
            return float(stub.f(x)), np.asarray(x, dtype=float)

        simplex = np.array([[1.2, 0.6], [1.4, 0.6], [0.2, 0.6]])
        f_vals = np.array([objective(v) for v in simplex])
        params = simplex.copy()
        n_restored = _restore_infeasible_vertices(evaluate, simplex, f_vals, params, 8)
        assert n_restored == 1
        assert np.isfinite(f_vals).all()
        assert simplex[2][0] >= 1.0

    def test_a_finite_simplex_is_left_alone(self):
        simplex = np.array([[1.2, 0.6], [1.4, 0.6], [1.3, 0.9]])
        f_vals = np.array([0.1, 0.2, 0.3])
        params = simplex.copy()
        before = simplex.copy()

        def evaluate(x):  # pragma: no cover - must not be reached
            raise AssertionError("a finite simplex needs no evaluation")

        assert _restore_infeasible_vertices(evaluate, simplex, f_vals, params, 8) == 0
        assert np.array_equal(simplex, before)


class TestTwoDimensionalGuard:
    def test_a_wider_guess_set_is_refused(self):
        stub = _SimplexStub(_bowl)
        stub.guess_symbols = ["L0_a", "L0_b", "L1_a"]
        with pytest.raises(NotImplementedError, match="2-D simplex"):
            BinaryLiquid.nelder_mead(stub, initial_guesses=[[0.0, 0.0]] * 3)


class TestInitialSimplexAnchor:
    """The ``L*_b`` guesses of an ``exp(-T/tau)`` format start on the S_xs(0 K) = 0 line."""

    @staticmethod
    def _triangle(monkeypatch, fmt_name, guess_keys):
        import gliquid.binary as binary

        monkeypatch.setattr(binary, "get_hull_rel_mid_depth", lambda ch: -20000.0)
        monkeypatch.setattr(binary, "get_hull_rel_enth_skew", lambda ch: 1000.0)
        hull = SimpleNamespace(facets=[object(), object()])
        fmt = PARAM_FORMATS[fmt_name]
        return build_init_triangle(guess_keys, hull, tau=DEFAULT_TAU, fmt=fmt), fmt

    def test_the_first_b_guess_is_the_zero_entropy_anchor(self, monkeypatch):
        triangle, _ = self._triangle(monkeypatch, "comb-exp", ("L0_b", "L1_a"))
        l0_a_first = -20000.0 * 2.0  # hull depth, doubled, with a stable compound present
        assert triangle[0][0] == pytest.approx(l0_a_first / DEFAULT_TAU)

    def test_the_second_b_guess_is_offset_away_from_zero(self, monkeypatch):
        triangle, _ = self._triangle(monkeypatch, "comb-exp", ("L0_b", "L1_a"))
        anchor = triangle[0][0]
        offset = triangle[2][0]
        assert offset != pytest.approx(anchor)
        assert math.copysign(1.0, offset - anchor) == math.copysign(1.0, anchor)
        assert abs(offset - anchor) >= DEFAULT_B_SPREAD_MIN

    def test_a_linear_format_keeps_the_tau_line(self, monkeypatch):
        triangle, _ = self._triangle(monkeypatch, "linear", ("L0_b", "L1_a"))
        l0_a_first = -20000.0 * 2.0
        assert triangle[0][0] == pytest.approx(l0_a_first / DEFAULT_TAU * -1.0321)


class TestOffsetVertexValue:
    def test_a_non_zero_value_is_scaled(self):
        assert _offset_vertex_value(-4.0, 10.0, 1.0) == pytest.approx(-3.2)

    def test_a_zero_value_takes_the_spread_in_the_anchor_sign(self):
        assert _offset_vertex_value(0.0, 10.0, -5.0) == pytest.approx(-10.0)
        assert _offset_vertex_value(0.0, 10.0, 5.0) == pytest.approx(10.0)
