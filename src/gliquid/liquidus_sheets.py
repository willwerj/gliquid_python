"""Per-phase liquidus temperatures of a ternary liquid, from its tangent planes.

Every phase's Gibbs energy is ``G = H - T*S`` with ``H`` and ``S`` fixed; the liquid's are its model's
H and S at the composition's linearization temperature ``T*(x)`` (:class:`TstarField`). A solid
point at composition ``c`` touches the plane tangent to the liquid at ``x`` at

    T = dH / dS,
    dH = H_c - [H_L(x) + grad H_L(x) . (c - x)],
    dS = S_c - [S_L(x) + grad S_L(x) . (c - x)] ,

where the ideal mixing entropy enters the tangent plane as ``-R * sum_i c_i ln x_i`` (so a liquid
lacking an element never touches a solid containing it). A phase's sheet ``T_P(x)`` is the hottest
contact over its points. The liquidus at ``x`` is ``max_P T_P(x)``, and ``P`` is the primary phase
where its sheet is the maximum.

The liquid's own points touch the same plane in the same way, and the hottest of those contacts
is the binodal (:class:`Binodal`): below it a liquid of composition ``x`` splits into two liquids.
The binodal at ``x`` takes every liquid composition's H and S at the one temperature ``T*(x)``,
or at the binodal itself where the two disagree.
Where the binodal is hotter than every solid's sheet, ``x`` lies in the two-liquid field
(:data:`TWO_LIQUIDS`) and the liquidus there is the binodal.

    sheets = LiquidusSheets.from_interpolation(ti)     # a TernaryLiquidInterpolation
    T = sheets(comp)                                   # (m, len(sheets.phases)), kelvin
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from gliquid.solution import R

#: Most (point, row) pairs a call evaluates at once.
CHUNK_PAIRS = 1_000_000
#: Name of the two-liquid column of a :class:`LiquidusSheets` call.
TWO_LIQUIDS = "L1 + L2"


class TstarField:
    """The liquid's linearization temperature T*(x) over ternary compositions.

    T*(x) is the x-weighted melting point ``tm . (1 - x0 - x1, x0, x1)``, plus a correction
    interpolated linearly between ``nodes`` (nearest node outside them) that brings it to
    ``values`` there. The composition gradient of T* is taken as that of the x-weighted melting
    point everywhere.

    Args:
        tm: (3,) component melting points, K.
        nodes: (n, 2) independent fractions where T* is known; ``None`` for the x-weighted
            melting point alone.
        values: (n,) T* at ``nodes``, K.
    """

    def __init__(self, tm, nodes=None, values=None):
        self.tm = np.asarray(tm, dtype=float)
        self.gradient = self.tm[1:] - self.tm[0]
        self._linear_fill = None
        if nodes is not None:
            nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
            correction = np.asarray(values, dtype=float) - self._linear(nodes)
            self._linear_fill = LinearNDInterpolator(nodes, correction)
            self._nearest_fill = NearestNDInterpolator(nodes, correction)

    def _linear(self, y) -> np.ndarray:
        return self.tm[0] + y @ self.gradient

    def __call__(self, y) -> np.ndarray:
        """(k,) T*, K, at the independent fractions ``y`` (k, 2)."""
        y = np.asarray(y, dtype=float).reshape(-1, 2)
        if self._linear_fill is None:
            return self._linear(y)
        correction = self._linear_fill(y)
        missing = ~np.isfinite(correction)
        if missing.any():
            correction[missing] = self._nearest_fill(y[missing])
        return self._linear(y) + correction

    def h_s_gradients(self, liquid_model, y, t=None) -> tuple[np.ndarray, ...]:
        """``liquid_model``'s nonideal H and S at ``(y, T*(y))``, or at ``(y, t)`` when ``t``
        (k,) is given, with their composition gradients along T*: ``(h, s, dh, ds)`` as
        :meth:`~gliquid.solution.SolutionModel.nonideal_h_s_gradients` returns them."""
        y = np.asarray(y, dtype=float).reshape(-1, 2)
        t = self(y) if t is None else np.asarray(t, dtype=float).reshape(-1)
        h, s, dh, ds = liquid_model.nonideal_h_s_gradients((y[:, 0], y[:, 1]), t)
        dh_dt, ds_dt = liquid_model.nonideal_h_s_temperature_derivatives((y[:, 0], y[:, 1]), t)
        dh = dh + dh_dt[None, :] * self.gradient[:, None]
        ds = ds + ds_dt[None, :] * self.gradient[:, None]
        return h, s, dh, ds


#: Largest mole fraction counted as zero: the sheets are evaluated with it set to 0.
SIDE_FRACTION = 1e-14


def _snap_to_sides(comp) -> np.ndarray:
    """A copy of ``comp`` (m, 2) with every mole fraction at most :data:`SIDE_FRACTION` set to
    0; where the first one is, ``1 - x0 - x1`` is not positive."""
    comp = np.array(comp, dtype=float).reshape(-1, 2)
    comp[comp <= SIDE_FRACTION] = 0.0
    first = np.flatnonzero(1 - comp.sum(axis=1) <= SIDE_FRACTION)
    x1 = np.where(comp[first, 1] >= 1 - SIDE_FRACTION, 1.0, comp[first, 1])
    x0 = 1.0 - x1
    for _ in range(4):
        x0 = np.where(x0 + x1 < 1.0, np.nextafter(x0, 2.0), x0)
    comp[first] = np.column_stack([x0, x1])
    return comp


def _on_sides(comp) -> np.ndarray:
    """(m, 3) whether each point of ``comp`` lies on the side of the composition triangle where
    each of its three mole fractions is zero."""
    full = np.column_stack([1 - comp.sum(axis=1), comp])
    return full <= SIDE_FRACTION


def _inside(points) -> np.ndarray:
    """Which of ``points`` (..., 2) lie in the composition triangle."""
    return (points >= 0).all(axis=-1) & (points.sum(axis=-1) <= 1)


def _contact_temperatures(d_h, d_s) -> np.ndarray:
    """Temperature at which a point ``d_h``, ``d_s`` above a tangent plane touches it: 0 where it
    never does, ``inf`` where it lies below the plane at every temperature. Overwrites
    ``d_h``."""
    falling = d_s < 0
    below = ~falling & (d_h < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        temps = np.divide(d_h, d_s, out=d_h, where=falling)
    temps[~falling] = 0.0
    temps[below] = np.inf
    np.nan_to_num(temps, copy=False, nan=0.0, posinf=np.inf)
    return np.maximum(temps, 0.0, out=temps)


#: How far, K, the binodal's grid estimate may trail ``floor`` at a point and still be refined.
BINODAL_MARGIN = 100.0
#: Second search seed: the best grid node farther than this many grid steps from the best one.
BINODAL_BASIN = 3
#: Contacts closer to ``x`` than this many grid steps are ignored by the scan and the refinement.
BINODAL_CLEARANCE = 1 / 8
#: Step, in mole fraction, of the finite differences that give the liquid's curvature.
CURVATURE_STEP = 1e-5
#: Distances, grid steps, from a liquid at which :meth:`Binodal._ridge` takes contacts.
BINODAL_RIDGE = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
#: Distance, mole fraction (max-norm), within which a point takes its grid node's binodal.
SELF_RADIUS = 1e-6
#: Spacing, K, of the temperatures the grid nodes' H and S are tabulated at to rank the binodal
#: scan's nodes; the two nodes it keeps are evaluated exactly.
BINODAL_KNOT = 25.0
#: Most, K, a binodal above ``T*(x)`` may move when every liquid is taken at the binodal itself;
#: one that moves more is solved self-consistently.
BINODAL_CONSISTENCY = 5.0
#: Grid steps (max-norm) within which a node's binodal counts for a point.
BINODAL_NEAR = 1.5
#: Most change, K, of a binodal taken at itself that still counts as self-consistent where it is
#: checked; also the least excess of T* over the floor at which an unclimbed point is searched.
BINODAL_CHECK = 25.0
#: Most rounds of the search next to re-entrant dome nodes at construction.
BINODAL_FLOOD_ROUNDS = 12
#: ``wide`` value of :meth:`Binodal._solve` that also climbs from the contact point of the
#: nearest re-entrant dome node to ``x``.
CONTACTS = "contacts"

_HEX = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]], dtype=float)
# Points around x, in grid steps, whose best contact seeds the wide solve's third climb.
_RING = np.vstack(
    [r * _HEX for r in (0.5, 1.0, 1.5, 2.0)]
    + [r * np.array([[1, 1], [-1, -1], [2, -1], [-2, 1], [1, -2], [-1, 2]]) / 2 for r in (1, 2, 3)]
)
_STENCIL = np.vstack([[0.0, 0.0], _HEX])
# Least-squares quadratic over the stencil; coefficients of 1, u, v, u^2, v^2, uv.
_FIT = np.linalg.pinv(
    np.column_stack([np.ones(len(_STENCIL)), _STENCIL, _STENCIL**2, _STENCIL.prod(axis=1)])
)


@dataclass(frozen=True)
class _Climb:
    """Climb of a binodal contact on the continuous liquid (:meth:`Binodal._climb`)."""

    #: Stencil radii, in grid steps.
    radii: tuple[float, ...] = (1 / 2, 1 / 8, 1 / 32)
    #: Most moves at each radius.
    moves: int = 4
    #: Least rise, K, for which it moves again at the same radius.
    rise: float = 1e-2
    #: Farthest a fitted peak may lie from the current point, in stencil radii.
    reach_steps: float = 4.0


CLIMB = _Climb()


@dataclass(frozen=True)
class _Polish:
    """Newton steps on the Gibbs gap from each climbed contact (:meth:`Binodal._polish`)."""

    #: Most Newton steps.
    steps: int = 8
    #: Most halvings of one Newton step.
    halvings: int = 6


POLISH = _Polish()


@dataclass(frozen=True)
class _SelfConsistent:
    """Bracket of a self-consistent binodal (:meth:`Binodal._self_consistent`)."""

    #: First step, K, above ``T*(x)``, doubled per widening.
    step: float = 100.0
    #: Most widenings.
    widenings: int = 8
    #: Width, K, the binodal is bisected to.
    tolerance: float = 1e-2


SELF_CONSISTENT = _SelfConsistent()


@dataclass(frozen=True)
class _Search:
    """Searches of the binodal from a start temperature (:meth:`Binodal._search`)."""

    #: Width, K, a searched binodal's bracket is closed to.
    tolerance: float = 0.5
    #: Most fixed-point steps of a downward search.
    descent: int = 8
    #: Most evaluations of the search for a two-liquid window under a descent that falls below
    #: its floor.
    hump_steps: int = 14


SEARCH = _Search()


@dataclass(frozen=True)
class _WideScan:
    """Side rows of the wide scan (:meth:`Binodal._wide_set`)."""

    #: Rows, in grid steps from each side of the composition triangle.
    side_rows: tuple[float, ...] = tuple(k / 8 for k in range(1, 16) if k != 8)
    #: Grid steps from a dome node within which side-row points are kept.
    side_reach: float = 3.0


WIDE_SCAN = _WideScan()


class Binodal:
    """Callable binodal temperature of a ternary liquid.

    At ``x`` it is the hottest temperature at which another liquid composition touches the plane
    tangent to the liquid at ``x``, every liquid composition taking its H and S at ``T*(x)``, or
    at that temperature itself where the two disagree (:meth:`__call__`). The search scans the
    grid ``nodes`` for the best contact, then climbs on the continuous liquid from that node and
    from the best node of a second basin, and keeps the hottest result or the spinodal limit
    (:meth:`spinodal`), whichever is hotter. Where the curvature of the liquid's total entropy is
    indefinite at some interior node, contacts along the ridge (:meth:`_ridge`) are climbed too,
    and each climbed contact is refined by Newton steps on the Gibbs gap (:meth:`_polish`).

    The binodal at the grid nodes is solved once, without a floor. Nodes that show no binodal next
    to a dome node whose binodal lies more than :data:`BINODAL_MARGIN` above their T* are searched
    again downward from it (:meth:`_search`). A dome node whose dome does not reach down to the
    coolest T* of its neighbours is re-entrant, and the points near it are searched again from
    its binodal too.

    Args:
        liquid_model: the liquid :class:`~gliquid.solution.SolutionModel` (3 components).
        tstar: the :class:`TstarField` giving the temperature the liquid is taken at for each
            ``x``.
        nodes: (n, 2) independent fractions of the liquid grid.
        spacing: the grid step, mole fraction.
    """

    def __init__(self, liquid_model, tstar, nodes, spacing):
        self.liquid_model = liquid_model
        self.tstar = tstar
        self.nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
        self.spacing = float(spacing)
        self._grid = self._node_set(self.nodes)
        node_t = tstar(self.nodes)
        lo, hi = float(node_t.min()) - BINODAL_KNOT, float(node_t.max()) + BINODAL_KNOT
        self._knots = lo + BINODAL_KNOT * np.arange(int(np.ceil((hi - lo) / BINODAL_KNOT)) + 1)
        self._grid["h"], self._grid["s"] = self._knot_h_s(self.nodes)
        self._wide = None
        self._memo = {}

    @cached_property
    def ceiling(self) -> float:
        """The hottest binodal over ``nodes`` plus :data:`BINODAL_MARGIN`, K; no two liquids
        coexist above it. Found with the binodal at every grid node (:meth:`_flood`), which
        also sets the node data the wide solves read."""
        with np.errstate(all="ignore"):
            self._node_b = self._value(self.nodes)
            return self._flood(self.tstar(self.nodes))

    @staticmethod
    def _node_set(points) -> dict:
        full = np.clip(np.column_stack([1 - points.sum(axis=1), points]), 0.0, None)
        return {
            "y": points,
            "full": full,
            "present": (full > 0).T.astype(float),
            "r2": (points**2).sum(axis=1),
            "tree": cKDTree(points),
        }

    def _knot_h_s(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(knots, n) H and S of ``points`` at every tabulated temperature."""
        n = len(points)
        h, s = self._h_s(np.tile(points, (len(self._knots), 1)), np.repeat(self._knots, n))
        return h.reshape(-1, n), s.reshape(-1, n)

    def _neighbours(self, comp) -> np.ndarray:
        """(m, k) indices of the nodes within :data:`BINODAL_NEAR` grid steps (max-norm) of each
        of ``comp``; ``len(nodes)`` pads."""
        k = min(int((2 * BINODAL_NEAR + 1) ** 2), len(self.nodes))
        _, idx = self._grid["tree"].query(
            comp, k=k, p=np.inf, distance_upper_bound=BINODAL_NEAR * self.spacing
        )
        return np.asarray(idx).reshape(len(comp), -1)

    def _near(self, comp, values) -> np.ndarray:
        """(m,) the hottest of node ``values`` within :data:`BINODAL_NEAR` grid steps."""
        return np.append(values, 0.0)[self._neighbours(comp)].max(axis=1)

    def _flood(self, node_t) -> float:
        """Flag the re-entrant dome nodes (``_node_re``: their binodal, else 0) and search the
        neighbours of dome nodes that show no binodal downward from the hottest neighbouring dome
        binodal, until no node changes; store where each
        re-entrant node's binodal is reached (``_node_y``) and return the :attr:`ceiling`."""
        n = len(self.nodes)
        idx = self._neighbours(self.nodes)
        idx = np.where(idx == np.arange(n)[:, None], n, idx)
        coolest = np.append(node_t, np.inf)[idx].min(axis=1)
        reentrant = np.zeros(n, dtype=bool)
        checked, tried = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        for _ in range(BINODAL_FLOOD_ROUNDS):
            b = np.where(np.isfinite(self._node_b), self._node_b, 0.0)
            self._dome = (b > 0) & (b >= node_t - BINODAL_MARGIN)
            rows = np.flatnonzero(self._dome & ~checked & (coolest < b - BINODAL_MARGIN))
            checked[rows] = True
            if len(rows):
                reentrant[rows] = self._solve(self.nodes[rows], coolest[rows]) < coolest[rows]
            top = np.append(np.where(reentrant | self._dome, b, 0.0), 0.0)[idx].max(axis=1)
            rows = np.flatnonzero(~tried & ~self._dome & (top > node_t + BINODAL_MARGIN))
            if not len(rows):
                break
            tried[rows] = True
            found = self._search(self.nodes[rows], top[rows], node_t[rows], wide=True)
            self._node_b[rows] = np.fmax(self._node_b[rows], found)
        ceiling = float(np.max(self._node_b, initial=0.0)) + BINODAL_MARGIN
        self._node_b = np.where(np.isfinite(self._node_b), self._node_b, 0.0)
        self._node_re = np.where(reentrant, self._node_b, 0.0)
        self._node_y = np.vstack([self.nodes, [[0.0, 0.0]]])
        rows = np.flatnonzero(reentrant)
        if len(rows):
            _, self._node_y[rows] = self._solve_points(
                self.nodes[rows], self._node_b[rows], wide=True
            )
        return ceiling

    def _h_s(self, y, t):
        """The liquid's H and total S at ``y`` (k, 2) and temperatures ``t`` (k,)."""
        h, s = self.liquid_model.nonideal_h_s((y[:, 0], y[:, 1]), t)
        full = np.clip(np.column_stack([1 - y.sum(axis=1), y]), 0.0, None)
        logs = np.log(np.where(full > 0, full, 1.0))
        return h, s - R * (full * logs).sum(axis=1)

    def _plane_gaps(self, tangent, nodes) -> tuple[np.ndarray, np.ndarray]:
        """(k, n) H and total S of every node of the set ``nodes`` above each tangent plane, the
        nodes taken at the plane's temperature, interpolated between the tabulated
        temperatures."""
        pos = np.clip((np.asarray(tangent["t"], float) - self._knots[0]) / BINODAL_KNOT, 0.0, None)
        k = np.minimum(pos.astype(int), len(self._knots) - 2)
        w = np.clip(pos - k, 0.0, 1.0)
        lo, hi = int(k.min()), int(k.max()) + 2
        rows = np.arange(len(k))
        y, full = nodes["y"], nodes["full"]
        weights = np.zeros((len(k), hi - lo + 6))
        weights[rows, k - lo] = 1 - w
        weights[rows, k - lo + 1] += w
        weights[:, -6:-4] = -tangent["d"]
        weights[:, -4] = -tangent["a"]
        d_h = weights[:, :-3] @ np.vstack([nodes["h"][lo:hi], y.T, np.ones((1, len(y)))])
        weights[:, -6:-4] = -tangent["e"]
        weights[:, -4:-1] = R * tangent["log_x"]
        weights[:, -1] = -tangent["b"]
        stacked = np.vstack([nodes["s"][lo:hi], y.T, full.T, np.ones((1, len(y)))])
        return d_h, weights @ stacked

    def _wide_set(self) -> dict:
        """The grid nodes plus points on rows parallel to each side at ``WIDE_SCAN.side_rows``
        grid steps from it, kept within ``WIDE_SCAN.side_reach`` grid steps of a dome node."""
        if self._wide is None:
            rows = []
            for f in WIDE_SCAN.side_rows:
                d = f * self.spacing
                u = np.arange(0.0, 1.0 - d + 1e-12, self.spacing)
                v = np.full_like(u, d)
                rows += [
                    np.column_stack([v, u]),
                    np.column_stack([u, v]),
                    np.column_stack([u, 1 - v - u]),
                ]
            extra = np.vstack(rows)
            domes = self.nodes[self._dome] if self._dome.any() else np.full((1, 2), 10.0)
            reach, _ = cKDTree(domes).query(
                extra, p=np.inf, distance_upper_bound=WIDE_SCAN.side_reach * self.spacing
            )
            extra = np.unique(
                np.round(extra[_inside(extra + 1e-12) & np.isfinite(reach)], 12), axis=0
            )
            self._wide = self._node_set(np.vstack([self.nodes, extra]))
            h, s = self._knot_h_s(extra) if len(extra) else (np.zeros((len(self._knots), 0)),) * 2
            self._wide["h"] = np.hstack([self._grid["h"], h])
            self._wide["s"] = np.hstack([self._grid["s"], s])
        return self._wide

    def _tangent(self, x, t) -> dict:
        """The planes tangent to the liquid at ``x`` (k, 2), taken at the temperatures ``t``
        (k,): H plane ``a + d . y``, S plane ``b + e . y - R sum_i y_i ln x_i``."""
        t = np.asarray(t, dtype=float)
        h, s, dh, ds = self.liquid_model.nonideal_h_s_gradients((x[:, 0], x[:, 1]), t)
        dh, ds = np.column_stack(dh), np.column_stack(ds)
        full = np.clip(np.column_stack([1 - x.sum(axis=1), x]), 0.0, None)
        absent = full <= 0
        return {
            "x": x,
            "t": t,
            "a": h - (dh * x).sum(axis=1),
            "d": dh,
            "b": s - (ds * x).sum(axis=1),
            "e": ds,
            "log_x": np.log(np.where(absent, 1.0, full)),
            "absent": absent,
        }

    @staticmethod
    def _take(tangent, rows) -> dict:
        return {key: value[rows] for key, value in tangent.items()}

    def _gaps(self, tangent, y) -> tuple[np.ndarray, np.ndarray]:
        """(k,) H and total S of liquid point ``y[i]`` (k, 2) above tangent plane ``i``, the point
        taken at the plane's temperature."""
        h, s = self._h_s(y, tangent["t"])
        full = np.clip(np.column_stack([1 - y.sum(axis=1), y]), 0.0, None)
        d_h = h - (tangent["a"] + (tangent["d"] * y).sum(axis=1))
        s_plane = tangent["b"] + (tangent["e"] * y).sum(axis=1)
        return d_h, s - (s_plane - R * (full * tangent["log_x"]).sum(axis=1))

    def _contact(self, tangent, y, clearance) -> np.ndarray:
        """(k,) contact temperature of liquid point ``y[i]`` with tangent plane ``i``; 0 within
        ``clearance`` (mole fraction) of the plane's point of tangency."""
        temps = _contact_temperatures(*self._gaps(tangent, y))
        full = np.clip(np.column_stack([1 - y.sum(axis=1), y]), 0.0, None)
        blocked = ((full > 0) & tangent["absent"]).any(axis=1)
        near = np.abs(y - tangent["x"]).max(axis=1) <= clearance
        return np.where(blocked | near, 0.0, temps)

    def _hessians(self, comp, t) -> tuple[np.ndarray, np.ndarray]:
        """(m, 2, 2) composition Hessians of the liquid's nonideal H and S at ``comp`` (m, 2)
        and temperatures ``t`` (m,)."""
        step = CURVATURE_STEP
        hess_h = np.zeros((len(comp), 2, 2))
        hess_s = np.zeros((len(comp), 2, 2))
        for k in range(2):
            shift = np.zeros(2)
            shift[k] = step
            upper, lower = comp + shift, comp - shift
            dh_up, ds_up = self.liquid_model.nonideal_gradients((upper[:, 0], upper[:, 1]), t)
            dh_lo, ds_lo = self.liquid_model.nonideal_gradients((lower[:, 0], lower[:, 1]), t)
            hess_h[:, :, k] = ((dh_up - dh_lo) / (2 * step)).T
            hess_s[:, :, k] = ((ds_up - ds_lo) / (2 * step)).T
        return (hess_h + hess_h.transpose(0, 2, 1)) / 2, (hess_s + hess_s.transpose(0, 2, 1)) / 2

    @staticmethod
    def _onset(full, hess_h, hess_s) -> tuple[np.ndarray, ...]:
        """At interior points ``full`` (m, 3) with nonideal Hessians ``hess_h``, ``hess_s``: ``c``
        (m, 2, 2), minus the curvature of the total entropy, ideal mixing included, and ``a``,
        ``b``, ``disc`` (m,), the coefficients and discriminant of det(hess_h + T c) = a T^2 + b T
        + det(hess_h)."""
        c = -hess_s + R * (
            1 / full[:, 0, None, None] * np.ones((2, 2))
            + np.einsum("ki,ij->kij", 1 / full[:, 1:], np.eye(2))
        )
        h = hess_h
        a = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
        b = h[:, 0, 0] * c[:, 1, 1] + h[:, 1, 1] * c[:, 0, 0] - 2 * h[:, 0, 1] * c[:, 0, 1]
        det_h = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] ** 2
        return c, a, b, b**2 - 4 * a * det_h

    @cached_property
    def _indefinite(self) -> bool:
        """Whether the curvature of the liquid's total entropy is indefinite at some interior
        node, taken at ``T*`` there."""
        full = np.column_stack([1 - self.nodes.sum(axis=1), self.nodes])
        rows = np.flatnonzero((full > 1e-12).all(axis=1))
        if not len(rows):
            return False
        hessians = self._hessians(self.nodes[rows], self.tstar(self.nodes[rows]))
        return bool((self._onset(full[rows], *hessians)[1] < 0).any())

    def _ridge(self, tangent) -> np.ndarray:
        """(k,) hottest contact with each tangent plane of liquids :data:`BINODAL_RIDGE` grid
        steps either side of its point of tangency ``x``, along the direction in which the
        liquid's Gibbs energy at ``x`` loses its curvature at the coolest temperature, where the
        curvature of its total entropy is indefinite there; 0 elsewhere."""
        x, t = tangent["x"], tangent["t"]
        out = np.zeros(len(x))
        full = np.column_stack([1 - x.sum(axis=1), x])
        rows = np.flatnonzero((full > 1e-12).all(axis=1))
        if not len(rows) or not self._indefinite:
            return out
        hess_h, hess_s = self._hessians(x[rows], t[rows])
        c, a, b, disc = self._onset(full[rows], hess_h, hess_s)
        keep = np.flatnonzero((a < 0) & (disc >= 0))
        if not len(keep):
            return out
        rows, a, b, disc, hess_h, c = (v[keep] for v in (rows, a, b, disc, hess_h, c))
        roots = (-b[:, None] + np.array([-1.0, 1.0]) * np.sqrt(disc)[:, None]) / (2 * a[:, None])
        onset = np.where(roots > 0, roots, np.inf).min(axis=1)
        keep = np.flatnonzero(np.isfinite(onset))
        if not len(keep):
            return out
        rows = rows[keep]
        w, v = np.linalg.eigh(hess_h[keep] + onset[keep, None, None] * c[keep])
        v = v[np.arange(len(rows)), :, np.abs(w).argmin(axis=1)]
        reach = self.spacing * np.concatenate([BINODAL_RIDGE, -np.asarray(BINODAL_RIDGE)])
        y = x[rows, None, :] + reach[None, :, None] * v[:, None, :]
        temps = self._contact(
            self._take(tangent, rows.repeat(len(reach))), y.reshape(-1, 2), 0.0
        ).reshape(len(rows), -1)
        out[rows] = np.where(_inside(y), temps, 0.0).max(axis=1)
        return out

    def spinodal(self, comp, t=None) -> np.ndarray:
        """(m,) spinodal temperature, K, at each of ``comp``: the hottest temperature at which the
        liquid's Gibbs energy, taken at the temperatures ``t`` (m,) or at ``T*`` there, loses its
        curvature in some direction, the limit of the contact temperature of liquids approaching
        ``x``; 0 where it never does.

        On an edge of the triangle only the direction along the edge counts; at a corner the
        result is 0.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        t = self.tstar(comp) if t is None else np.asarray(t, dtype=float)
        hess_h, hess_s = self._hessians(comp, t)
        full = np.column_stack([1 - comp.sum(axis=1), comp])
        present = full > 1e-12
        out = np.zeros(len(comp))

        interior = present.all(axis=1)
        if interior.any():
            c, a, b, disc = self._onset(full[interior], hess_h[interior], hess_s[interior])
            usable = (a > 0) & (c[:, 0, 0] > 0) & (disc >= 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                root = (-b + np.sqrt(np.where(usable, disc, 0.0))) / (2 * a)
            out[interior] = np.where(usable & np.isfinite(root), np.maximum(root, 0.0), 0.0)

        edge = present.sum(axis=1) == 2
        if edge.any():
            # the direction along the edge, in full fractions: +1 and -1 on the two present ones
            x = full[edge]
            sign = np.zeros_like(x)
            sign[np.arange(len(x)), present[edge].argmax(axis=1)] = 1.0
            sign[np.arange(len(x)), 2 - present[edge][:, ::-1].argmax(axis=1)] = -1.0
            d = sign[:, 1:]
            curve_h = np.einsum("ki,kij,kj->k", d, hess_h[edge], d)
            ideal = np.where(present[edge], sign**2 / np.where(present[edge], x, 1.0), 0.0).sum(
                axis=1
            )
            curve_s = np.einsum("ki,kij,kj->k", d, hess_s[edge], d) - R * ideal
            out[edge] = np.where((curve_s < 0) & (curve_h < 0), curve_h / curve_s, 0.0)
        return out

    def _scan(self, tangent, floor=None, nodes=None) -> list[np.ndarray]:
        """The best node for every tangent plane, its contact temperature, and the best node more
        than :data:`BINODAL_BASIN` grid steps from it with that node's contact, from the nodes'
        tabulated H and S; the second node is found only where the first contact is within
        :data:`BINODAL_MARGIN` of ``floor`` (everywhere when ``floor`` is None) and is the first
        one elsewhere. Nodes within :data:`BINODAL_CLEARANCE` grid steps of the plane's point of
        tangency are not contacts. ``nodes`` is a node set (:meth:`_node_set`), the grid's when
        None."""
        nodes = self._grid if nodes is None else nodes
        y_all = nodes["y"]
        x = tangent["x"]
        best_y, best_t = np.zeros_like(x), np.zeros(len(x))
        other_y, other_t = np.zeros_like(x), np.zeros(len(x))
        _, self_node = nodes["tree"].query(
            x, p=np.inf, distance_upper_bound=BINODAL_CLEARANCE * self.spacing
        )
        step = max(1, CHUNK_PAIRS // len(y_all))
        for lo in range(0, len(x), step):
            span = slice(lo, lo + step)
            tg = self._take(tangent, span)
            rows = np.arange(len(tg["x"]))
            d_h, d_s = self._plane_gaps(tg, nodes)
            temps = _contact_temperatures(d_h, d_s)
            edge = np.flatnonzero(tg["absent"].any(axis=1))
            if len(edge):
                blocked = tg["absent"][edge].astype(float) @ nodes["present"] > 0
                temps[edge] = np.where(blocked, 0.0, temps[edge])
            own = self_node[span] < len(y_all)
            temps[rows[own], self_node[span][own]] = 0.0
            best = temps.argmax(axis=1)
            best_y[span], best_t[span] = y_all[best], temps[rows, best]
            other_y[span], other_t[span] = best_y[span], best_t[span]
            wanted = rows if floor is None else rows[best_t[span] >= floor[span] - BINODAL_MARGIN]
            if len(wanted):
                y = y_all[best[wanted]]
                far2 = (y**2).sum(axis=1)[:, None] + nodes["r2"][None, :] - 2 * y @ y_all.T
                apart = np.where(far2 > (BINODAL_BASIN * self.spacing) ** 2, temps[wanted], -np.inf)
                other = apart.argmax(axis=1)
                other_y[lo + wanted] = y_all[other]
                other_t[lo + wanted] = apart[np.arange(len(wanted)), other]
        return [best_y, best_t, other_y, other_t]

    def _move(self, tangent, y, temps, step, push) -> tuple[np.ndarray, np.ndarray]:
        """One climbing move from each of ``y`` (k, 2) with contact ``temps``: the best of the
        point itself, a hex stencil of radius ``step`` around it, the peak of a quadratic fitted
        to that stencil, and the point ``2 * push`` (k, 2) away. Returns the new points and their
        contact temperatures."""
        k = len(y)
        rows = np.arange(k)
        clearance = BINODAL_CLEARANCE * self.spacing
        cand = y[:, None, :] + step * _HEX[None, :, :]
        around = self._contact(
            self._take(tangent, np.repeat(rows, len(_HEX))), cand.reshape(-1, 2), clearance
        )
        around = np.where(_inside(cand), around.reshape(k, len(_HEX)), -np.inf)
        values = np.column_stack([temps, around])
        fitted = np.isfinite(values).all(axis=1)
        _, bu, bv, cuu, cvv, cuv = (np.where(fitted[:, None], values, 0.0) @ _FIT.T).T
        det = 4 * cuu * cvv - cuv**2
        peaked = fitted & (cuu < 0) & (det > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(peaked, (-2 * cvv * bu + cuv * bv) / det, 0.0)
            v = np.where(peaked, (-2 * cuu * bv + cuv * bu) / det, 0.0)
        reach = np.maximum(1.0, np.hypot(u, v) / CLIMB.reach_steps)
        jumps = np.stack(
            [y + step * np.column_stack([u, v]) / reach[:, None], y + 2 * push], axis=1
        )
        allowed = _inside(jumps) & np.column_stack([peaked, (push != 0).any(axis=1)])
        at_jumps = self._contact(
            self._take(tangent, np.repeat(rows, 2)), jumps.reshape(-1, 2), clearance
        )
        at_jumps = np.where(allowed, at_jumps.reshape(k, 2), -np.inf)
        choices = np.column_stack([temps, around, at_jumps])
        points = np.concatenate([y[:, None, :], cand, jumps], axis=1)
        pick = choices.argmax(axis=1)
        return points[rows, pick], choices[rows, pick]

    def _climb(self, tangent, y, temps) -> tuple[np.ndarray, np.ndarray]:
        """The contact temperature climbed from ``y`` (k, 2) and the point it is reached at: at
        each radius of ``CLIMB.radii``, up to ``CLIMB.moves`` moves (:meth:`_move`)
        while it rises by more than ``CLIMB.rise``."""
        y, temps = y.copy(), np.maximum(temps, 0.0)
        for radius in CLIMB.radii:
            active = np.arange(len(y))
            push = np.zeros_like(y)
            for _ in range(CLIMB.moves):
                if not len(active):
                    break
                moved, reached = self._move(
                    self._take(tangent, active),
                    y[active],
                    temps[active],
                    radius * self.spacing,
                    push,
                )
                rose = reached > temps[active] + CLIMB.rise
                push = (moved - y[active])[rose]
                y[active], temps[active] = moved, reached
                active = active[rose]
        return temps, y

    def _polish(self, tangent, y) -> tuple[np.ndarray, np.ndarray]:
        """(k,) contact temperature with tangent plane ``i`` of the liquid point reached from
        ``y[i]`` (k, 2) by up to ``POLISH.steps`` damped Newton steps lowering the Gibbs gap
        ``d_h - T d_s`` at the plane's temperature ``T``, and those points (k, 2).

        Steps are taken only where the plane's point of tangency and ``y[i]`` are interior. A step
        uses the absolute eigenvalues of the gap's Hessian, is at most half a grid step
        (max-norm), keeps every fraction above a tenth of its value, and is halved up to
        ``POLISH.halvings`` times until the gap falls; a point stops at the first step that
        does not lower it.
        """
        y = np.array(y, dtype=float)
        t = np.asarray(tangent["t"], dtype=float)
        log_x = tangent["log_x"]
        interior = ~tangent["absent"].any(axis=1)
        interior &= (np.column_stack([1 - y.sum(axis=1), y]) > 0).all(axis=1) & np.isfinite(t)
        rows = np.flatnonzero(interior)
        gap = np.full(len(y), np.inf)
        if len(rows):
            d_h, d_s = self._gaps(self._take(tangent, rows), y[rows])
            gap[rows] = d_h - t[rows] * d_s
        for _ in range(POLISH.steps):
            if not len(rows):
                break
            p, tt = y[rows], t[rows]
            full = np.column_stack([1 - p.sum(axis=1), p])
            _, _, dh, ds = self.liquid_model.nonideal_h_s_gradients((p[:, 0], p[:, 1]), tt)
            ideal = np.log(full[:, 1:] / full[:, :1]) - (log_x[rows, 1:] - log_x[rows, :1])
            grad = (
                np.column_stack(dh)
                - tangent["d"][rows]
                - tt[:, None] * (np.column_stack(ds) - tangent["e"][rows])
                + R * tt[:, None] * ideal
            )
            hess_h, hess_s = self._hessians(p, tt)
            w, v = np.linalg.eigh(hess_h + tt[:, None, None] * self._onset(full, hess_h, hess_s)[0])
            w = np.maximum(np.abs(w), 1e-9 * np.abs(w).max(axis=1, keepdims=True) + 1e-12)
            step = -np.einsum("kij,kj,klj,kl->ki", v, 1 / w, v, grad)
            size = np.abs(step).max(axis=1)
            step *= np.minimum(1.0, 0.5 * self.spacing / np.where(size > 0, size, 1.0))[:, None]
            lowered = np.zeros(len(rows), dtype=bool)
            for _ in range(POLISH.halvings):
                todo = np.flatnonzero(~lowered)
                if not len(todo):
                    break
                trial = p[todo] + step[todo]
                trial_full = np.column_stack([1 - trial.sum(axis=1), trial])
                kept = (trial_full > 0.1 * full[todo]).all(axis=1)
                at = np.full(len(todo), np.inf)
                if kept.any():
                    r = rows[todo[kept]]
                    d_h, d_s = self._gaps(self._take(tangent, r), trial[kept])
                    at[kept] = d_h - t[r] * d_s
                fell = at < gap[rows[todo]]
                y[rows[todo[fell]]], gap[rows[todo[fell]]] = trial[fell], at[fell]
                lowered[todo[fell]] = True
                step[todo[~fell]] *= 0.5
            rows = rows[lowered]
        return self._contact(tangent, y, BINODAL_CLEARANCE * self.spacing), y

    def __call__(self, comp, floor=None) -> np.ndarray:
        """(m,) binodal temperature, K, at each of ``comp``.

        0 where no other liquid touches the tangent plane. Where one lies below it at every
        temperature, or where a binodal more than :data:`BINODAL_CONSISTENCY` above ``T*(x)``
        (and at or above ``floor``) moves by more than that when every liquid is taken at the
        binodal itself, the binodal is the self-consistent one (:meth:`_self_consistent`);
        ``inf`` where that has no bracket.

        With ``floor``, a grid node takes its binodal solved without one, and a point near a
        node binodal that reaches ``floor`` is searched again (:meth:`_floored_rows`).

        Args:
            comp: (m, 2) independent fractions.
            floor: (m,) temperatures, K. Where the grid scan puts the binodal more than
                :data:`BINODAL_MARGIN` below ``floor``, the scan's value is returned unrefined
                unless searched again as above; where ``floor`` is above :attr:`ceiling`, 0 is
                returned.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        if floor is None:
            return self._value(comp)
        floor = np.broadcast_to(np.asarray(floor, dtype=float), (len(comp),))
        out = np.zeros(len(comp))
        under = np.flatnonzero(floor <= self.ceiling)
        if len(under):
            out[under] = self._floored(comp[under], floor[under])
        return out

    def _floored(self, comp, floor) -> np.ndarray:
        """:meth:`__call__` with ``floor`` at most the :attr:`ceiling`, remembered per
        ``(comp, floor)`` row."""
        rows = np.ascontiguousarray(np.column_stack([comp, floor]))
        keys = [row.tobytes() for row in rows]
        missing = {}
        for i, key in enumerate(keys):
            if key not in self._memo and key not in missing:
                missing[key] = i
        if missing:
            first = np.fromiter(missing.values(), dtype=int, count=len(missing))
            solved = self._floored_rows(comp[first], floor[first])
            self._memo.update(zip(missing, solved.tolist()))
        return np.array([self._memo[key] for key in keys], dtype=float)

    def _floored_rows(self, comp, floor) -> np.ndarray:
        """(m,) :meth:`__call__` with ``floor``. A point within :data:`SELF_RADIUS` of a grid
        node takes that node's binodal. A point with a node binodal at or above
        ``floor - BINODAL_MARGIN`` within :data:`BINODAL_NEAR` grid steps is searched again
        (:meth:`_search`):

        - from ``T*`` where the grid scan left it unclimbed more than :data:`BINODAL_MARGIN`
          below ``floor`` while ``T*`` lies more than :data:`BINODAL_CHECK` above ``floor``;
        - from ``T*`` with the wide solve where its value, at or above ``floor``, lies more than
          :data:`BINODAL_MARGIN` below ``T*`` with a node binodal that far above ``T*`` and
          moves by more than :data:`BINODAL_CHECK` when every liquid is taken at it;
        - from that node binodal with the wide solve where its value still lies more than
          :data:`BINODAL_MARGIN` below ``T*`` with a node binodal that far above ``T*``, and
          the binodal solved with every liquid taken at that node binodal lies more than
          :data:`BINODAL_MARGIN` above its value.

        Next to a re-entrant node more than :data:`BINODAL_MARGIN` above ``T*`` it takes the
        hottest of the :data:`CONTACTS` searches from ``T*`` and from that node's binodal and of
        its value, unless that value lies more than :data:`BINODAL_CONSISTENCY` below ``T*`` and
        moves by more than :data:`BINODAL_CHECK` when every liquid is taken at it.
        """
        out = np.zeros(len(comp))
        dist, node = self._grid["tree"].query(comp, p=np.inf)
        at_node = (dist <= SELF_RADIUS) & (_on_sides(comp) == _on_sides(self.nodes[node])).all(
            axis=1
        )
        out[at_node] = self._node_b[node[at_node]]
        rows = np.flatnonzero(~at_node)
        if not len(rows):
            return out
        x, low = comp[rows], floor[rows]
        t = self.tstar(x)
        value = self._value(x, low)
        top, top_re = self._near(x, self._node_b), self._near(x, self._node_re)
        near = top >= low - BINODAL_MARGIN
        reentrant = (top_re >= low - BINODAL_MARGIN) & (top_re > t + BINODAL_MARGIN)
        scanned = near & ~reentrant & (value < low - BINODAL_MARGIN) & (t > low + BINODAL_CHECK)
        steep = near & ~reentrant & (top > t + BINODAL_MARGIN) & (value < t - BINODAL_MARGIN)
        cooler = (value >= low) & (steep | reentrant & (value < t - BINODAL_CONSISTENCY))
        with np.errstate(all="ignore"):
            r = np.flatnonzero(scanned)
            if len(r):
                value[r] = np.fmax(value[r], self._search(x[r], t[r], low[r], wide=False))
            r = np.flatnonzero(cooler)
            if len(r):
                moved = ~(np.abs(self._solve(x[r], value[r]) - value[r]) <= BINODAL_CHECK)
                value[r[moved]] = np.where(reentrant[r[moved]], -np.inf, value[r[moved]])
                checked = r[moved & ~reentrant[r]]
                if len(checked):
                    value[checked] = self._search(x[checked], t[checked], low[checked], wide=True)
            r = np.flatnonzero(near & ~reentrant & (top > t + BINODAL_MARGIN))
            r = r[~(value[r] >= t[r] - BINODAL_MARGIN)]
            if len(r):
                screen = self._solve(x[r], top[r], value[r] + 2 * BINODAL_MARGIN)
                r = r[screen > value[r] + BINODAL_MARGIN]
            if len(r):
                value[r] = np.fmax(value[r], self._search(x[r], top[r], low[r], wide=True))
            r = np.flatnonzero(reentrant)
            if len(r):
                found = np.fmax(
                    self._search(x[r], t[r], low[r], wide=CONTACTS),
                    self._search(x[r], top_re[r], low[r], wide=CONTACTS),
                )
                value[r] = np.fmax(value[r], found)
        out[rows] = value
        return out

    def _value(self, comp, floor=None) -> np.ndarray:
        """(m,) the binodal at ``comp`` from ``T*``: solved there (``floor`` as in
        :meth:`__call__`), then self-consistent where it moves when taken at itself."""
        t = self.tstar(comp)
        out = self._solve(comp, t, floor)
        again = np.full(len(comp), np.nan)
        live = np.isfinite(out) & (out > t + BINODAL_CONSISTENCY)
        if floor is not None:
            live &= out >= floor
        rows = np.flatnonzero(live)
        if len(rows):
            with np.errstate(all="ignore"):
                again[rows] = self._solve(comp[rows], out[rows])
        moved = live & ~(np.abs(again - out) <= BINODAL_CONSISTENCY)
        rows = np.flatnonzero(np.isposinf(out) | moved)
        if len(rows):
            out[rows] = self._self_consistent(
                comp[rows], t[rows], out[rows], again[rows], tolerance=SELF_CONSISTENT.tolerance
            )
        return out

    def _search(self, comp, start, floor, wide) -> np.ndarray:
        """(k,) the self-consistent binodal at ``comp`` nearest ``start`` (k,) from either side:
        solved at ``start`` without a floor (``wide`` as in :meth:`_solve`), searched upward
        (:meth:`_self_consistent`) where hotter and downward (:meth:`_descend`) where cooler but
        at or above ``floor``."""
        start = np.asarray(start, dtype=float)
        out = self._solve(comp, start, wide=wide)
        found = out.copy()
        rows = np.flatnonzero(np.isposinf(out) | (out > start + BINODAL_CONSISTENCY))
        if len(rows):
            again = np.full(len(rows), np.nan)
            finite = np.isfinite(out[rows])
            again[finite] = self._solve(comp[rows[finite]], out[rows[finite]], wide=wide)
            moved = np.isposinf(out[rows]) | ~(np.abs(again - out[rows]) <= BINODAL_CONSISTENCY)
            r = rows[moved]
            if len(r):
                found[r] = self._self_consistent(comp[r], start[r], out[r], again[moved], wide)
        rows = np.flatnonzero(
            np.isfinite(out) & (out < start - BINODAL_CONSISTENCY) & (out >= floor)
        )
        if len(rows):
            found[rows] = self._descend(
                comp[rows], start[rows], out[rows] - start[rows], floor[rows], wide
            )
        return found

    def _self_consistent(
        self, comp, t, out, again, wide=False, tolerance=SEARCH.tolerance
    ) -> np.ndarray:
        """(k,) the temperature ``T`` above ``t`` (k,) at which the binodal at ``comp`` (k, 2),
        every liquid taken at ``T``, is ``T``.

        With ``f(T)`` the binodal taken at ``T`` minus ``T``: ``f(t) = out - t`` is positive, and
        ``f(out) = again - out`` where ``again`` is the binodal taken at a finite ``out``. ``T``
        is bracketed between ``t`` and ``out`` where ``f(out)`` is not positive, and otherwise
        from ``out`` (or ``t``) upward in steps of ``SELF_CONSISTENT.step``, doubling, at most
        ``SELF_CONSISTENT.widenings`` times, and the bracket is closed (:meth:`_close`). ``inf``
        where no bracket is found. ``wide``: as in :meth:`_solve`; ``tolerance``: as in
        :meth:`_close`.
        """
        n = len(comp)
        low, at_low = np.array(t, dtype=float), out - t
        high, at_high = np.full(n, np.nan), np.full(n, np.nan)
        closed = np.isfinite(out) & (again <= out)
        high[closed], at_high[closed] = out[closed], again[closed] - out[closed]
        above = np.isfinite(out) & (again > out)
        low[above], at_low[above] = out[above], again[above] - out[above]
        step = np.full(n, SELF_CONSISTENT.step)
        with np.errstate(all="ignore"):
            for _ in range(SELF_CONSISTENT.widenings):
                rows = np.flatnonzero(~closed)
                if not len(rows):
                    break
                temps = low[rows] + step[rows]
                at = self._solve(comp[rows], temps, wide=wide) - temps
                inside = at > 0
                up, down = rows[inside], rows[~inside]
                low[up], at_low[up] = temps[inside], at[inside]
                step[up] *= 2
                high[down], at_high[down] = temps[~inside], at[~inside]
                closed[down] = True
        found = np.full(n, np.inf)
        found[closed] = self._close(
            comp[closed],
            low[closed],
            at_low[closed],
            high[closed],
            at_high[closed],
            wide,
            tolerance,
        )
        return found

    def _close(
        self, comp, low, at_low, high, at_high, wide, tolerance=SEARCH.tolerance
    ) -> np.ndarray:
        """(k,) the middle of each bracket ``[low, high]`` (``f(low) > 0 >= f(high)``, ``f`` as
        in :meth:`_self_consistent`) after false position with the Illinois halving, bisecting
        where a step leaves more than half the bracket of two steps before, until ``|f|`` or the
        bracket is at most ``tolerance``, K. ``wide``: as in :meth:`_solve`."""
        n = len(comp)
        with np.errstate(all="ignore"):
            kept, last, before = np.zeros(n), np.full(n, np.inf), np.full(n, np.inf)
            rows = np.flatnonzero(high - low > tolerance)
            while len(rows):
                lo, hi, width = low[rows], high[rows], high[rows] - low[rows]
                trial = hi - at_high[rows] * width / (at_high[rows] - at_low[rows])
                usable = (trial > lo) & (trial < hi) & (width <= 0.5 * before[rows])
                trial = np.where(usable, trial, 0.5 * (lo + hi))
                before[rows], last[rows] = last[rows], width
                at = self._solve(comp[rows], trial, wide=wide) - trial
                inside = at > 0
                up, down = rows[inside], rows[~inside]
                at_high[up] *= np.where(kept[up] < 0, 0.5, 1.0)
                at_low[down] *= np.where(kept[down] > 0, 0.5, 1.0)
                low[up], at_low[up], kept[up] = trial[inside], at[inside], -1.0
                high[down], at_high[down], kept[down] = trial[~inside], at[~inside], 1.0
                hit = np.abs(at) <= tolerance
                low[rows[hit]] = high[rows[hit]] = trial[hit]
                rows = rows[~hit & (high[rows] - low[rows] > tolerance)]
        return 0.5 * (low + high)

    def _descend(self, comp, high, at_high, floor, wide) -> np.ndarray:
        """(k,) the hottest self-consistent binodal below ``high`` (k,), where ``f(high) =
        at_high`` is negative: steps ``T <- T + f(T)``, secant-accelerated, at most
        ``SEARCH.descent`` of them, each kept at or above ``floor``, until ``|f|`` is within
        :data:`BINODAL_CONSISTENCY` (``T + f(T)``), a step falls below ``floor`` (``T + f(T)``),
        or ``f`` turns positive (the bracket is closed, :meth:`_close`). Where the descent ends
        below ``floor`` without a bracket from a ``high`` more than :data:`BINODAL_MARGIN` above
        it, and the binodal is :attr:`_indefinite`, ``(floor, high)`` is searched for a window
        where ``f`` is positive (:meth:`_hump`)."""
        n = len(comp)
        t, f = np.array(high, dtype=float), np.array(at_high, dtype=float)
        last_t, last_f = np.full(n, np.nan), np.full(n, np.nan)
        found = t + f
        bracket = np.zeros(n, dtype=bool)
        low, at_low = np.zeros(n), np.zeros(n)
        live = np.arange(n)
        for _ in range(SEARCH.descent):
            live = live[t[live] + f[live] >= floor[live]]
            if not len(live):
                break
            ta, fa = t[live], f[live]
            with np.errstate(all="ignore"):
                secant = ta - fa * (ta - last_t[live]) / (fa - last_f[live])
            usable = np.isfinite(secant) & (secant < ta + fa) & (secant > ta + 4 * fa)
            trial = np.maximum(np.where(usable, secant, ta + fa), floor[live])
            at = self._solve(comp[live], trial, wide=wide) - trial
            found[live] = trial + at
            rising = at > 0
            bracket[live[rising]] = True
            low[live[rising]], at_low[live[rising]] = trial[rising], at[rising]
            going = ~rising & (np.abs(at) > BINODAL_CONSISTENCY) & (trial > floor[live])
            last_t[live[going]], last_f[live[going]] = ta[going], fa[going]
            t[live[going]], f[live[going]] = trial[going], at[going]
            live = live[going]
        rows = np.flatnonzero(bracket)
        if len(rows):
            found[rows] = self._close(comp[rows], low[rows], at_low[rows], t[rows], f[rows], wide)
        rows = np.flatnonzero(~bracket & ~(found >= floor) & (high > floor + BINODAL_MARGIN))
        if len(rows) and self._indefinite:
            found[rows] = np.fmax(
                found[rows], self._hump(comp[rows], floor[rows], high[rows], at_high[rows], wide)
            )
        return found

    def _hump(self, comp, low, high, at_high, wide) -> np.ndarray:
        """(k,) the hottest self-consistent binodal found in ``(low, high)`` (k,) by maximising
        ``f`` (as in :meth:`_self_consistent`; ``f(high) = at_high`` is negative) with a
        golden-section search of at most ``SEARCH.hump_steps`` evaluations, stopping where
        ``f`` turns positive; the bracket between that temperature and the coolest hotter one at
        which ``f`` is not positive is closed (:meth:`_close`). ``-inf`` where ``f`` stays
        negative. ``wide``: as in :meth:`_solve`."""
        n = len(comp)
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = np.array(low, dtype=float), np.array(high, dtype=float)
        temps = np.full((n, SEARCH.hump_steps + 1), np.nan)
        values = np.full((n, SEARCH.hump_steps + 1), np.nan)
        temps[:, 0], values[:, 0] = b, at_high
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        fc, fd = np.full(n, -np.inf), np.full(n, -np.inf)
        live = np.arange(n)
        with np.errstate(all="ignore"):
            for k in range(1, SEARCH.hump_steps + 1):
                if not len(live):
                    break
                if k == 1:
                    probe = c[live]
                elif k == 2:
                    probe = d[live]
                else:
                    left = fc[live] >= fd[live]
                    b[live] = np.where(left, d[live], b[live])
                    a[live] = np.where(left, a[live], c[live])
                    c_new = np.where(left, b[live] - ratio * (b[live] - a[live]), d[live])
                    d_new = np.where(left, c[live], a[live] + ratio * (b[live] - a[live]))
                    fc[live] = fd[live] = np.where(left, fc[live], fd[live])
                    c[live], d[live] = c_new, d_new
                    probe = np.where(left, c[live], d[live])
                at = self._solve(comp[live], probe, wide=wide) - probe
                temps[live, k], values[live, k] = probe, at
                if k == 1:
                    fc[live] = at
                elif k == 2:
                    fd[live] = at
                else:
                    fc[live] = np.where(left, at, fc[live])
                    fd[live] = np.where(left, fd[live], at)
                live = live[~(at > 0)]
        positive = values > 0
        found = np.full(n, -np.inf)
        rows = np.flatnonzero(positive.any(axis=1))
        if not len(rows):
            return found
        t_pos = np.where(positive[rows], temps[rows], -np.inf).max(axis=1)
        k_pos = np.where(positive[rows], temps[rows], -np.inf).argmax(axis=1)
        f_pos = values[rows, k_pos]
        above = (temps[rows] > t_pos[:, None]) & ~(values[rows] > 0)
        k_up = np.where(above, temps[rows], np.inf).argmin(axis=1)
        found[rows] = self._close(
            comp[rows], t_pos, f_pos, temps[rows, k_up], values[rows, k_up], wide
        )
        return found

    def _node_contact(self, comp) -> np.ndarray:
        """(m, 2) the contact point of the nearest re-entrant dome node within
        :data:`BINODAL_NEAR` grid steps of each of ``comp``; ``comp`` where there is none."""
        idx = self._neighbours(comp)
        live = np.append(self._node_re, 0.0)[idx] > 0
        node = np.vstack([self.nodes, [[np.inf, np.inf]]])[idx]
        dist = np.where(live, np.abs(node - comp[:, None, :]).max(axis=2), np.inf)
        pick = idx[np.arange(len(comp)), dist.argmin(axis=1)]
        return np.where(live.any(axis=1)[:, None], self._node_y[pick], comp)

    def _solve(self, comp, t, floor=None, wide=False) -> np.ndarray:
        """(m,) binodal temperature, K, at each of ``comp`` (m, 2), every liquid taken at the
        temperatures ``t`` (m,); ``floor`` as in :meth:`__call__`, without the :attr:`ceiling`.
        ``wide``: the scan also covers the side rows near dome nodes (:meth:`_wide_set`), and the
        best of :data:`_RING` points around ``x`` seeds a third climb; :data:`CONTACTS` adds the
        contact point of the hottest re-entrant dome node nearby (:meth:`_node_contact`) as a
        fourth."""
        return self._solve_points(comp, t, floor, wide)[0]

    def _solve_points(self, comp, t, floor=None, wide=False) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`_solve` and (m, 2) the liquid point of the hottest climbed contact (``comp``
        where nothing is climbed)."""
        comp = np.asarray(comp, dtype=float)
        point = comp.copy()
        tangent = self._tangent(comp, t)
        y1, t1, y2, t2 = self._scan(tangent, floor, self._wide_set() if wide else None)
        clearance = BINODAL_CLEARANCE * self.spacing
        t1 = np.where(t1 > 0, self._contact(tangent, y1, clearance), t1)
        t2 = np.where(t2 > 0, self._contact(tangent, y2, clearance), t2)
        seeds, temps = [y1, y2], [t1, t2]
        if wide and len(comp):
            ring = comp[:, None, :] + self.spacing * _RING[None]
            c = self._contact(
                self._take(tangent, np.arange(len(comp)).repeat(len(_RING))),
                ring.reshape(-1, 2),
                clearance,
            ).reshape(len(comp), -1)
            c = np.where(_inside(ring), c, -1.0)
            k = c.argmax(axis=1)
            seeds.append(ring[np.arange(len(comp)), k])
            temps.append(np.maximum(c[np.arange(len(comp)), k], 0.0))
        if wide == CONTACTS and len(comp):
            seeds.append(self._node_contact(comp))
            temps.append(np.maximum(self._contact(tangent, seeds[-1], clearance), 0.0))
        out = np.fmax(np.fmax.reduce(np.stack(temps)), self._ridge(tangent))
        live = np.isfinite(out) & (out > 0)
        if floor is not None:
            live &= out >= floor - BINODAL_MARGIN
        rows = np.flatnonzero(live)
        if not len(rows):
            return out, point
        seeded = self._take(tangent, rows.repeat(len(seeds)))
        climbed, at = self._climb(
            seeded,
            np.stack([y[rows] for y in seeds], axis=1).reshape(-1, 2),
            np.stack([v[rows] for v in temps], axis=1).ravel(),
        )
        if self._indefinite:
            polished, moved = self._polish(seeded, at)
            better = polished > climbed
            climbed[better], at[better] = polished[better], moved[better]
        climbed, at = climbed.reshape(-1, len(seeds)), at.reshape(-1, len(seeds), 2)
        top = climbed.argmax(axis=1)
        point[rows] = at[np.arange(len(rows)), top]
        best = np.maximum(climbed.max(axis=1), self.spinodal(comp[rows], t[rows]))
        out[rows] = np.maximum(out[rows], best)
        return out, point


#: How far, K, the binodal may trail the solids' liquidus at its closest hull node and still get
#: a column from :meth:`LiquidusSheets.from_interpolation`.
BINODAL_REACH = 50.0


@dataclass(frozen=True)
class _SolidSheets:
    """Self-consistency of the solid sheets (:meth:`LiquidusSheets._solids`)."""

    #: Most, K, the hottest solid sheet may lie from T* before the solids are taken again with
    #: the liquid at that sheet's temperature ...
    consistency: float = 5.0
    #: ... at most this many times.
    rounds: int = 3


SOLID_SHEETS = _SolidSheets()


class LiquidusSheets:
    """Callable per-phase liquidus temperatures over ternary compositions.

    Args:
        liquid_model: the liquid :class:`~gliquid.solution.SolutionModel` (3 components).
        tstar: the :class:`TstarField` the liquid's H and S are taken at.
        names: (r,) phase name of each solid point.
        comp: (r, 2) independent fractions of each solid point.
        H: (r,) enthalpy of each solid point, J/mol-atom.
        S: (r,) entropy of each solid point, J/mol-atom/K.
        binodal: a :class:`Binodal` of the same liquid, called as the last column,
            :data:`TWO_LIQUIDS`; ``None`` for solids only.

    Attributes:
        phases: the solid phase names, in order of first appearance, then :data:`TWO_LIQUIDS`
            when there is a binodal; the columns of a call.
    """

    def __init__(self, liquid_model, tstar, names, comp, H, S, binodal=None):
        names = [str(name) for name in names]
        self.liquid_model = liquid_model
        self.tstar = tstar
        self.binodal = binodal
        self._solid_phases = tuple(dict.fromkeys(names))
        self.phases = self._solid_phases + ((TWO_LIQUIDS,) if binodal is not None else ())
        self._row_phase = np.array([self._solid_phases.index(name) for name in names], dtype=int)
        self._row_comp = np.asarray(comp, dtype=float).reshape(-1, 2)
        self._row_h = np.asarray(H, dtype=float)
        self._row_s = np.asarray(S, dtype=float)
        full = np.column_stack([1 - self._row_comp.sum(axis=1), self._row_comp])
        self._row_full = np.clip(full, 0.0, None)
        self._last = (None, None)

    @classmethod
    def from_interpolation(cls, ti) -> LiquidusSheets:
        """The sheets of a :class:`~gliquid.ternary.TernaryLiquidInterpolation`, over the solid
        rows its hull is built from, with the binodal of its liquid grid as a column where it
        comes within :data:`BINODAL_REACH` of the solids' liquidus at some hull node."""
        table = ti.hull_table()
        solids = table[table["Phase Name"] != "L"]
        model, tstar = ti.liquid_model(), ti.liquid_tstar_field()
        args = (
            model,
            tstar,
            solids["Phase Name"].to_numpy(),
            solids[["x0", "x1"]].to_numpy(dtype=float),
            solids["H"].to_numpy(dtype=float),
            solids["S"].to_numpy(dtype=float),
        )
        sheets = cls(*args)
        nodes = table.loc[table["Phase Name"] == "L", ["x0", "x1"]].to_numpy(dtype=float)
        binodal = Binodal(model, tstar, nodes, ti.delta)
        solid_liquidus = sheets.liquidus(nodes)
        if (binodal(nodes, floor=solid_liquidus) - solid_liquidus).max() > -BINODAL_REACH:
            return cls(*args, binodal=binodal)
        return sheets

    def point_temperatures(self, comp, t=None) -> np.ndarray:
        """(m, r) temperature, K, at which each solid point touches the liquid's tangent plane,
        the liquid taken at T* or, when given, at ``t`` (m,).

        0 where the point never touches it; ``inf`` where the point lies below it at every
        temperature.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        h, s, dh, ds = self.tstar.h_s_gradients(self.liquid_model, comp, t)
        step = self._row_comp[None, :, :] - comp[:, None, :]
        h_tangent = h[:, None] + dh[0][:, None] * step[..., 0] + dh[1][:, None] * step[..., 1]
        s_tangent = s[:, None] + ds[0][:, None] * step[..., 0] + ds[1][:, None] * step[..., 1]
        x_full = np.clip(np.column_stack([1 - comp.sum(axis=1), comp]), 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_x = np.log(x_full)
            ideal = np.where(
                self._row_full[None] > 0, self._row_full[None] * log_x[:, None, :], 0.0
            )
        s_tangent = s_tangent - R * ideal.sum(axis=-1)
        d_h = self._row_h[None, :] - h_tangent
        d_s = self._row_s[None, :] - s_tangent
        return _contact_temperatures(d_h, d_s)

    def _solid_pass(self, comp, t=None) -> np.ndarray:
        """(m, number of solid phases) hottest contact of each solid phase, K, the liquid taken
        at T* or at ``t`` (m,)."""
        out = np.zeros((len(comp), len(self._solid_phases)))
        step = max(1, CHUNK_PAIRS // max(1, len(self._row_h)))
        for start in range(0, len(comp), step):
            part = None if t is None else t[start : start + step]
            temps = self.point_temperatures(comp[start : start + step], part)
            np.maximum.at(out[start : start + step].T, self._row_phase, temps.T)
        return out

    def _solids(self, comp) -> np.ndarray:
        """(m, number of solid phases) liquidus temperature of each solid phase, K.

        Where the hottest solid lies more than ``SOLID_SHEETS.consistency`` from T*, the liquid is
        taken again at that hottest solid's temperature, at most ``SOLID_SHEETS.rounds`` times."""
        out = self._solid_pass(comp)
        if not len(comp) or not out.shape[1]:
            return out
        taken = self.tstar(comp)
        for _ in range(SOLID_SHEETS.rounds):
            top = out.max(axis=1)
            redo = np.flatnonzero(
                np.isfinite(top) & (top > 0) & (np.abs(top - taken) > SOLID_SHEETS.consistency)
            )
            if not len(redo):
                break
            taken[redo] = top[redo]
            out[redo] = self._solid_pass(comp[redo], top[redo])
        return out

    def __call__(self, comp) -> np.ndarray:
        """(m, len(phases)) liquidus temperature of each phase, K: its hottest point's contact,
        and the binodal in the :data:`TWO_LIQUIDS` column, at ``comp`` (m, 2) put on the sides
        of the composition triangle it lies within :data:`SIDE_FRACTION` of. A call at the
        compositions of the call before returns a copy of its result."""
        comp = _snap_to_sides(comp)
        key = comp.tobytes()
        if self._last[0] != key:
            out = self._solids(comp)
            if self.binodal is not None:
                out = np.column_stack([out, self.binodal(comp, floor=out.max(axis=1))])
            self._last = (key, out)
        return self._last[1].copy()

    def columns(self, comp, which) -> np.ndarray:
        """(m, k) liquidus temperature, K, of phase column ``which[i, j]`` at ``comp[i]``.

        The binodal is found only on the rows asking for it, refined wherever it comes within
        :data:`BINODAL_MARGIN` of the hottest other column asked for on the row. ``comp`` is put
        on the sides as in :meth:`__call__`.
        """
        comp = _snap_to_sides(comp)
        which = np.asarray(which, dtype=int).reshape(len(comp), -1)
        wanted = which == len(self._solid_phases)
        out = np.take_along_axis(self._solids(comp), np.where(wanted, 0, which), axis=1)
        rows = np.flatnonzero(wanted.any(axis=1))
        if len(rows):
            floor = np.where(wanted[rows], -np.inf, out[rows]).max(axis=1)
            found = self.binodal(comp[rows], floor=floor)
            out[rows] = np.where(wanted[rows], found[:, None], out[rows])
        return out

    def liquidus(self, comp) -> np.ndarray:
        """(m,) liquidus temperature, K: the hottest sheet."""
        return self(comp).max(axis=1)
