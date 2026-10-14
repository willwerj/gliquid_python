"""Per-phase liquidus temperatures of a ternary liquid, from its tangent planes.

Every phase's Gibbs energy is ``G = H - T*S`` with ``H`` and ``S`` fixed; the liquid's are taken at
the mean component melting point, as on the ternary hull. A solid point at composition ``c``
touches the plane tangent to the liquid at ``x`` at

    T = dH / dS,
    dH = H_c - [H_L(x) + grad H_L(x) . (c - x)],
    dS = S_c - [S_L(x) + grad S_L(x) . (c - x)] ,

where the ideal mixing entropy enters the tangent plane as ``-R * sum_i c_i ln x_i`` (so a liquid
lacking an element never touches a solid containing it). A phase's sheet ``T_P(x)`` is the hottest
contact over its points. The liquidus at ``x`` is ``max_P T_P(x)``, and ``P`` is the primary phase
where its sheet is the maximum.

The liquid's own points touch the same plane in the same way, and the hottest of those contacts
is the binodal (:class:`Binodal`): below it a liquid of composition ``x`` splits into two liquids.
Where the binodal is hotter than every solid's sheet, ``x`` lies in the two-liquid field
(:data:`TWO_LIQUIDS`) and the liquidus there is the binodal.

    sheets = LiquidusSheets.from_interpolation(ti)     # a TernaryLiquidInterpolation
    T = sheets(comp)                                   # (m, len(sheets.phases)), kelvin
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gliquid.solution import R

#: Most (point, row) pairs a call evaluates at once.
CHUNK_PAIRS = 1_000_000
#: Name of the two-liquid column of a :class:`LiquidusSheets` call.
TWO_LIQUIDS = "L1 + L2"
#: How far, K, the binodal may trail the solids' liquidus at every hull node and still get a
#: column from :meth:`LiquidusSheets.from_interpolation`.
BINODAL_REACH = 50.0
#: How far, K, the binodal's grid estimate may trail ``floor`` at a point and still be refined.
BINODAL_MARGIN = 100.0
#: Second search seed: the best grid node farther than this many grid steps from the best one.
BINODAL_BASIN = 3
#: Contacts closer to ``x`` than this many grid steps are ignored by the refinement.
BINODAL_CLEARANCE = 1 / 8
#: Step, in mole fraction, of the finite differences that give the liquid's curvature.
CURVATURE_STEP = 1e-5
#: Stencil radii, in grid steps, of the binodal refinement.
BINODAL_RADII = (1 / 2, 1 / 8, 1 / 32)
#: Most moves the binodal refinement makes at each radius.
BINODAL_MOVES = 4
#: Least rise, K, for which the binodal refinement moves again at the same radius.
BINODAL_RISE = 1e-2
#: Farthest a fitted peak may lie from the current point, in stencil radii.
BINODAL_REACH_STEPS = 4.0
#: Grid nodes this close to ``x``, in mole fraction, are not contacts.
SELF_RADIUS = 1e-6

_HEX = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]], dtype=float)
_STENCIL = np.vstack([[0.0, 0.0], _HEX])
# Least-squares quadratic over the stencil; coefficients of 1, u, v, u^2, v^2, uv.
_FIT = np.linalg.pinv(
    np.column_stack([np.ones(len(_STENCIL)), _STENCIL, _STENCIL**2, _STENCIL.prod(axis=1)])
)


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


class Binodal:
    """Callable binodal temperature of a ternary liquid.

    At ``x`` it is the hottest temperature at which another liquid composition touches the plane
    tangent to the liquid at ``x``. The search scans the grid ``nodes`` for the best contact,
    then climbs on the continuous liquid from that node and from the best node of a second basin,
    and keeps the hottest result or the spinodal limit (:meth:`spinodal`), whichever is hotter.

    Args:
        liquid_model: the liquid :class:`~gliquid.solution.SolutionModel` (3 components).
        temp: temperature (K) the liquid's H and S are taken at.
        nodes: (n, 2) independent fractions of the liquid grid.
        spacing: the grid step, mole fraction.

    Attributes:
        ceiling: the hottest binodal over ``nodes`` plus :data:`BINODAL_MARGIN`, K; no two
            liquids coexist above it.
    """

    def __init__(self, liquid_model, temp, nodes, spacing):
        self.liquid_model = liquid_model
        self.temp = float(temp)
        self.nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
        self.spacing = float(spacing)
        self._node_h, self._node_s = self._h_s(self.nodes)
        full = np.clip(np.column_stack([1 - self.nodes.sum(axis=1), self.nodes]), 0.0, None)
        self._node_full = full
        self._node_present = (full > 0).T.astype(float)
        self._node_r2 = (self.nodes**2).sum(axis=1)
        self._node_tree = cKDTree(self.nodes)
        self.ceiling = np.inf
        self.ceiling = float(np.max(self(self.nodes), initial=0.0)) + BINODAL_MARGIN

    def _h_s(self, y):
        """The liquid's H and total S at ``y`` (k, 2)."""
        h, s, _, _ = self.liquid_model.nonideal_h_s_gradients((y[:, 0], y[:, 1]), self.temp)
        full = np.clip(np.column_stack([1 - y.sum(axis=1), y]), 0.0, None)
        logs = np.log(np.where(full > 0, full, 1.0))
        return h, s - R * (full * logs).sum(axis=1)

    def _tangent(self, x) -> dict:
        """The planes tangent to the liquid at ``x`` (k, 2): H plane ``a + d . y``, S plane
        ``b + e . y - R sum_i y_i ln x_i``."""
        h, s, dh, ds = self.liquid_model.nonideal_h_s_gradients((x[:, 0], x[:, 1]), self.temp)
        dh, ds = np.column_stack(dh), np.column_stack(ds)
        full = np.clip(np.column_stack([1 - x.sum(axis=1), x]), 0.0, None)
        absent = full <= 0
        return {
            "x": x,
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

    def _contact(self, tangent, y, clearance=SELF_RADIUS) -> np.ndarray:
        """(k,) contact temperature of liquid point ``y[i]`` with tangent plane ``i``; 0 within
        ``clearance`` (mole fraction) of the plane's point of tangency."""
        h, s = self._h_s(y)
        full = np.clip(np.column_stack([1 - y.sum(axis=1), y]), 0.0, None)
        d_h = h - (tangent["a"] + (tangent["d"] * y).sum(axis=1))
        s_plane = tangent["b"] + (tangent["e"] * y).sum(axis=1)
        d_s = s - (s_plane - R * (full * tangent["log_x"]).sum(axis=1))
        temps = _contact_temperatures(d_h, d_s)
        blocked = ((full > 0) & tangent["absent"]).any(axis=1)
        near = np.abs(y - tangent["x"]).max(axis=1) <= clearance
        return np.where(blocked | near, 0.0, temps)

    def spinodal(self, comp) -> np.ndarray:
        """(m,) spinodal temperature, K, at each of ``comp``: the hottest temperature at which the
        liquid's Gibbs energy loses its curvature there in some direction, the limit of the
        contact temperature of liquids approaching ``x``; 0 where it never does.

        On an edge of the triangle only the direction along the edge counts; at a corner the
        result is 0.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        step = CURVATURE_STEP
        hess_h = np.zeros((len(comp), 2, 2))
        hess_s = np.zeros((len(comp), 2, 2))
        for k in range(2):
            shift = np.zeros(2)
            shift[k] = step
            upper, lower = comp + shift, comp - shift
            _, _, dh_up, ds_up = self.liquid_model.nonideal_h_s_gradients(
                (upper[:, 0], upper[:, 1]), self.temp
            )
            _, _, dh_lo, ds_lo = self.liquid_model.nonideal_h_s_gradients(
                (lower[:, 0], lower[:, 1]), self.temp
            )
            hess_h[:, :, k] = ((dh_up - dh_lo) / (2 * step)).T
            hess_s[:, :, k] = ((ds_up - ds_lo) / (2 * step)).T
        hess_h = (hess_h + hess_h.transpose(0, 2, 1)) / 2
        hess_s = (hess_s + hess_s.transpose(0, 2, 1)) / 2
        full = np.column_stack([1 - comp.sum(axis=1), comp])
        present = full > 1e-12
        out = np.zeros(len(comp))

        interior = present.all(axis=1)
        if interior.any():
            x = full[interior]
            # minus the curvature of the total entropy, ideal mixing included
            c = -hess_s[interior] + R * (
                1 / x[:, 0, None, None] * np.ones((2, 2))
                + np.einsum("ki,ij->kij", 1 / x[:, 1:], np.eye(2))
            )
            h = hess_h[interior]
            a = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
            b = h[:, 0, 0] * c[:, 1, 1] + h[:, 1, 1] * c[:, 0, 0] - 2 * h[:, 0, 1] * c[:, 0, 1]
            det_h = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] ** 2
            disc = b**2 - 4 * a * det_h
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

    def _scan(self, tangent, floor=None) -> list[np.ndarray]:
        """The best grid node for every tangent plane, its contact temperature, and the best node
        more than :data:`BINODAL_BASIN` grid steps from it with that node's contact; the second
        node is found only where the first contact is within :data:`BINODAL_MARGIN` of
        ``floor`` (everywhere when ``floor`` is None) and is the first one elsewhere."""
        x = tangent["x"]
        best_y, best_t = np.zeros_like(x), np.zeros(len(x))
        other_y, other_t = np.zeros_like(x), np.zeros(len(x))
        _, self_node = self._node_tree.query(x, distance_upper_bound=SELF_RADIUS)
        step = max(1, CHUNK_PAIRS // len(self.nodes))
        for lo in range(0, len(x), step):
            span = slice(lo, lo + step)
            tg = self._take(tangent, span)
            rows = np.arange(len(tg["x"]))
            d_h = tg["d"] @ self.nodes.T
            np.subtract(self._node_h[None, :], d_h, out=d_h)
            d_h -= tg["a"][:, None]
            d_s = tg["e"] @ self.nodes.T
            np.subtract(R * (tg["log_x"] @ self._node_full.T), d_s, out=d_s)
            d_s += self._node_s[None, :]
            d_s -= tg["b"][:, None]
            temps = _contact_temperatures(d_h, d_s)
            edge = np.flatnonzero(tg["absent"].any(axis=1))
            if len(edge):
                blocked = tg["absent"][edge].astype(float) @ self._node_present > 0
                temps[edge] = np.where(blocked, 0.0, temps[edge])
            own = self_node[span] < len(self.nodes)
            temps[rows[own], self_node[span][own]] = 0.0
            best = temps.argmax(axis=1)
            best_y[span], best_t[span] = self.nodes[best], temps[rows, best]
            other_y[span], other_t[span] = best_y[span], best_t[span]
            wanted = rows if floor is None else rows[best_t[span] >= floor[span] - BINODAL_MARGIN]
            if len(wanted):
                y = self.nodes[best[wanted]]
                far2 = (y**2).sum(axis=1)[:, None] + self._node_r2[None, :] - 2 * y @ self.nodes.T
                apart = np.where(far2 > (BINODAL_BASIN * self.spacing) ** 2, temps[wanted], -np.inf)
                other = apart.argmax(axis=1)
                other_y[lo + wanted] = self.nodes[other]
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
        reach = np.maximum(1.0, np.hypot(u, v) / BINODAL_REACH_STEPS)
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

    def _climb(self, tangent, y, temps) -> np.ndarray:
        """The contact temperature climbed from ``y`` (k, 2): at each radius of
        :data:`BINODAL_RADII`, up to :data:`BINODAL_MOVES` moves (:meth:`_move`) while it
        rises by more than :data:`BINODAL_RISE`."""
        y, temps = y.copy(), np.maximum(temps, 0.0)
        for radius in BINODAL_RADII:
            active = np.arange(len(y))
            push = np.zeros_like(y)
            for _ in range(BINODAL_MOVES):
                if not len(active):
                    break
                moved, reached = self._move(
                    self._take(tangent, active),
                    y[active],
                    temps[active],
                    radius * self.spacing,
                    push,
                )
                rose = reached > temps[active] + BINODAL_RISE
                push = (moved - y[active])[rose]
                y[active], temps[active] = moved, reached
                active = active[rose]
        return temps

    def __call__(self, comp, floor=None) -> np.ndarray:
        """(m,) binodal temperature, K, at each of ``comp``.

        0 where no other liquid touches the tangent plane; ``inf`` where one lies below it at
        every temperature.

        Args:
            comp: (m, 2) independent fractions.
            floor: (m,) temperatures, K. Where the grid scan puts the binodal more than
                :data:`BINODAL_MARGIN` below ``floor``, the scan's value is returned unrefined;
                where ``floor`` is above :attr:`ceiling`, 0 is returned.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        if floor is not None:
            floor = np.broadcast_to(np.asarray(floor, dtype=float), (len(comp),))
        out = np.zeros(len(comp))
        if floor is not None:
            under = np.flatnonzero(floor <= self.ceiling)
            if len(under) < len(comp):
                out[under] = self(comp[under], floor[under])
                return out
        tangent = self._tangent(comp)
        y1, t1, y2, t2 = self._scan(tangent, floor)
        out = t1.copy()
        live = np.isfinite(out) & (out > 0)
        if floor is not None:
            live &= out >= floor - BINODAL_MARGIN
        rows = np.flatnonzero(live)
        if not len(rows):
            return out
        climbed = self._climb(
            self._take(tangent, rows.repeat(2)),
            np.stack([y1[rows], y2[rows]], axis=1).reshape(-1, 2),
            np.stack([t1[rows], t2[rows]], axis=1).ravel(),
        )
        best = np.maximum(climbed.reshape(-1, 2).max(axis=1), self.spinodal(comp[rows]))
        out[rows] = np.maximum(out[rows], best)
        return out


class LiquidusSheets:
    """Callable per-phase liquidus temperatures over ternary compositions.

    Args:
        liquid_model: the liquid :class:`~gliquid.solution.SolutionModel` (3 components).
        temp: temperature (K) the liquid's H and S are taken at.
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

    def __init__(self, liquid_model, temp, names, comp, H, S, binodal=None):
        names = [str(name) for name in names]
        self.liquid_model = liquid_model
        self.temp = float(temp)
        self.binodal = binodal
        self._solid_phases = tuple(dict.fromkeys(names))
        self.phases = self._solid_phases + ((TWO_LIQUIDS,) if binodal is not None else ())
        self._row_phase = np.array([self._solid_phases.index(name) for name in names], dtype=int)
        self._row_comp = np.asarray(comp, dtype=float).reshape(-1, 2)
        self._row_h = np.asarray(H, dtype=float)
        self._row_s = np.asarray(S, dtype=float)
        full = np.column_stack([1 - self._row_comp.sum(axis=1), self._row_comp])
        self._row_full = np.clip(full, 0.0, None)

    @classmethod
    def from_interpolation(cls, ti) -> LiquidusSheets:
        """The sheets of a :class:`~gliquid.ternary.TernaryLiquidInterpolation`, over the solid
        rows its hull is built from, with the binodal of its liquid grid as a column where it
        comes within :data:`BINODAL_REACH` of the solids' liquidus at some hull node."""
        table = ti.hull_table()
        solids = table[table["Phase Name"] != "L"]
        model, temp = ti.liquid_model(), ti._tm_mean()
        args = (
            model,
            temp,
            solids["Phase Name"].to_numpy(),
            solids[["x0", "x1"]].to_numpy(dtype=float),
            solids["H"].to_numpy(dtype=float),
            solids["S"].to_numpy(dtype=float),
        )
        sheets = cls(*args)
        nodes = table.loc[table["Phase Name"] == "L", ["x0", "x1"]].to_numpy(dtype=float)
        binodal = Binodal(model, temp, nodes, ti.delta)
        solid_liquidus = sheets.liquidus(nodes)
        if (binodal(nodes, floor=solid_liquidus) - solid_liquidus).max() > -BINODAL_REACH:
            return cls(*args, binodal=binodal)
        return sheets

    def point_temperatures(self, comp) -> np.ndarray:
        """(m, r) temperature, K, at which each solid point touches the liquid's tangent plane.

        0 where the point never touches it; ``inf`` where the point lies below it at every
        temperature.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        h, s, dh, ds = self.liquid_model.nonideal_h_s_gradients((comp[:, 0], comp[:, 1]), self.temp)
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

    def _solids(self, comp) -> np.ndarray:
        """(m, number of solid phases) liquidus temperature of each solid phase, K."""
        out = np.zeros((len(comp), len(self._solid_phases)))
        step = max(1, CHUNK_PAIRS // max(1, len(self._row_h)))
        for start in range(0, len(comp), step):
            temps = self.point_temperatures(comp[start : start + step])
            np.maximum.at(out[start : start + step].T, self._row_phase, temps.T)
        return out

    def __call__(self, comp) -> np.ndarray:
        """(m, len(phases)) liquidus temperature of each phase, K: its hottest point's contact,
        and the binodal in the :data:`TWO_LIQUIDS` column."""
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        out = self._solids(comp)
        if self.binodal is None:
            return out
        return np.column_stack([out, self.binodal(comp, floor=out.max(axis=1))])

    def columns(self, comp, which) -> np.ndarray:
        """(m, k) liquidus temperature, K, of phase column ``which[i, j]`` at ``comp[i]``.

        The binodal is found only on the rows asking for it, refined wherever it comes within
        :data:`BINODAL_MARGIN` of the hottest other column asked for on the row.
        """
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
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
