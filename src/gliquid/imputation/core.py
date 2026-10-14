"""Dimension-agnostic pieces of phase-energy imputation, shared by binary and ternary systems.

:mod:`gliquid.imputation.binary_imputation` and :mod:`gliquid.imputation.ternary_imputation` both
build on this module; it imports neither.

Conventions:
  * A composition is either an element-keyed dict of atomic fractions, or a vector of
    INDEPENDENT fractions -- the fractions of every component after the first, in a model's
    component order. The second form is the ``X`` / ``x0, x1`` column convention of the
    H-S-x tables that ``BinaryLiquid.to_hsx`` and ``TernaryLiquidInterpolation.hsx_df`` hold.
  * Energies are J/mol-atom unless a name says eV. A solid phase carries S = 0, so its Gibbs
    energy equals its formation enthalpy at every temperature.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.spatial import ConvexHull

from gliquid.cache import IMPUTED_ID_PREFIX, canonical_entry_id
from gliquid.phase import EV_ATOM_TO_J_MOL

PROVENANCE_KEY = "provenance"
CLAMP_KEY = "clamp"
"""Provenance sub-key recording an enthalpy raised to keep a protected vertex stable."""
REPLACES_KEY = "replaces"
"""Entry-data key listing the DFT entry ids an imputed entry stands in for; a hull that places
the imputed entry leaves those entries out."""

HULL_GUARD_MARGIN_J = 1.0  # clearance of imputed H below the existing hull (J/mol-atom)
PROTECTED_VERTEX_MARGIN_J = 100.0  # clearance a clamp leaves a protected vertex (J/mol-atom)
CLAMP_TOLERANCE_J = 1.0  # bisection resolution for a clamped enthalpy (J/mol-atom)
CLAMP_BISECTION_STEPS = 60

FRACTION_FLOOR = 1e-6  # finite differences stay this far inside the composition simplex
_ENVELOPE_NORMAL_TOL = 1e-12
_BARYCENTRIC_TOL = 1e-9


# -- Compositions --


def normalize_fractions(spec, components: Sequence[str]) -> dict[str, float]:
    """Atomic fractions of ``spec`` over ``components``, in component order.

    Args:
        spec: a formula string (``'LaMnSi2'``), a ``Composition``, an element-keyed dict of
            amounts or fractions, a length-n sequence of amounts in component order, a
            length-(n-1) sequence of independent fractions, or -- for two components only --
            a float, read as the fraction of ``components[1]``.
        components: the system's element symbols.

    Returns:
        dict[str, float]: one fraction per component, summing to one.

    Raises:
        ValueError: ``spec`` names an element outside ``components``, carries a negative
            amount, sums to zero, or has the wrong length.
    """
    components = [str(c) for c in components]
    n = len(components)
    if isinstance(spec, (int, float, np.floating, np.integer)) and not isinstance(spec, bool):
        if n != 2:
            raise ValueError(f"a scalar composition needs two components, got {components}")
        x = float(spec)
        amounts = {components[0]: 1.0 - x, components[1]: x}
    elif isinstance(spec, (str, Composition)):
        comp = Composition(spec)
        amounts = {str(el): float(comp[el]) for el in comp.elements}
    elif isinstance(spec, dict):
        amounts = {str(Element(str(k))): float(v) for k, v in spec.items()}
    else:
        values = [float(v) for v in spec]
        if len(values) == n:
            amounts = dict(zip(components, values))
        elif len(values) == n - 1:
            amounts = {components[0]: 1.0 - sum(values), **dict(zip(components[1:], values))}
        else:
            raise ValueError(
                f"a composition vector needs {n} or {n - 1} values for {components}, "
                f"got {len(values)}"
            )

    foreign = sorted(set(amounts) - set(components))
    if foreign:
        raise ValueError(f"composition names {foreign}, outside the system {components}")
    if any(v < -FRACTION_FLOOR for v in amounts.values()):
        raise ValueError(f"composition carries a negative amount: {amounts}")
    total = sum(max(v, 0.0) for v in amounts.values())
    if total <= 0.0:
        raise ValueError(f"composition sums to zero: {amounts}")
    return {c: max(amounts.get(c, 0.0), 0.0) / total for c in components}


def independent_coords(fractions: dict[str, float], components: Sequence[str]) -> np.ndarray:
    """The independent-fraction vector of ``fractions``: every component after the first."""
    return np.array([float(fractions[c]) for c in list(components)[1:]], dtype=float)


# -- Liquid read-offs --


@dataclasses.dataclass(frozen=True)
class Envelope:
    """The lower convex envelope of a set of Gibbs energies at one composition.

    Attributes:
        value: the envelope, J/mol-atom.
        rows: indices of the points spanning the supporting facet.
        weights: barycentric weight of each supporting point; they sum to one.
        phases: the phase of each supporting point, when the caller supplied names.
        entropy: weight-averaged entropy of the support (J/mol-atom/K), when supplied.
    """

    value: float
    rows: tuple[int, ...]
    weights: tuple[float, ...]
    phases: tuple[str, ...] = ()
    entropy: float | None = None


def lower_envelope_at(points, g, x_query) -> Envelope:
    """The lower convex envelope of ``(points, g)`` at ``x_query``.

    The envelope is the maximum, at ``x_query``, over the planes of the lower facets of the
    convex hull of ``(x, g)``. Works for any number of independent coordinates, including one.

    Args:
        points: ``(m, d)`` independent-fraction coordinates, or ``(m,)`` when ``d == 1``.
        g: ``(m,)`` Gibbs energies.
        x_query: ``(d,)`` coordinates to read the envelope at; must lie inside the points'
            composition domain.

    Returns:
        Envelope: the value and the supporting points.

    Raises:
        ValueError: fewer than ``d + 2`` finite points, or no lower facet.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, None]
    g = np.asarray(g, dtype=float)
    xq = np.atleast_1d(np.asarray(x_query, dtype=float))
    keep = np.isfinite(g) & np.all(np.isfinite(pts), axis=1)
    index = np.flatnonzero(keep)
    pts, g = pts[keep], g[keep]
    d = pts.shape[1]
    if len(g) < d + 2:
        raise ValueError(f"need at least {d + 2} finite points, got {len(g)}")

    shift = float(g.min())
    scale = max(float(np.ptp(g)), 1.0)
    hull = ConvexHull(np.column_stack([pts, (g - shift) / scale]))
    lower = hull.equations[:, d] < -_ENVELOPE_NORMAL_TOL
    if not np.any(lower):
        raise ValueError("the points carry no lower facet")
    equations = hull.equations[lower]
    simplices = hull.simplices[lower]
    values = -(equations[:, :d] @ xq + equations[:, d + 1]) / equations[:, d]

    best = int(np.argmax(values))
    weights = _barycentric(pts[simplices[best]], xq)
    if weights.min() < -_BARYCENTRIC_TOL:
        top = values[best]
        tied = np.flatnonzero(values >= top - _BARYCENTRIC_TOL * max(1.0, abs(top)))
        for k in tied:
            w = _barycentric(pts[simplices[k]], xq)
            if w.min() >= -_BARYCENTRIC_TOL:
                best, weights = int(k), w
                break

    return Envelope(
        value=float(values[best] * scale + shift),
        rows=tuple(int(index[i]) for i in simplices[best]),
        weights=tuple(float(w) for w in weights),
    )


def _barycentric(vertices: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Barycentric weights of ``xq`` in the simplex spanned by ``vertices`` (``(d+1, d)``)."""
    system = np.vstack([vertices.T, np.ones(len(vertices))])
    rhs = np.append(xq, 1.0)
    weights, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return weights


def envelope_energy(
    table,
    x_query,
    temp: float,
    *,
    comp_cols: Sequence[str],
    phase_col: str = "Phase Name",
    exclude: Sequence[str] = (),
) -> Envelope:
    """The lower convex envelope of an H-S-x table's Gibbs energies at ``temp``.

    ``G = H - temp * S`` for every row, then :func:`lower_envelope_at`. A solid phase whose
    formation enthalpy equals the returned value is, at ``temp``, exactly on the hull the
    table describes: stable just below ``temp`` and not above it.

    Args:
        table: a DataFrame with ``comp_cols``, ``'S'``, ``'H'`` and ``phase_col``.
        x_query: independent-fraction coordinates, in ``comp_cols`` order.
        temp: temperature, K.
        comp_cols: the independent-fraction columns, e.g. ``['X']`` or ``['x0', 'x1']``.
        phase_col: the phase-name column.
        exclude: phase names left out of the envelope.

    Returns:
        Envelope: the value (J/mol-atom), the supporting rows' phases and weights, and their
        weight-averaged entropy.
    """
    rows = table[~table[phase_col].isin(list(exclude))] if len(exclude) else table
    points = rows[list(comp_cols)].to_numpy(dtype=float)
    s = rows["S"].to_numpy(dtype=float)
    g = rows["H"].to_numpy(dtype=float) - float(temp) * s
    env = lower_envelope_at(points, g, x_query)
    names = rows[phase_col].to_numpy()
    return dataclasses.replace(
        env,
        phases=tuple(str(names[i]) for i in env.rows),
        entropy=float(sum(w * s[i] for w, i in zip(env.weights, env.rows))),
    )


def tangent_value(gibbs: Callable[[np.ndarray], float], x_s, x_anchor, dx: float = 1e-3) -> float:
    """The liquid's tangent at ``x_anchor``, evaluated at ``x_s``.

    ``G(x_anchor) + dG/du (x_anchor) * |x_s - x_anchor|`` along the unit direction ``u`` from
    the anchor towards ``x_s``, by a central difference of half-width ``dx`` kept at least
    ``FRACTION_FLOOR`` inside the composition simplex. ``x_s == x_anchor`` returns
    ``G(x_anchor)``.

    Args:
        gibbs: ``G`` at one independent-fraction vector, J/mol-atom.
        x_s: independent-fraction coordinates the tangent is read at.
        x_anchor: independent-fraction coordinates the tangent touches the liquid at.
        dx: half-width of the central difference.

    Returns:
        float: the tangent's value at ``x_s``, J/mol-atom.

    Raises:
        ValueError: the anchor sits where no difference step fits inside the simplex.
    """
    xs = np.atleast_1d(np.asarray(x_s, dtype=float))
    xa = np.atleast_1d(np.asarray(x_anchor, dtype=float))
    if len(xs) == 1:
        # one independent fraction: the binary finite difference, operation for operation
        x_inv, x_target = float(xa[0]), float(xs[0])
        xl = min(max(x_inv - dx, FRACTION_FLOOR), 1.0 - FRACTION_FLOOR)
        xr = min(max(x_inv + dx, FRACTION_FLOOR), 1.0 - FRACTION_FLOOR)
        g_inv = gibbs(np.array([x_inv]))
        g_slope = (gibbs(np.array([xr])) - gibbs(np.array([xl]))) / (xr - xl)
        return g_inv + g_slope * (x_target - x_inv)

    delta = xs - xa
    norm = float(np.linalg.norm(delta))
    g_anchor = float(gibbs(xa))
    if norm == 0.0:
        return g_anchor
    u = delta / norm
    t_low, t_high = _step_bounds(xa, u)
    t_right, t_left = min(dx, t_high), max(-dx, t_low)
    if t_right <= t_left:
        raise ValueError(f"no finite-difference step fits inside the simplex at {xa.tolist()}")
    g_slope = (float(gibbs(xa + t_right * u)) - float(gibbs(xa + t_left * u))) / (t_right - t_left)
    return g_anchor + g_slope * norm


def _step_bounds(xa: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """The step interval ``[t_low, t_high]`` keeping every fraction of ``xa + t u`` in bounds."""
    full = np.concatenate([[1.0 - xa.sum()], xa])
    rate = np.concatenate([[-u.sum()], u])
    lo, hi = FRACTION_FLOOR, 1.0 - FRACTION_FLOOR
    t_low, t_high = -np.inf, np.inf
    for f, r in zip(full, rate):
        if r > 0:
            t_low, t_high = max(t_low, (lo - f) / r), min(t_high, (hi - f) / r)
        elif r < 0:
            t_low, t_high = max(t_low, (hi - f) / r), min(t_high, (lo - f) / r)
    return float(t_low), float(t_high)


# -- Entries against a hull --


def hull_ref_energy_per_atom(hull: PhaseDiagram, fractions: dict[str, float]) -> float:
    """Composition-weighted element reference energy per atom (eV/atom) on ``hull``.

    Args:
        hull: the hull supplying the elemental references.
        fractions: atomic fraction per element symbol, summing to one.

    Returns:
        float: the reference energy in eV/atom.

    Raises:
        KeyError: ``hull`` carries no reference for one of the elements.
    """
    el_refs = hull.el_refs
    total = 0.0
    for symbol, frac in fractions.items():
        total += frac * el_refs[Element(symbol)].energy_per_atom
    return total


def reanchor_imputed_entries(hull: PhaseDiagram, entry_dicts) -> list[ComputedEntry]:
    """Imputed entries rebuilt against ``hull``'s own elemental references.

    Each entry's absolute energy is recomputed from ``hull.el_refs``, its composition and
    its stored ``data['h_form_j_per_mol']``, so the rebuilt entry recovers that formation
    enthalpy via ``hull.get_form_energy_per_atom``. A stored energy is meaningless on a hull
    built from different elemental references -- replaying it unchanged shifts the formation
    energy by the reference delta.

    Args:
        hull: the hull the entries are to compete on.
        entry_dicts: tagged ``ComputedEntry.as_dict()`` payloads.

    Returns:
        list[ComputedEntry]: one re-anchored entry per dict, in the order given, keeping the
        source ``entry_id`` and ``data``.

    Raises:
        ValueError: an entry dict carries no ``data['h_form_j_per_mol']``.
        KeyError: ``hull`` carries no reference for one of an entry's elements.
    """
    rebuilt = []
    for entry_dict in entry_dicts:
        data = entry_dict.get("data") or {}
        if "h_form_j_per_mol" not in data:
            raise ValueError(
                f"imputed entry {entry_dict.get('entry_id')!r} carries no "
                f"data['h_form_j_per_mol']; its stored energy cannot be re-anchored."
            )
        source = ComputedEntry.from_dict(entry_dict)
        comp = source.composition
        fractions = {str(el): comp.get_atomic_fraction(el) for el in comp.elements}
        energy_per_atom = (
            hull_ref_energy_per_atom(hull, fractions)
            + float(data["h_form_j_per_mol"]) / EV_ATOM_TO_J_MOL
        )
        rebuilt.append(
            ComputedEntry(
                composition=comp,
                energy=energy_per_atom * comp.num_atoms,
                entry_id=source.entry_id,
                parameters=source.parameters,
                data=copy.deepcopy(data),
            )
        )
    return rebuilt


def tagged_entry_dict(
    composition: Composition, energy_per_atom: float, name: str, data: dict
) -> dict:
    """A tagged imputed ``ComputedEntry.as_dict()`` payload.

    Args:
        composition: the phase composition.
        energy_per_atom: absolute energy, eV/atom.
        name: display label; the ``entry_id`` is ``'imputed:<name>'``.
        data: the ``data`` block, stored as given.
    """
    entry = ComputedEntry(
        composition=composition,
        energy=energy_per_atom * composition.num_atoms,
        entry_id=f"{IMPUTED_ID_PREFIX}{name}",
        data=data,
    )
    return entry.as_dict()


def entry_fractions(entry_dict: dict) -> dict[str, float]:
    """An imputed entry's atomic fractions: ``data['fractions']`` when stored, else its
    composition's."""
    stored = (entry_dict.get("data") or {}).get("fractions")
    if isinstance(stored, dict) and stored:
        return {str(k): float(v) for k, v in stored.items()}
    comp = ComputedEntry.from_dict(entry_dict).composition
    return {str(el): comp.get_atomic_fraction(el) for el in comp.elements}


def entry_at_enthalpy(hull: PhaseDiagram, entry_dict: dict, h_form_j: float) -> dict:
    """A copy of ``entry_dict`` carrying ``h_form_j`` as its formation enthalpy on ``hull``."""
    out = copy.deepcopy(entry_dict)
    data = out.setdefault("data", {})
    data["h_form_j_per_mol"] = float(h_form_j)
    comp = ComputedEntry.from_dict(entry_dict).composition
    energy_per_atom = (
        hull_ref_energy_per_atom(hull, entry_fractions(entry_dict))
        + float(h_form_j) / EV_ATOM_TO_J_MOL
    )
    out["energy"] = energy_per_atom * comp.num_atoms
    return out


def replaced_entry_ids(entry_dicts) -> set[str]:
    """The DFT entry ids the given entries stand in for (``data['replaces']``), each as
    :func:`gliquid.cache.canonical_entry_id` spells it."""
    return {
        canonical_entry_id(entry_id)
        for entry_dict in entry_dicts
        for entry_id in (entry_dict.get("data") or {}).get(REPLACES_KEY, ())
    }


def augmented_hull(base: PhaseDiagram, entry_dicts) -> PhaseDiagram:
    """``base`` plus the given entry dicts, in ``base``'s element order."""
    return PhaseDiagram(
        list(base.all_entries) + [ComputedEntry.from_dict(e) for e in entry_dicts],
        elements=list(base.elements),
    )


def hull_with_reanchored_entries(
    hull: PhaseDiagram, entry_dicts
) -> tuple[PhaseDiagram, list[ComputedEntry]]:
    """``hull`` rebuilt with imputed entries re-anchored onto its elemental references.

    Args:
        hull: the hull the entries are to compete on.
        entry_dicts: tagged ``ComputedEntry.as_dict()`` payloads, each carrying
            ``data['h_form_j_per_mol']``.

    Returns:
        tuple: a new ``PhaseDiagram`` of ``hull.all_entries`` followed by the re-anchored
        entries, in ``hull``'s element order (rebuilt also for an empty list), and the
        re-anchored ``ComputedEntry`` list from :func:`reanchor_imputed_entries`.

    Raises:
        ValueError: an entry dict carries no ``data['h_form_j_per_mol']``.
        KeyError: ``hull`` carries no reference for one of an entry's elements.
    """
    entries = reanchor_imputed_entries(hull, entry_dicts)
    # element order must match the hull being replaced; pymatgen re-sorts it when omitted
    rebuilt = PhaseDiagram(list(hull.all_entries) + entries, elements=list(hull.elements))
    return rebuilt, entries


def phase_diagram_enthalpy_at(hull: PhaseDiagram, fractions: dict[str, float]) -> float:
    """Formation energy of ``hull``'s lower envelope at ``fractions``, J/mol-atom."""
    comp = Composition(fractions)
    absolute = hull.get_hull_energy_per_atom(comp)
    return float(absolute - hull_ref_energy_per_atom(hull, fractions)) * EV_ATOM_TO_J_MOL


def protected_clearance(hull: PhaseDiagram, protected: set[str]) -> float:
    """Smallest depth below the hull without it, among the protected compound vertices.

    Args:
        hull: the hull to measure the vertices on.
        protected: entry ids to measure; elemental vertices are skipped.

    Returns:
        float: the clearance in J/mol-atom, or infinity when ``hull`` carries no protected
        compound vertex.
    """
    shallowest = float("inf")
    for entry in hull.stable_entries:
        if str(entry.entry_id) not in protected or len(entry.composition.elements) < 2:
            continue
        reaction = hull.get_equilibrium_reaction_energy(entry)
        if reaction is not None:
            shallowest = min(shallowest, -float(reaction) * EV_ATOM_TO_J_MOL)
    return shallowest


# -- Admission against a growing hull --


@dataclasses.dataclass
class HullOps:
    """The hull operations admission and clamping need, bound to one system.

    Attributes:
        augmented: the system's hull plus a list of entry dicts.
        hull_enthalpy_at: the existing hull's formation enthalpy at an entry's composition.
        entry_at_enthalpy: a copy of an entry at a new formation enthalpy.
        protected_depth: the smallest clearance of the protected vertices on a hull.
    """

    augmented: Callable[[list[dict]], PhaseDiagram]
    hull_enthalpy_at: Callable[[dict], float]
    entry_at_enthalpy: Callable[[dict, float], dict]
    protected_depth: Callable[[PhaseDiagram, set[str]], float]


def clamp_bisect(
    ops: HullOps,
    entry_dict: dict,
    accepted: list[dict],
    protected: set[str],
    *,
    guard_margin_j: float = HULL_GUARD_MARGIN_J,
    protected_margin_j: float = PROTECTED_VERTEX_MARGIN_J,
    tolerance_j: float = CLAMP_TOLERANCE_J,
    steps: int = CLAMP_BISECTION_STEPS,
) -> dict | None:
    """The same phase at the deepest enthalpy that leaves every protected vertex a field.

    Bisects between the entry's enthalpy and the shallowest one that still clears the existing
    hull by ``guard_margin_j``. Raising the energy only makes the phase shallower, so the
    predicate is monotone. The target is ``protected_margin_j`` of clearance for every
    protected vertex; where no energy reaches it, the phase is clamped at the shallowest the
    guard allows.

    Args:
        ops: the system's hull operations.
        entry_dict: the candidate's tagged entry, at its inverted enthalpy.
        accepted: the entries already admitted.
        protected: entry ids that must stay stable.
        guard_margin_j: clearance the phase keeps below the existing hull.
        protected_margin_j: clearance targeted for every protected vertex.
        tolerance_j: bisection resolution.
        steps: bisection step cap.

    Returns:
        dict | None: the clamped entry with its clamp recorded on ``provenance``, or ``None``
        when no enthalpy satisfies both the protected vertices and the guard.
    """
    entry_id = str(entry_dict.get("entry_id"))
    h_inverted = float(entry_dict["data"]["h_form_j_per_mol"])
    h_shallowest = ops.hull_enthalpy_at(entry_dict) - guard_margin_j

    def survives(h_form_j: float, margin_j: float) -> tuple[bool, list[str]]:
        candidate = ops.entry_at_enthalpy(entry_dict, h_form_j)
        hull = ops.augmented(accepted + [candidate])
        stable = {str(e.entry_id) for e in hull.stable_entries}
        blockers = sorted(protected - stable)
        if entry_id not in stable or blockers:
            return False, blockers
        if margin_j <= 0.0:
            return True, blockers
        return ops.protected_depth(hull, protected) >= margin_j, blockers

    def clamped_at(h_form_j: float) -> dict:
        clamped = ops.entry_at_enthalpy(entry_dict, h_form_j)
        provenance = clamped["data"].setdefault(PROVENANCE_KEY, {})
        provenance[CLAMP_KEY] = {
            "inverted_h_form_j_per_mol": h_inverted,
            "clamped_h_form_j_per_mol": float(h_form_j),
            "blocked_by": survives(h_inverted, 0.0)[1],
        }
        return clamped

    if h_shallowest <= h_inverted:
        return None
    if not survives(h_shallowest, 0.0)[0]:
        return None
    if not survives(h_shallowest, protected_margin_j)[0]:
        return clamped_at(h_shallowest)

    deep, shallow = h_inverted, h_shallowest
    for _ in range(steps):
        if shallow - deep <= tolerance_j:
            break
        middle = 0.5 * (deep + shallow)
        if survives(middle, protected_margin_j)[0]:
            shallow = middle
        else:
            deep = middle
    return clamped_at(shallow)


def accept_growing(
    ops: HullOps,
    candidates: list[dict],
    protected: set[str],
    *,
    protect: bool,
    clamp: Callable[[dict, list[dict], set[str]], dict | None],
    vertex_noun: str = "matched vertex/vertices",
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Keep only the candidates that survive the hull as it accumulates.

    Candidates are taken strongest-first (largest ``margin_j``). Each is admitted only if the
    hull including it and everything admitted before still carries it, and -- when
    ``protect`` -- still carries every protected entry; a candidate that would displace one is
    handed to ``clamp`` and admitted at the clamped energy when that succeeds.

    Args:
        ops: the system's hull operations.
        candidates: ``{'entry_dict', 'name', 'margin_j'}`` records, in any order.
        protected: entry ids that must stay stable.
        protect: reject or clamp a candidate that displaces a protected entry. False admits it.
        clamp: ``(entry_dict, accepted, protected) -> entry_dict | None``.
        vertex_noun: how rejection reasons name a protected entry.

    Returns:
        tuple: the accepted entry dicts, and ``(name, reason)`` for each rejection.
    """
    accepted: list[dict] = []
    rejected: list[tuple[str, str]] = []

    for cand in sorted(candidates, key=lambda c: -c["margin_j"]):
        entry_id = str(cand["entry_dict"].get("entry_id"))
        stable = {
            str(e.entry_id) for e in ops.augmented(accepted + [cand["entry_dict"]]).stable_entries
        }
        if entry_id not in stable:
            rejected.append((cand["name"], "not stable on the hull once the others are in"))
            continue
        displaced = sorted(protected - stable)
        if displaced and protect:
            clamped = clamp(cand["entry_dict"], accepted, protected)
            if clamped is None:
                rejected.append(
                    (
                        cand["name"],
                        f"would displace {len(displaced)} {vertex_noun} and no "
                        f"clamped energy clears the hull: {', '.join(displaced[:3])}",
                    )
                )
                continue
            accepted.append(clamped)
            continue
        accepted.append(cand["entry_dict"])
    return accepted, rejected


# -- Provenance --


def package_version() -> str:
    """The installed gliquid version, or ``'unknown'`` when it cannot be resolved."""
    try:
        from gliquid import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - provenance falls back to 'unknown'
        return "unknown"


def provenance_header(calibration: str) -> dict:
    """The keys every imputed entry's provenance block opens with.

    Args:
        calibration: what the imputed enthalpy was calibrated against.

    Returns:
        dict: ``calibration`` and ``gliquid_version`` (:func:`package_version`), in that order.
    """
    return {"calibration": calibration, "gliquid_version": package_version()}


# relative tolerance within which recorded and current parameters count as equal
PARAM_RTOL = 1e-9


def params_moved(recorded, current, rtol: float) -> bool | None:
    """Whether two parameter vectors differ beyond ``rtol``; ``None`` when not comparable."""
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(current):
        return None
    return not all(
        abs(float(r) - float(c)) <= rtol * max(1.0, abs(float(c)))
        for r, c in zip(recorded, current)
    )
