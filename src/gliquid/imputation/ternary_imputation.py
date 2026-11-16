"""Phase-energy imputation for ternary systems.

Gives a solid phase of a :class:`~gliquid.ternary.TernaryLiquidInterpolation` the formation
enthalpy at which it leaves the ternary hull at a given temperature. The enthalpy is read off
the lower convex envelope, at that temperature and at the phase's composition, of every other
phase in the H-S-x table the hull is built from: the liquid surface, any solid-solution
surfaces, and the solid phases (:func:`gliquid.imputation.core.envelope_energy`). With a
``liquid_composition`` it is instead the liquid's tangent at that composition
(:func:`gliquid.imputation.core.tangent_value`), the ternary counterpart of the binary
peritectic/eutectic construction; a liquid composition equal to the phase's own gives the
congruent case ``G_liq(x, T)``.

Imputed phases always join the ternary object's ``imputed_entries``. They are also written to
its cache store when ``write_cache`` is on, which defaults to
:data:`gliquid.config.persist_ternary_imputed`. Give the ternary object a
:class:`gliquid.cache.OverlayBackend` as its store to keep written phases for one session only;
a later ternary object built on the same store with ``include_imputed=True`` reads them back.

A phase that would push a stable phase off the 0 K hull is not placed: the call warns, naming
the phases it would displace, and ``force=True`` places it anyway. ``hold_hull=True`` places it
at the deepest enthalpy that keeps those phases, with the entropy that still takes it off the
hull at its temperature.

A target with a ``solidus`` is placed between two temperatures: its enthalpy and entropy put it
on the envelope at its solidus and again at the temperature it leaves the hull at, so it is on
the hull between the two and above it at 0 K.

:func:`adjust_ternary_phase_temperature` moves the temperature an EXISTING phase leaves
the hull at: a DFT phase is replaced by an imputed one of the same name (its DFT entries
are left off every hull that places the imputed one, and dropping the imputed phase
restores them), and an imputed phase is re-imputed. :func:`edit_ternary_phases` applies a list
of additions, temperature changes and removals in one call.

Typical use::

    ti = TernaryLiquidInterpolation(["La", "Mn", "Si"], xs_mix=..., include_imputed=True)
    result = ti.impute_phase("LaMnSi2", 1473.0)
    result.accepted, result.enthalpy, result.realized_temperature_k
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from collections.abc import Sequence

import numpy as np
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as config
from gliquid.cache import CacheModeError, imputed_entry_label, is_imputed_entry_dict
from gliquid.imputation.core import (
    HULL_GUARD_MARGIN_J,
    PROVENANCE_KEY,
    REPLACES_KEY,
    HullOps,
    accept_growing,
    augmented_hull,
    clamp_bisect,
    entry_at_enthalpy,
    entry_fractions,
    envelope_energy,
    hull_ref_energy_per_atom,
    independent_coords,
    normalize_fractions,
    phase_diagram_enthalpy_at,
    protected_clearance,
    provenance_header,
    replaced_entry_ids,
    tagged_entry_dict,
    tangent_value,
)
from gliquid.phase import EV_ATOM_TO_J_MOL

logger = logging.getLogger(__name__)

CALIBRATIONS = {"envelope": "ternary_envelope", "tangent": "ternary_tangent"}
LIQUID_PHASE = "L"
DFT_TYPE = api.DEFAULT_DFT_TYPE
COMPOSITION_TOL = 1e-4  # largest fraction difference at which two compositions coincide
#: The sets of ``temperature``, ``solidus``, ``enthalpy`` and ``entropy`` a target may name.
_TARGET_KEYS = (
    ["temperature"],
    ["solidus", "temperature"],
    ["temperature", "enthalpy"],
    ["temperature", "entropy"],
    ["solidus", "enthalpy"],
    ["solidus", "entropy"],
    ["enthalpy", "entropy"],
)
MIN_SUPPORT_ENTROPY = 1e-9  # J/mol-atom/K; an envelope with less cannot move with temperature
HSX_TOLERANCE_K = 1.0  # realized-temperature window in hsx mode; gtx uses its T_incr
SOLIDUS_LIMIT_TOL_K = 1.0  # resolution of the nearest solidus reported on a rejection
# Largest share of the envelope's entropy at a phase's top temperature that the entropy solved
# from its solidus may take.
SOLIDUS_ENTROPY_SHARE = 0.9


@dataclasses.dataclass
class TernaryImputationResult:
    """The outcome of imputing one ternary phase.

    Attributes:
        name: the phase label, also the ``entry_id`` suffix.
        fractions: atomic fraction per component.
        temperature_k: the temperature the phase was asked to leave the hull at; ``None`` for
            a phase placed without one.
        solidus_k: the temperature the phase was asked to join the hull at on heating, K;
            ``None`` for a phase stable from 0 K.
        solidus_limit_k: on a rejection of ``solidus_k``, the nearest temperature, K, the phase
            can join the hull at: the lowest for ``'below_solidus'``, the highest for
            ``'no_entropy'``. ``None`` when there is none.
        method: ``'envelope'`` or ``'tangent'``.
        accepted: the phase joined the ternary object (or would have, with ``apply=False``).
        code: when accepted, ``'imputed'`` or ``'off_target'`` (the processed diagram shows
            the phase leaving the hull outside the temperature window); when rejected,
            ``'name_taken'``, ``'dft_covers'``, ``'occupied'``, ``'below_solidus'``,
            ``'unstable'``, ``'not_stable'``, ``'displaces_hull'``, ``'no_entropy'`` or
            ``'unrealized'``.
        reason: a sentence describing ``code``.
        enthalpy: the formation enthalpy the phase carries, J/mol-atom.
        inverted_enthalpy: the enthalpy the inversion produced, J/mol-atom.
        entropy: the entropy the phase carries, J/mol-atom/K. Nonzero where ``hold_hull`` held
            the enthalpy at the 0 K hull limit, and for a phase with a ``solidus_k``.
        support_entropy: the entropy of the envelope under the phase at ``temperature_k``,
            J/mol-atom/K. ``None`` for the tangent.
        displaces: names of the stable phases the inverted enthalpy pushes off the 0 K hull.
            Non-empty on an accepted phase only with ``force=True`` or a nonzero ``entropy``.
        decomposition: ``(phase, weight)`` pairs spanning the envelope at the phase's
            composition -- what it decomposes into at ``temperature_k``. Empty for the tangent.
        melts: the decomposition includes the liquid. ``None`` for the tangent.
        realized_temperature_k: the highest temperature the processed diagram shows the phase
            at, K; ``None`` when the diagram was not processed.
        entry_dict: the tagged ``ComputedEntry.as_dict()`` payload.
        written: the entry was written to the ternary object's cache store.
        replaces: DFT entry ids this phase stands in for, left off the hull while it is placed.
        previous_enthalpy: from :func:`adjust_ternary_phase_temperature`, the formation
            enthalpy the phase had on the hull before, J/mol-atom.
    """

    name: str
    fractions: dict[str, float]
    temperature_k: float | None
    method: str
    solidus_k: float | None = None
    solidus_limit_k: float | None = None
    accepted: bool = False
    code: str = ""
    reason: str = ""
    enthalpy: float | None = None
    inverted_enthalpy: float | None = None
    entropy: float = 0.0
    support_entropy: float | None = None
    displaces: tuple[str, ...] = ()
    decomposition: tuple[tuple[str, float], ...] = ()
    melts: bool | None = None
    realized_temperature_k: float | None = None
    entry_dict: dict | None = None
    written: bool = False
    replaces: tuple[str, ...] = ()
    previous_enthalpy: float | None = None

    def reject(self, code: str, reason: str) -> None:
        self.accepted, self.code, self.reason = False, code, reason


# -- Pieces --


def _default_name(spec, fractions: dict[str, float]) -> str:
    if isinstance(spec, (str, Composition)):
        return Composition(spec).reduced_formula
    return Composition({k: v for k, v in fractions.items() if v > 0}).reduced_formula


def _coincides(a: dict[str, float], b: dict[str, float]) -> bool:
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)) <= COMPOSITION_TOL


def _stable_compounds(hull, components) -> list[tuple[str, dict[str, float], bool]]:
    """``(entry_id, fractions, is_imputed)`` for every stable compound on ``hull``."""
    out = []
    for entry in hull.stable_entries:
        comp = entry.composition
        if len(comp.elements) < 2:
            continue
        fractions = {c: comp.get_atomic_fraction(c) for c in components}
        out.append((str(entry.entry_id), fractions, is_imputed_entry_dict(entry.as_dict())))
    return out


def _hull_ops(base) -> HullOps:
    return HullOps(
        augmented=lambda entry_dicts: augmented_hull(base, entry_dicts),
        hull_enthalpy_at=lambda entry_dict: phase_diagram_enthalpy_at(
            base, entry_fractions(entry_dict)
        ),
        entry_at_enthalpy=lambda entry_dict, h_form_j: entry_at_enthalpy(
            base, entry_dict, h_form_j
        ),
        protected_depth=protected_clearance,
    )


def _ternary_provenance(tli, record, liquid_fractions, margin_j, base) -> dict:
    return {
        **provenance_header(CALIBRATIONS[record.method]),
        "liquid": tli.imputation_params(),
        "anchor": {
            "method": record.method,
            "temp": None if record.temperature_k is None else float(record.temperature_k),
            "fractions": {k: float(v) for k, v in record.fractions.items()},
            "liquid_fractions": None
            if liquid_fractions is None
            else {k: float(v) for k, v in liquid_fractions.items()},
        },
        "margin_j": float(margin_j),
        "hull_entry_ids": sorted(str(e.entry_id) for e in base.stable_entries),
    }


def _liquid_at(tli, table, temp: float):
    """A copy of ``table`` whose liquid rows carry the liquid's H and S at ``temp`` (K)."""
    out = table.copy()
    liquid = (out["Phase Name"] == LIQUID_PHASE).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        h, s = tli.liquid_model().h_s_grid(
            (out["x0"].to_numpy(float)[liquid], out["x1"].to_numpy(float)[liquid]), float(temp)
        )
    out.loc[liquid, "H"] = np.where(np.isfinite(h), h, 0)
    out.loc[liquid, "S"] = np.where(np.isfinite(s), s, 0)
    return out


def _invert_ternary_phase_energy(
    tli, record, table, liquid_fractions, margin_j, dx, given
) -> float | None:
    """The enthalpy for ``record``, or ``None`` after rejecting it.

    ``given`` holds the target's ``enthalpy`` (J/mol-atom) and ``entropy`` (J/mol-atom/K), where
    it names them; the one it does not name is solved from the target's temperature or solidus
    and set on ``record``.
    """
    components = list(tli.components)
    if record.method == "tangent":
        return float(
            tangent_value(
                lambda x: tli.liquid_gibbs(x, record.temperature_k),
                independent_coords(record.fractions, components),
                independent_coords(liquid_fractions, components),
                dx,
            )
        )
    x_query = np.round(independent_coords(record.fractions, components), tli.HULL_COMP_DECIMALS)
    if record.temperature_k is None and record.solidus_k is None:
        record.entropy = float(given["entropy"])
        return float(given["enthalpy"])
    if record.temperature_k is None:
        return _join_at_solidus(tli, record, table, x_query, margin_j, given)
    env = envelope_energy(
        _liquid_at(tli, table, record.temperature_k),
        x_query,
        record.temperature_k,
        comp_cols=["x0", "x1"],
        exclude=[record.name],
    )
    record.decomposition = tuple(
        (p, float(w)) for p, w in zip(env.phases, env.weights) if w > 1e-12
    )
    record.melts = LIQUID_PHASE in env.phases
    record.support_entropy = float(env.entropy or 0.0)
    if record.support_entropy <= MIN_SUPPORT_ENTROPY:
        record.reject(
            "below_solidus",
            f"at {record.temperature_k:g} K every phase around this composition carries zero "
            f"entropy, so no enthalpy makes the phase leave the hull at that temperature",
        )
        return None
    if given:
        top = float(env.value) - margin_j
        if "enthalpy" in given:
            enthalpy = float(given["enthalpy"])
            record.entropy = (enthalpy - top) / record.temperature_k
        else:
            record.entropy = float(given["entropy"])
            enthalpy = top + record.temperature_k * record.entropy
        if not record.entropy < record.support_entropy:
            record.reject(
                "no_entropy",
                f"with an entropy of {record.entropy:.2f} J/mol-atom/K the phase would join the "
                f"hull at {record.temperature_k:g} K on heating, not leave it: the phases around "
                f"it carry {record.support_entropy:.2f} there",
            )
            return None
        return enthalpy
    if record.solidus_k is None:
        return float(env.value) - margin_j

    def solve(solidus) -> tuple[float | None, bool]:
        """The entropy that also puts the phase on the envelope at ``solidus`` (None where the
        liquid is absent there, or the phase would be off the hull between the two
        temperatures), and whether the liquid is there."""
        low = envelope_energy(
            _liquid_at(tli, table, solidus),
            x_query,
            solidus,
            comp_cols=["x0", "x1"],
            exclude=[record.name],
        )
        if not any(p == LIQUID_PHASE and w > 1e-12 for p, w in zip(low.phases, low.weights)):
            return None, False
        entropy = (float(low.value) - float(env.value)) / (record.temperature_k - solidus)
        ceiling = SOLIDUS_ENTROPY_SHARE * record.support_entropy
        fits = float(low.entropy or 0.0) - 1e-9 <= entropy <= ceiling
        return (entropy if fits else None), True

    def boundary(bad, good) -> float:
        """The temperature nearest ``bad`` that :func:`solve` accepts, given that it accepts
        ``good``."""
        while abs(good - bad) > SOLIDUS_LIMIT_TOL_K:
            middle = 0.5 * (good + bad)
            bad, good = (bad, middle) if solve(middle)[0] is not None else (middle, good)
        return good

    entropy, liquid = solve(record.solidus_k)
    if entropy is not None:
        record.entropy = entropy
        return float(env.value) - margin_j + record.temperature_k * entropy
    cold, hot = (1.0, record.solidus_k) if liquid else (record.solidus_k, record.temperature_k)
    while hot - cold > SOLIDUS_LIMIT_TOL_K:
        middle = 0.5 * (cold + hot)
        cold, hot = (cold, middle) if solve(middle)[1] else (middle, hot)
    if solve(hot)[0] is not None:
        record.solidus_limit_k = boundary(record.solidus_k, hot) if liquid else hot
    limit = record.solidus_limit_k
    if liquid:
        record.reject(
            "no_entropy",
            f"no single entropy puts the phase on the hull from {record.solidus_k:g} K to "
            f"{record.temperature_k:g} K and off it outside that range; "
            + (
                "no lower solidus does"
                if limit is None
                else f"the highest that does is {limit:.0f} K"
            ),
        )
    else:
        record.reject(
            "below_solidus",
            f"at {record.solidus_k:g} K no liquid reaches this composition, so the phase cannot "
            f"join the hull there; "
            + (
                "no higher solidus works"
                if limit is None
                else f"the lowest that works is {limit:.0f} K"
            ),
        )
    return None


def _join_at_solidus(tli, record, table, x_query, margin_j, given) -> float | None:
    """The enthalpy that puts ``record`` on the envelope at its solidus with the enthalpy or the
    entropy of ``given``, or ``None`` after rejecting it; the other one is set on ``record``."""
    low = envelope_energy(
        _liquid_at(tli, table, record.solidus_k),
        x_query,
        record.solidus_k,
        comp_cols=["x0", "x1"],
        exclude=[record.name],
    )
    record.decomposition = tuple(
        (p, float(w)) for p, w in zip(low.phases, low.weights) if w > 1e-12
    )
    record.melts = LIQUID_PHASE in low.phases
    record.support_entropy = float(low.entropy or 0.0)
    if not any(p == LIQUID_PHASE and w > 1e-12 for p, w in zip(low.phases, low.weights)):
        record.reject(
            "below_solidus",
            f"at {record.solidus_k:g} K no liquid reaches this composition, so the phase cannot "
            f"join the hull there",
        )
        return None
    bottom = float(low.value) - margin_j
    if "enthalpy" in given:
        enthalpy = float(given["enthalpy"])
        record.entropy = (enthalpy - bottom) / record.solidus_k
    else:
        record.entropy = float(given["entropy"])
        enthalpy = bottom + record.solidus_k * record.entropy
    if not record.entropy > record.support_entropy:
        record.reject(
            "no_entropy",
            f"with an entropy of {record.entropy:.2f} J/mol-atom/K the phase would leave the "
            f"hull at {record.solidus_k:g} K on heating, not join it: the phases around it carry "
            f"{record.support_entropy:.2f} there",
        )
        return None
    return enthalpy


def _realized_top_k(tli, name: str) -> float | None:
    try:
        tops = tli.get_inter_melting_temps([name])
    except ValueError:
        return None
    top = tops.get(name)
    return None if top is None else float(top) + 273.15


def _screen(tli, admitted: list[TernaryImputationResult]) -> None:
    """Process ``tli`` and drop, weakest first, any admitted phase the diagram does not show."""
    pending = list(admitted)
    for _ in range(len(admitted) + 1):
        tli.process_data()
        tops = {r.name: _realized_top_k(tli, r.name) for r in pending}
        missing = [r for r in pending if tops[r.name] is None]
        if not missing:
            for record in pending:
                record.realized_temperature_k = tops[record.name]
                if record.temperature_k is not None:
                    _check_window(tli, record)
            return
        dropped = missing[-1]
        dropped.reject("unrealized", "the processed diagram shows no field for this phase")
        pending.remove(dropped)
        tli.imputed_entries = [
            e for e in tli.imputed_entries if imputed_entry_label(e) != dropped.name
        ]
        tli.reset_derived_state()
        if not pending:
            return
    raise RuntimeError("the realizability screen did not settle")


def _check_window(tli, record: TernaryImputationResult) -> None:
    low = HSX_TOLERANCE_K if tli.hull_mode == "hsx" else float(tli.T_incr)
    realized, target = record.realized_temperature_k, record.temperature_k
    if not (target - low - 1e-6 <= realized <= target + HSX_TOLERANCE_K):
        record.code = "off_target"
        record.reason = (
            f"asked to leave the hull at {target:g} K; the processed diagram shows it up to "
            f"{realized:.1f} K"
        )


# -- Public API --


def impute_ternary_phases(
    tli,
    targets: Sequence[dict],
    *,
    force: bool = False,
    hold_hull: bool = False,
    write_cache: bool | None = None,
    apply: bool = True,
    screen: bool = True,
    margin_j: float = HULL_GUARD_MARGIN_J,
    dx: float = 1e-3,
) -> list[TernaryImputationResult]:
    """Impute ternary solid phases from their compositions and the temperatures they leave the
    hull at, or from an enthalpy or an entropy given for them.

    Each target's enthalpy is inverted (envelope, or tangent when it names a
    ``liquid_composition``) and must clear the 0 K solid hull. The survivors are admitted
    strongest-first. A target with a ``solidus`` gets the entropy that also puts it on the
    envelope at that temperature; it lies above the 0 K hull and is admitted without the 0 K
    checks. A target that names an ``enthalpy`` or an ``entropy`` carries it as given, with the
    other solved from its ``temperature`` or its ``solidus``, or carries both as given; above the
    0 K hull it is admitted without the 0 K checks and, unless ``force`` is set, needs a
    positive entropy; ``hold_hull`` does not move it. A phase that would push a stable phase --
    DFT or imputed, and stable before any entry a target ``replaces`` is left off -- off the 0 K
    hull is rejected with a warning naming those phases, unless ``force`` or ``hold_hull`` is
    set. With ``apply``, admitted phases replace any imputed phase of the same name in
    ``tli.imputed_entries`` and ``tli``'s derived state is reset, so the next
    ``process_data`` rebuilds with them. Rejections are also listed in
    ``tli.ternary_meta['imputation_rejected']``.

    Args:
        tli: a :class:`~gliquid.ternary.TernaryLiquidInterpolation`; mutated when ``apply``.
        targets: dicts with ``composition`` (a formula, ``Composition``, element dict, or
            fraction vector in ``tli.components`` order) and ``temperature`` (K), and
            optionally ``name`` (default: the reduced formula), ``liquid_composition``,
            ``replaces`` (DFT entry ids the phase stands in for, left off the hull), and
            ``solidus`` (K, below ``temperature``; envelope targets only): the temperature the
            phase joins the hull at on heating. An envelope target may name ``enthalpy`` (the
            formation enthalpy, J/mol-atom) or ``entropy`` (J/mol-atom/K) beside its
            ``temperature`` or, without a ``temperature``, beside its ``solidus``; or both and
            neither temperature.
        force: place a phase even when it pushes stable phases off the 0 K hull.
        hold_hull: place an envelope phase that would push stable phases off the 0 K hull at
            the deepest enthalpy that keeps them (:func:`gliquid.imputation.core.clamp_bisect`),
            with the entropy at which it still leaves the hull at its temperature. The phase
            is rejected when that entropy is not below the envelope's.
        write_cache: also write admitted phases to ``tli.data_dir``. ``None`` uses
            :data:`gliquid.config.persist_ternary_imputed`. Requires ``apply``.
        apply: add admitted phases to ``tli``. False computes only and leaves ``tli`` as it was.
        screen: with ``apply``, process the diagram, drop any admitted phase it does not show,
            and record each phase's ``realized_temperature_k``.
        margin_j: J/mol-atom. The envelope enthalpy is lowered by this much, so the phase is
            strictly stable below its temperature; it is also the clearance required below the
            0 K hull.
        dx: half-width of the tangent's finite difference.

    Returns:
        list[TernaryImputationResult]: one record per target, in the order given.

    Raises:
        ValueError: a composition is not one of ``tli``'s components' compositions, a
            temperature is not positive, a ``solidus`` is not below its ``temperature`` or
            comes with a ``liquid_composition``, or a target does not name one of: a
            ``temperature``; a ``temperature`` and one of ``solidus``, ``enthalpy`` and
            ``entropy``; a ``solidus`` and one of ``enthalpy`` and ``entropy``; an ``enthalpy``
            and an ``entropy``.
        CacheModeError: ``write_cache`` resolves on and ``tli.data_dir`` cannot be written.
    """
    write = bool(config.persist_ternary_imputed) if write_cache is None else bool(write_cache)
    write = write and apply
    if write and "write" not in getattr(tli.data_dir, "capabilities", frozenset()):
        raise CacheModeError(
            f"write_cache is on but the ternary store {tli.data_dir!r} cannot be written. Pass "
            f"write_cache=False, or give the ternary object a writable store or an "
            f"OverlayBackend."
        )

    components = list(tli.components)
    records, liquids, givens = [], [], []
    for target in targets:
        spec = target["composition"]
        fractions = normalize_fractions(spec, components)
        given = {k: float(target[k]) for k in ("enthalpy", "entropy") if target.get(k) is not None}
        solidus = target.get("solidus")
        named = sorted(k for k in ("temperature", "solidus") if target.get(k) is not None) + sorted(
            given
        )
        liquid = target.get("liquid_composition")
        if named not in _TARGET_KEYS or (given and liquid is not None):
            raise ValueError(
                f"a target names a temperature; a temperature and one of solidus, enthalpy and "
                f"entropy; a solidus and one of enthalpy and entropy; or an enthalpy and an "
                f"entropy (the last three without a liquid_composition); got {named}"
            )
        temp = None if target.get("temperature") is None else float(target["temperature"])
        if temp is not None and not temp > 0.0:
            raise ValueError(f"temperature must be positive (K), got {temp}")
        liquids.append(None if liquid is None else normalize_fractions(liquid, components))
        givens.append(given)
        ceiling = math.inf if temp is None else temp
        if solidus is not None and (liquid is not None or not 0.0 < float(solidus) < ceiling):
            raise ValueError(
                f"solidus must lie between 0 K and the temperature {temp} K of an envelope "
                f"target, got {solidus}"
            )
        records.append(
            TernaryImputationResult(
                name=str(target.get("name") or _default_name(spec, fractions)),
                fractions=fractions,
                temperature_k=temp,
                solidus_k=None if solidus is None else float(solidus),
                method="envelope" if liquid is None else "tangent",
                replaces=tuple(sorted(str(i) for i in target.get("replaces", ()))),
            )
        )

    targeted = {r.name for r in records}
    current = tli.all_imputed_entries()
    kept = [e for e in current if imputed_entry_label(e) not in targeted]
    replaced = {imputed_entry_label(e) for e in current} & targeted
    excluded = {eid for r in records for eid in r.replaces}
    base = tli.solid_hull(kept, exclude=excluded)
    table = tli.hull_table(kept, exclude=excluded)
    known = set(table["Phase Name"]) - replaced
    stable = _stable_compounds(base, components)
    protected = {eid for eid, _, _ in stable}
    if excluded:
        protected &= {str(e.entry_id) for e in tli.solid_hull(kept).stable_entries}
    names = {str(e.entry_id): api.hull_entry_name(e) for e in base.stable_entries}

    candidates = []
    for record, liquid_fractions, given in zip(records, liquids, givens):
        if record.name in known:
            record.reject("name_taken", f"{record.name!r} already names a phase of this system")
            continue
        covering = [
            (eid, imputed) for eid, fr, imputed in stable if _coincides(fr, record.fractions)
        ]
        if any(not imputed for _, imputed in covering):
            record.reject("dft_covers", "the DFT hull holds a stable phase at this composition")
            continue
        if covering:
            record.reject(
                "occupied", f"the imputed phase {covering[0][0]} sits at this composition"
            )
            continue

        h_form = _invert_ternary_phase_energy(
            tli, record, table, liquid_fractions, margin_j, dx, given
        )
        if h_form is None:
            continue
        record.inverted_enthalpy = h_form
        h_hull = phase_diagram_enthalpy_at(base, record.fractions)
        on_hull = h_form < h_hull - margin_j
        joins = record.solidus_k is not None
        floating = joins or bool(given) and not on_hull
        if given and floating and not joins and not record.entropy > 0.0 and not force:
            record.reject(
                "unstable",
                f"the enthalpy {h_form:.0f} J/mol-atom lies above the solid hull "
                f"({h_hull:.0f} J/mol-atom) at this composition, and without a positive entropy "
                f"the phase never joins it",
            )
            continue
        if joins and on_hull:
            record.reject(
                "no_entropy",
                f"the enthalpy {h_form:.0f} J/mol-atom that joins the hull at "
                f"{record.solidus_k:g} K is already on the 0 K hull",
            )
            continue
        if not floating and not h_form < h_hull - margin_j:
            record.reject(
                "unstable",
                f"the imputed enthalpy {h_form:.0f} J/mol-atom does not clear the solid hull "
                f"({h_hull:.0f} J/mol-atom) at this composition",
            )
            continue

        energy = hull_ref_energy_per_atom(base, record.fractions) + h_form / EV_ATOM_TO_J_MOL
        record.entry_dict = tagged_entry_dict(
            Composition(record.fractions),
            energy,
            record.name,
            {
                "imputed": True,
                "label": record.name,
                "fractions": {k: float(v) for k, v in record.fractions.items()},
                "h_form_j_per_mol": float(h_form),
                **({cache.ENTROPY_KEY: record.entropy} if record.entropy else {}),
                **({REPLACES_KEY: list(record.replaces)} if record.replaces else {}),
                PROVENANCE_KEY: _ternary_provenance(tli, record, liquid_fractions, margin_j, base),
            },
        )
        candidates.append(
            {
                "entry_dict": record.entry_dict,
                "name": record.name,
                "margin_j": h_hull - h_form,
                "floating": floating,
                "given": bool(given),
            }
        )

    ops = _hull_ops(base)
    by_name = {r.name: r for r in records}
    as_given = {c["name"] for c in candidates if c["given"]}

    def displaced_by(entry_dict, admitted) -> tuple[str, ...]:
        stable_now = {
            str(e.entry_id) for e in ops.augmented(admitted + [entry_dict]).stable_entries
        }
        return tuple(sorted(names[eid] for eid in protected - stable_now))

    def held(entry_dict, admitted, protected_ids):
        record = by_name[imputed_entry_label(entry_dict)]
        record.displaces = displaced_by(entry_dict, admitted)
        if not hold_hull or record.support_entropy is None or record.name in as_given:
            return None
        clamped = clamp_bisect(ops, entry_dict, admitted, protected_ids)
        if clamped is None:
            return None
        lift = float(clamped["data"]["h_form_j_per_mol"]) - record.inverted_enthalpy
        entropy = lift / record.temperature_k
        if not entropy < record.support_entropy:
            return None
        clamped["data"][cache.ENTROPY_KEY] = entropy
        return clamped

    accepted, rejected = accept_growing(
        ops,
        [c for c in candidates if not c["floating"]],
        protected,
        protect=not force,
        clamp=held,
        vertex_noun="phase(s)",
    )
    accepted += [c["entry_dict"] for c in candidates if c["floating"]]
    for name, reason in rejected:
        record = by_name[name]
        if not reason.startswith("would displace"):
            record.reject("not_stable", reason)
            continue
        if name in as_given:
            message = (
                f"Warning: placing phase {name} at {record.inverted_enthalpy:.0f} J/mol-atom "
                f"will displace {', '.join(record.displaces)} on the T = 0 K convex hull. If "
                f"you wish to continue, run with force=True."
            )
        else:
            message = (
                f"Warning: imputing phase {name} stable up to {record.temperature_k:g} K will "
                f"displace {', '.join(record.displaces)} on the T = 0 K convex hull. "
            ) + (
                "No entropy below the envelope's reaches that temperature at the enthalpy that "
                "keeps them."
                if hold_hull
                else "If you wish to continue, run with force=True."
            )
        record.reject("displaces_hull", message)
        warnings.warn(message, UserWarning, stacklevel=2)
    for entry_dict in accepted:
        record = by_name[imputed_entry_label(entry_dict)]
        record.entry_dict = entry_dict
        record.enthalpy = float(entry_dict["data"]["h_form_j_per_mol"])
        record.entropy = float(entry_dict["data"].get(cache.ENTROPY_KEY, 0.0))
        if not record.entropy and record.name not in as_given:
            record.displaces = displaced_by(entry_dict, [])
        record.accepted = True
        record.code = "imputed"

    admitted = [r for r in records if r.accepted]
    if apply and admitted:
        tli.imputed_entries = kept + [r.entry_dict for r in admitted]
        tli.reset_derived_state()
        if screen:
            _screen(tli, admitted)
    tli.ternary_meta["imputation_rejected"] = [
        (r.name, r.reason) for r in records if not r.accepted
    ]

    survivors = [r for r in records if r.accepted]
    if write and survivors:
        api.cache_imputed_entries(
            components, [r.entry_dict for r in survivors], DFT_TYPE, data_dir=tli.data_dir
        )
        for record in survivors:
            record.written = True
    for record in records:
        if not record.accepted:
            logger.info(f"skip {record.name}: {record.reason}")
    return records


def impute_ternary_phase(
    tli, composition, temperature: float | None = None, **kwargs
) -> TernaryImputationResult:
    """Impute one ternary phase that leaves the hull at ``temperature`` (K).

    Args:
        tli: the ternary object.
        composition: the phase composition (formula, ``Composition``, element dict, or
            fraction vector in ``tli.components`` order).
        temperature: K; ``None`` for a target that names none.
        **kwargs: ``name``, ``liquid_composition``, ``replaces``, ``solidus``, ``enthalpy`` and
            ``entropy`` describe the target; the rest go to :func:`impute_ternary_phases`.

    Returns:
        TernaryImputationResult: the outcome.
    """
    target = {"composition": composition, "temperature": temperature}
    for key in ("name", "liquid_composition", "replaces", "solidus", "enthalpy", "entropy"):
        if key in kwargs:
            target[key] = kwargs.pop(key)
    return impute_ternary_phases(tli, [target], **kwargs)[0]


def adjust_ternary_phase_temperature(
    tli, phase: str, temperature: float, **kwargs
) -> TernaryImputationResult:
    """Re-impute an existing solid phase so it leaves the hull at ``temperature`` (K).

    The phase keeps its name and composition. A DFT phase is replaced: every DFT entry at its
    composition is left off the hull (``replaces``) and an imputed entry takes its place, so
    dropping that entry (:func:`gliquid.imputation.lifecycle.drop_imputed_phases`) restores
    the DFT phase. An imputed phase is re-imputed in place.

    Args:
        tli: the ternary object.
        phase: the phase name, as it appears on the 0 K hull.
        temperature: K.
        **kwargs: passed to :func:`impute_ternary_phases` (``force``, ``write_cache``,
            ``apply``, ``screen``, ``margin_j``, ``dx``) or, as ``liquid_composition`` and
            ``solidus``, into its target.

    Returns:
        TernaryImputationResult: the outcome, with ``previous_enthalpy`` set.

    Raises:
        ValueError: ``phase`` is not a compound on the 0 K hull.
    """
    target, previous = _retarget(tli, tli.solid_hull(), phase, temperature)
    target.pop("temperature")
    record = impute_ternary_phase(tli, target.pop("composition"), temperature, **target, **kwargs)
    record.previous_enthalpy = previous
    return record


def _hull_compound(hull, phase: str):
    """The stable entry named ``phase`` on ``hull``."""
    matches = [e for e in hull.stable_entries if api.hull_entry_name(e) == phase]
    if not matches:
        names = sorted(api.hull_entry_name(e) for e in hull.stable_entries)
        raise ValueError(f"{phase!r} is not a phase on the 0 K hull; the hull holds {names}")
    if len(matches[0].composition.elements) < 2:
        raise ValueError(f"{phase!r} is an element; its melting point is not imputed")
    return matches[0]


def _dft_ids_at(hull, entry) -> tuple[str, ...]:
    """Ids of the DFT entries of ``hull`` at ``entry``'s composition."""
    reduced = entry.composition.reduced_composition
    return tuple(
        sorted(
            str(e.entry_id)
            for e in hull.all_entries
            if e.composition.reduced_composition == reduced
            and not is_imputed_entry_dict(e.as_dict())
        )
    )


def _retarget(tli, hull, phase: str, temperature: float | None) -> tuple[dict, float]:
    """The imputation target that moves ``phase`` to ``temperature`` (K; ``None`` for a target
    that names none), and the formation enthalpy it has on ``hull``, J/mol-atom."""
    entry = _hull_compound(hull, phase)
    if is_imputed_entry_dict(entry.as_dict()):
        replaces = tuple((entry.data or {}).get(REPLACES_KEY, ()))
    else:
        replaces = _dft_ids_at(hull, entry)
    target = {
        "composition": {c: entry.composition.get_atomic_fraction(c) for c in tli.components},
        "temperature": temperature,
        "name": phase,
        "replaces": replaces,
    }
    return target, float(hull.get_form_energy_per_atom(entry) * EV_ATOM_TO_J_MOL)


def edit_ternary_phases(tli, edits: Sequence[dict], **kwargs) -> list[TernaryImputationResult]:
    """Apply additions, temperature changes and removals of solid phases to ``tli``.

    Removals are applied first, then each addition and temperature change in the order given,
    so each is placed on the hull the ones before it left.

    Args:
        tli: the ternary object; mutated.
        edits: dicts, each one of ``{"remove": phase}`` (every DFT entry at the phase's
            composition is added to ``tli.excluded_entries``; an imputed phase of that name is
            dropped from ``tli.imputed_entries`` or, when the store holds it, added to
            ``tli.excluded_entries`` with the entries it stands in for), ``{"phase": phase,
            "temperature": K}`` (as
            :func:`adjust_ternary_phase_temperature`), or ``{"composition": ...,
            "temperature": K}`` with an optional ``"name"`` (as :func:`impute_ternary_phase`).
            The last two take ``"solidus"`` (K), ``"enthalpy"`` (J/mol-atom) and ``"entropy"``
            (J/mol-atom/K) in the combinations :func:`impute_ternary_phases` takes, with or
            without the ``"temperature"``, and ``"force": True`` to place that phase with
            ``force``.
        **kwargs: passed to :func:`impute_ternary_phases`.

    Returns:
        list[TernaryImputationResult]: one record per addition and temperature change, in the
        order given, with ``previous_enthalpy`` set on the temperature changes.

    Raises:
        ValueError: a removed or changed phase is not a compound on the 0 K hull.
    """
    hull = tli.solid_hull()
    session = {imputed_entry_label(e) for e in tli.imputed_entries}
    for phase in [e["remove"] for e in edits if "remove" in e]:
        if phase in session:
            tli.imputed_entries = [
                e for e in tli.imputed_entries if imputed_entry_label(e) != phase
            ]
            continue
        stored = [e for e in tli.all_imputed_entries() if imputed_entry_label(e) == phase]
        if stored:
            ids = {str(e.get("entry_id")) for e in stored} | replaced_entry_ids(stored)
        else:
            ids = set(_dft_ids_at(hull, _hull_compound(hull, phase)))
        ids = {cache.canonical_entry_id(i) for i in ids} - {None}
        tli.excluded_entries = sorted({*tli.excluded_entries, *ids})
    tli.reset_derived_state()

    records = []
    for edit in edits:
        if "remove" in edit:
            continue
        if "phase" in edit:
            target, before = _retarget(
                tli, tli.solid_hull(), edit["phase"], edit.get("temperature")
            )
        else:
            target = {"composition": edit["composition"], "temperature": edit.get("temperature")}
            before = None
            if edit.get("name"):
                target["name"] = edit["name"]
        for key in ("solidus", "enthalpy", "entropy"):
            if edit.get(key) is not None:
                target[key] = edit[key]
        forced = {"force": True} if edit.get("force") else {}
        record = impute_ternary_phases(tli, [target], **{**kwargs, **forced})[0]
        record.previous_enthalpy = before
        records.append(record)
    return records
