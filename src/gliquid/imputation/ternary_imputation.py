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
the phases it would displace, and ``force=True`` places it anyway.

:func:`adjust_ternary_phase_temperature` moves the temperature an EXISTING phase leaves
the hull at: a DFT phase is replaced by an imputed one of the same name (its DFT entries
are left off every hull that places the imputed one, and dropping the imputed phase
restores them), and an imputed phase is re-imputed.

Typical use::

    ti = TernaryLiquidInterpolation(["La", "Mn", "Si"], xs_mix=..., include_imputed=True)
    result = ti.impute_phase("LaMnSi2", 1473.0)
    result.accepted, result.enthalpy, result.realized_temperature_k
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Sequence

import numpy as np
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.config as config
from gliquid.cache import CacheModeError, imputed_entry_label, is_imputed_entry_dict
from gliquid.imputation.core import (
    HULL_GUARD_MARGIN_J,
    PROVENANCE_KEY,
    REPLACES_KEY,
    HullOps,
    accept_growing,
    augmented_hull,
    entry_at_enthalpy,
    entry_fractions,
    envelope_energy,
    hull_ref_energy_per_atom,
    independent_coords,
    normalize_fractions,
    phase_diagram_enthalpy_at,
    protected_clearance,
    provenance_header,
    tagged_entry_dict,
    tangent_value,
)
from gliquid.phase import EV_ATOM_TO_J_MOL

logger = logging.getLogger(__name__)

CALIBRATIONS = {"envelope": "ternary_envelope", "tangent": "ternary_tangent"}
LIQUID_PHASE = "L"
DFT_TYPE = api.DEFAULT_DFT_TYPE
COMPOSITION_TOL = 1e-4  # largest fraction difference at which two compositions coincide
MIN_SUPPORT_ENTROPY = 1e-9  # J/mol-atom/K; an envelope with less cannot move with temperature
HSX_TOLERANCE_K = 1.0  # realized-temperature window in hsx mode; gtx uses its T_incr


@dataclasses.dataclass
class TernaryImputationResult:
    """The outcome of imputing one ternary phase.

    Attributes:
        name: the phase label, also the ``entry_id`` suffix.
        fractions: atomic fraction per component.
        temperature_k: the temperature the phase was asked to leave the hull at.
        method: ``'envelope'`` or ``'tangent'``.
        accepted: the phase joined the ternary object (or would have, with ``apply=False``).
        code: when accepted, ``'imputed'`` or ``'off_target'`` (the processed diagram shows
            the phase leaving the hull outside the temperature window); when rejected,
            ``'name_taken'``, ``'dft_covers'``, ``'occupied'``, ``'below_solidus'``,
            ``'unstable'``, ``'not_stable'``, ``'displaces_hull'`` or ``'unrealized'``.
        reason: a sentence describing ``code``.
        enthalpy: the formation enthalpy the phase carries, J/mol-atom.
        inverted_enthalpy: the enthalpy the inversion produced, J/mol-atom.
        displaces: names of the stable phases this phase pushes off the 0 K hull. Non-empty
            on an accepted phase only with ``force=True``.
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
    temperature_k: float
    method: str
    accepted: bool = False
    code: str = ""
    reason: str = ""
    enthalpy: float | None = None
    inverted_enthalpy: float | None = None
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
            "temp": float(record.temperature_k),
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
    tli, record, table, liquid_fractions, margin_j, dx
) -> float | None:
    """The inverted enthalpy for ``record``, or ``None`` after rejecting it."""
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
    if (env.entropy or 0.0) <= MIN_SUPPORT_ENTROPY:
        record.reject(
            "below_solidus",
            f"at {record.temperature_k:g} K every phase around this composition carries zero "
            f"entropy, so no enthalpy makes the phase leave the hull at that temperature",
        )
        return None
    return float(env.value) - margin_j


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
    write_cache: bool | None = None,
    apply: bool = True,
    screen: bool = True,
    margin_j: float = HULL_GUARD_MARGIN_J,
    dx: float = 1e-3,
) -> list[TernaryImputationResult]:
    """Impute ternary solid phases from their compositions and the temperatures they leave the
    hull at.

    Each target's enthalpy is inverted (envelope, or tangent when it names a
    ``liquid_composition``) and must clear the 0 K solid hull. The survivors are admitted
    strongest-first. A phase that would push a stable phase -- DFT or imputed -- off the 0 K
    hull is rejected with a warning naming those phases, unless ``force`` is set. With
    ``apply``, admitted phases replace any imputed phase of the same name in
    ``tli.imputed_entries`` and ``tli``'s derived state is reset, so the next
    ``process_data`` rebuilds with them. Rejections are also listed in
    ``tli.ternary_meta['imputation_rejected']``.

    Args:
        tli: a :class:`~gliquid.ternary.TernaryLiquidInterpolation`; mutated when ``apply``.
        targets: dicts with ``composition`` (a formula, ``Composition``, element dict, or
            fraction vector in ``tli.components`` order) and ``temperature`` (K), and
            optionally ``name`` (default: the reduced formula), ``liquid_composition``, and
            ``replaces`` (DFT entry ids the phase stands in for, left off the hull).
        force: place a phase even when it pushes stable phases off the 0 K hull.
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
        ValueError: a composition is not one of ``tli``'s components' compositions, or a
            temperature is not positive.
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
    records, liquids = [], []
    for target in targets:
        spec = target["composition"]
        fractions = normalize_fractions(spec, components)
        temp = float(target["temperature"])
        if not temp > 0.0:
            raise ValueError(f"temperature must be positive (K), got {temp}")
        liquid = target.get("liquid_composition")
        liquids.append(None if liquid is None else normalize_fractions(liquid, components))
        records.append(
            TernaryImputationResult(
                name=str(target.get("name") or _default_name(spec, fractions)),
                fractions=fractions,
                temperature_k=temp,
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
    names = {str(e.entry_id): api.hull_entry_name(e) for e in base.stable_entries}

    candidates = []
    for record, liquid_fractions in zip(records, liquids):
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

        h_form = _invert_ternary_phase_energy(tli, record, table, liquid_fractions, margin_j, dx)
        if h_form is None:
            continue
        record.inverted_enthalpy = h_form
        h_hull = phase_diagram_enthalpy_at(base, record.fractions)
        if not h_form < h_hull - margin_j:
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
                **({REPLACES_KEY: list(record.replaces)} if record.replaces else {}),
                PROVENANCE_KEY: _ternary_provenance(tli, record, liquid_fractions, margin_j, base),
            },
        )
        candidates.append(
            {"entry_dict": record.entry_dict, "name": record.name, "margin_j": h_hull - h_form}
        )

    ops = _hull_ops(base)
    by_name = {r.name: r for r in records}

    def displaced_by(entry_dict, admitted) -> tuple[str, ...]:
        stable_now = {
            str(e.entry_id) for e in ops.augmented(admitted + [entry_dict]).stable_entries
        }
        return tuple(sorted(names[eid] for eid in protected - stable_now))

    def refuse(entry_dict, admitted, _protected):
        by_name[imputed_entry_label(entry_dict)].displaces = displaced_by(entry_dict, admitted)
        return None

    accepted, rejected = accept_growing(
        ops, candidates, protected, protect=not force, clamp=refuse, vertex_noun="phase(s)"
    )
    for name, reason in rejected:
        record = by_name[name]
        if not reason.startswith("would displace"):
            record.reject("not_stable", reason)
            continue
        message = (
            f"Warning: imputing phase {name} stable up to {record.temperature_k:g} K will "
            f"displace {', '.join(record.displaces)} on the T = 0 K convex hull. If you wish "
            f"to continue, run with force=True."
        )
        record.reject("displaces_hull", message)
        warnings.warn(message, UserWarning, stacklevel=2)
    for entry_dict in accepted:
        record = by_name[imputed_entry_label(entry_dict)]
        record.entry_dict = entry_dict
        record.enthalpy = float(entry_dict["data"]["h_form_j_per_mol"])
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


def impute_ternary_phase(tli, composition, temperature: float, **kwargs) -> TernaryImputationResult:
    """Impute one ternary phase that leaves the hull at ``temperature`` (K).

    Args:
        tli: the ternary object.
        composition: the phase composition (formula, ``Composition``, element dict, or
            fraction vector in ``tli.components`` order).
        temperature: K.
        **kwargs: ``name``, ``liquid_composition`` and ``replaces`` describe the target; the
            rest go to :func:`impute_ternary_phases`.

    Returns:
        TernaryImputationResult: the outcome.
    """
    target = {"composition": composition, "temperature": temperature}
    for key in ("name", "liquid_composition", "replaces"):
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
            ``apply``, ``screen``, ``margin_j``, ``dx``) or as ``liquid_composition``.

    Returns:
        TernaryImputationResult: the outcome, with ``previous_enthalpy`` set.

    Raises:
        ValueError: ``phase`` is not a compound on the 0 K hull.
    """
    components = list(tli.components)
    hull = tli.solid_hull()
    matches = [e for e in hull.stable_entries if api.hull_entry_name(e) == phase]
    if not matches:
        names = sorted(api.hull_entry_name(e) for e in hull.stable_entries)
        raise ValueError(f"{phase!r} is not a phase on the 0 K hull; the hull holds {names}")
    entry = matches[0]
    if len(entry.composition.elements) < 2:
        raise ValueError(f"{phase!r} is an element; its melting point is not imputed")
    fractions = {c: entry.composition.get_atomic_fraction(c) for c in components}
    if is_imputed_entry_dict(entry.as_dict()):
        replaces = tuple((entry.data or {}).get(REPLACES_KEY, ()))
    else:
        reduced = entry.composition.reduced_composition
        replaces = tuple(
            sorted(
                str(e.entry_id)
                for e in hull.all_entries
                if e.composition.reduced_composition == reduced
                and not is_imputed_entry_dict(e.as_dict())
            )
        )
    previous = hull.get_form_energy_per_atom(entry) * EV_ATOM_TO_J_MOL
    record = impute_ternary_phase(
        tli, fractions, temperature, name=phase, replaces=replaces, **kwargs
    )
    record.previous_enthalpy = float(previous)
    return record
