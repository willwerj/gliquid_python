"""A ternary system's solid entries as a list, and the phase edits that bring a system to one.

:func:`export_entries` lists the entries of a ternary object's 0 K solid hull as
``ComputedEntry`` dicts, imputed and user-defined phases included, each with its entropy under
``data[gliquid.cache.ENTROPY_KEY]``. The list is what ``pymatgen`` reads::

    PhaseDiagram([ComputedEntry.from_dict(d) for d in entries])

:func:`entry_list_edits` compares such a list with a ternary object and returns the edits
(:func:`gliquid.imputation.ternary_imputation.edit_ternary_phases`) that give the object the
list's phases.
"""

from __future__ import annotations

import dataclasses
import json

from monty.json import MontyEncoder
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.cache as cache
from gliquid.imputation.core import phase_diagram_enthalpy_at
from gliquid.phase import EV_ATOM_TO_J_MOL

#: Most, eV/atom, two reference energies of one element may differ and count as the same.
REFERENCE_TOL_EV = 1e-6
#: Most, J/mol-atom and J/mol-atom/K, a listed phase may differ from the object's and count as
#: unchanged.
ENTHALPY_TOL_J = 0.5
ENTROPY_TOL = 1e-6
#: Decimal places compositions are compared to.
COMPOSITION_DECIMALS = 6
#: The module a listed entry names as its class's.
ENTRY_MODULE = "pymatgen.entries.computed_entries"


def _entropy(entry) -> float:
    return float((getattr(entry, "data", None) or {}).get(cache.ENTROPY_KEY, 0.0))


def export_entries(tli) -> list[dict]:
    """The entries of ``tli``'s 0 K solid hull (:meth:`TernaryLiquidInterpolation.solid_hull`)
    as ``ComputedEntry`` payloads of plain JSON types.

    Every entry carries its total energy with its corrections in it (``correction`` 0 and no
    ``energy_adjustments``), and its entropy, J/mol-atom/K, under
    ``data[gliquid.cache.ENTROPY_KEY]`` (0 for an entry without one). Entries an imputed or
    user-defined phase stands in for, and excluded ones, are not listed.
    """
    out = []
    for entry in tli.solid_hull().all_entries:
        original = api.entry_original(entry)
        payload = {
            "@module": ENTRY_MODULE,
            "@class": "ComputedEntry",
            "composition": original.composition.as_dict(),
            "energy": float(original.energy),
            "correction": 0.0,
            "energy_adjustments": [],
            "entry_id": original.entry_id,
            "parameters": getattr(original, "parameters", None) or {},
            "data": {
                **(getattr(original, "data", None) or {}),
                cache.ENTROPY_KEY: _entropy(original),
            },
        }
        out.append(json.loads(json.dumps(payload, cls=MontyEncoder)))
    return out


@dataclasses.dataclass
class EntryListEdits:
    """What a list of entries changes on a ternary object.

    Attributes:
        edits: the edits, as :func:`gliquid.imputation.ternary_imputation.edit_ternary_phases`
            takes them, that give the object the list's phases: a ``{"remove": phase}`` per
            name of ``removed``, then a ``{"phase": phase, "enthalpy", "entropy", "force":
            True}`` or ``{"composition", "name", "enthalpy", "entropy", "force": True}`` per
            name of ``changed`` and of ``added``.
        changed: the object's phases the list gives another formation enthalpy or entropy.
        added: the phases the list holds at compositions where the object has none.
        removed: the object's phases at compositions where the list has no entry.
        unchanged: how many of the object's phases the list leaves as they are.
        ignored: ids of the listed entries that are not the object's own and change nothing:
            those with an element outside the system, and those above the object's 0 K hull,
            without an entropy, at a composition where the object has no phase.
        reference_shifts: element -> the list's reference energy minus the object's, eV/atom,
            for the elements where they differ by more than :data:`REFERENCE_TOL_EV`.
    """

    edits: list[dict] = dataclasses.field(default_factory=list)
    changed: list[str] = dataclasses.field(default_factory=list)
    added: list[str] = dataclasses.field(default_factory=list)
    removed: list[str] = dataclasses.field(default_factory=list)
    unchanged: int = 0
    ignored: list[str] = dataclasses.field(default_factory=list)
    reference_shifts: dict[str, float] = dataclasses.field(default_factory=dict)


def _phases(tli, hull) -> dict[tuple, dict]:
    """The compound phases of ``tli`` by composition: the stable entry of ``hull`` there, or
    the entry of lowest energy that carries an entropy."""
    components = list(tli.components)

    def described(entry, stable):
        fractions = {c: float(entry.composition.get_atomic_fraction(c)) for c in components}
        return {
            "name": api.hull_entry_name(entry),
            "fractions": fractions,
            "enthalpy": float(hull.get_form_energy_per_atom(entry)) * EV_ATOM_TO_J_MOL,
            "entropy": _entropy(entry),
            "stable": stable,
        }

    stable_ids = {id(entry) for entry in hull.stable_entries}
    found: dict[tuple, dict] = {}
    for entry in sorted(hull.all_entries, key=lambda e: e.energy_per_atom):
        stable = id(entry) in stable_ids
        if len(entry.composition.elements) < 2 or not (stable or _entropy(entry) > 0.0):
            continue
        phase = described(entry, stable)
        key = _key(phase["fractions"], components)
        if key not in found or (stable and not found[key]["stable"]):
            found[key] = phase
    return found


def _key(fractions, components) -> tuple:
    return tuple(round(fractions[c], COMPOSITION_DECIMALS) for c in components)


def _label(entry) -> str:
    data = getattr(entry, "data", None) or {}
    if data.get("label"):
        return str(data["label"])
    if cache.non_dft_entry_kind(entry) is not None:
        return cache.non_dft_entry_label(entry)
    return entry.composition.reduced_formula


def entry_list_edits(tli, entry_dicts) -> EntryListEdits:
    """The edits that give ``tli`` the phases of a list of entries.

    The list's phase at a composition is its entry of lowest energy there. Formation enthalpies
    are taken against the list's own reference energies (its elemental entry of lowest energy
    per atom; ``tli``'s for an element the list has none of), so a list on other references
    keeps its formation enthalpies. ``tli`` is not changed.

    Args:
        tli: a :class:`~gliquid.ternary.TernaryLiquidInterpolation`.
        entry_dicts: ``ComputedEntry.as_dict()`` payloads, as :func:`export_entries` lists them.

    Returns:
        EntryListEdits: the edits and what they change.

    Raises:
        ValueError: an item of ``entry_dicts`` is not a ``ComputedEntry`` payload.
    """
    components = list(tli.components)
    hull = tli.solid_hull()
    out = EntryListEdits()

    listed = []
    for n, payload in enumerate(entry_dicts):
        try:
            entry = ComputedEntry.from_dict(payload)
        except Exception as error:
            raise ValueError(f"item {n} of the list is not a ComputedEntry: {error}") from error
        if {str(el) for el in entry.composition.elements} <= set(components):
            listed.append(entry)
        else:
            out.ignored.append(str(entry.entry_id))

    own = {str(el): float(entry.energy_per_atom) for el, entry in hull.el_refs.items()}
    refs = dict(own)
    for symbol in components:
        energies = [
            float(e.energy_per_atom)
            for e in listed
            if [str(el) for el in e.composition.elements] == [symbol]
        ]
        if energies:
            refs[symbol] = min(energies)
    out.reference_shifts = {
        symbol: refs[symbol] - own[symbol]
        for symbol in components
        if abs(refs[symbol] - own[symbol]) > REFERENCE_TOL_EV
    }

    wanted: dict[tuple, dict] = {}
    for entry in sorted(listed, key=lambda e: e.energy_per_atom):
        if len(entry.composition.elements) < 2:
            continue
        fractions = {c: float(entry.composition.get_atomic_fraction(c)) for c in components}
        base = sum(fractions[c] * refs[c] for c in components)
        wanted.setdefault(
            _key(fractions, components),
            {
                "entry": entry,
                "fractions": fractions,
                "enthalpy": (float(entry.energy_per_atom) - base) * EV_ATOM_TO_J_MOL,
                "entropy": _entropy(entry),
            },
        )

    phases = _phases(tli, hull)
    taken = {phase["name"] for phase in phases.values()}
    held = {cache.canonical_entry_id(entry.entry_id) for entry in hull.all_entries}
    for key, phase in phases.items():
        if key not in wanted:
            out.removed.append(phase["name"])
            out.edits.append({"remove": phase["name"]})
    for key, want in wanted.items():
        given = {"enthalpy": want["enthalpy"], "entropy": want["entropy"], "force": True}
        phase = phases.get(key)
        if phase is None:
            above = want["enthalpy"] >= phase_diagram_enthalpy_at(hull, want["fractions"])
            if above and not want["entropy"] > 0.0:
                if cache.canonical_entry_id(want["entry"].entry_id) not in held:
                    out.ignored.append(str(want["entry"].entry_id))
                continue
            name, n = _label(want["entry"]), 1
            while name in taken:
                n += 1
                name = f"{_label(want['entry'])}-{n}"
            taken.add(name)
            out.added.append(name)
            out.edits.append({"composition": want["fractions"], "name": name, **given})
        elif (
            abs(want["enthalpy"] - phase["enthalpy"]) <= ENTHALPY_TOL_J
            and abs(want["entropy"] - phase["entropy"]) <= ENTROPY_TOL
        ):
            out.unchanged += 1
        else:
            out.changed.append(phase["name"])
            target = (
                {"phase": phase["name"]}
                if phase["stable"]
                else {"composition": phase["fractions"], "name": phase["name"]}
            )
            out.edits.append({**target, **given})
    return out
