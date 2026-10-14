"""The names the imputed-entry naming helpers give five entry shapes.

The rows are the dict helper ``gliquid.cache.imputed_entry_label`` (keyed ``entry_label``),
the hull-entry helper ``api.hull_entry_name``, and the name ``binary.build_phases_from_chull``
gives the phase it builds for the entry.

Four shapes are one imputed Cu-Mg entry at x_Mg = 0.5 and -30000 J/mol-atom on the
element-only Cu-Mg hull, differing only in their ``data`` block: ``labelled``, ``label_null``,
``no_label`` and ``data_none``. The fifth, ``non_imputed``, is the real MgCu2 entry of the full
Cu-Mg hull. Each cell is the helper's value, or the type of the exception it raises.

The last test checks, on the cached Cu-Mg and Al-Cu hulls (full, and stripped of their
interior compounds then imputed), that ``hull_entry_name`` gives every stable entry the name
``build_phases_from_chull`` gives its phase.
"""

from __future__ import annotations

import copy

import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.core as imp_core
from gliquid.binary import BinaryLiquid, build_phases_from_chull
from gliquid.cache import imputed_entry_label
from gliquid.phase import EV_ATOM_TO_J_MOL

from ._mpds_builders import strip_interior

# The helpers under test, one line each.
HELPERS = {
    "entry_label": imputed_entry_label,
    "hull_entry_name": api.hull_entry_name,
}
DICT_HELPER = "entry_label"
HULL_ENTRY_HELPER = "hull_entry_name"
INLINE = "build_phases_from_chull"
IMPUTE = imp_binary.impute_binary_phases
HULL_POINTS = imp_binary._hull_points
TAGGED_ENTRY_DICT = imp_core.tagged_entry_dict
HULL_REF_ENERGY = imp_core.hull_ref_energy_per_atom
EV = EV_ATOM_TO_J_MOL

# The liquid parameters test_imputation_bit_identity.py pins.
PARAMS = {
    "Cu-Mg": [-41302.26, -5.1627, -8148.05, 0.0],
    "Al-Cu": [-68087.2, -3.6819, 2945.42, 0.0],
}
PROBE_NAME = "Probe"
PROBE_H_FORM_J = -30000.0
REAL_FORMULA = "MgCu2"

SHAPE_DATA = {
    "labelled": {"imputed": True, "label": "Probe"},
    "label_null": {"imputed": True, "label": None},
    "no_label": {"imputed": True},
    "data_none": None,
}

# (helper, shape) -> ("value", name) or ("raises", exception type name)
CELLS = [
    ("entry_label", "labelled", ("value", "Probe")),
    ("entry_label", "label_null", ("value", "Probe")),
    ("entry_label", "no_label", ("value", "Probe")),
    ("entry_label", "data_none", ("value", "Probe")),
    ("entry_label", "non_imputed", ("value", "mp-1038-GGA")),
    ("hull_entry_name", "labelled", ("value", "Probe")),
    ("hull_entry_name", "label_null", ("value", "Probe")),
    ("hull_entry_name", "no_label", ("value", "Probe")),
    ("hull_entry_name", "data_none", ("value", "Mg0.5Cu0.5")),
    ("hull_entry_name", "non_imputed", ("value", "MgCu2")),
    ("build_phases_from_chull", "labelled", ("value", "Probe")),
    ("build_phases_from_chull", "label_null", ("value", "Probe")),
    ("build_phases_from_chull", "no_label", ("value", "Probe")),
    ("build_phases_from_chull", "data_none", ("value", "Mg0.5Cu0.5")),
    ("build_phases_from_chull", "non_imputed", ("value", "MgCu2")),
]


def load(system: str) -> BinaryLiquid:
    """The shipped system at its pinned liquid parameters."""
    return BinaryLiquid.from_cache(system, param_format="comb-exp", params=list(PARAMS[system]))


@pytest.fixture(scope="module")
def cu_mg() -> BinaryLiquid:
    return load("Cu-Mg")


def _shape(cu_mg: BinaryLiquid, shape: str):
    """``(entry_dict, hull, hull_entry)`` for one shape."""
    components = list(cu_mg.components)
    if shape == "non_imputed":
        real = next(
            e for e in cu_mg.dft_ch.stable_entries if e.composition.reduced_formula == REAL_FORMULA
        )
        return real.as_dict(), cu_mg.dft_ch, real

    elemental = [e for e in cu_mg.dft_ch.all_entries if len(e.composition.elements) == 1]
    elements = list(cu_mg.dft_ch.elements)
    fractions = {components[0]: 0.5, components[1]: 0.5}
    energy = (
        HULL_REF_ENERGY(PhaseDiagram(elemental, elements=elements), fractions) + PROBE_H_FORM_J / EV
    )
    entry_dict = TAGGED_ENTRY_DICT(
        Composition(fractions), energy, PROBE_NAME, {"imputed": True, "label": PROBE_NAME}
    )
    entry_dict["data"] = copy.deepcopy(SHAPE_DATA[shape])
    entry = ComputedEntry.from_dict(entry_dict)
    if shape == "data_none":
        entry.data = None
    hull = PhaseDiagram(elemental + [entry], elements=elements)
    hull_entry = next(e for e in hull.stable_entries if e.entry_id == entry_dict["entry_id"])
    return entry_dict, hull, hull_entry


def _inline_name(cu_mg: BinaryLiquid, hull, hull_entry) -> str | None:
    components = list(cu_mg.components)
    x = api.entry_frac_along(hull, hull_entry, components)[0]
    phases = build_phases_from_chull(hull, components, cu_mg.component_data)
    return next(
        p.name for p in phases if p.composition is not None and p.fraction_in(components)[0] == x
    )


def _cell(thunk) -> tuple:
    try:
        return ("value", thunk())
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__)


@pytest.mark.parametrize(
    ("helper", "shape", "expected"), CELLS, ids=[f"{h}-{s}" for h, s, _e in CELLS]
)
def test_label_cell(cu_mg, helper, shape, expected):
    """One helper on one entry shape gives the pinned name, or raises the pinned type."""
    entry_dict, hull, hull_entry = _shape(cu_mg, shape)
    if helper == DICT_HELPER:
        got = _cell(lambda: HELPERS[helper](copy.deepcopy(entry_dict)))
    elif helper == HULL_ENTRY_HELPER:
        got = _cell(lambda: HELPERS[helper](hull_entry))
    else:
        got = _cell(lambda: _inline_name(cu_mg, hull, hull_entry))
    assert got == expected


def test_the_probe_is_a_stable_vertex_of_the_element_only_hull(cu_mg):
    """The imputed shapes sit on the hull, so the inline cells name a built phase."""
    for shape in SHAPE_DATA:
        _entry_dict, hull, hull_entry = _shape(cu_mg, shape)
        assert hull_entry in hull.stable_entries
        assert len(hull.stable_entries) == 3


def _naming_pairs(bl: BinaryLiquid) -> list[tuple[str, str, str]]:
    """``(entry_id, hull_entry_name, built phase name)`` for every stable entry with a phase."""
    components = list(bl.components)
    phases = build_phases_from_chull(bl.dft_ch, components, bl.component_data)
    ladder = {p.name for c in components for p in bl.component_data[c].polymorphs}
    ladder_x = {p.fraction_in(components)[0] for p in phases if p.name in ladder}
    built = {
        p.fraction_in(components)[0]: p
        for p in phases
        if p.composition is not None and p.name not in ladder
    }
    pairs = []
    for entry in bl.dft_ch.stable_entries:
        x = api.entry_frac_along(bl.dft_ch, entry, components)[0]
        if x in ladder_x:
            continue
        pairs.append((str(entry.entry_id), HELPERS[HULL_ENTRY_HELPER](entry), built[x].name))
    return pairs


@pytest.fixture(scope="module")
def naming_systems():
    out = {}
    for system in PARAMS:
        full = load(system)
        work = strip_interior(copy.deepcopy(full))
        records = IMPUTE(work, write_cache=False)
        out[(system, "full")] = full
        out[(system, "stripped_then_imputed")] = work
        out[(system, "n_imputed")] = len(records)
    return out


@pytest.mark.parametrize("state", ["full", "stripped_then_imputed"])
@pytest.mark.parametrize("system", list(PARAMS))
def test_hull_entry_name_matches_the_inline_naming_on_cached_entries(naming_systems, system, state):
    """Every stable entry with a built phase gets the same name from both routes."""
    bl = naming_systems[(system, state)]
    pairs = _naming_pairs(bl)
    assert pairs, "no stable entry with a built phase; the comparison would be vacuous"
    if state == "stripped_then_imputed":
        assert naming_systems[(system, "n_imputed")] > 0
        assert any(entry_id.startswith("imputed:") for entry_id, _a, _b in pairs)
    assert [(i, a) for i, a, _b in pairs] == [(i, b) for i, _a, b in pairs]
