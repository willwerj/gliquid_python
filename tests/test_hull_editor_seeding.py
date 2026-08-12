"""ConvexHullEditor entropy seeding follows compositions, not construction order (S8).

The editor always displays alphabetically (x = fraction of _el_b, the alphabetically
second element). Seeding keyed the PHASE side by construction-order fraction but looked
up by alphabetical fraction — mirrored entropy seeds for reversed-order BinaryLiquids.
Everything here runs offline on a synthetic Hf-Zr hull.
"""

import pytest

ipywidgets = pytest.importorskip("ipywidgets")

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

from gliquid.binary import BinaryLiquid
from gliquid.hull_editor import J_PER_MOL_PER_EV, ConvexHullEditor
from gliquid.phase import UNARY, Phase

S_PHASE = 2.5  # J/mol-atom/K on the HfZr3 line compound


def _hull(elements):
    entries = [
        ComputedEntry(Composition("Hf"), 0.0),
        ComputedEntry(Composition("Zr"), 0.0),
        ComputedEntry(Composition("HfZr3"), -0.4),
    ]
    return PhaseDiagram(entries, elements=[Element(el) for el in elements])


def _bl(sys_name, components):
    phases = [
        Phase(
            phase_type="solid",
            name="HfZr3",
            composition=Composition("HfZr3"),
            enthalpy=-30000.0,
            entropy=S_PHASE,
        ),
        Phase(phase_type="liquid", name="L"),
    ]
    return BinaryLiquid(
        sys_name,
        components,
        component_data=UNARY.component_data(components),
        dft_ch=_hull(components),
        phases=phases,
        temp_range=[300, 3000],
    )


def _seeded_entropy(editor):
    """The editor's seeded entropy (eV/atom/K) on the HfZr3 entry."""
    for entry, s in zip(editor._entries, editor._entropies):
        if entry.composition.reduced_formula == "HfZr3":
            return s
    raise AssertionError("HfZr3 entry not found in editor state")


class TestEntropySeeding:
    def test_alphabetical_bl_seeds_unchanged(self):
        editor = ConvexHullEditor(_bl("Hf-Zr", ["Hf", "Zr"]))
        assert _seeded_entropy(editor) == pytest.approx(S_PHASE / J_PER_MOL_PER_EV)

    def test_reversed_bl_seeds_the_same_composition(self):
        editor = ConvexHullEditor(_bl("Zr-Hf", ["Zr", "Hf"]))
        assert _seeded_entropy(editor) == pytest.approx(S_PHASE / J_PER_MOL_PER_EV)

    def test_apply_round_trip_preserves_phases_both_orders(self):
        for sys_name, components in (("Hf-Zr", ["Hf", "Zr"]), ("Zr-Hf", ["Zr", "Hf"])):
            bl = _bl(sys_name, components)
            editor = ConvexHullEditor(bl)
            editor.apply()
            hfzr3 = next(p for p in bl.phases if p.name == "HfZr3")
            assert hfzr3.entropy == pytest.approx(S_PHASE, rel=1e-9)
            assert hfzr3.fraction_in(components)[0] == pytest.approx(
                Composition("HfZr3").get_atomic_fraction(components[1])
            )
