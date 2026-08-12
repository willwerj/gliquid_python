"""PD/CPD seam helpers (S6): one sanctioned way to read hulls, both flavors.

``pd_components`` / ``entry_original`` / ``entry_display_name`` / ``entry_frac_along``
must be exact drop-ins for the historical elemental idioms (the hull goldens gate that)
AND already give correct answers on a CompoundPhaseDiagram, so pseudo-binary consumers
need no rewrites later.
"""

import pytest
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api


@pytest.fixture(scope="module")
def elemental_pd():
    entries = [
        ComputedEntry(Composition("Hf"), 0.0),
        ComputedEntry(Composition("Zr"), 0.0),
        ComputedEntry(Composition("HfZr3"), -0.4),
    ]
    # reversed construction order on purpose: the axes follow the caller
    return PhaseDiagram(entries, elements=[Element("Zr"), Element("Hf")])


@pytest.fixture(scope="module")
def compound_pd():
    entries = [
        ComputedEntry(Composition("Cu"), 0.0),
        ComputedEntry(Composition("Mg"), 0.0),
        ComputedEntry(Composition("CuMg"), -0.8),
        ComputedEntry(Composition("CuMg2"), -0.9),
    ]
    return CompoundPhaseDiagram(entries, [Composition("CuMg"), Composition("Mg")])


class TestPdComponents:
    def test_elemental_follows_hull_order(self, elemental_pd):
        assert api.pd_components(elemental_pd) == ["Zr", "Hf"]

    def test_compound_gives_terminal_formulas(self, compound_pd):
        comps = api.pd_components(compound_pd)
        assert len(comps) == 2
        # pymatgen-normalized strings — identify via Composition equality
        assert Composition(comps[0]) == Composition("CuMg")
        assert Composition(comps[1]) == Composition("Mg")


class TestEntryIdentity:
    def test_elemental_original_is_self(self, elemental_pd):
        e = next(iter(elemental_pd.stable_entries))
        assert api.entry_original(e) is e
        assert api.entry_display_name(e) == e.composition.reduced_formula

    def test_compound_original_and_name(self, compound_pd):
        by_name = {api.entry_display_name(e): e for e in compound_pd.stable_entries}
        assert {"MgCu", "Mg"} <= set(by_name)
        cumg = by_name["MgCu"]
        original = api.entry_original(cumg)
        assert original is not cumg
        assert Composition(original.composition.reduced_formula) == Composition("CuMg")


class TestEntryFracAlong:
    def test_elemental_matches_get_atomic_fraction(self, elemental_pd):
        for e in elemental_pd.stable_entries:
            frac = api.entry_frac_along(elemental_pd, e, ["Zr", "Hf"])
            assert frac == (e.composition.get_atomic_fraction("Hf"),)
        hfzr3 = next(
            e for e in elemental_pd.stable_entries if e.composition.reduced_formula == "HfZr3"
        )
        assert api.entry_frac_along(elemental_pd, hfzr3, ["Zr", "Hf"])[0] == pytest.approx(0.25)
        assert api.entry_frac_along(elemental_pd, hfzr3, ["Hf", "Zr"])[0] == pytest.approx(0.75)

    def test_compound_pseudo_fractions(self, compound_pd):
        by_name = {api.entry_display_name(e): e for e in compound_pd.stable_entries}
        # terminals sit at the ends of the pseudo-binary axis
        assert api.entry_frac_along(compound_pd, by_name["MgCu"])[0] == pytest.approx(0.0)
        assert api.entry_frac_along(compound_pd, by_name["Mg"])[0] == pytest.approx(1.0)
        # CuMg2 = 2 atoms of (Cu0.5Mg0.5) + 1 atom of Mg on the normalized-terminal
        # basis -> x_Mg = 1/3 (pymatgen's atom-normalized convention, NOT 1/2)
        if "Mg2Cu" in by_name:
            assert api.entry_frac_along(compound_pd, by_name["Mg2Cu"])[0] == pytest.approx(1 / 3)
