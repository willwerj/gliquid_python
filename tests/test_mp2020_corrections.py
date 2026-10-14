"""Client-side MP2020 corrections in the GGA fetch path.

``gliquid.api.apply_mp2020_corrections`` recomputes ``correction`` / ``energy_adjustments`` with
oxidation states guessed from each composition, which is the convention of the shipped binary
entry corpus. The fixtures are the Ga-Ru-Si ternary entries as the Materials Project served them,
with the structures removed (``ga_ru_si_server_convention.json``: no correction on any Ru-Si
compound), and the entry id, correction and adjustment names a composition guess gives each of
them (``ga_ru_si_composition_guess.json``). Ru is more electronegative than Si, so the served
entries carry no Si anion correction on Ru-Si compounds while the composition guess assigns
0.071 eV per Si atom; Ga-Si and Ga-Ru agree in both.
"""

import copy
import json
from pathlib import Path

import pytest
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

import gliquid.api as api
from gliquid.cache import IMPUTED_ID_PREFIX, USER_ID_PREFIX, canonical_entry_id

from ._mpds_builders import install_fake_mprester

FIXTURES = Path(__file__).parent / "fixtures" / "mp2020"
SI_PER_ATOM_EV = 0.071


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def served():
    return _load("ga_ru_si_server_convention.json")


@pytest.fixture(scope="module")
def guessed():
    return _load("ga_ru_si_composition_guess.json")


def _elements(entry):
    return {el for el, n in entry["composition"].items() if n}


def _by_id(entries):
    return {canonical_entry_id(e["entry_id"]): e for e in entries}


class TestRuSiPositiveControl:
    def test_served_ru_si_compounds_carry_no_correction(self, served):
        ru_si = [e for e in served if _elements(e) == {"Ru", "Si"}]
        assert len(ru_si) == 13
        assert all(e["correction"] == 0.0 and not e["energy_adjustments"] for e in ru_si)

    def test_ru2si3_gains_the_si_anion_correction(self, served):
        corrected = _by_id(api.apply_mp2020_corrections(served))["mp-22192-GGA"]
        n_atoms = sum(corrected["composition"].values())
        per_atom = corrected["correction"] / n_atoms
        assert per_atom == pytest.approx(SI_PER_ATOM_EV * 0.6, abs=1e-9)
        assert per_atom * 96485.33212 == pytest.approx(4110.3, abs=0.1)
        assert [a["name"] for a in corrected["energy_adjustments"]] == [
            "MP2020 anion correction (Si)"
        ]

    def test_ten_of_thirteen_ru_si_compounds_are_corrected_by_their_si_content(self, served):
        corrected = [
            e
            for e in api.apply_mp2020_corrections(served)
            if _elements(e) == {"Ru", "Si"} and e["correction"]
        ]
        assert len(corrected) == 10
        for e in corrected:
            assert e["correction"] == pytest.approx(SI_PER_ATOM_EV * e["composition"]["Si"])

    def test_reproduces_the_composition_guess_fixture_entry_for_entry(self, served, guessed):
        out = api.apply_mp2020_corrections(served)
        assert len(out) == len(guessed)
        for got, want in zip(out, guessed):
            assert canonical_entry_id(got["entry_id"]) == canonical_entry_id(want["entry_id"])
            assert got["correction"] == pytest.approx(want["correction"], abs=1e-12)
            assert [a["name"] for a in got["energy_adjustments"]] == [
                a["name"] for a in want["energy_adjustments"]
            ]


class TestNegativeControl:
    def test_ga_si_and_ga_ru_corrections_do_not_move(self, served):
        before = _by_id(served)
        after = _by_id(api.apply_mp2020_corrections(served))
        checked = 0
        for cid, entry in before.items():
            if _elements(entry) in ({"Ga", "Si"}, {"Ga", "Ru"}):
                assert after[cid]["correction"] == pytest.approx(entry["correction"], abs=1e-12)
                checked += 1
        assert checked >= 9


class TestContract:
    def test_only_the_two_correction_keys_change(self, served):
        out = api.apply_mp2020_corrections(served)
        for before, after in zip(served, out):
            assert set(before) == set(after)
            for key in set(before) - {"correction", "energy_adjustments"}:
                assert after[key] == before[key]

    def test_input_is_not_mutated(self, served):
        snapshot = copy.deepcopy(served)
        api.apply_mp2020_corrections(served)
        assert served == snapshot

    def test_attached_oxidation_states_are_ignored(self, served):
        entry = copy.deepcopy(_by_id(served)["mp-22192-GGA"])
        want = _by_id(api.apply_mp2020_corrections([entry]))["mp-22192-GGA"]["correction"]
        for states in ({}, {"Ru": -1.0, "Si": 1.0}, {"Ru": 4.0, "Si": -2.0}):
            tagged = {**entry, "data": {"oxidation_states": states}}
            got = api.apply_mp2020_corrections([tagged])[0]
            assert got["correction"] == pytest.approx(want)
            assert got["data"] == {"oxidation_states": states}

    def test_an_entry_mp2020_rejects_is_returned_unchanged(self, served):
        entry = copy.deepcopy(_by_id(served)["mp-22192-GGA"])
        entry["parameters"] = {**entry["parameters"], "run_type": "R2SCAN"}
        assert api.apply_mp2020_corrections([entry]) == [entry]

    def test_an_entry_without_a_run_type_is_returned_unchanged(self):
        from pymatgen.entries.computed_entries import ComputedEntry

        entry = ComputedEntry(Composition("Cu2Mg"), -0.9, entry_id="s-CuMg2").as_dict()
        assert api.apply_mp2020_corrections([entry]) == [entry]

    @pytest.mark.parametrize("entry_id", [f"{IMPUTED_ID_PREFIX}Si3Ru2", f"{USER_ID_PREFIX}Si3Ru2"])
    def test_non_dft_entries_are_returned_unchanged(self, served, entry_id):
        entry = copy.deepcopy(_by_id(served)["mp-22192-GGA"])
        entry["entry_id"] = entry_id
        assert api.apply_mp2020_corrections([entry]) == [entry]

    def test_an_oxide_is_classified_from_its_structure(self):
        # Antifluorite Li2O: O on the fcc sites, Li on every (1/4 or 3/4)^3 site.
        o_sites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        li_sites = [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
        li2o = Structure(Lattice.cubic(4.62), ["O"] * 4 + ["Li"] * 8, o_sites + li_sites)
        entry = ComputedStructureEntry(
            li2o, -14.0, parameters={"run_type": "GGA", "potcar_spec": []}, entry_id="s-Li2O"
        ).as_dict()
        out = api.apply_mp2020_corrections([entry])[0]
        assert [a["name"] for a in out["energy_adjustments"]] == ["MP2020 anion correction (oxide)"]
        assert out["correction"] == pytest.approx(-0.687 * entry["composition"]["O"])
        assert out["structure"] == entry["structure"]


class TestFetchPath:
    @pytest.fixture
    def ru_si(self, served):
        return [e for e in served if _elements(e) <= {"Ru", "Si"}]

    def _install(self, monkeypatch, entries):
        install_fake_mprester(monkeypatch, lambda elements, criteria: copy.deepcopy(entries))

    def test_gga_fetch_returns_composition_guess_corrections(self, monkeypatch, ru_si):
        self._install(monkeypatch, ru_si)
        out = _by_id(api._get_dft_entries_from_components(["Ru", "Si"], "GGA"))
        entry = out["mp-22192-GGA"]
        assert entry["correction"] == pytest.approx(SI_PER_ATOM_EV * entry["composition"]["Si"])

    def test_r2scan_fetch_leaves_the_served_corrections(self, monkeypatch, ru_si):
        self._install(monkeypatch, ru_si)
        out = _by_id(api._get_dft_entries_from_components(["Ru", "Si"], "R2SCAN"))
        assert out["mp-22192-GGA"]["correction"] == 0.0

    def test_mixed_fetch_hands_the_scheme_the_served_entries(self, monkeypatch, ru_si):
        import sys
        import types

        seen = {}

        class _Scheme:
            def process_entries(self, entries, verbose=False):
                seen["corrections"] = {
                    canonical_entry_id(e["entry_id"]): e["correction"] for e in entries
                }
                return entries

        mod = types.ModuleType("pymatgen.entries.mixing_scheme")
        mod.MaterialsProjectDFTMixingScheme = _Scheme
        monkeypatch.setitem(sys.modules, "pymatgen.entries.mixing_scheme", mod)
        self._install(monkeypatch, ru_si)
        out = _by_id(api._get_dft_entries_from_components(["Ru", "Si"], "MIXED"))
        assert seen["corrections"]["mp-22192-GGA"] == 0.0
        assert out["mp-22192-GGA"]["correction"] == 0.0

    def test_keep_data_retains_the_served_data_block(self, monkeypatch, ru_si):
        tagged = [{**e, "data": {"oxide_type": "None"}} for e in ru_si]
        self._install(monkeypatch, tagged)
        kept = api._get_dft_entries_from_components(["Ru", "Si"], "GGA", keep_data=True)
        assert all(e["data"] == {"oxide_type": "None"} for e in kept)
        dropped = api._get_dft_entries_from_components(["Ru", "Si"], "GGA")
        assert all("data" not in e for e in dropped)
