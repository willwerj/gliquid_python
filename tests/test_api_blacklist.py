"""Tests for the Tier A spurious-structure blacklist in gliquid.api.

All offline. The blacklist (``config.spurious_structures_file``) must remove
blacklisted ELEMENTAL structures from both cache-read paths without rewriting
the cache file, matching by classic mp-id string, by dict-form entry ids, and —
for new-generation alpha ids that carry no classic id — by the (element,
spacegroup) of the entry's structure.
"""

import json

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

import gliquid.api as api
import gliquid.config as config


def _fcc_ag(energy_per_atom):
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.09), ["Ag"], [[0, 0, 0]])
    return ComputedStructureEntry(s, energy_per_atom * len(s), entry_id="mp-124-GGA")


def _hcp_ag(energy_per_atom, entry_id):
    s = Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(2.9, 4.7), ["Ag"], [[1 / 3, 2 / 3, 3 / 4]]
    )
    e = ComputedStructureEntry(s, energy_per_atom * len(s))
    d = e.as_dict()
    d["entry_id"] = entry_id
    return d


def _fcc_au(energy_per_atom):
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.08), ["Au"], [[0, 0, 0]])
    return ComputedStructureEntry(s, energy_per_atom * len(s), entry_id="mp-81-GGA")


@pytest.fixture()
def blacklist_file(tmp_path, monkeypatch):
    path = tmp_path / "spurious_structures.json"
    path.write_text(
        json.dumps(
            {
                "elements": {
                    "Ag": [
                        {
                            "material_id": "mp-8566",
                            "spacegroup_number": 194,
                            "reason": "test",
                            "source": "test",
                            "date": "2026-08-04",
                        }
                    ]
                }
            }
        )
    )
    monkeypatch.setattr(config, "spurious_structures_file", path)
    monkeypatch.setattr(api, "_spurious_cache", None)
    return path


def _write_cache(tmp_path, entries):
    cache = tmp_path / "Ag-Au_ENTRIES_MP_GGA.json"
    cache.write_text(json.dumps([e if isinstance(e, dict) else e.as_dict() for e in entries]))
    return cache


class TestSpuriousMatcher:
    def test_classic_string_id_with_suffix(self, blacklist_file):
        assert api._is_spurious_entry_dict({"composition": {"Ag": 4.0}, "entry_id": "mp-8566-GGA"})

    def test_dict_form_classic_id(self, blacklist_file):
        assert api._is_spurious_entry_dict(
            {
                "composition": {"Ag": 4.0},
                "entry_id": {"identifier": "mp-8566", "suffix": "GGA", "separator": "-"},
            }
        )

    def test_alpha_id_falls_back_to_structure_spacegroup(self, blacklist_file):
        hcp = _hcp_ag(-2.7, {"identifier": "mp-aaaazzzz", "suffix": "GGA", "separator": "-"})
        assert api._is_spurious_entry_dict(hcp)

    def test_id_prefix_must_match_exactly(self, blacklist_file):
        # mp-85 blacklisted for nobody: mp-8566 must not match an mp-85 entry or vice versa
        assert not api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0}, "entry_id": "mp-856-GGA"}
        )

    def test_compound_entries_never_match(self, blacklist_file):
        assert not api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0, "Au": 1.0}, "entry_id": "mp-8566-GGA"}
        )

    def test_no_blacklist_file_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "spurious_structures_file", tmp_path / "missing.json")
        monkeypatch.setattr(api, "_spurious_cache", None)
        assert not api._is_spurious_entry_dict(
            {"composition": {"Ag": 4.0}, "entry_id": "mp-8566-GGA"}
        )


class TestReadTimeGuards:
    def test_convexhull_drops_blacklisted_elemental_ref(self, tmp_path, blacklist_file):
        # 4H-Ag below FCC-Ag: unguarded, the hull's Ag reference would be the 4H entry.
        entries = [_hcp_ag(-2.75, "mp-8566-GGA"), _fcc_ag(-2.748), _fcc_au(-3.2)]
        cache = _write_cache(tmp_path, entries)
        pd, _ = api.get_dft_convexhull(["Ag", "Au"], data_dir=str(tmp_path))
        ag_refs = [e for e in pd.stable_entries if e.composition.reduced_formula == "Ag"]
        assert [e.entry_id for e in ag_refs] == ["mp-124-GGA"]
        # cache file itself is never rewritten by the read-time guard
        assert len(json.loads(cache.read_text())) == 3

    def test_structure_entries_drop_blacklisted(self, tmp_path, blacklist_file):
        _write_cache(tmp_path, [_hcp_ag(-2.75, "mp-8566-GGA"), _fcc_ag(-2.748), _fcc_au(-3.2)])
        got = api.get_dft_structure_entries(["Ag", "Au"], data_dir=str(tmp_path))
        assert sorted(e.entry_id for e in got) == ["mp-124-GGA", "mp-81-GGA"]


def _rhombo_ag(energy_per_atom, entry_id):
    # A theoretical 9R-like R-3m Ag: the artifact family entry caches carry but
    # the summary endpoint's theoretical=False filter never returns.
    s = Structure.from_spacegroup("R-3m", Lattice.hexagonal(2.9, 21.0), ["Ag"], [[0, 0, 0]])
    e = ComputedStructureEntry(s, energy_per_atom * len(s))
    d = e.as_dict()
    d["entry_id"] = entry_id
    return d


class TestAnchorConsistencyGuard:
    @pytest.fixture()
    def guard_file(self, tmp_path, monkeypatch):
        path = tmp_path / "spurious_structures.json"
        path.write_text(
            json.dumps(
                {
                    "elements": {},
                    "expected_gs_spacegroup": {"Ag": 225},
                }
            )
        )
        monkeypatch.setattr(config, "spurious_structures_file", path)
        monkeypatch.setattr(api, "_spurious_cache", None)
        return path

    def test_sub_anchor_theoretical_entry_is_dropped(self, tmp_path, guard_file):
        # 9R-Ag below FCC: no blacklist id/SG names it, but it undercuts the
        # expected-GS reference, so the guard drops it from the hull.
        entries = [
            _rhombo_ag(-2.752, {"identifier": "mp-aaaczzzz", "suffix": "GGA"}),
            _fcc_ag(-2.748),
            _fcc_au(-3.2),
        ]
        _write_cache(tmp_path, entries)
        pd, _ = api.get_dft_convexhull(["Ag", "Au"], data_dir=str(tmp_path))
        ag_refs = [e for e in pd.stable_entries if e.composition.reduced_formula == "Ag"]
        assert [e.entry_id for e in ag_refs] == ["mp-124-GGA"]

    def test_above_anchor_structures_are_kept(self, tmp_path, guard_file):
        # The guard only removes entries BELOW the expected-GS reference; real
        # metastable structures above it stay available.
        entries = [_rhombo_ag(-2.70, "mp-real-GGA"), _fcc_ag(-2.748), _fcc_au(-3.2)]
        _write_cache(tmp_path, entries)
        got = api.get_dft_structure_entries(["Ag", "Au"], data_dir=str(tmp_path))
        assert sorted(str(e.entry_id) for e in got) == ["mp-124-GGA", "mp-81-GGA", "mp-real-GGA"]

    def test_no_expected_entry_in_cache_drops_nothing(self, tmp_path, guard_file):
        # Without an expected-SG reference in the cache there is nothing to
        # anchor on; the guard must not guess.
        entries = [_rhombo_ag(-2.752, "mp-x-GGA"), _fcc_au(-3.2)]
        _write_cache(tmp_path, entries)
        got = api.get_dft_structure_entries(["Ag", "Au"], data_dir=str(tmp_path))
        assert sorted(str(e.entry_id) for e in got) == ["mp-81-GGA", "mp-x-GGA"]
