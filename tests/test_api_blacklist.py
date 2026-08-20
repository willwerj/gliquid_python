"""Tests for the Tier A spurious-structure blacklist in gliquid.api.

All offline. The blacklist (``config.spurious_structures_file``) must remove
blacklisted ELEMENTAL structures from both cache-read paths without rewriting
the cache file, matching by classic mp-id string, by dict-form entry ids, and —
for new-generation alpha ids that carry no classic id — by the (element,
spacegroup) of the entry's structure.

Its sibling ``compounds`` block carries non-elemental artifacts, which the
elemental path structurally cannot express (it refuses arity != 1). Those rules
apply at ANY arity and replaced the two element-specific Mg149 literals that
used to sit inline in ``api.py``.
"""

import json
import os
import time

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
        # An elements-only blacklist is an elemental policy: an arity-2 entry is
        # untouched even when its id is blacklisted for the elemental path.
        assert not api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0, "Au": 1.0}, "entry_id": "mp-8566-GGA"}
        )

    def test_elements_only_file_ignores_compound_composition_artifacts(self, blacklist_file):
        # Acceptance 4: a file carrying ONLY elemental rules behaves exactly as
        # before the compounds block existed — no compound rule, nothing filtered.
        assert api._spurious_structure_index()[3] == ()
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})

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


def _agau_b2(energy_per_atom, entry_id="mp-2647-GGA"):
    # One Ag + one Au in a CsCl-type cell: a real arity-2 entry the elemental
    # blacklist path can never reach.
    s = Structure(
        Lattice.cubic(3.2), ["Ag", "Au"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return ComputedStructureEntry(s, energy_per_atom * len(s), entry_id=entry_id)


def _compound_blacklist(tmp_path, monkeypatch, payload):
    path = tmp_path / "spurious_structures.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(config, "spurious_structures_file", path)
    monkeypatch.setattr(api, "_spurious_cache", None)
    return path


class TestCompoundRules:
    """The ``compounds`` block: three predicate forms, applied at any arity."""

    def test_compounds_only_file_still_filters(self, tmp_path, monkeypatch):
        # Acceptance 3 — guards the ``if not ids and not pairs: return False``
        # fast-out, which would make a compounds-only file a silent no-op.
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {"compounds": [{"match": {"element_count": {"Mg": 149}}, "reason": "test"}]},
        )
        ids, pairs, expected, compounds = api._spurious_structure_index()
        assert not ids and not pairs and not expected
        assert compounds  # positive control: the file really did parse
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})

    def test_element_count_leaves_other_elements_free(self, tmp_path, monkeypatch):
        _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"element_count": {"Mg": 149}}}]}
        )
        for other in ({"Cu": 1.0}, {"Al": 7.0}, {}, {"Cu": 1.0, "Zn": 3.0}):
            assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, **other}})

    def test_element_count_is_exact_not_a_threshold(self, tmp_path, monkeypatch):
        _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"element_count": {"Mg": 149}}}]}
        )
        for count in (148.0, 150.0, 1.0, 149.5):
            assert not api._is_spurious_entry_dict({"composition": {"Mg": count, "Cu": 1.0}})

    def test_element_count_matches_int_and_float_counts(self, tmp_path, monkeypatch):
        # Cached counts are floats; the literal this replaced compared them to an
        # int, so both spellings must match.
        _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"element_count": {"Mg": 149}}}]}
        )
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149, "Cu": 1}})
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})

    def test_composition_form_cannot_over_match(self, tmp_path, monkeypatch):
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {"compounds": [{"match": {"composition": {"Mg": 149, "Cu": 1}}}]},
        )
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})
        # same Mg count, different partner -> NOT matched (this is the whole point)
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Zn": 1.0}})
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 149.0}})

    def test_material_id_form_matches_compound_entries(self, tmp_path, monkeypatch):
        _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"material_id": "mp-2647"}}]}
        )
        assert api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0, "Au": 1.0}, "entry_id": "mp-2647-GGA"}
        )
        assert api._is_spurious_entry_dict(
            {
                "composition": {"Ag": 1.0, "Au": 1.0},
                "entry_id": {"identifier": "mp-2647", "suffix": "GGA", "separator": "-"},
            }
        )
        assert not api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0, "Au": 1.0}, "entry_id": "mp-264-GGA"}
        )

    def test_match_form_precedence_is_material_id_first(self, tmp_path, monkeypatch):
        # A record naming several forms is decided by the highest-precedence one.
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {
                "compounds": [
                    {
                        "match": {
                            "material_id": "mp-2647",
                            "composition": {"Ag": 9, "Au": 9},
                        }
                    }
                ]
            },
        )
        assert api._spurious_structure_index()[3] == (("material_id", "mp-2647"),)
        assert api._is_spurious_entry_dict(
            {"composition": {"Ag": 1.0, "Au": 1.0}, "entry_id": "mp-2647-GGA"}
        )
        assert not api._is_spurious_entry_dict({"composition": {"Ag": 9.0, "Au": 9.0}})

    def test_unusable_match_block_is_dropped_not_universal(self, tmp_path, monkeypatch, caplog):
        # An empty or unrecognized predicate must never degrade to "match all".
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {
                "compounds": [
                    {"match": {"element_count": {}}},
                    {"match": {"formula": "Mg149Cu"}},
                    {},
                ]
            },
        )
        with caplog.at_level("WARNING"):
            assert api._spurious_structure_index()[3] == ()
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})
        assert not api._is_spurious_entry_dict({"composition": {"Ag": 4.0}})

    def test_compound_rule_also_applies_to_elemental_entries(self, tmp_path, monkeypatch):
        # The literal this replaced tested every entry regardless of arity.
        _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"element_count": {"Mg": 149}}}]}
        )
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0}})

    def test_compound_and_element_rules_coexist(self, tmp_path, monkeypatch):
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {
                "elements": {"Ag": [{"material_id": "mp-8566", "spacegroup_number": 194}]},
                "compounds": [{"match": {"element_count": {"Mg": 149}}}],
            },
        )
        assert api._is_spurious_entry_dict({"composition": {"Ag": 4.0}, "entry_id": "mp-8566-GGA"})
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 2.0, "Cu": 1.0}})

    def test_index_cache_reindexes_when_the_file_changes(self, tmp_path, monkeypatch):
        # The memo tuple grew to 6 entries when compounds landed; a mis-shifted
        # unpack site yields a cache that never invalidates. Positive control
        # first: the rule must be live before we assert it goes away.
        path = _compound_blacklist(
            tmp_path, monkeypatch, {"compounds": [{"match": {"element_count": {"Mg": 149}}}]}
        )
        assert api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})
        assert api._spurious_structure_index()[3]  # memo warm
        assert api._spurious_structure_index()[3]  # served from the memo
        path.write_text(json.dumps({"compounds": [{"match": {"element_count": {"Mg": 7}}}]}))
        os.utime(path, (time.time() + 5, time.time() + 5))
        assert api._spurious_structure_index()[3] == (("element_count", (("Mg", 7),)),)
        assert not api._is_spurious_entry_dict({"composition": {"Mg": 149.0, "Cu": 1.0}})
        assert api._is_spurious_entry_dict({"composition": {"Mg": 7.0, "Cu": 1.0}})

    def test_read_path_drops_a_blacklisted_compound(self, tmp_path, monkeypatch):
        # End-to-end through the public read path, not just the predicate.
        _compound_blacklist(
            tmp_path,
            monkeypatch,
            {"compounds": [{"match": {"composition": {"Ag": 1, "Au": 1}}}]},
        )
        _write_cache(tmp_path, [_agau_b2(-3.4), _fcc_ag(-2.748), _fcc_au(-3.2)])
        got = api.get_dft_structure_entries(["Ag", "Au"], data_dir=str(tmp_path))
        assert sorted(str(e.entry_id) for e in got) == ["mp-124-GGA", "mp-81-GGA"]
        # the cache file itself is never rewritten
        assert len(json.loads((tmp_path / "Ag-Au_ENTRIES_MP_GGA.json").read_text())) == 3


class TestShippedBlacklistCarriesMg149:
    """The shipped reference table must still filter what the deleted literals did."""

    @pytest.fixture()
    def shipped(self, monkeypatch):
        monkeypatch.setattr(api, "_spurious_cache", None)
        assert config.spurious_structures_file is not None
        return config.spurious_structures_file

    def test_shipped_file_declares_the_compound_rule(self, shipped):
        raw = json.loads(shipped.read_text() if hasattr(shipped, "read_text") else open(shipped).read())
        assert "compounds" in raw, "the compounds block must be a SIBLING of elements"
        assert "compounds" not in raw.get("elements", {})
        assert any(
            rec.get("match", {}).get("element_count") == {"Mg": 149} for rec in raw["compounds"]
        )
        for rec in raw["compounds"]:
            assert rec.get("reason") and rec.get("source") and rec.get("date")

    def test_shipped_blacklist_filters_the_synthetic_artifact(self, shipped):
        artifact = {"composition": {"Mg": 149.0, "Cu": 1.0}, "entry_id": "mp-synthetic-GGA"}
        keeper = {"composition": {"Mg": 2.0, "Cu": 1.0}, "entry_id": "mp-keeper-GGA"}
        near_miss = {"composition": {"Mg": 148.0, "Cu": 1.0}, "entry_id": "mp-near-GGA"}
        got = api._filter_spurious_entries([artifact, keeper, near_miss])
        assert got == [keeper, near_miss]
