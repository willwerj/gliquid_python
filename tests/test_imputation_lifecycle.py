"""Round trips of imputed entries: provenance layout, staleness, dropping, the store.

Binary: Cu-Mg at its pinned liquid parameters, stripped of its interior compounds and imputed
with ``impute_binary_phases(..., write_cache=False)``. Ternary: Al-Mg-Si in hsx mode on the shipped
entries record, with ``config.persist_ternary_imputed`` off. Every store write goes to a
throwaway copy of the shipped record under ``tmp_path``.

Pinned:
  * the ``data`` and provenance key order of an imputed entry;
  * staleness: current when fresh, the reason once the fit moves or the provenance is gone;
  * ``drop_imputed_phases``: a copy with no imputed phase, the pre-imputation
    ``h_hull_interp`` bit for bit, fit state reset; a system with nothing imputed comes back
    as a copy with its fit state kept;
  * ``drop_imputed_phases`` on the ternary object: by name, unknown names, every name;
  * the store round trip of both dimensions through ``cache_imputed_entries`` /
    ``read_imputed_entries`` / ``drop_imputed_entries``.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.lifecycle as imp_lifecycle
import gliquid.imputation.ternary_imputation as imp_ternary
from gliquid.binary import BinaryLiquid
from gliquid.cache import is_imputed_entry_dict
from gliquid.ternary import TernaryLiquidInterpolation

from ._mpds_builders import strip_interior
from ._mpds_builders import without_module_paths as _without_module_paths

# The package names under test, one line each.
IMPUTE = imp_binary.impute_binary_phases
STALENESS = imp_lifecycle.imputed_entry_staleness
STALE_ENTRIES = imp_lifecycle.stale_imputed_entries
DROP = imp_lifecycle.drop_imputed_phases
HULL_POINTS = imp_binary._hull_points
PROVENANCE_KEY = imp_binary.PROVENANCE_KEY
TERNARY_STALENESS = imp_lifecycle.imputed_entry_staleness
TERNARY_DROP = imp_lifecycle.drop_imputed_phases
TERNARY_IMPUTE = imp_ternary.impute_ternary_phases
IS_IMPUTED = is_imputed_entry_dict
CACHE_IMPUTED = api.cache_imputed_entries
READ_IMPUTED = api.read_imputed_entries
DROP_IMPUTED = api.drop_imputed_entries

# The liquid parameters test_imputation_bit_identity.py pins.
PARAMS = {"Cu-Mg": [-41302.26, -5.1627, -8148.05, 0.0]}
SYSTEM = "Cu-Mg"

BINARY_IMPUTED_IDS = ["imputed:MgCu2", "imputed:Mg2Cu"]
BINARY_DATA_KEYS = ["imputed", "label", "x_b", "h_form_j_per_mol", "provenance"]
BINARY_PROVENANCE_KEYS = [
    "calibration",
    "gliquid_version",
    "param_format",
    "fit_params",
    "anchor",
    "inversion_arm",
]
MOVED = "fit parameters have moved"


AMS = ["Al", "Mg", "Si"]
AMS_MIXING = {
    "Al-Mg": [-20000.0, 0, 0, 0],
    "Mg-Si": [-30000.0, 0, 0, 0],
    "Si-Al": [-10000.0, 0, 0, 0],
}
AMS_TARGET = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
AMS_SECOND = {"Al": 0.6, "Mg": 0.2, "Si": 0.2}
TERNARY_DATA_KEYS = ["imputed", "label", "fractions", "h_form_j_per_mol", "provenance"]
TERNARY_PROVENANCE_KEYS = [
    "calibration",
    "gliquid_version",
    "liquid",
    "anchor",
    "margin_j",
    "hull_entry_ids",
]
TERNARY_RECORD = "Al-Mg-Si_ENTRIES_MP_GGA.json"

# Binary fit-state fields drop_imputed_phases resets, with their reset values.
RESET_FIELDS = [
    ("invariants", None),
    ("low_t_exp_phases", None),
    ("unsupported_comp_ranges", []),
    ("fit_unsupported_range_mode", None),
    ("coverage_report", None),
    ("full_comp_ss", False),
    ("guess_symbols", None),
    ("constraints", None),
    ("_ref_params", None),
    ("nmpath", None),
    ("hsx", None),
    ("init_error", False),
    ("skip_reason", None),
]
XS_MIX_RESET = [0.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def binary():
    """``(stripped, imputed, entry_dicts)`` for Cu-Mg."""
    full = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp", params=list(PARAMS[SYSTEM]))
    stripped = strip_interior(full)
    work = copy.deepcopy(stripped)
    records = IMPUTE(work, write_cache=False)
    return stripped, work, [copy.deepcopy(r.entry_dict) for r in records]


# -- Binary --


class TestBinaryEntries:
    def test_the_imputed_ids(self, binary):
        _stripped, _work, entries = binary
        assert [e["entry_id"] for e in entries] == BINARY_IMPUTED_IDS

    def test_data_and_provenance_key_order(self, binary):
        _stripped, _work, entries = binary
        for entry in entries:
            assert list(entry["data"]) == BINARY_DATA_KEYS
            assert list(entry["data"][PROVENANCE_KEY]) == BINARY_PROVENANCE_KEYS


class TestBinaryStaleness:
    def test_every_fresh_entry_is_current(self, binary):
        _stripped, work, entries = binary
        assert [STALENESS(work, e) for e in entries] == [None] * len(entries)

    def test_moving_one_liquid_parameter_makes_every_entry_stale(self, binary):
        _stripped, work, entries = binary
        moved = copy.deepcopy(work)
        params = [float(v) for v in moved.get_params()]
        params[0] += 1000.0
        moved.update_params(params)
        assert [STALENESS(moved, e) for e in entries] == [MOVED] * len(entries)

    def test_stale_imputed_entries_reports_imputed_entries_in_input_order(self, binary):
        _stripped, work, entries = binary
        moved = copy.deepcopy(work)
        params = [float(v) for v in moved.get_params()]
        params[0] += 1000.0
        moved.update_params(params)
        real = next(e for e in work.dft_ch.all_entries if not IS_IMPUTED(e.as_dict()))
        ordered = list(reversed(entries)) + [real.as_dict()]
        reported = STALE_ENTRIES(moved, ordered)
        assert [(e["entry_id"], reason) for e, reason in reported] == [
            (e["entry_id"], MOVED) for e in reversed(entries)
        ]
        assert all(got is want for (got, _r), want in zip(reported, reversed(entries)))


def _assert_reset(bl: BinaryLiquid) -> None:
    for name, value in RESET_FIELDS:
        got = getattr(bl, name)
        if value is None:
            assert got is None, name
        else:
            assert got == value, name
    assert list(bl.xs_mix.values) == XS_MIX_RESET, "xs_mix"
    assert all(p.points == [] for p in bl.phases), "phase.points"


class TestDropImputedPhases:
    def test_it_returns_a_new_object_and_leaves_the_source_imputed(self, binary):
        _stripped, work, _entries = binary
        n_before = sum(1 for p in work.phases if p.imputed)
        dropped = DROP(work)
        assert dropped is not work
        assert n_before and sum(1 for p in work.phases if p.imputed) == n_before

    def test_the_result_is_the_pre_imputation_hull(self, binary):
        stripped, work, _entries = binary
        dropped = DROP(work)
        assert not any(p.imputed for p in dropped.phases)
        assert np.array_equal(dropped.eqs["h_hull_interp"], stripped.eqs["h_hull_interp"])

    def test_the_fit_state_is_reset(self, binary):
        _stripped, work, _entries = binary
        _assert_reset(DROP(work))

    def test_in_place_returns_the_same_object(self, binary):
        _stripped, work, _entries = binary
        target = copy.deepcopy(work)
        assert DROP(target, in_place=True) is target
        assert not any(p.imputed for p in target.phases)

    def test_nothing_imputed_returns_a_copy_with_its_fit_state_kept(self, binary):
        stripped, _work, _entries = binary
        source = copy.deepcopy(stripped)
        sentinel = [{"type": "sentinel"}]
        source.invariants = sentinel
        back = DROP(source)
        assert back is not source
        assert back.invariants == sentinel and back.invariants is not sentinel
        assert list(back.xs_mix.values) == list(source.xs_mix.values)


@pytest.fixture
def binary_store(tmp_path) -> Path:
    """A flat directory store holding a copy of the shipped Cu-Mg entries record."""
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("needs a directory-mode cache corpus")
    record = Path(source) / f"{SYSTEM}_ENTRIES_MP_GGA.json"
    shutil.copy2(record, tmp_path / record.name)
    return tmp_path


def test_binary_store_round_trip(binary, binary_store):
    """Written, read back as written, read as current, then dropped one id at a time."""
    _stripped, work, entries = binary
    CACHE_IMPUTED(SYSTEM, entries, "GGA", data_dir=binary_store)
    back = READ_IMPUTED(SYSTEM, "GGA", data_dir=binary_store)
    assert _without_module_paths(back) == _without_module_paths(json.loads(json.dumps(entries)))
    assert [STALENESS(work, e) for e in back] == [None] * len(back)

    first = entries[0]["entry_id"]
    assert DROP_IMPUTED(SYSTEM, "GGA", entry_ids=[first], data_dir=binary_store) == 1
    left = READ_IMPUTED(SYSTEM, "GGA", data_dir=binary_store)
    assert [e["entry_id"] for e in left] == [e["entry_id"] for e in entries[1:]]
    assert DROP_IMPUTED(SYSTEM, "GGA", entry_ids=None, data_dir=binary_store) == len(left)
    assert READ_IMPUTED(SYSTEM, "GGA", data_dir=binary_store) == []


# -- Ternary --


def build(**kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(AMS),
        delta=0.1,
        xs_mix=AMS_MIXING,
        param_format="linear",
        order="given",
        hull_mode="hsx",
        T_incr=5,
        **kwargs,
    )


@pytest.fixture
def ternary_store(tmp_path) -> Path:
    """A flat directory store holding a copy of the shipped Al-Mg-Si entries record."""
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("needs a directory-mode cache corpus")
    record = Path(source) / TERNARY_RECORD
    shutil.copy2(record, tmp_path / record.name)
    return tmp_path


def _two_targets():
    return [
        {"composition": AMS_TARGET, "temperature": 1000.0, "name": "TA"},
        {"composition": AMS_SECOND, "temperature": 900.0, "name": "TB"},
    ]


class TestTernaryEntries:
    def test_data_and_provenance_key_order(self, no_persist):
        record = build().impute_phase(AMS_TARGET, 1200.0, name="TX", apply=False, write_cache=False)
        assert record.accepted, record.reason
        assert list(record.entry_dict["data"]) == TERNARY_DATA_KEYS
        assert list(record.entry_dict["data"][PROVENANCE_KEY]) == TERNARY_PROVENANCE_KEYS

    def test_an_entry_without_provenance_is_not_current(self, no_persist):
        ti = build()
        record = ti.impute_phase(AMS_TARGET, 1200.0, name="TX", apply=False, write_cache=False)
        bare = copy.deepcopy(record.entry_dict)
        del bare["data"][PROVENANCE_KEY]
        assert TERNARY_STALENESS(ti, record.entry_dict) is None
        assert TERNARY_STALENESS(ti, bare) == "no ternary provenance recorded"


class TestDropTernaryImputed:
    def test_an_unknown_name_removes_nothing(self, no_persist):
        ti = build()
        TERNARY_IMPUTE(ti, _two_targets(), screen=False, write_cache=False)
        before = list(ti.imputed_entries)
        assert TERNARY_DROP(ti, names=["NoSuchPhase"], write_cache=False) == []
        assert ti.imputed_entries == before

    def test_no_names_removes_every_explicit_entry_in_order(self, no_persist):
        ti = build()
        records = TERNARY_IMPUTE(ti, _two_targets(), screen=False, write_cache=False)
        assert [r.accepted for r in records] == [True, True], [r.reason for r in records]
        assert TERNARY_DROP(ti, names=None, write_cache=False) == ["TA", "TB"]
        assert ti.imputed_entries == []


def test_ternary_store_round_trip(no_persist, ternary_store):
    """Two phases written, both placed and current on a rebuild, one dropped from the store."""
    ti = build(data_dir=ternary_store)
    records = TERNARY_IMPUTE(ti, _two_targets(), screen=False, write_cache=True)
    assert [r.written for r in records] == [True, True], [r.reason for r in records]
    ids = [r.entry_dict["entry_id"] for r in records]
    assert [e["entry_id"] for e in READ_IMPUTED(AMS, "GGA", data_dir=ternary_store)] == ids

    reread = build(data_dir=ternary_store, include_imputed=True)
    placed = {str(e.entry_id) for e in reread.solid_hull().stable_entries}
    assert set(ids) <= placed
    for entry in reread.all_imputed_entries():
        assert TERNARY_STALENESS(reread, entry) is None

    assert TERNARY_DROP(ti, names=["TA"], write_cache=True) == ["TA"]
    assert [e["entry_id"] for e in READ_IMPUTED(AMS, "GGA", data_dir=ternary_store)] == [ids[1]]
