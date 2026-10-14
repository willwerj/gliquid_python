"""Where imputed phases are stored: the persistence knob, the session overlay store, and the
read twin of ``api.cache_imputed_entries``.

Runs offline against a temporary copy of the shipped Cu-Mg entries record, so no package
store is ever written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path

import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.config as config
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    KIND_IMPUTED_ENTRIES,
    CacheBackend,
    CacheKey,
    DirectoryBackend,
    OverlayBackend,
    is_imputed_entry_dict,
    resolve_backend,
)

SYSTEM = "Cu-Mg"
DFT_KEY = CacheKey(SYSTEM, KIND_DFT_ENTRIES, "GGA")
IMPUTED_KEY = CacheKey(SYSTEM, KIND_IMPUTED_ENTRIES, "GGA")


def imputed_entry(label: str, x_b: float, h_form_j: float) -> dict:
    return ComputedEntry(
        composition=Composition({"Cu": 1.0 - x_b, "Mg": x_b}),
        energy=0.0,
        entry_id=f"imputed:{label}",
        data={"imputed": True, "label": label, "h_form_j_per_mol": float(h_form_j)},
    ).as_dict()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def store(tmp_path) -> Path:
    """A flat directory store holding a copy of the shipped Cu-Mg entries record."""
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("needs a directory-mode cache corpus")
    record = Path(source) / f"{SYSTEM}_ENTRIES_MP_GGA.json"
    if not record.exists():
        pytest.skip(f"no shipped {record.name}")
    shutil.copy2(record, tmp_path / record.name)
    return tmp_path


@pytest.fixture
def record(store) -> Path:
    return store / f"{SYSTEM}_ENTRIES_MP_GGA.json"


# -- Persistence knob --


class TestPersistKnob:
    @pytest.mark.parametrize(
        "raw, expected",
        [("1", True), ("true", True), ("ON", True), ("0", False), ("off", False), ("", False)],
    )
    def test_recognized_spellings(self, monkeypatch, raw, expected):
        monkeypatch.setenv(config.PERSIST_TERNARY_IMPUTED_ENV_VAR, raw)
        assert config._initial_persist_ternary_imputed() is expected

    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv(config.PERSIST_TERNARY_IMPUTED_ENV_VAR, raising=False)
        assert config._initial_persist_ternary_imputed() is False

    def test_unrecognized_is_off_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(config.PERSIST_TERNARY_IMPUTED_ENV_VAR, "sometimes")
        with caplog.at_level(logging.WARNING, logger="gliquid.config"):
            assert config._initial_persist_ternary_imputed() is False
        assert config.PERSIST_TERNARY_IMPUTED_ENV_VAR in caplog.text

    def test_setter_round_trips(self):
        before = config.persist_ternary_imputed
        try:
            config.set_persist_ternary_imputed(True)
            assert config.persist_ternary_imputed is True
            config.set_persist_ternary_imputed(0)
            assert config.persist_ternary_imputed is False
        finally:
            config.set_persist_ternary_imputed(before)


# -- The overlay store --


class TestOverlayProtocol:
    def test_is_a_cache_backend(self, store):
        overlay = OverlayBackend(store)
        assert isinstance(overlay, CacheBackend)
        assert resolve_backend(overlay) is overlay

    def test_base_path_resolves_like_data_dir(self, store):
        overlay = OverlayBackend(store)
        assert isinstance(overlay.base, DirectoryBackend)

    def test_capabilities(self, store):
        caps = OverlayBackend(store).capabilities
        assert {"read", "write", "variants"} <= caps
        assert "paths" not in caps

    def test_deepcopy_shares_the_session(self, store):
        overlay = OverlayBackend(store)
        assert copy.deepcopy(overlay) is overlay


class TestOverlayReadsAndWrites:
    def test_reads_fall_through_to_the_base(self, store, record):
        overlay = OverlayBackend(store)
        assert overlay.exists(DFT_KEY)
        assert overlay.read_json(DFT_KEY) == json.loads(record.read_text())
        assert overlay.locate(DFT_KEY) == str(record)

    def test_writes_stay_in_memory(self, store, record):
        before = sha256(record)
        overlay = OverlayBackend(store, label="A")
        overlay.write_json(IMPUTED_KEY, [imputed_entry("X", 0.5, -1.0)])
        assert sha256(record) == before
        assert [e["entry_id"] for e in overlay.read_json(IMPUTED_KEY)] == ["imputed:X"]
        assert overlay.locate(IMPUTED_KEY).startswith("overlay[A]#")
        assert overlay.records == [IMPUTED_KEY]

    def test_two_overlays_on_one_base_are_isolated(self, store):
        a, b = OverlayBackend(store), OverlayBackend(store)
        a.write_json(IMPUTED_KEY, [imputed_entry("X", 0.5, -1.0)])
        assert b.read_json(IMPUTED_KEY) == []  # control: the base serves its own (empty) set
        assert a.read_json(IMPUTED_KEY) != b.read_json(IMPUTED_KEY)

    def test_records_are_copied_in_and_out(self, store):
        overlay = OverlayBackend(store)
        payload = [imputed_entry("X", 0.5, -1.0)]
        overlay.write_json(IMPUTED_KEY, payload)
        payload[0]["entry_id"] = "mutated after write"
        read = overlay.read_json(IMPUTED_KEY)
        read[0]["entry_id"] = "mutated after read"
        assert overlay.read_json(IMPUTED_KEY)[0]["entry_id"] == "imputed:X"

    def test_discard(self, store):
        overlay = OverlayBackend(store)
        overlay.write_json(IMPUTED_KEY, [imputed_entry("X", 0.5, -1.0)])
        overlay.discard(IMPUTED_KEY)
        assert overlay.records == []

    def test_variants_union(self, store):
        overlay = OverlayBackend(store)
        overlay.write_json(CacheKey(SYSTEM, KIND_DFT_ENTRIES, "R2SCAN"), [])
        assert overlay.variants(SYSTEM, KIND_DFT_ENTRIES) == ["GGA", "R2SCAN"]
        assert DirectoryBackend(store).variants(SYSTEM, KIND_DFT_ENTRIES) == ["GGA"]

    def test_threaded_writes_all_land(self, store):
        overlay = OverlayBackend(store)
        keys = [CacheKey(f"Cu-Mg{i}", KIND_DFT_ENTRIES, "GGA") for i in range(16)]
        threads = [
            threading.Thread(target=overlay.write_json, args=(k, [i])) for i, k in enumerate(keys)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(overlay.records, key=lambda k: k.sys_name) == sorted(
            keys, key=lambda k: k.sys_name
        )


class TestOverlayShadowsEmbeddedImputedEntries:
    """A directory base keeps imputed entries inside the DFT record."""

    @pytest.fixture
    def embedded(self, record) -> Path:
        raw = json.loads(record.read_text())
        raw.append(imputed_entry("OnDisk", 0.4, -9000.0))
        record.write_text(json.dumps(raw))
        return record

    def test_control_the_base_serves_its_embedded_entry(self, store, embedded):
        hull, _ = api.get_dft_convexhull(SYSTEM, include_imputed=True, data_dir=store)
        assert "imputed:OnDisk" in {str(e.entry_id) for e in hull.all_entries}

    def test_the_overlay_set_replaces_the_embedded_one(self, store, embedded):
        overlay = OverlayBackend(store)
        overlay.write_json(IMPUTED_KEY, [imputed_entry("InSession", 0.6, -9000.0)])
        dft = overlay.read_json(DFT_KEY)
        assert not any(is_imputed_entry_dict(e) for e in dft)
        hull, _ = api.get_dft_convexhull(SYSTEM, include_imputed=True, data_dir=overlay)
        ids = {str(e.entry_id) for e in hull.all_entries}
        assert "imputed:InSession" in ids
        assert "imputed:OnDisk" not in ids


class TestOverlayPassthrough:
    class _PooledBase(DirectoryBackend):
        def pool_covers(self, sys_name, dft_type):
            return sys_name == "Al-Mg-Si"

    def test_listed_readers_reach_the_base(self, store):
        overlay = OverlayBackend(self._PooledBase(store))
        assert overlay.pool_covers("Al-Mg-Si", "GGA") is True

    @pytest.mark.parametrize("name", ["path_for", "sys_location", "write_imputed_entries", "_root"])
    def test_other_names_do_not(self, store, name):
        with pytest.raises(AttributeError):
            getattr(OverlayBackend(store), name)


# -- api: write through an overlay, read back --


class TestReadImputedEntries:
    def test_absent_is_empty(self, store):
        assert api.read_imputed_entries(SYSTEM, data_dir=store) == []

    def test_round_trip_through_a_directory_store(self, store):
        api.cache_imputed_entries(SYSTEM, [imputed_entry("X", 0.5, -1.0)], data_dir=store)
        got = api.read_imputed_entries(SYSTEM, data_dir=store)
        assert [e["entry_id"] for e in got] == ["imputed:X"]

    def test_component_order_does_not_matter(self, store):
        api.cache_imputed_entries(SYSTEM, [imputed_entry("X", 0.5, -1.0)], data_dir=store)
        assert api.read_imputed_entries(["Mg", "Cu"], data_dir=store) == api.read_imputed_entries(
            SYSTEM, data_dir=store
        )

    def test_writes_through_an_overlay_leave_the_base_untouched(self, store, record):
        before = sha256(record)
        overlay = OverlayBackend(store)
        api.cache_imputed_entries(SYSTEM, [imputed_entry("X", 0.5, -1.0)], data_dir=overlay)
        assert sha256(record) == before
        assert [e["entry_id"] for e in api.read_imputed_entries(SYSTEM, data_dir=overlay)] == [
            "imputed:X"
        ]
        assert api.read_imputed_entries(SYSTEM, data_dir=store) == []
