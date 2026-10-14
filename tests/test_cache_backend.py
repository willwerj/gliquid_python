"""The record-level cache seam: ``CacheKey``, ``CacheBackend``, ``DirectoryBackend``.

The seam sits at RECORDS, not directories, so a single-file store answers it too. The three
on-disk filename conventions, INCLUDING the indexless ``<sys>.json`` MPDS naming (the
convention for a single-diagram store), are asserted on the actual strings.
"""

import json
import os

import pytest

import gliquid.api as api
import gliquid.config as config
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    KIND_IMPUTED_ENTRIES,
    CacheKey,
    DirectoryBackend,
    resolve_backend,
)

from ._mpds_builders import cu_mg_entry_pool


@pytest.fixture(autouse=True)
def _directory_mode(monkeypatch):
    """Pin ``cache_mode`` for the whole module rather than inheriting the session's: the root
    ``conftest.py``'s ``GLIQUID_TEST_SQLITE_STORE`` swap configures a single-file store
    process-wide. A test that sets the mode itself still wins.
    """
    monkeypatch.setattr(config, "cache_mode", "directory")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A temp corpus installed as the global one, restored afterwards."""
    monkeypatch.setattr(config, "cache_dir", tmp_path)
    monkeypatch.setattr(config, "dir_structure", "flat")
    return tmp_path


# -- Characterization: the on-disk behaviour every backend keeps --


class TestFilenamesAreByteIdentical:
    """``DirectoryBackend`` reproduces the on-disk filenames, asserted against the literal
    strings rather than against the backend itself."""

    def test_dft_entries_filename(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        key = CacheKey("Cu-Mg", "dft_entries", "GGA")
        assert backend.path_for(key) == tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json"

    def test_mpds_indexed_filename(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == tmp_path / "Cu-Mg_MPDS_PD_0.json"
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "3")) == tmp_path / "Cu-Mg_MPDS_PD_3.json"

    def test_mpds_indexless_filename(self, tmp_path):
        """``variant == ""`` is the indexless ``<sys>.json`` naming, not ``<sys>_MPDS_PD_.json``."""
        backend = DirectoryBackend(tmp_path)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "")) == tmp_path / "Cu-Mg.json"

    def test_nested_layout_puts_them_under_the_system_directory(self, tmp_path):
        backend = DirectoryBackend(tmp_path, dir_structure="nested")
        assert (
            backend.path_for(CacheKey("Cu-Mg", "dft_entries", "GGA"))
            == tmp_path / "Cu-Mg" / "Cu-Mg_ENTRIES_MP_GGA.json"
        )
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "")) == tmp_path / "Cu-Mg" / "Cu-Mg.json"

    def test_unknown_kind_is_rejected_rather_than_guessed(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        with pytest.raises(ValueError, match="kind"):
            backend.path_for(CacheKey("Cu-Mg", "not_a_kind", "0"))


# -- The record-level protocol --


class TestCacheKey:
    def test_is_frozen_and_hashable(self):
        import dataclasses

        key = CacheKey("Cu-Mg", "mpds", "0")
        with pytest.raises(dataclasses.FrozenInstanceError):
            key.sys_name = "Ag-V"
        assert {key, CacheKey("Cu-Mg", "mpds", "0")} == {key}, "equal keys must collapse"

    def test_variant_defaults_to_the_indexless_naming(self):
        assert CacheKey("Cu-Mg", "mpds").variant == ""


class TestDirectoryBackendRoundTrip:
    def test_write_then_read_exchanges_python_objects(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        key = CacheKey("Cu-Mg", "dft_entries", "GGA")
        assert backend.exists(key) is False
        payload = [{"composition": {"Cu": 1}, "energy": -1.0}]
        backend.write_json(key, payload)
        assert backend.exists(key) is True
        assert backend.read_json(key) == payload
        # ...and it really is the plain file on disk, not an opaque blob
        assert json.loads((tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").read_text()) == payload

    def test_read_json_of_a_missing_record_raises(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        with pytest.raises(FileNotFoundError):
            backend.read_json(CacheKey("Cu-Mg", "mpds", "0"))

    def test_write_is_atomic_leaving_no_scratch_file(self, tmp_path):
        """A failed write must not truncate the previous record (see api.TestAtomicCacheWrite)."""
        backend = DirectoryBackend(tmp_path)
        key = CacheKey("Cu-Mg", "dft_entries", "GGA")
        backend.write_json(key, [{"keep": True}])
        before = backend.path_for(key).read_bytes()
        with pytest.raises(TypeError):
            backend.write_json(key, [{"bad": {1, 2}}])
        assert backend.path_for(key).read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["Cu-Mg_ENTRIES_MP_GGA.json"]

    def test_nested_write_creates_the_system_directory(self, tmp_path):
        backend = DirectoryBackend(tmp_path, dir_structure="nested")
        key = CacheKey("Cu-Mg", "mpds", "0")
        backend.write_json(key, {"reference": None})
        assert (tmp_path / "Cu-Mg" / "Cu-Mg_MPDS_PD_0.json").exists()

    def test_locate_names_the_record(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        located = backend.locate(CacheKey("Cu-Mg", "mpds", "0"))
        assert isinstance(located, str)
        assert located.endswith("Cu-Mg_MPDS_PD_0.json")

    def test_capabilities_is_a_frozenset_declaring_writability(self, tmp_path):
        caps = DirectoryBackend(tmp_path).capabilities
        assert isinstance(caps, frozenset)
        assert {"read", "write", "paths"} <= caps


class TestDirectoryBackendVariants:
    def test_lists_mpds_variants_including_the_indexless_one(self, tmp_path):
        (tmp_path / "Cu-Mg.json").write_text("{}")
        (tmp_path / "Cu-Mg_MPDS_PD_0.json").write_text("{}")
        (tmp_path / "Cu-Mg_MPDS_PD_2.json").write_text("{}")
        backend = DirectoryBackend(tmp_path)
        assert sorted(backend.variants("Cu-Mg", "mpds")) == ["", "0", "2"]

    def test_lists_dft_variants(self, tmp_path):
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text("[]")
        (tmp_path / "Cu-Mg_ENTRIES_MP_R2SCAN.json").write_text("[]")
        backend = DirectoryBackend(tmp_path)
        assert sorted(backend.variants("Cu-Mg", "dft_entries")) == ["GGA", "R2SCAN"]

    def test_does_not_leak_another_systems_records(self, tmp_path):
        """POSITIVE CONTROL included: the real system's record IS found."""
        (tmp_path / "Cu-Mg_MPDS_PD_0.json").write_text("{}")
        (tmp_path / "Ag-V_MPDS_PD_1.json").write_text("{}")
        backend = DirectoryBackend(tmp_path)
        assert backend.variants("Cu-Mg", "mpds") == ["0"]  # control: found
        assert backend.variants("Ag-V", "mpds") == ["1"]  # and not confused

    def test_empty_store_lists_nothing(self, tmp_path):
        assert DirectoryBackend(tmp_path).variants("Cu-Mg", "mpds") == []

    def test_mpds_kind_does_not_match_a_dft_entries_file(self, tmp_path):
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text("[]")
        backend = DirectoryBackend(tmp_path)
        assert backend.variants("Cu-Mg", "mpds") == []
        assert backend.variants("Cu-Mg", "dft_entries") == ["GGA"]  # control


class TestFlatRootIsCreatedOnWrite:
    """A flat store's missing root is created by the first write and by nothing else."""

    PAYLOAD = [{"entry_id": "mp-1"}]

    def _reads_leave_root_absent_then_write_creates_it(self, backend, root, tmp_path):
        key = CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")
        assert backend.exists(key) is False
        with pytest.raises(FileNotFoundError):
            backend.read_json(key)
        assert backend.variants("Cu-Mg", KIND_DFT_ENTRIES) == []
        assert not (tmp_path / "a").exists(), "a read created the store root"

        backend.write_json(key, self.PAYLOAD)
        assert json.loads((root / "Cu-Mg_ENTRIES_MP_GGA.json").read_text()) == self.PAYLOAD

    def test_an_explicit_missing_root(self, tmp_path):
        root = tmp_path / "a" / "b"
        self._reads_leave_root_absent_then_write_creates_it(DirectoryBackend(root), root, tmp_path)

    def test_the_global_backend_after_set_cache_dir_to_a_missing_directory(
        self, tmp_path, monkeypatch
    ):
        for name in (
            "cache_dir",
            "cache_mode",
            "phase_transitions_file",
            "omegas_file",
            "spurious_structures_file",
        ):
            monkeypatch.setattr(config, name, getattr(config, name))
        monkeypatch.setattr(config, "dir_structure", "flat")
        root = tmp_path / "a" / "b"
        config.set_cache_dir(root)
        self._reads_leave_root_absent_then_write_creates_it(resolve_backend(None), root, tmp_path)

    def test_an_explicit_root_that_cannot_be_created_raises_the_os_error(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(OSError):
            DirectoryBackend(blocker).write_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA"), [])
        assert blocker.read_text() == "not a directory"


REAL_ENTRY = {"composition": {"Cu": 1.0}, "energy": -3.7, "entry_id": "mp-30"}
TAGGED_ENTRY = {
    "composition": {"Cu": 1.0, "Mg": 2.0},
    "energy": -13.4,
    "entry_id": "imputed:Mg2Cu",
    "data": {"imputed": True},
}


class TestDirectoryBackendImputedKind:
    """``KIND_IMPUTED_ENTRIES`` over a directory store, where the set lives INSIDE the
    DFT-entries record. The same seam a single-file store answers from its own table, so
    no caller has to ask which backend it holds."""

    def _record(self, tmp_path, entries):
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(entries))
        return DirectoryBackend(tmp_path)

    def test_reading_returns_only_the_tagged_entries(self, tmp_path):
        backend = self._record(tmp_path, [REAL_ENTRY, TAGGED_ENTRY])
        assert backend.read_json(CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA")) == [TAGGED_ENTRY]

    def test_the_two_kinds_name_the_same_file(self, tmp_path):
        backend = DirectoryBackend(tmp_path)
        assert backend.path_for(CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA")) == backend.path_for(
            CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")
        )

    def test_exists_follows_the_dft_record(self, tmp_path):
        key = CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA")
        assert DirectoryBackend(tmp_path).exists(key) is False
        assert self._record(tmp_path, [REAL_ENTRY]).exists(key) is True

    def test_writing_replaces_the_tagged_entries_and_keeps_the_real_ones(self, tmp_path):
        backend = self._record(tmp_path, [REAL_ENTRY, TAGGED_ENTRY])
        replacement = dict(TAGGED_ENTRY, entry_id="imputed:CuMg")
        backend.write_json(CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA"), [replacement])
        on_disk = json.loads((tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").read_text())
        assert on_disk == [REAL_ENTRY, replacement]

    def test_writing_an_empty_set_clears_the_tagged_entries(self, tmp_path):
        backend = self._record(tmp_path, [REAL_ENTRY, TAGGED_ENTRY])
        key = CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA")
        backend.write_json(key, [])
        assert backend.read_json(key) == []
        assert backend.read_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")) == [REAL_ENTRY]

    def test_variants_reports_only_a_record_that_carries_a_tagged_entry(self, tmp_path):
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps([REAL_ENTRY, TAGGED_ENTRY]))
        (tmp_path / "Cu-Mg_ENTRIES_MP_R2SCAN.json").write_text(json.dumps([REAL_ENTRY]))
        backend = DirectoryBackend(tmp_path)
        assert backend.variants("Cu-Mg", KIND_IMPUTED_ENTRIES) == ["GGA"]
        # control: both records exist for the DFT kind
        assert backend.variants("Cu-Mg", KIND_DFT_ENTRIES) == ["GGA", "R2SCAN"]


class TestResolveBackend:
    def test_none_means_the_configured_global_store(self, store):
        backend = resolve_backend(None)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == store / "Cu-Mg_MPDS_PD_0.json"

    def test_the_global_backend_tracks_later_config_changes(self, store, monkeypatch, tmp_path):
        """``set_cache_dir`` mid-session must be observed."""
        backend = resolve_backend(None)
        moved = tmp_path / "moved"
        moved.mkdir()
        monkeypatch.setattr(config, "cache_dir", moved)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == moved / "Cu-Mg_MPDS_PD_0.json"

    def test_a_path_means_a_FLAT_directory_backend(self, store, monkeypatch, tmp_path):
        """An explicit ``data_dir`` is flat inside it, whatever the global mode."""
        monkeypatch.setattr(config, "dir_structure", "nested")
        explicit = tmp_path / "explicit"
        backend = resolve_backend(explicit)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == explicit / "Cu-Mg_MPDS_PD_0.json"

    def test_a_str_works_the_same_as_a_path(self, tmp_path):
        backend = resolve_backend(str(tmp_path))
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == tmp_path / "Cu-Mg_MPDS_PD_0.json"

    def test_an_existing_backend_is_used_as_is(self, tmp_path):
        backend = DirectoryBackend(tmp_path, dir_structure="nested")
        assert resolve_backend(backend) is backend

    def test_sqlite_mode_pointed_at_a_directory_is_refused_rather_than_guessed(
        self, store, monkeypatch
    ):
        """``cache_mode`` and ``cache_dir`` must agree; ``set_cache_dir`` keeps them so.

        The store selected by ``SqliteBackend`` is covered in ``test_sqlite_backend.py``;
        what matters here is that a hand-assembled disagreement fails loudly.
        """
        from gliquid.cache import CacheModeError

        monkeypatch.setattr(config, "cache_mode", "sqlite")
        with pytest.raises(CacheModeError, match="DIRECTORY"):
            resolve_backend(None)

    def test_an_unknown_mode_is_refused(self, store, monkeypatch):
        from gliquid.cache import CacheModeError

        monkeypatch.setattr(config, "cache_mode", "parquet")
        with pytest.raises(CacheModeError, match="parquet"):
            resolve_backend(None)


class TestResolveCachePath:
    """``api.resolve_cache_path`` names the path of one record."""

    def test_returns_the_record_path_for_a_directory_store(self, store):
        got = api.resolve_cache_path(CacheKey("Cu-Mg", "mpds", "0"))
        assert got == store / "Cu-Mg_MPDS_PD_0.json"

    def test_honours_an_explicit_store_override(self, store, tmp_path):
        other = tmp_path / "other"
        got = api.resolve_cache_path(CacheKey("Cu-Mg", "dft_entries", "GGA"), cache=other)
        assert got == other / "Cu-Mg_ENTRIES_MP_GGA.json"

    def test_nested_mode_still_creates_the_system_dir(self, store, monkeypatch):
        monkeypatch.setattr(config, "dir_structure", "nested")
        got = api.resolve_cache_path(CacheKey("Cu-Mg", "mpds", "0"))
        assert got == store / "Cu-Mg" / "Cu-Mg_MPDS_PD_0.json"
        assert (store / "Cu-Mg").is_dir()

    def test_unconfigured_corpus_raises_naming_the_system(self, monkeypatch):
        monkeypatch.setattr(config, "cache_dir", None)
        with pytest.raises(config.ConfigError, match="Cu-Mg"):
            api.resolve_cache_path(CacheKey("Cu-Mg", "mpds", "0"))


class TestWriteSitesStillProduceTheHistoricalFiles:
    """End-to-end: the three api entry points write the on-disk filenames."""

    def test_cold_fetch_writes_the_entries_file(self, tmp_path, monkeypatch):
        pool = cu_mg_entry_pool()
        monkeypatch.setattr(api, "_get_dft_entries_from_components", lambda *a, **k: pool)
        api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["Cu-Mg_ENTRIES_MP_GGA.json"]

    def test_imputed_append_returns_the_same_path_string_as_before(self, tmp_path):
        cache = tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json"
        cache.write_text(json.dumps([{"composition": {"Cu": 1}, "energy": 0.0, "entry_id": "r"}]))
        written = api.cache_imputed_entries(
            ["Cu", "Mg"], [{"entry_id": "imputed:x", "composition": {"Cu": 1}}], data_dir=tmp_path
        )
        assert os.path.normpath(written) == os.path.normpath(str(cache))
        assert any(e["entry_id"] == "imputed:x" for e in json.loads(cache.read_text()))
