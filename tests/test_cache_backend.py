"""The record-level cache seam: ``CacheKey``, ``CacheBackend``, ``DirectoryBackend``.

Every cache read and write used to funnel through ``api._resolve_sys_dir(sys_name)``, which
answers *"which directory"*. A single-file store cannot answer that, so the seam sits one
level down — at RECORDS, not directories.

The first class here is a CHARACTERIZATION suite written against the pre-refactor code and
kept green through it. Two behaviours in it are load-bearing and easy to lose:

* ``_resolve_sys_dir`` calls ``os.makedirs(..., exist_ok=True)` in nested mode on the READ
  path, not just on write. ``dev/scripts/Fit_Binary_Systems.py`` relies on it for cold
  fetches into the workspace ``matrix_data`` store.
* the three on-disk filename conventions, INCLUDING the indexless ``<sys>.json`` MPDS
  naming, which is the legitimate convention for a single-diagram store.

Both are asserted on the actual strings, so a backend that "reproduces today's filenames"
has to actually do so.
"""

import json
import os
import warnings
from pathlib import Path

import pytest

import gliquid.api as api
import gliquid.config as config
from gliquid.cache import CacheKey, DirectoryBackend, resolve_backend


@pytest.fixture(autouse=True)
def _directory_mode(monkeypatch):
    """Pin ``cache_mode`` for the whole module rather than inheriting the session's.

    Every test here is about the DIRECTORY backend, and a test that installs a temp corpus
    but leaves the mode ambient is only testing what it means to test by coincidence -- the
    root ``conftest.py``'s ``GLIQUID_TEST_SQLITE_STORE`` swap configures a single-file store
    process-wide. A test that overrides the mode itself still wins: ``monkeypatch`` inside
    the test body applies after this fixture.
    """
    monkeypatch.setattr(config, "cache_mode", "directory")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A temp corpus installed as the global one, restored afterwards."""
    monkeypatch.setattr(config, "cache_dir", tmp_path)
    monkeypatch.setattr(config, "dir_structure", "flat")
    return tmp_path


# ---------------------------------------------------------------------------------------
# Characterization: behaviour that predates the seam and must survive it
# ---------------------------------------------------------------------------------------


class TestResolveSysDirCharacterization:
    """``api._resolve_sys_dir`` — the deprecated shim — keeps its exact old semantics."""

    def test_nested_mode_creates_the_system_dir_as_a_side_effect_of_a_READ(
        self, tmp_path, monkeypatch
    ):
        """The side effect ``dev/scripts/Fit_Binary_Systems.py`` depends on.

        Nothing here writes. A bare resolve — the read path — must still leave the system
        directory on disk, because the cold-fetch write that follows it in a campaign
        assumes the directory is already there.
        """
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "dir_structure", "nested")
        assert not (tmp_path / "Cu-Mg").exists()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            resolved = api._resolve_sys_dir("Cu-Mg")

        assert Path(resolved) == tmp_path / "Cu-Mg"
        assert (tmp_path / "Cu-Mg").is_dir(), "nested read path stopped creating the system dir"

    def test_flat_mode_does_not_create_anything(self, tmp_path, monkeypatch):
        """POSITIVE CONTROL for the test above: flat mode never made directories."""
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "dir_structure", "flat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            resolved = api._resolve_sys_dir("Cu-Mg")
        assert Path(resolved) == tmp_path
        assert list(tmp_path.iterdir()) == [], "flat mode created something it never used to"

    def test_explicit_data_dir_wins_and_is_flat(self, tmp_path, monkeypatch):
        """An explicit ``data_dir`` implies a FLAT layout inside it, in either mode."""
        other = tmp_path / "explicit"
        other.mkdir()
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "dir_structure", "nested")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            resolved = api._resolve_sys_dir("Cu-Mg", data_dir=other)
        assert Path(resolved) == other
        assert list(other.iterdir()) == []

    def test_unconfigured_corpus_raises_config_error_naming_the_system(self, monkeypatch):
        monkeypatch.setattr(config, "cache_dir", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(config.ConfigError) as exc:
                api._resolve_sys_dir("Cu-Mg")
        assert "Cu-Mg" in str(exc.value)

    def test_invalid_dir_structure_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "dir_structure", "sideways")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="dir_structure"):
                api._resolve_sys_dir("Cu-Mg")


class TestFilenamesAreByteIdentical:
    """``DirectoryBackend`` reproduces today's filenames — asserted on the STRINGS.

    These three conventions are the on-disk contract with ~30k cached files in the
    workspace stores. A backend that "reproduces the naming" is only useful if the
    reproduction is checked against the literals, not against itself.
    """

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


# ---------------------------------------------------------------------------------------
# The record-level protocol
# ---------------------------------------------------------------------------------------


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
        # ...and it really is the historical file on disk, not an opaque blob
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


class TestResolveBackend:
    def test_none_means_the_configured_global_store(self, store):
        backend = resolve_backend(None)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == store / "Cu-Mg_MPDS_PD_0.json"

    def test_the_global_backend_tracks_later_config_changes(self, store, monkeypatch, tmp_path):
        """``set_cache_dir`` mid-session must be observed, as ``config.data_dir`` was."""
        backend = resolve_backend(None)
        moved = tmp_path / "moved"
        moved.mkdir()
        monkeypatch.setattr(config, "cache_dir", moved)
        assert backend.path_for(CacheKey("Cu-Mg", "mpds", "0")) == moved / "Cu-Mg_MPDS_PD_0.json"

    def test_a_path_means_a_FLAT_directory_backend(self, store, monkeypatch, tmp_path):
        """Today's explicit-``data_dir`` semantics: flat inside it, whatever the global mode."""
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
    """``api.resolve_cache_path`` — the supported replacement for ``_resolve_sys_dir``."""

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
    """End-to-end: the three api entry points write exactly the filenames they used to."""

    def test_cold_fetch_writes_the_entries_file(self, tmp_path, monkeypatch):
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        pool = [
            ComputedEntry(Composition(f), e, entry_id=f"s-{f}").as_dict()
            for f, e in (("Cu", 0.0), ("Mg", 0.0), ("CuMg2", -0.9))
        ]
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
