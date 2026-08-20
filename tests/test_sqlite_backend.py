"""The single-file store: ``SqliteBackend`` and the ``python -m gliquid.cache`` CLI.

Three properties carry the weight here, and each is asserted against something other than
the code that produced it:

* **Losslessness.** Binary DFT payloads are stored per system, NOT deduplicated across
  systems: measured over 300 real caches, 3.58% of repeated ``entry_id``s carry differing
  payloads (fetched over time against different Materials Project database versions), so
  pooling them would silently pick one of several disagreeing copies. The proof is the
  sha256 of the UNCOMPRESSED bytes, re-derived here from the source object rather than read
  back from the row that stored it.
* **Read-mostly by design.** ``write_json`` raises unless the store was opened
  ``writable=True``. SQLite has one writer and ``dev/scripts/Fit_Binary_Systems.py`` fans
  campaigns over a ``ProcessPoolExecutor``. The error must name the fix, so that is
  asserted on the message, not just the type.
* **The MPDS variant rule is unchanged by the storage change.** ``variant=''`` shadows
  ``'0'``, ``pd_ind=0`` resolves a sole indexless record, and a nonzero index with no match
  raises naming the reachable handles — the same rule the directory backend implements,
  exercised through ``mpds.load_mpds_data`` against a sqlite store.
"""

import json
import math
import sqlite3
import zlib

import pytest

import gliquid.config as config
from gliquid.cache import (
    DFT_CODEC,
    KIND_DFT_ENTRIES,
    KIND_MPDS,
    SQLITE_SCHEMA_VERSION,
    CacheKey,
    CacheModeError,
    DirectoryBackend,
    SqliteBackend,
    SqliteStoreError,
    close_sqlite_backends,
    looks_like_system_name,
    main,
    mpds_header,
    parse_record_filename,
    resolve_backend,
    scan_directory_store,
)

ENTRIES = [
    {"composition": {"Cu": 1.0}, "energy": -3.7, "entry_id": "mp-30"},
    {"composition": {"Cu": 1.0, "Mg": 2.0}, "energy": -11.2, "entry_id": "mp-1234"},
]
DIAGRAM = {
    "chemical_elements": ["Cu", "Mg"],
    "temp": [300.0, 1400.0],
    "comp_range": [0.0, 100.0],
    "labels": [["L", [22.0, 951.0], None]],
    "shapes": [{"kind": "phase", "svgpath": "M 0,0 L 1,1"}],
    "entry": "C900001",
    "jcode": "1234",
    "year": "1991",
    "reference": {"entry": "https://mpds.io/entry/C900001"},
}

DFT_KEY = CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")
MPDS_KEY = CacheKey("Cu-Mg", KIND_MPDS, "0")


@pytest.fixture(autouse=True)
def _no_leaked_stores():
    """Read-only stores are cached per path; close them so tmp_path teardown can unlink."""
    yield
    close_sqlite_backends()


@pytest.fixture
def corpus(tmp_path):
    """A small flat directory corpus: two DFT records, three MPDS records, one neighbour."""
    root = tmp_path / "cache"
    root.mkdir()
    (root / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(ENTRIES))
    (root / "Ag-V_ENTRIES_MP_GGA.json").write_text(json.dumps(ENTRIES[:1]))
    (root / "Cu-Mg_MPDS_PD_0.json").write_text(json.dumps(DIAGRAM))
    (root / "Cu-Mg_MPDS_PD_1.json").write_text(json.dumps({"reference": None}))
    (root / "Ag-V.json").write_text(json.dumps(DIAGRAM))
    # A neighbour that is NOT a cache record and must not be migrated as one.
    (root / "fit_results_cache_comb-exp.json").write_text(json.dumps({"whatever": 1}))
    return root


@pytest.fixture
def store(tmp_path, corpus):
    """``corpus`` migrated into a single-file store."""
    dest = tmp_path / "store.sqlite"
    assert main(["migrate", "--from", str(corpus), "--to", str(dest)]) == 0
    return dest


def _writable(path) -> SqliteBackend:
    return SqliteBackend(path, create=True)


# ---------------------------------------------------------------------------------------
# Filename <-> key, and what counts as a cache record at all
# ---------------------------------------------------------------------------------------


class TestRecordNameParsing:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Cu-Mg_ENTRIES_MP_GGA.json", CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")),
            ("Cu-Mg_MPDS_PD_0.json", CacheKey("Cu-Mg", KIND_MPDS, "0")),
            ("Cu-Mg.json", CacheKey("Cu-Mg", KIND_MPDS, "")),
        ],
    )
    def test_the_three_conventions_round_trip(self, name, expected):
        from gliquid.cache import record_filename

        assert parse_record_filename(name) == expected
        assert record_filename(expected) == name

    def test_an_entries_file_is_not_read_as_an_indexless_diagram(self):
        """POSITIVE CONTROL for ordering: ``<sys>.json`` must be tried LAST."""
        assert parse_record_filename("Cu-Mg_ENTRIES_MP_GGA.json").kind == KIND_DFT_ENTRIES

    def test_a_pinned_system_rejects_a_lookalike(self):
        assert parse_record_filename("Cu-Mg_notes.json", "Cu-Mg") is None
        assert parse_record_filename("Cu-Mg.json", "Cu-Mg") is not None  # control

    @pytest.mark.parametrize("name", ["Cu-Mg", "Al-Mg-Si", "CuMg-Mg", "Mg2Si-Si"])
    def test_system_shaped_names_are_accepted(self, name):
        assert looks_like_system_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "fit_results_cache_comb-exp",
            "all_feature_dft_data",
            "cohesive_energies",
            "composite_fit_results-trimmed",
            "Cu",
            "Xx-Yy",
        ],
    )
    def test_non_system_names_are_rejected(self, name):
        assert not looks_like_system_name(name)


class TestScanDirectoryStore:
    def test_flat_scan_finds_records_and_sets_aside_the_rest(self, corpus):
        scan = scan_directory_store(corpus, "flat")
        assert len(scan.records) == 5
        assert [p.name for p in scan.ignored] == ["fit_results_cache_comb-exp.json"]

    def test_nested_scan_uses_the_directory_as_the_system(self, tmp_path):
        (tmp_path / "Cu-Mg").mkdir()
        (tmp_path / "Cu-Mg" / "Cu-Mg_ENTRIES_MP_GGA.json").write_text("[]")
        (tmp_path / "Cu-Mg" / "Cu-Mg_MPDS_PD_0.png").write_bytes(b"not json")
        (tmp_path / "identifiability_plots").mkdir()
        scan = scan_directory_store(tmp_path, "nested")
        assert [k for k, _ in scan.records] == [CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")]
        assert sorted(p.name for p in scan.ignored) == [
            "Cu-Mg_MPDS_PD_0.png",
            "identifiability_plots",
        ]

    def test_agrees_with_what_the_directory_backend_would_serve(self, corpus):
        """The scan is only trustworthy if it sees exactly what a live read sees."""
        backend = DirectoryBackend(corpus)
        scanned = {(k.sys_name, k.kind, k.variant) for k, _ in scan_directory_store(corpus).records}
        served = {
            (sys_name, kind, variant)
            for sys_name in ("Cu-Mg", "Ag-V")
            for kind in (KIND_DFT_ENTRIES, KIND_MPDS)
            for variant in backend.variants(sys_name, kind)
        }
        assert scanned == served


# ---------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------


class TestSchema:
    def test_both_record_tables_are_without_rowid_and_carry_no_secondary_index(self, store):
        conn = sqlite3.connect(str(store))
        sql = dict(
            conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
        )
        # Only the RECORD tables matter. `meta` is an ordinary rowid table, so SQLite gives
        # its TEXT PRIMARY KEY an implicit sqlite_autoindex -- that is the PK itself, not a
        # secondary index, and it holds six rows.
        record_indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name IN ('dft_entries', 'mpds_diagrams')"
        ).fetchall()
        conn.close()
        for table in ("dft_entries", "mpds_diagrams"):
            assert "WITHOUT ROWID" in sql[table].upper(), f"{table} lost WITHOUT ROWID"
        assert record_indices == [], (
            "a secondary index appeared; every access is an exact point read on the "
            "composite primary key and WITHOUT ROWID already makes that key the storage"
        )

    def test_user_version_pins_the_schema(self, store):
        conn = sqlite3.connect(str(store))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION
        conn.close()

    def test_meta_records_the_codec_and_the_mpds_mode(self, store):
        meta = SqliteBackend(store).meta()
        assert meta["dft_codec"] == DFT_CODEC
        assert meta["mpds_mode"] == "full"
        assert meta["schema_version"] == str(SQLITE_SCHEMA_VERSION)
        assert meta["source_dir_structure"] == "flat"

    def test_a_newer_schema_is_refused_rather_than_misread(self, store):
        conn = sqlite3.connect(str(store))
        conn.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        with pytest.raises(SqliteStoreError, match="NEWER gliquid"):
            SqliteBackend(store).exists(DFT_KEY)

    def test_the_current_schema_version_is_accepted(self, store):
        """POSITIVE CONTROL for the test above: only a GREATER version is refused."""
        assert SqliteBackend(store).exists(DFT_KEY)

    def test_a_sqlite_file_that_is_not_a_gliquid_store_is_refused(self, tmp_path):
        alien = tmp_path / "alien.sqlite"
        conn = sqlite3.connect(str(alien))
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.commit()
        conn.close()
        with pytest.raises(SqliteStoreError, match="not a gliquid cache store"):
            SqliteBackend(alien).exists(DFT_KEY)

    def test_a_file_that_is_not_sqlite_at_all_is_refused(self, tmp_path):
        junk = tmp_path / "junk.sqlite"
        junk.write_text("this is not a database")
        with pytest.raises(SqliteStoreError):
            SqliteBackend(junk).exists(DFT_KEY)

    def test_a_missing_store_names_how_to_build_one(self, tmp_path):
        with pytest.raises(SqliteStoreError, match="gliquid.cache migrate"):
            SqliteBackend(tmp_path / "absent.sqlite").exists(DFT_KEY)


# ---------------------------------------------------------------------------------------
# Losslessness
# ---------------------------------------------------------------------------------------


class TestLosslessness:
    def test_dft_round_trip_is_object_identical(self, tmp_path):
        backend = _writable(tmp_path / "s.sqlite")
        backend.write_json(DFT_KEY, ENTRIES)
        assert backend.read_json(DFT_KEY) == ENTRIES
        backend.close()

    def test_the_stored_sha256_is_of_the_UNCOMPRESSED_bytes(self, tmp_path):
        """The losslessness proof, re-derived independently of the row that stored it."""
        import hashlib

        backend = _writable(tmp_path / "s.sqlite")
        backend.write_json(DFT_KEY, ENTRIES)
        expected = hashlib.sha256(json.dumps(ENTRIES).encode("utf-8")).hexdigest()
        assert backend.stored_sha256(DFT_KEY) == expected
        # ...and it is NOT the sha of the compressed blob (the easy way to get this wrong)
        conn = backend._conn()
        blob = conn.execute("SELECT payload FROM dft_entries").fetchone()[0]
        assert hashlib.sha256(blob).hexdigest() != expected
        assert json.loads(zlib.decompress(blob)) == ENTRIES
        backend.close()

    def test_entries_are_NOT_deduplicated_across_systems(self, tmp_path):
        """3.58% of repeated ``entry_id``s carry DIFFERING payloads across real caches.

        Two systems that share an ``entry_id`` but disagree about its energy must each read
        back their own copy. A pooled entry table would hand both of them one of the two.
        """
        backend = _writable(tmp_path / "s.sqlite")
        old = [{"entry_id": "mp-30", "energy": -3.70, "composition": {"Cu": 1.0}}]
        new = [{"entry_id": "mp-30", "energy": -3.99, "composition": {"Cu": 1.0}}]
        backend.write_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA"), old)
        backend.write_json(CacheKey("Ag-Cu", KIND_DFT_ENTRIES, "GGA"), new)
        assert backend.read_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")) == old
        assert backend.read_json(CacheKey("Ag-Cu", KIND_DFT_ENTRIES, "GGA")) == new
        backend.close()

    def test_mpds_full_mode_keeps_the_whole_record_including_shapes(self, tmp_path):
        backend = _writable(tmp_path / "s.sqlite")
        backend.write_json(MPDS_KEY, DIAGRAM)
        assert backend.read_json(MPDS_KEY) == DIAGRAM
        backend.close()

    def test_the_no_diagram_placeholder_is_a_record_not_an_absence(self, tmp_path):
        """``load_mpds_data`` caches ``{"reference": None}``; dropping it re-fetches forever."""
        backend = _writable(tmp_path / "s.sqlite")
        key = CacheKey("Cu-Mg", KIND_MPDS, "")
        backend.write_json(key, {"reference": None})
        assert backend.exists(key)
        assert backend.read_json(key) == {"reference": None}
        backend.close()

    def test_header_columns_are_lifted_from_either_level(self):
        header = mpds_header(DIAGRAM)
        assert header["chemical_elements"] == ["Cu", "Mg"]
        assert header["jcode"] == "1234"
        assert "shapes" not in header
        # entry lives under reference in some records and at the top level in others
        assert mpds_header({"reference": {"entry": "X"}})["entry"] == "X"
        assert mpds_header({"reference": None}) == {"reference": None}

    def test_nan_survives_the_round_trip(self, tmp_path):
        """``json`` accepts the non-standard NaN literal in both directions."""
        backend = _writable(tmp_path / "s.sqlite")
        backend.write_json(DFT_KEY, [{"energy": float("nan")}])
        assert math.isnan(backend.read_json(DFT_KEY)[0]["energy"])
        backend.close()


# ---------------------------------------------------------------------------------------
# The protocol surface
# ---------------------------------------------------------------------------------------


class TestProtocol:
    def test_exists_and_read_of_an_absent_record(self, store):
        backend = SqliteBackend(store)
        assert backend.exists(DFT_KEY)
        assert not backend.exists(CacheKey("Pu-Sn", KIND_DFT_ENTRIES, "GGA"))
        with pytest.raises(FileNotFoundError):
            backend.read_json(CacheKey("Pu-Sn", KIND_DFT_ENTRIES, "GGA"))

    def test_variants_lists_what_the_store_holds(self, store):
        backend = SqliteBackend(store)
        assert backend.variants("Cu-Mg", KIND_MPDS) == ["0", "1"]
        assert backend.variants("Ag-V", KIND_MPDS) == [""]
        assert backend.variants("Cu-Mg", KIND_DFT_ENTRIES) == ["GGA"]
        assert backend.variants("Pu-Sn", KIND_MPDS) == []

    def test_variants_does_not_leak_another_systems_records(self, store):
        backend = SqliteBackend(store)
        assert backend.variants("Ag-V", KIND_DFT_ENTRIES) == ["GGA"]  # control: found
        assert backend.variants("Ag-V", KIND_MPDS) == [""]  # and not Cu-Mg's 0/1

    def test_unknown_kind_is_rejected_rather_than_guessed(self, store):
        backend = SqliteBackend(store)
        for call in (
            lambda: backend.exists(CacheKey("Cu-Mg", "not_a_kind", "0")),
            lambda: backend.read_json(CacheKey("Cu-Mg", "not_a_kind", "0")),
            lambda: backend.variants("Cu-Mg", "not_a_kind"),
        ):
            with pytest.raises(ValueError, match="kind"):
                call()

    def test_capabilities_declare_no_paths(self, store):
        caps = SqliteBackend(store).capabilities
        assert {"read", "variants"} <= caps
        assert "write" not in caps
        assert "paths" not in caps, "a row has no filesystem path; callers must degrade"
        assert "write" in SqliteBackend(store, writable=True).capabilities

    def test_locate_names_the_store_and_the_record(self, store):
        located = SqliteBackend(store).locate(MPDS_KEY)
        assert str(store) in located
        assert located.endswith("Cu-Mg_MPDS_PD_0.json")

    def test_resolve_cache_path_is_None_for_a_single_file_store(self, store):
        import gliquid.api as api

        assert api.resolve_cache_path(DFT_KEY, cache=store) is None

    def test_resolve_sys_dir_refuses_rather_than_fabricating_a_directory(self, store):
        import warnings

        import gliquid.api as api

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(CacheModeError, match="one file"):
                api._resolve_sys_dir("Cu-Mg", data_dir=store)


class TestReadOnlyByDesign:
    def test_write_json_on_a_read_only_store_names_the_fix(self, store):
        with pytest.raises(CacheModeError) as exc:
            SqliteBackend(store).write_json(DFT_KEY, ENTRIES)
        message = str(exc.value)
        assert "READ-ONLY" in message
        assert "gliquid.cache migrate" in message, "the error must name how to rebuild"
        assert "set_cache_dir" in message, "the error must name the directory-store escape"

    def test_bulk_write_is_refused_too(self, store):
        with pytest.raises(CacheModeError):
            with SqliteBackend(store).bulk_write():
                pass  # pragma: no cover - the context manager raises on entry

    def test_set_meta_is_refused_too(self, store):
        with pytest.raises(CacheModeError):
            SqliteBackend(store).set_meta("mpds_mode", "lean")

    def test_a_writable_store_does_write(self, store):
        """POSITIVE CONTROL: the refusal above is about the MODE, not about writes failing."""
        backend = SqliteBackend(store, writable=True)
        backend.write_json(DFT_KEY, [{"entry_id": "new"}])
        assert backend.read_json(DFT_KEY) == [{"entry_id": "new"}]
        backend.close()

    def test_a_rewrite_replaces_rather_than_duplicating(self, tmp_path):
        backend = _writable(tmp_path / "s.sqlite")
        backend.write_json(DFT_KEY, ENTRIES)
        backend.write_json(DFT_KEY, ENTRIES[:1])
        assert backend.read_json(DFT_KEY) == ENTRIES[:1]
        assert backend._conn().execute("SELECT COUNT(*) FROM dft_entries").fetchone()[0] == 1
        backend.close()


# ---------------------------------------------------------------------------------------
# Wiring into config / resolve_backend
# ---------------------------------------------------------------------------------------


class TestResolveBackendUnderSqliteMode:
    def test_set_cache_dir_to_a_store_file_selects_the_sqlite_backend(self, store, monkeypatch):
        monkeypatch.setattr(config, "cache_dir", store)
        monkeypatch.setattr(config, "cache_mode", "sqlite")
        backend = resolve_backend(None)
        assert isinstance(backend, SqliteBackend)
        assert backend.read_json(DFT_KEY) == ENTRIES

    def test_an_explicit_store_path_override_also_resolves_to_sqlite(self, store):
        assert isinstance(resolve_backend(store), SqliteBackend)
        assert isinstance(resolve_backend(str(store)), SqliteBackend)

    def test_a_directory_override_is_still_a_directory_backend(self, corpus):
        """POSITIVE CONTROL: the historical explicit-``data_dir`` meaning is unchanged."""
        assert isinstance(resolve_backend(corpus), DirectoryBackend)

    def test_sqlite_mode_pointed_at_a_directory_raises_rather_than_guessing(
        self, corpus, monkeypatch
    ):
        monkeypatch.setattr(config, "cache_dir", corpus)
        monkeypatch.setattr(config, "cache_mode", "sqlite")
        with pytest.raises(CacheModeError, match="DIRECTORY"):
            resolve_backend(None)

    def test_the_store_is_opened_once_per_path(self, store, monkeypatch):
        monkeypatch.setattr(config, "cache_dir", store)
        monkeypatch.setattr(config, "cache_mode", "sqlite")
        assert resolve_backend(None) is resolve_backend(None)

    def test_set_dir_structure_under_sqlite_logs_and_does_not_raise(self, store, caplog):
        """Re-asserted here because spec 04 is the first real consumer of that decision.

        20+ driver scripts under ``dev/scripts`` call ``set_dir_structure`` unconditionally
        at import; raising would make a single-file store unusable from every one of them
        for a setting that simply does not apply.
        """
        import logging

        # set_cache_dir is a real global mutation (it also re-resolves the reference
        # tables), so the previous values are captured and put back rather than reset to a
        # hardcoded path -- restoring to a guess would silently UN-SWAP the rest of a
        # session that conftest.py had pointed at a single-file store.
        saved = (config.cache_dir, config.cache_mode, config.dir_structure)
        try:
            config.set_cache_dir(store)
            assert config.cache_mode == "sqlite"
            before = config.dir_structure
            with caplog.at_level(logging.INFO, logger="gliquid.config"):
                config.set_dir_structure("nested")
            assert config.dir_structure == before, "the knob must stay inert, not flip"
            assert any("sqlite" in r.message for r in caplog.records)
        finally:
            config.set_cache_dir(saved[0])
            config.set_cache_mode(saved[1])
            config.dir_structure = saved[2]
        assert (config.cache_dir, config.cache_mode, config.dir_structure) == saved


class TestMpdsVariantRuleIsUnchangedBySqlite:
    """The shadowing rule from spec 01, re-run against the single-file store."""

    @pytest.fixture
    def sqlite_store(self, tmp_path, monkeypatch):
        def build(records):
            path = tmp_path / f"mpds-{len(list(tmp_path.iterdir()))}.sqlite"
            backend = SqliteBackend(path, create=True)
            for variant in records:
                backend.write_json(CacheKey("Cu-Mg", KIND_MPDS, variant), dict(DIAGRAM))
            backend.close()
            monkeypatch.setattr(config, "cache_dir", path)
            monkeypatch.setattr(config, "cache_mode", "sqlite")
            return path

        return build

    def test_indexless_shadows_pd0_and_warns(self, sqlite_store, caplog):
        import logging

        import gliquid.mpds as mpds

        path = sqlite_store(["", "0"])
        with caplog.at_level(logging.WARNING, logger="gliquid.mpds"):
            mpds.load_mpds_data("Cu-Mg", pd_ind=None)
        assert any("shadows" in r.message for r in caplog.records)
        assert any(str(path) in r.message for r in caplog.records), (
            "the warning must name the store it resolved in"
        )

    def test_no_warning_when_only_the_indexless_record_exists(self, sqlite_store, caplog):
        """POSITIVE CONTROL: indexless alone is the legitimate single-diagram convention."""
        import logging

        import gliquid.mpds as mpds

        sqlite_store([""])
        with caplog.at_level(logging.WARNING, logger="gliquid.mpds"):
            mpds.load_mpds_data("Cu-Mg", pd_ind=None)
        assert not any("shadows" in r.message for r in caplog.records)

    def test_pd_ind_0_resolves_a_sole_indexless_record(self, sqlite_store):
        import gliquid.mpds as mpds

        sqlite_store([""])
        loaded, _ = mpds.load_mpds_data("Cu-Mg", pd_ind=0)
        assert loaded["entry"] == "C900001", "pd_ind=0 must reach the indexless record"

    def test_a_nonzero_index_with_no_match_raises_naming_the_reachable_handles(self, sqlite_store):
        import gliquid.mpds as mpds

        sqlite_store([""])
        with pytest.raises(ValueError, match="pd_ind=0 or pd_ind=None"):
            mpds.load_mpds_data("Cu-Mg", pd_ind=3)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


class TestCli:
    def test_migrate_then_verify_reports_zero_mismatches(self, corpus, tmp_path, capsys):
        dest = tmp_path / "out.sqlite"
        assert main(["migrate", "--from", str(corpus), "--to", str(dest)]) == 0
        capsys.readouterr()
        assert main(["verify", "--directory", str(corpus), "--sqlite", str(dest)]) == 0
        out = capsys.readouterr().out
        assert "keys compared        : 5" in out
        assert "object mismatches    : 0" in out
        assert "dft sha256 mismatches: 0" in out

    def test_migrate_refuses_to_clobber_an_existing_store(self, corpus, store):
        assert main(["migrate", "--from", str(corpus), "--to", str(store)]) == 2
        assert main(["migrate", "--from", str(corpus), "--to", str(store), "--overwrite"]) == 0

    def test_migrate_reports_and_refuses_an_unparseable_source(self, corpus, tmp_path, capsys):
        """A truncated cache file is a wrong answer waiting to happen; skipping hides it."""
        truncated = corpus / "Ce-Ge-Ni_ENTRIES_MP_GGA.json"
        truncated.write_text(json.dumps(ENTRIES)[:40])
        dest = tmp_path / "out.sqlite"
        assert main(["migrate", "--from", str(corpus), "--to", str(dest)]) == 1
        out = capsys.readouterr().out
        assert "Ce-Ge-Ni_ENTRIES_MP_GGA.json" in out
        assert "REFUSED" in out
        assert not dest.exists(), "a refused migration must leave no store at all"
        assert not dest.with_name(dest.name + ".partial").exists()

    def test_skip_unparseable_is_explicit_and_still_reports(self, corpus, tmp_path, capsys):
        (corpus / "Ce-Ge-Ni_ENTRIES_MP_GGA.json").write_text(json.dumps(ENTRIES)[:40])
        dest = tmp_path / "out.sqlite"
        assert (
            main(["migrate", "--from", str(corpus), "--to", str(dest), "--skip-unparseable"]) == 0
        )
        out = capsys.readouterr().out
        assert "could not be parsed" in out
        assert "SKIPPED" in out
        assert dest.exists()

    def test_migrate_of_an_empty_scan_fails_loudly(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["migrate", "--from", str(empty), "--to", str(tmp_path / "o.sqlite")]) == 1
        assert "No cache records found" in capsys.readouterr().out

    def test_the_wrong_dir_structure_fails_loudly_and_names_the_right_one(self, tmp_path, capsys):
        """The silent near-miss this guard exists for: a nested corpus read as flat."""
        root = tmp_path / "nested"
        (root / "Cu-Mg").mkdir(parents=True)
        (root / "Cu-Mg" / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(ENTRIES))
        assert main(["migrate", "--from", str(root), "--to", str(tmp_path / "o.sqlite")]) == 1
        out = capsys.readouterr().out
        assert "--dir-structure nested would find 1" in out

    def test_nested_migrate_round_trips(self, tmp_path, capsys):
        root = tmp_path / "nested"
        (root / "Cu-Mg").mkdir(parents=True)
        (root / "Cu-Mg" / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(ENTRIES))
        (root / "Cu-Mg" / "Cu-Mg.json").write_text(json.dumps(DIAGRAM))
        dest = tmp_path / "o.sqlite"
        args = ["--from", str(root), "--to", str(dest), "--dir-structure", "nested"]
        assert main(["migrate", *args]) == 0
        capsys.readouterr()
        # verify picks the layout up from meta, without being told again
        assert main(["verify", "--directory", str(root), "--sqlite", str(dest)]) == 0
        assert "keys compared        : 2" in capsys.readouterr().out

    def test_verify_catches_a_tampered_record(self, corpus, store, capsys):
        backend = SqliteBackend(store, writable=True)
        backend.write_json(DFT_KEY, [{"entry_id": "wrong"}])
        backend.close()
        assert main(["verify", "--directory", str(corpus), "--sqlite", str(store)]) == 1
        out = capsys.readouterr().out
        assert "object mismatches    : 1" in out
        assert "dft sha256 mismatches: 1" in out

    def test_verify_catches_a_missing_record(self, corpus, store, capsys):
        backend = SqliteBackend(store, writable=True)
        backend._conn().execute("DELETE FROM dft_entries WHERE sys_name = 'Cu-Mg'")
        backend.close()
        assert main(["verify", "--directory", str(corpus), "--sqlite", str(store)]) == 1
        assert "missing from sqlite  : 1" in capsys.readouterr().out

    def test_verify_catches_an_extra_record(self, corpus, store, capsys):
        backend = SqliteBackend(store, writable=True)
        backend.write_json(CacheKey("Pu-Sn", KIND_DFT_ENTRIES, "GGA"), ENTRIES)
        backend.close()
        assert main(["verify", "--directory", str(corpus), "--sqlite", str(store)]) == 1
        assert "extra in sqlite      : 1" in capsys.readouterr().out

    def test_lean_mpds_mode_writes_lean_rows(self, corpus, tmp_path, capsys):
        """``--mpds-mode lean`` landed with the lean-record contract; DFT records are
        untouched by it. The reduction itself is pinned in test_mpds_lean_mode.py."""
        dest = tmp_path / "o.sqlite"
        args = ["--from", str(corpus), "--to", str(dest), "--mpds-mode", "lean"]
        assert main(["migrate", *args]) == 0
        assert "(mode=lean)" in capsys.readouterr().out
        backend = SqliteBackend(dest)
        try:
            assert backend.meta()["mpds_mode"] == "lean"
            assert backend.read_json(DFT_KEY) == ENTRIES, "DFT records are unaffected"
        finally:
            backend.close()

    def test_an_unknown_mpds_mode_is_refused_by_name(self, corpus, tmp_path, capsys):
        dest = tmp_path / "o.sqlite"
        args = ["--from", str(corpus), "--to", str(dest), "--mpds-mode", "svelte"]
        assert main(["migrate", *args]) == 2
        assert "not a known mode" in capsys.readouterr().out

    def test_info_describes_the_store(self, store, capsys):
        assert main(["info", str(store)]) == 0
        out = capsys.readouterr().out
        assert f"user_version         : {SQLITE_SCHEMA_VERSION}" in out
        assert "records                 : 2" in out  # dft_entries
        assert "mode full" in out
        assert "indexless (<sys>.json)  : 1" in out

    def test_info_on_a_non_store_reports_rather_than_tracebacks(self, tmp_path, capsys):
        junk = tmp_path / "junk.sqlite"
        junk.write_text("nope")
        assert main(["info", str(junk)]) == 2
        assert capsys.readouterr().out.startswith("ERROR: ")
