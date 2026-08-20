"""The pooled entry store: ``entry_pool`` and its coverage record.

Pooling deduplicates DFT entries by ``entry_id`` across systems and reassembles a system's
record as ``chemsys IN (<its non-empty subsets>)``. It is enabled for TERNARIES only, and
only because a drift census says so: 0.00% payload drift over 6,241 repeated ids in the 109
existing ternary caches, against 3.58% in the binary corpus. The tests here pin the three
things that make that reassembly trustworthy rather than merely small:

* **The query shape is the fetch's shape.** ``chemsys_subsets`` returns exactly the groups
  ``MPRester.get_entries_in_chemsys`` enumerates -- 7 for a ternary, 3 for a binary. A
  narrower query would return a smaller entry set than the fetch did, and the hull built
  from it would be quietly missing phases.
* **Coverage is recorded, never inferred.** A system whose entries all happen to be in the
  pool, because its neighbours contributed them, must still read as ABSENT unless it was
  actually pooled. This is the single most dangerous failure available here: the pool would
  return a plausible, non-empty entry list with every ternary compound missing, and the
  hull would be wrong with nothing raising.
* **Drift is reported, never resolved.** Two payloads for one ``entry_id`` is the condition
  that rules pooling out; ``write_pool_entries`` keeps the first and NAMES the conflict
  rather than picking a winner.

The end-to-end reconstruction against real data (109 systems, hull by hull) lives in
``dev/tests/test_ternary_entry_pool.py``, next to the corpus it needs.
"""

import json
import sqlite3

import pytest
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.config as config
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    KIND_MPDS,
    CacheKey,
    CacheModeError,
    SqliteBackend,
    canonical_entry_list,
    chemsys_key,
    chemsys_subsets,
    close_sqlite_backends,
    entry_chemsys,
    entry_list_sha256,
    entry_pool_id,
    main,
    system_elements,
)


# A miniature Ag-Cu-Zr: three elementals, three binaries, one true ternary. Energies are
# arranged so every entry is on the hull, which makes a dropped entry visible as a changed
# hull rather than as nothing at all.
def _entry(composition, energy, entry_id):
    """One cached entry, in the shape ``ComputedEntry.from_dict`` accepts.

    ``correction`` and ``energy_adjustments`` are not decoration: pymatgen raises
    ``KeyError: 'correction'`` without them, so an entry dict missing them would never
    reach a hull and these tests would be exercising a shape the corpus does not hold.
    """
    return {
        "@module": "pymatgen.entries.computed_entries",
        "@class": "ComputedEntry",
        "composition": composition,
        "energy": energy,
        "entry_id": entry_id,
        "correction": 0.0,
        "energy_adjustments": [],
    }


AG = _entry({"Ag": 1.0}, -2.8, "mp-124-GGA")
CU = _entry({"Cu": 1.0}, -4.1, "mp-30-GGA")
ZR = _entry({"Zr": 1.0}, -8.5, "mp-131-GGA")
AGCU = _entry({"Ag": 1.0, "Cu": 1.0}, -7.4, "mp-1001-GGA")
AGZR = _entry({"Ag": 1.0, "Zr": 2.0}, -20.5, "mp-1002-GGA")
CUZR = _entry({"Cu": 1.0, "Zr": 1.0}, -13.4, "mp-1003-GGA")
# The new-API shape: an entry_id that is a DICT, which a naive str() would key by its repr
# and a naive hash would refuse outright.
AGCUZR = _entry(
    {"Ag": 1.0, "Cu": 1.0, "Zr": 1.0}, -16.5, {"identifier": "mp-aaaaaaeu", "database_IDs": {}}
)
TERNARY_ENTRIES = [AG, CU, ZR, AGCU, AGZR, CUZR, AGCUZR]
TERNARY_SYS = "Ag-Cu-Zr"


def _create_plain_store(path):
    """A store with the base schema and NO pool, actually written to disk.

    ``SqliteBackend`` opens its connection lazily, so constructing one with ``create=True``
    and immediately closing it leaves no file at all -- the schema is created by the first
    query, not by ``__init__``. Touching ``meta()`` is what forces it.
    """
    backend = SqliteBackend(path, create=True)
    backend.meta()
    backend.close()
    return path


@pytest.fixture(autouse=True)
def _no_leaked_stores():
    yield
    close_sqlite_backends()


@pytest.fixture
def pooled_store(tmp_path):
    """A store holding ONLY a pool, covering Ag-Cu-Zr."""
    path = tmp_path / "pool.sqlite"
    backend = SqliteBackend(path, create=True)
    backend.ensure_entry_pool()
    backend.write_pool_entries(TERNARY_ENTRIES)
    backend.record_pool_system(TERNARY_SYS, "GGA", TERNARY_ENTRIES, source="test")
    backend.set_pool_meta("element_scope", json.dumps(["Ag", "Cu", "Zr"]))
    backend.close()
    return path


class TestChemsysQueryShape:
    """The read query must enumerate exactly what the fetch enumerated."""

    def test_ternary_has_seven_subsets(self):
        assert chemsys_subsets(["Cu", "Ag", "Zr"]) == [
            "Ag",
            "Ag-Cu",
            "Ag-Cu-Zr",
            "Ag-Zr",
            "Cu",
            "Cu-Zr",
            "Zr",
        ]

    def test_binary_has_three_subsets(self):
        assert chemsys_subsets(["Mg", "Cu"]) == ["Cu", "Cu-Mg", "Mg"]

    def test_unary_has_one_subset(self):
        assert chemsys_subsets(["Cu"]) == ["Cu"]

    def test_subset_count_is_two_to_the_n_minus_one(self):
        for elements in (["A"], ["Cu", "Mg"], ["Ag", "Cu", "Zr"], ["Ag", "Cu", "Mg", "Zr"]):
            assert len(chemsys_subsets(elements)) == 2 ** len(set(elements)) - 1

    def test_duplicate_elements_collapse(self):
        assert chemsys_subsets(["Cu", "Cu", "Mg"]) == chemsys_subsets(["Cu", "Mg"])

    def test_chemsys_key_is_sorted_and_deduplicated(self):
        assert chemsys_key(["Zr", "Ag", "Cu"]) == "Ag-Cu-Zr"
        assert chemsys_key(["Cu", "Cu"]) == "Cu"


class TestEntryIdentity:
    def test_string_entry_id(self):
        assert entry_pool_id(CU) == "mp-30-GGA"

    def test_dict_entry_id_is_keyed_canonically(self):
        """Two spellings of one new-API id must collide, not become two rows."""
        reordered = _entry(
            AGCUZR["composition"], AGCUZR["energy"], {"database_IDs": {}, "identifier": "mp-aaaaaaeu"}
        )
        assert entry_pool_id(reordered) == entry_pool_id(AGCUZR)

    @pytest.mark.parametrize("bad", [{}, {"entry_id": None}, {"entry_id": ""}])
    def test_unkeyable_entry_returns_none(self, bad):
        assert entry_pool_id(bad) is None

    def test_chemsys_ignores_zero_amounts(self):
        """An explicit 0.0 must not push an entry into a chemsys it does not belong to."""
        assert entry_chemsys({"composition": {"Cu": 1.0, "Mg": 0.0}}) == "Cu"

    def test_system_elements_expands_compound_components(self):
        assert system_elements("CuMg-Mg") == ["Cu", "Mg"]
        assert system_elements("Ag-Cu-Zr") == ["Ag", "Cu", "Zr"]

    def test_system_elements_rejects_a_non_system(self):
        assert system_elements("fit_results_cache_comb-exp") == []


class TestPoolRoundTrip:
    def test_pooled_read_reproduces_the_source_list(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        pooled = backend.read_json(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA"))
        assert entry_list_sha256(pooled) == entry_list_sha256(TERNARY_ENTRIES)
        assert len(pooled) == len(TERNARY_ENTRIES)
        backend.close()

    def test_pooled_read_is_a_stable_ORDER_not_merely_a_set(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        key = CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA")
        assert backend.read_json(key) == backend.read_json(key)
        assert backend.read_json(key) == canonical_entry_list(TERNARY_ENTRIES)
        backend.close()

    def test_exists_is_true_for_a_pooled_system(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        # get_dft_convexhull decides whether to FETCH on this answer alone.
        assert backend.exists(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA")) is True
        backend.close()

    def test_variants_agrees_with_exists(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        assert backend.variants(TERNARY_SYS, KIND_DFT_ENTRIES) == ["GGA"]
        backend.close()

    def test_mpds_reads_are_untouched_by_the_pool(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        assert backend.exists(CacheKey(TERNARY_SYS, KIND_MPDS, "0")) is False
        backend.close()


class TestCoverageIsRecordedNotInferred:
    """The pool's most dangerous failure mode, pinned.

    ``Ag-Cu`` and ``Ag-Cu-Ti`` both have entries sitting in the pool -- contributed by
    Ag-Cu-Zr -- yet neither was ever fetched. Answering them from those entries would hand
    back a hull with every unseen phase missing and nothing raising.
    """

    def test_an_unfetched_subsystem_reads_as_absent(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        key = CacheKey("Ag-Cu", KIND_DFT_ENTRIES, "GGA")
        assert backend.pool_covers("Ag-Cu", "GGA") is False
        assert backend.exists(key) is False
        with pytest.raises(FileNotFoundError):
            backend.read_json(key)
        backend.close()

    def test_an_unfetched_superset_reads_as_absent(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        key = CacheKey("Ag-Cu-Ti", KIND_DFT_ENTRIES, "GGA")
        assert backend.exists(key) is False
        with pytest.raises(FileNotFoundError):
            backend.read_json(key)
        backend.close()

    def test_the_entries_ARE_present_so_the_refusal_is_not_vacuous(self, pooled_store):
        """Without this, the two tests above would pass on an empty pool."""
        backend = SqliteBackend(pooled_store)
        reachable = backend.read_pool_entries(["Ag", "Cu"])
        assert {entry_pool_id(e) for e in reachable} == {
            entry_pool_id(e) for e in (AG, CU, AGCU)
        }
        backend.close()

    def test_a_different_dft_type_is_not_covered(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        assert backend.exists(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "R2SCAN")) is False
        backend.close()


class TestBinaryReadThroughTheSubsetQuery:
    """Acceptance: a binary served by the 3-subset ``IN`` matches the per-system blob.

    This proves the query SHAPE, and deliberately changes nothing about where binaries are
    actually read from -- they stay per-system blobs, because their measured 3.58% payload
    drift is exactly what pooling cannot survive.
    """

    def test_three_subset_read_equals_the_edge_of_the_source_record(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        from_pool = backend.read_pool_entries(["Ag", "Cu"])
        expected = [e for e in TERNARY_ENTRIES if entry_chemsys(e) in chemsys_subsets(["Ag", "Cu"])]
        assert entry_list_sha256(from_pool) == entry_list_sha256(expected)
        # ... and it must EXCLUDE everything containing Zr, or the "subset" query is a scan.
        assert all("Zr" not in (e.get("composition") or {}) for e in from_pool)
        backend.close()

    def test_a_blob_wins_over_the_pool_when_both_are_present(self, tmp_path):
        """Precedence, asserted rather than assumed.

        The blob is what a fetch actually returned for that system; the pool is a
        reassembly. Where both exist the un-reassembled one must win.
        """
        path = tmp_path / "both.sqlite"
        backend = SqliteBackend(path, create=True)
        backend.ensure_entry_pool()
        backend.write_pool_entries(TERNARY_ENTRIES)
        backend.record_pool_system(TERNARY_SYS, "GGA", TERNARY_ENTRIES, source="test")
        blob = [dict(CU, energy=-999.0)]
        backend.write_json(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA"), blob)
        backend.close()

        reader = SqliteBackend(path)
        assert reader.read_json(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA")) == blob
        reader.close()


class TestDriftIsReportedNotResolved:
    def test_a_conflicting_payload_is_named_and_the_first_kept(self, tmp_path):
        backend = SqliteBackend(tmp_path / "drift.sqlite", create=True)
        backend.ensure_entry_pool()
        backend.write_pool_entries([CU])
        written, drifted, unkeyable = backend.write_pool_entries([dict(CU, energy=-4.2)])
        assert written == 0
        assert drifted == ["mp-30-GGA"]
        assert unkeyable == []
        # The FIRST payload survives; nothing silently picked the newer one.
        assert backend.read_pool_entries(["Cu"]) == [CU]
        backend.close()

    def test_an_identical_repeat_is_not_drift(self, tmp_path):
        backend = SqliteBackend(tmp_path / "repeat.sqlite", create=True)
        backend.ensure_entry_pool()
        backend.write_pool_entries([CU])
        written, drifted, _ = backend.write_pool_entries([dict(CU)])
        assert (written, drifted) == (0, [])
        backend.close()

    def test_key_order_is_not_drift(self, tmp_path):
        """The same entry with its dict keys in another order is the same entry."""
        backend = SqliteBackend(tmp_path / "order.sqlite", create=True)
        backend.ensure_entry_pool()
        backend.write_pool_entries([CU])
        reordered = dict(reversed(list(CU.items())))
        _, drifted, _ = backend.write_pool_entries([reordered])
        assert drifted == []
        backend.close()

    def test_an_entry_without_an_id_is_reported_and_not_stored(self, tmp_path):
        backend = SqliteBackend(tmp_path / "unkeyed.sqlite", create=True)
        backend.ensure_entry_pool()
        orphan = {k: v for k, v in CU.items() if k != "entry_id"}
        written, drifted, unkeyable = backend.write_pool_entries([CU, orphan])
        assert written == 1
        assert unkeyable == [orphan]
        assert backend.read_pool_entries(["Cu"]) == [CU]
        backend.close()


class TestPoolIsOptional:
    def test_a_store_without_a_pool_says_so(self, tmp_path):
        backend = SqliteBackend(tmp_path / "plain.sqlite", create=True)
        assert backend.has_entry_pool is False
        assert backend.pool_meta() == {}
        assert backend.pool_stats() == {}
        assert backend.pool_systems() == []
        assert backend.pool_covers("Ag-Cu-Zr", "GGA") is False
        backend.close()

    def test_a_plain_store_has_no_pool_tables_at_all(self, tmp_path):
        """`has_entry_pool` must be about the FILE, not about this build of gliquid."""
        path = tmp_path / "plain.sqlite"
        _create_plain_store(path)
        conn = sqlite3.connect(str(path))
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        conn.close()
        assert "entry_pool" not in tables
        assert "dft_entries" in tables  # positive control: the schema WAS created

    def test_ensure_entry_pool_is_idempotent(self, tmp_path):
        backend = SqliteBackend(tmp_path / "twice.sqlite", create=True)
        backend.ensure_entry_pool()
        backend.write_pool_entries([CU])
        backend.ensure_entry_pool()
        assert backend.read_pool_entries(["Cu"]) == [CU]
        backend.close()

    def test_writing_a_pool_to_a_read_only_store_raises(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        with pytest.raises(CacheModeError):
            backend.write_pool_entries([CU])
        with pytest.raises(CacheModeError):
            backend.ensure_entry_pool()
        backend.close()


class TestStats:
    def test_stats_count_entries_by_arity(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        stats = backend.pool_stats()
        assert stats["entries"] == len(TERNARY_ENTRIES)
        assert stats["by_n_elements"] == {1: 3, 2: 3, 3: 1}
        assert stats["systems_covered"] == 1
        assert stats["entry_rows_reconstructed"] == len(TERNARY_ENTRIES)
        backend.close()

    def test_recorded_sha256_matches_a_pooled_read(self, pooled_store):
        backend = SqliteBackend(pooled_store)
        (_, _, _, recorded, _), = backend.pool_systems()
        pooled = backend.read_json(CacheKey(TERNARY_SYS, KIND_DFT_ENTRIES, "GGA"))
        assert recorded == entry_list_sha256(pooled)
        backend.close()


class TestCli:
    def test_info_reports_the_pool(self, pooled_store, capsys):
        assert main(["info", str(pooled_store)]) == 0
        out = capsys.readouterr().out
        assert "entry_pool" in out
        assert "systems covered         : 1" in out
        assert "element_scope" in out

    def test_info_on_a_poolless_store_prints_no_pool_section(self, tmp_path, capsys):
        path = tmp_path / "plain.sqlite"
        _create_plain_store(path)
        assert main(["info", str(path)]) == 0
        out = capsys.readouterr().out
        assert "entry_pool" not in out
        assert "mpds_diagrams" in out  # positive control: info DID run


class TestThroughTheApi:
    """The reason the pool exists: a hull, from a pool, with no network available."""

    def test_get_dft_convexhull_reads_a_pooled_ternary(self, pooled_store, monkeypatch):
        monkeypatch.setattr(config, "offline", True)  # any fetch attempt would now raise
        backend = SqliteBackend(pooled_store)
        hull, _ = api.get_dft_convexhull(["Ag", "Cu", "Zr"], "GGA", data_dir=backend)
        stable = {api.entry_display_name(e) for e in hull.stable_entries}
        # Every pooled entry reappears on the hull -- including the TRUE ternary, which is
        # the only part of the record no binary edge could have supplied.
        expected = {Composition(e["composition"]).reduced_formula for e in TERNARY_ENTRIES}
        assert stable == expected
        assert Composition(AGCUZR["composition"]).reduced_formula in stable
        backend.close()

    def test_an_uncovered_system_still_raises_rather_than_inventing_a_hull(
        self, pooled_store, monkeypatch
    ):
        monkeypatch.setattr(config, "offline", True)
        backend = SqliteBackend(pooled_store)
        with pytest.raises(config.OfflineError):
            api.get_dft_convexhull(["Ag", "Cu", "Ti"], "GGA", data_dir=backend)
        backend.close()
