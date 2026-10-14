"""Tests for gliquid.api: env-safe key resolution and lazy clients.

All tests run offline. Client-import hygiene is checked in the mixing-scheme subprocess, so
module state from the rest of the suite cannot mask an eager import regression.
"""

import json
import os
import sys
import types

import pytest

import gliquid.api as api
from gliquid.cache import KIND_DFT_ENTRIES, KIND_IMPUTED_ENTRIES, CacheKey, is_imputed_entry_dict

from ._fresh import ok, run_fresh
from ._mpds_builders import FakeMPRester, install_fake_mprester
from ._mpds_builders import cu_mg_entry_pool as _entry_pool


@pytest.fixture(autouse=True)
def _reset_api_module_state(monkeypatch):
    """Isolate the module-level singleton and one-shot dotenv flag per test."""
    monkeypatch.setattr(api, "_mpr", None)
    monkeypatch.setattr(api, "_dotenv_loaded", False)
    yield


class TestLoadDotenv:
    def test_never_overwrites_existing_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GLIQ_TEST_KEY", "from-environment")
        env_file = tmp_path / ".env"
        env_file.write_text("GLIQ_TEST_KEY=from-dotenv\n")
        api.load_dotenv(env_file)
        assert os.environ["GLIQ_TEST_KEY"] == "from-environment"

    def test_sets_missing_keys_and_strips_quotes(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GLIQ_TEST_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text('# comment line\nnot a kv line\nGLIQ_TEST_KEY="quoted-value"\n')
        api.load_dotenv(env_file)
        assert os.environ["GLIQ_TEST_KEY"] == "quoted-value"
        monkeypatch.delenv("GLIQ_TEST_KEY", raising=False)

    def test_missing_file_is_noop(self, tmp_path):
        api.load_dotenv(tmp_path / "does-not-exist.env")


class TestGetApiKey:
    def test_environment_wins(self, monkeypatch):
        monkeypatch.setenv("GLIQ_TEST_KEY", "env-value")
        assert api.get_api_key("GLIQ_TEST_KEY") == "env-value"

    def test_dotenv_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GLIQ_TEST_KEY", raising=False)
        (tmp_path / ".env").write_text("GLIQ_TEST_KEY=dotenv-value\n")
        monkeypatch.setattr(api.config, "project_root", tmp_path)
        assert api.get_api_key("GLIQ_TEST_KEY") == "dotenv-value"
        monkeypatch.delenv("GLIQ_TEST_KEY", raising=False)

    def test_missing_everywhere_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GLIQ_TEST_KEY", raising=False)
        monkeypatch.setattr(api.config, "project_root", tmp_path)  # no .env here
        assert api.get_api_key("GLIQ_TEST_KEY") is None


class TestLazyClients:
    def test_get_mpr_raises_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv(api.MP_KEY_VAR, raising=False)
        monkeypatch.setattr(api.config, "project_root", tmp_path)
        with pytest.raises(ValueError, match=api.MP_KEY_VAR):
            api.get_mpr()

    def test_mp_rester_raises_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv(api.MP_KEY_VAR, raising=False)
        monkeypatch.setattr(api.config, "project_root", tmp_path)
        with pytest.raises(ValueError, match=api.MP_KEY_VAR):
            api.mp_rester()

    def test_get_mpds_client_fails_cleanly_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv(api.MPDS_KEY_VAR, raising=False)
        monkeypatch.setattr(api.config, "project_root", tmp_path)
        # ValueError when mpds-client is installed; ImportError when it is not.
        with pytest.raises((ValueError, ImportError)):
            api.get_mpds_client()

    def test_mpds_api_error_is_exception_type(self):
        assert issubclass(api.mpds_api_error(), Exception)


class TestLegacyMontyModuleAliases:
    """The legacy pymatgen paths stay importable for UPSTREAM monty decodes: the Materials
    Project tags nested energy_adjustments records with `pymatgen.core.entries`, and
    MPRester.get_entries decodes them on a cache miss.
    """

    def test_legacy_paths_are_importable(self):
        from importlib import import_module

        for legacy in api._LEGACY_MONTY_MODULES:
            assert import_module(legacy) is not None

    def test_monty_decodes_a_legacy_tagged_payload(self):
        from monty.json import MontyDecoder

        payload = {
            "@module": "pymatgen.core.entries",
            "@class": "ConstantEnergyAdjustment",
            "value": -1.0,
            "uncertainty": 0.0,
            "name": "MP2020",
            "cls": {},
            "description": "",
        }
        assert MontyDecoder().process_decoded(payload) is not None

    def test_alias_targets_resolve_to_a_module_that_can_decode(self):
        """Each legacy path imports and carries the classes monty looks for on it. On a
        pymatgen that still ships the path, it is its own module, not the alias target.
        """
        import sys

        for legacy in api._LEGACY_MONTY_MODULES:
            module = sys.modules[legacy]
            assert module is not None
            # monty resolves @class off whatever this path points at.
            assert any(
                hasattr(module, name)
                for name in ("ComputedEntry", "ComputedStructureEntry", "ConstantEnergyAdjustment")
            ), f"{legacy} -> {module.__name__} exposes none of the entry classes monty needs"


class TestChemsysQueryShape:
    """The MPRester chemsys query is expressed in ELEMENT symbols: a compound component
    passed verbatim (e.g. 'CuMg') would query a nonexistent chemsys and return only the
    elemental entries.
    """

    ENTRIES = [
        ("Cu", 0.0),
        ("Mg", 0.0),
        ("CuMg", -0.8),
        ("CuMg2", -0.9),
    ]

    def _fake_rester(self, monkeypatch):
        """Stub MPRester whose chemsys expansion mirrors mp_api's, so the query shape matters."""
        import itertools

        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        pool = [
            ComputedEntry(Composition(f), e, entry_id=f"s-{f}").as_dict() for f, e in self.ENTRIES
        ]
        seen = []

        def fetch(elements, additional_criteria):
            seen.append(list(elements))
            unique = sorted(set(elements))
            allowed = {
                frozenset(combo)
                for n in range(1, len(unique) + 1)
                for combo in itertools.combinations(unique, n)
            }
            return [
                e for e in pool if frozenset(Composition(e["composition"]).as_dict()) in allowed
            ]

        install_fake_mprester(monkeypatch, fetch)
        return seen

    def test_elemental_components_query_the_same_elements(self, monkeypatch):
        seen = self._fake_rester(monkeypatch)
        fetched = api._get_dft_entries_from_components(["Mg", "Cu"], "GGA")
        assert len(seen) == 1
        assert set(seen[0]) == {"Cu", "Mg"}
        assert len(fetched) == len(self.ENTRIES)

    def test_compound_components_expand_to_their_elements(self, monkeypatch):
        seen = self._fake_rester(monkeypatch)
        api._get_dft_entries_from_components(["CuMg", "Mg"], "GGA")
        # NOT ['CuMg', 'Mg'] -- 'CuMg' is not a chemical system.
        assert seen == [["Cu", "Mg"]]

    def test_every_queried_symbol_is_a_single_element(self, monkeypatch):
        from pymatgen.core import Composition

        seen = self._fake_rester(monkeypatch)
        api._get_dft_entries_from_components(["CuMg", "Mg"], "GGA")
        assert all(len(Composition(sym).elements) == 1 for sym in seen[0])

    def test_cold_compound_fetch_builds_a_hull_and_caches_it(self, monkeypatch, tmp_path):
        """End-to-end: guards against the partial fetch poisoning the cache AND breaking the hull."""
        from pymatgen.core import Composition

        self._fake_rester(monkeypatch)
        ch, _ = api.get_dft_convexhull(["CuMg", "Mg"], "GGA", data_dir=tmp_path)
        assert [Composition(c) for c in api.pd_components(ch)] == [
            Composition("CuMg"),
            Composition("Mg"),
        ]
        cache = tmp_path / "CuMg-Mg_ENTRIES_MP_GGA.json"
        assert cache.exists()
        cached = json.loads(cache.read_text())
        comps = {Composition(e["composition"]).reduced_composition for e in cached}
        missing = [
            str(c)
            for c in (Composition("CuMg"), Composition("Mg"))
            if c.reduced_composition not in comps
        ]
        assert not missing, f"terminal entries missing from the cached fetch: {missing}"


class TestMixingSchemeImportIsBranchLocal:
    """Only the MIXED branch may need ``pymatgen.entries.mixing_scheme``: GGA and R2SCAN cold
    fetches survive its import failing. Blocked at the import system in a subprocess, which
    also checks that importing ``gliquid.api`` loads no remote client.
    """

    SCRIPT = """
import sys

import gliquid.api as api

leaked = [m for m in ("mp_api", "mpds_client", "requests") if m in sys.modules]
assert not leaked, f"eagerly imported: {leaked}"

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

POOL = [
    ComputedEntry(Composition(f), e, entry_id=f"s-{f}").as_dict()
    for f, e in (("Cu", 0.0), ("Mg", 0.0), ("CuMg2", -0.9))
]


class _FakeMPR:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_entries_in_chemsys(self, elements, additional_criteria=None):
        return list(POOL)


api.mp_rester = lambda api_key=None: _FakeMPR()
api.get_api_key = lambda name: "0" * 32

# 1. The property under test: neither branch may touch the mixing scheme.
for dft_type in ("GGA", "R2SCAN"):
    got = api._get_dft_entries_from_components(["Cu", "Mg"], dft_type)
    assert len(got) == len(POOL), f"{dft_type} returned {len(got)}"

# 2. POSITIVE CONTROL -- without this, an inert blocker would make (1) vacuous.
try:
    api._get_dft_entries_from_components(["Cu", "Mg"], "MIXED")
except ModuleNotFoundError as exc:
    assert "mixing_scheme" in str(exc), exc
else:
    raise AssertionError("blocker inert: MIXED succeeded with mixing_scheme blocked")

print("OK")
"""

    def test_gga_and_r2scan_work_without_the_mixing_scheme(self, tmp_path):
        ok(run_fresh(self.SCRIPT, tmp_path, blocked=("pymatgen.entries.mixing_scheme",)))


class TestMixedBranchUnionsBothThermoTypes:
    """MIXED fetches BOTH thermo types and hands their union to the mixing scheme.

    Pinned offline with a fake ``pymatgen.entries.mixing_scheme``, so the contract is
    checked without a live fetch: both sub-fetches happen, the scheme sees their union,
    and the scheme's OUTPUT (not the raw union) is what gliquid goes on to cache.
    """

    def _install(self, monkeypatch, scheme_out):
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        seen = {"thermo_types": [], "n_in": None, "verbose": None}

        class _FakeScheme:
            def process_entries(self, entries, verbose=False):
                seen["n_in"] = len(entries)
                seen["verbose"] = verbose
                return scheme_out(entries)

        mod = types.ModuleType("pymatgen.entries.mixing_scheme")
        mod.MaterialsProjectDFTMixingScheme = lambda: _FakeScheme()
        monkeypatch.setitem(sys.modules, "pymatgen.entries.mixing_scheme", mod)

        def fetch(elements, additional_criteria):
            tt = str((additional_criteria or {}).get("thermo_types", ["?"])[0])
            seen["thermo_types"].append(tt)
            n = 2 if "R2SCAN" in tt else 3  # distinct counts make the union checkable
            return [
                ComputedEntry(Composition("Cu"), -float(i), entry_id=f"{tt}-{i}").as_dict()
                for i in range(n)
            ]

        install_fake_mprester(monkeypatch, fetch)
        return seen

    def test_both_thermo_types_are_fetched_and_unioned(self, monkeypatch):
        seen = self._install(monkeypatch, scheme_out=lambda entries: entries)
        out = api._get_dft_entries_from_components(["Cu", "Mg"], "MIXED")
        assert len(seen["thermo_types"]) == 2
        joined = " ".join(seen["thermo_types"])
        assert "R2SCAN" in joined and "GGA" in joined
        assert seen["n_in"] == 5, "the scheme must receive the UNION of both fetches"
        assert len(out) == 5

    def test_the_schemes_output_is_what_gets_returned(self, monkeypatch):
        """A dropping scheme must shrink the result -- the raw union is not authoritative."""
        seen = self._install(monkeypatch, scheme_out=lambda entries: entries[:2])
        out = api._get_dft_entries_from_components(["Cu", "Mg"], "MIXED")
        assert seen["n_in"] == 5
        assert len(out) == 2

    def test_single_thermo_types_do_not_call_the_scheme(self, monkeypatch):
        seen = self._install(monkeypatch, scheme_out=lambda entries: entries)
        api._get_dft_entries_from_components(["Cu", "Mg"], "R2SCAN")
        assert seen["n_in"] is None, "R2SCAN must not run the mixing scheme"
        assert len(seen["thermo_types"]) == 1


class TestEmptyFetchIsNeverCached:
    """A fetch that yields zero entries RAISES, above every cache-write site, so no 2-byte
    ``[]`` cache is ever written.
    """

    def _fake_rester(self, monkeypatch, entries):
        install_fake_mprester(monkeypatch, lambda elements, criteria: list(entries))

    def test_empty_fetch_names_the_system_and_the_dft_type(self, monkeypatch):
        self._fake_rester(monkeypatch, [])
        with pytest.raises(ValueError) as exc:
            api._get_dft_entries_from_components(["Cu", "Mg"], "GGA")
        message = str(exc.value)
        assert "Cu-Mg" in message, message
        assert "GGA" in message, message

    @pytest.mark.parametrize("call", ["convexhull", "structure_entries", "imputed_append"])
    def test_no_cache_file_at_any_of_the_three_write_sites(self, monkeypatch, tmp_path, call):
        """Verified by LISTING the directory before and after, not by reading the code path."""
        self._fake_rester(monkeypatch, [])
        before = sorted(p.name for p in tmp_path.iterdir())
        assert before == []
        with pytest.raises(ValueError):
            if call == "convexhull":
                api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
            elif call == "structure_entries":
                api.get_dft_structure_entries(["Cu", "Mg"], "GGA", data_dir=tmp_path)
            else:
                api.cache_imputed_entries(
                    ["Cu", "Mg"], [{"entry_id": "imputed:x"}], data_dir=tmp_path
                )
        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before, f"an empty fetch was persisted: {after}"

    def test_control_a_nonempty_fetch_does_leave_a_file(self, monkeypatch, tmp_path):
        """POSITIVE CONTROL: without it, a listing assertion on a never-written dir is vacuous."""
        self._fake_rester(monkeypatch, _entry_pool())
        assert sorted(p.name for p in tmp_path.iterdir()) == []
        api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["Cu-Mg_ENTRIES_MP_GGA.json"]


class TestBlockedDftTypesFailBeforeAnyNetworkCall:
    """``R2SCAN`` and ``MIXED`` are recognized names that cannot be fetched; both fail before
    any network call (``api._BLOCKED_DFT_TYPES`` names each cause). The spy counts MPRester
    constructions through ``_construct_mprester``, the funnel ``mp_rester`` and ``get_mpr``
    share, and the positive control proves it fires.
    """

    def _client_spy(self, monkeypatch):
        constructions = []

        def _spy(api_key):
            constructions.append(len(api_key))  # length only; never the key itself
            return FakeMPRester(lambda elements, criteria: _entry_pool())

        monkeypatch.setattr(api, "_construct_mprester", _spy)
        monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)
        return constructions

    @pytest.mark.parametrize("dft_type", ["R2SCAN", "MIXED"])
    def test_convexhull_refuses_without_constructing_a_client(
        self, monkeypatch, tmp_path, dft_type
    ):
        constructions = self._client_spy(monkeypatch)
        with pytest.raises(ValueError) as exc:
            api.get_dft_convexhull(["Cu", "Mg"], dft_type, data_dir=tmp_path)
        assert constructions == [], "a Materials Project client was constructed anyway"
        assert sorted(p.name for p in tmp_path.iterdir()) == [], "something was still cached"
        message = str(exc.value)
        assert dft_type in message and "GGA" in message, message

    @pytest.mark.parametrize("dft_type", ["R2SCAN", "MIXED"])
    def test_structure_entries_refuses_without_constructing_a_client(
        self, monkeypatch, tmp_path, dft_type
    ):
        constructions = self._client_spy(monkeypatch)
        with pytest.raises(ValueError):
            api.get_dft_structure_entries(["Cu", "Mg"], dft_type, data_dir=tmp_path)
        assert constructions == []

    def test_control_a_gga_cold_fetch_does_construct_one(self, monkeypatch, tmp_path):
        """POSITIVE CONTROL: an inert spy would make every no-client assertion above vacuous."""
        constructions = self._client_spy(monkeypatch)
        api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
        assert constructions, "the spy never fired -- the no-client assertions prove nothing"

    def test_a_warm_blocked_cache_is_refused_with_the_blocked_type_error(
        self, monkeypatch, tmp_path
    ):
        """A warm 2-byte '[]' R2SCAN cache is refused with the blocked-type error. The
        MESSAGE is matched: PhaseDiagram raises a ValueError on an empty cache too.
        """
        (tmp_path / "Cu-Mg_ENTRIES_MP_R2SCAN.json").write_text("[]")
        constructions = self._client_spy(monkeypatch)
        with pytest.raises(ValueError) as exc:
            api.get_dft_convexhull(["Cu", "Mg"], "R2SCAN", data_dir=tmp_path)
        message = str(exc.value)
        assert "R2SCAN" in message and "ThermoType" in message, message
        assert constructions == []

    def test_each_cause_names_the_upstream_symbol_to_re_check(self):
        assert "ThermoType" in api._BLOCKED_DFT_TYPES["R2SCAN"]
        assert "entry_id" in api._BLOCKED_DFT_TYPES["MIXED"]

    def test_an_unrecognized_dft_type_still_lists_the_supported_names(self, tmp_path):
        with pytest.raises(ValueError, match="not currently supported"):
            api.get_dft_convexhull(["Cu", "Mg"], "PBE", data_dir=tmp_path)


class TestAtomicCacheWrite:
    """A failed cache write leaves the previous file intact and no scratch file behind;
    ``json.dump`` writes incrementally, so a direct write could leave a TRUNCATED cache.
    """

    # A set is not JSON-serializable, and json.dump raises only AFTER emitting the leading
    # bytes -- precisely the truncation shape being guarded against.
    UNSERIALIZABLE = [{"composition": {"Cu": 1}, "energy": -1.0, "bad": {1, 2}}]

    def test_failed_cold_fetch_leaves_no_partial_cache(self, monkeypatch, tmp_path):
        """The end-to-end shape: no file at all beats a truncated one read as valid."""
        monkeypatch.setattr(
            api, "_get_dft_entries_from_components", lambda *a, **k: self.UNSERIALIZABLE
        )
        with pytest.raises(TypeError):
            api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
        assert list(tmp_path.iterdir()) == [], "a partial cache was left on disk"

    def test_failed_imputed_append_keeps_the_real_entries(self, tmp_path):
        cache = tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json"
        cache.write_text(json.dumps([{"composition": {"Cu": 1}, "energy": 0.0, "entry_id": "r"}]))
        before = cache.read_bytes()
        with pytest.raises(TypeError):
            api.cache_imputed_entries(
                ["Cu", "Mg"], [{"entry_id": "imputed:x", "bad": {1}}], data_dir=tmp_path
            )
        assert cache.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == [cache.name]


# -- KIND_IMPUTED_ENTRIES on the hull path --


def _imputed_entry(formula="Cu2Mg", energy=-1.5, label="Probe"):
    """A tagged entry deep below the pool's hull, at a composition the pool has no vertex at."""
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    return ComputedEntry(
        Composition(formula),
        energy,
        entry_id=f"imputed:{label}",
        data={"imputed": True, "label": label},
    ).as_dict()


def _vertices(pd) -> list[tuple[str, float]]:
    """A hull's stable entries as ``(reduced formula, energy per atom)``, sorted."""
    return sorted(
        (api.entry_display_name(e), round(api.entry_original(e).energy_per_atom, 9))
        for e in pd.stable_entries
    )


class _RecordingBackend:
    """Delegates to ``inner`` and records the ``(method, kind)`` of every question asked."""

    def __init__(self, inner):
        self._inner = inner
        self.asked: list[tuple[str, str]] = []

    @property
    def kinds_touched(self) -> set[str]:
        return {kind for _, kind in self.asked}

    def exists(self, key):
        self.asked.append(("exists", key.kind))
        return self._inner.exists(key)

    def read_json(self, key):
        self.asked.append(("read_json", key.kind))
        return self._inner.read_json(key)

    def write_json(self, key, payload):
        self.asked.append(("write_json", key.kind))
        return self._inner.write_json(key, payload)

    def variants(self, sys_name, kind):
        self.asked.append(("variants", kind))
        return self._inner.variants(sys_name, kind)

    def locate(self, key):
        return self._inner.locate(key)


def _v2_store(path):
    from gliquid.cache import SqliteBackend

    backend = SqliteBackend(path, create=True)
    backend.write_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA"), _entry_pool())
    return backend


class TestImputedEntriesReachTheHullFromAStore:
    """``get_dft_convexhull(..., include_imputed=True)`` reads the store's imputed-entry
    record, so a single-file store's imputed phases are reachable and not merely written.

    The default stays DFT-only, asserted by INSTRUMENTING the backend."""

    @pytest.fixture
    def store(self, tmp_path):
        backend = _v2_store(tmp_path / "store.sqlite")
        backend.write_json(CacheKey("Cu-Mg", KIND_IMPUTED_ENTRIES, "GGA"), [_imputed_entry()])
        yield backend
        backend.close()

    def test_an_imputed_entry_becomes_a_stable_vertex_carrying_its_label(self, store):
        pd, _ = api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=store, include_imputed=True)
        stable = {api.entry_display_name(e): api.entry_original(e) for e in pd.stable_entries}
        assert "MgCu2" in stable, "the stored imputed entry never reached the hull"
        assert stable["MgCu2"].data["label"] == "Probe"

    def test_the_default_hull_matches_one_built_with_no_imputed_row_at_all(self, store, tmp_path):
        clean = _v2_store(tmp_path / "clean.sqlite")
        try:
            expected = _vertices(api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=clean)[0])
        finally:
            clean.close()
        default = _vertices(api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=store)[0])
        assert default == expected
        opted_in = api.get_dft_convexhull(
            ["Cu", "Mg"], "GGA", data_dir=store, include_imputed=True
        )[0]
        assert _vertices(opted_in) != expected, "POSITIVE CONTROL: the row is really there"

    def test_the_default_path_never_asks_the_store_about_the_imputed_kind(self, store):
        recorder = _RecordingBackend(store)
        api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=recorder, include_imputed=False)
        assert KIND_IMPUTED_ENTRIES not in recorder.kinds_touched, recorder.asked
        assert KIND_DFT_ENTRIES in recorder.kinds_touched, "control: the DFT kind WAS read"

        recorder.asked.clear()
        api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=recorder, include_imputed=True)
        assert KIND_IMPUTED_ENTRIES in recorder.kinds_touched, recorder.asked

    def test_a_write_through_the_api_round_trips_on_a_sqlite_store(self, tmp_path):
        backend = _v2_store(tmp_path / "rt.sqlite")
        try:
            api.cache_imputed_entries(["Cu", "Mg"], [_imputed_entry()], data_dir=backend)
            assert api.drop_imputed_entries(["Cu", "Mg"], data_dir=backend) == 1
            api.cache_imputed_entries(["Cu", "Mg"], [_imputed_entry()], data_dir=backend)
            pd, _ = api.get_dft_convexhull(
                ["Cu", "Mg"], "GGA", data_dir=backend, include_imputed=True
            )
            assert "MgCu2" in {api.entry_display_name(e) for e in pd.stable_entries}
            assert backend.read_json(CacheKey("Cu-Mg", KIND_DFT_ENTRIES, "GGA")) == _entry_pool()
        finally:
            backend.close()

    def test_a_v1_store_answers_no_imputed_entries_rather_than_raising(self, tmp_path):
        import sqlite3

        from gliquid.cache import SqliteBackend, close_sqlite_backends

        path = tmp_path / "v1.sqlite"
        _v2_store(path).close()
        conn = sqlite3.connect(str(path))
        conn.execute("DROP TABLE imputed_entries")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        backend = SqliteBackend(path)
        try:
            pd, _ = api.get_dft_convexhull(
                ["Cu", "Mg"], "GGA", data_dir=backend, include_imputed=True
            )
            assert _vertices(pd) == _vertices(
                api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=backend)[0]
            )
        finally:
            backend.close()
            close_sqlite_backends()

    def test_a_directory_store_round_trips_exactly_as_it_did(self, tmp_path):
        """Imputed entries tagged inside the DFT record reach the hull, and are invisible
        to the default."""
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(_entry_pool()))
        api.cache_imputed_entries(["Cu", "Mg"], [_imputed_entry()], data_dir=tmp_path)

        opted_in, _ = api.get_dft_convexhull(
            ["Cu", "Mg"], "GGA", data_dir=tmp_path, include_imputed=True
        )
        assert "MgCu2" in {api.entry_display_name(e) for e in opted_in.stable_entries}

        default, _ = api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=tmp_path)
        assert "MgCu2" not in {api.entry_display_name(e) for e in default.stable_entries}

        raw = json.loads((tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").read_text())
        assert [e for e in raw if not is_imputed_entry_dict(e)] == _entry_pool()
        assert api.drop_imputed_entries(["Cu", "Mg"], data_dir=tmp_path) == 1
        assert json.loads((tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").read_text()) == _entry_pool()
