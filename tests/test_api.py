"""Tests for gliquid.api: env-safe key resolution and lazy clients.

All tests run offline. The import-hygiene test spawns a subprocess so module state from the
rest of the suite cannot mask an eager import regression. The MPDS bibliography/jcode HTTP
helpers moved to dev/scripts/mpds_bib.py (tested by dev/tests/test_mpds_bib.py).
"""

import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import gliquid.api as api


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


def test_import_hygiene_no_eager_client_imports():
    """Importing gliquid.api must not import mp_api, mpds_client, or requests."""
    code = (
        "import sys; import gliquid.api; "
        "leaked = [m for m in ('mp_api', 'mpds_client', 'requests') if m in sys.modules]; "
        "assert not leaked, f'eagerly imported: {leaked}'"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.returncode == 0, result.stderr


class TestLegacyMontyModuleAliases:
    """The retired pymatgen paths must stay importable for UPSTREAM monty decodes.

    Regression guard for a cache-MISS-only failure: MPRester.get_entries decodes the
    server's response itself, and the Materials Project still tags nested
    energy_adjustments records with `pymatgen.core.entries`. Without the alias, monty
    raises ModuleNotFoundError and every uncached DFT fetch dies -- while a warm cache
    succeeds, which is exactly why it survived local testing.
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
        """Each legacy path resolves to *something usable* -- not necessarily the alias.

        Asserting ``sys.modules[legacy].__name__ == current`` was wrong: the aliaser
        deliberately leaves a legacy path alone when it genuinely still exists upstream, so
        on a pymatgen that still ships ``pymatgen.core.entries`` the name is its own, not the
        alias target. CI caught exactly that -- green on 3.10 where the path is gone, red on
        3.11+ where it is not. What actually matters is that the path imports and carries the
        classes monty will look for on it.
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
    """The MPRester chemsys query must be expressed in ELEMENT symbols.

    Cache-MISS-only regression guard. ``MPRester.get_entries_in_chemsys`` enumerates every
    chemsys substring of ``set(elements)``, so passing components verbatim queried a
    compound component as the nonexistent chemsys 'CuMg': the live API returned only the
    Mg entries, gliquid CACHED that partial set, and CompoundPhaseDiagram then died with
    "Missing terminal entries" -- far from the fetch that caused it. Warm caches hid it.
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

        class _FakeMPR:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_entries_in_chemsys(self, elements, additional_criteria=None):
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

        monkeypatch.setattr(api, "mp_rester", lambda api_key=None: _FakeMPR())
        monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)
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
        """End-to-end: the partial fetch used to poison the cache AND break the hull."""
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


class TestConstructMPResterFallback:
    """``_construct_mprester``'s ``TypeError`` fallback must announce what it degraded.

    The fallback yields ``monty_decode=True, use_document_model=True`` -- a DIFFERENT data
    shape (document models, not dicts) flowing into code that expects dicts. It is
    unreachable on every mp_api version tested, which is exactly why a silent one would be
    baffling if it ever fired.
    """

    def _fake_mp_api(self, monkeypatch, *, accept_kwargs: bool):
        class _FakeMPRester:
            def __init__(self, api_key, **kwargs):
                if kwargs and not accept_kwargs:
                    raise TypeError("__init__() got an unexpected keyword 'monty_decode'")
                self.api_key = api_key
                self.kwargs = kwargs

        client_mod = types.ModuleType("mp_api.client")
        client_mod.MPRester = _FakeMPRester
        parent = types.ModuleType("mp_api")
        parent.client = client_mod
        monkeypatch.setitem(sys.modules, "mp_api", parent)
        monkeypatch.setitem(sys.modules, "mp_api.client", client_mod)

    def test_modern_mp_api_keeps_the_kwargs_and_stays_quiet(self, monkeypatch, caplog):
        """Control: on a supported mp_api the preferred path is taken and warns nothing."""
        self._fake_mp_api(monkeypatch, accept_kwargs=True)
        with caplog.at_level(logging.WARNING, logger="gliquid.api"):
            client = api._construct_mprester("0" * 32)
        assert client.kwargs == {"monty_decode": False, "use_document_model": False}
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_fallback_warns_and_names_the_degradation(self, monkeypatch, caplog):
        self._fake_mp_api(monkeypatch, accept_kwargs=False)
        with caplog.at_level(logging.WARNING, logger="gliquid.api"):
            client = api._construct_mprester("0" * 32)
        assert client.kwargs == {}, "expected the degraded (defaulted) client"
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "the fallback fired silently"
        text = " ".join(warnings).lower()
        assert "monty_decode" in text and "use_document_model" in text
        assert "document model" in text, "the warning must name the data-shape change"
        assert "mp_api" in text


class TestMixingSchemeImportIsBranchLocal:
    """Only the MIXED branch may need ``pymatgen.entries.mixing_scheme``.

    A pymatgen release relocating that class must not break GGA and R2SCAN cold fetches
    for a feature they never touch -- the same shape as the ``emmet.core.thermo``
    coupling the ``emmet-core<0.86`` ceiling exists for. Blocked at the import system in a
    subprocess (the technique ``tests/test_editor_extra.py`` uses), because this session
    has pymatgen fully importable.
    """

    SCRIPT = """
import sys

BLOCKED = "pymatgen.entries.mixing_scheme"


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.modules.pop(BLOCKED, None)
sys.meta_path.insert(0, _Block())

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api

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

    def test_gga_and_r2scan_work_without_the_mixing_scheme(self):
        env = os.environ.copy()
        src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        assert "OK" in proc.stdout


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

        class _FakeMPR:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_entries_in_chemsys(self, elements, additional_criteria=None):
                tt = str((additional_criteria or {}).get("thermo_types", ["?"])[0])
                seen["thermo_types"].append(tt)
                n = 2 if "R2SCAN" in tt else 3  # distinct counts make the union checkable
                return [
                    ComputedEntry(Composition("Cu"), -float(i), entry_id=f"{tt}-{i}").as_dict()
                    for i in range(n)
                ]

        monkeypatch.setattr(api, "mp_rester", lambda api_key=None: _FakeMPR())
        monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)
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


def _entry_pool():
    """A minimal Cu-Mg entry pool, enough to build a hull and write a cache."""
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    return [
        ComputedEntry(Composition(f), e, entry_id=f"s-{f}").as_dict()
        for f, e in (("Cu", 0.0), ("Mg", 0.0), ("CuMg2", -0.9))
    ]


class TestEmptyFetchIsNeverCached:
    """A fetch that yields zero entries must RAISE, above every cache-write site.

    A 2-byte ``[]`` on disk is indistinguishable from a real cache on the next read, so a
    single bad fetch is served warm forever -- the same defect as the partial-chemsys fetch
    in spec 08b, and the file a cold R2SCAN fetch actually wrote before failing. Returning
    empty without raising is no better: the caller then dies further away, in
    ``PhaseDiagram``'s "Unable to build phase diagram without entries".
    """

    def _fake_rester(self, monkeypatch, entries):
        class _FakeMPR:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_entries_in_chemsys(self, elements, additional_criteria=None):
                return list(entries)

        monkeypatch.setattr(api, "mp_rester", lambda api_key=None: _FakeMPR())
        monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)

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
    """``R2SCAN`` and ``MIXED`` are recognized names that cannot be fetched.

    Both are broken UPSTREAM (``api._BLOCKED_DFT_TYPES`` carries the one-line diagnosis of
    each). Before this guard they failed deep inside the fetch with an error pointing
    nowhere near the cause, and R2SCAN wrote a ``[]`` cache on its way out. The spy here
    counts MPRester constructions -- ``_construct_mprester`` is the single funnel both
    ``mp_rester`` and ``get_mpr`` go through -- and the positive control proves it fires.
    """

    def _client_spy(self, monkeypatch):
        constructions = []

        class _FakeMPR:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_entries_in_chemsys(self, elements, additional_criteria=None):
                return _entry_pool()

        def _spy(api_key):
            constructions.append(len(api_key))  # length only; never the key itself
            return _FakeMPR()

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
        """The R2SCAN caches already on disk ARE the 2-byte '[]' this guard exists for.

        Unguarded this also raised ValueError -- but PhaseDiagram's "Unable to build phase
        diagram without entries", which names neither the dft_type nor the cause. Matching
        the MESSAGE is what makes this test red without the guard.
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

    def test_gga_is_recognized_and_not_blocked(self):
        assert "GGA" in api.SUPPORTED_DFT_TYPES
        assert "GGA" not in api._BLOCKED_DFT_TYPES

    def test_an_unrecognized_dft_type_still_lists_the_supported_names(self, tmp_path):
        with pytest.raises(ValueError, match="not currently supported"):
            api.get_dft_convexhull(["Cu", "Mg"], "PBE", data_dir=tmp_path)


class TestAtomicCacheWrite:
    """A failed cache write leaves the previous file intact and no scratch file behind.

    ``json.dump`` writes incrementally, so an interrupted write used to leave a TRUNCATED
    cache that later reads accept as valid. Concurrent cold fetches make this real here:
    ``dev/scripts/Fit_Binary_Systems.py`` fans campaigns over a ProcessPoolExecutor, and a
    campaign over uncached systems is a burst of concurrent cold fetches.
    """

    # A set is not JSON-serializable, and json.dump raises only AFTER emitting the leading
    # bytes -- precisely the truncation shape being guarded against.
    UNSERIALIZABLE = [{"composition": {"Cu": 1}, "energy": -1.0, "bad": {1, 2}}]

    def test_successful_write_leaves_exactly_one_file(self, tmp_path):
        target = tmp_path / "x.json"
        api._atomic_write_json(str(target), [{"a": 1}])
        assert json.loads(target.read_text()) == [{"a": 1}]
        assert [p.name for p in tmp_path.iterdir()] == ["x.json"], "scratch file survived"

    def test_failed_write_leaves_the_previous_file_byte_identical(self, tmp_path):
        target = tmp_path / "x.json"
        target.write_text('[{"keep": true}]')
        before = target.read_bytes()
        with pytest.raises(TypeError):
            api._atomic_write_json(str(target), self.UNSERIALIZABLE)
        assert target.read_bytes() == before, "previous cache was clobbered"
        assert [p.name for p in tmp_path.iterdir()] == ["x.json"], "scratch file survived"

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
