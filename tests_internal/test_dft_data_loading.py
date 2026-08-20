"""Tests for DFT convex-hull data loading: the live MP API path and the cached path.

Enforces both behaviors of :func:`gliquid.api.get_dft_convexhull` without
committing any DFT-entries fixtures (``data/*_ENTRIES_MP_GGA.json`` stay gitignored):

* The **API path** runs first -- a single live Materials Project fetch into a temp
  ``data_dir`` writes ``<sys>_ENTRIES_MP_GGA.json`` (the cache-miss branch).
* The **cache path** then reuses exactly that file and asserts the API is not called again.

Both require ``NEW_MP_API_KEY`` (locally via ``.env``/conda, on CI via repo secret), so
the module is skipped when the key is absent. The API path is the one that breaks on a
dependency-version mismatch (``ModuleNotFoundError: No module named 'pymatgen.core.entries'``).
"""

import os

import pytest

import gliquid.api as api
import gliquid.config as config

pytestmark = pytest.mark.needs_network

FIXTURE_SYSTEM = ["Ag", "V"]
FIXTURE_FILE = "Ag-V_ENTRIES_MP_GGA.json"


@pytest.fixture(scope="module")
def api_cached_dir(tmp_path_factory):
    """Fetch the DFT entries once from the live MP API into a temp cache dir.

    The first access is a cache miss -> live API fetch -> on-disk cache write. The
    cached-path test then reuses the file this fixture produced. Skips the whole module
    when no API key is available.
    """
    if not os.getenv("NEW_MP_API_KEY"):
        pytest.skip("NEW_MP_API_KEY not set")
    cache_dir = tmp_path_factory.mktemp("dft_cache")
    saved_data_dir = config.data_dir
    # Pin the mode as well as the location. This fixture IS a directory store -- it exists
    # to exercise the cold-fetch WRITE path, which a single-file store refuses by design --
    # and leaving the mode ambient would make it depend on how the session was configured
    # (the repo-root conftest.py can swap a sqlite store in process-wide).
    saved_cache_mode = config.cache_mode
    config.data_dir = cache_dir  # flat layout -> entries file lands directly in cache_dir
    config.set_cache_mode("directory")
    try:
        dft_ch, _ = api.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
        yield cache_dir, dft_ch
    finally:
        config.data_dir = saved_data_dir
        config.set_cache_mode(saved_cache_mode)


def test_api_fetch_builds_hull_and_writes_cache(api_cached_dir):
    """The live API path returns a valid hull and caches it to disk."""
    cache_dir, dft_ch = api_cached_dir
    assert dft_ch is not None
    assert len(dft_ch.stable_entries) >= 2  # at least the two terminal elements
    assert (cache_dir / FIXTURE_FILE).exists(), "API result should have been cached to disk"


def test_cache_hit_reuses_api_file_without_calling_api(api_cached_dir, monkeypatch):
    """The cached path loads the file the API call wrote and never calls the API again."""
    cache_dir, _ = api_cached_dir
    config.data_dir = cache_dir  # reuse the cache populated by the API fetch above

    def _fail(*args, **kwargs):
        raise AssertionError("MP API was called even though the cache file exists")

    monkeypatch.setattr(api, "_get_dft_entries_from_components", _fail)

    dft_ch, _ = api.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
    assert dft_ch is not None
    assert len(dft_ch.stable_entries) >= 2


# ---------------------------------------------------------------------------------------
# The other two dft_type values. Both are broken UPSTREAM as of emmet-core 0.85.1 /
# mp-api 0.45.13 / pymatgen 2025.10.7, each for its own diagnosed reason (spec 08c).
# They are xfailed rather than deleted so they stay executable canaries: ``raises=`` pins
# the exact failure SHAPE, so a change in how they break is a real failure, and an upstream
# fix shows up as XPASS instead of passing unnoticed.
#
# Spec 08d made the PUBLIC entry points (``get_dft_convexhull``,
# ``get_dft_structure_entries``) refuse both names outright, before any network call, so
# calling those here would only re-measure gliquid's own guard and could never XPASS again.
# The canaries therefore probe one layer down, at ``_get_dft_entries_from_components``,
# which 08d left reachable for exactly this purpose: it is the deepest point that still
# talks to the live API. The fast-fail guard itself is pinned offline in
# tests/test_api.py::TestBlockedDftTypesFailBeforeAnyNetworkCall.
# ---------------------------------------------------------------------------------------

R2SCAN_CAUSE = (
    "The Materials Project stores the r2SCAN thermo type as the literal 'r2SCAN', while "
    "emmet.core.thermo.ThermoType.R2SCAN is 'R2SCAN'. mp_api validates against the enum "
    "and forwards 'R2SCAN' verbatim, which matches no document, so the fetch returns 0 "
    "entries -- which 08d's empty-fetch guard now turns into a ValueError at the fetch "
    "instead of a PhaseDiagram failure one layer up. "
    "(Ge-Si demonstrably has 3 r2SCAN thermo docs; a GGA_GGA+U control returns 6.)"
)

MIXED_CAUSE = (
    "pymatgen's MaterialsProjectDFTMixingScheme._filter_and_sort_entries builds "
    "{e.entry_id for e in filtered_entries}, but new-API entries carry entry_id as a dict "
    "({'identifier': 'mp-aaaaalmn', 'suffix': 'GGA', ...}), which is unhashable -> "
    "TypeError: unhashable type: 'dict'. gliquid's own code already handles dict entry_ids "
    "(api._is_spurious_entry_dict); the mixing scheme does not. Note that R2SCAN_CAUSE also "
    "applies here: MIXED's r2SCAN half is empty, so even past this it would be GGA-only."
)


def _require_key():
    if not os.getenv("NEW_MP_API_KEY"):
        pytest.skip("NEW_MP_API_KEY not set")


@pytest.mark.xfail(raises=ValueError, reason=R2SCAN_CAUSE, strict=False)
def test_cold_r2scan_fetch_returns_entries():
    """Cold R2SCAN: does a live fetch return any r2SCAN entry at all? (Upstream: no.)"""
    _require_key()
    entries = api._get_dft_entries_from_components(FIXTURE_SYSTEM, "R2SCAN")
    assert entries, "an r2SCAN fetch returned nothing"


@pytest.mark.xfail(raises=TypeError, reason=MIXED_CAUSE, strict=False)
def test_cold_mixed_fetch_returns_entries():
    """Cold MIXED: the same, plus MaterialsProjectDFTMixingScheme over the cold set."""
    _require_key()
    entries = api._get_dft_entries_from_components(FIXTURE_SYSTEM, "MIXED")
    assert entries, "a MIXED fetch returned nothing"


def test_the_public_entry_point_refuses_both_blocked_types():
    """The other half of the canary: while the fetch is broken, the public API must say so.

    If either xfail above turns XPASS, this test is the one to delete alongside the
    matching ``api._BLOCKED_DFT_TYPES`` entry.
    """
    for dft_type in ("R2SCAN", "MIXED"):
        assert dft_type in api._BLOCKED_DFT_TYPES
