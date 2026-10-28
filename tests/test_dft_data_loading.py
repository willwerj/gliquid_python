"""Tests for DFT convex-hull data loading: the live MP API path and the cached path.

Enforces both behaviors of :func:`gliquid.api.get_dft_convexhull` without
committing any DFT-entries fixtures (``data/*_ENTRIES_MP_GGA.json`` stay gitignored):

* The **API path** runs first -- a single live Materials Project fetch into a temp
  ``cache_dir`` writes ``<sys>_ENTRIES_MP_GGA.json`` (the cache-miss branch).
* The **cache path** then reuses exactly that file and asserts the API is not called again.

Both require ``NEW_MP_API_KEY`` (locally via ``.env``/conda, on CI via repo secret), so
the module is skipped when the key is absent.
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
    saved_data_dir = config.cache_dir
    # Pin the mode too: this fixture exercises the directory store's cold-fetch write path, and the
    # repo-root conftest.py can swap a sqlite store in process-wide.
    saved_cache_mode = config.cache_mode
    config.cache_dir = cache_dir  # flat layout -> entries file lands directly in cache_dir
    config.set_cache_mode("directory")
    try:
        dft_ch, _ = api.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
        yield cache_dir, dft_ch
    finally:
        config.cache_dir = saved_data_dir
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
    config.cache_dir = cache_dir  # reuse the cache populated by the API fetch above

    def _fail(*args, **kwargs):
        raise AssertionError("MP API was called even though the cache file exists")

    monkeypatch.setattr(api, "_get_dft_entries_from_components", _fail)

    dft_ch, _ = api.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
    assert dft_ch is not None
    assert len(dft_ch.stable_entries) >= 2


# -- The other two dft_type values --
# Both fail upstream; each xfails with ``raises=`` so an upstream fix shows as a strict XPASS. They
# probe ``_get_dft_entries_from_components``, the deepest point reaching the live API.

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


@pytest.mark.xfail(raises=ValueError, reason=R2SCAN_CAUSE, strict=True)
def test_cold_r2scan_fetch_returns_entries():
    """Cold R2SCAN: does a live fetch return any r2SCAN entry at all? (Upstream: no.)"""
    _require_key()
    entries = api._get_dft_entries_from_components(FIXTURE_SYSTEM, "R2SCAN")
    assert entries, "an r2SCAN fetch returned nothing"


@pytest.mark.xfail(raises=TypeError, reason=MIXED_CAUSE, strict=True)
def test_cold_mixed_fetch_returns_entries():
    """Cold MIXED: the same, plus MaterialsProjectDFTMixingScheme over the cold set."""
    _require_key()
    entries = api._get_dft_entries_from_components(FIXTURE_SYSTEM, "MIXED")
    assert entries, "a MIXED fetch returned nothing"
