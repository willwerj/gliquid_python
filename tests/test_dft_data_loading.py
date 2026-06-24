"""Tests for DFT convex-hull data loading: the live MP API path and the cached path.

Enforces both behaviors of :func:`gliquid.load_binary_data.get_dft_convexhull` without
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

import gliquid.config as config
import gliquid.load_binary_data as lbd

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
    config.data_dir = cache_dir  # flat layout -> entries file lands directly in cache_dir
    try:
        dft_ch, _ = lbd.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
        yield cache_dir, dft_ch
    finally:
        config.data_dir = saved_data_dir


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

    monkeypatch.setattr(lbd, "_get_dft_entries_from_components", _fail)

    dft_ch, _ = lbd.get_dft_convexhull(FIXTURE_SYSTEM, "GGA")
    assert dft_ch is not None
    assert len(dft_ch.stable_entries) >= 2
