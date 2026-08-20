"""Repo-root pytest configuration: the optional single-file-store swap.

Set ``GLIQUID_TEST_SQLITE_STORE=<path>.sqlite`` and the whole suite runs against a SQLite
cache store instead of the ``cache/`` directory tree::

    python -m gliquid.cache migrate --from cache --to ../dev/data/store.sqlite
    GLIQUID_TEST_SQLITE_STORE=../dev/data/store.sqlite python -m pytest tests tests_internal

Unset (the default, and what CI and the acceptance gate run) this file does nothing at all,
so the normal suite is byte-identical to what it was before the swap existed.

Why an env-gated swap rather than a parametrized fixture: the store is a PROCESS-wide
configuration -- ``gliquid.config.cache_mode`` and ``cache_dir`` -- and every cached read in
the package resolves through it. Parametrizing would mean flipping global state between
tests and hoping every one of them restores it; a second whole-suite run cannot leak.

``@pytest.mark.directory_only`` marks the tests that are ABOUT the directory backend
(filenames on disk, ``dir_structure``, the atomic-write path, the cold-fetch write path).
They are skipped under the swap because they are not testing the thing being swapped. The
marker means "this test needs a directory tree", never "this test fails under sqlite" --
if a test fails under the swap for any other reason, that is a real finding.
"""

import os
from pathlib import Path

import pytest

_SQLITE_STORE_ENV = "GLIQUID_TEST_SQLITE_STORE"


def _configured_store() -> Path | None:
    value = os.environ.get(_SQLITE_STORE_ENV)
    return Path(value) if value else None


def pytest_configure(config):
    store = _configured_store()
    if store is None:
        return
    if not store.is_file():
        raise pytest.UsageError(
            f"{_SQLITE_STORE_ENV} is set to '{store}', which is not a file. Build one with "
            f"`python -m gliquid.cache migrate --from cache --to {store}`."
        )
    import gliquid.config as gliquid_config

    gliquid_config.set_cache_dir(store)
    if gliquid_config.cache_mode != "sqlite":  # pragma: no cover - defensive
        raise pytest.UsageError(
            f"set_cache_dir('{store}') left cache_mode {gliquid_config.cache_mode!r}; the "
            f"suite would have run against a directory store while claiming otherwise."
        )


def pytest_report_header(config):
    """Say which store the run used. A silent swap is indistinguishable from no swap."""
    store = _configured_store()
    if store is None:
        return None
    return f"gliquid cache store: SQLITE (single file) {store}"


def pytest_collection_modifyitems(config, items):
    if _configured_store() is None:
        return
    skip = pytest.mark.skip(
        reason=f"directory-only test, and {_SQLITE_STORE_ENV} selected a single-file store"
    )
    for item in items:
        if "directory_only" in item.keywords:
            item.add_marker(skip)
