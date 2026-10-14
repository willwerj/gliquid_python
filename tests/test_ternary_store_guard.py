"""A ternary on its own store refuses reads of the implicit default store.

``build_binary_edge_figures`` and ``TernaryLiquidInterpolation._reference_floor`` read the
global store. While that store is the implicit ``<cwd>/gliquid_cache`` default, both raise
``ConfigError`` naming both locations when the ternary's own store is somewhere else. An
explicitly configured global store, a ternary on the default store, and a reordered copy of
one stay silent.

The implicit state is simulated in-process by pointing ``config.cache_dir`` and the implicit
marker at the same uncreated path. The installed-package case, where the implicit default is
real, runs in ``test_config``'s default-store subprocess.
``TernaryLiquidInterpolation.edge_hull`` and ``BinaryLiquid.from_cache`` are tripwires
throughout: reaching either means the edge read went through.
"""

import logging
import warnings
from pathlib import Path

import pytest

import gliquid.config as config
import gliquid.ternary as ternary
from gliquid.cache import (
    KIND_MPDS,
    CacheKey,
    DirectoryBackend,
    OverlayBackend,
    SqliteBackend,
    close_sqlite_backends,
)
from gliquid.ternary import TernaryLiquidInterpolation, build_binary_edge_figures

COMPONENTS = ["Al", "Mg", "Si"]
IDEAL = {"Al-Mg": [0.0] * 4, "Mg-Si": [0.0] * 4, "Si-Al": [0.0] * 4}


class _EdgeReached(Exception):
    """Raised by the ``edge_hull`` and ``BinaryLiquid.from_cache`` tripwires."""


def _edge_tripwire(*args, **kwargs):
    raise _EdgeReached


class _OpaqueStore:
    """A store with the backend methods and no location gliquid can derive."""

    capabilities = frozenset({"read", "variants"})

    def exists(self, key):
        return False

    def read_json(self, key):
        raise FileNotFoundError(key)

    def write_json(self, key, payload):
        raise AssertionError("nothing is written in these tests")

    def variants(self, sys_name, kind):
        return []

    def locate(self, key):
        raise AssertionError("a store location is never derived through locate")


@pytest.fixture(autouse=True)
def _edges_are_tripwires(monkeypatch):
    """``TernaryLiquidInterpolation.edge_hull`` and ``BinaryLiquid.from_cache``, as
    ``ternary.py`` reads them, raise ``_EdgeReached``."""
    monkeypatch.setattr(ternary.TernaryLiquidInterpolation, "edge_hull", _edge_tripwire)
    monkeypatch.setattr(ternary.BinaryLiquid, "from_cache", staticmethod(_edge_tripwire))
    yield
    close_sqlite_backends()


@pytest.fixture
def default_store(tmp_path, monkeypatch):
    """The implicit ``<cwd>/gliquid_cache`` default store, at a path that is never created."""
    path = (tmp_path / "cwd" / config.DEFAULT_CACHE_DIR_NAME).resolve()
    monkeypatch.setattr(config, "cache_dir", path)
    monkeypatch.setattr(config, "_implicit_cache_dir", path)
    monkeypatch.setattr(config, "cache_mode", "directory")
    monkeypatch.setattr(config, "dir_structure", "flat")
    assert config._cache_dir_is_implicit()
    return path


@pytest.fixture
def explicit_other_store(default_store, tmp_path, monkeypatch):
    """``set_cache_dir`` to an existing directory other than the default; restored after."""
    for name in ("phase_transitions_file", "omegas_file", "spurious_structures_file"):
        monkeypatch.setattr(config, name, getattr(config, name))
    other = tmp_path / "global"
    other.mkdir()
    config.set_cache_dir(other)
    assert not config._cache_dir_is_implicit()
    return other


@pytest.fixture
def store_a(tmp_path):
    """A directory store for the ternary that is not the default store; never created."""
    return tmp_path / "store_a"


def _read_only_sqlite(tmp_path) -> Path:
    """A SQLite store file holding one record, for opening read-only."""
    path = tmp_path / "ternary.sqlite"
    writer = SqliteBackend(path, create=True)
    try:
        writer.write_json(CacheKey("Al-Mg", KIND_MPDS, ""), {"reference": None})
    finally:
        writer.close()
    return path


def _ternary(data_dir=None) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(COMPONENTS),
        data_dir=data_dir,
        delta=0.25,
        xs_mix=IDEAL,
        order="given",
        hull_mode="hsx",
        T_incr=100,
        param_format="linear",
    )


def _edge_figures(ti):
    return build_binary_edge_figures(ti)


def _reference_floor(ti):
    return ti._reference_floor()


READS = {"edge_figures": _edge_figures, "reference_floor": _reference_floor}


def _assert_refused(ti, read, *locations) -> str:
    """``read(ti)`` raises ``ConfigError`` naming every location and the remedy."""
    with pytest.raises(config.ConfigError) as info:
        read(ti)
    message = str(info.value)
    for location in locations:
        assert str(location) in message
    assert "gliquid.config.set_cache_dir(" in message
    return message


def _assert_silent(ti, caplog) -> None:
    """Both reads go through with no new warning and no new WARNING log record."""
    caplog.set_level(logging.WARNING)
    before = len(caplog.records)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_EdgeReached):
            build_binary_edge_figures(ti)
        floor = ti._reference_floor()
    assert isinstance(floor, float)
    assert [str(w.message) for w in caught] == []
    new_records = caplog.records[before:]
    assert [r.getMessage() for r in new_records if r.levelno >= logging.WARNING] == []


# -- Refused: the implicit default store and a ternary store somewhere else --


def test_edge_figures_refuse_the_default_store_for_a_ternary_on_its_own_store(
    default_store, store_a
):
    ti = _ternary(store_a)
    _assert_refused(ti, _edge_figures, store_a.resolve(), default_store)
    assert not store_a.exists()
    assert not default_store.exists()


def test_the_reference_floor_refuses_the_default_store_for_a_ternary_on_its_own_store(
    default_store, store_a
):
    ti = _ternary(store_a)
    _assert_refused(ti, _reference_floor, store_a.resolve(), default_store)
    assert not store_a.exists()
    assert not default_store.exists()


@pytest.mark.parametrize("read", sorted(READS))
@pytest.mark.parametrize("kind", ["read-only sqlite", "overlay", "nested directory"])
def test_every_store_kind_elsewhere_is_refused(default_store, store_a, tmp_path, kind, read):
    if kind == "read-only sqlite":
        location = _read_only_sqlite(tmp_path)
        ti = _ternary(location)
        assert isinstance(ti.data_dir, SqliteBackend)
        assert "write" not in ti.data_dir.capabilities
    elif kind == "overlay":
        location = store_a
        ti = _ternary(OverlayBackend(store_a))
    else:
        location = store_a
        ti = _ternary(DirectoryBackend(store_a, dir_structure="nested"))
    _assert_refused(ti, READS[read], location.resolve(), default_store)
    assert not store_a.exists()
    assert not default_store.exists()


@pytest.mark.parametrize("read", sorted(READS))
def test_a_store_without_a_derivable_location_is_refused(default_store, read):
    ti = _ternary(_OpaqueStore())
    message = _assert_refused(ti, READS[read], default_store)
    assert "_OpaqueStore" in message
    assert not default_store.exists()


# -- Silent: explicit stores, the default location itself, and copies of a default-store ternary --


@pytest.mark.parametrize("kind", ["directory", "read-only sqlite"])
def test_an_explicit_global_store_beside_a_ternary_store_is_silent(
    explicit_other_store, store_a, tmp_path, caplog, kind
):
    data_dir = store_a if kind == "directory" else _read_only_sqlite(tmp_path)
    _assert_silent(_ternary(data_dir), caplog)


def test_an_explicit_global_store_never_reaches_the_location_check(explicit_other_store, caplog):
    _assert_silent(_ternary(_OpaqueStore()), caplog)


@pytest.mark.parametrize("spelling", ["none", "path", "str", "unnormalized"])
def test_a_ternary_on_the_default_location_is_silent(default_store, caplog, spelling):
    data_dir = {
        "none": None,
        "path": default_store,
        "str": str(default_store),
        "unnormalized": default_store.parent / "sub" / ".." / default_store.name,
    }[spelling]
    _assert_silent(_ternary(data_dir), caplog)
    assert not default_store.exists()


def test_a_reordered_copy_of_a_default_store_ternary_is_silent(default_store, caplog):
    ti = _ternary()
    copy = ti.with_component_order(["Si", "Mg", "Al"])
    assert copy is not ti
    assert copy.data_dir is ti.data_dir
    _assert_silent(copy, caplog)
    assert not default_store.exists()


def test_an_overlay_over_the_default_store_is_silent(default_store, caplog):
    _assert_silent(_ternary(OverlayBackend(None)), caplog)
    assert not default_store.exists()
