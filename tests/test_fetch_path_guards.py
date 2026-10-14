"""The four remote entry points refuse a read-only store before fetching, and only an MPDS
``204`` is recorded as "no diagram".

Every test is hermetic: stores live in ``tmp_path``, the network entry points
(``api._get_dft_entries_from_components``, ``api.get_mpds_client``, ``api.mp_rester``) are
counters or tripwires, and ``api.get_api_key`` is stubbed. Tests that need the online path
switch offline mode off with ``monkeypatch``.
"""

import json
import logging
from pathlib import Path

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    KIND_MPDS,
    CacheKey,
    DirectoryBackend,
    SqliteBackend,
    close_sqlite_backends,
)

from ._mpds_builders import Tripwire
from ._mpds_builders import cu_mg_entry_pool as _entry_pool

SYSTEM = "Cu-Mg"
DFT_KEY = CacheKey(SYSTEM, KIND_DFT_ENTRIES, "GGA")
DFT_RECORD = "Cu-Mg_ENTRIES_MP_GGA.json"
MPDS_RECORD = {None: "Cu-Mg.json", 0: "Cu-Mg_MPDS_PD_0.json"}
MPDS_VARIANT = {None: "", 0: "0"}
MPDS_FIELDS = ["chemical_elements", "entry", "comp_range", "temp", "labels", "shapes", "reference"]


class _FakeAPIError(Exception):
    """Stands in for ``mpds_client.APIError(msg, code)``."""

    def __init__(self, msg, code=0):
        super().__init__(msg)
        self.msg = msg
        self.code = code


class _FakeMpdsClient:
    """An MPDS client whose ``get_data`` returns ``rows`` or raises ``error``."""

    def __init__(self, rows=(), error=None):
        self.rows = list(rows)
        self.error = error
        self.get_data_calls = 0

    def get_data(self, *args, **kwargs):
        self.get_data_calls += 1
        if self.error is not None:
            raise self.error
        return list(self.rows)


class _Network:
    """Replaces every network entry point in ``api`` with a counter or a tripwire."""

    def __init__(self, monkeypatch, *, mpds_key=True, client=None):
        self.dft_fetches = 0
        self.mpds_clients = 0
        self.client = client if client is not None else _FakeMpdsClient()
        self.mp_rester = Tripwire()
        monkeypatch.setattr(api, "_get_dft_entries_from_components", self._fetch)
        monkeypatch.setattr(api, "get_mpds_client", self._get_mpds_client)
        monkeypatch.setattr(api, "mp_rester", self.mp_rester)
        monkeypatch.setattr(api, "get_api_key", lambda name: "x" * 32 if mpds_key else None)

    def _fetch(self, components, dft_type, keep_data=False):
        self.dft_fetches += 1
        return _entry_pool()

    def _get_mpds_client(self, *args, **kwargs):
        self.mpds_clients += 1
        return self.client


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Online by default; every read-only store opened here is closed on teardown."""
    monkeypatch.setattr(config, "offline", False)
    yield
    close_sqlite_backends()


def _read_only_store(path: Path, records: dict) -> SqliteBackend:
    """A read-only ``SqliteBackend`` at ``path`` holding ``records`` (``CacheKey -> payload``)."""
    writer = SqliteBackend(path, create=True)
    try:
        for key, payload in records.items():
            writer.write_json(key, payload)
    finally:
        writer.close()
    backend = SqliteBackend(path)
    assert "write" not in backend.capabilities
    return backend


@pytest.fixture
def ro_store(tmp_path):
    """A read-only store on disk holding another system's DFT record, not Cu-Mg's."""
    path = tmp_path / "ro.sqlite"
    backend = _read_only_store(path, {CacheKey("Ag-V", KIND_DFT_ENTRIES, "GGA"): _entry_pool()})
    yield backend
    backend.close()


def _point_global_store_at(monkeypatch, path: Path, mode: str) -> None:
    monkeypatch.setattr(config, "cache_dir", path)
    monkeypatch.setattr(config, "cache_mode", mode)
    monkeypatch.setattr(config, "dir_structure", "flat")


DFT_CALLS = {
    "get_dft_convexhull": lambda store: api.get_dft_convexhull(SYSTEM, data_dir=store),
    "get_dft_structure_entries": lambda store: api.get_dft_structure_entries(
        SYSTEM, data_dir=store
    ),
    "cache_imputed_entries": lambda store: api.cache_imputed_entries(SYSTEM, [], data_dir=store),
}


# -- A read-only store fails before the fetch --


class TestReadOnlyStoreRefusesBeforeFetching:
    @pytest.mark.parametrize("name", list(DFT_CALLS))
    def test_a_dft_miss_raises_without_fetching(self, monkeypatch, ro_store, name):
        net = _Network(monkeypatch)
        with pytest.raises(config.CacheModeError) as excinfo:
            DFT_CALLS[name](ro_store)
        assert net.dft_fetches == 0
        message = str(excinfo.value)
        assert str(ro_store.path) in message
        assert DFT_RECORD in message

    def test_a_record_without_structures_is_not_refreshed(self, monkeypatch, tmp_path):
        store = _read_only_store(tmp_path / "held.sqlite", {DFT_KEY: _entry_pool()})
        try:
            net = _Network(monkeypatch)
            with pytest.raises(config.CacheModeError) as excinfo:
                api.get_dft_structure_entries(SYSTEM, data_dir=store)
            assert net.dft_fetches == 0
            assert DFT_RECORD in str(excinfo.value)
        finally:
            store.close()

    @pytest.mark.parametrize("pd_ind", [None, 0])
    def test_an_mpds_miss_raises_without_constructing_a_client(self, monkeypatch, ro_store, pd_ind):
        _point_global_store_at(monkeypatch, ro_store.path, "sqlite")
        net = _Network(monkeypatch)
        with pytest.raises(config.CacheModeError) as excinfo:
            mpds.load_mpds_data(SYSTEM, pd_ind=pd_ind)
        assert net.mpds_clients == 0
        assert net.client.get_data_calls == 0
        message = str(excinfo.value)
        assert str(ro_store.path) in message
        assert MPDS_RECORD[pd_ind] in message


class TestOfflineStillWins:
    @pytest.mark.parametrize("name", list(DFT_CALLS))
    def test_a_dft_miss_raises_the_offline_error(self, monkeypatch, ro_store, name):
        mp_rester = Tripwire()
        monkeypatch.setattr(api, "mp_rester", mp_rester)
        monkeypatch.setattr(api, "get_api_key", lambda name: "x" * 32)
        monkeypatch.setattr(config, "offline", True)
        with pytest.raises(config.OfflineError) as excinfo:
            DFT_CALLS[name](ro_store)
        assert type(excinfo.value) is config.OfflineError
        assert "Fetching Materials Project DFT entries for 'Cu-Mg'" in str(excinfo.value)
        assert mp_rester.calls == 0

    def test_an_mpds_miss_raises_the_offline_error(self, monkeypatch, ro_store):
        _point_global_store_at(monkeypatch, ro_store.path, "sqlite")
        client = Tripwire()
        monkeypatch.setattr(api, "get_mpds_client", client)
        monkeypatch.setattr(api, "get_api_key", lambda name: "x" * 32)
        monkeypatch.setattr(config, "offline", True)
        with pytest.raises(config.OfflineError) as excinfo:
            mpds.load_mpds_data(SYSTEM)
        assert type(excinfo.value) is config.OfflineError
        assert "Fetching the MPDS phase diagram for 'Cu-Mg'" in str(excinfo.value)
        assert client.calls == 0


class TestWritableStoreStillFetches:
    @pytest.mark.parametrize("name", list(DFT_CALLS))
    def test_a_dft_miss_fetches_once(self, monkeypatch, tmp_path, name):
        net = _Network(monkeypatch)
        DFT_CALLS[name](DirectoryBackend(tmp_path))
        assert net.dft_fetches == 1
        assert (tmp_path / DFT_RECORD).is_file()

    def test_an_mpds_miss_constructs_the_client_once(self, monkeypatch, tmp_path):
        _point_global_store_at(monkeypatch, tmp_path, "directory")
        monkeypatch.setattr(api, "mpds_api_error", lambda: _FakeAPIError)
        net = _Network(monkeypatch, client=_FakeMpdsClient(error=_FakeAPIError("No hits", 204)))
        mpds_json, _ = mpds.load_mpds_data(SYSTEM)
        assert net.mpds_clients == 1
        assert net.client.get_data_calls == 1
        assert mpds_json == {"reference": None}


def test_without_an_mpds_key_a_read_only_miss_returns_the_placeholder(monkeypatch, ro_store):
    _point_global_store_at(monkeypatch, ro_store.path, "sqlite")
    net = _Network(monkeypatch, mpds_key=False)
    mpds_json, (liquidus, _) = mpds.load_mpds_data(SYSTEM)
    assert mpds_json == {"reference": None}
    assert liquidus is None
    assert net.mpds_clients == 0
    assert ro_store.variants(SYSTEM, KIND_MPDS) == []


# -- Only an MPDS 204 is recorded as "no diagram" --


def _mpds_errors(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == mpds.logger.name and r.levelno == logging.ERROR]


class TestMpdsClientErrors:
    @pytest.mark.parametrize("pd_ind", [None, 0])
    @pytest.mark.parametrize("code", [0, 2, 401, 429, 500, 503])
    def test_a_failed_request_raises_and_caches_nothing(
        self, monkeypatch, tmp_path, caplog, code, pd_ind
    ):
        _point_global_store_at(monkeypatch, tmp_path, "directory")
        monkeypatch.setattr(api, "mpds_api_error", lambda: _FakeAPIError)
        error = _FakeAPIError(f"request failed ({code})", code)
        _Network(monkeypatch, client=_FakeMpdsClient(error=error))
        with caplog.at_level(logging.ERROR, logger=mpds.logger.name):
            with pytest.raises(_FakeAPIError) as excinfo:
                mpds.load_mpds_data(SYSTEM, pd_ind=pd_ind)
        assert excinfo.value is error
        assert DirectoryBackend(tmp_path).variants(SYSTEM, KIND_MPDS) == []
        errors = _mpds_errors(caplog)
        assert len(errors) == 1
        assert SYSTEM in errors[0].getMessage()
        assert f"code {code}" in errors[0].getMessage()

    def test_the_real_client_error_class_is_raised(self, monkeypatch, tmp_path):
        mpds_client = pytest.importorskip("mpds_client")
        _point_global_store_at(monkeypatch, tmp_path, "directory")
        error = mpds_client.APIError("Too Many Requests", 429)
        _Network(monkeypatch, client=_FakeMpdsClient(error=error))
        with pytest.raises(mpds_client.APIError) as excinfo:
            mpds.load_mpds_data(SYSTEM)
        assert excinfo.value.code == 429
        assert DirectoryBackend(tmp_path).variants(SYSTEM, KIND_MPDS) == []

    @pytest.mark.parametrize("pd_ind", [None, 0])
    def test_a_204_writes_the_placeholder_record(self, monkeypatch, tmp_path, caplog, pd_ind):
        _point_global_store_at(monkeypatch, tmp_path, "directory")
        monkeypatch.setattr(api, "mpds_api_error", lambda: _FakeAPIError)
        _Network(monkeypatch, client=_FakeMpdsClient(error=_FakeAPIError("No hits", 204)))
        with caplog.at_level(logging.INFO, logger=mpds.logger.name):
            mpds_json, (liquidus, _) = mpds.load_mpds_data(SYSTEM, pd_ind=pd_ind)
        assert mpds_json == {"reference": None}
        assert liquidus is None
        store = DirectoryBackend(tmp_path)
        assert store.variants(SYSTEM, KIND_MPDS) == [MPDS_VARIANT[pd_ind]]
        assert json.loads((tmp_path / MPDS_RECORD[pd_ind]).read_text()) == {"reference": None}
        assert "Got 0 hits" in caplog.text
        assert _mpds_errors(caplog) == []


# -- The MPDS cache log line names the cached diagram --


def _diagram_row(entry: str) -> list:
    """One ``get_data`` row, in ``MPDS_FIELDS`` order, copied from the shipped Cu-Mg diagram."""
    diagram = json.loads((Path(__file__).resolve().parents[1] / "cache" / "Cu-Mg.json").read_text())
    diagram["reference"] = {"entry": entry}
    return [diagram[field] for field in MPDS_FIELDS]


def test_the_cache_log_line_names_the_cached_entry(monkeypatch, tmp_path, caplog):
    first, second = "https://mpds.io/entry/C000001", "https://mpds.io/entry/C000002"
    _point_global_store_at(monkeypatch, tmp_path, "directory")
    _Network(monkeypatch, client=_FakeMpdsClient(rows=[_diagram_row(first), _diagram_row(second)]))
    with caplog.at_level(logging.INFO, logger=mpds.logger.name):
        mpds_json, _ = mpds.load_mpds_data(SYSTEM)
    assert mpds_json["reference"]["entry"] == first
    caching = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Caching binary phase data from entry at")
    ]
    assert len(caching) == 1
    assert first in caching[0]
    assert second not in caching[0]
