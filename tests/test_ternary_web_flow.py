"""Session-only ternary imputed phases over a read-only SQLite store.

A deployment serves a read-only single-file store. Each session wraps it in its own
:class:`~gliquid.cache.OverlayBackend`; imputing with ``write_cache=True`` writes to that
overlay, a ternary object rebuilt on the same overlay with ``include_imputed=True`` places the
phase again, another session's overlay does not see it, and the store file is never written.

Runs offline against a store migrated from the shipped Al-Mg-Si entries record.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

import gliquid.config as config
import gliquid.imputation.ternary_imputation as imp_ternary
from gliquid.cache import (
    CacheModeError,
    OverlayBackend,
    close_sqlite_backends,
    imputed_entry_label,
    main,
    resolve_backend,
)
from gliquid.ternary import TernaryLiquidInterpolation

COMPONENTS = ["Al", "Mg", "Si"]
MIXING = {
    "Al-Mg": [-20000.0, -4.0, -5000.0, 1.0],
    "Mg-Si": [-30000.0, -6.0, 0.0, 0.0],
    "Si-Al": [-10000.0, -2.0, 2000.0, 0.0],
}
TARGET = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
TEMP_K = 1100.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(store, **kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(COMPONENTS),
        store,
        delta=0.1,
        param_format="combined",
        xs_mix=MIXING,
        order="given",
        hull_mode="hsx",
        **kwargs,
    )


@pytest.fixture
def sqlite_store(tmp_path) -> Path:
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("needs a directory-mode cache corpus")
    record = Path(source) / "Al-Mg-Si_ENTRIES_MP_GGA.json"
    if not record.exists():
        pytest.skip(f"no shipped {record.name}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy2(record, corpus / record.name)
    dest = tmp_path / "store.sqlite"
    assert main(["migrate", "--from", str(corpus), "--to", str(dest)]) == 0
    yield dest
    close_sqlite_backends()


@pytest.fixture
def deployment():
    """Offline, persistence knob off -- the web app's settings."""
    offline, persist = config.offline, config.persist_ternary_imputed
    config.set_offline(True)
    config.set_persist_ternary_imputed(False)
    yield
    config.set_offline(offline)
    config.set_persist_ternary_imputed(persist)


class TestSessionStore:
    def test_a_session_keeps_its_phase_and_nothing_else_sees_it(self, deployment, sqlite_store):
        before = sha256(sqlite_store)
        base = resolve_backend(sqlite_store)
        assert "write" not in base.capabilities
        session_a = OverlayBackend(base, label="A")
        session_b = OverlayBackend(base, label="B")

        record = build(session_a, include_imputed=True).impute_phase(
            TARGET, TEMP_K, name="TX", write_cache=True
        )
        assert record.accepted and record.written, record.reason
        assert abs(record.realized_temperature_k - TEMP_K) <= imp_ternary.HSX_TOLERANCE_K

        rebuilt = build(session_a, include_imputed=True)
        placed = rebuilt.all_imputed_entries()
        assert [imputed_entry_label(e) for e in placed] == ["TX"]
        assert placed[0]["data"]["h_form_j_per_mol"] == record.enthalpy
        rebuilt.process_data()
        top = float(rebuilt.get_inter_melting_temps(["TX"])["TX"]) + 273.15
        assert abs(top - TEMP_K) <= imp_ternary.HSX_TOLERANCE_K

        assert build(session_b, include_imputed=True).all_imputed_entries() == []
        assert build(sqlite_store, include_imputed=True).all_imputed_entries() == []

        assert sha256(sqlite_store) == before
        leftovers = sorted(p.name for p in sqlite_store.parent.glob("store.sqlite*"))
        assert leftovers == ["store.sqlite"]

    def test_control_the_overlay_is_what_holds_the_phase(self, deployment, sqlite_store):
        session = OverlayBackend(sqlite_store)
        build(session).impute_phase(TARGET, TEMP_K, name="TX", write_cache=True)
        assert build(session, include_imputed=False).all_imputed_entries() == []
        session.discard()
        assert build(session, include_imputed=True).all_imputed_entries() == []

    def test_writing_to_the_read_only_store_is_refused(self, deployment, sqlite_store):
        ti = build(sqlite_store)
        with pytest.raises(CacheModeError):
            ti.impute_phase(TARGET, TEMP_K, name="TX", write_cache=True)
        assert ti.imputed_entries == []

    def test_knob_off_keeps_the_phase_on_the_object_only(self, deployment, sqlite_store):
        ti = build(sqlite_store)
        record = ti.impute_phase(TARGET, TEMP_K, name="TX")
        assert record.accepted and not record.written
        assert [imputed_entry_label(e) for e in ti.imputed_entries] == ["TX"]
