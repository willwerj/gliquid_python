"""Which cached diagram ``load_mpds_data`` resolves, for every ``pd_ind`` / store shape.

Two cache namings coexist: indexed ``<sys>_MPDS_PD_<n>.json``, and indexless ``<sys>.json``
wherever a store holds one diagram per system (the shipped ``cache/``). ``load_mpds_data``
serves both:

* ``pd_ind=None`` prefers the indexless file, so an indexless sibling SHADOWS PD_0; a warning
  fires on exactly that ambiguous case.
* ``pd_ind=0`` on an indexless-only store resolves the sole indexless record rather than
  falling through to the API branch and returning ``{"reference": None}``.

Everything runs offline against copies of the shipped Ag-V fixture in ``tmp_path``; the MPDS
client is monkeypatched to raise so any fall-through to the API branch fails loudly.
"""

import json
import logging
import shutil
from pathlib import Path

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds

SYSTEM = "Ag-V"
SENTINEL_ENTRY = "https://mpds.io/entry/C000000-INDEXED"


def _no_mpds_client(*args, **kwargs):
    raise AssertionError("fell through to the API branch: the live MPDS client was constructed")


@pytest.fixture
def flat_store(tmp_path, monkeypatch):
    """A flat cache dir the test fills, with the live MPDS client disarmed."""
    monkeypatch.setattr(api, "get_mpds_client", _no_mpds_client)
    original_dir, original_struct = config.cache_dir, config.dir_structure
    config.set_cache_dir(tmp_path)
    config.set_dir_structure("flat")
    try:
        yield tmp_path
    finally:
        config.set_cache_dir(original_dir)
        config.set_dir_structure(original_struct)


def _shipped_fixture() -> Path:
    """The shipped indexless Ag-V json in this checkout's ``cache/``."""
    return Path(__file__).resolve().parents[1] / "cache" / f"{SYSTEM}.json"


def _put_indexless(store: Path) -> None:
    shutil.copy(_shipped_fixture(), store / f"{SYSTEM}.json")


def _put_indexed(store: Path, ind: int = 0) -> None:
    """An indexed sibling that is distinguishable from the indexless one by entry alone."""
    js = json.loads(_shipped_fixture().read_text())
    js["reference"]["entry"] = SENTINEL_ENTRY
    (store / f"{SYSTEM}_MPDS_PD_{ind}.json").write_text(json.dumps(js))


def _entry(mpds_json: dict):
    return (mpds_json.get("reference") or {}).get("entry")


class TestIndexlessShadowingWarning:
    """``pd_ind=None`` warns only when an indexless file actually shadows an indexed one."""

    def test_shadowed_store_resolves_indexless_and_warns(self, flat_store, caplog):
        _put_indexless(flat_store)
        _put_indexed(flat_store)
        with caplog.at_level(logging.WARNING, logger=mpds.logger.name):
            mpds_json, _ = mpds.load_mpds_data(SYSTEM)
        # behaviour is unchanged — the indexless file still wins, but audibly (a warning)
        assert _entry(mpds_json) != SENTINEL_ENTRY
        assert "shadows" in caplog.text
        assert f"{SYSTEM}_MPDS_PD_0.json" in caplog.text

    def test_indexless_only_store_is_silent(self, flat_store, caplog):
        _put_indexless(flat_store)
        with caplog.at_level(logging.WARNING, logger=mpds.logger.name):
            mpds_json, (liq, _) = mpds.load_mpds_data(SYSTEM)
        assert _entry(mpds_json) is not None and liq
        assert "shadows" not in caplog.text, (
            "indexless is the legitimate naming for a single-diagram store"
        )

    def test_indexed_only_store_is_silent(self, flat_store, caplog):
        _put_indexed(flat_store)
        with caplog.at_level(logging.WARNING, logger=mpds.logger.name):
            mpds_json, _ = mpds.load_mpds_data(SYSTEM)
        assert _entry(mpds_json) == SENTINEL_ENTRY
        assert "shadows" not in caplog.text

    def test_explicit_index_ignores_the_indexless_sibling(self, flat_store, caplog):
        _put_indexless(flat_store)
        _put_indexed(flat_store)
        with caplog.at_level(logging.WARNING, logger=mpds.logger.name):
            mpds_json, _ = mpds.load_mpds_data(SYSTEM, pd_ind=0)
        assert _entry(mpds_json) == SENTINEL_ENTRY
        assert "shadows" not in caplog.text, "pinning the index is not the ambiguous case"


class TestExplicitIndexOnIndexlessStore:
    """``pd_ind=0`` must never resolve to an empty diagram where a cached one exists."""

    def test_pd_ind_zero_resolves_the_sole_indexless_diagram(self, flat_store):
        _put_indexless(flat_store)
        mpds_json, (liq, _) = mpds.load_mpds_data(SYSTEM, pd_ind=0)
        assert _entry(mpds_json) is not None, (
            "pd_ind=0 fell through to the API branch and returned an empty diagram"
        )
        assert liq, "pd_ind=0 must yield the same digitized liquidus as pd_ind=None"

    def test_pd_ind_zero_matches_pd_ind_none_on_an_indexless_store(self, flat_store):
        _put_indexless(flat_store)
        implicit, (implicit_liq, _) = mpds.load_mpds_data(SYSTEM)
        explicit, (explicit_liq, _) = mpds.load_mpds_data(SYSTEM, pd_ind=0)
        assert implicit == explicit
        assert implicit_liq == explicit_liq

    def test_nonzero_index_on_an_indexless_store_raises(self, flat_store):
        _put_indexless(flat_store)
        with pytest.raises(ValueError, match=r"pd_ind=1"):
            mpds.load_mpds_data(SYSTEM, pd_ind=1)

    def test_missing_index_beside_pd0_still_raises(self, flat_store):
        """The pre-existing guard is untouched when PD_0 is present."""
        _put_indexed(flat_store)
        with pytest.raises(ValueError, match=r"No matching json with pd_ind=3"):
            mpds.load_mpds_data(SYSTEM, pd_ind=3)

    def test_empty_store_still_reaches_the_api_branch(self, flat_store):
        """No cached file of either naming is a genuine cache miss, not a resolution bug."""
        if not api.get_api_key(api.MPDS_KEY_VAR):
            mpds_json, (liq, _) = mpds.load_mpds_data(SYSTEM, pd_ind=0)
            assert _entry(mpds_json) is None and liq is None
        else:
            with pytest.raises(AssertionError, match="fell through to the API branch"):
                mpds.load_mpds_data(SYSTEM, pd_ind=0)
