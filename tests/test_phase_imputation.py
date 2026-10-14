"""Smoke tests for the phase-energy imputation library (gliquid.imputation.binary_imputation).

Cu-Mg (two congruently melting intermetallics) is loaded via from_cache, with session-scoped
fixtures and small n_opts/max_iter.

The closed-loop recovery test removes a real DFT compound from both the hull and the phase
list, re-fits, then imputes it back from the experimental invariant and checks it reappears
as a tagged, dashed-line, stable vertex.

``TestFitIsolation`` asserts that a cached imputed entry does not change what a later fit
sees. It runs on a temporary cache copy, with ``include_imputed=True`` on the same cache as
the positive control that the entry was written.
"""

import copy
import inspect
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.config as config
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.core as imp_core
import gliquid.mpds as mpds
import gliquid.solid_coverage as solid_coverage
from gliquid.binary import (
    BinaryLiquid,
    BLPlotter,
    build_phases_from_chull,
    hull_field_disagreement,
)
from gliquid.cache import is_imputed_entry_dict
from gliquid.imputation.binary_imputation import (
    BinaryImputationResult,
    _binary_provenance,
    _controlling_invariant,
    _hull_points,
    _imputed_computed_entry,
    _liquid_gibbs,
    impute_binary_phases,
    invert_binary_phase_energy,
    list_binary_imputation_targets,
    select_best_fit,
)
from gliquid.imputation.lifecycle import (
    drop_imputed_phases,
    imputed_entry_staleness,
    stale_imputed_entries,
)
from gliquid.phase import EV_ATOM_TO_J_MOL, UNARY
from gliquid.plotting.binary_figs import imputed_hull_entries
from gliquid.solution import x_vals

from ._mpds_builders import SMOKE_FIT_KWARGS, strip_interior

SYSTEM = "Cu-Mg"


# -- Helpers --


def _interior_phases(bl):
    """Interior fixed-composition phases (a solution phase has composition None)."""
    return [
        p
        for p in bl.phases
        if p.composition is not None and 0.0 < round(float(bl._phase_x(p)), 6) < 1.0
    ]


def _strip_interior_from_cache(cache_dir):
    """Remove the interior DFT entry records from a cached entries file, so a later
    ``from_cache`` load builds the element-only hull ``strip_interior`` leaves in memory.

    Tagged imputed records are kept. Returns the removed entry ids.
    """
    path = Path(cache_dir) / f"{SYSTEM}_ENTRIES_MP_GGA.json"
    raw = json.loads(path.read_text())
    kept, removed = [], []
    for entry in raw:
        elements = [el for el, amt in (entry.get("composition") or {}).items() if amt]
        keep = len(elements) == 1 or is_imputed_entry_dict(entry)
        (kept if keep else removed).append(entry)
    assert removed, f"no interior {SYSTEM} records in {path.name} to strip"
    path.write_text(json.dumps(kept))
    return [e.get("entry_id") for e in removed]


# -- Shared fixtures (cu_mg_fit and cu_mg_full come from tests/conftest.py) --


@pytest.fixture
def stripped(cu_mg_full):
    """A copy of the fitted system with all interior DFT phases stripped (good liquid fit
    retained). Returns ``(bl, original_enthalpies)``."""
    bl = copy.deepcopy(cu_mg_full)
    orig = {p.name: p.enthalpy for p in _interior_phases(bl)}
    strip_interior(bl)
    return bl, orig


@pytest.fixture
def imputation_recovery(stripped):
    """Impute the stripped interior phases back from the experimental invariants, anchored
    on the retained full-hull liquid curve (write_cache=False).

    Returns ``(bl, original_enthalpies, imputed_list)``.
    """
    bl, orig = stripped
    imputed = impute_binary_phases(bl, fitting_data=None, write_cache=False)
    return bl, orig, imputed


# -- Detection + inversion units --


class TestInversionUnits:
    def test_congruent_inversion_lies_on_liquid(self, cu_mg_full):
        x_s, temp = 0.4, 1100.0
        inv = {"type": "cmp", "comp": x_s, "temp": temp, "phases": ["X"], "phase_comps": [x_s]}
        assert invert_binary_phase_energy(cu_mg_full, x_s, inv) == pytest.approx(
            _liquid_gibbs(cu_mg_full, x_s, temp)
        )

    def test_a_non_congruent_anchor_reads_the_tangent_not_the_curve(self, cu_mg_full):
        """The whole content of the non-``cmp`` branch, checked against its own definition.

        ``H = G_liq(x_inv, T) + G'_liq(x_inv, T) (x_s - x_inv)`` with the slope a central
        difference of half-width ``dx``. Asserting only that the result is finite would pass
        on a function that returned ``G_liq(x_s, T)``, which is the congruent answer.
        """
        x_inv, x_s, temp, dx = 0.30, 0.55, 1000.0, 1e-3
        inv = {"type": "per", "comp": x_inv, "temp": temp, "phases": ["X"], "phase_comps": [x_s]}
        g_inv = _liquid_gibbs(cu_mg_full, x_inv, temp)
        slope = (
            _liquid_gibbs(cu_mg_full, x_inv + dx, temp)
            - _liquid_gibbs(cu_mg_full, x_inv - dx, temp)
        ) / (2 * dx)
        expected = g_inv + slope * (x_s - x_inv)
        got = invert_binary_phase_energy(cu_mg_full, x_s, inv, dx=dx)
        assert got == pytest.approx(expected)
        # and it is NOT the congruent read-off, which is what a collapsed branch would give
        assert got != pytest.approx(_liquid_gibbs(cu_mg_full, x_s, temp))

    def test_a_eutectic_anchor_takes_the_same_tangent_branch(self, cu_mg_full):
        x_inv, x_s, temp = 0.30, 0.55, 1000.0
        base = {"comp": x_inv, "temp": temp, "phases": ["X"], "phase_comps": [x_s]}
        per = invert_binary_phase_energy(cu_mg_full, x_s, {"type": "per", **base})
        eut = invert_binary_phase_energy(cu_mg_full, x_s, {"type": "eut", **base})
        assert per == pytest.approx(eut)

    def test_imputed_entry_dict_is_tagged(self, cu_mg_full):
        ed = _imputed_computed_entry(cu_mg_full, "TestPhase", 0.5, -25000.0)
        assert is_imputed_entry_dict(ed)
        assert ed["entry_id"] == "imputed:TestPhase"
        assert ed["data"]["imputed"] is True
        # A plain DFT entry dict (no tag) must not be flagged.
        assert not is_imputed_entry_dict({"entry_id": "mp-123", "data": {}})

    def test_the_data_flag_alone_marks_an_entry_imputed(self):
        assert is_imputed_entry_dict({"entry_id": "mp-9999", "data": {"imputed": True}})

    def test_empty_fitting_data_returns_none(self, cu_mg_full):
        assert select_best_fit(copy.deepcopy(cu_mg_full), []) is None


# -- Closed-loop recovery --


class TestImputationRecovery:
    def test_detects_missing_phases(self, stripped):
        bl, orig = stripped
        targets = list_binary_imputation_targets(bl)
        detected = {ph["name"] for ph, _, _ in targets}
        # Both congruently melting Cu-Mg intermetallics are missing from the hull.
        assert detected, "Stripped interior phases should be detected as imputable"
        for name in orig:
            assert name in detected, f"{name} should be detected as imputable"

    def test_imputes_phases_back(self, imputation_recovery):
        _, orig, imputed = imputation_recovery
        assert imputed, "Expected imputed phases"
        assert all(isinstance(ip, BinaryImputationResult) for ip in imputed)
        imputed_names = {ip.name for ip in imputed}
        for name in orig:
            assert name in imputed_names, f"{name} should be imputed back"

    def test_imputed_phases_are_stable_and_tagged(self, imputation_recovery):
        bl, orig, imputed = imputation_recovery
        flagged = {p.name for p in bl.phases if p.imputed}
        for name in orig:
            assert name in flagged, f"{name} should be a tagged stable vertex in bl.phases"
        # Imputed formation enthalpies are exothermic.
        assert all(ip.enthalpy < 0.0 for ip in imputed)

    def test_recovered_enthalpy_is_in_range(self, imputation_recovery):
        # Congruent inversion (H = G_liq at the melting point) should land in the same
        # ballpark as the true DFT formation enthalpy — same sign, comparable magnitude.
        _, orig, imputed = imputation_recovery
        by_name = {ip.name: ip.enthalpy for ip in imputed}
        for name, h_dft in orig.items():
            h_imp = by_name[name]
            assert h_imp < 0.0 and h_dft < 0.0
            ratio = h_imp / h_dft
            assert 0.25 < ratio < 4.0, f"{name}: imputed {h_imp:.0f} vs DFT {h_dft:.0f} J/mol-atom"

    def test_imputed_entry_round_trips_and_filters(self, imputation_recovery):
        _, _, imputed = imputation_recovery
        ed = imputed[0].entry_dict
        assert is_imputed_entry_dict(ed)
        # include_imputed=False would drop it; the tag is what distinguishes it.
        kept = [e for e in [ed, {"entry_id": "mp-1", "data": {}}] if not is_imputed_entry_dict(e)]
        assert ed not in kept

    def test_hull_interp_matches_the_augmented_phase_list(self, imputation_recovery):
        bl, _, _ = imputation_recovery
        pts = _hull_points(bl)
        assert np.array_equal(
            bl.eqs["h_hull_interp"], np.interp(x_vals[1:-1], pts[:, 0], pts[:, 1])
        )

    def test_unmask_clears_the_unsupported_ranges(self, stripped):
        bl, _ = stripped
        bl.unsupported_comp_ranges = [[0.2, 0.5]]
        impute_binary_phases(bl, fitting_data=None, write_cache=False, unmask=True)
        assert bl.unsupported_comp_ranges == []

    def test_unmask_false_leaves_the_ranges_alone(self, stripped):
        bl, _ = stripped
        bl.unsupported_comp_ranges = [[0.2, 0.5]]
        impute_binary_phases(bl, fitting_data=None, write_cache=False, unmask=False)
        assert bl.unsupported_comp_ranges == [[0.2, 0.5]]

    def test_verbose_reports_through_the_logger_not_stdout(self, stripped, caplog, capsys):
        """The package reports through ``logging``; ``test_logging_boundary`` enforces it."""
        bl, orig = stripped
        with caplog.at_level(logging.INFO, logger=imp_binary.__name__):
            impute_binary_phases(bl, fitting_data=None, write_cache=False, verbose=True)
        messages = " ".join(r.getMessage() for r in caplog.records)
        for name in orig:
            assert name in messages
        assert capsys.readouterr().out == ""


# -- Fit isolation --


@pytest.fixture
def isolated_cache(tmp_path):
    """A throwaway copy of the Cu-Mg cache records, installed as the active corpus.

    The imputed-entry write in these tests must never touch the package cache, and the
    saved/restored cache dir keeps the rest of the suite on the real corpus.
    """
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("fit-isolation test needs a directory-mode cache corpus")
    copied = 0
    for path in sorted(Path(source).glob(f"{SYSTEM}*.json")):
        shutil.copy2(path, tmp_path / path.name)
        copied += 1
    if not copied:
        pytest.skip(f"no cached {SYSTEM} records to copy")

    saved_dir, saved_mode = config.cache_dir, config.cache_mode
    config.set_cache_dir(tmp_path)
    try:
        yield tmp_path
    finally:
        config.set_cache_dir(saved_dir)
        config.set_cache_mode(saved_mode)


class TestFitIsolation:
    def test_cached_imputed_entry_does_not_change_a_refit(self, cu_mg_fit, isolated_cache):
        """A cached imputed entry leaves a later fit BIT-IDENTICAL.

        ``include_imputed`` defaults False in ``api.get_dft_convexhull`` and
        ``BinaryLiquid.from_cache``, and ``bl.phases`` is populated once at construction. The
        fit before the write is the shared ``cu_mg_fit``, made on the package store.
        """
        before, fits_before = cu_mg_fit
        assert fits_before, f"{SYSTEM}: baseline fit produced nothing"
        params_before = list(before.get_params())
        phase_names_before = [p.name for p in before.phases]

        # A synthetic imputed vertex deep below the real hull, at a composition the real
        # hull has no vertex at — if it were ever admitted, the fit could not be identical.
        entry = _imputed_computed_entry(before, "ImputedProbe", 0.5, -60000.0)
        api.cache_imputed_entries(SYSTEM, [entry])

        after = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        fits_after = after.fit_parameters(**SMOKE_FIT_KWARGS)
        assert fits_after, f"{SYSTEM}: post-cache fit produced nothing"

        assert [p.name for p in after.phases] == phase_names_before
        assert not any(p.imputed for p in after.phases)
        assert list(after.get_params()) == params_before, (
            "a cached imputed entry changed the fitted parameters — fit isolation is broken"
        )

    def test_positive_control_include_imputed_sees_the_same_entry(self, isolated_cache):
        """The isolation assertion is only meaningful if the entry really is in the cache."""
        bl = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        if bl.init_error:
            pytest.skip(f"{SYSTEM}: init error on load")
        entry = _imputed_computed_entry(bl, "ImputedProbe", 0.5, -60000.0)
        api.cache_imputed_entries(SYSTEM, [entry])

        opted_in = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp", include_imputed=True)
        imputed_names = {p.name for p in opted_in.phases if p.imputed}
        assert "ImputedProbe" in imputed_names, (
            "include_imputed=True did not surface the cached entry — the isolation test "
            "above would pass vacuously"
        )

        default = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        assert not any(p.imputed for p in default.phases)

    def test_write_cache_round_trips_through_the_real_cache_path(self, stripped, isolated_cache):
        """``impute_binary_phases(write_cache=True)`` — the opt-in store write — writes tagged
        entries that a later DEFAULT load still cannot see.

        The compounds leave the cache as well as the in-memory hull, so what is written back
        is absent from the corpus a reload reads. The liquid curve is the full-hull fit.
        """
        bl, _ = stripped
        _strip_interior_from_cache(isolated_cache)
        before = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        phase_names_before = [p.name for p in before.phases]

        imputed = impute_binary_phases(bl, fitting_data=None, write_cache=True)
        assert imputed, "expected the stripped compounds to be imputed back"
        written = {ip.name for ip in imputed}

        raw = json.loads((isolated_cache / f"{SYSTEM}_ENTRIES_MP_GGA.json").read_text())
        tagged = [e for e in raw if is_imputed_entry_dict(e)]
        assert {e["entry_id"] for e in tagged} == {f"imputed:{n}" for n in written}

        opted_in = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp", include_imputed=True)
        assert {p.name for p in opted_in.phases if p.imputed} == written

        after = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        assert not any(p.imputed for p in after.phases)
        assert [p.name for p in after.phases] == phase_names_before

    def test_cache_write_is_idempotent_by_entry_id(self, isolated_cache):
        bl = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        if bl.init_error:
            pytest.skip(f"{SYSTEM}: init error on load")
        entry = _imputed_computed_entry(bl, "ImputedProbe", 0.5, -60000.0)
        api.cache_imputed_entries(SYSTEM, [entry])
        api.cache_imputed_entries(SYSTEM, [entry])

        opted_in = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp", include_imputed=True)
        names = [p.name for p in opted_in.phases if p.imputed]
        assert names.count("ImputedProbe") == 1


def _store_listing(store) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(Path(store).iterdir())}


class TestTheStoreIsWrittenOnlyWhenAsked:
    def test_write_cache_defaults_to_false(self):
        assert inspect.signature(impute_binary_phases).parameters["write_cache"].default is False

    def test_a_defaulted_call_leaves_the_entries_file_byte_identical(
        self, stripped, isolated_cache
    ):
        """A call that imputes at least one phase (the positive control) writes nothing."""
        bl, _ = stripped
        path = isolated_cache / f"{SYSTEM}_ENTRIES_MP_GGA.json"
        before = path.read_bytes()
        listing = _store_listing(isolated_cache)

        imputed = impute_binary_phases(bl, fitting_data=None)

        assert imputed, "expected the stripped compounds to be imputed back"
        assert path.read_bytes() == before
        assert _store_listing(isolated_cache) == listing


class TestTheAnchorIsSharedWithTheCoverageGate:
    """``_controlling_invariant`` is a thin call into ``mpds.controlling_invariant``; these
    pin its eutectic-fallback flag, not the join."""

    @staticmethod
    def _bl(invariants):
        """Only ``.invariants`` is read."""
        return SimpleNamespace(invariants=invariants)

    def test_a_phase_that_only_flanks_a_eutectic_is_still_anchored(self):
        eut = {
            "type": "eut",
            "comp": 0.107,
            "temp": 1000.0,
            "phases": ["A", "AB"],
            "phase_comps": [0.0, 0.400],
        }
        phase = {"name": "AB", "comp": 0.4}
        assert _controlling_invariant(self._bl([eut]), phase, 0.4) is eut
        # ... and the gate's own default does NOT take it, which is the whole reason for the flag
        assert mpds.controlling_invariant(phase, [eut]) is None

    def test_a_melting_record_still_wins(self):
        eut = {
            "type": "eut",
            "comp": 0.107,
            "temp": 1000.0,
            "phases": ["A", "AB"],
            "phase_comps": [0.0, 0.400],
        }
        per = {
            "type": "per",
            "comp": 0.234,
            "temp": 1200.0,
            "phases": ["AB"],
            "phase_comps": [0.400],
        }
        got = _controlling_invariant(self._bl([eut, per]), {"name": "AB", "comp": 0.4}, 0.4)
        assert got is per

    def test_a_phase_that_never_meets_the_liquid_is_not_imputable(self):
        far = {
            "type": "eut",
            "comp": 0.9,
            "temp": 1000.0,
            "phases": ["A2B", "B"],
            "phase_comps": [0.667, 1.0],
        }
        assert _controlling_invariant(self._bl([far]), {"name": "AB", "comp": 0.4}, 0.4) is None
        assert _controlling_invariant(self._bl([]), {"name": "AB", "comp": 0.4}, 0.4) is None


# -- Lifecycle: provenance, staleness, removal --


class TestProvenanceOnTheEntry:
    """A cached imputed entry records what produced it: its anchor and the fit it was read
    off."""

    def test_the_entry_records_its_anchor_and_its_fit(self, imputation_recovery):
        _bl, _orig, imputed = imputation_recovery
        assert imputed, "nothing imputed; the fixture cannot exercise provenance"
        for ip in imputed:
            prov = ip.entry_dict["data"][imp_binary.PROVENANCE_KEY]
            assert prov["calibration"] == "invariant"
            assert prov["gliquid_version"] != ""
            assert prov["param_format"] == "comb-exp"
            assert len(prov["fit_params"]) == 4
            assert prov["anchor"]["type"] == ip.invariant_type
            assert prov["anchor"]["comp"] == pytest.approx(ip.invariant_comp)
            assert prov["anchor"]["temp"] == pytest.approx(ip.invariant_temp)

    def test_an_unknown_anchor_is_recorded_as_null_not_omitted(self, cu_mg_full):
        prov = _binary_provenance(cu_mg_full, None)
        assert "anchor" in prov and prov["anchor"] is None


class TestStaleness:
    """An imputed energy is a function of (anchor, fitted liquid curve). Nothing else."""

    def _entry(self, imputed):
        return copy.deepcopy(imputed[0].entry_dict)

    def test_a_fresh_entry_is_current(self, imputation_recovery):
        bl, _orig, imputed = imputation_recovery
        assert imputed
        assert imputed_entry_staleness(bl, self._entry(imputed)) is None

    def test_moving_the_fit_makes_it_stale(self, imputation_recovery):
        bl, _orig, imputed = imputation_recovery
        entry = self._entry(imputed)
        entry["data"][imp_binary.PROVENANCE_KEY]["fit_params"][0] += 1000.0
        assert imputed_entry_staleness(bl, entry) == "fit parameters have moved"

    def test_moving_the_anchor_makes_it_stale(self, imputation_recovery):
        bl, _orig, imputed = imputation_recovery
        entry = self._entry(imputed)
        entry["data"][imp_binary.PROVENANCE_KEY]["anchor"]["temp"] += 25.0
        assert "anchor moved" in imputed_entry_staleness(bl, entry)

    def test_a_different_parameter_format_makes_it_stale(self, imputation_recovery):
        bl, _orig, imputed = imputation_recovery
        entry = self._entry(imputed)
        entry["data"][imp_binary.PROVENANCE_KEY]["param_format"] = "linear"
        assert "param_format" in imputed_entry_staleness(bl, entry)

    def test_an_entry_written_before_provenance_reads_unknown_not_current(
        self, imputation_recovery
    ):
        """An entry with no provenance reads as unknown, not as current."""
        bl, _orig, imputed = imputation_recovery
        entry = self._entry(imputed)
        del entry["data"][imp_binary.PROVENANCE_KEY]
        assert imputed_entry_staleness(bl, entry) == "no provenance recorded"

    def test_a_real_dft_entry_is_not_this_functions_to_judge(self, imputation_recovery):
        """Carries its own positive control: a stale IMPUTED entry alongside is reported."""
        bl, _orig, imputed = imputation_recovery
        real = {"entry_id": "mp-123", "data": {}}
        stale = self._entry(imputed)
        stale["data"][imp_binary.PROVENANCE_KEY]["fit_params"][0] += 1000.0
        reported = stale_imputed_entries(bl, [real, stale])
        assert [e["entry_id"] for e, _r in reported] == [stale["entry_id"]]


class TestTheImputationRoundTrip:
    """``drop_imputed_phases`` closes the symmetry ablate_phases / restore_phases has."""

    def test_dropping_returns_the_un_imputed_phase_set(self, stripped):
        bl, _orig = stripped
        before = sorted(p.name for p in bl.phases)
        imputed = impute_binary_phases(bl, fitting_data=None, write_cache=False)
        assert imputed, "nothing imputed; the round trip would be vacuous"
        assert sorted(p.name for p in bl.phases) != before
        back = drop_imputed_phases(bl)
        assert sorted(p.name for p in back.phases) == before

    def test_the_synthetic_vertices_leave_the_hull_too(self, stripped):
        """Not just ``phases``: a later build_phases_from_chull must not resurrect them."""
        bl, _orig = stripped
        impute_binary_phases(bl, fitting_data=None, write_cache=False)
        assert any(is_imputed_entry_dict(e.as_dict()) for e in bl.dft_ch.all_entries)
        back = drop_imputed_phases(bl)
        assert not any(is_imputed_entry_dict(e.as_dict()) for e in back.dft_ch.all_entries)
        rebuilt = build_phases_from_chull(back.dft_ch, back.components, back.component_data)
        assert not any(p.imputed for p in rebuilt)

    def test_it_leaves_the_source_alone_unless_asked(self, stripped):
        bl, _orig = stripped
        impute_binary_phases(bl, fitting_data=None, write_cache=False)
        n_imputed = sum(1 for p in bl.phases if p.imputed)
        assert n_imputed
        drop_imputed_phases(bl, in_place=False)
        assert sum(1 for p in bl.phases if p.imputed) == n_imputed

    def test_a_system_with_nothing_imputed_is_unchanged(self, cu_mg_full):
        back = drop_imputed_phases(cu_mg_full)
        assert sorted(p.name for p in back.phases) == sorted(p.name for p in cu_mg_full.phases)


class TestDroppingCachedEntries:
    def test_only_imputed_entries_are_removed(self, isolated_cache):
        bl = BinaryLiquid.from_cache(SYSTEM, param_format="comb-exp")
        if bl.init_error:
            pytest.skip(f"{SYSTEM}: init error on load")
        entry = _imputed_computed_entry(bl, "DropProbe", 0.5, -60000.0)
        api.cache_imputed_entries(SYSTEM, [entry])

        path = next(Path(isolated_cache).glob(f"{SYSTEM}_ENTRIES_MP_*.json"))
        n_real_before = sum(1 for e in json.loads(path.read_text()) if not is_imputed_entry_dict(e))
        assert n_real_before, "no real DFT entries to protect; the control is vacuous"

        assert api.drop_imputed_entries(SYSTEM) == 1
        after = json.loads(path.read_text())
        assert not any(is_imputed_entry_dict(e) for e in after)
        assert len(after) == n_real_before

    def test_dropping_nothing_is_not_an_error(self, isolated_cache):
        assert api.drop_imputed_entries(SYSTEM) == 0


# -- Which fit the energies are read off --


def _candidates(bl):
    """Three candidates whose lowest-objective member is not the first one."""
    names = bl.xs_mix.format.param_names
    base = bl.get_params()
    return [
        dict(zip(names, base), f=5.0, nmpath=None),
        dict(zip(names, [v * 0.5 for v in base]), f=1.0, nmpath=None),
        dict(zip(names, [v * 1.5 for v in base]), f=9.0, nmpath=None),
    ]


class TestWhichFitSuppliesTheCurve:
    """``select_best_fit`` picks the lowest-objective fit, the curve an imputed energy is
    read off."""

    def test_the_lowest_objective_fit_is_selected(self, cu_mg_full):
        bl = copy.deepcopy(cu_mg_full)
        cands = _candidates(bl)
        chosen = select_best_fit(bl, cands)
        assert chosen["f"] == 1.0
        assert bl.get_params() == pytest.approx(chosen_params(bl, chosen))

    def test_selection_is_not_positional(self, cu_mg_full):
        """``fit_parameters`` does not sort its output; the best is found by objective."""
        bl = copy.deepcopy(cu_mg_full)
        cands = _candidates(bl)
        assert cands[0]["f"] != min(c["f"] for c in cands)
        assert select_best_fit(bl, cands)["f"] == 1.0

    def test_empty_candidates_select_nothing(self, cu_mg_full):
        assert select_best_fit(copy.deepcopy(cu_mg_full), []) is None


# -- Hull element order, and replaying one model's entries onto another --


class TestTheRebuiltHullKeepsTheComponentOrder:
    """Every hull rebuild keeps the component order ``from_cache`` pinned. Cu-Mg is a system
    pymatgen's electronegativity order would reverse (Mg first)."""

    def test_the_fixture_is_one_pymatgen_would_reorder(self, stripped):
        """The control: on a system pymatgen leaves alone, the assertions below are vacuous."""
        bl, _ = stripped
        default = [str(el) for el in PhaseDiagram(bl.dft_ch.all_entries).elements]
        assert default != list(bl.components)
        assert [str(el) for el in bl.dft_ch.elements] == list(bl.components)

    def test_impute_phases_keeps_the_component_order(self, stripped):
        bl, _ = stripped
        assert impute_binary_phases(bl, fitting_data=None, write_cache=False)
        assert api.pd_components(bl.dft_ch) == list(bl.components)
        assert [str(el) for el in bl.dft_ch.elements] == list(bl.components)

    def test_the_phase_comparison_places_the_imputed_band_at_x_s(self, stripped):
        """The MP band of an imputed phase sits at ``x_s`` (fraction of ``components[1]``),
        not at ``1 - x_s``. The Cu-Mg fixture carries its MPDS record, so this is asserted
        rather than skipped."""
        bl, _ = stripped
        assert impute_binary_phases(bl, fitting_data=None, write_cache=False)
        _, (mp_phases, _, _) = mpds.get_low_temp_phase_data(bl.mpds_json, bl.dft_ch)
        entries = imputed_hull_entries(bl.dft_ch)
        assert entries
        for entry in entries:
            x_s = float(entry.data["x_b"])
            assert abs(x_s - 0.5) > 0.05, "a vertex at x=0.5 cannot tell the two frames apart"
            (lo, hi), _ = mp_phases[entry.name]
            assert lo == pytest.approx(x_s, abs=1e-6)
            assert hi == pytest.approx(x_s, abs=1e-6)

    def test_drop_imputed_phases_keeps_the_component_order(self, stripped):
        bl, _ = stripped
        assert impute_binary_phases(bl, fitting_data=None, write_cache=False)
        back = drop_imputed_phases(bl)
        assert [str(el) for el in back.dft_ch.elements] == list(bl.components)

    def test_augmented_hull_keeps_the_component_order(self, stripped):
        bl, _ = stripped
        probe = _imputed_computed_entry(bl, "Probe", 0.5, -60000.0)
        hull = imp_core.augmented_hull(bl.dft_ch, [probe])
        assert [str(el) for el in hull.elements] == list(bl.components)


class TestReplayingImputedEntriesOntoAnotherModel:
    """``apply_imputed_entries`` carries one model's imputed hull onto another model of the
    same system and leaves that model's liquid parameters alone."""

    @pytest.fixture
    def replayed(self, stripped):
        """``(source, target, imputed, fieldless)``: ``source`` imputed by ``impute_binary_phases``,
        ``target`` a copy with different liquid parameters given the same entry dicts."""
        source, _ = stripped
        assert any(source.get_params()), "an all-zero fit cannot be told from its half"
        target = copy.deepcopy(source)
        target.update_params([0.5 * v for v in target.get_params()])
        imputed = impute_binary_phases(source, fitting_data=None, write_cache=False)
        assert imputed, "nothing imputed; the replay would be vacuous"
        fieldless = imp_binary.apply_imputed_entries(target, [ip.entry_dict for ip in imputed])
        return source, target, imputed, fieldless

    def test_the_same_vertices_appear_on_the_target(self, replayed):
        source, target, imputed, _ = replayed
        assert {p.name for p in target.phases if p.imputed} == {ip.name for ip in imputed}
        by_name = {p.name: p for p in source.phases if p.imputed}
        for phase in target.phases:
            if not phase.imputed:
                continue
            assert target._phase_x(phase) == pytest.approx(source._phase_x(by_name[phase.name]))
            assert phase.enthalpy == pytest.approx(by_name[phase.name].enthalpy)

    def test_the_hull_carries_the_same_formation_energies(self, replayed):
        source, target, _, _ = replayed
        expected = {
            e.data["label"]: source.dft_ch.get_form_energy_per_atom(e)
            for e in imputed_hull_entries(source.dft_ch)
        }
        got = {
            e.data["label"]: target.dft_ch.get_form_energy_per_atom(e)
            for e in imputed_hull_entries(target.dft_ch)
        }
        assert set(got) == set(expected)
        for name, energy in expected.items():
            assert got[name] == pytest.approx(energy)

    def test_the_target_keeps_its_own_liquid_parameters(self, replayed):
        source, target, _, _ = replayed
        assert target.get_params() != source.get_params()

    def test_the_hull_keeps_the_component_order(self, replayed):
        _, target, _, _ = replayed
        assert [str(el) for el in target.dft_ch.elements] == list(target.components)

    def test_the_hull_interp_follows_the_new_phase_list(self, replayed):
        _, target, _, _ = replayed
        pts = _hull_points(target)
        assert np.array_equal(
            target.eqs["h_hull_interp"], np.interp(x_vals[1:-1], pts[:, 0], pts[:, 1])
        )

    def test_it_returns_the_fieldless_vertex_names(self, replayed):
        _, target, _, fieldless = replayed
        assert isinstance(fieldless, set)
        assert fieldless == set(hull_field_disagreement(target)["stable_without_field"])

    def test_unmask_clears_the_mask(self, stripped):
        bl, _ = stripped
        imputed = impute_binary_phases(copy.deepcopy(bl), fitting_data=None, write_cache=False)
        assert imputed
        bl.unsupported_comp_ranges = [[0.2, 0.5]]
        imp_binary.apply_imputed_entries(bl, [ip.entry_dict for ip in imputed])
        assert bl.unsupported_comp_ranges == []

    def test_unmask_false_leaves_the_mask_alone(self, stripped):
        bl, _ = stripped
        imputed = impute_binary_phases(copy.deepcopy(bl), fitting_data=None, write_cache=False)
        assert imputed
        bl.unsupported_comp_ranges = [[0.2, 0.5]]
        imp_binary.apply_imputed_entries(bl, [ip.entry_dict for ip in imputed], unmask=False)
        assert bl.unsupported_comp_ranges == [[0.2, 0.5]]

    def test_impute_phases_rebuilds_through_it(self):
        assert "screen_unrealizable_entries" in inspect.getsource(impute_binary_phases)


def _legend_line_colors(fig) -> dict[str, str]:
    """Trace name -> line color, first trace per name."""
    out = {}
    for trace in fig.data:
        color = getattr(getattr(trace, "line", None), "color", None)
        if trace.name and color and trace.name not in out:
            out[trace.name] = color
    return out


class TestADisplacedVertexKeepsItsPaletteSlot:
    """A DFT compound an imputed phase pushes off the hull still holds its palette slot, so
    every phase that survives draws in the color it has on the un-imputed hull."""

    @pytest.fixture
    def displaced(self, cu_mg_full):
        """``(before, after, gone, kept)``: ``cu_mg_full`` and a copy whose first interior
        compound ``gone`` an imputed probe displaces; ``kept`` names the interior compounds
        both hulls carry."""
        before = copy.deepcopy(cu_mg_full)
        before.update_phase_points()
        first = _interior_phases(before)[0]
        probe = _imputed_computed_entry(
            before, "Probe", float(before._phase_x(first)) - 0.03, 1.5 * first.enthalpy
        )
        after = copy.deepcopy(before)
        imp_binary.apply_imputed_entries(after, [probe])
        names_after = {p.name for p in after.phases}
        assert first.name not in names_after, "the probe displaced nothing"
        kept = [p.name for p in _interior_phases(before) if p.name in names_after]
        assert kept, "no interior compound survives the probe"
        return before, after, first.name, kept

    def test_the_plotter_palette_keeps_every_surviving_color(self, displaced):
        before, after, _, kept = displaced
        colors_before = BLPlotter(before)._phase_color_map()
        colors_after = BLPlotter(after)._phase_color_map()
        assert {n: colors_after[n] for n in kept} == {n: colors_before[n] for n in kept}

    def test_the_fit_liq_figure_keeps_every_surviving_color(self, displaced):
        before, after, _, kept = displaced
        drawn_before = _legend_line_colors(BLPlotter(before).get_plot("fit+liq"))
        drawn_after = _legend_line_colors(BLPlotter(after).get_plot("fit+liq"))
        shared = [n for n in kept if n in drawn_before and n in drawn_after]
        assert shared, "no surviving compound draws a field in both figures"
        assert {n: drawn_after[n] for n in shared} == {n: drawn_before[n] for n in shared}

    def test_the_displaced_compound_gets_no_trace(self, displaced):
        _, after, gone, _ = displaced
        assert gone not in _legend_line_colors(BLPlotter(after).get_plot("fit+liq"))


class TestOnlyMatchedVerticesAreProtected:
    """The displacement guard vetoes on behalf of hull vertices a low-T MPDS phase claims; a
    DFT vertex with no experimental counterpart does not block an imputed phase. The guard
    runs against stubbed hulls, and ``matched_hull_entry_ids`` is checked against the matcher
    it is built on.
    """

    @staticmethod
    def _cand(name, margin):
        return {
            "entry_dict": {"entry_id": f"imputed:{name}", "data": {"label": name}},
            "name": name,
            "margin_j": margin,
        }

    def _stub_hull(self, monkeypatch, stable_after):
        """``augmented_hull`` reports ``stable_after(labels)`` as the stable entry ids."""

        def fake(base, entry_dicts):
            labels = [e["data"]["label"] for e in entry_dicts]
            return SimpleNamespace(
                stable_entries=[SimpleNamespace(entry_id=eid) for eid in stable_after(labels)]
            )

        monkeypatch.setattr(imp_binary, "augmented_hull", fake)

    def test_a_candidate_displacing_a_matched_vertex_is_rejected(self, monkeypatch):
        # clamping (TestClampingOntoAMatchedHull) is what decides whether a matched blocker
        # is fatal; with no clamp available the guard must still reject.
        monkeypatch.setattr(imp_binary, "_clamp_onto_matched_hull", lambda *a, **k: None)
        monkeypatch.setattr(imp_binary, "matched_hull_entry_ids", lambda bl: {"mp-matched"})
        self._stub_hull(monkeypatch, lambda labels: ["imputed:A"])
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand("A", 10.0)]
        )
        assert accepted == []
        assert rejected[0][0] == "A"
        assert "matched vertex" in rejected[0][1] and "mp-matched" in rejected[0][1]

    def test_a_candidate_displacing_only_an_unmatched_vertex_is_accepted(self, monkeypatch):
        monkeypatch.setattr(imp_binary, "matched_hull_entry_ids", lambda bl: {"mp-matched"})
        self._stub_hull(monkeypatch, lambda labels: ["imputed:A", "mp-matched"])
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand("A", 10.0)]
        )
        assert [e["data"]["label"] for e in accepted] == ["A"]
        assert rejected == []

    def test_protect_dft_hull_false_still_skips_the_guard(self, monkeypatch):
        monkeypatch.setattr(imp_binary, "matched_hull_entry_ids", lambda bl: {"mp-matched"})
        self._stub_hull(monkeypatch, lambda labels: ["imputed:A"])
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand("A", 10.0)], protect_dft_hull=False
        )
        assert [e["data"]["label"] for e in accepted] == ["A"]
        assert rejected == []

    def test_a_candidate_that_is_not_itself_stable_is_still_rejected(self, monkeypatch):
        monkeypatch.setattr(imp_binary, "matched_hull_entry_ids", lambda bl: {"mp-matched"})
        self._stub_hull(monkeypatch, lambda labels: ["mp-matched"])
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand("A", 10.0)]
        )
        assert accepted == []
        assert rejected == [("A", "not stable on the hull once the others are in")]

    def test_the_helper_agrees_with_the_matcher_it_is_built_on(self, cu_mg_full):
        bl = cu_mg_full
        phases, unpaired_ht = bl.coverage_phase_inputs()
        assigned = solid_coverage.assign_dft_phases(
            phases,
            bl.components,
            bl._interior_dft_comps(),
            bl.dft_spacegroups(),
            unpaired_ht=unpaired_ht,
            rescuable=solid_coverage.rescuable_ss_indices(phases, bl.components),
        )
        claimed_keys = {round(float(c), 9) for m in assigned.values() for c in m.comps}
        expected = set()
        for entry in bl.dft_ch.stable_entries:
            x_b = float(api.entry_frac_along(bl.dft_ch, entry, bl.components)[0])
            if not 0.0 < round(x_b, 6) < 1.0 or round(x_b, 9) in claimed_keys:
                expected.add(str(entry.entry_id))
        assert imp_binary.matched_hull_entry_ids(bl) == expected

    def test_the_elemental_endpoints_are_always_matched(self, cu_mg_full):
        matched = imp_binary.matched_hull_entry_ids(cu_mg_full)
        for entry in cu_mg_full.dft_ch.stable_entries:
            if len(entry.composition.elements) == 1:
                assert str(entry.entry_id) in matched

    def test_the_matched_set_never_exceeds_the_stable_set(self, cu_mg_full):
        stable = {str(e.entry_id) for e in cu_mg_full.dft_ch.stable_entries}
        assert imp_binary.matched_hull_entry_ids(cu_mg_full) <= stable

    def test_a_system_with_no_mpds_phases_protects_everything(self, cu_mg_full, monkeypatch):
        bl = copy.deepcopy(cu_mg_full)
        monkeypatch.setattr(type(bl), "coverage_phase_inputs", lambda self, **kw: ([], frozenset()))
        assert imp_binary.matched_hull_entry_ids(bl) == {
            str(e.entry_id) for e in bl.dft_ch.stable_entries
        }


class TestTheTMatchedInversionGate:
    """The two-point common tangent is available per anchor type and is off by default."""

    @staticmethod
    def _inv(kind, x_inv=0.30, temp=1000.0, x_s=0.55):
        return {"type": kind, "comp": x_inv, "temp": temp, "phases": ["X"], "phase_comps": [x_s]}

    def test_the_default_gate_is_empty(self):
        assert imp_binary.T_MATCHED_ANCHOR_TYPES == ()

    def test_the_default_reproduces_the_shipped_construction(self, cu_mg_full):
        """With an empty gate the two branches must agree to floating-point equality."""
        for kind in ("per", "eut"):
            inv = self._inv(kind)
            x_inv, t_inv, x_s = inv["comp"], inv["temp"], 0.55
            g_inv = _liquid_gibbs(cu_mg_full, x_inv, t_inv)
            dx = 1e-3
            slope = (
                _liquid_gibbs(cu_mg_full, x_inv + dx, t_inv)
                - _liquid_gibbs(cu_mg_full, x_inv - dx, t_inv)
            ) / (2 * dx)
            assert invert_binary_phase_energy(cu_mg_full, x_s, inv) == pytest.approx(
                g_inv + slope * (x_s - x_inv)
            )

    def test_a_gated_type_takes_the_two_point_tangent(self, cu_mg_full):
        inv = self._inv("per")
        expected = imp_binary._tangent_through_far_neighbour(
            cu_mg_full, 0.55, inv["comp"], inv["temp"]
        )
        assert expected is not None, "Cu-Mg has no far neighbour; the fixture cannot test this"
        got = invert_binary_phase_energy(cu_mg_full, 0.55, inv, t_matched_types=("per",))
        assert got == pytest.approx(expected)
        # and it is NOT the shipped answer, which a dead gate would return
        assert got != pytest.approx(invert_binary_phase_energy(cu_mg_full, 0.55, inv))

    def test_the_gate_is_honoured_per_type(self, cu_mg_full):
        shipped = {
            k: invert_binary_phase_energy(cu_mg_full, 0.55, self._inv(k)) for k in ("per", "eut")
        }
        got = {
            k: invert_binary_phase_energy(cu_mg_full, 0.55, self._inv(k), t_matched_types=("per",))
            for k in ("per", "eut")
        }
        assert got["eut"] == pytest.approx(shipped["eut"])
        assert got["per"] != pytest.approx(shipped["per"])

    def test_a_congruent_anchor_is_the_closed_form_either_way(self, cu_mg_full):
        inv = self._inv("cmp", x_inv=0.4, x_s=0.4)
        closed = _liquid_gibbs(cu_mg_full, 0.4, inv["temp"])
        assert invert_binary_phase_energy(cu_mg_full, 0.4, inv) == pytest.approx(closed)
        assert invert_binary_phase_energy(
            cu_mg_full, 0.4, inv, t_matched_types=("cmp", "per", "eut")
        ) == pytest.approx(closed)

    def test_no_far_neighbour_falls_back_to_the_shipped_construction(self, cu_mg_full, monkeypatch):
        monkeypatch.setattr(imp_binary, "_tangent_through_far_neighbour", lambda *a, **k: None)
        inv = self._inv("per")
        assert invert_binary_phase_energy(
            cu_mg_full, 0.55, inv, t_matched_types=("per",)
        ) == pytest.approx(invert_binary_phase_energy(cu_mg_full, 0.55, inv))


class TestFarNeighbourPolymorph:
    """The two-point tangent reads a pure-element neighbour at its stable polymorph at T_inv.

    Fe-Ti on a synthetic hull holding TiFe alone. The Fe and Ti polymorph ladders are the
    bundled reference data, so ``bl.phases`` lists every polymorph at x=0 and x=1 in the order
    ``from_cache`` builds them. No cache and no network.
    """

    @pytest.fixture(scope="class")
    def fe_ti(self):
        components = ["Fe", "Ti"]
        ch = PhaseDiagram(
            [
                ComputedEntry(Composition("Fe"), 0.0),
                ComputedEntry(Composition("Ti"), 0.0),
                ComputedEntry(Composition("TiFe"), -0.84),
            ],
            elements=[Element(c) for c in components],
        )
        component_data = UNARY.component_data(components)
        bl = BinaryLiquid(
            "Fe-Ti",
            components,
            component_data=component_data,
            dft_ch=ch,
            phases=build_phases_from_chull(ch, components, component_data),
            param_format="comb-exp",
            params=[-40000.0, -5.0, 0.0, 0.0],
            temp_range=[300, 3000],
        )
        bl._rebuild_thermodynamic_expressions()
        return bl

    @staticmethod
    def _rows_at(bl, x):
        return [p for p in bl.phases if p.composition is not None and bl._phase_x(p) == x]

    def test_the_fixture_lists_a_hot_polymorph_first_at_x0(self, fe_ti):
        """The first row at x=0 is not the ground state, so a first-row pick is exercised."""
        fe, ti = self._rows_at(fe_ti, 0.0), self._rows_at(fe_ti, 1.0)
        assert len(fe) > 1 and len(ti) > 1
        assert fe[0].enthalpy > min(p.enthalpy for p in fe)

    @pytest.mark.parametrize(
        "x_inv, x_s, temp",
        [
            (0.40, 0.30, 1000.0),  # far side Fe, alpha-Fe stable
            (0.40, 0.30, 1300.0),  # far side Fe, gamma-Fe stable
            (0.60, 0.70, 1400.0),  # far side Ti, beta-Ti stable
        ],
    )
    def test_an_element_neighbour_is_read_at_its_stable_polymorph(self, fe_ti, x_inv, x_s, temp):
        x_far = 0.0 if x_inv > x_s else 1.0
        g_far = min(p.gibbs(temp) for p in self._rows_at(fe_ti, x_far))
        g_inv = _liquid_gibbs(fe_ti, x_inv, temp)
        expected = g_inv + (g_far - g_inv) / (x_far - x_inv) * (x_s - x_inv)
        inv = {"type": "per", "comp": x_inv, "temp": temp, "phases": ["X"], "phase_comps": [x_s]}
        got = invert_binary_phase_energy(fe_ti, x_s, inv, t_matched_types=("per",))
        assert got == pytest.approx(expected)

    def test_a_compound_neighbour_is_read_at_its_enthalpy(self, fe_ti):
        x_inv, x_s, temp = 0.20, 0.30, 1300.0
        tife = next(p for p in fe_ti.phases if p.name == "TiFe")
        g_inv = _liquid_gibbs(fe_ti, x_inv, temp)
        expected = g_inv + (tife.enthalpy - g_inv) / (0.5 - x_inv) * (x_s - x_inv)
        got = imp_binary._tangent_through_far_neighbour(fe_ti, x_s, x_inv, temp)
        assert got == pytest.approx(expected)


class TestClampingOntoAMatchedHull:
    """A matched blocker raises the candidate's energy instead of rejecting it outright.

    The bisection is driven against a stubbed hull whose stability rule is a threshold on the
    candidate's own energy, so the value the search lands on is known in advance.
    """

    BLOCKER = "mp-matched"
    HULL_H = -10_000.0

    def _cand(self, h_form_j, name="A"):
        entry = ComputedEntry(
            Composition({"Cu": 0.5, "Mg": 0.5}),
            0.0,
            entry_id=f"imputed:{name}",
            data={"label": name, "x_b": 0.5, "h_form_j_per_mol": float(h_form_j)},
        )
        return {"entry_dict": entry.as_dict(), "name": name, "margin_j": 10.0}

    def _stub(self, monkeypatch, blocker_survives_above):
        """Stub the hull: the blocker survives only above ``blocker_survives_above``, and its
        clearance below its neighbours' tie line grows one-for-one with the candidate's energy.
        """
        monkeypatch.setattr(imp_binary, "matched_hull_entry_ids", lambda bl: {self.BLOCKER})
        monkeypatch.setattr(imp_binary, "_phase_list_enthalpy_at", lambda bl, x: self.HULL_H)
        monkeypatch.setattr(imp_core, "hull_ref_energy_per_atom", lambda hull, fractions: 0.0)

        def fake(base, entry_dicts):
            h = float(entry_dicts[-1]["data"]["h_form_j_per_mol"])
            ids = [str(e["entry_id"]) for e in entry_dicts]
            if h > blocker_survives_above:
                ids.append(self.BLOCKER)
            return SimpleNamespace(
                stable_entries=[SimpleNamespace(entry_id=i) for i in ids], candidate_h=h
            )

        monkeypatch.setattr(imp_binary, "augmented_hull", fake)
        monkeypatch.setattr(
            imp_binary,
            "protected_clearance",
            lambda hull, protected: hull.candidate_h - blocker_survives_above,
        )

    def test_a_blocked_candidate_is_accepted_at_a_clamped_energy(self, monkeypatch):
        self._stub(monkeypatch, blocker_survives_above=-50_000.0)
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        assert rejected == []
        assert len(accepted) == 1
        clamped = float(accepted[0]["data"]["h_form_j_per_mol"])
        assert clamped == pytest.approx(
            -50_000.0 + imp_binary.PROTECTED_VERTEX_MARGIN_J, abs=imp_binary.CLAMP_TOLERANCE_J
        )
        # deepest value that works, not merely any value that works
        assert clamped < self.HULL_H - imp_binary.HULL_GUARD_MARGIN_J

    def test_an_unreachable_margin_clamps_at_the_shallowest_the_guard_allows(self, monkeypatch):
        """No energy gives the blocker its margin, so it is given every joule there is."""
        self._stub(monkeypatch, blocker_survives_above=-10_050.0)
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        assert rejected == []
        clamped = float(accepted[0]["data"]["h_form_j_per_mol"])
        assert clamped == pytest.approx(self.HULL_H - imp_binary.HULL_GUARD_MARGIN_J)

    def test_the_clamp_is_recorded_on_the_provenance(self, monkeypatch):
        self._stub(monkeypatch, blocker_survives_above=-50_000.0)
        accepted, _ = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        clamp = accepted[0]["data"][imp_binary.PROVENANCE_KEY][imp_binary.CLAMP_KEY]
        assert clamp["inverted_h_form_j_per_mol"] == -60_000.0
        assert clamp["clamped_h_form_j_per_mol"] == float(accepted[0]["data"]["h_form_j_per_mol"])
        assert clamp["blocked_by"] == [self.BLOCKER]

    def test_the_entry_energy_follows_the_clamped_enthalpy(self, monkeypatch):
        self._stub(monkeypatch, blocker_survives_above=-50_000.0)
        accepted, _ = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        h = float(accepted[0]["data"]["h_form_j_per_mol"])
        assert accepted[0]["energy"] == pytest.approx(h / EV_ATOM_TO_J_MOL)

    def test_a_blocker_no_energy_can_clear_is_still_rejected(self, monkeypatch):
        # the blocker never survives at any energy under the hull
        self._stub(monkeypatch, blocker_survives_above=self.HULL_H)
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        assert accepted == []
        assert rejected[0][0] == "A"
        assert "no clamped energy clears the hull" in rejected[0][1]
        assert self.BLOCKER in rejected[0][1]

    def test_an_unblocked_candidate_keeps_its_inverted_energy(self, monkeypatch):
        self._stub(monkeypatch, blocker_survives_above=-70_000.0)
        accepted, rejected = imp_binary.accept_against_growing_hull(
            SimpleNamespace(dft_ch=None), [self._cand(-60_000.0)]
        )
        assert rejected == []
        assert float(accepted[0]["data"]["h_form_j_per_mol"]) == -60_000.0
        assert imp_binary.CLAMP_KEY not in accepted[0]["data"].get(imp_binary.PROVENANCE_KEY, {})

    def test_the_record_carries_the_clamped_enthalpy_not_the_inverted_one(
        self, imputation_recovery
    ):
        _bl, _orig, imputed = imputation_recovery
        for record in imputed:
            assert record.enthalpy == pytest.approx(
                float(record.entry_dict["data"]["h_form_j_per_mol"])
            )

    def test_a_clamped_entry_is_not_read_as_stale(self, stripped):
        bl, _ = stripped
        imputed = impute_binary_phases(bl, fitting_data=None, write_cache=False)
        assert imputed, "nothing imputed; the fixture cannot exercise staleness"
        entry = copy.deepcopy(imputed[0].entry_dict)
        entry["data"][imp_binary.PROVENANCE_KEY][imp_binary.CLAMP_KEY] = {
            "inverted_h_form_j_per_mol": -99_999.0,
            "clamped_h_form_j_per_mol": entry["data"]["h_form_j_per_mol"],
            "blocked_by": ["mp-whatever"],
        }
        assert imputed_entry_staleness(bl, entry) is None


class TestTheRealizabilityScreen:
    """``screen_unrealizable_entries`` keeps only what the finished diagram draws.

    The loop runs against a scripted ``apply_imputed_entries``, so these pin which entries
    survive which passes; ``TestImputationRecovery`` carries the integration side.
    """

    @staticmethod
    def _entry(label):
        return {"entry_id": f"imputed:{label}", "data": {"imputed": True, "label": label}}

    @staticmethod
    def _bl():
        return SimpleNamespace(dft_ch=None, unsupported_comp_ranges=[], phases=[])

    def _script(self, monkeypatch, drawn_for, fieldless_for=None, protected=()):
        """Make ``apply_imputed_entries`` report ``drawn_for(labels)`` as the phase list."""

        def fake(bl, entry_dicts, *, unmask=True):
            labels = [e["data"]["label"] for e in entry_dicts]
            bl.phases = [
                SimpleNamespace(name=name, points=[[0.5, 900.0]]) for name in drawn_for(labels)
            ]
            return set((fieldless_for or (lambda _: set()))(labels))

        monkeypatch.setattr(imp_binary, "apply_imputed_entries", fake)
        monkeypatch.setattr(imp_binary, "_protected_vertex_names", lambda bl: set(protected))

    def test_an_entry_the_others_push_off_the_hull_is_dropped(self, monkeypatch):
        self._script(
            monkeypatch, lambda labels: [n for n in labels if n != "A" or "B" not in labels]
        )
        kept, dropped, _ = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["B"]
        assert dropped == [("A", "off the hull once every accepted phase is on it")]

    def test_a_vertex_with_no_field_is_dropped(self, monkeypatch):
        self._script(monkeypatch, lambda labels: labels, lambda _: {"A"})
        kept, dropped, _ = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["B"]
        assert dropped == [("A", "occupies no field on the T-x diagram")]

    def test_the_weakest_offender_goes_first_and_the_rest_get_another_pass(self, monkeypatch):
        # both are fieldless together; dropping the weaker one frees the stronger
        self._script(
            monkeypatch,
            lambda labels: labels,
            lambda labels: {"A", "B"} if len(labels) == 2 else set(),
        )
        kept, dropped, _ = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["A"]
        assert [name for name, _ in dropped] == ["B"]

    def test_nothing_is_dropped_when_everything_draws(self, monkeypatch):
        self._script(monkeypatch, lambda labels: labels)
        kept, dropped, fieldless = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["A", "B"]
        assert dropped == []
        assert fieldless == set()

    def test_a_fieldless_vertex_that_is_not_ours_is_reported_not_dropped(self, monkeypatch):
        self._script(monkeypatch, lambda labels: labels, lambda _: {"CuMg2"})
        kept, dropped, fieldless = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A")]
        )
        assert [e["data"]["label"] for e in kept] == ["A"]
        assert dropped == []

    def test_a_set_that_erases_a_protected_vertex_is_pared_back_until_it_draws(self, monkeypatch):
        """No single entry can be named for the erasure, so the weakest goes until it draws."""
        self._script(
            monkeypatch,
            lambda labels: labels,
            lambda labels: {"CuMg2"} if len(labels) == 2 else set(),
            protected=("CuMg2",),
        )
        kept, dropped, fieldless = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["A"]
        assert dropped == [("B", "erases the T-x field of the protected vertex CuMg2")]
        assert fieldless == set()

    def test_a_protected_vertex_already_fieldless_is_not_charged_to_the_imputation(
        self, monkeypatch
    ):
        """It draws nothing either way, so the screen has nothing to restore by dropping."""
        self._script(
            monkeypatch,
            lambda labels: labels,
            lambda _: {"CuMg2"},
            protected=("CuMg2",),
        )
        kept, dropped, _ = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["A", "B"]
        assert dropped == []

    def test_an_erased_vertex_that_is_unmatched_is_reported_not_defended(self, monkeypatch):
        """Protection covers matched vertices only; the screen must not widen it back."""
        self._script(
            monkeypatch,
            lambda labels: labels,
            lambda labels: {"CuMg2"} if len(labels) == 2 else set(),
        )
        kept, dropped, fieldless = imp_binary.screen_unrealizable_entries(
            self._bl(), [self._entry("A"), self._entry("B")]
        )
        assert [e["data"]["label"] for e in kept] == ["A", "B"]
        assert dropped == []
        assert fieldless == {"CuMg2"}
        assert fieldless == {"CuMg2"}

    def test_an_empty_set_settles_immediately(self, monkeypatch):
        self._script(monkeypatch, lambda labels: labels)
        kept, dropped, _ = imp_binary.screen_unrealizable_entries(self._bl(), [])
        assert kept == [] and dropped == []

    def test_every_imputed_phase_is_drawn(self, imputation_recovery):
        bl, _orig, imputed = imputation_recovery
        assert imputed, "nothing imputed; the fixture cannot exercise the screen"
        drawn = {p.name: p for p in bl.phases}
        for record in imputed:
            assert record.name in drawn, f"{record.name} was returned but is not on the hull"
            assert drawn[record.name].points, f"{record.name} was returned but draws no field"

    def test_the_cache_is_written_after_the_screen(self):
        source = inspect.getsource(impute_binary_phases)
        assert source.index("screen_unrealizable_entries") < source.index("cache_imputed_entries")


def chosen_params(bl, fit):
    return [fit[name] for name in bl.xs_mix.format.param_names]


class TestNearestArmSelection:
    """``nearest_arm_choices`` runs both arms and gives each phase the arm whose DRAWN field
    top is closest to its anchor.

    The two probe passes are scripted; ``TestImputationRecovery`` carries the integration side.
    """

    ANCHOR = 1000.0

    @staticmethod
    def _record(name, comp, anchor_type="per", anchor=1000.0):
        return BinaryImputationResult(
            name=name,
            comp=comp,
            enthalpy=-1000.0,
            invariant_type=anchor_type,
            invariant_comp=comp,
            invariant_temp=anchor,
            entry_dict={"entry_id": f"imputed:{name}", "data": {"label": name}},
        )

    def _script(self, monkeypatch, shipped, tmatched, tops):
        """Script both probe passes: ``tops`` maps ``(arm, name)`` to a drawn field top."""

        def fake_impute(bl, *, t_matched_types=(), **kwargs):
            bl.arm = "tmatched" if t_matched_types else "shipped"
            return tmatched if bl.arm == "tmatched" else shipped

        monkeypatch.setattr(imp_binary, "impute_binary_phases", fake_impute)
        monkeypatch.setattr(imp_binary, "_field_t_max", lambda bl, name: tops.get((bl.arm, name)))

    @staticmethod
    def _bl():
        return SimpleNamespace(arm=None)

    def test_the_arm_drawing_nearer_the_anchor_wins(self, monkeypatch):
        record = self._record("X", 0.5)
        self._script(
            monkeypatch,
            [record],
            [record],
            {("shipped", "X"): 1400.0, ("tmatched", "X"): 1010.0},
        )
        assert imp_binary.nearest_arm_choices(self._bl()) == {0.5: True}

    def test_an_undershoot_is_as_wrong_as_an_overshoot(self, monkeypatch):
        """Cooler is better only down to the anchor; the rule reads distance, not direction."""
        record = self._record("X", 0.5)
        self._script(
            monkeypatch,
            [record],
            [record],
            {("shipped", "X"): 1050.0, ("tmatched", "X"): 500.0},
        )
        assert imp_binary.nearest_arm_choices(self._bl()) == {0.5: False}

    def test_a_phase_only_one_arm_realizes_takes_that_arm(self, monkeypatch):
        record = self._record("X", 0.5)
        self._script(monkeypatch, [], [record], {("tmatched", "X"): 1200.0})
        assert imp_binary.nearest_arm_choices(self._bl()) == {0.5: True}

        self._script(monkeypatch, [record], [], {("shipped", "X"): 1200.0})
        assert imp_binary.nearest_arm_choices(self._bl()) == {0.5: False}

    def test_a_phase_with_no_drawn_field_falls_back_to_the_realizing_arm(self, monkeypatch):
        """A phase with no field has no temperature, which is not the same as a cold one."""
        record = self._record("X", 0.5)
        self._script(
            monkeypatch, [record], [record], {("shipped", "X"): 1400.0, ("tmatched", "X"): None}
        )
        assert imp_binary.nearest_arm_choices(self._bl()) == {0.5: False}

    @pytest.mark.parametrize("anchor_type", ["cmp", "eut"])
    def test_an_unselectable_anchor_is_never_switched(self, monkeypatch, anchor_type):
        """``cmp`` reduces to the closed form; ``eut`` is outside the characterised rule."""
        record = self._record("X", 0.5, anchor_type=anchor_type)
        self._script(
            monkeypatch,
            [record],
            [record],
            {("shipped", "X"): 1400.0, ("tmatched", "X"): 1000.0},
        )
        assert imp_binary.nearest_arm_choices(self._bl()) == {}

    def test_the_probes_do_not_mutate_the_system(self, monkeypatch):
        record = self._record("X", 0.5)
        self._script(
            monkeypatch,
            [record],
            [record],
            {("shipped", "X"): 1400.0, ("tmatched", "X"): 1010.0},
        )
        bl = self._bl()
        imp_binary.nearest_arm_choices(bl)
        assert bl.arm is None, "the arm probes must run on copies"


class TestInversionSelectionPlumbing:
    def test_an_unknown_selection_is_rejected(self, cu_mg_full):
        with pytest.raises(ValueError, match="inversion_selection"):
            impute_binary_phases(cu_mg_full, write_cache=False, inversion_selection="closest")

    def test_the_default_asks_for_no_arm_selection(self, stripped, monkeypatch):
        """The shipped default must not pay for the two extra passes."""
        called = []
        monkeypatch.setattr(
            imp_binary, "nearest_arm_choices", lambda bl, **kw: called.append(bl) or {}
        )
        bl, _ = stripped
        impute_binary_phases(bl, fitting_data=None, write_cache=False)
        assert called == []

    def test_every_imputed_phase_records_the_arm_it_took(self, stripped):
        bl, _ = stripped
        imputed = impute_binary_phases(
            bl, fitting_data=None, write_cache=False, inversion_selection="nearest"
        )
        assert imputed
        for record in imputed:
            provenance = record.entry_dict["data"][imp_binary.PROVENANCE_KEY]
            assert provenance[imp_binary.ARM_KEY] in ("shipped", "tmatched")

    def test_nearest_is_deterministic(self, cu_mg_full):
        runs = []
        for _ in range(2):
            bl = copy.deepcopy(cu_mg_full)
            strip_interior(bl)
            runs.append(
                {
                    r.name: float(r.entry_dict["data"]["h_form_j_per_mol"])
                    for r in impute_binary_phases(
                        bl, fitting_data=None, write_cache=False, inversion_selection="nearest"
                    )
                }
            )
        assert runs[0] == runs[1]


# -- Re-anchoring and the MPDS-free hull builder --


def _synthetic_entry(components, x_b, h_form_j, label):
    """A tagged imputed entry dict whose stored absolute energy is deliberately wrong."""
    a, b = components
    return ComputedEntry(
        composition=Composition({a: 1.0 - x_b, b: x_b}),
        energy=0.0,
        entry_id=f"imputed:{label}",
        data={
            "imputed": True,
            "label": label,
            "x_b": float(x_b),
            "h_form_j_per_mol": float(h_form_j),
        },
    ).as_dict()


def _shifted_references(hull, delta_ev_per_atom):
    """A copy of ``hull`` with every entry moved by a constant eV/atom.

    Formation energies are unchanged; the ABSOLUTE elemental reference energies are not,
    which is the only thing a stored imputed energy depends on.
    """
    entries = [
        ComputedEntry(
            composition=e.composition,
            energy=e.energy + delta_ev_per_atom * e.composition.num_atoms,
            entry_id=e.entry_id,
            data=dict(e.data or {}),
        )
        for e in hull.all_entries
    ]
    return PhaseDiagram(entries, elements=list(hull.elements))


def _stable_form_energies(hull):
    return {str(e.entry_id): hull.get_form_energy_per_atom(e) for e in hull.stable_entries}


class TestReanchoringImputedEntries:
    """A stored imputed energy is anchored to one hull's elemental references;
    ``reanchor_imputed_entries`` rebuilds it against whatever hull it is replayed onto."""

    SHIFT_EV = 0.37

    def test_a_zero_shift_returns_the_stored_energies_bit_for_bit(self, imputation_recovery):
        bl, _, imputed = imputation_recovery
        dicts = [ip.entry_dict for ip in imputed]
        assert dicts
        rebuilt = imp_binary.reanchor_imputed_entries(bl.dft_ch, dicts)
        for entry, source in zip(rebuilt, dicts):
            assert entry.energy == ComputedEntry.from_dict(source).energy

    def test_a_shifted_reference_still_recovers_the_formation_enthalpy(self, imputation_recovery):
        bl, _, imputed = imputation_recovery
        shifted = _shifted_references(bl.dft_ch, self.SHIFT_EV)
        dicts = [ip.entry_dict for ip in imputed]
        rebuilt = imp_binary.reanchor_imputed_entries(shifted, dicts)
        assert rebuilt
        for entry, source in zip(rebuilt, dicts):
            expected = float(source["data"]["h_form_j_per_mol"]) / EV_ATOM_TO_J_MOL
            assert shifted.get_form_energy_per_atom(entry) == pytest.approx(expected, abs=1e-9)

    def test_the_stored_energy_alone_would_have_been_wrong(self, imputation_recovery):
        """Positive control on the shift: the number this function exists to fix does move."""
        bl, _, imputed = imputation_recovery
        shifted = _shifted_references(bl.dft_ch, self.SHIFT_EV)
        for record in imputed:
            stale = ComputedEntry.from_dict(record.entry_dict)
            expected = float(record.entry_dict["data"]["h_form_j_per_mol"]) / EV_ATOM_TO_J_MOL
            assert shifted.get_form_energy_per_atom(stale) != pytest.approx(expected, abs=1e-6)

    def test_the_tags_and_ids_survive(self, imputation_recovery):
        bl, _, imputed = imputation_recovery
        dicts = [ip.entry_dict for ip in imputed]
        rebuilt = imp_binary.reanchor_imputed_entries(bl.dft_ch, dicts)
        for entry, source in zip(rebuilt, dicts):
            assert is_imputed_entry_dict(entry.as_dict())
            assert entry.entry_id == source["entry_id"]
            assert entry.data["label"] == source["data"]["label"]

    def test_an_entry_with_no_formation_enthalpy_raises(self, imputation_recovery):
        bl, _, imputed = imputation_recovery
        bad = copy.deepcopy(imputed[0].entry_dict)
        bad["data"].pop("h_form_j_per_mol")
        with pytest.raises(ValueError, match="h_form_j_per_mol"):
            imp_binary.reanchor_imputed_entries(bl.dft_ch, [bad])

    def test_an_empty_list_is_not_an_error(self, cu_mg_full):
        assert imp_binary.reanchor_imputed_entries(cu_mg_full.dft_ch, []) == []


class TestApplyRoutesThroughTheReanchor:
    def test_it_calls_the_public_primitive(self, stripped, monkeypatch):
        bl, _ = stripped
        imputed = impute_binary_phases(copy.deepcopy(bl), fitting_data=None, write_cache=False)
        assert imputed
        seen = []
        real = imp_core.reanchor_imputed_entries

        def spy(hull, entry_dicts):
            seen.append(len(entry_dicts))
            return real(hull, entry_dicts)

        monkeypatch.setattr(imp_core, "reanchor_imputed_entries", spy)
        imp_binary.apply_imputed_entries(bl, [ip.entry_dict for ip in imputed])
        assert seen == [len(imputed)]

    def test_the_matching_reference_path_is_bit_identical(self, stripped):
        bl, _ = stripped
        imputed = impute_binary_phases(copy.deepcopy(bl), fitting_data=None, write_cache=False)
        dicts = [ip.entry_dict for ip in imputed]
        assert dicts
        imp_binary.apply_imputed_entries(bl, dicts)
        got = {e.data["label"]: e.energy for e in imputed_hull_entries(bl.dft_ch)}
        for source in dicts:
            assert got[source["data"]["label"]] == ComputedEntry.from_dict(source).energy


class TestImputedHullBuilder:
    """``imputed_hull`` completes a hull from the DFT cache alone -- no MPDS record and no
    ``BinaryLiquid`` -- and must agree with the binary path entry for entry."""

    PROBE_X = 0.35
    PROBE_H_J = -80000.0

    def _entries(self, components):
        return [_synthetic_entry(components, self.PROBE_X, self.PROBE_H_J, "Probe")]

    @pytest.mark.parametrize("system", ["Cu-Mg", "Al-Cu"])
    def test_it_agrees_with_apply_imputed_entries(self, system):
        bl = BinaryLiquid.from_cache(system, param_format="comb-exp")
        if bl.init_error:
            pytest.skip(f"{system}: init error on load")
        entries = self._entries(bl.components)
        hull = imp_binary.imputed_hull(bl.components, entries)
        imp_binary.apply_imputed_entries(bl, entries)
        expected = _stable_form_energies(bl.dft_ch)
        got = _stable_form_energies(hull)
        assert set(got) == set(expected)
        for entry_id, energy in expected.items():
            assert got[entry_id] == pytest.approx(energy)

    def test_the_probe_reaches_the_hull(self):
        hull = imp_binary.imputed_hull(["Cu", "Mg"], self._entries(["Cu", "Mg"]))
        assert "imputed:Probe" in {str(e.entry_id) for e in hull.stable_entries}

    def test_the_stored_energy_is_replaced_by_the_re_anchored_one(self):
        """The probe goes in at ``energy=0``; the hull must read back its enthalpy."""
        hull = imp_binary.imputed_hull(["Cu", "Mg"], self._entries(["Cu", "Mg"]))
        probe = next(e for e in hull.stable_entries if str(e.entry_id) == "imputed:Probe")
        assert hull.get_form_energy_per_atom(probe) * EV_ATOM_TO_J_MOL == pytest.approx(
            self.PROBE_H_J, abs=1.0
        )

    def test_no_mpds_record_is_read(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("imputed_hull must not need an MPDS record")

        monkeypatch.setattr(mpds, "load_mpds_data", refuse)
        hull = imp_binary.imputed_hull(["Cu", "Mg"], self._entries(["Cu", "Mg"]))
        assert "imputed:Probe" in {str(e.entry_id) for e in hull.stable_entries}

    def test_the_element_order_is_pinned_not_re_sorted(self):
        """pymatgen re-sorts by electronegativity when ``elements=`` is omitted."""
        hull = imp_binary.imputed_hull(["Cu", "Mg"], self._entries(["Cu", "Mg"]))
        assert [str(el) for el in hull.elements] == ["Cu", "Mg"]

    def test_an_empty_entry_list_gives_the_dft_hull(self):
        dft, _ = api.get_dft_convexhull(["Cu", "Mg"], "GGA")
        hull = imp_binary.imputed_hull(["Cu", "Mg"], [])
        assert _stable_form_energies(hull) == _stable_form_energies(dft)

    def test_a_hyphenated_spec_is_accepted(self):
        hull = imp_binary.imputed_hull("Cu-Mg", self._entries(["Cu", "Mg"]))
        assert "imputed:Probe" in {str(e.entry_id) for e in hull.stable_entries}


class TestDftCoversReadsTheAssignment:
    """``_dft_covers`` reads :func:`gliquid.solid_coverage.assign_dft_phases`' matched compounds."""

    @staticmethod
    def _field(name, lo, hi, comp):
        return {"name": name, "comp": comp, "cbounds": [[lo, 1000.0], [hi, 1000.0]]}

    def test_an_assigned_compound_is_covered(self):
        na_pb = self._field("Na1.2Pb2.8", 0.6554, 0.7265, 0.7002)
        key = mpds.compound_key("Na1.2Pb2.8", ["Na", "Pb"])
        assert imp_binary._dft_covers(na_pb, ["Na", "Pb"], [0.75], {key}) is True
        assert imp_binary._dft_covers(na_pb, ["Na", "Pb"], [0.75], set()) is False

    def test_a_polymorph_is_covered_through_its_compound(self):
        ht = {"name": "NaPb3 ht", "comp": 0.75}
        key = mpds.compound_key("NaPb3 rt", ["Na", "Pb"])
        assert imp_binary._dft_covers(ht, ["Na", "Pb"], [], {key}) is True

    def test_an_unassigned_integer_formula_is_covered_at_its_stoichiometry(self):
        nap = {"name": "NaPb ht", "comp": 0.48}
        assert imp_binary._dft_covers(nap, ["Na", "Pb"], [0.50], set()) is True
        assert imp_binary._dft_covers(nap, ["Na", "Pb"], [0.52], set()) is False

    def test_an_uncovered_solution_field_is_still_not_an_imputation_target(self):
        cr_ir = self._field("Cr0.5Ir0.5", 0.3208, 0.6996, 0.6996)
        assert imp_binary._is_solution_field(cr_ir, ["Cr", "Ir"]) is True
