"""Tests for the phase-ablation primitives (gliquid.imputation.ablation): removing interior DFT
phases from a fitted system and putting them back.

Uses Cu-Mg (eutectic + congruent melter), which the package cache carries: the session fit
``cu_mg_full`` from tests/conftest.py.
"""

import copy

import numpy as np
import pytest

from gliquid.binary import BinaryLiquid
from gliquid.imputation.ablation import (
    ENDPOINT_COMPS,
    ablate_phases,
    list_ablatable_phases,
    restore_phases,
)
from gliquid.solution import x_vals

# -- Shared fixtures --


@pytest.fixture(scope="module")
def cu_mg_bl(cu_mg_full) -> BinaryLiquid:
    """``cu_mg_full``; skips when it has no digitized liquidus or no ablatable phase."""
    if not cu_mg_full.digitized_liq:
        pytest.skip("Cu-Mg: no digitized liquidus")
    if not list_ablatable_phases(cu_mg_full):
        pytest.skip("Cu-Mg: no ablatable interior phases")
    return cu_mg_full


@pytest.fixture(scope="session")
def cu_mg_target() -> list:
    """The Cu-Mg ablation target the ablation tests share."""
    return ["Mg2Cu"]


# -- Ablation primitive --


class TestAblatePrimitive:
    def test_list_ablatable_excludes_endpoints_and_solutions(self, cu_mg_bl):
        bl = cu_mg_bl
        for phase in list_ablatable_phases(bl):
            assert round(float(bl._phase_x(phase)), 6) not in ENDPOINT_COMPS
            assert phase.composition is not None
            assert not phase.is_solution

    def test_ablate_rebuilds_hull_interp(self, cu_mg_bl, cu_mg_target):
        base_interp = np.asarray(cu_mg_bl.eqs["h_hull_interp"])
        abl = ablate_phases(cu_mg_bl, cu_mg_target)
        new_interp = np.asarray(abl.eqs["h_hull_interp"])
        assert len(new_interp) == len(x_vals) - 2
        assert not np.allclose(new_interp, base_interp), "Hull interp should change after ablation"

    def test_ablate_does_not_mutate_baseline(self, cu_mg_bl, cu_mg_target):
        n_phases_before = len(cu_mg_bl.phases)
        params_before = list(cu_mg_bl.get_params())
        _ = ablate_phases(cu_mg_bl, cu_mg_target)
        assert len(cu_mg_bl.phases) == n_phases_before
        assert cu_mg_bl.get_params() == params_before

    def test_ablate_removes_target_phase(self, cu_mg_bl, cu_mg_target):
        abl = ablate_phases(cu_mg_bl, cu_mg_target)
        remaining = [p.name for p in abl.phases]
        for name in cu_mg_target:
            assert name not in remaining

    def test_ablate_resets_refit_caches(self, cu_mg_bl, cu_mg_target):
        abl = ablate_phases(cu_mg_bl, cu_mg_target)
        assert abl.invariants is None
        assert abl.low_t_exp_phases is None
        assert abl.unsupported_comp_ranges == []
        assert abl.coverage_report is None
        assert abl.constraints is None
        assert abl.guess_symbols is None
        assert abl.init_error is False
        assert abl.skip_reason is None

    def test_ablate_rejects_endpoint_target(self, cu_mg_bl):
        with pytest.raises(ValueError):
            ablate_phases(cu_mg_bl, [0.0])

    def test_ablate_rejects_unknown_phase(self, cu_mg_bl):
        with pytest.raises(ValueError):
            ablate_phases(cu_mg_bl, ["NotARealPhase"])

    def test_ablate_accepts_a_phase_object(self, cu_mg_bl):
        victim = list_ablatable_phases(cu_mg_bl)[0]
        abl = ablate_phases(cu_mg_bl, victim)
        assert victim.name not in [p.name for p in abl.phases]


class TestAblateRestoreRoundTrip:
    """Removal and insertion are exact inverses — the template this module imputes against."""

    def test_round_trip_restores_phase_list_and_hull(self, cu_mg_bl):
        source = copy.deepcopy(cu_mg_bl)
        names_before = [p.name for p in source.phases]
        hull_before = np.asarray(source.eqs["h_hull_interp"]).copy()
        victim = list_ablatable_phases(source)[0]

        abl = ablate_phases(source, victim.name)
        assert victim.name not in [p.name for p in abl.phases]
        assert not np.allclose(np.asarray(abl.eqs["h_hull_interp"]), hull_before)

        restored = restore_phases(abl, victim)
        assert [p.name for p in restored.phases] == names_before
        assert np.array_equal(np.asarray(restored.eqs["h_hull_interp"]), hull_before)

    def test_restore_keeps_fixed_phases_in_composition_order(self, cu_mg_bl):
        source = copy.deepcopy(cu_mg_bl)
        victim = list_ablatable_phases(source)[0]
        restored = restore_phases(ablate_phases(source, victim.name), victim)
        comps = [restored._phase_x(p) for p in restored.phases if p.composition is not None]
        assert comps == sorted(comps)
        assert restored.phases[-1].phase_type == "liquid"

    def test_restore_does_not_alias_the_caller_s_phase(self, cu_mg_bl):
        """The state reset clears `points`; inserting the caller's object by reference would
        reach back into whatever they took it from."""
        source = copy.deepcopy(cu_mg_bl)
        victim = list_ablatable_phases(source)[0]
        victim.points = [[0.3, 900.0]]
        restored = restore_phases(ablate_phases(source, victim.name), victim)
        assert victim.points == [[0.3, 900.0]]
        inserted = next(p for p in restored.phases if p.name == victim.name)
        assert inserted is not victim

    def test_restore_rejects_a_solution_phase(self, cu_mg_bl):
        source = copy.deepcopy(cu_mg_bl)
        liquid = source.phases[-1]
        with pytest.raises(ValueError):
            restore_phases(source, liquid)
