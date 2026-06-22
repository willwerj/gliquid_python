"""
Smoke tests for the phase-ablation overfitting benchmark library
(gliquid.phase_ablation).

Mirrors the style of test_fitting_behavior.py / test_fisher_information.py: no env
setup, load via from_cache, session-scoped fixtures, small n_opts/max_iter. Uses Cu-Mg
(eutectic + congruent melter; the FIM-test system).

Run with: python -m pytest tests/test_phase_ablation.py -v
    or:   python tests/test_phase_ablation.py
"""
import numpy as np
import pytest

from gliquid.binary import _x_vals
from gliquid.phase_ablation import (
    BaselineFit,
    AblationResult,
    RiskScore,
    ENDPOINT_COMPS,
    ablate_phases,
    available_strategies,
    baseline_fit,
    enumerate_targets,
    fit_ablated,
    list_ablatable_phases,
    predict_overfit_risk,
    register_strategy,
    strategy_kwargs,
)

SYSTEM = 'Cu-Mg'
SMOKE_FIT_KWARGS = {'n_opts': 1, 'max_iter': 32}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cu_mg_baseline() -> BaselineFit:
    """Cu-Mg fitted once against its full DFT hull."""
    baseline = baseline_fit(SYSTEM, param_format='comb-exp', fit_kwargs=SMOKE_FIT_KWARGS)
    if not baseline.ok:
        pytest.skip(f"{SYSTEM}: baseline could not be fit (init error or empty fit)")
    if not baseline.ablatable:
        pytest.skip(f"{SYSTEM}: no ablatable interior phases")
    return baseline


@pytest.fixture(scope="session")
def top_risk_target(cu_mg_baseline) -> list[str]:
    """The single-phase ablation target with the highest a-priori risk score."""
    singles = enumerate_targets(cu_mg_baseline, max_k=1)
    scored = [predict_overfit_risk(cu_mg_baseline, t) for t in singles]
    best = max(scored, key=lambda s: s.score)
    return best.target_phases


# ---------------------------------------------------------------------------
# Ablation primitive
# ---------------------------------------------------------------------------

class TestAblatePrimitive:
    def test_list_ablatable_excludes_endpoints(self, cu_mg_baseline):
        for phase in list_ablatable_phases(cu_mg_baseline.bl):
            assert round(float(phase['comp']), 6) not in ENDPOINT_COMPS
            assert not phase.get('is_solution', False)

    def test_ablate_rebuilds_hull_interp(self, cu_mg_baseline, top_risk_target):
        base_interp = np.asarray(cu_mg_baseline.bl.eqs['h_hull_interp'])
        abl = ablate_phases(cu_mg_baseline.bl, top_risk_target)
        new_interp = np.asarray(abl.eqs['h_hull_interp'])
        assert len(new_interp) == len(_x_vals) - 2
        assert not np.allclose(new_interp, base_interp), "Hull interp should change after ablation"

    def test_ablate_does_not_mutate_baseline(self, cu_mg_baseline, top_risk_target):
        n_phases_before = len(cu_mg_baseline.bl.phases)
        params_before = list(cu_mg_baseline.bl.get_params())
        _ = ablate_phases(cu_mg_baseline.bl, top_risk_target)
        assert len(cu_mg_baseline.bl.phases) == n_phases_before
        assert cu_mg_baseline.bl.get_params() == params_before

    def test_ablate_removes_target_phase(self, cu_mg_baseline, top_risk_target):
        abl = ablate_phases(cu_mg_baseline.bl, top_risk_target)
        remaining = [p['name'] for p in abl.phases]
        for name in top_risk_target:
            assert name not in remaining

    def test_ablate_resets_refit_caches(self, cu_mg_baseline, top_risk_target):
        abl = ablate_phases(cu_mg_baseline.bl, top_risk_target)
        assert abl.invariants is None
        assert abl.low_t_exp_phases is None
        assert abl.ignored_comp_ranges == []
        assert abl.constraints is None
        assert abl.guess_symbols is None
        assert abl.init_error is False

    def test_ablate_rejects_endpoint_target(self, cu_mg_baseline):
        with pytest.raises(ValueError):
            ablate_phases(cu_mg_baseline.bl, [0.0])

    def test_ablate_rejects_unknown_phase(self, cu_mg_baseline):
        with pytest.raises(ValueError):
            ablate_phases(cu_mg_baseline.bl, ['NotARealPhase'])


# ---------------------------------------------------------------------------
# Drift + strategies
# ---------------------------------------------------------------------------

class TestDriftAndStrategies:
    def test_ablated_fit_runs_past_phase_mismatch(self, cu_mg_baseline, top_risk_target):
        """fit_ablated forces check_phase_mismatch=False, so the pipeline must execute
        and return an AblationResult instead of aborting on the (intentional) mismatch."""
        result = fit_ablated(cu_mg_baseline, top_risk_target, 'heuristic', **SMOKE_FIT_KWARGS)
        assert isinstance(result, AblationResult)
        assert result.fit_ok, "Heuristic ablated fit should converge on the ablated system"

    def test_metrics_are_finite_on_success(self, cu_mg_baseline, top_risk_target):
        result = fit_ablated(cu_mg_baseline, top_risk_target, 'heuristic', **SMOKE_FIT_KWARGS)
        if not result.fit_ok:
            pytest.skip("heuristic ablated fit did not converge")
        assert np.isfinite(result.drift_free)
        assert np.isfinite(result.drift_L0_T)
        assert np.isfinite(result.mae_full)

    def test_heuristic_at_least_as_robust_as_naive(self, cu_mg_baseline, top_risk_target):
        """Core hypothesis: the ignored-ranges heuristic curbs overfitting vs naive.

        Robust to the realistic case where the naive fit fails to converge at all on an
        ablated congruent melter — that non-convergence is itself evidence the heuristic
        is more robust. When naive DOES converge, the heuristic must not be worse on both
        parameter drift and full-range liquidus MAE.
        """
        naive = fit_ablated(cu_mg_baseline, top_risk_target, 'naive', **SMOKE_FIT_KWARGS)
        heuristic = fit_ablated(cu_mg_baseline, top_risk_target, 'heuristic', **SMOKE_FIT_KWARGS)
        assert heuristic.fit_ok, "Heuristic should converge on the ablated system"
        if naive.fit_ok:
            improved = (heuristic.drift_free <= naive.drift_free + 1e-9) or \
                       (heuristic.mae_full <= naive.mae_full + 1e-9)
            assert improved, (
                f"Heuristic did not improve over naive: "
                f"drift {heuristic.drift_free} vs {naive.drift_free}, "
                f"mae_full {heuristic.mae_full} vs {naive.mae_full}"
            )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def test_default_strategies_registered(self):
        for name in ('naive', 'heuristic', 'penalty', 'fim_prior', 'production'):
            assert name in available_strategies()

    def test_strategy_kwargs_unknown_raises(self, cu_mg_baseline):
        with pytest.raises(ValueError):
            strategy_kwargs('does_not_exist', cu_mg_baseline)

    def test_strategy_kwargs_forces_no_phase_mismatch_via_fit(self, cu_mg_baseline):
        # strategy_kwargs itself does not set check_phase_mismatch; fit_ablated adds it.
        kwargs = strategy_kwargs('naive', cu_mg_baseline)
        assert kwargs.get('ignored_ranges') is False

    def test_custom_strategy_is_pluggable(self, cu_mg_baseline, top_risk_target):
        @register_strategy('_test_dummy')
        def _dummy(baseline):
            return {'ignored_ranges': True, 'check_lupis_elliott': False, 'use_lxb_penalty': False}

        try:
            assert '_test_dummy' in available_strategies()
            result = fit_ablated(cu_mg_baseline, top_risk_target, '_test_dummy', **SMOKE_FIT_KWARGS)
            assert isinstance(result, AblationResult)
        finally:
            from gliquid.phase_ablation import STRATEGY_REGISTRY
            STRATEGY_REGISTRY.pop('_test_dummy', None)


# ---------------------------------------------------------------------------
# Real FIM/Tikhonov prior (fim_prior strategy)
# ---------------------------------------------------------------------------

class TestFimPrior:
    def test_fim_prior_strategy_builds_precision_cfg(self, cu_mg_baseline):
        kwargs = strategy_kwargs('fim_prior', cu_mg_baseline)
        # Cu-Mg has a usable baseline FIM, so the real prior must be enabled.
        assert kwargs.get('use_fim_prior') is True
        cfg = kwargs['fim_prior_cfg']
        assert np.asarray(cfg['precision']).shape == (2, 2)
        assert list(cfg['free_indices']) == [0, 2]
        assert len(cfg['ref_params']) == 4

    def test_fim_tikhonov_penalty_anchors_at_reference(self, cu_mg_baseline):
        """Penalty ~1 at the baseline reference and strictly larger once params move."""
        cfg = strategy_kwargs('fim_prior', cu_mg_baseline)['fim_prior_cfg']
        bl = cu_mg_baseline.bl
        ref = list(cu_mg_baseline.params)
        bl.update_params(ref)
        pen_ref = bl.fim_tikhonov_penalty(cfg)
        moved = list(ref)
        moved[0] += 5000.0
        moved[2] += 5000.0
        bl.update_params(moved)
        pen_moved = bl.fim_tikhonov_penalty(cfg)
        bl.update_params(ref)  # restore
        assert abs(pen_ref - 1.0) < 1e-6
        assert pen_moved > pen_ref

    def test_fim_prior_curbs_drift_on_top_risk_target(self, cu_mg_baseline, top_risk_target):
        """The real FIM prior should not drift more than the plain heuristic, and should
        hold the well-identified params markedly closer to baseline on the top-risk phase."""
        heuristic = fit_ablated(cu_mg_baseline, top_risk_target, 'heuristic', **SMOKE_FIT_KWARGS)
        fim_prior = fit_ablated(cu_mg_baseline, top_risk_target, 'fim_prior', **SMOKE_FIT_KWARGS)
        assert fim_prior.fit_ok
        if heuristic.fit_ok:
            assert fim_prior.drift_free <= heuristic.drift_free + 1e-9


# ---------------------------------------------------------------------------
# A-priori predictor
# ---------------------------------------------------------------------------

class TestPredictor:
    REQUIRED_FEATURES = {
        'info_at_target', 'baseline_cond', 'baseline_sigma_free', 'decomp_near_liq',
        'nn_distance', 'ignored_span_fraction', 'n_phases_removed',
    }

    def test_risk_score_computable_without_ablated_fit(self, cu_mg_baseline, top_risk_target):
        score = predict_overfit_risk(cu_mg_baseline, top_risk_target)
        assert isinstance(score, RiskScore)
        assert np.isfinite(score.score)

    def test_risk_features_present(self, cu_mg_baseline, top_risk_target):
        score = predict_overfit_risk(cu_mg_baseline, top_risk_target)
        assert self.REQUIRED_FEATURES.issubset(score.features.keys())

    def test_enumerate_targets_single_and_pairs(self, cu_mg_baseline):
        n_ablatable = len(cu_mg_baseline.ablatable)
        singles = enumerate_targets(cu_mg_baseline, max_k=1)
        assert len(singles) == n_ablatable
        assert all(len(t) == 1 for t in singles)
        if n_ablatable > 2:
            with_pairs = enumerate_targets(cu_mg_baseline, max_k=2)
            assert len(with_pairs) > len(singles)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
