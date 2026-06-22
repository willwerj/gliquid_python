"""
Tests for the Fisher Information Matrix module (gliquid.fisher_information).

Run with: python -m pytest tests/test_fisher_information.py -v
    or:   python tests/test_fisher_information.py

Session-scoped fixture fits Cu-Mg once; most tests reuse it.
"""
import copy

import numpy as np
import pytest

from gliquid.binary import BinaryLiquid, a_sym, b_sym, c_sym, d_sym
from gliquid.fisher_information import (
    FIMResult,
    NMSensitivityResult,
    OptimalMeasurementResult,
    ConstraintComparisonResult,
    _interpolate_liquidus_at_x,
    _eval_liquidus_at_params,
    _free_param_indices,
    build_nm_path_parameter_precision,
    compute_jacobian,
    compute_fim,
    find_optimal_next_measurement,
    compute_nm_path_sensitivity,
    compare_constraint_sets,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bl_cu_mg_fitted():
    """Cu-Mg BinaryLiquid, fitted once for the whole session."""
    bl = BinaryLiquid.from_cache('Cu-Mg', param_format='comb-exp')
    if bl.init_error or not bl.digitized_liq:
        pytest.skip("Cu-Mg: init error or no liquidus data")
    bl.fit_parameters(n_opts=1, max_iter=32, verbose=False)
    return bl


@pytest.fixture(scope="session")
def fim_cu_mg(bl_cu_mg_fitted):
    """FIMResult for the fitted Cu-Mg system."""
    return compute_fim(bl_cu_mg_fitted)


# ---------------------------------------------------------------------------
# Helper: _interpolate_liquidus_at_x
# ---------------------------------------------------------------------------

class TestInterpolateLiquidus:
    def test_exact_match(self):
        pts = [[0.2, 900.0], [0.5, 1100.0], [0.8, 950.0]]
        assert _interpolate_liquidus_at_x(pts, 0.5) == pytest.approx(1100.0)

    def test_linear_interpolation(self):
        pts = [[0.0, 800.0], [1.0, 1200.0]]
        assert _interpolate_liquidus_at_x(pts, 0.5) == pytest.approx(1000.0)

    def test_out_of_range_left_returns_nan(self):
        pts = [[0.3, 900.0], [0.7, 950.0]]
        assert np.isnan(_interpolate_liquidus_at_x(pts, 0.1))

    def test_out_of_range_right_returns_nan(self):
        pts = [[0.3, 900.0], [0.7, 950.0]]
        assert np.isnan(_interpolate_liquidus_at_x(pts, 0.9))

    def test_empty_returns_nan(self):
        assert np.isnan(_interpolate_liquidus_at_x([], 0.5))


# ---------------------------------------------------------------------------
# Helper: _eval_liquidus_at_params
# ---------------------------------------------------------------------------

class TestEvalLiquidusAtParams:
    def test_does_not_mutate_bl(self, bl_cu_mg_fitted):
        params_before = bl_cu_mg_fitted.get_params()
        nmpath_before = bl_cu_mg_fitted.nmpath

        params_4 = list(bl_cu_mg_fitted._params)
        x_query = np.array([0.3, 0.5, 0.7])
        _eval_liquidus_at_params(bl_cu_mg_fitted, params_4, x_query)

        assert bl_cu_mg_fitted.get_params() == params_before
        assert bl_cu_mg_fitted.nmpath is nmpath_before

    def test_nominal_returns_finite_values(self, bl_cu_mg_fitted):
        params_4 = list(bl_cu_mg_fitted._params)
        x_query = np.array([0.3, 0.5, 0.7])
        T = _eval_liquidus_at_params(bl_cu_mg_fitted, params_4, x_query)
        assert T.shape == (3,)
        # At least some values should be finite (valid liquidus range)
        assert np.any(np.isfinite(T))

    def test_length_matches_query(self, bl_cu_mg_fitted):
        params_4 = list(bl_cu_mg_fitted._params)
        x_query = np.linspace(0.1, 0.9, 15)
        T = _eval_liquidus_at_params(bl_cu_mg_fitted, params_4, x_query)
        assert len(T) == 15


# ---------------------------------------------------------------------------
# compute_jacobian
# ---------------------------------------------------------------------------

class TestComputeJacobian:
    def test_shape(self, bl_cu_mg_fitted):
        x_query = np.linspace(0.2, 0.8, 10)
        J, valid_mask = compute_jacobian(bl_cu_mg_fitted, x_query)
        n_free = len(bl_cu_mg_fitted.guess_symbols)
        assert J.ndim == 2
        assert J.shape[1] == n_free
        assert len(valid_mask) == len(x_query)
        assert J.shape[0] == int(valid_mask.sum())

    def test_valid_mask_is_bool(self, bl_cu_mg_fitted):
        x_query = np.linspace(0.2, 0.8, 5)
        _, valid_mask = compute_jacobian(bl_cu_mg_fitted, x_query)
        assert valid_mask.dtype == bool

    def test_does_not_mutate_bl(self, bl_cu_mg_fitted):
        params_before = bl_cu_mg_fitted.get_params()
        x_query = np.linspace(0.2, 0.8, 5)
        compute_jacobian(bl_cu_mg_fitted, x_query)
        assert bl_cu_mg_fitted.get_params() == params_before

    def test_all_finite(self, bl_cu_mg_fitted):
        x_query = np.linspace(0.2, 0.8, 8)
        J, _ = compute_jacobian(bl_cu_mg_fitted, x_query)
        assert np.all(np.isfinite(J))


# ---------------------------------------------------------------------------
# compute_fim
# ---------------------------------------------------------------------------

class TestComputeFIM:
    def test_returns_fim_result(self, fim_cu_mg):
        assert isinstance(fim_cu_mg, FIMResult)

    def test_fim_is_square(self, bl_cu_mg_fitted, fim_cu_mg):
        n_free = len(bl_cu_mg_fitted.guess_symbols)
        assert fim_cu_mg.fim.shape == (n_free, n_free)

    def test_fim_is_symmetric(self, fim_cu_mg):
        assert np.allclose(fim_cu_mg.fim, fim_cu_mg.fim.T, atol=1e-10)

    def test_fim_is_psd(self, fim_cu_mg):
        assert np.all(fim_cu_mg.eigenvalues >= -1e-10)

    def test_param_names_correct_subset(self, fim_cu_mg):
        valid_names = {'L0_a', 'L0_b', 'L1_a', 'L1_b'}
        for name in fim_cu_mg.param_names:
            assert name in valid_names

    def test_param_names_count_matches_fim_shape(self, fim_cu_mg):
        assert len(fim_cu_mg.param_names) == fim_cu_mg.fim.shape[0]

    def test_param_variances_positive(self, fim_cu_mg):
        assert np.all(fim_cu_mg.param_variances > 0)

    def test_x_used_and_rejected_partition_input(self, bl_cu_mg_fitted):
        x_query = np.linspace(0.05, 0.95, 20)
        result = compute_fim(bl_cu_mg_fitted, x_compositions=x_query)
        total = len(result.x_used) + len(result.x_rejected)
        assert total == len(x_query)

    def test_x_rejected_outside_liquidus(self, bl_cu_mg_fitted):
        # x=0 and x=1 are endpoint compositions; they are often outside range
        # Use a few very-near-endpoint compositions that are likely out-of-range
        x_query = np.concatenate([np.linspace(0.01, 0.05, 3), np.linspace(0.2, 0.8, 10)])
        result = compute_fim(bl_cu_mg_fitted, x_compositions=x_query)
        # All accepted x must be such that T_model is finite
        params_4 = list(bl_cu_mg_fitted._params)
        T_check = _eval_liquidus_at_params(bl_cu_mg_fitted, params_4, result.x_used)
        assert np.all(np.isfinite(T_check))

    def test_condition_number_positive(self, fim_cu_mg):
        assert fim_cu_mg.condition_number > 0

    def test_sigma_stored(self, bl_cu_mg_fitted):
        result = compute_fim(bl_cu_mg_fitted, sigma=5.0)
        assert result.sigma == pytest.approx(5.0)

    def test_sigma_scales_fim_inversely(self, bl_cu_mg_fitted):
        r1 = compute_fim(bl_cu_mg_fitted, sigma=1.0)
        r2 = compute_fim(bl_cu_mg_fitted, sigma=2.0)
        # FIM = J^T J / sigma^2, so r2.fim = r1.fim / 4
        assert np.allclose(r2.fim, r1.fim / 4.0, rtol=1e-6)

    def test_more_data_increases_det_fim(self, bl_cu_mg_fitted):
        """Adding more measurement compositions should increase det(FIM)."""
        xs_small = np.linspace(0.25, 0.75, 5)
        xs_large = np.linspace(0.2, 0.8, 15)
        r_small = compute_fim(bl_cu_mg_fitted, x_compositions=xs_small)
        r_large = compute_fim(bl_cu_mg_fitted, x_compositions=xs_large)
        assert r_large.det_fim >= r_small.det_fim


# ---------------------------------------------------------------------------
# find_optimal_next_measurement
# ---------------------------------------------------------------------------

class TestFindOptimalNextMeasurement:
    def test_returns_correct_type(self, fim_cu_mg, bl_cu_mg_fitted):
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted)
        assert isinstance(opt, OptimalMeasurementResult)

    def test_scores_all_geq_one(self, fim_cu_mg, bl_cu_mg_fitted):
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted)
        assert np.all(opt.d_optimal_scores >= 1.0 - 1e-10)

    def test_scores_sorted_descending(self, fim_cu_mg, bl_cu_mg_fitted):
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted)
        diffs = np.diff(opt.d_optimal_scores)
        assert np.all(diffs <= 1e-10)

    def test_ranked_x_in_unit_interval(self, fim_cu_mg, bl_cu_mg_fitted):
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted)
        assert np.all((opt.ranked_x >= 0.0) & (opt.ranked_x <= 1.0))

    def test_custom_candidates(self, fim_cu_mg, bl_cu_mg_fitted):
        candidates = np.array([0.3, 0.45, 0.6])
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted,
                                             candidate_x=candidates)
        assert len(opt.ranked_x) == 3

    def test_jacobian_shape_matches(self, fim_cu_mg, bl_cu_mg_fitted):
        candidates = np.linspace(0.2, 0.8, 6)
        opt = find_optimal_next_measurement(fim_cu_mg, bl_cu_mg_fitted,
                                             candidate_x=candidates)
        n_free = len(bl_cu_mg_fitted.guess_symbols)
        assert opt.jacobian_at_candidates.shape == (6, n_free)

    def test_all_out_of_range_candidates_use_boundary_fallback(self, monkeypatch):
        fim_result = FIMResult(
            fim=np.eye(2),
            fim_inv=np.eye(2),
            jacobian=np.zeros((2, 2)),
            x_used=np.array([0.2, 0.8]),
            x_rejected=np.array([]),
            eigenvalues=np.array([1.0, 1.0]),
            eigenvectors=np.eye(2),
            condition_number=1.0,
            det_fim=1.0,
            param_variances=np.array([1.0, 1.0]),
            param_names=['L0_a', 'L0_b'],
            is_singular=False,
            sigma=1.0,
        )

        def fake_compute_jacobian(bl, x_compositions, free_param_indices=None, h_rel=1e-3):
            x_compositions = np.asarray(x_compositions, dtype=float)
            return np.zeros((len(x_compositions), 2)), np.zeros(len(x_compositions), dtype=bool)

        monkeypatch.setattr('gliquid.fisher_information.compute_jacobian', fake_compute_jacobian)

        opt = find_optimal_next_measurement(
            fim_result,
            bl=object(),
            candidate_x=np.array([0.05, 0.18, 0.50, 0.90]),
        )

        assert opt.n_out_of_range == 4
        assert opt.ranked_x[0] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# compute_nm_path_sensitivity
# ---------------------------------------------------------------------------

class TestNMPathSensitivity:
    def test_returns_correct_type(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        result = compute_nm_path_sensitivity(bl_cu_mg_fitted)
        assert isinstance(result, NMSensitivityResult)

    def test_sensitivity_sums_to_one(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        result = compute_nm_path_sensitivity(bl_cu_mg_fitted)
        assert result.per_param_sensitivity.sum() == pytest.approx(1.0, abs=1e-8)

    def test_sensitivity_length_matches_free_params(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        result = compute_nm_path_sensitivity(bl_cu_mg_fitted)
        assert len(result.per_param_sensitivity) == len(bl_cu_mg_fitted.guess_symbols)

    def test_sensitivity_all_nonnegative(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        result = compute_nm_path_sensitivity(bl_cu_mg_fitted)
        assert np.all(result.per_param_sensitivity >= 0)

    def test_param_names_match_guess_symbols(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        result = compute_nm_path_sensitivity(bl_cu_mg_fitted)
        assert len(result.param_names) == len(bl_cu_mg_fitted.guess_symbols)

    def test_raises_without_nmpath(self):
        bl = BinaryLiquid.from_cache('Cu-Mg', param_format='comb-exp')
        bl.nmpath = None
        with pytest.raises(ValueError, match="nmpath"):
            compute_nm_path_sensitivity(bl)


class TestNMPathParameterPrecision:
    def test_returns_vector_with_free_param_length(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        precision = build_nm_path_parameter_precision(bl_cu_mg_fitted)
        assert precision is not None
        assert precision.shape == (len(bl_cu_mg_fitted.guess_symbols),)

    def test_precision_is_nonnegative(self, bl_cu_mg_fitted):
        if bl_cu_mg_fitted.nmpath is None:
            pytest.skip("nmpath not populated")
        precision = build_nm_path_parameter_precision(bl_cu_mg_fitted)
        assert precision is not None
        assert np.all(precision >= 0.0)

    def test_compute_fim_accepts_parameter_prior_precision(self, bl_cu_mg_fitted):
        n_free = len(bl_cu_mg_fitted.guess_symbols)
        prior_precision = np.linspace(1.0, 2.0, n_free)
        result = compute_fim(
            bl_cu_mg_fitted,
            sigma=1.0,
            parameter_prior_precision=prior_precision,
            parameter_prior_strength=0.25,
        )
        assert isinstance(result, FIMResult)
        assert result.fim.shape == (n_free, n_free)

    def test_compute_fim_rejects_bad_prior_shape(self, bl_cu_mg_fitted):
        with pytest.raises(ValueError, match="parameter_prior_precision"):
            compute_fim(
                bl_cu_mg_fitted,
                sigma=1.0,
                parameter_prior_precision=np.array([1.0, 2.0]),
                parameter_prior_strength=0.25,
            )


# ---------------------------------------------------------------------------
# compare_constraint_sets
# ---------------------------------------------------------------------------

class TestCompareConstraintSets:
    def _make_unconstrained_copy(self, bl):
        """Return a copy of bl with all 4 parameters free (no constraints)."""
        bl2 = copy.deepcopy(bl)
        bl2.guess_symbols = [a_sym, b_sym, c_sym, d_sym]
        bl2.constraints = {}
        return bl2

    def test_returns_correct_type(self, bl_cu_mg_fitted):
        bl_free = self._make_unconstrained_copy(bl_cu_mg_fitted)
        result = compare_constraint_sets(
            [bl_cu_mg_fitted, bl_free],
            ["constrained", "unconstrained"],
            x_compositions=np.linspace(0.2, 0.8, 8),
        )
        assert isinstance(result, ConstraintComparisonResult)

    def test_correct_number_of_entries(self, bl_cu_mg_fitted):
        bl_free = self._make_unconstrained_copy(bl_cu_mg_fitted)
        result = compare_constraint_sets(
            [bl_cu_mg_fitted, bl_free],
            ["constrained", "unconstrained"],
            x_compositions=np.linspace(0.2, 0.8, 8),
        )
        assert len(result.fim_results) == 2
        assert len(result.det_fim_values) == 2
        assert len(result.labels) == 2

    def test_eigenvalue_table_shape(self, bl_cu_mg_fitted):
        bl_free = self._make_unconstrained_copy(bl_cu_mg_fitted)
        result = compare_constraint_sets(
            [bl_cu_mg_fitted, bl_free],
            ["constrained", "unconstrained"],
            x_compositions=np.linspace(0.2, 0.8, 8),
        )
        assert result.eigenvalue_table.shape == (2, 4)

    def test_param_variance_table_has_all_keys(self, bl_cu_mg_fitted):
        bl_free = self._make_unconstrained_copy(bl_cu_mg_fitted)
        result = compare_constraint_sets(
            [bl_cu_mg_fitted, bl_free],
            ["constrained", "unconstrained"],
            x_compositions=np.linspace(0.2, 0.8, 8),
        )
        for name in ['L0_a', 'L0_b', 'L1_a', 'L1_b']:
            assert name in result.param_variance_table
            assert len(result.param_variance_table[name]) == 2

    def test_constrained_params_are_nan_for_constrained_system(self, bl_cu_mg_fitted):
        bl_free = self._make_unconstrained_copy(bl_cu_mg_fitted)
        # The constrained bl uses guess_symbols → free params; non-free params get nan variance
        from gliquid.fisher_information import _free_param_indices, _ALL_PARAM_NAMES
        free_indices = set(_free_param_indices(bl_cu_mg_fitted))
        constrained_names = [_ALL_PARAM_NAMES[i] for i in range(4) if i not in free_indices]
        result = compare_constraint_sets(
            [bl_cu_mg_fitted, bl_free],
            ["constrained", "unconstrained"],
            x_compositions=np.linspace(0.2, 0.8, 8),
        )
        for name in constrained_names:
            # Index 0 is the constrained system — constrained params should have nan variance
            assert np.isnan(result.param_variance_table[name][0])

    def test_mismatched_lengths_raises(self, bl_cu_mg_fitted):
        with pytest.raises(ValueError, match="same length"):
            compare_constraint_sets([bl_cu_mg_fitted], ["a", "b"])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
