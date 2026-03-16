"""
Unit tests for BinaryLiquid fitting behavior across a diverse set of binary systems.

Uses the same data paths and config as the production Fit_Binary_Systems.py script.
Systems are drawn from the benchmark set in Benchmark_Models&Constraints.py and
span simple eutectics, peritectics, congruent melters, miscibility gaps, and
systems with varying numbers of DFT phases.

Run with: python -m pytest tests/test_fitting_behavior.py -v
    or:   python tests/test_fitting_behavior.py            (standalone)
"""
import pytest
import numpy as np

import gliquid.binary as binary
import gliquid.load_binary_data as lbd
from gliquid.binary import BinaryLiquid, a_sym, b_sym, c_sym, d_sym

# --- Test systems ------------------------------------------------------------
BENCHMARK_SYSTEMS = [
    'Ag-V',   # few DFT phases
    'Al-Cu',  # many DFT phases, well-studied
    'C-Nb',   # refractory carbide
    'Cr-Eu',  # rare-earth + transition metal
    'Cu-Mg',  # classic eutectic with congruent melter
]

DEFAULT_PFORMAT = 'comb-exp'
DEFAULT_N_OPTS = 5
DEFAULT_MAX_ITER = 64
VALID_PFORMATS = ['linear', 'combined', 'comb-exp']


# =============================================================================
# Helpers
# =============================================================================
def _load_bl(sys_name: str, pformat: str = DEFAULT_PFORMAT, **kwargs) -> BinaryLiquid:
    """Load a BinaryLiquid from cache with standard settings."""
    return BinaryLiquid.from_cache(sys_name, param_format=pformat, **kwargs)


def _fit_bl(bl: BinaryLiquid, n_opts: int = DEFAULT_N_OPTS, max_iter: int = DEFAULT_MAX_ITER,
            **kwargs) -> list[dict]:
    """Run parameter fitting with standard settings."""
    return bl.fit_parameters(n_opts=n_opts, verbose=False, max_iter=max_iter, **kwargs)


# =============================================================================
# Tests: Initialization
# =============================================================================
class TestBinaryLiquidInit:
    """Verify that BinaryLiquid objects initialize correctly from cached data."""

    @pytest.mark.parametrize("sys_name", BENCHMARK_SYSTEMS)
    def test_from_cache_loads(self, sys_name):
        bl = _load_bl(sys_name)
        assert bl.sys_name == '-'.join(sorted(sys_name.split('-')))
        assert len(bl.components) == 2
        assert bl.dft_ch is not None

    @pytest.mark.parametrize("sys_name", BENCHMARK_SYSTEMS)
    def test_has_digitized_liquidus(self, sys_name):
        bl = _load_bl(sys_name)
        if bl.digitized_liq:
            assert len(bl.digitized_liq) >= 3, "Digitized liquidus should have at least 3 points"

    @pytest.mark.parametrize("sys_name", BENCHMARK_SYSTEMS)
    def test_default_params_zero(self, sys_name):
        bl = _load_bl(sys_name)
        assert bl.get_params() == [0.0, 0.0, 0.0, 0.0]


# =============================================================================
# Tests: Input formatting and validation
# =============================================================================
class TestInputValidation:
    """Validate accepted and rejected system-name and parameter-input formats."""

    @pytest.mark.parametrize(
        "input_sys, expected_components, expected_sys, expected_order_changed",
        [
            ('Cu-Mg', ['Cu', 'Mg'], 'Cu-Mg', False),
            ('Mg-Cu', ['Mg', 'Cu'], 'Mg-Cu', True),
            (['Cu', 'Mg'], ['Cu', 'Mg'], 'Cu-Mg', False),
            (['Mg', 'Cu'], ['Mg', 'Cu'], 'Mg-Cu', True),
            ('Ag-V', ['Ag', 'V'], 'Ag-V', False),
            ('V-Ag', ['V', 'Ag'], 'V-Ag', True),
        ],
    )
    def test_system_name_alternative_formats(
        self,
        input_sys,
        expected_components,
        expected_sys,
        expected_order_changed,
    ):
        components, sys_name, order_changed = lbd.validate_and_format_binary_system(input_sys)
        assert components == expected_components
        assert sys_name == expected_sys
        assert order_changed is expected_order_changed

    @pytest.mark.parametrize(
        "bad_input",
        [
            'CuMg',
            'Cu-Mg-Zn',
            ['Cu'],
            ['Cu', 'Mg', 'Zn'],
            [1, 2],
            {'a': 'b'},
        ],
    )
    def test_wrong_system_formats_raise(self, bad_input):
        with pytest.raises(Exception):
            _load_bl(bad_input)

    @pytest.mark.parametrize("bad_sys", ['A-B', 'Xx-Cu', ['A', 'B'], ['Cu', 'Xx']])
    def test_incorrect_element_symbols_raise(self, bad_sys):
        with pytest.raises(Exception):
            _load_bl(bad_sys)

    @pytest.mark.parametrize("params", ([0, 0, 0, 0], (1, 2, 3, 4), [1.2, -0.5, 3.1, 0]))
    def test_validate_params_accepts_four_numeric_values(self, params):
        validated = binary.validate_binary_mixing_parameters(params)
        assert validated == [float(x) for x in params]

    @pytest.mark.parametrize(
        "bad_params",
        [
            None,
            '1,2,3,4',
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, '3', 4],
            [True, 0, 0, 0],
            {'L0_a': 1, 'L0_b': 2, 'L1_a': 3, 'L1_b': 4},
        ],
    )
    def test_validate_params_rejects_invalid_formats(self, bad_params):
        with pytest.raises(Exception):
            binary.validate_binary_mixing_parameters(bad_params)

    @pytest.mark.parametrize("params", ([0, 0, 0, 0], [10, -1, 5, 0]))
    def test_from_cache_param_validation_accepts_valid_formats(self, params):
        bl = _load_bl('Cu-Mg', params=params)
        assert bl.get_params() == [float(x) for x in params]

    @pytest.mark.parametrize("bad_params", ([1, 2, 3], [1, 2, '3', 4], {'a': 1}))
    def test_from_cache_param_validation_rejects_bad_formats(self, bad_params):
        with pytest.raises(Exception):
            _load_bl('Cu-Mg', params=bad_params)

    @pytest.mark.parametrize("pformat", VALID_PFORMATS)
    def test_valid_param_formats_initialize(self, pformat):
        bl = _load_bl('Cu-Mg', pformat=pformat)
        assert bl._param_format == pformat
        assert 'g_liquid' in bl.eqs

    @pytest.mark.parametrize("bad_pformat", ['exponential', 'pseudo', 'foo', '', None])
    def test_invalid_param_formats_raise(self, bad_pformat):
        with pytest.raises(Exception):
            _load_bl('Cu-Mg', pformat=bad_pformat)


# =============================================================================
# Tests: Objective function & penalty
# =============================================================================
class TestObjectiveFunction:
    """Test that f() behaves correctly with and without penalty."""

    @pytest.fixture
    def bl_cu_mg(self):
        bl = _load_bl('Cu-Mg')
        bl.guess_symbols = [a_sym, c_sym]
        import sympy as sp
        no_L0Sxs_eq = sp.Eq(sp.diff(bl.eqs['l0'], binary.t_sym).subs({binary.t_sym: 0}), 0)
        no_L1Sxs_eq = sp.Eq(sp.diff(bl.eqs['l1'], binary.t_sym).subs({binary.t_sym: 0}), 0)
        bl.constraints = sp.solve([no_L0Sxs_eq, no_L1Sxs_eq], (b_sym, d_sym),
                                  rational=False, simplify=False)
        return bl

    def test_f_returns_finite_for_zero_params(self, bl_cu_mg):
        """f() with zero params should return a finite MAE (bad fit, not inf)."""
        val = bl_cu_mg.f([0, 0], check_lupis_elliott=False)
        assert np.isfinite(val)


# =============================================================================
# Tests: Fitting behavior
# =============================================================================
class TestFittingBehavior:
    """End-to-end fitting tests on select benchmark systems."""

    @pytest.mark.parametrize("sys_name", ['Cu-Mg', 'Ag-V', 'C-Nb'])
    def test_fit_produces_results(self, sys_name):
        """Fitting should produce at least one result for well-behaved systems."""
        bl = _load_bl(sys_name)
        if bl.init_error or not bl.digitized_liq:
            pytest.skip(f"{sys_name}: init error or no liquidus data")
        fit_data = _fit_bl(bl, n_opts=3, max_iter=32)
        assert len(fit_data) > 0, f"No fits produced for {sys_name}"

    @pytest.mark.parametrize("sys_name", ['Cu-Mg', 'Ag-V', 'C-Nb'])
    def test_fit_mae_is_finite(self, sys_name):
        """All returned fits should have finite MAE."""
        bl = _load_bl(sys_name)
        if bl.init_error or not bl.digitized_liq:
            pytest.skip(f"{sys_name}: init error or no liquidus data")
        fit_data = _fit_bl(bl, n_opts=3, max_iter=32)
        for fit in fit_data:
            assert np.isfinite(fit['mae']), f"Infinite MAE in fit for {sys_name}"

    @pytest.mark.parametrize("sys_name", ['Cu-Mg', 'Ag-V', 'C-Nb'])
    def test_zero_params_produce_finite_mae(self, sys_name):
        """Zero-valued parameters should still produce a finite MAE."""
        bl = _load_bl(sys_name)
        if bl.init_error or not bl.digitized_liq:
            pytest.skip(f"{sys_name}: init error or no liquidus data")
        bl.update_params([0, 0, 0, 0])
        bl.update_phase_points()
        mae, *_ = bl.calculate_deviation_metrics()
        assert np.isfinite(mae), f"Zero-parameter MAE was non-finite for {sys_name}"

    def test_fitted_params_improve_over_zero_params(self):
        """For Cu-Mg, best fitted MAE should be lower than zero-parameter MAE."""
        bl = _load_bl('Cu-Mg')
        if bl.init_error or not bl.digitized_liq:
            pytest.skip("Cu-Mg: init error or no liquidus data")

        bl.update_params([0, 0, 0, 0])
        bl.update_phase_points()
        zero_mae, *_ = bl.calculate_deviation_metrics()

        fit_data = _fit_bl(bl, n_opts=3, max_iter=32)
        if not fit_data:
            pytest.skip("Cu-Mg: no fit results returned")

        best_fit = min(fit_data, key=lambda fit: fit.get('mae', float('inf')))
        assert best_fit['mae'] < zero_mae, (
            f"Expected fitted MAE < zero-parameter MAE, got {best_fit['mae']} vs {zero_mae}"
        )


# =============================================================================
# Standalone runner
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
