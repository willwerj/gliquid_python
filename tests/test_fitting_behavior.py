"""Unit tests for BinaryLiquid fitting behavior across a diverse set of binary systems.

Uses the same data paths and config as production fitting. Systems are drawn from a
benchmark set spanning simple eutectics, peritectics, congruent melters, miscibility
gaps, and systems with varying numbers of DFT phases. ``python tests/test_fitting_behavior.py``
runs it standalone.
"""

import numpy as np
import pytest

import gliquid.binary as binary
from gliquid.binary import BinaryLiquid
from gliquid.phase import validate_and_format_system
from gliquid.solution import l_sym

# --- Test systems ------------------------------------------------------------
BENCHMARK_SYSTEMS = [
    "Ag-V",  # few DFT phases
    "Al-Cu",  # many DFT phases, well-studied
    "C-Nb",  # refractory carbide
    "Cr-Eu",  # rare-earth + transition metal
    "Cu-Mg",  # classic eutectic with congruent melter
]

DEFAULT_PFORMAT = "comb-exp"
DEFAULT_N_OPTS = 5
DEFAULT_MAX_ITER = 64
VALID_PFORMATS = ["linear", "combined", "comb-exp"]


# -- Helpers --
def _load_bl(sys_name: str, pformat: str = DEFAULT_PFORMAT, **kwargs) -> BinaryLiquid:
    """Load a BinaryLiquid from cache with standard settings."""
    return BinaryLiquid.from_cache(sys_name, param_format=pformat, **kwargs)


def _fit_bl(
    bl: BinaryLiquid, n_opts: int = DEFAULT_N_OPTS, max_iter: int = DEFAULT_MAX_ITER, **kwargs
) -> list[dict]:
    """Run parameter fitting with standard settings."""
    return bl.fit_parameters(n_opts=n_opts, verbose=False, max_iter=max_iter, **kwargs)


# -- Tests: Initialization --
class TestBinaryLiquidInit:
    """Verify that BinaryLiquid objects initialize correctly from cached data."""

    @pytest.mark.parametrize("sys_name", BENCHMARK_SYSTEMS)
    def test_from_cache_loads(self, sys_name):
        bl = _load_bl(sys_name)
        assert bl.sys_name == "-".join(sorted(sys_name.split("-")))
        assert len(bl.components) == 2
        assert bl.dft_ch is not None
        if bl.digitized_liq:
            assert len(bl.digitized_liq) >= 3, "Digitized liquidus should have at least 3 points"
        assert bl.get_params() == [0.0, 0.0, 0.0, 0.0]


# -- Tests: Input formatting and validation --
class TestInputValidation:
    """Validate accepted and rejected system-name and parameter-input formats."""

    @pytest.mark.parametrize(
        "input_sys, expected_components, expected_sys, expected_order_changed",
        [
            ("Cu-Mg", ["Cu", "Mg"], "Cu-Mg", False),
            ("Mg-Cu", ["Mg", "Cu"], "Mg-Cu", True),
            (["Cu", "Mg"], ["Cu", "Mg"], "Cu-Mg", False),
            (["Mg", "Cu"], ["Mg", "Cu"], "Mg-Cu", True),
            ("Ag-V", ["Ag", "V"], "Ag-V", False),
            ("V-Ag", ["V", "Ag"], "V-Ag", True),
        ],
    )
    def test_system_name_alternative_formats(
        self,
        input_sys,
        expected_components,
        expected_sys,
        expected_order_changed,
    ):
        components, sys_name, order_changed = validate_and_format_system(input_sys)
        assert components == expected_components
        assert sys_name == expected_sys
        assert order_changed is expected_order_changed

    @pytest.mark.parametrize(
        "bad_input",
        [
            "CuMg",
            "Cu-Mg-Zn",
            ["Cu"],
            ["Cu", "Mg", "Zn"],
            [1, 2],
            {"a": "b"},
        ],
    )
    def test_wrong_system_formats_raise(self, bad_input):
        with pytest.raises(ValueError):
            _load_bl(bad_input)

    @pytest.mark.parametrize("bad_sys", ["A-B", "Xx-Cu", ["A", "B"], ["Cu", "Xx"]])
    def test_incorrect_element_symbols_raise(self, bad_sys):
        with pytest.raises(ValueError):
            _load_bl(bad_sys)

    # Parameter-vector validation lives on RKPolyExp (tests/test_redlich_kister.py).

    @pytest.mark.parametrize("params", ([0, 0, 0, 0], [10, -1, 5, 0]))
    def test_from_cache_param_validation_accepts_valid_formats(self, params):
        bl = _load_bl("Cu-Mg", params=params)
        assert bl.get_params() == [float(x) for x in params]

    @pytest.mark.parametrize("bad_params", ([1, 2, 3], [1, 2, "3", 4], {"a": 1}))
    def test_from_cache_param_validation_rejects_bad_formats(self, bad_params):
        with pytest.raises(ValueError):
            _load_bl("Cu-Mg", params=bad_params)

    @pytest.mark.parametrize("pformat", VALID_PFORMATS)
    def test_valid_param_formats_initialize(self, pformat):
        bl = _load_bl("Cu-Mg", pformat=pformat)
        assert bl.param_format == pformat
        assert "g_liquid" in bl.eqs

    @pytest.mark.parametrize("bad_pformat", ["exponential", "pseudo", "foo", "", None])
    def test_invalid_param_formats_raise(self, bad_pformat):
        with pytest.raises(ValueError):
            _load_bl("Cu-Mg", pformat=bad_pformat)


# -- Tests: Objective function & penalty --
class TestObjectiveFunction:
    """Test that f() behaves correctly with and without penalty."""

    @pytest.fixture
    def bl_cu_mg(self):
        bl = _load_bl("Cu-Mg")
        bl.guess_symbols = [l_sym("L0_a"), l_sym("L1_a")]
        import sympy as sp

        no_L0Sxs_eq = sp.Eq(sp.diff(bl.eqs["l0"], binary.t_sym).subs({binary.t_sym: 0}), 0)
        no_L1Sxs_eq = sp.Eq(sp.diff(bl.eqs["l1"], binary.t_sym).subs({binary.t_sym: 0}), 0)
        bl.constraints = sp.solve(
            [no_L0Sxs_eq, no_L1Sxs_eq],
            (l_sym("L0_b"), l_sym("L1_b")),
            rational=False,
            simplify=False,
        )
        return bl

    def test_f_returns_finite_for_zero_params(self, bl_cu_mg):
        """f() with zero params should return a finite MAE (bad fit, not inf)."""
        val = bl_cu_mg.f([0, 0], check_lupis_elliott=False)
        assert np.isfinite(val)

    def test_f_is_inf_when_the_liquidus_is_discontinuous(self):
        # L0_b above 2R makes the liquid entropy convex mid-axis, so no liquid joins the hull there
        params = [0.0, 25.0, 0.0, 0.0]
        bl = _load_bl("Cu-Mg", pformat="linear", params=params)
        bl.update_phase_points()
        assert not bl.h0_below_ch()
        assert not bl.liquidus_is_continuous()
        bl.guess_symbols = list(bl.xs_mix.format.symbols())
        assert bl.f(params) == float("inf")

    @staticmethod
    def _stub_objective(bl, mae):
        """Reduce f() to its penalty stage: every constraint passes and the MAE is fixed."""
        bl.guess_symbols = []
        bl.solve_params_from_constraints = lambda guessed_vals: None
        bl.h0_below_ch = lambda: False
        bl.update_phase_points = lambda: None
        bl.liquidus_is_continuous = lambda: True
        bl.calculate_deviation_metrics = lambda **kwargs: (mae, mae, 0.0, 0.0)
        return bl

    def test_f_multiplies_by_the_lupis_elliott_penalty_it_is_given_the_cfg_for(self):
        bl = self._stub_objective(_load_bl("Cu-Mg", pformat="linear"), 10.0)
        received = []
        bl.lupis_elliott_penalty = lambda cfg=None: received.append(cfg) or 2.0
        cfg = {"strength": 0.5, "hs_ratio": 1.0}
        val = bl.f([], lupis_elliott_cfg=cfg, suppress_unexpected_migs=False)
        assert received == [cfg]
        assert val == pytest.approx(20.0)

    def test_f_skips_the_lupis_elliott_penalty_when_disabled(self):
        bl = self._stub_objective(_load_bl("Cu-Mg", pformat="linear"), 10.0)
        received = []
        bl.lupis_elliott_penalty = lambda cfg=None: received.append(cfg) or 2.0
        val = bl.f([], check_lupis_elliott=False, suppress_unexpected_migs=False)
        assert received == []
        assert val == pytest.approx(10.0)


def _with_lupis_terms(bl, h0, s0, h1=0.0, s1=0.0):
    """Replace the L0/L1 enthalpy and entropy evaluators with constants."""
    bl.eqs["h_l0_lambdified"] = lambda *args: h0
    bl.eqs["s_l0_lambdified"] = lambda *args: s0
    bl.eqs["h_l1_lambdified"] = lambda *args: h1
    bl.eqs["s_l1_lambdified"] = lambda *args: s1
    return bl


# One order's term at h = 2, s = -1, hs_ratio = 1 and unit strength:
# 2 * sqrt(x^2 + y^2) * |x*y| / (|x| + |y|) with x = 2, y = -1.
UNIT_TERM = 4.0 * np.sqrt(5.0) / 3.0


class TestLupisElliottPenalty:
    """BinaryLiquid.lupis_elliott_penalty against its documented form."""

    def test_matching_signs_are_not_penalized(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, 1.0, -1.0, -3.0)
        cfg = {"strength": 1.0, "hs_ratio": 1.0, "l1": {"strength": 1.0}}
        assert bl.lupis_elliott_penalty(cfg) == 1.0

    def test_zero_parameters_are_not_penalized(self):
        bl = _load_bl("Cu-Mg", pformat="linear")
        assert bl.lupis_elliott_penalty({"strength": 1.0, "l1": {"strength": 1.0}}) == 1.0

    def test_opposite_signs_scale_with_strength(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, -1.0)
        penalty = bl.lupis_elliott_penalty({"strength": 0.5, "hs_ratio": 1.0})
        assert penalty == pytest.approx(1.0 + 0.5 * UNIT_TERM)

    def test_default_strength(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, -1.0)
        penalty = bl.lupis_elliott_penalty({"hs_ratio": 1.0})
        assert penalty == pytest.approx(1.0 + 7.5e-3 * UNIT_TERM)

    @pytest.mark.parametrize("cfg", [{"scale": 0.5}, {"l0": {"scale": 0.5}}])
    def test_the_removed_scale_key_raises(self, cfg):
        bl = _load_bl("Cu-Mg", pformat="linear")
        with pytest.raises(TypeError, match="'strength'"):
            bl.lupis_elliott_penalty(cfg)

    def test_hs_ratio_divides_the_enthalpy(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 4.0, -1.0)
        penalty = bl.lupis_elliott_penalty({"strength": 1.0, "hs_ratio": 2.0})
        assert penalty == pytest.approx(1.0 + UNIT_TERM)

    def test_a_per_term_strength_overrides_the_global_one(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, -1.0)
        cfg = {"strength": 1.0, "hs_ratio": 1.0, "l0": {"strength": 0.25}}
        assert bl.lupis_elliott_penalty(cfg) == pytest.approx(1.0 + 0.25 * UNIT_TERM)

    def test_l1_is_penalized_only_when_configured(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 1.0, 1.0, 2.0, -1.0)
        assert bl.lupis_elliott_penalty({"strength": 1.0, "hs_ratio": 1.0}) == 1.0
        cfg = {"strength": 1.0, "hs_ratio": 1.0, "l1": {"strength": 0.5}}
        assert bl.lupis_elliott_penalty(cfg) == pytest.approx(1.0 + 0.5 * UNIT_TERM)

    def test_the_orders_add_inside_one_factor(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, -1.0, 2.0, -1.0)
        cfg = {"strength": 1.0, "hs_ratio": 1.0, "l1": {"strength": 0.5}}
        assert bl.lupis_elliott_penalty(cfg) == pytest.approx(1.0 + 1.5 * UNIT_TERM)

    def test_comb_exp_inspects_l0_only(self):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="comb-exp"), 1.0, 1.0, 2.0, -1.0)
        cfg = {"strength": 1.0, "hs_ratio": 1.0, "l1": {"strength": 1.0}}
        assert bl.lupis_elliott_penalty(cfg) == 1.0

    @pytest.mark.parametrize("strength", [0.0, -1.0])
    def test_a_non_positive_strength_disables_the_term(self, strength):
        bl = _with_lupis_terms(_load_bl("Cu-Mg", pformat="linear"), 2.0, -1.0)
        assert bl.lupis_elliott_penalty({"strength": strength, "hs_ratio": 1.0}) == 1.0


# -- Tests: Fitting behavior --
# (n_opts, max_iter) per system. C-Nb returns no fit at (3, 32): do not lower its budget.
FIT_BUDGETS = {"Cu-Mg": (3, 32), "Ag-V": (3, 32), "C-Nb": (5, 64)}


def _load_with_liquidus(sys_name: str) -> BinaryLiquid:
    """``_load_bl(sys_name)``; skips on an init error or an empty digitized liquidus."""
    bl = _load_bl(sys_name)
    if bl.init_error or not bl.digitized_liq:
        pytest.skip(f"{sys_name}: init error or no liquidus data")
    return bl


def _zero_param_mae(sys_name: str) -> float:
    """MAE of a freshly loaded ``sys_name`` at zero parameters."""
    bl = _load_with_liquidus(sys_name)
    bl.update_params([0, 0, 0, 0])
    bl.update_phase_points()
    mae, *_ = bl.calculate_deviation_metrics()
    return mae


@pytest.fixture(scope="module")
def fit_once():
    """``fit_once(sys_name) -> list[dict]``: one fit per system per module at its
    ``FIT_BUDGETS`` entry. Cu-Mg runs with ``enable_multi_threading=True``."""
    fits = {}

    def fit(sys_name):
        if sys_name not in fits:
            n_opts, max_iter = FIT_BUDGETS[sys_name]
            fits[sys_name] = _fit_bl(
                _load_with_liquidus(sys_name),
                n_opts=n_opts,
                max_iter=max_iter,
                enable_multi_threading=sys_name == "Cu-Mg",
            )
        return fits[sys_name]

    return fit


class TestFittingBehavior:
    """End-to-end fitting tests on select benchmark systems, one fit per system."""

    @pytest.mark.parametrize("sys_name", list(FIT_BUDGETS))
    def test_fit_produces_results(self, fit_once, sys_name):
        """Fitting returns a non-empty list, and every fit has a finite MAE."""
        fit_data = fit_once(sys_name)
        assert isinstance(fit_data, list)
        assert len(fit_data) > 0, f"No fits produced for {sys_name}"
        for fit in fit_data:
            assert np.isfinite(fit["mae"]), f"Infinite MAE in fit for {sys_name}"

    @pytest.mark.parametrize("sys_name", list(FIT_BUDGETS))
    def test_zero_params_produce_finite_mae(self, sys_name):
        """Zero-valued parameters should still produce a finite MAE."""
        mae = _zero_param_mae(sys_name)
        assert np.isfinite(mae), f"Zero-parameter MAE was non-finite for {sys_name}"

    def test_fitted_params_improve_over_zero_params(self, fit_once):
        """For Cu-Mg, best fitted MAE should be lower than zero-parameter MAE."""
        zero_mae = _zero_param_mae("Cu-Mg")
        fit_data = fit_once("Cu-Mg")
        if not fit_data:
            pytest.skip("Cu-Mg: no fit results returned")

        best_fit = min(fit_data, key=lambda fit: fit.get("mae", float("inf")))
        assert best_fit["mae"] < zero_mae, (
            f"Expected fitted MAE < zero-parameter MAE, got {best_fit['mae']} vs {zero_mae}"
        )


# -- Standalone runner --
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
