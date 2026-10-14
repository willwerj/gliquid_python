"""The objective's guard against a generated two-liquid region the diagram does not expect.

``BinaryLiquid.expected_gap_ranges`` is where the assessed diagram allows two liquids;
``unexpected_gap_width`` is how much of the generated diagram's two-liquid span lies
outside those ranges; ``f`` scales such a fit up unless ``suppress_unexpected_migs`` is
False. Runs offline on the
shipped Ag-V and Cu-Mg caches; the corpus case needs the workspace store.
"""

import copy
import math

import pytest

import gliquid.config as config
from gliquid.binary import FIT_KWARGS, GAP_EXPONENT, GAP_WIDTH_SCALE, BinaryLiquid

AG_V_DOME = [54923.70, 6.8246, 608.58, 0.0]
GUESS = [AG_V_DOME[1], AG_V_DOME[2]]
NO_PENALTIES = {"check_lupis_elliott": False, "use_lxb_penalty": False}


def _prepared(system, params, invariants=(), **cache_kwargs):
    bl = BinaryLiquid.from_cache(system, param_format="comb-exp", **cache_kwargs)
    assert not bl.init_error
    bl.invariants = list(invariants)
    bl.low_t_exp_phases = []
    bl.update_params(params)
    bl.guess_symbols = bl.xs_mix.guess_symbols
    bl.constraints = None
    return bl


def _agv_mig():
    bl = BinaryLiquid.from_cache("Ag-V", param_format="comb-exp")
    invariants, _ = bl.find_invariant_points(verbose=False)
    return [iv for iv in invariants if iv["type"] == "mig"]


def _floating(mig):
    off = copy.deepcopy(mig)
    for foot in off["cbounds"]:
        foot[1] += 200.0
    return off


class TestExpectedGapRanges:
    def test_genuine_feet_give_their_span(self):
        bl = _prepared("Ag-V", AG_V_DOME, _agv_mig())
        (span,) = bl.expected_gap_ranges
        assert span == pytest.approx([0.0774836, 0.9218040000000001])

    def test_no_record_gives_nothing(self):
        assert _prepared("Ag-V", AG_V_DOME).expected_gap_ranges == []

    def test_feet_off_the_liquidus_give_the_whole_axis(self):
        bl = _prepared("Ag-V", AG_V_DOME, [_floating(_agv_mig()[0])])
        assert bl.expected_gap_ranges == [[0.0, 1.0]]

    def test_survives_deepcopy(self):
        bl = _prepared("Ag-V", AG_V_DOME, _agv_mig())
        assert copy.deepcopy(bl).expected_gap_ranges == bl.expected_gap_ranges


class TestUnexpectedGapWidth:
    def test_whole_dome_is_unexpected_without_a_record(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        bl.update_phase_points()
        ((lo, hi),) = bl.hsx.two_liquid_spans()
        assert bl.unexpected_gap_width(match_tol=0.0) == pytest.approx(hi - lo)

    def test_no_dome_means_no_width(self):
        bl = _prepared("Cu-Mg", [0.0, 0.0, 0.0, 0.0])
        bl.update_phase_points()
        assert bl.unexpected_gap_width() == 0.0


class TestObjective:
    def test_switch_off_restores_the_plain_objective(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        plain = bl.f(GUESS, suppress_unexpected_migs=False, **NO_PENALTIES)
        assert math.isfinite(plain)
        _, rmse, _, _ = bl.calculate_deviation_metrics()
        assert plain == pytest.approx(rmse)

    def test_the_objective_scales_by_the_unexpected_width(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        plain = bl.f(GUESS, suppress_unexpected_migs=False, **NO_PENALTIES)
        scaled = bl.f(GUESS, **NO_PENALTIES)
        width = bl.unexpected_gap_width()
        assert width > 0
        assert scaled == pytest.approx(plain * (1 + width / GAP_WIDTH_SCALE) ** GAP_EXPONENT)

    def test_default_is_on(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        plain = bl.f(GUESS, suppress_unexpected_migs=False, **NO_PENALTIES)
        assert bl.f(GUESS, **NO_PENALTIES) > plain

    def test_expected_dome_passes_untouched(self):
        bl = _prepared("Ag-V", AG_V_DOME, _agv_mig())
        plain = bl.f(GUESS, suppress_unexpected_migs=False, **NO_PENALTIES)
        assert bl.f(GUESS, **NO_PENALTIES) == pytest.approx(plain)

    def test_feet_off_the_liquidus_never_suppress(self):
        bl = _prepared("Ag-V", AG_V_DOME, [_floating(_agv_mig()[0])])
        plain = bl.f(GUESS, suppress_unexpected_migs=False, **NO_PENALTIES)
        assert bl.f(GUESS, **NO_PENALTIES) == pytest.approx(plain)

    def test_no_dome_system_is_unaffected(self):
        bl = _prepared("Cu-Mg", [0.0, 0.0, 0.0, 0.0])
        guess = [0.0, 0.0]
        plain = bl.f(guess, suppress_unexpected_migs=False, **NO_PENALTIES)
        assert bl.f(guess, **NO_PENALTIES) == plain


class TestFitOptions:
    def test_the_switch_is_accepted(self):
        assert "suppress_unexpected_migs" in FIT_KWARGS

    def test_an_unknown_option_still_raises(self):
        bl = BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp")
        with pytest.raises(TypeError):
            bl.fit_parameters(suppress_unexpected_domes=True)

    def test_match_tolerance_knob(self):
        previous = config.gap_match_tol
        try:
            config.set_gap_match_tol(0.05)
            assert config.gap_match_tol == 0.05
            with pytest.raises(ValueError):
                config.set_gap_match_tol(-0.1)
        finally:
            config.set_gap_match_tol(previous)


@pytest.mark.needs_cache
def test_pb_sn_rule_fit_is_rejected(matrix_data_store):
    config.set_solid_solutions(False)
    bl = BinaryLiquid.from_cache("Pb-Sn", param_format="comb-exp", pd_ind=0, tau=8000)
    assert not bl.init_error
    bl.invariants, bl.low_t_exp_phases = bl.find_invariant_points(verbose=False, t_tol=15)
    bl.unsupported_comp_ranges = [[0.0, 0.563476]]
    bl.update_params([6065.207387731687, 4.653917976346771, -1127.9927318433724, 0.0])
    bl.guess_symbols = bl.xs_mix.guess_symbols
    bl.constraints = None
    guess = [4.653917976346771, -1127.9927318433724]
    assert bl.expected_gap_ranges == []
    plain = bl.f(
        guess, suppress_unexpected_migs=False, unsupported_range_mode="cap", **NO_PENALTIES
    )
    assert math.isfinite(plain)
    assert bl.f(guess, unsupported_range_mode="cap", **NO_PENALTIES) > plain


class TestIsothermComparison:
    def _fake_mig(self, lo, hi, t_iso):
        return {
            "type": "mig",
            "comp": (lo + hi) / 2,
            "cbounds": [[lo, t_iso], [hi, t_iso]],
            "tbounds": [[(lo + hi) / 2, t_iso], [(lo + hi) / 2, t_iso + 500.0]],
            "phases": None,
            "phase_comps": None,
        }

    def test_assessed_dome_passes_at_its_isotherm(self):
        bl = _prepared("Ag-V", AG_V_DOME, _agv_mig())
        bl.update_phase_points()
        assert bl.unexpected_gap_width(0.05) == 0.0

    def test_isotherm_above_the_generated_dome_is_not_penalised(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        bl.invariants = [self._fake_mig(0.40, 0.60, 5000.0)]
        bl.digitized_liq = [[x, 5000.0] for x in (0.0, 0.5, 1.0)]
        bl.update_phase_points()
        assert bl.unexpected_gap_width(0.0) == 0.0

    def test_wide_tie_at_the_assessed_isotherm_is_penalised(self):
        bl = _prepared("Ag-V", AG_V_DOME)
        bl.invariants = [self._fake_mig(0.45, 0.55, 2500.0)]
        bl.digitized_liq = [[x, 2500.0] for x in (0.0, 0.5, 1.0)]
        bl.update_phase_points()
        assert bl.unexpected_gap_width(0.05) > 0.2
