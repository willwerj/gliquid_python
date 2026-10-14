"""The shared interface for the composition ranges with no solid free energy behind them.

Three things are pinned here:

  * ``BinaryLiquid.unsupported_comp_ranges`` is the attribute and the ``__init__`` kwarg;
    the removed ``ignored_comp_ranges`` spelling raises.
  * ``unsupported_range_mode``: ``'ignore'`` drops the unsupported points, ``'score'`` keeps
    and scores them, ``'cap'`` keeps them and counts only the residual above the assessed
    liquidus. The removed ``ignored_ranges`` bool raises.
  * Every fit carries the full-curve metrics and the windowed ones; the two agree exactly
    when nothing was masked, and the windowed pair is scored under the mode the optimizer
    actually ran with.

The capped objective's own arithmetic is pinned in
``tests/test_capped_unsupported_objective.py``.
"""

import pytest

import gliquid.binary as binary
import gliquid.solid_coverage as solid_coverage
from gliquid.binary import (
    DEFAULT_UNSUPPORTED_RANGE_MODE,
    FIT_KWARGS,
    MASK_FRACTION_CAP,
    UNSUPPORTED_RANGE_MODES,
    BinaryLiquid,
    resolve_unsupported_range_mode,
)

SYSTEM = "Hf-Zr"  # offline-cached, the same system the skip-reason contract drives
MASKED_SYSTEM = "Al-Cu"  # fits, and masks a real stretch of its liquidus
# The unmasked system is cu_mg_fit's Cu-Mg: its coverage report and invariant walk mask nothing.


def _report_with_ranges(ranges, min_supported_total=0.25):
    """A minimal ``SolidCoverageReport`` carrying only the ranges the mask reads."""
    return solid_coverage.SolidCoverageReport(
        span=(0.0, 1.0),
        unsupported_fraction=sum(hi - lo for lo, hi in ranges),
        unsupported_ranges=[list(r) for r in ranges],
        n_compounds=0,
        n_missing_compounds=0,
        phases=[],
        ss_models=(),
        thresholds={"min_supported_total": min_supported_total, "ignored_floor_slope": 0.0},
    )


def _paren_cmp(comp):
    """A terminal-phase cmp no DFT phase covers: the walk masks the flank away from 0.5."""
    return {"type": "cmp", "comp": comp, "temp": 1000.0, "phases": ["(X)"], "phase_comps": [comp]}


@pytest.fixture
def bl():
    return BinaryLiquid.from_cache(SYSTEM, pd_ind=0)


@pytest.fixture(scope="module")
def masked_fits():
    """``(system, fit_data)``: ``MASKED_SYSTEM`` fitted once per module under ``'cap'``."""
    system = BinaryLiquid.from_cache(MASKED_SYSTEM, param_format="comb-exp", pd_ind=0)
    return system, system.fit_parameters(n_opts=1, max_iter=64, unsupported_range_mode="cap")


class TestTheNameIsTheMeasurement:
    def test_the_attribute_is_the_new_name(self, bl):
        assert isinstance(bl.unsupported_comp_ranges, list)

    def test_the_removed_init_keyword_names_its_replacement(self):
        with pytest.raises(TypeError, match="use unsupported_comp_ranges"):
            BinaryLiquid("A-B", ["Cu", "Mg"], ignored_comp_ranges=[[0.2, 0.4]])


class TestTheModeReplacesTheBool:
    def test_the_default_is_score(self):
        assert DEFAULT_UNSUPPORTED_RANGE_MODE == "score"
        assert resolve_unsupported_range_mode({}) == "score"

    def test_the_kwarg_sets_are_swapped(self):
        assert "unsupported_range_mode" in FIT_KWARGS
        assert "ignored_ranges" not in FIT_KWARGS

    def test_the_removed_bool_raises_naming_the_mode(self):
        with pytest.raises(TypeError, match="unsupported_range_mode='ignore' or 'score'"):
            resolve_unsupported_range_mode({"ignored_ranges": True})

    def test_cap_resolves_like_the_other_two(self):
        assert resolve_unsupported_range_mode({"unsupported_range_mode": "cap"}) == "cap"

    def test_an_unrecognized_mode_raises(self):
        with pytest.raises(ValueError, match="unsupported_range_mode"):
            resolve_unsupported_range_mode({"unsupported_range_mode": "mask"})

    def test_the_modes_are_the_three_named_treatments(self):
        assert set(UNSUPPORTED_RANGE_MODES) == {"ignore", "cap", "score"}


class TestTheMetricHonoursTheMode:
    def test_score_keeps_the_unsupported_points(self, bl):
        full = bl.calculate_deviation_metrics()
        bl.unsupported_comp_ranges = [[0.30, 0.60]]
        assert bl.calculate_deviation_metrics(unsupported_range_mode="score") == full
        assert bl.calculate_deviation_metrics(unsupported_range_mode="ignore")[0] != pytest.approx(
            full[0], abs=1e-9
        )

    def test_cap_reaches_the_metric_and_forgives_only_downward(self, bl):
        full = bl.calculate_deviation_metrics()
        bl.unsupported_comp_ranges = [[0.30, 0.60]]
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert scored == full
        assert all(c <= s + 1e-12 for c, s in zip(capped, scored))

    def test_the_removed_bool_raises_at_the_metric(self, bl):
        with pytest.raises(TypeError, match="ignored_ranges"):
            bl.calculate_deviation_metrics(ignored_ranges=False)


class TestFitParametersReportsBothMetrics:
    """The windowed numbers ride alongside the full-curve ones on every completed fit."""

    def test_every_fit_carries_both(self, cu_mg_fit):
        _, fit_data = cu_mg_fit
        assert fit_data, "the fixture must produce at least one fit"
        for fit in fit_data:
            for key in ("mae", "rmse", "mape", "rmspe"):
                assert key in fit and f"{key}_window" in fit

    def test_they_agree_when_nothing_was_masked(self, cu_mg_fit):
        system, fit_data = cu_mg_fit
        assert not system.unsupported_comp_ranges, "fixture system must mask nothing"
        for fit in fit_data:
            for key in ("mae", "rmse", "mape", "rmspe"):
                assert fit[f"{key}_window"] == pytest.approx(fit[key], abs=1e-12)

    def test_they_part_company_when_something_was_masked(self, masked_fits):
        """The control on the test above: without a system that really masks, the
        identity would hold vacuously."""
        system, fit_data = masked_fits
        assert system.unsupported_comp_ranges, "fixture system must mask something"
        assert fit_data
        for fit in fit_data:
            assert fit["mae_window"] != pytest.approx(fit["mae"], abs=1e-9)

    def test_the_removed_bool_raises_at_the_fit(self, bl):
        with pytest.raises(TypeError, match="unsupported_range_mode"):
            bl.fit_parameters(n_opts=1, max_iter=2, ignored_ranges=False)

    def test_cap_is_accepted_by_the_fit(self, bl):
        assert isinstance(
            bl.fit_parameters(
                n_opts=1, max_iter=2, check_solid_coverage=False, unsupported_range_mode="cap"
            ),
            list,
        )

    def test_an_unrecognized_mode_still_raises_before_anything_is_fitted(self, bl):
        with pytest.raises(ValueError, match="unsupported_range_mode"):
            bl.fit_parameters(n_opts=1, max_iter=2, unsupported_range_mode="clip")


class TestTheWindowIsScoredUnderTheModeTheFitRanWith:
    """Under ``'cap'`` the windowed number is the CAPPED score, not the two-sided one."""

    def test_the_masked_system_still_fits_under_cap(self, masked_fits):
        system, fit_data = masked_fits
        assert system.unsupported_comp_ranges, "fixture system must mask something"
        assert fit_data

    def test_the_window_cannot_exceed_the_full_curve(self, masked_fits):
        """Both are scored on the same mesh under ``'cap'`` — the window takes the positive
        part where the full curve takes the magnitude."""
        _, fit_data = masked_fits
        for fit in fit_data:
            assert fit["mae_window"] <= fit["mae"] + 1e-9
            assert fit["mape_window"] <= fit["mape"] + 1e-9

    def test_the_window_is_the_capped_score_at_the_fitted_parameters(self, masked_fits):
        """Reload the winning parameters and re-score: the windowed number comes back
        under ``'cap'`` and the headline number under ``'score'``."""
        system, fit_data = masked_fits
        best = min(fit_data, key=lambda d: d["f"])
        reloaded = BinaryLiquid.from_cache(
            MASKED_SYSTEM,
            params=[float(best[p]) for p in ("L0_a", "L0_b", "L1_a", "L1_b")],
            param_format="comb-exp",
            pd_ind=0,
        )
        reloaded.unsupported_comp_ranges = [list(r) for r in system.unsupported_comp_ranges]
        reloaded.update_phase_points()
        assert reloaded.calculate_deviation_metrics(unsupported_range_mode="cap")[
            0
        ] == pytest.approx(best["mae_window"], abs=0.5)
        assert reloaded.calculate_deviation_metrics(unsupported_range_mode="score")[
            0
        ] == pytest.approx(best["mae"], abs=0.5)


class TestTheCoverageRangesReachTheMask:
    """``assess_solid_coverage`` computes them; ``fit_parameters`` does not discard them."""

    def test_a_coverage_report_feeds_the_mask(self, bl):
        bl.invariants, bl.low_t_exp_phases = [], []
        bl.coverage_report = _report_with_ranges([[0.20, 0.35]])
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert [0.20, 0.35] in bl.unsupported_comp_ranges

    def test_a_caller_supplied_list_is_used_as_is(self, bl):
        bl.invariants, bl.low_t_exp_phases = [], []
        bl.coverage_report = _report_with_ranges([[0.20, 0.35]])
        bl.unsupported_comp_ranges = [[0.80, 0.90]]
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.unsupported_comp_ranges == [[0.80, 0.90]]

    def test_they_do_not_reach_the_mask_fraction_cap(self, bl):
        """The coverage geometry is adjudicated by the gate's ``min_supported_total`` alone,
        so ``MASK_FRACTION_CAP`` cannot undercut an explicit ``coverage_thresholds``."""
        bl.invariants, bl.low_t_exp_phases = [], []
        bl.coverage_report = _report_with_ranges([[0.0, 1.0]])
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.skip_reason is None
        assert bl.unsupported_comp_ranges == [[0.0, 1.0]]


class TestTheMaskFractionCapRulesOnTheInvariantSourceOnly:
    """``MASK_FRACTION_CAP`` applies to the invariant-derived mask only, not to its union
    with the coverage mask."""

    def test_the_cap_is_a_named_constant(self):
        assert MASK_FRACTION_CAP == 0.55

    def _unbounded_walk(self, bl):
        """Two uncovered terminal-phase cmps with nothing beyond them: the walk masks
        [0, 0.45] and [0.55, 1], 0.9 of the span."""
        bl.invariants = [_paren_cmp(0.45), _paren_cmp(0.55)]
        bl.low_t_exp_phases = []
        bl.coverage_report = None
        return bl

    def test_the_invariant_derived_cap_still_fires(self, bl):
        """The capped objective did not quietly disable the skip: a wholly invariant-derived
        mask is still flagged at the shipped cap."""
        self._unbounded_walk(bl).fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.skip_reason == "mask_fraction_exceeded"

    def test_control_lifting_the_constant_admits_the_same_system(self, bl, monkeypatch):
        """Pins that ``fit_parameters`` reads the constant rather than a stale literal."""
        monkeypatch.setattr(binary, "MASK_FRACTION_CAP", 1.01)
        self._unbounded_walk(bl).fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.skip_reason is None
        assert bl.unsupported_comp_ranges == [[0, 0.45], [0.55, 1]]

    def test_a_wholly_coverage_derived_mask_escapes_even_a_zero_cap(self, monkeypatch):
        """The real-corpus control on the synthetic pin above: ``Al-Cu``'s mask comes
        entirely from its coverage report, and the cap never sees it."""
        monkeypatch.setattr(binary, "MASK_FRACTION_CAP", 0.0)
        system = BinaryLiquid.from_cache(MASKED_SYSTEM, param_format="comb-exp", pd_ind=0)
        system.fit_parameters(n_opts=1, max_iter=2)
        assert system.skip_reason is None
        assert system.unsupported_comp_ranges
        assert system.unsupported_comp_ranges == [
            list(r) for r in system.coverage_report.unsupported_ranges
        ]


class TestTheFloorRulesOnTheMergedMask:
    """``min_supported_total`` is taken twice: on the coverage-derived ranges, and again on
    those merged with the invariant-derived ones. A system whose two masks each clear the
    floor alone but not together is skipped as ``insufficient_union_coverage``."""

    @staticmethod
    def _stub_coverage(monkeypatch, ranges, floor, *, record=True):
        """Make gate one see a fixed report. ``record=False`` leaves ``coverage_report``
        None, which is the shape the merged-mask gate must tolerate."""

        def _assess(self, **overrides):
            report = _report_with_ranges(ranges, min_supported_total=floor)
            if record:
                self.coverage_report = report
            return report

        monkeypatch.setattr(BinaryLiquid, "assess_solid_coverage", _assess)

    @staticmethod
    def _flanked_walk(bl):
        """An uncovered terminal-phase cmp at 0.55 whose walk mask is bounded to [0.55, 0.75]."""
        bl.invariants = [_paren_cmp(0.55), {"type": "none", "comp": 0.75}]
        bl.low_t_exp_phases = []
        return bl

    def test_two_sources_that_each_pass_alone_are_skipped_together(self, bl, monkeypatch):
        self._stub_coverage(monkeypatch, [[0.0, 0.25]], 0.70)
        self._flanked_walk(bl)
        assert bl.fit_parameters(n_opts=1, max_iter=2) == []
        assert bl.skip_reason == binary.SKIP_UNION_COVERAGE
        assert bl.init_error is True
        assert bl.unsupported_comp_ranges == [[0.0, 0.25], [0.55, 0.75]]

    def test_a_merged_mask_that_clears_the_floor_still_fits(self, bl, monkeypatch):
        self._stub_coverage(monkeypatch, [[0.0, 0.25]], 0.50)
        self._flanked_walk(bl)
        bl.fit_parameters(n_opts=1, max_iter=2)
        assert bl.skip_reason is None
        assert bl.unsupported_comp_ranges == [[0.0, 0.25], [0.55, 0.75]]

    def test_overlapping_sources_are_merged_not_summed(self, bl, monkeypatch):
        """Masked width is 0.30, not the 0.50 the two range lists sum to."""
        self._stub_coverage(monkeypatch, [[0.50, 0.80]], 0.65)
        self._flanked_walk(bl)
        bl.fit_parameters(n_opts=1, max_iter=2)
        assert bl.skip_reason is None
        assert bl.unsupported_comp_ranges == [[0.50, 0.80]]

    def test_check_solid_coverage_false_disables_this_verdict_too(self, bl, monkeypatch):
        self._stub_coverage(monkeypatch, [[0.0, 0.25]], 0.70)
        self._flanked_walk(bl)
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.skip_reason is None

    def test_a_caller_supplied_mask_bypasses_it(self, bl, monkeypatch):
        """Like the mask-fraction cap, it rules on auto-derived ranges only."""
        self._stub_coverage(monkeypatch, [[0.0, 0.25]], 0.70)
        self._flanked_walk(bl)
        bl.unsupported_comp_ranges = [[0.90, 0.95]]
        bl.fit_parameters(n_opts=1, max_iter=2)
        assert bl.skip_reason is None
        assert bl.unsupported_comp_ranges == [[0.90, 0.95]]

    def test_an_unrecorded_report_does_not_raise(self, bl, monkeypatch):
        self._stub_coverage(monkeypatch, [[0.0, 0.25]], 0.70, record=False)
        self._flanked_walk(bl)
        bl.fit_parameters(n_opts=1, max_iter=2)
        assert bl.coverage_report is None
        assert bl.skip_reason is None
