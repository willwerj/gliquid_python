"""``unsupported_range_mode='cap'`` — the one-sided depression objective.

Inside a composition window with no solid free energy behind it, the missing solid propped
the real liquidus up, so the reconstruction should fall BELOW the assessed curve there.
``'cap'`` encodes exactly that: the capped points stay in the mesh and in both curves, and
only ``max(T_model - T_data, 0)`` counts. Below the curve is free.

What is pinned here:

  * an empty ``unsupported_comp_ranges`` makes all three modes bit-identical;
  * a model below the assessed curve throughout the window contributes exactly zero from
    it, and a model above contributes the full positive residual;
  * ``'cap'`` never scores worse than ``'score'`` on the same parameters, because it scores
    a subset of the same residual;
  * ``liquidus_holes`` is still an unconditional hard mask under ``'cap'``;
  * a mesh that is entirely capped reports ``inf``: every residual would be one-sided, and
    any sufficiently depressed curve would score zero.

The synthetic cases drive ``calculate_deviation_metrics`` through a bare object carrying
only the six attributes it reads, so the arithmetic is exact rather than approximately
recovered from a real diagram.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gliquid.binary import BinaryLiquid

MODES = ("ignore", "cap", "score")

# Fine enough that a mesh point's nearest neighbour on either curve is within 0.001 of it,
# so a window boundary cannot leak a residual across itself.
GRID = np.arange(0.002, 1.0, 0.002)

# The capped window, and the strictly narrower stretch the model deviates over. The margin
# keeps every deviating point inside the window under nearest-neighbour lookup.
WINDOW = [0.28, 0.62]
DEVIATION = (0.30, 0.60)
BASE_T = 1000.0


def _synthetic(model_offsets, unsupported=(), holes=None):
    """A BinaryLiquid carrying only what ``calculate_deviation_metrics`` reads.

    ``model_offsets`` maps a composition to the generated temperature offset from
    ``BASE_T``; the digitized curve is flat at ``BASE_T``.
    """
    bl = BinaryLiquid.__new__(BinaryLiquid)
    bl.sys_name = "Syn-Th"
    bl.init_error = False
    bl.assessed_crange_floor = 0.7
    bl.liq_coverage = {"holes": [list(h) for h in holes]} if holes else None
    bl.unsupported_comp_ranges = [list(r) for r in unsupported]
    bl.digitized_liq = [(float(x), BASE_T) for x in GRID]
    bl.phases = [SimpleNamespace(points=[(float(x), BASE_T + model_offsets(x)) for x in GRID])]
    return bl


def _inside(x):
    return DEVIATION[0] <= x <= DEVIATION[1]


def _below(x):
    return -100.0 if _inside(x) else 0.0


def _above(x):
    return 100.0 if _inside(x) else 0.0


def _tilted(x):
    """Deviates on both sides of the assessed curve inside the window."""
    if not _inside(x):
        return 0.0
    return -120.0 if x < 0.45 else 80.0


def _everywhere(x):
    return 40.0 * np.sin(12.0 * x) - 25.0


class TestAnEmptyMaskCannotMove:
    """With nothing unsupported, every mode takes the identical path."""

    def test_the_three_modes_are_bit_identical_on_a_synthetic_curve(self):
        bl = _synthetic(_everywhere)
        results = [bl.calculate_deviation_metrics(unsupported_range_mode=m) for m in MODES]
        assert all(np.isfinite(results[0]))
        assert results[0] == results[1] == results[2]

    @pytest.mark.parametrize("system", ["Hf-Zr", "Cu-Mg"])
    def test_the_three_modes_are_bit_identical_on_a_cached_system(self, system):
        bl = BinaryLiquid.from_cache(system, pd_ind=0)
        assert bl.unsupported_comp_ranges == [], "fixture system must mask nothing"
        results = [bl.calculate_deviation_metrics(unsupported_range_mode=m) for m in MODES]
        assert all(np.isfinite(results[0]))
        assert results[0] == results[1] == results[2]

    def test_holes_alone_still_leave_the_modes_identical(self):
        """A hole is masked under every mode, so it cannot separate them either."""
        bl = _synthetic(_everywhere, holes=[[0.40, 0.50]])
        results = [bl.calculate_deviation_metrics(unsupported_range_mode=m) for m in MODES]
        assert all(np.isfinite(results[0]))
        assert results[0] == results[1] == results[2]


class TestTheWindowIsScoredOneSided:
    def test_a_model_below_the_curve_contributes_nothing(self):
        bl = _synthetic(_below, unsupported=[WINDOW])
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        assert capped == (0.0, 0.0, 0.0, 0.0)

    def test_control_the_same_depression_is_scored_under_score(self):
        """Without this the assertion above would pass on a curve that never deviates."""
        bl = _synthetic(_below, unsupported=[WINDOW])
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert scored[0] > 0.0

    def test_a_model_above_the_curve_pays_the_full_residual(self):
        bl = _synthetic(_above, unsupported=[WINDOW])
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert capped == scored
        assert capped[0] > 0.0

    def test_only_the_positive_half_survives_a_two_sided_deviation(self):
        """The capped residual is the positive part point by point, not a per-window
        verdict: the stretch above the curve still pays while the stretch below does not."""
        bl = _synthetic(_tilted, unsupported=[WINDOW])
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert 0.0 < capped[0] < scored[0]

    def test_points_outside_the_window_keep_their_magnitude(self):
        """A depression OUTSIDE the mask is still a full two-sided residual."""
        outside = _synthetic(lambda x: -100.0 if x < 0.2 else 0.0, unsupported=[WINDOW])
        capped = outside.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = outside.calculate_deviation_metrics(unsupported_range_mode="score")
        assert capped == scored
        assert capped[0] > 0.0


class TestCapNeverScoresWorseThanScore:
    """It scores a subset of the same residual on the same mesh."""

    @pytest.mark.parametrize(
        "offsets", [_below, _above, _tilted, _everywhere], ids=["below", "above", "tilted", "wavy"]
    )
    @pytest.mark.parametrize(
        "unsupported",
        [[[0.28, 0.62]], [[0.05, 0.25]], [[0.10, 0.30], [0.55, 0.80]]],
        ids=["mid", "left", "split"],
    )
    def test_every_metric_is_no_larger(self, offsets, unsupported):
        bl = _synthetic(offsets, unsupported=unsupported)
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert all(np.isfinite(scored))
        assert all(c <= s + 1e-12 for c, s in zip(capped, scored))

    @pytest.mark.parametrize("system", ["Al-Cu", "Hf-Zr", "Cu-Mg"])
    @pytest.mark.parametrize("unsupported", [[[0.20, 0.45]], [[0.60, 0.85]]])
    def test_it_holds_on_cached_systems_too(self, system, unsupported):
        bl = BinaryLiquid.from_cache(system, pd_ind=0)
        bl.unsupported_comp_ranges = [list(r) for r in unsupported]
        capped = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        scored = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        assert all(np.isfinite(scored))
        assert all(c <= s + 1e-12 for c, s in zip(capped, scored))


class TestHolesStayHardUnderCap:
    def test_a_poison_inside_a_hole_is_inert(self):
        clean = _synthetic(_everywhere, holes=[[0.40, 0.50]]).calculate_deviation_metrics(
            unsupported_range_mode="cap"
        )
        poisoned = _synthetic(
            lambda x: 90000.0 if 0.40 <= x <= 0.50 else _everywhere(x), holes=[[0.40, 0.50]]
        ).calculate_deviation_metrics(unsupported_range_mode="cap")
        assert poisoned == clean

    def test_control_the_same_poison_lands_without_the_hole(self):
        clean = _synthetic(_everywhere).calculate_deviation_metrics(unsupported_range_mode="cap")
        poisoned = _synthetic(
            lambda x: 90000.0 if 0.40 <= x <= 0.50 else _everywhere(x)
        ).calculate_deviation_metrics(unsupported_range_mode="cap")
        assert poisoned[0] > 10 * clean[0]

    def test_a_hole_is_not_downgraded_to_a_cap_by_an_overlapping_range(self):
        """The hole is dropped even where an unsupported range covers the same stretch,
        so a depression inside it cannot be scored as free either."""
        clean = _synthetic(
            _everywhere, unsupported=[[0.35, 0.55]], holes=[[0.40, 0.50]]
        ).calculate_deviation_metrics(unsupported_range_mode="cap")
        poisoned = _synthetic(
            lambda x: 90000.0 if 0.40 <= x <= 0.50 else _everywhere(x),
            unsupported=[[0.35, 0.55]],
            holes=[[0.40, 0.50]],
        ).calculate_deviation_metrics(unsupported_range_mode="cap")
        assert poisoned == clean


class TestAnAllCappedMeshRefusesToScore:
    """A wholly one-sided objective is driven to zero by any sufficiently depressed curve,
    so it must not read downstream as a perfect fit."""

    def test_a_mask_covering_the_mesh_reports_inf(self):
        bl = _synthetic(_everywhere, unsupported=[[0.0, 1.0]])
        assert bl.calculate_deviation_metrics(unsupported_range_mode="cap") == (float("inf"),) * 4

    def test_a_depressed_curve_does_not_score_zero_instead(self):
        bl = _synthetic(lambda x: -500.0, unsupported=[[0.0, 1.0]])
        assert bl.calculate_deviation_metrics(unsupported_range_mode="cap") == (float("inf"),) * 4

    def test_sparse_data_does_not_waive_it(self):
        bl = _synthetic(_everywhere, unsupported=[[0.0, 1.0]])
        assert (
            bl.calculate_deviation_metrics(unsupported_range_mode="cap", allow_sparse_data=True)
            == (float("inf"),) * 4
        )

    def test_one_surviving_two_sided_point_is_enough_to_report(self):
        """The floor above counts capped points because they are scored; the guard rejects
        the case where that is ALL they are, not the case where they merely dominate."""
        bl = _synthetic(_everywhere, unsupported=[[0.05, 1.0]])
        metrics = bl.calculate_deviation_metrics(unsupported_range_mode="cap")
        assert all(np.isfinite(metrics))


class TestCappedPointsStayInTheMesh:
    """The structural repair: a mask that starves ``'ignore'`` does not starve ``'cap'``."""

    def test_a_wide_mask_that_returns_inf_under_ignore_still_scores_under_cap(self):
        bl = _synthetic(_everywhere, unsupported=[[0.09, 0.92]])
        assert (
            bl.calculate_deviation_metrics(unsupported_range_mode="ignore") == (float("inf"),) * 4
        )
        assert all(np.isfinite(bl.calculate_deviation_metrics(unsupported_range_mode="cap")))

    def test_a_mask_reaching_an_endpoint_clears_the_remaining_range_check(self):
        """``'ignore'`` shortens the surviving composition span below
        the objective's own span guard; ``'cap'`` keeps the span because it keeps the points."""
        bl = _synthetic(_everywhere, unsupported=[[0.58, 1.0]])
        assert (
            bl.calculate_deviation_metrics(unsupported_range_mode="ignore") == (float("inf"),) * 4
        )
        assert all(np.isfinite(bl.calculate_deviation_metrics(unsupported_range_mode="cap")))
