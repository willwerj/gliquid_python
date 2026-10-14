"""``min_comp_span`` scores a handful of measured liquidus points over a narrow span.

A measured liquidus is a few points over whatever composition span was measured.
``calculate_deviation_metrics`` reports ``inf`` when the comparison covers less than the
minimum composition span; ``min_comp_span=0`` together with ``allow_sparse_data=True``
scores the points however narrow their span.
"""

from types import SimpleNamespace

import numpy as np

from gliquid.binary import FIT_KWARGS, BinaryLiquid

GRID = np.arange(0.002, 1.0, 0.002)
BASE_T = 1000.0
OFFSET = 50.0
INF = (float("inf"),) * 4


def _measured(xs):
    """A BinaryLiquid whose digitized liquidus is ``xs`` at ``BASE_T`` and whose generated
    liquidus sits ``OFFSET`` above it everywhere."""
    bl = BinaryLiquid.__new__(BinaryLiquid)
    bl.sys_name = "Syn-Th"
    bl.init_error = False
    bl.assessed_crange_floor = 0.7
    bl.liq_coverage = None
    bl.unsupported_comp_ranges = []
    bl.digitized_liq = [(float(x), BASE_T) for x in xs]
    bl.phases = [SimpleNamespace(points=[(float(x), BASE_T + OFFSET) for x in GRID])]
    return bl


NARROW = (0.05, 0.15, 0.73)
WIDE = (0.05, 0.15, 0.80)


def test_a_narrow_measured_span_is_scored_with_min_comp_span_zero():
    mae, rmse, _, _ = _measured(NARROW).calculate_deviation_metrics(
        unsupported_range_mode="score", allow_sparse_data=True, min_comp_span=0
    )
    assert mae == OFFSET
    assert rmse == OFFSET


def test_positive_control_a_wide_measured_span_scores_the_same_by_default():
    mae, _, _, _ = _measured(WIDE).calculate_deviation_metrics(
        unsupported_range_mode="score", allow_sparse_data=True
    )
    assert mae == OFFSET


def test_sparse_data_alone_keeps_the_default_span_floor():
    bl = _measured(NARROW)
    assert (
        bl.calculate_deviation_metrics(unsupported_range_mode="score", allow_sparse_data=True)
        == INF
    )


def test_without_sparse_data_the_ten_point_floor_still_applies():
    bl = _measured(NARROW)
    assert bl.calculate_deviation_metrics(unsupported_range_mode="score", min_comp_span=0) == INF


def test_fit_parameters_accepts_the_keyword():
    assert "min_comp_span" in FIT_KWARGS
