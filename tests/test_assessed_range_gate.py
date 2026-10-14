"""The composition axis is partitioned three ways, and each part gates something different.

  ignored      no digitized liquidus at all -- outside the endpoints, or inside a hole the
               extractor refused to bridge. Gated by ``assessed_crange_floor``.
  unsupported  drawn liquidus whose conjugate solid has no evaluable free energy.
  supported    the rest. Gated by ``min_supported_total``.

Two things are easy to get wrong here and both are pinned below. A hole is IGNORED, not
supported -- scoring it as support credits a system for a curve nobody digitized. And the
BinaryLiquid attribute holding the ignored intervals is ``undigitized_comp_ranges``, kept
apart from ``unsupported_comp_ranges``, the solid-energy mask.
"""

import pytest

import gliquid.config as config
import gliquid.mpds as mpds

from ._mpds_builders import coverage_report, from_record


def _synthetic_hf_zr(regions):
    """A minimal Hf-Zr diagram whose real cached DFT hull is available offline."""
    return {
        "reference": {"entry": "https://mpds.io/entry/C-synthetic"},
        "comp_range": [0, 100],
        "temp": [0, 2600],
        "labels": [["HfZr<sub>2</sub>", [33.3, 1600]]],
        "chemical_elements": ["Hf", "Zr"],
        "entry": "C-synthetic",
        "jcode": "0000",
        "year": "1991",
        "shapes": [
            {
                "label": "L",
                "nphases": 1,
                "is_solid": False,
                "kind": "phase",
                "svgpath": "M " + " L ".join(f"{x},{tt}" for x, tt in r),
            }
            for r in regions
        ],
    }


CONTIGUOUS = [[[float(x), 2200.0 - 8.0 * x] for x in range(0, 101, 2)]]
TWO_REGIONS = [
    [[float(x), 2200.0 - 8.0 * x] for x in range(0, 11, 2)],
    [[float(x), 1400.0 + 4.0 * x] for x in range(35, 101, 5)],
]


def _from_cache(monkeypatch, regions, **kwargs):
    return from_record(monkeypatch, _synthetic_hf_zr(regions), **kwargs)


def _report(unsupported=(), ignored=None, span=(0.0, 1.0), floor=0.35):
    return coverage_report(unsupported, floor, 0.0, ignored, span)


class TestThePartition:
    def test_the_three_parts_tile_the_axis(self):
        rep = _report(unsupported=[[0.2, 0.5]], ignored=[[0.0, 0.1], [0.9, 1.0]])
        unsupported = 0.3
        assert rep.ignored_total == pytest.approx(0.2)
        assert rep.supported_total == pytest.approx(0.5)
        assert rep.ignored_total + rep.supported_total + unsupported == pytest.approx(1.0)

    def test_a_hole_is_ignored_not_supported(self):
        """Nothing was digitized there, so nothing can be scored against it either way."""
        rep = _report(ignored=[[0.4, 0.9]])
        assert rep.supported_total == pytest.approx(0.5)
        assert rep.supported_segments == [(0.0, 0.4), (0.9, 1.0)]
        assert rep.assessed_fraction == pytest.approx(0.5)

    def test_without_explicit_ranges_only_the_span_complement_is_ignored(self):
        """A report built by a caller that supplies no holes still partitions correctly."""
        rep = _report(span=(0.2, 0.8))
        assert rep.ignored_intervals == [(0.0, 0.2), (0.8, 1.0)]
        assert rep.supported_total == pytest.approx(0.6)

    def test_a_mask_running_past_the_endpoints_is_clipped(self):
        rep = _report(unsupported=[[-0.3, 0.4]], span=(0.0, 1.0))
        assert rep.supported_total == pytest.approx(0.6)


class TestIgnoredCompRanges:
    def test_holes_join_the_span_complement(self):
        got = mpds.ignored_comp_ranges([0.05, 0.95], {"holes": [[0.4, 0.6]]})
        assert got == [[0.0, 0.05], [0.4, 0.6], [0.95, 1.0]]

    def test_no_liquidus_ignores_the_whole_axis(self):
        assert mpds.ignored_comp_ranges(None) == [[0.0, 1.0]]

    def test_a_full_span_with_no_holes_ignores_nothing(self):
        assert mpds.ignored_comp_ranges([0.0, 1.0], {"holes": []}) == []

    def test_touching_intervals_merge(self):
        got = mpds.ignored_comp_ranges([0.2, 0.9], {"holes": [[0.1, 0.3]]})
        assert got == [[0.0, 0.3], [0.9, 1.0]]


class TestTheNameThatMustNotCollide:
    def test_a_hole_lands_in_the_undigitized_ranges_and_nowhere_else(self, monkeypatch):
        bl = _from_cache(monkeypatch, TWO_REGIONS)
        assert bl.undigitized_comp_ranges == [pytest.approx([0.10, 0.35], abs=1e-9)]
        assert bl.unsupported_comp_ranges == []
        assert bl.liquidus_holes == [pytest.approx([0.10, 0.35], abs=1e-9)]


class TestTheGateReadsIgnoredRange:
    def test_a_contiguous_curve_is_admitted(self, monkeypatch):
        bl = _from_cache(monkeypatch, CONTIGUOUS)
        assert bl.undigitized_comp_ranges == []
        assert bl.init_error is False

    def test_a_hole_counts_against_the_floor(self, monkeypatch):
        """Endpoint span alone is the whole axis here; only the hole rejects it."""
        from gliquid.binary import SKIP_NARROW_SPAN

        bl = _from_cache(monkeypatch, TWO_REGIONS, assessed_crange_floor=0.8)
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_NARROW_SPAN

    def test_the_floor_is_a_per_call_override(self, monkeypatch):
        bl = _from_cache(monkeypatch, TWO_REGIONS, assessed_crange_floor=0.5)
        assert bl.init_error is False

    def test_the_removed_kwarg_names_its_replacement(self, monkeypatch):
        with pytest.raises(TypeError, match="use assessed_crange_floor"):
            _from_cache(monkeypatch, TWO_REGIONS, comp_range_fit_lim=0.5)


class TestTheConfigDefaults:
    def test_the_shipped_floors(self):
        assert config.assessed_crange_floor == 0.7
        assert config.coverage_min_supported_total == 0.30


class TestTheFloorRisesWithIgnoredRange:
    """``min_supported_total + ignored_floor_slope * ignored_total`` is the coverage floor."""

    # 0.345 supported over [0, 0.76]; the rest of the axis is undigitized.
    UNSUPPORTED, IGNORED = [[0.05, 0.465]], [[0.76, 1.0]]

    def _report(self, slope):
        return coverage_report(self.UNSUPPORTED, 0.30, 0.0, self.IGNORED, slope=slope)

    def test_a_zero_slope_keeps_the_constant_floor(self):
        assert self._report(0.0).is_insufficient()[0] is False

    def test_the_slope_raises_the_floor_by_the_ignored_width(self):
        insufficient, reason = self._report(0.5).is_insufficient()
        assert insufficient is True
        assert "0.420 floor" in reason

    def test_no_ignored_range_leaves_the_floor_unmoved(self):
        report = coverage_report(self.UNSUPPORTED, 0.30, 0.0, [], slope=0.5)
        assert report.is_insufficient()[0] is False

    def test_the_shipped_slope_reaches_an_assessed_report(self):
        assert config.coverage_ignored_floor_slope == 0.5
        assert config.coverage_settings()["ignored_floor_slope"] == 0.5
        assert config.coverage_settings({"ignored_floor_slope": 0})["ignored_floor_slope"] == 0.0

    def test_a_negative_slope_is_refused(self):
        with pytest.raises(ValueError, match="ignored_floor_slope"):
            config.set_coverage_thresholds(ignored_floor_slope=-0.1)
