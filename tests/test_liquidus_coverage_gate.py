"""Liquidus interior-coverage gate: span-complete but interior-sparse data must not fit.

``mpds.extract_digitized_liquidus`` linearly fills every composition gap wider than 0.06
before anyone downstream can measure it, so a liquidus digitized only near the two pure
ends (Bi-Si class: liquid wedges at 0-7 and 92-100 at.%, nothing in between) reaches
``BinaryLiquid.from_cache`` as a full-span curve whose interior is ~85% fabricated — and
the endpoint-span gate (``comp_range_fit_lim``) admits it. ``from_cache`` therefore also
measures the PRE-fill stitched curve (``mpds.liquidus_coverage``) and flags ``init_error``
when its widest gap exceeds ``config.liquidus_max_gap`` or less than
``config.liquidus_min_coverage`` of the span is locally sampled.

Thresholds (0.45 / 0.50 at gap_tol 0.10) were calibrated on the 1433 span-admitted cached
systems: they newly reject only the 9 whose interiors are ~half fabricated or worse, and
leave ordinarily-sampled curves (typical spacing 0.01-0.03, corpus p95 max_gap 0.141)
untouched — see dev/scripts/_generated/scan_liquidus_coverage.py.
"""

import pytest

import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid


def _svgpath(points):
    """[[x_pct, T_celsius], ...] -> MPDS-style svgpath string."""
    return "M " + " L ".join(f"{x},{t}" for x, t in points)


def _make_json(points, elements=("Hf", "Zr"), temp=(0, 2600)):
    return {
        "reference": {"entry": "synthetic-test-json"},
        "chemical_elements": list(elements),
        "comp_range": [0, 100],
        "temp": list(temp),
        "labels": [],
        "shapes": [{"label": "L", "nphases": 1, "svgpath": _svgpath(points)}],
    }


# Bi-Si class: wedges near the pure ends, an ~85 at.% hole between them.
GAPPY_POINTS = [
    [0, 2233],
    [2, 2100],
    [4, 1950],
    [5.5, 1800],
    [7, 1700],
    [92, 1500],
    [94, 1600],
    [96, 1700],
    [98, 1780],
    [100, 1855],
]
# Ordinary well-sampled V-shaped liquidus at 2 at.% spacing.
DENSE_POINTS = [[x, 2233 - (2233 - 1400) * x / 50] for x in range(0, 50, 2)] + [
    [x, 1400 + (1855 - 1400) * (x - 50) / 50] for x in range(50, 101, 2)
]


class TestLiquidusCoverageMetrics:
    def test_gappy_interior_is_measured_pre_fill(self):
        cov = mpds.liquidus_coverage(_make_json(GAPPY_POINTS))
        assert cov["span"] == pytest.approx(1.0)
        assert cov["max_gap"] == pytest.approx(0.85)
        assert cov["covered_fraction"] == pytest.approx(0.15)
        assert cov["n_points"] == len(GAPPY_POINTS)

    def test_well_sampled_curve_is_fully_covered(self):
        cov = mpds.liquidus_coverage(_make_json(DENSE_POINTS))
        assert cov["max_gap"] == pytest.approx(0.02)
        assert cov["covered_fraction"] == pytest.approx(1.0)

    def test_fill_erases_the_gap_from_the_extracted_curve(self):
        """Why the endpoint-span gate cannot catch this class: post-fill, the hole is
        indistinguishable from data — full span, interior populated by interpolation."""
        extracted, _ = mpds.extract_digitized_liquidus(_make_json(GAPPY_POINTS))
        xs = [pt[0] for pt in extracted]
        assert max(xs) - min(xs) == pytest.approx(1.0)
        assert any(0.10 < x < 0.90 for x in xs), "fill fabricates interior points"
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        assert max(gaps) < 0.07, "post-fill the 0.85 hole is gone"

    def test_no_liquidus_returns_none(self):
        assert mpds.liquidus_coverage({"reference": None}) is None


class TestFromCacheGate:
    """End-to-end through BinaryLiquid.from_cache with the real extraction pipeline;
    only the cache/API load is stubbed so the jsons are synthetic."""

    def _from_cache(self, monkeypatch, points, **kwargs):
        mpds_json = _make_json(points)

        def fake_load(input, pd_ind=None):
            return mpds_json, mpds.extract_digitized_liquidus(mpds_json)

        monkeypatch.setattr(mpds, "load_mpds_data", fake_load)
        return BinaryLiquid.from_cache("Hf-Zr", **kwargs)

    def test_gappy_liquidus_is_rejected(self, monkeypatch, caplog):
        bl = self._from_cache(monkeypatch, GAPPY_POINTS)
        assert bl.init_error is True
        assert bl.liq_coverage["max_gap"] == pytest.approx(0.85)
        assert "interior-sparse" in caplog.text

    def test_well_sampled_liquidus_is_admitted(self, monkeypatch, caplog):
        bl = self._from_cache(monkeypatch, DENSE_POINTS)
        assert bl.init_error is False
        assert "interior-sparse" not in caplog.text

    def test_per_call_override_relaxes_the_gate(self, monkeypatch):
        bl = self._from_cache(monkeypatch, GAPPY_POINTS, liq_max_gap=0.90, liq_min_coverage=0.10)
        assert bl.init_error is False

    def test_rejection_survives_deviation_metrics(self, monkeypatch):
        """The flagged system reports inf deviation instead of scoring a fit against
        fabricated interior points."""
        bl = self._from_cache(monkeypatch, GAPPY_POINTS)
        assert bl.calculate_deviation_metrics() == (float("inf"),) * 4


class TestRealCachedSystem:
    def test_hf_zr_pd0_is_unaffected(self):
        """A real, ordinarily-sampled cached system passes the gate at from_cache time
        (its fit/skip verdict then belongs to the solid-coverage gate, not this one)."""
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        assert bl.init_error is False
        assert bl.liq_coverage is not None
        assert bl.liq_coverage["max_gap"] <= config.liquidus_max_gap
        assert bl.liq_coverage["covered_fraction"] >= config.liquidus_min_coverage


class TestConfigThresholds:
    def test_defaults_are_the_calibrated_values(self):
        assert config.liquidus_max_gap == pytest.approx(0.45)
        assert config.liquidus_min_coverage == pytest.approx(0.50)
        assert config.liquidus_gap_tol == pytest.approx(0.10)

    def test_setter_validates_and_is_partial(self):
        original = (config.liquidus_max_gap, config.liquidus_min_coverage, config.liquidus_gap_tol)
        try:
            with pytest.raises(ValueError):
                config.set_liquidus_coverage_thresholds(max_gap=0.0)
            with pytest.raises(ValueError):
                config.set_liquidus_coverage_thresholds(min_coverage=1.5)
            config.set_liquidus_coverage_thresholds(max_gap=0.30)
            assert config.liquidus_max_gap == pytest.approx(0.30)
            assert config.liquidus_min_coverage == pytest.approx(original[1])
        finally:
            config.set_liquidus_coverage_thresholds(
                max_gap=original[0], min_coverage=original[1], gap_tol=original[2]
            )

    def test_config_threshold_drives_the_gate(self, monkeypatch):
        """Tightening config alone flips a moderately-gappy curve to rejected."""
        moderate = [[x, 2233 - 8 * x] for x in range(0, 40, 2)] + [
            [x, 1855 - 6 * (100 - x)] for x in range(60, 101, 2)
        ]  # one 0.2 hole
        original = config.liquidus_max_gap
        mpds_json = _make_json(moderate)
        monkeypatch.setattr(
            mpds,
            "load_mpds_data",
            lambda input, pd_ind=None: (mpds_json, mpds.extract_digitized_liquidus(mpds_json)),
        )
        try:
            assert BinaryLiquid.from_cache("Hf-Zr").init_error is False
            config.set_liquidus_coverage_thresholds(max_gap=0.15)
            assert BinaryLiquid.from_cache("Hf-Zr").init_error is True
        finally:
            config.set_liquidus_coverage_thresholds(max_gap=original)
