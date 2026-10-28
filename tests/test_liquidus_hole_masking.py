"""Undigitized liquidus holes are masked out of the deviation metrics.

``extract_digitized_liquidus`` does not fabricate a liquidus across the hole between two
disjoint 'L' shapes, but the fit is still GRADED against that hole: the mesh in
``calculate_deviation_metrics`` spans the full endpoint range and picks each reference
temperature by nearest neighbour, so a mesh point inside a hole is compared against
whichever region edge happens to be closer — a flat step, invented by the metric rather
than by the extractor. This module pins the hole out of the number.

The pin is a POISON POINT: a wildly-wrong temperature planted inside the hole. If any mesh
point in the hole still contributed, nearest neighbour would pick the poison up and the
metrics would explode. Each test that asserts the poison is inert is paired with a control
that clears ``holes`` and shows the same poison does reach the metrics.

``TestReversedFrame`` covers the frame trap: ``liq_coverage`` is computed on the RAW json,
whose component frame may be reversed relative to the construction frame, and hole POSITIONS
are not mirror-invariant.
"""

import numpy as np
import pytest

import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid

from ._mpds_builders import from_record, liquidus_record
from ._mpds_builders import liquid_shape as _shape


def _make_json(regions, elements=("Hf", "Zr"), temp=(0, 2600)):
    return liquidus_record([_shape(r) for r in regions], elements, temp)


# One liquid field as two 'L' shapes around an undigitized hole (10-35 at.%), asymmetric and
# discontinuous across it, so a frame flip or a flat step is unmistakable. It still passes
# from_cache's interior-coverage gate and reaches the fit.
LEFT_REGION = [[x, 2233 - 20 * x] for x in range(0, 11, 2)]
RIGHT_REGION = [[x, 1500 + (1855 - 1500) * (x - 35) / 65] for x in range(35, 101, 5)]
HOLE = [0.10, 0.35]
IN_HOLE = 0.225  # hole midpoint, in the alphabetical (Hf-Zr) frame
MIRRORED_HOLE = [0.65, 0.90]
IN_MIRRORED_HOLE = 0.775

# The same points as one contiguous 'L' shape: one region, so the 25 at.% stretch is
# sparse sampling and gets filled. Nothing is masked. The A/B for "no hole, no change".
CONTIGUOUS = LEFT_REGION + RIGHT_REGION


def _holed_json():
    return _make_json([LEFT_REGION, RIGHT_REGION])


def _load(monkeypatch, mpds_json, spec="Hf-Zr", **kwargs):
    bl = from_record(monkeypatch, mpds_json, spec, **kwargs)
    bl.update_phase_points()
    return bl


def _poison(bl, x, temp=1e6):
    """Plant an absurd 'digitized' temperature at composition ``x``."""
    bl.digitized_liq = sorted(bl.digitized_liq + [[x, temp]])
    return bl


class TestCoverageReportsHoles:
    """``liquidus_coverage`` carries the hole intervals to whoever needs them."""

    def test_disjoint_regions_report_their_hole(self):
        cov = mpds.liquidus_coverage(_holed_json())
        assert cov["holes"] == [pytest.approx(HOLE, abs=1e-9)]

    def test_a_contiguous_field_has_no_holes(self):
        assert mpds.liquidus_coverage(_make_json([CONTIGUOUS]))["holes"] == []

    def test_three_regions_report_two_holes(self):
        cov = mpds.liquidus_coverage(
            _make_json(
                [
                    [[x, 2233 - 20 * x] for x in range(0, 11, 2)],
                    [[x, 1600] for x in range(30, 51, 2)],
                    [[x, 1500 + 7 * (x - 70)] for x in range(70, 101, 2)],
                ]
            )
        )
        assert len(cov["holes"]) == 2
        assert cov["holes"][0] == pytest.approx([0.10, 0.30], abs=1e-9)
        assert cov["holes"][1] == pytest.approx([0.50, 0.70], abs=1e-9)

    def test_a_gap_the_fill_would_bridge_is_not_a_hole(self):
        """The threshold is the fill trigger, not 'the regions were drawn separately': two
        regions 4 at.% apart leave a seam narrower than ``_FILL_GAP_X``, which is not a hole."""
        assert 0.04 < mpds._FILL_GAP_X
        near = _make_json(
            [
                [[x, 2233 - 20 * x] for x in range(0, 11, 2)],
                [[x, 1900 + 2 * x] for x in range(14, 101, 2)],
            ]
        )
        assert mpds.liquidus_coverage(near)["holes"] == []
        curve, _ = mpds.extract_digitized_liquidus(near)
        xs = [pt[0] for pt in curve]
        assert max(b - a for a, b in zip(xs, xs[1:])) == pytest.approx(0.04)

    def test_holes_reach_the_binaryliquid(self, monkeypatch):
        bl = _load(monkeypatch, _holed_json())
        assert bl.init_error is False, "the fixture must pass the coverage gate"
        assert bl.liquidus_holes == [pytest.approx(HOLE, abs=1e-9)]

    def test_missing_coverage_masks_nothing(self):
        """A BinaryLiquid built without a measurement must not invent one."""
        bl = BinaryLiquid.__new__(BinaryLiquid)
        bl.liq_coverage = None
        assert bl.liquidus_holes == []
        bl.liq_coverage = {"max_gap": 0.1}
        assert bl.liquidus_holes == []


class TestHolesAreMaskedOutOfTheMetrics:
    """No mesh point inside an undigitized hole may reach MAE / RMSE / MAPE / RMSPE."""

    def test_poison_inside_the_hole_is_inert(self, monkeypatch):
        bl = _load(monkeypatch, _holed_json())
        clean = bl.calculate_deviation_metrics()
        assert all(np.isfinite(clean))
        assert bl.calculate_deviation_metrics() == clean  # sanity: deterministic
        _poison(bl, IN_HOLE)
        assert bl.calculate_deviation_metrics() == clean

    def test_control_the_same_poison_lands_when_the_hole_is_not_masked(self, monkeypatch):
        """Without this the test above would pass vacuously on a mesh that never enters
        the hole. Clearing ``holes`` restores the pre-02b behaviour."""
        bl = _load(monkeypatch, _holed_json())
        bl.liq_coverage = dict(bl.liq_coverage, holes=[])
        clean = bl.calculate_deviation_metrics()
        _poison(bl, IN_HOLE)
        assert bl.calculate_deviation_metrics()[0] > 100 * clean[0]

    def test_masking_changes_the_metrics_it_is_supposed_to_change(self, monkeypatch):
        """Unmasked, the flat step across the hole is scored: dropping those mesh points
        moves MAE."""
        bl = _load(monkeypatch, _holed_json())
        masked = bl.calculate_deviation_metrics()
        bl.liq_coverage = dict(bl.liq_coverage, holes=[])
        assert bl.calculate_deviation_metrics()[0] != pytest.approx(masked[0], abs=1e-9)

    def test_holes_are_masked_even_in_score_mode(self, monkeypatch):
        """``unsupported_range_mode='score'`` asks for the metric without fit_parameters'
        judgement calls. A hole is not a judgement call — there is no data there either way."""
        bl = _load(monkeypatch, _holed_json())
        clean = bl.calculate_deviation_metrics(unsupported_range_mode="score")
        _poison(bl, IN_HOLE)
        assert bl.calculate_deviation_metrics(unsupported_range_mode="score") == clean

    def test_holes_and_unsupported_ranges_mask_together(self, monkeypatch):
        """Both sources go through one DROP mask under ``'ignore'``, so neither cancels the
        other. The mode is explicit because only ``'ignore'`` drops the second source; under
        ``'score'`` and ``'cap'`` the unsupported range stays in the mesh (``'cap'`` scores
        it one-sidedly: ``tests/test_capped_unsupported_objective.py``)."""
        bl = _load(monkeypatch, _holed_json())
        bl.unsupported_comp_ranges = [[0.60, 0.75]]
        clean = bl.calculate_deviation_metrics(unsupported_range_mode="ignore")
        assert all(np.isfinite(clean))
        _poison(bl, IN_HOLE)
        _poison(bl, 0.675)
        assert bl.calculate_deviation_metrics(unsupported_range_mode="ignore") == clean

    def test_data_outside_the_hole_still_counts(self, monkeypatch):
        """The mask must be the hole and nothing but the hole."""
        bl = _load(monkeypatch, _holed_json())
        clean = bl.calculate_deviation_metrics()
        _poison(bl, 0.50)  # inside the digitized right-hand region
        assert bl.calculate_deviation_metrics()[0] > clean[0]

    def test_an_over_masked_system_reports_inf_not_a_flattering_score(self, monkeypatch):
        """Bi-Si class. When masking leaves under ten mesh points the metric refuses to
        report, exactly as it already does when unsupported_comp_ranges over-masks."""
        bl = _load(monkeypatch, _holed_json())
        bl.liq_coverage = dict(bl.liq_coverage, holes=[[0.08, 0.95]])
        assert bl.calculate_deviation_metrics() == (float("inf"),) * 4

    def test_a_fully_masked_system_reports_inf_even_when_sparse_data_is_allowed(self, monkeypatch):
        """``allow_sparse_data`` waives the ten-point floor; it must not waive an empty
        mesh, whose means would be nan and would read downstream as a number."""
        bl = _load(monkeypatch, _holed_json())
        bl.liq_coverage = dict(bl.liq_coverage, holes=[[0.0, 1.0]])
        assert bl.calculate_deviation_metrics(allow_sparse_data=True) == (float("inf"),) * 4


class TestNoHoleNoChange:
    """A system without a hole takes the same path with or without the hole machinery."""

    def test_contiguous_field_masks_nothing(self, monkeypatch):
        bl = _load(monkeypatch, _make_json([CONTIGUOUS]))
        assert bl.liquidus_holes == []

    def test_metrics_are_identical_with_and_without_the_hole_machinery(self, monkeypatch):
        """The same object, once with ``liq_coverage`` present (holes empty) and once with
        it stripped entirely: byte-identical, so an unholed system cannot move."""
        bl = _load(monkeypatch, _make_json([CONTIGUOUS]))
        with_cov = bl.calculate_deviation_metrics()
        bl.liq_coverage = None
        assert bl.calculate_deviation_metrics() == with_cov
        assert all(np.isfinite(with_cov))

    @pytest.mark.parametrize("system", ["Hf-Zr"])
    def test_real_single_l_system_has_no_holes(self, system):
        bl = BinaryLiquid.from_cache(system, pd_ind=0)
        assert bl.liq_coverage["holes"] == []
        assert bl.liquidus_holes == []


class TestReversedFrame:
    """The frame trap: ``liq_coverage`` is computed on the raw json, not the fit frame. The
    hole sits at 0.10-0.35 of the alphabetical Hf-Zr frame, so in the Zr-Hf frame it must sit
    at 0.65-0.90.
    """

    def test_mirror_liquidus_coverage_moves_positions_and_leaves_scalars(self):
        cov = mpds.liquidus_coverage(_holed_json())
        flipped = mpds.mirror_liquidus_coverage(cov)
        assert flipped["holes"] == [pytest.approx(MIRRORED_HOLE, abs=1e-9)]
        assert flipped["x_min"] == pytest.approx(1 - cov["x_max"])
        assert flipped["x_max"] == pytest.approx(1 - cov["x_min"])
        for key in ("span", "n_points", "max_gap", "covered_fraction"):
            assert flipped[key] == cov[key]

    def test_mirror_is_an_involution(self):
        cov = mpds.liquidus_coverage(_holed_json())
        back = mpds.mirror_liquidus_coverage(mpds.mirror_liquidus_coverage(cov))
        assert back["holes"] == [pytest.approx(HOLE, abs=1e-9)]
        assert back["x_min"] == pytest.approx(cov["x_min"])

    def test_mirror_orders_multiple_holes_ascending(self):
        flipped = mpds.mirror_liquidus_coverage({"holes": [[0.1, 0.2], [0.6, 0.7]]})
        assert flipped["holes"] == [pytest.approx([0.3, 0.4]), pytest.approx([0.8, 0.9])]

    def test_from_cache_stores_the_hole_in_the_construction_frame(self, monkeypatch):
        bl = _load(monkeypatch, _holed_json(), spec="Zr-Hf")
        assert bl.components == ["Zr", "Hf"]
        assert bl.liquidus_holes == [pytest.approx(MIRRORED_HOLE, abs=1e-9)]

    def test_with_component_order_reframes_the_hole(self, monkeypatch):
        bl = _load(monkeypatch, _holed_json())
        clone = bl.with_component_order(["Zr", "Hf"])
        assert clone.liquidus_holes == [pytest.approx(MIRRORED_HOLE, abs=1e-9)]
        back = clone.with_component_order(["Hf", "Zr"])
        assert back.liquidus_holes == [pytest.approx(HOLE, abs=1e-9)]

    def test_the_hole_masks_the_end_of_the_diagram_that_has_no_data(self, monkeypatch):
        """Poison the TRUE hole of the reversed frame. An unmirrored mask sits at the
        other end, so the poison lands and the metrics explode."""
        bl = _load(monkeypatch, _holed_json(), spec="Zr-Hf")
        clean = bl.calculate_deviation_metrics()
        assert all(np.isfinite(clean))
        _poison(bl, IN_MIRRORED_HOLE)
        assert bl.calculate_deviation_metrics() == clean

    def test_the_metric_is_the_same_number_in_either_frame(self, monkeypatch):
        """The sharpest statement of the trap: a system and its mirror image are the same
        physical system, so they must score the same. They do not if the hole does not
        travel with the liquidus."""
        params = [-30000.0, -5.0, 0.0, 0.0]
        kw = dict(params=params, param_format="comb-exp")
        forward = _load(monkeypatch, _holed_json(), spec="Hf-Zr", **kw)
        reverse = _load(monkeypatch, _holed_json(), spec="Zr-Hf", **kw)
        f_mae, _, f_mape, _ = forward.calculate_deviation_metrics()
        r_mae, _, r_mape, _ = reverse.calculate_deviation_metrics()
        assert np.isfinite(f_mae)
        assert r_mae == pytest.approx(f_mae, rel=1e-6)
        assert r_mape == pytest.approx(f_mape, rel=1e-6)


class TestRealDisjointSystems:
    """Hole POSITIONS in cached digitizations, read straight off disk rather than through
    ``from_cache``. Only Hf-Zr ships with the package; the rest need the maintainer's
    per-system corpus store.
    """

    @staticmethod
    def _cached(system):
        import json
        from pathlib import Path

        import gliquid.api as api
        from gliquid.cache import CacheKey

        name = f"{system}_MPDS_PD_0.json"
        candidates = [api.resolve_cache_path(CacheKey(system, "mpds", "0"))]
        root = config.project_root
        parents = Path(root).resolve().parents if root is not None else ()
        candidates += [parent / "matrix_data" / system / name for parent in parents]
        # resolve_cache_path answers None for a store with no filesystem paths — a real
        # answer, so it is filtered rather than assumed away.
        path = next((p for p in candidates if p is not None and p.exists()), None)
        if path is None:
            pytest.skip(f"{system} not in the local MPDS cache")
        return json.loads(path.read_text())

    @pytest.mark.needs_cache
    @pytest.mark.parametrize(
        "system, hole",
        [
            ("Ir-Th", [0.2856, 0.3904]),
            ("Pd-Se", [0.5646, 0.9537]),
            ("Ca-Nd", [0.0452, 0.4794]),
        ],
    )
    def test_cached_system_carries_its_hole(self, system, hole):
        cov = mpds.liquidus_coverage(self._cached(system))
        assert cov["holes"] == [pytest.approx(hole, abs=1e-3)]

    @pytest.mark.needs_cache
    def test_a_real_hole_mirrors_to_the_other_end(self):
        cov = mpds.liquidus_coverage(self._cached("Pd-Se"))
        ((f_lo, f_hi),) = cov["holes"]
        ((r_lo, r_hi),) = mpds.mirror_liquidus_coverage(cov)["holes"]
        assert r_lo == pytest.approx(1 - f_hi) and r_hi == pytest.approx(1 - f_lo)
        # Pd-Se's hole is far from centre, so the mirror is not a no-op in disguise.
        assert abs(r_lo - f_lo) > 0.5


class TestGateIsUnaffected:
    """The interior-coverage gate reads mirror-invariant scalars and must not move."""

    def test_thresholds_still_reject_the_bi_si_class(self):
        gappy = _make_json(
            [
                [[x, 2233 - 20 * x] for x in range(0, 8, 2)],
                [[x, 1500 + 20 * (x - 92)] for x in range(92, 101, 2)],
            ]
        )
        cov = mpds.liquidus_coverage(gappy)
        assert cov["max_gap"] > config.liquidus_max_gap
        assert cov["covered_fraction"] < config.liquidus_min_coverage
        assert cov["holes"] == [pytest.approx([0.06, 0.92], abs=1e-9)]
