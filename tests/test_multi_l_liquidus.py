"""Multi-shape liquid fields: the liquidus is the lower envelope of ALL 'L' shapes.

MPDS draws the liquid of a monotectic / immiscible / terminally-truncated diagram as
SEVERAL disjoint one-phase 'L' shapes. ``mpds.extract_digitized_liquidus`` used to read
only the first of them (``next(... if b.get('label') == 'L')``) and so reported whichever
sliver happened to come first in the shape list: Ag-Fe spanned 0.006 of the composition
range instead of 1.0, Bi-Si 0.075, Er-Ta 0.169.

Two properties are pinned here. The union must be complete -- both wedges reach the
extraction, and nothing digitized is thrown away. And it must stay honest -- the hole
between disjoint wedges is not digitized data, so no point is invented in it (not by the
envelope and not by the fill step), ``is_partial`` says the curve does not cover the range
continuously, and ``liquidus_coverage`` still measures the hole (that is what makes the
Bi-Si class fail ``from_cache``'s interior-coverage gate now that it arrives span-complete).

``TestRegionConfinedFill`` pins the distinction the fill step has to make: densifying a
sparse patch INSIDE one liquid region is interpolation between two points on the same
curve and stays; bridging the hole BETWEEN two regions is fabrication and is gone.
"""

import json
from pathlib import Path

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.cache import CacheKey
from gliquid.plotting.binary_tx import _ASSESSED_GAP_PCT, _assessed_liquidus_segments


def _load_cached(system):
    """A cached ``<sys>_MPDS_PD_0.json``, or skip when it is not on this machine.

    Only Hf-Zr ships with the package; the rest of the corpus lives in the workspace's
    nested ``matrix_data`` store next to the checkout, so those pins are opportunistic.
    """
    name = f"{system}_MPDS_PD_0.json"
    candidates = [api.resolve_cache_path(CacheKey(system, "mpds", "0"))]
    candidates += [
        parent / "matrix_data" / system / name
        for parent in Path(config.project_root).resolve().parents
    ]
    # resolve_cache_path answers None for a store with no filesystem paths — a real answer,
    # so it is filtered rather than assumed away.
    path = next((p for p in candidates if p is not None and p.exists()), None)
    if path is None:
        pytest.skip(f"{system} not in the local MPDS cache")
    return json.loads(path.read_text())


def _svgpath(points):
    """[[x_pct, T_celsius], ...] -> MPDS-style svgpath string."""
    return "M " + " L ".join(f"{x},{t}" for x, t in points)


def _shape(points, label="L", kind="phase", **extra):
    return {
        "label": label,
        "nphases": 1,
        "is_solid": False,
        "kind": kind,
        "svgpath": _svgpath(points),
        **extra,
    }


def _make_json(shapes, elements=("Hf", "Zr"), temp=(0, 2600)):
    return {
        "reference": {"entry": "synthetic-test-json"},
        "chemical_elements": list(elements),
        "comp_range": [0, 100],
        "temp": list(temp),
        "labels": [],
        "shapes": list(shapes),
    }


# Two terminal liquid wedges with an 84 at.% hole between them (Bi-Si / Ag-Fe class).
LEFT_WEDGE = [[0, 2233], [2, 2150], [4, 2080], [6, 2020], [8, 1980]]
RIGHT_WEDGE = [[92, 1600], [94, 1660], [96, 1730], [98, 1800], [100, 1855]]
# One continuous liquid field, drawn as a single 'L' shape closed along the frame top.
SINGLE_L = [
    [0, 2233],
    [10, 2050],
    [25, 1800],
    [40, 1620],
    [55, 1560],
    [70, 1640],
    [85, 1760],
    [100, 1855],
    [100, 2600],
    [0, 2600],
]

# Sparsely digitized terminal wedges: 10 at.% between consecutive points (wider than the
# 6 at.% fill trigger) INSIDE each wedge, and a 40 at.% hole BETWEEN them. Splitting the
# same eight points one way or the other is the whole experiment in TestRegionConfinedFill.
SPARSE_LEFT = [[0, 2233], [10, 2100], [20, 2000], [30, 1900]]
SPARSE_RIGHT = [[70, 1600], [80, 1700], [90, 1780], [100, 1855]]


def _gaps(liq):
    xs = [pt[0] for pt in liq]
    return [b - a for a, b in zip(xs, xs[1:])]


class TestDisjointWedges:
    """Two disjoint 'L' shapes -> one curve spanning both, with the hole preserved."""

    def test_extraction_spans_both_wedges(self):
        liq, _ = mpds.extract_digitized_liquidus(
            _make_json([_shape(LEFT_WEDGE), _shape(RIGHT_WEDGE)])
        )
        xs = [pt[0] for pt in liq]
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(1.0)
        assert max(xs) - min(xs) == pytest.approx(1.0)
        # The digitized endpoints of BOTH wedges survive, at their own temperatures.
        assert [0.0, 2233 + 273.15] in liq
        assert [1.0, 1855 + 273.15] in liq
        assert [0.08, 1980 + 273.15] in liq
        assert [0.92, 1600 + 273.15] in liq

    def test_first_shape_alone_would_be_a_sliver(self):
        """Guard on the defect itself: either wedge on its own spans <= 0.08."""
        for wedge in (LEFT_WEDGE, RIGHT_WEDGE):
            liq, is_partial = mpds.extract_digitized_liquidus(_make_json([_shape(wedge)]))
            xs = [pt[0] for pt in liq]
            assert max(xs) - min(xs) <= 0.08
            assert is_partial is True

    def test_nothing_is_invented_in_the_hole(self):
        """Pre-fill, the union is exactly the 10 digitized points -- the 0.84 hole is
        still a hole, and ``liquidus_coverage`` can therefore still see it."""
        cov = mpds.liquidus_coverage(_make_json([_shape(LEFT_WEDGE), _shape(RIGHT_WEDGE)]))
        assert cov["n_points"] == len(LEFT_WEDGE) + len(RIGHT_WEDGE)
        assert cov["span"] == pytest.approx(1.0)
        assert cov["max_gap"] == pytest.approx(0.84)
        assert cov["covered_fraction"] == pytest.approx(0.16)

    def test_is_partial_reflects_the_gap_not_the_endpoints(self):
        """The union reaches both pure ends, so the endpoint test alone would say
        'complete'; the curve is still partial because the branches do not join up."""
        liq, is_partial = mpds.extract_digitized_liquidus(
            _make_json([_shape(LEFT_WEDGE), _shape(RIGHT_WEDGE)])
        )
        assert liq[0][0] == pytest.approx(0.0) and liq[-1][0] == pytest.approx(1.0)
        assert is_partial is True

    def test_shape_order_does_not_matter(self):
        forward, p1 = mpds.extract_digitized_liquidus(
            _make_json([_shape(LEFT_WEDGE), _shape(RIGHT_WEDGE)])
        )
        reverse, p2 = mpds.extract_digitized_liquidus(
            _make_json([_shape(RIGHT_WEDGE), _shape(LEFT_WEDGE)])
        )
        assert forward == reverse
        assert p1 == p2 is True

    def test_gappy_union_still_fails_the_interior_coverage_gate(self):
        """The plumbing the fix must not break: a span-complete but half-fabricated
        curve is exactly what ``liquidus_coverage`` exists to flag."""
        cov = mpds.liquidus_coverage(_make_json([_shape(LEFT_WEDGE), _shape(RIGHT_WEDGE)]))
        assert cov["max_gap"] > config.liquidus_max_gap
        assert cov["covered_fraction"] < config.liquidus_min_coverage

    def test_a_shape_too_small_to_be_a_curve_is_skipped_not_fatal(self):
        """Co-Pb class: the first 'L' shape has under three interior points, so the old
        extractor returned nothing at all. The other shape must still be extracted."""
        liq, is_partial = mpds.extract_digitized_liquidus(
            _make_json([_shape([[0, 2233], [1, 2200]]), _shape(RIGHT_WEDGE)])
        )
        assert liq is not None
        assert min(pt[0] for pt in liq) == pytest.approx(0.92)
        assert is_partial is True

    def test_no_usable_shape_reports_insufficient_data(self):
        liq, is_partial = mpds.extract_digitized_liquidus(
            _make_json([_shape([[0, 2233], [1, 2200]])])
        )
        assert liq is None
        assert is_partial is True


class TestRegionConfinedFill:
    """Densification is legitimate WITHIN a liquid region and fabrication BETWEEN two.

    Every test here uses the same eight digitized points. Drawn as one 'L' shape they are
    one region and the 40 at.% gap is sparse sampling, so it is filled; drawn as two 'L'
    shapes they are two regions and the same gap is a hole no shape covers, so it is not.
    Nothing but the region structure differs, which is why the fill step cannot decide
    from the flat point list alone.
    """

    ONE_SHAPE = SPARSE_LEFT + SPARSE_RIGHT

    def test_wide_gap_inside_one_branch_is_still_filled(self):
        liq, is_partial = mpds.extract_digitized_liquidus(_make_json([_shape(self.ONE_SHAPE)]))
        assert max(_gaps(liq)) <= mpds._FILL_GAP_X
        assert len(liq) > len(self.ONE_SHAPE)
        assert is_partial is False
        # The 0.30-0.70 stretch is interpolated, so it carries points.
        assert any(0.30 < pt[0] < 0.70 for pt in liq)

    def test_gap_between_two_branches_is_not_filled(self):
        liq, is_partial = mpds.extract_digitized_liquidus(
            _make_json([_shape(SPARSE_LEFT), _shape(SPARSE_RIGHT)])
        )
        assert max(_gaps(liq)) == pytest.approx(0.40)
        assert is_partial is True
        # Nothing at all is invented across the hole.
        assert not any(0.30 < pt[0] < 0.70 for pt in liq)

    def test_the_two_wedges_are_each_still_densified(self):
        """The fix confines the fill, it does not disable it: the 10 at.% sampling inside
        each wedge is still interpolated up to the fill step."""
        liq, _ = mpds.extract_digitized_liquidus(
            _make_json([_shape(SPARSE_LEFT), _shape(SPARSE_RIGHT)])
        )
        in_hole = [g for g in _gaps(liq) if g > mpds._FILL_GAP_X]
        assert in_hole == [pytest.approx(0.40)], "only the inter-region hole may stay wide"
        assert len(liq) > len(SPARSE_LEFT) + len(SPARSE_RIGHT)

    def test_endpoints_of_both_wedges_survive_unmoved(self):
        liq, _ = mpds.extract_digitized_liquidus(
            _make_json([_shape(SPARSE_LEFT), _shape(SPARSE_RIGHT)])
        )
        assert [0.30, 1900 + 273.15] in liq
        assert [0.70, 1600 + 273.15] in liq

    def test_span_is_unchanged_by_confining_the_fill(self):
        """Only the interior moves: the endpoint span the ``comp_range_fit_lim`` gate reads
        is set by the outermost digitized points, which the fill never touched."""
        liq, _ = mpds.extract_digitized_liquidus(
            _make_json([_shape(SPARSE_LEFT), _shape(SPARSE_RIGHT)])
        )
        xs = [pt[0] for pt in liq]
        assert max(xs) - min(xs) == pytest.approx(1.0)

    def test_overlapping_shapes_are_one_region_and_are_filled(self):
        """Two shapes that overlap in composition merge into a single covered interval, so
        the seam between them is interpolation, not a hole."""
        hot = [[0, 2233], [45, 2000], [60, 1990], [100, 1855]]
        cool = [[0, 2233], [40, 1500], [55, 1520], [100, 1855]]
        liq, is_partial = mpds.extract_digitized_liquidus(_make_json([_shape(hot), _shape(cool)]))
        assert max(_gaps(liq)) <= mpds._FILL_GAP_X
        assert is_partial is False


class TestPlotBreak:
    """A plotter that connects consecutive points must not draw across an undigitized hole.

    Removing the fill leaves one digitized point on each side of the hole, so the break has
    to happen in the renderer as well. Both T-x plotters route the assessed liquidus through
    ``_assessed_liquidus_segments`` and emit one trace per segment.
    """

    def test_break_threshold_tracks_the_fill_threshold(self):
        """The renderer's break rule is only correct because the extractor guarantees no
        in-region gap survives wider than ``_FILL_GAP_X``. Pin them together: this module
        cannot import mpds, so the constant is duplicated there as a literal."""
        assert _ASSESSED_GAP_PCT == pytest.approx(mpds._FILL_GAP_X * 100)

    def _plot_points(self, mpds_json):
        liq, _ = mpds.extract_digitized_liquidus(mpds_json)
        return [[pt[0] * 100, pt[1] - 273.15] for pt in liq]

    def test_disjoint_wedges_render_as_two_traces(self):
        pts = self._plot_points(_make_json([_shape(SPARSE_LEFT), _shape(SPARSE_RIGHT)]))
        segments = _assessed_liquidus_segments(pts)
        assert len(segments) == 2
        assert max(p[0] for p in segments[0]) == pytest.approx(30.0)
        assert min(p[0] for p in segments[1]) == pytest.approx(70.0)
        # No segment straddles the hole, so no line can be drawn through it.
        assert sum(len(s) for s in segments) == len(pts)

    def test_a_contiguous_liquidus_stays_one_trace(self):
        pts = self._plot_points(_make_json([_shape(SINGLE_L)]))
        assert len(_assessed_liquidus_segments(pts)) == 1

    def test_same_points_one_region_stay_one_trace(self):
        """The A/B of the fill fix, seen from the renderer."""
        pts = self._plot_points(_make_json([_shape(SPARSE_LEFT + SPARSE_RIGHT)]))
        assert len(_assessed_liquidus_segments(pts)) == 1

    def test_empty_input_is_not_a_trace(self):
        assert _assessed_liquidus_segments([]) == []

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system", ["Ag-Fe", "Er-Ta", "Bi-Si", "Ir-Th"])
    def test_real_disjoint_systems_break(self, system):
        segments = _assessed_liquidus_segments(self._plot_points(_load_cached(system)))
        assert len(segments) >= 2

    # Hf-Zr ships with the package; the other two only resolve from matrix_data.
    @pytest.mark.parametrize(
        "system",
        [
            "Hf-Zr",
            pytest.param("Al-Tl", marks=pytest.mark.needs_cache),
            pytest.param("Ho-Tm", marks=pytest.mark.needs_cache),
        ],
    )
    def test_real_single_l_systems_do_not_break(self, system):
        segments = _assessed_liquidus_segments(self._plot_points(_load_cached(system)))
        assert len(segments) == 1


class TestSingleShapeRegression:
    """One 'L' shape must behave exactly as it did before multi-shape extraction."""

    def test_single_shape_curve_is_complete_and_not_partial(self):
        liq, is_partial = mpds.extract_digitized_liquidus(_make_json([_shape(SINGLE_L)]))
        xs = [pt[0] for pt in liq]
        assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(1.0)
        assert is_partial is False
        # Every digitized (non-frame) vertex is present, unmoved.
        for x, t in SINGLE_L[:8]:
            assert [x / 100, t + 273.15] in liq

    def test_envelope_is_a_strict_no_op_for_one_branch(self):
        """With a single branch the multi-shape path returns that branch itself, so a
        single-'L' diagram cannot be perturbed by the envelope logic."""
        branch = [[0.0, 2506.15], [0.5, 1773.15], [1.0, 2128.15]]
        assert mpds._lower_envelope([branch]) is branch

    @pytest.mark.parametrize(
        "system, span, n_points, is_partial",
        [
            ("Hf-Zr", 1.0, 18, False),  # ships with the package
            pytest.param("Al-Tl", 0.99770, 137, False, marks=pytest.mark.needs_cache),
            pytest.param("Ho-Tm", 1.0, 41, False, marks=pytest.mark.needs_cache),
        ],
    )
    def test_real_single_l_systems_are_byte_stable(self, system, span, n_points, is_partial):
        """Pinned against the values the first-'L'-only extractor produced for cached
        single-'L' diagrams."""
        mpds_json = _load_cached(system)
        assert len(mpds.liquid_shape_paths(mpds_json)) == 1
        liq, partial = mpds.extract_digitized_liquidus(mpds_json)
        xs = [pt[0] for pt in liq]
        assert max(xs) - min(xs) == pytest.approx(span, abs=1e-4)
        assert len(liq) == n_points
        assert partial is is_partial


class TestFrameEdgeStripping:
    """Frame-boundary points close a wedge; they are canvas, not liquidus."""

    def test_frame_lid_and_floor_points_are_dropped(self):
        wedge = [[0, 2233], [4, 2080], [8, 1980], [8, 2600], [0, 2600]]
        liq, _ = mpds.extract_digitized_liquidus(_make_json([_shape(wedge)]))
        assert len(liq) == 3
        assert max(pt[1] for pt in liq) == pytest.approx(2233 + 273.15)

    def test_frame_points_are_stripped_from_every_shape(self):
        """Not just the first one -- each wedge is closed along the frame in its own
        svgpath, so a lid left on the second shape would ride into the envelope."""
        left = LEFT_WEDGE + [[8, 2600], [0, 2600]]
        right = [[92, 0], [100, 0]] + RIGHT_WEDGE
        json_2 = _make_json([_shape(left), _shape(right)])
        liq, _ = mpds.extract_digitized_liquidus(json_2)
        frame_top, frame_bottom = 2600 + 273.15, 0 + 273.15
        assert all(pt[1] < frame_top - 4 for pt in liq)
        assert all(pt[1] > frame_bottom + 4 for pt in liq)
        assert mpds.liquidus_coverage(json_2)["n_points"] == len(LEFT_WEDGE) + len(RIGHT_WEDGE)


class TestOverlappingShapes:
    """Where two 'L' shapes cover the same composition, the liquidus is the cooler one:
    the liquid field lies above the liquidus."""

    HOT = [[0, 2233], [50, 2000], [100, 1855]]
    COOL = [[0, 2233], [50, 1500], [100, 1855]]

    def test_cooler_branch_wins_the_overlap(self):
        liq, is_partial = mpds.extract_digitized_liquidus(
            _make_json([_shape(self.HOT), _shape(self.COOL)])
        )
        mid = [pt[1] for pt in liq if pt[0] == pytest.approx(0.5)]
        assert mid == [pytest.approx(1500 + 273.15)]
        assert is_partial is False, "overlapping shapes leave no hole"

    def test_result_is_symmetric_in_shape_order(self):
        a, _ = mpds.extract_digitized_liquidus(_make_json([_shape(self.HOT), _shape(self.COOL)]))
        b, _ = mpds.extract_digitized_liquidus(_make_json([_shape(self.COOL), _shape(self.HOT)]))
        assert a == b


class TestShapeSelection:
    """Only shapes labelled exactly 'L' whose kind is a phase field are liquid."""

    def test_two_phase_dome_labels_are_not_liquid(self):
        json_1 = _make_json(
            [
                _shape(LEFT_WEDGE),
                _shape([[20, 1200], [50, 1400], [80, 1200]], label="L1 + L2"),
                _shape(RIGHT_WEDGE),
            ]
        )
        assert len(mpds.liquid_shape_paths(json_1)) == 2
        cov = mpds.liquidus_coverage(json_1)
        assert cov["n_points"] == len(LEFT_WEDGE) + len(RIGHT_WEDGE)

    def test_mislabelled_solid_compound_is_not_liquid(self):
        """Five cached diagrams carry a ``kind='compound'`` shape labelled 'L'; unioned
        in, its sliver would drag the envelope down onto a solid."""
        json_1 = _make_json(
            [
                _shape(SINGLE_L),
                _shape([[50, 400], [50, 900], [51, 900]], kind="compound", is_solid=True),
            ]
        )
        assert len(mpds.liquid_shape_paths(json_1)) == 1
        liq, _ = mpds.extract_digitized_liquidus(json_1)
        assert min(pt[1] for pt in liq) > 900 + 273.15

    def test_no_liquid_shape_at_all(self):
        json_1 = _make_json([_shape(LEFT_WEDGE, label="L1 + L2")])
        assert mpds.liquid_shape_paths(json_1) == []
        assert mpds.extract_digitized_liquidus(json_1) == (None, False)


class TestRealMultiShapeSystems:
    """The systems the defect was found on, read from the local MPDS cache."""

    @pytest.mark.needs_cache
    @pytest.mark.parametrize(
        "system, min_span, is_partial",
        [
            ("Ag-Fe", 0.999, True),  # was 0.006
            ("Er-Ta", 0.999, True),  # was 0.169
            ("Bi-Si", 0.999, True),  # was 0.075
            ("Ir-Th", 0.99, True),  # was 0.609
        ],
    )
    def test_span_recovers(self, system, min_span, is_partial):
        mpds_json = _load_cached(system)
        assert len(mpds.liquid_shape_paths(mpds_json)) > 1
        liq, partial = mpds.extract_digitized_liquidus(mpds_json)
        xs = [pt[0] for pt in liq]
        assert max(xs) - min(xs) >= min_span
        assert partial is is_partial
