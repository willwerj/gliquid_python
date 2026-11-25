"""Ring-closure artifacts: a parsed liquidus branch carries one temperature per composition.

An MPDS ``'L'`` svgpath is a closed RING, not a curve: having traced the liquidus it closes
the liquid field by running up a pure-element axis and back across a top lid. The frame strip
only catches a lid sitting ON the frame, and ``within_tol_from_line``'s zero-width fallback
judges a vertical stitch candidate on the gap between its two ANCHORS, so a closure vertex can
reach the parsed branch as a spike at the edge of a fit+liq / pred+liq plot.

``mpds._collapse_columns`` removes them from one invariant -- the liquidus is single-valued
in composition, and the liquid field lies ABOVE it -- so it can only ever DROP a point.
``TestDiagramInvariants`` pins that the span, ``covered`` and ``is_partial`` a fit reads are
untouched.

At a pure end it picks by distance from ``t_fusion``, not by temperature, because a terminal
solid solution merged into the same ring also reaches the axis. It does not touch a ring that merges the liquid with a
solid field along a shared edge rather than at a single axis column.
"""

import pytest

import gliquid.mpds as mpds
from gliquid.phase import UNARY

from ._mpds_builders import liquid_shape as _shape
from ._mpds_builders import liquidus_record

# An edge spike, as a plot shows it: a local temperature maximum at least SPIKE_T_K above
# both neighbours whose excursion is narrower than SPIKE_X on at least one side, or a
# terminal point that hot above its only neighbour.
SPIKE_X, SPIKE_T_K = 3e-3, 40.0


def _make_json(shapes, elements=("Be", "Ge"), temp=(500, 1405)):
    return liquidus_record(shapes, elements, temp)


def _spikes(curve):
    """Every edge spike in a curve, as ``(index, x, T_celsius)``."""
    found = []
    for i in range(1, len(curve) - 1):
        before, point, after = curve[i - 1], curve[i], curve[i + 1]
        narrow = abs(point[0] - before[0]) <= SPIKE_X or abs(after[0] - point[0]) <= SPIKE_X
        if narrow and point[1] - before[1] >= SPIKE_T_K and point[1] - after[1] >= SPIKE_T_K:
            found.append((i, point[0], point[1] - 273.15))
    if len(curve) >= 2:
        if abs(curve[1][0] - curve[0][0]) <= SPIKE_X and curve[0][1] - curve[1][1] >= SPIKE_T_K:
            found.append((0, curve[0][0], curve[0][1] - 273.15))
        if abs(curve[-1][0] - curve[-2][0]) <= SPIKE_X and curve[-1][1] - curve[-2][1] >= SPIKE_T_K:
            found.append((len(curve) - 1, curve[-1][0], curve[-1][1] - 273.15))
    return found


@pytest.fixture
def no_collapse(monkeypatch):
    """``_shape_liquidus`` without the closure-edge collapse, for the positive controls."""
    monkeypatch.setattr(mpds, "_collapse_columns", lambda branch, mpds_json: sorted(branch))


# A Be-rich ring: up the Be axis from the melting point to a 1395 degC lid that the 1405 degC
# frame strip cannot see, across the lid, then the liquidus back down through a eutectic to
# the Be axis. (0, 1395) is the artifact.
LEFT_RISER = [
    [0, 1289.0],
    [0, 1395.0],
    [20.5, 1395.0],
    [17, 1350],
    [14, 1300],
    [10.5, 1210],
    [7, 1240],
    [4, 1265],
    [0, 1289.0],
]
# The mirror case, up the Yb axis to a 1287.5 degC lid under a 1400 degC frame. (100, 1287.5)
# is the artifact; (65.5, 1287.5) is the corner where a miscibility-gap branch meets the lid
# and must survive.
RIGHT_RISER = [
    [100, 819.0],
    [100, 1287.5],
    [65.5, 1287.5],
    [71.0, 1183.0],
    [80, 1130],
    [90, 1030],
    [96, 930],
    [98.6, 822.0],
    [100, 819.0],
]
# A steep stretch of drawn liquidus near the Y axis: four vertices spanning 112 degC within
# 0.03 at.% of one another, all at INTERIOR compositions.
STEEP_NEAR_AXIS = [
    [97.9, 2200],
    [98.05, 2168.0],
    [99.0, 1935.0],
    [99.39, 1722.0],
    [99.4, 1685.0],
    [99.395, 1648.0],
    [99.37, 1610.0],
    [99.87, 1521.5],
    [100, 1523.5],
    [100, 2200],
    [97.9, 2200],
]


class TestCollapseColumnsUnit:
    """The rule itself, on hand-built branches."""

    BE_GE = _make_json([], elements=("Be", "Ge"))

    def test_interior_duplicate_keeps_the_colder_point(self):
        branch = [[0.2, 1500.0], [0.4, 1400.0], [0.4, 1900.0], [0.6, 1450.0]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert kept == [[0.2, 1500.0], [0.4, 1400.0], [0.6, 1450.0]]

    def test_axis_column_keeps_the_melting_point_not_the_coldest(self):
        """An Mn-axis column holding a 1100 degC solidus vertex and the 1246 degC liquidus end,
        within 1 K of Mn's t_fusion. Taking the coldest swaps the melting point for a solidus."""
        er_mn = _make_json([], elements=("Er", "Mn"))
        branch = [[0.5, 1600.0], [0.9998, 1100.0 + 273.15], [0.9998, 1246.0 + 273.15]]
        kept = mpds._collapse_columns(branch, er_mn)
        assert len(kept) == 2
        assert kept[-1][1] == pytest.approx(1246.0 + 273.15)
        assert abs(kept[-1][1] - UNARY["Mn"].t_fusion) < 1.0

    def test_axis_column_still_drops_a_hotter_lid_vertex(self):
        """The t_fusion anchor is not a licence for the lid: Be melts at 1560 K, so the
        1670 K axis riser loses to the 1562 K liquidus end."""
        branch = [[0.0, 1562.0], [0.0, 1670.0], [0.1, 1483.0]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert kept == [[0.0, 1562.0], [0.1, 1483.0]]

    def test_axis_window_keeps_dense_sampling_of_a_steep_terminal_branch(self):
        """Two digitized points 1 K apart near the Ag axis, at 0.04 and 0.08 at.%. They sit
        inside the axis WINDOW at different compositions, so only a real spike may collapse
        them -- below ``_AXIS_MIN_SPREAD_K`` the window is dense sampling and every point is
        data."""
        ag_v = _make_json([], elements=("Ag", "V"))
        branch = [[0.0004, 964.5 + 273.15], [0.0008, 963.5 + 273.15], [0.5, 1800.0]]
        assert mpds._collapse_columns(branch, ag_v) == sorted(branch)

    def test_axis_window_still_collapses_a_real_spike(self):
        """A closure edge that lands at a DIFFERENT composition from the liquidus end (0.9998
        vs 1.0) but 675 K above it, so the window must collapse."""
        dy_yb = _make_json([], elements=("Dy", "Yb"))
        branch = [[0.5, 1500.0], [0.9998, 815.0 + 273.15], [1.0, 1490.0 + 273.15]]
        kept = mpds._collapse_columns(branch, dy_yb)
        assert kept == [[0.5, 1500.0], [0.9998, 815.0 + 273.15]]

    def test_distinct_near_axis_interior_points_all_survive(self):
        """Over-removal guard: steep sampling at DISTINCT interior compositions is data."""
        branch = [
            [0.9939, 1995.0],
            [0.994, 1958.0],
            [0.99395, 1921.0],
            [0.9937, 1883.0],
        ]
        kept = mpds._collapse_columns(branch, _make_json([], elements=("Eu", "Y")))
        assert sorted(kept) == sorted(branch)

    def test_only_ever_drops(self):
        """The contract ``_lower_envelope`` also keeps: no point is moved or invented, and
        no composition comes back hotter than it went in."""
        branch = [[0.0, 1562.0], [0.0, 1670.0], [0.3, 1450.0], [0.3, 1900.0], [1.0, 1211.0]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert all(point in branch for point in kept)
        coldest = {}
        for x, t in branch:
            coldest[x] = min(t, coldest.get(x, t))
        assert all(t <= coldest[x] for x, t in kept)

    def test_columns_of_one_pass_through_untouched(self):
        branch = [[0.0, 1560.0], [0.25, 1400.0], [0.75, 1300.0], [1.0, 1211.0]]
        assert mpds._collapse_columns(branch, self.BE_GE) == branch


class TestAxisRiser:
    """``_shape_liquidus`` on a closed ring whose closure runs up a pure-element axis."""

    def test_left_axis_riser_is_dropped(self):
        mpds_json = _make_json([_shape(LEFT_RISER)])
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        at_axis = [point for point in branch if point[0] == 0.0]
        assert at_axis == [[0.0, 1289.0 + 273.15]], "one point at the Be axis, the liquidus one"
        assert not _spikes(branch)

    def test_left_axis_riser_is_there_without_the_collapse(self, no_collapse):
        """Positive control: the fixture really does reproduce the artifact."""
        mpds_json = _make_json([_shape(LEFT_RISER)])
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [0.0, 1395.0 + 273.15] in branch
        assert _spikes(branch)

    def test_right_axis_riser_under_a_non_frame_lid_is_dropped(self):
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [point for point in branch if point[0] == 1.0] == [[1.0, 819.0 + 273.15]]
        assert not _spikes(branch)

    def test_right_axis_riser_is_there_without_the_collapse(self, no_collapse):
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [1.0, 1287.5 + 273.15] in branch
        assert _spikes(branch)

    def test_the_real_lid_corner_survives(self):
        """(65.5, 1287.5) is where the miscibility-gap branch meets the lid -- the only
        thing the shape draws at that composition, so it is liquidus, not closure."""
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [0.655, 1287.5 + 273.15] in branch

    def test_steep_near_axis_sampling_survives(self):
        """Over-removal guard at shape level: a shape whose near-axis points sit at
        distinct compositions keeps every one of them."""
        mpds_json = _make_json([_shape(STEEP_NEAR_AXIS)], elements=("Eu", "Y"), temp=(0, 2200))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        for x_pct, t_c in ([99.39, 1722.0], [99.4, 1685.0]):
            nearest = min(branch, key=lambda point: abs(point[0] - x_pct / 100))
            assert nearest[0] == pytest.approx(x_pct / 100)
            assert nearest[1] == pytest.approx(t_c + 273.15)


class TestDiagramInvariants:
    """What a fit reads off the diagram is unchanged."""

    def test_span_covered_and_is_partial_are_unchanged(self, monkeypatch):
        mpds_json = _make_json([_shape(LEFT_RISER)])
        after, partial_after = mpds.extract_digitized_liquidus(mpds_json)
        _, _, covered_after = mpds._stitched_liquidus(mpds_json, quiet=True)

        monkeypatch.setattr(mpds, "_collapse_columns", lambda branch, js: sorted(branch))
        before, partial_before = mpds.extract_digitized_liquidus(mpds_json)
        _, _, covered_before = mpds._stitched_liquidus(mpds_json, quiet=True)

        assert covered_after == covered_before
        assert partial_after == partial_before
        assert before[0][0] == after[0][0] and before[-1][0] == after[-1][0]
        assert len(after) < len(before), "the riser really was in the pre-fix curve"

    def test_a_clean_shape_is_left_alone(self):
        """No spike to remove -> the curve keeps every digitized point, so the collapse is
        not a general-purpose filter."""
        clean = [[0, 1289.0], [25, 1200], [50, 1150], [75, 1050], [100, 938.0]]
        mpds_json = _make_json([_shape(clean)])
        curve, _ = mpds.extract_digitized_liquidus(mpds_json)
        for x, t in clean:
            assert [x / 100, t + 273.15] in curve
        assert not _spikes(curve)


class TestNarrowWedge:
    """A few-vertex 'L' wedge whose sectioning leaves no two-point run is read from its ring."""

    # Triangle on the right axis: hot tip and melting vertex share the axis column.
    TRIANGLE = [[99.96, 4100.0], [99.77, 2650.0], [99.96, 2627.0], [99.96, 4100.0]]
    # Full-width band: wider than _WEDGE_MAX_SPAN_X, its ring is not a liquidus.
    BAND = [
        [99.96, 1345.0],
        [99.96, 1413.0],
        [0.04, 1413.0],
        [0.04, 1345.0],
        [99.96, 1345.0],
    ]

    def test_triangle_keeps_its_floor_not_its_tip(self):
        mpds_json = _make_json([_shape(self.TRIANGLE)], elements=("Dy", "Mo"), temp=(0, 5000))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert sum(branch, []) == pytest.approx([0.9977, 2650.0 + 273.15, 0.9996, 2627.0 + 273.15])

    def test_the_tip_was_kept_without_the_ring_fallback(self, monkeypatch):
        monkeypatch.setattr(mpds, "_WEDGE_MAX_SPAN_X", 0.0)
        mpds_json = _make_json([_shape(self.TRIANGLE)], elements=("Dy", "Mo"), temp=(0, 5000))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert sum(branch, []) == pytest.approx([0.9996, 4100.0 + 273.15])

    def test_wide_band_is_not_rebuilt(self):
        mpds_json = _make_json([_shape(self.BAND)], elements=("Li", "Si"), temp=(200, 1414))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert len(branch) == 1
