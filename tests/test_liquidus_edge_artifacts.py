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
solid solution merged into the same ring also reaches the axis
(``TestCachedCorpus::test_er_mn_...``). It does not touch a ring that merges the liquid with a
solid field along a shared edge rather than at a single axis column.
"""

import pytest

import gliquid.mpds as mpds
from gliquid.phase import UNARY

from ._mpds_builders import liquid_shape as _shape
from ._mpds_builders import liquidus_record
from ._mpds_builders import load_cached_mpds as _load_cached

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


# Be-Ge PD_0 shape 0, thinned: up the Be axis from the melting point to a 1396.35 degC lid
# that the 1405 degC frame strip cannot see, across the lid, then the real liquidus back
# down through the eutectic to the Be axis. (0, 1396.35) is the artifact.
LEFT_RISER = [
    [0, 1288.9],
    [0, 1396.35],
    [20.4918, 1396.35],
    [17, 1350],
    [14, 1300],
    [10.5, 1210],
    [7, 1240],
    [4, 1265],
    [0, 1288.9],
]
# Gd-Yb PD_0 shape 1, thinned: the mirror case, up the Yb axis to a 1287.48 degC lid under
# a 1400 degC frame. (100, 1287.48) is the artifact; (65.5547, 1287.48) is the real corner
# where the miscibility-gap branch meets the lid and must survive.
RIGHT_RISER = [
    [100, 818.855],
    [100, 1287.48],
    [65.5547, 1287.48],
    [70.9874, 1183.17],
    [80, 1130],
    [90, 1030],
    [96, 930],
    [98.5703, 821.896],
    [100, 818.855],
]
# Eu-Y PD_0 shape 1: a genuinely steep stretch of drawn liquidus, four vertices spanning
# 112 degC within 0.03 at.% of one another. All INTERIOR compositions, all real data.
STEEP_NEAR_AXIS = [
    [97.865, 2200],
    [98.0426, 2167.81],
    [99.019, 1934.97],
    [99.388, 1721.94],
    [99.398, 1684.95],
    [99.393, 1647.51],
    [99.371, 1609.58],
    [99.8666, 1521.49],
    [100, 1523.72],
    [100, 2200],
    [97.865, 2200],
]


class TestCollapseColumnsUnit:
    """The rule itself, on hand-built branches."""

    BE_GE = _make_json([], elements=("Be", "Ge"))

    def test_interior_duplicate_keeps_the_colder_point(self):
        branch = [[0.2, 1500.0], [0.4, 1400.0], [0.4, 1900.0], [0.6, 1450.0]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert kept == [[0.2, 1500.0], [0.4, 1400.0], [0.6, 1450.0]]

    def test_axis_column_keeps_the_melting_point_not_the_coldest(self):
        """Er-Mn PD_0's x=0.9998 column: 1100.05 degC and 1246.43 degC against Mn's
        1245.85 degC t_fusion. Taking the coldest swaps the melting point for a solidus."""
        er_mn = _make_json([], elements=("Er", "Mn"))
        branch = [[0.5, 1600.0], [0.99981, 1100.05 + 273.15], [0.99981, 1246.43 + 273.15]]
        kept = mpds._collapse_columns(branch, er_mn)
        assert len(kept) == 2
        assert kept[-1][1] == pytest.approx(1246.43 + 273.15)
        assert abs(kept[-1][1] - UNARY["Mn"].t_fusion) < 1.0

    def test_axis_column_still_drops_a_hotter_lid_vertex(self):
        """The t_fusion anchor is not a licence for the lid: Be melts at 1560 K, so the
        1669.5 K axis riser loses to the 1562.05 K liquidus end."""
        branch = [[0.0, 1562.05], [0.0, 1669.5], [0.1, 1483.15]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert kept == [[0.0, 1562.05], [0.1, 1483.15]]

    def test_axis_window_keeps_dense_sampling_of_a_steep_terminal_branch(self):
        """Ag-V PD_0: (0.038 at.%, 964.83 degC) and (0.079 at.%, 963.73 degC) are 1.1 K
        apart and both digitized. They sit inside the axis WINDOW at different
        compositions, so only a real spike may collapse them -- below
        ``_AXIS_MIN_SPREAD_K`` the window is dense sampling and every point is data."""
        ag_v = _make_json([], elements=("Ag", "V"))
        branch = [[0.00038, 964.83 + 273.15], [0.00079, 963.73 + 273.15], [0.5, 1800.0]]
        assert mpds._collapse_columns(branch, ag_v) == sorted(branch)

    def test_axis_window_still_collapses_a_real_spike(self):
        """Dy-Yb PD_0's class: the closure edge lands at a DIFFERENT composition from the
        liquidus end (0.9998 vs 1.0) but 675 K above it, so the window must collapse."""
        dy_yb = _make_json([], elements=("Dy", "Yb"))
        branch = [[0.5, 1500.0], [0.9998, 813.8 + 273.15], [1.0, 1488.9 + 273.15]]
        kept = mpds._collapse_columns(branch, dy_yb)
        assert kept == [[0.5, 1500.0], [0.9998, 813.8 + 273.15]]

    def test_distinct_near_axis_interior_points_all_survive(self):
        """Over-removal guard: steep sampling at DISTINCT interior compositions is data."""
        branch = [
            [0.99388, 1995.09],
            [0.99398, 1958.1],
            [0.99393, 1920.66],
            [0.99371, 1882.73],
        ]
        kept = mpds._collapse_columns(branch, _make_json([], elements=("Eu", "Y")))
        assert sorted(kept) == sorted(branch)

    def test_only_ever_drops(self):
        """The contract ``_lower_envelope`` also keeps: no point is moved or invented, and
        no composition comes back hotter than it went in."""
        branch = [[0.0, 1562.05], [0.0, 1669.5], [0.3, 1450.0], [0.3, 1900.0], [1.0, 1211.4]]
        kept = mpds._collapse_columns(branch, self.BE_GE)
        assert all(point in branch for point in kept)
        coldest = {}
        for x, t in branch:
            coldest[x] = min(t, coldest.get(x, t))
        assert all(t <= coldest[x] for x, t in kept)

    def test_columns_of_one_pass_through_untouched(self):
        branch = [[0.0, 1560.0], [0.25, 1400.0], [0.75, 1300.0], [1.0, 1211.4]]
        assert mpds._collapse_columns(branch, self.BE_GE) == branch


class TestAxisRiser:
    """``_shape_liquidus`` on a closed ring whose closure runs up a pure-element axis."""

    def test_left_axis_riser_is_dropped(self):
        mpds_json = _make_json([_shape(LEFT_RISER)])
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        at_axis = [point for point in branch if point[0] == 0.0]
        assert at_axis == [[0.0, 1288.9 + 273.15]], "one point at the Be axis, the liquidus one"
        assert not _spikes(branch)

    def test_left_axis_riser_is_there_without_the_collapse(self, no_collapse):
        """Positive control: the fixture really does reproduce the artifact."""
        mpds_json = _make_json([_shape(LEFT_RISER)])
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [0.0, 1396.35 + 273.15] in branch
        assert _spikes(branch)

    def test_right_axis_riser_under_a_non_frame_lid_is_dropped(self):
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [point for point in branch if point[0] == 1.0] == [[1.0, 818.855 + 273.15]]
        assert not _spikes(branch)

    def test_right_axis_riser_is_there_without_the_collapse(self, no_collapse):
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [1.0, 1287.48 + 273.15] in branch
        assert _spikes(branch)

    def test_the_real_lid_corner_survives(self):
        """(65.5547, 1287.48) is where the miscibility-gap branch meets the lid -- the only
        thing the shape draws at that composition, so it is liquidus, not closure."""
        mpds_json = _make_json([_shape(RIGHT_RISER)], elements=("Gd", "Yb"), temp=(200, 1400))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        assert [0.655547, 1287.48 + 273.15] in branch

    def test_steep_near_axis_sampling_survives(self):
        """Over-removal guard at shape level: a shape whose near-axis points sit at
        distinct compositions keeps every one of them."""
        mpds_json = _make_json([_shape(STEEP_NEAR_AXIS)], elements=("Eu", "Y"), temp=(0, 2200))
        branch = mpds._shape_liquidus(mpds_json["shapes"][0]["svgpath"], mpds_json)
        for x_pct, t_c in ([99.388, 1721.94], [99.398, 1684.95]):
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
        clean = [[0, 1288.9], [25, 1200], [50, 1150], [75, 1050], [100, 938.25]]
        mpds_json = _make_json([_shape(clean)])
        curve, _ = mpds.extract_digitized_liquidus(mpds_json)
        for x, t in clean:
            assert [x / 100, t + 273.15] in curve
        assert not _spikes(curve)


class TestCachedCorpus:
    """Opportunistic pins on the real jsons that carry this artifact."""

    REPORTED = ["Be-Ge", "Gd-Yb", "Er-Yb", "Eu-Y", "Ir-Mn"]

    @pytest.mark.parametrize("system", REPORTED)
    def test_no_edge_spike_survives(self, system):
        curve, _ = mpds.extract_digitized_liquidus(_load_cached(system))
        assert _spikes(curve) == []

    @pytest.mark.parametrize("system", REPORTED)
    def test_the_spike_was_there_before(self, system, no_collapse):
        """Positive control on the corpus, so a pin cannot pass by the artifact vanishing
        for some unrelated reason."""
        curve, _ = mpds.extract_digitized_liquidus(_load_cached(system))
        assert _spikes(curve), "this system is only a pin because it carried the defect"

    def test_er_mn_keeps_the_melting_point_over_a_colder_axis_vertex(self):
        """Er-Mn PD_0's Mn axis carries 1100.05 degC and 1246.43 degC; the endpoint is the
        one within 1 K of Mn's t_fusion, not the coldest."""
        curve, _ = mpds.extract_digitized_liquidus(_load_cached("Er-Mn"))
        assert curve[-1][0] == pytest.approx(1.0, abs=mpds._AXIS_X_TOL)
        assert abs(curve[-1][1] - UNARY["Mn"].t_fusion) < 1.0

    def test_ag_v_keeps_both_of_its_near_axis_points(self):
        """Corpus pin for the axis-window guard. Ag-V PD_0 draws two liquidus points inside
        0.1 at.% of the Ag axis, 1.1 K apart; both must survive."""
        curve, _ = mpds.extract_digitized_liquidus(_load_cached("Ag-V"))
        near_axis = [point for point in curve if point[0] <= mpds._AXIS_X_TOL]
        assert len(near_axis) == 2, "dense terminal sampling is data, not a closure edge"
        spread = max(p[1] for p in near_axis) - min(p[1] for p in near_axis)
        assert spread < mpds._AXIS_MIN_SPREAD_K

    def test_ir_mn_interior_spike_is_gone(self):
        """Not every artifact is at an axis. Ir-Mn PD_0 draws two temperatures at
        x=0.798323: the 1263.64 degC liquidus and a 1396.35 degC closure vertex."""
        curve, _ = mpds.extract_digitized_liquidus(_load_cached("Ir-Mn"))
        at_spike = [point for point in curve if point[0] == pytest.approx(0.798323, abs=1e-6)]
        assert len(at_spike) == 1, "one temperature at that composition, not two"
        assert at_spike[0][1] == pytest.approx(1263.64 + 273.15)

    def test_lu_yb_both_ends_land_on_their_melting_points(self):
        """Lu-Yb PD_0 lands on both melting points.

        The Yb end is a ring-closure spike, removed by ``_collapse_columns``. At the Lu end,
        shape 0 is one ring merging the liquid field with the ``(Lu)`` solid solution, which
        the json's ``split_merged_one_phase_ring`` repair splits at the pinch; without it the
        parser walks out along the ``(Lu)`` solvus, so this pin is also that repair's canary.
        """
        mpds_json = _load_cached("Lu-Yb")
        repair = mpds_json.get("gliquid_repair")
        if not (repair and repair.get("op") == "split_merged_one_phase_ring"):
            pytest.skip("this cached Lu-Yb predates the merged-ring repair")
        curve, is_partial = mpds.extract_digitized_liquidus(mpds_json)
        assert abs(curve[0][1] - UNARY["Lu"].t_fusion) < 1.0, "Lu end: merged ring repaired"
        assert abs(curve[-1][1] - UNARY["Yb"].t_fusion) < 1.0, "Yb end: the spike is gone"
        assert not _spikes(curve)

    def test_lu_yb_keeps_its_miscibility_gap_open(self):
        """Lu-Yb's two disjoint 'L' fields keep the L1+L2 miscibility gap between them open,
        and ``is_partial`` stays True. Needs the json repair: unrepaired, the left region is
        the ``(Lu)`` solvus at 1.8-6.6 at.%, not the liquid field at 0-34.4 at.%.
        """
        mpds_json = _load_cached("Lu-Yb")
        if "gliquid_repair" not in mpds_json:
            pytest.skip("this cached Lu-Yb predates the merged-ring repair")
        _, is_partial, covered = mpds._stitched_liquidus(mpds_json, quiet=True)
        assert is_partial is True
        assert len(covered) == 2, "two disjoint liquid fields"
        assert covered[0][0] == pytest.approx(0.0, abs=1e-3)
        assert covered[-1][1] == pytest.approx(1.0, abs=1e-3)
        gap = covered[1][0] - covered[0][1]
        assert gap > mpds._FILL_GAP_X, "the miscibility gap is a hole, not sparse sampling"
        curve, _ = mpds.extract_digitized_liquidus(mpds_json)
        assert not any(covered[0][1] < point[0] < covered[1][0] for point in curve), (
            "nothing may be invented across the gap"
        )
