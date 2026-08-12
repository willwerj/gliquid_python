"""Solid-solution envelope rendering on the binary T-X plot.

The SS phase field is drawn as a CLOSED, hatch-filled polygon rather than a pair of
independent upper/lower polylines. A single branch splitter feeds both boundaries, so
the two can no longer disagree about where a composition gap falls (the Hf-Y / Ru-Y /
Hf-W / Cr-W disconnection class).

Topology contract, checked against real hull output in ``fixtures/ss_envelope_pins.json``:

  * one polygon per contiguous composition branch;
  * 3 or 4 corner vertices per polygon -- 4 when the branch ends bluntly or spans the
    whole composition axis, 3 when the band pinches to an apex at its solubility limit;
  * a branch that reaches a composition-axis edge is extended to exactly x=0 / x=100 and
    contributes two on-edge vertices; every other terminus is an off-edge extremum.

Tie-line admission is no longer geometric. A tie that touches an SS field survives only
when it is an invariant REACTION: a eutectic, a peritectic, an invariant carrying two
liquid vertices (the L1+L2 monotectic, whose remaining vertex is the solution phase
itself), or the EXTREMAL member of an L + S1 + S2 family (one liquid vertex plus two
SEPARATED compositions of the same solution phase -- a solid miscibility gap meeting the
liquidus, Cr-W's ~1932 C horizontal). A 'Misc Gaps' solvus or 'Solid Ties' pair on an SS
field walks a continuous boundary one grid step at a time -- Cr-W emits 75, Hf-W 64 -- and
the field's own hatched polygon already draws where that boundary runs.

Ties are also merged: two sources emitting the same horizontal with slightly different
extents collapse to one trace spanning their union, and a polymorph tie terminates on any
SS field boundary in its way instead of crossing it.

This file is the TOPOLOGY / PREDICATE half: no test here draws a figure, so collecting it
costs nothing. The classes that render one live in
``tests_internal/test_ss_envelope_rendering.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gliquid.plotting.binary_tx as btx
from gliquid.binary import (
    _apex_from_invariant,
    _assemblage_across,
    _edge_inside_ss_field,
    _facet_assemblages,
    _match_edge_interval,
    _split_indices,
    _ss_boundary_crossings,
    _ss_family_maxima,
    _ss_minimum_anchor,
    _ss_minimum_tie_allowed,
    _ss_regions,
    _ss_solid_pair_phase,
    _ss_tie_allowed,
)

PINS_PATH = Path(__file__).parent / "fixtures" / "ss_envelope_pins.json"
PINS = json.loads(PINS_PATH.read_text(encoding="utf-8"))
SYSTEMS = sorted(PINS)


def envelope(system: str, ss_name: str | None = None):
    """``(x_pct, t_lo, t_hi)`` arrays for one pinned solid-solution field."""
    rec = PINS[system]
    ss_name = ss_name or rec["ss_phases"][0]
    env = rec["envelopes"][ss_name]
    return (
        np.array(env["x_pct"], float),
        np.array(env["t_lo"], float),
        np.array(env["t_hi"], float),
    )


def regions_for(system: str):
    """``{ss_phase: [region, ...]}`` for a pinned system."""
    rec = PINS[system]
    return {name: _ss_regions(*envelope(system, name)) for name in rec["ss_phases"]}


def ring_contains_samples(region, x_pct, t_lo, t_hi, tol=1e-6):
    """Every sampled band point of the branch lies on or inside the region ring."""
    from matplotlib.path import Path as MplPath

    path = MplPath(np.column_stack([region["x"], region["t"]]))
    lo_x, hi_x = region["x"].min(), region["x"].max()
    inside = (x_pct >= lo_x - tol) & (x_pct <= hi_x + tol)
    pts = np.column_stack(
        [
            np.concatenate([x_pct[inside], x_pct[inside]]),
            np.concatenate([t_lo[inside], t_hi[inside]]),
        ]
    )
    return bool(
        path.contains_points(pts, radius=1e-3).all()
        or path.contains_points(pts, radius=-1e-3).all()
    )


# ---------------------------------------------------------------------------
# _split_indices -- the single branch splitter shared by both boundaries
# ---------------------------------------------------------------------------
class TestSplitIndices:
    def test_contiguous_run_is_one_branch(self):
        assert _split_indices(np.arange(0.0, 11.0)) == [(0, 11)]

    def test_gap_opens_a_new_branch(self):
        xs = np.array([0.0, 1.0, 2.0, 70.0, 71.0, 72.0])
        assert _split_indices(xs) == [(0, 3), (3, 6)]

    def test_degenerate_inputs(self):
        assert _split_indices(np.array([])) == []
        assert _split_indices(np.array([5.0])) == [(0, 1)]

    def test_indices_partition_the_input(self):
        xs = np.array([0.0, 1.0, 2.0, 40.0, 41.0, 90.0])
        spans = _split_indices(xs)
        assert spans[0][0] == 0 and spans[-1][1] == xs.size
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))

    def test_upper_and_lower_boundaries_split_identically(self):
        """The disconnection bug: both boundaries must break at the same indices."""
        x_pct, t_lo, t_hi = envelope("Cr-W")
        regions = _ss_regions(x_pct, t_lo, t_hi)
        assert len(regions) == 2
        for region in regions:
            # a ring is one closed loop -> its x-extent has no interior gap
            xs = np.sort(np.unique(region["x"]))
            assert np.max(np.diff(xs)) <= 1.5 * np.median(np.diff(xs)) + 1e-9

    def test_two_samples_across_the_diagram_are_two_branches(self):
        """Mo-Y / W-Y BCC: sampled at 0.2 at.% and at 100 at.%, nothing between.

        The gap is then the ONLY spacing present, so inferring the grid from the array
        makes the threshold 1.5x the gap and the cut can never fire -- which welded the
        two terminal branches into one diagram-spanning quadrilateral.
        """
        xs = np.array([0.2, 100.0])
        assert _split_indices(xs) == [(0, 1), (1, 2)]
        assert _split_indices(xs, grid_step=0.2) == [(0, 1), (1, 2)]

    def test_two_adjacent_samples_stay_one_branch(self):
        """The over-cut this must not become: Hf-W's HCP field really is supported at
        exactly 0.2 and 0.4 at.% -- one grid step apart, one branch."""
        xs = np.array([0.2, 0.4])
        assert _split_indices(xs) == [(0, 2)]
        assert _split_indices(xs, grid_step=0.2) == [(0, 2)]

    def test_caller_grid_step_overrides_the_per_array_inference(self):
        xs = np.array([0.0, 5.0, 10.0])
        assert _split_indices(xs) == [(0, 3)]  # inferred median spacing 5 -> no cut
        assert _split_indices(xs, grid_step=1.0) == [(0, 1), (1, 2), (2, 3)]

    def test_explicit_gap_threshold_still_wins_over_the_grid_step(self):
        """The digitized liquidus passes _ASSESSED_GAP_PCT because it is sampled at
        whatever spacing the diagram was traced at; a grid-relative rule would cut it at
        every mildly uneven patch. That override must outrank a supplied grid step."""
        xs = np.array([0.0, 5.0, 10.0])
        assert _split_indices(xs, 6.0, grid_step=0.2) == [(0, 3)]
        assert _split_indices(xs, 3.0, grid_step=99.0) == [(0, 1), (1, 2), (2, 3)]


# ---------------------------------------------------------------------------
# Where the composition grid step comes from
# ---------------------------------------------------------------------------
class TestGridStep:
    def test_two_samples_cannot_report_a_step_coarser_than_the_base_grid(self):
        """A two-sample array's only spacing IS the candidate gap, so it is no evidence
        of the grid. Capped at the 1 at.% base composition grid."""
        assert btx._grid_step(np.array([0.2, 100.0])) == pytest.approx(1.0)

    def test_two_adjacent_samples_keep_their_finer_spacing(self):
        """The cap is one-sided: a finer real spacing is still believed."""
        assert btx._grid_step(np.array([0.2, 0.4])) == pytest.approx(0.2)

    def test_three_or_more_samples_use_the_median_unchanged(self):
        assert btx._grid_step(np.array([0.0, 5.0, 10.0, 15.0])) == pytest.approx(5.0)

    def test_degenerate_arrays_fall_back_to_one_at_pct(self):
        assert btx._grid_step(np.array([5.0])) == pytest.approx(1.0)
        assert btx._grid_step(np.array([5.0, 5.0])) == pytest.approx(1.0)


class TestHullGridStep:
    def test_the_finest_spacing_wins_over_the_median(self):
        """A presentation hull refines only its SOLUTION phases, so an axis carrying a
        two-sample solution field is mostly coarse liquid nodes. Mo-Y's median is 1.0 and
        Hf-Y's is 0.2 off the same 0.2 at.% presentation grid; the minimum is not."""
        axis = np.concatenate([np.arange(0.0, 101.0, 1.0), np.array([0.2])])
        assert btx._hull_grid_step(axis) == pytest.approx(0.2)
        assert btx._hull_grid_step(np.arange(0.0, 100.2, 0.2)) == pytest.approx(0.2)

    def test_float_noise_duplicates_do_not_collapse_the_step(self):
        """The coarse and refined grids are built by different arithmetic, so a shared
        node can appear twice a few ulps apart; unrounded, the minimum would be ~1e-16."""
        axis = np.array([0.0, 1.0, 1.0 + 1e-14, 2.0, 3.0])
        assert btx._hull_grid_step(axis) == pytest.approx(1.0)

    def test_degenerate_axis_falls_back_to_one_at_pct(self):
        assert btx._hull_grid_step(np.array([])) == pytest.approx(1.0)
        assert btx._hull_grid_step(np.array([50.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _ss_regions -- synthetic topologies
# ---------------------------------------------------------------------------
class TestRegionTopology:
    def test_full_range_band_is_one_quadrilateral_on_both_edges(self):
        x = np.linspace(0.0, 100.0, 21)
        lo = 800.0 + 2.0 * x
        hi = 1400.0 + 2.0 * x
        (region,) = _ss_regions(x, lo, hi)
        verts = region["vertices"]
        assert len(verts) == 4
        assert all(v["on_edge"] for v in verts)
        assert sorted({round(v["x"], 6) for v in verts}) == [0.0, 100.0]

    def test_blunt_terminus_keeps_two_off_edge_vertices(self):
        x = np.arange(0.0, 6.0)
        lo = np.array([1000.0, 1150.0, 1300.0, 1450.0, 1600.0, 1750.0])
        hi = np.array([2000.0, 2050.0, 2100.0, 2150.0, 2200.0, 2250.0])
        (region,) = _ss_regions(x, lo, hi)
        verts = region["vertices"]
        assert len(verts) == 4
        off = [v for v in verts if not v["on_edge"]]
        assert len(off) == 2
        assert {v["kind"] for v in off} == {"blunt"}
        assert all(v["x"] == pytest.approx(5.0) for v in off)

    def test_pinched_terminus_collapses_to_one_apex(self):
        x = np.arange(0.0, 6.0)
        hi = np.array([2000.0, 1990.0, 1980.0, 1970.0, 1960.0, 1950.0])
        lo = np.array([1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 1949.0])
        (region,) = _ss_regions(x, lo, hi)
        verts = region["vertices"]
        assert len(verts) == 3
        (apex,) = [v for v in verts if not v["on_edge"]]
        assert apex["kind"] == "apex"
        # the apex sits just past the last sampled composition, where the band closes
        assert 5.0 <= apex["x"] <= 5.0 + 3.0
        assert 1940.0 <= apex["t"] <= 1960.0

    def test_near_edge_branch_is_extended_to_the_axis(self):
        x = np.arange(1.0, 7.0)
        lo = 900.0 + 50.0 * x
        hi = 2000.0 - 10.0 * x
        (region,) = _ss_regions(x, lo, hi)
        edge = [v for v in region["vertices"] if v["on_edge"]]
        assert len(edge) == 2
        assert all(v["x"] == pytest.approx(0.0) for v in edge)
        assert region["x"].min() == pytest.approx(0.0)

    def test_disconnected_branches_give_one_region_each(self):
        x = np.concatenate([np.arange(0.0, 6.0), np.arange(94.0, 101.0)])
        lo = np.concatenate([np.linspace(900, 1400, 6), np.linspace(1400, 900, 7)])
        hi = np.concatenate([np.linspace(2000, 1900, 6), np.linspace(1900, 2000, 7)])
        regions = _ss_regions(x, lo, hi)
        assert len(regions) == 2
        assert regions[0]["x"].max() < regions[1]["x"].min()

    def test_ring_is_closed_and_finite(self):
        x = np.linspace(0.0, 100.0, 11)
        (region,) = _ss_regions(x, 900.0 + 3.0 * x, 1800.0 + 2.0 * x)
        assert region["x"][0] == pytest.approx(region["x"][-1])
        assert region["t"][0] == pytest.approx(region["t"][-1])
        assert np.isfinite(region["x"]).all() and np.isfinite(region["t"]).all()

    def test_single_column_branch_does_not_raise(self):
        regions = _ss_regions(np.array([50.0]), np.array([1000.0]), np.array([1500.0]))
        assert all(np.isfinite(r["x"]).all() for r in regions)

    def test_two_terminal_samples_give_one_region_per_axis(self):
        """Mo-Y's BCC field, to scale: BCC-Mo at 0.2 at.% and beta-Y at 100 at.%.

        Welded it drew a quadrilateral whose upper and lower edges were straight chords
        across the whole diagram, reading as a phase stable everywhere.
        """
        x = np.array([0.2, 100.0])
        lo = np.array([1712.1, 1477.9])
        hi = np.array([2521.6, 1521.8])
        regions = _ss_regions(x, lo, hi, grid_step=0.2)
        assert len(regions) == 2
        left, right = sorted(regions, key=lambda r: r["x"].min())
        assert left["x"].max() < right["x"].min()  # nothing crosses the interior
        assert left["x"].min() == pytest.approx(0.0)  # anchored on the Mo axis
        assert right["x"].max() == pytest.approx(100.0)  # and on the Y axis
        # each branch keeps its OWN temperature extent, not the chord between the two
        assert left["t"].min() == pytest.approx(1712.1)
        assert right["t"].max() == pytest.approx(1521.8)

    def test_two_adjacent_samples_stay_one_region(self):
        """Hf-W's HCP field: two samples one grid step apart are one branch, not two."""
        regions = _ss_regions(
            np.array([0.2, 0.4]),
            np.array([646.0, 700.0]),
            np.array([2009.8, 1900.0]),
            grid_step=0.2,
        )
        assert len(regions) == 1

    def test_single_sample_branch_closes_into_a_ring(self):
        """After the split, Mo-Y's Y-side branch is supported by ONE composition. It must
        still emit a closed, finite ring and a well-formed corner list."""
        (region,) = _ss_regions(
            np.array([100.0]), np.array([1477.9]), np.array([1521.8]), grid_step=0.2
        )
        assert region["x"][0] == pytest.approx(region["x"][-1])
        assert region["t"][0] == pytest.approx(region["t"][-1])
        assert region["x"].size >= 3
        assert np.isfinite(region["x"]).all() and np.isfinite(region["t"]).all()
        assert len(region["vertices"]) in (3, 4)
        assert all(np.isfinite([v["x"], v["t"]]).all() for v in region["vertices"])
        # the branch spans the field's true temperature extent at that composition
        assert region["t"].min() == pytest.approx(1477.9)
        assert region["t"].max() == pytest.approx(1521.8)

    def test_no_ring_segment_spans_more_than_one_grid_gap(self):
        """The defect, stated geometrically: a welded field's ring steps 99.8 at.% between
        consecutive points. Every emitted ring must stay within the splitter's threshold."""
        step = 0.2
        limit = max(1.5 * step, 0.8) + 1e-9
        cases = [
            (np.array([0.2, 100.0]), np.array([1712.1, 1477.9]), np.array([2521.6, 1521.8])),
            # exact counts, not np.arange stop values: float accumulation decides whether
            # the last node lands inside the half-open interval
            (
                np.concatenate([np.arange(15) * step, 97.0 + np.arange(16) * step]),
                np.concatenate([np.linspace(900, 1400, 15), np.linspace(1400, 900, 16)]),
                np.concatenate([np.linspace(2000, 1900, 15), np.linspace(1900, 2000, 16)]),
            ),
        ]
        for x, lo, hi in cases:
            regions = _ss_regions(x, lo, hi, grid_step=step)
            assert len(regions) == 2
            for region in regions:
                assert np.max(np.abs(np.diff(region["x"]))) <= limit


# ---------------------------------------------------------------------------
# _ss_regions -- the four systems the disconnected boundaries were reported on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system", SYSTEMS)
class TestRealSystemTopology:
    def test_every_region_has_three_or_four_vertices(self, system):
        for name, regions in regions_for(system).items():
            assert regions, f"{system}/{name} produced no region"
            for region in regions:
                n = len(region["vertices"])
                assert n in (3, 4), f"{system}/{name}: {n} vertices, expected 3 or 4"

    def test_off_edge_vertices_are_the_complement_of_the_axis_edge_ones(self, system):
        for regions in regions_for(system).values():
            for region in regions:
                verts = region["vertices"]
                on_edge = [v for v in verts if v["on_edge"]]
                off_edge = [v for v in verts if not v["on_edge"]]
                # a field is anchored at an element edge: either it spans the whole axis
                # (4 on-edge) or one end is on an edge and the other is an extremum.
                assert len(on_edge) in (2, 4)
                assert len(off_edge) == len(verts) - len(on_edge)
                assert len(off_edge) <= 2
                for v in on_edge:
                    assert v["x"] == pytest.approx(0.0) or v["x"] == pytest.approx(100.0)
                for v in off_edge:
                    assert v["kind"] in ("apex", "blunt")

    def test_regions_stay_inside_the_composition_axis(self, system):
        for regions in regions_for(system).values():
            for region in regions:
                assert region["x"].min() >= -1e-6
                assert region["x"].max() <= 100.0 + 1e-6

    def test_ring_encloses_the_sampled_band(self, system):
        for name, regions in regions_for(system).items():
            x_pct, t_lo, t_hi = envelope(system, name)
            for region in regions:
                assert ring_contains_samples(region, x_pct, t_lo, t_hi)

    def test_branch_count_matches_the_composition_gaps(self, system):
        for name, regions in regions_for(system).items():
            x_pct, _, _ = envelope(system, name)
            assert len(regions) == len(_split_indices(np.sort(x_pct)))


def test_full_range_solid_solution_touches_both_axis_edges():
    """Hf-W's W-rich lobe and Cr-W's lobes each anchor on exactly one element edge."""
    cr_w = regions_for("Cr-W")["BCC"]
    left, right = sorted(cr_w, key=lambda r: r["x"].min())
    assert left["x"].min() == pytest.approx(0.0)
    assert right["x"].max() == pytest.approx(100.0)
    # the miscibility gap closes at the solubility limits, not at the axis
    assert [len(r["vertices"]) for r in (left, right)] == [3, 3]


def test_hf_y_terminates_bluntly_on_the_hf_side_and_pinches_on_the_y_side():
    left, right = sorted(regions_for("Hf-Y")["HCP"], key=lambda r: r["x"].min())
    assert {v["kind"] for v in left["vertices"] if not v["on_edge"]} == {"blunt"}
    assert {v["kind"] for v in right["vertices"] if not v["on_edge"]} == {"apex"}
    assert len(left["vertices"]) == 4 and len(right["vertices"]) == 3


# ---------------------------------------------------------------------------
# Axis corners: snap to the component polymorph, never extrapolate past its transition
# ---------------------------------------------------------------------------
ALPHA_HF = (733.4, 1742.9)  # hcp ground state
BETA_HF = (1742.9, 2232.8)  # bcc, above the transition
HF_EDGE = {0.0: [ALPHA_HF, BETA_HF], 100.0: []}


class TestMatchEdgeInterval:
    def test_picks_the_polymorph_it_overlaps_most(self):
        # an hcp branch riding alpha-Hf
        assert _match_edge_interval(HF_EDGE, 0.0, 733.4, 1667.6) == ALPHA_HF
        # a bcc branch riding beta-Hf
        assert _match_edge_interval(HF_EDGE, 0.0, 1800.0, 2200.0) == BETA_HF

    def test_no_overlap_returns_none(self):
        assert _match_edge_interval(HF_EDGE, 0.0, 100.0, 300.0) is None

    def test_missing_or_empty_data_returns_none(self):
        assert _match_edge_interval(None, 0.0, 800.0, 1500.0) is None
        assert _match_edge_interval({}, 0.0, 800.0, 1500.0) is None
        assert _match_edge_interval(HF_EDGE, 100.0, 800.0, 1500.0) is None

    def test_touching_intervals_do_not_count_as_overlap(self):
        """The polymorphs share the transition temperature; a zero-width touch is not a
        match, otherwise the choice between them would be arbitrary."""
        assert _match_edge_interval({0.0: [ALPHA_HF]}, 0.0, 1742.9, 1900.0) is None


class TestEdgeSnapping:
    def _branch(self, xs, lo, hi, **kw):
        (region,) = _ss_regions(np.array(xs, float), np.array(lo, float), np.array(hi, float), **kw)
        return region

    def test_branch_short_of_the_axis_snaps_to_the_polymorph(self):
        # steep boundaries that would extrapolate past the alpha->beta transition
        region = self._branch([1.0, 2.0], [900.0, 1200.0], [1650.0, 1450.0], edge_intervals=HF_EDGE)
        edge = sorted(v["t"] for v in region["vertices"] if v["on_edge"])
        assert edge == pytest.approx(list(ALPHA_HF))

    def test_snapping_never_exceeds_the_transition_temperature(self):
        region = self._branch([1.0, 2.0], [900.0, 1200.0], [1650.0, 1450.0], edge_intervals=HF_EDGE)
        assert max(v["t"] for v in region["vertices"] if v["on_edge"]) <= ALPHA_HF[1]

    def test_corner_sampled_at_the_axis_is_left_alone(self):
        """A bcc field present at x=0 must keep its hull values, not be dragged onto the
        hcp ground state that happens to sit at the same composition."""
        region = self._branch(
            [0.0, 1.0, 2.0],
            [1742.9, 1700.0, 1650.0],
            [2232.8, 2200.0, 2150.0],
            edge_intervals=HF_EDGE,
        )
        edge = sorted(v["t"] for v in region["vertices"] if v["on_edge"])
        assert edge == pytest.approx([1742.9, 2232.8])

    def test_without_edge_data_extrapolation_still_applies(self):
        region = self._branch([1.0, 2.0], [900.0, 1200.0], [1650.0, 1450.0])
        edge = sorted(v["t"] for v in region["vertices"] if v["on_edge"])
        assert edge[1] > ALPHA_HF[1]  # the un-snapped overshoot this feature removes

    def test_snapping_is_per_branch_not_per_field(self):
        """Two branches of one field can degenerate into different polymorphs."""
        xs = [1.0, 2.0, 98.0, 99.0]
        region_l, region_r = _ss_regions(
            np.array(xs, float),
            np.array([900.0, 1200.0, 1200.0, 900.0], float),
            np.array([1650.0, 1450.0, 1450.0, 1650.0], float),
            edge_intervals={0.0: [ALPHA_HF], 100.0: [(500.0, 1400.0)]},
        )
        assert sorted(v["t"] for v in region_l["vertices"] if v["on_edge"]) == pytest.approx(
            list(ALPHA_HF)
        )
        assert sorted(v["t"] for v in region_r["vertices"] if v["on_edge"]) == pytest.approx(
            [500.0, 1400.0]
        )


# ---------------------------------------------------------------------------
# Apex snapping: a blunt face straddling a three-phase invariant closes onto it
# ---------------------------------------------------------------------------
def hf_y_bcc_terminus():
    """Hf-Y's BCC terminus, refined: solvus rising ~35 K/at.%, solidus plunging -236."""
    xs = np.array([20.8, 21.0, 21.2, 21.4])
    lo = np.array([1305.4, 1312.4, 1319.4, 1326.3])
    hi = np.array([1602.4, 1571.9, 1535.6, 1488.4])
    return xs, lo, hi


EUTECTIC = (21.4, 1330.7)  # ['BCC','L','HCP']


class TestApexFromInvariant:
    def test_no_invariants_leaves_the_face_alone(self):
        xs, lo, hi = hf_y_bcc_terminus()
        assert (
            _apex_from_invariant(None, 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2)
            is None
        )
        assert (
            _apex_from_invariant([], 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2)
            is None
        )

    def test_invariant_off_the_face_is_ignored(self):
        xs, lo, hi = hf_y_bcc_terminus()
        # below the solvus end, so not on the vertical face at all
        assert (
            _apex_from_invariant(
                [(21.4, 1200.0)], 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2
            )
            is None
        )

    def test_invariant_at_a_distant_composition_is_ignored(self):
        xs, lo, hi = hf_y_bcc_terminus()
        assert (
            _apex_from_invariant(
                [(89.0, 1400.0)], 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2
            )
            is None
        )

    def test_apex_takes_the_invariant_temperature(self):
        xs, lo, hi = hf_y_bcc_terminus()
        apex_x, apex_t = _apex_from_invariant(
            [EUTECTIC], 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2
        )
        assert apex_t == pytest.approx(1330.7)

    def test_composition_comes_from_the_nearer_boundary(self):
        """The eutectic is 4.4 K from the solvus end and 158 K from the plunging solidus;
        extending the near, well-conditioned one puts the apex just past the last sample,
        not the ~0.6 at.% the raw two-boundary intersection would give."""
        xs, lo, hi = hf_y_bcc_terminus()
        apex_x, _ = _apex_from_invariant(
            [EUTECTIC], 21.4, 1326.3, 1488.4, xs, lo, hi, from_left=False, step=0.2
        )
        assert apex_x == pytest.approx(21.4 + (1330.7 - 1326.3) / 34.75, abs=0.02)
        assert 21.4 <= apex_x <= 21.7

    def test_implausible_reach_is_rejected(self):
        """A nearly flat boundary would place the apex arbitrarily far out."""
        xs = np.array([20.0, 21.0])
        lo = np.array([1326.0, 1326.1])  # ~0.1 K per at.% -> reach ~46 at.%
        hi = np.array([1600.0, 1500.0])
        assert (
            _apex_from_invariant(
                [(21.0, 1330.7)], 21.0, 1326.1, 1500.0, xs, lo, hi, from_left=False, step=0.2
            )
            is None
        )


class TestApexSnappingIntegration:
    def _region(self, apex_invariants):
        xs, lo, hi = hf_y_bcc_terminus()
        # prepend an axis-anchored run so the branch has a left edge to close on
        pad_x = np.arange(0.0, 20.8, 0.2)
        pad_lo = np.linspace(1180.0, 1305.0, pad_x.size)
        pad_hi = np.linspace(2230.0, 1602.0, pad_x.size)
        (region,) = _ss_regions(
            np.concatenate([pad_x, xs]),
            np.concatenate([pad_lo, lo]),
            np.concatenate([pad_hi, hi]),
            apex_invariants=apex_invariants,
        )
        return region

    def test_face_closes_to_a_single_apex_at_the_invariant(self):
        region = self._region([EUTECTIC])
        assert len(region["vertices"]) == 3
        (apex,) = [v for v in region["vertices"] if not v["on_edge"]]
        assert apex["kind"] == "apex"
        assert apex["t"] == pytest.approx(1330.7)

    def test_without_the_invariant_the_face_stays_blunt(self):
        region = self._region(None)
        assert len(region["vertices"]) == 4
        assert {v["kind"] for v in region["vertices"] if not v["on_edge"]} == {"blunt"}

    def test_snapped_region_still_satisfies_the_vertex_contract(self):
        region = self._region([EUTECTIC])
        assert len(region["vertices"]) in (3, 4)
        assert region["x"][0] == pytest.approx(region["x"][-1])
        assert region["t"][0] == pytest.approx(region["t"][-1])


# ---------------------------------------------------------------------------
# Tie-line admission at SS extrema
# ---------------------------------------------------------------------------
def family_maxima(system: str):
    """The per-phase L + S1 + S2 extremum plot_tx would pre-compute for a pinned system."""
    rec = PINS[system]
    return _ss_family_maxima(
        [(inv["key"], inv["x_pct"], inv["phases"], inv["temp"]) for inv in rec["invariants"]],
        regions_for(system),
    )


def kept_ties(system: str):
    """The invariant entries that survive the SS tie filter, as ``(temp, x0, x1)``.

    Wired exactly as plot_tx wires it: the family maxima are resolved once over the whole
    pinned invariant list, then handed to every per-invariant call.
    """
    rec = PINS[system]
    regions = regions_for(system)
    t_range = tuple(rec["conds"])
    fam = family_maxima(system)
    kept = []
    for inv in rec["invariants"]:
        if _ss_tie_allowed(
            inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"], fam
        ):
            kept.append((inv["temp"], min(inv["x_pct"]), max(inv["x_pct"])))
    return kept


@pytest.mark.parametrize("system", SYSTEMS)
class TestTieAdmission:
    def test_ties_without_a_solid_solution_phase_are_never_dropped(self, system):
        rec = PINS[system]
        regions = regions_for(system)
        t_range = tuple(rec["conds"])
        for inv in rec["invariants"]:
            if not any(p in regions for p in inv["phases"]):
                assert (
                    _ss_tie_allowed(
                        inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"]
                    )
                    is True
                )

    def test_every_kept_ss_tie_is_an_invariant_reaction(self, system):
        """The contract: a surviving SS-touching tie is a eutectic, a peritectic, an
        L1+L2 horizontal, or the extremal member of an L + S1 + S2 family. Nothing is
        admitted on the field's geometry any more."""
        rec = PINS[system]
        regions = regions_for(system)
        t_range = tuple(rec["conds"])
        fam = family_maxima(system)
        for inv in rec["invariants"]:
            if not any(p in regions for p in inv["phases"]):
                continue
            if not _ss_tie_allowed(
                inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"], fam
            ):
                continue
            assert (
                inv["key"] in ("Eutectics", "Peritectics")
                or sum(1 for p in inv["phases"] if p == "L") >= 2
                or _ss_solid_pair_phase(inv["x_pct"], inv["phases"], regions) in fam
            ), (
                f"{system}: kept a {inv['key']} tie at T={inv['temp']} "
                f"({inv['phases']}) that is not an invariant reaction"
            )

    def test_ss_touching_solvus_entries_are_always_rejected(self, system):
        """A 'Misc Gaps' / 'Solid Ties' entry on an SS field walks a CONTINUOUS boundary
        one grid step at a time; the field's own hatched polygon already draws it."""
        rec = PINS[system]
        regions = regions_for(system)
        t_range = tuple(rec["conds"])
        fam = family_maxima(system)
        for inv in rec["invariants"]:
            if inv["key"] in ("Eutectics", "Peritectics"):
                continue
            if not any(p in regions for p in inv["phases"]):
                continue
            if sum(1 for p in inv["phases"] if p == "L") >= 2:
                continue  # the monotectic exemption
            if _ss_solid_pair_phase(inv["x_pct"], inv["phases"], regions) is not None:
                continue  # the L + S1 + S2 exemption, checked on its own below
            assert (
                _ss_tie_allowed(
                    inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"], fam
                )
                is False
            )

    def test_at_most_one_l_plus_two_solid_tie_per_ss_field(self, system):
        """The acceptance criterion the fan would violate: clause 4 draws ONE tie per
        solution field, however many slices the emitter handed over for it."""
        rec = PINS[system]
        regions = regions_for(system)
        t_range = tuple(rec["conds"])
        fam = family_maxima(system)
        assert set(fam) <= set(rec["ss_phases"])
        drawn: dict[str, int] = {}
        for inv in rec["invariants"]:
            if inv["key"] in ("Eutectics", "Peritectics"):
                continue
            phase = _ss_solid_pair_phase(inv["x_pct"], inv["phases"], regions)
            if phase is None:
                continue
            if _ss_tie_allowed(
                inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"], fam
            ):
                drawn[phase] = drawn.get(phase, 0) + 1
        assert all(n == 1 for n in drawn.values()), f"{system}: {drawn}"

    def test_filter_is_a_strict_subset(self, system):
        assert len(kept_ties(system)) <= len(PINS[system]["invariants"])


def test_miscibility_gap_fan_collapses():
    """Cr-W / Hf-W emitted one tie per temperature step across the gap; that must stop."""
    for system, before_min in (("Cr-W", 70), ("Hf-W", 60)):
        before = len(PINS[system]["invariants"])
        after = len(kept_ties(system))
        assert before >= before_min
        assert after <= 8, f"{system}: {after} ties survived, expected a handful"


def test_cr_w_collapses_to_its_peritectic_and_one_l_plus_two_bcc():
    """The acceptance case, on real hull output.

    Cr-W's 75 'Misc Gaps' entries walk a SOLID (BCC+BCC) miscibility gap. 46 of them are
    ('L','BCC','BCC'); 45 are collapsed facets of the W-rich two-phase band whose BCC
    "pair" is one 1 at.% sampling step wide, and exactly ONE is the reaction where the gap
    meets the liquidus -- BCC at 29 and 71 at.%, 42 at.% apart. That one is admitted, at
    the family maximum, alongside the ['Cr','BCC','W'] peritectic. Nothing else survives.
    """
    kept = sorted(kept_ties("Cr-W"))
    assert len(kept) == 2
    (t_per, x0_per, x1_per), (t_inv, x0_inv, x1_inv) = kept
    assert t_per == pytest.approx(559.27, abs=0.5)
    assert (x0_per, x1_per) == pytest.approx((0.0, 100.0))
    assert t_inv == pytest.approx(1932.4, abs=0.5)
    assert (x0_inv, x1_inv) == pytest.approx((14.0, 71.0))

    rec = PINS["Cr-W"]
    regions = regions_for("Cr-W")
    shaped = [
        i
        for i in rec["invariants"]
        if len(i["phases"]) == 3 and sorted(i["phases"]) == ["BCC", "BCC", "L"]
    ]
    genuine = [i for i in shaped if _ss_solid_pair_phase(i["x_pct"], i["phases"], regions) == "BCC"]
    assert len(shaped) == 46 and len(genuine) == 1
    assert family_maxima("Cr-W") == {"BCC": pytest.approx(1932.4, abs=0.5)}


def test_three_phase_invariants_are_never_filtered():
    """A eutectic/peritectic puts a participating SS phase at its solubility limit.

    Hf-Y's BCC terminus is a blunt face spanning ~1298-1629 C and the eutectic sits at
    1330.6 C -- inside the face, at neither corner -- so a purely geometric corner test
    discards the one invariant that actually locates the maximum-composition vertex.
    """
    for system in SYSTEMS:
        rec = PINS[system]
        regions = regions_for(system)
        t_range = tuple(rec["conds"])
        for inv in rec["invariants"]:
            if inv["key"] not in ("Eutectics", "Peritectics"):
                continue
            if not any(p in regions for p in inv["phases"]):
                continue
            assert (
                _ss_tie_allowed(
                    inv["x_pct"], inv["phases"], inv["temp"], regions, t_range, inv["key"]
                )
                is True
            ), f"{system}: {inv['key']} at T={inv['temp']} was filtered out"


def test_ru_y_keeps_the_hcp_eutectic():
    """['HCP', 'L', 'YRu2'] at 1913.4 C -- HCP is at its maximum composition there."""
    kept = kept_ties("Ru-Y")
    assert any(abs(t - 1913.4) < 1.0 for t, _, _ in kept)


def test_hf_y_keeps_its_peritectic_and_one_l_plus_two_hcp():
    """The 4 at.% solubility limit is located by the ['beta-Hf (bcc)', 'HCP', 'L']
    peritectic at 1978 C, not by the ten solvus slices that used to be admitted at the
    field's composition extrema. One of those ten IS a reaction -- ['HCP','L','HCP'] at
    1322.7 C with the HCP pair at 4 and 96 at.% -- and clause 4 admits exactly it."""
    rec = PINS["Hf-Y"]
    assert sum(1 for i in rec["invariants"] if i["key"] == "Misc Gaps") == 10
    kept = sorted(kept_ties("Hf-Y"))
    assert len(kept) == 2
    (t_inv, x0_inv, x1_inv), (t_per, x0_per, x1_per) = kept
    assert t_per == pytest.approx(1978.15, abs=0.5)
    assert (x0_per, x1_per) == pytest.approx((0.0, 30.0))
    assert t_inv == pytest.approx(1322.69, abs=0.5)
    assert (x0_inv, x1_inv) == pytest.approx((4.0, 96.0))


# ---------------------------------------------------------------------------
# The liquid-liquid (monotectic) exemption -- Sc-V's ['L', 'L', 'BCC'] at 1459 C
# ---------------------------------------------------------------------------
def _box_region(x_lo, x_hi, t_lo, t_hi):
    """A rectangular stand-in field: only its ring matters to the helpers under test."""
    return {
        "x": np.array([x_lo, x_hi, x_hi, x_lo, x_lo], float),
        "t": np.array([t_lo, t_lo, t_hi, t_hi, t_lo], float),
        "vertices": [],
        "edge_anchors": [],
        "vertex_anchors": [],
        "edge_tol": 1.5,
    }


SC_V_REGIONS = {"BCC": [_box_region(90.0, 100.0, 1200.0, 1900.0)]}
SC_V_RANGE = (1000.0, 2000.0)


class TestLiquidLiquidExemption:
    def test_two_liquid_invariant_on_an_ss_phase_is_admitted(self):
        """Sc-V's monotectic terminates ON the BCC solution, so every geometric test
        rejected the one horizontal that marks the miscibility gap."""
        assert (
            _ss_tie_allowed(
                [19.0, 56.0, 99.8], ["L", "L", "BCC"], 1459.4, SC_V_REGIONS, SC_V_RANGE, "Misc Gaps"
            )
            is True
        )

    def test_the_exemption_does_not_depend_on_where_the_solid_sits(self):
        for comp in (90.0, 95.0, 99.8, 100.0):
            assert (
                _ss_tie_allowed(
                    [19.0, 56.0, comp],
                    ["L", "L", "BCC"],
                    1459.4,
                    SC_V_REGIONS,
                    SC_V_RANGE,
                    "Misc Gaps",
                )
                is True
            )

    def test_a_single_liquid_vertex_is_not_enough(self):
        """One liquid vertex is not the L1+L2 exemption. Without a family maximum (the
        pre-clause-4 call shape) such an entry is rejected outright."""
        assert (
            _ss_tie_allowed(
                [90.0, 95.0, 99.0],
                ["BCC", "BCC", "L"],
                1500.0,
                SC_V_REGIONS,
                SC_V_RANGE,
                "Misc Gaps",
            )
            is False
        )

    def test_solid_only_gap_on_an_ss_phase_is_rejected(self):
        assert (
            _ss_tie_allowed(
                [90.0, 95.0, 99.8],
                ["BCC", "BCC", "HfW2"],
                1459.4,
                SC_V_REGIONS,
                SC_V_RANGE,
                "Misc Gaps",
            )
            is False
        )

    def test_two_liquid_invariant_without_any_ss_phase_still_passes(self):
        assert (
            _ss_tie_allowed(
                [19.0, 56.0, 99.8], ["L", "L", "V"], 1459.4, SC_V_REGIONS, SC_V_RANGE, "Misc Gaps"
            )
            is True
        )

    def test_solid_ties_key_is_filtered_the_same_way(self):
        assert (
            _ss_tie_allowed(
                [92.0, 99.0], ["BCC", "W"], 1500.0, SC_V_REGIONS, SC_V_RANGE, "Solid Ties"
            )
            is False
        )


# ---------------------------------------------------------------------------
# The L + S1 + S2 invariant -- Cr-W's ('BCC','BCC','L') at ~1932 C
# ---------------------------------------------------------------------------
# The real Cr-W family, in at.% W, as plot_tx sees it on the FITTED hull: one genuine
# reaction (BCC pair 42.4 at.% apart) and three collapsed facets of the two-phase L + BCC
# band whose "pair" is a single 0.2 at.% sampling step. Only the first may be drawn.
CR_W_REGIONS = {"BCC": [_box_region(0.0, 100.0, 500.0, 3400.0)]}
CR_W_RANGE = (500.0, 3500.0)
CR_W_FAMILY = [
    ("Misc Gaps", [98.0, 99.4, 99.6], ["L", "BCC", "BCC"], 3389.846),  # collapsed facet
    ("Misc Gaps", [14.0, 28.8, 71.2], ["L", "BCC", "BCC"], 1932.378),  # THE reaction
    ("Misc Gaps", [14.0, 28.6, 28.8], ["L", "BCC", "BCC"], 1932.245),  # collapsed facet
    ("Misc Gaps", [9.0, 10.4, 10.6], ["L", "BCC", "BCC"], 1893.413),  # collapsed facet
]


def _cr_w_kept(family=CR_W_FAMILY, regions=CR_W_REGIONS):
    fam = _ss_family_maxima(family, regions)
    return [
        (t, c) for _k, c, p, t in family if _ss_tie_allowed(c, p, t, regions, CR_W_RANGE, _k, fam)
    ]


class TestLPlusTwoSolidTies:
    def test_the_solid_pair_must_be_genuinely_apart(self):
        """A 0.2 at.% "pair" is one sampling step of a continuous boundary, not a gap."""
        assert _ss_solid_pair_phase([14.0, 28.8, 71.2], ["L", "BCC", "BCC"], CR_W_REGIONS) == "BCC"
        assert _ss_solid_pair_phase([98.0, 99.4, 99.6], ["L", "BCC", "BCC"], CR_W_REGIONS) is None

    def test_two_different_solids_are_not_this_reaction(self):
        assert _ss_solid_pair_phase([14.0, 28.8, 71.2], ["L", "BCC", "HfW2"], CR_W_REGIONS) is None

    def test_the_solid_must_be_a_solution_phase(self):
        assert _ss_solid_pair_phase([14.0, 28.8, 71.2], ["L", "Cr", "Cr"], CR_W_REGIONS) is None

    def test_a_two_vertex_entry_is_not_this_reaction(self):
        assert _ss_solid_pair_phase([28.8, 71.2], ["BCC", "BCC"], CR_W_REGIONS) is None

    def test_the_family_maximum_is_the_only_genuine_member(self):
        """The hottest entry of Cr-W's bucket is a collapsed facet at 3389 C; the family
        maximum must skip it and land on the 1932 C reaction."""
        assert _ss_family_maxima(CR_W_FAMILY, CR_W_REGIONS) == {"BCC": 1932.378}

    def test_only_the_extremal_member_is_admitted(self):
        kept = _cr_w_kept()
        assert len(kept) == 1
        ((temp, comps),) = kept
        assert temp == pytest.approx(1932.378)
        assert (min(comps), max(comps)) == pytest.approx((14.0, 71.2))

    def test_a_lower_member_of_the_same_family_is_rejected(self):
        """The test that proves the fan cannot return: a second genuine-width member
        below the extremum draws nothing."""
        family = CR_W_FAMILY + [("Misc Gaps", [12.0, 30.0, 69.0], ["L", "BCC", "BCC"], 1800.0)]
        kept = _cr_w_kept(family)
        assert len(kept) == 1
        assert kept[0][0] == pytest.approx(1932.378)

    def test_members_inside_the_merge_tolerance_of_the_extremum_are_kept(self):
        """The extremum is matched within _TIE_MERGE_T_FRAC of the plotted span, because
        _add_tie would merge anything that close into one trace anyway."""
        tol = btx._TIE_MERGE_T_FRAC * (CR_W_RANGE[1] - CR_W_RANGE[0])
        family = CR_W_FAMILY + [
            ("Misc Gaps", [14.0, 28.9, 71.1], ["L", "BCC", "BCC"], 1932.378 - 0.4 * tol)
        ]
        assert len(_cr_w_kept(family)) == 2
        family = CR_W_FAMILY + [
            ("Misc Gaps", [14.0, 28.9, 71.1], ["L", "BCC", "BCC"], 1932.378 - 4.0 * tol)
        ]
        assert len(_cr_w_kept(family)) == 1

    def test_the_families_of_two_phases_are_selected_independently(self):
        regions = {"BCC": CR_W_REGIONS["BCC"], "HCP": [_box_region(0.0, 100.0, 500.0, 3400.0)]}
        family = CR_W_FAMILY + [
            ("Misc Gaps", [5.0, 20.0, 60.0], ["L", "HCP", "HCP"], 1500.0),
            ("Misc Gaps", [5.0, 22.0, 58.0], ["L", "HCP", "HCP"], 1400.0),
        ]
        assert _ss_family_maxima(family, regions) == {"BCC": 1932.378, "HCP": 1500.0}

    def test_always_keys_do_not_set_the_bar(self):
        """A eutectic/peritectic draws unconditionally, so letting one define the maximum
        could suppress a genuine solvus-family reaction below it."""
        family = CR_W_FAMILY + [("Peritectics", [20.0, 30.0, 80.0], ["L", "BCC", "BCC"], 3000.0)]
        assert _ss_family_maxima(family, CR_W_REGIONS) == {"BCC": 1932.378}

    def test_without_a_family_maximum_the_clause_cannot_fire(self):
        """The argument is defaulted, so every pre-clause-4 caller keeps its behavior."""
        assert (
            _ss_tie_allowed(
                [14.0, 28.8, 71.2],
                ["L", "BCC", "BCC"],
                1932.378,
                CR_W_REGIONS,
                CR_W_RANGE,
                "Misc Gaps",
            )
            is False
        )

    def test_a_family_with_no_genuine_member_admits_nothing(self):
        family = [e for e in CR_W_FAMILY if e[3] != 1932.378]
        assert _ss_family_maxima(family, CR_W_REGIONS) == {}
        assert _cr_w_kept(family) == []


# ---------------------------------------------------------------------------
# Solid-solution fields as polymorph-tie boundaries
# ---------------------------------------------------------------------------
MN_BCC = {"BCC": [_box_region(6.0, 18.0, 700.0, 1250.0)]}


class TestSsBoundaryCrossings:
    def test_a_horizontal_through_the_field_returns_both_boundaries(self):
        xs = sorted(_ss_boundary_crossings(MN_BCC, 900.0, temp_tol=1.0))
        assert xs == pytest.approx([6.0, 18.0])

    def test_a_horizontal_clear_of_the_field_returns_nothing(self):
        assert _ss_boundary_crossings(MN_BCC, 400.0, temp_tol=1.0) == []

    def test_no_ss_phases_means_no_candidates(self):
        assert _ss_boundary_crossings({}, 900.0, temp_tol=1.0) == []

    def test_every_region_of_every_phase_is_walked(self):
        two = {
            "BCC": [_box_region(6.0, 18.0, 700.0, 1250.0)],
            "HCP": [_box_region(60.0, 80.0, 700.0, 1250.0)],
        }
        xs = sorted(_ss_boundary_crossings(two, 900.0, temp_tol=1.0))
        assert xs == pytest.approx([6.0, 18.0, 60.0, 80.0])


class TestEdgeInsideSsField:
    def test_edge_clear_of_the_field_is_not_inside(self):
        """Mn-Si at the alpha->beta transition: the (Mn) field spans 6-18 at.%, so the tie
        is drawn and clamped onto its near boundary rather than suppressed."""
        assert _edge_inside_ss_field(MN_BCC, 900.0, 0.0, temp_tol=1.0) is False

    def test_edge_covered_by_the_field_is_inside(self):
        """An axis-anchored field: the pure element IS the solution phase there."""
        anchored = {"BCC": [_box_region(0.0, 12.5, 1132.0, 1246.0)]}
        assert _edge_inside_ss_field(anchored, 1200.0, 0.0, temp_tol=1.0) is True

    def test_the_on_edge_corner_itself_counts_as_covered(self):
        """Mn-Si's beta-Mn tops out exactly at the field's lower on-edge corner."""
        anchored = {"BCC": [_box_region(0.0, 12.5, 1132.8, 1246.0)]}
        assert _edge_inside_ss_field(anchored, 1132.8, 0.0, temp_tol=0.5) is True

    def test_the_opposite_edge_is_unaffected(self):
        anchored = {"BCC": [_box_region(0.0, 12.5, 1132.0, 1246.0)]}
        assert _edge_inside_ss_field(anchored, 1200.0, 100.0, temp_tol=1.0) is False

    def test_a_temperature_above_the_field_is_not_inside(self):
        anchored = {"BCC": [_box_region(0.0, 12.5, 1132.0, 1246.0)]}
        assert _edge_inside_ss_field(anchored, 1400.0, 0.0, temp_tol=1.0) is False

    def test_no_ss_phases_never_suppresses(self):
        assert _edge_inside_ss_field({}, 1200.0, 0.0, temp_tol=1.0) is False


# ---------------------------------------------------------------------------
# The eutectoid horizontal at a branch's interior temperature minimum
# ---------------------------------------------------------------------------
# The synthetic field: an 80 at.%-wide band sampled every 1 at.%, flat on top at 900 C,
# its lower boundary a V bottoming out at 600 C in the middle. Both termini are blunt, so
# neither terminus anchor covers the V's floor and the branch carries exactly one
# 'lower_min' anchor -- the shape Y-Zr's and Hf-Y's BCC fields have.
_MIN_STEP = 1.0
_MIN_FLOOR = 600.0
_MIN_TOP = 900.0


def _facet(t, *vertices):
    """One hull facet as the three ``df_tx`` rows that share its temperature."""
    return [[x / 100.0, float(t), label] for x, label in vertices]


def _v_region(x_min=50.0, x0=10.0, x1=90.0, slope=5.0, step=_MIN_STEP):
    """One branch whose lower boundary bottoms out at ``x_min``."""
    xs = np.arange(x0, x1 + 0.5 * step, step)
    lo = _MIN_FLOOR + slope * np.abs(xs - x_min)
    hi = np.full(xs.shape, _MIN_TOP)
    return _ss_regions(xs, lo, hi, grid_step=step)[0]


# (temp, x_lo, x_hi, phase set) rows in the shape _facet_assemblages returns. The pair
# straddling 600 C is what admission rule 1 reads. Both span the whole axis, so every
# candidate composition below has a KNOWN assemblage on both sides and rule 1 passes --
# whatever else rejects it is then the clause under test.
_BELOW = (500.0, 0.0, 100.0, frozenset({"CA", "CB"}))
_CHANGED = [_BELOW, (650.0, 0.0, 60.0, frozenset({"CA", "SS"}))]
_UNCHANGED = [_BELOW, (650.0, 0.0, 100.0, frozenset({"CA", "CB"}))]
_WIDE_RANGE = (0.0, 2000.0)


def _assemblage_changes(region, assemblages, t_range=_WIDE_RANGE):
    """Whether admission rule 1 passes for a region -- used to attribute a rejection."""
    anchor = _ss_minimum_anchor(region)
    eps = btx._SS_MIN_T_EPS_FRAC * (t_range[1] - t_range[0])
    sides = [
        _assemblage_across(assemblages, anchor["x"], anchor["t"], above=up, temp_eps=eps)
        for up in (False, True)
    ]
    return all(s is not None for s in sides) and sides[0] != sides[1]


def _allowed(region, assemblages, t_range=_WIDE_RANGE):
    return _ss_minimum_tie_allowed(
        _ss_minimum_anchor(region), region, assemblages, (0.0, 100.0), t_range
    )


class TestMinimumTieAdmission:
    def test_the_branch_carries_exactly_one_lower_minimum_anchor(self):
        region = _v_region()
        mins = [a for a in region["vertex_anchors"] if a.get("kind") == "lower_min"]
        assert len(mins) == 1
        assert mins[0]["x"] == pytest.approx(50.0)
        assert mins[0]["t"] == pytest.approx(_MIN_FLOOR)

    def test_an_interior_minimum_with_a_changing_assemblage_is_admitted(self):
        """Y-Zr: HCP+HCP below 768 C, BCC+HCP above it -- a eutectoid."""
        assert _allowed(_v_region(), _CHANGED) is True

    def test_an_interior_minimum_with_an_unchanged_assemblage_is_rejected(self):
        """The pinch-apex case: a field narrowing to a point INSIDE one two-phase region
        (Sc-V's Sc-side BCC sliver) marks no reaction."""
        assert _allowed(_v_region(), _UNCHANGED) is False

    def test_an_unknown_side_is_never_evidence_of_a_change(self):
        """Nothing below the minimum in the hull means the assemblage there was never
        computed -- Mn-Si's BCC bottoms out at the coldest facet in the system."""
        assert _allowed(_v_region(), [_CHANGED[1]]) is False
        assert _allowed(_v_region(), [_BELOW]) is False
        assert _allowed(_v_region(), []) is False

    def test_a_minimum_within_the_edge_window_of_an_axis_is_rejected(self):
        """Ti-V: the BCC lower boundary is monotonic and its coldest sample sits one grid
        step from the axis, where the boundary is still running off the edge."""
        region = _v_region(x_min=1.0, x0=0.0, x1=40.0)
        anchor = _ss_minimum_anchor(region)
        assert anchor["x"] == pytest.approx(1.0)
        assert anchor["x"] - 0.0 <= region["edge_tol"]  # inside the window
        assert _assemblage_changes(region, _CHANGED)  # rule 1 passed; the window rejects
        assert _allowed(region, _CHANGED) is False

    def test_the_same_minimum_moved_inward_is_admitted(self):
        """The edge window is the only thing rejecting the case above."""
        region = _v_region(x_min=4.0, x0=0.0, x1=40.0)
        assert _ss_minimum_anchor(region)["x"] - 0.0 > region["edge_tol"]
        assert _allowed(region, _CHANGED) is True

    def test_a_minimum_at_the_plotted_temperature_floor_is_rejected(self):
        """A field the bottom of the frame cuts off has its coldest SAMPLE there, not its
        coldest point."""
        region = _v_region()
        assert _assemblage_changes(region, _CHANGED, t_range=(595.0, 2000.0))
        assert _allowed(region, _CHANGED, t_range=(595.0, 2000.0)) is False

    def test_the_same_minimum_clear_of_the_floor_is_admitted(self):
        assert _allowed(_v_region(), _CHANGED, t_range=(400.0, 2000.0)) is True

    def test_a_minimum_sitting_on_a_terminus_never_becomes_a_candidate(self):
        """A monotonic branch bottoms out AT its terminus, which the terminus anchor
        already covers -- so no 'lower_min' anchor is recorded at all."""
        xs = np.arange(10.0, 90.5, _MIN_STEP)
        region = _ss_regions(
            xs, 900.0 - 2.0 * xs, np.full(xs.shape, _MIN_TOP), grid_step=_MIN_STEP
        )[0]
        assert _ss_minimum_anchor(region) is None
        assert _ss_minimum_tie_allowed(None, region, _CHANGED, (0.0, 100.0), _WIDE_RANGE) is False


class TestFacetAssemblages:
    def test_consecutive_triples_become_one_facet_each(self):
        rows = _facet(700.0, (10, "SS"), (40, "CA"), (90, "CB"))
        rows += _facet(500.0, (5, "CA"), (5, "CA"), (95, "CB"))
        got = _facet_assemblages(pd.DataFrame(rows, columns=["x", "t", "label"]))
        assert got == [
            (700.0, 10.0, 90.0, frozenset({"SS", "CA", "CB"})),
            (500.0, 5.0, 95.0, frozenset({"CA", "CB"})),
        ]

    def test_a_repeated_label_is_one_phase_at_both_ends_of_its_own_field(self):
        rows = _facet(700.0, (10, "SS"), (12, "SS"), (90, "CB"))
        (facet,) = _facet_assemblages(pd.DataFrame(rows, columns=["x", "t", "label"]))
        assert facet[3] == frozenset({"SS", "CB"})

    def test_a_frame_that_is_not_whole_facets_yields_nothing(self):
        rows = _facet(700.0, (10, "SS"), (40, "CA"), (90, "CB"))[:2]
        assert _facet_assemblages(pd.DataFrame(rows, columns=["x", "t", "label"])) == []

    def test_a_triple_whose_temperatures_disagree_yields_nothing(self):
        rows = _facet(700.0, (10, "SS"), (40, "CA"), (90, "CB"))
        rows[1][1] = 701.0
        assert _facet_assemblages(pd.DataFrame(rows, columns=["x", "t", "label"])) == []

    def test_an_empty_frame_yields_nothing(self):
        assert _facet_assemblages(pd.DataFrame(columns=["x", "t", "label"])) == []


class TestAssemblageAcross:
    ROWS = [
        (500.0, 5.0, 95.0, frozenset({"CA", "CB"})),
        (600.0, 5.0, 95.0, frozenset({"CA", "SS", "CB"})),
        (650.0, 5.0, 60.0, frozenset({"CA", "SS"})),
        (700.0, 70.0, 95.0, frozenset({"SS", "CB"})),
    ]

    def test_the_nearest_facet_in_each_direction_wins(self):
        assert _assemblage_across(self.ROWS, 50.0, 600.0, above=True, temp_eps=1e-3) == frozenset(
            {"CA", "SS"}
        )
        assert _assemblage_across(self.ROWS, 50.0, 600.0, above=False, temp_eps=1e-3) == frozenset(
            {"CA", "CB"}
        )

    def test_a_facet_that_does_not_span_the_composition_is_skipped(self):
        # the 650 C facet stops at 60 at.%, so at 75 the next one up is the 700 C band
        assert _assemblage_across(self.ROWS, 75.0, 600.0, above=True, temp_eps=1e-3) == frozenset(
            {"SS", "CB"}
        )

    def test_facets_at_the_temperature_itself_belong_to_neither_side(self):
        assert _assemblage_across(self.ROWS, 50.0, 600.0, above=True, temp_eps=60.0) is None
        assert _assemblage_across(self.ROWS, 50.0, 600.0, above=False, temp_eps=200.0) is None

    def test_nothing_on_that_side_returns_none(self):
        assert _assemblage_across(self.ROWS, 50.0, 100.0, above=False, temp_eps=1e-3) is None
