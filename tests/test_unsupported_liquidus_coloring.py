"""Recoloring the assessed liquidus where no solid free energy stands behind it.

``BinaryLiquid.unsupported_comp_ranges`` names the composition stretches the objective
either drops or scores one-sidedly. ``plot_tx`` sub-splits each digitized region at the range
edges and draws the inside runs in ``ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR``:

  * the edge is INTERPOLATED onto the curve and shared by both sub-runs, so the recolor
    is a colour change and not a one-sample gap that reads as a break in the curve; and
  * an empty range list is a no-op: the hole splitting from ``_assessed_liquidus_segments``
    is untouched.

The two splits compose: a run breaks at an undigitized hole AND at a range edge.
"""

import pytest

from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.plotting.binary_tx import (
    _ASSESSED_DASH,
    _ASSESSED_GAP_PCT,
    _ASSESSED_UNSUPPORTED_DASH,
    _assessed_liquidus_segments,
    _normalize_ranges_pct,
    _unsupported_liquidus_runs,
)
from gliquid.plotting.style import (
    ASSESSED_LIQUIDUS_COLOR,
    ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR,
)

# A straight line in (at.%, degC) so an interpolated boundary temperature is exact.
RAMP = [[float(x), 1000.0 + 10.0 * x] for x in range(0, 101, 5)]


def _flatten(runs):
    """Sub-run points concatenated with the shared boundary vertices collapsed."""
    out = []
    for points, _unsupported in runs:
        for point in points:
            if not out or out[-1] != point:
                out.append(point)
    return out


class TestRangeConversion:
    """The ranges arrive in mole fraction and the curve is drawn in at.%."""

    def test_mole_fraction_becomes_at_pct(self):
        assert _normalize_ranges_pct([[0.25, 0.6]]) == [(25.0, 60.0)]

    def test_a_reversed_pair_is_ordered(self):
        assert _normalize_ranges_pct([[0.6, 0.25]]) == [(25.0, 60.0)]

    def test_a_zero_width_range_is_dropped(self):
        assert _normalize_ranges_pct([[0.4, 0.4]]) == []

    def test_none_and_empty_are_both_no_ranges(self):
        assert _normalize_ranges_pct(None) == []
        assert _normalize_ranges_pct([]) == []


class TestSubRunSplit:
    """``_unsupported_liquidus_runs`` over one already-hole-split digitized region."""

    def test_no_ranges_returns_the_segment_unchanged(self):
        runs = _unsupported_liquidus_runs(RAMP, [])
        assert len(runs) == 1
        points, unsupported = runs[0]
        assert unsupported is False
        assert points == RAMP

    def test_an_empty_segment_yields_no_runs(self):
        assert _unsupported_liquidus_runs([], [(20.0, 40.0)]) == []

    def test_a_range_off_the_segment_leaves_one_supported_run(self):
        runs = _unsupported_liquidus_runs(RAMP, [(120.0, 140.0)])
        assert [flag for _pts, flag in runs] == [False]
        assert _flatten(runs) == RAMP

    def test_a_range_covering_everything_leaves_one_unsupported_run(self):
        runs = _unsupported_liquidus_runs(RAMP, [(-1.0, 101.0)])
        assert [flag for _pts, flag in runs] == [True]
        assert _flatten(runs) == RAMP

    def test_one_interior_range_gives_three_alternating_runs(self):
        runs = _unsupported_liquidus_runs(RAMP, [(30.0, 60.0)])
        assert [flag for _pts, flag in runs] == [False, True, False]

    def test_two_ranges_alternate(self):
        runs = _unsupported_liquidus_runs(RAMP, [(10.0, 20.0), (60.0, 80.0)])
        assert [flag for _pts, flag in runs] == [False, True, False, True, False]

    def test_a_range_reaching_the_segment_start_opens_unsupported(self):
        runs = _unsupported_liquidus_runs(RAMP, [(-5.0, 40.0)])
        assert [flag for _pts, flag in runs] == [True, False]

    def test_the_boundary_point_is_the_last_of_one_run_and_the_first_of_the_next(self):
        runs = _unsupported_liquidus_runs(RAMP, [(32.0, 61.0)])
        assert [flag for _pts, flag in runs] == [False, True, False]
        assert runs[0][0][-1] == runs[1][0][0]
        assert runs[1][0][-1] == runs[2][0][0]
        assert runs[0][0][-1][0] == pytest.approx(32.0)
        assert runs[1][0][-1][0] == pytest.approx(61.0)

    def test_the_boundary_temperature_is_interpolated_onto_the_curve(self):
        # 32 and 61 at.% fall BETWEEN samples (the ramp is sampled every 5 at.%), so the
        # shared vertex only lands on the curve if it is interpolated rather than snapped.
        runs = _unsupported_liquidus_runs(RAMP, [(32.0, 61.0)])
        assert runs[0][0][-1][1] == pytest.approx(1000.0 + 10.0 * 32.0)
        assert runs[1][0][-1][1] == pytest.approx(1000.0 + 10.0 * 61.0)

    def test_a_range_narrower_than_the_sampling_still_splits(self):
        # Both edges land inside a single sample interval: the densification has to insert
        # two vertices into that one interval, in order.
        runs = _unsupported_liquidus_runs(RAMP, [(31.0, 34.0)])
        assert [flag for _pts, flag in runs] == [False, True, False]
        assert [p[0] for p in runs[1][0]] == pytest.approx([31.0, 34.0])

    def test_an_edge_landing_on_a_sample_does_not_duplicate_it(self):
        runs = _unsupported_liquidus_runs(RAMP, [(30.0, 60.0)])
        assert [p[0] for p in runs[1][0]] == pytest.approx([float(x) for x in range(30, 61, 5)])

    def test_the_split_is_continuous_and_adds_only_the_boundary_vertices(self):
        runs = _unsupported_liquidus_runs(RAMP, [(32.0, 61.0)])
        rebuilt = _flatten(runs)
        assert [p[0] for p in rebuilt] == sorted(p[0] for p in rebuilt)
        # Every original sample survives, and exactly the two edges are new.
        assert set(map(tuple, RAMP)) <= set(map(tuple, rebuilt))
        added = [p[0] for p in rebuilt if p not in RAMP]
        assert added == pytest.approx([32.0, 61.0])

    def test_a_single_point_segment_is_classified_not_dropped(self):
        assert _unsupported_liquidus_runs([[50.0, 1500.0]], [(40.0, 60.0)]) == [
            ([[50.0, 1500.0]], True)
        ]
        assert _unsupported_liquidus_runs([[50.0, 1500.0]], [(10.0, 20.0)]) == [
            ([[50.0, 1500.0]], False)
        ]


class TestComposedWithHoleSplit:
    """A system with BOTH an undigitized hole and an unsupported range splits on both."""

    # Two digitized wedges either side of a hole far wider than the fill threshold.
    HOLED = [[float(x), 1000.0 + 10.0 * x] for x in range(0, 31, 5)] + [
        [float(x), 1000.0 + 10.0 * x] for x in range(70, 101, 5)
    ]

    def test_the_fixture_really_has_a_hole(self):
        gaps = [b[0] - a[0] for a, b in zip(self.HOLED, self.HOLED[1:])]
        assert max(gaps) > _ASSESSED_GAP_PCT
        assert len(_assessed_liquidus_segments(self.HOLED)) == 2

    def test_the_hole_survives_the_recolor_split(self):
        ranges = [(20.0, 80.0)]
        runs = [
            run
            for segment in _assessed_liquidus_segments(self.HOLED)
            for run in _unsupported_liquidus_runs(segment, ranges)
        ]
        # Left wedge: supported to 20, unsupported 20->30. Right wedge: unsupported
        # 70->80, supported 80->100. Four runs, and none of them crosses the hole.
        assert [flag for _pts, flag in runs] == [False, True, True, False]
        for points, _flag in runs:
            spans = [b[0] - a[0] for a, b in zip(points, points[1:])]
            assert not spans or max(spans) <= _ASSESSED_GAP_PCT

    def test_a_range_sitting_entirely_in_the_hole_changes_nothing(self):
        ranges = [(40.0, 60.0)]
        runs = [
            run
            for segment in _assessed_liquidus_segments(self.HOLED)
            for run in _unsupported_liquidus_runs(segment, ranges)
        ]
        assert [flag for _pts, flag in runs] == [False, False]
        assert [pts for pts, _flag in runs] == _assessed_liquidus_segments(self.HOLED)


def _assessed_liquidus_traces(fig):
    """The drawn assessed-liquidus traces, in draw order.

    Selected by colour alone, so the selector stays valid as the line styles change: the
    legend proxies carry x=[None], and the drawn curve is everything else in one of the two
    assessed reds. Which style each run carries is pinned separately, in TestLineStyle.
    """
    reds = {ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR}
    return [
        trace
        for trace in fig.data
        if trace.line.color in reds
        and trace.x is not None
        and any(value is not None for value in trace.x)
    ]


def _assessed_liquidus_colors(fig):
    """Colours of the drawn assessed-liquidus traces, in draw order."""
    return [trace.line.color for trace in _assessed_liquidus_traces(fig)]


def _render(unsupported_ranges):
    """A fresh Hf-Zr ``fit+liq`` figure; ``plot_tx`` mutates conds, so never reuse one."""
    bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, params=[-15000, 0, 0, 0], param_format="linear")
    bl.update_phase_points()
    if unsupported_ranges is not None:
        bl.unsupported_comp_ranges = unsupported_ranges
    return BLPlotter(bl).get_plot("fit+liq")


@pytest.fixture(scope="module")
def render():
    """``render(ranges)``: the ``_render(ranges)`` figure, built once per module for each
    distinct ``ranges``. The figure is shared: read it, never change it."""
    figures = {}

    def get(unsupported_ranges):
        key = None if unsupported_ranges is None else tuple(map(tuple, unsupported_ranges))
        if key not in figures:
            figures[key] = _render(unsupported_ranges)
        return figures[key]

    return get


class TestFigure:
    """End to end on a packaged system, through the ``binary.py`` dispatch."""

    def test_hf_zr_masks_nothing_by_default(self):
        # Anti-vacuity for the no-op test below: the control side must really be empty.
        bl = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, params=[-15000, 0, 0, 0], param_format="linear"
        )
        assert bl.unsupported_comp_ranges == []

    def test_empty_ranges_render_identically_to_no_ranges(self, render):
        assert render(None).to_json() == render([]).to_json()

    def test_a_masked_system_renders_two_assessed_colors(self, render):
        colors = _assessed_liquidus_colors(render([[0.3, 0.6]]))
        assert set(colors) == {ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR}
        assert colors.count(ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR) == 1

    def test_an_unmasked_system_renders_one_assessed_color(self, render):
        assert set(_assessed_liquidus_colors(render([]))) == {ASSESSED_LIQUIDUS_COLOR}

    def test_the_recolored_curve_covers_the_same_points(self, render):
        """Visual continuity, read off the figure: the split traces join end to end and
        reproduce the unsplit curve plus the two interpolated boundary vertices."""
        plain = render([])
        split = render([[0.3, 0.6]])
        reds = {ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR}

        def curve(fig):
            out = []
            for trace in fig.data:
                if (
                    trace.line.color not in reds
                    or trace.x is None
                    or all(value is None for value in trace.x)
                ):
                    continue
                for x, y in zip(trace.x, trace.y):
                    if not out or out[-1] != (x, y):
                        out.append((x, y))
            return out

        before, after = curve(plain), curve(split)
        assert before  # anti-vacuity: there is an assessed liquidus to split
        assert set(before) <= set(after)
        added = sorted(x for x, _y in set(after) - set(before))
        assert added == pytest.approx([30.0, 60.0])

    def test_no_legend_entry_is_added(self, render):
        names = lambda fig: [t.name for t in fig.data if t.showlegend]  # noqa: E731
        assert names(render([[0.3, 0.6]])) == names(render([]))
        assert ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR not in [
            t.line.color for t in render([[0.3, 0.6]]).data if t.showlegend
        ]


class TestLineStyle:
    """Unsupported runs are dash-dotted as well as lighter, so the distinction survives a
    thumbnail and a greyscale print, where the two reds are nearly the same value."""

    def test_the_shipped_styles_are_dash_and_dashdot(self):
        # Pinned as literals: these are the rendered values, not an indirection.
        assert _ASSESSED_DASH == "dash"
        assert _ASSESSED_UNSUPPORTED_DASH == "dashdot"
        assert _ASSESSED_DASH != _ASSESSED_UNSUPPORTED_DASH  # anti-vacuity for the pairing

    def test_each_run_carries_the_style_that_matches_its_colour(self, render):
        traces = _assessed_liquidus_traces(render([[0.3, 0.6]]))
        styles = {(trace.line.color, getattr(trace.line, "dash", None)) for trace in traces}
        assert styles == {
            (ASSESSED_LIQUIDUS_COLOR, _ASSESSED_DASH),
            (ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR, _ASSESSED_UNSUPPORTED_DASH),
        }

    def test_an_unmasked_system_has_no_dash_dotted_assessed_run(self, render):
        traces = _assessed_liquidus_traces(render([]))
        assert traces  # anti-vacuity: there is an assessed liquidus to style
        assert {getattr(trace.line, "dash", None) for trace in traces} == {_ASSESSED_DASH}

    def test_the_recolored_traces_match_the_others_in_every_other_respect(self, render):
        fig = render([[0.3, 0.6]])
        reds = {ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR}
        styles = {
            (t.line.dash, t.line.width, t.mode, t.showlegend, t.name)
            for t in fig.data
            if getattr(t.line, "dash", None) == "dash"
            and t.line.color in reds
            and t.x is not None
            and any(value is not None for value in t.x)
        }
        assert len(styles) == 1
