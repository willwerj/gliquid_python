import bisect
import copy

import numpy as np
import pandas as pd
import pytest

import gliquid.plotting.binary_tx as binary_tx
from gliquid.binary import (
    BinaryLiquid,
    BLPlotter,
    build_polymorph_transitions,
    plot_tx,
)
from gliquid.hsx import HSX
from gliquid.imputation.binary_imputation import impute_binary_phases
from gliquid.plotting.binary_tx import (
    _BOTTOM_PAD_FRAC,
    _LIQ_PAD_FRAC,
    _PLOT_H_PX,
    _PLOT_W_PX,
    _WIDE_BOX_HALF_W,
    _assign_compound_sides,
    _column_blocked_bands,
    _estimate_label_box,
    _free_intervals,
    _horizontal_anchor,
    _label_metrics,
    _OrientationLatch,
    _pack_labels,
    _pad_range_for_liquidus,
    _place_compound_y,
    _place_liquid_label,
    _resolve_label_collisions,
    _subtract_bands,
)
from gliquid.plotting.style import abbreviate_phase_name

from ._mpds_builders import strip_interior


def _make_hsx() -> HSX:
    data_dict = {
        "phases": ["Mn", "Zr", "L", "alpha Mn (bcc)", "beta Mn (bcc)", "ZrMn2"],
        "comps": ["Mn", "Zr"],
        "data": [
            [0.0, 0.0, 0.0, "Mn"],
            [1.0, 0.0, 0.0, "Zr"],
            [0.0, 1.0, 2.0, "L"],
            [1.0, 1.0, 2.1, "L"],
            [0.0, 0.5, 1.0, "alpha Mn (bcc)"],
            [0.0, 0.7, 1.2, "beta Mn (bcc)"],
            [0.4, 0.6, 1.1, "ZrMn2"],
        ],
    }
    return HSX(data_dict=data_dict, conds=[0.0, 1500.0])


# -- abbreviate_phase_name --
def test_abbreviate_phase_name_rules():
    names = ["alpha Mn (bcc)", "beta Mn (bcc)", "delta Fe (bcc)", "ZrMn2", "Liquid"]

    # Mn has two polymorphs present -> keep greek; structure kept (already short).
    assert abbreviate_phase_name("alpha Mn (bcc)", names) == "α-Mn (bcc)"
    # Fe has a single phase present -> drop greek; structure kept.
    assert abbreviate_phase_name("delta Fe (bcc)", names) == "Fe (bcc)"
    assert abbreviate_phase_name("ZrMn2", names) == "ZrMn<sub>2</sub>"
    assert abbreviate_phase_name("Liquid", names) == "L"
    assert abbreviate_phase_name("L", names) == "L"


def test_abbreviate_phase_name_hyphen_form_and_structure_abbrev():
    # Hyphen-separated greek prefix (real-data form), single polymorph -> drop greek, abbrev struct.
    single = ["alpha-Ga (orthorhombic)", "SmGa2", "L"]
    assert abbreviate_phase_name("alpha-Ga (orthorhombic)", single) == "Ga (ortho)"

    # Two polymorphs of the same element present -> keep greek.
    paired = ["alpha-Ga (orthorhombic)", "beta-Ga (tetragonal)", "SmGa2", "L"]
    assert abbreviate_phase_name("alpha-Ga (orthorhombic)", paired) == "α-Ga (ortho)"
    assert abbreviate_phase_name("beta-Ga (tetragonal)", paired) == "β-Ga (tetra)"

    # Long Strukturbericht structure -> compact token.
    mns = ["beta-Mn (complex cubic A13)", "delta-Mn (bcc)", "alpha-Mn (complex cubic A12)", "L"]
    assert abbreviate_phase_name("beta-Mn (complex cubic A13)", mns) == "β-Mn (A13)"
    assert abbreviate_phase_name("alpha-Mn (complex cubic A12)", mns) == "α-Mn (A12)"
    assert abbreviate_phase_name("delta-Mn (bcc)", mns) == "δ-Mn (bcc)"


def test_abbreviate_phase_name_struct_element_form():
    # An unnamed polymorph -> bare element, structure parenthesised.
    names = ["Al (fcc)", "Cr (bcc)", "In (Fm-3m)", "L"]
    assert abbreviate_phase_name("Al (fcc)", names) == "Al (fcc)"
    assert abbreviate_phase_name("Cr (bcc)", names) == "Cr (bcc)"
    # Space-group symbols are dropped rather than shown as a structure; none ship in
    # phase_transitions.json.
    assert abbreviate_phase_name("In (Fm-3m)", names) == "In"
    assert abbreviate_phase_name("alpha-Pu (P6_3/mmc)", ["alpha-Pu (P6_3/mmc)", "L"]) == "Pu"


def test_abbreviate_phase_name_compound_with_tag():
    names = ["Mn3Sn", "Mn2Sn ht", "MnSn2", "L"]
    assert abbreviate_phase_name("Mn2Sn ht", names) == "Mn<sub>2</sub>Sn ht"
    assert abbreviate_phase_name("Mn3Sn", names) == "Mn<sub>3</sub>Sn"
    assert abbreviate_phase_name("MnSn2", names) == "MnSn<sub>2</sub>"
    assert abbreviate_phase_name("Sm3Ga2", names) == "Sm<sub>3</sub>Ga<sub>2</sub>"


@pytest.mark.parametrize(
    ("name", "components", "expected"),
    [
        ("Fe1.7Ge", None, "Fe<sub>1.7</sub>Ge"),
        ("CeSi1.67", None, "CeSi<sub>1.67</sub>"),
        ("Sr7Pb11.8", None, "Sr<sub>7</sub>Pb<sub>11.8</sub>"),
        ("Zn58.7Y12.65", ["Y", "Zn"], "Y<sub>0.177</sub>Zn<sub>0.823</sub>"),
    ],
)
def test_abbreviate_phase_name_subscripts_a_decimal_run_whole(name, components, expected):
    # No components: the raw name, as in the ternary legend. With components: the binary
    # field label, here one that stays in atomic fractions.
    assert abbreviate_phase_name(name, [name, "L"], components) == expected


def test_abbreviate_phase_name_bare_element():
    # A bare element symbol (ground state with no greek/structure) -> just the symbol.
    assert abbreviate_phase_name("Ga", ["Ga", "SmGa2", "L"]) == "Ga"


def test_abbreviate_phase_name_capitalised_structure_word_form():
    # "diamond cubic" is a real structure, abbreviated rather than dropped.
    names = ["Si (diamond cubic)", "Ge (diamond cubic)", "Cr3Ge", "L"]
    assert abbreviate_phase_name("Si (diamond cubic)", names) == "Si (dc)"
    assert abbreviate_phase_name("Ge (diamond cubic)", names) == "Ge (dc)"
    assert abbreviate_phase_name("C (graphite)", ["C (graphite)", "L"]) == "C (gra)"
    # A genuine compound is unaffected.
    assert abbreviate_phase_name("Cr3Ge", names) == "Cr<sub>3</sub>Ge"
    # Greek polymorph naming must NOT be captured by the structure-word branch.
    assert (
        abbreviate_phase_name("alpha Mn (bcc)", ["alpha Mn (bcc)", "beta Mn (bcc)"]) == "α-Mn (bcc)"
    )


# -- _resolve_label_collisions --
def _vlabel(x, y, text):
    return {
        "x": x,
        "y": y,
        "text": text,
        "xanchor": "center",
        "yanchor": "middle",
        "textangle": -90,
        "font_size": 12,
        "font_color": "black",
    }


def test_resolve_label_collisions_separates_overlaps():
    labels = [_vlabel(10.0, 400.0, "(Mn)"), _vlabel(10.0, 400.0, "ZrMn<sub>2</sub>")]

    resolved = _resolve_label_collisions(
        labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0), max_iterations=20
    )

    assert resolved[0]["y"] != pytest.approx(resolved[1]["y"], abs=1e-6)


def test_resolve_label_collisions_three_stacked():
    labels = [_vlabel(10.0, 400.0, "A"), _vlabel(10.0, 400.0, "B"), _vlabel(10.0, 400.0, "C")]

    resolved = _resolve_label_collisions(
        labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0), max_iterations=50
    )

    ys = sorted(round(lbl["y"], 4) for lbl in resolved)
    assert len(set(ys)) == 3, "all three labels should end at distinct y positions"
    # displaced labels should be flagged for a leader arrow
    assert any(lbl.get("showarrow") for lbl in resolved)


def test_resolve_label_collisions_keeps_non_overlapping_put():
    labels = [_vlabel(10.0, 200.0, "A"), _vlabel(90.0, 1200.0, "B")]
    resolved = _resolve_label_collisions(labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0))
    assert resolved[0]["y"] == pytest.approx(200.0)
    assert resolved[1]["y"] == pytest.approx(1200.0)
    assert not any(lbl.get("showarrow") for lbl in resolved)


# -- _estimate_label_box (lock the fragile font/pixel constants) --
def test_estimate_label_box_constants():
    label = _vlabel(50.0, 750.0, "Xx")  # 2 glyphs, vertical text
    cx, cy, half_w, half_h = _estimate_label_box(label, xlim=(0.0, 100.0), ylim=(0.0, 1500.0))

    # Vertical text: width is driven by line height, height by glyph advance.
    # w_px = 1.2*12 = 14.4 -> half_w = (14.4 / (615/100)) / 2
    # h_px = 2*0.6*12 = 14.4 -> half_h = (14.4 / (456/1500)) / 2
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(750.0)
    assert half_w == pytest.approx((14.4 / (_PLOT_W_PX / 100.0)) / 2.0, abs=1e-6)
    assert half_h == pytest.approx((14.4 / (_PLOT_H_PX / 1500.0)) / 2.0, abs=1e-6)


def test_estimate_label_box_subscripts_count_less():
    plain = _vlabel(50.0, 750.0, "ABC")
    subbed = _vlabel(50.0, 750.0, "A<sub>BC</sub>")
    # Horizontal text so width reflects glyph count directly.
    plain["textangle"] = 0
    subbed["textangle"] = 0
    _, _, hw_plain, _ = _estimate_label_box(plain, (0.0, 100.0), (0.0, 1500.0))
    _, _, hw_sub, _ = _estimate_label_box(subbed, (0.0, 100.0), (0.0, 1500.0))
    assert hw_sub < hw_plain


# -- _label_metrics (both-orientation wrapper around _estimate_label_box) --
def test_label_metrics_vertical_matches_estimate_label_box():
    xlim, ylim = (0.0, 100.0), (0.0, 1500.0)
    (half_w, half_h_frac), _ = _label_metrics("ZrMn<sub>2</sub>", xlim, ylim, font_size=12)
    probe = {
        "x": 0.0,
        "y": 0.0,
        "text": "ZrMn<sub>2</sub>",
        "textangle": -90,
        "font_size": 12,
        "xanchor": "center",
        "yanchor": "middle",
    }
    _, _, exp_half_w, exp_half_h = _estimate_label_box(probe, xlim, ylim)
    assert half_w == pytest.approx(exp_half_w)
    assert half_h_frac == pytest.approx(exp_half_h / (ylim[1] - ylim[0]))


def test_label_metrics_horizontal_matches_estimate_label_box():
    xlim, ylim = (0.0, 100.0), (0.0, 1500.0)
    _, (half_w, half_h_frac) = _label_metrics("ZrMn<sub>2</sub>", xlim, ylim, font_size=12)
    probe = {
        "x": 0.0,
        "y": 0.0,
        "text": "ZrMn<sub>2</sub>",
        "textangle": 0,
        "font_size": 12,
        "xanchor": "center",
        "yanchor": "middle",
    }
    _, _, exp_half_w, exp_half_h = _estimate_label_box(probe, xlim, ylim)
    assert half_w == pytest.approx(exp_half_w)
    assert half_h_frac == pytest.approx(exp_half_h / (ylim[1] - ylim[0]))


def test_label_metrics_orientations_swap_half_w_and_half_h():
    # A vertical label is TALL and NARROW; horizontal is WIDE and SHORT for the same text.
    (v_half_w, v_half_h_frac), (h_half_w, h_half_h_frac) = _label_metrics(
        "AgTh<sub>2</sub>", (0.0, 100.0), (0.0, 1500.0), font_size=12
    )
    assert h_half_w > v_half_w
    assert h_half_h_frac < v_half_h_frac


# -- _assign_compound_sides: distinct two-phase gaps for clustered compounds --
def _gap_index(x, comps, xlim=(0.0, 100.0)):
    """Which inter-compound interval of ``[xlim0] + sorted(comps) + [xlim1]`` contains x."""
    bounds = [xlim[0]] + sorted(comps) + [xlim[1]]
    return bisect.bisect_right(bounds, x) - 1


def test_assign_compound_sides_clustered_get_distinct_gaps():
    comps = [19.0, 21.0, 24.0, 30.0]
    half_ws = [1.17] * 4
    xs = _assign_compound_sides(comps, half_ws, xlim=(0.0, 100.0))
    gaps = [_gap_index(x, comps) for x in xs]
    assert len(set(gaps)) == len(comps), f"labels share a gap: xs={xs}, gaps={gaps}"


def test_assign_compound_sides_edge_compounds_forced_inward():
    comps = [5.0, 95.0]
    half_ws = [1.17, 1.17]
    xs = _assign_compound_sides(comps, half_ws, xlim=(0.0, 100.0))
    assert xs[0] > 5.0, "compound near the left edge must place its label to the right (inward)"
    assert xs[1] < 95.0, "compound near the right edge must place its label to the left (inward)"


# -- _free_intervals (shared by _pack_labels and _place_compound_y) --
def test_free_intervals_no_cuts_returns_whole_range():
    assert _free_intervals(0.0, 100.0, [], 5.0) == [(0.0, 100.0)]


def test_free_intervals_ignores_cuts_outside_the_open_range():
    assert _free_intervals(0.0, 100.0, [0.0, 100.0, -10.0, 200.0], 5.0) == [(0.0, 100.0)]


def test_free_intervals_splits_around_a_single_cut():
    assert _free_intervals(0.0, 100.0, [40.0], 5.0) == [(0.0, 35.0), (45.0, 100.0)]


def test_free_intervals_splits_around_multiple_cuts_in_sorted_order():
    # Cuts given out of order; the split still proceeds low to high.
    assert _free_intervals(0.0, 100.0, [70.0, 30.0], 5.0) == [
        (0.0, 25.0),
        (35.0, 65.0),
        (75.0, 100.0),
    ]


def test_free_intervals_drops_a_band_a_cut_margin_collapses():
    # Two cuts 6 apart with margin 5: the middle band (2..4) would need lo > hi, so it is
    # omitted rather than returned inverted.
    assert _free_intervals(0.0, 100.0, [10.0, 16.0], 5.0) == [(0.0, 5.0), (21.0, 100.0)]


def test_free_intervals_merges_adjacent_cuts_within_one_margin():
    # Two cuts closer together than 2*margin: the shared band is consumed by the running
    # cursor (lo) rather than producing a negative-width interval.
    assert _free_intervals(0.0, 100.0, [10.0, 12.0], 5.0) == [(0.0, 5.0), (17.0, 100.0)]


def test_free_intervals_empty_when_ceiling_at_or_below_floor():
    assert _free_intervals(100.0, 100.0, [], 5.0) == []
    assert _free_intervals(100.0, 50.0, [], 5.0) == []


# -- _place_compound_y: bottom pad, and skipping up past a blocking tie --
def test_place_compound_y_bottom_pad_when_clear():
    # No tie lines: the label sits at the bottom region with exactly ``pad`` above the floor.
    pad, half_h, margin = 15.0, 50.0, 12.0
    y = _place_compound_y(
        c0=0.0, ceiling=1000.0, tie_temps=[], half_h=half_h, pad=pad, margin=margin
    )
    assert y == pytest.approx(pad + half_h)  # box bottom == c0 + pad


def test_place_compound_y_skips_blocked_bottom_region():
    # A tie at 40 makes the bottom region (0..28) too short; the label must lift into the next
    # tie-free region above the tie, not sit inside the tie.
    y = _place_compound_y(
        c0=0.0, ceiling=1000.0, tie_temps=[40.0], half_h=50.0, pad=15.0, margin=12.0
    )
    assert y is not None
    assert y - 50.0 > 40.0, "label box must clear the blocking tie at T=40"
    assert y == pytest.approx(52.0 + 50.0)  # bottom of the (52..1000) region + half_h


def test_place_compound_y_returns_none_when_nothing_fits():
    # Ceiling barely above the floor: no region is tall enough for the label.
    y = _place_compound_y(c0=0.0, ceiling=20.0, tie_temps=[], half_h=50.0, pad=15.0, margin=12.0)
    assert y is None


# -- _resolve_label_collisions bottom pad --
def test_resolve_label_collisions_applies_bottom_pad():
    ylim = (0.0, 1500.0)
    # A bottom-anchored compound label placed flush with the floor should be lifted by the pad.
    label = _vlabel(50.0, 0.0, "ZrMn<sub>2</sub>")
    label["yanchor"] = "bottom"
    resolved = _resolve_label_collisions([label], xlim=(0.0, 100.0), ylim=ylim)
    _, cy, _, half_h = _estimate_label_box(resolved[0], (0.0, 100.0), ylim)
    box_bottom = cy - half_h
    assert box_bottom == pytest.approx(ylim[0] + _BOTTOM_PAD_FRAC * (ylim[1] - ylim[0]), abs=1e-6)


# -- plot_tx: one monotone path per solid-phase trace --
def _is_monotone(values: list) -> bool:
    """True if ``values`` is non-decreasing throughout or non-increasing throughout."""
    return all(a <= b for a, b in zip(values, values[1:])) or all(
        a >= b for a, b in zip(values, values[1:])
    )


class TestSolidPhaseTraceMonotonicity:
    """Each solid-phase trace on a fit+liq figure is drawn from ``df_tx``, which is built
    in Qhull facet-enumeration order rather than temperature order, and gets a low-extension
    point appended at the end regardless of its temperature. Every such trace must still read
    back as one monotone path in ``t``.
    """

    SOLID_PHASE_NAMES = {"alpha-Hf (hcp)", "beta-Hf (bcc)", "alpha-Zr (hcp)", "beta-Zr (bcc)"}

    @pytest.fixture(scope="module")
    def hf_zr_fit_liq(self):
        bl = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
        )
        return BLPlotter(bl).get_plot("fit+liq")

    def test_every_solid_phase_trace_is_monotone(self, hf_zr_fit_liq):
        seen = {tr.name for tr in hf_zr_fit_liq.data if tr.name in self.SOLID_PHASE_NAMES}
        assert seen == self.SOLID_PHASE_NAMES
        for trace in hf_zr_fit_liq.data:
            if trace.name in self.SOLID_PHASE_NAMES:
                ys = list(trace.y)
                assert _is_monotone(ys), f"{trace.name} trace is not monotone in t: {ys}"


class TestImputedPhaseTraceIsSingleMonotonePath:
    """An imputed phase's boundary is drawn dashed; it must stay one monotone path too, so
    the dash pattern is not doubled over part of its range.
    """

    @pytest.fixture(scope="module")
    def imputed_cu_mg(self, cu_mg_full):
        bl = strip_interior(copy.deepcopy(cu_mg_full))
        if not impute_binary_phases(bl, fitting_data=None, write_cache=False):
            pytest.skip("Cu-Mg: nothing was imputed")
        return bl

    def test_imputed_trace_is_monotone_and_uniformly_dashed(self, imputed_cu_mg):
        bl = imputed_cu_mg
        imputed_names = {p.name for p in bl.phases if p.imputed}
        assert imputed_names
        fig = BLPlotter(bl).get_plot("fit+liq")
        found = 0
        for trace in fig.data:
            if trace.name in imputed_names:
                found += 1
                ys = list(trace.y)
                assert _is_monotone(ys), f"{trace.name} trace is not monotone in t: {ys}"
                dash = trace.line.dash if trace.line is not None else None
                assert dash == "4px,2px"
        assert found == len(imputed_names)

    def test_the_legend_key_carries_the_same_dash(self, imputed_cu_mg):
        fig = BLPlotter(imputed_cu_mg).get_plot("fit+liq")
        (key,) = [tr for tr in fig.data if tr.name == "Imputed"]
        assert key.line.dash == "4px,2px"


class TestGetPlotLiquidusKeywords:
    """The liquidus figure types take no keyword arguments, as ``'pc'`` does: a keyword passed
    to ``get_plot`` is ignored and the figure is the one drawn without it.
    """

    @pytest.fixture(scope="class")
    def plotter(self):
        bl = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
        )
        return BLPlotter(bl)

    @pytest.mark.parametrize("plot_type", ["fit+liq"])
    def test_keywords_leave_the_figure_unchanged(self, plotter, plot_type):
        base = plotter.get_plot(plot_type)
        fig = plotter.get_plot(plot_type, width=900)
        assert [tr.name for tr in fig.data] == [tr.name for tr in base.data]


class TestPadRangeForLiquidus:
    """``_pad_range_for_liquidus`` widens a temperature range until the generated liquidus
    clears both bounds by ``pad_frac`` of the final span, and never narrows it.
    """

    P = _LIQ_PAD_FRAC
    TOL = 1e-9

    def _assert_contract(self, conds_in, max_liq, min_liq, out):
        c0, c1 = out
        span = c1 - c0
        assert c0 <= conds_in[0] + self.TOL
        assert c1 >= conds_in[1] - self.TOL
        assert c1 - max_liq >= self.P * span - self.TOL
        assert min_liq - c0 >= self.P * span - self.TOL

    def test_already_satisfied_is_unchanged(self):
        conds_in = (-500.0, 1500.0)
        out = _pad_range_for_liquidus(conds_in, max_liq=1000.0, min_liq=0.0)
        assert out == pytest.approx(conds_in)
        self._assert_contract(conds_in, 1000.0, 0.0, out)

    def test_needs_top_only(self):
        conds_in = (-500.0, 1000.0)
        out = _pad_range_for_liquidus(conds_in, max_liq=1000.0, min_liq=0.0)
        assert out[0] == pytest.approx(-500.0)
        assert out[1] > 1000.0
        self._assert_contract(conds_in, 1000.0, 0.0, out)

    def test_needs_bottom_only(self):
        conds_in = (0.0, 2000.0)
        out = _pad_range_for_liquidus(conds_in, max_liq=1000.0, min_liq=0.0)
        assert out[1] == pytest.approx(2000.0)
        assert out[0] < 0.0
        self._assert_contract(conds_in, 1000.0, 0.0, out)

    def test_needs_both_matches_closed_form(self):
        conds_in = (0.0, 1000.0)
        max_liq, min_liq = 1000.0, 0.0
        out = _pad_range_for_liquidus(conds_in, max_liq=max_liq, min_liq=min_liq)
        denom = 1.0 - 2.0 * self.P
        assert out[0] == pytest.approx(((1 - self.P) * min_liq - self.P * max_liq) / denom)
        assert out[1] == pytest.approx(((1 - self.P) * max_liq - self.P * min_liq) / denom)
        assert out[1] - max_liq == pytest.approx(self.P * (out[1] - out[0]))
        assert min_liq - out[0] == pytest.approx(self.P * (out[1] - out[0]))
        self._assert_contract(conds_in, max_liq, min_liq, out)

    def test_lopsided_floor_still_clears_the_top(self):
        """A reference floor far below the liquidus inflates the span; the two-sided closed
        form alone would leave the top clearance under ``pad_frac`` of it."""
        conds_in = (700.0, 1700.0)
        max_liq, min_liq = 1670.0, 1100.0
        denom = 1.0 - 2.0 * self.P
        two_sided_c1 = ((1 - self.P) * max_liq - self.P * min_liq) / denom
        assert two_sided_c1 - max_liq < self.P * (two_sided_c1 - conds_in[0])
        out = _pad_range_for_liquidus(conds_in, max_liq=max_liq, min_liq=min_liq)
        assert out[0] == pytest.approx(700.0)
        self._assert_contract(conds_in, max_liq, min_liq, out)

    def test_lopsided_ceiling_still_clears_the_bottom(self):
        conds_in = (600.0, 6000.0)
        max_liq, min_liq = 3185.0, 1400.0
        out = _pad_range_for_liquidus(conds_in, max_liq=max_liq, min_liq=min_liq)
        assert out[1] == pytest.approx(6000.0)
        self._assert_contract(conds_in, max_liq, min_liq, out)

    def test_flat_liquidus_collapses_to_the_input_bounds(self):
        conds_in = (0.0, 1000.0)
        out = _pad_range_for_liquidus(conds_in, max_liq=500.0, min_liq=500.0)
        self._assert_contract(conds_in, 500.0, 500.0, out)

    def test_non_finite_extremes_leave_conds_alone(self):
        conds_in = (0.0, 1000.0)
        assert _pad_range_for_liquidus(conds_in, np.nan, 0.0) == pytest.approx(conds_in)
        assert _pad_range_for_liquidus(conds_in, 1000.0, -np.inf) == pytest.approx(conds_in)

    @pytest.mark.parametrize("c0_in,c1_in", [(-4000.0, 400.0), (900.0, 9000.0), (940.0, 960.0)])
    @pytest.mark.parametrize("max_liq,min_liq", [(1000.0, 0.0), (950.0, 949.0), (2719.0, -38.8)])
    def test_contract_holds_over_a_grid(self, c0_in, c1_in, max_liq, min_liq):
        conds_in = (c0_in, c1_in)
        out = _pad_range_for_liquidus(conds_in, max_liq=max_liq, min_liq=min_liq)
        self._assert_contract(conds_in, max_liq, min_liq, out)


class TestPlotTxTemperatureRange:
    """``plot_tx`` establishes the plotted temperature range from three rules: the reference
    range is a bound it may widen past but never narrow inside, the generated liquidus is
    cleared from both bounds by ``_LIQ_PAD_FRAC`` of the final span, and a solid phase only
    lowers the floor while it is still stable at or above it.
    """

    TOL = 1e-6

    @pytest.fixture(scope="class")
    def hf_zr(self):
        return BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
        )

    @staticmethod
    def _render(bl, conds_in, mpds):
        """Render fit+liq with an explicit starting range; returns (conds_out, hsx)."""
        bl.hsx = None
        bl.update_phase_points()
        bl.hsx.conds = list(conds_in)
        fig = plot_tx(
            bl.hsx,
            polymorph_transitions=build_polymorph_transitions(bl),
            components=bl.components,
            mpds_temp_range=mpds,
        )
        return [float(v) for v in fig.layout.yaxis.range], bl.hsx

    @staticmethod
    def _liquidus_extremes(hsx):
        """The generated liquidus min/max in °C, as ``plot_tx`` reads them."""
        liq = hsx.df_tx[hsx.df_tx["label"] == "L"].copy()
        liq.sort_values(by=["x", "t"], inplace=True)
        liq.drop_duplicates(subset="x", keep="first", inplace=True)
        return float(liq["t"].min()), float(liq["t"].max())

    @staticmethod
    def _phase_top(hsx, name):
        sub = hsx.df_tx[hsx.df_tx["label"] == name]
        assert not sub.empty, f"{name} is not in the computed diagram"
        return float(sub["t"].max())

    def test_plotted_range_contains_the_reference(self, hf_zr):
        conds, _ = self._render(hf_zr, [800.0, 2400.0], (800.0, 2400.0))
        assert conds[0] <= 800.0 + self.TOL
        assert conds[1] >= 2400.0 - self.TOL

    def test_reference_floor_only_widens(self, hf_zr):
        """A reference minimum ABOVE the starting floor does not raise it."""
        conds, _ = self._render(hf_zr, [800.0, 2400.0], (1000.0, 2400.0))
        assert conds[0] <= 800.0 + self.TOL

    def test_headroom_cap_does_not_narrow_below_the_reference(self, hf_zr):
        conds, hsx = self._render(hf_zr, [800.0, 2400.0], (800.0, 3000.0))
        _, max_liq = self._liquidus_extremes(hsx)
        # Padding the liquidus alone would put the top well under the reference maximum.
        assert max_liq + 0.18 * (max_liq - conds[0]) < 3000.0
        assert conds[1] >= 3000.0 - self.TOL

    @pytest.mark.parametrize(
        "conds_in,mpds",
        [
            ([800.0, 2400.0], (800.0, 2400.0)),
            ([800.0, 2400.0], None),
            # A reference range the liquidus runs straight out of at both ends.
            ([1100.0, 2200.0], (1100.0, 2200.0)),
        ],
    )
    def test_generated_liquidus_clears_both_bounds(self, hf_zr, conds_in, mpds):
        conds, hsx = self._render(hf_zr, conds_in, mpds)
        min_liq, max_liq = self._liquidus_extremes(hsx)
        span = conds[1] - conds[0]
        assert (conds[1] - max_liq) / span >= _LIQ_PAD_FRAC - self.TOL
        assert (min_liq - conds[0]) / span >= _LIQ_PAD_FRAC - self.TOL

    def test_phase_below_the_floor_does_not_lower_it(self, hf_zr):
        """alpha-Zr decomposes far below a 1900 °C floor, so it may not pull the floor down
        to meet itself; the range settles on the liquidus padding instead."""
        conds, hsx = self._render(hf_zr, [1900.0, 2400.0], (1900.0, 2400.0))
        assert conds[0] > self._phase_top(hsx, "alpha-Zr (hcp)")

    def test_phase_inside_the_window_does_lower_the_floor(self, hf_zr):
        """The same phase, with a floor it is still stable above, does lower the range."""
        conds, hsx = self._render(hf_zr, [800.0, 2400.0], (800.0, 2400.0))
        assert conds[0] < self._phase_top(hsx, "alpha-Zr (hcp)")

    def test_label_lowering_is_a_fixed_point(self, hf_zr):
        """Re-rendering from the lowered range lowers it no further: the escalation solves
        for a floor whose own span makes the label fit, rather than ratcheting down."""
        once, _ = self._render(hf_zr, [800.0, 2400.0], (800.0, 2400.0))
        twice, _ = self._render(hf_zr, once, (800.0, 2400.0))
        assert twice[0] == pytest.approx(once[0], abs=1e-6)
        assert twice[1] == pytest.approx(once[1], abs=1e-6)


# -- _subtract_bands (the blocked= support shared by _pack_labels/_place_compound_y) --
def test_subtract_bands_is_identity_without_blocked():
    intervals = [(0.0, 100.0), (200.0, 300.0)]
    assert _subtract_bands(intervals, None) == intervals
    assert _subtract_bands(intervals, []) == intervals


def test_subtract_bands_splits_and_trims():
    assert _subtract_bands([(0.0, 100.0)], [(20.0, 30.0), (50.0, 60.0)]) == [
        (0.0, 20.0),
        (30.0, 50.0),
        (60.0, 100.0),
    ]
    assert _subtract_bands([(0.0, 100.0)], [(-10.0, 20.0)]) == [(20.0, 100.0)]
    assert _subtract_bands([(0.0, 100.0)], [(80.0, 200.0)]) == [(0.0, 80.0)]


def test_subtract_bands_drops_a_fully_covered_interval():
    assert _subtract_bands([(10.0, 20.0), (40.0, 50.0)], [(0.0, 30.0)]) == [(40.0, 50.0)]


def test_pack_labels_blocked_withholds_a_band():
    regions, half_hs = [(0.0, 1000.0)], [50.0]
    plain = _pack_labels(regions, half_hs, 1000.0, [], 0.0, 12.0)
    blocked = _pack_labels(regions, half_hs, 1000.0, [], 0.0, 12.0, blocked=[(0.0, 600.0)])
    assert plain[0] == pytest.approx(500.0)
    assert blocked[0] is not None
    assert blocked[0] > plain[0], "the withheld band must push the label above it"
    assert blocked[0] - 50.0 >= 600.0


def test_place_compound_y_blocked_withholds_the_bottom_region():
    plain = _place_compound_y(0.0, 1000.0, [], half_h=50.0, pad=15.0, margin=12.0)
    blocked = _place_compound_y(
        0.0, 1000.0, [], half_h=50.0, pad=15.0, margin=12.0, blocked=[(0.0, 400.0)]
    )
    assert plain == pytest.approx(65.0)
    assert blocked is not None and blocked - 50.0 >= 400.0


# -- _column_blocked_bands (a compound trace inside a polymorph label's own column) --
def test_column_blocked_bands_blocks_a_trace_inside_the_column():
    traces = [{"comp": 2.0, "t_min": -100.0, "t_max": 900.0}]
    assert _column_blocked_bands(0.33, 2.67, traces, clear=0.5) == [(-100.0, 900.0)]


def test_column_blocked_bands_ignores_a_trace_outside_the_column():
    traces = [{"comp": 40.0, "t_min": -100.0, "t_max": 900.0}]
    assert _column_blocked_bands(0.33, 2.67, traces, clear=0.5) == []


def test_column_blocked_bands_honors_the_clear_margin():
    traces = [{"comp": 3.0, "t_min": 0.0, "t_max": 500.0}]
    assert _column_blocked_bands(0.33, 2.67, traces, clear=0.1) == []
    assert _column_blocked_bands(0.33, 2.67, traces, clear=0.5) == [(0.0, 500.0)]


def test_column_blocked_bands_includes_every_matching_trace_imputed_or_not():
    traces = [
        {"comp": 1.5, "t_min": -273.15, "t_max": 800.0, "imputed": True},
        {"comp": 2.2, "t_min": 100.0, "t_max": 300.0},
        {"comp": 50.0, "t_min": 0.0, "t_max": 1000.0},
    ]
    assert _column_blocked_bands(0.33, 2.67, traces, clear=0.5) == [
        (-273.15, 800.0),
        (100.0, 300.0),
    ]


def test_column_blocked_bands_empty_for_no_traces():
    assert _column_blocked_bands(0.33, 2.67, [], clear=0.5) == []


# -- plot_tx final emission: a 'lower' placement that never resolves to a concrete y --
class TestStuckLowerFallsBackToFloat:
    """A polymorph label whose in-band placement never succeeds, however far ``conds[0]``
    is lowered, still emits at a finite position instead of raising ``KeyError``."""

    @pytest.fixture
    def hf_zr(self):
        return BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, params=[-15000.0, 0.0, 0.0, 0.0], param_format="linear"
        )

    def test_a_permanently_unplaceable_polymorph_label_floats(self, hf_zr, monkeypatch):
        def never_places(
            regions, half_hs, ceiling, tie_temps, floor, margin, gap=0.0, blocked=None
        ):
            return [None] * len(regions)

        monkeypatch.setattr(binary_tx, "_pack_labels", never_places)
        hf_zr.hsx = None
        hf_zr.update_phase_points()
        fig = plot_tx(
            hf_zr.hsx,
            polymorph_transitions=build_polymorph_transitions(hf_zr),
            components=hf_zr.components,
        )
        floated = [a for a in fig.layout.annotations if a.showarrow]
        assert len(floated) == 4, "all four Hf/Zr polymorph labels should float"
        assert all(np.isfinite(a.y) for a in floated)


# -- _horizontal_anchor (a horizontal label's band, growing inward from an element edge) --
def test_horizontal_anchor_sides_read_away_from_their_edge():
    lo = _horizontal_anchor(0, half_w=5.0, obstacles=[], obstacle_half_w=1.2)
    hi = _horizontal_anchor(100, half_w=5.0, obstacles=[], obstacle_half_w=1.2)
    assert lo == (1.5, "left")  # box [1.5, 11.5], reading rightward off the 0 edge
    assert hi == (98.5, "right")  # box [88.5, 98.5], reading leftward off the 100 edge


def test_horizontal_anchor_steps_inward_past_an_obstacle():
    ax, xanchor = _horizontal_anchor(100, half_w=5.0, obstacles=[98.5], obstacle_half_w=1.2)
    assert xanchor == "right"
    assert ax <= 98.5 - 1.2 - 0.4 + 1e-9
    assert ax > 50.0


def test_horizontal_anchor_clears_a_compound_column_on_both_flanks():
    ax, _ = _horizontal_anchor(100, half_w=5.0, obstacles=[98.5, 75.0], obstacle_half_w=1.2)
    box_lo, box_hi = ax - 10.0, ax
    for o in (98.5, 75.0):
        assert o + 1.6 <= box_lo or o - 1.6 >= box_hi


def test_horizontal_anchor_refuses_a_box_wider_than_its_half_of_the_axis():
    assert _horizontal_anchor(0, half_w=30.0, obstacles=[], obstacle_half_w=1.2) is None
    assert _horizontal_anchor(100, half_w=30.0, obstacles=[], obstacle_half_w=1.2) is None


def test_horizontal_anchor_gives_up_before_the_midpoint():
    obstacles = [float(x) for x in range(0, 52, 2)]
    assert _horizontal_anchor(0, half_w=5.0, obstacles=obstacles, obstacle_half_w=1.2) is None


def test_horizontal_anchor_never_crosses_the_midpoint():
    for side in (0, 100):
        band = _horizontal_anchor(side, half_w=20.0, obstacles=[], obstacle_half_w=1.2)
        assert band is not None
        ax, xanchor = band
        box_lo, box_hi = (ax, ax + 40.0) if xanchor == "left" else (ax - 40.0, ax)
        assert 0.0 <= box_lo <= box_hi <= 100.0
        assert box_hi <= 50.0 or box_lo >= 50.0


# -- _OrientationLatch (one-way V->H; the layout fixed point's non-oscillation guarantee) --
def test_orientation_latch_defaults_to_vertical():
    latch = _OrientationLatch()
    assert latch.angle("a", False) == -90
    assert not latch.is_horizontal("a")


def test_orientation_latch_is_absorbing():
    latch = _OrientationLatch()
    assert latch.angle("a", True) == 0
    for prefer in (False, True, False, False, True):
        assert latch.angle("a", prefer) == 0
    assert latch.is_horizontal("a")


def test_orientation_latch_is_monotone_over_a_sequence_of_preferences():
    """A key's angle never returns to vertical, whatever preference sequence it is asked,
    so the joint (conds, latch) state of the layout fixed point cannot oscillate."""
    latch = _OrientationLatch()
    seen = [latch.angle("k", prefer) for prefer in (False, False, True, False, True, False)]
    assert seen == [-90, -90, 0, 0, 0, 0]
    assert all(b >= a for a, b in zip(seen, seen[1:])), "angle is non-decreasing (-90 -> 0)"


def test_orientation_latch_keys_are_independent():
    latch = _OrientationLatch()
    latch.angle(("poly", 100, 0), True)
    assert latch.angle(("poly", 100, 1), False) == -90
    assert latch.angle(("cmpd", 0), False) == -90
    assert latch.angle(("poly", 100, 0), False) == 0


# -- Mixed-orientation label boxes in _resolve_label_collisions --
def _hlabel(x, y, text, xanchor="center"):
    return {
        "x": x,
        "y": y,
        "text": text,
        "xanchor": xanchor,
        "yanchor": "middle",
        "textangle": 0,
        "font_size": 12,
        "font_color": "black",
    }


def test_resolve_label_collisions_leaves_x_alone_for_disjoint_horizontal_boxes():
    """Horizontal boxes sharing a y band but not an x band stay exactly where they are:
    x-disjointness is guaranteed at the layout stage, so there is no x-separation pass."""
    labels = [_hlabel(10.0, 700.0, "alpha-Mn"), _hlabel(85.0, 700.0, "beta-Mn")]
    resolved = _resolve_label_collisions(
        labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0), max_iterations=20
    )
    assert resolved[0]["x"] == pytest.approx(10.0)
    assert resolved[1]["x"] == pytest.approx(85.0)
    assert resolved[0]["y"] == pytest.approx(700.0)
    assert resolved[1]["y"] == pytest.approx(700.0)
    assert not any(lbl.get("showarrow") for lbl in resolved)


def test_resolve_label_collisions_separates_overlapping_horizontal_boxes_in_y_only():
    labels = [_hlabel(50.0, 700.0, "alpha-Mn"), _hlabel(52.0, 700.0, "beta-Mn")]
    resolved = _resolve_label_collisions(
        labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0), max_iterations=20
    )
    assert resolved[0]["x"] == pytest.approx(50.0)
    assert resolved[1]["x"] == pytest.approx(52.0)
    assert resolved[0]["y"] != pytest.approx(resolved[1]["y"], abs=1e-6)


def test_apply_ceiling_samples_a_wide_box_between_its_cardinal_positions():
    """A box wider than ``_WIDE_BOX_HALF_W`` is sampled on a finer grid, so a liquidus dip
    strictly between ``cx - half_w``, ``cx`` and ``cx + half_w`` still pulls the label down."""
    text = "ABCDEFGH"
    _, (half_w, _) = _label_metrics(text, (0.0, 100.0), (0.0, 1500.0), font_size=12)
    assert half_w > _WIDE_BOX_HALF_W
    cx = 50.0
    dip_lo, dip_hi = cx - 0.55 * half_w, cx - 0.45 * half_w
    for cardinal in (cx - half_w, cx, cx + half_w):
        assert not dip_lo <= cardinal <= dip_hi, "the dip must miss every cardinal sample"

    def ceiling(x):
        return 500.0 if dip_lo <= x <= dip_hi else 1400.0

    resolved = _resolve_label_collisions(
        [_hlabel(cx, 1200.0, text)], xlim=(0.0, 100.0), ylim=(0.0, 1500.0), ceiling=ceiling
    )
    _, cy, _, half_h = _estimate_label_box(resolved[0], (0.0, 100.0), (0.0, 1500.0))
    assert cy + half_h < 500.0, "the box top must clear the dip the finer grid found"


def test_apply_ceiling_keeps_three_samples_for_a_narrow_box():
    """A vertical label's box never reaches the wide-box threshold, so it keeps the three
    cardinal samples and a dip between them does not move it."""
    text = "Mn"
    (half_w, _), _ = _label_metrics(text, (0.0, 100.0), (0.0, 1500.0), font_size=12)
    assert half_w <= _WIDE_BOX_HALF_W
    cx = 50.0
    dip_lo, dip_hi = cx - 0.55 * half_w, cx - 0.45 * half_w

    def ceiling(x):
        return 500.0 if dip_lo <= x <= dip_hi else 1400.0

    resolved = _resolve_label_collisions(
        [_vlabel(cx, 1200.0, text)], xlim=(0.0, 100.0), ylim=(0.0, 1500.0), ceiling=ceiling
    )
    _, cy, _, half_h = _estimate_label_box(resolved[0], (0.0, 100.0), (0.0, 1500.0))
    assert cy + half_h > 500.0


# -- _place_liquid_label blocked= (the floated-label stack holds the L label off) --
def _flat_liquidus(temp: float = 500.0) -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0.0, 100.0, 51), "t": [temp] * 51})


def test_place_liquid_label_ignores_an_empty_blocked_list():
    liq = _flat_liquidus()
    plain = _place_liquid_label(liq, None, (0.0, 100.0), (0.0, 1500.0))
    same = _place_liquid_label(liq, None, (0.0, 100.0), (0.0, 1500.0), blocked=[])
    assert plain == same


def test_place_liquid_label_avoids_a_blocked_span():
    liq = _flat_liquidus()
    x_plain, _ = _place_liquid_label(liq, None, (0.0, 100.0), (0.0, 1500.0))
    x_blocked, _ = _place_liquid_label(
        liq, None, (0.0, 100.0), (0.0, 1500.0), blocked=[(30.0, 100.0)]
    )
    assert x_plain >= 30.0, "a flat liquidus centres the label inside the span blocked below"
    assert x_blocked < 30.0


def test_place_liquid_label_falls_back_when_everything_is_blocked():
    liq = _flat_liquidus()
    x, y = _place_liquid_label(liq, None, (0.0, 100.0), (0.0, 1500.0), blocked=[(-10.0, 110.0)])
    assert 0.0 <= x <= 100.0
    assert y > 500.0


def _compound_hsx(h_ab: float, s_ab: float) -> HSX:
    rows = [[0.0, 0.0, 0.0, "A"], [1.0, 0.0, 0.0, "B"]]
    for x in np.linspace(0.0, 1.0, 21):
        rows.append([float(x), 10 + 8 * x * (1 - x), 10000 - 4000 * x * (1 - x), "L"])
    rows.append([0.5, s_ab, h_ab, "AB"])
    return HSX({"phases": ["A", "B", "L", "AB"], "comps": ["A", "B"], "data": rows}, [0.0, 1000.0])


@pytest.mark.parametrize(
    "h_ab, s_ab, bottom",
    [(2000.0, 5.0, 2000.0 / 5.0 - 273.15), (-2000.0, 0.0, -273.15)],
    ids=["entropy-stabilized", "stable-at-0K"],
)
def test_compound_line_starts_where_the_compound_becomes_stable(h_ab, s_ab, bottom):
    fig = plot_tx(_compound_hsx(h_ab, s_ab))
    (trace,) = [tr for tr in fig.data if tr.name == "AB"]
    assert min(trace.y) == pytest.approx(bottom, abs=1e-6)


def test_zero_kelvin_stable_is_the_lower_h_hull():
    assert binary_tx._zero_kelvin_stable(_compound_hsx(2000.0, 5.0)) == {"A", "B"}
    assert binary_tx._zero_kelvin_stable(_compound_hsx(-2000.0, 0.0)) == {"A", "B", "AB"}
