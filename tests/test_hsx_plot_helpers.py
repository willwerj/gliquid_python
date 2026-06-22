import bisect

import pytest

from gliquid.hsx import HSX, _estimate_label_box, _BOTTOM_PAD_FRAC, _PLOT_W_PX, _PLOT_H_PX


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


# ---------------------------------------------------------------------------
# _abbreviate_phase_name
# ---------------------------------------------------------------------------
def test_abbreviate_phase_name_rules():
    hsx = _make_hsx()
    names = ["alpha Mn (bcc)", "beta Mn (bcc)", "delta Fe (bcc)", "ZrMn2", "Liquid"]

    # Mn has two polymorphs present -> keep greek; structure kept (already short).
    assert hsx._abbreviate_phase_name("alpha Mn (bcc)", names) == "(αMn) bcc"
    # Fe has a single phase present -> drop greek; structure kept.
    assert hsx._abbreviate_phase_name("delta Fe (bcc)", names) == "(Fe) bcc"
    assert hsx._abbreviate_phase_name("ZrMn2", names) == "ZrMn<sub>2</sub>"
    assert hsx._abbreviate_phase_name("Liquid", names) == "L"
    assert hsx._abbreviate_phase_name("L", names) == "L"


def test_abbreviate_phase_name_hyphen_form_and_structure_abbrev():
    hsx = _make_hsx()
    # Hyphen-separated greek prefix (real-data form), single polymorph -> drop greek, abbrev struct.
    single = ["alpha-Ga (orthorhombic)", "SmGa2", "L"]
    assert hsx._abbreviate_phase_name("alpha-Ga (orthorhombic)", single) == "(Ga) ortho"

    # Two polymorphs of the same element present -> keep greek.
    paired = ["alpha-Ga (orthorhombic)", "beta-Ga (tetragonal)", "SmGa2", "L"]
    assert hsx._abbreviate_phase_name("alpha-Ga (orthorhombic)", paired) == "(αGa) ortho"
    assert hsx._abbreviate_phase_name("beta-Ga (tetragonal)", paired) == "(βGa) tetra"

    # Long Strukturbericht structure -> compact token.
    mns = ["beta-Mn (complex cubic A13)", "delta-Mn (bcc)", "alpha-Mn (complex cubic A12)", "L"]
    assert hsx._abbreviate_phase_name("beta-Mn (complex cubic A13)", mns) == "(βMn) A13"
    assert hsx._abbreviate_phase_name("alpha-Mn (complex cubic A12)", mns) == "(αMn) A12"
    assert hsx._abbreviate_phase_name("delta-Mn (bcc)", mns) == "(δMn) bcc"


def test_abbreviate_phase_name_struct_element_form():
    hsx = _make_hsx()
    # "<structure>-<Element>" naming (no greek prefix) -> parenthesised element + structure.
    names = ["fcc-Al", "bcc-Cr", "In (Fm-3m)", "L"]
    assert hsx._abbreviate_phase_name("fcc-Al", names) == "(Al) fcc"
    assert hsx._abbreviate_phase_name("bcc-Cr", names) == "(Cr) bcc"
    # Space-group structure symbols are dropped (not a crystal-system name).
    assert hsx._abbreviate_phase_name("In (Fm-3m)", names) == "(In)"
    assert hsx._abbreviate_phase_name("alpha-Pu (P6_3/mmc)", ["alpha-Pu (P6_3/mmc)", "L"]) == "(Pu)"


def test_abbreviate_phase_name_compound_with_tag():
    hsx = _make_hsx()
    names = ["Mn3Sn", "Mn2Sn ht", "MnSn2", "L"]
    assert hsx._abbreviate_phase_name("Mn2Sn ht", names) == "Mn<sub>2</sub>Sn ht"
    assert hsx._abbreviate_phase_name("Mn3Sn", names) == "Mn<sub>3</sub>Sn"
    assert hsx._abbreviate_phase_name("MnSn2", names) == "MnSn<sub>2</sub>"
    assert hsx._abbreviate_phase_name("Sm3Ga2", names) == "Sm<sub>3</sub>Ga<sub>2</sub>"


def test_abbreviate_phase_name_bare_element():
    hsx = _make_hsx()
    # A bare element symbol (ground state with no greek/structure) -> parenthesised.
    assert hsx._abbreviate_phase_name("Ga", ["Ga", "SmGa2", "L"]) == "(Ga)"


def test_abbreviate_phase_name_capitalised_structure_word_form():
    hsx = _make_hsx()
    # "<Structure words> <Element>" naming (capitalised multi-word structure, trailing element).
    # The long descriptor is dropped -> just the parenthesised element, like other ground states.
    names = ["Diamond cubic Si", "Diamond cubic Ge", "Cr3Ge", "L"]
    assert hsx._abbreviate_phase_name("Diamond cubic Si", names) == "(Si)"
    assert hsx._abbreviate_phase_name("Diamond cubic Ge", names) == "(Ge)"
    # A genuine compound is unaffected.
    assert hsx._abbreviate_phase_name("Cr3Ge", names) == "Cr<sub>3</sub>Ge"
    # Greek polymorph naming must NOT be captured by the structure-word branch.
    assert hsx._abbreviate_phase_name("alpha Mn (bcc)", ["alpha Mn (bcc)", "beta Mn (bcc)"]) == "(αMn) bcc"


# ---------------------------------------------------------------------------
# _merge_close_values
# ---------------------------------------------------------------------------
def test_merge_close_values_groups_with_mean():
    hsx = _make_hsx()
    merged = hsx._merge_close_values([10.0, 10.2, 12.0, 12.1, 20.0], tol=0.3)

    assert len(merged) == 3
    assert merged[0] == pytest.approx(10.1, abs=1e-6)
    assert merged[1] == pytest.approx(12.05, abs=1e-6)
    assert merged[2] == pytest.approx(20.0, abs=1e-6)


def test_merge_close_values_edge_cases():
    hsx = _make_hsx()
    assert hsx._merge_close_values([], tol=1.0) == []
    assert hsx._merge_close_values([7.0], tol=1.0) == [7.0]
    # all within tol of their neighbour -> single mean
    assert hsx._merge_close_values([1.0, 1.5, 2.0], tol=0.6) == pytest.approx([1.5])
    # none within tol -> unchanged
    assert hsx._merge_close_values([1.0, 5.0, 9.0], tol=0.5) == pytest.approx([1.0, 5.0, 9.0])


# ---------------------------------------------------------------------------
# _detect_tie_lines
# ---------------------------------------------------------------------------
def test_detect_tie_lines_uses_crossings_not_full_xlim():
    hsx = _make_hsx()
    boundary_curves = [
        [(0.0, 1000.0), (50.0, 800.0), (100.0, 900.0)],
        [(20.0, 600.0), (20.0, 900.0)],
        [(80.0, 700.0), (80.0, 950.0)],
    ]

    tie_lines = hsx._detect_tie_lines(
        invariant_temps=[850.0],
        boundary_curves=boundary_curves,
        plot_xlim=(0.0, 100.0),
        temp_tol=0.1,
        x_tol=0.25,
    )

    assert len(tie_lines) == 1
    tl = tie_lines[0]
    assert tl["x_start"] == pytest.approx(20.0, abs=1e-4)
    assert tl["x_end"] == pytest.approx(80.0, abs=1e-4)


def test_detect_tie_lines_single_solid_pairs_with_liquidus():
    hsx = _make_hsx()
    # One solid boundary at x=30 crossing T=850; the liquidus crosses 850 once, at x=75.
    boundary_curves = [
        [(0.0, 1000.0), (50.0, 900.0), (100.0, 800.0)],  # liquidus -> crossing at x=75
        [(30.0, 600.0), (30.0, 900.0)],                  # lone solid boundary -> crossing 30
    ]
    tie_lines = hsx._detect_tie_lines(
        invariant_temps=[850.0], boundary_curves=boundary_curves,
        plot_xlim=(0.0, 100.0), temp_tol=0.1, x_tol=0.25,
    )
    assert len(tie_lines) == 1
    assert tie_lines[0]["x_start"] == pytest.approx(30.0, abs=1e-4)
    assert tie_lines[0]["x_end"] == pytest.approx(75.0, abs=1e-4)


def test_detect_tie_lines_no_crossing_returns_empty():
    hsx = _make_hsx()
    boundary_curves = [[(20.0, 600.0), (20.0, 700.0)], [(80.0, 600.0), (80.0, 700.0)]]
    tie_lines = hsx._detect_tie_lines(
        invariant_temps=[1200.0], boundary_curves=boundary_curves,
        plot_xlim=(0.0, 100.0), temp_tol=1.0, x_tol=0.25,
    )
    assert tie_lines == []


# ---------------------------------------------------------------------------
# _resolve_label_collisions
# ---------------------------------------------------------------------------
def _vlabel(x, y, text):
    return {
        "x": x, "y": y, "text": text,
        "xanchor": "center", "yanchor": "middle",
        "textangle": -90, "font_size": 12, "font_color": "black",
    }


def test_resolve_label_collisions_separates_overlaps():
    hsx = _make_hsx()
    labels = [_vlabel(10.0, 400.0, "(Mn)"), _vlabel(10.0, 400.0, "ZrMn<sub>2</sub>")]

    resolved = hsx._resolve_label_collisions(labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0),
                                             max_iterations=20)

    assert resolved[0]["y"] != pytest.approx(resolved[1]["y"], abs=1e-6)


def test_resolve_label_collisions_three_stacked():
    hsx = _make_hsx()
    labels = [_vlabel(10.0, 400.0, "A"), _vlabel(10.0, 400.0, "B"), _vlabel(10.0, 400.0, "C")]

    resolved = hsx._resolve_label_collisions(labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0),
                                             max_iterations=50)

    ys = sorted(round(l["y"], 4) for l in resolved)
    assert len(set(ys)) == 3, "all three labels should end at distinct y positions"
    # displaced labels should be flagged for a leader arrow
    assert any(l.get("showarrow") for l in resolved)


def test_resolve_label_collisions_keeps_non_overlapping_put():
    hsx = _make_hsx()
    labels = [_vlabel(10.0, 200.0, "A"), _vlabel(90.0, 1200.0, "B")]
    resolved = hsx._resolve_label_collisions(labels, xlim=(0.0, 100.0), ylim=(0.0, 1500.0))
    assert resolved[0]["y"] == pytest.approx(200.0)
    assert resolved[1]["y"] == pytest.approx(1200.0)
    assert not any(l.get("showarrow") for l in resolved)


# ---------------------------------------------------------------------------
# _estimate_label_box (lock the fragile font/pixel constants)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _assign_compound_sides (Bug 3: distinct two-phase gaps for clustered compounds)
# ---------------------------------------------------------------------------
def _gap_index(x, comps, xlim=(0.0, 100.0)):
    """Which inter-compound interval of ``[xlim0] + sorted(comps) + [xlim1]`` contains x."""
    bounds = [xlim[0]] + sorted(comps) + [xlim[1]]
    return bisect.bisect_right(bounds, x) - 1


def test_assign_compound_sides_clustered_get_distinct_gaps():
    comps = [19.0, 21.0, 24.0, 30.0]
    half_ws = [1.17] * 4
    xs = HSX._assign_compound_sides(comps, half_ws, xlim=(0.0, 100.0))
    gaps = [_gap_index(x, comps) for x in xs]
    assert len(set(gaps)) == len(comps), f"labels share a gap: xs={xs}, gaps={gaps}"


def test_assign_compound_sides_edge_compounds_forced_inward():
    comps = [5.0, 95.0]
    half_ws = [1.17, 1.17]
    xs = HSX._assign_compound_sides(comps, half_ws, xlim=(0.0, 100.0))
    assert xs[0] > 5.0, "compound near the left edge must place its label to the right (inward)"
    assert xs[1] < 95.0, "compound near the right edge must place its label to the left (inward)"


# ---------------------------------------------------------------------------
# _place_compound_y (Bugs 1 & 2: bottom pad + skip up past a blocking tie)
# ---------------------------------------------------------------------------
def test_place_compound_y_bottom_pad_when_clear():
    # No tie lines: the label sits at the bottom region with exactly ``pad`` above the floor.
    pad, half_h, margin = 15.0, 50.0, 12.0
    y = HSX._place_compound_y(c0=0.0, ceiling=1000.0, tie_temps=[], half_h=half_h,
                              pad=pad, margin=margin)
    assert y == pytest.approx(pad + half_h)          # box bottom == c0 + pad


def test_place_compound_y_skips_blocked_bottom_region():
    # A tie at 40 makes the bottom region (0..28) too short; the label must lift into the next
    # tie-free region above the tie, not sit inside the tie.
    y = HSX._place_compound_y(c0=0.0, ceiling=1000.0, tie_temps=[40.0], half_h=50.0,
                              pad=15.0, margin=12.0)
    assert y is not None
    assert y - 50.0 > 40.0, "label box must clear the blocking tie at T=40"
    assert y == pytest.approx(52.0 + 50.0)           # bottom of the (52..1000) region + half_h


def test_place_compound_y_returns_none_when_nothing_fits():
    # Ceiling barely above the floor: no region is tall enough for the label.
    y = HSX._place_compound_y(c0=0.0, ceiling=20.0, tie_temps=[], half_h=50.0,
                              pad=15.0, margin=12.0)
    assert y is None


# ---------------------------------------------------------------------------
# _resolve_label_collisions bottom pad (Bug 1)
# ---------------------------------------------------------------------------
def test_resolve_label_collisions_applies_bottom_pad():
    hsx = _make_hsx()
    ylim = (0.0, 1500.0)
    # A bottom-anchored compound label placed flush with the floor should be lifted by the pad.
    label = _vlabel(50.0, 0.0, "ZrMn<sub>2</sub>")
    label["yanchor"] = "bottom"
    resolved = hsx._resolve_label_collisions([label], xlim=(0.0, 100.0), ylim=ylim)
    _, cy, _, half_h = _estimate_label_box(resolved[0], (0.0, 100.0), ylim)
    box_bottom = cy - half_h
    assert box_bottom == pytest.approx(ylim[0] + _BOTTOM_PAD_FRAC * (ylim[1] - ylim[0]), abs=1e-6)
