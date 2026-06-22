"""Integration tests for HSX.plot_tx label/tie-line/legend/L-label placement.

These build a real ``fit+liq`` figure for cached binary systems and introspect the returned
``go.Figure`` (no rendering needed) to catch the four defects the placement work targets:

  P1  phase labels overlapping each other or the liquidus
  P2  missing tie lines (solid<->liquidus / invariants / polymorph transitions)
  P3  the legend overlapping a liquidus curve
  P4  the 'L' label pinned at x=50 / outside the liquid field

Reference systems (Ga-Sm, Fe-Tb, Mn-Sn) and the elemental polymorph data live in the nested
``matrix_data`` cache one level above the repo. When that cache is absent the whole module is
skipped, so the suite stays portable.

Run:  python -m pytest tests/test_plot_tx_integration.py -v
"""
import functools
from pathlib import Path

import numpy as np
import pytest

import gliquid.config as config
from gliquid.hsx import _estimate_label_box, _box_overlap, _PLOT_W_PX, _PLOT_H_PX


# --------------------------------------------------------------------------- data location
def _find_g_liquid_root() -> Path | None:
    for p in [Path(config.data_dir).resolve()] + list(Path(config.data_dir).resolve().parents):
        if p.name == "G_liquid":
            return p
    return None


_ROOT = _find_g_liquid_root()
_MATRIX = (_ROOT / "matrix_data") if _ROOT else None

pytestmark = pytest.mark.skipif(
    _MATRIX is None or not _MATRIX.exists() or not (_MATRIX / "phase_transitions.json").exists(),
    reason="nested matrix_data cache (with polymorph data) not available",
)

# REFERENCE_SYSTEMS = ["Ga-Sm", "Fe-Tb", "Mn-Sn"]
# EXTRA_SYSTEMS = ["Al-Cu", "Cr-Eu"]
# ALL_SYSTEMS = REFERENCE_SYSTEMS + EXTRA_SYSTEMS
ALL_SYSTEMS = [
    "Ag-Co",
    "Ag-Cu",
    "Ag-La",
    "Ag-Mo",
    "Ag-Na",
    "Ag-Pb",
    "Ag-Ti",
    "Al-Cu",
    "Al-Fe",
    "Al-In",
    "Al-Si",
    "Al-Tm",
    "Al-Zn",
    "Au-Cr",
    "Au-Er",
    "Au-Gd",
    "Au-Hf",
    "Au-In",
    "Au-Nd",
    "Au-Pb",
    "Au-Sm",
    "B-Cd",
    "B-Mn",
    "B-Nb",
    "B-Re",
    "B-Ru",
    "B-Ta",
    "Ba-Ge",
    "Ba-Zn",
    "Be-Fe",
    "Be-Ru",
    "Be-V",
    "Bi-Sr",
    "Bi-Tm",
    "C-Fe",
    "C-Ta",
    "C-Zr",
    "Ca-La",
    "Ca-Si",
    "Ce-Cr",
    "Ce-Ga",
    "Ce-In",
    "Ce-Sn",
    "Co-Nd",
    "Co-Sn",
    "Cr-Ge",
    "Cr-Hf",
    "Cr-Ho",
    "Cr-Nd",
    "Cu-Eu",
    "Cu-Hg",
    "Cu-Ir",
    "Cu-La",
    "Cu-Lu",
    "Cu-Pr",
    "Cu-Ru",
    "Cu-Si",
    "Dy-Ir",
    "Dy-Mn",
    "Dy-Ru",
    "Dy-Ti",
    "Er-Fe",
    "Er-Ga",
    "Er-Rh",
    "Fe-Ge",
    "Ga-Lu",
    "Ga-Sn",
    "Ga-V",
    "Gd-Mg",
    "Gd-Zn",
    "Ge-In",
    "Ge-La",
    "Ge-Nb",
    "Ge-Pb",
    "Ge-Sm",
    "Hf-Rh",
    "Hf-Y",
    "Hg-Li",
    "Hg-Ni",
    "Hg-Rb",
    "Hg-Sr",
    "Hg-V",
    "Hg-Y",
    "Ho-In",
    "In-Si",
    "In-Tm",
    "La-Sn",
    "La-Tl",
    "Li-Si",
    "Lu-Mg",
    "Mg-Tb",
    "Mg-Tm",
    "Mn-Si",
    "Mn-Ta",
    "Mo-Nd",
    "Nb-Os",
    "Nb-Ru",
    "Ni-Sc",
    "Ni-Sm",
    "Os-Ti",
    "Pb-Sc",
    "Si-Sn",
    "Si-Ti",
    "Si-V",
    "Sn-Ti",
    "Tl-Zn",
    "Ir-Li",
    "Cr-La",
    "Mg-Rh",
    "Sn-Y",
]

# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def matrix_data_cache():
    """Point gliquid at the nested matrix_data cache (with polymorphs) for this module only."""
    import gliquid.load_binary_data as lbd

    saved = (config.data_dir, config.dir_structure, config.phase_transitions_file)
    config.set_data_dir(_MATRIX)
    config.set_dir_structure("nested")
    lbd.reload_element_data()
    try:
        yield
    finally:
        config.data_dir, config.dir_structure, config.phase_transitions_file = saved
        lbd.reload_element_data(require=False)


def _system_cached(name: str) -> bool:
    sys_dir = _MATRIX / "-".join(sorted(name.split("-")))
    if not sys_dir.exists():
        return False
    sn = sys_dir.name
    has_mpds = (sys_dir / f"{sn}.json").exists() or (sys_dir / f"{sn}_MPDS_PD_0.json").exists()
    has_dft = (sys_dir / f"{sn}_ENTRIES_MP_GGA.json").exists()
    return has_mpds and has_dft


# --- Fitted parameters come from the SAME source the matrix plotter's main function uses
# (dev/scripts/Interactive_Matrix_Plotter.py): the fitted-systems workbook under
# ``<G_liquid>/binary_fitting_data/``. For each system the lowest-``mae`` row supplies
# ``[L0_a, L0_b, L1_a, L1_b]``; systems absent from the workbook are skipped. ---
_PARAM_FORMAT = "comb-exp"
_FITTED_WORKBOOK = (
    (_ROOT / "binary_fitting_data" / "fixed_refs_20_opts_p1.5se8-filtered-matrix.xlsx")
    if _ROOT else None
)


@functools.lru_cache(maxsize=1)
def _workbook_params() -> dict:
    """Map ``system -> [L0_a, L0_b, L1_a, L1_b]`` (lowest-mae fit), keyed by both element orderings."""
    import pandas as pd

    if _FITTED_WORKBOOK is None or not _FITTED_WORKBOOK.exists():
        return {}
    df = pd.read_excel(_FITTED_WORKBOOK)
    out: dict[str, list[float]] = {}
    for sysname, grp in df.dropna(subset=["system"]).groupby("system"):
        row = grp.sort_values("mae").iloc[0]
        params = [float(row[c]) for c in ("L0_a", "L0_b", "L1_a", "L1_b")]
        parts = str(sysname).split("-")
        if len(parts) != 2:
            continue
        out[f"{parts[0]}-{parts[1]}"] = params
        out[f"{parts[1]}-{parts[0]}"] = params
    return out


def _fitted_bl(name):
    """Return a BinaryLiquid for ``name`` with the workbook's fitted params applied."""
    from gliquid.binary import BinaryLiquid

    params = _workbook_params().get(name)
    if params is None:
        wb = _FITTED_WORKBOOK.name if _FITTED_WORKBOOK else "fitted-systems workbook"
        pytest.skip(f"{name}: no fitted parameters in {wb}")
    bl = BinaryLiquid.from_cache(name, param_format=_PARAM_FORMAT, pd_ind=0)
    bl.update_params(list(params))
    return bl


@pytest.fixture
def figure(matrix_data_cache, request):
    """Build a fit+liq figure (fitted params, cached) for a system."""
    name = request.param
    if not _system_cached(name):
        pytest.skip(f"{name} not cached in matrix_data")
    from gliquid.binary import BLPlotter

    bl = _fitted_bl(name)
    fig = BLPlotter(bl).get_plot("fit+liq")
    return name, bl, fig


# --------------------------------------------------------------------------- introspection helpers
def _ylim(fig):
    return tuple(float(v) for v in fig.layout.yaxis.range)


_XLIM = (0.0, 100.0)


def _phase_label_annotations(fig):
    """Data-coordinate phase labels (excludes the paper-anchored component names and 'L')."""
    out = []
    for a in fig.layout.annotations:
        if a.xref == "paper" or a.yref == "paper" or a.text == "L":
            continue
        out.append(a)
    return out


def _ann_to_label(a, ylim):
    # For arrow annotations plotly draws the text at the pixel offset (ax, ay) from the anchor
    # (x, y); recover the rendered text position in data coords (ay is positive-downward).
    x, y = float(a.x), float(a.y)
    if a.showarrow:
        x += (a.ax or 0) / (_PLOT_W_PX / (_XLIM[1] - _XLIM[0]))
        y -= (a.ay or 0) / (_PLOT_H_PX / (ylim[1] - ylim[0]))
    return {
        "x": x, "y": y, "text": a.text,
        "xanchor": a.xanchor or "center", "yanchor": a.yanchor or "middle",
        "textangle": a.textangle or 0,
        "font_size": (a.font.size if a.font and a.font.size else 12),
    }


def _liquidus_xy(fig, colors):
    for tr in fig.data:
        line = getattr(tr, "line", None)
        if line is not None and getattr(line, "color", None) in colors and tr.x is not None and len(tr.x) > 2:
            return np.array(tr.x, dtype=float), np.array(tr.y, dtype=float)
    return None


def _fitted_liquidus(fig):
    return _liquidus_xy(fig, {"cornflowerblue", "#117733"})


def _assessed_liquidus(fig):
    return _liquidus_xy(fig, {"#B82E2E"})


def _silver_tielines(fig):
    out = []
    for tr in fig.data:
        line = getattr(tr, "line", None)
        if line is not None and str(getattr(line, "color", "")).lower() == "silver" and tr.x is not None:
            out.append((np.array(tr.x, dtype=float), np.array(tr.y, dtype=float)))
    return out


def _segment_intersects_box(p0, p1, box):
    """True if segment p0->p1 intersects/enters the AABB ``box=(cx,cy,hw,hh)`` (data coords)."""
    cx, cy, hw, hh = box
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh
    (ax, ay), (bx, by) = p0, p1
    # Quick reject by bounding box of the segment.
    if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
        return False
    # Either endpoint inside?
    if (x0 <= ax <= x1 and y0 <= ay <= y1) or (x0 <= bx <= x1 and y0 <= by <= y1):
        return True
    # Liang-Barsky clip against the rectangle.
    dx, dy = bx - ax, by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return t0 <= t1


def _polyline_intersects_box(xs, ys, box):
    return any(_segment_intersects_box((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), box)
              for i in range(len(xs) - 1))


# --------------------------------------------------------------------------- P1: label overlaps
@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p1_phase_labels_do_not_overlap_each_other(figure):
    name, _bl, fig = figure
    ylim = _ylim(fig)
    labels = [_ann_to_label(a, ylim) for a in _phase_label_annotations(fig)]
    boxes = [_estimate_label_box(l, _XLIM, ylim) for l in labels]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            pen_x, pen_y = _box_overlap(boxes[i], boxes[j])
            if pen_x > 0 and pen_y > 0:
                overlaps.append((labels[i]["text"], labels[j]["text"]))
    assert not overlaps, f"{name}: overlapping phase labels: {overlaps}"


@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p1_labels_clear_of_lines(figure):
    """No label text box may overlap either liquidus curve; element/polymorph solid-solution
    labels (those rendered as "(El)...") additionally must not cross any tie line."""
    name, _bl, fig = figure
    ylim = _ylim(fig)
    gen = _fitted_liquidus(fig)
    asd = _assessed_liquidus(fig)
    ties = _silver_tielines(fig)
    bad = []
    for a in _phase_label_annotations(fig):
        box = _estimate_label_box(_ann_to_label(a, ylim), _XLIM, ylim)
        if gen is not None and _polyline_intersects_box(gen[0], gen[1], box):
            bad.append((a.text, "generated liquidus"))
        if asd is not None and _polyline_intersects_box(asd[0], asd[1], box):
            bad.append((a.text, "assessed liquidus"))
        # No phase label (element, polymorph, or compound) may cross a tie line: bug-2 placement
        # lifts a label into the next two-phase region rather than letting a tie bisect its box.
        for tx, ty in ties:
            if _polyline_intersects_box(tx, ty, box):
                bad.append((a.text, f"tie line @{ty[0]:.0f}C"))
                break
    assert not bad, f"{name}: labels overlapping lines: {bad}"


def _is_compound_text(text: str) -> bool:
    """Compound formula labels (e.g. 'Al<sub>2</sub>Cu'); excludes '(El)' solid-solution and 'L'."""
    t = text.strip()
    return bool(t) and not t.startswith("(") and t != "L"


def _tie_below(ties, cx, hw, bb, y_floor):
    """True if a silver tie line crosses the column [cx-hw, cx+hw] strictly below box-bottom bb."""
    return any(y_floor + 1e-6 < ys[0] < bb - 1e-6 and xs.min() <= cx + hw and xs.max() >= cx - hw
               for xs, ys in ties)


@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p1_compound_labels_keep_consistent_bottom_pad(figure):
    """Bug 1: compound labels keep a consistent ~1.5% gap above the lower boundary — never flush
    with the axis (gap too small), and a bottom-region label sits at exactly the pad (not an
    inconsistent oversized gap). A label legitimately lifted above a tie line is exempt."""
    from gliquid.hsx import _BOTTOM_PAD_FRAC

    name, _bl, fig = figure
    ylim = _ylim(fig)
    span = ylim[1] - ylim[0]
    pad = _BOTTOM_PAD_FRAC * span
    ties = _silver_tielines(fig)
    labels = []  # (cx, hw, box_bottom, text)
    for a in _phase_label_annotations(fig):
        if a.showarrow or not _is_compound_text(a.text):
            continue  # floated/relocated labels are not bottom-anchored
        cx, cy, hw, hh = _estimate_label_box(_ann_to_label(a, ylim), _XLIM, ylim)
        labels.append((cx, hw, cy - hh, a.text))
    if not labels:
        pytest.skip(f"{name}: no bottom-anchored compound labels")
    # (a) The pad is never violated (gap never smaller than ~1.5%, never on the axis).
    for cx, hw, bb, text in labels:
        assert bb >= ylim[0] + pad - 0.005 * span, (
            f"{name}: '{text}' sits too close to the bottom (gap {bb - ylim[0]:.1f} < pad {pad:.1f})")
    # (b) Consistency: the lowest compound label, when nothing lies below it in its column, sits at
    # exactly the pad. (Labels lifted above a tie, or stacked above a lower one, are exempt.)
    cx, hw, bb, text = min(labels, key=lambda c: c[2])
    if not _tie_below(ties, cx, hw, bb, ylim[0]):
        assert bb <= ylim[0] + pad + 0.02 * span, (
            f"{name}: lowest compound label '{text}' gap {bb - ylim[0]:.1f} >> pad {pad:.1f}")


@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p4_leader_arrows_do_not_cross(figure):
    """Bug 4: relocated (arrowed) labels on a side are assigned to slots monotonically, so their
    leader arrows do not cross and no two relocated boxes coincide."""
    name, _bl, fig = figure
    ylim = _ylim(fig)
    # Group arrowed phase labels by side using the arrow anchor (home) composition.
    sides = {0: [], 100: []}
    for a in _phase_label_annotations(fig):
        if not a.showarrow:
            continue
        home_x, home_y = float(a.x), float(a.y)
        text_y = _ann_to_label(a, ylim)["y"]
        sides[0 if home_x < 50 else 100].append((home_y, text_y, a.text))
    for side, items in sides.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])  # by home (anchor) temperature
        text_ys = [t[1] for t in items]
        # Monotonic non-decreasing text positions => non-crossing leader arrows.
        assert all(text_ys[i] <= text_ys[i + 1] + 1e-6 for i in range(len(text_ys) - 1)), (
            f"{name}: crossing leader arrows on side {side}: {[(round(h,0), round(ty,0), tx) for h,ty,tx in items]}")


# --------------------------------------------------------------------------- P2: tie lines
@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p2_tielines_present_for_each_invariant(figure):
    name, bl, fig = figure
    inv, _, _ = bl.hsx.liquidus_invariants()
    expected_temps = []
    for key in ("Eutectics", "Peritectics", "Misc Gaps", "Solid Ties"):
        expected_temps += [e[0] for e in inv.get(key, [])]
    if not expected_temps:
        pytest.skip(f"{name}: no eutectic/peritectic invariants to check")

    ties = _silver_tielines(fig)
    assert ties, f"{name}: no tie lines drawn at all"
    tie_temps = [float(np.mean(t[1])) for t in ties]
    missing = [round(T, 1) for T in expected_temps
               if not any(abs(T - tt) <= 3.0 for tt in tie_temps)]
    assert not missing, f"{name}: invariants without a tie line at T={missing} (have {sorted(set(round(t,1) for t in tie_temps))})"


# A eutectic/peritectic between two adjacent solids collapses to a 2-vertex (solid-solid) hull
# simplex whenever the liquid vertex is not distinct -- close-set line compounds, or a eutectic
# pinned just below a low-melting element. ``liquidus_invariants`` used to process only 3-vertex
# simplices, so those horizontal ties went undrawn. Each case below names two adjacent solids
# (by composition window) that must now be bridged by a tie.
_SOLID_TIE_CASES = {
    "Cu-Eu": (48, 69),   # EuCu (50) <-> Eu2Cu (67), a eutectic between two congruent compounds
    "Ga-V": (0, 18),     # (Ga) <-> V8Ga41, eutectic pinned just below Ga's 30C melt
    "Ga-Lu": (0, 27),    # (Ga) <-> LuGa3, same low-melting-element case
}


@pytest.mark.parametrize("name", sorted(_SOLID_TIE_CASES))
def test_p2_collapsed_simplex_solid_ties_present(matrix_data_cache, name):
    """Regression: ties between adjacent solids that collapse to a 2-vertex hull simplex."""
    if not _system_cached(name):
        pytest.skip(f"{name} not cached in matrix_data")
    from gliquid.binary import BLPlotter

    lo, hi = _SOLID_TIE_CASES[name]
    bl = _fitted_bl(name)
    inv = bl.hsx.liquidus_invariants()[0]
    span_ok = lambda lo_c, hi_c: lo - 1.5 <= lo_c and hi_c <= hi + 1.5
    detected = any(span_ok(min(c) * 100, max(c) * 100) for _, _, c, _ in inv.get("Solid Ties", []))
    assert detected, (
        f"{name}: no Solid Tie detected within x=[{lo},{hi}]; "
        f"got {[(round(min(c)*100), round(max(c)*100)) for _,_,c,_ in inv.get('Solid Ties', [])]}")

    fig = BLPlotter(bl).get_plot("fit+liq")
    ties = _silver_tielines(fig)
    drawn = any(lo - 1.5 <= xs.min() and xs.max() <= hi + 1.5 for xs, _ in ties)
    assert drawn, f"{name}: Solid Tie within x=[{lo},{hi}] detected but not drawn"


# In a (near-)immiscible system with no compound between the elements, the lower-melting element
# melts into a wide L+(opposite solid) field: a degenerate-eutectic horizontal pinned at that
# element's melting point. It surfaces as a 2-vertex element-to-element solid simplex (the hottest
# tie for that terminal pair) and must be drawn full width. Used to be dropped as an "element to
# element" artifact -- see the Solid Ties recovery in HSX.liquidus_invariants.
_ELEMENT_MELT_TIE_CASES = {
    "Hg-V": -38.8,    # Hg melts at -38.8 C; V stays solid -> wide L+(V) tie across the diagram
    "Ag-Mo": 961.8,   # Ag melts at 961.8 C; Mo stays solid -> wide L+(Mo) tie
}


@pytest.mark.parametrize("name", sorted(_ELEMENT_MELT_TIE_CASES))
def test_p2_immiscible_element_melt_tie(matrix_data_cache, name):
    """Regression: the degenerate tie at the lower element's melting point in an immiscible system."""
    if not _system_cached(name):
        pytest.skip(f"{name} not cached in matrix_data")
    from gliquid.binary import BLPlotter

    t_melt = _ELEMENT_MELT_TIE_CASES[name]
    bl = _fitted_bl(name)
    inv = bl.hsx.liquidus_invariants()[0]
    detected = any(min(c) < 0.05 and max(c) > 0.95 and abs(t - t_melt) <= 5.0
                   for t, _, c, _ in inv.get("Solid Ties", []))
    assert detected, (
        f"{name}: no full-width Solid Tie near the lower-element melt {t_melt}C; got "
        f"{[(round(t), round(min(c) * 100), round(max(c) * 100)) for t, _, c, _ in inv.get('Solid Ties', [])]}")

    fig = BLPlotter(bl).get_plot("fit+liq")
    ties = _silver_tielines(fig)
    drawn = any((xs.max() - xs.min()) > 95 and abs(float(ys[0]) - t_melt) <= 5.0 for xs, ys in ties)
    assert drawn, f"{name}: element-melt tie at {t_melt}C detected but not drawn"


# A floated/relocated elemental-polymorph label (drawn with a leader arrow at the inward column fx)
# must dodge any compound sharing that composition, else the label and arrow land on the compound's
# vertical line/label (Be-V: the floated (βBe) bcc vs Be12V at 8 at%).
_POLYMORPH_DODGE_CASES = {
    "Be-V": ("bcc", 8.0),   # (βBe) bcc must stay clear of the Be12V column at 8 at%
}


@pytest.mark.parametrize("name", sorted(_POLYMORPH_DODGE_CASES))
def test_p1_floated_polymorph_label_dodges_compound(matrix_data_cache, name):
    """Regression: a floated polymorph label must not share a column with a same-composition compound."""
    if not _system_cached(name):
        pytest.skip(f"{name} not cached in matrix_data")
    from gliquid.binary import BLPlotter

    sub, comp = _POLYMORPH_DODGE_CASES[name]
    bl = _fitted_bl(name)
    fig = BLPlotter(bl).get_plot("fit+liq")
    ylim = _ylim(fig)
    arrowed = [a for a in _phase_label_annotations(fig)
               if a.showarrow and a.text.startswith("(") and sub in a.text]
    assert arrowed, f"{name}: expected a floated polymorph label containing {sub!r}"
    for a in arrowed:
        cx, _, hw, _ = _estimate_label_box(_ann_to_label(a, ylim), _XLIM, ylim)
        gap = abs(cx - comp) - hw
        assert gap >= 1.0, (
            f"{name}: floated label {a.text!r} at x={cx:.1f} sits on the compound column "
            f"at {comp} at% (clear gap {gap:.1f} at%)")


@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p2_tielines_are_local_not_full_width(figure):
    name, _bl, fig = figure
    ties = _silver_tielines(fig)
    # Full-width tie lines are legitimate for a simple eutectic and for polymorph transitions in a
    # compound-free system (each spans to the opposite element), but each must be a distinct
    # isotherm. A large number of distinct full-width temperatures indicates the over-draw bug.
    full_width_temps = {round(float(ys[0]), 1) for xs, ys in ties if (xs.max() - xs.min()) > 99.0}
    assert len(full_width_temps) <= 8, (
        f"{name}: {len(full_width_temps)} distinct full-width tie temperatures (expected <=8)")


# --------------------------------------------------------------------------- P3: legend
@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p3_legend_clears_liquidus(figure):
    name, _bl, fig = figure
    leg = fig.layout.legend
    if leg.orientation == "h":
        # Legend was lifted above the plot area; that is the accepted overflow fallback.
        assert leg.y is not None and leg.y >= 1.0
        return

    ylim = _ylim(fig)
    span_x, span_y = _XLIM[1] - _XLIM[0], ylim[1] - ylim[0]
    # Reconstruct the legend rectangle from its ACTUAL entries (1 when there is no assessed
    # liquidus, 2 otherwise) — mirroring HSX._place_legend's w_px/h_px formulas.
    entries = [tr for tr in fig.data if getattr(tr, "showlegend", None) and getattr(tr, "name", None)]
    n_entries = max(len(entries), 1)
    max_chars = max((len(tr.name) for tr in entries), default=len("Fitted Liquidus"))
    w_data = (max_chars * 0.6 * 15 + 40) / (615 / span_x)
    h_data = (n_entries * 21 + 12) / (456 / span_y)
    x_anchor = float(leg.x) * span_x
    y_anchor = ylim[0] + float(leg.y) * span_y
    x0 = x_anchor - w_data if leg.xanchor == "right" else x_anchor
    x1 = x_anchor if leg.xanchor == "right" else x_anchor + w_data
    y_bottom = y_anchor - h_data  # yanchor top

    for liq in (_fitted_liquidus(fig), _assessed_liquidus(fig)):
        if liq is None:
            continue
        xs, ys = liq
        inside = [(x, y) for x, y in zip(xs, ys) if x0 <= x <= x1 and y >= y_bottom]
        assert not inside, f"{name}: liquidus enters legend box near {inside[:3]}"


# --------------------------------------------------------------------------- P4: L label
@pytest.mark.parametrize("figure", ALL_SYSTEMS, indirect=True)
def test_p4_liquid_label_in_open_field(figure):
    name, _bl, fig = figure
    ylim = _ylim(fig)
    l_ann = next((a for a in fig.layout.annotations if a.text == "L"), None)
    assert l_ann is not None, f"{name}: no 'L' label"
    assert 0 < l_ann.x < 100, f"{name}: L at edge x={l_ann.x}"
    assert ylim[0] < l_ann.y < ylim[1], f"{name}: L outside plot y-range"

    liq = _fitted_liquidus(fig)
    if liq is not None:
        xs, ys = liq
        order = np.argsort(xs)
        top_at_l = float(np.interp(l_ann.x, xs[order], ys[order]))
        assert l_ann.y > top_at_l - 1e-6, (
            f"{name}: L (y={l_ann.y:.0f}) is below the liquidus (y={top_at_l:.0f}) at x={l_ann.x:.0f}")


# --------------------------------------------------------------------------- visual (opt-in)
@pytest.mark.visual
@pytest.mark.parametrize("name", ALL_SYSTEMS)
def test_visual_render(matrix_data_cache, name):
    if not _system_cached(name):
        pytest.skip(f"{name} not cached")
    from gliquid.binary import BLPlotter

    out_dir = Path(__file__).parent / "_figures"
    out_dir.mkdir(exist_ok=True)
    bl = _fitted_bl(name)
    out = out_dir / f"{name}.svg"
    try:
        BLPlotter(bl).write_image("fit+liq", str(out), image_format="svg", export_timeout_s=120)
    except Exception as exc:
        pytest.skip(f"image export unavailable: {exc}")
    assert out.exists() and out.stat().st_size > 0
