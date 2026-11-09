"""gliquid.liquidus_sheets -- per-phase liquidus temperatures from the liquid's tangent planes.

The sheets must reproduce the liquid model the hull is built from, agree with the hull on which
phase is tied to the liquid, never lie below the hull's liquidus, and keep a liquid that lacks an
element from touching a solid containing it. Runs offline (cached Al-Mg-Si MP entries; unary
registry solid solutions for Hf-Ti-Zr).
"""

import numpy as np
import pytest

import gliquid.liquidus_sheets as ls
import gliquid.ternary as tmod
from gliquid.liquidus_sheets import (
    BINODAL_CONSISTENCY,
    BINODAL_MARGIN,
    TWO_LIQUIDS,
    Binodal,
    LiquidusSheets,
    TstarField,
)
from gliquid.solution import R

AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}


def _grid(n):
    return np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])


@pytest.fixture(scope="module")
def ams():
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        hull_mode="hsx",
        param_format="linear",
    )
    plotter.process_data()
    plotter.get_plot("tx")
    return plotter, LiquidusSheets.from_interpolation(plotter._ti)


def test_split_reproduces_the_liquid_model(ams):
    plotter, sheets = ams
    comp = _grid(10)
    comp = comp[(comp.min(axis=1) > 0) & (comp.sum(axis=1) < 1)]
    model = plotter._ti.liquid_model()
    h, s, _, _ = sheets.tstar.h_s_gradients(model, comp)
    h_grid, s_grid = model.h_s_grid((comp[:, 0], comp[:, 1]), sheets.tstar(comp))
    full = np.column_stack([1 - comp.sum(axis=1), comp])
    assert h == pytest.approx(h_grid, rel=1e-9)
    assert s - R * (full * np.log(full)).sum(axis=1) == pytest.approx(s_grid, rel=1e-9)


def test_gradients_match_finite_differences(ams):
    plotter, sheets = ams
    model = plotter._ti.liquid_model()
    x = np.array([[0.2, 0.3], [0.5, 0.1], [0.1, 0.7]])
    _, _, dh, ds = sheets.tstar.h_s_gradients(model, x)
    step = 1e-6
    for k in range(2):
        shift = np.zeros(2)
        shift[k] = step
        hi = sheets.tstar.h_s_gradients(model, x + shift)
        lo = sheets.tstar.h_s_gradients(model, x - shift)
        assert dh[k] == pytest.approx((hi[0] - lo[0]) / (2 * step), rel=1e-5, abs=1e-3)
        assert ds[k] == pytest.approx((hi[1] - lo[1]) / (2 * step), rel=1e-5, abs=1e-6)


def test_primary_phase_is_tied_to_the_liquid_on_the_hull(ams):
    plotter, sheets = ams
    liq = plotter.liq_plotting_df
    temps = sheets(liq[["x0_orig", "x1_orig"]].to_numpy(float))
    primary = np.array(sheets.phases)[temps.argmax(axis=1)]
    for phase, tied in zip(primary, liq["coexistent_phases"]):
        assert phase in tied.split(", ")


def test_hull_liquidus_never_lies_above_the_sheets(ams):
    plotter, sheets = ams
    liq = plotter.liq_plotting_df
    liquidus = sheets.liquidus(liq[["x0_orig", "x1_orig"]].to_numpy(float)) - 273.15
    assert np.isfinite(liquidus).all()
    assert (liq["T"].to_numpy() <= liquidus + 1e-6).all()


def test_corners_melt_at_the_hull_melting_points(ams):
    plotter, sheets = ams
    liq = plotter.liq_plotting_df.set_index(["x0_orig", "x1_orig"])
    for corner in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]:
        liquidus = sheets.liquidus(np.array([corner]))[0] - 273.15
        assert liquidus == pytest.approx(liq.loc[corner, "T"], abs=1e-6)


def test_an_edge_liquid_never_touches_a_solid_with_the_missing_element(ams):
    _, sheets = ams
    rows = sheets._row_comp
    full = np.column_stack([1 - rows.sum(axis=1), rows])
    t = np.linspace(0, 1, 21)
    edges = {
        0: np.column_stack([t, 1 - t]),
        1: np.column_stack([np.zeros_like(t), t]),
        2: np.column_stack([t, np.zeros_like(t)]),
    }
    for missing, comp in edges.items():
        temps = sheets.point_temperatures(comp)
        assert (temps[:, full[:, missing] > 1e-12] == 0).all()


def test_solid_solution_phases_bind_at_their_hottest_point():
    plotter = tmod.TLIPlotter.from_components(
        ["Hf", "Ti", "Zr"],
        delta=0.1,
        xs_mix={"Hf-Ti": [2000.0, 0, 0, 0], "Ti-Zr": [-1000.0, 0, 0, 0], "Zr-Hf": [500.0, 0, 0, 0]},
        hull_mode="hsx",
        solid_solutions=True,
        ss_kwargs={"ref_mode": "from_unary_db"},
        param_format="linear",
    )
    plotter.process_data()
    plotter.get_plot("tx")
    sheets = LiquidusSheets.from_interpolation(plotter._ti)
    assert max(np.bincount(sheets._row_phase)) > 1
    liq = plotter.liq_plotting_df
    comp = liq[["x0_orig", "x1_orig"]].to_numpy(float)
    temps = sheets(comp)
    per_point = sheets.point_temperatures(comp)
    for k in range(len(sheets.phases)):
        assert temps[:, k] == pytest.approx(per_point[:, sheets._row_phase == k].max(axis=1))
    primary = np.array(sheets.phases)[temps.argmax(axis=1)]
    for phase, tied in zip(primary, liq["coexistent_phases"]):
        assert phase in tied.split(", ")
    assert (liq["T"].to_numpy() <= temps.max(axis=1) - 273.15 + 1e-6).all()


def test_chunked_evaluation_matches_one_pass(ams, monkeypatch):
    import gliquid.liquidus_sheets as module

    _, sheets = ams
    comp = np.random.default_rng(0).dirichlet([1, 1, 1], 500)[:, 1:]
    whole = sheets(comp)
    monkeypatch.setattr(module, "CHUNK_PAIRS", 7 * len(sheets._row_h))
    monkeypatch.setattr(sheets, "_last", (None, None))
    assert np.array_equal(sheets(comp), whole)


def test_a_repeated_call_returns_a_copy_of_the_result_at_those_compositions(ams):
    _, sheets = ams
    comp = _grid(10)
    first = sheets(comp)
    expected = first.copy()
    first[:] = 0.0
    assert np.array_equal(sheets(comp), expected)
    assert np.array_equal(sheets(comp[::-1])[::-1], expected)


def test_a_kohler_liquidus_figure_raises_no_runtime_warning():
    import warnings

    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        interp_scheme="kohler",
        hull_mode="hsx",
        param_format="linear",
    )
    plotter.process_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plotter.get_plot("tx")


def _gap_sheets(al_mg, delta=0.1, si_al=AMS_LD["Si-Al"]):
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=delta,
        xs_mix={k: list(v) for k, v in {**AMS_LD, "Al-Mg": al_mg, "Si-Al": si_al}.items()},
        hull_mode="hsx",
        param_format="linear",
    )
    plotter.process_data()
    return LiquidusSheets.from_interpolation(plotter._ti)


def test_no_two_liquid_column_without_a_miscibility_gap(ams):
    _, sheets = ams
    assert TWO_LIQUIDS not in sheets.phases
    assert sheets.binodal is None


def test_binodal_and_spinodal_of_a_regular_solution_edge_match_the_closed_forms():
    """A regular Al-Mg liquid, L = 25 kJ: on the edge the binodal is L (1 - 2x) / (R ln((1-x)/x))
    and the spinodal 2 L x (1 - x) / R; the binodal is 0 where its floor is above its ceiling."""
    sheets = _gap_sheets([25000.0, 0.0, 0.0, 0.0])
    assert sheets.phases[-1] == TWO_LIQUIDS
    x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45])
    edge = np.column_stack([x, np.zeros_like(x)])
    binodal = 25000.0 * (1 - 2 * x) / (R * np.log((1 - x) / x))
    assert sheets.binodal(edge) == pytest.approx(binodal, abs=1e-6)
    assert sheets.binodal.spinodal(edge) == pytest.approx(2 * 25000.0 * x * (1 - x) / R, abs=1e-3)
    floor = np.r_[sheets.binodal.ceiling + 1.0, np.zeros(len(x) - 1)]
    assert sheets.binodal(edge, floor) == pytest.approx(np.r_[0.0, binodal[1:]], abs=1e-6)


def _brute_binodal(binodal, points, n=1000):
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    grid = np.column_stack([i[i + j <= n], j[i + j <= n]]) / n
    full = np.column_stack([1 - grid.sum(axis=1), grid])
    tangent = binodal._tangent(points, binodal.tstar(points))
    out = np.zeros(len(points))
    for k in range(len(points)):
        h, s = binodal._h_s(grid, np.full(len(grid), tangent["t"][k]))
        d_h = h - (tangent["a"][k] + grid @ tangent["d"][k])
        d_s = s - (tangent["b"][k] + grid @ tangent["e"][k] - R * (full @ tangent["log_x"][k]))
        with np.errstate(divide="ignore", invalid="ignore"):
            temps = np.where(d_s < 0, d_h / d_s, np.where(d_h < 0, np.inf, 0.0))
        temps = np.maximum(np.nan_to_num(temps, nan=0.0), 0.0)
        temps[((full > 0) & tangent["absent"][k]).any(axis=1)] = 0.0
        temps[np.abs(grid - points[k]).max(axis=1) <= 1e-6] = 0.0
        out[k] = temps.max()
    return out


def test_binodal_matches_a_brute_force_search():
    sheets = _gap_sheets([25000.0, 1.0, 2000.0, -0.5])
    points = np.random.default_rng(3).dirichlet([1, 1, 1], 400)[:, 1:]
    temps = sheets(points)
    competing = points[temps[:, -1] > temps[:, :-1].max(axis=1) - 30][:30]
    assert len(competing) >= 10
    found = sheets.binodal(competing)
    assert found == pytest.approx(_brute_binodal(sheets.binodal, competing), abs=5.0)


def _bi_ce_sn_binodal(sn_bi_entropy, t=1405.0, n=40):
    xs = {
        "Bi-Ce": [-179800.956, -22.4751, -32784.517, 0.0],
        "Ce-Sn": [-144691.343, -13.3173, 20026.731, 0.0],
        "Sn-Bi": [-8068.073, sn_bi_entropy, 511.42, 0.0],
    }
    ti = tmod.TernaryLiquidInterpolation(
        ["Bi", "Ce", "Sn"], xs_mix=xs, param_format="combined", delta=1 / n, interp_scheme="linear"
    )
    ti.init_ref_data()
    nodes = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    field = TstarField(ti.ref_data["T"], nodes, np.full(len(nodes), t))
    return Binodal(ti.liquid_model(), field, nodes, 1 / n)


def test_a_binodal_held_up_by_a_ridge_thinner_than_the_grid_matches_a_fine_scan():
    """Where the liquid's total entropy has indefinite curvature, two liquids touch only along a
    sliver narrower than a hundredth of a grid step; the binodal there matches a 1e-4 scan."""
    binodal = _bi_ce_sn_binodal(23.9818)
    point = np.array([[0.025, 0.475]])
    tangent = binodal._tangent(point, np.array([1405.0]))
    y = point + np.mgrid[-0.03:0.03:601j, -0.03:0.03:601j].reshape(2, -1).T
    y = y[(y > 0).all(axis=1) & (y.sum(axis=1) < 1) & (np.abs(y - point).max(axis=1) > 1e-6)]
    scan = binodal._contact(binodal._take(tangent, np.zeros(len(y), int)), y, 0.0).max()
    assert scan > 1405.0
    assert binodal(point)[0] == pytest.approx(scan, abs=5.0)
    assert not _bi_ce_sn_binodal(0.0)._indefinite


def test_ridge_contacts_of_a_batch_match_each_point_alone():
    """Points where the curvature stays definite, mixed in ahead of indefinite ones, leave the
    ridge contacts of the others unchanged."""
    binodal = _bi_ce_sn_binodal(23.9818)
    points = np.array([[0.3, 0.3], [0.1, 0.2], [0.025, 0.475], [0.6, 0.1], [0.03, 0.46]])
    t = np.full(len(points), 1405.0)
    batch = binodal._ridge(binodal._tangent(points, t))
    alone = [binodal._ridge(binodal._tangent(p[None], t[:1]))[0] for p in points]
    assert (batch[[2, 4]] > 1405.0).all()
    assert batch == pytest.approx(alone, abs=1e-6)


def test_a_binodal_contact_in_a_valley_two_grid_steps_away_is_found():
    """Two liquids touch only in a valley under a thousandth wide, two grid steps from the point;
    the solve at T matches a 1e-4 scan of the valley."""
    binodal = _bi_ce_sn_binodal(23.9818)
    point, t = np.array([[0.0239, 0.5018]]), 1390.0
    tangent = binodal._tangent(point, np.array([t]))
    y = np.mgrid[0.020:0.035:1e-4, 0.40:0.50:1e-4].reshape(2, -1).T
    scan = binodal._contact(binodal._take(tangent, np.zeros(len(y), int)), y, 0.0).max()
    assert scan > t
    assert binodal._solve(point, np.array([t]))[0] == pytest.approx(scan, abs=5.0)


def test_a_descent_past_a_narrow_two_liquid_window_finds_its_upper_root():
    """``f(T)`` is positive only in a window 7 K wide far under the start, which the descent's
    steps jump; the window's upper root is returned."""
    profile_t = np.arange(719.2, 1960.0, 50.0)
    profile_f = np.array(
        [-719, -749, -774, -795, -812, -824, -829, -825, -809, -776, -714, -598, -342]
        + [6, -30, -58, -83, -108, -133, -158, -182, -207, -233, -259, -285],
        dtype=float,
    )

    def f(t):
        return np.interp(np.asarray(t, float), profile_t, profile_f)

    binodal = object.__new__(Binodal)
    binodal.__dict__["_indefinite"] = True
    binodal._solve = lambda comp, t, floor=None, wide=False: np.asarray(t, float) + f(t)
    start = np.array([1915.4])
    found = binodal._descend(np.zeros((1, 2)), start, f(start), np.array([719.2]), False)
    assert found[0] == pytest.approx(profile_t[13] + 50.0 * 6 / 36, abs=1.0)


def test_the_binodal_rises_no_higher_than_the_liquid_spinodal():
    """No two liquids coexist above the liquid's hottest spinodal point, with T* at the node
    liquidus across a Si-Al miscibility gap."""
    ti = tmod.TernaryLiquidInterpolation(
        ["Al", "Mg", "Si"],
        xs_mix={
            "Al-Mg": [-10000.0, 0.0, 0.0, 0.0],
            "Mg-Si": [-10000.0, 0.0, 0.0, 0.0],
            "Si-Al": [60000.0, -5.0, 3000.0, 0.0],
        },
        param_format="comb-exp",
        delta=0.05,
        hull_mode="hsx",
    )
    ti.process_data()
    nodes = ti.liquid_tstar[["x0", "x1"]].to_numpy(float)
    binodal = Binodal(ti.liquid_model(), ti.liquid_tstar_field(), nodes, ti.delta)
    assert binodal(nodes).max() <= binodal.spinodal(nodes).max() + 10.0


def test_chosen_columns_match_the_full_call():
    sheets = _gap_sheets([25000.0, 1.0, 2000.0, -0.5])
    rng = np.random.default_rng(5)
    points = rng.dirichlet([1, 1, 1], 300)[:, 1:]
    full = sheets(points)
    gap = len(sheets.phases) - 1
    which = np.column_stack(
        [rng.integers(0, gap, len(points)), rng.integers(0, gap + 1, len(points))]
    )
    picked = sheets.columns(points, which)
    expected = np.take_along_axis(full, which, axis=1)
    solid = which < gap
    assert picked[solid] == pytest.approx(expected[solid], abs=1e-9)
    competing = (which == gap) & (
        full[:, -1:] >= full[:, :-1].max(axis=1, keepdims=True) - BINODAL_MARGIN
    )
    assert competing.any()
    assert picked[competing] == pytest.approx(expected[competing], abs=1e-6)


def test_grid_nodes_take_the_binodal_solved_without_a_floor():
    binodal = _gap_sheets([25000.0, 1.0, 2000.0, -0.5]).binodal
    nodes = binodal.nodes
    free = binodal(nodes)
    floor = np.random.default_rng(7).uniform(0.0, binodal.ceiling, len(nodes))
    assert binodal(nodes, floor=floor) == pytest.approx(np.where(np.isfinite(free), free, 0.0))


@pytest.mark.parametrize("wide", [False, True])
def test_a_search_from_above_lands_on_the_self_consistent_binodal(wide):
    """A regular Al-Mg liquid has the same binodal at every temperature it is taken at, so the
    downward search from 300 K above it lands on it."""
    binodal = _gap_sheets([25000.0, 0.0, 0.0, 0.0]).binodal
    x = np.array([0.1, 0.2, 0.3])
    edge = np.column_stack([x, np.zeros_like(x)])
    exact = 25000.0 * (1 - 2 * x) / (R * np.log((1 - x) / x))
    found = binodal._search(edge, exact + 300.0, exact - 100.0, wide=wide)
    assert found == pytest.approx(exact, abs=BINODAL_CONSISTENCY)


def test_a_binodal_the_grid_scan_leaves_far_below_its_floor_is_still_found():
    binodal = _gap_sheets([25000.0, 1.0, 2000.0, -0.5]).binodal
    points = np.array([[0.4941, 0.0728], [0.5144, 0.0464], [0.5074, 0.0727]])
    free = binodal(points)
    floor = free - 60.0
    with np.errstate(all="ignore"):
        scanned = binodal._solve(points, binodal.tstar(points), floor)
    assert (scanned < floor - BINODAL_MARGIN).all()
    assert binodal(points, floor=floor) == pytest.approx(free, abs=BINODAL_CONSISTENCY)


def test_a_binodal_missed_at_a_cool_tstar_is_searched_from_the_dome_node_nearby(monkeypatch):
    binodal = _gap_sheets([25000.0, 1.0, 2000.0, -0.5]).binodal
    assert binodal.ceiling > 0
    x = binodal.nodes[binodal._node_b > 0] + [0.3 * binodal.spacing, 0.0]
    x = x[(x > 0).all(axis=1) & (x.sum(axis=1) < 1)]
    expected = binodal(x)
    keep = expected > binodal.tstar(x) - BINODAL_MARGIN
    x, expected = x[keep], expected[keep]
    assert len(x) >= 3
    tstar, solve = binodal.tstar, binodal._solve

    def cooler_off_node(y, t=None):
        out = tstar(y, t) if t is not None else tstar(y)
        on = binodal._grid["tree"].query(np.atleast_2d(y), p=np.inf)[0] <= 1e-9
        return np.where(on, out, out - 3 * BINODAL_MARGIN)

    def missed_when_cool(comp, temps, floor=None, wide=False):
        temps = np.broadcast_to(np.asarray(temps, dtype=float), (len(comp),))
        cut = cooler_off_node(comp) + BINODAL_MARGIN / 2
        return np.where(temps > cut, solve(comp, temps, floor, wide), 0.0)

    monkeypatch.setattr(binodal, "tstar", cooler_off_node)
    monkeypatch.setattr(binodal, "_solve", missed_when_cool)
    found = binodal(x, floor=np.full(len(x), -np.inf))
    assert found == pytest.approx(expected, abs=BINODAL_CONSISTENCY)


def test_a_floored_row_asked_again_returns_its_solved_value_without_solving():
    binodal = _gap_sheets([25000.0, 1.0, 2000.0, -0.5]).binodal
    points = np.random.default_rng(11).dirichlet([1, 1, 1], 40)[:, 1:]
    floor = np.full(len(points), 0.5 * binodal.ceiling)
    first = binodal(points, floor=floor)
    solved = binodal._floored_rows(points, floor)
    binodal._floored_rows = None
    again = binodal(np.vstack([points[::-1], points]), floor=np.r_[floor, floor])
    assert np.array_equal(again, np.r_[first[::-1], first])
    assert solved == pytest.approx(first, abs=1e-6)


def test_a_point_a_hair_inside_a_side_takes_its_own_binodal():
    """With the gap on Si-Al, the Al-Mg side has no two liquids but the liquid a hair inside it
    does, Si being the second liquid's partner."""
    binodal = _gap_sheets(AMS_LD["Al-Mg"], si_al=[60000.0, 0.0, 0.0, 0.0]).binodal
    side = binodal.nodes[(binodal.nodes[:, 1] == 0) & (binodal.nodes[:, 0] > 0)][:3]
    inside = side + [0.0, 9e-7]
    free = binodal(inside)
    assert (binodal(side, floor=np.zeros(len(side))) == 0).all()
    assert (free > 100.0).all()
    assert binodal(inside, floor=np.zeros(len(side))) == pytest.approx(free, abs=1e-6)


def test_a_binodal_whose_first_scan_seed_fails_takes_the_second(monkeypatch):
    sheets = _gap_sheets([25000.0, 1.0, 2000.0, -0.5])
    binodal = sheets.binodal
    at_nodes = sheets(binodal.nodes)
    two = np.flatnonzero(at_nodes[:, -1] > at_nodes[:, :-1].max(axis=1) + 20.0)
    x = binodal.nodes[two] + [0.2 * binodal.spacing, 0.0]
    x = x[(x > 0).all(axis=1) & (x.sum(axis=1) < 1)]
    floor = sheets(x)[:, :-1].max(axis=1)
    t = binodal.tstar(x)
    expected = binodal._solve(x, t, floor)
    clear = expected > floor + 20.0
    x, floor, t, expected = x[clear], floor[clear], t[clear], expected[clear]
    assert len(x) >= 3
    scan = binodal._scan

    def first_seed_fails(tangent, floor=None, nodes=None):
        y1, t1, _, _ = scan(tangent, floor, nodes)
        return [tangent["x"].copy(), np.ones_like(t1), y1, t1]

    monkeypatch.setattr(binodal, "_scan", first_seed_fails)
    assert binodal._solve(x, t, floor) == pytest.approx(expected, abs=1.0)


def test_the_contacts_solve_climbs_from_the_nearest_reentrant_node_contact(monkeypatch):
    sheets = _gap_sheets([25000.0, 1.0, 2000.0, -0.5])
    binodal = sheets.binodal
    assert binodal.ceiling > 0
    at_nodes = sheets(binodal.nodes)
    two = np.flatnonzero(at_nodes[:, -1] > at_nodes[:, :-1].max(axis=1) + 20.0)
    x = binodal.nodes[two] + [0.2 * binodal.spacing, 0.0]
    keep = (x > 0).all(axis=1) & (x.sum(axis=1) < 1)
    x, two = x[keep], two[keep]
    t = binodal.tstar(x)
    expected, contact = binodal._solve_points(x, t)
    assert len(x) >= 3 and (expected > 0).all()
    node_re, node_y = np.zeros_like(binodal._node_re), binodal._node_y.copy()
    node_re[two], node_y[two] = 1.0, contact
    monkeypatch.setattr(binodal, "_node_re", node_re)
    monkeypatch.setattr(binodal, "_node_y", node_y)

    def blind(tangent, floor=None, nodes=None):
        none = np.zeros(len(tangent["x"]))
        return [tangent["x"].copy(), none, tangent["x"].copy(), none.copy()]

    monkeypatch.setattr(binodal, "_scan", blind)
    monkeypatch.setattr(ls, "_RING", np.array([[-50.0, -50.0]]))
    assert (binodal._solve(x, t, wide=True) == 0).all()
    assert binodal._solve(x, t, wide=ls.CONTACTS) == pytest.approx(expected, abs=1.0)
