"""The ternary node liquidus, solid melting points and binodal solved with the liquid's Gibbs
energy at temperature, in both hull modes, and the partition's rules at the sides of the
composition triangle.

The reference liquidus and melting points come from lower hulls built here with
``scipy.spatial.ConvexHull`` over ``proc_df``, the liquid's Gibbs energy taken from
``liquid_gibbs`` at each temperature. Runs offline (cached Al-Mg-Si MP entries; unary registry
solid solutions for Hf-Ti-Zr).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull

import gliquid.ternary as tmod
from gliquid.liquidus_sheets import Binodal, LiquidusSheets, TstarField, _snap_to_sides
from gliquid.plotting.liquidus_fields import ADDED_VERTEX_RISE, _cap_added
from gliquid.solution import SolutionModel

# A Si-Al miscibility gap in a liquid whose excess decays with temperature.
AMS_GAP = {
    "Al-Mg": [-10000.0, 0.0, 0.0, 0.0],
    "Mg-Si": [-10000.0, 0.0, 0.0, 0.0],
    "Si-Al": [60000.0, -5.0, 3000.0, 0.0],
}
HTZ = {"Hf-Ti": [2000.0, -1.0, 0, 0], "Ti-Zr": [-1000.0, 0, 0, 0], "Zr-Hf": [500.0, 0, 0, 0]}


def _build(components, xs_mix, mode, **kwargs):
    ti = tmod.TernaryLiquidInterpolation(
        components, xs_mix=xs_mix, param_format="comb-exp", hull_mode=mode, **kwargs
    )
    ti.process_data()
    return ti


def _node_liquidus(ti) -> np.ndarray:
    """Each liquid ``proc_df`` row's liquidus on the processed hull, K."""
    frames = pd.concat(ti.equil_df_list)
    liquid = frames[frames["Phase"] == "L"]
    lowest = liquid.groupby([liquid["x0_orig"].round(4), liquid["x1_orig"].round(4)])["T"].min()
    rows = ti.proc_df[ti.proc_df["Phase"] == "L"]
    keys = zip(rows["x0"].round(4), rows["x1"].round(4))
    return np.array([lowest.get(key, np.nan) for key in keys]) + 273.15


def _on_reference_hull(ti, temp) -> np.ndarray:
    """Which ``proc_df`` rows are vertices of a lower facet of the convex hull at ``temp`` (K)."""
    df = ti.proc_df
    xy, h, s = df[["x0", "x1"]].to_numpy(float), df["H"].to_numpy(), df["S"].to_numpy()
    liquid = (df["Phase"] == "L").to_numpy()
    g = h - temp * s
    g[liquid] = ti.liquid_gibbs(xy[liquid], temp)
    hull = ConvexHull(np.column_stack([xy, g]), qhull_options="Qt")
    out = np.zeros(len(g), dtype=bool)
    out[hull.simplices[hull.equations[:, 2] < -1e-9].ravel()] = True
    return out


def _reference_liquidus(ti, step=50.0, tol=0.05) -> np.ndarray:
    """Each liquid ``proc_df`` row's liquidus, K: the lowest temperature at which it is a vertex
    of a lower facet of the convex hull, bisected to ``tol`` from a ``step`` ladder that ends one
    rung after every row is on; NaN where it is on at the bottom of the ladder."""
    rows = np.flatnonzero((ti.proc_df["Phase"] == "L").to_numpy())

    def on(temp):
        return _on_reference_hull(ti, temp)[rows]

    temps, ladder = [step], [on(step)]
    while not (ladder[-1].all() and len(ladder) > 1 and ladder[-2].all()):
        temps.append(temps[-1] + step)
        ladder.append(on(temps[-1]))
    temps, ladder = np.array(temps), np.array(ladder)
    out = np.full(len(rows), np.nan)
    for i in np.flatnonzero(~ladder[0] & ladder[-1]):
        k = len(temps) - 1 - np.argmax(~ladder[::-1, i])
        low, high = temps[k], temps[k + 1]
        while high - low > tol:
            mid = 0.5 * (low + high)
            low, high = (low, mid) if on(mid)[i] else (mid, high)
        out[i] = 0.5 * (low + high)
    return out


def _ams(mode):
    return tmod.TernaryLiquidInterpolation(
        ["Al", "Mg", "Si"],
        xs_mix=AMS_GAP,
        param_format="comb-exp",
        hull_mode=mode,
        delta=0.1,
        T_incr=20,
    )


@pytest.fixture(scope="module")
def ams_hsx():
    """Al-Mg-Si processed in hull_mode 'hsx'."""
    ti = _ams("hsx")
    ti.process_data()
    return ti


@pytest.fixture(scope="module")
def ams_gtx():
    """Al-Mg-Si processed in hull_mode 'gtx'."""
    ti = _ams("gtx")
    ti.process_data()
    return ti


@pytest.fixture(scope="module")
def reference_of():
    """``reference_of(ti)``: the reference liquidus of Al-Mg-Si's liquid nodes, from the first
    processed model asked; both hull modes share ``proc_df``."""
    found = {}

    def reference_of(ti):
        if "liquidus" not in found:
            found["liquidus"] = _reference_liquidus(ti)
        return found["liquidus"]

    return reference_of


def test_gtx_node_liquidus_is_within_a_kelvin_of_the_reference(ams_gtx, reference_of):
    reference = reference_of(ams_gtx)
    found = _node_liquidus(ams_gtx)
    assert np.isfinite(found).all()
    assert np.isfinite(reference).sum() > 0.9 * len(reference)
    assert np.nanmax(np.abs(found - reference)) <= 1.0


def test_hsx_node_liquidus_is_within_the_check_spacing_of_the_reference(ams_hsx, reference_of):
    reference = reference_of(ams_hsx)
    found = _node_liquidus(ams_hsx)
    assert np.isfinite(found).all()
    assert np.nanmax(np.abs(found - reference)) <= tmod.NODE_LIQUIDUS.check_k
    assert np.nanmedian(np.abs(found - reference)) <= 0.1


def test_the_facet_solve_moves_no_node_with_the_hull_own_linearized_gibbs_energy(
    ams_hsx, monkeypatch
):
    ti = ams_hsx
    rows = np.flatnonzero((ti.proc_df["Phase"] == "L").to_numpy())
    _, simplices, temps = ti._liquid_hull()
    facet_T, facet = ti._row_facets(simplices, temps, rows)
    exact = ti._facet_crossing(simplices, facet, facet_T, rows)

    def linearized(temps, rows=None):
        rows = np.arange(len(ti.proc_df)) if rows is None else rows
        return ti.proc_df["H"].to_numpy()[rows] - temps * ti.proc_df["S"].to_numpy()[rows]

    monkeypatch.setattr(ti, "_row_gibbs", linearized)
    control = ti._facet_crossing(simplices, facet, facet_T, rows)
    inside = np.isfinite(control) & (facet_T >= tmod.NODE_LIQUIDUS.search_k[0])
    assert inside.sum() > 0.9 * len(rows)
    assert np.abs(control - facet_T)[inside].max() <= tmod.NODE_LIQUIDUS.tol_k
    assert np.nanmax(np.abs(exact - facet_T)) > 10.0


def test_a_node_the_hsx_hull_leaves_out_is_solved_on_lower_hulls_and_drawn(ams_hsx):
    ti = ams_hsx
    rows = np.flatnonzero((ti.proc_df["Phase"] == "L").to_numpy())
    df_tx, simplices, temps = ti._liquid_hull()
    full, _ = ti._hsx_liquidus(simplices, temps, rows)
    xy = ti.proc_df[["x0", "x1"]].to_numpy(float)[rows]
    i = int(np.flatnonzero(np.isclose(xy, [0.3, 0.3]).all(axis=1))[0])
    keep = ~(simplices == rows[i]).any(axis=1)
    liquidus, facet = ti._hsx_liquidus(simplices[keep], temps[keep], rows)
    assert facet[i] == -1
    assert liquidus[i] == pytest.approx(full[i], abs=tmod.NODE_LIQUIDUS.hull_tol_k)
    kept_tx = df_tx[np.repeat(keep, 4)].reset_index(drop=True)
    frame = ti._with_node_liquidus(kept_tx, simplices[keep], facet, rows, liquidus)
    own = frame[np.isclose(frame[["x0", "x1"]].to_numpy(float), xy[i]).all(axis=1)]
    assert len(own) == 4
    assert (own["label"] == "L").all() and (own["t"] == liquidus[i]).all()


def test_a_node_back_on_the_gtx_hull_after_leaving_it_takes_its_last_move_onto_it():
    corners = [[3, 0, 1], [3, 1, 2], [3, 2, 0]]
    rungs = [np.array(corners), np.array([[0, 1, 2]]), np.array([[0, 1, 2]]), np.array(corners)]
    proc_df = pd.DataFrame({"x0": [0.0, 1.0, 0.0, 1 / 3], "x1": [0.0, 0.0, 1.0, 1 / 3]})
    T_grid = np.array([100.0, 200.0, 300.0, 400.0])
    frames = [
        pd.DataFrame({"T": np.full(rung.size, temp - 273.15)}) for rung, temp in zip(rungs, T_grid)
    ]

    def row_gibbs(temps, rows):
        temps, rows = np.broadcast_arrays(np.asarray(temps, float), np.asarray(rows))
        return np.where(rows == 3, 350.0 - temps, 0.0)

    ti = SimpleNamespace(proc_df=proc_df, T_grid=T_grid, equil_df_list=frames, _row_gibbs=row_gibbs)
    liquidus = tmod.TernaryLiquidInterpolation._rung_liquidus(ti, rungs, np.array([3]))
    assert liquidus == pytest.approx([350.0], abs=tmod.NODE_LIQUIDUS.tol_k)
    node_rows = [frame["T"][rung.ravel() == 3] for frame, rung in zip(frames, rungs)]
    assert pd.concat(node_rows).to_numpy() == pytest.approx(liquidus[0] - 273.15, abs=1e-9)


def test_gtx_liquid_tstar_is_the_sheets_liquidus_on_the_refined_nodes(ams_gtx):
    ti = ams_gtx
    nodes = ti.liquid_tstar[["x0", "x1"]].to_numpy(float)
    rows = np.flatnonzero((ti.proc_df["Phase"] == "L").to_numpy())
    refined = ti.liquid_tstar.copy()
    ti._set_liquid_tstar(rows, _node_liquidus(ti))
    try:
        on_sheets = LiquidusSheets.from_interpolation(ti).liquidus(nodes)
    finally:
        ti.liquid_tstar = refined
    assert refined["tstar"].to_numpy() == pytest.approx(on_sheets, abs=1e-6)
    assert np.abs(on_sheets - _node_liquidus(ti)).max() > 1.0


def test_both_hull_modes_build_the_same_liquidus_sheets(ams_hsx, ams_gtx):
    grid = tmod.generate_comp_grid(0.05)
    comp = np.column_stack([grid["B"], grid["C"]])
    hsx = LiquidusSheets.from_interpolation(ams_hsx).liquidus(comp)
    gtx = LiquidusSheets.from_interpolation(ams_gtx).liquidus(comp)
    assert hsx == pytest.approx(gtx, abs=0.01)


def test_a_mole_fraction_round_off_from_a_side_is_put_on_the_side(ams_hsx):
    off = np.array([[0.3, np.nextafter(0.7, 0.0)], [1e-16, 0.6], [0.4, 0.6]])
    snapped = _snap_to_sides(off)
    assert 1 - snapped[0].sum() <= 0.0
    assert snapped[1] == pytest.approx([0.0, 0.6], abs=0.0)
    sheets = LiquidusSheets.from_interpolation(ams_hsx)
    assert sheets(off[:1]) == pytest.approx(sheets(np.array([[0.3, 0.7]])), rel=1e-9)
    raw = sheets.point_temperatures(off[:1])
    on_side = sheets.point_temperatures(np.array([[0.3, 0.7]]))
    assert (raw > on_side).any()


def test_hsx_solid_melting_points_are_where_each_solid_leaves_the_reference_hull(ams_hsx):
    ti = ams_hsx
    df = ti.proc_df
    rows = np.flatnonzero((df["Phase"] != "L").to_numpy())
    names = df["Phase"].to_numpy()[rows]
    tops = ti.get_inter_melting_temps(list(names))
    _, simplices, temps = ti._liquid_hull()
    facet_top, _ = ti._row_facets(simplices, temps, rows, hottest=True)
    found, reference = np.array([tops[name] + 273.15 for name in names]), []
    for row, top in zip(rows, found):
        low, high = top - 5.0, top + 5.0
        assert _on_reference_hull(ti, low)[row] and not _on_reference_hull(ti, high)[row]
        while high - low > 0.02:
            mid = 0.5 * (low + high)
            low, high = (mid, high) if _on_reference_hull(ti, mid)[row] else (low, mid)
        reference.append(0.5 * (low + high))
    assert found == pytest.approx(reference, abs=0.05)
    assert np.abs(facet_top - reference).max() > 0.25


def test_a_binodal_far_above_tstar_is_the_liquid_gap_at_its_own_temperature():
    ti = _ams("hsx")
    nodes = tmod.generate_comp_grid(0.1)
    tstar = TstarField([933.47, 923.0, 1687.0])
    binodal = Binodal(ti.liquid_model(), tstar, np.column_stack([nodes["B"], nodes["C"]]), 0.1)
    comp = np.array([[0.0, 0.2], [0.0, 0.5]])
    y = np.linspace(0.0, 1.0, 2001)
    edge = np.column_stack([np.zeros_like(y), y])

    def split(x, temp):
        g = ti.liquid_gibbs(edge, temp)
        hull = ConvexHull(np.column_stack([y, g]))
        lower = np.unique(hull.simplices[hull.equations[:, 1] < 0].ravel())
        lower = lower[np.argsort(y[lower])]
        return ti.liquid_gibbs([0.0, x], temp) > np.interp(x, y[lower], g[lower]) + 1e-6

    reference = []
    for x in comp[:, 1]:
        low, high = 1000.0, 4000.0
        while high - low > 0.05:
            mid = 0.5 * (low + high)
            low, high = (mid, high) if split(x, mid) else (low, mid)
        reference.append(0.5 * (low + high))
    assert binodal(comp) == pytest.approx(reference, abs=1.0)
    assert np.abs(binodal._solve(comp, tstar(comp)) - reference).min() > 5.0


def test_the_cap_lowers_an_added_needle_to_its_ring_and_leaves_input_nodes_alone():
    faces = np.array([[6, k, (k + 1) % 6] for k in range(6)])
    T = np.r_[np.full(6, 1000.0) + np.arange(6), 1500.0]
    assert _cap_added(T, faces, first=6)[6] == pytest.approx(1005.0 + ADDED_VERTEX_RISE)
    assert _cap_added(T, faces, first=6)[:6] == pytest.approx(T[:6])
    assert _cap_added(T, faces, first=7) == pytest.approx(T)


def test_the_cap_holds_every_added_vertex_of_a_cluster_to_its_ring():
    faces = np.array([[6, k, (k + 1) % 6] for k in range(5)] + [[6, 5, 7], [7, 5, 0], [7, 0, 6]])
    T = np.r_[np.full(6, 1000.0) + np.arange(6), 2000.0, 1500.0]
    capped = _cap_added(T, faces, first=6)
    ring = np.full(len(T), -np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
        np.maximum.at(ring, faces[:, a], capped[faces[:, b]])
    assert capped[6:] == pytest.approx([1500.0 + ADDED_VERTEX_RISE, 1500.0])
    assert (capped[6:] <= ring[6:] + ADDED_VERTEX_RISE).all()


@pytest.mark.parametrize("mode", ["hsx", "gtx"])
def test_a_solid_solution_ternary_solves_every_node_on_exact_solution_surfaces(mode):
    ti = _build(
        ["Hf", "Ti", "Zr"],
        HTZ,
        mode,
        delta=0.2,
        solid_solutions=True,
        ss_kwargs={"ref_mode": "from_unary_db"},
    )
    assert set(ti.ss_models) & set(pd.concat(ti.equil_df_list)["Phase"])
    assert np.isfinite(_node_liquidus(ti)).all()
    phase, model = next(iter(ti.ss_models.items()))
    surface = SolutionModel.from_ss_model(ti.components, model, ti.interp_scheme)
    rows = np.flatnonzero((ti.proc_df["Phase"] == phase).to_numpy())
    x = ti.proc_df[["x0", "x1"]].to_numpy(float)[rows]
    for temp in (500.0, 2500.0):
        h, s = surface.h_s_grid((x[:, 0], x[:, 1]), temp)
        assert ti._row_gibbs(temp, rows) == pytest.approx(h - temp * s, abs=1e-6)
