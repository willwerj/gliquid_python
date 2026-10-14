"""gliquid.liquidus_sheets -- per-phase liquidus temperatures from the liquid's tangent planes.

The sheets must reproduce the liquid model the hull is built from, agree with the hull on which
phase is tied to the liquid, never lie below the hull's liquidus, and keep a liquid that lacks an
element from touching a solid containing it. Runs offline (cached Al-Mg-Si MP entries; unary
registry solid solutions for Hf-Ti-Zr).
"""

import numpy as np
import pytest

import gliquid.ternary as tmod
from gliquid.liquidus_sheets import BINODAL_MARGIN, TWO_LIQUIDS, LiquidusSheets
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
    h, s, _, _ = model.nonideal_h_s_gradients((comp[:, 0], comp[:, 1]), sheets.temp)
    h_grid, s_grid = model.h_s_grid((comp[:, 0], comp[:, 1]), sheets.temp)
    full = np.column_stack([1 - comp.sum(axis=1), comp])
    assert h == pytest.approx(h_grid, rel=1e-9)
    assert s - R * (full * np.log(full)).sum(axis=1) == pytest.approx(s_grid, rel=1e-9)


def test_gradients_match_finite_differences(ams):
    plotter, sheets = ams
    model = plotter._ti.liquid_model()
    x = np.array([[0.2, 0.3], [0.5, 0.1], [0.1, 0.7]])
    _, _, dh, ds = model.nonideal_h_s_gradients((x[:, 0], x[:, 1]), sheets.temp)
    step = 1e-6
    for k in range(2):
        shift = np.zeros(2)
        shift[k] = step
        hi = model.nonideal_h_s_gradients(((x + shift)[:, 0], (x + shift)[:, 1]), sheets.temp)
        lo = model.nonideal_h_s_gradients(((x - shift)[:, 0], (x - shift)[:, 1]), sheets.temp)
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
    assert np.array_equal(sheets(comp), whole)


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


def _gap_sheets(al_mg, delta=0.1):
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=delta,
        xs_mix={k: list(v) for k, v in {**AMS_LD, "Al-Mg": al_mg}.items()},
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
    and the spinodal 2 L x (1 - x) / R."""
    sheets = _gap_sheets([25000.0, 0.0, 0.0, 0.0])
    assert sheets.phases[-1] == TWO_LIQUIDS
    x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45])
    edge = np.column_stack([x, np.zeros_like(x)])
    binodal = 25000.0 * (1 - 2 * x) / (R * np.log((1 - x) / x))
    assert sheets.binodal(edge) == pytest.approx(binodal, abs=1e-6)
    assert sheets.binodal.spinodal(edge) == pytest.approx(2 * 25000.0 * x * (1 - x) / R, abs=1e-3)


def _brute_binodal(binodal, points, n=1000):
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    grid = np.column_stack([i[i + j <= n], j[i + j <= n]]) / n
    h, s = binodal._h_s(grid)
    full = np.column_stack([1 - grid.sum(axis=1), grid])
    tangent = binodal._tangent(points)
    out = np.zeros(len(points))
    for k in range(len(points)):
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
