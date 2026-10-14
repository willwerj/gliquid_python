"""Two temperature windows: ``T_grid`` is evaluated on, ``temp_range`` is displayed.

``_init_sys`` screens the grid against the liquidus rather than sizing it from the component
melting points. A composition whose liquidus sits above the last rung is solid in every slice,
emits no liquid row, and leaves the surface entirely -- no exception, no clipped temperature,
and ``tricontourf`` then interpolates its neighbours across the gap. The grid must therefore
reach the top of the liquidus, and these pin that it does.

``temp_range`` is that same window's two ends: the grid is evaluated over exactly the range a
figure shows. Its floor is the lower of the liquidus and the frame the three edges' references
are drawn in, so anything stable only below that floor is not evaluated; ``temp_range=(0.0,
None)`` anchors it at absolute zero.

Only gtx pays for the screen, because only gtx has rungs to size. hsx takes its window from
its own facet temperatures in ``_process_data_hsx``, so on that path ``temp_range`` is final
after ``process_data`` rather than after ``_init_sys``.

Runs offline against the two ternary caches that ship for tests; the window arithmetic itself
is exercised on hand-built frames, so it needs no cache at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gliquid.ternary import (
    PREDICTED_EDGE_FLOOR_MARGIN_K,
    PRESCREEN_TOL_K,
    WINDOW_MARGIN_SLICES,
    TernaryLiquidInterpolation,
)

from ._mpds_builders import IDEAL_MIX
from ._mpds_builders import ideal_ternary as _make_ti

ABS_ZERO_C = -273.15
SYSTEMS = sorted(IDEAL_MIX)
T_INCR = 100


@pytest.fixture(scope="module")
def prepared():
    out = {}
    for sys_name in SYSTEMS:
        ti = _make_ti(sys_name)
        ti.interpolate()
        ti._init_sys()
        out[sys_name] = ti
    return out


# -- the window arithmetic, on frames whose answer is known by hand --

CORNERS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


def _hand_built(liquid_h, liquid_s, solid_h=(0.0, 0.0, 0.0)):
    """A three-corner frame whose corner ``i`` melts at ``liquid_h[i] / liquid_s[i]``."""
    rows = []
    for (x0, x1), h in zip(CORNERS, solid_h):
        rows.append({"x0": x0, "x1": x1, "S": 0.0, "H": h, "Phase": "solid"})
    for (x0, x1), h, s in zip(CORNERS, liquid_h, liquid_s):
        rows.append({"x0": x0, "x1": x1, "S": s, "H": h, "Phase": "L"})
    ti = _make_ti("Hf-Ti-Zr")
    ti.proc_df = pd.DataFrame(rows)
    return ti


def test_no_liquid_sits_on_the_hull_at_absolute_zero():
    ti = _hand_built([100.0, 600.0, 1000.0], [1.0, 2.0, 2.0])
    assert ti._liquid_nodes_on_hull(0.0) == 0


def test_every_liquid_sits_on_the_hull_once_it_is_hot_enough():
    ti = _hand_built([100.0, 600.0, 1000.0], [1.0, 2.0, 2.0])
    assert ti._liquid_nodes_on_hull(10_000.0) == 3


def test_the_bracket_finds_the_coldest_and_hottest_corner():
    ti = _hand_built([100.0, 600.0, 1000.0], [1.0, 2.0, 2.0])
    floor, ceiling = ti._liquidus_bracket()
    assert floor == pytest.approx(100.0, abs=PRESCREEN_TOL_K)
    assert ceiling == pytest.approx(500.0, abs=PRESCREEN_TOL_K)


def test_the_bracket_converges_from_above_so_the_ceiling_is_never_short():
    """A ceiling BELOW the liquidus is the failure the screen exists to prevent."""
    ti = _hand_built([100.0, 600.0, 1000.0], [1.0, 2.0, 2.0])
    _, ceiling = ti._liquidus_bracket()
    assert ceiling >= 500.0


def test_a_liquid_with_no_entropy_advantage_never_wins_and_reports_nothing():
    ti = _hand_built([100.0, 600.0, 1000.0], [0.0, 0.0, 0.0])
    assert ti._liquidus_bracket() == (None, None)


def test_the_floor_takes_the_lower_of_the_liquidus_and_the_references(monkeypatch):
    ti = _make_ti("Hf-Ti-Zr")
    monkeypatch.setattr(ti, "_reference_floor", lambda: 300.0)
    assert ti._resolve_window(900.0, 2000.0)[0] == 300.0
    monkeypatch.setattr(ti, "_reference_floor", lambda: 1200.0)
    assert ti._resolve_window(900.0, 2000.0)[0] == 900.0


def test_the_floor_is_clamped_at_absolute_zero(monkeypatch):
    ti = _make_ti("Hf-Ti-Zr")
    monkeypatch.setattr(ti, "_reference_floor", lambda: -500.0)
    assert ti._resolve_window(900.0, 2000.0)[0] == 0.0


def test_a_derived_ceiling_carries_the_margin_and_lands_on_the_lattice(monkeypatch):
    ti = _make_ti("Hf-Ti-Zr")
    monkeypatch.setattr(ti, "_reference_floor", lambda: 300.0)
    ceiling = ti._resolve_window(900.0, 2001.0)[1]
    assert ceiling == (21 + WINDOW_MARGIN_SLICES) * T_INCR
    assert ceiling > 2001.0


def test_a_pinned_ceiling_is_honoured_exactly():
    """An explicit end is the caller's, so the margin must not silently widen it."""
    ti = _make_ti("Hf-Ti-Zr", temp_range=(0.0, 2600.0))
    assert ti._resolve_window(900.0, 2000.0) == (0.0, 2600.0)


def test_the_window_falls_back_when_nothing_is_liquid(monkeypatch):
    """With no liquidus to screen against, the ceiling comes off the melting points."""
    ti = _make_ti("Hf-Ti-Zr")
    ti.init_ref_data()  # _init_sys reaches this branch only after interpolate()
    monkeypatch.setattr(ti, "_reference_floor", lambda: None)
    low, high = ti._resolve_window(None, None)
    assert low == high
    assert high > float(np.max(ti.ref_data["T"]))


# -- the reference floor is the lowest of the three edges --


def _stub_edges(monkeypatch, ti, by_pair):
    """Serve each edge a cached MPDS record, or none at all."""
    import gliquid.cache as cache
    import gliquid.mpds as mpds
    import gliquid.ternary as ternary

    class _Backend:
        def variants(self, sys_name, kind):
            return ["0"] if by_pair.get(sys_name) is not None else []

    monkeypatch.setattr(ternary.cache, "resolve_backend", lambda _=None: _Backend())
    monkeypatch.setattr(
        ternary.mpds,
        "load_mpds_data",
        lambda elements, **kw: ({"temp": by_pair["-".join(sorted(elements))]}, (None, False)),
    )
    assert cache and mpds  # the module attributes patched above are the ones read
    return ti


def test_the_reference_floor_is_the_lowest_edge_axis(monkeypatch):
    ti = _make_ti("Hf-Ti-Zr")
    _stub_edges(
        monkeypatch,
        ti,
        {"Hf-Ti": [500.0, 2500.0], "Ti-Zr": [100.0, 2000.0], "Hf-Zr": [800.0, 2400.0]},
    )
    assert ti._reference_floor() == pytest.approx(100.0 + 273.15)


def test_a_predicted_edge_falls_back_below_its_lower_melting_point(monkeypatch):
    """An edge with no cached diagram contributes the floor the binary plotter would give it."""
    from gliquid.phase import UNARY

    ti = _make_ti("Hf-Ti-Zr")
    _stub_edges(monkeypatch, ti, {"Hf-Ti": None, "Ti-Zr": None, "Hf-Zr": None})
    coldest = min(float(UNARY[element].t_fusion) for element in ("Hf", "Ti", "Zr"))
    assert ti._reference_floor() == pytest.approx(coldest - PREDICTED_EDGE_FLOOR_MARGIN_K)


def test_one_digitized_edge_still_sets_the_floor_when_it_is_lowest(monkeypatch):
    ti = _make_ti("Hf-Ti-Zr")
    _stub_edges(monkeypatch, ti, {"Hf-Ti": [-100.0, 2500.0], "Ti-Zr": None, "Hf-Zr": None})
    assert ti._reference_floor() == pytest.approx(-100.0 + 273.15)


# -- the grid reaches the liquidus, and keeps every invariant it had --


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_no_composition_is_still_solid_at_the_top_of_the_grid(prepared, sys_name):
    """The screen's whole job: nothing may be left above the last rung."""
    ti = prepared[sys_name]
    top = float(np.asarray(ti.T_grid, dtype=float)[-1])
    n_liquid = int((ti.proc_df["Phase"] == "L").sum())
    assert ti._liquid_nodes_on_hull(top) == n_liquid


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_grid_clears_the_liquidus_by_the_margin_it_advertises(prepared, sys_name):
    ti = prepared[sys_name]
    _, ceiling = ti._liquidus_bracket()
    top = float(np.asarray(ti.T_grid, dtype=float)[-1])
    assert top >= ceiling
    assert top - ceiling <= (WINDOW_MARGIN_SLICES + 1) * ti.T_incr


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_grid_starts_on_the_lattice_at_or_above_absolute_zero(prepared, sys_name):
    grid = np.asarray(prepared[sys_name].T_grid, dtype=float)
    assert grid[0] >= 0.0
    assert grid[0] % prepared[sys_name].T_incr == pytest.approx(0.0, abs=1e-9)
    assert (grid >= 0.0).all()


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_conds_still_tracks_the_screened_grid(prepared, sys_name):
    ti = prepared[sys_name]
    grid = np.asarray(ti.T_grid, dtype=float)
    assert float(ti.conds[0]) == pytest.approx(grid[0] + ABS_ZERO_C, abs=1e-9)
    assert float(ti.conds[1]) == pytest.approx(grid[-1] + ABS_ZERO_C, abs=1e-9)


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_display_window_contains_the_whole_liquidus(prepared, sys_name):
    ti = prepared[sys_name]
    floor, ceiling = ti._liquidus_bracket()
    assert ti.temp_range[0] <= floor
    assert ti.temp_range[1] >= ceiling


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_evaluated_range_is_the_displayed_range(prepared, sys_name):
    """One window: every rung the hull is evaluated on is inside what a figure shows."""
    ti = prepared[sys_name]
    grid = np.asarray(ti.T_grid, dtype=float)
    assert ti.temp_range == [pytest.approx(grid[0]), pytest.approx(grid[-1])]


# -- an explicit window wins --


def test_an_explicit_temp_range_overrides_the_derived_one():
    ti = _make_ti("Hf-Ti-Zr", temp_range=[400.0, 2600.0])
    ti.interpolate()
    ti._init_sys()
    assert ti.temp_range == [400.0, 2600.0]


def test_the_lower_margin_widens_the_window_downward():
    """Downward has something to reveal, since the floor is not absolute zero."""
    base, wider = _make_ti("Hf-Ti-Zr"), _make_ti("Hf-Ti-Zr")
    for ti in (base, wider):
        ti.interpolate()
    wider.temp_slider = [300, 0]
    base._init_sys()
    wider._init_sys()
    assert wider.temp_range[0] < base.temp_range[0]
    assert wider.temp_range[0] >= 0.0


def test_an_explicit_temp_range_survives_a_reframe():
    """``with_component_order`` rebuilds the model, so anything it drops is silently lost."""
    ti = _make_ti("Hf-Ti-Zr", temp_range=[400.0, 2600.0])
    assert ti.with_component_order(["Zr", "Ti", "Hf"])._temp_range_arg == [400.0, 2600.0]


def test_without_an_override_the_window_is_derived_per_system():
    ti = _make_ti("Hf-Ti-Zr")
    assert ti.temp_range == []
    ti.interpolate()
    ti._init_sys()
    assert len(ti.temp_range) == 2 and ti.temp_range[1] > ti.temp_range[0]


# -- only gtx pays for the screen --


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_hsx_does_not_run_the_screen(monkeypatch, sys_name):
    """The bisection sizes a grid hsx never reads, so it must not be charged for it."""
    ti = _make_ti(sys_name, hull_mode="hsx")
    ti.interpolate()
    calls = []
    monkeypatch.setattr(ti, "_liquidus_bracket", lambda: calls.append(1) or (None, None))
    ti._init_sys()
    assert calls == []


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_gtx_does_run_the_screen(monkeypatch, sys_name):
    ti = _make_ti(sys_name, hull_mode="gtx")
    ti.interpolate()
    calls = []
    real = ti._liquidus_bracket
    monkeypatch.setattr(ti, "_liquidus_bracket", lambda: calls.append(1) or real())
    ti._init_sys()
    assert calls == [1]


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_hsx_window_starts_from_the_melting_points(sys_name):
    ti = _make_ti(sys_name, hull_mode="hsx")
    ti.interpolate()
    ti._init_sys()
    assert ti.temp_range[1] >= float(np.max(ti.ref_data["T"]))


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_the_hsx_window_is_re_laid_on_its_own_facet_temperatures(sys_name):
    """After process_data the window must bracket the liquidus hsx actually returned."""
    ti = _make_ti(sys_name, hull_mode="hsx")
    ti.interpolate()
    ti._init_sys()
    ti._process_data_hsx()
    liquid = ti.equil_df_list[0]
    temps = liquid.loc[liquid["Phase"] == "L", "T"] + 273.15
    assert ti.temp_range[0] <= float(temps.min())
    assert ti.temp_range[1] >= float(temps.max())


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_conds_still_tracks_the_grid_after_the_hsx_re_lay(sys_name):
    """The one invariant has to survive a window that moves twice."""
    ti = _make_ti(sys_name, hull_mode="hsx")
    ti.interpolate()
    ti._init_sys()
    ti._process_data_hsx()
    grid = np.asarray(ti.T_grid, dtype=float)
    assert float(ti.conds[0]) == pytest.approx(grid[0] + ABS_ZERO_C, abs=1e-9)
    assert float(ti.conds[1]) == pytest.approx(grid[-1] + ABS_ZERO_C, abs=1e-9)
    assert ti.temp_range == [pytest.approx(grid[0]), pytest.approx(grid[-1])]


def test_a_pinned_end_survives_the_hsx_re_lay():
    ti = _make_ti("Hf-Ti-Zr", hull_mode="hsx", temp_range=(0.0, 3000.0))
    ti.interpolate()
    ti._init_sys()
    ti._process_data_hsx()
    assert ti.temp_range == [0.0, 3000.0]


def test_the_hsx_re_lay_never_contracts_the_window():
    """A caller who asked for more room keeps it; the re-lay may only add.

    Driven from a deliberately over-wide starting window, because the natural one is already
    close to the liquidus and would not catch a contraction.
    """
    ti = _make_ti("Hf-Ti-Zr", hull_mode="hsx")
    ti.interpolate()
    ti._init_sys()
    ti._window = (0.0, 9000.0)
    ti._apply_window(*ti._window)
    before = np.asarray(ti.T_grid, dtype=float)
    ti._process_data_hsx()
    after = np.asarray(ti.T_grid, dtype=float)
    assert after[0] <= before[0]
    assert after[-1] >= before[-1]
    assert ti.temp_range == [pytest.approx(after[0]), pytest.approx(after[-1])]


def test_a_lower_floor_asked_for_up_front_survives_the_hsx_re_lay():
    """The absolute-zero case: an initialized floor of 0 K must not be raised."""
    ti = _make_ti("Hf-Ti-Zr", hull_mode="hsx", temp_range=(0.0, None))
    ti.interpolate()
    ti._init_sys()
    ti._process_data_hsx()
    assert ti.temp_range[0] == pytest.approx(0.0)
    assert float(ti.conds[0]) == pytest.approx(ABS_ZERO_C, abs=1e-9)


def test_the_slider_widening_survives_the_hsx_re_lay():
    base, wider = (_make_ti("Hf-Ti-Zr", hull_mode="hsx") for _ in range(2))
    for ti in (base, wider):
        ti.interpolate()
    wider.temp_slider = [400, 500]
    for ti in (base, wider):
        ti._init_sys()
        ti._process_data_hsx()
    assert wider.temp_range[0] <= base.temp_range[0]
    assert wider.temp_range[1] >= base.temp_range[1] + 499.0


@pytest.mark.parametrize("sys_name", SYSTEMS)
def test_every_ceiling_carries_the_same_headroom(sys_name):
    """Screened, melting-point and facet ceilings all go through one padding rule."""
    ti = _make_ti(sys_name)
    assert ti._pad_ceiling(1000.0) == 1000.0 + WINDOW_MARGIN_SLICES * T_INCR
    assert ti._pad_ceiling(1000.1) == 1100.0 + WINDOW_MARGIN_SLICES * T_INCR
    gtx, hsx = _make_ti(sys_name, "gtx"), _make_ti(sys_name, "hsx")
    for ti in (gtx, hsx):
        ti.interpolate()
        ti._init_sys()
    hsx._process_data_hsx()
    # Both paths end within one rung of each other, having padded the same liquidus.
    assert abs(gtx.temp_range[1] - hsx.temp_range[1]) <= T_INCR


def test_the_melting_point_fallback_is_no_wider_than_the_screen():
    """The melting-point fallback pads like everything else."""
    ti = _make_ti("Hf-Ti-Zr")
    ti.init_ref_data()
    hottest = float(np.max(ti.ref_data["T"]))
    _, high = ti._resolve_window(None, None)
    assert high == ti._pad_ceiling(hottest)
    assert high - hottest <= (WINDOW_MARGIN_SLICES + 1) * T_INCR


# -- the degenerate-facet trap --


def _facet_frame(rows):
    """A facet frame shaped like the one ``_process_data_hsx`` builds, in kelvin."""
    return pd.DataFrame(rows, columns=["x0", "x1", "T", "Phase"])


def test_the_span_reduces_each_composition_to_its_coldest_liquid_row():
    frame = _facet_frame(
        [
            (0.0, 0.0, 900.0, "L"),
            (0.0, 0.0, 4.2e7, "L"),  # near-degenerate facet through the same composition
            (0.5, 0.0, 1200.0, "L"),
            (0.5, 0.0, 8.0e6, "L"),
        ]
    )
    assert TernaryLiquidInterpolation._liquidus_span(frame) == (900.0, 1200.0)


def test_a_raw_maximum_would_have_taken_the_degenerate_facet():
    """Guards the reduction itself: without it the span runs to the artifact."""
    frame = _facet_frame([(0.0, 0.0, 900.0, "L"), (0.0, 0.0, 4.2e7, "L")])
    assert float(frame["T"].max()) == pytest.approx(4.2e7)
    assert TernaryLiquidInterpolation._liquidus_span(frame)[1] == 900.0


def test_solid_rows_do_not_enter_the_span():
    frame = _facet_frame(
        [
            (0.0, 0.0, 900.0, "L"),
            (0.0, 0.0, 30.0, "Hf"),
            (0.5, 0.0, 5000.0, "Ti"),
        ]
    )
    assert TernaryLiquidInterpolation._liquidus_span(frame) == (900.0, 900.0)


def test_a_frame_with_no_liquid_yields_no_span():
    assert (
        TernaryLiquidInterpolation._liquidus_span(_facet_frame([(0.0, 0.0, 900.0, "Hf")])) is None
    )
