"""One temperature convention across the ternary pipeline: ``T_grid`` KELVIN, ``conds`` CELSIUS.

``TernaryLiquidInterpolation._init_sys`` builds a KELVIN evaluation grid -- ``G = H - T * S``
needs an absolute temperature, and ``_eval_hull``/``get_convex_hull`` key their per-temperature
slices by it -- while every equilibrium frame is converted to CELSIUS on the way out
(``temp_df["T"] - 273.15``). ``conds`` is the plotted temperature WINDOW, and it used to be
carried in Kelvin yet consumed as a Celsius quantity in three places: the solid-segment floor
in ``TLIPlotter._plot_tx``, and the base triangle plus the z-axis range in
``render_tx_surface``.

Because ``np.min([0, min_temp - 200])`` evaluates to 0 for every system whose elements all melt
above 200 K, ``conds[0]`` read as 0 C when it was meant as 0 K -- so the plot floor sat 273.15 C
ABOVE the bottom of the grid. Two consequences, both covered below: a phase stable only at very
low temperature (Ti's spurious ``Ti (P6/mmm)`` DFT ground state) had its segment floored above
its own ceiling, and ``render_tx_surface`` carried a matching ``- 200`` on the z-axis range.

``conds`` is now DERIVED from the grid -- the same window, in the frame the data is in -- so the
two cannot drift and nothing the hull produces can land outside the plotted range.

Runs offline against the two ternary caches that ship for tests. Hf-Ti-Zr is the positive
control (Ti carries a polymorph whose whole stability window sits below 0 C); Al-Mg-Si is the
negative control (every solid there is stable at 0 C, which is why its frozen figure pins never
moved).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gliquid.plotting import ternary_surface
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

ABS_ZERO_C = -273.15
_IDEAL = [0.0, 0.0, 0.0, 0.0]  # 'linear' [L0_a, L0_b, L1_a, L1_b]: an ideal solution
MIX = {
    "Hf-Ti-Zr": {"Hf-Ti": _IDEAL, "Ti-Zr": _IDEAL, "Zr-Hf": _IDEAL},
    "Al-Mg-Si": {"Al-Mg": _IDEAL, "Mg-Si": _IDEAL, "Si-Al": _IDEAL},
}
SYSTEMS = sorted(MIX)
DELTA = 0.25  # coarse: the convention is grid-independent, and this keeps the hulls cheap
T_INCR = 100

# The floor the old code used: ``np.min([0, min_temp - 200])`` is 0 for both systems here, and
# it was compared against Celsius frames. A phase whose ceiling is below this is what the unit
# error inverted.
OLD_KELVIN_FLOOR_READ_AS_C = 0.0


def _make_ti(sys_name, hull_mode="gtx"):
    return TernaryLiquidInterpolation(
        sys_name.split("-"),
        delta=DELTA,
        interp_scheme="linear",
        param_format="linear",
        xs_mix=MIX[sys_name],
        T_incr=T_INCR,
        hull_mode=hull_mode,
        order="given",
    )


def _build(sys_name):
    """A processed plotter plus its tx figure.

    ``fit_or_pred`` is left empty on purpose: ``_init_sys`` only builds the binary edge
    sub-figures when both it and ``xs_mix`` are populated, and those are not under test here.
    """
    ti = _make_ti(sys_name)
    ti.interpolate()
    p = TLIPlotter(ti, order="given", T_incr=T_INCR)
    p.process_data()
    return p, p.get_plot("tx")  # get_plot also builds solid_plotting_df / liq_plotting_df


@pytest.fixture(scope="module")
def built():
    return {name: _build(name) for name in SYSTEMS}


@pytest.fixture(scope="module")
def plotters(built):
    return {name: p for name, (p, _) in built.items()}


@pytest.fixture(scope="module")
def figures(built):
    return {name: fig for name, (_, fig) in built.items()}


def _raw_solid(plotter):
    df = pd.concat(plotter.equil_df_list, ignore_index=True)
    return df[df["Phase"] != "L"]


def _drawn_floor(conds):
    """Where the base triangle and the corner labels are DRAWN -- the floor lifted clear of
    plotly's clip plane. Read from the renderer so the test cannot drift from it."""
    lo, hi = float(conds[0]), float(conds[1])
    return lo + ternary_surface.FLOOR_INSET_FRAC * (hi - lo)


class TestCondsIsTheGridInCelsius:
    """The single invariant the whole convention reduces to."""

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_conds_is_exactly_the_grid_window_converted(self, plotters, sys_name):
        p = plotters[sys_name]
        grid = np.asarray(p.T_grid, dtype=float)
        assert float(p.conds[0]) == pytest.approx(grid[0] + ABS_ZERO_C, abs=1e-9)
        assert float(p.conds[1]) == pytest.approx(grid[-1] + ABS_ZERO_C, abs=1e-9)

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_holds_in_hsx_mode_too(self, sys_name):
        """hull_mode='hsx' takes the analytic route but shares ``_init_sys``, and it hands
        ``conds`` straight to HSX -- where binary.py's convention is already Celsius."""
        ti = _make_ti(sys_name, hull_mode="hsx")
        ti._init_sys()  # settles the grid and conds; no hull needed for the invariant
        grid = np.asarray(ti.T_grid, dtype=float)
        assert float(ti.conds[0]) == pytest.approx(grid[0] + ABS_ZERO_C, abs=1e-9)
        assert float(ti.conds[1]) == pytest.approx(grid[-1] + ABS_ZERO_C, abs=1e-9)

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_the_grid_is_anchored_at_absolute_zero(self, plotters, sys_name):
        """Every solid is stable at 0 K, so no phase's stability window can fall off the
        bottom of the grid -- which is what lets the plot floor BE the grid floor."""
        p = plotters[sys_name]
        assert float(np.asarray(p.T_grid, dtype=float)[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(p.conds[0]) == pytest.approx(ABS_ZERO_C, abs=1e-9)

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_no_grid_temperature_is_a_negative_absolute_temperature(self, plotters, sys_name):
        """``G = H - T * S`` below 0 K is meaningless. The retired
        ``np.min([0, min_temp - 200])`` produced exactly that for any system containing an
        element that melts below 200 K -- 10 of the 86 in the unary registry do."""
        grid = np.asarray(plotters[sys_name].T_grid, dtype=float)
        assert (grid >= 0.0).all(), f"{sys_name}: grid dips to {grid.min()} K"

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_every_grid_temperature_is_evaluated(self, plotters, sys_name):
        """``_eval_hull`` used to carry an ``if T < self.conds[0]: continue`` guard comparing a
        Kelvin grid value against ``conds``. It never fired; this pins that it never could."""
        p = plotters[sys_name]
        assert len(p.equil_df_list) == len(p.T_grid)


class TestTheFloorIsClampedAtAbsoluteZero:
    """``temp_slider`` is a margin on the plotted window. Downward it has nothing to reveal --
    the floor is already 0 K -- and honouring it would drive ``G = H - T * S`` to a negative
    absolute temperature, flipping the sign of the entropy term for those slices."""

    @pytest.mark.parametrize("lower_margin", [200, 500, 1000])
    def test_a_lower_margin_cannot_push_the_grid_below_absolute_zero(self, lower_margin):
        ti = _make_ti("Hf-Ti-Zr")
        ti.temp_slider = [lower_margin, 0]
        ti.interpolate()
        ti._init_sys()
        grid = np.asarray(ti.T_grid, dtype=float)
        assert grid[0] == pytest.approx(0.0, abs=1e-9), (
            f"temp_slider=({lower_margin}, 0) started the grid at {grid[0]} K"
        )
        assert (grid >= 0.0).all()
        assert float(ti.conds[0]) == pytest.approx(ABS_ZERO_C, abs=1e-9)

    def test_the_upper_margin_still_widens_the_window(self):
        """The clamp must not neuter the margin in the direction that has room."""
        base, wider = _make_ti("Hf-Ti-Zr"), _make_ti("Hf-Ti-Zr")
        for t in (base, wider):
            t.interpolate()
        wider.temp_slider = [0, 300]
        base._init_sys()
        wider._init_sys()
        assert float(wider.conds[1]) > float(base.conds[1]) + 299.0
        assert float(wider.conds[0]) == pytest.approx(float(base.conds[0]), abs=1e-9)

    def test_a_negative_lower_margin_still_raises_the_floor(self):
        """Asking for a floor ABOVE absolute zero is coherent, so it is preserved -- the clamp
        is one-sided, not a blanket ignore of temp_slider[0]."""
        ti = _make_ti("Hf-Ti-Zr")
        ti.temp_slider = [-500, 0]
        ti.interpolate()
        ti._init_sys()
        assert float(np.asarray(ti.T_grid, dtype=float)[0]) == pytest.approx(500.0, abs=1e-9)
        assert float(ti.conds[0]) == pytest.approx(500.0 + ABS_ZERO_C, abs=1e-9)

    @pytest.mark.parametrize("lower_margin", [0, 400])
    def test_conds_still_tracks_the_grid_under_any_margin(self, lower_margin):
        """The one invariant must survive the clamp."""
        ti = _make_ti("Hf-Ti-Zr")
        ti.temp_slider = [lower_margin, 150]
        ti.interpolate()
        ti._init_sys()
        grid = np.asarray(ti.T_grid, dtype=float)
        assert float(ti.conds[0]) == pytest.approx(grid[0] + ABS_ZERO_C, abs=1e-9)
        assert float(ti.conds[1]) == pytest.approx(grid[-1] + ABS_ZERO_C, abs=1e-9)


class TestEachLabelMarksItsOwnVertex:
    """The corner labels are ordered ``[components[0], components[2], components[1]]`` against
    an x array of ``[0, 0.5, 1]`` -- which looks transposed but is not. Checked against the
    COMPOSITION TRANSFORM, since a 3-D camera decides where a vertex lands on screen and a
    rendered image cannot settle it.
    """

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_label_positions_are_their_own_components_vertices(self, figures, plotters, sys_name):
        p = plotters[sys_name]
        # hsx_df carries x0 = fraction of components[1], x1 = fraction of components[2], and
        # cartesian_to_ternary maps (x0, x1) -> (x0 + x1/2, x1*sqrt(3)/2).
        vertices = {}
        for i, el in enumerate(p.components):
            x0, x1 = (1.0 if i == 1 else 0.0), (1.0 if i == 2 else 0.0)
            vertices[el] = (x0 + 0.5 * x1, x1 * np.sqrt(3) / 2)

        (labels,) = [t for t in figures[sys_name].data if getattr(t, "mode", None) == "text"]
        drawn = 0
        for lx, ly, raw in zip(labels.x, labels.y, labels.text):
            if not raw:
                continue
            el = raw.replace("<b>", "").replace("</b>", "")
            assert el in vertices, f"label {el!r} is not a component of {sys_name}"
            ex, ey = vertices[el]
            d = float(np.hypot(float(lx) - ex, float(ly) - ey))
            nearest = min(vertices, key=lambda k: np.hypot(float(lx) - vertices[k][0],
                                                           float(ly) - vertices[k][1]))
            assert nearest == el, (
                f"{sys_name}: label {el} at ({lx}, {ly}) is nearest {nearest}'s vertex "
                f"{vertices[nearest]}, not its own {vertices[el]}"
            )
            assert d < 0.05, f"{sys_name}: label {el} sits {d:.3f} from its vertex"
            drawn += 1
        assert drawn == 3, f"{sys_name}: {drawn} component labels drawn, expected 3"

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_premise_the_three_vertices_are_distinguishable(self, plotters, sys_name):
        """Guards the test above: if two vertices coincided, 'nearest' could not tell a correct
        mapping from a transposed one."""
        pts = []
        for i in range(3):
            x0, x1 = (1.0 if i == 1 else 0.0), (1.0 if i == 2 else 0.0)
            pts.append((x0 + 0.5 * x1, x1 * np.sqrt(3) / 2))
        for a in range(3):
            for b in range(a + 1, 3):
                assert np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) > 0.4


class TestCondsAndTheFramesAgree:
    """conds and the equilibrium frames must be the same quantity in the same unit."""

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_premise_the_frames_really_reach_the_bottom_of_the_grid(self, plotters, sys_name):
        """Without this the containment test below could pass for want of any low-T data."""
        raw = pd.concat(plotters[sys_name].equil_df_list, ignore_index=True)
        assert float(raw["T"].min()) == pytest.approx(ABS_ZERO_C, abs=1e-6), (
            "no frame reaches absolute zero -- the containment assertion is vacuous"
        )

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_no_equilibrium_temperature_falls_outside_conds(self, plotters, sys_name):
        p = plotters[sys_name]
        raw = pd.concat(p.equil_df_list, ignore_index=True)
        lo, hi = float(p.conds[0]), float(p.conds[1])
        assert float(raw["T"].min()) >= lo - 1e-6, (
            f"{sys_name}: hull reaches {raw['T'].min()} C, below the plotted floor {lo}"
        )
        assert float(raw["T"].max()) <= hi + 1e-6, (
            f"{sys_name}: hull reaches {raw['T'].max()} C, above the plotted ceiling {hi}"
        )


class TestSolidSegmentsRestOnTheFloor:
    """``_plot_tx`` floors the lowest solid segment at ``conds[0]``; with conds in the wrong
    unit that floor could sit above the segment's own ceiling and invert it."""

    def test_premise_a_phase_is_stable_only_below_the_old_kelvin_floor(self, plotters):
        """The positive control, asserted rather than assumed: Hf-Ti-Zr must carry a phase
        whose whole stability window sits below the floor the old code used. Without one, the
        unit error is unobservable and every assertion in this class is vacuous."""
        ceilings = _raw_solid(plotters["Hf-Ti-Zr"]).groupby("Phase")["T"].max()
        below = ceilings[ceilings < OLD_KELVIN_FLOOR_READ_AS_C]
        assert not below.empty, (
            "no Hf-Ti-Zr phase tops out below 0 C any more -- the unit error has nothing to "
            f"bite on. Ceilings: {ceilings.to_dict()}"
        )

    def test_negative_control_has_no_such_phase(self, plotters):
        """Al-Mg-Si's every solid is stable at 0 C, so the old floor was above nothing --
        which is why its frozen figure pins never moved."""
        ceilings = _raw_solid(plotters["Al-Mg-Si"]).groupby("Phase")["T"].max()
        assert (ceilings >= OLD_KELVIN_FLOOR_READ_AS_C).all(), (
            f"Al-Mg-Si is no longer a negative control: {ceilings.to_dict()}"
        )

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_lowest_segment_starts_exactly_at_the_floor_everywhere(self, plotters, sys_name):
        """One flat statement, no exceptions: at every composition the solid stack starts on
        the plot floor. The Kelvin/Celsius fallback existed because this could not be said."""
        p = plotters[sys_name]
        floor = float(p.conds[0])
        starts = p.solid_plotting_df.groupby(["x0", "x1"])["T"].min()
        off = starts[(starts - floor).abs() > 1e-6]
        assert off.empty, (
            f"{sys_name}: {len(off)} of {len(starts)} composition(s) do not start at the floor "
            f"{floor}: {off.head().to_dict()}"
        )

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_no_segment_is_inverted(self, plotters, sys_name):
        """The visible symptom of a floor placed above a ceiling."""
        p = plotters[sys_name]
        for (x0, x1, phase), grp in p.solid_plotting_df.groupby(["x0", "x1", "Phase"]):
            lo, hi = float(grp["T"].min()), float(grp["T"].max())
            assert lo <= hi + 1e-6, f"{sys_name}: {phase} at ({x0}, {x1}) runs {lo} -> {hi}"


class TestZAxisRangeMatchesTheData:
    """``render_tx_surface`` reads ``conds`` as Celsius. Its z-axis range used to be
    ``conds -/+ 200`` at both ends: the top 200 was compensating for the unit mismatch, and the
    bottom 200 was clearance for corner labels anchored 150 BELOW the floor. With the floor at
    absolute zero there is nothing below to borrow, so the labels take their offset in screen
    space and the axis is the window itself."""

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_range_is_exactly_conds(self, figures, plotters, sys_name):
        conds = plotters[sys_name].conds
        lo, hi = figures[sys_name].layout.scene.zaxis.range
        assert float(lo) == pytest.approx(float(conds[0]), abs=1e-9)
        assert float(hi) == pytest.approx(float(conds[1]), abs=1e-9)

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_nothing_is_drawn_below_absolute_zero(self, figures, sys_name):
        """The point of the exercise: no coordinate anywhere in the figure -- data, base
        triangle, corner-label anchor, or axis bound -- may be a temperature that cannot
        exist."""
        fig = figures[sys_name]
        assert float(fig.layout.scene.zaxis.range[0]) >= ABS_ZERO_C - 1e-9
        for i, trace in enumerate(fig.data):
            vals = [
                float(v)
                for v in np.ravel(getattr(trace, "z", None) if trace.z is not None else [])
                if v is not None
            ]
            assert all(v >= ABS_ZERO_C - 1e-9 for v in vals), (
                f"{sys_name}: trace {i} ({trace.type}) has z below absolute zero: "
                f"{min(vals)}"
            )

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_corner_labels_are_anchored_on_the_floor_and_offset_in_screen_space(
        self, figures, plotters, sys_name
    ):
        """A z anchor below the floor would put the labels outside the axis range, and plotly
        CLIPS out-of-range 3-D text -- the component labels would silently vanish."""
        conds = plotters[sys_name].conds
        expected = _drawn_floor(conds)
        labels = [t for t in figures[sys_name].data if getattr(t, "mode", None) == "text"]
        assert len(labels) == 1, "the corner-label trace is no longer identifiable by mode"
        assert all(float(v) == pytest.approx(expected, abs=1e-9) for v in labels[0].z)
        assert labels[0].textposition == "bottom center", (
            "the label offset must stay in SCREEN space; a data-space drop below the floor "
            "would be an impossible temperature and would be clipped"
        )
        assert [t for t in labels[0].text if t], "no component labels drawn"

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_every_drawn_z_is_inside_the_range(self, figures, sys_name):
        """The failure this catches: with the floor in the wrong unit, the one segment that
        fell back to the true grid bottom was drawn BELOW the axis and clipped away."""
        fig = figures[sys_name]
        lo, hi = (float(v) for v in fig.layout.scene.zaxis.range)
        checked = 0
        for i, trace in enumerate(fig.data):
            z = getattr(trace, "z", None)
            if z is None:
                continue
            vals = [float(v) for v in np.ravel(z) if v is not None]
            if not vals:
                continue  # legend-only marker traces carry z=[None]
            checked += 1
            assert min(vals) >= lo - 1e-6, (
                f"{sys_name}: trace {i} ({trace.type}) draws down to {min(vals)}, below the "
                f"axis floor {lo}"
            )
            assert max(vals) <= hi + 1e-6, (
                f"{sys_name}: trace {i} ({trace.type}) draws up to {max(vals)}, above the "
                f"axis ceiling {hi}"
            )
        assert checked, f"{sys_name}: no trace carried z values -- assertion was vacuous"

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_base_triangle_sits_on_the_floor(self, figures, plotters, sys_name):
        """Lifted off ``conds[0]`` by FLOOR_INSET_FRAC and no more: exactly at the bound its
        two back edges land on plotly's clip plane and are not drawn at all, and the inset is
        how the axis gets to stop at absolute zero without opening room below it."""
        conds = plotters[sys_name].conds
        axes = [t for t in figures[sys_name].data if getattr(t, "name", None) == "axes"]
        assert len(axes) == 1, "the base-triangle trace is no longer identifiable by name"
        assert all(float(v) == pytest.approx(_drawn_floor(conds), abs=1e-9) for v in axes[0].z)
        # Strictly inside the range, and close enough to the floor to read as sitting on it.
        assert float(axes[0].z[0]) > float(conds[0])
        assert float(axes[0].z[0]) - float(conds[0]) < 0.005 * (float(conds[1]) - float(conds[0]))

    @pytest.mark.parametrize("sys_name", SYSTEMS)
    def test_the_solid_segments_are_not_lifted_with_the_decoration(self, plotters, sys_name):
        """The inset is a drawing nudge for the base triangle only. The DATA -- the solid-phase
        segment floors -- must still sit on the true grid floor."""
        p = plotters[sys_name]
        assert float(p.solid_plotting_df["T"].min()) == pytest.approx(
            float(p.conds[0]), abs=1e-9
        )
