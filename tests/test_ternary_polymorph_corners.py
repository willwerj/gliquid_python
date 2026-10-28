"""Elemental polymorph ladders must enter the TERNARY hull, not just the binary one.

``TernaryLiquidInterpolation`` references its liquid to ``UNARY[el].h_liq``/``s_liq``, which
are cumulative above the DFT GROUND-STATE polymorph. Its solid side must carry the same
ladder, via ``component_data[comp].polymorphs``, or a pure corner melts at ``h_liq / s_liq``
instead of ``t_fusion``:

    T_corner = h_liq / s_liq  <  t_fusion   whenever the ladder has a sub-melting transition

because every such transition contributes ``dH_i / T_i`` with ``T_i < T_melt``.
``gliquid.binary`` is not affected -- ``build_phases_from_chull`` inserts
``component_data[comp].polymorphs`` explicitly, and ``get_ternary_form_en`` is the hull
consumer this module pairs it with.

Runs offline against the two ternary caches that ship for tests:
  Hf-Ti-Zr -- ALL THREE elements carry a sub-melting bcc transition (positive control)
  Al-Mg-Si -- NONE of the three does (negative control: nothing may move)

Every fixture runs in both hull modes: gtx snaps each corner to the T_incr grid, hsx reads it
off a facet and lands on t_fusion itself.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import gliquid.api as api
import gliquid.solution as solution
from gliquid.phase import UNARY
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

AFFECTED = ["Hf", "Ti", "Zr"]  # every corner has a bcc transition below melting
UNAFFECTED = ["Al", "Mg", "Si"]  # no sub-melting transition on any ladder
# 'linear' takes [L0_a, L0_b, L1_a, L1_b]. All zero: an ideal solution, so the corner
# temperature is decided purely by the reference states and the hull -- which is the thing
# under test. Any mixing model gives the same corners (excess terms vanish at x = 1).
_IDEAL = [0.0, 0.0, 0.0, 0.0]
ZERO_MIX = {"Hf-Ti": _IDEAL, "Ti-Zr": _IDEAL, "Zr-Hf": _IDEAL}
ZERO_MIX_AMS = {"Al-Mg": _IDEAL, "Mg-Si": _IDEAL, "Si-Al": _IDEAL}
T_INCR = 10
HULL_MODES = ["gtx", "hsx"]
#: hsx corner temperatures sit on t_fusion to within the reference data's own rounding.
HSX_TOL_K = 0.05


def _expected_corner(el, hull_mode):
    """The rendered corner temperature (K): t_fusion, rounded up to the grid under gtx."""
    t_fusion = UNARY[el].t_fusion
    if hull_mode == "gtx":
        return pytest.approx(math.ceil(t_fusion / T_INCR) * T_INCR, abs=1e-6)
    return pytest.approx(t_fusion, abs=HSX_TOL_K)


def _sub_melting_steps(symbol):
    ref = UNARY[symbol]
    return [
        p
        for p in ref.polymorphs
        if p.t_transition and 0 < p.t_transition < ref.t_fusion and (p.delta_h or 0) > 0
    ]


def _corner_temps(components, xs_mix, hull_mode, **kwargs):
    """Rendered liquidus temperature (K) at each pure-element vertex."""
    ti = TernaryLiquidInterpolation(
        list(components),
        delta=0.25,  # coarse: corners are vertices of any grid, and this keeps the hull cheap
        interp_scheme="linear",
        param_format="linear",
        xs_mix=xs_mix,
        T_incr=T_INCR,
        hull_mode=hull_mode,
        order="given",
        **kwargs,
    )
    ti.interpolate()
    ti.process_data()
    import pandas as pd

    df = pd.concat(ti.equil_df_list, ignore_index=True)
    liq = (
        df[df["Phase"] == "L"]
        .sort_values("T")
        .drop_duplicates(subset=["x0_orig", "x1_orig"], keep="first")
    )
    lookup = {
        (round(float(r.x0_orig), 4), round(float(r.x1_orig), 4)): float(r.T) + 273.15
        for r in liq.itertuples()
    }
    return {
        components[0]: lookup.get((0.0, 0.0)),
        components[1]: lookup.get((1.0, 0.0)),
        components[2]: lookup.get((0.0, 1.0)),
    }, ti


@pytest.fixture(scope="module", params=HULL_MODES)
def htz(request):
    return _corner_temps(AFFECTED, ZERO_MIX, request.param)


@pytest.fixture(scope="module", params=HULL_MODES)
def ams(request):
    return _corner_temps(UNAFFECTED, ZERO_MIX_AMS, request.param)


class TestTheDefectIsRealInTheData:
    """Guards the premise: without these, the tests below could pass vacuously."""

    @pytest.mark.parametrize("el", AFFECTED)
    def test_positive_control_element_has_a_sub_melting_transition(self, el):
        assert _sub_melting_steps(el), f"{el} carries none -- it cannot exercise the fix"

    @pytest.mark.parametrize("el", UNAFFECTED)
    def test_negative_control_element_has_none(self, el):
        assert not _sub_melting_steps(el), f"{el} gained one -- it is no longer a control"

    @pytest.mark.parametrize("el", AFFECTED)
    def test_h_over_s_would_be_wrong(self, el):
        """The broken value and the right one must be far apart, or the assertions below
        cannot tell them apart."""
        ref = UNARY[el]
        assert ref.t_fusion - ref.h_liq / ref.s_liq > 50.0


class TestCornersReachTFusion:
    @pytest.mark.parametrize("el", AFFECTED)
    def test_affected_corner_melts_at_t_fusion(self, htz, el):
        corners, ti = htz
        assert corners[el] is not None, f"no liquid vertex found for {el}"
        assert corners[el] == _expected_corner(el, ti.hull_mode)

    @pytest.mark.parametrize("el", AFFECTED)
    def test_affected_corner_is_not_the_h_over_s_value(self, htz, el):
        """Guards against melting at the extrapolated ground state."""
        corners, _ = htz
        assert abs(corners[el] - UNARY[el].h_liq / UNARY[el].s_liq) > T_INCR

    @pytest.mark.parametrize("el", UNAFFECTED)
    def test_unaffected_corner_unchanged(self, ams, el):
        corners, ti = ams
        assert corners[el] == _expected_corner(el, ti.hull_mode)


class TestLadderOnTheHull:
    def test_high_temperature_polymorphs_are_line_compounds(self, htz):
        _, ti = htz
        names = set(ti.tern_mp_df["Phase Name"])
        for expected in ("beta-Hf (bcc)", "beta-Ti (bcc)", "beta-Zr (bcc)"):
            assert expected in names, f"{expected} missing from the ternary hull: {sorted(names)}"

    def test_ground_states_replace_the_hull_elemental_entries(self, htz):
        """Mirrors build_phases_from_chull: the ladder OWNS the vertex, so the bare
        element symbol must not also appear."""
        _, ti = htz
        names = set(ti.tern_mp_df["Phase Name"])
        assert {"alpha-Hf (hcp)", "alpha-Ti (hcp)", "alpha-Zr (hcp)"} <= names
        assert not ({"Hf", "Ti", "Zr"} & names), f"bare element entry survived: {names}"

    def test_polymorph_names_stay_distinct(self, htz):
        """get_ternary_form_en dedupes by phase name keeping min H; a collision would
        silently drop the high-temperature polymorph and reinstate the wrong corner
        temperature."""
        _, ti = htz
        names = list(ti.tern_mp_df["Phase Name"])
        assert len(names) == len(set(names))

    def test_deepest_formation_energy_is_the_dft_hulls(self, htz):
        """Computed over ALL stable entries, so injecting positive-enthalpy polymorphs
        cannot move it -- and min() cannot hit an empty sequence when a system has no
        stable compounds."""
        _, ti = htz
        pdia, _ = api.get_dft_convexhull(list(AFFECTED), "GGA", data_dir=ti.data_dir)
        expected = min(pdia.get_form_energy_per_atom(e) * 96485.0 for e in pdia.stable_entries)
        assert ti.ternary_meta["deepest_formation_energy"] == pytest.approx(expected)


class TestSolidSolutionExclusion:
    """A structure must never be both an SS surface and a fixed-composition line compound."""

    def test_ss_covered_spacegroups_are_not_line_compounds(self):
        ti = TernaryLiquidInterpolation(
            AFFECTED,
            delta=0.25,
            xs_mix=ZERO_MIX,
            param_format="linear",
            solid_solutions=True,
            order="given",
        )
        ti.interpolate()
        assert ti.ss_models, "no SS models built -- the exclusion is untested"
        excluded = {solution.SS_SPACEGROUPS[p] for p in ti.ss_models}
        names = set(ti.tern_mp_df["Phase Name"])
        for el in AFFECTED:
            for poly in UNARY[el].polymorphs:
                if poly.spacegroup_number in excluded:
                    assert poly.name not in names, (
                        f"{poly.name} is covered by an SS surface (sg "
                        f"{poly.spacegroup_number}) yet also appears as a line compound"
                    )

    def test_without_solid_solutions_nothing_is_excluded(self, htz):
        """Positive control for the test above: with SS off, those same polymorphs DO
        appear -- so the exclusion assertion is not passing for want of any candidates."""
        _, ti = htz
        assert not ti.ss_models
        names = set(ti.tern_mp_df["Phase Name"])
        assert {"beta-Hf (bcc)", "beta-Ti (bcc)", "beta-Zr (bcc)"} <= names


# -- The ladder reaches the figure --
def _plotter(ti):
    """A TLIPlotter over the processed ``_corner_temps`` model ``ti``, its tx frame built."""
    p = TLIPlotter(ti, order="given")
    p.get_plot("tx")  # builds solid_plotting_df as a side effect
    return p


@pytest.fixture(scope="module")
def htz_plot(htz):
    return _plotter(htz[1])


@pytest.fixture(scope="module")
def ams_plot(ams):
    return _plotter(ams[1])


def _corner_xy(components, el):
    i = list(components).index(el)
    return (1.0 if i == 1 else 0.0, 1.0 if i == 2 else 0.0)


def _raw_solid(plotter):
    """The hull's solid rows inside the plotted temperature window."""
    df = pd.concat(plotter.equil_df_list, ignore_index=True)
    return df[(df["Phase"] != "L") & (df["T"] >= plotter.conds[0])]


def _corner_rows(plotter, components, el):
    """(raw_rows, rendered_rows) at ``el``'s vertex, both keyed on the SAME columns.

    Only the equilibrium frames carry ``x0_orig``/``x1_orig``; the floor rows ``_plot_tx``
    appends set just x0/x1/T/Phase/Colors, so those rows have NaN there and a filter on the
    _orig columns silently drops exactly the rows a flooring assertion is about. Both frames
    do share the post-``cartesian_to_ternary`` x0/x1, so resolve the corner's transformed
    coordinates from the raw frame once and key everything on those.
    """
    import numpy as np

    cx, cy = _corner_xy(components, el)
    raw = _raw_solid(plotter)
    seed = raw[
        np.isclose(raw["x0_orig"], cx, atol=1e-6) & np.isclose(raw["x1_orig"], cy, atol=1e-6)
    ]
    assert not seed.empty, f"no solid row at {el}'s vertex -- fixture or convention changed"
    tx0, tx1 = float(seed["x0"].iloc[0]), float(seed["x1"].iloc[0])

    def _pick(df):
        return df[np.isclose(df["x0"], tx0, atol=1e-6) & np.isclose(df["x1"], tx1, atol=1e-6)]

    return _pick(raw), _pick(plotter.solid_plotting_df)


class TestLadderSurvivesIntoTheFigure:
    """The figure keeps every polymorph of the ladder the hull finds, although all the
    polymorphs of one element share its corner."""

    @pytest.mark.parametrize("el", AFFECTED)
    def test_premise_hull_really_carries_a_ladder_here(self, htz_plot, el):
        """Without this the survival assertions below could pass vacuously."""
        at, _ = _corner_rows(htz_plot, AFFECTED, el)
        assert at["Phase"].nunique() >= 2, (
            f"{el}'s corner carries {sorted(at['Phase'].unique())} on the hull -- "
            "a single phase cannot exercise the dedupe"
        )

    @pytest.mark.parametrize("el", AFFECTED)
    def test_every_hull_phase_at_the_corner_is_rendered(self, htz_plot, el):
        raw_at, rendered_at = _corner_rows(htz_plot, AFFECTED, el)
        raw, rendered = set(raw_at["Phase"]), set(rendered_at["Phase"])
        assert rendered == raw, f"{el}: dropped {sorted(raw - rendered)} between hull and figure"

    @pytest.mark.parametrize("el", AFFECTED)
    def test_ladder_segments_stack_instead_of_all_starting_at_the_floor(self, htz_plot, el):
        """Each polymorph occupies its OWN temperature band: the ground state starts at the
        plot floor, every higher polymorph where the one below it ends."""
        _, at = _corner_rows(htz_plot, AFFECTED, el)
        spans = sorted(
            ((float(g["T"].min()), float(g["T"].max()), ph) for ph, g in at.groupby("Phase")),
            key=lambda s: s[1],
        )
        assert len(spans) >= 2, f"{el}: only {spans} -- nothing to stack"
        floor = float(htz_plot.conds[0])

        # The stack starts on the plot floor (conds[0], the grid bottom in Celsius) and its segments
        # tile the corner without inverting or overlapping.
        assert spans[0][0] == pytest.approx(floor, abs=1e-6), (
            f"{el}: stack starts at {spans[0][0]}, not on the plot floor {floor}"
        )
        for lo, hi, ph in spans:
            assert lo <= hi + 1e-6, f"{el}: {ph} segment is inverted ({lo} -> {hi})"
        for (_lo_prev, hi_prev, ph_prev), (lo, _hi, ph) in zip(spans, spans[1:]):
            assert lo == pytest.approx(hi_prev, abs=T_INCR + 1e-6), (
                f"{el}: {ph} starts at {lo} but {ph_prev} ends at {hi_prev} -- not contiguous"
            )
            assert lo > spans[0][0] + 1e-6, (
                f"{el}: {ph} still starts at the bottom of the stack -- segments overlap and "
                f"the {ph_prev}->{ph} transition is invisible"
            )

    def test_no_solid_phase_is_lost_anywhere(self, htz_plot):
        """Not just corners: the dedupe is global, so nothing may vanish at any composition."""
        raw = set(_raw_solid(htz_plot)["Phase"])
        rendered = set(htz_plot.solid_plotting_df["Phase"])
        assert rendered == raw, f"lost {sorted(raw - rendered)}"

    @pytest.mark.parametrize("el", UNAFFECTED)
    def test_negative_control_corner_is_untouched(self, ams_plot, el):
        """No ladder means nothing to keep: exactly one phase, still floored at conds[0].

        This is what protects test_tliplotter_figures.py, whose pins are
        frozen on this very system.
        """
        raw, at = _corner_rows(ams_plot, UNAFFECTED, el)
        assert raw["Phase"].nunique() == 1, (
            f"{el} is no longer a control: {sorted(set(raw['Phase']))}"
        )
        assert at["Phase"].nunique() == 1
        assert float(at["T"].min()) == pytest.approx(float(ams_plot.conds[0]), abs=1e-6)

    def test_negative_control_system_has_no_stacked_composition_at_all(self, ams_plot):
        """The pinned fixture's invariant, asserted rather than assumed: if no composition
        in Al-Mg-Si carries two phases, the per-phase dedupe cannot move its figure pins."""
        raw = _raw_solid(ams_plot)
        stacked = raw.groupby(["x0", "x1"])["Phase"].nunique()
        assert (stacked <= 1).all(), (
            f"{int((stacked > 1).sum())} composition(s) carry >1 phase -- "
            "the figure pins CAN move and must be re-frozen deliberately"
        )
