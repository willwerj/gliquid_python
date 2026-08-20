"""Elemental polymorph ladders must enter the TERNARY hull, not just the binary one.

``TernaryLiquidInterpolation`` references its liquid to ``UNARY[el].h_liq``/``s_liq``, which
are cumulative above the DFT GROUND-STATE polymorph. Its solid side used to come from
``pdia.stable_entries`` alone -- at a pure-element vertex, the ground state at formation
energy 0 and nothing else. The liquid therefore crossed the EXTRAPOLATED ground state and a
pure corner melted at ``h_liq / s_liq`` instead of ``t_fusion``:

    T_corner = h_liq / s_liq  <  t_fusion   whenever the ladder has a sub-melting transition

because every such transition contributes ``dH_i / T_i`` with ``T_i < T_melt``. Measured 74 K
low for Sn, 870 K for Ti, across 27 of the 84 elements carrying a liquid reference.
``gliquid.binary`` was never affected -- ``build_phases_from_chull`` inserts
``component_data[comp].polymorphs`` explicitly, and ``get_ternary_form_en`` was the only hull
consumer in the package with no paired ``.polymorphs`` injection.

Runs offline against the two ternary caches that ship for tests:
  Hf-Ti-Zr -- ALL THREE elements carry a sub-melting bcc transition (positive control)
  Al-Mg-Si -- NONE of the three does (negative control: nothing may move)
"""

from __future__ import annotations

import math

import pytest

import gliquid.api as api
import gliquid.solution as solution
from gliquid.phase import UNARY
from gliquid.ternary import TernaryLiquidInterpolation

AFFECTED = ["Hf", "Ti", "Zr"]  # every corner has a bcc transition below melting
UNAFFECTED = ["Al", "Mg", "Si"]  # no sub-melting transition on any ladder
# 'linear' takes [L0_a, L0_b, L1_a, L1_b]. All zero: an ideal solution, so the corner
# temperature is decided purely by the reference states and the hull -- which is the thing
# under test. Any mixing model gives the same corners (excess terms vanish at x = 1).
_IDEAL = [0.0, 0.0, 0.0, 0.0]
ZERO_MIX = {"Hf-Ti": _IDEAL, "Ti-Zr": _IDEAL, "Zr-Hf": _IDEAL}
ZERO_MIX_AMS = {"Al-Mg": _IDEAL, "Mg-Si": _IDEAL, "Si-Al": _IDEAL}
T_INCR = 10


def _sub_melting_steps(symbol):
    ref = UNARY[symbol]
    return [
        p
        for p in ref.polymorphs
        if p.t_transition and 0 < p.t_transition < ref.t_fusion and (p.delta_h or 0) > 0
    ]


def _corner_temps(components, xs_mix, **kwargs):
    """Rendered liquidus temperature (K) at each pure-element vertex."""
    ti = TernaryLiquidInterpolation(
        list(components),
        delta=0.25,  # coarse: corners are vertices of any grid, and this keeps the hull cheap
        interp_scheme="linear",
        param_format="linear",
        xs_mix=xs_mix,
        T_incr=T_INCR,
        order="given",
        **kwargs,
    )
    ti.interpolate()
    ti.process_data()
    import pandas as pd

    df = pd.concat(ti.equil_df_list, ignore_index=True)
    liq = df[df["Phase"] == "L"].sort_values("T").drop_duplicates(
        subset=["x0_orig", "x1_orig"], keep="first"
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


@pytest.fixture(scope="module")
def htz():
    return _corner_temps(AFFECTED, ZERO_MIX)


@pytest.fixture(scope="module")
def ams():
    return _corner_temps(UNAFFECTED, ZERO_MIX_AMS)


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
        corners, _ = htz
        assert corners[el] is not None, f"no liquid vertex found for {el}"
        # The hull snaps to the T_incr grid, so the corner is t_fusion rounded UP.
        expected = math.ceil(UNARY[el].t_fusion / T_INCR) * T_INCR
        assert corners[el] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("el", AFFECTED)
    def test_affected_corner_is_not_the_h_over_s_value(self, htz, el):
        """The specific regression: melting at the extrapolated ground state."""
        corners, _ = htz
        broken = math.ceil(UNARY[el].h_liq / UNARY[el].s_liq / T_INCR) * T_INCR
        assert corners[el] != pytest.approx(broken, abs=1e-6)

    @pytest.mark.parametrize("el", UNAFFECTED)
    def test_unaffected_corner_unchanged(self, ams, el):
        corners, _ = ams
        expected = math.ceil(UNARY[el].t_fusion / T_INCR) * T_INCR
        assert corners[el] == pytest.approx(expected, abs=1e-6)


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
        silently drop the high-temperature polymorph and reinstate the defect."""
        _, ti = htz
        names = list(ti.tern_mp_df["Phase Name"])
        assert len(names) == len(set(names))

    def test_deepest_formation_energy_is_the_dft_hulls(self, htz):
        """Computed over ALL stable entries, so injecting positive-enthalpy polymorphs
        cannot move it -- and min() cannot hit an empty sequence when a system has no
        stable compounds."""
        _, ti = htz
        pdia, _ = api.get_dft_convexhull(list(AFFECTED), "GGA", data_dir=ti.data_dir)
        expected = min(
            pdia.get_form_energy_per_atom(e) * 96485.0 for e in pdia.stable_entries
        )
        assert ti.ternary_meta["deepest_formation_energy"] == pytest.approx(expected)


class TestSolidSolutionExclusion:
    """A structure must never be both an SS surface and a fixed-composition line compound."""

    def test_ss_covered_spacegroups_are_not_line_compounds(self):
        ti = TernaryLiquidInterpolation(
            AFFECTED, delta=0.25, xs_mix=ZERO_MIX, param_format="linear",
            solid_solutions=True, order="given",
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
