"""The two equations a ``eut`` invariant contributes, checked for self-consistency.

``BinaryLiquid._eut_constraint_equations`` emits the slope equation (the liquid free-energy
slope at the eutectic equals the slope through the two solids) and the tangent equation
(the lower-energy solid lies on the tangent to the liquid free energy at the eutectic). A
diagram generated from known parameters yields eutectics at which each equation holds at
the true parameters, its residual a small fraction of RT, and from which the tangent
equation recovers ``L0_a``. Runs offline on the shipped Cu-Mg cache.
"""

import pytest
import sympy as sp

from gliquid.binary import BinaryLiquid, t_sym, xb_sym

from ._mpds_builders import by_order as _by_order
from ._mpds_builders import recovered_L0_a as _recovered_L0_a
from ._mpds_builders import residual_rt as _residual_rt

TRUTH = [-30000.0, -5.0, -8000.0, 0.0]
RESIDUAL_RT = 0.25
RTOL = 0.05


def _fixed_phase_at(bl, x):
    fixed = [p for p in bl.phases if p.composition is not None and not p.is_solution]
    return min(fixed, key=lambda p: abs(bl._phase_x(p) - x))


@pytest.fixture(scope="module")
def eutectics():
    """Cu-Mg at the true parameters and every eutectic its generated diagram carries."""
    bl = BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp", pd_ind=0)
    assert not bl.init_error
    bl.update_params(TRUTH)
    bl.update_phase_points()
    entries = bl.hsx.liquidus_invariants()[0]["Eutectics"]
    assert entries, "the fixture parameters generate no eutectic"
    records = []
    for temp_c, x_liq, comps, phases in entries:
        solids = sorted((float(c), p) for c, p in zip(comps, phases) if p != "L")
        record = {
            "type": "eut",
            "comp": float(x_liq),
            "temp": float(temp_c) + 273.15,
            "phases": [solids[0][1], solids[1][1]],
            "phase_comps": [solids[0][0], solids[1][0]],
        }
        records.append(
            (record, _fixed_phase_at(bl, solids[0][0]), _fixed_phase_at(bl, solids[1][0]))
        )
    return bl, records


class TestSelfConsistency:
    def test_both_equations_hold_at_the_true_parameters(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            for entry in bl._eut_constraint_equations(record, lhs, rhs):
                assert _residual_rt(entry, TRUTH) < RESIDUAL_RT, (record, entry[0])

    def test_tangent_equation_recovers_L0_a(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            entries = bl._eut_constraint_equations(record, lhs, rhs)
            (entry,) = [e for e in entries if e[1].startswith("0th order")]
            assert _recovered_L0_a(entry, TRUTH) == pytest.approx(TRUTH[0], rel=RTOL), record

    def test_tangent_without_the_slope_term_lands_farther(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            entries = bl._eut_constraint_equations(record, lhs, rhs)
            (tangent,) = [e for e in entries if e[1].startswith("0th order")]
            x_e, t_e = record["comp"], record["temp"]
            solid = lhs if tangent[1].endswith("lhs") else rhs
            x_s, g_s = bl._phase_x(solid), bl._solid_gibbs_at_T(solid, t_e)
            g_liq = bl.eqs["g_liquid"].subs({xb_sym: x_e, t_sym: t_e})
            flat = ["", "", t_e, sp.Eq(g_liq + g_liq * (x_s - x_e), g_s)]
            assert _residual_rt(flat, TRUTH) > 3 * _residual_rt(tangent, TRUTH)


class TestShape:
    def test_one_slope_and_one_tangent_entry(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            names = [e[0] for e in bl._eut_constraint_equations(record, lhs, rhs)]
            x = round(record["comp"], 2)
            assert names == [f"eut - {x} 1st", f"eut - {x} 0th"]

    def test_tangent_anchors_on_the_lower_energy_solid(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            entries = _by_order(bl._eut_constraint_equations(record, lhs, rhs))
            g1 = bl._solid_gibbs_at_T(lhs, record["temp"])
            g3 = bl._solid_gibbs_at_T(rhs, record["temp"])
            side = "0th order lhs" if g1 <= g3 else "0th order rhs"
            assert side in entries

    def test_entries_carry_the_invariant_temperature(self, eutectics):
        bl, records = eutectics
        for record, lhs, rhs in records:
            for entry in bl._eut_constraint_equations(record, lhs, rhs):
                assert entry[2] == pytest.approx(record["temp"])
