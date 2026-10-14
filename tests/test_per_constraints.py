"""The equation a ``per`` invariant contributes, checked for self-consistency.

``BinaryLiquid._per_constraint_equation`` emits the tangent equation: the forming compound
lies on the tangent to the liquid free energy at the invariant liquid composition. A diagram
generated from known parameters yields peritectics at which the equation holds at the true
parameters, its residual a small fraction of RT, and from which it recovers ``L0_a``.
``_reacting_solid`` finds the solid the liquid reacts with. Runs offline on the shipped
Al-Cu and C-Nb caches.
"""

import math

import pytest

from gliquid.binary import BinaryLiquid

from ._mpds_builders import recovered_L0_a as _recovered_L0_a
from ._mpds_builders import residual_rt as _residual_rt

TRUTH = {
    "Al-Cu": [-40000.0, -8.0, -10000.0, 0.0],
    "C-Nb": [-90000.0, -15.0, -20000.0, 0.0],
}
RESIDUAL_RT = 0.25
RTOL = 0.05
TANGENT = -math.inf  # t_tol forcing the tangent form of the 0th equation
CONGRUENT = math.inf  # t_tol forcing the pseudo-congruent form


def _fixed_phases(bl):
    return [p for p in bl.phases if p.composition is not None and not p.is_solution]


def _fixed_phase_at(bl, x):
    return min(_fixed_phases(bl), key=lambda p: abs(bl._phase_x(p) - x))


def _load(system):
    bl = BinaryLiquid.from_cache(system, param_format="comb-exp", pd_ind=0)
    assert not bl.init_error
    bl.update_params(TRUTH[system])
    bl.update_phase_points()
    return bl


@pytest.fixture(scope="module", params=sorted(TRUTH))
def peritectics(request):
    """A system at its true parameters and every peritectic its generated diagram carries.

    Each record comes with the forming compound's hull phase and the composition of the
    solid the liquid reacts with, read off the generated three-phase tie.
    """
    bl = _load(request.param)
    entries = bl.hsx.liquidus_invariants()[0]["Peritectics"]
    assert entries, f"{request.param}: the fixture parameters generate no peritectic"
    records = []
    for temp_c, _, comps, phases in entries:
        tie = sorted((float(c), p) for c, p in zip(comps, phases))
        liq = next(i for i, (_, p) in enumerate(tie) if p == "L")
        assert liq != 1, f"{request.param}: liquid inside the tie {tie}"
        beta, alpha = tie[1], (tie[2] if liq == 0 else tie[0])
        record = {
            "type": "per",
            "comp": tie[liq][0],
            "temp": float(temp_c) + 273.15,
            "phases": [beta[1]],
            "phase_comps": [beta[0]],
        }
        records.append((record, _fixed_phase_at(bl, beta[0]), alpha[0]))
    return bl, TRUTH[request.param], records


class TestSelfConsistency:
    def test_the_tangent_equation_holds_at_the_true_parameters(self, peritectics):
        bl, params, records = peritectics
        for record, per_phase, _ in records:
            entry = bl._per_constraint_equation(record, per_phase, t_tol=TANGENT)
            assert entry[1] == "0th order"
            assert _residual_rt(entry, params) < RESIDUAL_RT, (record, entry[0])

    def test_tangent_equation_recovers_L0_a(self, peritectics):
        bl, params, records = peritectics
        for record, per_phase, _ in records:
            entry = bl._per_constraint_equation(record, per_phase, t_tol=TANGENT)
            assert _recovered_L0_a(entry, params) == pytest.approx(params[0], rel=RTOL), record

    def test_pseudo_congruent_form_lands_farther_than_the_tangent(self, peritectics):
        bl, params, records = peritectics
        for record, per_phase, _ in records:
            pseudo = bl._per_constraint_equation(record, per_phase, t_tol=CONGRUENT)
            tangent = bl._per_constraint_equation(record, per_phase, t_tol=TANGENT)
            assert pseudo[1] == "H-S partition 0th order"
            assert _residual_rt(pseudo, params) > _residual_rt(tangent, params), record


class TestReactingSolid:
    def test_is_the_hull_neighbour_beyond_the_compound(self, peritectics):
        bl, _, records = peritectics
        for record, per_phase, x_alpha in records:
            found = bl._reacting_solid(record["comp"], per_phase)
            assert found is not None
            assert found["x"] == pytest.approx(x_alpha, abs=0.011)
            assert isinstance(found["matches_mpds"], bool)

    def test_pure_component_when_nothing_lies_between(self):
        bl = _load("Al-Cu")
        found = bl._reacting_solid(0.7, _fixed_phase_at(bl, 0.75))
        assert found["x"] == 1.0
        assert bl._phase_x(found["phase"]) == 1.0

    def test_lies_on_the_far_side_of_the_compound(self):
        bl = _load("Al-Cu")
        found = bl._reacting_solid(0.9, _fixed_phase_at(bl, 0.75))
        assert found["x"] == pytest.approx(0.6923, abs=0.001)

    def test_side_follows_the_digitized_compound(self):
        bl = _load("Al-Cu")
        per_phase = _fixed_phase_at(bl, 0.5)
        assert bl._reacting_solid(0.49, per_phase)["x"] == pytest.approx(0.6923, abs=0.001)
        found = bl._reacting_solid(0.49, per_phase, x_beta_ref=0.48)
        assert found["x"] == pytest.approx(0.3333, abs=0.001)

    def test_none_beyond_an_endpoint(self):
        bl = _load("Al-Cu")
        assert bl._reacting_solid(0.9, _fixed_phase_at(bl, 1.0)) is None

    def test_digitized_neighbour_off_the_hull_is_flagged(self):
        bl = _load("Al-Cu")
        per_phase = _fixed_phase_at(bl, 0.5)
        previous = bl.low_t_exp_phases
        try:
            bl.low_t_exp_phases = [{"name": "AlCu", "comp": 0.5}, {"name": "X", "comp": 0.60}]
            found = bl._reacting_solid(0.45, per_phase)
            assert found["x"] == pytest.approx(0.6923, abs=0.001)
            assert found["x_mpds"] == 0.60
            assert found["matches_mpds"] is False
            bl.low_t_exp_phases = [{"name": "Al4Cu9", "comp": 0.69}]
            assert bl._reacting_solid(0.45, per_phase)["matches_mpds"] is True
        finally:
            bl.low_t_exp_phases = previous


class TestSelection:
    def test_the_entry_is_named_after_the_invariant(self, peritectics):
        bl, _, records = peritectics
        for record, per_phase, _ in records:
            entry = bl._per_constraint_equation(record, per_phase, t_tol=TANGENT)
            assert entry[0] == f"per - {round(record['comp'], 2)} 0th"
            assert entry[2] == pytest.approx(record["temp"])
