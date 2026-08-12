"""Tests for gliquid.phase.Phase — the generalized phase representation.

A Phase is valid for any number of components: fixed-stoichiometry phases
(elemental polymorphs, line compounds) carry a pymatgen Composition and scalar
energetics; solution phases (liquid, gas, solid solutions) have composition=None
and evaluate through an attached solution model. The chemistry lives in the
Composition; evaluation axes (e.g. x_B) are DERIVED via fraction_in().

Run with: python -m pytest tests/test_phase.py -v
"""

import pickle
import sys
from pathlib import Path

import pytest
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gliquid.phase import UNARY, ComponentRef, Phase  # noqa: E402


class TestFixedCompositionPhases:
    def test_elemental_polymorph(self):
        p = Phase(
            name="HCP",
            phase_type="solid",
            composition=Composition("Ti"),
            enthalpy=1463.64,
            entropy=8.13,
            t_transition=180.0,
            spacegroup_number=194,
        )
        assert not p.is_solution
        assert p.composition.reduced_formula == "Ti"
        assert p.gibbs(1000.0) == pytest.approx(1463.64 - 1000.0 * 8.13)

    def test_line_compound_from_formula(self):
        p = Phase(
            name="Al2Cu",
            phase_type="solid",
            composition=Composition("Al2Cu"),
            enthalpy=-15000.0,
            entropy=0.0,
        )
        assert not p.is_solution
        assert p.composition.get_atomic_fraction("Cu") == pytest.approx(1 / 3)

    def test_fraction_in_binary_axis(self):
        p = Phase(name="Al2Cu", phase_type="solid", composition=Composition("Al2Cu"))
        assert p.fraction_in(["Al", "Cu"]) == (pytest.approx(1 / 3),)

    def test_fraction_in_endpoints_are_exact(self):
        a = Phase(name="FCC", phase_type="solid", composition=Composition("Al"))
        assert a.fraction_in(["Al", "Cu"]) == (0.0,)
        assert a.fraction_in(["Cu", "Al"]) == (1.0,)

    def test_fraction_in_ternary_axes(self):
        p = Phase(name="AlCuMg", phase_type="solid", composition=Composition("AlCuMg"))
        x = p.fraction_in(["Al", "Cu", "Mg"])
        assert x == (pytest.approx(1 / 3), pytest.approx(1 / 3))

    def test_fraction_in_element_absent_from_phase(self):
        p = Phase(name="Laves", phase_type="solid", composition=Composition("Cu2Mg"))
        x_b, x_c = p.fraction_in(["Al", "Cu", "Mg"])
        assert x_b == pytest.approx(2 / 3) and x_c == pytest.approx(1 / 3)


class TestSolutionPhases:
    def test_no_composition_means_solution(self):
        liq = Phase(name="L", phase_type="liquid")
        assert liq.is_solution
        assert liq.composition is None

    def test_fraction_in_raises_for_solution_phases(self):
        liq = Phase(name="L", phase_type="liquid")
        with pytest.raises(ValueError):
            liq.fraction_in(["Al", "Cu"])

    def test_model_slot_holds_a_solution_model(self):
        from gliquid.solution import RKPolyExp, SolutionModel, t_sym

        m = SolutionModel(
            ("Hf", "Zr"), (0 * t_sym, 0 * t_sym), {(0, 1): RKPolyExp("regular", [8000.0])}
        )
        p = Phase(name="BCC", phase_type="solid", model=m)
        assert p.is_solution and p.model.components == ("Hf", "Zr")


class TestLadderCompatibility:
    """The elemental reference-ladder behavior PhaseRef carried."""

    def test_from_json_with_composition(self):
        p = Phase.from_json(
            {
                "phase_type": "solid",
                "common_name": "BCC",
                "transition_temperature_K": 1200.0,
                "enthalpy_J_per_mol": 3000.0,
                "entropy_J_per_mol_K": 2.5,
                "delta_H_J_per_mol": 3000.0,
                "spacegroup_number": 229,
            },
            composition=Composition("Hf"),
        )
        assert p.name == "BCC" and p.t_transition == 1200.0
        assert p.composition.reduced_formula == "Hf"
        assert not p.is_solution

    def test_registry_phases_carry_their_element_composition(self):
        ref = UNARY["Al"]
        assert ref.phases, "registry should load Al phases"
        assert all(
            p.composition is not None and p.composition.reduced_formula == "Al" for p in ref.phases
        )

    def test_element_ref_ladder_logic_survives(self):
        """H is the stored cumulative liquid enthalpy exactly; the stepwise S
        reconstruction agrees with the stored reference to data-rounding precision
        (phase_transitions.json stores rounded entropies)."""
        ref = UNARY["Al"]
        h, s = ref.liquid_ref_from_solids()
        assert h == pytest.approx(ref.h_liq, rel=1e-12)
        assert s == pytest.approx(ref.s_liq, rel=1e-3)

    def test_imputed_defaults_false_and_points_are_per_instance(self):
        p1 = Phase(name="A", phase_type="solid", composition=Composition("Al"))
        p2 = Phase(name="B", phase_type="solid", composition=Composition("Cu"))
        assert p1.imputed is False
        p1.points.append([0.0, 300.0])
        assert p2.points == []


def test_pickle_round_trip_with_composition():
    p = Phase(
        name="Al2Cu",
        phase_type="solid",
        composition=Composition("Al2Cu"),
        enthalpy=-15000.0,
        entropy=0.0,
        imputed=True,
    )
    clone = pickle.loads(pickle.dumps(p))
    assert clone.name == "Al2Cu" and clone.imputed is True
    assert clone.composition == p.composition
    assert clone.fraction_in(["Al", "Cu"]) == (pytest.approx(1 / 3),)


def test_element_ref_builds_from_phases():
    """ComponentRef is constructed from a few Phase objects (the user's framing)."""
    phases = [
        Phase(
            name="GS",
            phase_type="solid",
            composition=Composition("Fe"),
            t_transition=0.0,
            delta_h=0.0,
            enthalpy=0.0,
            entropy=0.0,
        ),
        Phase(
            name="liquid",
            phase_type="liquid",
            composition=Composition("Fe"),
            t_transition=1811.0,
            delta_h=13810.0,
            enthalpy=13810.0,
            entropy=13810.0 / 1811.0,
        ),
    ]
    ref = ComponentRef("Fe", phases)
    assert ref.t_fusion == 1811.0
    assert ref.h_liq == pytest.approx(13810.0)
