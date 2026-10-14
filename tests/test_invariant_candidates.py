"""Which invariant-derived equations become Nelder-Mead candidates.

``BinaryLiquid._invariant_constraint_ics`` turns the invariant equations ``fit_parameters``
collects into initial conditions, one per candidate constraint set. A format consuming two
invariant equations per candidate pairs the hottest equation with each other one; a format
consuming one equation makes every equation its own candidate, solved together with the
format's identity constraints. Runs offline on the shipped Cu-Mg cache.
"""

import math

import pytest
import sympy as sp

from gliquid.binary import BinaryLiquid
from gliquid.solution import l_sym

L0_a, L0_b, L1_a, L1_b = (l_sym(n) for n in ("L0_a", "L0_b", "L1_a", "L1_b"))


def _bl(param_format):
    bl = BinaryLiquid.from_cache("Cu-Mg", param_format=param_format)
    assert not bl.init_error
    bl.f = lambda guess, **kwargs: 1.0
    return bl


def _two_eq_equations():
    """Three solvable equations for a two-equation format: the hottest fixes L0_a + L1_a."""
    return [
        ["eut - 0.2 1st", "1st order", 900.0, sp.Eq(L0_a - L1_a, -1000.0)],
        ["cmp - 0.3 0th", "0th order", 1100.0, sp.Eq(L0_a + L1_a, -3000.0)],
        ["eut - 0.6 1st", "1st order", 800.0, sp.Eq(L0_a - L1_a, -2000.0)],
    ]


def _one_eq_equations(n):
    return [
        [f"eut - {i} 1st", "1st order", 700.0 + 100.0 * i, sp.Eq(L0_a, -1000.0 * (i + 1))]
        for i in range(n)
    ]


class TestTwoEquationFormat:
    def test_hottest_equation_pairs_with_each_other_one(self):
        bl = _bl("linear")
        ics = bl._invariant_constraint_ics(_two_eq_equations())
        assert len(ics) == 2
        for ic in ics:
            names = [entry[0] for entry in ic["constrs"]]
            assert "cmp - 0.3 0th" in names
            assert len(names) == 2

    def test_lone_equation_yields_no_candidate(self):
        bl = _bl("linear")
        assert bl._invariant_constraint_ics(_two_eq_equations()[:1]) == []


class TestOneEquationFormat:
    def test_every_equation_is_its_own_candidate(self):
        bl = _bl("comb-exp")
        ics = bl._invariant_constraint_ics(_one_eq_equations(3))
        assert sorted(ic["constrs"][0][0] for ic in ics) == sorted(
            e[0] for e in _one_eq_equations(3)
        )

    def test_lone_equation_is_a_candidate(self):
        bl = _bl("comb-exp")
        ics = bl._invariant_constraint_ics(_one_eq_equations(1))
        assert len(ics) == 1
        assert ics[0]["constrs"][0][0] == "eut - 0 1st"

    def test_identity_constraints_complete_each_candidate(self):
        bl = _bl("comb-exp")
        ics = bl._invariant_constraint_ics(_one_eq_equations(2))
        for ic in ics:
            orders = [entry[1] for entry in ic["constrs"][1:]]
            assert orders and all(o == "0th order" for o in orders)
            assert all(math.isinf(entry[2]) for entry in ic["constrs"][1:])


class TestCandidateShape:
    def test_false_equations_are_dropped(self):
        bl = _bl("comb-exp")
        false_entry = ["per - 0.5 0th", "0th order", 950.0, sp.false]
        with_false = bl._invariant_constraint_ics(_one_eq_equations(3) + [false_entry])
        without = bl._invariant_constraint_ics(_one_eq_equations(3))
        assert all(entry[3] is not sp.false for ic in with_false for entry in ic["constrs"])
        assert [ic["constrs"][0][0] for ic in with_false] == [ic["constrs"][0][0] for ic in without]

    def test_infinite_initial_objective_skips_the_candidate(self):
        bl = _bl("comb-exp")
        bl.f = lambda guess, **kwargs: float("inf")
        assert bl._invariant_constraint_ics(_one_eq_equations(3)) == []
        assert bl.dropped_candidates == ["eut - 0 1st", "eut - 1 1st", "eut - 2 1st"]

    def test_candidate_carries_objective_constraints_and_simplex(self):
        bl = _bl("comb-exp")
        ics = bl._invariant_constraint_ics(_one_eq_equations(3))
        assert ics
        for ic in ics:
            assert set(ic) == {"f", "constrs", "init_tri"}
            assert ic["f"] == pytest.approx(1.0)
            assert ic["init_tri"] == bl.init_triangle


class TestAnUndeterminedCandidate:
    """A candidate survives only when solving its equations determines every solve symbol."""

    def test_a_contradiction_drops_the_candidate(self):
        bl = _bl("comb-exp")
        contradiction = ["mig - 0.3 2nd", "2nd order", 1200.0, sp.Eq(L1_b, 5.0)]
        assert bl._invariant_constraint_ics([contradiction]) == []
        assert bl.dropped_candidates == ["mig - 0.3 2nd"]

    def test_an_equation_in_guess_symbols_alone_drops_the_candidate(self):
        bl = _bl("comb-exp")
        guess_only = ["mig - 0.3 3rd", "3rd order", 1200.0, sp.Eq(L1_a, -3000.0)]
        assert bl._invariant_constraint_ics([guess_only]) == []
        assert bl.dropped_candidates == ["mig - 0.3 3rd"]

    def test_a_partial_solve_drops_the_candidate_in_a_two_equation_format(self):
        bl = _bl("linear")
        hottest = ["cmp - 0.3 0th", "0th order", 1100.0, sp.Eq(L0_a + L1_a, -3000.0)]
        guess_only = ["eut - 0.2 1st", "1st order", 900.0, sp.Eq(L0_b, 1.0)]
        assert bl._invariant_constraint_ics([guess_only, hottest]) == []
        assert bl.dropped_candidates == ["eut - 0.2 1st"]

    def test_an_inconsistent_pair_drops_the_candidate(self):
        bl = _bl("linear")
        hottest = ["cmp - 0.3 0th", "0th order", 1100.0, sp.Eq(L0_a + L1_a, -3000.0)]
        inconsistent = ["eut - 0.2 1st", "1st order", 900.0, sp.Eq(L0_a + L1_a, -1000.0)]
        assert bl._invariant_constraint_ics([inconsistent, hottest]) == []
        assert bl.dropped_candidates == ["eut - 0.2 1st"]

    def test_the_solvable_candidates_beside_it_survive(self):
        bl = _bl("comb-exp")
        guess_only = ["mig - 0.3 3rd", "3rd order", 1200.0, sp.Eq(L1_a, -3000.0)]
        ics = bl._invariant_constraint_ics(_one_eq_equations(3) + [guess_only])
        assert [ic["constrs"][0][0] for ic in ics] == [e[0] for e in _one_eq_equations(3)]
        assert bl.dropped_candidates == ["mig - 0.3 3rd"]

    def test_a_determined_candidate_is_not_dropped(self):
        bl = _bl("comb-exp")
        determined = ["eut - 0.4 1st", "1st order", 1200.0, sp.Eq(L0_a, -1000.0)]
        ics = bl._invariant_constraint_ics([determined])
        assert [ic["constrs"][0][0] for ic in ics] == ["eut - 0.4 1st"]
        assert bl.dropped_candidates == []

    def test_a_survivor_determines_every_solve_symbol(self):
        bl = _bl("comb-exp")
        solve_syms = tuple(bl.xs_mix.solve_symbols)
        ics = bl._invariant_constraint_ics(_one_eq_equations(2))
        assert ics
        for ic in ics:
            solved = sp.solve(
                [entry[3] for entry in ic["constrs"]], solve_syms, rational=False, simplify=False
            )
            assert all(sym in solved for sym in solve_syms)
