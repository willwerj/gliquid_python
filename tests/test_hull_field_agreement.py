"""Two hulls describe the same solids, and they can disagree.

``bl.phases`` is built from ``dft_ch.stable_entries``, so every interior fixed-composition
phase in it is a T=0 convex-hull vertex by construction. The HSX hull is then built from those
same names and decides for itself which of them reach the T-x diagram. A vertex clearing the
T=0 hull by a hair can occupy no finite field once temperature and the liquid are in play, and
it is then counted as present everywhere while appearing in no plot.

``hull_field_disagreement`` reports that gap. It is a diagnostic, not a guard: a narrow field is
sometimes the honest answer, and the function's job is to stop the two hulls disagreeing in
silence.
"""

import pytest
from pymatgen.core import Composition

from gliquid.binary import hull_field_disagreement
from gliquid.phase import Phase


class _Fake:
    """The surface ``hull_field_disagreement`` reads."""

    def __init__(self, phases, fractions):
        self.phases = phases
        self._fractions = fractions

    def _phase_x(self, phase):
        return self._fractions[phase.name]


def _model(spec):
    """``spec`` maps name -> (x, has_points).  ``x`` None marks a solution phase."""
    phases, fractions = [], {}
    for name, (x, has_points) in spec.items():
        phases.append(
            Phase(
                phase_type="liquid" if name == "L" else "solid",
                name=name,
                composition=None if x is None else Composition({"Cu": 1 - x, "Mg": x}),
                points=[[0.5, 900.0]] if has_points else [],
            )
        )
        fractions[name] = 0.0 if x is None else x
    return _Fake(phases, fractions)


class TestWhatItReports:
    def test_an_interior_vertex_with_no_field_is_named(self):
        bl = _model({"AB": (0.5, False), "AB2": (0.7, True)})
        assert hull_field_disagreement(bl)["stable_without_field"] == ["AB"]

    def test_a_fully_drawn_hull_reports_nothing(self):
        """The control: a function that always reported would satisfy the case above."""
        bl = _model({"AB": (0.5, True), "AB2": (0.7, True)})
        assert hull_field_disagreement(bl)["stable_without_field"] == []

    def test_endpoints_are_not_interior_and_never_reported(self):
        """A pure-element vertex is an anchor, not a compound with a field to occupy."""
        bl = _model({"A": (0.0, False), "B": (1.0, False), "AB": (0.5, True)})
        assert hull_field_disagreement(bl)["stable_without_field"] == []

    def test_the_liquid_is_not_an_interior_vertex(self):
        bl = _model({"L": (None, True), "AB": (0.5, True)})
        out = hull_field_disagreement(bl)
        assert out["stable_without_field"] == []
        assert "L" not in out["field_without_stable"]

    def test_a_drawn_solution_phase_is_reported_for_inspection_not_as_a_fault(self):
        """Solution phases legitimately draw a field without being fixed-composition
        vertices, so they land in the second bucket rather than the first."""
        bl = _model({"(A)": (None, True), "AB": (0.5, True)})
        out = hull_field_disagreement(bl)
        assert out["stable_without_field"] == []
        assert out["field_without_stable"] == ["(A)"]


class TestItIsNotAnImputationArtifact:
    def test_the_check_reads_phases_not_the_imputed_flag(self):
        """The disagreement is a property of the two hulls, not of imputation, so the check
        reads every interior vertex rather than only the imputed ones."""
        bl = _model({"AB": (0.5, False)})
        for phase in bl.phases:
            assert phase.imputed is False
        assert hull_field_disagreement(bl)["stable_without_field"] == ["AB"]


class TestOrderingIsStable:
    def test_names_come_back_in_phase_order(self):
        bl = _model({"AB": (0.3, False), "AB2": (0.5, False), "AB3": (0.7, False)})
        assert hull_field_disagreement(bl)["stable_without_field"] == ["AB", "AB2", "AB3"]


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_a_boundary_composition_is_excluded(x):
    bl = _model({"edge": (x, False), "AB": (0.5, True)})
    assert hull_field_disagreement(bl)["stable_without_field"] == []
