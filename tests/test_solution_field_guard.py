"""``_is_solution_field``: a formula-labelled field that is drawn as a solid solution.

``list_binary_imputation_targets`` inserts an imputed phase at ONE composition, the reference field's
max-temperature vertex. A field with a wide homogeneity range has no such composition, so
the guard keeps it out of the target list.

The three conditions are independent and each is load-bearing:

  * **Wide.** A narrow field's vertex is at an edge trivially.
  * **Off-centre.** A compound with a homogeneity range melts congruently near its own
    composition, so its hot vertex is an interior apex.
  * **Away from the stated composition.** A vertex hard against an edge that sits exactly
    on the label's own composition is a compound, not a solution.

Fields are synthetic; ``cbounds`` mirrors the ``identify_mpds_phases`` shape,
``[[x_min, T], [x_max, T]]``.
"""

import pytest

from gliquid.imputation.binary_imputation import _is_solution_field, _measured_comp


def field(name, comp, lo, hi):
    return {"type": "ss", "name": name, "comp": comp, "cbounds": [[lo, 300.0], [hi, 300.0]]}


# Wide range, hot vertex against an edge, far from the stated composition.
SOLUTIONS = [
    (field("Mg0.1In0.9", 0.7672, 0.0662, 0.7672), ["In", "Mg"]),
    (field("YC0.33", 0.3223, 0.2773, 0.7840), ["Y", "C"]),
    (field("Mn1.2Ga0.8 rt", 0.9996, 0.5576, 0.9996), ["Ga", "Mn"]),
    (field("Zn0.8Mn1.2 ht", 0.0004, 0.0004, 0.3445), ["Mn", "Zn"]),
    (field("U0.4Pu0.6 rt", 0.7478, 0.2650, 0.7623), ["Pu", "U"]),
    (field("Fe13Ge3 rt", 0.0000, 0.0000, 0.2177), ["Fe", "Ge"]),
    (field("Mn2.8Ir1.2 rt", 0.6350, 0.6350, 0.9996), ["Ir", "Mn"]),
    (field("Cd0.5Hg0.5 rt", 0.3018, 0.3018, 0.8708), ["Cd", "Hg"]),
]

COMPOUNDS = [
    # hot vertex is an interior apex
    (field("MgIn rt", 0.5280, 0.4106, 0.6167), ["In", "Mg"]),
    (field("Zn0.7Mn0.3 ht", 0.5299, 0.4205, 0.6296), ["Mn", "Zn"]),
    (field("Fe0.92Si2 ht", 0.7091, 0.6971, 0.7295), ["Fe", "Si"]),
    (field("Fe6.5Ge4 ht", 0.4274, 0.4114, 0.4373), ["Fe", "Ge"]),
]


@pytest.mark.parametrize("phase,components", SOLUTIONS)
def test_wide_edge_vertex_is_a_solution(phase, components):
    assert _is_solution_field(phase, components) is True


@pytest.mark.parametrize("phase,components", COMPOUNDS)
def test_compounds_are_kept(phase, components):
    assert _is_solution_field(phase, components) is False


def test_a_line_compound_has_no_range():
    assert (
        _is_solution_field(
            {"name": "Mg5In2", "comp": 0.7131, "tbounds": [[0.7131, 273.0]]}, ["In", "Mg"]
        )
        is False
    )


def test_narrow_field_with_an_edge_vertex_is_kept():
    """Every line-ish field puts its vertex at an edge; width is what makes it meaningful."""
    assert _is_solution_field(field("Mn2Ga5 rt", 0.2804, 0.2804, 0.3058), ["Ga", "Mn"]) is False


def test_edge_vertex_on_its_own_composition_is_kept():
    """Mn-Zn's ZnMn rt: vertex hard against the right edge, exactly at its composition."""
    assert _is_solution_field(field("ZnMn rt", 0.5000, 0.3654, 0.5000), ["Mn", "Zn"]) is False


def test_terminal_name_is_not_parsed_as_a_formula():
    assert _is_solution_field(field("(Mn, Ir)", 0.9996, 0.6350, 0.9996), ["Ir", "Mn"]) is False


def test_label_over_other_elements_is_not_a_solution():
    assert _is_solution_field(field("Al2Cu", 0.9000, 0.1000, 0.9000), ["Ir", "Mn"]) is False


def test_reversed_cbounds_are_tolerated():
    lo_first = field("Mg0.1In0.9", 0.7672, 0.0662, 0.7672)
    hi_first = field("Mg0.1In0.9", 0.7672, 0.7672, 0.0662)
    assert _is_solution_field(lo_first, ["In", "Mg"]) is True
    assert _is_solution_field(hi_first, ["In", "Mg"]) is True


class TestMeasuredComp:
    """Fractional subscripts carry the composition ``formula_comp`` deliberately drops."""

    @pytest.mark.parametrize(
        "label,components,want",
        [
            ("Cd0.5Hg0.5 rt", ["Cd", "Hg"], 0.5),
            ("Ta0.75Ir0.25", ["Ta", "Ir"], 0.25),
            ("Mn1.2Ga0.8 rt", ["Ga", "Mn"], 0.6),
            ("In0.2Sn0.8", ["In", "Sn"], 0.8),
            ("Mn3Rh rt", ["Rh", "Mn"], 0.75),
        ],
    )
    def test_parses(self, label, components, want):
        assert _measured_comp(label, components) == pytest.approx(want, abs=1e-9)

    @pytest.mark.parametrize(
        "label,components",
        [
            ("(Mn)", ["Ir", "Mn"]),
            ("(Mn, Ir) ht", ["Ir", "Mn"]),
            ("Al2Cu", ["Ir", "Mn"]),
            ("", ["Ir", "Mn"]),
            (None, ["Ir", "Mn"]),
            ("L", ["Ir", "Mn"]),
        ],
    )
    def test_returns_none(self, label, components):
        assert _measured_comp(label, components) is None
