"""One rule writes every compound label in the binary figures.

A compound reaches the figures from three places -- pymatgen's ``reduced_formula`` on the DFT
hull, the MPDS phase name on an imputed vertex, and the MPDS phase name again on the T-x field
labels -- and each spells it differently. ``reduced_formula`` orders elements by
ELECTRONEGATIVITY, so a panel whose axis runs Au to Ga labels its vertices ``Ga2Au7``; an MPDS
name carries polymorph suffixes (``rt``, ``ht``) and decimal stoichiometry (``Ag2.4Ce0.6``).

``standard_formula`` is the single rule: the system's own component order, whole-number
stoichiometry, nothing outside ``[A-Za-z0-9.]``. Every case below carries the form it must NOT
produce.
"""

import re

import pytest
from pymatgen.core import Composition

import gliquid.plotting.style as style
from gliquid.plotting.binary_figs import standardized_vertex_label
from gliquid.plotting.style import standard_formula

# (source label, system components, expected label). Every entry is a real MPDS phase name
# except the last, which is the composition-only path.
WORKED_EXAMPLES = [
    ("Cu33Al17 rt", ["Al", "Cu"], "Al17Cu33"),
    ("Au7Ga2 rt", ["Au", "Ga"], "Au7Ga2"),
    ("Ag2.4Ce0.6", ["Ag", "Ce"], "Ag4Ce"),
    ("YSi1.67 ht", ["Si", "Y"], "Si5Y3"),
    ("Co1.5Sn ht", ["Co", "Sn"], "Co3Sn2"),
    ("Zn0.5Co0.5 rt", ["Co", "Zn"], "CoZn"),
    ("Tl0.8Sn0.2", ["Sn", "Tl"], "SnTl4"),
    ("Au0.87In0.13 ht", ["Au", "In"], "Au87In13"),
    ("Ni0.97Ga3.62", ["Ga", "Ni"], "Ga362Ni97"),
    ("NbCo2 / Nb1.1Co1.9 ht", ["Co", "Nb"], "Co2Nb"),
]

#: The composition-only path: an entry with no MPDS label, at a digitized fraction that has no
#: small integer ratio. Written out as atomic fractions instead.
FRACTIONAL_EXAMPLE = (
    Composition({"Au": 1 - 0.309569, "Ga": 0.309569}),
    ["Au", "Ga"],
    "Au0.690Ga0.310",
)


class TestTheWorkedExamples:
    @pytest.mark.parametrize("source,components,expected", WORKED_EXAMPLES)
    def test_each_label_standardizes(self, source, components, expected):
        assert standard_formula(source, components) == expected

    def test_a_composition_with_no_label_writes_atomic_fractions(self):
        source, components, expected = FRACTIONAL_EXAMPLE
        assert standard_formula(source, components) == expected

    def test_the_component_order_decides_the_element_order(self):
        """The control for every case above: the SAME label reads both ways round.

        Nothing in ``Cu33Al17`` says which element comes first -- only the system does.
        """
        assert standard_formula("Cu33Al17 rt", ["Al", "Cu"]) == "Al17Cu33"
        assert standard_formula("Cu33Al17 rt", ["Cu", "Al"]) == "Cu33Al17"

    def test_electronegativity_order_is_not_what_comes_out(self):
        """``reduced_formula`` is the form being replaced; it must not survive anywhere."""
        assert Composition("Au7Ga2").reduced_formula == "Ga2Au7"
        assert standard_formula(Composition("Au7Ga2"), ["Au", "Ga"]) == "Au7Ga2"


class TestTheOutputAlphabet:
    @pytest.mark.parametrize(
        "source,components", [(s, c) for s, c, _ in WORKED_EXAMPLES] + [FRACTIONAL_EXAMPLE[:2]]
    )
    def test_no_label_carries_a_suffix_or_a_separator(self, source, components):
        label = standard_formula(source, components)
        assert re.fullmatch(r"[A-Za-z0-9.]+", label), label

    @pytest.mark.parametrize("suffix", ["rt", "ht", "lt", "stab", "m-orth"])
    def test_every_polymorph_suffix_is_dropped(self, suffix):
        assert standard_formula(f"AlCu {suffix}", ["Al", "Cu"]) == "AlCu"


class TestTheThreeDigitCap:
    def test_999_stays_an_integer_formula(self):
        assert standard_formula("Ga999Ni1", ["Ga", "Ni"]) == "Ga999Ni"

    def test_1000_drops_to_atomic_fractions(self):
        assert standard_formula("Ga1000Ni1", ["Ga", "Ni"]) == "Ga0.999Ni0.000999"

    def test_the_fractions_carry_three_significant_figures(self):
        source, components, _ = FRACTIONAL_EXAMPLE
        label = standard_formula(source, components)
        digits = [d.lstrip("0.") for d in re.findall(r"\d+\.\d+", label)]
        assert digits == ["690", "310"]

    def test_a_label_just_inside_the_cap_is_not_prettified(self):
        """``Ga362Ni97`` is inside the rule; no second heuristic may shorten it."""
        assert standard_formula("Ni0.97Ga3.62", ["Ga", "Ni"]) == "Ga362Ni97"


class TestTheOverCapReReduction:
    """Past the integer cap the fractions get one more chance to be a small whole-number ratio.

    ``Cd82.25Ni14.49`` reduces exactly to 1175:207, past the cap, but its atomic fractions are
    17:3, which is a formula. The re-reduction is taken only when it stays small (not
    ``Y177Zn823``). Both branches carry a real corpus label.
    """

    def test_a_small_ratio_is_re_reduced(self):
        assert standard_formula("Cd82.25Ni14.49", ["Cd", "Ni"]) == "Cd17Ni3"

    def test_a_large_ratio_keeps_its_fractions(self):
        assert standard_formula("Zn58.7Y12.65", ["Y", "Zn"]) == "Y0.177Zn0.823"

    def test_another_large_ratio_keeps_its_fractions(self):
        assert standard_formula("YGe1.567", ["Ge", "Y"]) == "Ge0.610Y0.390"

    def test_the_cap_is_what_decides(self, monkeypatch):
        monkeypatch.setattr(style, "_MAX_REREDUCED_INTEGER", 3)
        assert "." in standard_formula("Cd82.25Ni14.49", ["Cd", "Ni"])
        monkeypatch.setattr(style, "_MAX_REREDUCED_INTEGER", 17)
        assert standard_formula("Cd82.25Ni14.49", ["Cd", "Ni"]) == "Cd17Ni3"

    def test_nothing_under_the_integer_cap_is_touched(self):
        # the re-reduction is reachable only from the over-cap branch
        assert standard_formula("Ni0.97Ga3.62", ["Ga", "Ni"]) == "Ga362Ni97"
        assert standard_formula("Ga999Ni1", ["Ga", "Ni"]) == "Ga999Ni"


class TestAMiss:
    def test_an_unparseable_label_is_rejected(self):
        with pytest.raises(ValueError):
            standard_formula("SomeMpdsLabel", ["Cu", "Mg"])

    def test_a_label_naming_a_foreign_element_is_rejected(self):
        with pytest.raises(ValueError):
            standard_formula("Mg2Cu", ["Al", "Cu"])

    def test_a_label_that_is_only_a_suffix_is_rejected(self):
        with pytest.raises(ValueError):
            standard_formula("rt", ["Cu", "Mg"])

    def test_a_hull_vertex_falls_back_to_its_composition(self):
        """A miss on the label must not lose the vertex its label; the composition stands in."""

        class _Entry:
            composition = Composition({"Cu": 2, "Mg": 1})
            data = {"imputed": True, "label": "SomeMpdsLabel"}

        assert standardized_vertex_label(_Entry(), ["Cu", "Mg"]) == "Cu2Mg"

    def test_a_hull_vertex_prefers_a_label_that_parses(self):
        """The control: with a usable label the composition is NOT what gets written.

        An imputed entry sits at the digitized fractional composition, which has no small
        integer ratio -- the MPDS label is the stoichiometry the phase actually has.
        """

        class _Entry:
            composition = Composition({"Cu": 0.34, "Mg": 0.66})
            data = {"imputed": True, "label": "Mg2Cu"}

        assert standardized_vertex_label(_Entry(), ["Cu", "Mg"]) == "CuMg2"

    def test_a_dft_vertex_ignores_any_label(self):
        class _Entry:
            composition = Composition({"Cu": 2, "Mg": 1})
            entry_id = "mp-9998"
            data = {"label": "Mg2Cu"}

        assert standardized_vertex_label(_Entry(), ["Cu", "Mg"]) == "Cu2Mg"


class TestTheTxFieldLabels:
    """``abbreviate_phase_name`` gains the compound path, and keeps everything else."""

    def test_a_compound_standardizes_when_the_components_are_known(self):
        from gliquid.plotting.style import abbreviate_phase_name

        assert (
            abbreviate_phase_name("Cu33Al17 rt", ["Cu33Al17 rt", "L"], components=["Al", "Cu"])
            == "Al<sub>17</sub>Cu<sub>33</sub>"
        )

    def test_an_elemental_polymorph_is_untouched_by_the_compound_path(self):
        from gliquid.plotting.style import abbreviate_phase_name

        names = ["alpha Mn (bcc)", "beta Mn (bcc)", "L"]
        assert abbreviate_phase_name("alpha Mn (bcc)", names, components=["Mn", "Sn"]) == (
            "α-Mn (bcc)"
        )

    def test_a_name_that_will_not_parse_keeps_todays_label(self):
        from gliquid.plotting.style import abbreviate_phase_name

        assert (
            abbreviate_phase_name("SomeMpdsLabel", ["SomeMpdsLabel", "L"], components=["Cu", "Mg"])
            == "SomeMpdsLabel"
        )
