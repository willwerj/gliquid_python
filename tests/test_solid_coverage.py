"""Unit tests for solid_coverage.assess_solid_coverage on synthetic phase diagrams.

Synthetic JSONs keep these offline and make each support rule testable in isolation --
no cache, no DFT hull, no BinaryLiquid. See test_solid_coverage_gate.py for the
end-to-end wiring through fit_parameters.
"""

import math
from functools import partial

import pytest

import gliquid.config as config
from gliquid import mpds, solid_coverage

from ._mpds_builders import eutectic_inv, melting_inv
from ._mpds_builders import lc_phase as _lc_phase

SPAN = [0.0, 1.0]


def ss_phase(name, lo, hi, spacegroup=None, pearson=None):
    """A digitized solid-solution field spanning [lo, hi]."""
    return {
        "type": "ss",
        "name": name,
        "comp": hi,
        "cbounds": [[lo, 1000.0], [hi, 1000.0]],
        "tbounds": [[lo, 500.0], [hi, 1500.0]],
        "structure": {"prototype": name, "spacegroup": spacegroup, "pearson": pearson},
    }


def field_at(name, lo, hi, comp, spacegroup=None, pearson=None):
    """A digitized field spanning ``[lo, hi]`` whose recorded ``comp`` is stated explicitly.

    ``ss_phase`` records ``comp`` at the field's upper edge. MPDS records it at the field's
    maximum-temperature point, which is the pure end for a terminal solution and the lower
    edge for a peritectic ordered field -- Co-Zn's ``Zn11Co2`` spans 0.690-0.853 and is
    recorded at 0.690.
    """
    phase = ss_phase(name, lo, hi, spacegroup=spacegroup, pearson=pearson)
    phase["comp"] = comp
    return phase


lc_phase = partial(_lc_phase, t_lo=500.0, t_hi=1500.0)


def comp_phase(name, comp, spacegroup=None, pearson=None):
    """A pure component's own phase, drawn as a boundary line rather than an area."""
    return {
        "type": "comp",
        "name": name,
        "comp": comp,
        "tbounds": [[comp, 500.0], [comp, 1500.0]],
        "structure": {"prototype": name, "spacegroup": spacegroup, "pearson": pearson},
    }


def assess(phases, invariants=(), dft_comps=(), ss_models=(), **kw):
    """Assess with the field-ceiling narrowing and the extension cap OFF unless a case asks,
    so the bounding rules, the borders-liquid predicate and DFT matching alone decide the
    expected intervals. Each is exercised in its own class."""
    knobs = {name: kw.pop(name) for name in list(kw) if name in config._COVERAGE_KNOBS}
    thresholds = {
        "field_ceiling_tol": math.inf,
        "ss_extension_cap": math.inf,
        **knobs,
        **(kw.pop("thresholds", None) or {}),
    }
    return solid_coverage.assess_solid_coverage(
        list(phases),
        list(invariants),
        SPAN,
        list(dft_comps),
        ss_models,
        thresholds=thresholds,
        **kw,
    )


class TestSolidSolutionFields:
    def test_wide_field_without_a_model_is_unsupported(self):
        """The Lu-Nd class: a field spelled '(Lu)' that measurably spans the whole axis."""
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)])
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "no_ss_model:HCP"

    def test_wide_field_with_a_matching_model_is_supported(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)], ss_models=("HCP",))
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason == "ss_model:HCP"

    def test_model_for_a_different_structure_does_not_help(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=229)], ss_models=("HCP",))
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "no_ss_model:BCC"

    def test_narrow_field_is_not_scored(self):
        rep = assess([ss_phase("(A)", 0.0, 0.05, spacegroup=194)])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason == "narrow_field"

    def test_partial_fields_sum_over_their_union(self):
        """Two terminal fields, each recorded at its pure end and ended by its own eutectic
        (0.30, 0.52), so each primary field is its cbounds and the union arithmetic is under
        test."""
        rep = assess(
            [
                field_at("(A)", 0.0, 0.30, comp=0.0, spacegroup=194),
                field_at("(B)", 0.52, 1.00, comp=1.0, spacegroup=229),
            ],
            invariants=[{"comp": 0.30}, {"comp": 0.52}],
        )
        assert rep.unsupported_fraction == pytest.approx(0.78)
        assert rep.unsupported_ranges == [[0.0, 0.30], [0.52, 1.0]]

    def test_overlapping_fields_are_not_double_counted(self):
        rep = assess(
            [
                ss_phase("(A)", 0.0, 0.60, spacegroup=194),
                ss_phase("(B)", 0.40, 1.00, spacegroup=229),
            ]
        )
        assert rep.unsupported_fraction == pytest.approx(1.0)


class TestUnresolvedStructureFallback:
    """MPDS leaves '(A, B)' shapes structurally unresolved."""

    def test_unresolved_field_is_supported_when_any_model_is_loaded(self):
        rep = assess([ss_phase("(A, B)", 0.0, 1.0)], ss_models=("HCP",))
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason == "unknown_structure_ss_loaded"

    def test_unresolved_field_is_unsupported_with_no_models(self):
        rep = assess([ss_phase("(A, B)", 0.0, 1.0)])
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "unknown_structure_no_ss"


class TestDftRescueOfSolutionFields:
    """The width-capped DFT rescue, which reaches only COMPONENT solution fields.

    A field labelled by formula is an ordered intermetallic and takes the compound path
    (see TestOrderedFieldsWithHomogeneityRanges), so what is left here is a '(X)' or
    '(A, B)' field whose resolved spacegroup is not a solid-solution structure.
    """

    def test_solution_field_is_rescued_by_a_nearby_dft_phase(self):
        # width 0.15: wider than ss_narrow_tol (0.10), within ss_rescue_max_width (0.25)
        rep = assess([ss_phase("(A, B)", 0.45, 0.60, spacegroup=71)], dft_comps=[0.55])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason.startswith("dft_phase@")

    def test_solution_field_without_a_nearby_dft_phase_stays_unsupported(self):
        """The rescue verdict is unchanged; the interval it masks is the primary
        field between the flanking invariants, not the 0.20-wide cbounds."""
        rep = assess(
            [ss_phase("(A, B)", 0.30, 0.50, spacegroup=71)],
            invariants=[{"comp": 0.25}, {"comp": 0.60}],
            dft_comps=[0.90],
        )
        assert rep.phases[0].supported is False
        assert rep.phases[0].reason == "no_ss_model:sg71 field:invariant/invariant"
        assert rep.unsupported_ranges == [[0.25, 0.60]]
        assert rep.unsupported_fraction == pytest.approx(0.35)

    def test_wide_field_is_not_rescued_however_close_the_dft_phase(self):
        """Uncapped, this would score 0.00 and declare complete solid solutions with no
        solid-solution models (Ag-Au, Ta-W, Se-Te, Bi-Sb, As-Sb) fittable -- precisely the
        failure this gate exists to catch."""
        rep = assess([ss_phase("(A,B)", 0.0, 1.0, spacegroup=71)], dft_comps=[0.5])
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "too_wide_to_rescue:sg71"

    def test_rescue_width_cap_is_tunable(self):
        """Uncapped the field is supported and masks nothing; capped it masks its
        primary field ``[0.10, 0.75]``, which is wider than its 0.40-wide cbounds."""
        phases = [ss_phase("(A, B)", 0.20, 0.60, spacegroup=71)]
        invariants = [{"comp": 0.10}, {"comp": 0.75}]
        capped = assess(phases, invariants=invariants, dft_comps=[0.4])
        assert capped.phases[0].supported is False
        assert capped.unsupported_ranges == [[0.10, 0.75]]
        assert capped.unsupported_fraction == pytest.approx(0.65)
        rescued = assess(phases, invariants=invariants, dft_comps=[0.4], ss_rescue_max_width=0.5)
        assert rescued.phases[0].supported is True
        assert rescued.unsupported_fraction == pytest.approx(0.0)


class TestOrderedFieldsWithHomogeneityRanges:
    """A field MPDS labels by formula is an ordered intermetallic, not a solution: it is
    checked against the DFT hull whatever its width or spacegroup."""

    def test_ordered_field_with_a_dft_counterpart_is_supported(self):
        rep = assess([ss_phase("A3B5", 0.45, 0.60, spacegroup=71)], dft_comps=[0.55])
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.550"
        assert rep.unsupported_ranges == []
        assert (rep.n_compounds, rep.n_missing_compounds) == (1, 0)

    def test_ordered_field_without_a_dft_counterpart_masks_its_field(self):
        rep = assess(
            [ss_phase("A3B5", 0.45, 0.60, spacegroup=71)],
            invariants=[{"comp": 0.30}, {"comp": 0.80}],
            dft_comps=[0.05],
        )
        assert rep.phases[0].supported is False
        assert rep.phases[0].reason == "ordered_field:no_dft_phase:invariant/invariant"
        assert rep.unsupported_ranges == [[0.30, 0.80]]
        assert (rep.n_compounds, rep.n_missing_compounds) == (1, 1)

    def test_an_unsupported_ordered_field_masks_its_drawn_width_too(self):
        """The In-Mg shape: a field drawn wider than the invariants that flank its comp.

        The homogeneity range is measured and the invariant bound is inferred, so the mask
        is their UNION -- the drawn field cannot be narrowed away by a nearer invariant.
        """
        rep = assess(
            [field_at("A3B5", 0.10, 0.80, comp=0.78, spacegroup=71)],
            invariants=[{"comp": 0.40}, {"comp": 0.90}],
        )
        assert rep.phases[0].supported is False
        assert rep.phases[0].interval == (0.10, 0.90)
        assert rep.unsupported_ranges == [[0.10, 0.90]]

    def test_the_invariant_field_still_wins_where_it_is_the_wider_one(self):
        """POSITIVE CONTROL: the union, not a replacement by the drawn width."""
        rep = assess(
            [field_at("A3B5", 0.45, 0.60, comp=0.50, spacegroup=71)],
            invariants=[{"comp": 0.30}, {"comp": 0.80}],
        )
        assert rep.phases[0].interval == (0.30, 0.80)
        assert rep.unsupported_ranges == [[0.30, 0.80]]

    def test_a_narrow_ordered_field_is_no_longer_waved_through(self):
        """The Ir-Sm case: a 0.015-wide ordered field is checked against the hull, which
        does not hold it, rather than passing as `narrow_field`."""
        rep = assess(
            [ss_phase("A2B", 0.3185, 0.3334, spacegroup=None)],
            invariants=[
                melting_inv("cmp", 0.3334, "A2B", 0.3334),
                {"comp": 0.2003},
                {"comp": 0.6350},
            ],
            dft_comps=[0.625, 0.75],
        )
        assert rep.phases[0].supported is False
        assert rep.phases[0].reason == "ordered_field:no_dft_phase:invariant/invariant"

    def test_a_congruent_ordered_field_is_bounded_by_its_flanking_invariants(self):
        """Not by its own 0.015-wide homogeneity range: the masked field is the primary
        crystallization field, [0.2003, 0.6350], which is 29x the cbounds width."""
        rep = assess(
            [ss_phase("A2B", 0.3185, 0.3334, spacegroup=None)],
            invariants=[
                melting_inv("cmp", 0.3334, "A2B", 0.3334),
                {"comp": 0.2003},
                {"comp": 0.6350},
            ],
            dft_comps=[],
        )
        assert rep.unsupported_ranges == [[0.2003, 0.6350]]

    def test_a_peritectic_ordered_field_is_relocated_like_a_line_compound(self):
        """The Hf-Mn Hf2Mn case: stoichiometry 0.362, peritectic liquid 0.2412, so the
        field runs from the peritectic AWAY from the stoichiometry."""
        rep = assess(
            [ss_phase("A2B", 0.3327, 0.3624, spacegroup=227)],
            invariants=[
                melting_inv("per", 0.2412, "A2B", 0.3624),
                {"comp": 0.2202},
                {"comp": 0.6716},
            ],
            dft_comps=[],
        )
        phase = rep.phases[0]
        assert phase.comp == pytest.approx(0.3624)
        assert phase.reason == "ordered_field:no_dft_phase:invariant/peritectic"
        assert rep.unsupported_ranges == [[0.2202, 0.2412]]

    def test_a_solid_solution_spacegroup_does_not_reject_a_dft_covered_ordered_field(self):
        """The Hf-Mn HfMn2 case: sg 194 with no HCP model loaded is supported by the hull
        compound at 0.667, not rejected as `no_ss_model:HCP`."""
        rep = assess([ss_phase("AB2", 0.620, 0.753, spacegroup=194)], dft_comps=[0.667])
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.667"

    def test_the_rescue_width_cap_does_not_reach_an_ordered_field(self):
        """Mn-Zr's ZrMn2 (0.209-0.400) and a 0.10-0.90 field are both matched to the hull."""
        rep = assess([ss_phase("AB2", 0.209, 0.400, spacegroup=194)], dft_comps=[0.333])
        assert rep.phases[0].supported is True
        wide = assess([ss_phase("AB2", 0.10, 0.90, spacegroup=194)], dft_comps=[0.90])
        assert wide.phases[0].supported is True
        assert wide.phases[0].reason == "ordered_field:dft_phase@0.900"

    def test_a_terminal_field_with_a_narrow_range_is_unchanged(self):
        rep = assess([ss_phase("(A) rt", 0.0, 0.05, spacegroup=194)], dft_comps=[])
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "narrow_field"
        assert (rep.n_compounds, rep.n_missing_compounds) == (0, 0)

    def test_a_full_range_solution_field_is_unchanged(self):
        """Cr-Ti's '(Ti, Cr)': '('-prefixed, structurally unresolved, and not a compound."""
        phases = [ss_phase("(Ti, Cr)", 0.0, 1.0)]
        rep = assess(phases, elements=["Cr", "Ti"])
        assert rep.phases[0].reason == "unknown_structure_no_ss"
        assert rep.unsupported_ranges == [[0.0, 1.0]]
        assert (rep.n_compounds, rep.n_missing_compounds) == (0, 0)
        loaded = assess(phases, ss_models=("HCP",), elements=["Cr", "Ti"])
        assert loaded.phases[0].reason == "unknown_structure_ss_loaded"

    def test_a_suffixed_full_range_field_is_still_a_solution(self):
        rep = assess([ss_phase("(Sr, Ca) ht", 0.0, 1.0, spacegroup=194)], elements=["Ca", "Sr"])
        assert rep.phases[0].reason == "no_ss_model:HCP"
        assert rep.n_compounds == 0

    def test_named_elements_decide_when_they_are_supplied(self):
        """With the symbols in hand a '('-label is checked against them; without them the
        leading parenthesis decides on its own."""
        phases = [ss_phase("(Ti, Cr)", 0.30, 0.50, spacegroup=71)]
        assert assess(phases, elements=["Cr", "Ti"], dft_comps=[]).n_compounds == 0
        assert assess(phases, elements=["Fe", "Ni"], dft_comps=[]).n_compounds == 1
        assert assess(phases, dft_comps=[]).n_compounds == 0


class TestMissingCompounds:
    def test_compound_with_a_dft_counterpart_is_supported(self):
        rep = assess([lc_phase("AB", 0.5)], dft_comps=[0.505])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.n_missing_compounds == 0 and rep.n_compounds == 1

    def test_missing_compound_masks_only_its_primary_crystallization_field(self):
        """Between the flanking invariants -- not out to the next supported anchor. Past a
        eutectic the conjugate solid is a different phase whose energy may be known, so that
        stretch of liquidus still constrains the fit. This is Mn-Y at 0.53, not 0.92."""
        invariants = [{"comp": 0.12}, {"comp": 0.285}, {"comp": 0.647}]
        rep = assess([lc_phase("AB2", 0.333)], invariants=invariants, dft_comps=[0.077])
        assert rep.unsupported_ranges == [[0.285, 0.647]]
        assert rep.unsupported_fraction == pytest.approx(0.362)

    def test_missing_compound_falls_back_to_supported_anchors_without_invariants(self):
        rep = assess([lc_phase("AB2", 0.333)], dft_comps=[0.077])
        assert rep.unsupported_ranges == [[0.077, 1.0]]

    def test_endpoints_always_anchor(self):
        rep = assess([lc_phase("AB", 0.5)])
        assert rep.unsupported_ranges == [[0.0, 1.0]]
        assert rep.n_missing_compounds == 1


class TestFlankingBoundFallback:
    """Each side of a missing compound's field takes the first rule that yields a bound:
    the nearest invariant, else the nearest local minimum of the liquidus (the eutectic the
    invariant finder did not report), else the nearest supported anchor. The rule that fired
    is recorded in ``reason`` so the audit trail names it."""

    # Minima at 0.2 and 0.6; the compound at 0.4 sits on the rise between them.
    LIQ = [
        [0.0, 1200.0],
        [0.1, 1100.0],
        [0.2, 1000.0],
        [0.3, 1100.0],
        [0.4, 1300.0],
        [0.5, 1200.0],
        [0.6, 1100.0],
        [0.7, 1200.0],
        [0.8, 1300.0],
        [1.0, 1400.0],
    ]
    # Strictly rising, so it has no local minimum at all -- the Ge-W shape.
    MONOTONE_LIQ = [[0.0, 1200.0], [0.4, 2000.0], [1.0, 3700.0]]

    def test_invariants_on_both_sides_win_over_the_liquidus(self):
        """The invariant-bounded path is untouched: a liquidus minimum never overrides an
        invariant, so passing the curve changes nothing here."""
        invariants = [{"comp": 0.25}, {"comp": 0.55}]
        rep = assess([lc_phase("A3B2", 0.4)], invariants=invariants, liquidus=self.LIQ)
        assert rep.unsupported_ranges == [[0.25, 0.55]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/invariant"
        without = assess([lc_phase("A3B2", 0.4)], invariants=invariants)
        assert without.unsupported_ranges == rep.unsupported_ranges

    def test_one_flanking_invariant_leaves_the_other_side_to_the_minimum(self):
        rep = assess([lc_phase("A3B2", 0.4)], invariants=[{"comp": 0.25}], liquidus=self.LIQ)
        assert rep.unsupported_ranges == [[0.25, 0.6]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/minimum"

    def test_liquidus_minima_bound_the_field_when_no_invariant_does(self):
        """The eutectics the invariant finder missed still show on the digitized curve."""
        rep = assess([lc_phase("A3B2", 0.4)], liquidus=self.LIQ)
        assert rep.unsupported_ranges == [[0.2, 0.6]]
        assert rep.phases[0].reason == "no_dft_phase:minimum/minimum"
        assert rep.unsupported_fraction == pytest.approx(0.4)

    def test_a_monotone_liquidus_falls_back_to_the_endpoints(self):
        """The Ge-W shape: one missing compound, no invariant and no minimum on either
        side, so the anchors bound it and the whole span is masked."""
        rep = assess([lc_phase("A3B2", 0.4)], liquidus=self.MONOTONE_LIQ)
        assert rep.unsupported_ranges == [[0.0, 1.0]]
        assert rep.phases[0].reason == "no_dft_phase:endpoint/endpoint"
        assert rep.unsupported_fraction == pytest.approx(1.0)


class TestControllingInvariant:
    """Which invariant record identifies the compound it melts."""

    def test_a_peritectic_naming_the_compound_is_returned(self):
        inv = melting_inv("per", 0.234, "Be2Ag", 0.730)
        assert mpds.controlling_invariant(lc_phase("Be2Ag", 0.730), [inv]) is inv

    def test_a_congruent_point_naming_the_compound_is_returned(self):
        inv = melting_inv("cmp", 0.4, "AB", 0.4)
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [inv]) is inv

    def test_a_eutectic_naming_the_compound_is_not_controlling(self):
        """A eutectic lists the two phases it is BETWEEN, not one it melts."""
        eut = eutectic_inv(0.107, phases=("A", "Be2Ag"), phase_comps=(0.0, 0.730))
        assert mpds.controlling_invariant(lc_phase("Be2Ag", 0.730), [eut]) is None

    def test_a_miscibility_gap_naming_the_compound_is_not_controlling(self):
        mig = {"type": "mig", "comp": 0.5, "phases": ["AB"], "phase_comps": [0.4]}
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [mig]) is None

    def test_a_record_for_a_different_compound_is_not_returned(self):
        inv = melting_inv("per", 0.3, "A2B", 0.667)
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [inv]) is None

    def test_a_same_named_shape_at_another_composition_is_not_returned(self):
        """The Al-Dy case: DyAl2 labelled at both 0.333 and 0.666."""
        inv = melting_inv("per", 0.2, "DyAl2", 0.666)
        assert mpds.controlling_invariant(lc_phase("DyAl2", 0.333), [inv]) is None

    def test_the_nearest_record_wins_among_several(self):
        far = melting_inv("per", 0.2, "AB", 0.415)
        near = melting_inv("per", 0.9, "AB", 0.401)
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [far, near]) is near

    def test_a_thermal_form_binds_to_the_record_naming_the_bare_compound(self):
        """The Li-Zn shape: an 'ss' shape keeps its polymorph suffix, the record does not."""
        inv = melting_inv("per", 0.7215, "Li2Zn5", 0.6996)
        assert mpds.controlling_invariant(lc_phase("Li2Zn5 ht", 0.7093), [inv]) is inv

    def test_a_bare_shape_binds_to_a_record_naming_a_thermal_form(self):
        inv = melting_inv("per", 0.7215, "Li2Zn5 rt", 0.6996)
        assert mpds.controlling_invariant(lc_phase("Li2Zn5", 0.7093), [inv]) is inv

    def test_stripping_the_suffix_does_not_reach_a_different_compound(self):
        """POSITIVE CONTROL: the composition check still separates the two DyAl2 shapes."""
        inv = melting_inv("per", 0.2, "DyAl2 rt", 0.666)
        assert mpds.controlling_invariant(lc_phase("DyAl2 ht", 0.333), [inv]) is None

    def test_no_invariants_at_all(self):
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), []) is None
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), None) is None


class TestTheEutecticFallback:
    """``allow_eutectic`` -- an anchor wherever the phase borders the liquid, not just where
    something melts it.

    Field bounding wants the record that MELTS a compound and takes the default. Imputation
    wants an anchor anywhere the phase meets the liquid, because a phase that only ever
    appears at a eutectic still has a tangent to read the energy off. A eutectic does not name
    the phase it flanks, so it is matched on stoichiometry alone.
    """

    FLANKING = eutectic_inv(0.107, phases=("A", "AB"), phase_comps=(0.0, 0.400))

    def test_it_is_returned_only_when_asked_for(self):
        phase = lc_phase("AB", 0.4)
        assert mpds.controlling_invariant(phase, [self.FLANKING]) is None
        assert (
            mpds.controlling_invariant(phase, [self.FLANKING], allow_eutectic=True) is self.FLANKING
        )

    def test_a_melting_record_still_wins_over_it(self):
        melt = melting_inv("per", 0.234, "AB", 0.400)
        got = mpds.controlling_invariant(
            lc_phase("AB", 0.4), [self.FLANKING, melt], allow_eutectic=True
        )
        assert got is melt

    def test_the_nearest_eutectic_by_LIQUID_composition_wins(self):
        """Resolved on the liquid composition, which is what the inversion is anchored on."""
        near = eutectic_inv(0.35, phases=("A", "AB"), phase_comps=(0.0, 0.400))
        got = mpds.controlling_invariant(
            lc_phase("AB", 0.4), [self.FLANKING, near], allow_eutectic=True
        )
        assert got is near

    def test_a_eutectic_the_phase_does_not_flank_is_not_returned(self):
        other = eutectic_inv(0.9, phases=("A2B", "B"), phase_comps=(0.667, 1.0))
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [other], allow_eutectic=True) is None

    def test_the_composition_tolerance_is_honoured(self):
        near_miss = eutectic_inv(0.107, phases=("A", "AB"), phase_comps=(0.0, 0.430))
        phase = lc_phase("AB", 0.4)
        assert mpds.controlling_invariant(phase, [near_miss], allow_eutectic=True) is None
        assert (
            mpds.controlling_invariant(phase, [near_miss], allow_eutectic=True, comp_tol=0.05)
            is near_miss
        )

    def test_a_eutectic_with_no_recorded_stoichiometry_is_not_returned(self):
        bare = eutectic_inv(0.107)
        assert mpds.controlling_invariant(lc_phase("AB", 0.4), [bare], allow_eutectic=True) is None


class TestPeritecticFieldBounding:
    """A missing compound's field is located from its own melting invariant.

    A peritectic melter's stoichiometry lies OUTSIDE its primary crystallization field, so
    keying the flanking search on the stoichiometry returns the complement of the field --
    usually a wide interval running out to an endpoint. The field runs from the peritectic
    liquid composition AWAY from the stoichiometry instead.
    """

    # A single minimum at 0.1, so a side with no invariant lands on the second rule.
    LIQ_MIN_AT_01 = [[0.0, 1200.0], [0.1, 1000.0], [0.5, 1400.0], [1.0, 1600.0]]
    # Strictly rising: no minimum anywhere, so that side falls through to an anchor.
    MONOTONE_LIQ = [[0.0, 1200.0], [0.4, 2000.0], [1.0, 3700.0]]

    def test_stoichiometry_above_the_peritectic_liquid_masks_the_left_side(self):
        """Ag-Be: `L + (Be) -> Be2Ag` at x=0.234 with Be2Ag at x=0.730, eutectic at 0.107.
        The field is [0.107, 0.234]; everything Be-richer is primary (Be)."""
        invariants = [
            eutectic_inv(0.107, phases=("Ag", "Be2Ag"), phase_comps=(0.0, 0.730)),
            melting_inv("per", 0.234, "Be2Ag", 0.730),
        ]
        rep = assess([lc_phase("Be2Ag", 0.730)], invariants=invariants)
        assert rep.unsupported_ranges == [[0.107, 0.234]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/peritectic"
        assert rep.largest_supported_segment == pytest.approx(1.0 - 0.234)

    def test_stoichiometry_below_the_peritectic_liquid_masks_the_right_side(self):
        """Ni-Sr: SrNi at x=0.500 melts peritectically at x=0.774, eutectic at 0.885."""
        invariants = [
            melting_inv("per", 0.774, "SrNi", 0.500),
            eutectic_inv(0.885, phases=("SrNi", "Sr"), phase_comps=(0.500, 1.0)),
        ]
        rep = assess([lc_phase("SrNi", 0.500)], invariants=invariants)
        assert rep.unsupported_ranges == [[0.774, 0.885]]
        assert rep.phases[0].reason == "no_dft_phase:peritectic/invariant"
        assert rep.largest_supported_segment == pytest.approx(0.774)

    def test_a_congruent_melter_is_bounded_from_its_stoichiometry(self):
        """The safety property: a congruent record must leave the field exactly where the
        stoichiometry-keyed search put it, phase record included."""
        flanking = [
            eutectic_inv(0.2, phases=("A", "AB"), phase_comps=(0.0, 0.4)),
            eutectic_inv(0.6, phases=("AB", "B"), phase_comps=(0.4, 1.0)),
        ]
        congruent = assess(
            [lc_phase("AB", 0.4)],
            invariants=[*flanking, melting_inv("cmp", 0.4, "AB", 0.4)],
            liquidus=self.LIQ_MIN_AT_01,
        )
        unclassified = assess(
            [lc_phase("AB", 0.4)],
            invariants=[{"comp": 0.2}, {"comp": 0.4}, {"comp": 0.6}],
            liquidus=self.LIQ_MIN_AT_01,
        )
        assert congruent.phases[0] == unclassified.phases[0]
        assert congruent.unsupported_ranges == unclassified.unsupported_ranges == [[0.2, 0.6]]
        assert congruent.unsupported_fraction == unclassified.unsupported_fraction
        assert congruent.phases[0].reason == "no_dft_phase:invariant/invariant"

    def test_a_peritectic_at_its_own_stoichiometry_is_bounded_the_same_way(self):
        """A degenerate peritectic has no side to move to, so it takes the congruent path."""
        invariants = [
            eutectic_inv(0.2),
            melting_inv("per", 0.4, "AB", 0.4),
            eutectic_inv(0.6),
        ]
        rep = assess([lc_phase("AB", 0.4)], invariants=invariants)
        assert rep.unsupported_ranges == [[0.2, 0.6]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/invariant"

    def test_no_further_invariant_on_the_field_side_falls_to_the_minimum(self):
        invariants = [melting_inv("per", 0.3, "AB2", 0.7)]
        rep = assess([lc_phase("AB2", 0.7)], invariants=invariants, liquidus=self.LIQ_MIN_AT_01)
        assert rep.unsupported_ranges == [[0.1, 0.3]]
        assert rep.phases[0].reason == "no_dft_phase:minimum/peritectic"

    def test_no_minimum_either_falls_to_the_anchor(self):
        invariants = [melting_inv("per", 0.3, "AB2", 0.7)]
        rep = assess([lc_phase("AB2", 0.7)], invariants=invariants, liquidus=self.MONOTONE_LIQ)
        assert rep.unsupported_ranges == [[0.0, 0.3]]
        assert rep.phases[0].reason == "no_dft_phase:endpoint/peritectic"

    def test_a_dft_compound_anchors_the_far_side_of_a_peritectic_field(self):
        # Both phases are placed above this liquidus: what is under test is where the
        # field ENDS, and a phase below the curve over its own field owns none of it.
        invariants = [melting_inv("per", 0.8, "AB2", 0.35)]
        rep = assess(
            [
                at_temp(lc_phase("AB2", 0.35), 500.0, 4000.0),
                at_temp(lc_phase("A2B", 0.9), 500.0, 4000.0),
            ],
            invariants=invariants,
            dft_comps=[0.9],
            liquidus=self.MONOTONE_LIQ,
        )
        assert rep.unsupported_ranges == [[0.8, 0.9]]
        assert rep.phases[0].reason == "no_dft_phase:peritectic/endpoint"

    def test_a_compound_with_no_controlling_invariant_is_unchanged(self):
        """Only records naming THIS compound relocate its field; a peritectic belonging to
        another phase leaves the stoichiometry-keyed search in place."""
        invariants = [
            eutectic_inv(0.2),
            melting_inv("per", 0.25, "A3B", 0.75),
            eutectic_inv(0.6),
        ]
        rep = assess([lc_phase("AB", 0.4)], invariants=invariants)
        assert rep.unsupported_ranges == [[0.25, 0.6]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/invariant"

    def test_a_stoichiometry_below_the_span_is_not_relocated(self):
        """Ru-Yb: the liquidus is digitized only from x=0.520 while YbRu sits at 0.500, so
        which side of its peritectic the field falls on is not something the data answers."""
        invariants = [
            melting_inv("per", 0.541, "YbRu", 0.500),
            eutectic_inv(0.700),
        ]
        rep = solid_coverage.assess_solid_coverage(
            [lc_phase("YbRu", 0.500)], invariants, [0.520, 1.0], [], ()
        )
        assert rep.unsupported_ranges == [[0.0, 0.541]]
        assert rep.phases[0].reason == "no_dft_phase:endpoint/invariant"

    def test_a_stoichiometry_above_the_span_is_not_relocated(self):
        invariants = [melting_inv("per", 0.30, "A2B", 0.80), eutectic_inv(0.15)]
        rep = solid_coverage.assess_solid_coverage(
            [lc_phase("A2B", 0.80)], invariants, [0.0, 0.70], [], ()
        )
        assert rep.unsupported_ranges == [[0.30, 1.0]]
        assert rep.phases[0].reason == "no_dft_phase:invariant/endpoint"

    def test_a_stoichiometry_on_the_span_edge_still_relocates(self):
        """The guard reads the span literally: an endpoint is inside it."""
        invariants = [melting_inv("per", 0.60, "AB", 0.40), eutectic_inv(0.85)]
        rep = solid_coverage.assess_solid_coverage(
            [lc_phase("AB", 0.40)], invariants, [0.40, 1.0], [], ()
        )
        assert rep.unsupported_ranges == [[0.60, 0.85]]
        assert rep.phases[0].reason == "no_dft_phase:peritectic/invariant"

    def test_the_guard_leaves_a_congruent_melter_alone(self):
        """A congruent compound never relocates, so being off-span changes nothing."""
        invariants = [
            eutectic_inv(0.55),
            melting_inv("cmp", 0.30, "AB", 0.30),
            eutectic_inv(0.15),
        ]
        off_span = solid_coverage.assess_solid_coverage(
            [lc_phase("AB", 0.30)], invariants, [0.50, 1.0], [], ()
        )
        assert off_span.phases[0].interval == (0.15, 0.55)
        assert off_span.phases[0].reason == "no_dft_phase:invariant/invariant"

    def test_a_supported_compound_is_never_relocated(self):
        invariants = [melting_inv("per", 0.1, "AB", 0.4)]
        rep = assess([lc_phase("AB", 0.4)], invariants=invariants, dft_comps=[0.41])
        assert rep.unsupported_ranges == []
        assert rep.phases[0].interval == (0.4, 0.4)


class TestSurvivingSegmentGeometry:
    """The statistics the verdict is taken on, derived from span and unsupported_ranges."""

    def test_nothing_masked_is_one_full_segment(self):
        rep = assess([lc_phase("AB", 0.5)], dft_comps=[0.5])
        assert rep.supported_segments == [(0.0, 1.0)]
        assert rep.largest_supported_segment == pytest.approx(1.0)
        assert rep.n_supported_segments == 1
        assert rep.supported_fraction == pytest.approx(1.0)

    def test_an_interior_hole_leaves_two_segments(self):
        rep = assess(
            [lc_phase("AB", 0.5)],
            invariants=[{"comp": 0.4}, {"comp": 0.7}],
            dft_comps=[],
        )
        assert rep.supported_segments == [(0.0, 0.4), (0.7, 1.0)]
        assert rep.largest_supported_segment == pytest.approx(0.4)
        assert rep.n_supported_segments == 2

    def test_a_fully_masked_liquidus_has_no_segments(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)])
        assert rep.supported_segments == []
        assert rep.largest_supported_segment == 0.0
        assert rep.n_supported_segments == 0

    def test_ranges_running_past_the_span_are_clipped_first(self):
        """A missing compound's masked field can overhang the digitized endpoints."""
        rep = solid_coverage.assess_solid_coverage(
            [lc_phase("AB", 0.5)], [], [0.2, 0.8], [], (), thresholds={"min_supported_total": 0.1}
        )
        assert rep.unsupported_ranges == [[0.0, 1.0]]
        assert rep.supported_segments == []
        assert rep.largest_supported_segment == 0.0

    def test_segments_are_reported_on_the_json_view(self):
        rep = assess([lc_phase("AB", 0.5)], invariants=[{"comp": 0.4}, {"comp": 0.7}], dft_comps=[])
        payload = rep.as_dict()
        assert payload["supported_segments"] == [[0.0, 0.4], [0.7, 1.0]]
        assert payload["largest_supported_segment"] == pytest.approx(0.4)
        assert payload["n_supported_segments"] == 2


class TestThresholdDecision:
    def test_a_fully_masked_liquidus_is_disqualifying(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)])
        insufficient, reason = rep.is_insufficient()
        assert insufficient is True and "constraining liquidus totals" in reason

    def test_a_wide_hole_flanked_by_a_usable_branch_is_admitted(self):
        """The Mn-Pr / Ru-Sn shape: most of the axis is masked, one branch survives whole."""
        rep = assess(
            [lc_phase("AB", 0.2)], invariants=[{"comp": 0.05}, {"comp": 0.65}], dft_comps=[]
        )
        assert rep.unsupported_fraction == pytest.approx(0.6)
        assert rep.largest_supported_segment == pytest.approx(0.35)
        assert rep.is_insufficient()[0] is False

    def test_support_arriving_in_pieces_still_counts(self):
        """Three narrow survivors constrain the parameters as much as one wide one.

        Each field is recorded at its interior maximum and flanked by the invariants that
        end it, so its primary field IS its cbounds.
        """
        rep = assess(
            [
                field_at("(A)", 0.10, 0.40, comp=0.25, spacegroup=194),
                field_at("(B)", 0.60, 0.90, comp=0.75, spacegroup=194),
            ],
            invariants=[{"comp": 0.10}, {"comp": 0.40}, {"comp": 0.60}, {"comp": 0.90}],
        )
        assert rep.unsupported_ranges == [[0.10, 0.40], [0.60, 0.90]]
        assert rep.unsupported_fraction == pytest.approx(0.6)
        assert rep.largest_supported_segment == pytest.approx(0.2)
        assert rep.supported_total == pytest.approx(0.4)
        assert rep.is_insufficient()[0] is False

    def test_too_little_support_in_total_is_disqualifying(self):
        """The same geometry scaled down: the pieces do not add up to the floor."""
        rep = assess(
            [
                field_at("(A)", 0.05, 0.45, comp=0.25, spacegroup=194),
                field_at("(B)", 0.55, 0.95, comp=0.75, spacegroup=194),
            ],
            invariants=[{"comp": 0.05}, {"comp": 0.45}, {"comp": 0.55}, {"comp": 0.95}],
        )
        assert rep.supported_total == pytest.approx(0.2)
        assert rep.is_insufficient()[0] is True

    def test_the_compound_count_no_longer_gates(self):
        """The Rb-Sn class: 2 of 3 compounds off the hull, admitted on its surviving
        [0, 0.6] branch."""
        rep = assess(
            [lc_phase("A12B17", 0.60), lc_phase("AB2", 0.667), lc_phase("AB4", 0.80)],
            invariants=[{"comp": 0.6}, {"comp": 0.75}, {"comp": 0.955}],
            dft_comps=[0.60],
        )
        assert (rep.n_compounds, rep.n_missing_compounds) == (3, 2)
        assert rep.missing_compound_fraction == pytest.approx(2 / 3)
        assert rep.unsupported_fraction == pytest.approx(0.355)
        assert rep.largest_supported_segment == pytest.approx(0.6)
        assert rep.is_insufficient()[0] is False

    def test_a_zero_floor_admits_everything(self):
        """The escape hatch that keeps a report assessed but ungated."""
        rep = assess(
            [ss_phase("(A)", 0.0, 1.0, spacegroup=194)],
            thresholds={"min_supported_total": 0.0},
        )
        assert rep.largest_supported_segment == 0.0
        assert rep.is_insufficient()[0] is False

    def test_an_override_names_one_key_and_keeps_the_rest(self):
        rep = assess([lc_phase("AB", 0.5)], thresholds={"min_supported_total": 0.9})
        assert rep.thresholds["dft_cover_tol"] == config.coverage_dft_cover_tol
        assert rep.thresholds["min_supported_total"] == 0.9

    def test_clean_system_passes(self):
        rep = assess([lc_phase("AB", 0.5)], dft_comps=[0.5])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.is_insufficient()[0] is False


class TestReportPlumbing:
    def test_span_clips_the_masked_measure(self):
        """A partial liquidus is scored over what was actually digitized.

        The terminal field is recorded at its pure end and ended by the eutectic at
        0.5, so its primary field is its cbounds and the clip is what this measures.
        """
        rep = solid_coverage.assess_solid_coverage(
            [field_at("(A)", 0.0, 0.5, comp=0.0, spacegroup=194)],
            [{"comp": 0.5}],
            [0.25, 0.75],
            [],
            (),
        )
        assert rep.unsupported_ranges == [[0.0, 0.5]]
        assert rep.unsupported_fraction == pytest.approx(0.5)  # [0.25,0.5] of [0.25,0.75]

    def test_empty_phase_list_is_fully_supported(self):
        rep = assess([])
        assert rep.unsupported_fraction == 0.0 and rep.phases == []
        assert rep.is_insufficient()[0] is False

    def test_as_dict_is_json_safe(self):
        import json

        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194), lc_phase("AB", 0.5)])
        json.dumps(rep.as_dict())  # must not raise

    def test_report_records_the_loaded_models(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)], ss_models=("HCP", "BCC"))
        assert rep.ss_models == ("HCP", "BCC")

    def test_non_canonical_pearson_is_flagged_but_not_disqualifying(self):
        """Spacegroup alone is not a structure (225 can be rock salt rather than FCC). Audit
        only -- gating on Pearson is too aggressive on real FCC solutions.

        Spelled '(A, B)': a formula-labelled field never reaches the spacegroup rules.
        """
        rep = assess(
            [ss_phase("(A, B)", 0.2, 0.5, spacegroup=225, pearson="cF8")], ss_models=("FCC",)
        )
        assert rep.phases[0].supported is True
        assert "pearson=cF8" in rep.phases[0].reason


class TestComponentPhases:
    """A pure component's phase is not a line compound, including a terminal phase MPDS
    digitizes as a `kind='compound'` boundary line at x = 0 or 1."""

    def test_component_phase_is_supported_and_is_not_a_compound(self):
        rep = assess([comp_phase("(B)", 1.0)])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "component_phase"
        assert (rep.n_compounds, rep.n_missing_compounds) == (0, 0)

    def test_component_phase_masks_nothing_even_with_no_dft_at_all(self):
        rep = assess([comp_phase("(A)", 0.0), comp_phase("(B)", 1.0)], dft_comps=[])
        assert rep.unsupported_ranges == []
        assert rep.is_insufficient()[0] is False

    def test_cr_ti_regression(self):
        """Cr-Ti: '(Ti) rt' is the right-hand frame edge, and TiCr2 is on the DFT hull.

        The Laves columns are narrow fields; the '(Ti, Cr)' field has no resolved structure
        and rides on the loaded models. Nothing is masked.
        """
        phases = [
            ss_phase("(Ti, Cr)", 0.0, 1.0),
            ss_phase("TiCr2 ht1", 0.339, 0.361, spacegroup=194, pearson="hP24"),
            ss_phase("TiCr2 rt", 0.351, 0.371, spacegroup=227, pearson="cF24"),
            comp_phase("(Ti)", 1.0, spacegroup=194, pearson="hP2"),
        ]
        rep = assess(phases, dft_comps=[1 / 3], ss_models=("BCC", "FCC", "HCP"))
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.n_missing_compounds == 0
        assert rep.is_insufficient()[0] is False

    def test_a_real_compound_at_extreme_composition_is_still_scored(self):
        """The rule must be name-based: DyB66 sits at x = 0.02 and is a genuine compound,
        so a terminal-composition rule would wrongly exempt it."""
        rep = assess([lc_phase("DyB66", 0.02)], dft_comps=[0.5])
        assert rep.phases[0].kind == "lc"
        assert rep.phases[0].supported is False
        assert rep.n_missing_compounds == 1

    def test_component_phase_does_not_rescue_a_missing_neighbour(self):
        """Exempting '(B)' must not also exempt a real compound next to it."""
        rep = assess([lc_phase("AB", 0.5), comp_phase("(B)", 1.0)], dft_comps=[])
        assert rep.n_missing_compounds == 1
        assert rep.unsupported_ranges == [[0.0, 1.0]]


class TestComponentLabelRecognition:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("(Ti)", True),
            ("(Cr)", True),
            ("(Ti) rt", True),
            ("(Ti, Cr)", True),  # the full-range solution field
            ("(Cr, Ti) ht1", True),
            ("(Ti,Cr)", True),
            ("(Ti,", False),  # '(Ti, Cr)' arrives split on whitespace
            ("TiCr2", False),
            ("(V)", False),
            ("(Ti, V)", False),  # one symbol is not a component of this system
            ("Ti", False),
            ("TiCr2 (Ti)", False),  # the group must LEAD the label
            ("()", False),
            ("", False),
        ],
    )
    def test_only_a_component_solution_label_matches(self, name, expected):
        assert mpds.is_component_phase_label(name, ["Cr", "Ti"]) is expected

    def test_no_elements_matches_nothing(self):
        assert mpds.is_component_phase_label("(Ti)", None) is False
        assert mpds.is_component_phase_label("(Ti)", []) is False
        assert mpds.is_component_phase_label("(Ti, Cr)", None) is False


class TestComponentPhaseTyping:
    """identify_mpds_phases must emit 'comp' for a component drawn as a boundary line."""

    @staticmethod
    def _json(with_elements=True):
        j = {
            "reference": {"entry": "synthetic"},
            "temp": [400.0, 1600.0],
            "comp_range": [0.0, 100.0],
            "shapes": [
                # The Cr-Ti case: a component phase digitized as the frame edge.
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "(Ti) rt",
                    "kind": "compound",
                    "phase": "Ti/194/hP2",
                    "svgpath": "100,500 100,1450",
                },
                # A genuine line compound, same shape kind.
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "TiCr2 rt",
                    "kind": "compound",
                    "phase": "TiCr2/227/cF24",
                    "svgpath": "35,500 35,1450",
                },
            ],
        }
        if with_elements:
            j["chemical_elements"] = ["Cr", "Ti"]
        return j

    def test_component_line_is_typed_comp_and_compound_stays_lc(self):
        by_name = {p["name"]: p for p in mpds.identify_mpds_phases(self._json())}
        assert by_name["(Ti)"]["type"] == "comp"
        assert by_name["TiCr2"]["type"] == "lc"

    def test_a_two_component_boundary_line_is_untouched_by_the_widened_label_rule(self):
        """The name reaching the rule is `label.split()[0]`, so a '(Ti, Cr)' line arrives as
        '(Ti,' and still types 'lc'. No cached json draws a two-component label as a
        boundary line, so this pins that the widening cannot reach identification."""
        j = self._json()
        j["shapes"][0]["label"] = "(Ti, Cr) rt"
        by_name = {p["name"]: p for p in mpds.identify_mpds_phases(j)}
        assert by_name["(Ti,"]["type"] == "lc"

    def test_elements_default_to_the_json_frame_block(self):
        """Without chemical_elements the rule cannot fire and the phase stays 'lc'."""
        phases = mpds.identify_mpds_phases(self._json(with_elements=False))
        assert next(p for p in phases if p["name"] == "(Ti)")["type"] == "lc"

    def test_explicit_elements_override_a_missing_frame_block(self):
        phases = mpds.identify_mpds_phases(self._json(with_elements=False), elements=["Cr", "Ti"])
        assert next(p for p in phases if p["name"] == "(Ti)")["type"] == "comp"

    def test_comp_phases_keep_the_line_compound_key_set(self):
        """'comp' splits out of 'lc' — it must not acquire cbounds or lose tbounds."""
        for phase in mpds.identify_mpds_phases(self._json()):
            assert set(phase) == {"type", "name", "comp", "tbounds"}

    def test_comp_phases_survive_the_frame_mirror(self):
        rich = mpds.identify_mpds_phases(self._json(), with_structure=True)
        mirrored = mpds.mirror_mpds_phases(rich)
        ti = next(p for p in mirrored if p["name"] == "(Ti)")
        assert ti["type"] == "comp" and ti["comp"] == pytest.approx(0.0)


class TestStructureKeyIsAdditive:
    """with_structure must stay opt-in: the characterization pins compare phase dicts by
    exact key set, so an always-on key would break every pinned low_t_exp_phases list."""

    @staticmethod
    def _json():
        return {
            "reference": {"entry": "synthetic"},
            "chemical_elements": ["Hf", "Zr"],
            "temp": [400.0, 1600.0],
            "comp_range": [0.0, 100.0],
            "shapes": [
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "(Hf)",
                    "kind": "phase",
                    "phase": "Hf/194/hP2",
                    "svgpath": "0,500 0,1400 20,1400 20,500",
                },
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "HfZr3 rt",
                    "kind": "compound",
                    "phase": "HfZr3/225/cF4",
                    "svgpath": "75,500 75,1450",
                },
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "(Hf, Zr)",
                    "kind": "phase",
                    "svgpath": "30,500 30,1450 60,1450 60,500",
                },
            ],
        }

    def test_default_output_has_the_historical_key_sets(self):
        for phase in mpds.identify_mpds_phases(self._json()):
            expected = {"type", "name", "comp", "tbounds"}
            if phase["type"] == "ss":
                expected.add("cbounds")
            assert set(phase) == expected

    def test_structure_is_the_only_difference(self):
        base = mpds.identify_mpds_phases(self._json())
        rich = mpds.identify_mpds_phases(self._json(), with_structure=True)
        assert len(base) == len(rich)
        for b, r in zip(base, rich):
            r = dict(r)
            assert set(r.pop("structure")) == {"prototype", "spacegroup", "pearson"}
            assert b == r

    def test_unresolved_shape_yields_all_none(self):
        rich = mpds.identify_mpds_phases(self._json(), with_structure=True)
        unresolved = next(p for p in rich if p["name"] == "(Hf, Zr)")
        assert unresolved["structure"] == {"prototype": None, "spacegroup": None, "pearson": None}

    def test_structure_survives_the_frame_mirror(self):
        rich = mpds.identify_mpds_phases(self._json(), with_structure=True)
        mirrored = mpds.mirror_mpds_phases(rich)
        assert {p["name"]: p["structure"] for p in mirrored} == {
            p["name"]: p["structure"] for p in rich
        }

    @pytest.mark.parametrize(
        "shape",
        [{}, {"phase": None}, {"phase": "garbage"}, {"phase": "A/notanumber/cF4"}, {"phase": 42}],
    )
    def test_malformed_identifiers_degrade_instead_of_raising(self, shape):
        assert mpds.parse_shape_structure(shape) == {
            "prototype": None,
            "spacegroup": None,
            "pearson": None,
        }

    def test_identifier_is_parsed_into_its_three_parts(self):
        parsed = mpds.parse_shape_structure({"phase": "Th0.805Tm0.195/225/cF4"})
        assert parsed == {"prototype": "Th0.805Tm0.195", "spacegroup": 225, "pearson": "cF4"}


class TestMergeRanges:
    @pytest.mark.parametrize(
        "ranges,expected",
        [
            ([], []),
            ([[0.2, 0.4]], [[0.2, 0.4]]),
            ([[0.2, 0.4], [0.3, 0.6]], [[0.2, 0.6]]),
            ([[0.6, 0.8], [0.1, 0.2]], [[0.1, 0.2], [0.6, 0.8]]),
            ([[0.1, 0.3], [0.3, 0.5]], [[0.1, 0.5]]),  # adjacent
            ([[0.4, 0.2]], [[0.2, 0.4]]),  # unordered input
        ],
    )
    def test_merge(self, ranges, expected):
        assert mpds.merge_ranges(ranges) == expected


def at_temp(phase, t_lo, t_hi):
    """The same synthetic phase with its digitized field placed between two temperatures."""
    placed = dict(phase)
    comp = phase["comp"]
    placed["tbounds"] = [[comp, t_lo], [comp, t_hi]]
    return placed


def monotone_liquidus(t_min, t_max=None):
    """A falling digitized liquidus, so it carries no interior local minimum to bound on."""
    return [[0.0, t_min + 800.0 if t_max is None else t_max], [1.0, t_min]]


class TestPhasesThatNeverBorderTheLiquid:
    """A solid constrains the liquidus only where its field actually touches the liquid.

    A compound that decomposes below the minimum DIGITIZED liquidus temperature is never in
    equilibrium with any point the objective scores, so its absence from the DFT hull masks
    nothing. The comparison is against the digitized curve, the one the fit scores.
    """

    def test_a_phase_below_the_liquidus_minimum_is_not_masked(self):
        """The Ge-W case, to its measured temperatures."""
        rep = assess(
            [at_temp(lc_phase("W2Ge3", 0.6), 700.0, 1022.6)],
            liquidus=monotone_liquidus(1202.5),
        )
        phase = rep.phases[0]
        assert phase.supported is False
        assert phase.borders_liquid is False
        assert phase.reason == "subliquidus:no_dft_phase:endpoint/endpoint"
        assert rep.unsupported_ranges == []
        assert rep.largest_supported_segment == pytest.approx(1.0)
        # Still a compound the hull does not hold; the counts are provenance, not the mask.
        assert (rep.n_compounds, rep.n_missing_compounds) == (1, 1)

    def test_a_phase_above_the_liquidus_minimum_is_still_masked(self):
        """The Ir-Sm case: SmIr2 tops out 1551 K ABOVE the minimum, so nothing changes."""
        rep = assess(
            [at_temp(lc_phase("SmIr2", 0.3334), 900.0, 2673.3)],
            invariants=[{"comp": 0.2003}, {"comp": 0.6350}],
            liquidus=monotone_liquidus(1122.4),
        )
        phase = rep.phases[0]
        assert phase.supported is False
        assert phase.borders_liquid is True
        assert phase.reason == "no_dft_phase:invariant/invariant"
        assert rep.unsupported_ranges == [[0.2003, 0.6350]]

    def test_a_phase_inside_the_margin_band_is_kept_masked_by_the_margin(self):
        """10 K below the liquidus minimum: excluded at the strict criterion, kept masked by
        a 25 K guard band. Both directions are pinned."""
        phases = [at_temp(lc_phase("AB2", 0.5), 900.0, 990.0)]
        liq = monotone_liquidus(1000.0)
        strict = assess(phases, liquidus=liq, subliquidus_margin=0.0)
        assert strict.phases[0].borders_liquid is False
        assert strict.unsupported_ranges == []
        guarded = assess(phases, liquidus=liq, subliquidus_margin=25.0)
        assert guarded.phases[0].borders_liquid is True
        assert guarded.unsupported_ranges == [[0.0, 1.0]]

    def test_an_infinite_margin_restores_the_unconditional_mask(self):
        """The off switch the three-arm measurement needs: 06g behaviour, exactly."""
        rep = assess(
            [at_temp(lc_phase("W2Ge3", 0.6), 700.0, 1022.6)],
            liquidus=monotone_liquidus(1202.5),
            subliquidus_margin=math.inf,
        )
        assert rep.phases[0].borders_liquid is True
        assert rep.unsupported_ranges == [[0.0, 1.0]]

    def test_the_criterion_cannot_fire_without_a_liquidus(self):
        """No digitized curve, no minimum to compare against -- so the mask stands."""
        rep = assess([at_temp(lc_phase("W2Ge3", 0.6), 700.0, 1022.6)])
        assert rep.phases[0].borders_liquid is True
        assert rep.unsupported_ranges == [[0.0, 1.0]]

    def test_a_phase_with_no_tbounds_keeps_its_mask(self):
        """The criterion may only ever REMOVE a mask, so missing data must not remove one."""
        phase = at_temp(lc_phase("W2Ge3", 0.6), 700.0, 1022.6)
        phase["tbounds"] = []
        rep = assess([phase], liquidus=monotone_liquidus(1202.5))
        assert rep.phases[0].borders_liquid is True
        assert rep.unsupported_ranges == [[0.0, 1.0]]

    def test_a_supported_phase_below_the_liquidus_reads_as_not_bordering(self):
        """The flag is computed for a supported phase too, and masks nothing either way."""
        rep = assess(
            [at_temp(lc_phase("AB", 0.5), 700.0, 900.0)],
            dft_comps=[0.5],
            liquidus=monotone_liquidus(1202.5),
        )
        assert rep.phases[0].supported is True
        assert rep.phases[0].borders_liquid is False
        assert rep.unsupported_ranges == []

    def test_it_reaches_an_unmodelled_solid_solution_field_too(self):
        """Not a compound rule: any unsupported phase below the liquidus masks nothing.

        The ``field:`` suffix records that, with no invariant on either side, the field runs
        to both anchors, even though the phase is excluded and masks nothing.
        """
        field = ss_phase("(A)", 0.0, 0.60, spacegroup=225)
        field["tbounds"] = [[0.0, 400.0], [0.60, 900.0]]
        rep = assess([field], liquidus=monotone_liquidus(1200.0))
        assert rep.phases[0].supported is False
        assert rep.phases[0].borders_liquid is False
        assert rep.phases[0].reason == "subliquidus:no_ss_model:FCC field:endpoint/endpoint"
        assert rep.phases[0].interval == (0.0, 1.0)
        assert rep.unsupported_ranges == []

    def test_the_margin_is_carried_on_the_report_thresholds(self):
        rep = assess([lc_phase("AB", 0.5)], subliquidus_margin=25.0)
        assert rep.thresholds["subliquidus_margin"] == pytest.approx(25.0)
        assert rep.as_dict()["phases"][0]["borders_liquid"] is True

    def test_the_default_margin_comes_from_config(self, monkeypatch):
        phases = [at_temp(lc_phase("AB2", 0.5), 900.0, 990.0)]
        liq = monotone_liquidus(1000.0)
        monkeypatch.setattr(config, "coverage_subliquidus_margin", 25.0)
        assert assess(phases, liquidus=liq).unsupported_ranges == [[0.0, 1.0]]
        monkeypatch.setattr(config, "coverage_subliquidus_margin", 0.0)
        assert assess(phases, liquidus=liq).unsupported_ranges == []


def shaped_liquidus(points):
    """A digitized liquidus with an interior shape, so local and global minima differ."""
    return [[float(x), float(t)] for x, t in points]


class TestFieldExtentAndDftMatching:
    """What a digitized field COVERS, and which DFT phase it is matched to.

    * **A** -- an ordered field is matched to DFT anywhere in its drawn extent, not only at
      its recorded ``comp`` (Co-Zn ``Zn11Co2``: recorded at 0.6900, drawn over 0.690-0.853,
      on the hull at 0.8462). A can only ADD support.
    * **B** -- matching takes the NEAREST DFT composition in tolerance (Hg-Sm ``Hg2Sm`` at
      0.3429 matches ``SmHg2`` at 0.3333, not ``SmHg3`` at 0.250).
    * **C** -- an unsupported COMPONENT solution field masks the UNION of its cbounds and its
      invariant-bounded primary field (Hg-In ``(Hg) lt``: dissolves to 0.1607, melts at
      0.1424, next invariant at 0.2601). C can only ADD mask.
    * **D** -- the sub-liquidus test compares a phase's ceiling with the liquidus minimum over
      its OWN field (Hg-Sm ``Hg6.5Sm``: 361.8 K, against 472.2 K over 0.0017-0.1513 and a
      global minimum of 319.9 K). D can only REMOVE mask.
    """

    # ---- A: a field is matched across its range, not at one point ----------------------
    def test_a_dft_phase_inside_the_field_supports_it_however_far_from_comp(self):
        """The Co-Zn ``Zn11Co2`` case, to its measured numbers."""
        rep = assess([field_at("Zn11Co2", 0.690, 0.853, comp=0.690)], dft_comps=[0.8462])
        # This distance alone, checked only against the recorded comp, would exceed
        # the tolerance and miss the match.
        assert abs(0.690 - 0.8462) > config.coverage_dft_cover_tol
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.846"
        assert rep.unsupported_ranges == []

    def test_a_dft_phase_outside_the_widened_field_still_does_not_support_it(self):
        rep = assess(
            [field_at("Zn11Co2", 0.690, 0.853, comp=0.690)],
            invariants=[{"comp": 0.60}, {"comp": 0.97}],
            dft_comps=[0.30],
        )
        assert rep.phases[0].supported is False
        assert rep.unsupported_ranges == [[0.60, 0.97]]

    def test_the_field_is_widened_by_exactly_dft_edge_tol_at_each_edge(self):
        tol = config.coverage_dft_edge_tol
        for hull, supported in (
            (0.60 + tol - 1e-9, True),
            (0.60 + tol + 1e-4, False),
            (0.40 - tol + 1e-9, True),
            (0.40 - tol - 1e-4, False),
        ):
            rep = assess([field_at("AB", 0.40, 0.60, comp=0.40)], dft_comps=[hull])
            assert rep.phases[0].supported is supported, hull

    def test_a_line_compound_is_still_matched_on_its_own_composition(self):
        """A is about fields with a homogeneity range; a line compound has none."""
        assert assess([lc_phase("AB", 0.50)], dft_comps=[0.65]).phases[0].supported is False
        assert assess([lc_phase("AB", 0.50)], dft_comps=[0.505]).phases[0].supported is True

    # ---- B: nearest, not first ---------------------------------------------------------
    def test_the_nearest_dft_composition_wins_for_a_line_compound(self):
        """The Hg-Sm ``Hg2Sm`` case: ``SmHg3`` at 0.250 and ``SmHg2`` at 0.3333 are both in
        tolerance of 0.3429, and 0.250 came first in composition order."""
        rep = assess([lc_phase("Hg2Sm", 0.3429)], dft_comps=[0.25, 0.3333])
        assert rep.phases[0].reason == "dft_phase@0.333"

    def test_a_dft_phase_inside_the_field_beats_a_nearer_one_outside_it(self):
        """The Co-Zn ``Zn13Co`` case: distance is measured to the FIELD, not to ``comp``.

        MPDS draws ``Zn13Co`` over 0.8708-0.8929 and records it at 0.8708. The hull holds
        ``Zn11Co2`` at 0.8462, ``Zn53Co7`` at 0.8833 and ``Zn13Co`` at 0.9286. 0.8462 is
        nearest to the recorded point (0.0246); 0.8833 lies inside the drawn field and is
        the match.
        """
        rep = assess(
            [field_at("Zn13Co", 0.8708, 0.8929, comp=0.8708)],
            dft_comps=[0.8462, 0.8833, 0.9286],
        )
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.883"

    def test_ties_inside_the_field_are_broken_by_nearness_to_comp(self):
        rep = assess([field_at("AB", 0.40, 0.60, comp=0.58)], dft_comps=[0.42, 0.57])
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.570"

    def test_the_solution_field_rescue_also_takes_the_nearest(self):
        """The width-capped rescue of an '(A, B)' field read the first match too."""
        rep = assess(
            [ss_phase("(A, B)", 0.45, 0.60, spacegroup=71)],
            dft_comps=[0.40, 0.55],
            elements=["A", "B"],
        )
        assert rep.phases[0].reason == "dft_phase@0.550"

    # ---- C: a solution field masks the liquidus it controls ----------------------------
    def test_an_unsupported_solution_field_reaches_its_flanking_invariant(self):
        """The Hg-In ``(Hg) lt`` case: cbounds end at 0.1607, the controlled liquidus at
        0.2601, and the peritectic there is what ends the field."""
        rep = assess(
            [field_at("(Hg) lt", 0.0004, 0.1607, comp=0.1424, spacegroup=166)],
            invariants=[melting_inv("cmp", 0.1424, "(Hg) lt", 0.1424), {"comp": 0.2601}],
            elements=["Hg", "In"],
        )
        assert rep.phases[0].supported is False
        assert rep.phases[0].interval == (pytest.approx(0.0), pytest.approx(0.2601))
        assert rep.phases[0].reason == "no_ss_model:sg166 field:endpoint/invariant"
        assert rep.unsupported_ranges == [[0.0, 0.2601]]

    def test_the_extension_is_a_union_and_never_narrows_the_cbounds_mask(self):
        """A field whose invariant-bounded primary field is NARROWER than its cbounds keeps
        the cbounds (Cr-Ti's axis-spanning '(Ti, Cr)'): C is a union, not a replacement."""
        rep = assess(
            [field_at("(Ti, Cr)", 0.0, 1.0, comp=0.5)],
            invariants=[{"comp": 0.3}, {"comp": 0.7}],
            elements=["Cr", "Ti"],
        )
        assert rep.unsupported_ranges == [[0.0, 1.0]]
        assert rep.phases[0].reason == "unknown_structure_no_ss"

    def test_a_supported_solution_field_keeps_its_cbounds_interval(self):
        rep = assess(
            [field_at("(A)", 0.0, 0.30, comp=0.0, spacegroup=194)],
            invariants=[{"comp": 0.80}],
            ss_models=("HCP",),
            elements=["A", "B"],
        )
        assert rep.phases[0].supported is True
        assert rep.phases[0].interval == (0.0, 0.30)
        assert rep.unsupported_ranges == []

    def test_the_reason_records_the_bounding_rules_only_when_the_field_extends(self):
        extended = assess(
            [field_at("(A)", 0.0, 0.30, comp=0.30, spacegroup=194)],
            invariants=[{"comp": 0.80}],
            elements=["A", "B"],
        )
        assert extended.phases[0].reason == "no_ss_model:HCP field:endpoint/invariant"
        assert extended.unsupported_ranges == [[0.0, 0.80]]
        unmoved = assess([field_at("(A)", 0.0, 1.0, comp=1.0, spacegroup=194)], elements=["A", "B"])
        assert unmoved.phases[0].reason == "no_ss_model:HCP"
        assert unmoved.unsupported_ranges == [[0.0, 1.0]]

    def test_a_solution_field_and_a_compound_share_one_bounding_path(self):
        """The same invariants, the same interval, whichever path the phase took."""
        invariants = [{"comp": 0.25}, {"comp": 0.60}]
        solution = assess(
            [field_at("(A, B)", 0.40, 0.55, comp=0.42, spacegroup=71)],
            invariants=invariants,
            elements=["A", "B"],
        )
        compound = assess([lc_phase("AB", 0.42)], invariants=invariants)
        assert solution.unsupported_ranges == compound.unsupported_ranges == [[0.25, 0.60]]

    # ---- D: the LOCAL liquidus minimum -------------------------------------------------
    def test_a_phase_below_the_liquidus_over_its_own_field_is_excluded(self):
        """The Hg-Sm ``Hg6.5Sm`` case: 361.8 K clears the global 319.9 K and loses to the
        472.2 K minimum over its own field."""
        liq = shaped_liquidus(
            [(0.0, 319.9), (0.05, 700.0), (0.10, 1000.0), (0.20, 1200.0), (1.0, 1100.0)]
        )
        rep = assess(
            [at_temp(lc_phase("Hg6.5Sm", 0.10), 300.0, 361.8)],
            invariants=[{"comp": 0.02}, {"comp": 0.15}],
            liquidus=liq,
        )
        phase = rep.phases[0]
        assert phase.supported is False
        assert phase.borders_liquid is False
        assert phase.reason == "subliquidus:no_dft_phase:invariant/invariant"
        assert rep.unsupported_ranges == []
        # The fact the GLOBAL comparison rests on, and why it keeps the mask.
        assert 361.8 > min(t for _, t in liq)

    def test_a_phase_bordering_the_liquid_over_its_own_field_is_still_masked(self):
        """The Ir-Sm ``SmIr2`` case: 2673.3 K against a local minimum of 1623.4 K."""
        liq = shaped_liquidus(
            [(0.0, 2500.0), (0.25, 2400.0), (0.45, 1900.0), (0.63, 1623.4), (1.0, 1122.4)]
        )
        rep = assess(
            [at_temp(lc_phase("SmIr2", 0.3334), 900.0, 2673.3)],
            invariants=[{"comp": 0.2003}, {"comp": 0.6350}],
            liquidus=liq,
        )
        assert rep.phases[0].borders_liquid is True
        assert rep.unsupported_ranges == [[0.2003, 0.6350]]

    def test_the_margin_band_is_applied_to_the_local_minimum(self):
        liq = shaped_liquidus([(0.0, 500.0), (0.45, 990.0), (0.55, 995.0), (1.0, 900.0)])
        phases = [at_temp(lc_phase("AB2", 0.5), 900.0, 980.0)]
        invariants = [{"comp": 0.40}, {"comp": 0.60}]
        strict = assess(phases, invariants=invariants, liquidus=liq, subliquidus_margin=0.0)
        assert strict.phases[0].borders_liquid is False  # 990 - 980 = 10 K below
        guarded = assess(phases, invariants=invariants, liquidus=liq, subliquidus_margin=25.0)
        assert guarded.phases[0].borders_liquid is True

    def test_the_local_minimum_is_read_over_the_field_c_widened(self):
        """C and D compose: the field D evaluates is the one C extended."""
        field = field_at("(A)", 0.0, 0.20, comp=0.20, spacegroup=194)
        field["tbounds"] = [[0.0, 400.0], [0.20, 950.0]]
        liq = shaped_liquidus([(0.0, 1000.0), (0.20, 1100.0), (0.60, 900.0), (1.0, 1200.0)])
        rep = assess([field], invariants=[{"comp": 0.60}], liquidus=liq, elements=["A", "B"])
        assert rep.phases[0].interval == (pytest.approx(0.0), pytest.approx(0.60))
        # The extension reaches the 900 K point, so the 950 K ceiling borders the liquid.
        assert rep.phases[0].borders_liquid is True
        assert rep.unsupported_ranges == [[0.0, 0.60]]


class TestFieldWindowIsSeparateFromThePointWindow:
    """``dft_edge_tol`` bounds how far off a drawn range or point a hull phase may lie and still
    sit on it; ``dft_structure_tol`` how far a hull phase sharing the space group may lie."""

    CR_IR = dict(lo=0.3208, hi=0.6996, comp=0.6996, spacegroup=194, pearson="hP2")

    def _cr_ir(self, **kw):
        f = self.CR_IR
        phase = field_at(
            "Cr0.5Ir0.5",
            f["lo"],
            f["hi"],
            comp=f["comp"],
            spacegroup=f["spacegroup"],
            pearson=f["pearson"],
        )
        return assess([phase], dft_comps=[0.75], elements=["Cr", "Ir"], **kw)

    def test_a_compound_past_the_field_edge_does_not_cover_it(self):
        rep = self._cr_ir()
        assert rep.phases[0].supported is False
        assert rep.phases[0].reason.startswith("ordered_field:no_dft_phase")

    def test_a_different_structure_past_the_edge_does_not_cover_it(self):
        rep = self._cr_ir(dft_spacegroups={0.75: 221})
        assert rep.phases[0].supported is False

    def test_the_override_reaches_the_ordered_field_matcher(self):
        rep = self._cr_ir(dft_edge_tol=0.10)
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.750"
        assert rep.phases[0].dft_via == "on"
        assert rep.phases[0].dft_structure == "unknown"

    def test_a_point_phase_off_its_composition_needs_a_structure_match(self):
        phase = lc_phase("AB", 0.50, spacegroup=221)
        assert assess([phase], dft_comps=[0.54]).phases[0].supported is False
        rep = assess([phase], dft_comps=[0.54], dft_spacegroups={0.54: 221})
        assert rep.phases[0].supported is True
        assert rep.phases[0].dft_via == "structure"
        far = assess([phase], dft_comps=[0.58], dft_spacegroups={0.58: 221})
        assert far.phases[0].supported is False

    def test_the_point_window_does_not_widen_a_field(self):
        rep = assess([field_at("AB", 0.40, 0.60, comp=0.40)], dft_comps=[0.65], dft_cover_tol=0.5)
        assert rep.phases[0].supported is False

    def test_the_solution_field_rescue_uses_the_edge_window(self):
        phases = [ss_phase("(A, B)", 0.45, 0.60, spacegroup=71)]
        near = assess(phases, dft_comps=[0.605], elements=["A", "B"])
        assert near.phases[0].reason == "dft_phase@0.605"
        far = assess(phases, dft_comps=[0.65], elements=["A", "B"])
        assert far.phases[0].supported is False
        widened = assess(phases, dft_comps=[0.65], elements=["A", "B"], dft_edge_tol=0.10)
        assert widened.phases[0].reason == "dft_phase@0.650"
        same = assess(phases, dft_comps=[0.64], elements=["A", "B"], dft_spacegroups={0.64: 71})
        assert same.phases[0].reason == "dft_phase@0.640"

    def test_the_report_records_both_windows(self):
        rep = assess([])
        assert rep.thresholds["dft_edge_tol"] == config.coverage_dft_edge_tol
        assert rep.thresholds["dft_structure_tol"] == config.coverage_dft_structure_tol
        rep = assess([], dft_edge_tol=0.0, dft_structure_tol=0.02)
        assert (rep.thresholds["dft_edge_tol"], rep.thresholds["dft_structure_tol"]) == (0.0, 0.02)

    def test_a_same_structure_compound_just_past_the_field_edge_covers_it(self):
        """``Na1.2Pb2.8`` is drawn to 0.7265 and cP4 NaPb3 sits at 0.75."""
        phase = field_at("Na1.2Pb2.8", 0.6554, 0.7265, comp=0.7002, spacegroup=221, pearson="cP4")
        rep = assess([phase], dft_comps=[0.75], elements=["Na", "Pb"], dft_spacegroups={0.75: 221})
        assert rep.phases[0].supported is True
        assert rep.phases[0].reason == "ordered_field:dft_phase@0.750"
        assert rep.phases[0].dft_via == "structure"
        bare = assess([phase], dft_comps=[0.75], elements=["Na", "Pb"])
        assert bare.phases[0].supported is False

    def test_a_shared_space_group_beats_a_nearer_edge(self):
        """Li-Zn: 0.75 is on both fields; only the farther one shares its space group."""
        near = field_at("Li2Zn5", 0.7093, 0.7453, comp=0.7093)
        same = field_at("Li0.2Zn0.8", 0.7592, 0.9186, comp=0.7592, spacegroup=194)
        got = solid_coverage.assign_dft_phases([near, same], ["Li", "Zn"], [0.75], {0.75: 194})
        assert list(got) == [1]
        assert (got[1].via, got[1].structure) == ("on", "match")

    def test_an_exact_stoichiometry_beats_a_shared_space_group(self):
        exact = lc_phase("LiZn3", 0.75)
        same = field_at("Li0.2Zn0.8", 0.755, 0.90, comp=0.755, spacegroup=194)
        got = solid_coverage.assign_dft_phases([exact, same], ["Li", "Zn"], [0.75], {0.75: 194})
        assert list(got) == [0]

    def test_a_triclinic_space_group_is_not_a_structure_match(self):
        phase = field_at("AB", 0.40, 0.60, comp=0.40, spacegroup=2)
        rep = assess([phase], dft_comps=[0.63], dft_spacegroups={0.63: 2})
        assert rep.phases[0].supported is False

    def test_matched_dft_comps_reads_the_windows_from_config(self):
        phases = [field_at("AB", 0.40, 0.60, comp=0.40, spacegroup=71)]
        tol = config.coverage_dft_edge_tol
        assert solid_coverage.matched_dft_comps(phases, [], [0.60 + tol - 1e-9]) == {
            0.60 + tol - 1e-9
        }
        assert solid_coverage.matched_dft_comps(phases, [], [0.60 + tol + 1e-4]) == set()
        assert solid_coverage.matched_dft_comps(phases, [], [0.64], dft_spacegroups={0.64: 71}) == {
            0.64
        }


class TestBordersLiquidIsMeasuredForEveryPhase:
    """``borders_liquid`` is a fact about the diagram, not about the hull.

    It is computed for every phase, supported or not -- never left as an unverified
    ``True``. Anything reading the flag to decide what a phase may own can therefore
    trust it.
    """

    @staticmethod
    def _liq(temp=1000.0, n=101):
        return [[i / (n - 1), temp] for i in range(n)]

    def test_a_supported_compound_reaching_the_liquidus_is_flagged(self):
        phase = lc_phase("AB", 0.5)
        phase["tbounds"] = [[0.5, 300.0], [0.5, 1000.0]]
        rep = assess([phase], dft_comps=[0.5], liquidus=self._liq(1000.0))
        assert rep.phases[0].borders_liquid is True


class TestExtensionCapAndWidthGuard:
    """An unsupported component field's extension is capped, unless the drawn fields dominate.

    Every case here names ``ss_extension_cap`` explicitly; the module helper switches the cap
    off so the bounding cases above keep pinning the union arithmetic on its own.
    """

    def test_the_cap_clips_the_extension_past_the_drawn_field(self):
        rep = assess(
            [field_at("(A)", 0.0, 0.30, comp=0.30, spacegroup=194)],
            invariants=[{"comp": 0.80}],
            elements=["A", "B"],
            ss_extension_cap=0.05,
        )
        assert rep.phases[0].interval == (pytest.approx(0.0), pytest.approx(0.35))
        assert rep.unsupported_ranges[0][1] == pytest.approx(0.35)
        assert rep.phases[0].reason == "no_ss_model:HCP field:endpoint/invariant cap:0.05"
        assert rep.thresholds["ss_guard_fired"] is False

    def test_an_extension_shorter_than_the_cap_is_untouched(self):
        rep = assess(
            [field_at("(A)", 0.0, 0.30, comp=0.30, spacegroup=194)],
            invariants=[{"comp": 0.33}],
            elements=["A", "B"],
            ss_extension_cap=0.05,
        )
        assert rep.unsupported_ranges == [[0.0, 0.33]]
        assert "cap:" not in rep.phases[0].reason

    def test_the_cap_never_narrows_the_drawn_field_itself(self):
        rep = assess(
            [field_at("(A)", 0.0, 0.40, comp=0.0, spacegroup=194)],
            invariants=[{"comp": 0.35}],
            elements=["A", "B"],
            ss_extension_cap=0.05,
        )
        assert rep.unsupported_ranges == [[0.0, 0.40]]
        assert "cap:" not in rep.phases[0].reason

    def test_the_guard_keeps_the_union_when_the_drawn_fields_dominate(self):
        phases = [
            field_at("(A)", 0.0, 0.45, comp=0.0, spacegroup=225),
            field_at("(B)", 0.62, 1.0, comp=1.0, spacegroup=194),
        ]
        guarded = assess(phases, elements=["A", "B"], ss_extension_cap=0.05, ss_width_guard=0.67)
        assert guarded.thresholds["ss_drawn_width"] == pytest.approx(0.83)
        assert guarded.thresholds["ss_guard_fired"] is True
        assert guarded.unsupported_ranges == [[0.0, 1.0]]
        assert all("cap:" not in p.reason for p in guarded.phases)

        capped = assess(phases, elements=["A", "B"], ss_extension_cap=0.05, ss_width_guard=0.90)
        assert capped.thresholds["ss_guard_fired"] is False
        assert capped.unsupported_ranges[0][1] == pytest.approx(0.50)
        assert capped.unsupported_ranges[1][0] == pytest.approx(0.57)
        assert capped.supported_total == pytest.approx(0.07)

    def test_only_masked_fields_count_toward_the_guard(self):
        phases = [
            field_at("(A)", 0.0, 0.45, comp=0.0, spacegroup=225),
            field_at("(B)", 0.62, 1.0, comp=1.0, spacegroup=194),
        ]
        rep = assess(
            phases,
            elements=["A", "B"],
            ss_models=("FCC",),
            ss_extension_cap=0.05,
            ss_width_guard=0.67,
        )
        assert rep.phases[0].supported is True
        assert rep.thresholds["ss_drawn_width"] == pytest.approx(0.38)
        assert rep.thresholds["ss_guard_fired"] is False
        assert len(rep.unsupported_ranges) == 1
        assert rep.unsupported_ranges[0][0] == pytest.approx(0.57)

    def test_the_cap_applies_to_a_two_component_field_below_the_guard(self):
        rep = assess(
            [ss_phase("(A, B)", 0.30, 0.50, spacegroup=71)],
            invariants=[{"comp": 0.25}, {"comp": 0.60}],
            dft_comps=[0.90],
            ss_extension_cap=0.05,
        )
        assert rep.unsupported_ranges[0][0] == pytest.approx(0.25)
        assert rep.unsupported_ranges[0][1] == pytest.approx(0.55)

    def test_config_supplies_the_default_and_math_inf_restores_the_full_extension(self):
        phases = [field_at("(A)", 0.0, 0.30, comp=0.30, spacegroup=194)]
        invariants = [{"comp": 0.80}]
        full = assess(phases, invariants=invariants, elements=["A", "B"], ss_extension_cap=math.inf)
        assert full.unsupported_ranges == [[0.0, 0.80]]

        shipped = solid_coverage.assess_solid_coverage(
            phases,
            invariants,
            SPAN,
            [],
            (),
            elements=["A", "B"],
            thresholds={"field_ceiling_tol": math.inf},
        )
        assert shipped.thresholds["ss_extension_cap"] == 0.05
        assert shipped.thresholds["ss_width_guard"] == 0.67
        assert shipped.unsupported_ranges[0][1] == pytest.approx(0.35)

        old = (config.coverage_ss_extension_cap, config.coverage_ss_width_guard)
        try:
            config.set_coverage_thresholds(ss_extension_cap=0.10)
            wider = solid_coverage.assess_solid_coverage(
                phases,
                invariants,
                SPAN,
                [],
                (),
                elements=["A", "B"],
                thresholds={"field_ceiling_tol": math.inf},
            )
            assert wider.unsupported_ranges[0][1] == pytest.approx(0.40)
            with pytest.raises(ValueError):
                config.set_coverage_thresholds(ss_extension_cap=-0.1)
            with pytest.raises(ValueError):
                config.set_coverage_thresholds(ss_width_guard=0.0)
        finally:
            config.set_coverage_thresholds(ss_extension_cap=old[0], ss_width_guard=old[1])


class TestAPrintedTwoLiquidLabel:
    """``printed_gap_comps`` refuses a masked field that runs across a printed gap.

    Inside a printed ``L1 + L2`` the conjugate is a second liquid, so the field is no
    evidence about the missing solid. The refusal is all-or-nothing on the interval, the
    same shape the invariant walk's exception takes.

    These cases carry no ``mig`` record, so they pin the path where the printed label is the
    only thing locating the region; :class:`TestAPrintedGapIsFoundByItsExtent` pins the path
    where a record locates it instead.
    """

    PHASES = [lc_phase("AB", 0.50)]
    INVARIANTS = [eutectic_inv(0.20), eutectic_inv(0.80)]

    def test_the_field_masks_when_nothing_is_printed(self):
        """POSITIVE CONTROL: the same geometry, with no label, still masks."""
        rep = assess(self.PHASES, self.INVARIANTS)
        assert rep.unsupported_ranges == [pytest.approx([0.20, 0.80])]
        assert rep.phases[0].reason == "no_dft_phase:invariant/invariant"

    def test_a_label_inside_the_field_refuses_it_whole(self):
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[0.50])
        assert rep.unsupported_ranges == []
        assert rep.phases[0].reason.endswith("printed_gap")

    def test_the_refusal_does_not_trim_the_interval(self):
        """All or nothing: a refused field contributes no narrowed remainder either."""
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[0.25])
        assert rep.unsupported_ranges == []
        assert rep.unsupported_fraction == pytest.approx(0.0)

    def test_a_label_outside_the_field_masks_as_before(self):
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[0.90])
        assert rep.unsupported_ranges == [pytest.approx([0.20, 0.80])]
        assert "printed_gap" not in rep.phases[0].reason

    @pytest.mark.parametrize("comp", [0.20, 0.80])
    def test_a_label_on_either_edge_refuses(self, comp):
        """The bound is closed, as ``mask_walked`` reads it."""
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[comp])
        assert rep.unsupported_ranges == []

    def test_it_refuses_only_the_field_that_spans_the_label(self):
        """A second unsupported compound, clear of the label, keeps its own mask."""
        phases = [lc_phase("AB", 0.50), lc_phase("AB3", 0.90)]
        invariants = [eutectic_inv(0.20), eutectic_inv(0.80), eutectic_inv(0.95)]
        rep = assess(phases, invariants, printed_gap_comps=[0.50])
        assert rep.unsupported_ranges == [pytest.approx([0.80, 0.95])]
        assert rep.n_missing_compounds == 2

    def test_a_refused_field_is_still_counted_as_missing(self):
        """Refusing the MASK does not invent support: the compound is still unsupported."""
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[0.50])
        assert rep.n_compounds == 1
        assert rep.n_missing_compounds == 1
        assert rep.phases[0].supported is False

    def test_the_comps_are_recorded_on_the_report(self):
        rep = assess(self.PHASES, self.INVARIANTS, printed_gap_comps=[0.50])
        assert rep.thresholds["printed_gap_comps"] == pytest.approx([0.50])

    def test_no_comps_is_the_default(self):
        rep = assess(self.PHASES, self.INVARIANTS)
        assert rep.thresholds["printed_gap_comps"] == []


def printed_dome(lo, hi, temp=1000.0):
    """A ``mig`` record whose ``cbounds`` give a printed two-liquid region its extent."""
    return {
        "type": "mig",
        "comp": 0.5 * (lo + hi),
        "cbounds": [[lo, temp], [hi, temp]],
        "tbounds": [[lo, temp], [0.5 * (lo + hi), temp + 200.0]],
        "phases": ["L1", "L2"],
        "phase_comps": [lo, hi],
    }


class TestAPrintedGapIsFoundByItsExtent:
    """Where a ``mig`` record locates the printed region, its extent decides, not the label.

    A leader arrow prints a label outside the lens it names, so the label's composition can
    sit inside a field the region never reaches. ``cbounds`` say where the region is. None of
    these cases supplies a liquidus, so no record's feet are confirmed -- which is the point:
    an extent is a statement about composition, and the feet resting on the liquidus is a
    statement about temperature.
    """

    PHASES = [lc_phase("AB", 0.30)]
    FLANKS = [eutectic_inv(0.24), eutectic_inv(0.40)]

    def test_a_label_outside_its_own_lens_leaves_a_clear_field_masked(self):
        """The Mn-Y case: the label prints inside a field the dome never reaches."""
        rep = assess(
            self.PHASES, [*self.FLANKS, printed_dome(0.17, 0.23)], printed_gap_comps=[0.25]
        )
        assert rep.unsupported_ranges == [pytest.approx([0.24, 0.40])]
        assert "printed_gap" not in rep.phases[0].reason

    def test_a_region_reaching_the_field_refuses_it(self):
        """POSITIVE CONTROL: the extent decides against a field as readily as for it."""
        rep = assess(self.PHASES, [printed_dome(0.20, 0.40)], printed_gap_comps=[0.25])
        assert rep.unsupported_ranges == []
        assert rep.phases[0].reason.endswith("printed_gap")

    def test_the_extent_decides_even_where_the_label_is_far_away(self):
        rep = assess(self.PHASES, [printed_dome(0.20, 0.40)], printed_gap_comps=[0.95])
        assert rep.unsupported_ranges == []

    def test_a_foot_on_a_pure_element_states_no_extent(self):
        """The In-Rb shape: feet at both ends locate nothing, so the label decides alone."""
        rep = assess(self.PHASES, [*self.FLANKS, printed_dome(0.0, 1.0)], printed_gap_comps=[0.25])
        assert rep.unsupported_ranges == []
        assert rep.phases[0].reason.endswith("printed_gap")

    def test_that_record_does_not_refuse_a_field_the_label_misses(self):
        """POSITIVE CONTROL on the fallback: an unusable extent is not a blanket refusal."""
        rep = assess(self.PHASES, [*self.FLANKS, printed_dome(0.0, 1.0)], printed_gap_comps=[0.95])
        assert rep.unsupported_ranges == [pytest.approx([0.24, 0.40])]

    def test_without_a_printed_label_a_record_refuses_nothing(self):
        """POSITIVE CONTROL: a record alone is not a printed region."""
        rep = assess(self.PHASES, [printed_dome(0.20, 0.40)])
        assert rep.unsupported_ranges == [pytest.approx([0.20, 0.40])]
        assert "printed_gap" not in rep.phases[0].reason


def test_mpds_has_no_solid_coverage_names():
    """The coverage names resolve only in ``gliquid.solid_coverage``, never through ``mpds``."""
    for name in ("assess_solid_coverage", "SolidCoverageReport", "PhaseCoverage"):
        with pytest.raises(AttributeError, match=name):
            getattr(mpds, name)
