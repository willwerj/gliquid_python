"""Unit tests for mpds.assess_solid_coverage on synthetic phase diagrams.

Synthetic JSONs keep these offline and make each support rule testable in isolation --
no cache, no DFT hull, no BinaryLiquid. See test_solid_coverage_gate.py for the
end-to-end wiring through fit_parameters.
"""

import pytest

from gliquid import mpds

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


def lc_phase(name, comp, spacegroup=None, pearson=None):
    """A digitized line compound at `comp`."""
    return {
        "type": "lc",
        "name": name,
        "comp": comp,
        "tbounds": [[comp, 500.0], [comp, 1500.0]],
        "structure": {"prototype": name, "spacegroup": spacegroup, "pearson": pearson},
    }


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
    return mpds.assess_solid_coverage(
        list(phases), list(invariants), SPAN, list(dft_comps), ss_models, **kw
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
        rep = assess(
            [
                ss_phase("(A)", 0.0, 0.30, spacegroup=194),
                ss_phase("(B)", 0.52, 1.00, spacegroup=229),
            ]
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
    """MPDS leaves '(A, B)' shapes structurally unresolved; ~13.5% of cached fields."""

    def test_unresolved_field_is_supported_when_any_model_is_loaded(self):
        rep = assess([ss_phase("(A, B)", 0.0, 1.0)], ss_models=("HCP",))
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason == "unknown_structure_ss_loaded"

    def test_unresolved_field_is_unsupported_with_no_models(self):
        rep = assess([ss_phase("(A, B)", 0.0, 1.0)])
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "unknown_structure_no_ss"


class TestDftRescueOfOrderedPhases:
    """An ordered compound with a homogeneity range is fine if DFT has it -- but the rescue
    must be width-capped, or one interior compound would excuse an arbitrarily wide field."""

    def test_ordered_field_is_rescued_by_a_nearby_dft_phase(self):
        # width 0.15: wider than ss_narrow_tol (0.10), within ss_rescue_max_width (0.25)
        rep = assess([ss_phase("A3B5", 0.45, 0.60, spacegroup=71)], dft_comps=[0.55])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.phases[0].reason.startswith("dft_phase@")

    def test_ordered_field_without_a_nearby_dft_phase_stays_unsupported(self):
        rep = assess([ss_phase("A3B5", 0.30, 0.50, spacegroup=71)], dft_comps=[0.90])
        assert rep.unsupported_fraction == pytest.approx(0.20)
        assert rep.phases[0].reason == "no_ss_model:sg71"

    def test_wide_field_is_not_rescued_however_close_the_dft_phase(self):
        """Regression: uncapped, this scored 0.00 and complete solid solutions with no
        solid-solution models (Ag-Au, Ta-W, Se-Te, Bi-Sb, As-Sb) were declared fittable --
        precisely the failure this gate exists to catch."""
        rep = assess([ss_phase("(A,B)", 0.0, 1.0, spacegroup=71)], dft_comps=[0.5])
        assert rep.unsupported_fraction == pytest.approx(1.0)
        assert rep.phases[0].reason == "too_wide_to_rescue:sg71"

    def test_rescue_width_cap_is_tunable(self):
        phases = [ss_phase("A3B5", 0.20, 0.60, spacegroup=71)]
        assert assess(phases, dft_comps=[0.4]).unsupported_fraction == pytest.approx(0.40)
        rescued = assess(phases, dft_comps=[0.4], ss_rescue_max_width=0.5)
        assert rescued.unsupported_fraction == pytest.approx(0.0)


class TestMissingCompounds:
    def test_compound_with_a_dft_counterpart_is_supported(self):
        rep = assess([lc_phase("AB", 0.5)], dft_comps=[0.52])
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


class TestThresholdDecision:
    def test_high_unsupported_fraction_is_disqualifying(self):
        rep = assess([ss_phase("(A)", 0.0, 1.0, spacegroup=194)])
        insufficient, reason = rep.is_insufficient()
        assert insufficient is True and "no solid reference" in reason

    def test_many_missing_compounds_disqualify_below_the_fraction_cap(self):
        """The Rb-Sn class: each missing compound owns a modest field, but collectively the
        hull cannot represent the solid side."""
        invariants = [{"comp": 0.6}, {"comp": 0.75}, {"comp": 0.955}]
        rep = assess(
            [lc_phase("A12B17", 0.60), lc_phase("AB2", 0.667), lc_phase("AB4", 0.80)],
            invariants=invariants,
            dft_comps=[0.5],
        )
        assert rep.unsupported_fraction < rep.thresholds["skip_frac"]
        insufficient, reason = rep.is_insufficient()
        assert insufficient is True and "no DFT counterpart" in reason

    def test_single_missing_compound_does_not_trip_the_count_rule(self):
        """min_missing guards marginal systems (Eu-Pb: one missing compound, 7% of the axis)."""
        rep = assess(
            [lc_phase("AB", 0.5), lc_phase("AB3", 0.75)],
            invariants=[{"comp": 0.45}, {"comp": 0.55}],
            dft_comps=[0.75],
        )
        assert rep.n_missing_compounds == 1
        assert rep.is_insufficient()[0] is False

    def test_clean_system_passes(self):
        rep = assess([lc_phase("AB", 0.5)], dft_comps=[0.5])
        assert rep.unsupported_fraction == pytest.approx(0.0)
        assert rep.is_insufficient()[0] is False


class TestReportPlumbing:
    def test_span_clips_the_masked_measure(self):
        """A partial liquidus is scored over what was actually digitized."""
        rep = mpds.assess_solid_coverage(
            [ss_phase("(A)", 0.0, 0.5, spacegroup=194)], [], [0.25, 0.75], [], ()
        )
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
        """Spacegroup alone is not a structure (NbC is 225 but rock salt, not FCC). Audit
        only -- gating on Pearson was measured as too aggressive on real FCC solutions."""
        rep = assess([ss_phase("NbC", 0.2, 0.5, spacegroup=225, pearson="cF8")], ss_models=("FCC",))
        assert rep.phases[0].supported is True
        assert "pearson=cF8" in rep.phases[0].reason


class TestComponentPhases:
    """A pure component's phase is not a line compound.

    MPDS routinely digitizes a terminal phase as a `kind='compound'` boundary line. Typed
    as 'lc' it became a reported compound at x = 0 or 1 that `covered_by_dft` could never
    match, because `dft_comps` is interior-only by construction.
    """

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

        As a line compound this masked [0.3333, 1.0] = 67% > the 45% cap and skipped a
        system that fits at MAE 11.9 K. The Laves columns are narrow fields; the '(Ti, Cr)'
        field has no resolved structure and rides on the loaded models.
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
            ("(Ti,", False),  # '(Ti, Cr)' arrives split on whitespace
            ("TiCr2", False),
            ("(V)", False),
            ("Ti", False),
            ("", False),
        ],
    )
    def test_only_a_bare_component_label_matches(self, name, expected):
        assert mpds.is_component_phase_label(name, ["Cr", "Ti"]) is expected

    def test_no_elements_matches_nothing(self):
        assert mpds.is_component_phase_label("(Ti)", None) is False
        assert mpds.is_component_phase_label("(Ti)", []) is False


class TestComponentPhaseTyping:
    """identify_mpds_phases must emit 'comp' for a component drawn as a boundary line."""

    @staticmethod
    def _json(with_elements=True):
        j = {
            "reference": {"entry": "synthetic"},
            "temp": [400.0, 1600.0],
            "comp_range": [0.0, 100.0],
            "shapes": [
                # The Cr-Ti defect: a component phase digitized as the frame edge.
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

    def test_elements_default_to_the_json_frame_block(self):
        """Without chemical_elements the rule cannot fire and degrades to the old 'lc'."""
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
