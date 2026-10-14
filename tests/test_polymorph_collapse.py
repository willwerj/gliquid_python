"""Polymorph collapse: one phase per compound, identified by the low-temperature form.

MPDS digitizes each thermal form of a compound as its own shape: C-Ce draws two CeC2 lines,
and C-La names 'LaC2 rt' in the labels block but digitizes only the ht dome, whose bottom is
far above the low-temperature cutoff. Each compound becomes one phase.

Fixtures are synthetic but shaped on the real jsons; the two C-Ce carbide lines span the full
temperature axis 10 K apart, which defeats any temperature-based ranking of the forms.
"""

import pytest

from gliquid import mpds

from ._mpds_builders import field_shape as _dome
from ._mpds_builders import line_shape as _line
from ._mpds_builders import mpds_record


def _json(shapes, temp=(200.0, 2600.0), labels=(), elements=("C", "Ce")):
    return mpds_record(shapes, elements, temp, labels)


def _collapse(data, **kwargs):
    return mpds.collapse_polymorphs(
        mpds.identify_mpds_phases(data, with_polymorph=True), data, **kwargs
    )


class TestLabelParsing:
    def test_subscript_markup_is_stripped(self):
        """The labels block carries markup; shape['label'] does not. Both must join."""
        assert mpds.split_phase_label("LaC<sub>2</sub> rt") == ("LaC2", "rt")
        assert mpds.split_phase_label("Ce2C3") == ("Ce2C3", "")

    def test_component_labels_keep_their_parenthesis(self):
        assert mpds.split_phase_label("(Ce) ht1") == ("(Ce)", "ht1")

    def test_empty_label_degrades_quietly(self):
        assert mpds.split_phase_label("") == ("", "")
        assert mpds.split_phase_label(None) == ("", "")


class TestPolymorphRank:
    def test_thermal_forms_order_by_temperature(self):
        ranks = [mpds.polymorph_rank(s) for s in ("lt", "rt", "", "ht", "ht1", "ht2")]
        assert ranks == sorted(ranks)
        assert mpds.polymorph_rank("rt") == mpds.polymorph_rank("")

    def test_high_pressure_never_wins_the_low_temperature_slot(self):
        assert mpds.polymorph_rank("hp") > mpds.polymorph_rank("ht2")

    def test_embedded_token_is_found(self):
        """Ga-Pt labels a boundary 'Pt3Ga Ga + rt'."""
        assert mpds.polymorph_rank("ga + rt") == mpds.polymorph_rank("rt")

    def test_unrecognized_suffix_ranks_as_an_ordinary_compound(self):
        """'HfFe2 hex1' is a structure note, not a thermal form."""
        assert mpds.polymorph_rank("hex1") == mpds.polymorph_rank("")


class TestCeC2Collapse:
    """C-Ce: both CeC2 lines span the full temperature axis, 10 K apart."""

    @staticmethod
    def _data():
        return _json(
            [
                _line("CeC2 rt", 33.3016, 218.246, 2249.67, "CeC2/139/tI6"),
                _line("CeC2 ht", 33.3016, 199.392, 2239.94, "CeC2/225/cF36"),
                _line("Ce2C3", 39.9924, 199.392, 1899.34, "Ce2C3/220/cI40"),
            ]
        )

    def test_one_entry_survives_per_compound(self):
        by_name = [p["name"] for p in _collapse(self._data())]
        assert by_name.count("CeC2") == 1
        assert by_name.count("Ce2C3") == 1

    def test_identity_comes_from_the_rt_form_not_the_colder_line(self):
        """The ht line is digitized 19 K LOWER, so ranking by temperature picks wrong."""
        cec2 = next(p for p in _collapse(self._data()) if p["name"] == "CeC2")
        assert cec2["tbounds"][0][1] == pytest.approx(491.396)  # rt bottom, not 472.5

    def test_upper_bound_is_the_compound_melting_point(self):
        cec2 = next(p for p in _collapse(self._data()) if p["name"] == "CeC2")
        assert cec2["tbounds"][1][1] == pytest.approx(2522.82)


class TestLabelOnlyLowTemperatureForm:
    """C-La: only the ht dome is digitized; 'LaC2 rt' exists solely as a label."""

    @staticmethod
    def _data():
        return _json(
            [
                _dome(
                    "LaC2 ht",
                    [(33.28, 1081.65), (34.95, 1500.0), (33.65, 2362.45), (33.28, 1081.65)],
                    "LaC2/225/cF36",
                )
            ],
            temp=(0, 4500),
            labels=[
                ["LaC<sub>2</sub> rt", [32.5636, 102.2625], 12681],
                ["LaC<sub>2</sub> ht", [32.0577, 1348.5668], 20226],
            ],
            elements=("C", "La"),
        )

    def test_the_compound_reaches_the_low_temperature_tables(self):
        lac2 = next(p for p in _collapse(self._data()) if p["name"] == "LaC2")
        assert lac2["tbounds"][0][1] < mpds.low_temp_threshold(self._data())

    def test_lower_bound_comes_from_the_rt_label(self):
        lac2 = next(p for p in _collapse(self._data()) if p["name"] == "LaC2")
        assert lac2["tbounds"][0][1] == pytest.approx(375.4125)

    def test_upper_bound_stays_the_digitized_melting_point(self):
        lac2 = next(p for p in _collapse(self._data()) if p["name"] == "LaC2")
        assert lac2["tbounds"][1][1] == pytest.approx(2635.6)

    def test_a_hot_label_alone_does_not_rescue_the_compound(self):
        """Only an rt/lt label below the cutoff counts; the ht label must not."""
        data = self._data()
        data["labels"] = [["LaC<sub>2</sub> ht", [32.0577, 1348.5668], 20226]]
        lac2 = next(p for p in _collapse(data) if p["name"].startswith("LaC2"))
        assert lac2["tbounds"][0][1] > mpds.low_temp_threshold(data)


class TestWhatMustNotCollapse:
    def test_same_name_at_different_compositions_stays_separate(self):
        """Al-Dy labels DyAl2 at both 0.333 and 0.666, Hg-Zr HgZr3 at 0.25 and 0.75 --
        a mirrored-frame digitization artifact, not a pair of polymorphs."""
        data = _json(
            [_line("DyAl2", 33.3, 200.0, 1500.0), _line("DyAl2", 66.6, 200.0, 989.0)],
            elements=("Al", "Dy"),
        )
        assert len([p for p in _collapse(data) if p["name"] == "DyAl2"]) == 2

    def test_component_solid_solutions_pass_through_untouched(self):
        """'(Ce) ht1' / '(Ce) ht2' are terminal fields keyed by their own labels."""
        data = _json(
            [
                _dome("(Ce) ht1", [(95.2, 201.2), (99.9, 450.0), (95.2, 201.2)]),
                _dome("(Ce) ht2", [(96.8, 725.3), (99.9, 793.7), (96.8, 725.3)]),
            ]
        )
        names = {p["name"] for p in _collapse(data)}
        assert names == {"(Ce) ht1", "(Ce) ht2"}


class TestKeySetIsPreserved:
    """Consumers pin phase dicts by exact key set; an uncollapsed compound must not move."""

    def test_ordinary_compound_keeps_the_historical_keys(self):
        data = _json([_line("Ce2C3", 39.9924, 199.392, 1899.34)])
        assert set(_collapse(data)[0]) == {"type", "name", "comp", "tbounds"}

    def test_the_internal_polymorph_key_never_escapes(self):
        data = TestCeC2Collapse._data()
        assert all("polymorph" not in p for p in _collapse(data))

    def test_collapse_does_not_mutate_its_input(self):
        data = TestCeC2Collapse._data()
        phases = mpds.identify_mpds_phases(data, with_polymorph=True)
        mpds.collapse_polymorphs(phases, data)
        assert all("polymorph" in p for p in phases)


class TestOptInKeys:
    def test_polymorph_key_is_off_by_default(self):
        data = _json([_line("CeC2 rt", 33.3016, 218.246, 2249.67)])
        assert "polymorph" not in mpds.identify_mpds_phases(data)[0]

    def test_polymorph_key_records_the_suffix(self):
        data = _json([_line("CeC2 rt", 33.3016, 218.246, 2249.67)])
        assert mpds.identify_mpds_phases(data, with_polymorph=True)[0]["polymorph"] == "rt"

    def test_it_composes_with_with_structure(self):
        data = _json([_line("CeC2 ht", 33.3016, 199.392, 2239.94, "CeC2/225/cF36")])
        phase = mpds.identify_mpds_phases(data, with_structure=True, with_polymorph=True)[0]
        assert phase["polymorph"] == "ht"
        assert phase["structure"]["pearson"] == "cF36"


class TestDegenerateInput:
    def test_empty_phase_list_round_trips(self):
        assert mpds.collapse_polymorphs([], _json([])) == []

    def test_malformed_label_entries_are_skipped(self):
        data = _json([_line("CeC2 ht", 33.3016, 199.392, 2239.94)])
        data["labels"] = [["CeC2 rt"], ["CeC2 rt", None], "not-a-label", []]
        assert len(_collapse(data)) == 1

    def test_missing_labels_block_is_tolerated(self):
        data = _json([_line("CeC2 ht", 33.3016, 199.392, 2239.94)])
        del data["labels"]
        assert len(_collapse(data)) == 1


class TestPairingByNormalisedFormula:
    """Two spellings of one formula are one compound; two compositions are not."""

    @staticmethod
    def _bi_mg():
        return _json(
            [_line("Mg3Bi2 rt", 60.0, 100.0, 700.0), _line("Bi2Mg3 ht", 60.0, 700.0, 823.0)],
            elements=("Bi", "Mg"),
        )

    def test_two_spellings_at_one_composition_are_one_cluster(self):
        (merged,) = _collapse(self._bi_mg())
        assert merged["name"] == "Mg3Bi2"
        assert merged["tbounds"][0][1] == pytest.approx(373.15)
        assert merged["tbounds"][1][1] == pytest.approx(1096.15)

    def test_elements_default_to_the_json_frame(self):
        data = self._bi_mg()
        phases = mpds.identify_mpds_phases(data, with_polymorph=True)
        assert len(mpds.collapse_polymorphs(phases, data)) == 1
        assert len(mpds.collapse_polymorphs(phases, data, elements=["Mg", "Bi"])) == 1

    @pytest.mark.parametrize("tol,count", [(0.02, 2), (0.04, 1)])
    def test_a_wide_pair_clusters_only_within_tolerance(self, tol, count):
        data = _json(
            [_line("Cu9Al4 rt", 69.0, 200.0, 800.0), _line("Cu9Al4 ht", 72.45, 800.0, 1030.0)],
            elements=("Al", "Cu"),
        )
        assert len(_collapse(data, comp_tol=tol)) == count

    @pytest.mark.parametrize("tol", [0.02, 0.04, 0.1, 0.3])
    def test_same_name_at_different_compositions_stays_separate_at_every_tolerance(self, tol):
        data = _json(
            [_line("DyAl2", 33.3, 200.0, 1500.0), _line("DyAl2", 66.6, 200.0, 989.0)],
            elements=("Al", "Dy"),
        )
        assert len(_collapse(data, comp_tol=tol)) == 2

    def test_an_unparseable_label_keys_as_its_own_text(self):
        data = _json(
            [_line("XY2 rt", 33.3, 300.0, 800.0), _line("XY2 ht", 33.3, 800.0, 1200.0)],
            elements=("X", "Y"),
        )
        assert [p["name"] for p in _collapse(data)] == ["XY2"]
