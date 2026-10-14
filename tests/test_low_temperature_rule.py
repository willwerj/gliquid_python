"""One low-temperature rule: ``is_low_temperature_phase`` at every call site.

The relative fraction is ``config.coverage_low_temp_axis_fraction``, thermal forms pair by
normalised formula within ``config.coverage_polymorph_pair_tol``, and an unpaired
high-temperature form is never a low-temperature phase -- not by the fraction, not by the
room-temperature clause, and not through a cooler label's temperature.

Fixtures are synthetic; element symbols are real so ``standard_formula`` can parse them.
"""

import inspect
from types import SimpleNamespace

import pytest

import gliquid.config as config
from gliquid import mpds, solid_coverage

from ._mpds_builders import bare_field_shape as _dome
from ._mpds_builders import bare_line_shape as _line
from ._mpds_builders import compound_hull, mpds_record

AXIS = (0.0, 1100.0)  # threshold at the shipped fraction: 273.15 + 0.05 * 1100 = 328.15 K
COMP_DATA = {"Al": SimpleNamespace(t_fusion=933.0), "Cu": SimpleNamespace(t_fusion=1358.0)}
TEMP_RANGE = [273.15, 1373.15]
FLAT_LIQ = [[0.0, 933.0], [0.5, 1100.0], [1.0, 1358.0]]


def _liquid(points):
    return {
        "nphases": 1,
        "is_solid": True,
        "kind": "boundary",
        "label": "L",
        "svgpath": " ".join(f"{x},{y}" for x, y in points),
    }


def _json(shapes, elements=("Al", "Cu"), temp=AXIS, labels=()):
    return mpds_record(shapes, elements, temp, labels)


def _raw(data):
    return mpds.identify_mpds_phases(data, with_polymorph=True)


def _collapsed(data, **kwargs):
    return mpds.collapse_polymorphs(_raw(data), data, **kwargs)


def _unpaired(data, **kwargs):
    return mpds.mark_unpaired_ht_forms(
        _raw(data), data["chemical_elements"], mpds_json=data, **kwargs
    )


def _hull(elements=("Al", "Cu")):
    return compound_hull(("Al", "Cu"), "Al2Cu", elements=elements)


@pytest.fixture()
def restore_fraction():
    before = config.coverage_low_temp_axis_fraction
    yield
    config.set_coverage_thresholds(low_temp_axis_fraction=before)


class TestFractionKnob:
    def test_shipped_defaults(self):
        assert config.coverage_low_temp_axis_fraction == 0.05
        assert config.coverage_polymorph_pair_tol == 0.04

    def test_the_shipped_threshold(self):
        assert mpds.low_temp_threshold(_json([])) == pytest.approx(328.15)

    def test_threshold_follows_config(self, restore_fraction):
        config.set_coverage_thresholds(low_temp_axis_fraction=0.10)
        assert mpds.low_temp_threshold(_json([])) == pytest.approx(273.15 + 110.0)

    def test_explicit_fraction_wins_over_config(self, restore_fraction):
        config.set_coverage_thresholds(low_temp_axis_fraction=0.10)
        assert mpds.low_temp_threshold(_json([]), 0.02) == pytest.approx(273.15 + 22.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_the_fraction_is_validated(self, bad):
        with pytest.raises(ValueError):
            config.set_coverage_thresholds(low_temp_axis_fraction=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_the_pair_tolerance_is_validated(self, bad):
        with pytest.raises(ValueError):
            config.set_coverage_thresholds(polymorph_pair_tol=bad)

    def test_collapse_reads_the_tolerance_from_config_by_default(self):
        params = inspect.signature(mpds.collapse_polymorphs).parameters
        assert params["comp_tol"].default is None
        data = _json(
            [_line("Cu9Al4 rt", 69.0, 0.0, 700.0), _line("Cu9Al4 ht", 72.45, 700.0, 1030.0)]
        )
        assert len(_collapsed(data)) == 1
        assert len(_collapsed(data, comp_tol=0.02)) == 2

    def test_only_the_pair_tolerance_has_a_per_call_override(self):
        assert config.coverage_settings({"polymorph_pair_tol": 0.01})["polymorph_pair_tol"] == 0.01
        with pytest.raises(ValueError, match="low_temp_axis_fraction"):
            config.coverage_settings({"low_temp_axis_fraction": 0.02})

    def test_the_coverage_report_records_both_knobs(self):
        shipped = solid_coverage.assess_solid_coverage([], [], [0.0, 1.0], [])
        assert (
            shipped.thresholds["low_temp_axis_fraction"] == config.coverage_low_temp_axis_fraction
        )
        assert shipped.thresholds["polymorph_pair_tol"] == config.coverage_polymorph_pair_tol
        overridden = solid_coverage.assess_solid_coverage(
            [], [], [0.0, 1.0], [], thresholds={"polymorph_pair_tol": 0.01}
        )
        assert overridden.thresholds["polymorph_pair_tol"] == 0.01


class TestUnpairedHighTemperatureForms:
    """A ht form with no rt/lt sibling within the pair tolerance is not low-temperature."""

    def test_a_lone_ht_form_is_never_low_temperature(self):
        data = _json([_line("AlCu ht", 50.0, -23.15, 800.0)])
        unpaired = _unpaired(data)
        assert unpaired == {("AlCu", 0.5)}
        (phase,) = _collapsed(data)
        threshold = mpds.low_temp_threshold(data)
        assert mpds.reaches_room_temperature(phase), "fixture must pass the floor on its own"
        assert mpds.is_low_temperature_phase(phase, threshold)
        assert not mpds.is_low_temperature_phase(phase, threshold, unpaired_ht=unpaired)

    def test_an_rt_sibling_within_tolerance_pairs_it(self):
        data = _json([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 51.0, 0.0, 500.0)])
        assert _unpaired(data) == frozenset()
        (merged,) = _collapsed(data)
        assert merged["name"] == "AlCu"
        assert mpds.is_low_temperature_phase(
            merged, mpds.low_temp_threshold(data), unpaired_ht=_unpaired(data)
        )

    def test_an_rt_sibling_beyond_tolerance_leaves_it_unpaired(self):
        data = _json([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 56.0, 0.0, 500.0)])
        unpaired = _unpaired(data)
        assert unpaired == {("AlCu", 0.5)}
        threshold = mpds.low_temp_threshold(data)
        verdicts = {
            round(p["comp"], 3): mpds.is_low_temperature_phase(p, threshold, unpaired_ht=unpaired)
            for p in _collapsed(data)
        }
        assert verdicts == {0.5: False, 0.56: True}

    def test_an_explicit_tolerance_widens_the_pairing(self):
        data = _json([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 56.0, 0.0, 500.0)])
        assert _unpaired(data, pair_tol=0.10) == frozenset()

    def test_an_unsuffixed_compound_is_never_unpaired(self):
        data = _json([_line("Al2Cu", 33.3, 400.0, 800.0), _line("AlCu3", 75.0, 400.0, 800.0)])
        assert _unpaired(data) == frozenset()

    def test_a_labels_only_rt_form_pairs_the_ht_shape(self):
        """C-La class: the rt form exists solely as a label beside the ht dome."""
        data = _json(
            [
                _dome(
                    "LaC2 ht",
                    [(33.28, 1081.65), (34.95, 1500.0), (33.65, 2362.45), (33.28, 1081.65)],
                )
            ],
            elements=("C", "La"),
            temp=(0.0, 4500.0),
            labels=[["LaC<sub>2</sub> rt", [32.5636, 102.2625], 12681]],
        )
        assert _unpaired(data) == frozenset()
        (lac2,) = _collapsed(data)
        assert lac2["name"] == "LaC2"
        assert mpds.is_low_temperature_phase(
            lac2, mpds.low_temp_threshold(data), unpaired_ht=_unpaired(data)
        )

    def test_a_labels_entry_beyond_tolerance_does_not_pair(self):
        """The exclusion is outermost: a cooler label still lowers the merged bottom, and
        the form is still not low-temperature."""
        data = _json(
            [
                _dome(
                    "LaC2 ht",
                    [(33.28, 1081.65), (34.95, 1500.0), (33.65, 2362.45), (33.28, 1081.65)],
                )
            ],
            elements=("C", "La"),
            temp=(0.0, 4500.0),
            labels=[["LaC<sub>2</sub> rt", [40.0, 102.2625], 12681]],
        )
        unpaired = _unpaired(data)
        ((base, comp),) = unpaired
        assert (base, comp) == ("LaC2", pytest.approx(0.3365))
        (lac2,) = _collapsed(data)
        threshold = mpds.low_temp_threshold(data)
        assert lac2["tbounds"][0][1] < threshold, "label recovery still ran"
        assert not mpds.is_low_temperature_phase(lac2, threshold, unpaired_ht=unpaired)

    def test_the_predicate_needs_no_elements(self):
        phase = {
            "type": "lc",
            "name": "AlCu",
            "comp": 0.5,
            "tbounds": [[0.5, 250.0], [0.5, 1000.0]],
        }
        assert mpds.is_low_temperature_phase(phase, 328.15)
        assert not mpds.is_low_temperature_phase(phase, 328.15, unpaired_ht={("AlCu", 0.5)})
        assert mpds.is_low_temperature_phase(phase, 328.15, unpaired_ht={("AlCu", 0.55)})

    @pytest.mark.parametrize("kind", ["mig", "cmp", "eut"])
    def test_only_solid_phase_types_qualify(self, kind):
        phase = {
            "type": kind,
            "name": "AlCu",
            "comp": 0.5,
            "tbounds": [[0.5, 250.0], [0.5, 1000.0]],
        }
        assert not mpds.is_low_temperature_phase(phase, 328.15)

    def test_the_exclusion_survives_a_frame_mirror(self):
        """``identify_invariant_points`` evaluates the rule in the caller's frame."""
        data = _json([_line("Al3Cu2 ht", 40.0, -23.15, 800.0), _line("Al2Cu", 33.3, 0.0, 600.0)])
        for components, liq in (
            (["Al", "Cu"], FLAT_LIQ),
            (["Cu", "Al"], mpds.mirror_liquidus(FLAT_LIQ)),
        ):
            _, low_t, _ = mpds.identify_invariant_points(
                data, components, [list(p) for p in liq], COMP_DATA, TEMP_RANGE
            )
            assert [p["name"] for p in low_t] == ["Al2Cu"], components


class TestCallSitesAgree:
    """The invariant search, the phase tables and the predicate name the same phases."""

    @staticmethod
    def _data():
        return _json(
            [
                _liquid([(0, 660), (33.3, 700), (50, 600), (69, 900), (75, 1000), (100, 1085)]),
                _line("Al2Cu", 33.3, 0.0, 600.0),
                _line("AlCu3 ht", 75.0, 0.0, 1000.0),
                _line("Cu9Al4 rt", 69.0, 0.0, 700.0),
                _line("Cu9Al4 ht", 72.45, 700.0, 1000.0),
                _dome("AlCu ht", [(48, 500), (52, 500), (52, 800), (48, 800), (48, 500)]),
            ],
            labels=[["AlCu rt", [50.0, 20.0], 1]],
        )

    EXPECTED = {"Al2Cu", "Cu9Al4", "AlCu"}

    def test_the_predicate(self):
        data = self._data()
        threshold = mpds.low_temp_threshold(data)
        unpaired = _unpaired(data)
        names = {
            p["name"]
            for p in _collapsed(data)
            if mpds.is_low_temperature_phase(p, threshold, unpaired_ht=unpaired)
        }
        assert names == self.EXPECTED

    def test_identify_invariant_points(self):
        data = self._data()
        liq = [
            [x / 100, t + 273.15]
            for x, t in [(0, 660), (33.3, 700), (50, 600), (69, 900), (75, 1000), (100, 1085)]
        ]
        _, low_t, _ = mpds.identify_invariant_points(data, ["Al", "Cu"], liq, COMP_DATA, TEMP_RANGE)
        assert {p["name"] for p in low_t if "(" not in p["name"]} == self.EXPECTED

    def test_get_low_temp_phase_data(self):
        (congruent, incongruent, _, _), _ = mpds.get_low_temp_phase_data(self._data(), _hull())
        assert set(congruent) | set(incongruent) == self.EXPECTED


class TestAssignDftPairsSpellings:
    """Two spellings of one formula draw on ONE hull entry."""

    @staticmethod
    def _assign(names, elements=("Bi", "Mg"), hull=(0.6,)):
        phases = [
            {"type": "lc", "name": name, "comp": 0.6, "tbounds": [[0.6, 500.0], [0.6, 1000.0]]}
            for name in names
        ]
        assigned = solid_coverage.assign_dft_phases(phases, list(elements), list(hull))
        return {i: m.hull for i, m in assigned.items()}

    def test_integer_spellings_share_the_entry(self):
        assert self._assign(["Mg3Bi2", "Bi2Mg3"]) == {0: 0.6, 1: 0.6}

    def test_a_fractional_spelling_shares_the_entry(self):
        assert self._assign(["Mg3Bi2", "Mg1.5Bi"]) == {0: 0.6, 1: 0.6}

    def test_without_elements_the_spellings_compete(self):
        phases = [
            {"type": "lc", "name": name, "comp": 0.6, "tbounds": [[0.6, 500.0], [0.6, 1000.0]]}
            for name in ["Mg3Bi2", "Bi2Mg3"]
        ]
        assert len(solid_coverage.assign_dft_phases(phases, [], [0.6])) == 1

    def test_two_compounds_keep_their_own_entries(self):
        phases = [
            {
                "type": "lc",
                "name": "DyAl2",
                "comp": 1 / 3,
                "tbounds": [[1 / 3, 500.0], [1 / 3, 1000.0]],
            },
            {
                "type": "lc",
                "name": "AlDy2",
                "comp": 2 / 3,
                "tbounds": [[2 / 3, 500.0], [2 / 3, 1000.0]],
            },
        ]
        got = solid_coverage.assign_dft_phases(phases, ["Al", "Dy"], [1 / 3, 2 / 3])
        assert {i: m.hull for i, m in got.items()} == {
            0: pytest.approx(1 / 3),
            1: pytest.approx(2 / 3),
        }
