"""An unpaired high-temperature form is unsupported and masks its primary field.

A shape whose only form is a high-temperature polymorph has no 0 K ground state the DFT hull
could hold. It is never a low-temperature phase, it never claims a hull entry, it anchors no
melting invariant, and its primary crystallization field is masked. A genuine pair -- a cooler
sibling drawn as a shape or named in the ``labels`` block -- stays one merged, supported phase.

Tested at each layer: the predicate, the low-temperature rule, ``assess_solid_coverage``,
``BinaryLiquid.from_cache`` with its invariants and fit, and the cached corpus. The fixtures
are synthetic except in ``TestTheCachedCorpus``; element symbols are real so
``standard_formula`` can parse them.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest

import gliquid.api as api
from gliquid import mpds, solid_coverage
from gliquid.binary import SKIP_MASK_FRACTION, BinaryLiquid

from ._mpds_builders import bare_field_shape as _dome
from ._mpds_builders import bare_line_shape as _line
from ._mpds_builders import collapsed_phases as _collapsed
from ._mpds_builders import (
    compound_hull,
    eutectic_inv,
    lc_phase,
    melting_inv,
    mpds_record,
    stub_mpds_load,
    svgpath,
)
from ._mpds_builders import corpus_coverage_report as _report
from ._mpds_builders import unpaired_ht_forms as _unpaired

ELEMENTS = ("Al", "Cu")
AXIS = (0.0, 1100.0)
# [at.% Cu, C]: eutectics at 30 and 70, a congruent maximum at 50, 2 at.% sampling.
LIQUIDUS = (
    [[x, 660.0 - 2.0 * x] for x in range(0, 30, 2)]
    + [[x, 600.0 + 10.0 * (x - 30)] for x in range(30, 50, 2)]
    + [[x, 800.0 - 10.0 * (x - 50)] for x in range(50, 70, 2)]
    + [[x, 600.0 + (485.0 / 30.0) * (x - 70)] for x in range(70, 101, 2)]
)
FIELD = (0.3, 0.7)  # the unpaired compound's primary field, eutectic to eutectic
COMP_DATA = {"Al": SimpleNamespace(t_fusion=933.0), "Cu": SimpleNamespace(t_fusion=1358.0)}
TEMP_RANGE = [273.15, 1373.15]
FLAT_LIQ = [[0.0, 933.0], [0.5, 1100.0], [1.0, 1358.0]]


def _json(shapes, labels=()):
    liquid = {"label": "L", "nphases": 1, "svgpath": svgpath(LIQUIDUS)}
    return mpds_record([liquid, *shapes], ELEMENTS, AXIS, labels)


UNPAIRED = _json([_line("AlCu ht", 50.0, 0.0, 800.0)])
PAIRED_BY_SHAPE = _json([_line("AlCu ht", 50.0, 400.0, 800.0), _line("AlCu rt", 51.0, 0.0, 400.0)])
PAIRED_BY_LABEL = _json(
    [_line("AlCu ht", 50.0, 400.0, 800.0)], labels=[["AlCu rt", [50.0, 20.0], 1]]
)
SIBLING_BEYOND_TOL = _json(
    [_line("AlCu ht", 50.0, 400.0, 800.0), _line("AlCu rt", 56.0, 0.0, 400.0)]
)


def _record(shapes, elements=ELEMENTS, temp=AXIS, labels=()):
    """A synthetic MPDS record holding only ``shapes`` and ``labels``."""
    return mpds_record(shapes, elements, temp, labels)


def _hull():
    return compound_hull(("Al", "Cu"), "AlCu", elements=ELEMENTS)


_lc = partial(lc_phase, t_lo=500.0, t_hi=1500.0)


def _assess(phases, invariants=(), dft_comps=(), **kwargs):
    """``assess_solid_coverage`` on [0, 1] with the field-ceiling narrowing and the extension
    cap off."""
    return solid_coverage.assess_solid_coverage(
        list(phases),
        list(invariants),
        [0.0, 1.0],
        list(dft_comps),
        (),
        thresholds={"field_ceiling_tol": math.inf, "ss_extension_cap": math.inf},
        **kwargs,
    )


@pytest.fixture()
def system(monkeypatch):
    """``BinaryLiquid.from_cache`` over a synthetic json and hull; everything else is real."""
    hull = _hull()

    def build(data, **kwargs):
        stub_mpds_load(monkeypatch, data)
        monkeypatch.setattr(api, "get_dft_convexhull", lambda system, *args, **kw: (hull, {}))
        return BinaryLiquid.from_cache("Al-Cu", param_format="comb-exp", pd_ind=0, **kwargs)

    return build


def _coverage(bl, **overrides):
    bl.find_invariant_points(verbose=False)
    return bl.assess_solid_coverage(**overrides)


class TestThePredicate:
    def test_matches_on_base_label_and_composition(self):
        keys = frozenset({("AlCu", 0.5)})
        assert mpds.is_unpaired_ht_form({"name": "AlCu ht", "comp": 0.5}, keys)
        assert mpds.is_unpaired_ht_form({"name": "AlCu", "comp": 0.5}, keys)
        assert not mpds.is_unpaired_ht_form({"name": "AlCu", "comp": 0.55}, keys)
        assert not mpds.is_unpaired_ht_form({"name": "Al2Cu", "comp": 0.5}, keys)

    def test_an_empty_set_or_a_missing_composition_never_matches(self):
        assert not mpds.is_unpaired_ht_form({"name": "AlCu", "comp": 0.5}, frozenset())
        assert not mpds.is_unpaired_ht_form({"name": "AlCu"}, frozenset({("AlCu", 0.5)}))


class TestTheLowTemperatureRule:
    """A ht form with no rt/lt sibling within the pair tolerance is not low-temperature."""

    def test_a_lone_ht_form_is_never_low_temperature(self):
        data = _record([_line("AlCu ht", 50.0, -23.15, 800.0)])
        unpaired = _unpaired(data)
        assert unpaired == {("AlCu", 0.5)}
        (phase,) = _collapsed(data)
        threshold = mpds.low_temp_threshold(data)
        assert mpds.reaches_room_temperature(phase), "fixture must pass the floor on its own"
        assert mpds.is_low_temperature_phase(phase, threshold)
        assert not mpds.is_low_temperature_phase(phase, threshold, unpaired_ht=unpaired)

    def test_an_rt_sibling_within_tolerance_pairs_it(self):
        data = _record([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 51.0, 0.0, 500.0)])
        assert _unpaired(data) == frozenset()
        (merged,) = _collapsed(data)
        assert merged["name"] == "AlCu"
        assert mpds.is_low_temperature_phase(
            merged, mpds.low_temp_threshold(data), unpaired_ht=_unpaired(data)
        )

    def test_an_rt_sibling_beyond_tolerance_leaves_it_unpaired(self):
        data = _record([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 56.0, 0.0, 500.0)])
        unpaired = _unpaired(data)
        assert unpaired == {("AlCu", 0.5)}
        threshold = mpds.low_temp_threshold(data)
        verdicts = {
            round(p["comp"], 3): mpds.is_low_temperature_phase(p, threshold, unpaired_ht=unpaired)
            for p in _collapsed(data)
        }
        assert verdicts == {0.5: False, 0.56: True}

    def test_an_explicit_tolerance_widens_the_pairing(self):
        data = _record([_line("AlCu ht", 50.0, -23.15, 800.0), _line("AlCu rt", 56.0, 0.0, 500.0)])
        assert _unpaired(data, pair_tol=0.10) == frozenset()

    def test_an_unsuffixed_compound_is_never_unpaired(self):
        data = _record([_line("Al2Cu", 33.3, 400.0, 800.0), _line("AlCu3", 75.0, 400.0, 800.0)])
        assert _unpaired(data) == frozenset()

    def test_a_labels_only_rt_form_pairs_the_ht_shape(self):
        """C-La class: the rt form exists solely as a label beside the ht dome."""
        data = _record(
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
        data = _record(
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
        data = _record([_line("Al3Cu2 ht", 40.0, -23.15, 800.0), _line("Al2Cu", 33.3, 0.0, 600.0)])
        for components, liq in (
            (["Al", "Cu"], FLAT_LIQ),
            (["Cu", "Al"], mpds.mirror_liquidus(FLAT_LIQ)),
        ):
            _, low_t, _ = mpds.identify_invariant_points(
                data, components, [list(p) for p in liq], COMP_DATA, TEMP_RANGE
            )
            assert [p["name"] for p in low_t] == ["Al2Cu"], components


class TestAssessSolidCoverage:
    """A shape listed in ``unpaired_ht`` never claims the hull and masks its primary field."""

    UNPAIRED = frozenset({("A2B", 0.40)})

    def test_it_is_unsupported_over_its_primary_field(self):
        rep = _assess(
            [_lc("A2B", 0.40)],
            [eutectic_inv(0.20), eutectic_inv(0.60)],
            dft_comps=[0.40],
            unpaired_ht=self.UNPAIRED,
        )
        (rec,) = rep.phases
        assert rec.supported is False
        assert rec.reason == "unpaired_ht_polymorph:invariant/invariant"
        assert rec.interval == (0.20, 0.60)
        assert rep.unsupported_ranges == [[0.20, 0.60]]
        assert (rep.n_compounds, rep.n_missing_compounds) == (0, 0)

    def test_a_peritectic_form_is_relocated_like_a_missing_compound(self):
        rep = _assess(
            [_lc("A2B", 0.40)],
            [eutectic_inv(0.10), melting_inv("per", 0.30, "A2B", 0.40)],
            dft_comps=[0.40],
            unpaired_ht=self.UNPAIRED,
        )
        (rec,) = rep.phases
        assert rec.interval == (0.10, 0.30)
        assert rec.reason == "unpaired_ht_polymorph:invariant/peritectic"

    def test_the_hull_entry_goes_to_the_nearest_other_claimant(self):
        phases = [_lc("A2B", 0.40), _lc("AB", 0.405)]
        assigned = solid_coverage.assign_dft_phases(phases, [], [0.40], unpaired_ht=self.UNPAIRED)
        assert {i: m.hull for i, m in assigned.items()} == {1: 0.40}
        rep = _assess(phases, dft_comps=[0.40], unpaired_ht=self.UNPAIRED)
        assert [(p.name, p.supported) for p in rep.phases] == [("A2B", False), ("AB", True)]
        assert rep.phases[1].reason == "dft_phase@0.400"

    def test_a_paired_shape_is_scored_on_the_ordinary_rules(self):
        phases = [_lc("A2B", 0.40)]
        assert _assess(phases, dft_comps=[0.40]) == _assess(
            phases, dft_comps=[0.40], unpaired_ht=frozenset()
        )
        assert _assess(phases, dft_comps=[0.40]).phases[0].reason == "dft_phase@0.400"


class TestCoverage:
    def test_the_form_is_unsupported_and_holds_no_hull_entry(self, system):
        bl = system(UNPAIRED)
        cov = _coverage(bl)
        (record,) = cov.phases
        assert record.name == "AlCu"
        assert record.supported is False
        assert record.borders_liquid is True
        assert record.reason == "unpaired_ht_polymorph:invariant/invariant"
        assert record.interval == pytest.approx(FIELD)
        assert cov.unsupported_ranges == [pytest.approx(list(FIELD))]
        assert (cov.n_compounds, cov.n_missing_compounds) == (0, 0)

    def test_a_sibling_shape_within_tolerance_keeps_it_supported(self, system):
        bl = system(PAIRED_BY_SHAPE)
        cov = _coverage(bl)
        assert [p.name for p in cov.phases] == ["AlCu", "AlCu"]
        assert all(p.supported for p in cov.phases)
        hot = next(p for p in cov.phases if p.comp == pytest.approx(0.50))
        cool = next(p for p in cov.phases if p.comp == pytest.approx(0.51))
        assert hot.reason == "dft_phase@0.500"
        # Only the hotter form meets the liquid: the cooler one's field ends far below it.
        assert cool.reason == "subliquidus:dft_phase@0.500"
        assert cov.unsupported_ranges == []
        assert cov.n_missing_compounds == 0

    def test_a_sibling_in_the_labels_block_pairs_it(self, system):
        bl = system(PAIRED_BY_LABEL)
        cov = _coverage(bl)
        (record,) = cov.phases
        assert record.supported is True
        assert record.reason == "dft_phase@0.500"
        assert cov.unsupported_ranges == []

    def test_the_pair_tolerance_override_reaches_the_rule(self, system):
        bl = system(SIBLING_BEYOND_TOL)
        default = _coverage(bl)
        widened = _coverage(bl, thresholds={"polymorph_pair_tol": 0.10})
        assert [p.reason for p in default.phases if "unpaired_ht_polymorph" in p.reason]
        assert default.unsupported_ranges == [pytest.approx(list(FIELD))]
        assert not [p for p in widened.phases if "unpaired_ht_polymorph" in p.reason]
        assert widened.unsupported_ranges == []


class TestInvariants:
    def test_no_melting_record_is_built_on_the_unpaired_shape(self, system):
        invariants, low_t = system(UNPAIRED).find_invariant_points(verbose=False)
        assert low_t == []
        assert [iv["type"] for iv in invariants] == ["eut", "eut"]
        assert not any("AlCu" in (iv.get("phases") or []) for iv in invariants)

    def test_the_paired_control_melts_congruently(self, system):
        invariants, low_t = system(PAIRED_BY_SHAPE).find_invariant_points(verbose=False)
        assert [p["name"] for p in low_t] == ["AlCu"]
        assert [iv["phases"] for iv in invariants if iv["type"] == "cmp"] == [["AlCu"]]


class TestFitParameters:
    def test_the_field_is_masked(self, system):
        bl = system(UNPAIRED)
        bl.fit_parameters(n_opts=1, max_iter=2, verbose=False)
        assert bl.init_error is False
        assert bl.skip_reason is None
        assert bl.skip_reason != SKIP_MASK_FRACTION
        assert bl.unsupported_comp_ranges == [pytest.approx(list(FIELD))]
        assert bl.coverage_report.is_insufficient()[0] is False

    def test_the_paired_control_masks_nothing(self, system):
        bl = system(PAIRED_BY_SHAPE)
        bl.fit_parameters(n_opts=1, max_iter=2, verbose=False)
        assert bl.init_error is False
        assert bl.unsupported_comp_ranges == []


class TestTheCachedCorpus:
    """A form with no cooler sibling holds no hull entry and masks its primary field.

    Each entry is (system, the unpaired form, the hull entry it fails to claim). Au-In's entry
    at 0.400 is ``Au3In2``'s by formula, but the only drawn ``Au3In2`` is the ht form, so no
    phase claims the entry.
    """

    CASES = [("Au-In", "Au3In2 ht", 0.400)]

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system,form,hull", CASES)
    def test_the_form_is_unsupported_and_claims_no_entry(self, system, form, hull, matrix_data_env):
        cov = _report(system)
        record = next(p for p in cov.phases if p.name == form)
        assert record.supported is False
        assert "unpaired_ht_polymorph:" in record.reason
        assert "dft_phase@" not in record.reason
        assert not any((p.reason or "").endswith(f"dft_phase@{hull:.3f}") for p in cov.phases)
        assert cov.is_insufficient()[0] is False

    @pytest.mark.needs_cache
    def test_a_paired_ht_form_shares_its_rt_forms_entry(self, matrix_data_env):
        cov = _report("B-Mo")
        by_name = {p.name: p for p in cov.phases}
        for name in ("MoB2 rt", "MoB2 ht"):
            assert by_name[name].supported is True
            assert by_name[name].reason.endswith("dft_phase@0.333")
