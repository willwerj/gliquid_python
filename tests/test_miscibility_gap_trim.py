"""A two-liquid dome is not evidence about a missing solid, so it leaves the mask.

``identify_invariant_points`` records a miscibility gap as a ``mig`` invariant whose ``comp``
is the dome APEX. ``bounding_invariant_comps`` keeps that value out of the compositions at which an
invariant may bound a neighbouring field, so an invariant-bounded field stops at a FOOT. A mask
reaching a dome from any other source -- a liquidus minimum, an anchor endpoint, a solution
field's ceiling -- still runs across it, and the trim removes the dome from that mask.

The trim is gated on the feet: the invariant matcher accepts any two-phase shape whose top is
near a liquidus maximum, so an ordinary ``L + solid`` field can be recorded as ``mig``. A
genuine dome meets the liquidus at BOTH feet on the monotectic isotherm, and that is what is
checked.
"""

from functools import partial

import pytest

import gliquid.config as config
import gliquid.mpds as mpds
import gliquid.solid_coverage as solid_coverage

from ._mpds_builders import lc_phase as _lc_phase

SPAN = [0.0, 1.0]


lc_phase = partial(_lc_phase, t_lo=500.0, t_hi=1500.0)


def flat_liquidus(temp=1000.0, n=101):
    """A liquidus at one temperature, so a foot's deviation is exactly temp - foot_T."""
    return [[i / (n - 1), temp] for i in range(n)]


def mig(lo, hi, temp, apex=None):
    return {
        "type": "mig",
        "comp": apex if apex is not None else 0.5 * (lo + hi),
        "cbounds": [[lo, temp], [hi, temp]],
        "tbounds": [[lo, temp], [0.5 * (lo + hi), temp + 200.0]],
        "phases": ["L1", "L2"],
        "phase_comps": [lo, hi],
    }


def assess(phases, invariants=(), liquidus=None, **kw):
    kw.setdefault("liquidus", flat_liquidus() if liquidus is None else liquidus)
    return solid_coverage.assess_solid_coverage(list(phases), list(invariants), SPAN, [], (), **kw)


class TestSubtractRanges:
    def test_a_cut_splits_a_range(self):
        assert mpds.subtract_ranges([[0.0, 1.0]], [[0.4, 0.6]]) == [[0.0, 0.4], [0.6, 1.0]]

    def test_no_cuts_is_a_merge(self):
        assert mpds.subtract_ranges([[0.0, 0.4], [0.3, 0.7]], []) == [[0.0, 0.7]]

    def test_a_cut_covering_everything_leaves_nothing(self):
        assert mpds.subtract_ranges([[0.2, 0.5]], [[0.0, 1.0]]) == []

    def test_a_disjoint_cut_changes_nothing(self):
        assert mpds.subtract_ranges([[0.0, 0.2]], [[0.5, 0.9]]) == [[0.0, 0.2]]


class TestTheFootGate:
    def test_a_dome_whose_feet_sit_on_the_liquidus_is_genuine(self):
        iv = mig(0.3, 0.8, 1000.0)
        assert mpds.mig_feet_on_liquidus(iv, flat_liquidus(1000.0), 20.0) is True

    def test_a_foot_off_the_liquidus_is_refused(self):
        """An `L + solid` field recorded as a gap: its foot is nowhere near the isotherm."""
        iv = mig(0.3, 0.8, 900.0)
        assert mpds.mig_feet_on_liquidus(iv, flat_liquidus(1000.0), 20.0) is False

    def test_the_tolerance_is_the_knob(self):
        iv = mig(0.3, 0.8, 985.0)
        assert mpds.mig_feet_on_liquidus(iv, flat_liquidus(1000.0), 20.0) is True
        assert mpds.mig_feet_on_liquidus(iv, flat_liquidus(1000.0), 5.0) is False

    def test_a_non_mig_record_is_never_genuine(self):
        assert (
            mpds.mig_feet_on_liquidus({"type": "eut", "comp": 0.5}, flat_liquidus(), 20.0) is False
        )

    def test_no_liquidus_means_nothing_can_be_confirmed(self):
        assert mpds.mig_feet_on_liquidus(mig(0.3, 0.8, 1000.0), [], 20.0) is False

    def test_only_confirmed_feet_are_returned(self):
        invs = [mig(0.1, 0.3, 1000.0), mig(0.6, 0.9, 500.0)]
        got = mpds.validated_mig_feet(invs, flat_liquidus(1000.0), 20.0)
        assert got == [(0.1, 0.3)]


class TestAnchoredDomes:
    """A dome whose feet both rest on supported liquidus leaves the mask and IS support."""

    def _case(self, **kw):
        """A missing compound whose whole field lies INSIDE the dome."""
        return assess(
            [lc_phase("AB", 0.6)],
            invariants=[
                {"comp": 0.5, "type": "eut"},
                {"comp": 0.7, "type": "eut"},
                mig(0.3, 0.9, 1000.0),
            ],
            **kw,
        )

    def test_the_mask_inside_the_dome_is_removed(self):
        rep = self._case()
        assert rep.unsupported_ranges == []
        assert rep.gap_ranges == []
        assert rep.supported_total == pytest.approx(1.0)
        assert any("mig_trimmed" in p.reason for p in rep.phases)

    def test_keeping_gaps_restores_the_mask(self):
        rep = self._case(thresholds={"mask_miscibility_gaps": True})
        assert rep.unsupported_ranges == [[0.5, 0.7]]
        assert rep.supported_total == pytest.approx(0.8)


class TestFloatingDomes:
    """A dome with a foot on an unknown solid is neither masked nor supported."""

    def _case(self, **kw):
        """The compound's field runs up to the dome, so the near foot sits on the mask edge."""
        return assess(
            [lc_phase("AB", 0.2)],
            invariants=[{"comp": 0.05, "type": "eut"}, mig(0.4, 0.9, 1000.0)],
            **kw,
        )

    def test_the_dome_becomes_its_own_category(self):
        rep = self._case()
        assert rep.gap_ranges == [[0.4, 0.9]]
        assert rep.gap_total == pytest.approx(0.5)

    def test_it_is_not_counted_as_support(self):
        rep = self._case()
        assert rep.supported_total == pytest.approx(0.15)
        assert rep.supported_segments == [(0.0, 0.05), (0.9, 1.0)]

    def test_the_untrimmed_mask_stops_at_the_near_foot(self):
        """Keeping gaps leaves the mask in place, and it still ends at the foot, not the apex."""
        rep = self._case(thresholds={"mask_miscibility_gaps": True})
        assert rep.unsupported_ranges == [[0.05, 0.4]]
        assert rep.gap_ranges == []

    def test_a_refused_dome_is_neither_trimmed_nor_set_aside(self):
        """Feet 100 K off the liquidus: an `L + solid` field, not a gap."""
        rep = assess(
            [lc_phase("AB", 0.2)],
            invariants=[{"comp": 0.05, "type": "eut"}, mig(0.4, 0.9, 900.0)],
        )
        assert rep.unsupported_ranges == [[0.05, 0.4]]
        assert rep.gap_ranges == []
        assert not any("mig_trimmed" in p.reason for p in rep.phases)

    def test_the_thresholds_record_what_was_applied(self):
        rep = self._case()
        assert rep.thresholds["mask_miscibility_gaps"] is False
        assert rep.thresholds["mig_foot_tol"] == config.coverage_mig_foot_tol


class TestTheShippedDefaults:
    def test_gaps_are_trimmed_by_default(self):
        assert config.coverage_mask_miscibility_gaps is False

    def test_the_foot_tolerance_ships_at_twenty_kelvin(self):
        assert config.coverage_mig_foot_tol == pytest.approx(20.0)


class TestADomeCountsTowardTheGate:
    """A confirmed dome constrains the liquid, so it does not count against fittability.

    :attr:`supported_segments` cuts ``gap_ranges`` out alongside the mask, so a wide dome
    drives ``supported_total`` down and can push a system under ``min_supported_total``. The
    fitter meanwhile builds a curvature equation at the apex and a slope equation across the
    feet from that same record, so the gate reads ``constrained_total`` instead.
    """

    def _case(self, **kw):
        """The compound's field runs into the dome, so the dome is set aside as its own."""
        return assess(
            [lc_phase("AB", 0.2)],
            invariants=[{"comp": 0.05, "type": "eut"}, mig(0.4, 0.9, 1000.0)],
            **kw,
        )

    def test_a_dome_carries_a_short_supported_run_over_the_floor(self):
        rep = self._case(thresholds={"min_supported_total": 0.30})
        assert rep.gap_ranges == [[0.4, 0.9]]
        assert rep.supported_total == pytest.approx(0.15)
        assert rep.constrained_total == pytest.approx(0.65)
        assert rep.supported_total < 0.30
        assert rep.is_insufficient()[0] is False

    def test_counting_the_dome_is_still_bounded(self):
        """POSITIVE CONTROL: the dome counts, and counting it is still not always enough."""
        rep = self._case(thresholds={"min_supported_total": 0.70})
        assert rep.constrained_total == pytest.approx(0.65)
        assert rep.is_insufficient()[0] is True

    def test_a_dome_whose_feet_miss_the_liquidus_does_not_count(self):
        """POSITIVE CONTROL: an unvalidated record reaches neither gap_ranges nor the gate."""
        rep = assess(
            [lc_phase("AB", 0.2)],
            invariants=[{"comp": 0.05, "type": "eut"}, mig(0.4, 0.9, 900.0)],
        )
        assert rep.gap_ranges == []
        assert rep.constrained_total == pytest.approx(rep.supported_total)

    def test_a_dome_over_undigitized_liquidus_does_not_count(self):
        """The total is a UNION, not a sum: a dome over an ignored stretch adds nothing."""
        rep = self._case(ignored_ranges=[[0.4, 0.9]])
        assert rep.gap_total == pytest.approx(0.5)
        assert rep.constrained_total == pytest.approx(rep.supported_total)
        assert rep.constrained_total < rep.supported_total + rep.gap_total

    def test_the_reason_names_both_parts(self):
        reason = self._case(thresholds={"min_supported_total": 0.70}).is_insufficient()[1]
        assert "constraining liquidus totals" in reason
        assert "dome" in reason


class TestAnApexIsNotAFieldBoundary:
    """A ``mig`` contributes its feet, not its apex, where an invariant bounds a field.

    A dome's apex is a temperature maximum in the middle of the gap, so bounding a
    neighbouring missing compound's primary field there starts the mask inside the dome, on
    liquidus whose conjugate is a second liquid rather than the absent solid.
    """

    DOME = mig(0.10, 0.40, 1000.0)  # apex 0.25, feet 0.10 and 0.40
    RIGHT_EUT = {"type": "eut", "comp": 0.80}

    def test_the_feet_replace_the_apex(self):
        assert self.DOME["comp"] == pytest.approx(0.25)
        assert mpds.bounding_invariant_comps([self.DOME, self.RIGHT_EUT]) == pytest.approx(
            [0.10, 0.40, 0.80]
        )

    def test_every_other_type_contributes_its_comp(self):
        invariants = [{"type": "eut", "comp": 0.2}, {"type": "per", "comp": 0.7}]
        assert mpds.bounding_invariant_comps(invariants) == pytest.approx([0.2, 0.7])

    def test_a_record_with_no_feet_contributes_nothing(self):
        """POSITIVE CONTROL: the apex is dropped whether or not feet arrive to replace it."""
        assert mpds.bounding_invariant_comps([{"type": "mig", "comp": 0.25}]) == []

    def test_the_comps_are_ordered_and_deduplicated(self):
        both = [mig(0.40, 0.10, 1000.0), {"type": "eut", "comp": 0.40}]
        assert mpds.bounding_invariant_comps(both) == pytest.approx([0.10, 0.40])

    def test_no_invariants_is_empty(self):
        assert mpds.bounding_invariant_comps([]) == []
        assert mpds.bounding_invariant_comps(None) == []

    def test_a_field_right_of_a_dome_stops_at_the_near_foot(self):
        rep = assess([lc_phase("AB", 0.60)], [self.DOME, self.RIGHT_EUT])
        assert rep.unsupported_ranges == [pytest.approx([0.40, 0.80])]

    def test_a_field_left_of_a_dome_stops_at_the_near_foot(self):
        invariants = [{"type": "eut", "comp": 0.02}, mig(0.60, 0.90, 1000.0)]
        rep = assess([lc_phase("AB", 0.30)], invariants)
        assert rep.unsupported_ranges == [pytest.approx([0.02, 0.60])]

    def test_a_field_inside_a_dome_spans_it_foot_to_foot(self):
        """A compound crystallizing from two liquids is bounded by the monotectic feet."""
        rep = assess([lc_phase("AB", 0.30)], [self.DOME, self.RIGHT_EUT])
        assert rep.unsupported_ranges == [pytest.approx([0.10, 0.40])]
        assert rep.gap_ranges == [pytest.approx([0.10, 0.40])]
        assert "mig_trimmed" in rep.phases[0].reason

    def test_an_unconfirmed_dome_still_bounds_at_its_feet(self):
        """The foot gate decides whether a dome leaves the mask, not where a field ends.

        ``bounding_invariant_comps`` reads ``mig_feet``, which does not check the isotherm, so a
        record the gate refuses still keeps its apex out of the bounding compositions.
        """
        dome = mig(0.10, 0.40, 880.0)
        assert mpds.mig_feet_on_liquidus(dome, flat_liquidus(1000.0), 20.0) is False
        assert mpds.bounding_invariant_comps([dome, self.RIGHT_EUT]) == pytest.approx(
            [0.10, 0.40, 0.80]
        )
        rep = assess([lc_phase("AB", 0.60)], [dome, self.RIGHT_EUT])
        assert rep.unsupported_ranges == [pytest.approx([0.40, 0.80])]
        assert rep.gap_ranges == []


class TestAConfirmedDomeStaysWithTheTrim:
    """A dome the liquidus confirms is the trim's to settle, not the printed-gap refusal's.

    The trim weighs each foot, clearing an anchored dome from the mask and recording a
    floating one in ``gap_ranges``. Na-Si's masked field coincides with its dome, and its
    label prints well clear of both.
    """

    def _case(self, foot_temp, **kw):
        """A compound whose field IS the dome, with the label printed away from it."""
        return assess(
            [lc_phase("AB", 0.6)],
            invariants=[mig(0.5, 0.7, foot_temp)],
            printed_gap_comps=[0.2],
            **kw,
        )

    def test_a_confirmed_dome_is_left_to_the_trim(self):
        rep = self._case(1000.0)
        assert mpds.mig_feet_on_liquidus(mig(0.5, 0.7, 1000.0), flat_liquidus(), 20.0) is True
        assert "printed_gap" not in rep.phases[0].reason
        assert rep.gap_ranges == [[0.5, 0.7]]

    def test_a_dome_the_liquidus_does_not_confirm_falls_to_the_refusal(self):
        """POSITIVE CONTROL: unconfirmed, so its extent is the refusal's to use."""
        rep = self._case(1400.0)
        assert mpds.mig_feet_on_liquidus(mig(0.5, 0.7, 1400.0), flat_liquidus(), 20.0) is False
        assert rep.phases[0].reason.endswith("printed_gap")
        assert rep.gap_ranges == []

    def test_the_label_alone_would_not_have_refused_either_one(self):
        """POSITIVE CONTROL: the label sits outside the field, so only an extent can refuse."""
        assert not 0.5 <= 0.2 <= 0.7
        rep = assess([lc_phase("AB", 0.6)], invariants=[mig(0.5, 0.7, 1400.0)])
        assert "printed_gap" not in rep.phases[0].reason
