"""A phase may bound its neighbour's field only if it has an energy AND meets the liquid.

A missing compound's primary crystallization field is bounded by invariants, which do not
know that another compound lies between. Where one does, the stretch past it is conjugate to
THAT compound and is not the missing phase's to mask.

Two conditions gate which phases are entitled to do the bounding, and both are load-bearing:

* **a real free energy.** ``narrow_field``, ``component_phase`` and
  ``unknown_structure_ss_loaded`` are support VERDICTS -- they say a phase need not be scored,
  not that its energy is known.
* **it meets the liquid.** A solid whose whole field lies below the liquidus is never the
  conjugate of any point on it, so it owns none of it -- even with a hull entry behind it.
"""

import pytest

import gliquid.config as config
import gliquid.solid_coverage as solid_coverage

from ._mpds_builders import lc_phase as lc
from ._mpds_builders import ss_phase as ss

SPAN = [0.0, 1.0]


def flat(temp=1000.0, n=101):
    return [[i / (n - 1), temp] for i in range(n)]


def assess(phases, invariants=(), dft_comps=(), **kw):
    kw.setdefault("liquidus", flat())
    kw.setdefault("elements", ["A", "B"])
    return solid_coverage.assess_solid_coverage(
        list(phases), list(invariants), SPAN, list(dft_comps), kw.pop("ss_models", ()), **kw
    )


class TestWhoMayBound:
    def test_an_energy_backed_phase_that_meets_the_liquid_may(self):
        rec = solid_coverage.PhaseCoverage(
            name="AB",
            kind="lc",
            comp=0.5,
            interval=(0.5, 0.5),
            supported=True,
            reason="dft_phase@0.500",
            borders_liquid=True,
        )
        assert solid_coverage._may_bound(rec) is True

    @pytest.mark.parametrize(
        "reason", ["narrow_field", "component_phase", "unknown_structure_ss_loaded"]
    )
    def test_a_support_verdict_may_not(self, reason):
        """These say the phase need not be scored, not what its energy is."""
        rec = solid_coverage.PhaseCoverage(
            name="(A)",
            kind="ss",
            comp=0.02,
            interval=(0.0, 0.04),
            supported=True,
            reason=reason,
            borders_liquid=True,
        )
        assert solid_coverage._may_bound(rec) is False

    def test_a_sub_liquidus_phase_may_not_even_with_an_energy(self):
        rec = solid_coverage.PhaseCoverage(
            name="AB",
            kind="lc",
            comp=0.5,
            interval=(0.5, 0.5),
            supported=True,
            reason="dft_phase@0.500",
            borders_liquid=False,
        )
        assert solid_coverage._may_bound(rec) is False

    def test_an_unsupported_phase_may_not(self):
        rec = solid_coverage.PhaseCoverage(
            name="AB",
            kind="lc",
            comp=0.5,
            interval=(0.2, 0.8),
            supported=False,
            reason="no_dft_phase:invariant/invariant",
            borders_liquid=True,
        )
        assert solid_coverage._may_bound(rec) is False

    def test_ss_model_support_counts_but_no_ss_model_does_not(self):
        yes = solid_coverage.PhaseCoverage(
            name="(A)",
            kind="ss",
            comp=0.1,
            interval=(0.0, 0.2),
            supported=True,
            reason="ss_model:FCC",
            borders_liquid=True,
        )
        no = solid_coverage.PhaseCoverage(
            name="(A)",
            kind="ss",
            comp=0.1,
            interval=(0.0, 0.2),
            supported=False,
            reason="no_ss_model:FCC",
            borders_liquid=True,
        )
        assert solid_coverage._may_bound(yes) is True
        assert solid_coverage._may_bound(no) is False


class TestTheHomeEnd:
    """Which side of a split field the masking phase keeps."""

    def test_a_congruent_field_spans_its_own_composition(self):
        assert solid_coverage._home_end((0.2, 0.8), 0.5) == pytest.approx(0.5)

    def test_a_peritectic_field_keeps_the_end_nearer_the_stoichiometry(self):
        """The field runs from the invariant AWAY from the compound, so comp is outside it."""
        assert solid_coverage._home_end((0.006, 0.131), 0.76) == pytest.approx(0.131)
        assert solid_coverage._home_end((0.80, 0.99), 0.10) == pytest.approx(0.80)


class TestTheTrim:
    def _case(self, **kw):
        """AB2 is missing and its field runs over AB, which the hull holds."""
        return assess(
            [lc("AB", 0.5), lc("AB2", 0.7)],
            invariants=[{"comp": 0.3, "type": "eut"}, {"comp": 0.9, "type": "eut"}],
            dft_comps=[0.5],
            **kw,
        )

    def test_the_mask_stops_at_the_supported_compound(self):
        rep = self._case()
        assert rep.unsupported_ranges == [[0.5, 0.9]]
        assert any("bounded" in p.reason for p in rep.phases if not p.supported)

    def test_turning_it_off_restores_the_untrimmed_mask(self):
        rep = self._case(thresholds={"bound_by_supported": False})
        assert rep.unsupported_ranges == [[0.3, 0.9]]
        assert not any("bounded" in p.reason for p in rep.phases)

    def test_a_narrow_field_anchor_does_not_trim(self):
        """The Ho-Zr shape: a narrow field sits INSIDE a mask and must not cut it."""
        rep = assess(
            [ss("(A)", 0.30, 0.34, spacegroup=194), lc("AB2", 0.7)],
            invariants=[{"comp": 0.1, "type": "eut"}, {"comp": 0.9, "type": "eut"}],
        )
        narrow = next(p for p in rep.phases if p.name == "(A)")
        assert narrow.supported is True and narrow.reason == "narrow_field"
        # AB2's field spans the narrow phase; an ineligible anchor leaves it whole
        assert [0.1, 0.9] in rep.unsupported_ranges
        assert not any("bounded" in p.reason for p in rep.phases)

    def test_the_same_geometry_with_an_energy_backed_anchor_does_trim(self):
        """Control on the same span: a hull-backed compound in the field DOES cut it."""
        rep = assess(
            [lc("AB", 0.5), lc("AB2", 0.7)],
            invariants=[{"comp": 0.1, "type": "eut"}, {"comp": 0.9, "type": "eut"}],
            dft_comps=[0.5],
        )
        anchor = next(p for p in rep.phases if p.name == "AB")
        assert anchor.supported is True and anchor.borders_liquid is True
        assert rep.unsupported_ranges == [[0.5, 0.9]]
        assert any("bounded" in p.reason for p in rep.phases if not p.supported)

    def test_a_sub_liquidus_supported_compound_does_not_trim(self):
        """The Dy-Mg shape: it has a hull entry but never meets the liquid."""
        cold = lc("AB", 0.5, t_lo=300.0, t_hi=400.0)
        rep = assess(
            [cold, lc("AB2", 0.7)],
            invariants=[{"comp": 0.3, "type": "eut"}, {"comp": 0.9, "type": "eut"}],
            dft_comps=[0.5],
        )
        anchor = next(p for p in rep.phases if p.name == "AB")
        assert anchor.supported is True and anchor.borders_liquid is False
        assert rep.unsupported_ranges == [[0.3, 0.9]]

    def test_the_thresholds_record_what_was_applied(self):
        assert self._case().thresholds["bound_by_supported"] is True


class TestTheShippedDefault:
    def test_bounding_is_on(self):
        assert config.coverage_bound_by_supported is True


class TestTheFieldStopsAtThePhaseCeiling:
    """A phase is the primary solid only where the liquidus is no hotter than it can exist.

    ``_phase_borders_liquid`` asks the all-or-nothing form of this -- does the phase reach the
    liquidus ANYWHERE in its field -- and then lets it mask the whole field. Past the point
    where the liquidus climbs above the phase's own ceiling the conjugate solid is something
    else, and that stretch is not this phase's to mask.
    """

    @staticmethod
    def _rising(t_lo=400.0, t_hi=2400.0, n=201):
        """A liquidus climbing steadily from A to B."""
        return [[i / (n - 1), t_lo + (t_hi - t_lo) * i / (n - 1)] for i in range(n)]

    def test_the_mask_stops_where_the_liquidus_passes_the_ceiling(self):
        liq = self._rising()
        phase = lc(
            "AB", 0.9, t_lo=300.0, t_hi=1400.0
        )  # ceiling 1400 K -> liquidus hits it at x=0.5
        rep = assess([phase], invariants=[{"comp": 0.1, "type": "eut"}], liquidus=liq)
        lo, hi = rep.unsupported_ranges[0]
        assert lo == pytest.approx(0.1, abs=0.02)
        assert hi == pytest.approx(0.5, abs=0.02)
        assert any("ceiling" in p.reason for p in rep.phases)

    def test_a_phase_hotter_than_the_whole_liquidus_is_untrimmed(self):
        liq = self._rising()
        phase = lc("AB", 0.9, t_lo=300.0, t_hi=3000.0)
        rep = assess([phase], invariants=[{"comp": 0.1, "type": "eut"}], liquidus=liq)
        assert not any("ceiling" in p.reason for p in rep.phases)

    def test_a_phase_below_the_whole_liquidus_masks_nothing(self):
        """The degenerate case: an empty trim is the same verdict borders_liquid=False gives."""
        liq = self._rising(t_lo=1500.0, t_hi=2400.0)
        phase = lc("AB", 0.5, t_lo=300.0, t_hi=800.0)
        rep = assess([phase], invariants=[{"comp": 0.1, "type": "eut"}], liquidus=liq)
        assert rep.unsupported_ranges == []
        assert rep.phases[0].borders_liquid is False
        assert rep.phases[0].reason.startswith("subliquidus:")

    def test_the_tolerance_is_its_own_knob(self):
        """Widening it must not also widen the borders-liquid predicate.

        ``subliquidus_margin`` decides whether a phase masks at all; this tolerance decides
        whether it may BOUND its neighbours.
        """
        liq = self._rising()
        phase = lc("AB", 0.9, t_lo=300.0, t_hi=1400.0)
        tight = assess(
            [phase],
            invariants=[{"comp": 0.1, "type": "eut"}],
            liquidus=liq,
            thresholds={"field_ceiling_tol": 0.0},
        )
        loose = assess(
            [phase],
            invariants=[{"comp": 0.1, "type": "eut"}],
            liquidus=liq,
            thresholds={"field_ceiling_tol": 400.0},
        )
        assert loose.unsupported_ranges[0][1] > tight.unsupported_ranges[0][1]

    def test_the_thresholds_record_both(self):
        rep = assess([lc("AB", 0.5)], invariants=[{"comp": 0.1, "type": "eut"}])
        assert rep.thresholds["field_ceiling_tol"] == config.coverage_field_ceiling_tol
        assert rep.thresholds["subliquidus_margin"] == config.coverage_subliquidus_margin


class TestTheShippedTolerances:
    def test_the_ceiling_tolerance(self):
        assert config.coverage_field_ceiling_tol == pytest.approx(10.0)

    def test_the_subliquidus_margin_stays_strict(self):
        """Raising it would admit more phases as bounding anchors, which is a separate call."""
        assert config.coverage_subliquidus_margin == pytest.approx(0.0)
