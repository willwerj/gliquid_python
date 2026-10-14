"""``borders_liquid`` against the liquidus ABOVE the phase, not the diagram's coldest point.

A solid supports a stretch of liquidus only where its own stability field spans the
temperature of that stretch. The reference is therefore the liquidus over the field the
phase would own, and where the digitization drops no point inside that field -- a zero-width
line compound, a narrow drawn field -- the curve is interpolated at the phase's own
composition rather than replaced by the diagram-wide minimum, which any phase clears.

A supported compound that fails the test keeps its support verdict: it has a free energy,
it is simply not the solid in equilibrium with the liquidus above it, so it neither masks
nor bounds a neighbour.
"""

import math

import pytest

import gliquid.config as config
from gliquid import solid_coverage

from ._mpds_builders import lc_phase as lc
from ._mpds_builders import ss_phase as ss

SPAN = [0.0, 1.0]


def sparse_liquidus(points):
    """A liquidus digitized only at the given ``(x, T)`` pairs."""
    return [[float(x), float(t)] for x, t in points]


def assess(phases, invariants=(), dft_comps=(), **kw):
    kw.setdefault("elements", ["A", "B"])
    knobs = {name: kw.pop(name) for name in list(kw) if name in config._COVERAGE_KNOBS}
    thresholds = {"ss_extension_cap": math.inf, **knobs}
    return solid_coverage.assess_solid_coverage(
        list(phases),
        list(invariants),
        SPAN,
        list(dft_comps),
        kw.pop("ss_models", ()),
        thresholds=thresholds,
        **kw,
    )


class TestTheReferenceIsTheCurveAboveThePhase:
    """The Bi-Pr / Co-La shape: a supported compound far below the liquidus at its own x."""

    # A liquidus falling steeply from a hot maximum, digitized coarsely enough that no
    # point lands on the compound at 0.6.
    LIQ = sparse_liquidus([(0.0, 600.0), (0.5, 2100.0), (0.7, 1500.0), (1.0, 1200.0)])

    def test_a_supported_line_compound_below_it_does_not_border_the_liquid(self):
        rep = assess(
            [lc("A2B3", 0.6, t_hi=1500.0)],
            dft_comps=[0.6],
            liquidus=self.LIQ,
            dft_cover_tol=0.02,
        )
        record = rep.phases[0]
        assert record.supported is True
        assert record.borders_liquid is False
        assert record.reason == "subliquidus:dft_phase@0.600"
        # the diagram-wide minimum lies below the compound's ceiling; the curve above it does not
        assert min(t for _x, t in self.LIQ) < 1500.0

    def test_a_compound_that_reaches_the_curve_above_it_still_borders(self):
        rep = assess(
            [lc("A2B3", 0.6, t_hi=1900.0)],
            dft_comps=[0.6],
            liquidus=self.LIQ,
            dft_cover_tol=0.02,
        )
        assert rep.phases[0].borders_liquid is True
        assert rep.phases[0].reason == "dft_phase@0.600"

    def test_it_cannot_bound_a_neighbours_field_once_excluded(self):
        """A phase that never meets the liquid owns none of it, so it trims nothing."""
        phases = [lc("A2B3", 0.6, t_hi=1500.0), lc("AB", 0.55, t_hi=2200.0)]
        invariants = [{"comp": 0.52, "type": "eut"}, {"comp": 0.95, "type": "eut"}]
        rep = assess(
            phases,
            invariants=invariants,
            dft_comps=[0.6],
            liquidus=self.LIQ,
            dft_cover_tol=0.02,
        )
        anchor = next(p for p in rep.phases if p.name == "A2B3")
        assert anchor.supported is True and anchor.borders_liquid is False
        assert rep.unsupported_ranges == [[0.52, 0.95]]

    def test_the_same_geometry_with_a_bordering_anchor_does_bound(self):
        """POSITIVE CONTROL: only the anchor's own temperature differs."""
        phases = [lc("A2B3", 0.6, t_hi=1900.0), lc("AB", 0.55, t_hi=2200.0)]
        invariants = [{"comp": 0.52, "type": "eut"}, {"comp": 0.95, "type": "eut"}]
        rep = assess(
            phases,
            invariants=invariants,
            dft_comps=[0.6],
            liquidus=self.LIQ,
            dft_cover_tol=0.02,
        )
        anchor = next(p for p in rep.phases if p.name == "A2B3")
        assert anchor.borders_liquid is True
        assert rep.unsupported_ranges == [[0.52, 0.6]]


class TestTheDigitizationBandOnAnInterpolatedReference:
    """A congruent compound's top and the liquidus apex are two readings of one point.

    The interpolated reference carries ``field_ceiling_tol`` of slack so a few kelvin of
    digitization error cannot strip a congruently melting compound of its bounding rights;
    the band is finite, so a compound genuinely below the curve is still excluded.
    """

    LIQ = sparse_liquidus([(0.0, 600.0), (0.4, 1990.0), (0.6, 1990.0), (1.0, 600.0)])

    @pytest.mark.parametrize(
        "t_hi,expected", [(1990.0, True), (1989.0, True), (1985.0, True), (1960.0, False)]
    )
    def test_the_band_is_ten_kelvin_wide(self, t_hi, expected):
        rep = assess([lc("AB", 0.5, t_hi=t_hi)], dft_comps=[0.5], liquidus=self.LIQ)
        assert rep.phases[0].borders_liquid is expected

    def test_the_band_is_the_field_ceiling_tolerance(self):
        phase, kw = lc("AB", 0.5, t_hi=1960.0), {"dft_comps": [0.5], "liquidus": self.LIQ}
        assert assess([phase], field_ceiling_tol=0.0, **kw).phases[0].borders_liquid is False
        assert assess([phase], field_ceiling_tol=50.0, **kw).phases[0].borders_liquid is True

    def test_the_subliquidus_margin_still_gates_a_drawn_field(self):
        """The band applies to the interpolated reference only, never to a sampled one."""
        field = ss("(A, B)", 0.3, 0.7, t_hi=1985.0, spacegroup=225)
        kw = {"liquidus": self.LIQ, "ss_models": ("FCC",)}
        assert assess([field], **kw).phases[0].borders_liquid is False
        assert assess([field], subliquidus_margin=25.0, **kw).phases[0].borders_liquid is True


class TestADrawnFieldKeepsTheMinimumOverItsOwnInterval:
    """Where the digitization covers the field, nothing about the reference changed."""

    LIQ = sparse_liquidus([(x / 20.0, 2000.0 - 900.0 * abs(x / 20.0 - 0.5)) for x in range(21)])

    def test_a_field_spanning_a_cold_stretch_is_measured_on_that_stretch(self):
        field = ss("AB", 0.0, 0.3, comp=0.15, t_hi=1700.0, spacegroup=225)
        rep = assess([field], liquidus=self.LIQ, ss_models=("FCC",))
        assert rep.phases[0].borders_liquid is True

    def test_a_field_below_its_own_stretch_is_excluded(self):
        field = ss("AB", 0.4, 0.6, comp=0.5, t_hi=1500.0, spacegroup=225)
        rep = assess([field], liquidus=self.LIQ, ss_models=("FCC",))
        assert rep.phases[0].borders_liquid is False
