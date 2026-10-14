"""A diagram that draws no phase at all does not thereby support its liquidus.

``assess_solid_coverage`` masks per drawn phase, so a diagram with nothing digitized reaches
the end of the loop having masked nothing and reads as fully supported. That is true only
vacuously, and it is the difference between two opposite situations:

* the DFT hull holds an interior entry -- a compound IS stable somewhere under the liquidus and
  none of it was digitized, so no stretch has a demonstrated solid reference;
* the hull is empty -- no compound exists, the liquidus is conjugate to terminal solutions
  whose energies come from the unary reference, and full support is the honest answer.

The second case is the control: a simple eutectic that draws no compound stays supported.
"""

import pytest

from gliquid import solid_coverage

SPAN = [0.0, 1.0]
LIQUIDUS = [[x / 20.0, 1000.0 - 200.0 * (x / 20.0) * (1 - x / 20.0)] for x in range(21)]


def _assess(phases, dft_comps):
    return solid_coverage.assess_solid_coverage(
        phases, [], SPAN, dft_comps, (), liquidus=LIQUIDUS, elements=["A", "B"]
    )


class TestNothingDrawn:
    def test_an_interior_hull_entry_with_no_drawn_phase_is_unsupported(self):
        report = _assess([], [0.5])
        assert report.largest_supported_segment == pytest.approx(0.0)
        assert report.unsupported_fraction == pytest.approx(1.0)

    def test_it_is_reported_as_insufficient(self):
        report = _assess([], [0.5])
        report.thresholds["min_supported_total"] = 0.35
        verdict, _reason = report.is_insufficient()
        assert verdict

    @pytest.mark.parametrize("hull", [[0.25], [0.5, 0.75], [0.1, 0.4, 0.9]])
    def test_any_interior_entry_triggers_it(self, hull):
        assert _assess([], hull).largest_supported_segment == pytest.approx(0.0)


class TestTheEmptyHullControl:
    """A system with no compounds must keep its full support."""

    def test_no_phases_and_no_hull_stays_supported(self):
        report = _assess([], [])
        assert report.unsupported_fraction == pytest.approx(0.0)
        assert report.largest_supported_segment == pytest.approx(1.0)

    @pytest.mark.parametrize("endpoints", [[0.0], [1.0], [0.0, 1.0]])
    def test_endpoint_only_entries_are_not_interior(self, endpoints):
        """Hull compositions at the pure elements are not compounds under the liquidus."""
        report = _assess([], endpoints)
        assert report.largest_supported_segment == pytest.approx(1.0)


class TestDrawnPhasesAreUnaffected:
    def test_a_drawn_supported_compound_still_supports(self):
        phase = {
            "name": "AB",
            "type": "lc",
            "comp": 0.5,
            "tbounds": [[0.5, 300.0], [0.5, 900.0]],
            "structure": {},
        }
        report = _assess([phase], [0.5])
        assert report.unsupported_fraction < 1.0
