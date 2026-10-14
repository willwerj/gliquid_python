"""A congruent melting point at the end of the digitized curve.

``find_local_maxima`` needs a rise AND a fall, so it never returns the first or last point.
Where the digitized liquidus simply BEGINS at a compound's melting point the maximum is real —
the data starts there, the diagram does not — and missing it sends the phase to peritectic
recovery, which projects it onto whatever other composition shares its ceiling temperature.

Two boundaries make the rule safe, and both are load-bearing:

* **a pure end is not a compound.** The maximum at x=0 or x=1 is that element melting.
* **a terminal solid solution is not a compound either.** A liquidus stopping short of the axis
  puts its endpoint within matching distance of ``(X)``/``(X) rt``, which the assignment loop
  would otherwise mark congruently melting.

Endpoint maxima reach the congruent-melting assignment only. The miscibility-gap search reads
the interior maxima, where a dome apex actually lives.
"""

import pytest

from gliquid.mpds import _AXIS_X_TOL, _endpoint_maxima, _is_terminal_phase, find_local_maxima


def _curve(*pairs):
    return [[float(x), float(t)] for x, t in pairs]


class TestWhichEndsQualify:
    def test_a_descending_start_is_a_maximum(self):
        """Mn-Th: the curve begins at the compound's melting point and falls away."""
        got = _endpoint_maxima(_curve((0.208, 1523), (0.4, 1300), (0.8, 1184)))
        assert [p[0] for p in got] == [0.208]

    def test_an_ascending_end_is_a_maximum(self):
        got = _endpoint_maxima(_curve((0.2, 1184), (0.5, 1300), (0.79, 1523)))
        assert [p[0] for p in got] == [0.79]

    def test_both_ends_can_qualify(self):
        got = _endpoint_maxima(_curve((0.2, 1500), (0.5, 1100), (0.8, 1450)))
        assert [p[0] for p in got] == [0.2, 0.8]

    def test_a_rising_start_is_not_a_maximum(self):
        """The control: an end the curve climbs away from is not a peak."""
        assert _endpoint_maxima(_curve((0.2, 1100), (0.5, 1300), (0.8, 1500))) == [[0.8, 1500.0]]

    def test_a_flat_start_is_not_a_maximum(self):
        assert _endpoint_maxima(_curve((0.2, 1500), (0.3, 1500), (0.8, 1100))) == []

    @pytest.mark.parametrize("x", [0.0, _AXIS_X_TOL / 2, 1.0, 1.0 - _AXIS_X_TOL / 2])
    def test_a_pure_end_never_qualifies(self, x):
        """At x=0 or x=1 the maximum is that element melting, not a compound."""
        other = 0.5
        curve = _curve((x, 1800), (other, 1200)) if x < 0.5 else _curve((other, 1200), (x, 1800))
        assert _endpoint_maxima(curve) == []

    def test_too_few_points_is_not_an_error(self):
        assert _endpoint_maxima([]) == []
        assert _endpoint_maxima(_curve((0.5, 1200))) == []

    def test_it_never_returns_what_the_interior_search_already_found(self):
        """Interior and endpoint maxima are disjoint by construction, so a phase cannot be
        assigned twice from the same peak."""
        curve = _curve((0.1, 1500), (0.3, 1200), (0.5, 1400), (0.7, 1200), (0.9, 1500))
        interior = {p[0] for p in find_local_maxima(curve)}
        ends = {p[0] for p in _endpoint_maxima(curve)}
        assert interior == {0.5}
        assert ends == {0.1, 0.9}
        assert not (interior & ends)


class TestTerminalPhasesAreRefused:
    """The 0.97 case: a liquidus stopping short of the axis must not christen ``(X)``."""

    @pytest.mark.parametrize(
        "phase",
        [
            {"name": "(Th)", "type": "ss", "comp": 0.97},
            {"name": "(Th) rt", "type": "ss", "comp": 0.97},
            {"name": "(Sc) ht", "type": "ss", "comp": 0.03},
            {"name": "Th", "type": "comp", "comp": 0.97},
            {"name": "Whatever", "type": "lc", "comp": 1.0 - _AXIS_X_TOL / 2},
            {"name": "Whatever", "type": "lc", "comp": _AXIS_X_TOL / 2},
        ],
    )
    def test_a_terminal_phase_is_recognised(self, phase):
        assert _is_terminal_phase(phase) is True

    @pytest.mark.parametrize(
        "phase",
        [
            {"name": "Th6Mn23", "type": "lc", "comp": 0.208},
            {"name": "ThMn2", "type": "lc", "comp": 0.3334},
            {"name": "Cu9Lu2", "type": "lc", "comp": 0.95},
        ],
    )
    def test_a_real_compound_is_not(self, phase):
        """The control: a guard that refused everything would satisfy the cases above, and
        would also refuse the compound near the edge that this whole rule exists to catch."""
        assert _is_terminal_phase(phase) is False

    def test_a_phase_with_no_composition_is_not_terminal(self):
        assert _is_terminal_phase({"name": "L", "type": "ss", "comp": None}) is False
