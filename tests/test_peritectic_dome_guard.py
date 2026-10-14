"""The peritectic anchor a phase gets when a congruent dome stands between it and the liquid.

``identify_invariant_points`` matches a phase's field top to the liquidus and records the
nearest crossing as that phase's peritectic. ``_unclaimed_maximum_between`` is what stops the
match reaching across another compound's congruent dome to the liquid on its far side, for a
dome the congruent pass did not claim; a dome it did claim is already blocked by the phase
entering ``stable_phase_comps``. Pure geometry — no cache and no network.
"""

import pytest

from gliquid.mpds import _CONGRUENT_MATCH_TOL, _unclaimed_maximum_between


class TestUnclaimedMaximumBetween:
    def test_dome_between_phase_and_endpoint_separates(self):
        assert _unclaimed_maximum_between([[0.50, 2184.0]], 0.438, 0.658)

    def test_direction_does_not_matter(self):
        assert _unclaimed_maximum_between([[0.50, 2184.0]], 0.658, 0.438)

    def test_dome_outside_the_interval_does_not_separate(self):
        assert not _unclaimed_maximum_between([[0.80, 2184.0]], 0.438, 0.658)
        assert not _unclaimed_maximum_between([[0.20, 2184.0]], 0.438, 0.658)

    def test_phases_own_dome_does_not_separate(self):
        """A maximum within tol of the phase is its own congruent point."""
        near = 0.438 + _CONGRUENT_MATCH_TOL / 2
        assert not _unclaimed_maximum_between([[near, 2184.0]], 0.438, 0.658)

    def test_dome_just_beyond_tol_separates(self):
        beyond = 0.438 + _CONGRUENT_MATCH_TOL * 2
        assert _unclaimed_maximum_between([[beyond, 2184.0]], 0.438, 0.658)

    def test_endpoint_coincident_with_a_dome_does_not_separate(self):
        """A maximum AT the endpoint is the endpoint, not something between."""
        assert not _unclaimed_maximum_between([[0.658, 2184.0]], 0.438, 0.658)

    def test_no_maxima_never_separates(self):
        assert not _unclaimed_maximum_between([], 0.438, 0.658)

    @pytest.mark.parametrize("maxima", [[[0.30, 1.0], [0.50, 2.0]], [[0.50, 2.0], [0.90, 1.0]]])
    def test_any_one_dome_in_the_interval_separates(self, maxima):
        assert _unclaimed_maximum_between(maxima, 0.438, 0.658)


class TestPeritecticEndpointFiltering:
    """The filter as ``identify_invariant_points`` applies it to a candidate endpoint list."""

    @staticmethod
    def _surviving(endpoints, maxima, phase_comp):
        return [ep for ep in endpoints if not _unclaimed_maximum_between(maxima, phase_comp, ep[0])]

    def test_endpoint_across_a_dome_is_dropped_and_a_nearer_one_kept(self):
        endpoints = [[0.658, 1687.75], [0.400, 1687.75]]
        survivors = self._surviving(endpoints, [[0.502, 2184.45]], 0.4382)
        assert survivors == [[0.400, 1687.75]]

    def test_every_endpoint_across_a_dome_leaves_no_anchor(self):
        """With no survivor the phase records no peritectic, which is the intended outcome
        for a phase that never borders the liquid."""
        endpoints = [[0.658, 1687.75], [0.700, 1687.75]]
        assert self._surviving(endpoints, [[0.502, 2184.45]], 0.4382) == []

    def test_unrelated_dome_leaves_the_anchor_alone(self):
        endpoints = [[0.400, 1687.75]]
        assert self._surviving(endpoints, [[0.900, 2000.0]], 0.4382) == endpoints
