"""The absolute room-temperature floor beside the relative low-temperature threshold.

``low_temp_threshold`` is a fraction of the diagram's DECLARED temperature axis, so a diagram
whose axis is wrong carries the threshold with it. Au-Zn declares a bottom of -300 C, which is
below absolute zero and puts the relative cutoff at 43 K -- compounds drawn all the way down
to 273 K then read as "not low temperature" while one drawn to -1 K reads as low temperature.

``reaches_room_temperature`` is the floor that does not depend on the declared frame: a phase
whose field extends to 298.15 K or below is stable at room temperature whatever the axis says.

Fixtures reproduce the Au-Zn frame rather than a synthetic one, because the effect only appears
when the declared axis puts the relative threshold below room temperature.
"""

import pytest

from gliquid import mpds

AU_ZN_TEMP = (-300.0, 1100.0)  # the real declared axis; threshold lands at 43.15 K
SANE_TEMP = (0.0, 1100.0)


def _phase(bottom_k, top_k=1000.0, name="Au3Zn ht"):
    return {"name": name, "type": "lc", "comp": 0.25, "tbounds": [[0.25, bottom_k], [0.25, top_k]]}


def _json(temp):
    return {"temp": list(temp)}


class TestThresholdIsFrameDependent:
    def test_a_below_zero_axis_drives_the_cutoff_under_room_temperature(self):
        assert mpds.low_temp_threshold(_json(AU_ZN_TEMP)) == pytest.approx(43.15)

    def test_a_sane_axis_leaves_the_cutoff_above_room_temperature(self):
        assert mpds.low_temp_threshold(_json(SANE_TEMP)) > mpds.ROOM_TEMPERATURE


class TestReachesRoomTemperature:
    def test_a_field_drawn_to_freezing_point_qualifies(self):
        assert mpds.reaches_room_temperature(_phase(273.15))

    def test_the_boundary_is_inclusive(self):
        assert mpds.reaches_room_temperature(_phase(mpds.ROOM_TEMPERATURE))

    def test_a_field_that_stops_above_room_temperature_does_not(self):
        assert not mpds.reaches_room_temperature(_phase(400.0))

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"tbounds": []},
            {"tbounds": [[]]},
            {"tbounds": [[0.25, None]]},
            {"tbounds": [[0.25, "warm"]]},
        ],
    )
    def test_unusable_bounds_are_not_room_temperature(self, bad):
        assert not mpds.reaches_room_temperature(bad)


class TestTheClauseCoversWhatTheThresholdMisses:
    """The pair, on the frame that motivates it."""

    def test_a_phase_the_relative_rule_rejects_is_caught_by_the_floor(self):
        phase = _phase(273.15)
        threshold = mpds.low_temp_threshold(_json(AU_ZN_TEMP))
        assert phase["tbounds"][0][1] >= threshold, "fixture must fail the relative rule"
        assert mpds.reaches_room_temperature(phase)

    def test_the_floor_does_not_admit_a_genuinely_high_temperature_phase(self):
        """The clause must widen the list, not empty the criterion."""
        phase = _phase(900.0)
        threshold = mpds.low_temp_threshold(_json(AU_ZN_TEMP))
        assert phase["tbounds"][0][1] >= threshold
        assert not mpds.reaches_room_temperature(phase)

    def test_a_sane_frame_needs_no_floor(self):
        """On a well-declared axis the relative rule already admits a room-temperature phase."""
        phase = _phase(273.15)
        assert phase["tbounds"][0][1] < mpds.low_temp_threshold(_json(SANE_TEMP))
