"""Characterization pins for BinaryLiquid.find_invariant_points.

Frozen pre-move by dev/scripts/_scratch/freeze_invariant_pins.py into
fixtures/invariant_points_pins.json, as the behavior-preservation gate for moving the
algorithm into gliquid.mpds.identify_invariant_points (the method stays as a thin
wrapper). The pinned VALUES are the invariant: identical invariant/low-T structures
(floats at rtol <= 1e-9), identical init_error semantics, and the returned lists ARE
the instance attributes. Everything runs offline against the shipped data caches.
"""

import json
import sys
from pathlib import Path

import pytest

from gliquid.binary import BinaryLiquid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PINS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "invariant_points_pins.json"
    ).read_text()
)


@pytest.mark.parametrize("name", sorted(PINS))
def test_pinned_case(name):
    case = PINS[name]
    bl = BinaryLiquid.from_cache(case["system"], **case["cache_kwargs"])
    assert bool(bl.init_error) == case["init_error_before"]
    invariants, low_t = bl.find_invariant_points(**case["call_kwargs"])
    pu.assert_deep_approx(case["invariants"], invariants)
    pu.assert_deep_approx(case["low_t_exp_phases"], low_t)
    # 'init_error_after' was frozen when full-comp-SS detection set init_error; the
    # detection now notifies via full_comp_ss instead (init_error stays False), so the
    # pinned flag maps to the union. The fixture bytes stay untouched (pins are API).
    assert bool(bl.init_error or bl.full_comp_ss) == case["init_error_after"]
    assert (invariants is bl.invariants and low_t is bl.low_t_exp_phases) == case[
        "returned_is_attribute"
    ]


class TestNoReferenceContract:
    """mpds_json without a reference: warns, returns ([], []), touches NO attributes."""

    def test_early_return_leaves_state_untouched(self, caplog):
        bl = object.__new__(BinaryLiquid)  # attribute-free shell; only mpds_json is read
        bl.mpds_json = {"reference": None}
        assert bl.find_invariant_points() == ([], [])
        assert not hasattr(bl, "invariants")
        assert not hasattr(bl, "low_t_exp_phases")
        assert not hasattr(bl, "init_error")
        assert "does not contain any data" in caplog.text  # logger.warning since the logging wave
