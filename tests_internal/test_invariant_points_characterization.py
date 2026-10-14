"""``BinaryLiquid.find_invariant_points``, the wrapper over
:func:`gliquid.mpds.identify_invariant_points`.

The wrapper passes the instance's inputs and its keywords through, assigns the returned lists to
``invariants`` and ``low_t_exp_phases`` and returns them, and records a full-composition solid
solution on ``full_comp_ss`` with ``init_error`` left unset. The values themselves are pinned in
``test_decomposition_pins.py``. Everything runs offline against the shipped data caches.
"""

import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid


def test_the_wrapper_forwards_its_inputs_and_assigns_the_results(monkeypatch):
    calls = []
    identify = mpds.identify_invariant_points

    def spy(*args, **kwargs):
        calls.append((args, kwargs, identify(*args, **kwargs)))
        return calls[-1][2]

    monkeypatch.setattr(mpds, "identify_invariant_points", spy)
    for call_kwargs, full_comp_ss in (({"t_tol": 25, "check_full_ss": False}, False), ({}, True)):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        inputs = (bl.mpds_json, bl.components, bl.digitized_liq, bl.component_data, bl.temp_range)
        invariants, low_t = bl.find_invariant_points(**call_kwargs)
        ((args, kwargs, result),) = calls
        calls.clear()
        assert all(a is b for a, b in zip(args, inputs, strict=True))
        assert kwargs == {
            "verbose": False,
            "t_tol": 15,
            "check_full_ss": True,
            **call_kwargs,
            "liquidus_holes": bl.liquidus_holes,
        }
        assert invariants is result[0] is bl.invariants
        assert low_t is result[1] is bl.low_t_exp_phases
        assert result[2] is full_comp_ss
        assert bl.full_comp_ss is full_comp_ss
        assert not bl.init_error


class TestNoReferenceContract:
    """mpds_json without a reference: warns, returns ([], []), touches NO attributes."""

    def test_early_return_leaves_state_untouched(self, caplog):
        bl = object.__new__(BinaryLiquid)  # attribute-free shell; only mpds_json is read
        bl.mpds_json = {"reference": None}
        assert bl.find_invariant_points() == ([], [])
        assert not hasattr(bl, "invariants")
        assert not hasattr(bl, "low_t_exp_phases")
        assert not hasattr(bl, "init_error")
        assert "does not contain any data" in caplog.text  # emitted via logger.warning
