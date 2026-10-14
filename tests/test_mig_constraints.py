"""The two dome equations a ``mig`` invariant contributes, and when each applies.

``BinaryLiquid._mig_constraint_equations`` emits the bottom-of-dome equation (equal liquid
free-energy slope at the two feet on the monotectic isotherm) when both feet sit on the
digitized liquidus, and the top-of-dome equation (zero curvature at the apex) when the
record carries a known apex. Self-consistency: a diagram generated from known parameters
yields feet and an apex from which each equation recovers ``L0_a`` at the true
``(L0_b, L1_a)``. Runs offline on the shipped Ag-V and Hf-Zr caches.
"""

import copy
import math

import pytest

from gliquid import mpds
from gliquid.binary import BinaryLiquid

from ._mpds_builders import by_order as _by_order
from ._mpds_builders import recovered_L0_a as _recovered_L0_a

# Parameters that open a two-liquid dome on each shipped diagram.
TRUTH = {
    "Ag-V": ([54923.70, 6.8246, 608.58, 0.0], {}),
    "Hf-Zr": ([45000.0, 3.0, -5000.0, 0.0], {"pd_ind": 0}),
}
BOTTOM_RTOL = 0.01
TOP_RTOL = 0.02


def _dome(system):
    """A BinaryLiquid at its true parameters plus the mig record its own diagram implies."""
    params, kwargs = TRUTH[system]
    bl = BinaryLiquid.from_cache(system, param_format="comb-exp", **kwargs)
    assert not bl.init_error
    bl.update_params(params)
    bl.update_phase_points()
    ties = [
        entry
        for entry in bl.hsx.liquidus_invariants()[0]["Misc Gaps"]
        if list(entry[3]).count("L") == 2
    ]
    assert ties, f"{system}: the fixture parameters open no two-liquid dome"
    coldest = min(ties, key=lambda entry: entry[0])
    x1, x3 = sorted(c for c, p in zip(coldest[2], coldest[3]) if p == "L")
    t_mono = coldest[0] + 273.15
    inside = [(x, t) for x, t in bl.phases[-1].points if x1 < x < x3]
    xa, ta = max(inside, key=lambda point: point[1])
    record = {
        "type": "mig",
        "comp": xa,
        "cbounds": [[x1, t_mono], [x3, t_mono]],
        "tbounds": [[xa, t_mono], [xa, ta]],
        "phases": None,
        "phase_comps": None,
        "apex": [xa, ta],
        "partial": False,
        "source": "label",
    }
    return bl, params, record


@pytest.fixture(scope="module", params=sorted(TRUTH))
def dome(request):
    return _dome(request.param)


class TestRecovery:
    def test_bottom_of_dome_recovers_L0_a(self, dome):
        bl, params, record = dome
        bottom = _by_order(bl._mig_constraint_equations(record, foot_tol=math.inf))["1st order"]
        assert _recovered_L0_a(bottom, params) == pytest.approx(params[0], rel=BOTTOM_RTOL)

    def test_top_of_dome_recovers_L0_a(self, dome):
        bl, params, record = dome
        top = _by_order(bl._mig_constraint_equations(record, foot_tol=math.inf))["2nd order"]
        assert _recovered_L0_a(top, params) == pytest.approx(params[0], rel=TOP_RTOL)

    def test_shifted_feet_land_farther(self, dome):
        bl, params, record = dome
        shifted = copy.deepcopy(record)
        for foot in shifted["cbounds"]:
            foot[0] += 0.05
        true_eq = _by_order(bl._mig_constraint_equations(record, foot_tol=math.inf))["1st order"]
        shifted_eq = _by_order(bl._mig_constraint_equations(shifted, foot_tol=math.inf))[
            "1st order"
        ]
        err_true = abs(_recovered_L0_a(true_eq, params) - params[0])
        err_shifted = abs(_recovered_L0_a(shifted_eq, params) - params[0])
        assert err_shifted > 3 * err_true

    def test_equations_carry_their_temperatures(self, dome):
        bl, params, record = dome
        entries = _by_order(bl._mig_constraint_equations(record, foot_tol=math.inf))
        assert entries["1st order"][2] == pytest.approx(record["cbounds"][0][1])
        assert entries["2nd order"][2] == pytest.approx(record["apex"][1])


class TestSelection:
    def test_complete_dome_emits_both_equations(self, dome):
        bl, _, record = dome
        orders = {entry[1] for entry in bl._mig_constraint_equations(record, foot_tol=math.inf)}
        assert orders == {"1st order", "2nd order"}

    def test_partial_dome_emits_only_the_feet_equation(self, dome):
        bl, _, record = dome
        partial = {**record, "apex": None, "partial": True}
        orders = [entry[1] for entry in bl._mig_constraint_equations(partial, foot_tol=math.inf)]
        assert orders == ["1st order"]

    def test_feet_off_the_liquidus_emit_no_feet_equation(self, dome):
        bl, _, record = dome
        floating = copy.deepcopy(record)
        for foot in floating["cbounds"]:
            foot[1] += 100.0
        orders = [entry[1] for entry in bl._mig_constraint_equations(floating, foot_tol=20.0)]
        assert "1st order" not in orders

    def test_feet_on_different_isotherms_emit_no_feet_equation(self, dome):
        bl, _, record = dome
        skewed = copy.deepcopy(record)
        skewed["cbounds"][1][1] += 100.0
        orders = [entry[1] for entry in bl._mig_constraint_equations(skewed, foot_tol=20.0)]
        assert "1st order" not in orders

    def test_feet_on_the_liquidus_pass_the_gate(self):
        bl = BinaryLiquid.from_cache("Ag-V", param_format="comb-exp")
        invariants, _ = bl.find_invariant_points(verbose=False)
        (record,) = [iv for iv in invariants if iv["type"] == "mig"]
        orders = {entry[1] for entry in bl._mig_constraint_equations(record, foot_tol=20.0)}
        assert "1st order" in orders

    def test_old_style_record_keeps_both_equations(self, dome):
        bl, _, record = dome
        old = {k: v for k, v in record.items() if k not in ("apex", "partial", "source")}
        orders = {entry[1] for entry in bl._mig_constraint_equations(old, foot_tol=math.inf)}
        assert orders == {"1st order", "2nd order"}


class TestMigApex:
    def test_apex_key_wins(self):
        assert mpds.mig_apex(
            {"type": "mig", "apex": [0.5, 2000.0], "tbounds": [[0.5, 1.0], [0.5, 2.0]]}
        ) == [
            0.5,
            2000.0,
        ]

    def test_partial_without_apex_is_unknown(self):
        assert (
            mpds.mig_apex({"type": "mig", "partial": True, "tbounds": [[0.5, 1.0], [0.5, 2.0]]})
            is None
        )

    def test_old_record_reads_tbounds(self):
        assert mpds.mig_apex({"type": "mig", "tbounds": [[0.5, 1.0], [0.5, 2.0]]}) == [0.5, 2.0]

    def test_record_without_bounds_is_unknown(self):
        assert mpds.mig_apex({"type": "mig"}) is None
