"""Tests for gliquid.ternary.

Focused on the unary reference-state path (``TernaryLiquidInterpolation.init_ref_data``),
asserted against stable values from ``data/phase_transitions.json``.
"""

import pytest

import gliquid.ternary as ht


def _ref_data_for(elements):
    """Build ref_data via init_ref_data without running the (heavier) __init__."""
    ti = ht.TernaryLiquidInterpolation.__new__(ht.TernaryLiquidInterpolation)
    ti.components = list(elements)
    ti.init_ref_data()
    return ti.ref_data


def test_init_ref_data_reference_values():
    ref = _ref_data_for(["Al", "Cu", "Mg"])
    # Stable values from data/phase_transitions.json (unchanged by the refactor).
    assert list(ref["H"]) == pytest.approx([10710.0, 13260.0, 8480.0])
    assert list(ref["T"]) == pytest.approx([933.5, 1357.77, 923.0])
    assert list(ref["S"]) == pytest.approx([11.473, 9.766, 9.187], abs=1e-2)


def test_init_ref_data_preserves_element_order():
    # init_ref_data must map arrays positionally to self.tern_sys (order matters downstream).
    ref = _ref_data_for(["Mg", "Al", "Cu"])
    assert list(ref["T"]) == pytest.approx([923.0, 933.5, 1357.77])


def test_init_ref_data_unknown_element_defaults_zero():
    ref = _ref_data_for(["Al", "Xx", "Cu"])
    assert list(ref["H"])[1] == 0.0
    assert list(ref["S"])[1] == 0.0
    assert list(ref["T"])[1] == 0.0
