"""``xs_mix`` and ``fit_or_pred`` accept each binary pair in either order.

A pair named against the cyclic edge convention (``'Al-Si'`` where the edge is ``'Si-Al'``)
is re-keyed onto its edge, with the odd Redlich-Kister orders flipped for a mixing model.
Offline: nothing here interpolates.
"""

import pytest

import gliquid.ternary as tmod

AL_SI = [-13430.0, -1.516, -966.7, 0.5]
AL_MG = [804.7, 0.08031, 258.5, 0.0]
MG_SI = [-45190.0, 9.047, 1482.0, 0.0]


def _interp(xs_mix, **kwargs):
    return tmod.TernaryLiquidInterpolation(
        ["Al", "Mg", "Si"], xs_mix=xs_mix, param_format="comb-exp", **kwargs
    )


def test_an_inverse_pair_is_rekeyed_with_odd_orders_flipped():
    ti = _interp({"Al-Si": AL_SI, "Al-Mg": AL_MG, "Mg-Si": MG_SI})
    assert list(ti.xs_mix) == ["Si-Al", "Al-Mg", "Mg-Si"]
    assert list(ti.xs_mix["Si-Al"].values) == [AL_SI[0], AL_SI[1], -AL_SI[2], -AL_SI[3]]
    assert list(ti.xs_mix["Al-Mg"].values) == AL_MG


def test_either_order_builds_the_same_models():
    from_inverse = _interp({"Al-Si": AL_SI, "Al-Mg": AL_MG, "Mg-Si": MG_SI})
    swapped = tmod.RKPolyExp("comb-exp", AL_SI).swapped()
    from_edges = _interp({"Si-Al": swapped, "Al-Mg": AL_MG, "Mg-Si": MG_SI})
    assert {k: list(v.values) for k, v in from_inverse.xs_mix.items()} == {
        k: list(v.values) for k, v in from_edges.xs_mix.items()
    }


def test_fit_or_pred_is_rekeyed_without_change():
    ti = _interp({"Al-Si": AL_SI}, fit_or_pred={"Al-Si": "fit", "Mg-Al": "pred"})
    assert ti.fit_or_pred == {"Si-Al": "fit", "Al-Mg": "pred"}


def test_a_pair_given_in_both_orders_is_rejected():
    with pytest.raises(ValueError, match="both orders"):
        _interp({"Al-Si": AL_SI, "Si-Al": AL_SI})


def test_the_triplet_and_foreign_keys_are_kept_as_given():
    oriented = tmod.orient_to_edges(
        {"Al-Mg-Si": 1.0, "Cu-Zn": 2.0, "Mg-Al": 3.0}, ["Al-Mg", "Mg-Si", "Si-Al"]
    )
    assert oriented == {"Al-Mg-Si": 1.0, "Cu-Zn": 2.0, "Al-Mg": 3.0}


def test_a_triplet_in_any_order_is_rekeyed_onto_the_edges():
    oriented = tmod.orient_to_edges({"Si-Al-Mg": 1.0, "Cu-Zn-Al": 2.0}, ["Al-Mg", "Mg-Si", "Si-Al"])
    assert oriented == {"Al-Mg-Si": 1.0, "Cu-Zn-Al": 2.0}


def test_a_triplet_given_in_two_orders_is_rejected():
    with pytest.raises(ValueError, match="both orders"):
        _interp({"Al-Mg-Si": [1.0], "Si-Mg-Al": [1.0]})
