"""Tests for TernaryLiquidInterpolation.from_binaries — construction from 3 BinaryLiquid edges.

Runs offline: BinaryLiquid instances are constructed directly (no MPDS/liquidus data),
which is all from_binaries needs — components, mixing model, tau, ss flags.

Run with: python -m pytest tests/test_ternary_from_binaries.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gliquid.api as api  # noqa: E402
from gliquid.binary import BinaryLiquid  # noqa: E402
from gliquid.phase import UNARY, Phase  # noqa: E402
from gliquid.ternary import TernaryLiquidInterpolation  # noqa: E402

L_SETS = {
    ("Hf", "Ti"): [8000.0, -1.5, 3000.0, 0.5],
    ("Hf", "Zr"): [5000.0, 0.0, -2500.0, 0.75],
    ("Ti", "Zr"): [-12000.0, 2.0, -1500.0, 0.25],
}


@pytest.fixture(scope="module")
def dummy_ch():
    ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return ch


def _bl(pair, dummy_ch, params=None, param_format="linear", tau=8000):
    return BinaryLiquid(
        "-".join(pair),
        list(pair),
        component_data=UNARY.component_data(list(pair)),
        dft_ch=dummy_ch,
        phases=[Phase(phase_type="liquid", name="L")],
        temp_range=[300, 3000],
        params=list(L_SETS[tuple(pair)]) if params is None else params,
        param_format=param_format,
        tau=tau,
    )


class TestFromBinaries:
    def test_components_and_edge_orientation(self, dummy_ch):
        binaries = [_bl(p, dummy_ch) for p in L_SETS]
        tli = TernaryLiquidInterpolation.from_binaries(binaries)
        assert tli.components == ["Hf", "Ti", "Zr"]
        assert tli.binary_systems == ["Hf-Ti", "Ti-Zr", "Zr-Hf"]
        # Alphabetical edges carry the binary's params unchanged...
        assert tli.xs_mix["Hf-Ti"].values == L_SETS[("Hf", "Ti")]
        assert tli.xs_mix["Ti-Zr"].values == L_SETS[("Ti", "Zr")]
        # ...the inverted cyclic edge (Zr-Hf vs stored Hf-Zr) flips odd orders.
        hf_zr = L_SETS[("Hf", "Zr")]
        assert tli.xs_mix["Zr-Hf"].values == [hf_zr[0], hf_zr[1], -hf_zr[2], -hf_zr[3]]

    def test_format_and_tau_are_inherited(self, dummy_ch):
        binaries = [
            _bl(
                p,
                dummy_ch,
                param_format="comb-exp",
                params=[v if i != 3 else 0.0 for i, v in enumerate(L_SETS[p])],
                tau=5000,
            )
            for p in L_SETS
        ]
        tli = TernaryLiquidInterpolation.from_binaries(binaries)
        assert tli.param_format == "comb-exp"
        assert tli.tau == 5000

    def test_mixed_formats_raise(self, dummy_ch):
        pairs = list(L_SETS)
        binaries = [
            _bl(pairs[0], dummy_ch, param_format="linear"),
            _bl(pairs[1], dummy_ch, param_format="comb-exp", params=[1.0, 2.0, 3.0, 0.0]),
            _bl(pairs[2], dummy_ch, param_format="linear"),
        ]
        with pytest.raises(ValueError, match="format"):
            TernaryLiquidInterpolation.from_binaries(binaries)

    def test_wrong_element_span_raises(self, dummy_ch):
        binaries = [_bl(("Hf", "Ti"), dummy_ch)] * 3
        with pytest.raises(ValueError):
            TernaryLiquidInterpolation.from_binaries(binaries)

    def test_mixing_models_are_independent_copies(self, dummy_ch):
        binaries = [_bl(p, dummy_ch) for p in L_SETS]
        tli = TernaryLiquidInterpolation.from_binaries(binaries)
        tli.xs_mix["Hf-Ti"]["L0_a"] = 999.0
        assert binaries[0].xs_mix["L0_a"] == L_SETS[("Hf", "Ti")][0]

    def test_surface_matches_explicit_construction(self, dummy_ch):
        binaries = [_bl(p, dummy_ch) for p in L_SETS]
        tli = TernaryLiquidInterpolation.from_binaries(binaries, delta=0.25)
        tli.interpolate_liquid_surface()

        explicit = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"],
            delta=0.25,
            param_format="linear",
            xs_mix={
                "Hf-Ti": list(L_SETS[("Hf", "Ti")]),
                "Ti-Zr": list(L_SETS[("Ti", "Zr")]),
                "Zr-Hf": [
                    L_SETS[("Hf", "Zr")][0],
                    L_SETS[("Hf", "Zr")][1],
                    -L_SETS[("Hf", "Zr")][2],
                    -L_SETS[("Hf", "Zr")][3],
                ],
            },
        )
        explicit.interpolate_liquid_surface()
        np.testing.assert_allclose(
            tli.hsx_df["H"].to_numpy(), explicit.hsx_df["H"].to_numpy(), rtol=0
        )
        np.testing.assert_allclose(
            tli.hsx_df["S"].to_numpy(), explicit.hsx_df["S"].to_numpy(), rtol=0
        )


def _swapped(params):
    """Odd RK orders negate under a component-order swap of the 4-vector layout."""
    return [params[0], params[1], -params[2], -params[3]]


class TestConstructionOrderPreserved:
    """TLI keeps the GIVEN component order (it used to force-sort); from_binaries stays
    alphabetical by default but accepts a components= override."""

    UNSORTED = ["Ti", "Hf", "Zr"]

    def test_components_and_cyclic_edges_follow_given_order(self):
        tli = TernaryLiquidInterpolation(
            self.UNSORTED,
            xs_mix={
                "Ti-Hf": _swapped(L_SETS[("Hf", "Ti")]),
                "Hf-Zr": list(L_SETS[("Hf", "Zr")]),
                "Zr-Ti": _swapped(L_SETS[("Ti", "Zr")]),
            },
        )
        assert tli.components == ["Ti", "Hf", "Zr"]
        assert tli.binary_systems == ["Ti-Hf", "Hf-Zr", "Zr-Ti"]

    def test_surface_equals_sorted_frame_under_permutation(self):
        sorted_tli = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"],
            delta=0.25,
            xs_mix={
                "Hf-Ti": list(L_SETS[("Hf", "Ti")]),
                "Ti-Zr": list(L_SETS[("Ti", "Zr")]),
                "Zr-Hf": _swapped(L_SETS[("Hf", "Zr")]),
            },
        )
        sorted_tli.interpolate_liquid_surface()
        unsorted_tli = TernaryLiquidInterpolation(
            self.UNSORTED,
            delta=0.25,
            xs_mix={
                "Ti-Hf": _swapped(L_SETS[("Hf", "Ti")]),
                "Hf-Zr": list(L_SETS[("Hf", "Zr")]),
                "Zr-Ti": _swapped(L_SETS[("Ti", "Zr")]),
            },
        )
        unsorted_tli.interpolate_liquid_surface()

        def by_ti_zr(tli):
            """{(x_Ti, x_Zr): (H, S)} — a frame-independent key for each grid point."""
            comps = tli.components
            out = {}
            for _, row in tli.hsx_df.iterrows():
                fr = {comps[1]: row["x0"], comps[2]: row["x1"]}
                fr[comps[0]] = 1.0 - row["x0"] - row["x1"]
                out[(round(fr["Ti"], 6), round(fr["Zr"], 6))] = (row["H"], row["S"])
            return out

        surf_s, surf_u = by_ti_zr(sorted_tli), by_ti_zr(unsorted_tli)
        assert set(surf_s) == set(surf_u)
        for key in surf_s:
            assert surf_u[key][0] == pytest.approx(surf_s[key][0], rel=1e-9, abs=1e-8)
            assert surf_u[key][1] == pytest.approx(surf_s[key][1], rel=1e-9, abs=1e-10)

    def test_from_binaries_components_override(self, dummy_ch):
        binaries = [_bl(p, dummy_ch) for p in L_SETS]
        tli = TernaryLiquidInterpolation.from_binaries(binaries, components=self.UNSORTED)
        assert tli.components == ["Ti", "Hf", "Zr"]
        assert tli.binary_systems == ["Ti-Hf", "Hf-Zr", "Zr-Ti"]
        # edge orientations follow the override's cyclic frame
        assert tli.xs_mix["Ti-Hf"].values == _swapped(L_SETS[("Hf", "Ti")])
        assert tli.xs_mix["Hf-Zr"].values == L_SETS[("Hf", "Zr")]
        assert tli.xs_mix["Zr-Ti"].values == _swapped(L_SETS[("Ti", "Zr")])

    def test_from_binaries_bad_override_raises(self, dummy_ch):
        binaries = [_bl(p, dummy_ch) for p in L_SETS]
        with pytest.raises(ValueError, match="span"):
            TernaryLiquidInterpolation.from_binaries(binaries, components=["Ti", "Hf", "Nb"])


class TestWithComponentOrder:
    """with_component_order re-frames a TLI onto a new component order (mirrors
    BinaryLiquid.with_component_order): cyclic edges reorient (odd RK orders flip on an
    inverted edge), the 'A-B-C' triplet key re-joins (symmetric, unchanged), and the
    interpolated surface is preserved under the coordinate permutation."""

    UNSORTED = ["Ti", "Hf", "Zr"]

    def _unsorted_tli(self, **kwargs):
        return TernaryLiquidInterpolation(
            self.UNSORTED,
            xs_mix={
                "Ti-Hf": _swapped(L_SETS[("Hf", "Ti")]),
                "Hf-Zr": list(L_SETS[("Hf", "Zr")]),
                "Zr-Ti": _swapped(L_SETS[("Ti", "Zr")]),
            },
            **kwargs,
        )

    def test_reframe_to_alphabetical_edges(self):
        reframed = self._unsorted_tli().with_component_order("alphabetical")
        assert reframed.components == ["Hf", "Ti", "Zr"]
        assert reframed.binary_systems == ["Hf-Ti", "Ti-Zr", "Zr-Hf"]
        # edges now match the alphabetical-construction reference
        assert reframed.xs_mix["Hf-Ti"].values == L_SETS[("Hf", "Ti")]
        assert reframed.xs_mix["Ti-Zr"].values == L_SETS[("Ti", "Zr")]
        assert reframed.xs_mix["Zr-Hf"].values == _swapped(L_SETS[("Hf", "Zr")])

    def test_noop_returns_self(self):
        tli = self._unsorted_tli()
        assert tli.with_component_order("given") is tli
        assert tli.with_component_order(["Ti", "Hf", "Zr"]) is tli

    def test_triplet_key_reframed(self):
        unsorted = TernaryLiquidInterpolation(
            self.UNSORTED,
            ternary_l0=5000.0,
            xs_mix={"Ti-Hf": [0.0] * 4, "Hf-Zr": [0.0] * 4, "Zr-Ti": [0.0] * 4},
        )
        reframed = unsorted.with_component_order("alphabetical")
        assert "Ti-Hf-Zr" not in reframed.xs_mix
        assert reframed.xs_mix["Hf-Ti-Zr"].values == [5000.0]

    def test_surface_preserved_under_reframe(self):
        unsorted = TernaryLiquidInterpolation(
            self.UNSORTED,
            delta=0.25,
            xs_mix={
                "Ti-Hf": _swapped(L_SETS[("Hf", "Ti")]),
                "Hf-Zr": list(L_SETS[("Hf", "Zr")]),
                "Zr-Ti": _swapped(L_SETS[("Ti", "Zr")]),
            },
        )
        reframed = unsorted.with_component_order("alphabetical")
        unsorted.interpolate_liquid_surface()
        reframed.interpolate_liquid_surface()

        def by_ti_zr(tli):
            comps = tli.components
            out = {}
            for _, row in tli.hsx_df.iterrows():
                fr = {comps[1]: row["x0"], comps[2]: row["x1"]}
                fr[comps[0]] = 1.0 - row["x0"] - row["x1"]
                out[(round(fr["Ti"], 6), round(fr["Zr"], 6))] = (row["H"], row["S"])
            return out

        su, sr = by_ti_zr(unsorted), by_ti_zr(reframed)
        assert set(su) == set(sr)
        for key in su:
            assert sr[key][0] == pytest.approx(su[key][0], rel=1e-9, abs=1e-8)
            assert sr[key][1] == pytest.approx(su[key][1], rel=1e-9, abs=1e-10)
