"""Characterization pins for component-ordering semantics (pre-restructuring freeze).

Written at the start of the compound-forward-compat refactor, BEFORE any code move:
(a) a reversed-order DIRECT-constructed BinaryLiquid — the liquid path is already
    orientation-correct today; these pins keep it that way through the restructuring
    (from_cache with reversed input is exercised separately once cache keys are
    canonicalized — today it cannot construct offline).
(b) the ternary DFT-entry cache honors TernaryLiquidInterpolation's own ``data_dir``
    (flat layout) — the read-path contract the unified api loader must preserve.
(c) the ternary-L0 analytic oracle: a nonzero ``ternary_l0`` shifts the interpolated
    liquid enthalpy surface by exactly ``l0 * x_A * x_B * x_C`` and leaves entropy
    untouched. No pre-existing pin exercises a nonzero ternary_l0, so this is the
    certificate that must stay green when the term moves into the solution model.

Runs fully offline (unary registry + shipped Hf-Zr / Al-Mg-Si data fixtures).
"""

import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.config as config
from gliquid.binary import BinaryLiquid, _x_vals
from gliquid.phase import UNARY, Phase
from gliquid.solution import t_sym
from gliquid.ternary import TernaryLiquidInterpolation

RTOL = 1e-9


def _direct_bl(sys_name, components, ch):
    """BinaryLiquid built directly (no cache access), mirroring test_binary_solution."""
    return BinaryLiquid(
        sys_name,
        components,
        component_data=UNARY.component_data(components),
        dft_ch=ch,
        phases=[Phase(phase_type="liquid", name="L")],
        temp_range=[300, 3000],
    )


@pytest.fixture(scope="module")
def hfzr_hull():
    ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return ch


@pytest.fixture(scope="module")
def bl_alpha(hfzr_hull):
    return _direct_bl("Hf-Zr", ["Hf", "Zr"], hfzr_hull)


@pytest.fixture(scope="module")
def bl_rev(hfzr_hull):
    return _direct_bl("Zr-Hf", ["Zr", "Hf"], hfzr_hull)


class TestReversedDirectConstruction:
    """Construction order is authoritative: refs, HSX axis and phase fractions follow it."""

    def test_phase_fraction_axis_follows_component_order(self, bl_alpha, bl_rev):
        p = Phase(phase_type="solid", name="Hf3Zr", composition=Composition("Hf3Zr"))
        assert bl_alpha._phase_x(p) == pytest.approx(0.25, rel=RTOL)  # fraction of Zr
        assert bl_rev._phase_x(p) == pytest.approx(0.75, rel=RTOL)  # fraction of Hf

    def test_hsx_endpoints_follow_component_order(self, bl_alpha, bl_rev):
        data_a = bl_alpha.to_HSX()
        data_r = bl_rev.to_HSX()
        assert data_a["X"] == [float(x) for x in _x_vals]
        assert data_r["X"] == [float(x) for x in _x_vals]
        assert data_a["H"][0] == pytest.approx(UNARY["Hf"].h_liq, rel=RTOL)
        assert data_a["H"][-1] == pytest.approx(UNARY["Zr"].h_liq, rel=RTOL)
        assert data_r["H"][0] == pytest.approx(UNARY["Zr"].h_liq, rel=RTOL)
        assert data_r["H"][-1] == pytest.approx(UNARY["Hf"].h_liq, rel=RTOL)

    def test_rebuilt_eqs_reference_exprs_follow_component_order(self, bl_alpha, bl_rev):
        """The unpickle path wires component refs into eqs in construction order."""
        clone_a = pickle.loads(pickle.dumps(bl_alpha))
        clone_r = pickle.loads(pickle.dumps(bl_rev))
        assert sp.simplify(clone_a.eqs["ga"] - UNARY["Hf"].gibbs_ref_expr(t_sym)) == 0
        assert sp.simplify(clone_a.eqs["gb"] - UNARY["Zr"].gibbs_ref_expr(t_sym)) == 0
        assert sp.simplify(clone_r.eqs["ga"] - UNARY["Zr"].gibbs_ref_expr(t_sym)) == 0
        assert sp.simplify(clone_r.eqs["gb"] - UNARY["Hf"].gibbs_ref_expr(t_sym)) == 0

    def test_rebuilt_liquid_surface_mirrors(self, bl_alpha, bl_rev):
        """With zero mixing params the liquid H is the reference tie-line: reversing the
        component order mirrors it exactly (H_rev(x) == H_alpha(1 - x))."""
        clone_a = pickle.loads(pickle.dumps(bl_alpha))
        clone_r = pickle.loads(pickle.dumps(bl_rev))
        x = np.array([0.25, 0.5, 0.75])
        t = 1500.0
        h_a = clone_a.eqs["h_liq_lambdified"](x, t, *clone_a.xs_mix.values)
        h_r = clone_r.eqs["h_liq_lambdified"](1 - x, t, *clone_r.xs_mix.values)
        np.testing.assert_allclose(
            np.asarray(h_r, dtype=float), np.asarray(h_a, dtype=float), rtol=RTOL
        )


class TestTernaryCacheDataDir:
    """The ternary DFT cache read honors the instance's own data_dir (flat layout)."""

    @pytest.mark.directory_only  # seeds tmp_path by copying a FILE out of the corpus
    def test_get_ternary_form_en_reads_instance_data_dir(self, tmp_path, monkeypatch):
        src = Path(config.data_dir) / "Al-Mg-Si_ENTRIES_MP_GGA.json"
        assert src.exists(), f"fixture missing: {src}"
        shutil.copy(src, tmp_path / "Al-Mg-Si_ENTRIES_MP_GGA.json")

        import gliquid.api as api

        def _no_api():
            raise AssertionError("cache miss: get_ternary_form_en hit the live API")

        monkeypatch.setattr(api, "get_mpr", _no_api)
        tli = TernaryLiquidInterpolation(["Al", "Mg", "Si"], data_dir=tmp_path)
        df = tli.get_ternary_form_en(tli.components)
        assert not df.empty
        assert list(df.columns) == ["x0", "x1", "S", "H", "Phase Name"]
        assert (df["S"] == 0).all()
        assert {"Al", "Mg", "Si"} <= set(df["Phase Name"])


ZERO_EDGES = {"Hf-Ti": [0.0] * 4, "Ti-Zr": [0.0] * 4, "Zr-Hf": [0.0] * 4}
L0_TEST_VALUE = 8000.0


class TestTernaryL0Oracle:
    """H(l0) - H(0) == l0 * x_A * x_B * x_C exactly; S is untouched (T-independent term)."""

    @pytest.fixture(scope="class")
    def surfaces(self):
        base = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"], xs_mix={k: list(v) for k, v in ZERO_EDGES.items()}, delta=0.25
        )
        base.interpolate_liquid_surface()
        shifted = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"],
            xs_mix={k: list(v) for k, v in ZERO_EDGES.items()},
            delta=0.25,
            ternary_l0=L0_TEST_VALUE,
        )
        shifted.interpolate_liquid_surface()
        return base.hsx_df, shifted.hsx_df

    def test_grids_identical(self, surfaces):
        df0, df1 = surfaces
        assert (df0["x0"] == df1["x0"]).all()
        assert (df0["x1"] == df1["x1"]).all()

    def test_h_shift_is_exact_three_body_term(self, surfaces):
        df0, df1 = surfaces
        x0 = df0["x0"].to_numpy(dtype=float)
        x1 = df0["x1"].to_numpy(dtype=float)
        expected = L0_TEST_VALUE * (1 - x0 - x1) * x0 * x1
        diff = df1["H"].to_numpy(dtype=float) - df0["H"].to_numpy(dtype=float)
        # atol floor absorbs float cancellation of the ~1e4-magnitude H values the
        # difference is formed from; it is ~1e-12 relative to them.
        np.testing.assert_allclose(diff, expected, rtol=RTOL, atol=1e-8)

    def test_s_untouched(self, surfaces):
        df0, df1 = surfaces
        np.testing.assert_allclose(
            df1["S"].to_numpy(dtype=float), df0["S"].to_numpy(dtype=float), rtol=RTOL, atol=1e-12
        )


class TestTripletKeyIntegration:
    """The ternary L0 lives IN the mixing/solution model (S10): an explicit 'A-B-C'
    triplet entry, the ternary_l0= folding sugar, and the analytic oracle all agree."""

    def _surface(self, **kwargs):
        tli = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"],
            xs_mix={
                **{k: list(v) for k, v in ZERO_EDGES.items()},
                **kwargs.pop("extra_mixing", {}),
            },
            delta=0.25,
            **kwargs,
        )
        tli.interpolate_liquid_surface()
        return tli

    def test_sugar_folds_into_mixing_no_separate_field(self):
        tli = self._surface(ternary_l0=L0_TEST_VALUE)
        assert not hasattr(tli, "ternary_l0")
        assert "Hf-Ti-Zr" in tli.xs_mix
        assert tli.xs_mix["Hf-Ti-Zr"].values == [L0_TEST_VALUE]

    def test_explicit_triplet_equals_sugar(self):
        via_sugar = self._surface(ternary_l0=L0_TEST_VALUE)
        via_key = self._surface(extra_mixing={"Hf-Ti-Zr": [L0_TEST_VALUE]})
        np.testing.assert_allclose(
            via_key.hsx_df["H"].to_numpy(), via_sugar.hsx_df["H"].to_numpy(), rtol=0
        )
        np.testing.assert_allclose(
            via_key.hsx_df["S"].to_numpy(), via_sugar.hsx_df["S"].to_numpy(), rtol=0
        )

    def test_triplet_term_is_permutation_symmetric(self):
        base = self._surface(extra_mixing={"Hf-Ti-Zr": [L0_TEST_VALUE]})
        permuted = TernaryLiquidInterpolation(
            ["Ti", "Hf", "Zr"],
            delta=0.25,
            xs_mix={
                "Ti-Hf": [0.0] * 4,
                "Hf-Zr": [0.0] * 4,
                "Zr-Ti": [0.0] * 4,
                "Ti-Hf-Zr": [L0_TEST_VALUE],
            },
        )
        permuted.interpolate_liquid_surface()

        def keyed(tli):
            comps = tli.components
            out = {}
            for _, row in tli.hsx_df.iterrows():
                fr = {comps[1]: row["x0"], comps[2]: row["x1"]}
                fr[comps[0]] = 1.0 - row["x0"] - row["x1"]
                out[(round(fr["Ti"], 6), round(fr["Zr"], 6))] = (row["H"], row["S"])
            return out

        surf_b, surf_p = keyed(base), keyed(permuted)
        assert set(surf_b) == set(surf_p)
        for key in surf_b:
            assert surf_p[key][0] == pytest.approx(surf_b[key][0], rel=RTOL, abs=1e-8)
            assert surf_p[key][1] == pytest.approx(surf_b[key][1], rel=RTOL, abs=1e-10)

    def test_triplet_with_higher_orders_raises(self):
        from gliquid.solution import RKPolyExp, SolutionModel, t_sym

        with pytest.raises(ValueError, match="order-0"):
            SolutionModel(
                ("Hf", "Ti", "Zr"),
                (0 * t_sym,) * 3,
                {
                    "Hf-Ti": RKPolyExp("linear"),
                    "Ti-Zr": RKPolyExp("linear"),
                    "Zr-Hf": RKPolyExp("linear"),
                    "Hf-Ti-Zr": RKPolyExp("linear"),
                },
            )
