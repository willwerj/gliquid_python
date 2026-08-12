"""Characterization pins for the binary HSX/hull pipeline and the liquid-expression builder.

Frozen pre-refactor by dev/scripts/_scratch/freeze_refactor_pins.py into
fixtures/hull_numerics_pins.json. These are the behavior-preservation gates of the
nd-reductionist-refactor: hull simplex SETS exact, all numerics at rtol <= 1e-9.
Everything runs offline against the shipped Hf-Zr / Cu-Mg data fixtures.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gliquid.binary import BinaryLiquid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PINS = json.loads((Path(__file__).parent / "fixtures" / "hull_numerics_pins.json").read_text())

LIQ_PARAM_SETS = (
    [0.0, 0.0, 0.0, 0.0],
    [-30000.0, 5.0, -8000.0, 2.0],
    [15000.0, -3.0, 4000.0, -1.0],
)
LIQ_TEMPS = (600.0, 1200.0, 2400.0)


class TestLowerHullGoldens:
    @pytest.mark.parametrize("name", ["coarse_noisy", "ternary_simplex", "degenerate"])
    def test_simplices_exact(self, name):
        blk = PINS["gliq_lowerhull3_goldens"][name]
        hull_fn = pu.get_lower_hull_fn()
        pts = np.array(blk["points"], dtype=float)
        got_nv = hull_fn(pts.copy(), vertical_simplices=False)
        assert pu.canonical_simplices(got_nv) == pu.canonical_simplices(
            blk["simplices_no_vertical"]
        )
        got_wv = hull_fn(pts.copy(), vertical_simplices=True)
        assert pu.canonical_simplices(got_wv) == pu.canonical_simplices(
            blk["simplices_with_vertical"]
        )


@pytest.fixture(scope="module")
def bl_hfzr():
    return BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)


@pytest.fixture(scope="module")
def bl_cumg():
    return BinaryLiquid.from_cache(
        "Cu-Mg", params=[-30000.0, 5.0, -8000.0, 2.0], param_format="linear"
    )


@pytest.fixture(scope="module")
def bl_ss():
    return BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )


def _check_hsx_pipeline(bl, pin, include_to_hsx):
    bl.update_phase_points()
    hsx = bl.hsx
    assert pu.canonical_simplices(hsx.hull()) == pu.canonical_simplices(pin["hull_simplices"])
    df_tx, _final_phases, _valid, temps = hsx.compute_tx()
    assert list(df_tx.columns) == pin["df_tx_columns"]
    pu.assert_deep_approx(pin["df_tx_values"], df_tx.values)
    pu.assert_deep_approx(pin["temps"], np.asarray(temps))
    inv_points, _combined, counts = hsx.liquidus_invariants()
    pu.assert_deep_approx(pin["invariant_points"], inv_points)
    pu.assert_deep_approx(pin["invariant_counts"], counts)
    pu.assert_deep_approx(pin["phase_points"], hsx.get_phase_points())
    if include_to_hsx:
        pu.assert_deep_approx(pin["to_hsx"], bl.to_HSX())


def _check_liquid_builder(eqs, pin_block):
    x_inner = pu.get_x_vals()[1:-1]
    for T in LIQ_TEMPS:
        for i, params in enumerate(LIQ_PARAM_SETS):
            key = f"T{int(T)}_p{i}"
            h = np.broadcast_to(
                np.asarray(eqs["h_liq_lambdified"](x_inner, T, *params), dtype=float), x_inner.shape
            )
            s = np.broadcast_to(
                np.asarray(eqs["s_liq_lambdified"](x_inner, T, *params), dtype=float), x_inner.shape
            )
            pu.assert_deep_approx(pin_block[key]["h"], h)
            pu.assert_deep_approx(pin_block[key]["s"], s)


class TestHfZrPipeline:
    def test_hsx_pipeline(self, bl_hfzr):
        _check_hsx_pipeline(bl_hfzr, PINS["HfZr"], include_to_hsx=True)

    def test_liquid_builder_linear(self, bl_hfzr):
        _check_liquid_builder(bl_hfzr.eqs, PINS["HfZr"]["liquid_builder"])

    def test_liquid_builder_combexp_default_tau(self, bl_hfzr):
        _, t_sym, _ = pu.get_builder_and_symbols()
        eqs = pu.build_combexp_eqs(
            bl_hfzr.component_data["Hf"].gibbs_ref_expr(t_sym),
            bl_hfzr.component_data["Zr"].gibbs_ref_expr(t_sym),
            tau=8000,
        )
        _check_liquid_builder(eqs, PINS["HfZr_combexp_tau8000"]["liquid_builder"])

    def test_liquid_builder_combexp_tau3000(self, bl_hfzr):
        """The tau certificate: kwarg (post-refactor) must reproduce what the retired
        module-global monkeypatch produced (frozen pre-refactor)."""
        _, t_sym, _ = pu.get_builder_and_symbols()
        eqs = pu.build_combexp_eqs(
            bl_hfzr.component_data["Hf"].gibbs_ref_expr(t_sym),
            bl_hfzr.component_data["Zr"].gibbs_ref_expr(t_sym),
            tau=3000,
        )
        _check_liquid_builder(eqs, PINS["HfZr_combexp_tau3000"]["liquid_builder"])


class TestCuMgPipeline:
    def test_hsx_pipeline(self, bl_cumg):
        _check_hsx_pipeline(bl_cumg, PINS["CuMg"], include_to_hsx=False)


class TestTernaryHSXSmoke:
    """The generalized HSX consuming a ternary (x0, x1, S, H) table — new capability."""

    def test_four_vertex_compute_tx(self):
        from gliquid.hsx import HSX

        tmod = pu.get_ternary_mod()
        ti = tmod.TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"],
            xs_mix={"Hf-Ti": [0.0] * 4, "Ti-Zr": [0.0] * 4, "Zr-Hf": [0.0] * 4},
            delta=0.25,
        )
        ti.interpolate_liquid_surface()
        table = ti.hsx_df[["x0", "x1", "S", "H", "Phase Name"]]
        hsx = HSX(
            {"data": table, "phases": ["L"], "comps": ["Hf", "Ti", "Zr"]}, conds=[0.0, 2400.0]
        )
        df_tx, _phases, simplices, temps = hsx.compute_tx()
        assert simplices.shape[1] == 4  # 4-vertex facets in (x0, x1, S, H)
        assert list(df_tx.columns) == ["x0", "x1", "t", "label"]
        assert len(temps) > 0 and np.isfinite(temps).all()
        with pytest.raises(NotImplementedError):
            hsx.liquidus_invariants()
        with pytest.raises(NotImplementedError):
            hsx.get_phase_points()

    def test_wrong_column_count_raises(self):
        from gliquid.hsx import HSX

        bad = {
            "data": {"X": [0.0, 1.0], "S": [0.0, 0.0], "H": [0.0, 0.0], "Phase Name": ["L", "L"]},
            "phases": ["L"],
            "comps": ["Hf", "Ti", "Zr"],
        }
        with pytest.raises(ValueError, match="positional columns"):
            HSX(bad, conds=[0.0, 2400.0])


class TestSolidSolutionPins:
    def test_ss_model_values(self, bl_ss):
        pinned = PINS["HfZr_ss"]["ss_model_dicts"]
        assert set(bl_ss.ss_models) == set(pinned)
        for phase, model in bl_ss.ss_models.items():
            pu.assert_deep_approx(
                pu.canonical_ss_model(pinned[phase], ["Hf", "Zr"]),
                pu.canonical_ss_model(model, ["Hf", "Zr"]),
            )

    def test_h_s_grids(self, bl_ss):
        for phase, blk in PINS["HfZr_ss"]["h_s_grids"].items():
            h, s = bl_ss.solid_solution_h_s(phase)
            pu.assert_deep_approx(blk["h"], h)
            pu.assert_deep_approx(blk["s"], s)

    def test_gibbs_grid(self, bl_ss):
        x = pu.get_x_vals()
        for phase, pinned in PINS["HfZr_ss"]["gibbs_1500K"].items():
            pu.assert_deep_approx(pinned, bl_ss.solid_solution_gibbs(phase, x, 1500.0))

    def test_reconciled_liquid_refs(self, bl_ss):
        for el, (h_pin, s_pin) in PINS["HfZr_ss"]["reconciled_liquid_refs"].items():
            pu.assert_deep_approx(
                [h_pin, s_pin], [bl_ss.component_data[el].h_liq, bl_ss.component_data[el].s_liq]
            )

    def test_hsx_pipeline(self, bl_ss):
        _check_hsx_pipeline(bl_ss, PINS["HfZr_ss"], include_to_hsx=True)
