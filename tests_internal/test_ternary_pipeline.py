"""Characterization pins for the ternary interpolation + GTX hull pipeline and SS models.

Frozen pre-refactor by dev/scripts/_scratch/freeze_refactor_pins.py into
fixtures/ternary_pipeline_pins.json. Gates: rtol <= 1e-9 numerics; simplex SETS exact.
Runs offline (unary registry + shipped omegas_hcp.json + cached Al-Mg-Si MP entries).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PINS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ternary_pipeline_pins.json"
    ).read_text()
)

TERN_HTZ = ["Hf", "Ti", "Zr"]
ALPHA_PAIRS = ["Hf-Ti", "Hf-Zr", "Ti-Zr"]
ZERO_LD = {
    "Hf-Ti": [0.0, 0.0, 0.0, 0.0],
    "Ti-Zr": [0.0, 0.0, 0.0, 0.0],
    "Zr-Hf": [0.0, 0.0, 0.0, 0.0],
}
NONZERO_LD = {
    "Hf-Ti": [8000.0, -1.5, 3000.0, 0.5],
    "Ti-Zr": [-12000.0, 2.0, -1500.0, 0.25],
    "Zr-Hf": [5000.0, 0.0, 2500.0, -0.75],
}

LIQUID_CONFIGS = {
    "HfTiZr_liquid_zeros_d025": (ZERO_LD, 0.25, "linear", "linear"),
    "HfTiZr_liquid_nonzero_d010_linear": (NONZERO_LD, 0.10, "linear", "linear"),
    "HfTiZr_liquid_nonzero_d025_muggianu": (NONZERO_LD, 0.25, "muggianu", "linear"),
    "HfTiZr_liquid_nonzero_d025_kohler": (NONZERO_LD, 0.25, "kohler", "linear"),
    "HfTiZr_liquid_nonzero_d025_combexp": (NONZERO_LD, 0.25, "linear", "comb-exp"),
}


class TestLiquidSurfacePins:
    @pytest.mark.parametrize("key", sorted(LIQUID_CONFIGS))
    def test_surface(self, key):
        l_dict, delta, interp_type, param_format = LIQUID_CONFIGS[key]
        tmod = pu.get_ternary_mod()
        ti = tmod.TernaryLiquidInterpolation(
            TERN_HTZ,
            xs_mix={k: list(v) for k, v in l_dict.items()},
            delta=delta,
            interp_scheme=interp_type,
            param_format=param_format,
        )
        ti.interpolate_liquid_surface()
        pin = PINS[key]
        df = ti.hsx_df
        pu.assert_deep_approx(pin["x0"], df["x0"].to_numpy(dtype=float))
        pu.assert_deep_approx(pin["x1"], df["x1"].to_numpy(dtype=float))
        pu.assert_deep_approx(pin["H"], df["H"].to_numpy(dtype=float))
        pu.assert_deep_approx(pin["S"], df["S"].to_numpy(dtype=float))
        assert df["Phase Name"].tolist() == pin["phase"]


class TestSSModelPins:
    @pytest.mark.parametrize("ref_mode", ["from_unary_db", "from_omegas_file"])
    def test_edge_models(self, ref_mode):
        sd = pu.get_solution_mod()
        from gliquid.phase import UNARY

        pinned = PINS[f"ss_{pu.FIXTURE_REF_MODE[ref_mode]}"]["edge_models"]
        for pair in ALPHA_PAIRS:
            comps = pair.split("-")
            live = sd.load_solid_solution_models(
                comps, UNARY.component_data(comps), ref_mode=ref_mode
            )
            assert set(live) == set(pinned[pair]), f"{ref_mode}/{pair}: phase set changed"
            for phase, model in live.items():
                pu.assert_deep_approx(
                    pu.canonical_ss_model(pinned[pair][phase], comps),
                    pu.canonical_ss_model(model, comps),
                )

    @pytest.mark.parametrize("ref_mode", ["from_unary_db", "from_omegas_file"])
    def test_merged_ternary_models(self, ref_mode):
        """Single-pass N=3 loading (post-refactor) must equal the retired 3-edge merge
        (frozen pre-refactor)."""
        sd = pu.get_solution_mod()
        from gliquid.phase import UNARY

        pinned = PINS[f"ss_{pu.FIXTURE_REF_MODE[ref_mode]}"]["merged_ternary"]
        if hasattr(sd, "build_ternary_ss_models"):
            edge_models = {}
            for pair in ALPHA_PAIRS:
                comps = pair.split("-")
                edge_models[pair] = sd.load_solid_solution_models(
                    comps, UNARY.component_data(comps), ref_mode=ref_mode
                )
            live = sd.build_ternary_ss_models(TERN_HTZ, edge_models)
        else:
            live = sd.load_solid_solution_models(
                sorted(TERN_HTZ), UNARY.component_data(sorted(TERN_HTZ)), ref_mode=ref_mode
            )
        assert set(live) == set(pinned), f"{ref_mode}: merged phase set changed"
        for phase, model in live.items():
            pu.assert_deep_approx(
                pu.canonical_ss_model(pinned[phase], TERN_HTZ),
                pu.canonical_ss_model(model, TERN_HTZ),
            )


AMS_PIN = PINS["AlMgSi_gtx_d010_T100"]
AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}


@pytest.fixture(scope="module")
def ams_plotter():
    tmod = pu.get_ternary_mod()
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"], delta=0.1, T_incr=100, xs_mix={k: list(v) for k, v in AMS_LD.items()}
    )
    plotter.process_data()
    return plotter


class TestAlMgSiGtxPipeline:
    def test_grid_and_sizes(self, ams_plotter):
        pu.assert_deep_approx(AMS_PIN["T_grid"], np.asarray(ams_plotter.T_grid, dtype=float))
        assert len(ams_plotter.equil_df_list) == AMS_PIN["n_equil_dfs"]
        assert len(ams_plotter.hsx_df) == AMS_PIN["hsx_df_len"]
        assert ams_plotter.phase_names == AMS_PIN["phases"]

    def test_equil_slices(self, ams_plotter):
        for idx_str, blk in AMS_PIN["equil_slices"].items():
            df = ams_plotter.equil_df_list[int(idx_str)]
            live = df.drop(columns=["Colors"])
            assert list(live.columns) == [c for c in blk["columns"] if c != "Colors"]
            pu.assert_deep_approx(blk["values"], live.values)

    def test_hull_input_slices(self, ams_plotter):
        """The exact per-T hull inputs and their simplex sets — the live-data golden for
        the absorbed lower-hull function."""
        hull_fn = pu.get_lower_hull_fn()
        t_by_key = {f"{t:.1f}": float(t) for t in np.asarray(ams_plotter.T_grid, dtype=float)}
        for t_key, blk in AMS_PIN["hull_input_slices"].items():
            live_pts = np.array(ams_plotter.df_Tgroups[t_by_key[t_key]][["x0", "x1", "G"]])
            pu.assert_deep_approx(blk["points"], live_pts)
            got = hull_fn(np.array(blk["points"], dtype=float), vertical_simplices=False)
            assert pu.canonical_simplices(got) == pu.canonical_simplices(blk["simplices"])


class TestInterMeltingTemps:
    """Characterization for get_inter_melting_temps (secondary-track S1; zero prior coverage).

    Runs after get_plot('tx') — today the method needs _plot_tx state to have been built;
    the executable spec below (max T per phase over the concatenated equilibrium slices)
    is the behavior contract that must survive the processing→TLI move.
    """

    def test_matches_max_temp_per_phase(self, ams_plotter):
        ams_plotter.get_plot("tx")
        solids = [p for p in ams_plotter.phase_names if p != "L"]
        got = ams_plotter.get_inter_melting_temps(solids)
        concat = pd.concat(ams_plotter.equil_df_list, ignore_index=True)
        expected = {}
        for phase in solids:
            sub = concat[concat["Phase"] == phase]
            if not sub.empty:
                expected[phase] = sub["T"].max()
        assert got == expected
        assert got, "spec test is vacuous — no solid phase present in the hull slices"

    def test_unknown_phase_raises_valueerror(self, ams_plotter):
        ams_plotter.get_plot("tx")
        with pytest.raises(ValueError, match="not found in the system phases"):
            ams_plotter.get_inter_melting_temps(["NotAPhase"])
