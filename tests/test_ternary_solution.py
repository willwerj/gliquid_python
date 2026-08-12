"""Tests for ternary solid-solution surfaces (gliquid.ternary + the n-ary solution loader).

Everything runs offline: from_unary_db edge models need only the unary registry plus the
shipped data/omegas_hcp.json (which carries the Hf-Ti / Hf-Zr / Ti-Zr pairs).
"""

import numpy as np
import pytest
import sympy as sp

import gliquid.solution as sd
import gliquid.ternary as ht
from gliquid.binary import BinaryLiquid
from gliquid.solution import R, SolutionModel, comp_symbols, t_sym
from gliquid.ternary import TernaryLiquidInterpolation

x1_sym, x2_sym = comp_symbols(3)
from gliquid.phase import UNARY, Phase

TERN = ["Hf", "Ti", "Zr"]
PAIRS = ["Hf-Ti", "Hf-Zr", "Ti-Zr"]


@pytest.fixture(scope="module")
def edge_models():
    models = {}
    for pair in PAIRS:
        components = pair.split("-")
        models[pair] = sd.load_solid_solution_models(
            components, UNARY.component_data(components), ref_mode="from_unary_db"
        )
    return models


@pytest.fixture(scope="module")
def tern_models():
    """Single-pass n-ary load over all three elements (replaces the retired 3-edge merge)."""
    return sd.load_solid_solution_models(
        sorted(TERN), UNARY.component_data(sorted(TERN)), ref_mode="from_unary_db"
    )


class TestTernarySSModels:
    def test_phases_and_schema(self, tern_models):
        # Since the 'updated Ti reference' data commit (2f85901), Ti carries an HCP-194
        # polymorph (dH 1463.64 J/mol, T_tr 180 K) above its spacegroup-191 DFT ground state,
        # so BCC and HCP resolve for all three elements straight from the element db. None of
        # Hf/Ti/Zr has an FCC polymorph, but since the tiered-policy rebuild the unary DB's
        # lattice_stabilities block bakes in the omegas FCC energies at build time, so the
        # resolver supplies those references itself instead of the runtime fallback.
        assert set(tern_models) == {"BCC", "HCP", "FCC"}
        for model in tern_models.values():
            assert set(model["omega"]) == set(PAIRS)
            assert set(model["delta_h"]) == set(TERN)
            assert set(model["delta_s"]) == set(TERN)
        assert all("lattice_stability" in tern_models["FCC"]["refs"][el]["source"] for el in TERN)

    def test_uncovered_phase_skipped_per_edge(self, edge_models):
        # Data-reality pin for the from_unary_db resolver: every Hf-Ti-Zr edge carries BCC and
        # HCP from the element db, plus FCC via the omegas fallback.
        for pair in PAIRS:
            assert set(edge_models[pair]) == {"BCC", "HCP", "FCC"}

    def test_missing_pair_omega_stays_edge_only(self, tmp_path):
        # from_unary_db refs resolve BCC+HCP for Hf-Ti-Zr, but this omegas file covers every
        # pair only for BCC -> HCP is skipped with a warning (stays edge-only).
        import json as _json

        data = {
            "omegas": {
                "BCC": {"Hf-Ti": -0.010, "Hf-Zr": -0.012, "Ti-Zr": -0.008},
                "HCP": {"Hf-Zr": -0.005},
            }
        }
        omegas_path = tmp_path / "omegas_partial.json"
        omegas_path.write_text(_json.dumps(data), encoding="utf-8")
        models = sd.load_solid_solution_models(
            sorted(TERN),
            UNARY.component_data(sorted(TERN)),
            omegas_path=omegas_path,
            ref_mode="from_unary_db",
        )
        assert set(models) == {"BCC"}


class TestTernarySSExpressions:
    @pytest.mark.parametrize("scheme", ["linear", "muggianu", "kohler"])
    def test_edge_reduction_matches_binary(self, tern_models, scheme):
        """On each binary edge the ternary surface must equal the binary regular-solution model."""
        model = tern_models["BCC"]
        exprs = SolutionModel.from_ss_model(TERN, model, scheme).expressions()
        g = sp.lambdify([x1_sym, x2_sym, t_sym], exprs["g_liquid"], "numpy")

        # x1 = x_Ti, x2 = x_Zr (sorted ternary: A=Hf, B=Ti, C=Zr); x = fraction of the
        # second (alphabetically later) element of each pair, matching BinaryLiquid's x_b.
        edge_maps = {
            "Hf-Ti": lambda x: (x, 0.0 * x),
            "Hf-Zr": lambda x: (0.0 * x, x),
            "Ti-Zr": lambda x: (1.0 - x, x),
        }
        x = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        for pair, mapper in edge_maps.items():
            components = pair.split("-")
            bl = BinaryLiquid(
                pair,
                components,
                ss_models={
                    "BCC": {
                        "omega": {pair: model["omega"][pair]},
                        "delta_h": {el: model["delta_h"][el] for el in components},
                        "delta_s": {el: model["delta_s"][el] for el in components},
                    }
                },
                component_data=UNARY.component_data(components),
                dft_ch=_dummy_ch(),
                phases=[Phase(phase_type="liquid", name="L")],
                temp_range=[300, 3000],
            )
            g_binary = bl.solid_solution_gibbs("BCC", x, 1500.0)
            x1_e, x2_e = mapper(x)
            g_ternary = np.asarray(g(x1_e, x2_e, 1500.0), dtype=float)
            assert g_ternary == pytest.approx(g_binary.tolist(), rel=1e-9), (
                f"{scheme}: ternary surface does not reduce to the {pair} binary edge"
            )

    def test_scheme_invariance_for_regular_model(self, tern_models):
        """With l1 = 0, linear / Muggianu / Kohler surfaces are identical."""
        model = tern_models["BCC"]
        grids = np.array([[0.2, 0.3], [0.1, 0.1], [0.4, 0.25], [1 / 3, 1 / 3]])
        results = []
        for scheme in ("linear", "muggianu", "kohler"):
            exprs = SolutionModel.from_ss_model(TERN, model, scheme).expressions()
            g = sp.lambdify([x1_sym, x2_sym, t_sym], exprs["g_liquid"], "numpy")
            results.append(np.array([float(g(x1, x2, 1200.0)) for x1, x2 in grids]))
        assert results[0] == pytest.approx(results[1].tolist(), rel=1e-9)
        assert results[0] == pytest.approx(results[2].tolist(), rel=1e-9)

    def test_config_entropy_at_equimolar(self, tern_models):
        model = tern_models["BCC"]
        exprs = SolutionModel.from_ss_model(TERN, model, "linear").expressions()
        s = sp.lambdify([x1_sym, x2_sym, t_sym], exprs["s_liquid"], "numpy")
        x = 1.0 / 3.0
        s_offsets = sum(model["delta_s"][el] * x for el in TERN)
        s_ideal = -R * 3 * (x * np.log(x))
        assert float(s(x, x, 1000.0)) == pytest.approx(s_offsets + s_ideal, rel=1e-9)

    def test_h_surface_t_independent(self, tern_models):
        exprs = SolutionModel.from_ss_model(TERN, tern_models["BCC"], "linear").expressions()
        h = sp.lambdify([x1_sym, x2_sym, t_sym], exprs["h_liquid"], "numpy")
        assert float(h(0.2, 0.3, 500.0)) == pytest.approx(float(h(0.2, 0.3, 2500.0)), rel=1e-12)


def _dummy_ch():
    """Minimal PhaseDiagram for BinaryLiquid direct construction (init_triangle needs one)."""
    import gliquid.api as api

    ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return ch


class TestTernaryInterpolationSS:
    @pytest.fixture(scope="class")
    def interp(self):
        zeros = {pair: [0, 0, 0, 0] for pair in ("Hf-Ti", "Ti-Zr", "Zr-Hf")}
        # from_omegas_file resolves all three phases on every edge (the omegas file's elements
        # block covers Hf/Ti/Zr for BCC/FCC/HCP), exercising the multi-phase surface path.
        ti = TernaryLiquidInterpolation(
            TERN, xs_mix=zeros, solid_solutions=True, ss_kwargs={"ref_mode": "from_omegas_file"}
        )
        ti.interpolate_liquid_surface()
        n_liquid = len(ti.hsx_df)
        ti.append_solid_solution_surfaces()
        ti._n_liquid_rows = n_liquid
        return ti

    def test_ss_rows_appended_per_phase(self, interp):
        counts = interp.hsx_df["Phase Name"].value_counts()
        assert counts["L"] == interp._n_liquid_rows
        for phase in interp.ss_models:
            assert counts[phase] == interp._n_liquid_rows, (
                f"{phase} should cover the same grid as the liquid"
            )
        assert set(interp.ss_models) == {"BCC", "FCC", "HCP"}

    def test_ss_rows_finite_and_schema(self, interp):
        ss_rows = interp.hsx_df[interp.hsx_df["Phase Name"] != "L"]
        assert list(interp.hsx_df.columns) == ["x0", "x1", "S", "H", "Phase Name"]
        assert np.isfinite(ss_rows["H"].to_numpy(dtype=float)).all()
        assert np.isfinite(ss_rows["S"].to_numpy(dtype=float)).all()

    def test_flag_off_is_unchanged(self):
        zeros = {pair: [0, 0, 0, 0] for pair in ("Hf-Ti", "Ti-Zr", "Zr-Hf")}
        ti_off = TernaryLiquidInterpolation(TERN, xs_mix=zeros)
        ti_off.interpolate_liquid_surface()
        assert set(ti_off.hsx_df["Phase Name"].unique()) == {"L"}
        assert ti_off.solid_solutions is False and ti_off.ss_models == {}


class TestTernarySSPlotting:
    def test_init_sys_assigns_fixed_ss_colors(self):
        from gliquid.binary import SS_FIXED_COLORS
        from gliquid.ternary import TLIPlotter

        zeros = {pair: [0, 0, 0, 0] for pair in ("Hf-Ti", "Ti-Zr", "Zr-Hf")}
        ti = TernaryLiquidInterpolation(
            TERN, xs_mix=zeros, solid_solutions=True, ss_kwargs={"ref_mode": "from_omegas_file"}
        )
        ti.interpolate_liquid_surface()
        ti.append_solid_solution_surfaces()
        plotter = TLIPlotter(ti)
        plotter._init_sys()  # colors every phase in hsx_df; SS phases get reserved colors
        for ss_name in plotter.ss_models:
            assert plotter.color_map[ss_name] == SS_FIXED_COLORS[ss_name]
        assert plotter.color_map["L"] == "cornflowerblue"
        ss_rows = plotter.hsx_df[plotter.hsx_df["Phase"] == "BCC"]
        assert (ss_rows["Colors"] == SS_FIXED_COLORS["BCC"]).all()


def test_build_polymorph_transitions_elementref_native():
    """Finding 13: the annotation helper must use ComponentRef/PhaseRef attribute access."""
    bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
    transitions = ht.build_polymorph_transitions(bl)
    assert transitions, "Hf/Zr both have polymorph ladders"
    sample = transitions[0]
    assert {"name", "comp_x_pct", "transition_temp_C", "ground_state_name"} <= set(sample)
    assert all(np.isfinite(t["transition_temp_C"]) for t in transitions)
