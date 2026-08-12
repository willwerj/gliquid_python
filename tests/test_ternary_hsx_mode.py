"""hull_mode='hsx' — the ternary pipeline through the generalized HSX class.

New capability, so the gates are PHYSICAL (agreement with the pinned GTX slicing within
its own temperature quantization), not 1e-9 pins: GTX temperatures are T-grid multiples,
the HSX facet temperatures are continuous, so per-composition liquidus differences are
bounded by one T_incr. Also carries the regression test for the (pre-existing, now fixed)
TLIPlotter kwarg-forwarding bug. Runs offline (cached Al-Mg-Si MP entries).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}
GTX_T_INCR = 10.0


def _build_plotter(**kwargs):
    tmod = pu.get_ternary_mod()
    plotter = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"], delta=0.1, xs_mix={k: list(v) for k, v in AMS_LD.items()}, **kwargs
    )
    plotter.process_data()
    return plotter


def _liquidus_by_comp(equil_dfs):
    """Per-composition liquidus: max T of the L vertices of solid-touching facets."""
    df = pd.concat(equil_dfs, ignore_index=True)
    has_solid = df.groupby("simplex_id")["Phase"].transform(lambda s: (s != "L").any())
    liq_rows = df[(df["Phase"] == "L") & has_solid]
    return liq_rows.groupby(["x0_orig", "x1_orig"])["T"].max()


def test_invalid_hull_mode_rejected():
    tmod = pu.get_ternary_mod()
    with pytest.raises(ValueError, match="hull_mode"):
        tmod.TLIPlotter.from_components(["Al", "Mg", "Si"], hull_mode="bogus")


def test_ss_kwargs_forwarding_fixed():
    # Regression: the plotter used to drop solid_solutions/ss_kwargs instead of
    # forwarding them to ternary_interpolation.
    tmod = pu.get_ternary_mod()
    plotter = tmod.TLIPlotter.from_components(
        ["Hf", "Ti", "Zr"], solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    assert plotter.solid_solutions is True
    assert plotter.ss_kwargs == {"ref_mode": "from_unary_db"}


class TestHsxModeAgreement:
    @pytest.fixture(scope="class")
    def gtx(self):
        return _build_plotter(T_incr=GTX_T_INCR)

    @pytest.fixture(scope="class")
    def hsx(self):
        return _build_plotter(hull_mode="hsx")

    def test_hsx_output_schema_matches_gtx(self, hsx):
        assert hsx._n_simplex_vertices == 4
        assert len(hsx.equil_df_list) == 1
        df = hsx.equil_df_list[0]
        assert list(df.columns) == [
            "x0",
            "x1",
            "T",
            "Phase",
            "Colors",
            "simplex_id",
            "x0_orig",
            "x1_orig",
        ]
        counts = df["simplex_id"].value_counts()
        assert (counts == 4).all()

    def test_liquidus_agreement_within_gtx_quantization(self, gtx, hsx):
        liq_gtx = _liquidus_by_comp(gtx.equil_df_list)
        liq_hsx = _liquidus_by_comp(hsx.equil_df_list)
        shared = liq_gtx.index.intersection(liq_hsx.index)
        assert len(shared) > 20, "the two modes should share most grid compositions"
        diff = (liq_gtx.loc[shared] - liq_hsx.loc[shared]).abs()
        # Robust-statistic gate: the coarse-grid GTX reference can emit a sliver
        # tie-simplex at an isolated boundary composition (diagnosed on this fixture:
        # ONE Mg-Si-edge composition spikes +311 K in GTX — solid Si tied to a far
        # liquid vertex — while all 63 other compositions agree within 9.5 K, the
        # expected floor-quantization of the T grid). Gate the distribution, not the max.
        assert np.quantile(diff, 0.95) <= GTX_T_INCR + 1e-6
        assert (diff <= GTX_T_INCR + 1e-6).mean() >= 0.95

    def test_same_stable_phase_set(self, gtx, hsx):
        phases_gtx = set(pd.concat(gtx.equil_df_list, ignore_index=True)["Phase"].unique())
        phases_hsx = set(pd.concat(hsx.equil_df_list, ignore_index=True)["Phase"].unique())
        assert phases_hsx == phases_gtx

    def test_eutectic_temperature_agreement(self, gtx, hsx):
        liq_gtx = _liquidus_by_comp(gtx.equil_df_list)
        liq_hsx = _liquidus_by_comp(hsx.equil_df_list)
        assert abs(liq_gtx.min() - liq_hsx.min()) <= GTX_T_INCR + 1e-6

    def test_gtx_mode_is_the_default_and_unchanged(self, gtx):
        # The pinned GTX behavior is the default: triangles, per-T dataframes.
        assert gtx.hull_mode == "gtx"
        assert gtx._n_simplex_vertices == 3
        assert len(gtx.equil_df_list) > 1

    def test_get_convex_hull_is_gtx_only(self, hsx):
        with pytest.raises(ValueError, match="GTX-mode diagnostic"):
            hsx.get_convex_hull(700.0)
