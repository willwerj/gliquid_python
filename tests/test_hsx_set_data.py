"""set_data equivalence certificate for the in-place HSX update (structural-refactor
Task 2): swapping new H/S/X rows into a live HSX must be indistinguishable — exactly,
not approximately — from constructing a fresh HSX with the same data, and
BinaryLiquid.update_phase_points must reuse its HSX across parameter updates.
Runs offline on the shipped Cu-Mg cache.
"""

import numpy as np
import pandas as pd

from gliquid.binary import BinaryLiquid
from gliquid.hsx import HSX

PARAMS_A = [-30000.0, 5.0, -8000.0, 2.0]
PARAMS_B = [15000.0, -3.0, 4000.0, -1.0]


def _fresh_bl(params):
    return BinaryLiquid.from_cache("Cu-Mg", params=params, param_format="linear")


class TestSetDataEquivalence:
    def test_set_data_matches_fresh_construction(self):
        bl_a, bl_b = _fresh_bl(PARAMS_A), _fresh_bl(PARAMS_B)
        names = [p.name for p in bl_a.phases]
        conds = [bl_a.temp_range[0] - 273.15, bl_a.temp_range[-1] - 273.15]
        data_b = bl_b.to_HSX()

        mut = HSX({"data": bl_a.to_HSX(), "phases": names, "comps": bl_a.components}, conds)
        mut.compute_tx()  # populate every derived cache so set_data must reset them all
        mut.set_data(data_b)
        ref = HSX({"data": data_b, "phases": names, "comps": bl_b.components}, conds)

        pd.testing.assert_frame_equal(mut.df, ref.df, check_exact=True)
        assert np.array_equal(mut.points, ref.points)
        assert np.array_equal(mut.liq_points, ref.liq_points)
        assert np.array_equal(mut.inter_points, ref.inter_points)

        out_mut = mut.compute_tx()
        out_ref = ref.compute_tx()
        pd.testing.assert_frame_equal(out_mut[0], out_ref[0], check_exact=True)
        assert np.array_equal(out_mut[1], out_ref[1])  # facet phase labels
        assert np.array_equal(out_mut[2], out_ref[2])  # valid simplices
        assert np.array_equal(out_mut[3], out_ref[3])  # temps
        assert mut.get_phase_points() == ref.get_phase_points()

    def test_update_params_reuses_hsx_and_matches_fresh(self):
        bl = _fresh_bl(PARAMS_A)
        bl.update_phase_points()
        first_hsx = bl.hsx
        bl.update_params(PARAMS_B)
        assert bl.hsx is first_hsx, "update_phase_points should update HSX in place"

        ref = _fresh_bl(PARAMS_B)
        ref.update_phase_points()
        assert bl.hsx.get_phase_points() == ref.hsx.get_phase_points()
        for p_live, p_ref in zip(bl.phases, ref.phases):
            assert p_live.name == p_ref.name
            assert p_live.points == p_ref.points

    def test_phase_list_change_falls_back_to_reconstruction(self):
        bl = _fresh_bl(PARAMS_A)
        bl.update_phase_points()
        first_hsx = bl.hsx
        bl.phases = [p for p in bl.phases if p.name != bl.phases[-1].name]
        bl.update_phase_points()
        assert bl.hsx is not first_hsx, "a changed phase set must rebuild the HSX"
        assert bl.hsx.phases == [p.name for p in bl.phases]
