"""Tests for first-class solid-solution phases in BinaryLiquid.

Model math is pinned against the dev-code values frozen in
fixtures/ss_characterization_pins.json (same construction as test_ss_characterization.py, but
through the package BinaryLiquid). from_cache runs offline against the Hf-Zr data/ fixtures.

The figure-level half -- ``TestSSPlotter``, which asserts on traces, reserved colors and
legend entries -- is maintainer machinery and lives in
``tests_internal/test_binary_solution_plotting.py``.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import gliquid.api as api
from gliquid.binary import BinaryLiquid, _x_vals
from gliquid.phase import SS_SPACEGROUPS, UNARY, Phase
from gliquid.solution import SolutionModel

PINS = json.loads(
    (Path(__file__).parent / "fixtures" / "ss_characterization_pins.json").read_text()
)

SS_MODELS = {
    "BCC": {
        "omega": {"Hf-Zr": 10000.0},
        "delta_h": {"Hf": 2000.0, "Zr": 4000.0},
        "delta_s": {"Hf": 0.5, "Zr": 1.0},
    },
    "HCP": {
        "omega": {"Hf-Zr": -5000.0},
        "delta_h": {"Hf": 0.0, "Zr": 1000.0},
        "delta_s": {"Hf": 0.0, "Zr": 0.25},
    },
}
X_TEST = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.fixture(scope="module")
def direct_instance():
    """BinaryLiquid built exactly like the dev characterization instance."""
    ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return BinaryLiquid(
        "Hf-Zr",
        ["Hf", "Zr"],
        ss_models=SS_MODELS,
        component_data=UNARY.component_data(["Hf", "Zr"]),
        dft_ch=ch,
        phases=[Phase(phase_type="liquid", name="L")],
        temp_range=[300, 3000],
    )


@pytest.fixture(scope="module")
def ss_from_cache():
    return BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )


class TestModelMathMatchesDevPins:
    def test_h_s(self, direct_instance):
        pin = PINS["pin1"]
        h, s = direct_instance.solid_solution_h_s("BCC", x_vals=X_TEST)
        assert h.tolist() == pytest.approx(pin["BCC_h"], rel=1e-9)
        assert s.tolist() == pytest.approx(pin["BCC_s"], rel=1e-9)
        h2, s2 = direct_instance.solid_solution_h_s("HCP", x_vals=X_TEST)
        assert h2.tolist() == pytest.approx(pin["HCP_h"], rel=1e-9)
        assert s2.tolist() == pytest.approx(pin["HCP_s"], rel=1e-9)

    def test_gibbs(self, direct_instance):
        g = direct_instance.solid_solution_gibbs("BCC", X_TEST, 1500.0)
        assert g.tolist() == pytest.approx(PINS["pin1"]["BCC_g_1500K"], rel=1e-9)

    def test_unknown_phase_raises(self, direct_instance):
        with pytest.raises(KeyError):
            direct_instance.solid_solution_h_s("SIGMA")

    def test_hsx_rows_match_dev_pins(self, direct_instance):
        pin = PINS["pin2"]
        assert [p.name for p in direct_instance.phases] == pin["phase_order"]
        data = direct_instance.to_HSX()
        n = pin["n_x_vals"]
        assert len(data["X"]) == pin["total_rows"]
        assert data["Phase Name"][n : n + 3] + data["Phase Name"][-3:] == pin["phase_names_tail"]
        assert data["H"][n] == pytest.approx(pin["bcc_first_h"])
        assert data["H"][n + n // 2] == pytest.approx(pin["bcc_mid_h"])
        assert data["S"][-1] == pytest.approx(pin["hcp_last_s"])
        assert data["X"][n : 2 * n] == [float(x) for x in _x_vals]


class TestFromCacheSolidSolutions:
    def test_ss_models_loaded_and_reconciled(self, ss_from_cache):
        bl = ss_from_cache
        # BCC/HCP from the element db; FCC via the omegas fallback (neither Hf nor Zr has an
        # FCC polymorph, but the omegas file carries FCC energies for both).
        assert set(bl.ss_models) == {"BCC", "HCP", "FCC"}
        pin = PINS["pin3"]["element-db"]["reconciled"]
        for el in ("Hf", "Zr"):
            assert bl.component_data[el].h_liq == pytest.approx(pin[el]["H_liq"], rel=1e-9)
            assert bl.component_data[el].s_liq == pytest.approx(pin[el]["S_liq"], rel=1e-9)
            # from_unary_db does not reconcile: the liquid reference IS the registry's, untouched.
            assert bl.component_data[el].h_liq == pytest.approx(UNARY[el].h_liq, rel=1e-12)
            assert bl.component_data[el].s_liq == pytest.approx(UNARY[el].s_liq, rel=1e-12)

    def test_covered_structures_are_not_line_compounds(self, ss_from_cache):
        excluded = {SS_SPACEGROUPS[p] for p in ss_from_cache.ss_models}
        for phase in ss_from_cache.phases:
            if not phase.is_solution:
                assert phase.spacegroup_number not in excluded, (
                    f"{phase.name} should be covered by the SS phase, not a line compound"
                )
        names = [p.name for p in ss_from_cache.phases]
        assert names[-1] == "L"
        assert {"BCC", "HCP"} <= set(names)

    def test_hsx_contains_ss_grids(self, ss_from_cache):
        data = ss_from_cache.to_HSX()
        n = len(_x_vals)
        for name in ("BCC", "HCP"):
            rows = [i for i, p in enumerate(data["Phase Name"]) if p == name]
            assert len(rows) == n
            h_expected, s_expected = ss_from_cache.solid_solution_h_s(name)
            assert data["H"][rows[0] : rows[0] + n] == pytest.approx(h_expected.tolist())
            assert data["S"][rows[0] : rows[0] + n] == pytest.approx(s_expected.tolist())

    def test_update_phase_points_populates_ss_phases(self, ss_from_cache):
        ss_from_cache.update_phase_points()
        for phase in ss_from_cache.phases:
            if phase.name in ("BCC", "HCP"):
                assert phase.points, f"no phase points computed for {phase.name}"

    def test_pickle_round_trip(self, ss_from_cache):
        clone = pickle.loads(pickle.dumps(ss_from_cache))
        assert clone.ss_models == ss_from_cache.ss_models
        # eqs are rebuilt from the reconciled component_data on unpickle
        assert clone.component_data["Hf"].h_liq == pytest.approx(
            ss_from_cache.component_data["Hf"].h_liq
        )
        data = clone.to_HSX()
        assert "BCC" in data["Phase Name"] and "HCP" in data["Phase Name"]


class TestIgnoreSs:
    """ss_in_hull / fit_parameters(ignore_ss=True): SS phases leave the hull but stay
    loaded (ss_models) for plotting. Fresh instances per test — the toggle mutates state."""

    @staticmethod
    def _fresh_ss():
        return BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
        )

    def test_toggle_round_trip(self):
        bl = self._fresh_ss()
        bl.update_phase_points()
        assert "BCC" in bl.hsx.phases and "HCP" in bl.hsx.phases

        bl.ss_in_hull = False
        bl.update_phase_points()
        assert "BCC" not in bl.hsx.phases and "HCP" not in bl.hsx.phases
        assert "BCC" not in bl.to_HSX()["Phase Name"]
        for phase in bl.phases:
            if phase.is_solution and phase.name != "L":
                assert phase.points == []

        bl.ss_in_hull = True
        bl.update_phase_points()
        assert "BCC" in bl.hsx.phases
        assert "BCC" in bl.to_HSX()["Phase Name"]
        assert next(p for p in bl.phases if p.name == "BCC").points

    def test_fit_parameters_ignore_ss_runs_ss_free(self):
        bl = self._fresh_ss()
        result = bl.fit_parameters(ignore_ss=True, n_opts=1, max_iter=4)
        assert isinstance(result, list)
        assert bl.ss_in_hull is False
        assert "BCC" not in bl.hsx.phases and "HCP" not in bl.hsx.phases
        assert bl.ss_models  # models stay loaded for plotting


class TestReversedOrderSS:
    """SS models follow the CONSTRUCTION component order (bug: from_ss_model force-sorted,
    mirroring the SS x-axis relative to the liquid for non-alphabetical BinaryLiquids)."""

    def test_from_ss_model_preserves_component_order(self):
        model_a = SolutionModel.from_ss_model(["Hf", "Zr"], SS_MODELS["BCC"])
        model_r = SolutionModel.from_ss_model(["Zr", "Hf"], SS_MODELS["BCC"])
        assert model_a.components == ("Hf", "Zr")
        assert model_r.components == ("Zr", "Hf")
        h_a, s_a = model_a.h_s_grid((X_TEST,), 1500.0)
        h_r, s_r = model_r.h_s_grid((X_TEST,), 1500.0)
        np.testing.assert_allclose(h_r, h_a[::-1], rtol=1e-9)
        np.testing.assert_allclose(s_r, s_a[::-1], rtol=1e-9)

    def test_reversed_bl_ss_h_s_mirrors_the_pinned_values(self, direct_instance):
        ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
        bl_rev = BinaryLiquid(
            "Zr-Hf",
            ["Zr", "Hf"],
            ss_models=SS_MODELS,
            component_data=UNARY.component_data(["Zr", "Hf"]),
            dft_ch=ch,
            phases=[Phase(phase_type="liquid", name="L")],
            temp_range=[300, 3000],
        )
        pin = PINS["pin1"]
        for name, h_key, s_key in (("BCC", "BCC_h", "BCC_s"), ("HCP", "HCP_h", "HCP_s")):
            h_r, s_r = bl_rev.solid_solution_h_s(name, x_vals=X_TEST)
            assert h_r.tolist() == pytest.approx(pin[h_key][::-1], rel=1e-9)
            assert s_r.tolist() == pytest.approx(pin[s_key][::-1], rel=1e-9)
        # the alphabetical instance keeps matching the pins un-mirrored
        h_a, _ = direct_instance.solid_solution_h_s("BCC", x_vals=X_TEST)
        assert h_a.tolist() == pytest.approx(pin["BCC_h"], rel=1e-9)


class TestFlagOffIsUnchanged:
    def test_default_has_no_ss_state(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        assert bl.ss_models == {}
        data = bl.to_HSX()
        assert "BCC" not in data["Phase Name"]  # SS names only appear via ss_models
        # endpoint polymorphs stay as line compounds when the flag is off
        solution_names = [p.name for p in bl.phases if p.is_solution]
        assert solution_names == ["L"]

    def test_flag_off_matches_plain_construction(self):
        a = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        b = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=False, ss_kwargs=None)
        assert a.to_HSX() == b.to_HSX()
        assert [p.name for p in a.phases] == [p.name for p in b.phases]
