"""Canonical (alphabetical) disk-cache keys + reversed-frame mirroring (red->green, S4).

Construction order is authoritative for every in-memory object, but ON-DISK cache keys
canonicalize to the alphabetical system name, so 'Zr-Hf' and 'Hf-Zr' hit the SAME cache
files (matrix_data grew 12 duplicate reversed-order dirs before this). MPDS-derived
artifacts (digitized liquidus, low-T phase data) are stored in the json's own
alphabetical frame and mirrored into the construction frame at their consumption
boundaries. Everything runs offline against the shipped Hf-Zr / Al-Mg-Si fixtures.
"""

import shutil
from pathlib import Path

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid

RTOL = 1e-9


def _no_fetch(*args, **kwargs):
    raise AssertionError("cache miss: the live MP API was called")


def _no_mpds_client(*args, **kwargs):
    raise AssertionError("cache miss: the live MPDS client was constructed")


class TestCanonicalCacheKeys:
    def test_dft_cache_reversed_binary_hits_canonical_file(self, tmp_path, monkeypatch):
        src = Path(config.data_dir) / "Hf-Zr_ENTRIES_MP_GGA.json"
        shutil.copy(src, tmp_path / "Hf-Zr_ENTRIES_MP_GGA.json")
        monkeypatch.setattr(api, "_get_dft_entries_from_components", _no_fetch)
        ch, _ = api.get_dft_convexhull(["Zr", "Hf"], "GGA", data_dir=tmp_path)
        # hull axes follow the CALLER's order even though the file key is canonical
        assert [str(el) for el in ch.elements] == ["Zr", "Hf"]
        assert len(ch.stable_entries) >= 2

    def test_dft_cache_reversed_ternary_hits_canonical_file(self, tmp_path, monkeypatch):
        src = Path(config.data_dir) / "Al-Mg-Si_ENTRIES_MP_GGA.json"
        shutil.copy(src, tmp_path / "Al-Mg-Si_ENTRIES_MP_GGA.json")
        monkeypatch.setattr(api, "_get_dft_entries_from_components", _no_fetch)
        ch, _ = api.get_dft_convexhull(["Si", "Al", "Mg"], "GGA", data_dir=tmp_path)
        assert [str(el) for el in ch.elements] == ["Si", "Al", "Mg"]
        assert len(ch.stable_entries) >= 3

    def test_mpds_cache_reversed_input_hits_canonical_file(self, monkeypatch):
        monkeypatch.setattr(api, "get_mpds_client", _no_mpds_client)
        mpds_json, (liq, _) = mpds.load_mpds_data("Zr-Hf", pd_ind=0)
        # raw json stays in its own (alphabetical) frame — consumers mirror at use
        assert mpds_json["chemical_elements"] == ["Hf", "Zr"]
        assert liq, "digitized liquidus should load from the canonical cache file"


class TestReversedFrameMirroring:
    @pytest.fixture(scope="class")
    def bl_pair(self):
        bl_a = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl_r = BinaryLiquid.from_cache("Zr-Hf", pd_ind=0)
        return bl_a, bl_r

    def test_reversed_from_cache_constructs_offline(self, bl_pair):
        bl_a, bl_r = bl_pair
        assert bl_r.components == ["Zr", "Hf"]
        assert bl_r.sys_name == "Zr-Hf"
        assert bl_r.digitized_liq, "reversed construction must find the canonical MPDS cache"

    def test_digitized_liquidus_is_mirrored(self, bl_pair):
        bl_a, bl_r = bl_pair
        expected = [[1 - x, t] for x, t in reversed(bl_a.digitized_liq)]
        assert len(bl_r.digitized_liq) == len(expected)
        for (xr, tr), (xe, te) in zip(bl_r.digitized_liq, expected):
            assert xr == pytest.approx(xe, abs=1e-12)
            assert tr == pytest.approx(te, rel=RTOL)
        # endpoint temperatures swap sides
        assert bl_r.digitized_liq[0][1] == pytest.approx(bl_a.digitized_liq[-1][1], rel=RTOL)
        assert bl_r.digitized_liq[-1][1] == pytest.approx(bl_a.digitized_liq[0][1], rel=RTOL)

    def test_raw_mpds_json_keeps_its_own_frame(self, bl_pair):
        _, bl_r = bl_pair
        assert bl_r.mpds_json["chemical_elements"] == ["Hf", "Zr"]


class TestMirrorHelpers:
    PHASES = [
        {"type": "lc", "name": "A3B", "comp": 0.25, "tbounds": [[0.25, 800.0], [0.25, 1300.0]]},
        {
            "type": "ss",
            "name": "beta",
            "comp": 0.6,
            "cbounds": [[0.5, 900.0], [0.7, 950.0]],
            "tbounds": [[0.6, 700.0], [0.6, 1500.0]],
        },
    ]

    def test_mirror_is_involution_and_flips_axes(self):
        mirrored = mpds.mirror_mpds_phases(self.PHASES)
        assert [p["comp"] for p in mirrored] == [0.4, 0.75]  # re-sorted by comp
        beta = next(p for p in mirrored if p["name"] == "beta")
        assert beta["cbounds"] == [[pytest.approx(0.3), 950.0], [pytest.approx(0.5), 900.0]]
        assert beta["tbounds"][0] == [pytest.approx(0.4), 700.0]
        back = mpds.mirror_mpds_phases(mirrored)
        assert [p["name"] for p in back] == [p["name"] for p in self.PHASES]
        for orig, rt in zip(self.PHASES, back):
            assert rt["comp"] == pytest.approx(orig["comp"])

    def test_frame_check(self):
        j = {"chemical_elements": ["Hf", "Zr"]}
        assert mpds.mpds_frame_matches(j, ["Hf", "Zr"])
        assert not mpds.mpds_frame_matches(j, ["Zr", "Hf"])
        assert mpds.mpds_frame_matches({}, ["Zr", "Hf"])  # no frame info -> trust caller


class TestLowTempPhaseDataFrame:
    """get_low_temp_phase_data normalizes its MPDS side to the hull's component frame."""

    @staticmethod
    def _synthetic_json():
        return {
            "reference": {"entry": "synthetic"},
            "chemical_elements": ["Hf", "Zr"],
            "temp": [400.0, 1600.0],
            "comp_range": [0.0, 100.0],
            "shapes": [
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "L",
                    "kind": "boundary",
                    "svgpath": "0,1400 25,1300 50,1200 75,1350 100,1500",
                },
                {
                    "nphases": 1,
                    "is_solid": True,
                    "label": "HfZr3 rt",
                    "kind": "compound",
                    "svgpath": "30,500 30,1450",
                },
            ],
        }

    def test_mpds_side_follows_hull_frame(self, tmp_path, monkeypatch):
        src = Path(config.data_dir) / "Hf-Zr_ENTRIES_MP_GGA.json"
        shutil.copy(src, tmp_path / "Hf-Zr_ENTRIES_MP_GGA.json")
        monkeypatch.setattr(api, "_get_dft_entries_from_components", _no_fetch)
        ch_a, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA", data_dir=tmp_path)
        ch_r, _ = api.get_dft_convexhull(["Zr", "Hf"], "GGA", data_dir=tmp_path)
        j = self._synthetic_json()

        (cong_a, incong_a, tmax_a), _ = mpds.get_low_temp_phase_data(j, ch_a)
        (cong_r, incong_r, tmax_r), _ = mpds.get_low_temp_phase_data(j, ch_r)

        all_a = {**cong_a, **incong_a}
        all_r = {**cong_r, **incong_r}
        assert all_a, "synthetic phase should be picked up"
        assert set(all_a) == set(all_r)
        assert tmax_r == pytest.approx(tmax_a, rel=RTOL)
        for name, ((c1, c2), t) in all_a.items():
            (r1, r2), tr = all_r[name]
            assert (r1, r2) == (pytest.approx(1 - c2), pytest.approx(1 - c1))
            assert tr == pytest.approx(t, rel=RTOL)

    def test_component_phases_are_not_melting_compounds(self, tmp_path, monkeypatch):
        """A pure component is neither congruent nor incongruent, and must not set the
        temperature scale either. '(Zr) rt' here is the frame edge spanning the full
        temperature axis -- exactly Cr-Ti's defect -- so leaving it in would pin
        max_phase_temp to the top of the diagram and squash the real compound bars."""
        src = Path(config.data_dir) / "Hf-Zr_ENTRIES_MP_GGA.json"
        shutil.copy(src, tmp_path / "Hf-Zr_ENTRIES_MP_GGA.json")
        monkeypatch.setattr(api, "_get_dft_entries_from_components", _no_fetch)
        ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA", data_dir=tmp_path)

        j = self._synthetic_json()
        (cong_ref, incong_ref, tmax_ref), _ = mpds.get_low_temp_phase_data(j, ch)

        j_with_component = self._synthetic_json()
        j_with_component["shapes"].append(
            {
                "nphases": 1,
                "is_solid": True,
                "label": "(Zr) rt",
                "kind": "compound",
                "svgpath": "100,450 100,1590",
            }
        )
        (cong, incong, tmax), _ = mpds.get_low_temp_phase_data(j_with_component, ch)

        assert "(Zr)" not in {**cong, **incong}
        assert (set(cong), set(incong)) == (set(cong_ref), set(incong_ref))
        assert tmax == pytest.approx(tmax_ref, rel=RTOL)
