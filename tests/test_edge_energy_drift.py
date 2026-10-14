"""Energy drift between the binary edges and the ternary DFT hull.

``cache.entry_energies_by_id`` and ``cache.entry_energy_drift`` compare entries across stores by
canonical id; ``TernaryLiquidInterpolation`` warns once when an entry the binary edges and the
ternary hull share carries two different energies per atom.
"""

import logging

import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
from gliquid.binary import BinaryLiquid
from gliquid.cache import (
    IMPUTED_ID_PREFIX,
    USER_ID_PREFIX,
    canonical_entry_id,
    entry_energies_by_id,
    entry_energy_drift,
)
from gliquid.phase import UNARY, Phase
from gliquid.ternary import TernaryLiquidInterpolation

NEW_API_ID = {"identifier": "mp-aaaaacap", "suffix": "GGA", "separator": "-"}
CLASSIC_ID = "mp-1367-GGA"
SHIFT_EV = 0.0426


def _payload(formula, energy, entry_id, correction=None, adjustments=None):
    entry = {
        "composition": Composition(formula).as_dict(),
        "energy": energy,
        "entry_id": entry_id,
    }
    if correction is not None:
        entry["correction"] = correction
    if adjustments is not None:
        entry["energy_adjustments"] = adjustments
    return entry


class TestEntryEnergiesById:
    def test_classic_and_new_api_ids_share_a_key(self):
        a = entry_energies_by_id([_payload("Ru2Si3", -10.0, CLASSIC_ID)])
        b = entry_energies_by_id([_payload("Ru2Si3", -10.0, NEW_API_ID)])
        assert list(a) == list(b) == [canonical_entry_id(CLASSIC_ID)]

    def test_energy_is_per_atom_and_includes_the_correction(self):
        energies = entry_energies_by_id([_payload("Ru2Si3", -10.0, CLASSIC_ID, correction=0.213)])
        assert energies[CLASSIC_ID] == pytest.approx((-10.0 + 0.213) / 5)

    def test_correction_falls_back_to_the_adjustments(self):
        adjustments = [
            {"adj_per_atom": 0.071, "n_atoms": 3.0, "name": "MP2020 anion correction (Si)"}
        ]
        energies = entry_energies_by_id(
            [_payload("Ru2Si3", -10.0, CLASSIC_ID, adjustments=adjustments)]
        )
        assert energies[CLASSIC_ID] == pytest.approx((-10.0 + 0.213) / 5)

    def test_non_dft_and_unidentified_entries_are_skipped(self):
        entries = [
            _payload("Ru2Si3", -10.0, f"{IMPUTED_ID_PREFIX}Si3Ru2"),
            _payload("Ru2Si3", -10.0, f"{USER_ID_PREFIX}Si3Ru2"),
            _payload("Ru2Si3", -10.0, None),
            _payload("RuSi", -9.0, "mp-189-GGA"),
        ]
        assert list(entry_energies_by_id(entries)) == ["mp-189-GGA"]

    def test_pymatgen_entries_are_read_through_energy_per_atom(self):
        entry = ComputedEntry(Composition("Ru2Si3"), -10.0, correction=0.213, entry_id=CLASSIC_ID)
        assert entry_energies_by_id([entry])[CLASSIC_ID] == pytest.approx(entry.energy_per_atom)


class TestEntryEnergyDrift:
    def test_only_shared_ids_beyond_the_tolerance_are_reported(self):
        reference = {"a": -1.0, "b": -2.0, "c": -3.0}
        other = {"a": -1.0 + 5e-7, "b": -2.0 + SHIFT_EV, "d": -9.0}
        assert entry_energy_drift(reference, other) == [("b", -2.0, -2.0 + SHIFT_EV)]

    def test_largest_difference_comes_first(self):
        reference = {"a": 0.0, "b": 0.0, "c": 0.0}
        other = {"a": 0.01, "b": -0.05, "c": 0.02}
        assert [row[0] for row in entry_energy_drift(reference, other)] == ["b", "c", "a"]

    def test_identical_maps_report_nothing(self):
        energies = {"a": -1.0, "b": -2.0}
        assert entry_energy_drift(energies, dict(energies)) == []


@pytest.fixture(scope="module")
def edge_hull():
    hull, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return hull


@pytest.fixture(scope="module")
def ternary_hull():
    hull, _ = api.get_dft_convexhull(["Hf", "Ti", "Zr"], "GGA")
    return hull


def _bl(pair, hull):
    return BinaryLiquid(
        "-".join(pair),
        list(pair),
        component_data=UNARY.component_data(list(pair)),
        dft_ch=hull,
        phases=[Phase(phase_type="liquid", name="L")],
        temp_range=[300, 3000],
        params=[8000.0, -1.5, 3000.0, 0.5],
        param_format="linear",
        tau=8000,
    )


def _ternary(edge_hull):
    pairs = [("Hf", "Ti"), ("Hf", "Zr"), ("Ti", "Zr")]
    return TernaryLiquidInterpolation.from_binaries([_bl(p, edge_hull) for p in pairs])


class TestTernaryWarning:
    def test_from_binaries_records_the_edge_entry_energies(self, edge_hull):
        ternary = _ternary(edge_hull)
        assert ternary._edge_entry_energies == entry_energies_by_id(edge_hull.all_entries)
        assert ternary._edge_entry_energies

    def test_a_reordered_copy_keeps_the_edge_entry_energies(self, edge_hull):
        ternary = _ternary(edge_hull)
        reordered = ternary.with_component_order(["Zr", "Ti", "Hf"])
        assert reordered is not ternary
        assert reordered._edge_entry_energies == ternary._edge_entry_energies

    def test_matching_energies_log_nothing(self, edge_hull, ternary_hull, caplog):
        ternary = _ternary(edge_hull)
        ternary._edge_entry_energies = entry_energies_by_id(ternary_hull.all_entries)
        with caplog.at_level(logging.WARNING, logger="gliquid.ternary"):
            ternary._warn_on_edge_energy_drift(ternary_hull)
        assert not caplog.records

    def test_a_shifted_edge_entry_is_named_in_the_warning(self, edge_hull, ternary_hull, caplog):
        ternary = _ternary(edge_hull)
        energies = entry_energies_by_id(ternary_hull.all_entries)
        shifted = next(key for key in energies if key.startswith("mp-"))
        ternary._edge_entry_energies = {**energies, shifted: energies[shifted] - SHIFT_EV}
        with caplog.at_level(logging.WARNING, logger="gliquid.ternary"):
            ternary._warn_on_edge_energy_drift(ternary_hull)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "1 entries" in message and shifted in message
        assert f"{SHIFT_EV * 96485.33212:+.0f} J/mol-atom" in message

    def test_the_hull_load_warns_once(self, edge_hull, ternary_hull, caplog):
        ternary = _ternary(edge_hull)
        energies = entry_energies_by_id(ternary_hull.all_entries)
        shifted = next(key for key in energies if key.startswith("mp-"))
        ternary._edge_entry_energies = {**energies, shifted: energies[shifted] - SHIFT_EV}
        with caplog.at_level(logging.WARNING, logger="gliquid.ternary"):
            ternary.solid_hull()
            ternary.solid_hull()
        assert len([r for r in caplog.records if "binary edges" in r.getMessage()]) == 1
