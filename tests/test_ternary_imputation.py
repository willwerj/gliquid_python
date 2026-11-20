"""Ternary phase-energy imputation (``gliquid.imputation.ternary_imputation``).

The headline property: a phase imputed at temperature T leaves the processed ternary hull at T.
Checked in both hull backends, with a control that moves the energy and must move the
temperature, and as a round trip -- a phase injected at a known enthalpy, its drawn top read
off, the phase removed and imputed back at that top, must recover the enthalpy.

Runs offline against the shipped Al-Mg-Si entries record, which has no stable ternary
compound, so every imputed phase is synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.config as config
import gliquid.imputation.core as imp_core
import gliquid.imputation.lifecycle as imp_lifecycle
import gliquid.imputation.ternary_imputation as imp_ternary
import gliquid.ternary as ternary_module
from gliquid.binary import build_phases_from_chull
from gliquid.cache import (
    ENTROPY_KEY,
    CacheModeError,
    DirectoryBackend,
    canonical_entry_id,
    imputed_entry_label,
)
from gliquid.phase import EV_ATOM_TO_J_MOL, UNARY
from gliquid.plotting import style
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

COMPONENTS = ["Al", "Mg", "Si"]
IDEAL = {"Al-Mg": [0.0] * 4, "Mg-Si": [0.0] * 4, "Si-Al": [0.0] * 4}
MIXING = {"Al-Mg": [-20000.0, 0, 0, 0], "Mg-Si": [-30000.0, 0, 0, 0], "Si-Al": [-10000.0, 0, 0, 0]}
COMB_EXP = {
    "Al-Mg": [-20000.0, -10.0, 0, 0],
    "Mg-Si": [-60000.0, -20.0, 0, 0],
    "Si-Al": [-10000.0, -5.0, 0, 0],
}
TARGET = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
MG2SI = {"Al": 0.0, "Mg": 2 / 3, "Si": 1 / 3}
DELTA = 0.1
T_INCR = 5


def build(mode="hsx", xs_mix=IDEAL, param_format="linear", **kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(COMPONENTS),
        delta=DELTA,
        xs_mix=xs_mix,
        param_format=param_format,
        order="given",
        hull_mode=mode,
        T_incr=T_INCR,
        **kwargs,
    )


def top_k(ti, name) -> float:
    ti.process_data()
    return float(ti.get_inter_melting_temps([name])[name]) + 273.15


def probe_entry(ti, fractions, h_form_j, name="TX") -> dict:
    """A tagged entry at ``h_form_j`` on ``ti``'s DFT hull."""
    hull = ti.solid_hull([])
    energy = imp_core.hull_ref_energy_per_atom(hull, fractions) + h_form_j / EV_ATOM_TO_J_MOL
    from pymatgen.core import Composition

    return imp_core.tagged_entry_dict(
        Composition(fractions),
        energy,
        name,
        {"imputed": True, "label": name, "fractions": fractions, "h_form_j_per_mol": h_form_j},
    )


def shipped_record() -> Path:
    """The shipped Al-Mg-Si entries record; skips without a directory-mode corpus holding it."""
    source = config.cache_dir
    if source is None or config.cache_mode != "directory":
        pytest.skip("needs a directory-mode cache corpus")
    record = Path(source) / "Al-Mg-Si_ENTRIES_MP_GGA.json"
    if not record.exists():
        pytest.skip(f"no shipped {record.name}")
    return record


@pytest.fixture
def store(tmp_path) -> Path:
    """A flat directory store holding a copy of the shipped Al-Mg-Si entries record."""
    record = shipped_record()
    shutil.copy2(record, tmp_path / record.name)
    return tmp_path


# -- The phase leaves the hull at the requested temperature --


class TestRealizedTemperature:
    @pytest.mark.parametrize("temp", [900.0])
    def test_hsx(self, no_persist, temp):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, temp, name="TX")
        assert record.accepted and record.code == "imputed", record.reason
        assert record.melts is True
        assert abs(record.realized_temperature_k - temp) <= imp_ternary.HSX_TOLERANCE_K

    def test_hsx_with_a_liquid_nonlinear_in_temperature(self, no_persist):
        ti = build("hsx", xs_mix=COMB_EXP, param_format="comb-exp")
        record = ti.impute_phase(TARGET, 800.0, name="TX")
        assert record.accepted and record.code == "imputed", record.reason
        assert abs(record.realized_temperature_k - 800.0) <= imp_ternary.HSX_TOLERANCE_K

    @pytest.mark.parametrize("temp", [1100.0])
    def test_gtx(self, no_persist, temp):
        ti = build("gtx")
        record = ti.impute_phase(TARGET, temp, name="TX")
        assert record.accepted and record.code == "imputed", record.reason
        assert temp - T_INCR < record.realized_temperature_k <= temp + 1e-6

    def test_control_a_shallower_energy_melts_lower(self, no_persist):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, 1100.0, name="TX")
        shifted = build("hsx", imputed_entries=[probe_entry(ti, TARGET, record.enthalpy + 500.0)])
        assert top_k(shifted, "TX") < record.realized_temperature_k - 10.0

    def test_hotter_means_deeper(self, no_persist):
        cold = build("hsx").impute_phase(TARGET, 900.0, name="TX", apply=False)
        hot = build("hsx").impute_phase(TARGET, 1100.0, name="TX", apply=False)
        assert hot.enthalpy < cold.enthalpy

    def test_composition_off_the_rounding_grid(self, no_persist):
        sevenths = {"Al": 1 / 7, "Mg": 2 / 7, "Si": 4 / 7}
        record = build("hsx").impute_phase(sevenths, 1000.0, name="TX")
        assert record.accepted, record.reason
        assert abs(record.realized_temperature_k - 1000.0) <= imp_ternary.HSX_TOLERANCE_K

    def test_plotter_imputes_onto_its_own_model(self, no_persist):
        plotter = TLIPlotter.from_components(
            COMPONENTS,
            delta=DELTA,
            xs_mix=IDEAL,
            order="given",
            hull_mode="hsx",
            param_format="linear",
        )
        record = plotter.impute_phase(TARGET, 1000.0, name="TX")
        assert record.accepted
        assert [imputed_entry_label(e) for e in plotter._ti.imputed_entries] == ["TX"]

    def test_the_liquidus_partition_is_rebuilt_only_after_an_imputation(
        self, no_persist, monkeypatch
    ):
        plotter = TLIPlotter.from_components(
            COMPONENTS,
            delta=DELTA,
            xs_mix=IDEAL,
            order="given",
            hull_mode="hsx",
            param_format="linear",
        )
        builds = []
        real = TLIPlotter._build_partition

        def counted(self, triangles):
            builds.append(len(triangles))
            return real(self, triangles)

        monkeypatch.setattr(TLIPlotter, "_build_partition", counted)
        plotter.get_plot("tx")
        plotter.get_plot("tx")
        assert len(builds) == 1
        plotter.impute_phase(TARGET, 1000.0, name="TX")
        plotter.get_plot("tx")
        assert len(builds) == 2

    @pytest.mark.parametrize("editor", ["model", "other_plotter"])
    def test_the_liquidus_partition_follows_edits_not_made_through_this_plotter(
        self, no_persist, monkeypatch, editor
    ):
        plotter = TLIPlotter.from_components(
            COMPONENTS,
            delta=DELTA,
            xs_mix=IDEAL,
            order="given",
            hull_mode="hsx",
            param_format="linear",
        )
        other = TLIPlotter(plotter._ti, order="given")
        builds = []
        real = TLIPlotter._build_partition

        def counted(self, triangles):
            builds.append(self)
            return real(self, triangles)

        monkeypatch.setattr(TLIPlotter, "_build_partition", counted)
        plotter.get_plot("tx")
        assert builds == [plotter]
        (plotter._ti if editor == "model" else other).impute_phase(TARGET, 1000.0, name="TX")
        plotter.get_plot("tx")
        assert builds == [plotter, plotter]
        assert "TX" in plotter._liquidus_partition(plotter.triangulation.simplices).phases


class TestRoundTrip:
    @pytest.mark.parametrize(
        "h0,xs_mix",
        [(-25000.0, IDEAL), (-40000.0, MIXING)],
        ids=["-25000.0-ideal", "-40000.0-mixing"],
    )
    def test_injected_enthalpy_is_recovered_from_its_own_top(self, no_persist, xs_mix, h0):
        injected = build("hsx", xs_mix=xs_mix)
        injected.imputed_entries = [probe_entry(injected, TARGET, h0)]
        t0 = top_k(injected, "TX")
        record = build("hsx", xs_mix=xs_mix).impute_phase(TARGET, t0, name="TX", apply=False)
        assert record.accepted, record.reason
        assert record.enthalpy == pytest.approx(h0, abs=10.0)


# -- Rejections --


class TestRejections:
    def test_below_the_solidus(self, no_persist):
        record = build("hsx").impute_phase(TARGET, 300.0, name="TX")
        assert (record.accepted, record.code) == (False, "below_solidus")

    def test_a_dft_phase_already_there(self, no_persist):
        record = build("hsx").impute_phase(MG2SI, 1200.0, name="TX")
        assert (record.accepted, record.code) == (False, "dft_covers")

    def test_a_name_already_in_use(self, no_persist):
        record = build("hsx").impute_phase(TARGET, 1000.0, name="Mg2Si")
        assert (record.accepted, record.code) == (False, "name_taken")

    def test_an_energy_above_the_hull(self, no_persist, monkeypatch):
        monkeypatch.setattr(imp_ternary, "_invert_ternary_phase_energy", lambda *a, **k: 5000.0)
        record = build("hsx").impute_phase(TARGET, 1000.0, name="TX")
        assert (record.accepted, record.code) == (False, "unstable")

    def test_rejections_are_listed_on_the_model(self, no_persist):
        ti = build("hsx")
        ti.impute_phase(MG2SI, 1200.0, name="TX")
        assert [name for name, _ in ti.ternary_meta["imputation_rejected"]] == ["TX"]

    def test_invalid_input_raises(self, no_persist):
        with pytest.raises(ValueError):
            build("hsx").impute_phase({"Fe": 1.0}, 1000.0)
        with pytest.raises(ValueError):
            build("hsx").impute_phase(TARGET, 0.0)


SYNTHETIC_ID = "mp-synthetic-1"
SYNTHETIC_FRACTIONS = {"Al": 0.25, "Mg": 0.5, "Si": 0.25}  # Mg2AlSi


@pytest.fixture(scope="session")
def synthetic_record(tmp_path_factory) -> Path:
    """The shipped record plus one untagged stable ternary entry, Mg2AlSi at -20 kJ/mol-atom."""
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    shipped = shipped_record()
    directory = tmp_path_factory.mktemp("synthetic_store")
    record = directory / shipped.name
    shutil.copy2(shipped, record)
    hull = build("hsx", data_dir=directory).solid_hull([])
    per_atom = imp_core.hull_ref_energy_per_atom(hull, SYNTHETIC_FRACTIONS) + (
        -20000.0 / EV_ATOM_TO_J_MOL
    )
    entry = ComputedEntry(Composition("Mg2AlSi"), 4 * per_atom, entry_id=SYNTHETIC_ID)
    raw = json.loads(record.read_text())
    raw.append(entry.as_dict())
    record.write_text(json.dumps(raw))
    return record


@pytest.fixture
def synthetic_store(store, synthetic_record):
    """``store`` with its record replaced by ``synthetic_record``."""
    shutil.copyfile(synthetic_record, store / synthetic_record.name)
    return store


def synthetic_stable(ti) -> bool:
    return SYNTHETIC_ID in {str(e.entry_id) for e in ti.solid_hull().stable_entries}


class TestProtection:
    DISPLACER = {"Al": 0.2, "Mg": 0.45, "Si": 0.35}

    def test_by_default_it_warns_and_the_phase_stays_out(self, no_persist, synthetic_store):
        ti = build("hsx", data_dir=synthetic_store)
        with pytest.warns(
            UserWarning, match=r"TX stable up to 2500 K will displace Mg2AlSi.*force=True"
        ):
            record = ti.impute_phase(self.DISPLACER, 2500.0, name="TX", screen=False)
        assert (record.accepted, record.code) == (False, "displaces_hull")
        assert [n for n in record.displaces if n.startswith("Mg2AlSi")]
        assert ti.imputed_entries == [] and synthetic_stable(ti)

    def test_force_places_it_and_names_what_fell(self, no_persist, synthetic_store):
        ti = build("hsx", data_dir=synthetic_store)
        record = ti.impute_phase(self.DISPLACER, 2500.0, name="TX", screen=False, force=True)
        assert record.accepted and record.code == "imputed"
        assert record.enthalpy == record.inverted_enthalpy
        assert [n for n in record.displaces if n.startswith("Mg2AlSi")]
        assert not synthetic_stable(ti)
        assert record.enthalpy < record.hull_limit_enthalpy < record.hull_enthalpy

    def test_the_hull_limit_force_reports_keeps_what_fell(self, no_persist, synthetic_store):
        ti = build("hsx", data_dir=synthetic_store)
        forced = ti.impute_phase(
            self.DISPLACER, 2500.0, name="TX", screen=False, force=True, apply=False
        )
        kept = ti.impute_phase(
            self.DISPLACER,
            name="TX",
            screen=False,
            enthalpy=forced.hull_limit_enthalpy,
            entropy=0.0,
        )
        assert kept.accepted and kept.displaces == () and kept.hull_limit_enthalpy is None
        assert synthetic_stable(ti)

    def test_an_imputed_phase_is_protected_too(self, no_persist):
        ti = build("hsx")
        assert ti.impute_phase(SYNTHETIC_FRACTIONS, 1200.0, name="P", screen=False).accepted
        with pytest.warns(UserWarning, match="will displace P on"):
            record = ti.impute_phase(self.DISPLACER, 2500.0, name="TX", screen=False)
        assert record.code == "displaces_hull" and record.displaces == ("P",)
        assert [imputed_entry_label(e) for e in ti.imputed_entries] == ["P"]

    def test_control_a_phase_that_displaces_nothing_is_silent(self, no_persist):
        ti = build("hsx")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = ti.impute_phase(TARGET, 1000.0, name="TX", screen=False)
        assert not [w for w in caught if "will displace" in str(w.message)]
        assert record.accepted and record.displaces == ()

    def test_an_edge_phase_is_never_displaced(self, no_persist):
        ti = build("hsx")
        ti.impute_phase({"Al": 0.02, "Mg": 0.65, "Si": 0.33}, 6000.0, name="TX", screen=False)
        assert any(e.composition.reduced_formula == "Mg2Si" for e in ti.solid_hull().stable_entries)

    def test_a_dft_ternary_phase_covers_its_composition(self, no_persist, synthetic_store):
        record = build("hsx", data_dir=synthetic_store).impute_phase(SYNTHETIC_FRACTIONS, 1500.0)
        assert (record.accepted, record.code) == (False, "dft_covers")


class TestDftAblation:
    """A real hull entry, removed and imputed back from its own drawn melting temperature."""

    def test_the_dft_enthalpy_is_recovered(self, no_persist, synthetic_store, store):
        with_it = build("hsx", data_dir=synthetic_store)
        with_it.process_data()
        name = next(n for n in with_it.phase_names if n.startswith("Mg2AlSi"))
        t_melt = float(with_it.get_inter_melting_temps([name])[name]) + 273.15

        record_path = store / "Al-Mg-Si_ENTRIES_MP_GGA.json"
        raw = json.loads(record_path.read_text())
        kept = [e for e in raw if e.get("entry_id") != SYNTHETIC_ID]
        assert len(kept) == len(raw) - 1
        record_path.write_text(json.dumps(kept))

        plain = build("hsx", data_dir=store)
        record = plain.impute_phase(SYNTHETIC_FRACTIONS, t_melt, name="Imputed", apply=False)
        assert record.accepted, record.reason
        assert record.enthalpy == pytest.approx(-20000.0, abs=10.0)


class TestAdjustPhaseTemperature:
    """An existing DFT phase moved to a new top temperature, and moved back."""

    @staticmethod
    def synthetic(store):
        ti = build("hsx", data_dir=store)
        ti.process_data()
        name = next(n for n in ti.phase_names if n.startswith("Mg2AlSi"))
        return ti, name, top_k(ti, name)

    def test_a_dft_phase_moves_to_the_new_temperature(self, no_persist, synthetic_store):
        ti, name, t0 = self.synthetic(synthetic_store)
        record = ti.adjust_phase_temperature(name, t0 - 150.0)
        assert record.accepted, record.reason
        assert record.replaces == (SYNTHETIC_ID,)
        assert record.previous_enthalpy == pytest.approx(-20000.0, abs=1e-6)
        assert record.enthalpy > record.previous_enthalpy
        assert abs(record.realized_temperature_k - (t0 - 150.0)) <= imp_ternary.HSX_TOLERANCE_K
        assert ti.imputed_phase_names() == [name]
        assert SYNTHETIC_ID not in {str(e.entry_id) for e in ti.solid_hull().all_entries}

    def test_control_its_own_temperature_returns_the_dft_enthalpy(
        self, no_persist, synthetic_store
    ):
        ti, name, t0 = self.synthetic(synthetic_store)
        record = ti.adjust_phase_temperature(name, t0, apply=False)
        assert record.accepted, record.reason
        assert record.enthalpy == pytest.approx(-20000.0, abs=10.0)

    def test_dropping_it_restores_the_dft_phase(self, no_persist, synthetic_store):
        ti, name, t0 = self.synthetic(synthetic_store)
        ti.adjust_phase_temperature(name, t0 - 150.0)
        assert not synthetic_stable(ti)
        assert imp_lifecycle.drop_imputed_phases(ti, names=[name]) == [name]
        assert synthetic_stable(ti)
        assert abs(top_k(ti, name) - t0) <= imp_ternary.HSX_TOLERANCE_K

    def test_a_saved_adjustment_is_placed_again(self, no_persist, synthetic_store):
        ti, name, t0 = self.synthetic(synthetic_store)
        ti.adjust_phase_temperature(name, t0 - 150.0, write_cache=True)
        reread = build("hsx", data_dir=synthetic_store, include_imputed=True)
        assert not synthetic_stable(reread)
        assert abs(top_k(reread, name) - (t0 - 150.0)) <= imp_ternary.HSX_TOLERANCE_K

    def test_an_adjusted_phase_can_be_adjusted_again(self, no_persist, synthetic_store):
        ti, name, t0 = self.synthetic(synthetic_store)
        ti.adjust_phase_temperature(name, t0 - 150.0)
        record = ti.adjust_phase_temperature(name, t0 - 50.0)
        assert record.accepted, record.reason
        assert record.replaces == (SYNTHETIC_ID,)
        assert abs(record.realized_temperature_k - (t0 - 50.0)) <= imp_ternary.HSX_TOLERANCE_K
        assert [imputed_entry_label(e) for e in ti.imputed_entries] == [name]

    def test_an_unknown_phase_or_an_element_raises(self, no_persist):
        ti = build("hsx")
        with pytest.raises(ValueError, match="not a phase on the 0 K hull"):
            ti.adjust_phase_temperature("NoSuchPhase", 1000.0)
        element = next(
            e for e in ti.solid_hull().stable_entries if len(e.composition.elements) == 1
        )
        from gliquid.api import hull_entry_name

        with pytest.raises(ValueError, match="is an element"):
            ti.adjust_phase_temperature(hull_entry_name(element), 1000.0)


class TestHoldHull:
    """A phase that would displace a stable phase, held at the 0 K hull limit with an entropy."""

    DISPLACER = TestProtection.DISPLACER

    def test_it_is_placed_with_an_entropy_and_the_hull_keeps_its_phases(
        self, no_persist, synthetic_store
    ):
        ti = build("hsx", data_dir=synthetic_store)
        record = ti.impute_phase(self.DISPLACER, 2500.0, name="TX", screen=False, hold_hull=True)
        assert record.accepted and record.code == "imputed", record.reason
        assert record.enthalpy > record.inverted_enthalpy
        lift = record.enthalpy - record.inverted_enthalpy
        assert record.entropy == pytest.approx(lift / 2500.0)
        assert 0.0 < record.entropy < record.support_entropy
        assert [n for n in record.displaces if n.startswith("Mg2AlSi")]
        assert synthetic_stable(ti)

    def test_it_leaves_the_hull_at_its_temperature(self, no_persist, synthetic_store):
        ti = build("hsx", data_dir=synthetic_store)
        ti.impute_phase(self.DISPLACER, 2500.0, name="TX", screen=False, hold_hull=True)
        assert abs(top_k(ti, "TX") - 2500.0) <= imp_ternary.HSX_TOLERANCE_K

    def test_control_a_phase_that_displaces_nothing_carries_no_entropy(self, no_persist):
        record = build("hsx").impute_phase(TARGET, 1000.0, name="TX", screen=False, hold_hull=True)
        assert record.accepted and record.entropy == 0.0 and record.displaces == ()
        assert record.hull_limit_enthalpy is None and record.enthalpy < record.hull_enthalpy

    def test_with_force_it_is_held_where_an_entropy_reaches_and_placed_as_it_is_otherwise(
        self, no_persist, synthetic_store, monkeypatch
    ):
        ti = build("hsx", data_dir=synthetic_store)
        both = dict(name="TX", screen=False, hold_hull=True, force=True, apply=False)
        held = ti.impute_phase(self.DISPLACER, 2500.0, **both)
        assert held.accepted and held.entropy > 0.0
        assert held.enthalpy == held.hull_limit_enthalpy > held.inverted_enthalpy

        refined = imp_ternary._HullRows.refined

        def flat(rows, temp):
            return dataclasses.replace(refined(rows, temp), entropy=held.entropy)

        monkeypatch.setattr(imp_ternary._HullRows, "refined", flat)
        placed = ti.impute_phase(self.DISPLACER, 2500.0, **both)
        assert placed.accepted and placed.entropy == 0.0
        assert placed.enthalpy == placed.inverted_enthalpy < placed.hull_limit_enthalpy
        assert [n for n in placed.displaces if n.startswith("Mg2AlSi")]


class TestSolidus:
    """A phase placed on the hull between a solidus and the temperature it leaves it at."""

    TOP, SOLIDUS = 1200.0, 900.0

    def placed(self):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, self.TOP, name="TX", solidus=self.SOLIDUS, screen=False)
        return ti, record

    def test_it_carries_an_entropy_and_sits_above_the_0_k_hull(self, no_persist):
        ti, record = self.placed()
        assert record.accepted and record.code == "imputed", record.reason
        assert record.solidus_k == self.SOLIDUS and record.entropy > 0.0
        assert record.entropy <= imp_ternary.SOLIDUS_ENTROPY_SHARE * record.support_entropy
        assert "TX" not in {
            imputed_entry_label(e.as_dict()) for e in ti.solid_hull().stable_entries
        }
        assert ti.imputed_phase_names() == ["TX"]

    def test_it_is_on_the_hull_between_the_two_temperatures_only(self, no_persist):
        ti, _ = self.placed()
        assert abs(top_k(ti, "TX") - self.TOP) <= imp_ternary.HSX_TOLERANCE_K
        frame = ti.equil_df_list[0]
        lowest = float(frame.loc[frame["Phase"] == "TX", "T"].min()) + 273.15
        assert abs(lowest - self.SOLIDUS) <= 5.0

    def test_control_without_a_solidus_it_is_stable_from_0_k(self, no_persist):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, self.TOP, name="TX", screen=False)
        assert record.entropy == 0.0 and record.solidus_k is None
        assert "TX" in {imputed_entry_label(e.as_dict()) for e in ti.solid_hull().stable_entries}

    def test_below_the_liquid_it_is_refused_with_the_lowest_solidus_that_works(self, no_persist):
        record = build("hsx").impute_phase(TARGET, self.TOP, name="TX", solidus=300.0, screen=False)
        assert (record.accepted, record.code) == (False, "below_solidus")
        assert 300.0 < record.solidus_limit_k < self.TOP
        retry = build("hsx").impute_phase(
            TARGET, self.TOP, name="TX", solidus=record.solidus_limit_k, screen=False
        )
        assert retry.accepted, retry.reason

    def test_a_solidus_at_or_above_the_top_raises(self, no_persist):
        with pytest.raises(ValueError, match="solidus must lie between"):
            build("hsx").impute_phase(TARGET, self.TOP, name="TX", solidus=self.TOP)

    def test_edit_phases_passes_the_solidus(self, no_persist):
        ti = build("hsx")
        (record,) = ti.edit_phases(
            [
                {
                    "composition": TARGET,
                    "temperature": self.TOP,
                    "solidus": self.SOLIDUS,
                    "name": "TX",
                }
            ],
            screen=False,
        )
        assert record.accepted and record.solidus_k == self.SOLIDUS


class TestGivenEnergies:
    """A phase placed with an enthalpy or an entropy given for it."""

    TOP, SOLIDUS = TestSolidus.TOP, TestSolidus.SOLIDUS

    def test_both_given_are_carried_as_given(self, no_persist):
        ti = build("hsx")
        plain = build("hsx").impute_phase(TARGET, 1000.0, name="TX", screen=False)
        record = ti.impute_phase(
            TARGET, name="TX", enthalpy=plain.enthalpy - 500.0, entropy=2.0, screen=False
        )
        assert record.accepted and record.temperature_k is None, record.reason
        assert (record.enthalpy, record.entropy) == (plain.enthalpy - 500.0, 2.0)
        table = ti.hull_table()
        assert table.loc[table["Phase Name"] == "TX", "S"].tolist() == [2.0]

    def test_an_entropy_with_a_temperature_solves_the_enthalpy_and_leaves_at_it(self, no_persist):
        plain = build("hsx").impute_phase(TARGET, 900.0, name="TX", screen=False)
        ti = build("hsx")
        record = ti.impute_phase(TARGET, 900.0, name="TX", entropy=3.0)
        assert record.accepted and record.entropy == 3.0, record.reason
        assert record.enthalpy == pytest.approx(plain.enthalpy + 900.0 * 3.0, abs=1e-6)
        assert abs(record.realized_temperature_k - 900.0) <= imp_ternary.HSX_TOLERANCE_K

    def test_an_enthalpy_with_a_temperature_solves_the_entropy(self, no_persist):
        plain = build("hsx").impute_phase(TARGET, 900.0, name="TX", screen=False)
        record = build("hsx").impute_phase(
            TARGET, 900.0, name="TX", enthalpy=plain.enthalpy + 2700.0, screen=False
        )
        assert record.accepted, record.reason
        assert record.entropy == pytest.approx(3.0, abs=1e-9)

    @pytest.mark.parametrize("given", ["enthalpy", "entropy"])
    def test_one_of_them_with_a_solidus_solves_the_other(self, no_persist, given):
        both = build("hsx").impute_phase(
            TARGET, self.TOP, name="TX", solidus=self.SOLIDUS, screen=False
        )
        record = build("hsx").impute_phase(
            TARGET, name="TX", solidus=self.SOLIDUS, screen=False, **{given: getattr(both, given)}
        )
        assert record.accepted and record.temperature_k is None, record.reason
        assert record.enthalpy == pytest.approx(both.enthalpy, abs=1e-6)
        assert record.entropy == pytest.approx(both.entropy, abs=1e-9)

    def test_above_the_hull_without_an_entropy_it_is_refused(self, no_persist):
        record = build("hsx").impute_phase(
            TARGET, name="TX", enthalpy=5000.0, entropy=0.0, screen=False
        )
        assert (record.accepted, record.code) == (False, "unstable")

    def test_an_entropy_that_would_join_at_the_temperature_is_refused(self, no_persist):
        record = build("hsx").impute_phase(TARGET, 900.0, name="TX", entropy=1e4, screen=False)
        assert (record.accepted, record.code) == (False, "no_entropy")

    def test_a_given_enthalpy_that_displaces_a_phase_is_not_held_at_the_hull(
        self, no_persist, synthetic_store
    ):
        held = build("hsx", data_dir=synthetic_store).impute_phase(
            TestHoldHull.DISPLACER, 2500.0, name="TX", screen=False, hold_hull=True
        )
        with pytest.warns(UserWarning, match="will displace"):
            record = build("hsx", data_dir=synthetic_store).impute_phase(
                TestHoldHull.DISPLACER,
                name="TX",
                enthalpy=held.inverted_enthalpy,
                entropy=0.0,
                screen=False,
                hold_hull=True,
            )
        assert (record.accepted, record.code) == (False, "displaces_hull")
        assert [n for n in record.displaces if n.startswith("Mg2AlSi")]

    @pytest.mark.parametrize(
        "target",
        [
            {"enthalpy": -1e4},
            {"solidus": 900.0},
            {"temperature": 900.0, "solidus": 800.0, "entropy": 1.0},
        ],
    )
    def test_any_other_set_of_targets_raises(self, no_persist, target):
        with pytest.raises(ValueError, match="a target names"):
            imp_ternary.impute_ternary_phases(build("hsx"), [{"composition": TARGET, **target}])

    def test_edit_phases_gives_an_existing_phase_an_enthalpy_and_an_entropy(
        self, no_persist, synthetic_store
    ):
        ti, name, _ = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        (record,) = ti.edit_phases(
            [{"phase": name, "enthalpy": -19000.0, "entropy": 1.5}], screen=False
        )
        assert record.accepted and record.replaces == (SYNTHETIC_ID,), record.reason
        assert (record.enthalpy, record.entropy) == (-19000.0, 1.5)
        assert record.previous_enthalpy == pytest.approx(-20000.0, abs=1e-6)


class TestEntryList:
    """A system's entries listed for pymatgen, and a list read back as phase edits."""

    def test_the_list_is_the_hull_pymatgen_builds_and_carries_the_entropies(self, no_persist):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, 1000.0, name="TX", entropy=2.0, screen=False)
        entries = json.loads(json.dumps(ti.export_entries()))
        hull = PhaseDiagram([ComputedEntry.from_dict(d) for d in entries])

        def stable(diagram):
            return sorted(canonical_entry_id(e.entry_id) for e in diagram.stable_entries)

        assert stable(hull) == stable(ti.solid_hull())
        placed = next(e for e in hull.all_entries if e.entry_id == "imputed:TX")
        assert hull.get_form_energy_per_atom(placed) * EV_ATOM_TO_J_MOL == pytest.approx(
            record.enthalpy, abs=1e-6
        )
        entropies = {d["entry_id"] == "imputed:TX": d["data"][ENTROPY_KEY] for d in entries}
        assert entropies == {True: 2.0, False: 0.0}
        assert {(d["correction"], len(d["energy_adjustments"])) for d in entries} == {(0.0, 0)}
        assert {d["@module"] for d in entries} == {"pymatgen.entries.computed_entries"}

    def test_control_a_systems_own_list_changes_nothing(self, no_persist, synthetic_store):
        ti = build("hsx", data_dir=synthetic_store)
        found = ti.entry_list_edits(ti.export_entries())
        assert (found.edits, found.reference_shifts, found.ignored) == ([], {}, [])
        assert found.unchanged >= 1

    def test_a_fresh_system_given_an_edited_systems_list_gets_its_phases(
        self, no_persist, synthetic_store
    ):
        edited, name, _ = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        edited.edit_phases(
            [
                {"phase": name, "enthalpy": -19000.0, "entropy": 1.5},
                {"composition": TARGET, "temperature": 1000.0, "name": "TX"},
            ],
            screen=False,
        )
        entries = edited.export_entries()
        fresh = build("hsx", data_dir=synthetic_store)
        found = fresh.entry_list_edits(entries)
        assert (found.changed, found.added, found.removed) == ([name], ["TX"], [])
        records = fresh.edit_phases(found.edits, screen=False)
        assert all(r.accepted for r in records), [r.reason for r in records]
        assert fresh.entry_list_edits(entries).edits == []

    def test_a_phase_the_list_leaves_out_is_removed(self, no_persist, synthetic_store):
        ti, name, _ = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        entries = [d for d in ti.export_entries() if d["entry_id"] != SYNTHETIC_ID]
        found = ti.entry_list_edits(entries)
        assert found.removed == [name] and found.edits == [{"remove": name}]
        ti.edit_phases(found.edits)
        assert not synthetic_stable(ti)

    def test_a_list_on_other_references_keeps_its_formation_enthalpies(
        self, no_persist, synthetic_store
    ):
        ti = build("hsx", data_dir=synthetic_store)
        entries = ti.export_entries()
        for payload in entries:
            payload["energy"] += 0.01 * ComputedEntry.from_dict(payload).composition["Al"]
        found = ti.entry_list_edits(entries)
        assert found.reference_shifts == pytest.approx({"Al": 0.01})
        assert found.edits == []

    def test_listed_entries_that_change_nothing_are_ignored(self, no_persist):
        ti = build("hsx")
        high = probe_entry(ti, TARGET, 5000.0, name="HIGH")
        other = ComputedEntry("Fe2O3", -1.0, entry_id="other").as_dict()
        found = ti.entry_list_edits(ti.export_entries() + [high, other])
        assert found.edits == [] and sorted(found.ignored) == ["imputed:HIGH", "other"]

    def test_a_forced_edit_places_a_phase_that_displaces_another(self, no_persist, synthetic_store):
        held = build("hsx", data_dir=synthetic_store).impute_phase(
            TestHoldHull.DISPLACER, 2500.0, name="TX", screen=False, hold_hull=True
        )
        ti = build("hsx", data_dir=synthetic_store)
        edit = {
            "composition": TestHoldHull.DISPLACER,
            "name": "TX",
            "enthalpy": held.inverted_enthalpy,
            "entropy": 0.0,
            "force": True,
        }
        (record,) = ti.edit_phases([edit], screen=False, hold_hull=True)
        assert record.accepted and not synthetic_stable(ti)

    def test_remove_takes_out_a_phase_the_store_imputed_and_what_it_stands_in_for(
        self, no_persist, synthetic_store
    ):
        ti, name, t0 = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        ti.adjust_phase_temperature(name, t0 - 150.0, write_cache=True)
        reread = build("hsx", data_dir=synthetic_store, include_imputed=True)
        assert name in reread.imputed_phase_names()
        reread.edit_phases([{"remove": name}])
        reread.process_data()
        assert name not in reread.phase_names and not synthetic_stable(reread)


class TestEntropyOnTheEdges:
    """An imputed edge phase carrying an entropy is placed above the 0 K hull, on the ternary
    and on its binary edge."""

    EDGE = {"Al": 0.0, "Mg": 0.5, "Si": 0.5}

    def lifted(self, entropy):
        ti = build("hsx")
        entry = probe_entry(ti, self.EDGE, 2000.0, name="EX")
        if entropy:
            entry["data"][ENTROPY_KEY] = entropy
        ti.imputed_entries = [entry]
        ti.reset_derived_state()
        return ti

    def edge_phases(self, ti):
        pair = ["Mg", "Si"]
        return build_phases_from_chull(ti.edge_hull("Mg-Si"), pair, UNARY.component_data(pair))

    def test_it_is_a_row_of_the_ternary_table_with_its_entropy(self, no_persist):
        table = self.lifted(6.0).hull_table()
        assert table.loc[table["Phase Name"] == "EX", "S"].tolist() == [6.0]

    def test_it_is_a_phase_of_the_binary_edge_with_its_entropy(self, no_persist):
        phase = next(p for p in self.edge_phases(self.lifted(6.0)) if p.name == "EX")
        assert phase.entropy == 6.0 and phase.enthalpy == pytest.approx(2000.0, abs=1e-6)

    def test_control_without_an_entropy_it_is_left_out(self, no_persist):
        ti = self.lifted(0.0)
        assert "EX" not in set(ti.hull_table()["Phase Name"])
        assert "EX" not in {p.name for p in self.edge_phases(ti)}


class TestEditPhases:
    """Additions, temperature changes and removals applied in one call."""

    def test_a_change_and_an_addition_are_placed_together(self, no_persist, synthetic_store):
        ti, name, t0 = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        records = ti.edit_phases(
            [
                {"phase": name, "temperature": t0 - 150.0},
                {"composition": TARGET, "temperature": t0, "name": "TX"},
            ],
            screen=False,
        )
        assert [r.name for r in records] == [name, "TX"]
        assert all(r.accepted for r in records), [r.reason for r in records]
        assert records[0].previous_enthalpy == pytest.approx(-20000.0, abs=1e-6)
        assert records[1].previous_enthalpy is None
        assert sorted(ti.imputed_phase_names()) == sorted([name, "TX"])

    def test_remove_leaves_the_dft_phase_off_the_hull(self, no_persist, synthetic_store):
        ti, name, _ = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        assert ti.edit_phases([{"remove": name}]) == []
        assert not synthetic_stable(ti)
        ti.process_data()
        assert name not in ti.phase_names

    def test_remove_drops_a_phase_imputed_on_the_model(self, no_persist):
        ti = build("hsx")
        ti.impute_phase(TARGET, 1000.0, name="TX", screen=False)
        ti.edit_phases([{"remove": "TX"}])
        assert ti.imputed_entries == []

    def test_removing_an_unknown_phase_raises(self, no_persist):
        with pytest.raises(ValueError, match="not a phase on the 0 K hull"):
            build("hsx").edit_phases([{"remove": "NoSuchPhase"}])

    def test_a_phase_exposed_by_lifting_the_changed_one_out_is_not_protected(
        self, no_persist, synthetic_store, synthetic_record
    ):
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        fractions = {"Al": 0.2, "Mg": 0.5, "Si": 0.3}
        ti = build("hsx", data_dir=synthetic_store)
        covered = imp_core.phase_diagram_enthalpy_at(ti.solid_hull(), fractions)
        bare = imp_core.phase_diagram_enthalpy_at(ti.solid_hull(exclude=[SYNTHETIC_ID]), fractions)
        assert covered < bare - 1000.0
        per_atom = imp_core.hull_ref_energy_per_atom(ti.solid_hull([]), fractions) + (
            0.5 * (covered + bare) / EV_ATOM_TO_J_MOL
        )
        hidden = ComputedEntry(Composition("Mg5Al2Si3"), 10 * per_atom, entry_id="mp-synthetic-2")
        record_file = synthetic_store / synthetic_record.name
        raw = json.loads(record_file.read_text())
        raw.append(hidden.as_dict())
        record_file.write_text(json.dumps(raw))

        ti, name, t0 = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        assert "mp-synthetic-2" not in {str(e.entry_id) for e in ti.solid_hull().stable_entries}
        record = ti.adjust_phase_temperature(name, t0 - 20.0, screen=False)
        assert record.accepted, record.reason
        assert record.displaces == ()


class TestLiquidusRanges:
    """The temperature span of each phase's liquidus field."""

    @staticmethod
    def plotter():
        return TLIPlotter.from_components(
            COMPONENTS,
            delta=DELTA,
            xs_mix=IDEAL,
            order="given",
            hull_mode="hsx",
            param_format="linear",
        )

    def test_empty_before_a_plot(self, no_persist):
        assert self.plotter().liquidus_ranges() == {}

    def test_an_imputed_phase_spans_up_to_its_temperature(self, no_persist):
        plotter = self.plotter()
        plotter.impute_phase(TARGET, 1000.0, name="TX")
        plotter.get_plot("tx")
        ranges = plotter.liquidus_ranges()
        low, high = ranges["TX"]
        assert high == pytest.approx(1000.0 - 273.15, abs=5.0)
        assert low < high
        assert set(ranges) <= set(plotter.phase_names)

    def test_an_element_tops_out_at_its_melting_point(self, no_persist):
        plotter = self.plotter()
        plotter.get_plot("tx")
        ranges = plotter.liquidus_ranges()
        melting = {
            name: top for name, (_, top) in ranges.items() if name.split(" ")[0] in COMPONENTS
        }
        assert melting
        tops = plotter.get_inter_melting_temps(list(melting))
        for name, top in melting.items():
            assert top == pytest.approx(tops[name], abs=imp_ternary.HSX_TOLERANCE_K)


class TestPhaseTops:
    """The highest temperature each solid phase is stable at, and what it becomes there."""

    def test_an_imputed_phase_tops_out_at_its_temperature_into_its_decomposition(self, no_persist):
        ti = build("hsx")
        record = ti.impute_phase(TARGET, 1000.0, name="TX", screen=False)
        top = imp_ternary.phase_tops(ti, ["TX"])["TX"]
        assert top.temperature_k == pytest.approx(1000.0, abs=0.5)
        merged = {}
        for phase, weight in record.decomposition:
            merged[phase] = merged.get(phase, 0.0) + weight
        assert dict(top.products) == pytest.approx(merged, abs=1e-6)
        assert top.kind == imp_ternary._top_kind(merged)

    def test_the_processed_diagram_start_finds_the_same_tops(self, no_persist):
        ti = build("hsx")
        cold = imp_ternary.phase_tops(ti)
        ti.process_data()
        hinted = imp_ternary.phase_tops(ti)
        assert set(hinted) == set(cold) and hinted
        for name, top in hinted.items():
            assert top.temperature_k == pytest.approx(cold[name].temperature_k, abs=0.05)
            assert top.kind == cold[name].kind

    def test_a_solid_above_the_hull_without_an_entropy_is_never_stable(self, no_persist):
        assert imp_ternary.phase_top(build("hsx"), TARGET, 50000.0, 0.0) is None

    def test_a_temperature_below_the_liquid_is_refused_with_the_lowest_that_works(self, no_persist):
        ti = build("hsx")
        tol = imp_ternary.TOP_TOL_K

        def record(temp):
            return ti.impute_phase(TARGET, temp, name="TX", apply=False, screen=False)

        cold = record(300.0)
        assert cold.code == "below_solidus" and cold.lowest_top_k > 300.0
        assert record(cold.lowest_top_k - 2 * tol).code == "below_solidus"
        assert record(cold.lowest_top_k).code != "below_solidus"

    def test_a_phase_given_its_enthalpy_below_the_liquid_decomposes_into_solids(self, no_persist):
        ti = build("hsx")
        deep = imp_core.phase_diagram_enthalpy_at(ti.solid_hull(), TARGET) - 2000.0
        target = {"composition": TARGET, "temperature": 300.0, "enthalpy": deep, "name": "TX"}
        record = imp_ternary.impute_ternary_phases(ti, [target], screen=False, refine_liquid=True)[
            0
        ]
        assert record.accepted and record.entropy < 0.0, record.reason
        assert "L" not in dict(record.decomposition)
        top = imp_ternary.phase_tops(ti, ["TX"], refine_liquid=True)["TX"]
        assert top.temperature_k == pytest.approx(300.0, abs=imp_ternary.TOP_TOL_K)
        assert top.kind == "decomposition"

    def test_a_phase_given_its_enthalpy_tops_out_without_the_phase_it_pushes_off(self, no_persist):
        ti = build("hsx")
        ti.impute_phase({"Al": 0.2, "Mg": 0.45, "Si": 0.35}, 1300.0, name="QX", screen=False)
        deep = imp_core.phase_diagram_enthalpy_at(ti.solid_hull(), TARGET) - 3000.0
        target = {"composition": TARGET, "temperature": 1000.0, "enthalpy": deep, "name": "PX"}
        record = imp_ternary.impute_ternary_phases(
            ti, [target], force=True, screen=False, refine_liquid=True
        )[0]
        assert record.accepted and "QX" in record.displaces, record.reason
        assert "QX" not in dict(record.decomposition)
        top = imp_ternary.phase_tops(ti, ["PX"], refine_liquid=True)["PX"]
        assert abs(top.temperature_k - 1000.0) <= imp_ternary.TOP_MARGIN_K + imp_ternary.TOP_TOL_K

    def test_a_phase_stable_in_a_window_keeps_the_top_its_search_lands_on(self, monkeypatch):
        class Window:
            def __init__(self, *args):
                pass

            def envelope(self, temp, lattice=None):
                above, below = 5.0 * (temp - 1000.0), 800.0 - temp
                if above >= below:
                    return imp_core.Envelope(-(above + 1e-13), (0,), (1.0,), ("L",), 5.0)
                return imp_core.Envelope(-(below + 1e-13), (1,), (1.0,), ("A",), -1.0)

            def lattice(self, env):
                return None

        monkeypatch.setattr(imp_ternary, "_HullRows", Window)
        top = imp_ternary._phase_top(None, None, None, 0.0, 0.0, [], 1075.0)
        assert top.temperature_k == pytest.approx(1000.0, abs=imp_ternary.TOP_TOL_K)
        assert dict(top.products) == {"L": 1.0}

    @pytest.mark.parametrize(
        "products, kind",
        [
            ({"L": 1.0}, "congruent"),
            ({"L": 0.6, "A": 0.4}, "incongruent"),
            ({"A": 1.0}, "transition"),
            ({"A": 0.5, "B": 0.5}, "decomposition"),
        ],
    )
    def test_the_kind_follows_from_the_products(self, products, kind):
        assert imp_ternary._top_kind(products) == kind


class TestRefinedLiquid:
    """``refine_liquid``: a phase is placed against the liquid itself, not against its grid."""

    OFF_GRID = {"Al": 0.23, "Mg": 0.41, "Si": 0.36}

    def placed(self, temp, **kwargs):
        ti = build("hsx")
        return ti, ti.impute_phase(self.OFF_GRID, temp, name="TX", screen=False, **kwargs)

    def test_a_phase_that_melts_alone_carries_the_liquids_energy_at_its_composition(
        self, no_persist
    ):
        ti, record = self.placed(1500.0, refine_liquid=True)
        liquid = float(ti.liquid_gibbs([self.OFF_GRID["Mg"], self.OFF_GRID["Si"]], 1500.0))
        assert {phase for phase, _ in record.decomposition} == {"L"}
        assert record.enthalpy == pytest.approx(liquid - imp_core.HULL_GUARD_MARGIN_J, abs=1e-6)
        assert self.placed(1500.0)[1].enthalpy > record.enthalpy + 10.0

    @pytest.mark.parametrize("temp", [900.0, 1500.0])
    def test_it_tops_out_at_its_temperature_against_the_refined_liquid(self, no_persist, temp):
        ti, record = self.placed(temp, refine_liquid=True)
        assert record.accepted, record.reason
        top = imp_ternary.phase_tops(ti, ["TX"], refine_liquid=True)["TX"]
        reach = imp_ternary.TOP_MARGIN_K + imp_ternary.TOP_TOL_K
        assert abs(top.temperature_k - temp) <= reach
        assert imp_ternary.phase_tops(ti, ["TX"])["TX"].temperature_k > temp + 1.0


class TestSolidusTriangles:
    """The triangles of three solid phases the liquid leaves."""

    @staticmethod
    def plotter():
        return TLIPlotter.from_components(
            COMPONENTS,
            delta=DELTA,
            xs_mix=IDEAL,
            order="given",
            hull_mode="gtx",
            param_format="linear",
        )

    @staticmethod
    def liquid_over_plane(ti, triangle) -> float:
        """Gibbs energy, J/mol, of the triangle's liquid over the plane through its solids."""
        xy, enthalpy, entropy, is_liquid = ti._row_arrays()
        phase = ti.proc_df["Phase"].to_numpy(object)
        kelvin = triangle["T"] + 273.15
        rows = [
            next(
                r
                for r in np.flatnonzero(~is_liquid)
                if phase[r] == name and np.allclose(xy[r], corner, atol=1e-9)
            )
            for name, corner in zip(triangle["phases"], triangle["comp"])
        ]
        plane = np.linalg.solve(
            np.column_stack([np.ones(3), xy[rows]]), enthalpy[rows] - kelvin * entropy[rows]
        )
        return ti.liquid_gibbs(triangle["liquid"], kelvin) - plane @ [1.0, *triangle["liquid"]]

    def test_empty_before_a_plot(self, no_persist):
        assert self.plotter().solidus_triangles() == []

    def test_the_triangles_tile_the_system_where_the_liquid_leaves(self, no_persist):
        plotter = self.plotter()
        plotter.get_plot("tx")
        triangles = plotter.solidus_triangles()
        temps = [triangle["T"] for triangle in triangles]
        assert temps == sorted(temps, reverse=True)
        area = sum(abs(np.linalg.det(t["comp"][1:] - t["comp"][0])) for t in triangles)
        assert area == pytest.approx(1.0, abs=1e-9)
        for triangle in triangles:
            assert self.liquid_over_plane(plotter._ti, triangle) == pytest.approx(0.0, abs=0.5)

        ranges = plotter.liquidus_ranges()
        for triangle in triangles:
            for phase in triangle["phases"]:
                assert ranges[phase][0] <= triangle["T"]

    def test_solidus_draws_the_binary_tie_lines_inside_the_window_on_their_sides(self, no_persist):
        plotter = self.plotter()
        plain = plotter.get_plot("tx")
        assert "solidus" not in {trace.name for trace in plain.data}
        assert "solidus" not in {trace.name for trace in plotter.get_plot("tx", solidus=True).data}

        low, high = plotter.conds
        inside = 0.5 * (low + high)
        plotter._edge_tie_lines = {
            "Mg-Si": [(0.0, 50.0, inside), (20.0, 20.0, inside), (0.0, 50.0, high + 100.0)],
            "Al-Si": [(40.0, 100.0, low)],
        }
        (drawn,) = [t for t in plotter.get_plot("tx", solidus=True).data if t.name == "solidus"]
        assert list(drawn.z) == [inside, inside, None, low, low]
        assert [x for x in drawn.x if x is not None] == pytest.approx([1.0, 0.75, 0.2, 0.5])
        height = np.sqrt(3) / 2
        assert [y for y in drawn.y if y is not None] == pytest.approx(
            [0.0, 0.5 * height, 0.4 * height, height]
        )

    def test_a_joint_under_a_hotter_sheet_gives_no_triangle(self, no_persist, monkeypatch):
        plotter = self.plotter()
        plotter.get_plot("tx")
        partition, sheets = plotter._partition_cache[2], plotter._partition_sheets
        monkeypatch.setattr(TLIPlotter, "_unjoined_solidus", lambda self, joined: [])
        assert len(plotter._solidus(partition, sheets)) >= 3

        class Raised:
            """``sheets`` with every sheet read 50 K hotter by a whole call."""

            columns = staticmethod(sheets.columns)
            contact_points = staticmethod(sheets.contact_points)

            def __call__(self, comp):
                return sheets(comp) + 50.0

        assert plotter._solidus(partition, Raised()) == []

    def test_a_triangle_the_solids_hull_does_not_hold_is_taken_again(self, no_persist, monkeypatch):
        plotter = self.plotter()
        plotter.get_plot("tx")
        expected = {t["phases"]: t["T"] for t in plotter.solidus_triangles()}
        ti = plotter._ti
        xy, _, _, is_liquid = ti._row_arrays()
        corners = [
            next(r for r in np.flatnonzero(~is_liquid) if np.allclose(xy[r], corner))
            for corner in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
        ]
        facets = ternary_module._solid_facets
        floor = max(plotter.conds[0] + 273.15, 1.0)
        monkeypatch.setattr(
            ternary_module,
            "_solid_facets",
            lambda ti, temp: np.array([corners]) if temp == floor else facets(ti, temp),
        )
        solved = {t["phases"]: t["T"] for t in plotter._unjoined_solidus([])}
        assert set(solved) == set(expected)
        for phases, temp in solved.items():
            assert temp == pytest.approx(expected[phases], abs=1e-2)

    @pytest.mark.parametrize("inner, held", [(-100.0, False), (100.0, True)])
    def test_a_triangle_with_a_solid_below_its_plane_is_off_the_solids_hull(self, inner, held):
        class Rows:
            @staticmethod
            def _row_arrays():
                xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, 0.3], [0.5, 0.25]])
                enthalpy = np.array([0.0, 0.0, 0.0, inner, -900.0])
                liquid = np.array([False, False, False, False, True])
                return xy, enthalpy, np.zeros(5), liquid

        on_hull = ternary_module._on_solids_hull(Rows(), np.array([[0, 1, 2]]), np.array([500.0]))
        assert on_hull.tolist() == [held]

    def test_a_lower_solid_at_a_corner_of_a_triangle_leaves_it_on_the_solids_hull(self):
        class Rows:
            @staticmethod
            def _row_arrays():
                xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.25]])
                enthalpy = np.array([0.0, 0.0, 0.0, -100.0, -900.0])
                liquid = np.array([False, False, False, False, True])
                return xy, enthalpy, np.zeros(5), liquid

        on_hull = ternary_module._on_solids_hull(Rows(), np.array([[0, 1, 2]]), np.array([500.0]))
        assert on_hull.tolist() == [True]

    def test_a_lower_solid_beyond_a_corner_of_a_triangle_takes_it_off_the_solids_hull(self):
        class Rows:
            @staticmethod
            def _row_arrays():
                xy = np.array([[0.0, 0.5], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.25]])
                enthalpy = np.array([0.0, 0.0, 0.0, -100.0, -900.0])
                liquid = np.array([False, False, False, False, True])
                return xy, enthalpy, np.zeros(5), liquid

        on_hull = ternary_module._on_solids_hull(Rows(), np.array([[0, 1, 2]]), np.array([500.0]))
        assert on_hull.tolist() == [False]

    @pytest.mark.parametrize(
        "twin_enthalpy, twin_entropy, held", [(-100.0, 0.0, False), (-499.5, -1.0, True)]
    )
    def test_a_triangle_holds_while_its_solids_are_the_stable_ones_at_its_temperature(
        self, twin_enthalpy, twin_entropy, held
    ):
        """A second solid at one corner, of lower Gibbs energy at 500 K or only below it."""

        class Rows:
            @staticmethod
            def _row_arrays():
                xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.25]])
                enthalpy = np.array([0.0, 0.0, 0.0, twin_enthalpy, -900.0])
                entropy = np.array([0.0, 0.0, 0.0, twin_entropy, 0.0])
                liquid = np.array([False, False, False, False, True])
                return xy, enthalpy, entropy, liquid

        holds = ternary_module._holds(Rows(), np.array([[0, 1, 2]]), np.array([500.0]))
        assert holds.tolist() == [held]

    def test_a_contact_is_taken_through_the_solids_stable_at_its_temperature(
        self, no_persist, monkeypatch
    ):
        plotter = self.plotter()
        plotter.get_plot("tx")
        ti = plotter._ti
        first, other = ternary_module._solid_facets(ti, max(plotter.conds[0] + 273.15, 1.0))[:2]
        alone, _, _ = ternary_module._plane_contacts(ti, np.array([first, other]))
        asked = []

        def stable(ti, rows, temps):
            """The rows asked about, until a contact is known; then ``other``."""
            asked.append(np.array(temps))
            return rows if len(asked) == 1 else np.array([other])

        monkeypatch.setattr(ternary_module, "_stable_rows", stable)
        temps, _, rows = ternary_module._plane_contacts(ti, np.array([first]))
        assert rows.tolist() == [other.tolist()]
        assert asked[1] == pytest.approx(alone[:1], abs=1e-6)
        assert temps == pytest.approx(alone[1:], abs=1e-6)

    def test_three_solids_of_one_energy_are_one_triangle(self):
        """Three elements with no compound: the solids' hull is flat."""

        class Rows:
            @staticmethod
            def _row_arrays():
                xy = np.array([[0.5, 0.25], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
                enthalpy = np.array([-900.0, 0.0, 0.0, 0.0, 150.0])
                liquid = np.array([True, False, False, False, False])
                return xy, enthalpy, np.zeros(5), liquid

        facets = ternary_module._solid_facets(Rows(), 500.0)
        assert np.sort(facets, axis=1).tolist() == [[1, 2, 3]]

    def test_the_solids_hull_alone_gives_the_triangles_of_the_joints(self, no_persist):
        plotter = self.plotter()
        plotter.get_plot("tx")
        joined = {t["phases"]: t for t in plotter.solidus_triangles()}
        solved = {t["phases"]: t for t in plotter._unjoined_solidus([])}
        assert set(solved) == set(joined) and len(joined) >= 3
        for phases, triangle in solved.items():
            assert triangle["T"] == pytest.approx(joined[phases]["T"], abs=1e-2)
            assert triangle["liquid"] == pytest.approx(joined[phases]["liquid"], abs=1e-4)


# -- Tangent, apply and replace --


class TestTangent:
    def test_congruent_tangent_is_the_liquid(self, no_persist):
        ti = build("hsx", xs_mix=MIXING)
        record = ti.impute_phase(TARGET, 1400.0, name="TX", liquid_composition=TARGET, apply=False)
        coords = imp_core.independent_coords(TARGET, COMPONENTS)
        assert record.method == "tangent"
        assert record.inverted_enthalpy == ti.liquid_gibbs(coords, 1400.0)
        assert record.melts is None and record.decomposition == ()

    def test_liquid_gibbs_matches_the_grid(self):
        ti = build("hsx", xs_mix=MIXING)
        ti.interpolate()
        row = ti.hsx_df[(ti.hsx_df["Phase Name"] == "L")].iloc[57]
        g = ti.liquid_gibbs([row["x0"], row["x1"]], 1000.0)
        assert g == pytest.approx(row["H"] - 1000.0 * row["S"], abs=1e-6)


class TestApply:
    def test_apply_false_leaves_the_model_alone(self, no_persist):
        ti = build("hsx")
        ti.interpolate()
        before = ti.hsx_df
        record = ti.impute_phase(TARGET, 1000.0, name="TX", apply=False)
        assert record.accepted and record.realized_temperature_k is None
        assert ti.imputed_entries == [] and ti.hsx_df is before

    def test_reimputing_replaces_by_name(self, no_persist):
        ti = build("hsx")
        first = ti.impute_phase(TARGET, 900.0, name="TX")
        second = ti.impute_phase(TARGET, 1100.0, name="TX")
        assert second.accepted, second.reason
        assert [imputed_entry_label(e) for e in ti.imputed_entries] == ["TX"]
        assert (
            ti.imputed_entries[0]["data"]["h_form_j_per_mol"] == second.enthalpy != first.enthalpy
        )
        assert abs(second.realized_temperature_k - 1100.0) <= imp_ternary.HSX_TOLERANCE_K

    def test_drop(self, no_persist):
        ti = build("hsx")
        ti.impute_phase(TARGET, 1000.0, name="TX")
        assert imp_lifecycle.drop_imputed_phases(ti, names=["TX"]) == ["TX"]
        assert ti.imputed_entries == [] and not hasattr(ti, "hsx_df")


# -- Persistence --


class TestPersistence:
    def test_knob_off_writes_nothing(self, no_persist, store):
        before = (store / "Al-Mg-Si_ENTRIES_MP_GGA.json").read_bytes()
        record = build("hsx", data_dir=store).impute_phase(TARGET, 1000.0, name="TX")
        assert record.accepted and not record.written
        assert (store / "Al-Mg-Si_ENTRIES_MP_GGA.json").read_bytes() == before

    def test_write_cache_true_writes_and_reads_back(self, no_persist, store):
        record = build("hsx", data_dir=store).impute_phase(
            TARGET, 1000.0, name="TX", write_cache=True
        )
        assert record.written
        reread = build("hsx", data_dir=store, include_imputed=True)
        assert [imputed_entry_label(e) for e in reread.all_imputed_entries()] == ["TX"]
        assert abs(top_k(reread, "TX") - 1000.0) <= imp_ternary.HSX_TOLERANCE_K

    def test_control_without_include_imputed_it_is_not_placed(self, no_persist, store):
        build("hsx", data_dir=store).impute_phase(TARGET, 1000.0, name="TX", write_cache=True)
        plain = build("hsx", data_dir=store)
        plain.interpolate()
        assert "TX" not in set(plain.hsx_df["Phase Name"])

    def test_knob_on_writes_by_default(self, store):
        before = config.persist_ternary_imputed
        try:
            config.set_persist_ternary_imputed(True)
            record = build("hsx", data_dir=store).impute_phase(TARGET, 1000.0, name="TX")
        finally:
            config.set_persist_ternary_imputed(before)
        assert record.written

    def test_unwritable_store_raises_before_anything_changes(self, no_persist, store):
        class ReadOnly(DirectoryBackend):
            @property
            def capabilities(self):
                return frozenset({"read"})

        ti = build("hsx", data_dir=ReadOnly(store))
        with pytest.raises(CacheModeError):
            ti.impute_phase(TARGET, 1000.0, name="TX", write_cache=True)
        assert ti.imputed_entries == []

    def test_include_imputed_travels_with_the_component_order(self):
        ti = build("hsx", include_imputed=True)
        assert ti.with_component_order(["Si", "Mg", "Al"]).include_imputed is True


def stored_edge_entry(store, name="EDGE") -> dict:
    """A tagged imputed MgSi entry at -40 kJ/mol-atom, written to ``store`` under 'Mg-Si'."""
    from gliquid import api
    from gliquid.cache import KIND_DFT_ENTRIES, CacheKey

    backend = DirectoryBackend(store)
    backend.write_json(CacheKey("Mg-Si", KIND_DFT_ENTRIES, "GGA"), [])
    entry = probe_entry(
        build("hsx", data_dir=store), {"Al": 0.0, "Mg": 0.5, "Si": 0.5}, -40000.0, name
    )
    entry["data"]["x_b"] = 0.5
    api.cache_imputed_entries(["Mg", "Si"], [entry], data_dir=store)
    return entry


class TestEdgeRecords:
    def test_include_imputed_places_a_stored_edge_phase(self, no_persist, store):
        stored_edge_entry(store)
        ti = build("hsx", data_dir=store, include_imputed=True)
        assert [imputed_entry_label(e) for e in ti.all_imputed_entries()] == ["EDGE"]
        assert ti.imputed_phase_names() == ["EDGE"]

    def test_control_without_include_imputed_it_is_not_read(self, no_persist, store):
        stored_edge_entry(store)
        ti = build("hsx", data_dir=store)
        assert ti.all_imputed_entries() == [] and ti.imputed_phase_names() == []

    def test_an_explicit_copy_is_placed_once(self, no_persist, store):
        entry = stored_edge_entry(store)
        ti = build("hsx", data_dir=store, include_imputed=True, imputed_entries=[entry])
        assert [imputed_entry_label(e) for e in ti.all_imputed_entries()] == ["EDGE"]


class TestStaleness:
    def test_edges_are_keyed_alphabetically(self):
        params = build("hsx", xs_mix=MIXING).imputation_params()
        assert list(params["edges"]) == ["Al-Mg", "Al-Si", "Mg-Si"]
        assert params["edges"]["Al-Si"] == [-10000.0, 0.0, 0.0, 0.0]

    def test_reordering_does_not_read_as_stale(self, no_persist):
        ti = build("hsx", xs_mix=MIXING)
        record = ti.impute_phase(TARGET, 1200.0, name="TX", apply=False)
        reordered = ti.with_component_order(["Si", "Mg", "Al"])
        assert imp_lifecycle.imputed_entry_staleness(reordered, record.entry_dict) is None

    @pytest.mark.parametrize("param_format, stale", [("comb-exp", True), ("linear", False)])
    def test_an_entry_without_the_liquid_linearization_record(
        self, no_persist, param_format, stale
    ):
        ti = TernaryLiquidInterpolation(
            list(COMPONENTS),
            delta=DELTA,
            xs_mix=MIXING,
            param_format=param_format,
            order="given",
            hull_mode="hsx",
            T_incr=T_INCR,
        )
        record = ti.impute_phase(TARGET, 1200.0, name="TX", apply=False)
        assert imp_lifecycle.imputed_entry_staleness(ti, record.entry_dict) is None
        unrecorded = json.loads(json.dumps(record.entry_dict))
        del unrecorded["data"][imp_core.PROVENANCE_KEY]["liquid"]["liquid_gibbs"]
        reason = imp_lifecycle.imputed_entry_staleness(ti, unrecorded)
        assert (reason is not None and "liquid_gibbs" in reason) is stale


# -- Plotting --


def dashed(fig) -> list:
    """The solid-phase lines of ``fig`` that are broken into dashes at one composition."""
    return [
        t
        for t in fig.data
        if t.type == "scatter3d"
        and (t.hovertemplate or "").startswith("<b>Phase:")
        and None in t.z
        and len({x for x in t.x if x is not None}) == 1
    ]


def tx_figure(ti, order="given"):
    return TLIPlotter(ti, order=order).get_plot("tx")


class TestPlot:
    def test_an_imputed_phase_is_drawn_dashed(self, no_persist):
        ti = build("hsx")
        ti.impute_phase(TARGET, 1000.0, name="TX")
        assert ti.imputed_phase_names() == ["TX"]
        fig = tx_figure(ti, order=["Si", "Mg", "Al"])
        lines = dashed(fig)
        assert len(lines) == 1 and "TX (imputed)" in lines[0].hovertemplate
        assert "TX (imputed)" in {t.name for t in fig.data}
        assert ti.modified_phase_names() == []
        low, high = fig.layout.scene.zaxis.range
        ends = [z for z in lines[0].z if z is not None]
        assert ends[1] - ends[0] == pytest.approx(style.SOLID_LINE_DASH_3D[0] * (high - low))
        assert ends[2] - ends[1] == pytest.approx(style.SOLID_LINE_DASH_3D[1] * (high - low))

    def test_a_phase_standing_in_for_a_dft_entry_is_named_modified(
        self, no_persist, synthetic_store
    ):
        ti, name, t0 = TestAdjustPhaseTemperature.synthetic(synthetic_store)
        assert ti.adjust_phase_temperature(name, t0 - 150.0).accepted
        assert ti.modified_phase_names() == ti.imputed_phase_names() == [name]
        fig = tx_figure(ti)
        (line,) = dashed(fig)
        assert "(modified)" in line.hovertemplate
        names = [t.name or "" for t in fig.data]
        assert any("(modified)" in n for n in names) and not any("(imputed)" in n for n in names)

    def test_control_without_imputation_nothing_is_dashed(self):
        ti = build("hsx")
        assert ti.imputed_phase_names() == []
        fig = tx_figure(ti)
        assert dashed(fig) == []
        assert not any("(imputed)" in (t.name or "") for t in fig.data)

    def test_a_dropped_phase_is_no_longer_named(self, no_persist):
        ti = build("hsx")
        ti.impute_phase(TARGET, 1000.0, name="TX")
        imp_lifecycle.drop_imputed_phases(ti, names=["TX"])
        assert ti.imputed_phase_names() == []

    def test_an_imputed_phase_recolors_nothing_else(self, no_persist):
        plain = build("hsx")
        plain.process_data()
        ti = build("hsx")
        ti.impute_phase(TARGET, 1000.0, name="A0")
        assert ti.color_map["A0"] not in plain.color_map.values()
        assert {p: c for p, c in ti.color_map.items() if p != "A0"} == plain.color_map


def test_envelope_decomposition_names_the_liquid(no_persist):
    record = build("hsx").impute_phase(TARGET, 1000.0, name="TX", apply=False)
    names = [p for p, _ in record.decomposition]
    assert "L" in names
    assert sum(w for _, w in record.decomposition) == pytest.approx(1.0)
    assert np.isfinite(record.enthalpy)
