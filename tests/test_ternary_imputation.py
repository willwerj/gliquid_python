"""Ternary phase-energy imputation (``gliquid.imputation.ternary_imputation``).

The headline property: a phase imputed at temperature T leaves the processed ternary hull at T.
Checked in both hull backends, with a control that moves the energy and must move the
temperature, and as a round trip -- a phase injected at a known enthalpy, its drawn top read
off, the phase removed and imputed back at that top, must recover the enthalpy.

Runs offline against the shipped Al-Mg-Si entries record, which has no stable ternary
compound, so every imputed phase is synthetic.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import gliquid.config as config
import gliquid.imputation.core as imp_core
import gliquid.imputation.lifecycle as imp_lifecycle
import gliquid.imputation.ternary_imputation as imp_ternary
from gliquid.cache import CacheModeError, DirectoryBackend, imputed_entry_label
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

COMPONENTS = ["Al", "Mg", "Si"]
IDEAL = {"Al-Mg": [0.0] * 4, "Mg-Si": [0.0] * 4, "Si-Al": [0.0] * 4}
MIXING = {"Al-Mg": [-20000.0, 0, 0, 0], "Mg-Si": [-30000.0, 0, 0, 0], "Si-Al": [-10000.0, 0, 0, 0]}
TARGET = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
MG2SI = {"Al": 0.0, "Mg": 2 / 3, "Si": 1 / 3}
DELTA = 0.1
T_INCR = 5


def build(mode="hsx", xs_mix=IDEAL, **kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(COMPONENTS),
        delta=DELTA,
        xs_mix=xs_mix,
        param_format="linear",
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


# -- Plotting --


def dashed(fig) -> list:
    return [t for t in fig.data if t.type == "scatter3d" and t.line.dash == "dash"]


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
