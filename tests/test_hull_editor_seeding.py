"""ConvexHullEditor seeds each phase's entropy onto one entry at its composition, in either
component order; ``apply`` keeps the model consistent with the edited hull; added names stay
unique. Runs offline on a synthetic Hf-Zr hull and the cached Hf-Zr and Cu-Mg models.
"""

import io
import sys

import numpy as np
import pytest

ipywidgets = pytest.importorskip("ipywidgets")

from pymatgen.core import Composition

from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.hull_editor import J_PER_MOL_PER_EV, ConvexHullEditor
from gliquid.phase import UNARY, Phase

from ._mpds_builders import compound_hull

S_PHASE = 2.5  # J/mol-atom/K on the HfZr3 line compound


def _hull(elements):
    return compound_hull(("Hf", "Zr"), "HfZr3", -0.4, elements)


def _bl(sys_name, components):
    phases = [
        Phase(
            phase_type="solid",
            name="HfZr3",
            composition=Composition("HfZr3"),
            enthalpy=-30000.0,
            entropy=S_PHASE,
        ),
        Phase(phase_type="liquid", name="L"),
    ]
    return BinaryLiquid(
        sys_name,
        components,
        component_data=UNARY.component_data(components),
        dft_ch=_hull(components),
        phases=phases,
        temp_range=[300, 3000],
    )


_NO_OP_MODELS = {"Hf-Zr": {"pd_ind": 0, "solid_solutions": False}, "Cu-Mg": {}}


def _phase_rows(bl):
    """``(name, x, enthalpy, entropy)`` per phase of ``bl``, in model order."""
    return [
        (
            p.name,
            None if p.composition is None else float(p.fraction_in(bl.components)[0]),
            p.enthalpy,
            p.entropy,
        )
        for p in bl.phases
    ]


def _seeded_entropy(editor):
    """The editor's seeded entropy (eV/atom/K) on the HfZr3 entry."""
    for entry, s in zip(editor._entries, editor._entropies):
        if entry.composition.reduced_formula == "HfZr3":
            return s
    raise AssertionError("HfZr3 entry not found in editor state")


class TestEntropySeeding:
    def test_alphabetical_bl_seeds_unchanged(self):
        editor = ConvexHullEditor(_bl("Hf-Zr", ["Hf", "Zr"]))
        assert _seeded_entropy(editor) == pytest.approx(S_PHASE / J_PER_MOL_PER_EV)

    def test_reversed_bl_seeds_the_same_composition(self):
        editor = ConvexHullEditor(_bl("Zr-Hf", ["Zr", "Hf"]))
        assert _seeded_entropy(editor) == pytest.approx(S_PHASE / J_PER_MOL_PER_EV)

    def test_apply_round_trip_preserves_phases_both_orders(self):
        for sys_name, components in (("Hf-Zr", ["Hf", "Zr"]), ("Zr-Hf", ["Zr", "Hf"])):
            bl = _bl(sys_name, components)
            editor = ConvexHullEditor(bl)
            editor.apply()
            hfzr3 = next(p for p in bl.phases if p.name == "HfZr3")
            assert hfzr3.entropy == pytest.approx(S_PHASE, rel=1e-9)
            assert hfzr3.fraction_in(components)[0] == pytest.approx(
                Composition("HfZr3").get_atomic_fraction(components[1])
            )

    @pytest.mark.parametrize("name", ["Mg2Cu", "CuMg2"])
    def test_a_no_op_apply_keeps_a_compound_entropy_among_twins(self, name):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        (phase,) = [p for p in bl.phases if p.name == "Mg2Cu"]
        phase.name, phase.entropy = name, 5.0
        ConvexHullEditor(bl).apply()
        (mg2cu,) = [p for p in bl.phases if p.name == "Mg2Cu"]
        assert mg2cu.entropy == pytest.approx(5.0)

    def test_the_temperature_alone_drops_no_compound(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=False)
        editor = ConvexHullEditor(bl)
        assert not any(
            s for e, s in zip(editor._entries, editor._entropies) if e.composition.is_element
        )
        editor.add_entry("HfZr3", -0.01)
        editor.set_temperature(1500.0)
        editor.apply()
        assert "HfZr3" in {p.name for p in bl.phases}


class TestApplyRefreshesTheModel:
    def test_removing_the_compound_refreshes_the_hsx(self):
        bl = _bl("Hf-Zr", ["Hf", "Zr"])
        bl.update_phase_points()
        assert "HfZr3" in bl.hsx.phases
        editor = ConvexHullEditor(bl)
        (index,) = [
            i for i, e in enumerate(editor._entries) if e.composition.reduced_formula == "HfZr3"
        ]
        editor.remove_entry(index)
        editor.apply()
        assert bl.hsx.phases == bl._hsx_phase_names()
        assert "HfZr3" not in bl.hsx.phases

    def test_a_model_without_a_computed_hsx_applies_without_computing_one(self):
        bl = _bl("Hf-Zr", ["Hf", "Zr"])
        bl.temp_range = []
        ConvexHullEditor(bl).apply()
        assert bl.hsx is None

    def test_a_no_op_apply_keeps_the_solid_solutions(self, ss_from_cache):
        bl = ss_from_cache
        bl.update_phase_points()
        names = [p.name for p in bl.phases]
        interp = bl.eqs["h_hull_interp"].copy()
        ConvexHullEditor(bl).apply()
        assert [p.name for p in bl.phases] == names
        assert all(p.model is not None for p in bl.phases if p.name in bl.ss_models)
        assert np.array_equal(bl.eqs["h_hull_interp"], interp)

    @pytest.mark.parametrize("system", list(_NO_OP_MODELS))
    def test_a_no_op_apply_keeps_the_polymorph_order_and_hull_interp(self, system):
        bl = BinaryLiquid.from_cache(system, **_NO_OP_MODELS[system])
        rows = _phase_rows(bl)
        interp = bl.eqs["h_hull_interp"].copy()
        ConvexHullEditor(bl).apply()
        assert _phase_rows(bl) == rows
        assert np.array_equal(bl.eqs["h_hull_interp"], interp)
        by_name = {p.name: p for p in bl.phases}
        polymorphs = [poly for c in bl.components for poly in bl.component_data[c].polymorphs]
        assert polymorphs
        for poly in polymorphs:
            phase = by_name[poly.name]
            assert phase is not poly
            assert (phase.enthalpy, phase.entropy) == (poly.enthalpy, poly.entropy)

    def test_removing_a_compound_raises_the_hull_interp_and_recolours_the_plotter(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        bl.update_phase_points()
        plotter = BLPlotter(bl)
        colours = plotter._phase_color_map()
        interp = bl.eqs["h_hull_interp"].copy()
        editor = ConvexHullEditor(bl)
        (index,) = [i for i, e in enumerate(editor._entries) if e.name == "MgCu2"]
        editor.remove_entry(index)
        editor.apply()
        assert np.all(bl.eqs["h_hull_interp"] >= interp - 1e-9)
        assert (bl.eqs["h_hull_interp"] - interp).max() > 1000.0
        assert plotter._phase_color_map() == BLPlotter(bl)._phase_color_map() != colours

    def test_a_compound_unstable_at_the_editor_temperature_stays_in_the_model(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        interp = bl.eqs["h_hull_interp"].copy()
        editor = ConvexHullEditor(bl)
        (index,) = [i for i, e in enumerate(editor._entries) if e.name == "MgCu2"]
        editor.modify_entry(index, new_entropy=-50.0, units="J/mol")
        editor.set_temperature(1500.0)
        assert index not in editor._stable_idx
        editor.apply()
        assert "MgCu2" in {p.name for p in bl.phases}
        assert np.array_equal(bl.eqs["h_hull_interp"], interp)

    def test_an_entropy_stabilised_addition_joins_without_moving_the_hull_interp(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        interp = bl.eqs["h_hull_interp"].copy()
        editor = ConvexHullEditor(bl)
        editor.add_entry("CuMg", 1930.0, name="X_hiT", entropy=20.0, units="J/mol")
        editor.set_temperature(1500.0)
        editor.apply()
        assert "X_hiT" in {p.name for p in bl.phases}
        assert np.array_equal(bl.eqs["h_hull_interp"], interp)

    def test_a_proxied_model_is_refreshed(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        bl.update_phase_points()
        editor = ConvexHullEditor(_Proxy(bl))
        (index,) = [i for i, e in enumerate(editor._entries) if e.name == "MgCu2"]
        editor.remove_entry(index)
        editor.apply()
        assert "MgCu2" not in bl.hsx.phases

    def test_a_named_high_temperature_polymorph_joins_its_0k_compound(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        editor = ConvexHullEditor(bl)
        editor.add_entry("MgCu2", -12061.0, name="MgCu2_hiT", entropy=10.0, units="J/mol")
        editor.set_temperature(1200.0)
        editor.apply()
        assert {"MgCu2", "MgCu2_hiT"} <= {p.name for p in bl.phases}

    def test_a_same_named_high_temperature_twin_leaves_the_0k_phase(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        editor = ConvexHullEditor(bl)
        editor.add_entry("MgCu2", -12061.0, entropy=10.0, units="J/mol")
        editor.set_temperature(1200.0)
        editor.apply()
        (mgcu2,) = [p for p in bl.phases if p.name == "MgCu2"]
        assert mgcu2.entropy == 0.0

    def test_reapplying_an_editor_leaves_the_model_unchanged(self):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        editor = ConvexHullEditor(bl)
        editor.add_entry("CuMg", 1930.0, name="X_hiT", entropy=20.0, units="J/mol")
        editor.set_temperature(1500.0)
        editor.apply()
        first = [(p.name, p.enthalpy, p.entropy) for p in bl.phases]
        again = ConvexHullEditor(bl)
        again.set_temperature(1500.0)
        again.apply()
        assert [(p.name, p.enthalpy, p.entropy) for p in bl.phases] == first


class TestDisplay:
    def test_a_compound_kept_from_0k_is_marked(self):
        editor = ConvexHullEditor(BinaryLiquid.from_cache("Cu-Mg"))
        index = next(e["index"] for e in editor.entries if e["name"] == "MgCu2" and e["stable"])
        editor.modify_entry(index, new_entropy=-50.0, units="J/mol")
        editor.set_temperature(1500.0)
        row = editor.entries[index]
        assert (row["stable"], row["stable_at_0k"]) == (False, True)
        assert {e["index"] for e in editor.entries if e["stable_at_0k"]} == {
            e.attribute for e in editor._make_gibbs(0.0)[0].stable_entries
        }
        assert editor._table_html.value.count("<td>🟡</td>") == 1


class TestAddedNames:
    @pytest.mark.parametrize("name", ["L", "BCC", "beta-Hf (bcc)", "HfZr3"])
    def test_a_name_in_use_elsewhere_is_rejected(self, ss_from_cache, name):
        editor = ConvexHullEditor(ss_from_cache)
        if name == "HfZr3":
            editor.add_entry("HfZr3", -0.1)
        with pytest.raises(ValueError, match="already used"):
            editor.add_entry("Hf3Zr", -0.1, name=name)

    def test_the_formula_name_at_the_same_composition_is_accepted(self):
        editor = ConvexHullEditor(_bl("Hf-Zr", ["Hf", "Zr"]))
        editor.add_entry("HfZr3", -0.5)
        assert sum(e.composition.reduced_formula == "HfZr3" for e in editor._entries) == 2

    def test_a_name_at_the_same_fractional_composition_is_accepted(self):
        editor = ConvexHullEditor(BinaryLiquid.from_cache("Cu-Mg"))
        editor.add_entry("Cu0.75Mg0.25", -0.05, name="MgCu3")
        editor.add_entry("MgCu3", -0.06)
        assert editor._entries[-1].composition.reduced_formula == "MgCu3"


def test_the_change_log_survives_a_console_that_cannot_encode_its_symbols(monkeypatch):
    editor = ConvexHullEditor(_bl("Hf-Zr", ["Hf", "Zr"]))
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    editor._log("⚠ Undone ↩")
    stdout.flush()
    assert stdout.buffer.getvalue().decode("cp1252").startswith("? Undone ?")


class _Proxy:
    """Forwards every attribute of a model, so it is not an instance of the model's class."""

    def __init__(self, target):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        setattr(self._target, name, value)
