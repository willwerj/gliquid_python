"""Imputed binary phases must be able to compete on a TERNARY hull.

``TernaryLiquidInterpolation`` builds its solid side from the shared DFT entries cache, which
holds real DFT entries only. An imputed phase carries an ABSOLUTE energy anchored to the
elemental references of the hull it was inverted on, so replaying it onto a ternary hull built
from a different entries snapshot without re-anchoring shifts its formation energy by the
reference delta and produces a plausible, wrong number.

Runs offline against the Al-Mg-Si cache that ships for tests. Every injected entry is built
with a deliberately meaningless absolute energy (``energy=0``).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.hull_editor as hull_editor
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.core as imp_core
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

COMPONENTS = ["Al", "Mg", "Si"]
_IDEAL = [0.0, 0.0, 0.0, 0.0]
ZERO_MIX = {"Al-Mg": _IDEAL, "Mg-Si": _IDEAL, "Si-Al": _IDEAL}
DELTA = 0.25  # coarse: the injected phase sits on a vertex of this grid
T_INCR = 20

DEEP_LABEL = "MgSi(imputed)"
DEEP_H_J = -140000.0  # J/mol-atom: far below anything on the Al-Mg-Si hull
SHALLOW_LABEL = "AlSi(imputed)"
SHALLOW_H_J = 50000.0  # positive formation enthalpy: cannot reach the hull


def entry_dict(elements, x_b: float, h_form_j: float, label: str) -> dict:
    """A tagged imputed entry dict whose stored absolute energy is deliberately wrong."""
    a, b = elements
    return ComputedEntry(
        composition=Composition({a: 1.0 - x_b, b: x_b}),
        energy=0.0,
        entry_id=f"imputed:{label}",
        data={
            "imputed": True,
            "label": label,
            "x_b": float(x_b),
            "h_form_j_per_mol": float(h_form_j),
        },
    ).as_dict()


def build(imputed_entries=None, **kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(COMPONENTS),
        delta=DELTA,
        interp_scheme="linear",
        param_format="linear",
        xs_mix=ZERO_MIX,
        T_incr=T_INCR,
        order="given",
        imputed_entries=imputed_entries,
        **kwargs,
    )


@pytest.fixture(scope="module")
def deep_entry():
    return entry_dict(["Mg", "Si"], 0.5, DEEP_H_J, DEEP_LABEL)


@pytest.fixture(scope="module")
def shallow_entry():
    return entry_dict(["Al", "Si"], 0.5, SHALLOW_H_J, SHALLOW_LABEL)


@pytest.fixture(scope="module")
def plain():
    ti = build()
    return ti, ti.get_ternary_form_en(ti.components)


@pytest.fixture(scope="module")
def injected(deep_entry):
    ti = build([deep_entry])
    return ti, ti.get_ternary_form_en(ti.components)


class TestInjectedEntryReachesTheHull:
    def test_the_count_of_entries_handed_in_is_reported(self, injected):
        ti, _ = injected
        assert ti.ternary_meta["n_imputed_entries"] == 1

    def test_the_count_that_reached_the_hull_is_reported(self, injected):
        ti, _ = injected
        assert ti.ternary_meta["n_imputed_stable"] == 1

    def test_the_phase_is_listed_under_its_mpds_label(self, injected):
        _, df = injected
        assert DEEP_LABEL in set(df["Phase Name"])

    def test_the_energy_is_re_anchored_not_the_stored_one(self, injected):
        """The entry went in at ``energy=0``; the hull must carry its FORMATION enthalpy."""
        _, df = injected
        row = df[df["Phase Name"] == DEEP_LABEL].iloc[0]
        assert float(row["H"]) == pytest.approx(DEEP_H_J, abs=1.0)

    def test_the_solid_hull_moves(self, plain, injected):
        _, before = plain
        _, after = injected
        assert set(after["Phase Name"]) != set(before["Phase Name"])

    def test_it_displaces_the_dft_vertex_it_undercuts(self, plain, injected):
        """The injected phase competes on the hull, it is not merely appended to the frame."""
        _, before = plain
        _, after = injected
        assert "Mg2Si" in set(before["Phase Name"])
        assert "Mg2Si" not in set(after["Phase Name"])

    def test_the_hull_metadata_is_taken_after_injection(self, plain, injected):
        ti_before, _ = plain
        ti_after, _ = injected
        assert ti_after.ternary_meta["deepest_formation_energy"] == pytest.approx(DEEP_H_J, abs=1.0)
        assert ti_before.ternary_meta["deepest_formation_energy"] > DEEP_H_J
        assert "n_ternary_compounds" in ti_after.ternary_meta

    def test_the_edge_liquidus_moves(self, deep_entry):
        before = _edge_liquidus(build())
        after = _edge_liquidus(build([deep_entry]))
        assert math.isfinite(before)
        assert after > before


class TestAnInjectedEntryIsNeverDroppedSilently:
    def test_an_entry_that_misses_the_hull_is_still_counted(self, shallow_entry):
        ti = build([shallow_entry])
        ti.get_ternary_form_en(ti.components)
        assert ti.ternary_meta["n_imputed_entries"] == 1
        assert ti.ternary_meta["n_imputed_stable"] == 0

    def test_an_entry_with_no_formation_enthalpy_raises(self):
        bad = entry_dict(["Mg", "Si"], 0.5, DEEP_H_J, "Bad")
        bad["data"].pop("h_form_j_per_mol")
        ti = build([bad])
        with pytest.raises(ValueError, match="h_form_j_per_mol"):
            ti.get_ternary_form_en(ti.components)


class TestTheDefaultPathDoesNotMove:
    """Negative control: with no imputed entries nothing about the solid side changes."""

    def test_the_counts_read_zero(self, plain):
        ti, _ = plain
        assert ti.ternary_meta["n_imputed_entries"] == 0
        assert ti.ternary_meta["n_imputed_stable"] == 0

    def test_the_frame_is_unchanged(self, plain):
        ti, df = plain
        again = TernaryLiquidInterpolation(
            list(COMPONENTS),
            delta=DELTA,
            interp_scheme="linear",
            param_format="linear",
            xs_mix=ZERO_MIX,
            T_incr=T_INCR,
            order="given",
        )
        pd.testing.assert_frame_equal(again.get_ternary_form_en(again.components), df)
        assert ti.components == again.components

    def test_no_dft_phase_is_renamed(self, plain):
        _, df = plain
        assert all(isinstance(name, str) and name for name in df["Phase Name"])


class TestTheListTravelsWithTheModel:
    def test_with_component_order_carries_it(self, deep_entry):
        ti = build([deep_entry])
        reframed = ti.with_component_order("Si-Al-Mg")
        assert reframed.imputed_entries == [deep_entry]

    def test_the_plotter_passes_it_through(self, deep_entry):
        plotter = TLIPlotter.from_components(
            list(COMPONENTS),
            delta=DELTA,
            interp_scheme="linear",
            param_format="linear",
            xs_mix=ZERO_MIX,
            T_incr=T_INCR,
            imputed_entries=[deep_entry],
        )
        assert plotter._ti.imputed_entries == [deep_entry]

    def test_from_binaries_collects_them_only_when_asked(self, deep_entry):
        binaries = [_FakeBinary(("Al", "Mg"), []), _FakeBinary(("Mg", "Si"), [deep_entry])]
        binaries.append(_FakeBinary(("Si", "Al"), []))
        off = TernaryLiquidInterpolation.from_binaries(binaries, delta=DELTA)
        on = TernaryLiquidInterpolation.from_binaries(binaries, delta=DELTA, include_imputed=True)
        assert off.imputed_entries == []
        assert [e["entry_id"] for e in on.imputed_entries] == [f"imputed:{DEEP_LABEL}"]


class _FakeHull:
    def __init__(self, entry_dicts):
        self.all_entries = [ComputedEntry.from_dict(d) for d in entry_dicts]


class _FakeBinary:
    """The narrow surface ``from_binaries`` reads: components, mixing model, format, hull."""

    def __init__(self, components, imputed_entry_dicts):
        from gliquid.solution import RKPolyExp

        self.components = list(components)
        self.xs_mix = RKPolyExp("linear", list(_IDEAL))
        self.param_format = "linear"
        self.tau = self.xs_mix.tau
        self.ss_models = {}
        self.dft_ch = _FakeHull(imputed_entry_dicts)


def _edge_liquidus(ti) -> float:
    """Lowest rendered liquid temperature (K) at the Mg-Si edge midpoint.

    Infinity when the midpoint is solid everywhere on the temperature grid.
    """
    ti.interpolate()
    ti.process_data()
    df = pd.concat(ti.equil_df_list, ignore_index=True)
    liq = (
        df[df["Phase"] == "L"]
        .sort_values("T")
        .drop_duplicates(subset=["x0_orig", "x1_orig"], keep="first")
    )
    for row in liq.itertuples():
        if round(float(row.x0_orig), 4) == 0.5 and round(float(row.x1_orig), 4) == 0.5:
            return float(row.T) + 273.15
    return math.inf


def test_the_conversion_constant_is_the_shared_one():
    assert hull_editor.J_PER_MOL_PER_EV is EV_ATOM_TO_J_MOL
    assert not hasattr(imp_core, "J_PER_MOL_PER_EV")
    assert not hasattr(imp_binary, "J_PER_MOL_PER_EV")
