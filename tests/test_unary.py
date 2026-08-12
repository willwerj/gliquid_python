"""
Unit tests for gliquid.phase -- the unified unary reference-state module.

Covers the scalar liquid reference API consumed by binary/ternary fitting and the
per-phase / entropy-reconciliation API that underpins solid-solution support.
"""

import sys
from pathlib import Path

import pytest
import sympy as sp

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

import gliquid.mpds as mpds
import gliquid.phase as phase
from gliquid.phase import UNARY, ComponentRef, Phase


def test_registry_loads_elements():
    assert len(UNARY.elements) > 50
    al = UNARY["Al"]
    assert isinstance(al, ComponentRef)
    # Values sourced from phase_transitions.json (default data dir).
    assert al.t_fusion == pytest.approx(933.5)
    assert al.h_liq == pytest.approx(10710.0)
    assert al.polymorphs, "Al should expose at least its ground-state polymorph"


def test_unknown_element_defaults_to_zero():
    xx = UNARY["Xx"]
    assert xx.h_liq == 0.0 and xx.s_liq == 0.0
    assert xx.t_fusion == 0.0 and xx.t_vaporization == 0.0
    assert xx.polymorphs == []


def test_component_data_returns_independent_copies():
    cd = UNARY.component_data(["Cu", "Mg"])
    assert set(cd) == {"Cu", "Mg"}
    assert all(isinstance(v, ComponentRef) for v in cd.values())
    # Mutating a per-system copy must not corrupt the shared registry.
    cd["Cu"].phases = []
    assert UNARY["Cu"].phases, "registry ComponentRef was mutated via a component_data copy"


def test_gibbs_ref_expr_matches_h_minus_ts():
    t = sp.Symbol("T")
    cu = UNARY["Cu"]
    expr = cu.gibbs_ref_expr(t)
    assert sp.simplify(expr - (cu.h_liq - t * cu.s_liq)) == 0


def test_phaseref_gibbs():
    p = Phase(phase_type="solid", enthalpy=1000.0, entropy=3.0)
    assert p.gibbs(1000) == pytest.approx(1000.0 - 1000 * 3.0)


def test_per_phase_solid_lookup():
    # Find any element carrying a solid polymorph with a recorded spacegroup, and confirm the
    # per-phase lookup (used by solid-solution reference states) returns that exact phase.
    for ref in UNARY.elements.values():
        for p in ref.phases:
            if p.phase_type == "solid" and p.spacegroup_number is not None:
                assert ref.solid_phase(p.spacegroup_number) is not None
                assert ref.solid_phase(p.spacegroup_number).spacegroup_number == p.spacegroup_number
                return
    pytest.skip("no solid phase with a spacegroup number in the dataset")


def test_liquid_ref_from_solids_reproduces_stored_liquid():
    # Forward-compat contract: rebuilding the liquid reference from the full solid ladder plus
    # fusion reproduces the stored cumulative h_liq/s_liq (to within JSON rounding of the steps).
    tested = 0
    for ref in UNARY.elements.values():
        solids = [p for p in ref.polymorphs if p.t_transition and p.t_transition > 0]
        if ref.liquid and solids:
            h, s = ref.liquid_ref_from_solids()
            assert h == pytest.approx(ref.h_liq)  # H is taken directly -> exact
            assert s == pytest.approx(ref.s_liq, abs=1e-2)  # S recomputed from rounded ΔH/T
            tested += 1
    assert tested > 0, "expected at least one element with a solid-solid transition"


def test_unknown_module_attr_still_raises():
    with pytest.raises(AttributeError):
        _ = mpds.some_nonexistent_attribute


def test_ss_constants():
    assert phase.SS_SPACEGROUPS == {"BCC": 229, "FCC": 225, "HCP": 194}
    assert phase.SS_SYMBOLS == {"BCC": "Im-3m", "FCC": "Fm-3m", "HCP": "P6_3/mmc"}
    assert phase.EV_ATOM_TO_J_MOL == pytest.approx(96485.0)


def test_copy_produces_fresh_phaserefs():
    cu = UNARY["Cu"]
    dup = cu.copy()
    assert dup.phases == cu.phases
    assert all(a is not b for a, b in zip(dup.phases, cu.phases)), (
        "copy() must not share Phase instances with the registry"
    )


def test_with_liquid_ref_overrides_copy_not_registry():
    cu = UNARY["Cu"]
    h0, s0 = cu.h_liq, cu.s_liq
    mod = cu.with_liquid_ref(12345.0, 6.789)
    assert mod.h_liq == pytest.approx(12345.0)
    assert mod.s_liq == pytest.approx(6.789)
    # gibbs_ref_expr reflects the override on the copy
    t = sp.Symbol("T")
    assert sp.simplify(mod.gibbs_ref_expr(t) - (12345.0 - t * 6.789)) == 0
    # registry (and the original instance) are untouched
    assert cu.h_liq == pytest.approx(h0) and cu.s_liq == pytest.approx(s0)
    assert UNARY["Cu"].h_liq == pytest.approx(h0)


def test_registry_polymorphs_carry_structure_metadata():
    for ref in UNARY.elements.values():
        for p in ref.polymorphs:
            if p.spacegroup_number is not None:
                assert not p.is_solution
                assert p.composition is not None
                # single-element chemistry (reduced_formula can be diatomic, e.g. 'H2')
                assert [el.symbol for el in p.composition.elements] == [ref.symbol]
                return
    pytest.skip("no polymorph with a spacegroup number in the dataset")
