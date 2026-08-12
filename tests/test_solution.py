"""Tests for gliquid.solution: the ComponentRef-native solid-solution reference resolvers.

Re-pins the dev-code numbers frozen in fixtures/ss_characterization_pins.json through the
package implementation — the pins must reproduce bit-for-bit (rtol 1e-9). Runs entirely
offline against the Hf-Zr fixtures shipped in the package data/ directory.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pin_utils as pu  # noqa: E402

import gliquid.api as api
import gliquid.solution as sd
from gliquid.phase import UNARY, ComponentRef, Phase

PINS = json.loads(
    (Path(__file__).parent / "fixtures" / "ss_characterization_pins.json").read_text()
)


def _approx_deep(actual, expected, rel=1e-9, path=""):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual) == set(expected), f"{path}: key mismatch {set(actual) ^ set(expected)}"
        for k in expected:
            _approx_deep(actual[k], expected[k], rel, f"{path}.{k}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: length mismatch"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _approx_deep(a, e, rel, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=rel, abs=1e-12), (
            f"{path}: {actual} != {expected}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_constants_come_from_unary():
    import gliquid.phase as phase

    assert sd.ALL_SS_PHASES == list(phase.SS_SPACEGROUPS)
    assert sd.EV_ATOM_TO_J_MOL is phase.EV_ATOM_TO_J_MOL
    assert sd.SS_SPACEGROUPS is phase.SS_SPACEGROUPS


def _pin_to_keyed(pin_models, components):
    """The frozen dev-pin models (binary-locked schema) re-expressed in the keyed schema.

    Same VALUES, new packaging (omega/delta_h/delta_s maps) — the fixture itself stays
    byte-frozen for provenance.
    """
    comps = sorted(components)
    pair = "-".join(comps)
    keyed = {}
    for phase, model in pin_models.items():
        keyed[phase] = {
            "refs": model["refs"],
            "ref_mode": model["ref_mode"],
            "omega": {pair: model["omega_jmol"]},
            "delta_h": {comps[0]: model["deltaH_a_jmol"], comps[1]: model["deltaH_b_jmol"]},
            "delta_s": {comps[0]: model["deltaS_a_jmol_k"], comps[1]: model["deltaS_b_jmol_k"]},
        }
    return keyed


@pytest.mark.parametrize("ref_mode", ["from_omegas_file", "from_unary_db"])
def test_ref_modes_match_dev_pins(ref_mode):
    component_data = UNARY.component_data(["Hf", "Zr"])
    models = sd.load_solid_solution_models(
        components=["Hf", "Zr"], component_data=component_data, ref_mode=ref_mode
    )
    # fixtures are byte-frozen under the old labels; translate the lookup key and relabel
    # the pin's ref_mode/source strings to the new API names for the value comparison.
    pin = PINS["pin3"][pu.FIXTURE_REF_MODE[ref_mode]]
    _approx_deep(
        models,
        pu.relabel_ref_modes(_pin_to_keyed(pin["models"], ["Hf", "Zr"])),
        path=f"{ref_mode}.models",
    )
    reconciled = {
        el: {"H_liq": component_data[el].h_liq, "S_liq": component_data[el].s_liq}
        for el in component_data
    }
    _approx_deep(reconciled, pin["reconciled"], path=f"{ref_mode}.reconciled")


def test_binary_cache_mode_matches_dev_pins():
    entries = api.get_dft_structure_entries("Hf-Zr")  # package data/ fixture, offline
    component_data = UNARY.component_data(["Hf", "Zr"])
    models = sd.load_solid_solution_models(
        components=["Hf", "Zr"],
        component_data=component_data,
        entries=entries,
        ref_mode="from_dft_entries",
    )
    pin = PINS["pin3"]["binary-cache"]  # frozen fixture key stays under the old label
    _approx_deep(
        models,
        pu.relabel_ref_modes(_pin_to_keyed(pin["models"], ["Hf", "Zr"])),
        path="from_dft_entries.models",
    )
    reconciled = {
        el: {"H_liq": component_data[el].h_liq, "S_liq": component_data[el].s_liq}
        for el in component_data
    }
    _approx_deep(reconciled, pin["reconciled"], path="binary-cache.reconciled")


def test_lattice_stability_block_supplies_missing_phases():
    # Since the tiered-policy rebuild, from_unary_db never reads config.omegas_file at
    # runtime: structures the ladder lacks come from the unary DB's builder-baked
    # lattice_stabilities block (same omegas energies, same anchoring math, baked in).
    component_data = UNARY.component_data(["Hf", "Zr"])
    models = sd.load_solid_solution_models(
        components=["Hf", "Zr"], component_data=component_data, ref_mode="from_unary_db"
    )
    # Neither Hf nor Zr has an FCC polymorph on its ladder; the block supplies both refs.
    assert set(models) == {"BCC", "HCP", "FCC"}
    assert "lattice_stability" in models["FCC"]["refs"]["Hf"]["source"]
    assert models["FCC"]["refs"]["Hf"]["material_id"] == "omegas_hcp"
    assert "lattice_stability" not in models["BCC"]["refs"]["Hf"].get("source", "")


def test_reconciliation_does_not_corrupt_registry():
    # from_omegas_file: resolver enthalpies come from the omegas file, so the reconciled
    # liquid reference genuinely diverges from the stored one. from_unary_db does NOT reconcile
    # at all -- its refs already ARE the stored ladder, on the same zero (see
    # load_solid_solution_models); re-deriving from an SS-phase subset could only lose steps.
    h0, s0 = UNARY["Hf"].h_liq, UNARY["Hf"].s_liq
    component_data = UNARY.component_data(["Hf", "Zr"])
    sd.load_solid_solution_models(
        components=["Hf", "Zr"], component_data=component_data, ref_mode="from_omegas_file"
    )
    assert component_data["Hf"].h_liq != pytest.approx(h0)  # copy was reconciled...
    assert UNARY["Hf"].h_liq == pytest.approx(h0)  # ...registry untouched
    assert UNARY["Hf"].s_liq == pytest.approx(s0)


def test_reconciled_liquid_ref_synthetic_ladder():
    # Mirrors pin4: one solid step (cum 3000 J/mol at 1200 K) + fusion (20000 J/mol at 2000 K)
    ref = ComponentRef(
        "Xx",
        [
            Phase(phase_type="solid", t_transition=0, delta_h=0.0, spacegroup_number=194),
            Phase(phase_type="solid", t_transition=1200.0, delta_h=3000.0, spacegroup_number=229),
            Phase(phase_type="liquid", t_transition=2000.0, delta_h=20000.0, enthalpy=23000.0),
        ],
    )
    h, s = sd._reconciled_liquid_ref(ref, [(3000.0, 1200.0)])
    pin = PINS["pin4"]
    assert h == pytest.approx(pin["h_liq"])  # 23000
    assert s == pytest.approx(pin["s_liq"])  # 3000/1200 + 20000/2000 = 12.5


def test_ordered_solid_steps_matches_dev_pin():
    hf = UNARY["Hf"].copy()
    steps = sd._ordered_solid_steps(
        hf, "Hf", {"BCC": {"Hf": {"delta_h_jmol": 5000.0}}, "HCP": {"Hf": {"delta_h_jmol": 0.0}}}
    )
    # t_melt from the registry (2500 in the dev pin was an explicit arg; Hf melts at 2506 K,
    # so the BCC step at 2016 K survives either way and HCP (T=0) is excluded)
    _approx_deep(
        [list(t) for t in steps],
        [list(t) for t in PINS["pin4"]["ordered_steps_hf"]],
        path="ordered_steps_hf",
    )


def test_get_dft_structure_entries_offline_and_typed():
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    entries = api.get_dft_structure_entries("Hf-Zr")
    assert entries and all(isinstance(e, ComputedStructureEntry) for e in entries)
    assert all(e.structure is not None for e in entries)


# ---------------------------------------------------------------------------------------
# Reference-frame invariants.
#
# The pure-element edge of a fitted diagram is not a tabulated melting point -- it is the hull
# crossing of G_liquid with the lowest endpoint solid. It therefore lands on t_fusion only while
# the SS phase references and the elemental polymorph line compounds share one reference frame.
# The Hf-Zr fixtures above cannot see a break in that frame: both elements have exactly ONE solid
# transition below melting, so their step and cumulative enthalpies coincide. Ti, Mn, Fe, La, U
# and Pu do not, and publishing step values for them moved Ti's edge by 458 K.
#
# These tests are driven off the whole unary registry so a multi-transition element cannot slip
# through again, and they sweep every subset of {BCC, FCC, HCP} because which SS phases survive
# packaging -- and therefore which polymorphs are excluded from the line compounds -- is a
# property of the PARTNER element, not of the element under test.
# ---------------------------------------------------------------------------------------
def _elements_with_ss_polymorphs():
    return sorted(
        el
        for el, ref in UNARY.elements.items()
        if ref.t_fusion and any(sd._find_matching_polymorph(p, ref) for p in sd.ALL_SS_PHASES)
    )


def _endpoint_melting_point(ref, phase_refs, el, subset):
    """T where the liquid overtakes the last endpoint solid, given `subset` SS phases loaded.

    Mirrors the hull: a polymorph whose spacegroup a loaded SS phase covers is dropped from the
    line compounds (binary.build_phases_from_chull) and represented by that phase's g_ref instead.
    The liquid is last to be overtaken, so the melting point is the MAX crossing, not the min.
    """
    kept = {p: r[el] for p, r in phase_refs.items() if p in subset and el in r}
    covered = {sd.SS_SPACEGROUPS[p] for p in kept}
    solids = [
        (p.enthalpy or 0.0, p.entropy or 0.0)
        for p in ref.polymorphs
        if p.spacegroup_number not in covered
    ]
    solids += [(r["delta_h_jmol"], r["delta_s_jmol_k"]) for r in kept.values()]
    crossings = [(ref.h_liq - h) / (ref.s_liq - s) for h, s in solids if ref.s_liq - s > 1e-9]
    return max(crossings) if crossings else None


@pytest.mark.parametrize("el", _elements_with_ss_polymorphs())
def test_stored_ladder_is_internally_consistent(el):
    """Every element's stored (h_liq, s_liq) must be reproducible from its own full ladder.

    This is the property the from_unary_db frame relies on: because its refs ARE the stored
    ladder, the liquid reference is already in their frame and must not be re-derived. A DB entry
    whose cumulative values disagree with its steps would break that silently.
    """
    ref = UNARY[el]
    h, s = ref.liquid_ref_from_solids()  # None -> the element's full solid ladder
    # phase_transitions.json stores entropies rounded to 4 decimals, so the re-derived sum agrees
    # only to that precision; the tolerance tracks the file's storage precision, not the physics.
    assert h == pytest.approx(ref.h_liq, abs=1e-6)
    assert s == pytest.approx(ref.s_liq, abs=1e-3)


@pytest.mark.parametrize("el", _elements_with_ss_polymorphs())
def test_endpoint_melting_point_holds_under_every_ss_subset(el):
    """The pure-element edge must land on t_fusion whichever SS phases the partner leaves."""
    import itertools

    ref = UNARY[el]
    _, phase_refs = sd._resolve_refs_db([el], UNARY.component_data([el]))
    phases = sorted(phase_refs)
    for n in range(1, len(phases) + 1):
        for subset in itertools.combinations(phases, n):
            t = _endpoint_melting_point(ref, phase_refs, el, set(subset))
            assert t == pytest.approx(ref.t_fusion, abs=1.0), (
                f"{el} melts at {t:.1f} K with SS phases {subset} loaded, "
                f"but its reference melting point is {ref.t_fusion} K"
            )


def test_unary_db_publishes_cumulative_not_step_references():
    """Ti is the discriminating case: two transitions below melting, so step != cumulative."""
    _, phase_refs = sd._resolve_refs_db(["Ti"], UNARY.component_data(["Ti"]))
    beta_ti = UNARY["Ti"].solid_phase(229)
    assert beta_ti.delta_h != pytest.approx(beta_ti.enthalpy)  # the two really do differ
    assert phase_refs["BCC"]["Ti"]["delta_h_jmol"] == pytest.approx(beta_ti.enthalpy)
    assert phase_refs["BCC"]["Ti"]["delta_s_jmol_k"] == pytest.approx(beta_ti.entropy)


def test_duplicate_spacegroup_resolves_to_the_pre_melt_polymorph():
    """Fe has alpha-Fe and delta-Fe, both Im-3m/229.

    The SS phase must reference delta-Fe -- the solid actually in equilibrium with the liquid --
    because the spacegroup exclusion removes BOTH from the line compounds.
    """
    fe = UNARY["Fe"]
    matches = [p for p in fe.polymorphs if p.spacegroup_number == 229]
    assert len(matches) > 1, "fixture assumes Fe still carries two BCC polymorphs"
    picked = sd._find_matching_polymorph("BCC", fe)
    assert picked.t_transition == max(
        p.t_transition for p in matches if p.t_transition < fe.t_fusion
    )
    assert picked.enthalpy > 0  # delta-Fe, not the alpha-Fe ground state


def test_reconciled_liquid_ref_two_step_ladder():
    """Two steps distinguish cumulative from step inputs; the single-step pin above cannot.

    _reconciled_liquid_ref takes CUMULATIVE enthalpies and differences them, so feeding it the
    per-transition steps understates both H and S.
    """
    ref = ComponentRef(
        "Xx",
        [
            Phase(phase_type="solid", t_transition=0, delta_h=0.0, spacegroup_number=191),
            Phase(phase_type="solid", t_transition=200.0, delta_h=1000.0, spacegroup_number=194),
            Phase(phase_type="solid", t_transition=1000.0, delta_h=4000.0, spacegroup_number=229),
            Phase(phase_type="liquid", t_transition=2000.0, delta_h=20000.0, enthalpy=25000.0),
        ],
    )
    # cumulative inputs: HCP at 1000, BCC at 1000 + 4000 = 5000
    h, s = sd._reconciled_liquid_ref(ref, [(1000.0, 200.0), (5000.0, 1000.0)])
    assert h == pytest.approx(5000.0 + 20000.0)
    assert s == pytest.approx(1000.0 / 200.0 + 4000.0 / 1000.0 + 20000.0 / 2000.0)


# ---------------------------------------------------------------------------------------
# Omegas fallback: an element that never exhibits a structure has no polymorph entry for it, so
# the primary resolver produces no reference and the whole SS phase is dropped -- blocking a
# solid solution the omegas file could support. Since the tiered-policy rebuild this path
# serves the from_dft_entries / legacy modes only (from_unary_db bakes the same energies into
# its lattice_stabilities block at build time), so these tests drive it with synthetic refs
# rather than live registry elements, whose coverage is now complete.
# ---------------------------------------------------------------------------------------
def _omegas_elements_block(**per_phase):
    """{phase: {element: eV/atom}} shaped like the omegas file's `elements` block."""
    return {phase: dict(vals) for phase, vals in per_phase.items()}


def _bcc_only_phase_refs(el: str, bcc_ev: float, delta_h: float) -> dict:
    return {
        "BCC": {
            el: sd._make_phase_ref(
                ss_phase="BCC",
                material_id="mp-test",
                energy_ev_per_atom=bcc_ev,
                delta_h_jmol=delta_h,
                delta_s_jmol_k=1.0,
            )
        }
    }


def test_fallback_anchors_a_missing_phase_on_the_primary_ladder():
    """A missing structure anchors onto the shared phase's cumulative enthalpy."""
    bcc_ev = -9.5
    phase_refs = _bcc_only_phase_refs("Xx", bcc_ev, delta_h=1234.0)
    # FCC placed 0.05 eV/atom above BCC; the anchor shifts that onto BCC's cumulative enthalpy.
    elements = _omegas_elements_block(BCC={"Xx": bcc_ev}, FCC={"Xx": bcc_ev + 0.05})
    sd._apply_omegas_fallback(["Xx"], phase_refs, elements, "from_dft_entries")

    expected = 1234.0 + 0.05 * sd.EV_ATOM_TO_J_MOL
    assert phase_refs["FCC"]["Xx"]["delta_h_jmol"] == pytest.approx(expected)
    assert phase_refs["FCC"]["Xx"]["delta_s_jmol_k"] == 0.0  # never-exhibited structure
    assert "omegas_fallback" in phase_refs["FCC"]["Xx"]["source"]


def test_fallback_uses_the_omegas_zero_when_no_phase_is_shared():
    """No BCC/FCC/HCP reference at all -> the legacy omegas zero, as _resolve_refs_legacy."""
    phase_refs = {}
    elements = _omegas_elements_block(BCC={"Xx": -11.0}, FCC={"Xx": -10.8})
    sd._apply_omegas_fallback(["Xx"], phase_refs, elements, "from_dft_entries")
    # lowest omegas energy becomes the zero, exactly as _resolve_refs_legacy does
    assert phase_refs["BCC"]["Xx"]["delta_h_jmol"] == pytest.approx(0.0)
    assert phase_refs["FCC"]["Xx"]["delta_h_jmol"] == pytest.approx(0.2 * sd.EV_ATOM_TO_J_MOL)


def test_fallback_refuses_a_phase_below_the_primary_ground_state(caplog):
    """A negative anchored enthalpy means the two sources disagree about the ground state.

    The phase stays edge-only and the conflict is logged. (For from_unary_db this guard's
    job moved into the BUILDER as a hard error; the runtime path keeps it for the modes
    that still resolve references at load time.)
    """
    bcc_ev = -9.5
    phase_refs = _bcc_only_phase_refs("Xx", bcc_ev, delta_h=1234.0)
    elements = _omegas_elements_block(BCC={"Xx": bcc_ev}, FCC={"Xx": bcc_ev - 0.05})
    with caplog.at_level("WARNING"):
        sd._apply_omegas_fallback(["Xx"], phase_refs, elements, "from_dft_entries")
    assert "FCC" not in phase_refs
    assert "disagree about Xx's ground state" in caplog.text


def test_fallback_never_overwrites_a_primary_reference():
    component_data = UNARY.component_data(["Cr"])
    _, phase_refs = sd._resolve_refs_db(["Cr"], component_data)
    before = dict(phase_refs["BCC"]["Cr"])
    elements = _omegas_elements_block(BCC={"Cr": -99.0})  # absurd value, must be ignored
    sd._apply_omegas_fallback(["Cr"], phase_refs, elements, "from_unary_db")
    assert phase_refs["BCC"]["Cr"] == before


@pytest.mark.parametrize("ref_mode", ["from_unary_db", "from_omegas_file"])
def test_uncovered_system_is_a_clean_noop(ref_mode):
    """A system the omegas file does not cover must return {} WITHOUT mutating
    component_data and WITHOUT raising — so 'SS enabled + uncovered' == 'SS off'.

    Cr-Eu is uncovered by the shipped 3-element (Hf/Ti/Zr) omegas fixture: Eu is absent.
    Before the coverage gate, from_unary_db raised KeyError (after already mutating Cr's
    s_liq) and from_omegas_file raised on the empty available_phases set.
    """
    component_data = UNARY.component_data(["Cr", "Eu"])
    before = {el: (r.h_liq, r.s_liq) for el, r in component_data.items()}
    models = sd.load_solid_solution_models(
        components=["Cr", "Eu"], component_data=component_data, ref_mode=ref_mode
    )
    assert models == {}
    after = {el: (r.h_liq, r.s_liq) for el, r in component_data.items()}
    assert after == before  # no reconciliation mutation; no exception raised
