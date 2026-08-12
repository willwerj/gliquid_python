"""Tier B lattice-stability block: loader (phase.py) and resolver (solution.py).

All synthetic — valid against any phase_transitions.json vintage. The block
supersedes the runtime omegas fallback for ``from_unary_db``: builder-baked
cumulative enthalpies above the element anchor, no transition temperature,
kept OUT of ``polymorphs``/hull/liquid machinery by construction.

Entropy is 0 by the recalculation convention (nothing to divide by without a
transition temperature), except where the builder's scoped metastable-entropy
exception emits a negative SGTE value — see ``TestMetastableEntropyException``.
"""

import json

import pytest

import gliquid.config as config
import gliquid.solution as sd
from gliquid.phase import ComponentRef, Phase, UnaryData

SYNTH = {
    "elements": {
        "Fe": {
            "symbol": "Fe",
            "phases": [
                {
                    "phase_type": "solid",
                    "common_name": "alpha-Fe (bcc)",
                    "spacegroup_number": 229,
                    "spacegroup_symbol": "Im-3m",
                    "transition_temperature_K": 0,
                    "enthalpy_J_per_mol": 0.0,
                    "entropy_J_per_mol_K": 0.0,
                    "delta_H_J_per_mol": 0.0,
                    "delta_S_J_per_mol_K": 0.0,
                    "materials_project_id": "mp-13",
                },
                {
                    "phase_type": "liquid",
                    "common_name": "Liquid Iron",
                    "transition_temperature_K": 1811.0,
                    "enthalpy_J_per_mol": 13810.0,
                    "entropy_J_per_mol_K": 7.6256,
                    "delta_H_J_per_mol": 13810.0,
                    "delta_S_J_per_mol_K": 7.6256,
                },
            ],
            "lattice_stabilities": {
                "HCP": {
                    "spacegroup_number": 194,
                    "spacegroup_symbol": "P6_3/mmc",
                    "delta_H_J_per_mol": 9436.84,
                    "delta_S_J_per_mol_K": 0.0,
                    "transition_temperature_K": None,
                    "materials_project_id": "mp-136",
                    "metadata": {"source": "DFT (Materials Project)", "basis": "element anchor"},
                },
                "FCC": {
                    "spacegroup_number": 225,
                    "spacegroup_symbol": "Fm-3m",
                    "delta_H_J_per_mol": 5000.0,
                    "delta_S_J_per_mol_K": 0.0,
                    "transition_temperature_K": None,
                    "materials_project_id": "omegas_hcp",
                    "metadata": {"source": "omegas_hcp.json", "basis": "anchored on HCP"},
                },
            },
        },
    },
}


@pytest.fixture()
def synth_registry(tmp_path, monkeypatch):
    path = tmp_path / "phase_transitions.json"
    path.write_text(json.dumps(SYNTH))
    monkeypatch.setattr(config, "phase_transitions_file", path)
    return UnaryData(require=True)


class TestLoader:
    def test_block_parses_into_separate_field(self, synth_registry):
        ref = synth_registry["Fe"]
        assert len(ref.lattice_stabilities) == 2
        hcp = next(p for p in ref.lattice_stabilities if p.spacegroup_number == 194)
        assert hcp.enthalpy == 9436.84 and hcp.entropy == 0.0
        assert hcp.t_transition is None
        assert hcp.imputed is False and hcp.source == "DFT (Materials Project)"
        fcc = next(p for p in ref.lattice_stabilities if p.spacegroup_number == 225)
        assert fcc.imputed is True  # omegas-sourced entries carry the dashed-provenance flag

    def test_block_never_leaks_into_ladder_machinery(self, synth_registry):
        ref = synth_registry["Fe"]
        # polymorphs filter requires a transition temperature -> block excluded,
        # so hull line compounds and liquid reconstruction never see it.
        assert {p.spacegroup_number for p in ref.polymorphs} == {229}
        assert ref.solid_phase(194) is None
        h, s = ref.liquid_ref_from_solids()
        assert (h, s) == (13810.0, pytest.approx(13810.0 / 1811.0))

    def test_copy_carries_fresh_block_instances(self, synth_registry):
        ref = synth_registry["Fe"]
        dup = ref.copy()
        assert [p.spacegroup_number for p in dup.lattice_stabilities] == [
            p.spacegroup_number for p in ref.lattice_stabilities
        ]
        assert dup.lattice_stabilities[0] is not ref.lattice_stabilities[0]


class TestResolver:
    def test_block_fills_slots_the_ladder_misses(self, synth_registry):
        component_data = {"Fe": synth_registry["Fe"].copy()}
        _, phase_refs = sd._resolve_refs_db(["Fe"], component_data)
        assert phase_refs["BCC"]["Fe"]["delta_h_jmol"] == 0.0  # from the ladder
        assert phase_refs["HCP"]["Fe"]["delta_h_jmol"] == 9436.84  # from the block
        assert phase_refs["HCP"]["Fe"]["delta_s_jmol_k"] == 0.0
        assert "lattice_stability" in phase_refs["HCP"]["Fe"]["source"]
        assert phase_refs["FCC"]["Fe"]["material_id"] == "omegas_hcp"

    def test_ladder_always_wins_over_the_block(self, synth_registry):
        ref = synth_registry["Fe"].copy()
        # Give the block a bogus BCC entry; the ladder's BCC must still win.
        ref.lattice_stabilities.append(
            Phase(phase_type="solid", spacegroup_number=229, enthalpy=99999.0, entropy=0.0)
        )
        _, phase_refs = sd._resolve_refs_db(["Fe"], {"Fe": ref})
        assert phase_refs["BCC"]["Fe"]["delta_h_jmol"] == 0.0

    def test_empty_block_changes_nothing(self):
        ref = ComponentRef(
            "Xx",
            [
                Phase(
                    phase_type="solid",
                    t_transition=0,
                    delta_h=0.0,
                    enthalpy=0.0,
                    entropy=0.0,
                    spacegroup_number=229,
                ),
                Phase(
                    phase_type="liquid",
                    t_transition=1000.0,
                    delta_h=10000.0,
                    enthalpy=10000.0,
                    entropy=10.0,
                ),
            ],
        )
        _, phase_refs = sd._resolve_refs_db(["Xx"], {"Xx": ref})
        assert set(phase_refs) == {"BCC"}


# --------------------------------------------------------------------------------------
# Scoped metastable-entropy exception (04-metastable-entropy-exception)
# --------------------------------------------------------------------------------------
# The builder may emit a NEGATIVE delta_S on a lattice stability for a metastable
# structure, where dS = dH / T_trans is undefined. Both the loader and the resolver
# must carry that value through instead of re-flattening it to 0, and it must stay out
# of the ladder machinery exactly as a zero-entropy entry does. Synthetic throughout:
# which elements are in the exception is the DATABASE's business, not the package's.


def _synth_with_entropy(hcp_entropy):
    """SYNTH with the HCP lattice stability's delta_S set (or the key removed if None)."""
    data = json.loads(json.dumps(SYNTH))
    hcp = data["elements"]["Fe"]["lattice_stabilities"]["HCP"]
    if hcp_entropy is None:
        del hcp["delta_S_J_per_mol_K"]
    else:
        hcp["delta_S_J_per_mol_K"] = hcp_entropy
    return data


@pytest.fixture()
def registry_factory(tmp_path, monkeypatch):
    def build(data):
        path = tmp_path / "phase_transitions.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr(config, "phase_transitions_file", path)
        return UnaryData(require=True)

    return build


class TestMetastableEntropyException:
    def test_loader_carries_a_negative_entropy_through(self, registry_factory):
        ref = registry_factory(_synth_with_entropy(-1.6))["Fe"]
        hcp = next(p for p in ref.lattice_stabilities if p.spacegroup_number == 194)
        assert hcp.entropy == -1.6
        # The enthalpy is NOT touched by the exception.
        assert hcp.enthalpy == 9436.84

    @pytest.mark.parametrize("stored", [None, 0.0])
    def test_absent_or_zero_entropy_stays_zero(self, registry_factory, stored):
        """An older database file (or one predating the field) must not move."""
        ref = registry_factory(_synth_with_entropy(stored))["Fe"]
        hcp = next(p for p in ref.lattice_stabilities if p.spacegroup_number == 194)
        assert hcp.entropy == 0.0

    def test_resolver_publishes_the_negative_entropy(self, registry_factory):
        reg = registry_factory(_synth_with_entropy(-1.6))
        _, phase_refs = sd._resolve_refs_db(["Fe"], {"Fe": reg["Fe"].copy()})
        assert phase_refs["HCP"]["Fe"]["delta_s_jmol_k"] == -1.6
        assert phase_refs["HCP"]["Fe"]["delta_h_jmol"] == 9436.84
        # Entries outside the exception are unmoved.
        assert phase_refs["FCC"]["Fe"]["delta_s_jmol_k"] == 0.0
        assert phase_refs["BCC"]["Fe"]["delta_s_jmol_k"] == 0.0

    def test_negative_entropy_still_never_reaches_ladder_machinery(self, registry_factory):
        """No transition temperature => still out of polymorphs, liquid reconstruction
        and _ordered_solid_steps, so the liquid reference cannot move."""
        reg = registry_factory(_synth_with_entropy(-1.6))
        ref = reg["Fe"].copy()
        assert {p.spacegroup_number for p in ref.polymorphs} == {229}
        assert ref.solid_phase(194) is None
        _, phase_refs = sd._resolve_refs_db(["Fe"], {"Fe": ref})
        assert sd._ordered_solid_steps(ref, "Fe", phase_refs) == []
        assert ref.liquid_ref_from_solids() == (13810.0, pytest.approx(13810.0 / 1811.0))

    def test_negative_entropy_cannot_overtake_the_ground_state(self, registry_factory):
        """The safety argument, checked rather than assumed.

        The endpoint melting reconstruction is max((h_liq - dH) / (s_liq - dS)) over the
        candidate solids. A NEGATIVE dS only enlarges the denominator, so the metastable
        reference's crossing sits BELOW the ground state's and the element still melts at
        its own t_fusion. A positive dS would not be safe — hence the builder's sign guard.
        """
        reg = registry_factory(_synth_with_entropy(-1.6))
        ref = reg["Fe"].copy()
        _, phase_refs = sd._resolve_refs_db(["Fe"], {"Fe": ref})
        h_liq, s_liq = ref.h_liq, ref.s_liq
        crossings = {
            p: (h_liq - r["Fe"]["delta_h_jmol"]) / (s_liq - r["Fe"]["delta_s_jmol_k"])
            for p, r in phase_refs.items()
        }
        # abs=0.01 K: SYNTH's s_liq is the 4-dp rounded 7.6256, not 13810/1811 exactly.
        assert crossings["BCC"] == pytest.approx(ref.t_fusion, abs=0.01)  # the ground state
        assert max(crossings.values()) == pytest.approx(ref.t_fusion, abs=0.01)
        assert crossings["HCP"] < crossings["BCC"]


# --------------------------------------------------------------------------------------
# Virtual ladder phases (05-virtual-polymorphs)
# --------------------------------------------------------------------------------------
# A real equilibrium polymorph that Materials Project has no entry for is injected onto
# the ladder by the builder with a synthetic ``materials_project_id`` ("virtual-Y-beta").
# From the package's point of view it must behave as an ORDINARY ladder phase: resolved by
# spacegroup, preferred over any lattice stability of the same structure, kept in the
# crossing pool as a line compound when no SS model covers it, and never flagged
# ``imputed`` (that flag means "omegas-file fallback", not "not from MP"). Synthetic
# throughout: WHICH elements carry a virtual phase is the DATABASE's business.


def _synth_with_virtual_bcc():
    """A hcp-ground-state element with a virtual bcc successor below melting.

    Modelled on Y: alpha (hcp, H = 0) -> beta (bcc, 4886.73 J/mol at 1751 K) -> liquid
    (16286.73 J/mol at 1795 K), plus an fcc lattice stability the ladder does not cover.
    """
    return {
        "elements": {
            "Y": {
                "symbol": "Y",
                "phases": [
                    {
                        "phase_type": "solid",
                        "common_name": "alpha-Y (hcp)",
                        "spacegroup_number": 194,
                        "spacegroup_symbol": "P6_3/mmc",
                        "transition_temperature_K": 0,
                        "enthalpy_J_per_mol": 0.0,
                        "entropy_J_per_mol_K": 0.0,
                        "delta_H_J_per_mol": 0.0,
                        "delta_S_J_per_mol_K": 0.0,
                        "materials_project_id": "mp-112",
                    },
                    {
                        "phase_type": "solid",
                        "common_name": "beta-Y (bcc)",
                        "spacegroup_number": 229,
                        "spacegroup_symbol": "Im-3m",
                        "transition_temperature_K": 1751,
                        "enthalpy_J_per_mol": 4886.73,
                        "entropy_J_per_mol_K": 2.7908,
                        "delta_H_J_per_mol": 4886.73,
                        "delta_S_J_per_mol_K": 2.7908,
                        "materials_project_id": "virtual-Y-beta",
                    },
                    {
                        "phase_type": "liquid",
                        "common_name": "Liquid Y",
                        "transition_temperature_K": 1795.0,
                        "enthalpy_J_per_mol": 16286.73,
                        "entropy_J_per_mol_K": 9.1418,
                        "delta_H_J_per_mol": 11400.0,
                        "delta_S_J_per_mol_K": 6.351,
                    },
                ],
                "lattice_stabilities": {
                    "FCC": {
                        "spacegroup_number": 225,
                        "spacegroup_symbol": "Fm-3m",
                        "delta_H_J_per_mol": 9438.0,
                        "delta_S_J_per_mol_K": 0.0,
                        "transition_temperature_K": None,
                        "materials_project_id": "mp-9",
                        "metadata": {
                            "source": "DFT (Materials Project)",
                            "basis": "element anchor",
                        },
                    },
                },
            },
        },
    }


class TestVirtualLadderPhase:
    def test_virtual_id_is_a_polymorph_not_an_imputed_reference(self, registry_factory):
        ref = registry_factory(_synth_with_virtual_bcc())["Y"]
        beta = ref.solid_phase(229)
        assert beta is not None and beta.material_id == "virtual-Y-beta"
        assert beta.imputed is False  # 'imputed' means omegas fallback, not 'not MP'
        assert {p.spacegroup_number for p in ref.polymorphs} == {194, 229}

    def test_resolver_publishes_the_virtual_phase_with_its_real_entropy(self, registry_factory):
        reg = registry_factory(_synth_with_virtual_bcc())
        _, phase_refs = sd._resolve_refs_db(["Y"], {"Y": reg["Y"].copy()})
        assert phase_refs["BCC"]["Y"]["delta_h_jmol"] == 4886.73
        # NOT 0.0: a ladder phase's entropy comes from dH / T_trans, unlike a lattice stability.
        assert phase_refs["BCC"]["Y"]["delta_s_jmol_k"] == 2.7908
        # 'source' marks NON-primary provenance, so a ladder-resolved phase carries none;
        # the FCC lattice stability does.
        assert "source" not in phase_refs["BCC"]["Y"]
        assert "lattice_stability" in phase_refs["FCC"]["Y"]["source"]
        assert phase_refs["FCC"]["Y"]["delta_s_jmol_k"] == 0.0

    def test_liquid_reference_is_reproduced_from_the_extended_ladder(self, registry_factory):
        """The injected step must be part of H_liq/S_liq, not an unaccounted extra."""
        ref = registry_factory(_synth_with_virtual_bcc())["Y"]
        h, s = ref.liquid_ref_from_solids()
        assert h == pytest.approx(16286.73, abs=1e-6)
        assert s == pytest.approx(4886.73 / 1751 + 11400.0 / 1795.0, abs=1e-4)

    def test_endpoint_melting_survives_every_loaded_subset(self, registry_factory):
        """The load-bearing property: inserting a ladder step must not move the endpoint.

        Whatever subset of {BCC, FCC, HCP} a partner leaves loaded, the element still melts
        at its own t_fusion, because any polymorph NOT covered by a loaded SS phase stays in
        the crossing pool as a line compound. With HCP alone loaded it is the injected bcc
        phase that supplies the crossing -- which is exactly the case that would break if a
        virtual phase were treated as a metastable reference instead of a ladder step.
        """
        import itertools

        reg = registry_factory(_synth_with_virtual_bcc())
        ref = reg["Y"].copy()
        _, phase_refs = sd._resolve_refs_db(["Y"], {"Y": ref})
        h_liq, s_liq = ref.h_liq, ref.s_liq
        names = sorted(phase_refs)
        for r in range(len(names) + 1):
            for subset in itertools.combinations(names, r):
                kept = {p: phase_refs[p]["Y"] for p in subset}
                covered = {sd.SS_SPACEGROUPS[p] for p in kept}
                solids = [
                    (p.enthalpy or 0.0, p.entropy or 0.0)
                    for p in ref.polymorphs
                    if p.spacegroup_number not in covered
                ]
                solids += [(v["delta_h_jmol"], v["delta_s_jmol_k"]) for v in kept.values()]
                crossings = [(h_liq - h) / (s_liq - s) for h, s in solids if s_liq - s > 1e-9]
                assert max(crossings) == pytest.approx(ref.t_fusion, abs=0.02), subset
