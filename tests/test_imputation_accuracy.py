"""Accuracy pins for phase-energy imputation, against the DFT entries it stands in for.

Each subject holds one interior compound out of its own DFT hull, imputes it back from the
experimental invariant, and compares the result with the entry that was removed. Liquid
parameters are pinned rather than fitted, so a change in the optimizer cannot move an
accuracy pin and an accuracy regression cannot hide behind a re-fit.

This module gates the shipped subjects only and carries no claim about the method's
accuracy at large.

Everything runs offline against the shipped Cu-Mg / Al-Cu / Pb-Rb / Bi-La data fixtures.
"""

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram

import gliquid.imputation.binary_imputation as imp_binary
from gliquid.binary import BinaryLiquid, build_phases_from_chull
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.solution import x_vals

pytestmark = pytest.mark.pins

PARAM_FORMAT = "comb-exp"
STRIP_TOL = 1e-4  # a held-out compound takes every entry at its composition, not just the vertex
MATCH_TOL = 0.02  # composition window pairing an imputed phase with the entry it replaces

ANCHOR_TOLERANCE_J = {"cmp": 100.0, "per": 5000.0, "eut": 5000.0}
"""Ceiling on |imputed - DFT| in J/mol-atom, by the invariant the inversion is anchored at.

A congruent anchor reads the liquid's Gibbs energy at the compound's own composition. The
other two extrapolate the liquid's tangent from the invariant composition to the compound,
and are allowed the wider band that reach costs.
"""

SUBJECTS = {
    "Cu-Mg": {
        "params": [-41302.2616880771, -5.162709622017944, -8148.050427961623, 0.0],
        "scored": [
            {
                "held": "MgCu2",
                "x": 0.333333,
                "imputed_as": "MgCu2",
                "anchor": "cmp",
                "dft_h_form_j": -14061.2574,
                "imputed_h_form_j": -14103.3511,
                "clamped": False,
            },
            {
                "held": "Mg2Cu",
                "x": 0.666667,
                "imputed_as": "Mg2Cu",
                "anchor": "cmp",
                "dft_h_form_j": -10819.6598,
                "imputed_h_form_j": -10817.8832,
                "clamped": False,
            },
        ],
        "unreachable": [],
    },
    "Al-Cu": {
        "params": [-68087.20479694149, -3.6819269965385653, 2945.4157631308426, 0.0],
        "scored": [
            {
                "held": "Al2Cu",
                "x": 0.333333,
                "imputed_as": "CuAl2",
                "anchor": "per",
                "dft_h_form_j": -17180.7202,
                "imputed_h_form_j": -16430.6884,
                "clamped": False,
            },
        ],
        # The MPDS phase whose field covers each of these peaks further than MATCH_TOL from
        # the DFT stoichiometry, so no imputed phase pairs with the entry that was removed.
        "unreachable": [
            {"held": "AlCu", "x": 0.5},
            {"held": "Al4Cu9", "x": 0.692308},
            {"held": "AlCu3", "x": 0.75},
        ],
    },
    "Pb-Rb": {
        "params": [-53513.451772942535, -6.688814675517017, -40559.502187713966, 0.0],
        "scored": [
            {
                "held": "RbPb",
                "x": 0.5,
                "imputed_as": "RbPb",
                "anchor": "cmp",
                "dft_h_form_j": -21395.3811,
                "imputed_h_form_j": -21396.1115,
                "clamped": False,
            },
        ],
        "unreachable": [{"held": "Rb4Pb9", "x": 0.307692}],
    },
    "Bi-La": {
        "params": [-287901.79447405384, -35.98772430925674, -71831.34895841485, 0.0],
        "scored": [
            {
                "held": "LaBi",
                "x": 0.5,
                "imputed_as": "LaBi",
                "anchor": "per",
                "dft_h_form_j": -101059.8399,
                "imputed_h_form_j": -98168.7896,
                "clamped": False,
            },
            {
                "held": "La2Bi",
                "x": 0.666667,
                "imputed_as": "La2Bi",
                "anchor": "per",
                "dft_h_form_j": -73361.9681,
                "imputed_h_form_j": -71799.6272,
                "clamped": True,
            },
        ],
        "unreachable": [{"held": "LaBi3", "x": 0.25}],
    },
}

_SCORED = [(system, row) for system, blk in SUBJECTS.items() for row in blk["scored"]]
_UNREACHABLE = [(system, row) for system, blk in SUBJECTS.items() for row in blk["unreachable"]]


def _ids(pairs):
    return [f"{system}-{row['held']}" for system, row in pairs]


def _x_of(bl, entry) -> float:
    """An entry's fraction along the system's own composition axis."""
    return float(entry.composition.get_atomic_fraction(bl.components[1]))


def _strip(bl, held_x: float) -> None:
    """Remove every entry at ``held_x`` from the hull and the phase list.

    Dropping the hull VERTEX alone does not make a compound missing: a metastable entry at
    the same composition takes its place and the hull is unchanged.
    """
    keep = [
        entry
        for entry in bl.dft_ch.all_entries
        if len(entry.composition.elements) == 1 or abs(_x_of(bl, entry) - held_x) > STRIP_TOL
    ]
    bl.dft_ch = PhaseDiagram(keep, elements=list(bl.dft_ch.elements))
    bl.phases = build_phases_from_chull(bl.dft_ch, bl.components, bl.component_data)
    points = imp_binary._hull_points(bl)
    bl.eqs["h_hull_interp"] = np.interp(x_vals[1:-1], points[:, 0], points[:, 1])
    bl.invariants = None
    bl.low_t_exp_phases = None
    bl.hsx = None


def _load(system: str):
    bl = BinaryLiquid.from_cache(
        system.split("-"),
        params=list(SUBJECTS[system]["params"]),
        param_format=PARAM_FORMAT,
        pd_ind=0,
    )
    bl.update_phase_points()
    return bl


def _holdout(system: str, held_x: float):
    """Impute the compound at ``held_x`` back onto a hull it has been removed from."""
    bl = _load(system)
    _strip(bl, held_x)
    imputed = imp_binary.impute_binary_phases(bl, fitting_data=None, write_cache=False)
    return next((r for r in imputed if abs(float(r.comp) - held_x) <= MATCH_TOL), None)


def _clamp_of(record) -> dict | None:
    provenance = (record.entry_dict.get("data") or {}).get(imp_binary.PROVENANCE_KEY) or {}
    return provenance.get(imp_binary.CLAMP_KEY)


@pytest.fixture(scope="module")
def imputed():
    """Every pinned hold-out, run once."""
    return {
        (system, row["held"]): _holdout(system, row["x"]) for system, row in _SCORED + _UNREACHABLE
    }


@pytest.fixture(scope="module")
def dft_truth():
    """Each subject's DFT formation enthalpies in J/mol-atom, keyed by (system, phase)."""
    out = {}
    for system in SUBJECTS:
        bl = _load(system)
        for entry in bl.dft_ch.stable_entries:
            if len(entry.composition.elements) > 1:
                out[(system, entry.name)] = (
                    float(bl.dft_ch.get_form_energy_per_atom(entry)) * EV_ATOM_TO_J_MOL
                )
    return out


@pytest.mark.parametrize("system,row", _SCORED, ids=_ids(_SCORED))
class TestHeldOutCompoundIsRecovered:
    def test_the_dft_entry_it_is_scored_against_is_the_pinned_one(self, system, row, dft_truth):
        """A moved DFT fixture would rewrite the error without touching the imputation."""
        assert dft_truth[(system, row["held"])] == pytest.approx(row["dft_h_form_j"], rel=1e-6)

    def test_it_comes_back_under_the_pinned_name_and_anchor(self, system, row, imputed):
        record = imputed[(system, row["held"])]
        assert record is not None, f"{system}: {row['held']} was not imputed back"
        assert record.name == row["imputed_as"]
        assert record.invariant_type == row["anchor"]

    def test_the_error_against_dft_is_within_the_anchor_tolerance(
        self, system, row, imputed, dft_truth
    ):
        record = imputed[(system, row["held"])]
        got = float(record.entry_dict["data"]["h_form_j_per_mol"])
        error = got - dft_truth[(system, row["held"])]
        ceiling = ANCHOR_TOLERANCE_J[row["anchor"]]
        assert abs(error) <= ceiling, (
            f"{system} {row['held']} ({row['anchor']}): imputed enthalpy is off by "
            f"{error:.1f} J/mol-atom, past the {ceiling:.0f} J ceiling"
        )

    def test_the_imputed_enthalpy_matches_the_pin(self, system, row, imputed):
        """The tripwire: the tolerance above is wide enough to absorb a real regression."""
        record = imputed[(system, row["held"])]
        got = float(record.entry_dict["data"]["h_form_j_per_mol"])
        assert got == pytest.approx(row["imputed_h_form_j"], rel=1e-6)

    def test_the_clamp_state_matches_the_pin(self, system, row, imputed):
        """A phase that starts or stops being clamped changes which energy it carries."""
        assert (_clamp_of(imputed[(system, row["held"])]) is not None) is row["clamped"]


@pytest.mark.parametrize("system,row", _UNREACHABLE, ids=_ids(_UNREACHABLE))
def test_an_unreachable_compound_stays_unreachable(system, row, imputed):
    """Reach is a property of the method, not an accident, so it is pinned in both directions."""
    assert imputed[(system, row["held"])] is None, (
        f"{system}: {row['held']} is now imputable — it was not, so either reach improved "
        f"or the pairing window moved. Re-pin it as a scored subject if the former."
    )
