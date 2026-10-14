"""Bit-identity pins for binary phase-energy imputation.

Frozen by freeze_imputation_bitid_pins.py into fixtures/imputation_bitid_pins.json, which
calls :func:`compute_pins` below. Every float is compared exactly: JSON stores floats by their
round-trip ``repr``. The liquid parameters are pinned, so no fit is involved, and everything
runs offline against the shipped Cu-Mg and Al-Cu data fixtures.

Blocks per system:
  * ``liquid_gibbs``: ``_liquid_gibbs`` on a composition x temperature grid.
  * ``synthetic``: ``invert_binary_phase_energy`` on hand-built anchors, including anchors within
    ``dx`` of either edge, under each ``t_matched_types`` setting.
  * ``targets``: every imputable phase of the interior-stripped system -- both inversion arms,
    the hull guard, the tagged entry and an entry re-energized by ``entry_at_enthalpy``.
  * ``impute_shipped`` / ``impute_nearest``: full ``impute_binary_phases`` output and the finished
    phase list.
  * ``clamp``: a deliberately over-deep probe through ``_clamp_onto_matched_hull`` and
    ``accept_against_growing_hull`` on the full hull.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.core as imp_core
from gliquid.binary import BinaryLiquid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mpds_builders import (
    matches_pin,  # noqa: E402
    strip_interior,  # noqa: E402
)
from _mpds_builders import without_module_paths as _without_module_paths  # noqa: E402

pytestmark = pytest.mark.pins

PINS_PATH = Path(__file__).parent / "fixtures" / "imputation_bitid_pins.json"

PARAMS = {
    "Cu-Mg": [-41302.26, -5.1627, -8148.05, 0.0],
    "Al-Cu": [-68087.2, -3.6819, 2945.42, 0.0],
}
GIBBS_X = (1e-6, 0.013, 0.25, 0.5, 0.731, 1.0 - 1e-6)
GIBBS_T = (700.0, 1100.0, 1500.0)
SYNTHETIC_ANCHORS = (
    ({"type": "per", "comp": 0.0004, "temp": 950.0}, 0.2),
    ({"type": "eut", "comp": 0.9997, "temp": 800.0}, 0.8),
    ({"type": "cmp", "comp": 0.4, "temp": 1000.0}, 0.4),
    ({"type": "per", "comp": 0.55, "temp": 1050.0}, 0.4),
    ({"type": "eut", "comp": 0.35, "temp": 900.0}, 0.62),
)
ARMS = ((), ("per",), ("per", "eut"))
CLAMP_PROBES = (0.3, 0.55)
CLAMP_DEPTH_J = 20000.0
VERSION_MASK = "<masked>"


def load(system: str) -> BinaryLiquid:
    """The shipped system at its pinned liquid parameters."""
    return BinaryLiquid.from_cache(system, param_format="comb-exp", params=PARAMS[system])


def masked(entry_dict: dict | None) -> dict | None:
    """``entry_dict`` with the installed package version masked out of its provenance."""
    if entry_dict is None:
        return None
    out = copy.deepcopy(entry_dict)
    provenance = (out.get("data") or {}).get(imp_binary.PROVENANCE_KEY)
    if isinstance(provenance, dict) and "gliquid_version" in provenance:
        provenance["gliquid_version"] = VERSION_MASK
    return out


def entry_record(entry_dict: dict | None) -> dict | None:
    """A masked entry plus the key order of its ``data`` block."""
    if entry_dict is None:
        return None
    out = masked(entry_dict)
    return {"entry": out, "data_keys": list((out.get("data") or {}).keys())}


def imputed_records(records) -> list[dict]:
    return [
        {
            "name": r.name,
            "comp": float(r.comp),
            "enthalpy": float(r.enthalpy),
            "invariant_type": r.invariant_type,
            "invariant_comp": float(r.invariant_comp),
            "invariant_temp": float(r.invariant_temp),
            "entry": entry_record(r.entry_dict),
        }
        for r in records
    ]


def phase_list(bl: BinaryLiquid) -> list[list]:
    return sorted(
        [p.name, None if p.enthalpy is None else float(p.enthalpy), bool(p.imputed)]
        for p in bl.phases
    )


def compute_system(system: str) -> dict:
    """Every pinned quantity for one system."""
    bl = load(system)
    block: dict = {"params": list(PARAMS[system])}

    block["liquid_gibbs"] = [
        [float(imp_binary._liquid_gibbs(bl, x, t)) for x in GIBBS_X] for t in GIBBS_T
    ]
    block["synthetic"] = [
        [
            float(imp_binary.invert_binary_phase_energy(bl, x_s, inv, t_matched_types=arm))
            for inv, x_s in SYNTHETIC_ANCHORS
        ]
        for arm in ARMS
    ]

    stripped = copy.deepcopy(bl)
    strip_interior(stripped)
    targets = []
    for phase, inv, x_s in imp_binary.list_binary_imputation_targets(stripped):
        h_shipped = float(imp_binary.invert_binary_phase_energy(stripped, x_s, inv))
        h_tmatched = float(
            imp_binary.invert_binary_phase_energy(
                stripped, x_s, inv, t_matched_types=("per", "eut")
            )
        )
        entry = imp_binary._imputed_computed_entry(stripped, phase["name"], x_s, h_shipped, inv)
        targets.append(
            {
                "name": phase["name"],
                "x_s": float(x_s),
                "invariant": {k: inv[k] for k in ("type", "comp", "temp")},
                "h_shipped": h_shipped,
                "h_tmatched": h_tmatched,
                "hull_at": float(imp_binary._phase_list_enthalpy_at(stripped, x_s)),
                "below": bool(imp_binary._below_existing_hull(stripped, x_s, h_shipped)),
                "entry": entry_record(entry),
                "reenergized": entry_record(
                    imp_core.entry_at_enthalpy(stripped.dft_ch, entry, h_shipped + 500.0)
                ),
            }
        )
    block["targets"] = targets

    for selection in imp_binary.INVERSION_SELECTIONS:
        work = copy.deepcopy(stripped)
        records = imp_binary.impute_binary_phases(
            work, write_cache=False, inversion_selection=selection
        )
        block[f"impute_{selection}"] = {
            "records": imputed_records(records),
            "phases": phase_list(work),
        }

    protected = imp_binary.matched_hull_entry_ids(bl)
    clamp = []
    for x_s in CLAMP_PROBES:
        name = f"Probe{x_s}"
        h_deep = float(imp_binary._phase_list_enthalpy_at(bl, x_s)) - CLAMP_DEPTH_J
        deep = imp_binary._imputed_computed_entry(bl, name, x_s, h_deep)
        clamped = imp_binary._clamp_onto_matched_hull(bl, deep, [], protected)
        accepted, rejected = imp_binary.accept_against_growing_hull(
            bl, [{"entry_dict": deep, "name": name, "margin_j": CLAMP_DEPTH_J}]
        )
        clamp.append(
            {
                "x_s": x_s,
                "h_deep": h_deep,
                "clamped": entry_record(clamped),
                "accepted": [entry_record(e) for e in accepted],
                "rejected": [list(r) for r in rejected],
            }
        )
    block["clamp"] = clamp
    return block


def compute_pins() -> dict:
    """Every pinned quantity, keyed by system, in the layout the fixture stores."""
    return {system: compute_system(system) for system in PARAMS}


def _normalized(value):
    """``value`` as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(PINS_PATH.read_text())


@pytest.fixture(scope="module", params=list(PARAMS))
def current(request) -> tuple[str, dict]:
    return request.param, _normalized(compute_system(request.param))


BLOCKS = (
    "params",
    "liquid_gibbs",
    "synthetic",
    "targets",
    "impute_shipped",
    "impute_nearest",
    "clamp",
)


class TestBinaryImputationBitIdentity:
    @pytest.mark.parametrize("block", BLOCKS)
    def test_block_matches_frozen(self, frozen, current, block):
        system, values = current
        actual, pinned = (
            _without_module_paths(values[block]),
            _without_module_paths(frozen[system][block]),
        )
        assert matches_pin(actual, pinned), (actual, pinned)

    def test_fixture_exercises_every_path(self, frozen):
        for system, block in frozen.items():
            assert block["targets"], f"{system}: no imputable phase after stripping"
            assert block["impute_shipped"]["records"], f"{system}: nothing imputed"
        clamps = [c for block in frozen.values() for c in block["clamp"]]
        assert any(c["clamped"] is not None for c in clamps), "no probe reached the clamp"
