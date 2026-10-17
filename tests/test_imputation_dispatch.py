"""The dimension-generic imputation API: dispatch, the record union and the ladder.

Checked:
  * ``impute_phases``, ``drop_imputed_phases``, ``imputed_entry_staleness`` and
    ``stale_imputed_entries`` in ``gliquid.imputation.lifecycle`` call the binary
    implementation for a ``BinaryLiquid`` and the ternary one for a
    ``TernaryLiquidInterpolation``, and raise ``TypeError`` for any other model;
  * ``impute_phases`` raises ``TypeError`` for a binary system given ``targets`` and for a
    ternary system without them;
  * ``stale_imputed_entries`` judges the imputed entries of a ternary model;
  * both record types are instances of the ``ImputationResult`` union, and ``dataclasses.asdict``
    keeps their field order;
  * ``phase.polymorph_ladder`` keeps the registry order, applies the exclusion and returns copies.

Binary: shipped Cu-Mg at the liquid parameters test_imputation_bit_identity.py pins. Ternary:
Al-Mg-Si in hsx mode on the shipped entries record, with ``config.persist_ternary_imputed`` off.
"""

from __future__ import annotations

import dataclasses

import pytest

import gliquid.imputation.lifecycle as imp_lifecycle
from gliquid.binary import BinaryLiquid
from gliquid.imputation.binary_imputation import BinaryImputationResult
from gliquid.imputation.ternary_imputation import TernaryImputationResult
from gliquid.phase import UNARY, polymorph_ladder
from gliquid.ternary import TernaryLiquidInterpolation

# The package names under test, one line each.
IMPUTE = imp_lifecycle.impute_phases
DROP = imp_lifecycle.drop_imputed_phases
STALENESS = imp_lifecycle.imputed_entry_staleness
STALE_ENTRIES = imp_lifecycle.stale_imputed_entries
IMPUTATION_RESULT = imp_lifecycle.ImputationResult
LADDER = polymorph_ladder

CU_MG_PARAMS = [-41302.26, -5.1627, -8148.05, 0.0]
AMS = ["Al", "Mg", "Si"]
AMS_MIXING = {
    "Al-Mg": [-20000.0, 0, 0, 0],
    "Mg-Si": [-30000.0, 0, 0, 0],
    "Si-Al": [-10000.0, 0, 0, 0],
}
AMS_TARGET = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
MOVED_REASON = "Mg-Si parameters have moved"

# (dispatcher, binary implementation, ternary implementation), as lifecycle names.
ROUTES = {
    "impute_phases": ("impute_binary_phases", "impute_ternary_phases"),
    "drop_imputed_phases": ("_drop_binary_imputed_phases", "_drop_ternary_imputed_phases"),
    "imputed_entry_staleness": (
        "_binary_imputed_entry_staleness",
        "_ternary_imputed_entry_staleness",
    ),
}

IMPUTED_ID = "imputed:Probe"

LADDER_ELEMENT = "Mn"
LADDER_SPACEGROUPS = [217, 213, 225, 229]


@pytest.fixture(scope="module")
def cu_mg() -> BinaryLiquid:
    return BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp", params=list(CU_MG_PARAMS))


def build(xs_mix=AMS_MIXING) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(AMS),
        delta=0.1,
        xs_mix=xs_mix,
        order="given",
        hull_mode="hsx",
        T_incr=5,
        param_format="linear",
    )


@pytest.fixture(scope="module")
def ams() -> TernaryLiquidInterpolation:
    return build()


def _spy(monkeypatch, name):
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return name

    monkeypatch.setattr(imp_lifecycle, name, spy)
    return calls


def _dispatch(dispatcher, model):
    if dispatcher == "impute_phases":
        targets = None if isinstance(model, BinaryLiquid) else [{"composition": "X"}]
        return IMPUTE(model, targets, flag=1)
    if dispatcher == "drop_imputed_phases":
        return DROP(model, flag=1)
    return STALENESS(model, {"entry_id": IMPUTED_ID})


# -- Dispatch --


@pytest.mark.parametrize("dispatcher", list(ROUTES))
def test_a_binary_system_goes_to_the_binary_implementation(monkeypatch, cu_mg, dispatcher):
    binary, ternary = ROUTES[dispatcher]
    binary_calls, ternary_calls = _spy(monkeypatch, binary), _spy(monkeypatch, ternary)
    assert _dispatch(dispatcher, cu_mg) == binary
    assert ternary_calls == []
    (args, kwargs), *rest = binary_calls
    assert rest == [] and args[0] is cu_mg
    if dispatcher != "imputed_entry_staleness":
        assert kwargs == {"flag": 1}


@pytest.mark.parametrize("dispatcher", list(ROUTES))
def test_a_ternary_system_goes_to_the_ternary_implementation(monkeypatch, ams, dispatcher):
    binary, ternary = ROUTES[dispatcher]
    binary_calls, ternary_calls = _spy(monkeypatch, binary), _spy(monkeypatch, ternary)
    assert _dispatch(dispatcher, ams) == ternary
    assert binary_calls == []
    (args, kwargs), *rest = ternary_calls
    assert rest == [] and args[0] is ams
    if dispatcher == "impute_phases":
        assert args[1] == [{"composition": "X"}] and kwargs == {"flag": 1}


@pytest.mark.parametrize(
    "call",
    [
        lambda model: IMPUTE(model),
        lambda model: DROP(model),
        lambda model: STALENESS(model, {"entry_id": IMPUTED_ID}),
        lambda model: STALE_ENTRIES(model, [{"entry_id": IMPUTED_ID}]),
    ],
    ids=["impute_phases", "drop_imputed_phases", "imputed_entry_staleness", "stale"],
)
def test_another_model_type_raises(call):
    with pytest.raises(TypeError, match="BinaryLiquid or a TernaryLiquidInterpolation"):
        call({"not": "a model"})


def test_a_binary_system_takes_no_targets(monkeypatch, cu_mg):
    calls = _spy(monkeypatch, "impute_binary_phases")
    with pytest.raises(TypeError, match="fitting_data="):
        IMPUTE(cu_mg, [{"fit": 1}])
    assert calls == []


def test_a_ternary_system_needs_targets(monkeypatch, ams):
    calls = _spy(monkeypatch, "impute_ternary_phases")
    with pytest.raises(TypeError, match="targets"):
        IMPUTE(ams)
    assert calls == []


def test_stale_imputed_entries_judges_a_ternary_model(no_persist):
    ti = build()
    record = ti.impute_phase(AMS_TARGET, 1200.0, name="TX", apply=False, write_cache=False)
    assert record.accepted, record.reason
    real = {"entry_id": "mp-1234-GGA", "data": {}}
    assert STALE_ENTRIES(ti, [real, record.entry_dict]) == []
    moved = build({**AMS_MIXING, "Mg-Si": [-31000.0, 0, 0, 0]})
    reported = STALE_ENTRIES(moved, [real, record.entry_dict])
    assert [(entry["entry_id"], reason) for entry, reason in reported] == [
        (record.entry_dict["entry_id"], MOVED_REASON)
    ]


# -- Records --


def _records():
    binary = BinaryImputationResult(
        name="Probe",
        comp=0.5,
        enthalpy=-1000.0,
        invariant_type="per",
        invariant_comp=0.4,
        invariant_temp=900.0,
        entry_dict={"entry_id": IMPUTED_ID, "data": {"label": "Probe"}},
    )
    ternary = TernaryImputationResult(
        name="Probe", fractions=dict(AMS_TARGET), temperature_k=1000.0, method="envelope"
    )
    return binary, ternary


def test_both_record_types_are_imputation_results():
    for record in _records():
        assert isinstance(record, IMPUTATION_RESULT)
    assert not isinstance({"name": "Probe"}, IMPUTATION_RESULT)


def test_asdict_keeps_the_field_order():
    for record in _records():
        fields = [f.name for f in dataclasses.fields(record)]
        assert list(dataclasses.asdict(record)) == fields


# -- Ladder --


def test_the_ladder_keeps_the_registry_order():
    registry = UNARY[LADDER_ELEMENT]
    ladder = LADDER(registry)
    assert [p.spacegroup_number for p in ladder] == LADDER_SPACEGROUPS
    assert [p.name for p in ladder] == [p.name for p in registry.polymorphs]
    assert LADDER(registry, set()) == ladder


def test_the_ladder_applies_the_exclusion():
    ladder = LADDER(UNARY[LADDER_ELEMENT], {213})
    assert [p.spacegroup_number for p in ladder] == [217, 225, 229]


def test_the_ladder_returns_copies():
    registry = UNARY[LADDER_ELEMENT]
    before = [list(p.points) for p in registry.polymorphs]
    ladder = LADDER(registry)
    for copied, original in zip(ladder, registry.polymorphs):
        assert copied is not original and copied.points == []
    ladder[0].points.append((0.0, 1.0))
    assert [list(p.points) for p in registry.polymorphs] == before
