"""How each builder of ``eqs['h_hull_interp']`` relates to the sorted rebuild.

The reference is :func:`_reference_interp`: the ``(x, enthalpy)`` points of the
fixed-composition phases, sorted by x and interpolated on the interior composition grid, as
the ablation rebuild computes them on a deep copy. Every assertion is bit equality
(``np.array_equal``) or its negation.

Builders and the states they are checked in:
  * ``BinaryLiquid.from_cache``: Cu-Mg, Cu-Mg loaded as ``'Mg-Cu'``, Hf-Zr with solid
    solutions off and on.
  * ``BinaryLiquid._rebuild_thermodynamic_expressions``, reached through the pickle round trip
    (``__setstate__``) and through ``with_component_order``. A ``with_component_order`` clone
    keeps its source's phase order, so on a system whose clone phase list is not in x order
    the stored value differs from the reference.
  * The ablation rebuild (``ablate_phases`` / ``restore_phases``), which also resets the fit
    state listed in ``RESET_FIELDS``.
  * ``apply_imputed_entries``.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.imputation.ablation as imp_ablation
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.lifecycle as imp_lifecycle
from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.solution import x_vals

from ._mpds_builders import strip_interior

# The package names under test, one line each.
REBUILD_HULL_INTERP = BinaryLiquid.rebuild_hull_interp
ABLATE = imp_ablation.ablate_phases
RESTORE = imp_ablation.restore_phases
LIST_ABLATABLE = imp_ablation.list_ablatable_phases
APPLY_IMPUTED = imp_binary.apply_imputed_entries
IMPUTE = imp_binary.impute_binary_phases
DROP = imp_lifecycle.drop_imputed_phases
HULL_POINTS = imp_binary._hull_points

# The liquid parameters test_imputation_bit_identity.py pins.
PARAMS = {"Cu-Mg": [-41302.26, -5.1627, -8148.05, 0.0]}
SS_KWARGS = {"ref_mode": "from_unary_db"}

# Fit-state fields the ablation rebuild resets, with the value each is reset to.
RESET_FIELDS = [
    ("invariants", None),
    ("low_t_exp_phases", None),
    ("unsupported_comp_ranges", []),
    ("fit_unsupported_range_mode", None),
    ("coverage_report", None),
    ("full_comp_ss", False),
    ("guess_symbols", None),
    ("constraints", None),
    ("_ref_params", None),
    ("nmpath", None),
    ("hsx", None),
    ("init_error", False),
    ("skip_reason", None),
]
XS_MIX_RESET = [0.0, 0.0, 0.0, 0.0]
SENTINELS = {
    "invariants": [{"type": "sentinel"}],
    "low_t_exp_phases": [{"name": "sentinel"}],
    "unsupported_comp_ranges": [[0.1, 0.2]],
    "fit_unsupported_range_mode": "score",
    "coverage_report": "sentinel",
    "full_comp_ss": True,
    "guess_symbols": ["sentinel"],
    "constraints": ["sentinel"],
    "_ref_params": [1.0, 2.0],
    "nmpath": np.zeros((2, 4)),
    "hsx": "sentinel",
    "init_error": True,
    "skip_reason": "sentinel",
}
HF_ZR_SS_PHASES = ["BCC", "FCC", "HCP"]


def _reference_interp(bl: BinaryLiquid) -> np.ndarray:
    """What the sorted rebuild stores for ``bl``'s current phase list."""
    work = copy.deepcopy(bl)
    REBUILD_HULL_INTERP(work)
    return work.eqs["h_hull_interp"]


def _stored(bl: BinaryLiquid) -> np.ndarray:
    return bl.eqs["h_hull_interp"]


def _roundtrip(bl: BinaryLiquid) -> BinaryLiquid:
    return pickle.loads(pickle.dumps(bl))


def load(spec: str) -> BinaryLiquid:
    return BinaryLiquid.from_cache(spec, param_format="comb-exp", params=list(PARAMS["Cu-Mg"]))


def load_hf_zr(ss: bool) -> BinaryLiquid:
    return BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=ss, ss_kwargs=dict(SS_KWARGS))


@pytest.fixture(scope="module")
def systems() -> dict[str, BinaryLiquid]:
    return {
        "Cu-Mg": load("Cu-Mg"),
        "Mg-Cu": load("Mg-Cu"),
        "Hf-Zr ss_off": load_hf_zr(False),
        "Hf-Zr ss_on": load_hf_zr(True),
    }


@pytest.fixture(scope="module")
def cu_mg(systems) -> BinaryLiquid:
    return systems["Cu-Mg"]


@pytest.fixture(scope="module")
def ablation(cu_mg):
    """``(target, ablated, restored)`` for the first ablatable Cu-Mg phase."""
    target = LIST_ABLATABLE(cu_mg)[0]
    ablated = ABLATE(cu_mg, target.name)
    restored = RESTORE(ablated, [target])
    return target, ablated, restored


@pytest.fixture(scope="module")
def imputed(cu_mg):
    """``(stripped, imputed, records)``: Cu-Mg stripped of its interior compounds, then imputed."""
    stripped = strip_interior(copy.deepcopy(cu_mg))
    work = copy.deepcopy(stripped)
    records = IMPUTE(work, write_cache=False)
    assert records, "nothing imputed; the imputed states would be vacuous"
    return stripped, work, records


SYSTEM_KEYS = ["Cu-Mg", "Mg-Cu", "Hf-Zr ss_off", "Hf-Zr ss_on"]


class TestFromCache:
    @pytest.mark.parametrize("key", SYSTEM_KEYS)
    def test_the_stored_value_is_the_reference(self, systems, key):
        bl = systems[key]
        assert np.array_equal(_stored(bl), _reference_interp(bl))


class TestUnpickle:
    @pytest.mark.parametrize("key", SYSTEM_KEYS)
    def test_a_round_trip_stores_the_reference_and_the_source_value(self, systems, key):
        bl = systems[key]
        back = _roundtrip(bl)
        assert np.array_equal(_stored(back), _reference_interp(back))
        assert np.array_equal(_stored(back), _stored(bl))

    @pytest.mark.parametrize("which", ["ablated", "restored"])
    def test_ablated_and_restored_round_trips(self, ablation, which):
        _target, ablated, restored = ablation
        bl = ablated if which == "ablated" else restored
        back = _roundtrip(bl)
        assert np.array_equal(_stored(back), _reference_interp(back))
        assert np.array_equal(_stored(back), _stored(bl))

    def test_a_stripped_then_imputed_round_trip(self, imputed):
        _stripped, work, _records = imputed
        back = _roundtrip(work)
        assert np.array_equal(_stored(back), _reference_interp(back))
        assert np.array_equal(_stored(back), _stored(work))

    def test_a_round_trip_after_dropping_the_imputed_phases(self, imputed):
        _stripped, work, _records = imputed
        dropped = DROP(work)
        back = _roundtrip(dropped)
        assert np.array_equal(_stored(back), _reference_interp(back))
        assert np.array_equal(_stored(back), _stored(dropped))


class TestWithComponentOrder:
    def test_a_reversed_cu_mg_clone_differs_from_the_reference(self, cu_mg):
        clone = cu_mg.with_component_order(["Mg", "Cu"])
        assert not np.array_equal(_stored(clone), _reference_interp(clone))

    def test_the_reversed_cu_mg_clone_keeps_the_source_phase_order(self, cu_mg):
        clone = cu_mg.with_component_order(["Mg", "Cu"])
        fixed = [p for p in clone.phases if p.composition is not None]
        assert [p.name for p in fixed] == ["Cu (fcc)", "MgCu2", "Mg2Cu", "Mg (hcp)"]
        assert [float(clone._phase_x(p)) for p in fixed] == [
            1.0,
            0.6666666666666666,
            0.3333333333333333,
            0.0,
        ]

    def test_a_clone_round_trip_keeps_the_clone_value(self, cu_mg):
        clone = cu_mg.with_component_order(["Mg", "Cu"])
        assert np.array_equal(_stored(_roundtrip(clone)), _stored(clone))

    @pytest.mark.parametrize(("key", "equal"), [("Hf-Zr ss_on", True), ("Hf-Zr ss_off", False)])
    def test_reversed_hf_zr(self, systems, key, equal):
        bl = systems[key]
        clone = bl.with_component_order(list(reversed(bl.components)))
        assert np.array_equal(_stored(clone), _reference_interp(clone)) is equal


def test_removing_the_x0_phase_by_hand_then_rebuilding_differs_from_the_reference(cu_mg):
    """``_rebuild_thermodynamic_expressions`` pads x = 0 and x = 1 with zero enthalpy."""
    work = copy.deepcopy(cu_mg)
    at_zero = [
        p for p in work.phases if p.composition is not None and float(work._phase_x(p)) == 0.0
    ]
    assert [p.name for p in at_zero] == ["Cu (fcc)"]
    work.phases = [p for p in work.phases if all(p is not q for q in at_zero)]
    work._rebuild_thermodynamic_expressions()
    assert not np.array_equal(_stored(work), _reference_interp(work))


class TestAblationRebuild:
    def test_ablate_stores_the_reference(self, ablation):
        _target, ablated, _restored = ablation
        assert np.array_equal(_stored(ablated), _reference_interp(ablated))

    def test_restore_stores_the_reference_and_the_from_cache_value(self, cu_mg, ablation):
        _target, _ablated, restored = ablation
        assert np.array_equal(_stored(restored), _reference_interp(restored))
        assert np.array_equal(_stored(restored), _stored(cu_mg))


class TestApplyImputedEntries:
    def test_the_cu_mg_imputed_entries_on_the_stripped_system(self, imputed):
        stripped, _work, records = imputed
        replay = copy.deepcopy(stripped)
        APPLY_IMPUTED(replay, [r.entry_dict for r in records])
        assert np.array_equal(_stored(replay), _reference_interp(replay))

    @pytest.mark.parametrize("key", ["Cu-Mg", "Hf-Zr ss_on"])
    def test_no_entries(self, systems, key):
        work = copy.deepcopy(systems[key])
        APPLY_IMPUTED(work, [])
        assert np.array_equal(_stored(work), _reference_interp(work))

    def test_no_entries_on_hf_zr_with_solid_solutions_keeps_the_phase_list(self, systems):
        """The rebuilt list is the loaded one: solid solutions kept, covered polymorphs out."""
        source = systems["Hf-Zr ss_on"]
        work = copy.deepcopy(source)
        APPLY_IMPUTED(work, [])
        assert [p.name for p in work.phases] == [p.name for p in source.phases]
        assert sorted(work.ss_models) == HF_ZR_SS_PHASES
        assert np.array_equal(_stored(work), _stored(source))

    def test_an_imputed_entry_on_hf_zr_keeps_the_solid_solutions_drawn(self, systems):
        work = copy.deepcopy(systems["Hf-Zr ss_on"])
        entry = ComputedEntry(
            Composition("HfZr"),
            0.0,
            entry_id="imputed:HfZr",
            data={"imputed": True, "label": "HfZr", "x_b": 0.5, "h_form_j_per_mol": -20000.0},
        )
        APPLY_IMPUTED(work, [entry.as_dict()])
        by_name = {p.name: p for p in work.phases}
        assert "HfZr" in by_name
        assert all(by_name[name].model is not None for name in HF_ZR_SS_PHASES)
        assert not any("(bcc)" in name or "(hcp)" in name for name in by_name)
        drawn = {t.name for t in BLPlotter(work).get_plot("fit+liq").data}
        assert {"BCC (Hf, Zr)", "HCP (Hf, Zr)", "HfZr"} <= drawn


def _sentinel_copy(bl: BinaryLiquid) -> BinaryLiquid:
    work = copy.deepcopy(bl)
    for name, value in SENTINELS.items():
        setattr(work, name, copy.deepcopy(value))
    for phase in work.phases:
        phase.points = [[0.5, 1000.0]]
    work.xs_mix.update(list(PARAMS["Cu-Mg"]))
    assert list(work.xs_mix.values) != XS_MIX_RESET
    return work


def _assert_reset(bl: BinaryLiquid) -> None:
    for name, value in RESET_FIELDS:
        got = getattr(bl, name)
        if value is None:
            assert got is None, name
        else:
            assert got == value and type(got) is type(value), name
    assert list(bl.xs_mix.values) == XS_MIX_RESET, "xs_mix"
    assert all(p.points == [] for p in bl.phases), "phase.points"


class TestTheResetContract:
    def test_ablate_resets_every_field(self, cu_mg):
        work = _sentinel_copy(cu_mg)
        _assert_reset(ABLATE(work, LIST_ABLATABLE(work)[0].name))

    def test_restore_resets_every_field(self, cu_mg, ablation):
        target, ablated, _restored = ablation
        work = _sentinel_copy(ablated)
        _assert_reset(RESTORE(work, [target]))


class TestBinaryLiquidMethods:
    """``BinaryLiquid.rebuild_hull_interp`` and ``BinaryLiquid.reset_fit_state`` called directly."""

    def test_rebuild_reproduces_the_from_cache_value(self, cu_mg):
        work = copy.deepcopy(cu_mg)
        work.eqs["h_hull_interp"] = np.zeros_like(_stored(cu_mg))
        work.rebuild_hull_interp()
        assert np.array_equal(_stored(work), _stored(cu_mg))

    def test_rebuild_follows_the_phase_list(self, cu_mg):
        work = copy.deepcopy(cu_mg)
        interior = LIST_ABLATABLE(work)[0]
        work.phases = [p for p in work.phases if p is not interior]
        work.rebuild_hull_interp()
        pts = HULL_POINTS(work)
        assert np.array_equal(_stored(work), np.interp(x_vals[1:-1], pts[:, 0], pts[:, 1]))
        assert not np.array_equal(_stored(work), _stored(cu_mg))

    def test_rebuild_without_a_fixed_composition_phase_raises_and_leaves_eqs(self, cu_mg):
        work = copy.deepcopy(cu_mg)
        work.phases = [p for p in work.phases if p.composition is None]
        before = _stored(work)
        keys = list(work.eqs)
        with pytest.raises(ValueError, match="no fixed-composition phases"):
            work.rebuild_hull_interp()
        assert _stored(work) is before
        assert list(work.eqs) == keys

    def test_fit_state_reset_clears_every_field(self, cu_mg):
        work = _sentinel_copy(cu_mg)
        assert work.reset_fit_state() is None
        _assert_reset(work)
