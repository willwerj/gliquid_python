"""Exact behaviour at the decision seams of the coverage resolvers, ``fit_parameters``,
the miscibility-gap walk, ``BinaryLiquid.find_invariant_points``, the flank bounds and
``plot_tx``.

Every observation is compared in :func:`canon` form, which keeps ``int`` / ``float`` /
``float64`` apart and never rounds. Each case drives an entry point that stays public (or a
named private caller of the seam) and pins its output. The labelled seams, whose tests
carry ``latent`` in their names:

  * L1 -- a per-call ``low_temp_axis_fraction`` raises; ``dft_cover_tol`` and a
    function-level ``polymorph_pair_tol`` are recorded on ``report.thresholds`` and do not
    move the gate;
  * L2 -- ``fit_parameters`` masks under the per-fit
    ``coverage_thresholds['dft_cover_tol']``;
  * L3 -- the right-hand miscibility-gap walk skips two segments where the left skips one;
  * L4 -- a vertical liquidus segment on either walk raises ``ZeroDivisionError``;
  * L5 -- ``identify_invariant_points`` retypes the returned phase dicts (pinned in
    ``test_decomposition_pins.py``);
  * L6 -- ``fit_parameters`` logs a failing attempt and returns ``[]`` however many run;
  * L7 -- the flank fallbacks of ``fit_parameters`` are the ints ``0`` and ``1``.

The expectations are read into ``EXPECTED`` from ``fixtures/decomposition_seams_expected.json``.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import sympy

import gliquid.binary as binary
import gliquid.config as config
import gliquid.mpds as mpds
import gliquid.plotting.binary_tx as binary_tx
import gliquid.solid_coverage as solid_coverage
from gliquid.binary import BinaryLiquid, plot_tx, record_tie_lines

from ._mpds_builders import FakeHsx as _FakeHsx
from ._mpds_builders import coverage_report
from ._mpds_builders import facet_rows as _facet
from .test_miscibility_gap_detection import COMP_DATA, TEMP_RANGE, _json, _svg

ONE_THIRD = 1.0 / 3.0
TWO_THIRDS = 2.0 / 3.0

EXPECTED = json.loads(
    (Path(__file__).parent / "fixtures" / "decomposition_seams_expected.json").read_text(
        encoding="utf-8"
    )
)


# -- Typed canonical form and the probes --
def _sort_key(item):
    return json.dumps(item, sort_keys=True)


def canon(x):
    """``x`` as nested dicts, lists and ``"<type>:<repr>"`` strings, compared exactly.

    Dicts become key-sorted dicts with ``str`` keys; lists stay lists; tuples become
    ``{"tuple": [...]}``; sets become ``{"set": [...]}`` sorted; ndarrays become
    ``{"ndarray": dtype, "items": [...]}``; dataclass instances become
    ``{"dataclass": name, "fields": {...}}``; numpy scalars become
    ``"<type>:<repr of .item()>"`` and every other value ``"<type>:<repr>"``.
    """
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, list):
        return [canon(v) for v in x]
    if isinstance(x, tuple):
        return {"tuple": [canon(v) for v in x]}
    if isinstance(x, (set, frozenset)):
        return {"set": sorted((canon(v) for v in x), key=_sort_key)}
    if isinstance(x, np.ndarray):
        return {"ndarray": str(x.dtype), "items": [canon(v) for v in x.tolist()]}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {
            "dataclass": type(x).__name__,
            "fields": {f.name: canon(getattr(x, f.name)) for f in dataclasses.fields(x)},
        }
    if isinstance(x, np.generic):
        return f"{type(x).__name__}:{x.item()!r}"
    return f"{type(x).__name__}:{x!r}"


def digest(obj) -> str:
    """First 16 hex digits of the SHA-256 of ``canon(obj)`` serialised with sorted keys."""
    text = json.dumps(canon(obj), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def eq_digest(eq) -> str:
    """First 16 hex digits of the SHA-256 of ``sympy.srepr(eq)``."""
    return hashlib.sha256(sympy.srepr(eq).encode("utf-8")).hexdigest()[:16]


class _ProbeStop(Exception):
    """Raised by the fit probe in place of candidate building."""


def fit_probe(monkeypatch, bl, *, full_srepr=False, **fit_kwargs):
    """Everything ``fit_parameters`` decides before optimisation.

    ``bl._invariant_constraint_ics`` is replaced by a recorder that keeps
    ``[name, order, canon(temperature), equation]`` per constraint equation (the equation
    as ``sympy.srepr`` with ``full_srepr``, else as :func:`eq_digest`) and raises
    :class:`_ProbeStop`.

    Returns:
        dict: ``result`` (``"probe_stop"`` or ``canon`` of the returned value), ``eqs``,
        ``unsupported`` (``canon(bl.unsupported_comp_ranges)``) and ``skip_reason``.
    """
    rows = []

    def record(eqs, **kwargs):
        for name, order, temp, eq in eqs:
            rows.append(
                [name, order, canon(temp), sympy.srepr(eq) if full_srepr else eq_digest(eq)]
            )
        raise _ProbeStop

    monkeypatch.setattr(bl, "_invariant_constraint_ics", record)
    try:
        result = canon(bl.fit_parameters(**fit_kwargs))
    except _ProbeStop:
        result = "probe_stop"
    return {
        "result": result,
        "eqs": rows,
        "unsupported": canon(bl.unsupported_comp_ranges),
        "skip_reason": canon(bl.skip_reason),
    }


def _inv(kind, comp, phases=None, phase_comps=None, temp=1000.0, **extra):
    record = {"type": kind, "comp": comp, "temp": temp}
    if phases is not None:
        record["phases"] = list(phases)
    if phase_comps is not None:
        record["phase_comps"] = list(phase_comps)
    record.update(extra)
    return record


def _none(comp):
    """An invariant that bounds a neighbouring field and matches no walk branch."""
    return {"type": "none", "comp": comp}


def _paren_cmp(comp):
    """A terminal-phase cmp no DFT phase covers: the walk masks the flank away from 0.5."""
    return _inv("cmp", comp, ["(X)"], [comp])


def _report(ranges, min_supported_total=0.25):
    return coverage_report(ranges, min_supported_total, sum(hi - lo for lo, hi in ranges))


def walk_probe(monkeypatch, invariants, low_t=(), *, system="Cu-Mg", prepare=None, **fit_kwargs):
    """:func:`fit_probe` on a fresh ``system`` whose invariant walk reads ``invariants``."""
    bl = BinaryLiquid.from_cache(system, param_format="linear")
    bl.invariants = copy.deepcopy(list(invariants))
    bl.low_t_exp_phases = copy.deepcopy(list(low_t))
    if prepare is not None:
        prepare(bl, monkeypatch)
    fit_kwargs.setdefault("check_solid_coverage", False)
    return fit_probe(monkeypatch, bl, **fit_kwargs)


# -- Coverage-default resolvers --
@dataclasses.dataclass(frozen=True)
class _Site:
    id: str
    knob: str
    run: object
    explicit: object
    patched: object


def _assessed_threshold(param):
    def run(value):
        report = solid_coverage.assess_solid_coverage(
            [], [], [0.0, 1.0], [], thresholds={param: value}
        )
        return report.thresholds[param]

    return run


_SS_FIELDS = [
    {
        "type": "ss",
        "name": "(Cu)",
        "comp": 0.0,
        "cbounds": [[0.0, 900.0], [0.15, 900.0]],
        "structure": {"spacegroup": 99},
    },
    {
        "type": "ss",
        "name": "(Mg)",
        "comp": 1.0,
        "cbounds": [[0.72, 700.0], [1.0, 700.0]],
        "structure": {"spacegroup": 99},
    },
    {"type": "ss", "name": "(Cu)", "comp": 0.1, "structure": {"spacegroup": 99}},
    {
        "type": "ss",
        "name": "CuMg",
        "comp": 0.5,
        "cbounds": [[0.4, 800.0], [0.55, 800.0]],
        "structure": {"spacegroup": 99},
    },
]


def _rescuable(param):
    return lambda value: solid_coverage.rescuable_ss_indices(
        copy.deepcopy(_SS_FIELDS), ["Cu", "Mg"], **{param: value}
    )


_ORDERED_FIELD = [
    {
        "type": "ss",
        "name": "CuMg",
        "comp": 0.5,
        "cbounds": [[0.45, 900.0], [0.55, 900.0]],
        "structure": {"spacegroup": 99},
    }
]


def _assigned(param):
    return lambda value: solid_coverage.assign_dft_phases(
        copy.deepcopy(_ORDERED_FIELD), ["Cu", "Mg"], [0.57], {0.57: 99}, **{param: value}
    )


_POLYMORPH_SHAPES = [
    {
        "type": "lc",
        "name": "Mg2Cu",
        "comp": 0.66,
        "tbounds": [[0.66, 300.0], [0.66, 700.0]],
        "polymorph": "rt",
    },
    {
        "type": "lc",
        "name": "Mg2Cu",
        "comp": 0.69,
        "tbounds": [[0.69, 700.0], [0.69, 900.0]],
        "polymorph": "ht",
    },
]


def _collapsed(value):
    return mpds.collapse_polymorphs(
        copy.deepcopy(_POLYMORPH_SHAPES), None, comp_tol=value, elements=["Cu", "Mg"]
    )


_FOOT_LIQUIDUS = [[0.0, 1000.0], [0.2, 900.0], [0.5, 1100.0], [0.8, 925.0], [1.0, 1000.0]]
_FOOT_MIG = {"type": "mig", "comp": 0.5, "cbounds": [[0.2, 900.0], [0.8, 925.0]], "apex": None}


def _mig_headers(value):
    bl = BinaryLiquid.from_cache("Cu-Mg")
    bl.digitized_liq = copy.deepcopy(_FOOT_LIQUIDUS)
    return [entry[:3] for entry in bl._mig_constraint_equations(dict(_FOOT_MIG), value)]


_ASSESS = "gliquid.solid_coverage:assess_solid_coverage"
SITES = [
    _Site(f"{_ASSESS}:ss_narrow_tol", "ss_narrow_tol", None, 0.123, 0.137),
    _Site(f"{_ASSESS}:dft_cover_tol", "dft_cover_tol", None, 0.123, 0.137),
    _Site(f"{_ASSESS}:dft_edge_tol", "dft_edge_tol", None, 0.023, 0.017),
    _Site(f"{_ASSESS}:dft_structure_tol", "dft_structure_tol", None, 0.063, 0.071),
    _Site(f"{_ASSESS}:ss_rescue_max_width", "ss_rescue_max_width", None, 0.33, 0.29),
    _Site(f"{_ASSESS}:subliquidus_margin", "subliquidus_margin", None, 7, 12.5),
    _Site(f"{_ASSESS}:mask_miscibility_gaps", "mask_miscibility_gaps", None, True, True),
    _Site(f"{_ASSESS}:mig_foot_tol", "mig_foot_tol", None, 33.0, 27.0),
    _Site(f"{_ASSESS}:bound_by_supported", "bound_by_supported", None, False, False),
    _Site(f"{_ASSESS}:field_ceiling_tol", "field_ceiling_tol", None, 3.0, 4.0),
    _Site(f"{_ASSESS}:ss_extension_cap", "ss_extension_cap", None, 0.08, 0.09),
    _Site(f"{_ASSESS}:ss_width_guard", "ss_width_guard", None, 0.5, 0.55),
    _Site(f"{_ASSESS}:polymorph_pair_tol", "polymorph_pair_tol", None, 0.03, 0.02),
]
SITES = [dataclasses.replace(s, run=_assessed_threshold(s.id.rsplit(":", 1)[1])) for s in SITES]
SITES += [
    _Site(
        "gliquid.solid_coverage:rescuable_ss_indices:ss_narrow_tol",
        "ss_narrow_tol",
        _rescuable("ss_narrow_tol"),
        0.16,
        0.2,
    ),
    _Site(
        "gliquid.solid_coverage:rescuable_ss_indices:ss_rescue_max_width",
        "ss_rescue_max_width",
        _rescuable("ss_rescue_max_width"),
        0.12,
        0.3,
    ),
    _Site(
        "gliquid.solid_coverage:assign_dft_phases:edge_tol",
        "dft_edge_tol",
        _assigned("edge_tol"),
        0.03,
        0.025,
    ),
    _Site(
        "gliquid.solid_coverage:assign_dft_phases:structure_tol",
        "dft_structure_tol",
        _assigned("structure_tol"),
        0.015,
        0.012,
    ),
    _Site(
        "gliquid.mpds:low_temp_threshold:fraction",
        "low_temp_axis_fraction",
        lambda value: mpds.low_temp_threshold({"temp": [100.0, 1100.0]}, value),
        0.07,
        0.09,
    ),
    _Site(
        "gliquid.mpds:_collapse_polymorph_clusters:comp_tol",
        "polymorph_pair_tol",
        _collapsed,
        0.025,
        0.02,
    ),
    _Site(
        "gliquid.binary:BinaryLiquid._mig_constraint_equations:foot_tol",
        "mig_foot_tol",
        _mig_headers,
        30.0,
        26.0,
    ),
]


class TestCoverageResolvers:
    """One case per genuine ``None -> config.coverage_<knob>`` site, plus the asymmetric
    site and the direct reads."""

    @pytest.mark.parametrize("site", SITES, ids=[s.id for s in SITES])
    def test_explicit_value_is_used_and_none_reads_config_at_call_time(self, site, monkeypatch):
        expected = EXPECTED["resolvers"][site.id]
        default = canon(site.run(None))
        explicit = canon(site.run(site.explicit))
        assert default == expected["default"]
        assert explicit == expected["explicit"]
        assert explicit != default
        monkeypatch.setattr(config, f"coverage_{site.knob}", site.patched)
        patched = canon(site.run(None))
        assert patched == expected["patched"]
        assert patched != default

    @pytest.mark.parametrize("pair_tol", [None, 0.025, 0, 1], ids=repr)
    def test_asymmetric_pair_tol_site(self, pair_tol):
        shapes = [
            {"name": "Mg2Cu", "comp": 0.66, "polymorph": "ht"},
            {"name": "Mg2Cu", "comp": 0.69, "polymorph": "rt"},
        ]
        got = canon(mpds.mark_unpaired_ht_forms(shapes, ["Cu", "Mg"], pair_tol=pair_tol))
        assert got == EXPECTED["asymmetric"][repr(pair_tol)]

    def test_asymmetric_pair_tol_none_reads_config_at_call_time(self, monkeypatch):
        monkeypatch.setattr(config, "coverage_polymorph_pair_tol", 0.02)
        shapes = [
            {"name": "Mg2Cu", "comp": 0.66, "polymorph": "ht"},
            {"name": "Mg2Cu", "comp": 0.69, "polymorph": "rt"},
        ]
        got = canon(mpds.mark_unpaired_ht_forms(shapes, ["Cu", "Mg"]))
        assert got == EXPECTED["asymmetric"]["patched"]

    @pytest.mark.parametrize("foot_tol", [None, 10.0])
    def test_direct_read_expected_gap_ranges_follows_config(self, foot_tol, monkeypatch):
        if foot_tol is not None:
            monkeypatch.setattr(config, "coverage_mig_foot_tol", foot_tol)
        bl = BinaryLiquid.from_cache("Cu-Mg")
        bl.digitized_liq = copy.deepcopy(_FOOT_LIQUIDUS)
        bl.invariants = [
            {"type": "eut", "comp": 0.1},
            {"type": "mig", "comp": 0.5, "cbounds": [[0.2, 915.0], [0.8, 925.0]], "apex": None},
        ]
        got = canon(bl.expected_gap_ranges)
        assert got == EXPECTED["direct"][f"expected_gap_ranges:{foot_tol!r}"]

    @pytest.mark.parametrize("key", ["min_supported_total"])
    def test_direct_read_threshold_dict_follows_config(self, key, monkeypatch):
        monkeypatch.setattr(config, f"coverage_{key}", 7)
        report = solid_coverage.assess_solid_coverage([], [], [0.0, 1.0], [])
        assert canon(report.thresholds[key]) == "int:7"
        override = solid_coverage.assess_solid_coverage(
            [], [], [0.0, 1.0], [], thresholds={key: 0.5}
        )
        assert canon(override.thresholds[key]) == "float:0.5"


# -- Preserved coverage defects (L1, L2) --
def _assessed(system, **overrides):
    bl = BinaryLiquid.from_cache(system)
    bl.find_invariant_points()
    return bl, bl.assess_solid_coverage(**overrides)


def _without(report_canon, knob):
    out = copy.deepcopy(report_canon)
    del out["fields"]["thresholds"][knob]
    return out


class TestLatentCoverageBugs:
    @pytest.mark.parametrize("system", ["Al-Cu", "Bi-La"])
    def test_dft_cover_tol_is_recorded_and_does_not_move_the_gate(self, system):
        """A dft_cover_tol override changes report.thresholds['dft_cover_tol'] and nothing else."""
        knob = "dft_cover_tol"
        _, base = _assessed(system)
        _, over = _assessed(system, thresholds={knob: 0.001})
        assert canon(over.thresholds[knob]) == canon(0.001)
        assert _without(canon(over), knob) == _without(canon(base), knob)

    def test_l1_a_per_call_low_temp_fraction_raises(self):
        with pytest.raises(ValueError, match="low_temp_axis_fraction cannot be overridden"):
            _assessed("Al-Cu", thresholds={"low_temp_axis_fraction": 0.0})

    def test_latent_l1_control_config_low_temp_fraction_moves_al_cu(self, monkeypatch):
        """The same low_temp_axis_fraction, set in config before the invariants are found,
        changes the Al-Cu report outside thresholds."""
        _, base = _assessed("Al-Cu")
        monkeypatch.setattr(config, "coverage_low_temp_axis_fraction", 0.0)
        _, moved = _assessed("Al-Cu")
        knob = "low_temp_axis_fraction"
        assert _without(canon(moved), knob) != _without(canon(base), knob)

    @pytest.mark.parametrize("system", ["Al-Cu", "Bi-La"])
    def test_latent_l1_dft_cover_tol_config_value_moves_nothing(self, system, monkeypatch):
        """No reader of config.coverage_dft_cover_tol sits on the coverage path, so even the
        config value changes only report.thresholds."""
        _, base = _assessed(system)
        monkeypatch.setattr(config, "coverage_dft_cover_tol", 0.001)
        _, moved = _assessed(system)
        knob = "dft_cover_tol"
        assert _without(canon(moved), knob) == _without(canon(base), knob)

    def test_latent_l1_function_level_polymorph_pair_tol_is_recorded_not_applied(self):
        """At solid_coverage.assess_solid_coverage a polymorph_pair_tol override changes
        report.thresholds['polymorph_pair_tol'] and nothing else."""
        bl = BinaryLiquid.from_cache("Al-Cu")
        bl.find_invariant_points()
        phases, unpaired = bl.coverage_phase_inputs()
        span = [bl.digitized_liq[0][0], bl.digitized_liq[-1][0]]

        def assess(**overrides):
            return solid_coverage.assess_solid_coverage(
                copy.deepcopy(phases),
                copy.deepcopy(bl.invariants),
                span,
                bl._interior_dft_comps(),
                bl.ss_models,
                liquidus=bl.digitized_liq,
                elements=bl.components,
                unpaired_ht=unpaired,
                **overrides,
            )

        base, over = assess(), assess(thresholds={"polymorph_pair_tol": 0.01})
        knob = "polymorph_pair_tol"
        assert canon(over.thresholds[knob]) == "float:0.01"
        assert _without(canon(over), knob) == _without(canon(base), knob)

    def test_method_level_polymorph_pair_tol_override_report(self):
        _, report = _assessed("Al-Cu", thresholds={"polymorph_pair_tol": 0.01})
        _, base = _assessed("Al-Cu")
        assert digest(report) == EXPECTED["latent"]["al_cu_pair_tol_0.01"]
        assert digest(base) == EXPECTED["latent"]["al_cu_base"]
        assert digest(report) != digest(base)

    @staticmethod
    def _l2_probe(monkeypatch, **fit_kwargs):
        invariants = [_inv("cmp", 0.40, ["(X)"], [0.40])]
        return walk_probe(monkeypatch, invariants, **fit_kwargs)

    def test_latent_l2_default_cover_tol_masks_nothing(self, monkeypatch):
        """A '(' congruent point within 0.10 of the 1/3 DFT phase masks nothing."""
        got = self._l2_probe(monkeypatch)
        assert got == EXPECTED["latent"]["l2_default"]
        assert got["unsupported"] == []

    def test_latent_l2_control_config_cover_tol_masks(self, monkeypatch):
        """With config.coverage_dft_cover_tol at 0.001 the same input masks."""
        monkeypatch.setattr(config, "coverage_dft_cover_tol", 0.001)
        got = self._l2_probe(monkeypatch)
        assert got == EXPECTED["latent"]["l2_config"]
        assert got["unsupported"] != []

    def test_l2_per_fit_cover_tol_masks_like_config(self, monkeypatch):
        """fit_parameters(coverage_thresholds={'dft_cover_tol': 0.001}) masks as config does."""
        got = self._l2_probe(monkeypatch, coverage_thresholds={"dft_cover_tol": 0.001})
        assert got == EXPECTED["latent"]["l2_config"]

    def test_the_removed_per_fit_keyword_names_its_replacement(self, monkeypatch):
        with pytest.raises(TypeError, match="coverage_thresholds=\\{'dft_cover_tol'"):
            self._l2_probe(monkeypatch, coverage_dft_cover_tol=0.001)


# -- fit_parameters' closures and the walk around them --
def _twin(first):
    """Insert a copy of MgCu2 with a lower enthalpy before or after the original."""

    def prepare(bl, _monkeypatch):
        index = next(i for i, p in enumerate(bl.phases) if p.name == "MgCu2")
        original = bl.phases[index]
        twin = dataclasses.replace(original, name="MgCu2-twin", enthalpy=original.enthalpy - 1000.0)
        bl.phases.insert(index if first else index + 1, twin)

    return prepare


def _labels_at(*comps):
    def prepare(_bl, monkeypatch):
        monkeypatch.setattr(binary.mpds, "two_liquid_label_comps", lambda *_a, **_k: list(comps))

    return prepare


def _cover_tol(value):
    def prepare(_bl, monkeypatch):
        monkeypatch.setattr(config, "coverage_dft_cover_tol", value)

    return prepare


def _set_attr(name, value):
    def prepare(bl, _monkeypatch):
        setattr(bl, name, value)

    return prepare


def _only_solution_phases(bl, _monkeypatch):
    bl.phases = [p for p in bl.phases if p.composition is None]


def _cap(value):
    def prepare(_bl, monkeypatch):
        monkeypatch.setattr(binary, "MASK_FRACTION_CAP", value)

    return prepare


def _coverage_report(ranges):
    def prepare(bl, _monkeypatch):
        bl.coverage_report = _report(ranges)

    return prepare


def _caller_ranges(ranges):
    def prepare(bl, _monkeypatch):
        bl.unsupported_comp_ranges = [list(r) for r in ranges]

    return prepare


def _union_gate(ranges, floor):
    def prepare(_bl, monkeypatch):
        def assess(self, **overrides):
            self.coverage_report = _report(ranges, min_supported_total=floor)
            return self.coverage_report

        monkeypatch.setattr(BinaryLiquid, "assess_solid_coverage", assess)

    return prepare


_EXACT_TOL = abs(ONE_THIRD - 0.4)
_WALK_CASES = {
    # find_nearest_phase
    "nearest_cmp_within_0.02": ([_inv("cmp", ONE_THIRD + 0.019, ["MgCu2"])], (), {}),
    "nearest_cmp_beyond_0.02": ([_inv("cmp", ONE_THIRD + 0.021, ["MgCu2"])], (), {}),
    "nearest_per_within_0.04": ([_inv("per", 0.30, ["P"], [ONE_THIRD + 0.039])], (), {}),
    "nearest_per_beyond_0.04": ([_inv("per", 0.30, ["P"], [ONE_THIRD + 0.041])], (), {}),
    "nearest_eut_left_within_0.04": (
        [_inv("eut", 0.5, ["MgCu2", "Mg2Cu"], [ONE_THIRD + 0.039, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "nearest_eut_left_beyond_0.04": (
        [_inv("eut", 0.5, ["MgCu2", "Mg2Cu"], [ONE_THIRD + 0.041, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "nearest_eut_right_within_0.04": (
        [_inv("eut", 0.5, ["MgCu2", "Mg2Cu"], [ONE_THIRD, TWO_THIRDS - 0.039], temp=800.0)],
        (),
        {},
    ),
    "nearest_eut_right_beyond_0.04": (
        [_inv("eut", 0.5, ["MgCu2", "Mg2Cu"], [ONE_THIRD, TWO_THIRDS - 0.041], temp=800.0)],
        (),
        {},
    ),
    "nearest_tie_twin_first": (
        [_inv("cmp", ONE_THIRD, ["MgCu2"])],
        (),
        {"prepare": _twin(first=True)},
    ),
    "nearest_tie_twin_second": (
        [_inv("cmp", ONE_THIRD, ["MgCu2"])],
        (),
        {"prepare": _twin(first=False)},
    ),
    "nearest_with_no_fixed_phase": (
        [_inv("cmp", ONE_THIRD, ["MgCu2"])],
        (),
        {"prepare": _only_solution_phases},
    ),
    # flank_bound (fit_parameters' invariant-flank calls)
    "flank_5e-7_not_candidates": (
        [_none(0.3), _none(0.45 - 5e-7), _paren_cmp(0.45)],
        (),
        {},
    ),
    "flank_2e-6_candidates": (
        [_none(0.3), _none(0.45 - 2e-6), _paren_cmp(0.45)],
        (),
        {},
    ),
    "flank_no_candidate_cap_skips": ([_paren_cmp(0.45), _paren_cmp(0.55)], (), {}),
    # _dft_comp_within
    "covers_at_exact_tolerance": (
        [_inv("cmp", 0.4, ["(X)"], [0.4])],
        (),
        {"prepare": _cover_tol(_EXACT_TOL)},
    ),
    "covers_one_ulp_below_tolerance": (
        [_inv("cmp", 0.4, ["(X)"], [0.4])],
        (),
        {"prepare": _cover_tol(math.nextafter(_EXACT_TOL, 0.0))},
    ),
    "cmp_emits": (
        [_none(0.2), _inv("cmp", ONE_THIRD + 0.01, ["HT"]), _none(0.5)],
        (),
        {},
    ),
    "init_error_returns_empty": (
        [_inv("cmp", ONE_THIRD, ["MgCu2"])],
        (),
        {"prepare": _set_attr("init_error", True)},
    ),
    "no_liquidus_returns_empty": (
        [_inv("cmp", ONE_THIRD, ["MgCu2"])],
        (),
        {"prepare": _set_attr("digitized_liq", None)},
    ),
    "cmp_per_eut_emit": (
        [
            _inv("cmp", ONE_THIRD + 0.01, ["HT"]),
            _inv("per", 0.30, ["HT"], [ONE_THIRD]),
            _inv("eut", 0.5, ["HT", "Mg2Cu"], [ONE_THIRD, TWO_THIRDS], temp=800.0),
        ],
        (),
        {},
    ),
    # mask_walked
    "mask_walked_no_label": ([_none(0.1), _inv("cmp", 0.2, ["(Cu)"], [0.2])], (), {}),
    "mask_walked_label_at_lo_minus_1e-9": (
        [_none(0.1), _inv("cmp", 0.2, ["(Cu)"], [0.2])],
        (),
        {"prepare": _labels_at(0.1 - 1e-9)},
    ),
    "mask_walked_label_at_hi_plus_1e-9": (
        [_none(0.1), _inv("cmp", 0.2, ["(Cu)"], [0.2])],
        (),
        {"prepare": _labels_at(0.2 + 1e-9)},
    ),
    "mask_walked_label_at_lo_minus_3e-9": (
        [_none(0.1), _inv("cmp", 0.2, ["(Cu)"], [0.2])],
        (),
        {"prepare": _labels_at(0.1 - 3e-9)},
    ),
    "mask_walked_label_at_hi_plus_3e-9": (
        [_none(0.1), _inv("cmp", 0.2, ["(Cu)"], [0.2])],
        (),
        {"prepare": _labels_at(0.2 + 3e-9)},
    ),
    # the walk's branch table
    "cmp_paren_right": ([_inv("cmp", 0.8, ["(Mg)"], [0.8]), _none(0.9)], (), {}),
    "cmp_paren_at_half": ([_inv("cmp", 0.5, ["(X)"], [0.5])], (), {}),
    "cmp_paren_covered": ([_inv("cmp", 0.4, ["(X)"], [0.4])], (), {}),
    "cmp_paren_caller_supplied_ranges": (
        [_inv("cmp", ONE_THIRD + 0.01, ["(Cu)"], [ONE_THIRD + 0.01])],
        (),
        {"prepare": _caller_ranges([[0.9, 0.95]])},
    ),
    "per_paren_phase_below": (
        [_none(0.05), _inv("per", 0.2, ["(Cu)"], [0.1]), _none(0.35)],
        (),
        {},
    ),
    "per_paren_phase_above": (
        [_none(0.05), _inv("per", 0.2, ["(Cu)"], [0.3]), _none(0.35)],
        (),
        {},
    ),
    "per_paren_phase_equal": (
        [_none(0.05), _inv("per", 0.2, ["(Cu)"], [0.2]), _none(0.35)],
        (),
        {},
    ),
    "eut_none_in_phase_comps": (
        [_none(0.3), _inv("eut", 0.5, ["(Cu)", None], [0.45, None], temp=800.0), _none(0.7)],
        (),
        {},
    ),
    "eut_left_missing_uncovered": (
        [_none(0.3), _inv("eut", 0.5, ["X", "Mg2Cu"], [0.45, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "eut_left_past_comp_covered": (
        [_none(0.1), _inv("eut", 0.3, ["MgCu2", "Mg2Cu"], [0.3, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "eut_left_past_comp_uncovered": (
        [_none(0.1), _inv("eut", 0.3, ["MgCu2", "Mg2Cu"], [0.3, TWO_THIRDS], temp=800.0)],
        (),
        {"prepare": _cover_tol(0.01)},
    ),
    "eut_left_paren_beyond_0.05": (
        [_inv("eut", 0.5, ["(Cu)", "Mg2Cu"], [ONE_THIRD - 0.03, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "eut_left_paren_within_0.05": (
        [_inv("eut", 0.5, ["(Cu)", "Mg2Cu"], [0.04, TWO_THIRDS], temp=800.0)],
        (),
        {},
    ),
    "eut_right_past_comp_covered": (
        [_inv("eut", 0.7, ["MgCu2", "Mg2Cu"], [ONE_THIRD, 0.7], temp=800.0), _none(0.9)],
        (),
        {},
    ),
    "eut_right_past_comp_uncovered": (
        [_inv("eut", 0.7, ["MgCu2", "Mg2Cu"], [ONE_THIRD, 0.7], temp=800.0), _none(0.9)],
        (),
        {"prepare": _cover_tol(0.01)},
    ),
    "eut_right_paren_below_0.95": (
        [_inv("eut", 0.5, ["MgCu2", "(Mg)"], [ONE_THIRD, TWO_THIRDS + 0.03], temp=800.0)],
        (),
        {},
    ),
    "eut_right_paren_within_0.95": (
        [_inv("eut", 0.5, ["MgCu2", "(Mg)"], [ONE_THIRD, 0.97], temp=800.0)],
        (),
        {},
    ),
    "mig_delegates": (
        [
            {
                "type": "mig",
                "comp": 0.5,
                "cbounds": [[0.3, 1000.0], [0.7, 1000.0]],
                "apex": [0.5, 1200.0],
            }
        ],
        (),
        {},
    ),
    # post-walk merge, cap and union gate
    "merge_touching_within_1e-9": (
        [],
        (),
        {"prepare": _coverage_report([[0.2, 0.3], [0.3 + 1e-9, 0.4]])},
    ),
    "merge_gap_of_2e-9_stays_apart": (
        [],
        (),
        {"prepare": _coverage_report([[0.2, 0.3], [0.3 + 2e-9, 0.4]])},
    ),
    "merge_union_with_coverage_ranges": (
        [_none(0.3), _paren_cmp(0.45)],
        (),
        {"prepare": _coverage_report([[0.4, 0.6]])},
    ),
    "union_gate_skips": (
        [_paren_cmp(0.55), _none(0.75)],
        (),
        {"prepare": _union_gate([[0.0, 0.25]], 0.70), "check_solid_coverage": True},
    ),
    "union_gate_admits": (
        [_paren_cmp(0.55), _none(0.75)],
        (),
        {"prepare": _union_gate([[0.0, 0.25]], 0.50), "check_solid_coverage": True},
    ),
}


def run_walk_case(monkeypatch, name):
    invariants, low_t, kwargs = _WALK_CASES[name]
    return walk_probe(monkeypatch, invariants, low_t, **dict(kwargs))


class TestFitClosureSeams:
    @pytest.mark.parametrize("name", sorted(_WALK_CASES))
    def test_walk_case(self, name, monkeypatch):
        assert run_walk_case(monkeypatch, name) == EXPECTED["walk"][name]

    def test_the_liquid_phase_carries_no_composition(self):
        """find_nearest_phase skips it: every walk case above runs with it in self.phases."""
        bl = BinaryLiquid.from_cache("Cu-Mg")
        assert [p.name for p in bl.phases if p.composition is None] == ["L"]

    def test_an_unknown_keyword_raises_before_anything_is_touched(self, monkeypatch):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        with pytest.raises(TypeError, match="unexpected keyword argument: 'bogus'"):
            fit_probe(monkeypatch, bl, bogus=1)

    def test_a_lean_record_raises(self, monkeypatch):
        bl = BinaryLiquid.from_cache("Cu-Mg")
        bl.mpds_json = mpds.lean_record(bl.mpds_json)
        with pytest.raises(config.CacheModeError, match="LEAN MPDS record"):
            fit_probe(monkeypatch, bl)

    def test_latent_l7_flank_fallbacks_are_ints(self, monkeypatch):
        """With no invariant beyond them the walk's masks end at the ints 0 and 1."""
        got = walk_probe(monkeypatch, [_paren_cmp(0.45), _paren_cmp(0.55)], prepare=_cap(1.01))
        assert got["unsupported"] == [["int:0", "float:0.45"], ["float:0.55", "int:1"]]
        assert got == EXPECTED["walk_extra"]["l7_int_fallbacks"]


# -- fit_parameters when every attempt fails (L6) --
def _exhausted(monkeypatch, *, n_ics, nelder_mead, worker, **fit_kwargs):
    """A Cu-Mg fit whose candidates, H-S partition and workers are stubbed."""
    bl = BinaryLiquid.from_cache("Cu-Mg")
    fake_ics = [
        {
            "f": float(i + 1),
            "constrs": [[f"fake {i}", "0th order", 1000.0 + i, sympy.true]],
            "init_tri": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        }
        for i in range(n_ics)
    ]
    monkeypatch.setattr(
        BinaryLiquid, "_invariant_constraint_ics", lambda self, eqs, **kw: list(fake_ics)
    )
    monkeypatch.setattr(BinaryLiquid, "nelder_mead", nelder_mead)
    monkeypatch.setattr(binary, "_run_single_optimization_worker", worker)
    fit_kwargs.setdefault("check_solid_coverage", False)
    fit_kwargs.setdefault("init_objective_cutoff", None)
    return bl, bl.fit_parameters(**fit_kwargs)


def _cheap_nelder_mead(self, *args, **kwargs):
    return 1.0, {}, np.zeros((1, 1, 1))


def _failing_nelder_mead(self, *args, **kwargs):
    raise RuntimeError("stub H-S partition failure")


def _raising_worker(*args, **kwargs):
    raise ValueError("stub worker failure")


def _none_worker(*args, **kwargs):
    return None


def _result_worker(task_index, selected_ics, bl_template, *args):
    """A result dict carrying the template's own parameters; attempt 2 scores lower."""
    names = bl_template.xs_mix.format.param_names
    return {
        "f": 10.0 - task_index,
        "nmpath": np.full((1, 1, 1), float(task_index)),
        **{name: bl_template.xs_mix[name] for name in names},
    }


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


class TestFitExhaustedAttempts:
    def test_latent_l6_two_sequential_failures_return_empty(self, monkeypatch, caplog):
        """Two attempts that both raise are logged at ERROR and the call returns []."""
        with caplog.at_level(logging.ERROR, logger="gliquid"):
            _, out = _exhausted(
                monkeypatch,
                n_ics=1,
                nelder_mead=_cheap_nelder_mead,
                worker=_raising_worker,
                n_opts=2,
            )
        assert out == []
        assert _errors(caplog) == EXPECTED["exhausted"]["sequential_errors"]

    def test_latent_l6_two_threaded_failures_return_empty(self, monkeypatch, caplog):
        """The same with enable_multi_threading=True."""
        with caplog.at_level(logging.ERROR, logger="gliquid"):
            _, out = _exhausted(
                monkeypatch,
                n_ics=1,
                nelder_mead=_cheap_nelder_mead,
                worker=_raising_worker,
                n_opts=2,
                enable_multi_threading=True,
            )
        assert out == []
        assert sorted(_errors(caplog)) == EXPECTED["exhausted"]["threaded_errors"]

    def test_l6_single_failure_is_logged_and_returns_empty(self, monkeypatch, caplog):
        """A single attempt that raises is logged at ERROR and the call returns []."""
        with caplog.at_level(logging.ERROR, logger="gliquid"):
            _, out = _exhausted(
                monkeypatch,
                n_ics=0,
                nelder_mead=_cheap_nelder_mead,
                worker=_raising_worker,
                n_opts=1,
            )
        assert out == []
        assert _errors(caplog) == EXPECTED["exhausted"]["sequential_errors"][:1]

    def test_latent_l6_worker_returning_none_gives_empty(self, monkeypatch):
        """A worker that returns None leaves the call returning []."""
        _, out = _exhausted(
            monkeypatch, n_ics=0, nelder_mead=_cheap_nelder_mead, worker=_none_worker, n_opts=1
        )
        assert out == []

    @pytest.mark.parametrize("threaded", [False, True])
    def test_latent_l6_workers_returning_none_give_empty(self, threaded, monkeypatch, caplog):
        """Two attempts whose workers return None leave the call returning [] silently."""
        with caplog.at_level(logging.ERROR, logger="gliquid"):
            _, out = _exhausted(
                monkeypatch,
                n_ics=1,
                nelder_mead=_cheap_nelder_mead,
                worker=_none_worker,
                n_opts=2,
                enable_multi_threading=threaded,
            )
        assert out == []
        assert _errors(caplog) == []

    @pytest.mark.parametrize("threaded", [False, True])
    def test_the_lowest_objective_attempt_is_kept(self, threaded, monkeypatch):
        bl, out = _exhausted(
            monkeypatch,
            n_ics=1,
            nelder_mead=_cheap_nelder_mead,
            worker=_result_worker,
            n_opts=2,
            enable_multi_threading=threaded,
        )
        assert sorted(canon(r["f"]) for r in out) == ["float:10.0", "float:9.0"]
        assert canon(bl.nmpath) == {"ndarray": "float64", "items": [[["float:1.0"]]]}

    def test_latent_l6_zero_candidates_give_empty(self, monkeypatch, caplog):
        """No invariant candidate and a failing H-S partition leave nothing to run: []."""
        with caplog.at_level(logging.ERROR, logger="gliquid"):
            _, out = _exhausted(
                monkeypatch,
                n_ics=0,
                nelder_mead=_failing_nelder_mead,
                worker=_raising_worker,
                n_opts=2,
            )
        assert out == []
        assert _errors(caplog) == EXPECTED["exhausted"]["zero_candidate_errors"]


# -- The miscibility-gap walk (L3, L4) --
_LABEL_AT = {
    "left": ("L1 + L2", 35.0, 800.0),
    "right": ("L1 + L2", 65.0, 800.0),
}

# min at index 2, apex at 3, far-flank crossing in segment (3, 4)
_ADJ_LEFT = [
    [0.0, 1173.0],
    [0.1, 1080.0],
    [0.2, 1000.0],
    [0.35, 1200.0],
    [0.5, 950.0],
    [0.7, 1050.0],
    [1.0, 1173.0],
]
# the mirror: min at index 4, apex at 3, far-flank crossing in segment (2, 3)
_ADJ_RIGHT = [
    [0.0, 1173.0],
    [0.3, 1050.0],
    [0.5, 950.0],
    [0.65, 1200.0],
    [0.8, 1000.0],
    [0.9, 1080.0],
    [1.0, 1173.0],
]
# crossings four segments from the minimum
_FAR_LEFT = [
    [0.0, 1173.0],
    [0.1, 1080.0],
    [0.2, 1000.0],
    [0.3, 1100.0],
    [0.4, 1150.0],
    [0.5, 1200.0],
    [0.6, 1150.0],
    [0.7, 1050.0],
    [0.8, 900.0],
    [1.0, 1173.0],
]
_FAR_RIGHT = [
    [0.0, 1173.0],
    [0.2, 900.0],
    [0.3, 1050.0],
    [0.4, 1150.0],
    [0.5, 1200.0],
    [0.6, 1150.0],
    [0.7, 1100.0],
    [0.8, 1000.0],
    [0.9, 1080.0],
    [1.0, 1173.0],
]
# a vertical segment carries the crossing
_VERTICAL_LEFT = [
    [0.0, 1173.0],
    [0.1, 1080.0],
    [0.2, 1000.0],
    [0.3, 1200.0],
    [0.4, 1150.0],
    [0.4, 900.0],
    [1.0, 1173.0],
]
_VERTICAL_RIGHT = [
    [0.0, 1173.0],
    [0.6, 900.0],
    [0.6, 1150.0],
    [0.7, 1200.0],
    [0.8, 1000.0],
    [0.9, 1080.0],
    [1.0, 1173.0],
]


def mig_walk(liquidus, apex_index, min_index, side, polygons=(), labels=None, more_maxima=()):
    """``_miscibility_gap_records`` on a label record, by default with no two-phase polygon.

    Returns ``(records, maxima, minima, liquidus)`` after the call, all in canon form.
    """
    liq = copy.deepcopy(liquidus)
    maxima = [liq[apex_index]] + [liq[i] for i in more_maxima]
    minima = [liq[min_index]]
    records = mpds._miscibility_gap_records(
        _json(labels if labels is not None else [_LABEL_AT[side]], [_svg(p) for p in polygons]),
        ["A", "B"],
        liq,
        maxima,
        minima,
        [],
        COMP_DATA,
        TEMP_RANGE,
        t_tol=15,
        holes=[],
    )
    return canon({"records": records, "maxima": maxima, "minima": minima, "liquidus": liq})


class TestMigWalk:
    def test_left_walk_finds_the_crossing_next_to_the_apex(self):
        got = mig_walk(_ADJ_LEFT, 3, 2, "left")
        assert got == EXPECTED["mig"]["adjacent_left"]
        assert len(got["records"]) == 1

    def test_latent_l3_right_walk_misses_the_crossing_next_to_the_apex(self):
        """The mirrored right walk never tests the segment next to the apex: no record."""
        got = mig_walk(_ADJ_RIGHT, 3, 4, "right")
        assert got == EXPECTED["mig"]["adjacent_right"]
        assert got["records"] == []

    def test_far_crossing_found_on_the_left(self):
        assert mig_walk(_FAR_LEFT, 5, 2, "left") == EXPECTED["mig"]["far_left"]

    def test_far_crossing_found_on_the_right(self):
        assert mig_walk(_FAR_RIGHT, 4, 7, "right") == EXPECTED["mig"]["far_right"]

    def test_polygons_that_miss_leave_the_feet_to_the_walk_and_a_near_top_sets_tbounds(self):
        """A polygon topping out too far below the apex is passed over; one topping out near
        it that does not bracket it still replaces tbounds, so the record's apex is that
        polygon's top, and the walk supplies the feet."""
        polygons = [
            [[0.2, 1000.0], [0.35, 1150.0], [0.47, 1000.0]],
            [[0.33, 1100.0], [0.34, 1195.0], [0.345, 1150.0]],
        ]
        got = mig_walk(_ADJ_LEFT, 3, 2, "left", polygons=polygons)
        assert got == EXPECTED["mig"]["polygons_miss"]

    def test_a_label_inside_a_recorded_dome_is_skipped(self):
        labels = [("L1 + L2", 50.0, 800.0), ("L1 + L2", 60.0, 800.0)]
        got = mig_walk(_FAR_LEFT, 5, 2, "left", labels=labels, more_maxima=(9,))
        assert got == EXPECTED["mig"]["label_inside_dome"]

    def test_latent_l4_vertical_segment_on_the_left_walk_divides_by_zero(self):
        """A vertical crossing segment on the left walk raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            mig_walk(_VERTICAL_LEFT, 3, 2, "left")

    def test_latent_l4_vertical_segment_on_the_right_walk_divides_by_zero(self):
        """A vertical crossing segment on the right walk raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            mig_walk(_VERTICAL_RIGHT, 3, 4, "right")


# -- The congruent pass of identify_invariant_points on synthetic records --
_DOME_LIQ = [
    [0.0, 1173.0],
    [0.1, 1080.0],
    [0.2, 1000.0],
    [0.3, 1120.0],
    [0.4, 1180.0],
    [0.5, 1200.0],
    [0.6, 1180.0],
    [0.7, 1120.0],
    [0.8, 1000.0],
    [0.9, 1080.0],
    [1.0, 1173.0],
]


def _with_solids(labels, polygons, solids):
    record = _json(labels, [_svg(p) for p in polygons])
    record["shapes"].extend(
        {"kind": kind, "label": name, "nphases": 1, "is_solid": True, "svgpath": _svg(points)}
        for kind, name, points in solids
    )
    return record


def invariant_pass(record, liquidus):
    invariants, low_t, full = mpds.identify_invariant_points(
        record, ["A", "B"], copy.deepcopy(liquidus), COMP_DATA, TEMP_RANGE
    )
    return canon({"invariants": invariants, "low_t_exp_phases": low_t, "full_comp_ss": full})


_PASS_CASES = {
    "syntectic_apex_is_not_congruent": lambda: invariant_pass(
        _with_solids(
            [("L1 + L2", 50.0, 800.0)],
            [[[0.2, 1000.0], [0.5, 1200.0], [0.8, 1000.0]]],
            [("phase", "AB", [[0.5, 700.0], [0.5, 1190.0]])],
        ),
        _DOME_LIQ,
    ),
    "apex_without_label_is_congruent": lambda: invariant_pass(
        _with_solids([], [], [("phase", "AB", [[0.5, 700.0], [0.5, 1190.0]])]), _DOME_LIQ
    ),
    "terminal_endpoint_maximum_is_skipped": lambda: invariant_pass(
        _with_solids(
            [],
            [],
            [("phase", "(A)", [[0.0, 500.0], [0.04, 500.0], [0.04, 1250.0], [0.0, 1300.0]])],
        ),
        [[0.03, 1300.0], [0.2, 1000.0], [0.5, 1100.0], [1.0, 1173.0]],
    ),
    "reference_none_returns_nothing": lambda: invariant_pass(
        {**_json([], []), "reference": None}, _DOME_LIQ
    ),
}


class TestInvariantPassSeams:
    @pytest.mark.parametrize("name", sorted(_PASS_CASES))
    def test_pass_case(self, name):
        assert _PASS_CASES[name]() == EXPECTED["passes"][name]


# -- BinaryLiquid.find_invariant_points, the wrapper over identify_invariant_points --
class TestFindInvariantPointsWrapper:
    """The wrapper passes the instance's inputs and its keywords through, assigns the returned
    lists to ``invariants`` and ``low_t_exp_phases`` and returns them, and records a
    full-composition solid solution on ``full_comp_ss`` with ``init_error`` left unset. The
    values are pinned in ``test_decomposition_pins.py``."""

    def test_the_wrapper_forwards_its_inputs_and_assigns_the_results(self, monkeypatch):
        calls = []
        identify = mpds.identify_invariant_points

        def spy(*args, **kwargs):
            calls.append((args, kwargs, identify(*args, **kwargs)))
            return calls[-1][2]

        monkeypatch.setattr(mpds, "identify_invariant_points", spy)
        for call_kwargs, full_comp_ss in (
            ({"t_tol": 25, "check_full_ss": False}, False),
            ({}, True),
        ):
            bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
            inputs = (
                bl.mpds_json,
                bl.components,
                bl.digitized_liq,
                bl.component_data,
                bl.temp_range,
            )
            invariants, low_t = bl.find_invariant_points(**call_kwargs)
            ((args, kwargs, result),) = calls
            calls.clear()
            assert all(a is b for a, b in zip(args, inputs, strict=True))
            assert kwargs == {
                "verbose": False,
                "t_tol": 15,
                "check_full_ss": True,
                **call_kwargs,
                "liquidus_holes": bl.liquidus_holes,
            }
            assert invariants is result[0] is bl.invariants
            assert low_t is result[1] is bl.low_t_exp_phases
            assert result[2] is full_comp_ss
            assert bl.full_comp_ss is full_comp_ss
            assert not bl.init_error

    def test_no_reference_returns_empty_and_leaves_state_untouched(self, caplog):
        """mpds_json without a reference: warns, returns ([], []), touches NO attributes."""
        bl = object.__new__(BinaryLiquid)  # attribute-free shell; only mpds_json is read
        bl.mpds_json = {"reference": None}
        assert bl.find_invariant_points() == ([], [])
        assert not hasattr(bl, "invariants")
        assert not hasattr(bl, "low_t_exp_phases")
        assert not hasattr(bl, "init_error")
        assert "does not contain any data" in caplog.text  # emitted via logger.warning


# -- flank_bound on several candidates on one side, and at the SEGMENT_TOL window edge --
_LO_SPAN = np.float64(0.05)
_HI_SPAN = np.float64(0.95)
_FLANK_CASES = {
    "left_duplicates": lambda: mpds.flank_bound(
        0.5,
        "left",
        (([0.3, 0.3, 0.1], "invariant"), ([], "minimum"), ([], "endpoint")),
        tol=1e-9,
        default=(0.0, "endpoint"),
    ),
    "right_duplicates": lambda: mpds.flank_bound(
        0.5,
        "right",
        (([0.7, 0.7, 0.9], "invariant"), ([], "minimum"), ([], "endpoint")),
        tol=1e-9,
        default=(1.0, "endpoint"),
    ),
    "window_at_segment_tol_excluded": lambda: (
        mpds.flank_bound(
            0.5,
            "left",
            (([0.5 - mpds.SEGMENT_TOL, 0.5 + mpds.SEGMENT_TOL], None),),
            tol=mpds.SEGMENT_TOL,
            default=(0.0, None),
        )[0],
        mpds.flank_bound(
            0.5,
            "right",
            (([0.5 - mpds.SEGMENT_TOL, 0.5 + mpds.SEGMENT_TOL], None),),
            tol=mpds.SEGMENT_TOL,
            default=(1.0, None),
        )[0],
    ),
    "window_at_twice_segment_tol_included": lambda: (
        mpds.flank_bound(
            0.5,
            "left",
            (([0.5 - 2 * mpds.SEGMENT_TOL, 0.5 + 2 * mpds.SEGMENT_TOL], None),),
            tol=mpds.SEGMENT_TOL,
            default=(0.0, None),
        )[0],
        mpds.flank_bound(
            0.5,
            "right",
            (([0.5 - 2 * mpds.SEGMENT_TOL, 0.5 + 2 * mpds.SEGMENT_TOL], None),),
            tol=mpds.SEGMENT_TOL,
            default=(1.0, None),
        )[0],
    ),
    "window_duplicates_int_float64": lambda: (
        mpds.flank_bound(
            0.5,
            "left",
            (([0, 0, np.float64(0.2), 1, np.float64(0.8)], None),),
            tol=mpds.SEGMENT_TOL,
            default=(0.0, None),
        )[0],
        mpds.flank_bound(
            0.5,
            "right",
            (([0, 0, np.float64(0.2), 1, np.float64(0.8)], None),),
            tol=mpds.SEGMENT_TOL,
            default=(1.0, None),
        )[0],
    ),
}


class TestFlankFunctionSeams:
    @pytest.mark.parametrize("name", sorted(_FLANK_CASES))
    def test_flank_case(self, name):
        assert canon(_FLANK_CASES[name]()) == EXPECTED["flank"][name]

    def test_window_fallback_returns_the_span_objects(self):
        lo, hi = (
            mpds.flank_bound(
                0.5, "left", (([0.5], None),), tol=mpds.SEGMENT_TOL, default=(_LO_SPAN, None)
            )[0],
            mpds.flank_bound(
                0.5, "right", (([0.5], None),), tol=mpds.SEGMENT_TOL, default=(_HI_SPAN, None)
            )[0],
        )
        assert lo is _LO_SPAN
        assert hi is _HI_SPAN


# -- mpds.flank_bound's own contract --
class TestFlankBoundContract:
    @pytest.mark.parametrize("side", ["up", "Left", None])
    def test_an_unknown_side_raises(self, side):
        with pytest.raises(ValueError):
            mpds.flank_bound(0.5, side, (([0.3, 0.7], "r"),), tol=1e-9, default=(0.0, "d"))

    @pytest.mark.parametrize(
        ("side", "expected"), [("left", (0.1, "second")), ("right", (0.9, "second"))]
    )
    def test_the_first_tier_holding_a_candidate_on_the_side_decides(self, side, expected):
        tiers = (([0.5], "first"), ([0.1, 0.9], "second"), ([0.4, 0.6], "third"))
        assert mpds.flank_bound(0.5, side, tiers, tol=1e-9, default=(None, None)) == expected

    @pytest.mark.parametrize(
        ("side", "at_tol", "beyond"),
        [("left", 0.5 - 1e-6, 0.5 - 2e-6), ("right", 0.5 + 1e-6, 0.5 + 2e-6)],
    )
    def test_a_candidate_exactly_tol_away_is_not_on_the_side(self, side, at_tol, beyond):
        default = (None, "none")
        assert mpds.flank_bound(0.5, side, (([at_tol], "r"),), tol=1e-6, default=default) is default
        assert mpds.flank_bound(
            0.5, side, (([at_tol, beyond], "r"),), tol=1e-6, default=default
        ) == (beyond, "r")

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_default_is_returned_as_the_same_object(self, side):
        default = (np.float64(0.05), "endpoint")
        opposite = [0.9] if side == "left" else [0.1]
        assert mpds.flank_bound(0.5, side, (), tol=1e-9, default=default) is default
        tiers = (([], "a"), ([0.5], "b"), (opposite, "c"))
        assert mpds.flank_bound(0.5, side, tiers, tol=1e-9, default=default) is default

    @pytest.mark.parametrize(
        ("side", "candidate"),
        [("left", np.float64(0.3)), ("left", 0), ("right", np.float64(0.7)), ("right", 1)],
        ids=["left-float64", "left-int", "right-float64", "right-int"],
    )
    def test_the_bound_is_the_candidate_element_itself(self, side, candidate):
        bound, rule = mpds.flank_bound(
            0.5, side, (([candidate], "r"),), tol=1e-9, default=(None, None)
        )
        assert bound is candidate
        assert type(bound) is type(candidate)
        assert rule == "r"

    @pytest.mark.parametrize(("side", "value"), [("left", 0.3), ("right", 0.7)])
    def test_the_first_of_equal_extremes_is_returned(self, side, value):
        orders = (
            [float(repr(value)), np.float64(value)],
            [np.float64(value), float(repr(value))],
            [float(repr(value)), float(repr(value))],
        )
        for candidates in orders:
            bound, _ = mpds.flank_bound(
                0.5, side, ((candidates, "r"),), tol=1e-9, default=(None, None)
            )
            assert bound is candidates[0]

    def test_tol_and_default_are_keyword_only_and_required(self):
        with pytest.raises(TypeError):
            mpds.flank_bound(0.5, "left", (), default=(0.0, None))
        with pytest.raises(TypeError):
            mpds.flank_bound(0.5, "left", (), tol=1e-9)
        with pytest.raises(TypeError):
            mpds.flank_bound(0.5, "left", (), 1e-9, (0.0, None))


# -- The flank bounds through the functions that keep calling them --
_INV_COMPS = [0.2, 0.4, 0.7, 0.8]


def _missing_field(phase_comp, invariants, x_lo=0.0, x_hi=1.0):
    return solid_coverage._missing_compound_field(
        {"name": "P", "comp": phase_comp},
        phase_comp,
        invariants,
        x_lo,
        x_hi,
        _INV_COMPS,
        [0.1, 0.9],
        [0.0, 1.0],
    )


_ENTRY_CASES = {
    "missing_per_liquid_below_stoichiometry": lambda: _missing_field(
        0.6, [_inv("per", 0.4, ["P"], [0.6])]
    ),
    "missing_per_liquid_above_stoichiometry": lambda: _missing_field(
        0.6, [_inv("per", 0.7, ["P"], [0.6])]
    ),
    "missing_per_off_span": lambda: _missing_field(
        0.6, [_inv("per", 0.4, ["P"], [0.6])], x_lo=0.65, x_hi=1.0
    ),
    "missing_no_controlling_invariant": lambda: _missing_field(0.6, []),
    "missing_cmp": lambda: _missing_field(0.6, [_inv("cmp", 0.6, ["P"], [0.6])]),
}


class TestFlankThroughEntryPoints:
    @pytest.mark.parametrize("name", sorted(_ENTRY_CASES))
    def test_entry_case(self, name):
        assert canon(_ENTRY_CASES[name]()) == EXPECTED["entry"][name]


# -- plot_tx's closures, on hand-built facet tables --


_INV_KEYS = ("Eutectics", "Peritectics", "Congruent Melting", "Misc Gaps", "Solid Ties")


def _inv_points(eutectics=()):
    points = {key: [] for key in _INV_KEYS}
    points["Eutectics"] = [list(e) for e in eutectics]
    return points


def _line(x, t_lo, t_hi, label, step=25.0):
    temps = list(np.arange(t_lo, t_hi, step)) + [t_hi]
    return [[x, float(t), label] for t in temps]


def _v_liquidus(x_min=0.0, x_max=1.0, step=0.05):
    xs = [round(x, 10) for x in np.arange(x_min, x_max + 1e-9, step)]
    return [[x, 1400.0 - 1200.0 * min(x, 1.0 - x), "L"] for x in xs]


def polymorph_hull(
    names=("Alo", "Ahi", "Blo", "Bhi"),
    compounds=(("AB", 0.5, -273.15, 800.0),),
    liquidus=None,
    eutectics=((800.0, 0.5, [0.25, 0.5, 0.75], ["L", "AB", "L"]),),
):
    """Polymorph lines at both axes, compound lines, and a V-shaped liquidus."""
    a_lo, a_hi, b_lo, b_hi = names
    rows = list(liquidus if liquidus is not None else _v_liquidus())
    rows += _line(0.0, -273.15, 600.0, a_lo) + _line(0.0, 600.0, 1400.0, a_hi)
    rows += _line(1.0, -273.15, 500.0, b_lo) + _line(1.0, 500.0, 1400.0, b_hi)
    for label, x, t_lo, t_hi in compounds:
        rows += _line(x, t_lo, t_hi, label)
    phases = ["A", "B", "L", a_lo, a_hi, b_lo, b_hi] + [c[0] for c in compounds]
    transitions = [
        {"name": a_hi, "comp_x_pct": 0, "transition_temp_C": 600.0, "ground_state_name": a_lo},
        {"name": b_hi, "comp_x_pct": 100, "transition_temp_C": 500.0, "ground_state_name": b_lo},
    ]
    return _FakeHsx(rows, phases, _inv_points(eutectics)), transitions


def figure_capture(fig, hsx, ties):
    annotations = [
        [a.text, a.x, a.y, a.textangle, a.showarrow, a.ax, a.ay] for a in fig.layout.annotations
    ]
    return canon({"annotations": annotations, "conds": list(hsx.conds), "ties": ties})


def _pack_mode(mode):
    """A ``_pack_labels`` stand-in.

    ``real`` passes through; ``never`` places nothing; ``no_inband`` refuses the calls
    that carry a ``blocked`` list and passes the others through; ``horizontal_only``
    passes through only a ``blocked``-list call that directly follows a call without one.
    """
    real = binary_tx._pack_labels
    state = {"after_unblocked": False}

    def fake(regions, half_hs, ceiling, tie_temps, floor, margin, gap=0.0, blocked=None):
        args = (regions, half_hs, ceiling, tie_temps, floor, margin, gap, blocked)
        if mode == "real":
            return real(*args)
        if mode == "never":
            return [None] * len(regions)
        if mode == "no_inband":
            return [None] * len(regions) if blocked is not None else real(*args)
        if mode == "horizontal_only":
            if blocked is None:
                state["after_unblocked"] = True
                return [None] * len(regions)
            horizontal = state["after_unblocked"]
            state["after_unblocked"] = False
            return real(*args) if horizontal else [None] * len(regions)
        raise ValueError(mode)

    return fake


def render(monkeypatch, hsx, *, pack="real", spy=None, **plot_kwargs):
    fake = _pack_mode(pack)
    if spy is not None:
        inner = fake

        def fake(regions, half_hs, ceiling, tie_temps, floor, margin, gap=0.0, blocked=None):
            out = inner(regions, half_hs, ceiling, tie_temps, floor, margin, gap, blocked)
            spy.append([len(regions), ceiling])
            return out

    monkeypatch.setattr(binary_tx, "_pack_labels", fake)
    with record_tie_lines() as ties:
        fig = plot_tx(hsx, **plot_kwargs)
    return fig, ties


def _tie_hull(eutectics):
    rows = _v_liquidus() + _line(0.0, -273.15, 1400.0, "A") + _line(1.0, -273.15, 1400.0, "B")
    return _FakeHsx(rows, ["A", "B", "L"], _inv_points(eutectics))


def _eut(temp, lo, hi):
    mid = 0.5 * (lo + hi)
    return [temp, mid, [lo, mid, hi], ["A", "L", "B"]]


# The plotted span of the tie hull while its ties are drawn: its own conds, which the
# liquidus padding leaves unchanged.
_TIE_STAGE_SPAN = 1500.0


def _tie_case(eutectics):
    def run(monkeypatch):
        hsx = _tie_hull(eutectics)
        fig, ties = render(monkeypatch, hsx)
        silver = [
            [list(tr.x), list(tr.y)]
            for tr in fig.data
            if getattr(getattr(tr, "line", None), "color", None) == "Silver"
        ]
        return canon({"ties": ties, "traces": silver})

    return run


def _hull_case(pack="real", *, spy=False, **hull_kwargs):
    def run(monkeypatch):
        hsx, transitions = polymorph_hull(**hull_kwargs)
        calls = [] if spy else None
        fig, ties = render(
            monkeypatch,
            hsx,
            pack=pack,
            spy=calls,
            polymorph_transitions=transitions,
            components=["A", "B"],
        )
        out = figure_capture(fig, hsx, ties)
        if spy:
            out["pack_calls"] = canon(calls)
        return out

    return run


_LONG = "A" + "x" * 70
_LONGER = "C" + "y" * 120
_TOL_PROBE = 900.0
_PLOT_CASES = {
    "tie_touching_spans_merge": _tie_case([_eut(900.0, 0.02, 0.20), _eut(900.0, 0.20, 0.50)]),
    "tie_overlapping_spans_merge": _tie_case([_eut(900.0, 0.10, 0.60), _eut(900.0, 0.12, 0.70)]),
    "tie_disjoint_spans_stay_apart": _tie_case([_eut(900.0, 0.02, 0.20), _eut(900.0, 0.50, 0.70)]),
    "tie_contained_span_keeps_trace": _tie_case([_eut(900.0, 0.10, 0.80), _eut(900.0, 0.30, 0.50)]),
    "tie_three_way_chain": _tie_case(
        [_eut(900.0, 0.10, 0.30), _eut(900.0, 0.50, 0.70), _eut(900.0, 0.30, 0.50)]
    ),
    "tie_just_inside_tolerance": "tolerance:0.99",
    "tie_just_outside_tolerance": "tolerance:1.01",
    "layout_real": _hull_case("real", spy=True),
    "layout_never_places": _hull_case("never", spy=True),
    "layout_no_inband": _hull_case("no_inband"),
    "layout_horizontal_only": _hull_case("horizontal_only"),
    "layout_long_names_never_places": _hull_case(
        "never",
        names=(_LONG + "lo", _LONG + "hi", "Blo", "Bhi"),
        compounds=((_LONGER, 0.5, -273.15, 800.0),),
    ),
    "layout_extra_compounds": _hull_case(
        "real",
        compounds=(
            ("AB", 0.5, -273.15, 800.0),
            ("AC", 0.08, -273.15, 900.0),
            ("AD", 0.03, -273.15, 850.0),
            ("AE", 0.6, -273.15, -100.0),
        ),
    ),
    "layout_stacked_compound_real": _hull_case(
        "real", compounds=(("ABlo", 0.5, -273.15, 500.0), ("ABhi", 0.5, 500.0, 800.0))
    ),
    "layout_stacked_compound_never": _hull_case(
        "never", compounds=(("ABlo", 0.5, -273.15, 500.0), ("ABhi", 0.5, 500.0, 800.0))
    ),
    "layout_dense_ties_over_compound": _hull_case(
        "real",
        eutectics=[(t, 0.5, [0.45, 0.5, 0.55], ["L", "AB", "L"]) for t in range(-260, 800, 40)],
    ),
    "env_liquidus_short_of_the_a_axis": _hull_case(
        "real", spy=True, liquidus=_v_liquidus(x_min=0.05)
    ),
    "env_liquidus_partly_short_of_the_a_axis": _hull_case(
        "real",
        spy=True,
        liquidus=[[0.02, 1380.0, "L"], [0.025, 1370.0, "L"]] + _v_liquidus(x_min=0.05),
    ),
    "env_v_notch_under_the_a_label": _hull_case(
        "real",
        spy=True,
        liquidus=[[0.0, 1400.0, "L"], [0.015, 700.0, "L"], [0.03, 1400.0, "L"]]
        + _v_liquidus(x_min=0.05),
    ),
}


def _eutectoid_rows(liquidus_t, left_compound=True, right_compound=True):
    """An SS band from 10 to 90 at.% whose floor is a V bottoming out at 600 C at 50 at.%,
    optional line compounds at 5 and 95 at.%, a ``LOW`` facet below the minimum and an
    ``SS`` facet above it, under a liquidus at ``liquidus_t(x)``."""
    rows = []
    for x in range(0, 101, 5):
        rows += _facet(liquidus_t(x), (x, "L"), (x, "L"), (x, "L"))
    for t in (0.0, 500.0, 1000.0):
        if left_compound:
            rows += _facet(t, (5, "CA"), (5, "CA"), (5, "CA"))
        if right_compound:
            rows += _facet(t, (95, "CB"), (95, "CB"), (95, "CB"))
    for x in range(10, 91):
        rows += _facet(600.0 + 5.0 * abs(x - 50), (x, "SS"), (x, "SS"), (x, "SS"))
        rows += _facet(900.0, (x, "SS"), (x, "SS"), (x, "SS"))
    rows += _facet(300.0, (40, "LOW"), (50, "LOW"), (60, "LOW"))
    rows += _facet(650.0, (50, "SS"), (52, "SS"), (55, "SS"))
    return rows


def _ss_case(liquidus_t, **kwargs):
    def run(monkeypatch):
        hsx = _FakeHsx(
            _eutectoid_rows(liquidus_t, **kwargs),
            ["A", "B", "L", "CA", "SS", "CB", "LOW"],
            _inv_points(),
        )
        fig, ties = render(monkeypatch, hsx, ss_phases={"SS"})
        return figure_capture(fig, hsx, ties)

    return run


def _flat_liquidus(x):
    return 1200.0


def _dropped_liquidus(x):
    return 1200.0 if x <= 40 else 500.0


_PLOT_CASES.update(
    {
        "ss_minimum_bracketed": _ss_case(_flat_liquidus),
        "ss_minimum_axis_fallback": _ss_case(_flat_liquidus, left_compound=False),
        "ss_minimum_unbracketed": _ss_case(_dropped_liquidus, right_compound=False),
    }
)


def run_plot_case(monkeypatch, name):
    case = _PLOT_CASES[name]
    if isinstance(case, str):
        factor = float(case.split(":")[1])
        tol = binary_tx._TIE_MERGE_T_FRAC * _TIE_STAGE_SPAN
        case = _tie_case(
            [_eut(_TOL_PROBE, 0.10, 0.60), _eut(_TOL_PROBE + factor * tol, 0.12, 0.70)]
        )
    return case(monkeypatch)


class TestPlotTxClosureSeams:
    @pytest.mark.parametrize("name", sorted(_PLOT_CASES))
    def test_plot_case(self, name, monkeypatch):
        assert run_plot_case(monkeypatch, name) == EXPECTED["plot"][name]
