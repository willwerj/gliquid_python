"""Imputing, judging and removing imputed phases, for binary and ternary systems.

:func:`impute_phases`, :func:`drop_imputed_phases`, :func:`imputed_entry_staleness` and
:func:`stale_imputed_entries` take a :class:`~gliquid.binary.BinaryLiquid` or a
:class:`~gliquid.ternary.TernaryLiquidInterpolation` and call the implementation for its
dimension. :data:`ImputationResult` is the record type either dimension's imputation returns.
"""

from __future__ import annotations

import copy
from typing import TypeAlias

from pymatgen.analysis.phase_diagram import PhaseDiagram

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid
from gliquid.cache import IMPUTED_ID_PREFIX, imputed_entry_label, is_imputed_entry_dict
from gliquid.imputation.binary_imputation import BinaryImputationResult, impute_binary_phases
from gliquid.imputation.core import PARAM_RTOL, PROVENANCE_KEY, params_moved
from gliquid.imputation.ternary_imputation import (
    CALIBRATIONS,
    DFT_TYPE,
    TernaryImputationResult,
    impute_ternary_phases,
)

#: The result of imputing one phase, binary or ternary.
ImputationResult: TypeAlias = BinaryImputationResult | TernaryImputationResult

# -- Lifecycle: staleness and removal --

_ANCHOR_COMP_TOL = 1e-6
_ANCHOR_TEMP_TOL = 1e-6


def _binary_imputed_entry_staleness(bl: BinaryLiquid, entry_dict: dict) -> str | None:
    """Why a cached imputed entry no longer describes ``bl``, or ``None`` if it still does.

    An imputed enthalpy is a function of the controlling invariant and the fitted liquid
    parameters (see :func:`gliquid.imputation.binary_imputation._binary_provenance`), so it is stale
    exactly when either has moved -- a re-digitized diagram that shifts an invariant, or a
    refit of the liquid.

    Requires ``bl.invariants`` / ``bl.low_t_exp_phases`` to be populated (they are after
    ``find_invariant_points`` or a fit); a system with no invariants reads every anchored
    entry as stale.

    Args:
        bl: the system the entry is cached against.
        entry_dict: a tagged ``ComputedEntry.as_dict()`` payload.

    Returns:
        str | None: a short reason, or ``None`` when the entry is current.
    """
    data = entry_dict.get("data") or {}
    prov = data.get(PROVENANCE_KEY)
    if not isinstance(prov, dict):
        return "no provenance recorded"

    if prov.get("param_format") != bl.param_format:
        return f"param_format {prov.get('param_format')!r} != {bl.param_format!r}"

    moved = params_moved(prov.get("fit_params"), [float(v) for v in bl.get_params()], PARAM_RTOL)
    if moved is None:
        return "fit parameters not comparable"
    if moved:
        return "fit parameters have moved"

    anchor = prov.get("anchor")
    if anchor is None:
        return "no anchor recorded"
    name = data.get("label")
    phase = next(
        (p for p in (bl.low_t_exp_phases or []) if p.get("name") == name),
        None,
    )
    if phase is None:
        return f"{name!r} is no longer a low-temperature phase"
    current_inv = mpds.controlling_invariant(phase, bl.invariants, allow_eutectic=True)
    if current_inv is None:
        return f"{name!r} no longer has a controlling invariant"
    if (
        current_inv["type"] != anchor.get("type")
        or abs(float(current_inv["comp"]) - float(anchor.get("comp", float("nan"))))
        > _ANCHOR_COMP_TOL
        or abs(float(current_inv["temp"]) - float(anchor.get("temp", float("nan"))))
        > _ANCHOR_TEMP_TOL
    ):
        return (
            f"anchor moved {anchor.get('type')}@{anchor.get('comp')} -> "
            f"{current_inv['type']}@{float(current_inv['comp']):.6f}"
        )
    return None


def _drop_binary_imputed_phases(bl: BinaryLiquid, in_place: bool = False) -> BinaryLiquid:
    """Remove every imputed phase from ``bl``, the in-memory inverse of :func:`impute_phases`.

    The system returns to its un-imputed state without a reload. Both halves of what
    ``impute_phases`` changed are undone: the synthetic vertices leave ``dft_ch`` as well as
    ``phases``, so a later ``build_phases_from_chull`` cannot resurrect them.

    Fit state is reset (through ``ablate_phases``); ``unsupported_comp_ranges`` is recomputed
    by the next fit from the reduced hull, not restored.

    Args:
        bl: system to strip.
        in_place: mutate ``bl`` when True, else work on a deepcopy.

    Returns:
        BinaryLiquid: the system with no imputed phases (unchanged if it had none).
    """
    import gliquid.imputation.ablation as imp_ablation

    work = bl if in_place else copy.deepcopy(bl)
    targets = [p for p in work.phases if getattr(p, "imputed", False)]
    if not targets:
        return work

    kept = [e for e in work.dft_ch.all_entries if not is_imputed_entry_dict(e.as_dict())]
    work.dft_ch = PhaseDiagram(kept, elements=list(work.dft_ch.elements))
    return imp_ablation.ablate_phases(work, targets, in_place=True)


def _drop_ternary_imputed_phases(tli, names=None, *, write_cache: bool | None = None) -> list[str]:
    """Remove imputed phases from ``tli`` (and from its store when ``write_cache``).

    Args:
        tli: the ternary object; its derived state is reset when anything is removed.
        names: phase names to remove; ``None`` removes every explicit imputed phase.
        write_cache: also remove them from ``tli.data_dir``. ``None`` uses
            :data:`gliquid.config.persist_ternary_imputed`.

    Returns:
        list[str]: the names removed from ``tli``.
    """
    wanted = None if names is None else {str(n) for n in names}
    removed = [
        imputed_entry_label(e)
        for e in tli.imputed_entries
        if wanted is None or imputed_entry_label(e) in wanted
    ]
    if removed:
        tli.imputed_entries = [
            e for e in tli.imputed_entries if imputed_entry_label(e) not in removed
        ]
        tli.reset_derived_state()
    write = bool(config.persist_ternary_imputed) if write_cache is None else bool(write_cache)
    if write:
        ids = None if wanted is None else [f"{IMPUTED_ID_PREFIX}{n}" for n in sorted(wanted)]
        api.drop_imputed_entries(
            list(tli.components), DFT_TYPE, entry_ids=ids, data_dir=tli.data_dir
        )
    return removed


def _ternary_imputed_entry_staleness(tli, entry_dict: dict) -> str | None:
    """Why an imputed ternary entry no longer describes ``tli``'s liquid, or ``None``.

    The energy is a function of the liquid model it was read off (recorded under
    ``provenance['liquid']``); it is stale when any of that has moved.

    Args:
        tli: the ternary object.
        entry_dict: a tagged entry written by :func:`impute_ternary_phases`.

    Returns:
        str | None: a short reason, or ``None`` when the entry is current.
    """
    provenance = (entry_dict.get("data") or {}).get(PROVENANCE_KEY)
    if (
        not isinstance(provenance, dict)
        or provenance.get("calibration") not in CALIBRATIONS.values()
    ):
        return "no ternary provenance recorded"
    recorded, current = provenance.get("liquid") or {}, tli.imputation_params()
    for key in ("param_format", "interp_scheme", "solid_solutions", "liquid_gibbs"):
        if recorded.get(key) != current[key]:
            return f"{key} {recorded.get(key)!r} != {current[key]!r}"
    for key in ("tau", "delta"):
        if recorded.get(key) is None:
            return f"{key} not recorded"
        if params_moved([recorded[key]], [current[key]], PARAM_RTOL):
            return f"{key} has moved"
    if set(recorded.get("edges") or {}) != set(current["edges"]):
        return "edge set differs"
    for edge, values in current["edges"].items():
        moved = params_moved(recorded["edges"][edge], values, PARAM_RTOL)
        if moved is None:
            return f"{edge} parameters not comparable"
        if moved:
            return f"{edge} parameters have moved"
    if (recorded.get("triplet") is None) != (current["triplet"] is None):
        return "ternary interaction term added or removed"
    if current["triplet"] is not None and params_moved(
        recorded["triplet"], current["triplet"], PARAM_RTOL
    ):
        return "ternary interaction term has moved"
    if provenance.get("anchor") is None:
        return "no anchor recorded"
    return None


# -- Dispatch on the model's dimension --


def _dimension(model) -> str:
    """The dimension whose imputation implementation serves ``model``.

    Args:
        model: a :class:`~gliquid.binary.BinaryLiquid` or a
            :class:`~gliquid.ternary.TernaryLiquidInterpolation`.

    Returns:
        str: ``'binary'`` or ``'ternary'``.

    Raises:
        TypeError: ``model`` is neither.
    """
    if isinstance(model, BinaryLiquid):
        return "binary"
    from gliquid.ternary import TernaryLiquidInterpolation

    if isinstance(model, TernaryLiquidInterpolation):
        return "ternary"
    raise TypeError(
        f"expected a BinaryLiquid or a TernaryLiquidInterpolation, got {type(model).__name__}"
    )


def impute_phases(model, targets=None, **kw) -> list[ImputationResult]:
    """Impute solid phases into ``model`` and return one record per imputed phase or target.

    Binary (a :class:`~gliquid.binary.BinaryLiquid`): the targets are the phases
    :func:`~gliquid.imputation.binary_imputation.list_binary_imputation_targets` detects, so
    ``targets`` must be ``None``. The keywords are those of
    :func:`~gliquid.imputation.binary_imputation.impute_binary_phases`: ``fitting_data``,
    ``write_cache``, ``unmask``, ``verbose``, ``protect_dft_hull``, ``t_matched_types`` and
    ``inversion_selection``. Returns ``list[BinaryImputationResult]``,
    one per phase inserted.

    Ternary (a :class:`~gliquid.ternary.TernaryLiquidInterpolation`): ``targets`` is required,
    in the form :func:`~gliquid.imputation.ternary_imputation.impute_ternary_phases` takes. The
    keywords are ``force``, ``write_cache``, ``apply``, ``screen``, ``margin_j`` and ``dx``.
    Returns ``list[TernaryImputationResult]``, one record per target in the order given.

    Args:
        model: the system to impute into; mutated as its implementation documents.
        targets: the ternary target dicts; ``None`` for a binary system.
        **kw: the implementation's keywords.

    Returns:
        list[ImputationResult]: the records.

    Raises:
        TypeError: ``model`` is neither dimension, ``targets`` is given for a binary system, or
            ``targets`` is missing for a ternary one.
    """
    if _dimension(model) == "binary":
        if targets is not None:
            raise TypeError(
                "a binary system's targets are detected from its MPDS record; pass candidate "
                "fits as fitting_data=, not as targets"
            )
        return impute_binary_phases(model, **kw)
    if targets is None:
        raise TypeError("a ternary system needs targets: dicts with composition and temperature")
    return impute_ternary_phases(model, targets, **kw)


def drop_imputed_phases(model, **kw) -> BinaryLiquid | list[str]:
    """Remove imputed phases from ``model``.

    Binary: every imputed phase leaves the phase list and the hull, and the fit state is reset.
    The keyword is ``in_place`` (default ``False``: work on a deep copy). Returns the
    :class:`~gliquid.binary.BinaryLiquid` with no imputed phase.

    Ternary: the keywords are ``names`` (default ``None``: every explicit imputed phase) and
    ``write_cache`` (also remove them from the store; ``None`` uses
    :data:`gliquid.config.persist_ternary_imputed`). Returns the names removed.

    :func:`gliquid.api.drop_imputed_entries` removes imputed entries from a store instead.

    Args:
        model: a :class:`~gliquid.binary.BinaryLiquid` or a
            :class:`~gliquid.ternary.TernaryLiquidInterpolation`.
        **kw: the implementation's keywords.

    Returns:
        BinaryLiquid | list[str]: per dimension, as above.

    Raises:
        TypeError: ``model`` is neither dimension.
    """
    if _dimension(model) == "binary":
        return _drop_binary_imputed_phases(model, **kw)
    return _drop_ternary_imputed_phases(model, **kw)


def imputed_entry_staleness(model, entry_dict: dict) -> str | None:
    """Why a cached imputed entry no longer describes ``model``, or ``None`` if it still does.

    Binary: the entry is stale when its anchor invariant or the fitted liquid parameters have
    moved; ``model.invariants`` / ``model.low_t_exp_phases`` must be populated. Ternary: the
    entry is stale when the liquid model it was read off has moved; the hull is not read.

    Args:
        model: a :class:`~gliquid.binary.BinaryLiquid` or a
            :class:`~gliquid.ternary.TernaryLiquidInterpolation`.
        entry_dict: a tagged ``ComputedEntry.as_dict()`` payload.

    Returns:
        str | None: a short reason, or ``None`` when the entry is current.

    Raises:
        TypeError: ``model`` is neither dimension.
    """
    if _dimension(model) == "binary":
        return _binary_imputed_entry_staleness(model, entry_dict)
    return _ternary_imputed_entry_staleness(model, entry_dict)


def stale_imputed_entries(model, entry_dicts) -> list[tuple[dict, str]]:
    """Every entry in ``entry_dicts`` that no longer describes ``model``, with its reason.

    Non-imputed entries are ignored rather than reported. For a ternary model the reason comes from
    the recorded liquid model only; the hull is not read.

    Args:
        model: a :class:`~gliquid.binary.BinaryLiquid` or a
            :class:`~gliquid.ternary.TernaryLiquidInterpolation`.
        entry_dicts: ``ComputedEntry.as_dict()`` payloads, imputed or not.

    Returns:
        list[tuple[dict, str]]: ``(entry, reason)`` for each stale imputed entry, in input
        order.

    Raises:
        TypeError: an imputed entry is given and ``model`` is neither dimension.
    """
    out = []
    for entry in entry_dicts or []:
        if not is_imputed_entry_dict(entry):
            continue
        reason = imputed_entry_staleness(model, entry)
        if reason is not None:
            out.append((entry, reason))
    return out
