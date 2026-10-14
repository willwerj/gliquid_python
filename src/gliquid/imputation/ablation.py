"""
Remove interior DFT phases from a binary system's hull and put them back.

:func:`ablate_phases` removes one or more interior fixed-composition phases and resets the
fit state for a clean re-fit; :func:`restore_phases` is its exact inverse. Both edit
``bl.phases`` and rebuild ``eqs['h_hull_interp']`` through
``BinaryLiquid.rebuild_hull_interp``. :func:`list_ablatable_phases` lists the phases that may
be removed: pure-element endpoints (x 0.0 / 1.0) and solution phases never are.
"""

from __future__ import annotations

import copy

from gliquid.binary import BinaryLiquid
from gliquid.phase import Phase

ENDPOINT_COMPS = (0.0, 1.0)


# -- Ablation primitive --


def _phase_x(bl: BinaryLiquid, phase: Phase) -> float:
    """The phase's binary evaluation-axis fraction in ``bl``'s component order."""
    return bl._phase_x(phase)


def list_ablatable_phases(bl: BinaryLiquid) -> list[Phase]:
    """
    Return the interior fixed-composition phases that may be ablated.

    Pure-element polymorph endpoints (x 0.0 / 1.0) anchor the hull and the liquid
    endpoints and are never ablatable; solution phases (the liquid and any solid
    solution, both ``composition is None``) are excluded too.

    Args:
        bl (BinaryLiquid): System whose phases to inspect.

    Returns:
        list[Phase]: References into ``bl.phases`` (read-only use intended).
    """
    return [
        p
        for p in bl.phases
        if p.composition is not None and round(float(_phase_x(bl, p)), 6) not in ENDPOINT_COMPS
    ]


def _resolve_targets(bl: BinaryLiquid, targets, tol: float = 0.02) -> list[Phase]:
    """
    Resolve target specifiers to concrete ablatable phases in ``bl.phases``.

    Args:
        bl (BinaryLiquid): System to resolve against.
        targets: A single target or an iterable of targets, each a phase name (str), a
            B-mole-fraction composition (float, matched to the nearest ablatable phase
            within ``tol``), or a ``Phase`` (matched by name).
        tol (float): Composition match tolerance for float targets.

    Returns:
        list[Phase]: The matched ablatable phases (object identity preserved).

    Raises:
        ValueError: If a target is an endpoint/solution phase or matches nothing.
    """
    if isinstance(targets, (str, float, int, Phase)):
        targets = [targets]

    ablatable = list_ablatable_phases(bl)
    resolved: list[Phase] = []
    for target in targets:
        name = target.name if isinstance(target, Phase) else target
        if isinstance(name, str):
            matches = [p for p in ablatable if p.name == name]
            if not matches:
                raise ValueError(
                    f"Target phase '{name}' is not an ablatable interior phase of "
                    f"{bl.sys_name}. Ablatable: {[p.name for p in ablatable]}"
                )
            resolved.append(matches[0])
        elif isinstance(name, (float, int)) and not isinstance(name, bool):
            comp = round(float(name), 6)
            if comp in ENDPOINT_COMPS:
                raise ValueError(f"Endpoint composition {comp} is not ablatable.")
            if not ablatable:
                raise ValueError(f"{bl.sys_name} has no ablatable interior phases.")
            nearest = min(ablatable, key=lambda p: abs(_phase_x(bl, p) - comp))
            if abs(_phase_x(bl, nearest) - comp) > tol:
                raise ValueError(
                    f"No ablatable phase within tol={tol} of composition {comp} in "
                    f"{bl.sys_name} (nearest at {_phase_x(bl, nearest)})."
                )
            resolved.append(nearest)
        else:
            raise ValueError(
                f"Target must be a phase name (str), composition (float) or Phase; got {target!r}"
            )
    return resolved


def ablate_phases(
    bl: BinaryLiquid, targets, in_place: bool = False, tol: float = 0.02
) -> BinaryLiquid:
    """
    Remove one or more interior DFT phases and reset all DFT-dependent state for a
    clean re-fit.

    The exact inverse of :func:`restore_phases`: both edit ``bl.phases`` and then rebuild
    ``eqs['h_hull_interp']`` through ``BinaryLiquid.rebuild_hull_interp``.

    Args:
        bl (BinaryLiquid): Source system (a fitted full-hull baseline is typical).
        targets: Phase name(s) (str), composition(s) (float) or ``Phase`` object(s) to
            remove. See ``_resolve_targets``.
        in_place (bool): If False (default), operate on a deepcopy so ``bl`` is left
            untouched. If True, mutate ``bl`` directly.
        tol (float): Composition match tolerance for float targets.

    Returns:
        BinaryLiquid: The ablated system (the deepcopy when ``in_place`` is False).

    Notes:
        ``bl.dft_ch`` (the original pymatgen hull) is intentionally left untouched;
        downstream code reasons about the present phases via ``bl.phases``.
    """
    work = bl if in_place else copy.deepcopy(bl)

    to_remove = _resolve_targets(work, targets, tol=tol)
    remove_ids = {id(p) for p in to_remove}
    work.phases = [p for p in work.phases if id(p) not in remove_ids]

    work.rebuild_hull_interp()
    work.reset_fit_state()
    return work


def restore_phases(bl: BinaryLiquid, phases, in_place: bool = False) -> BinaryLiquid:
    """
    Insert fixed-composition phases back into the hull and rebuild it.

    The inverse of :func:`ablate_phases`: a phase is placed in composition order among the
    fixed-composition phases, the liquid (and any solution phase) stays last, and
    ``eqs['h_hull_interp']`` is rebuilt through the same helper the removal used.

    Args:
        bl (BinaryLiquid): System to insert into.
        phases: A ``Phase`` or iterable of ``Phase`` objects with a fixed composition.
        in_place (bool): If False (default), operate on a deepcopy AND insert copies of
            ``phases``, so nothing the caller holds is aliased into the result. True inserts
            the caller's objects themselves, and the state reset clears their ``points``.

    Returns:
        BinaryLiquid: The system carrying the restored phases.

    Raises:
        ValueError: If a phase carries no composition (a solution phase has no hull vertex).
    """
    work = bl if in_place else copy.deepcopy(bl)
    if isinstance(phases, Phase):
        phases = [phases]

    for source_phase in phases:
        if source_phase.composition is None:
            raise ValueError(
                f"Cannot restore solution phase '{source_phase.name}' as a hull vertex; it "
                f"has no fixed composition."
            )
        phase = source_phase if in_place else copy.deepcopy(source_phase)
        target_x = _phase_x(work, phase)
        insert_at = len(work.phases)
        for index, existing in enumerate(work.phases):
            if existing.composition is None or _phase_x(work, existing) > target_x:
                insert_at = index
                break
        work.phases.insert(insert_at, phase)

    work.rebuild_hull_interp()
    work.reset_fit_state()
    return work
