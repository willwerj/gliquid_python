"""Phase-energy imputation for the gliquid binary liquid model.

Inverse of :mod:`gliquid.imputation.ablation`. For a low-temperature solid phase that the MPDS
diagram shows bordering the liquid but the DFT convex hull lacks, the formation enthalpy is
the value at which the fitted liquid free-energy curve reproduces the phase's controlling
invariant (congruent, peritectic or eutectic point). The phase is inserted as a tagged
synthetic hull vertex (entry_id ``"imputed:..."``, ``data={'imputed': True}``) at the
composition of the maximum-temperature point of its MPDS phase, normalized to one atom.

Conventions:
  * Invariants come from ``find_invariant_points``: each dict has
    ``type``/``comp``/``temp``/``phases``/``phase_comps``. ``cmp`` and ``per`` carry
    ``phases == [name]`` and ``phase_comps == [x_s]``; ``eut`` carries both flanks.
  * Low-temperature stability is ``mpds.is_low_temperature_phase``, via ``bl.low_t_exp_phases``.
  * The liquid Gibbs energy comes from ``eqs['h_liq_lambdified']`` /
    ``eqs['s_liq_lambdified']`` in the frame of the solid hull enthalpies. With ``S = 0`` for
    compounds, the invariant tangent value is the imputed enthalpy.
  * ``bl.phases`` holds ``gliquid.phase.Phase`` objects; a solution phase has
    ``composition is None``.

Imputed entries do not reach a fit: ``api.get_dft_convexhull`` and
``BinaryLiquid.from_cache`` default to ``include_imputed=False``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.mpds as mpds
import gliquid.solid_coverage as solid_coverage
from gliquid.binary import (
    BinaryLiquid,
    build_phases_from_chull,
    hull_field_disagreement,
)
from gliquid.cache import imputed_entry_label
from gliquid.imputation.core import (
    CLAMP_BISECTION_STEPS,
    CLAMP_KEY,  # noqa: F401 - re-exported
    CLAMP_TOLERANCE_J,
    HULL_GUARD_MARGIN_J,
    PROTECTED_VERTEX_MARGIN_J,
    PROVENANCE_KEY,
    HullOps,
    accept_growing,
    augmented_hull,
    clamp_bisect,
    entry_at_enthalpy,
    hull_ref_energy_per_atom,
    hull_with_reanchored_entries,
    protected_clearance,
    provenance_header,
    reanchor_imputed_entries,  # noqa: F401 - re-exported
    tagged_entry_dict,
    tangent_value,
)
from gliquid.mpds import FORMULA_TOL, formula_comp
from gliquid.phase import EV_ATOM_TO_J_MOL, validate_and_format_system
from gliquid.solution import SS_SPACEGROUPS

logger = logging.getLogger(__name__)

# -- Constants --

CALIBRATION = "invariant"
"""The one calibration target. Recorded on every entry written from 0.2.1 onward so a
cache holding entries from several versions stays self-describing; entries written
before it carry no provenance block at all and read as unknown, not as current."""

T_MATCHED_ANCHOR_TYPES: tuple[str, ...] = ()
"""Anchor types whose inversion takes the two-point common tangent rather than the liquid's
differentiated slope. ``cmp`` never needs it -- the construction reduces to the closed form
there -- so only ``per`` and ``eut`` are meaningful. Pass ``t_matched_types`` to
:func:`invert_binary_phase_energy` to override per call."""

T_MATCHED_SELECTABLE_TYPES: tuple[str, ...] = ("per",)
"""Anchor types :func:`nearest_arm_choices` may switch to the two-point common tangent.

``cmp`` is excluded because the construction reduces to the closed form there, so the two arms
are the same expression. ``eut`` is excluded because the selection rule was characterised over
peritectic anchors alone; widening it here would ship a behaviour no measurement covers."""

INVERSION_SELECTIONS = ("shipped", "nearest")
DEFAULT_INVERSION_SELECTION = "shipped"
"""How :func:`impute_binary_phases` picks between the two inversions. ``shipped`` applies
``t_matched_types`` uniformly; ``nearest`` decides per phase, see
:func:`nearest_arm_choices`."""

ARM_KEY = "inversion_arm"


@dataclasses.dataclass
class BinaryImputationResult:
    """The result of imputing one binary solid phase from the experimental liquidus.

    The phase itself joins ``bl.phases`` as a :class:`gliquid.phase.Phase`, built from
    ``entry_dict`` when the hull is rebuilt.
    """

    name: str  # display label (the reference MPDS phase name)
    comp: float  # B-mole fraction (max-T point of the reference phase)
    enthalpy: float  # imputed formation enthalpy, J/mol-atom
    invariant_type: str  # 'cmp' | 'per' | 'eut' (the anchor used for inversion)
    invariant_comp: float  # liquid composition the tangent is anchored at
    invariant_temp: float  # K
    entry_dict: dict  # tagged ComputedEntry.as_dict()


# -- Fit selection --


def select_best_fit(bl: BinaryLiquid, fitting_data: list[dict]) -> dict | None:
    """Set ``bl`` to the lowest-objective candidate -- the one ``fit_parameters`` itself keeps.

    Args:
        bl: System whose parameters to set (mutated in place).
        fitting_data: The list returned by ``BinaryLiquid.fit_parameters``, in any order.

    Returns:
        The chosen fit dict, or ``None`` if ``fitting_data`` is empty.
    """
    if not fitting_data:
        return None
    best = min(fitting_data, key=lambda fit: fit["f"])
    bl.update_params([best[name] for name in bl.xs_mix.format.param_names])
    bl.nmpath = best.get("nmpath")
    return best


# -- Liquid Gibbs evaluation + invariant inversion --


def _liquid_gibbs(bl: BinaryLiquid, x, temp: float):
    """Molar liquid Gibbs energy ``G_liq(x, temp)`` (J/mol-atom), DFT-T=0K referenced.

    Mirrors the H/S evaluation in ``BinaryLiquid.to_hsx`` (``h_liq_lambdified`` /
    ``s_liq_lambdified``) and forms ``G = H - T S`` at the requested temperature.
    """
    params = bl.get_params()
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    h = np.asarray(bl.eqs["h_liq_lambdified"](xa, temp, *params)).flatten()
    s = np.asarray(bl.eqs["s_liq_lambdified"](xa, temp, *params)).flatten()
    g = h - temp * s
    return float(g[0]) if g.size == 1 else g


def _tangent_through_far_neighbour(bl: BinaryLiquid, x_s: float, x_inv: float, t_inv: float):
    """The invariant's common tangent, fixed by the liquid and the compound's far neighbour.

    At a peritectic or a eutectic the liquid and both neighbouring solids lie on one common
    tangent at the invariant temperature, so the line is determined by two points rather than
    by a slope: the liquid at ``(x_inv, t_inv)``, and the existing hull vertex nearest ``x_s``
    on the side AWAY from ``x_inv`` -- the solid the invariant reaction involves alongside the
    compound, at its Gibbs energy at ``t_inv``. At a pure element that is the polymorph stable
    at ``t_inv``, read as the fit's invariant constraints read it.

    Returns:
        float | None: the line's value at ``x_s``, or ``None`` when no hull vertex lies on
        that side.
    """
    points = _hull_points(bl)
    if x_inv < x_s:
        far = [p for p in points if p[0] > x_s + 1e-9]
        neighbour = min(far, key=lambda p: p[0]) if len(far) else None
    else:
        far = [p for p in points if p[0] < x_s - 1e-9]
        neighbour = max(far, key=lambda p: p[0]) if len(far) else None
    if neighbour is None or abs(float(neighbour[0]) - x_inv) < 1e-9:
        return None
    x_far = float(neighbour[0])
    # an element lists every polymorph at one x: never read the first row's enthalpy there.
    g_far = (
        bl._stable_solid_gibbs_at_T(x_far, t_inv) if x_far in (0.0, 1.0) else float(neighbour[1])
    )
    g_inv = _liquid_gibbs(bl, x_inv, t_inv)
    slope = (g_far - g_inv) / (x_far - x_inv)
    return g_inv + slope * (x_s - x_inv)


def invert_binary_phase_energy(
    bl: BinaryLiquid,
    x_s: float,
    invariant: dict,
    dx: float = 1e-3,
    t_matched_types: tuple[str, ...] = T_MATCHED_ANCHOR_TYPES,
) -> float:
    """Impute the formation enthalpy (J/mol-atom) at composition ``x_s``.

    The compound (entropy 0) lies on the liquid common tangent at its controlling invariant
    ``(x_inv, T_inv)``. Congruent (``cmp``): ``H = G_liq(x_s, T_inv)``. Peritectic / eutectic:
    ``H = G_liq(x_inv, T_inv) + G_liq'(x_inv, T_inv) (x_s - x_inv)``. An anchor type named in
    ``t_matched_types`` takes that tangent from :func:`_tangent_through_far_neighbour`, the line
    through two anchored points; ``cmp`` is the same either way.

    Args:
        bl: fitted model. Left unchanged.
        x_s: the phase composition.
        invariant: its controlling invariant, from :func:`gliquid.mpds.controlling_invariant`.
        dx: half-width of the finite difference for the tangent slope.
        t_matched_types: anchor types taking the two-point tangent. Empty by default.

    Returns:
        float: the formation enthalpy in J/mol-atom.
    """
    x_inv = float(invariant["comp"])
    t_inv = float(invariant["temp"])
    if invariant["type"] == "cmp":
        return _liquid_gibbs(bl, x_s, t_inv)
    if invariant["type"] in (t_matched_types or ()):
        matched = _tangent_through_far_neighbour(bl, x_s, x_inv, t_inv)
        if matched is not None:
            return matched
    return tangent_value(lambda x: _liquid_gibbs(bl, x, t_inv), [x_s], [x_inv], dx)


# -- Detection --


def _controlling_invariant(
    bl: BinaryLiquid, phase: dict, x_s: float, comp_tol: float = 0.02
) -> dict | None:
    """The invariant that anchors the inversion for a missing phase, or ``None``.

    The phase's own melting record where one matches, else a eutectic it flanks. A phase
    with no liquid-bordering invariant is not imputable. ``x_s`` is ``phase['comp']``.
    """
    return mpds.controlling_invariant(phase, bl.invariants, allow_eutectic=True, comp_tol=comp_tol)


# A drawn composition range at least this wide is a homogeneity range, not digitization slop.
_SOLUTION_MIN_SPAN = 0.17
# The max-temperature vertex counts as an EDGE of a field when it lies in this outer fraction.
_SOLUTION_EDGE = 0.15
# ...and only when it is also at least this far from the label's own stated composition.
_SOLUTION_STOICH_TOL = 0.05


def _measured_comp(label: str, components: list[str]) -> float | None:
    """The composition a phase label states, INCLUDING fractional subscripts.

    Args:
        label: an MPDS phase label, polymorph suffix and all.
        components: the two component symbols, low composition first.

    Returns:
        float | None: fraction of ``components[1]``, or None for a label that is not a
        formula over exactly these two elements.
    """
    # parsed here rather than by formula_comp, which rejects fractional subscripts
    base = mpds.split_phase_label(label or "")[0]
    if not base or base.startswith("("):
        return None
    try:
        amounts = Composition(base).get_el_amt_dict()
    except Exception:  # noqa: BLE001 - an unparseable label is not treated as a solution
        return None
    if not amounts or not set(amounts) <= set(components):
        return None
    total = sum(amounts.values())
    return amounts.get(components[1], 0.0) / total if total > 0 else None


def _is_solution_field(phase: dict, components: list[str]) -> bool:
    """Whether a formula-labelled MPDS field is a solid solution rather than a compound.

    An imputed phase is inserted at ONE composition, the reference field's
    max-temperature vertex. A field with a wide homogeneity range has no such
    composition, so imputing a hull vertex for one places a compound the diagram never
    showed.

    A compound with a homogeneity range melts congruently near its own composition, so its
    hot vertex is an interior apex of the drawn range; a solution's hot vertex is an edge.
    All three of wide, off-centre and away from the stated composition must hold.

    Args:
        phase: an ``identify_mpds_phases`` entry. ``cbounds`` holds the drawn composition
            extremes; a line compound has none.
        components: the two component symbols, low composition first.

    Returns:
        bool: True when the field must not be offered as an imputation target.
    """
    bounds = phase.get("cbounds")
    if not bounds:
        return False
    lo, hi = float(bounds[0][0]), float(bounds[1][0])
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    if span < _SOLUTION_MIN_SPAN:
        return False
    stated = _measured_comp(phase.get("name"), components)
    if stated is None:
        return False
    comp = float(phase["comp"])
    at_edge = not (_SOLUTION_EDGE <= (comp - lo) / span <= 1.0 - _SOLUTION_EDGE)
    return at_edge and abs(comp - stated) >= _SOLUTION_STOICH_TOL


def assigned_compound_keys(bl: BinaryLiquid) -> set[str]:
    """The compounds :func:`gliquid.solid_coverage.assign_dft_phases` gives a DFT hull phase.

    Args:
        bl: the system whose MPDS phases, hull and hull space groups to read.

    Returns:
        set[str]: :func:`gliquid.mpds.compound_key` of every matched MPDS phase.
    """
    phases, assigned = bl.dft_assignment()
    return {mpds.compound_key(phases[i].get("name") or "", bl.components) for i in assigned}


def _dft_covers(phase: dict, components: list[str], dft_comps, assigned_keys) -> bool:
    """Whether the hull holds this phase's compound.

    A phase whose compound the assignment matched is covered. A phase the assignment never
    considers (an unpaired high-temperature form) is covered by a hull phase at its integer
    formula's stoichiometry.
    """
    if mpds.compound_key(phase.get("name") or "", components) in assigned_keys:
        return True
    formula = formula_comp(phase.get("name"), components)
    return formula is not None and any(abs(formula - d) <= FORMULA_TOL for d in dft_comps)


def list_binary_imputation_targets(bl: BinaryLiquid) -> list[tuple[dict, dict, float]]:
    """Missing low-T-stable phases that border the liquid, with their anchor invariant.

    A phase is imputable when it (1) is low-temperature stable per ``bl.low_t_exp_phases``,
    (2) is a compound rather than a solution field -- neither an ``(A, B)`` terminal name
    nor a wide field failing :func:`_is_solution_field` -- (3) has no DFT hull phase under
    :func:`gliquid.solid_coverage.assign_dft_phases` (:func:`_dft_covers`), and (4)
    participates in a ``cmp``/``per``/``eut`` invariant. Its composition ``x_s`` is the
    reference phase's max-temperature point (``phase['comp']``).

    Args:
        bl: the system whose MPDS phases and hull to read.

    Returns:
        List of ``(phase, invariant, x_s)`` tuples.
    """
    if not bl.digitized_liq or not bl.mpds_json.get("reference"):
        return []
    if bl.invariants is None or bl.low_t_exp_phases is None:
        bl.find_invariant_points()

    dft_comps = [
        bl._phase_x(p)
        for p in bl.phases
        if p.composition is not None and 0.0 < round(bl._phase_x(p), 6) < 1.0
    ]
    assigned_keys = assigned_compound_keys(bl)

    out: list[tuple[dict, dict, float]] = []
    for phase in bl.low_t_exp_phases or []:
        if "(" in phase["name"]:  # terminal solid solution, not an imputable compound
            continue
        if _is_solution_field(phase, bl.components):  # named as a compound, drawn as a solution
            continue
        x_s = float(phase["comp"])
        if not (0.0 < x_s < 1.0) or _dft_covers(phase, bl.components, dft_comps, assigned_keys):
            continue
        inv = _controlling_invariant(bl, phase, x_s)
        if inv is None:  # does not border the liquid -> not imputable
            continue
        out.append((phase, inv, x_s))
    return out


# -- Synthetic entry construction + stability guard --


def _ref_energy_per_atom(bl: BinaryLiquid, x_s: float) -> float:
    """Composition-weighted element reference energy per atom (eV/atom) from the hull."""
    a, b = bl.components
    return hull_ref_energy_per_atom(bl.dft_ch, {a: 1.0 - x_s, b: x_s})


def imputed_hull(components, entry_dicts, *, dft_type: str = "GGA", data_dir=None) -> PhaseDiagram:
    """The cached DFT hull for ``components`` with imputed entries re-anchored onto it.

    Needs no MPDS record and no :class:`~gliquid.binary.BinaryLiquid`, so any system with
    cached DFT entries can be completed. The hull is built in sorted component order, and the
    rebuild pins that order -- pymatgen re-sorts by electronegativity when ``elements=`` is
    omitted.

    Args:
        components: system spec -- a hyphenated string (``'A-B'``) or a list of components.
        entry_dicts: tagged ``ComputedEntry.as_dict()`` payloads, each carrying
            ``data['h_form_j_per_mol']``. An empty list returns the DFT hull itself.
        dft_type: functional selecting the cache record.
        data_dir: explicit cache STORE override -- a directory (flat layout inside it) or a
            ``CacheBackend``.

    Returns:
        PhaseDiagram: the DFT entries plus the re-anchored imputed entries.

    Raises:
        ValueError: an entry dict carries no ``data['h_form_j_per_mol']``.
    """
    names, _, _ = validate_and_format_system(components, allow_compounds=True)
    hull, _ = api.get_dft_convexhull(sorted(names), dft_type, data_dir=data_dir)
    if not entry_dicts:
        return hull
    return hull_with_reanchored_entries(hull, entry_dicts)[0]


def _binary_provenance(bl: BinaryLiquid, invariant: dict | None) -> dict:
    """What an imputed energy is a function of, recorded so staleness is decidable.

    An imputed enthalpy is read off the fitted liquid curve at an experimental invariant, so
    it is a function of exactly two things: the anchor the diagram supplies and the liquid
    parameters the fit supplies. Both are recorded, so
    :func:`gliquid.imputation.lifecycle.imputed_entry_staleness` can decide staleness
    without re-deriving the energy.

    Args:
        bl: the fitted model the energy was read from.
        invariant: the controlling invariant, or ``None`` when it is not known.

    Returns:
        dict: JSON-serializable provenance for the entry's ``data`` block.
    """
    anchor = None
    if invariant is not None:
        anchor = {
            "type": invariant["type"],
            "comp": float(invariant["comp"]),
            "temp": float(invariant["temp"]),
        }
    return {
        **provenance_header(CALIBRATION),
        "param_format": bl.param_format,
        "fit_params": [float(v) for v in bl.get_params()],
        "anchor": anchor,
    }


def _imputed_computed_entry(
    bl: BinaryLiquid, name: str, x_s: float, h_form_j: float, invariant: dict | None = None
) -> dict:
    """Tagged ``ComputedEntry`` dict for an imputed phase.

    The composition is normalized to a fractional ``Composition`` (one atom total) at
    exactly ``x_s`` so plotting places the phase at its fractional composition. The
    absolute energy is the element reference plus the imputed formation enthalpy
    (converted eV/atom), so the rebuilt hull recovers ``h_form_j`` via
    ``get_form_energy_per_atom``.

    Args:
        bl: the fitted model the energy was read from.
        name: display label, also the ``entry_id`` suffix.
        x_s: the phase composition.
        h_form_j: the imputed formation enthalpy, J/mol-atom.
        invariant: the controlling invariant, recorded in the provenance block. Omitting it
            leaves ``anchor`` null, which
            :func:`gliquid.imputation.lifecycle.imputed_entry_staleness` reads as unknown rather
            than as current.
    """
    a, b = bl.components
    comp = Composition({a: 1.0 - x_s, b: x_s})  # fractional; num_atoms == 1
    energy_per_atom = _ref_energy_per_atom(bl, x_s) + h_form_j / EV_ATOM_TO_J_MOL
    return tagged_entry_dict(
        comp,
        energy_per_atom,
        name,
        {
            "imputed": True,
            "label": name,
            "x_b": float(x_s),
            "h_form_j_per_mol": float(h_form_j),
            PROVENANCE_KEY: _binary_provenance(bl, invariant),
        },
    )


def _comp_key(x_s: float) -> float:
    """The composition a phase is looked up by between imputation passes."""
    return round(float(x_s), 6)


def _field_t_max(bl: BinaryLiquid, name: str) -> float | None:
    """Top of the T-x field the finished diagram draws for ``name``, in K.

    ``None`` when the phase is absent or occupies no field, which is not the same as a cold
    one: a phase with no field has no temperature to compare.
    """
    phase = next((p for p in bl.phases if p.name == name), None)
    if phase is None or not phase.points:
        return None
    return max(float(point[1]) for point in phase.points)


def nearest_arm_choices(bl: BinaryLiquid, **kwargs) -> dict[float, bool]:
    """Which compositions take the two-point common tangent, by nearest drawn field top.

    Both inversions are run to completion on throwaway copies of ``bl``, and each phase is
    given the arm whose DRAWN field top lands closest to its anchor temperature, in either
    direction. The comparison is on the drawn diagram, not on the two enthalpies.

    A phase only one arm realizes takes that arm. ``cmp`` anchors are never switched.

    Args:
        bl: the system to probe. It is NOT mutated; each arm runs on a deep copy.
        **kwargs: forwarded to :func:`impute_binary_phases` for the probe passes, so the choice is
            made under the same guard, masking and coverage settings as the real run.

    Returns:
        dict: composition (rounded) -> True when the T-matched arm is chosen. A composition
        absent from the mapping keeps the caller's ``t_matched_types``.
    """
    probes: dict[str, dict[float, tuple[BinaryImputationResult, float | None]]] = {}
    for label, types in (("shipped", ()), ("tmatched", T_MATCHED_SELECTABLE_TYPES)):
        probe = copy.deepcopy(bl)
        records = impute_binary_phases(
            probe,
            t_matched_types=types,
            inversion_selection="shipped",
            write_cache=False,
            verbose=False,
            **kwargs,
        )
        probes[label] = {_comp_key(r.comp): (r, _field_t_max(probe, r.name)) for r in records}

    choices: dict[float, bool] = {}
    for key in set(probes["shipped"]) | set(probes["tmatched"]):
        shipped, tmatched = probes["shipped"].get(key), probes["tmatched"].get(key)
        record = (shipped or tmatched)[0]
        if record.invariant_type not in T_MATCHED_SELECTABLE_TYPES:
            continue
        if shipped and tmatched and shipped[1] is not None and tmatched[1] is not None:
            anchor = float(record.invariant_temp)
            choices[key] = abs(tmatched[1] - anchor) < abs(shipped[1] - anchor)
        else:
            choices[key] = tmatched is not None and shipped is None
    return choices


def _hull_points(bl: BinaryLiquid) -> np.ndarray:
    """Current solid-hull ``[comp, enthalpy]`` vertices, composition-ordered."""
    return np.array(
        sorted(
            ([bl._phase_x(p), p.enthalpy] for p in bl.phases if p.composition is not None),
            key=lambda pt: pt[0],
        )
    )


def _phase_list_enthalpy_at(bl: BinaryLiquid, x_s: float) -> float:
    """Interpolated existing solid-hull formation enthalpy (J/mol-atom) at ``x_s``.

    Built from the present ``bl.phases`` (the missing-phase state that detection sees), so
    the guard stays consistent with the hull the fit was anchored on, independent of
    ``bl.dft_ch``.
    """
    pts = _hull_points(bl)
    return float(np.interp(x_s, pts[:, 0], pts[:, 1]))


def _below_existing_hull(
    bl: BinaryLiquid, x_s: float, h_form_j: float, margin_j: float = HULL_GUARD_MARGIN_J
) -> bool:
    """True if ``h_form_j`` sits below the current solid hull at ``x_s`` (would be stable).

    When the inverted energy is above the existing hull the phase would not register as a
    stable vertex; that signals an inconsistent invariant or stoichiometry and the phase
    is skipped rather than forced.
    """
    return h_form_j < _phase_list_enthalpy_at(bl, x_s) - margin_j


# -- Admission onto the hull --


def _clamp_onto_matched_hull(
    bl: BinaryLiquid, entry_dict: dict, accepted: list[dict], protected: set[str]
) -> dict | None:
    """The same phase at the deepest enthalpy that leaves every matched vertex a field.

    The energy is raised until the hull carries both the phase and every matched vertex. The
    search is a bisection between the inverted value and the shallowest energy that still
    clears the existing hull by ``HULL_GUARD_MARGIN_J``; raising the energy can only make the
    phase shallower, so the predicate is monotone and the bisection is exact to
    ``CLAMP_TOLERANCE_J``.

    Stability alone does not put a vertex on the diagram: one admitted with no clearance below
    its neighbours' tie line occupies no field. The bisection therefore targets
    ``PROTECTED_VERTEX_MARGIN_J`` of clearance, and where no energy reaches it the phase is
    clamped at the shallowest the hull guard allows, which is the most clearance available.

    Args:
        bl: the system being imputed into.
        entry_dict: the candidate's tagged entry, at its inverted enthalpy.
        accepted: the entries already admitted.
        protected: entry ids that must stay stable.

    Returns:
        dict | None: the clamped entry with its clamp recorded on ``provenance``, or ``None``
        when no enthalpy satisfies both the blockers and the hull guard.
    """
    return clamp_bisect(
        _binary_ops(bl),
        entry_dict,
        accepted,
        protected,
        guard_margin_j=HULL_GUARD_MARGIN_J,
        protected_margin_j=PROTECTED_VERTEX_MARGIN_J,
        tolerance_j=CLAMP_TOLERANCE_J,
        steps=CLAMP_BISECTION_STEPS,
    )


def _binary_ops(bl: BinaryLiquid) -> HullOps:
    """The hull operations of a binary system, each resolved from this module at call time."""
    return HullOps(
        augmented=lambda entry_dicts: augmented_hull(bl.dft_ch, entry_dicts),
        hull_enthalpy_at=lambda entry_dict: _phase_list_enthalpy_at(
            bl, float(entry_dict["data"]["x_b"])
        ),
        entry_at_enthalpy=lambda entry_dict, h_form_j: entry_at_enthalpy(
            bl.dft_ch, entry_dict, h_form_j
        ),
        protected_depth=lambda hull, protected: protected_clearance(hull, protected),
    )


def matched_hull_entry_ids(bl: BinaryLiquid) -> set[str]:
    """Entry ids of the hull vertices a low-temperature MPDS phase stands behind.

    The two elemental endpoints are always included: they bound the hull and no imputed
    interior phase can displace them. Every other stable entry is included only when
    :func:`gliquid.solid_coverage.matched_dft_comps` claims its composition, so the
    complement is the hull's unmatched vertices -- compositions the DFT hull holds that no
    reported compound corresponds to.

    Falls back to every stable entry when the system carries no usable MPDS record, which
    is the conservative reading: with nothing to match against, nothing is unmatched.

    Args:
        bl: the system whose hull and MPDS record to read.

    Returns:
        set[str]: the matched entry ids.
    """
    all_ids = {str(e.entry_id) for e in bl.dft_ch.stable_entries}
    try:
        mpds_phases, unpaired_ht = bl.coverage_phase_inputs()
    except Exception:
        return all_ids
    if not mpds_phases:
        return all_ids

    claimed = solid_coverage.matched_dft_comps(
        mpds_phases,
        bl.components,
        bl._interior_dft_comps(),
        dft_spacegroups=bl.dft_spacegroups(),
        unpaired_ht=unpaired_ht,
    )
    claimed_keys = {round(float(c), 9) for c in claimed}

    matched = set()
    for entry in bl.dft_ch.stable_entries:
        x_b = float(api.entry_frac_along(bl.dft_ch, entry, bl.components)[0])
        if not 0.0 < round(x_b, 6) < 1.0 or round(x_b, 9) in claimed_keys:
            matched.add(str(entry.entry_id))
    return matched


def accept_against_growing_hull(
    bl: BinaryLiquid, candidates: list[dict], protect_dft_hull: bool = True
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Keep only the candidates that survive the hull as it ACTUALLY accumulates.

    Candidates are taken strongest-first (deepest below the starting hull), so a marginal
    phase never displaces a confident one, and each is admitted only if the hull including it
    and every candidate admitted before still carries it and still carries every MATCHED
    solid that was stable to begin with. Matched means a low-temperature MPDS phase stands
    behind the vertex (:func:`matched_hull_entry_ids`); an unmatched vertex does not veto a
    candidate.

    Args:
        bl: the system being imputed into; its ``dft_ch`` supplies the starting hull.
        candidates: ``{'entry_dict', 'name', 'margin_j'}`` records, in any order.
        protect_dft_hull: reject a candidate that would displace a matched entry the starting
            hull held. False admits it and lets the DFT vertex fall.

    Returns:
        tuple: the accepted entry dicts, and ``(name, reason)`` for each rejection.
    """
    protected = matched_hull_entry_ids(bl) if protect_dft_hull else set()
    return accept_growing(
        _binary_ops(bl),
        candidates,
        protected,
        protect=protect_dft_hull,
        clamp=lambda entry_dict, accepted, prot: _clamp_onto_matched_hull(
            bl, entry_dict, accepted, prot
        ),
    )


def _protected_vertex_names(bl: BinaryLiquid) -> set[str]:
    """Phase names of the hull vertices a low-temperature MPDS phase stands behind.

    :func:`matched_hull_entry_ids` keys on entry id and
    :func:`gliquid.binary.hull_field_disagreement` reports phase names, so the two are joined
    here through the entries themselves.

    Args:
        bl: the system whose hull to read, before any imputed entry is applied.

    Returns:
        set[str]: the names ``build_phases_from_chull`` gives those vertices.
    """
    protected = matched_hull_entry_ids(bl)
    return {
        str(api.entry_original(entry).name)
        for entry in bl.dft_ch.stable_entries
        if str(entry.entry_id) in protected
    }


def screen_unrealizable_entries(
    bl: BinaryLiquid, entry_dicts: list[dict], *, unmask: bool = True
) -> tuple[list[dict], list[tuple[str, str]], set[str]]:
    """Keep only the imputed entries the finished diagram actually draws.

    Each pass rebuilds the hull from the surviving entries and drops the weakest entry when an
    imputed entry draws no T-x field, or when a matched DFT vertex that drew a field before
    imputation draws none. Passes repeat until none drops. ``bl`` is left rebuilt on the
    survivors, as :func:`apply_imputed_entries` leaves it.

    Args:
        bl: System to rebuild (mutated in place).
        entry_dicts: Tagged entry payloads, strongest-first (as
            :func:`accept_against_growing_hull` returns them).
        unmask: Passed through to :func:`apply_imputed_entries`.

    Returns:
        tuple: the surviving entry dicts, ``(name, reason)`` for each entry dropped, and the
        names of non-imputed vertices the T-x diagram draws no field for.

    Raises:
        RuntimeError: if the passes do not settle.
    """
    base_ch = bl.dft_ch
    base_unsupported = [list(r) for r in (bl.unsupported_comp_ranges or [])]
    protected = _protected_vertex_names(bl)
    kept = list(entry_dicts)
    dropped: list[tuple[str, str]] = []

    bl.unsupported_comp_ranges = [list(r) for r in base_unsupported]
    already_fieldless = apply_imputed_entries(bl, [], unmask=unmask)

    for _ in range(len(entry_dicts) + 1):
        bl.dft_ch = base_ch
        bl.unsupported_comp_ranges = [list(r) for r in base_unsupported]
        fieldless = apply_imputed_entries(bl, kept, unmask=unmask)
        drawn = {phase.name for phase in bl.phases}

        offenders = []
        for entry_dict in kept:
            name = imputed_entry_label(entry_dict)
            if name not in drawn:
                offenders.append(
                    (entry_dict, name, "off the hull once every accepted phase is on it")
                )
            elif name in fieldless:
                offenders.append((entry_dict, name, "occupies no field on the T-x diagram"))
        erased = sorted((protected & fieldless) - already_fieldless)
        if erased and kept:
            offenders.append(
                (
                    kept[-1],
                    imputed_entry_label(kept[-1]),
                    f"erases the T-x field of the protected vertex {erased[0]}",
                )
            )
        if not offenders:
            return (
                kept,
                dropped,
                {n for n in fieldless if n not in {imputed_entry_label(e) for e in kept}},
            )

        entry_dict, name, reason = offenders[-1]
        kept = [e for e in kept if e is not entry_dict]
        dropped.append((name, reason))

    raise RuntimeError("realizability screen did not settle")


def apply_imputed_entries(
    bl: BinaryLiquid, entry_dicts: list[dict], *, unmask: bool = True
) -> set[str]:
    """Rebuild ``bl``'s hull and phase list with the given imputed entries, in place.

    The hull is rebuilt from ``bl.dft_ch.all_entries`` plus one ``ComputedEntry`` per dict,
    in the element order of the hull being replaced. ``bl.phases`` and
    ``bl.eqs['h_hull_interp']`` are recomputed from it, ``bl.hsx`` is reset and the phase
    points are recomputed. As in ``BinaryLiquid.from_cache``, ``bl``'s solid-solution phases
    are kept and the elemental polymorphs they cover are left out. The liquid parameters are
    not touched, so one model's imputed entries (``BinaryImputationResult.entry_dict``) can be
    replayed onto another model of the same system:
    ``apply_imputed_entries(other_bl, [p.entry_dict for p in imputed])``.

    Args:
        bl: System to rebuild (mutated in place).
        entry_dicts: Tagged ``ComputedEntry.as_dict()`` payloads to add to the hull, each
            carrying ``data['h_form_j_per_mol']``; they are re-anchored onto ``bl``'s
            elemental references by :func:`reanchor_imputed_entries`.
        unmask: Clear ``bl.unsupported_comp_ranges`` so the completed hull is evaluated
            against the full liquidus.

    Returns:
        set[str]: Names of phases that are stable vertices on the T=0 hull but occupy no
        field on the T-x diagram.

    Raises:
        ValueError: an entry dict carries no ``data['h_form_j_per_mol']``.
    """
    ch, _ = hull_with_reanchored_entries(bl.dft_ch, entry_dicts)
    bl.dft_ch = ch
    bl.phases = build_phases_from_chull(
        ch,
        bl.components,
        bl.component_data,
        exclude_spacegroups={SS_SPACEGROUPS[p] for p in bl.ss_models} or None,
    )
    bl._ensure_solid_solution_phases()
    bl.rebuild_hull_interp()
    if unmask:
        bl.unsupported_comp_ranges = []
    bl.hsx = None
    bl.update_phase_points()
    return set(hull_field_disagreement(bl)["stable_without_field"])


# -- Orchestrator --


def impute_binary_phases(
    bl: BinaryLiquid,
    fitting_data: list[dict] | None = None,
    write_cache: bool = False,
    unmask: bool = True,
    verbose: bool = False,
    protect_dft_hull: bool = True,
    t_matched_types: tuple[str, ...] = T_MATCHED_ANCHOR_TYPES,
    inversion_selection: str = DEFAULT_INVERSION_SELECTION,
) -> list[BinaryImputationResult]:
    """Impute missing low-T phases into ``bl`` in place and return what was inserted.

    Call it after a masked fit of ``bl``. Steps: (1) anchor on the lowest-objective candidate
    fit when several are supplied, (2) detect imputable phases, (3) invert each energy at its
    controlling invariant, (4) skip any that would not be a stable vertex, (5) rebuild
    ``dft_ch`` / ``phases`` / ``h_hull_interp`` from the augmented hull, clear the masks and
    recompute phase points, dropping any entry the finished diagram does not draw
    (:func:`screen_unrealizable_entries`), (6) optionally append the surviving tagged entries to
    the store's DFT record. The liquid is not re-fit.

    Args:
        bl: System to impute into (mutated in place).
        fitting_data: Candidate fits from ``fit_parameters``; with more than one, the
            lowest-objective candidate is used (:func:`select_best_fit`). ``None`` uses the
            current ``bl`` parameters.
        write_cache: Append the tagged imputed entries to the store's DFT record. They stay out
            of ``get_dft_convexhull``'s default hull.
        unmask: Clear ``bl.unsupported_comp_ranges`` after inserting phases.
        verbose: Report each skipped or inserted phase through ``logger.info``.
        protect_dft_hull: Reject an imputed phase that would push an existing hull vertex off
            the hull (:func:`accept_against_growing_hull`).
        t_matched_types: Anchor types whose inversion takes the two-point common tangent
            (:func:`invert_binary_phase_energy`).
        inversion_selection: ``'shipped'`` applies ``t_matched_types`` to every phase;
            ``'nearest'`` decides per phase by drawn field top (:func:`nearest_arm_choices`) and
            runs two extra imputation passes on copies of ``bl``.

    Returns:
        List of ``BinaryImputationResult`` records inserted (may be empty). Every record is a
        stable vertex of the finished hull that occupies at least one point of the T-x diagram.

    Raises:
        NotImplementedError: if either component is not elemental.
    ValueError: for an ``inversion_selection`` outside ``INVERSION_SELECTIONS``.
    """
    if any(len(Composition(c).elements) > 1 for c in bl.components):
        raise NotImplementedError(
            "Phase-energy imputation is only supported for elemental components."
        )
    if inversion_selection not in INVERSION_SELECTIONS:
        raise ValueError(
            f"inversion_selection must be one of {INVERSION_SELECTIONS}, got "
            f"{inversion_selection!r}"
        )

    if fitting_data and len(fitting_data) > 1:
        select_best_fit(bl, fitting_data)

    arm_choices: dict[float, bool] = {}
    if inversion_selection == "nearest":
        arm_choices = nearest_arm_choices(
            bl,
            unmask=unmask,
            protect_dft_hull=protect_dft_hull,
        )

    targets = list_binary_imputation_targets(bl)

    imputed: list[BinaryImputationResult] = []
    candidates: list[dict] = []
    for phase, inv, x_s in targets:
        chosen = arm_choices.get(_comp_key(x_s))
        arm_types = (
            t_matched_types if chosen is None else (T_MATCHED_SELECTABLE_TYPES if chosen else ())
        )
        h_form = float(invert_binary_phase_energy(bl, x_s, inv, t_matched_types=arm_types))
        if not _below_existing_hull(bl, x_s, h_form):
            if verbose:
                logger.info(
                    f"skip {phase['name']} @ x={x_s:.3f}: imputed H={h_form:.0f} J/mol-atom "
                    f"not below existing hull (inconsistent invariant)"
                )
            continue
        ed = _imputed_computed_entry(bl, phase["name"], x_s, h_form, invariant=inv)
        ed["data"][PROVENANCE_KEY][ARM_KEY] = "tmatched" if arm_types else "shipped"
        candidates.append(
            {
                "entry_dict": ed,
                "name": phase["name"],
                "margin_j": float(_phase_list_enthalpy_at(bl, x_s) - h_form),
            }
        )
        imputed.append(
            BinaryImputationResult(
                name=phase["name"],
                comp=x_s,
                enthalpy=h_form,
                invariant_type=inv["type"],
                invariant_comp=float(inv["comp"]),
                invariant_temp=float(inv["temp"]),
                entry_dict=ed,
            )
        )

    # Re-score every candidate against the hull as it accumulates.
    entry_dicts, rejected = accept_against_growing_hull(
        bl, candidates, protect_dft_hull=protect_dft_hull
    )
    # Records take the entry and enthalpy the guard accepted, which may be clamped.
    kept = {str(e.get("entry_id")): e for e in entry_dicts}
    imputed = [
        dataclasses.replace(
            record,
            entry_dict=kept[str(record.entry_dict.get("entry_id"))],
            enthalpy=float(
                kept[str(record.entry_dict.get("entry_id"))]["data"]["h_form_j_per_mol"]
            ),
        )
        for record in imputed
        if str(record.entry_dict.get("entry_id")) in kept
    ]
    for name, reason in rejected:
        logger.info(f"skip {name}: {reason}")
    if verbose:
        for record in imputed:
            logger.info(
                f"impute {record.name} @ x={record.comp:.3f}: "
                f"H={record.enthalpy:.0f} J/mol-atom "
                f"(anchor {record.invariant_type} @ x={record.invariant_comp:.3f}, "
                f"T={record.invariant_temp:.0f} K)"
            )

    if not imputed:
        return imputed

    entry_dicts, unrealizable, fieldless = screen_unrealizable_entries(
        bl, entry_dicts, unmask=unmask
    )
    kept_ids = {str(e.get("entry_id")) for e in entry_dicts}
    imputed = [p for p in imputed if str(p.entry_dict.get("entry_id")) in kept_ids]
    for name, reason in unrealizable:
        logger.info(f"skip {name}: {reason}")
    for name in sorted(fieldless):
        logger.info(
            f"{name} is a stable vertex but occupies no field on the T-x diagram; "
            f"it will not appear in any plot"
        )

    if write_cache and entry_dicts:
        api.cache_imputed_entries(bl.sys_name, entry_dicts, dft_type=bl.dft_type)
    return imputed
