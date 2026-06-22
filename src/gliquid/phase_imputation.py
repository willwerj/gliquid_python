"""Phase-energy imputation for the gliquid binary liquid model.

Inverse of :mod:`phase_ablation`. When the DFT convex hull is MISSING a low-temperature
stable solid phase that the experimental (MPDS) diagram shows bordering the liquid, this
module imputes the formation enthalpy that phase must have for the fitted liquid
free-energy curve to reproduce the controlling experimental invariant (congruent,
peritectic, or eutectic point). The imputed phase is inserted as a tagged synthetic
vertex so the shared DFT entries cache still builds a pymatgen ``PhaseDiagram`` without
issue, while remaining distinguishable from real DFT entries (entry_id ``"imputed:..."``,
``data={'imputed': True}``). Plotting renders imputed phases with a dashed line.

Grounding (src/gliquid/binary.py, load_binary_data.py):
  * Invariants come from ``find_invariant_points`` (MPDS-derived, DFT-independent): each
    dict has ``type``/``comp``/``temp``/``phases``/``phase_comps``. ``cmp`` and ``per``
    carry ``phases == [name]`` and ``phase_comps == [x_s]``; ``eut`` carries both flanks.
  * Low-temperature stability uses the same bottom-10% T-band gate that
    ``find_invariant_points`` / ``get_low_temp_phase_data`` apply, via
    ``bl.low_t_exp_phases``.
  * The liquid Gibbs energy is evaluated from ``eqs['h_liq_lambdified']`` /
    ``eqs['s_liq_lambdified']`` in the same DFT-T=0K-referenced frame as the solid hull
    enthalpies (``96485 * get_form_energy_per_atom``). With the compound convention
    ``S = 0`` the solid Gibbs energy is T-independent and equals the stored formation
    enthalpy, so the invariant tangent value IS the imputed enthalpy.

The composition of an imputed phase is the composition of the maximum-temperature point
of the reference MPDS phase (``phase['comp']``), stored as a normalized fractional
``Composition`` (one atom total) so plotting places it at the exact fractional
composition. The liquid curve used for inversion is, when several fits are supplied, the
candidate with the optimal (largest) FIM determinant rather than the lowest-objective
fit. No liquid re-fit is performed after insertion.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

import gliquid.load_binary_data as lbd
from gliquid.binary import BinaryLiquid, build_phases_from_chull, _x_vals
from gliquid.fisher_information import compute_fim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREE_PARAM_INDICES = [0, 2]            # L0_a, L1_a (mirrors the production FIM convention)
DEFAULT_FIM_SIGMA = 5.0
DEFAULT_DFT_COVER_TOL = 0.10           # "missing" tolerance; matches the masking convention
J_PER_MOL_PER_EV = 96485.0
HULL_GUARD_MARGIN_J = 1.0              # imputed H must clear the existing hull by this (J/mol-atom)


@dataclasses.dataclass
class ImputedPhase:
    """A solid phase imputed onto the DFT hull from the experimental liquidus."""
    name: str                  # display label (the reference MPDS phase name)
    comp: float                # B-mole fraction (max-T point of the reference phase)
    enthalpy: float            # imputed formation enthalpy, J/mol-atom
    invariant_type: str        # 'cmp' | 'per' | 'eut' (the anchor used for inversion)
    invariant_comp: float      # liquid composition the tangent is anchored at
    invariant_temp: float      # K
    entry_dict: dict           # tagged ComputedEntry.as_dict()


# ---------------------------------------------------------------------------
# Optimal-FIM fit selection
# ---------------------------------------------------------------------------

def select_optimal_fim_fit(bl: BinaryLiquid, fitting_data: list[dict],
                           sigma: float = DEFAULT_FIM_SIGMA,
                           free_param_indices: list[int] | None = None) -> dict | None:
    """Set ``bl`` to the candidate fit with the largest FIM determinant (D-optimality).

    ``fit_parameters`` returns every candidate fit and leaves ``bl`` at the lowest-objective
    one. For imputation we anchor instead on the most identifiable liquid curve, since the
    inverted energies are read directly off that curve.

    Args:
        bl: System whose parameters to set to the chosen candidate (mutated in place).
        fitting_data: The list returned by ``BinaryLiquid.fit_parameters``.
        sigma: Temperature uncertainty (K) passed to ``compute_fim``.
        free_param_indices: Free-parameter indices for the FIM (default ``[0, 2]``).

    Returns:
        The chosen fit dict, or ``None`` if ``fitting_data`` is empty.
    """
    if not fitting_data:
        return None
    indices = free_param_indices if free_param_indices is not None else FREE_PARAM_INDICES
    best, best_score = fitting_data[0], -np.inf   # always return a valid candidate
    for fit in fitting_data:
        params = [fit['L0_a'], fit['L0_b'], fit['L1_a'], fit['L1_b']]
        try:
            bl.update_params(params)
            score = compute_fim(bl, sigma=sigma, free_param_indices=indices).det_fim
        except Exception:
            score = -np.inf
        if score > best_score:
            best, best_score = fit, score
    if best is not None:
        bl.update_params([best['L0_a'], best['L0_b'], best['L1_a'], best['L1_b']])
        bl.nmpath = best.get('nmpath')
    return best


# ---------------------------------------------------------------------------
# Liquid Gibbs evaluation + invariant inversion
# ---------------------------------------------------------------------------

def _liquid_gibbs(bl: BinaryLiquid, x, temp: float):
    """Molar liquid Gibbs energy ``G_liq(x, temp)`` (J/mol-atom), DFT-T=0K referenced.

    Mirrors the H/S evaluation in ``BinaryLiquid.to_HSX`` (``h_liq_lambdified`` /
    ``s_liq_lambdified``) and forms ``G = H - T S`` at the requested temperature.
    """
    params = [bl.get_L0_a(), bl.get_L0_b(), bl.get_L1_a(), bl.get_L1_b()]
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    h = np.asarray(bl.eqs['h_liq_lambdified'](xa, temp, *params)).flatten()
    s = np.asarray(bl.eqs['s_liq_lambdified'](xa, temp, *params)).flatten()
    g = h - temp * s
    return float(g[0]) if g.size == 1 else g


def invert_phase_energy(bl: BinaryLiquid, x_s: float, invariant: dict, dx: float = 1e-3) -> float:
    """Impute the formation enthalpy (J/mol-atom) at composition ``x_s``.

    The compound (entropy 0) must lie on the liquid common tangent at its controlling
    invariant ``(x_inv, T_inv)``:

    * congruent (``cmp``): the solid lies on the liquid curve at its own composition, so
      ``H = G_liq(x_s, T_inv)``;
    * peritectic / eutectic: the solid lies on the tangent to the liquid at the invariant
      liquid composition, so ``H = G_liq(x_inv, T_inv) + G_liq'(x_inv, T_inv) (x_s - x_inv)``.

    ``x_inv``/``T_inv`` are experimental, so the anchor does not depend on the (possibly
    extrapolated) liquid shape inside the missing-phase gap.
    """
    x_inv = float(invariant['comp'])
    t_inv = float(invariant['temp'])
    if invariant['type'] == 'cmp':
        return _liquid_gibbs(bl, x_s, t_inv)
    xl = min(max(x_inv - dx, 1e-6), 1.0 - 1e-6)
    xr = min(max(x_inv + dx, 1e-6), 1.0 - 1e-6)
    g_inv = _liquid_gibbs(bl, x_inv, t_inv)
    g_slope = (_liquid_gibbs(bl, xr, t_inv) - _liquid_gibbs(bl, xl, t_inv)) / (xr - xl)
    return g_inv + g_slope * (x_s - x_inv)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _controlling_invariant(bl: BinaryLiquid, phase: dict, x_s: float,
                           comp_tol: float = 0.02) -> dict | None:
    """The invariant that anchors the inversion for a missing phase, or ``None``.

    Preference order: the phase's own congruent or peritectic invariant (where it appears
    by name), then a eutectic it flanks (a ``phase_comps`` entry within ``comp_tol`` of
    ``x_s``). A phase with no liquid-bordering invariant is not imputable.
    """
    name = phase['name']
    invs = bl.invariants or []
    for kind in ('cmp', 'per'):
        for inv in invs:
            if inv['type'] == kind and name in (inv.get('phases') or []):
                return inv
    cands = [inv for inv in invs if inv['type'] == 'eut'
             and any(pc is not None and abs(pc - x_s) <= comp_tol
                     for pc in (inv.get('phase_comps') or []))]
    if cands:
        return min(cands, key=lambda inv: abs(float(inv['comp']) - x_s))
    return None


def list_imputable_phases(bl: BinaryLiquid,
                          dft_cover_tol: float = DEFAULT_DFT_COVER_TOL) -> list[tuple[dict, dict, float]]:
    """Missing low-T-stable phases that border the liquid, with their anchor invariant.

    A phase is imputable when it (1) is low-temperature stable per ``bl.low_t_exp_phases``,
    (2) has no interior DFT compound within ``dft_cover_tol`` (the masking "missing"
    convention), and (3) participates in a ``cmp``/``per``/``eut`` invariant. Its
    composition ``x_s`` is the reference phase's max-temperature point (``phase['comp']``).

    Returns:
        List of ``(phase, invariant, x_s)`` tuples.
    """
    if not bl.digitized_liq or not bl.mpds_json.get('reference'):
        return []
    if bl.invariants is None or bl.low_t_exp_phases is None:
        bl.find_invariant_points()

    dft_comps = [p['comp'] for p in bl.phases
                 if not p.get('is_solution') and 0.0 < round(p['comp'], 6) < 1.0]

    def dft_covers(comp: float) -> bool:
        return any(abs(comp - d) <= dft_cover_tol for d in dft_comps)

    out: list[tuple[dict, dict, float]] = []
    for phase in (bl.low_t_exp_phases or []):
        if '(' in phase['name']:            # terminal solid solution, not an imputable compound
            continue
        x_s = float(phase['comp'])
        if not (0.0 < x_s < 1.0) or dft_covers(x_s):
            continue
        inv = _controlling_invariant(bl, phase, x_s)
        if inv is None:                     # does not border the liquid -> not imputable
            continue
        out.append((phase, inv, x_s))
    return out


# ---------------------------------------------------------------------------
# Synthetic entry construction + stability guard
# ---------------------------------------------------------------------------

def _ref_energy_per_atom(bl: BinaryLiquid, x_s: float) -> float:
    """Composition-weighted element reference energy per atom (eV/atom) from the hull."""
    a, b = bl.components
    el_refs = bl.dft_ch.el_refs
    return ((1.0 - x_s) * el_refs[Element(a)].energy_per_atom
            + x_s * el_refs[Element(b)].energy_per_atom)


def _imputed_computed_entry(bl: BinaryLiquid, name: str, x_s: float, h_form_j: float) -> dict:
    """Tagged ``ComputedEntry`` dict for an imputed phase.

    The composition is normalized to a fractional ``Composition`` (one atom total) at
    exactly ``x_s`` so plotting places the phase at its fractional composition. The
    absolute energy is the element reference plus the imputed formation enthalpy
    (converted eV/atom), so the rebuilt hull recovers ``h_form_j`` via
    ``get_form_energy_per_atom``.
    """
    a, b = bl.components
    comp = Composition({a: 1.0 - x_s, b: x_s})          # fractional; num_atoms == 1
    energy_per_atom = _ref_energy_per_atom(bl, x_s) + h_form_j / J_PER_MOL_PER_EV
    entry = ComputedEntry(
        composition=comp,
        energy=energy_per_atom * comp.num_atoms,
        entry_id=f"imputed:{name}",
        data={'imputed': True, 'label': name, 'x_b': float(x_s),
              'h_form_j_per_mol': float(h_form_j)},
    )
    return entry.as_dict()


def _hull_enthalpy_at(bl: BinaryLiquid, x_s: float) -> float:
    """Interpolated existing solid-hull formation enthalpy (J/mol-atom) at ``x_s``.

    Built from the present ``bl.phases`` (the missing-phase state that detection sees), so
    the guard stays consistent with the hull the fit was anchored on, independent of
    ``bl.dft_ch``.
    """
    pts = sorted(([p['comp'], p['enthalpy']] for p in bl.phases if 'comp' in p),
                 key=lambda q: q[0])
    xs = [q[0] for q in pts]
    hs = [q[1] for q in pts]
    return float(np.interp(x_s, xs, hs))


def _below_existing_hull(bl: BinaryLiquid, x_s: float, h_form_j: float,
                         margin_j: float = HULL_GUARD_MARGIN_J) -> bool:
    """True if ``h_form_j`` sits below the current solid hull at ``x_s`` (would be stable).

    When the inverted energy is above the existing hull the phase would not register as a
    stable vertex; that signals an inconsistent invariant or stoichiometry and the phase
    is skipped rather than forced.
    """
    return h_form_j < _hull_enthalpy_at(bl, x_s) - margin_j


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def impute_phases(bl: BinaryLiquid, fitting_data: list[dict] | None = None,
                  sigma: float = DEFAULT_FIM_SIGMA, free_param_indices: list[int] | None = None,
                  dft_cover_tol: float = DEFAULT_DFT_COVER_TOL, write_cache: bool = True,
                  unmask: bool = True, verbose: bool = False) -> list[ImputedPhase]:
    """Impute missing low-T phases into ``bl`` in place and return what was inserted.

    Designed for the masked-fit workflow: fit ``bl`` with masking (so the liquid params come
    cleanly from the representable liquidus), then call this to fill the masked gaps. The
    inversion reads the fitted liquid curve at the EXPERIMENTAL invariants, so it inherently
    uses the full (unmasked) information regardless of which liquidus points the fit masked.

    Steps: (1) anchor on the optimal-FIM candidate fit when several are supplied,
    (2) detect imputable phases, (3) invert each energy at its controlling invariant,
    (4) skip any that would not be a stable vertex (guard), (5) optionally append the
    tagged entries to the shared DFT cache, (6) rebuild ``dft_ch`` / ``phases`` /
    ``h_hull_interp`` from the augmented hull, clear the masks (the gaps are now filled), and
    recompute phase points. No liquid re-fit is performed (by design).

    Args:
        bl: System to impute into (mutated in place).
        fitting_data: Candidate fits from ``fit_parameters``; if more than one, the
            optimal-FIM candidate is selected before inversion. If ``None``, the current
            ``bl`` parameters are used.
        sigma, free_param_indices: FIM settings for candidate selection.
        dft_cover_tol: "Missing" tolerance shared with the masking heuristic.
        write_cache: Append imputed entries to ``<sys>_ENTRIES_MP_<type>.json`` for reuse.
        unmask: After inserting phases, clear ``bl.ignored_comp_ranges`` so the completed
            hull is evaluated against the full liquidus (the "unmask for imputation" step).
        verbose: Print skipped/inserted phases.

    Returns:
        List of ``ImputedPhase`` records actually inserted (may be empty).
    """
    if any(len(Composition(c).elements) > 1 for c in bl.components):
        raise NotImplementedError("Phase-energy imputation is only supported for elemental components.")

    if fitting_data and len(fitting_data) > 1:
        select_optimal_fim_fit(bl, fitting_data, sigma=sigma, free_param_indices=free_param_indices)

    targets = list_imputable_phases(bl, dft_cover_tol=dft_cover_tol)

    imputed: list[ImputedPhase] = []
    entry_dicts: list[dict] = []
    for phase, inv, x_s in targets:
        h_form = float(invert_phase_energy(bl, x_s, inv))
        if not _below_existing_hull(bl, x_s, h_form):
            if verbose:
                print(f"  skip {phase['name']} @ x={x_s:.3f}: imputed H={h_form:.0f} J/mol-atom "
                      f"not below existing hull (inconsistent invariant)")
            continue
        ed = _imputed_computed_entry(bl, phase['name'], x_s, h_form)
        entry_dicts.append(ed)
        imputed.append(ImputedPhase(
            name=phase['name'], comp=x_s, enthalpy=h_form, invariant_type=inv['type'],
            invariant_comp=float(inv['comp']), invariant_temp=float(inv['temp']), entry_dict=ed))
        if verbose:
            print(f"  impute {phase['name']} @ x={x_s:.3f}: H={h_form:.0f} J/mol-atom "
                  f"(anchor {inv['type']} @ x={float(inv['comp']):.3f}, T={float(inv['temp']):.0f} K)")

    if not imputed:
        return imputed

    if write_cache:
        lbd.cache_imputed_entries(bl.sys_name, entry_dicts, dft_type=bl.dft_type)

    # Rebuild the hull from the existing entries plus the imputed vertices. Done in memory
    # (equivalent to reloading the augmented cache) so a failed/skipped cache write never
    # leaves bl inconsistent.
    all_entries = list(bl.dft_ch.all_entries) + [ComputedEntry.from_dict(e) for e in entry_dicts]
    ch = PhaseDiagram(all_entries)
    bl.dft_ch = ch
    bl.phases = build_phases_from_chull(ch, bl.components, bl.component_data)
    hull_points = np.array([[p['comp'], p['enthalpy']] for p in bl.phases if 'comp' in p])
    bl.eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])
    if unmask:
        bl.ignored_comp_ranges = []   # gaps are now filled; evaluate against the full liquidus
    bl.hsx = None
    bl.update_phase_points()
    return imputed
