"""
Authors: Joshua Willwerth, Shibo Tan, Abrar Rauf
Description: This script is designed for the thermodynamic modeling of two-component systems.
It provides tools for fitting the non-ideal mixing parameters of the liquid phase from T=0K DFT-calculated phases and
digitized equilibrium phase boundary data. The data stored and produced may be visualized using the BLPlotter class
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import copy
import logging
import math
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
import gliquid.plotting.export as plot_export
import gliquid.solid_coverage as solid_coverage
import gliquid.solution as solution
from gliquid.cache import (
    imputed_entry_label,
    non_dft_entry_kind,
    non_dft_entry_label,
)
from gliquid.hsx import HSX
from gliquid.phase import (
    EV_ATOM_TO_J_MOL,
    UNARY,
    ComponentRef,
    Phase,
    non_dft_kind,
    polymorph_ladder,
    resolve_component_order,
    validate_and_format_system,
)
from gliquid.solution import (
    DEFAULT_TAU,
    DEFAULT_TSTAR_LIQ_TOL,
    DEFAULT_TSTAR_MAX_HULLS,
    DEFAULT_TSTAR_TOL,
    ParamFormat,
    RKPolyExp,
    SolutionModel,
    comp_symbols,
    l_sym,
    t_sym,
)

if typing.TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_x_step = solution.x_step
_x_vals = solution.x_vals
xb_sym = comp_symbols(2)[0]

# Smallest composition span calculate_deviation_metrics scores.
_OBJECTIVE_MIN_SPAN = 0.7

# Causes of a BinaryLiquid ``init_error``, recorded in ``BinaryLiquid.skip_reason``.
SKIP_NO_LIQUIDUS = "no_liquidus"  # from_cache: no digitized liquidus at all
SKIP_NARROW_SPAN = "narrow_liquidus_span"  # from_cache: assessed range < assessed_crange_floor
SKIP_SPARSE_LIQUIDUS = "sparse_liquidus_interior"  # from_cache: max_gap / covered_fraction gate
SKIP_SOLID_COVERAGE = "insufficient_solid_coverage"  # fit_parameters: solid-energy coverage gate
SKIP_MASK_FRACTION = "mask_fraction_exceeded"  # fit_parameters: unsupported-range mask cap
SKIP_UNION_COVERAGE = (
    "insufficient_union_coverage"  # fit_parameters: coverage floor on the merged mask
)
SKIP_REASONS = (
    SKIP_NO_LIQUIDUS,
    SKIP_NARROW_SPAN,
    SKIP_SPARSE_LIQUIDUS,
    SKIP_SOLID_COVERAGE,
    SKIP_MASK_FRACTION,
    SKIP_UNION_COVERAGE,
)

# Every keyword ``BinaryLiquid.fit_parameters`` accepts through ``**kwargs``; any other raises.
FIT_KWARGS = frozenset(
    {
        # read by fit_parameters itself
        "ignore_ss",
        "disable_inv_constrs",
        "check_full_ss",
        "check_solid_coverage",
        "coverage_thresholds",
        "init_objective_cutoff",
        # forwarded to nelder_mead
        "max_iter",
        # forwarded to f
        "check_lupis_elliott",
        "lupis_elliott_cfg",
        "use_lxb_penalty",
        "lxb_penalty_cfg",
        "suppress_unexpected_migs",
        # forwarded through f to calculate_deviation_metrics
        "unsupported_range_mode",
        "allow_sparse_data",
        "min_comp_span",
    }
)

# How the objective treats composition ranges with no solid free energy behind them:
#   'ignore' -- drop those points from the mesh and from both curves.
#   'score'  -- score them like any other point (the default).
#   'cap'    -- score only the residual above the assessed liquidus.
UNSUPPORTED_RANGE_MODES = ("ignore", "cap", "score")
DEFAULT_UNSUPPORTED_RANGE_MODE = "score"

# A candidate whose initial-simplex objective exceeds this multiple of the best candidate's
# is not optimised; None runs every candidate.
DEFAULT_INIT_OBJECTIVE_CUTOFF = 1e5

# Largest fraction of the digitized-liquidus span the invariant-derived unsupported ranges may
# mask; above it fit_parameters flags SKIP_MASK_FRACTION and returns [].
MASK_FRACTION_CAP = 0.55

# Composition slack for the nearest bounding invariant either side of a masked range.
_INV_FLANK_TOL = 1e-6

# f() scales the objective by (1 + width / GAP_WIDTH_SCALE) ** GAP_EXPONENT, width being
# BinaryLiquid.unexpected_gap_width().
GAP_WIDTH_SCALE = 0.01
GAP_EXPONENT = 2.0


def _binodal_at(ties, temp):
    """``[lo, hi]`` of a dome's tie line at ``temp``, interpolated between the bracketing
    ties; None when ``temp`` lies outside the ties' temperature range."""
    if not ties:
        return None
    ties = sorted(ties, key=lambda tie: tie[2])
    if temp < ties[0][2] or temp > ties[-1][2]:
        return None
    ts = [tie[2] for tie in ties]
    lo = float(np.interp(temp, ts, [tie[0] for tie in ties]))
    hi = float(np.interp(temp, ts, [tie[1] for tie in ties]))
    return [lo, hi]


# Golden-ratio conjugate, (sqrt(5) - 1) / 2.
DEFAULT_ASYM_SCALE = float((np.sqrt(5) - 1) / 2.0)

# Smallest extent (J/mol/K) between the two L*_b values of an initial simplex.
DEFAULT_B_SPREAD_MIN = 10.0

# Bisection steps spent placing one infeasible initial simplex vertex.
DEFAULT_RESTORE_MAX_HALVINGS = 8


def _offset_vertex_value(value: float, spread_min: float, sign_of: float) -> float:
    """``0.8 * value``, or ``value`` moved ``spread_min`` toward zero when that is the larger
    move; ``spread_min`` carrying the sign of ``sign_of`` when ``value`` is 0."""
    if abs(0.2 * value) < spread_min:
        return value - math.copysign(spread_min, value if value else -sign_of)
    return value * 0.8


def _lower_hull_vertices(xs: np.ndarray, gs: np.ndarray) -> list[int]:
    """Indices of the strict lower-hull vertices of the points ``(xs, gs)``, in x order."""
    order = np.lexsort((gs, xs))
    hull = []
    for idx in order:
        x, g = xs[idx], gs[idx]
        if hull and x == xs[hull[-1]]:
            continue
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (xs[b] - xs[a]) * (g - gs[a]) - (gs[b] - gs[a]) * (x - xs[a]) <= 0:
                hull.pop()
            else:
                break
        hull.append(int(idx))
    return hull


def _grid_liquidus(points, grid: np.ndarray) -> np.ndarray:
    """The lowest liquid temperature at each ``grid`` composition among ``points``
    (``[x, T]`` rows); NaN where the liquid has no point."""
    lookup = {}
    for xv, tv in np.asarray(points or [], dtype=float).reshape(-1, 2):
        key = round(float(xv), 9)
        lookup[key] = min(lookup.get(key, np.inf), float(tv))
    return np.array([lookup.get(round(float(xv), 9), np.nan) for xv in grid])


def _restore_infeasible_vertices(evaluate, simplex, f_vals, params, max_halvings) -> int:
    """Move each infinite vertex of a simplex onto the feasible side of the constraint.

    A vertex is first mirrored through the centroid of the finite vertices, which keeps the
    simplex's size; when the mirror image is infinite too, the vertex is bisected toward the
    best finite vertex and the finite midpoint nearest the infeasible side is kept. A vertex
    neither step can place is left where it was.

    Args:
        evaluate: ``x -> (f, params)``.
        simplex: vertices, one per row; modified in place.
        f_vals: objective per vertex; modified in place.
        params: parameter vector per vertex; modified in place.
        max_halvings: bisection steps spent per infinite vertex.

    Returns:
        int: how many vertices were placed.
    """
    finite = np.isfinite(f_vals)
    if finite.all() or not finite.any():
        return 0
    ibest = int(np.argmin(np.where(finite, f_vals, np.inf)))
    centroid = simplex[finite].mean(axis=0)
    n_restored = 0
    for i in np.flatnonzero(~finite):
        mirror = 2.0 * centroid - simplex[i]
        f_m, p_m = evaluate(mirror)
        if np.isfinite(f_m):
            simplex[i], f_vals[i], params[i] = mirror, f_m, p_m
            n_restored += 1
            continue
        infeasible, feasible = simplex[i].copy(), simplex[ibest].copy()
        found = None
        for _ in range(max_halvings):
            mid = 0.5 * (infeasible + feasible)
            f_mid, p_mid = evaluate(mid)
            if np.isfinite(f_mid):
                feasible, found = mid, (f_mid, p_mid)
            else:
                infeasible = mid
        if found is not None:
            simplex[i], (f_vals[i], params[i]) = feasible, found
            n_restored += 1
    return n_restored


def resolve_unsupported_range_mode(kwargs: dict) -> str:
    """The unsupported-range treatment named by ``kwargs``, one of ``UNSUPPORTED_RANGE_MODES``.

    Raises:
        ValueError: for an unrecognized value.
        TypeError: for the ``ignored_ranges`` keyword removed in 0.3.0.
    """
    if "ignored_ranges" in kwargs:
        raise TypeError(
            "'ignored_ranges' was removed in 0.3.0; use unsupported_range_mode='ignore' or 'score'"
        )
    mode = kwargs.get("unsupported_range_mode")
    if mode is None:
        mode = DEFAULT_UNSUPPORTED_RANGE_MODE
    if mode not in UNSUPPORTED_RANGE_MODES:
        raise ValueError(
            f"unsupported_range_mode must be one of {UNSUPPORTED_RANGE_MODES}; got {mode!r}."
        )
    return mode


def _dft_comp_within(comp, dft_comps, tol):
    """True if a stable interior DFT compound sits within ``tol`` of ``comp``."""
    return any(abs(c - comp) <= tol for c in dft_comps)


def _mask_walked(ranges, lo, hi, gap_comps):
    """Append a walked range unless it spans a printed two-liquid label."""
    if any(lo - 1e-9 <= x <= hi + 1e-9 for x in gap_comps):
        return
    ranges.append([lo, hi])


def get_hull_rel_enth_skew(dft_ch: PhaseDiagram) -> float:
    """
    Calculate the enthalpy skew of the DFT T=0K convex hull, relative to the ideal liquid enthalpy.
    """
    comps = api.pd_components(dft_ch)
    hull_skew = (UNARY[comps[0]].h_liq - UNARY[comps[1]].h_liq) / 4.0
    for e in dft_ch.stable_entries:
        xb_comp = e.composition.fractional_composition.as_dict().get(comps[1], 0)
        form_energy_j_mol = dft_ch.get_form_energy_per_atom(e) * EV_ATOM_TO_J_MOL
        hull_skew += (xb_comp - 0.5) * form_energy_j_mol
    return float(hull_skew)


def get_hull_rel_mid_depth(dft_ch: PhaseDiagram) -> float:
    """
    Calculate the depth at the middle composition of the DFT T=0K convex hull, relative to the ideal liquid enthalpy.
    """
    comps = api.pd_components(dft_ch)
    lhs_ref = dft_ch.get_hull_energy_per_atom(Composition({comps[0]: 1}))
    rhs_ref = dft_ch.get_hull_energy_per_atom(Composition({comps[1]: 1}))
    hull_mid_ref = dft_ch.get_hull_energy_per_atom(Composition({c: 0.5 for c in comps}))
    ends_j_mol = lhs_ref * EV_ATOM_TO_J_MOL + UNARY[comps[0]].h_liq + rhs_ref * EV_ATOM_TO_J_MOL
    e_depth = hull_mid_ref * EV_ATOM_TO_J_MOL - (ends_j_mol + UNARY[comps[1]].h_liq) / 2
    return float(e_depth)


def build_init_triangle(
    guess_keys,
    dft_ch: PhaseDiagram = None,
    asym_scale=DEFAULT_ASYM_SCALE,
    invert_scale=-1.0321,
    tau: float = DEFAULT_TAU,
    fmt: ParamFormat | None = None,
    b_spread_min: float = DEFAULT_B_SPREAD_MIN,
) -> list[list[float]]:
    """Initial Nelder-Mead simplex from DFT convex-hull statistics.

    ``guess_keys`` names the two guessed parameters, normally the format's
    ``guess_params``. The ``L*_a`` guesses come from the hull's depth and
    skew; the ``L*_b`` guesses are ``invert_scale * L*_a / tau``, the tau line of a linear
    term. With ``fmt`` given, an order whose term carries an ``exp(-T/tau)`` factor
    (``'combined'``, ``'comb-exp'``) is anchored at S_xs(0 K) = 0 instead: the first ``L*_b``
    guess is ``L*_a / tau`` and the second is offset away from zero, in ``L*_a``'s sign, by the
    linear rule's spread floored at ``b_spread_min``.
    """
    if dft_ch is None:
        raise ValueError(
            "DFT convex hull data must be provided to build initial simplex based on hull features."
        )

    stable_compounds = len(dft_ch.facets) != 1
    re_depth = get_hull_rel_mid_depth(dft_ch) * 2.0
    l0_sign = 1.0 if stable_compounds else invert_scale
    re_skew = get_hull_rel_enth_skew(dft_ch) * 4.0

    param_guesses = {
        "L0_a": [re_depth * l0_sign, re_depth * asym_scale],
        "L1_a": [re_skew, re_skew * asym_scale * invert_scale],
    }
    param_guesses.update(
        {
            "L0_b": [g / tau * invert_scale for g in param_guesses["L0_a"]],
            "L1_b": [g / tau * invert_scale for g in param_guesses["L1_a"]],
        }
    )  # tau scales the T-linear guesses
    if fmt is not None:
        for order in (0, 1):
            a_key, b_key = f"L{order}_a", f"L{order}_b"
            if b_key in fmt.param_names and fmt.order_expr(order, tau).has(sp.exp):
                b_lo, b_hi = param_guesses[b_key]
                spread = max(abs(b_lo - b_hi), b_spread_min)
                anchor = param_guesses[a_key][0] / tau
                param_guesses[b_key] = [anchor, anchor + math.copysign(spread, anchor)]

    pg1 = param_guesses[guess_keys[0]]
    pg2 = param_guesses[guess_keys[1]]

    return [[pg1[0], pg2[0]], [pg1[0], pg2[1]], [pg1[1], pg2[1]]]


def hull_field_disagreement(bl) -> dict[str, list[str]]:
    """Interior phases the T=0 hull (``dft_ch``) and the T-x hull (``hsx``) disagree about.

    Call after ``update_phase_points``: before it no phase carries ``points``.

    Args:
        bl: a ``BinaryLiquid`` whose ``phases`` carry points.

    Returns:
        dict: ``stable_without_field`` names interior T=0 vertices that the T-x hull draws
        nowhere. ``field_without_stable`` names phases HSX draws that are not interior
        vertices, such as elemental polymorphs and solution phases.
    """
    interior, drawn = [], []
    for phase in bl.phases:
        if phase.name == "L":
            continue
        if phase.points:
            drawn.append(phase.name)
        if phase.composition is None:
            continue
        if 0.0 < float(bl._phase_x(phase)) < 1.0:
            interior.append(phase)
    return {
        "stable_without_field": [p.name for p in interior if not p.points],
        "field_without_stable": [name for name in drawn if name not in {p.name for p in interior}],
    }


def hull_phase_fields(entry) -> dict:
    """The identity fields of the solid ``Phase`` a hull entry becomes.

    Args:
        entry: a hull entry; a transformed entry is read through its original.

    Returns:
        dict: ``name``, ``imputed`` and ``non_dft``. An entry tagged ``data['imputed']`` is
        named by :func:`gliquid.cache.imputed_entry_label` and a user-defined one by
        :func:`gliquid.cache.non_dft_entry_label`; every other entry keeps its ``name``.
        ``non_dft`` is :func:`gliquid.cache.non_dft_entry_kind`.
    """
    original = api.entry_original(entry)
    entry_data = getattr(original, "data", None) or {}
    if entry_data.get("imputed"):
        return {
            "name": imputed_entry_label({"entry_id": original.entry_id, "data": entry_data}),
            "imputed": True,
            "non_dft": "imputed",
        }
    kind = non_dft_entry_kind(original)
    name = non_dft_entry_label(original) if kind == "user" else original.name
    return {"name": name, "imputed": False, "non_dft": kind}


def _endpoint_polymorphs(
    components: list[str],
    component_data: dict[str, ComponentRef],
    exclude_spacegroups: set[int] | None = None,
) -> list[Phase]:
    """Every component's polymorph ladder as solid phases, in phase-list order.

    After a stable sort by composition, each component's ground state is the polymorph
    nearest the interior of the composition axis.

    Args:
        components (list): The two component names.
        component_data (dict): Mapping of component symbol to ``ComponentRef``.
        exclude_spacegroups (set[int] | None): Polymorphs whose spacegroup number is in this
            set are skipped.

    Returns:
        list[Phase]: The polymorphs, unsorted.
    """
    phases = []
    for comp in components:
        for polymorph in polymorph_ladder(component_data[comp], exclude_spacegroups):
            # The insert(-1) ordering must be preserved.
            phases.insert(-1, polymorph)
    return phases


def build_phases_from_chull(
    ch: PhaseDiagram,
    components: list[str],
    component_data: dict[str, ComponentRef],
    exclude_spacegroups: set[int] | None = None,
) -> list[Phase]:
    """
    Builds the system's phase list as ``Phase`` objects.

    Args:
        ch (PhaseDiagram): A pymatgen PhaseDiagram object containing stable entries.
        components (list): List of component names; each phase's evaluation-axis
            fraction derives from its Composition via ``Phase.fraction_in``.
        component_data (dict): Mapping of component symbol to ``ComponentRef``, used to add polymorphs as explicit phases.
        exclude_spacegroups (set[int] | None): Endpoint polymorphs whose spacegroup number is in
            this set are skipped.

    Returns:
        list[Phase]: Elemental polymorphs and line compounds sorted by composition,
        with the 'L' liquid sentinel appended last.
    """
    phases = _endpoint_polymorphs(components, component_data, exclude_spacegroups)
    for entry in ch.stable_entries:
        composition = api.entry_frac_along(ch, entry, components)[0]
        if composition in [
            p.fraction_in(components)[0] for p in phases
        ]:  # Skip if a polymorph of the pure element is already added
            continue
        phases.insert(
            -1,
            Phase(
                phase_type="solid",
                composition=api.entry_original(entry).composition,
                enthalpy=EV_ATOM_TO_J_MOL * ch.get_form_energy_per_atom(entry),
                entropy=0,
                **hull_phase_fields(entry),
            ),
        )
    phases.sort(key=lambda p: p.fraction_in(components)[0])
    phases.append(Phase(phase_type="liquid", name="L"))
    return phases


class BinaryLiquid:
    """
    Represents a binary liquid system for thermodynamic modeling and phase diagram generation.

    Attributes:
        init_error (bool): Flag indicating if an error occurred during initialization.
        skip_reason (str | None): The ``SKIP_*`` cause that set ``init_error`` first;
            ``None`` while it is False.
        full_comp_ss (bool): Set by ``find_invariant_points`` when the MPDS labels report a
            full-composition solid solution.
        sys_name (str): Binary system name.
        components (list): List of component names.
        component_data (dict): Thermodynamic data for components.
        mean_elt_tm (float): Mean elemental melting temperature.
        liquid_tstar (np.ndarray | None): Linearization temperature (K) of each interior grid
            row in the last ``update_phase_points``.
        pd_ind (int | None): Index of the MPDS phase diagram used.
        mpds_json (dict): MPDS phase equilibrium data for the system.
        digitized_liq (list): Digitized liquidus data points.
        max_liq_temp (float | None): Maximum temperature on the liquidus line.
        min_liq_temp (float | None): Minimum temperature on the liquidus line.
        temp_range (list): Temperature range for calculations.
        assessed_crange_floor (float): Minimum fraction of the composition axis that must
            carry digitized liquidus for the system to be considered fittable.
        liq_coverage (dict | None): ``mpds.liquidus_coverage`` metrics of the pre-fill
            digitized liquidus (``max_gap``, ``covered_fraction``, ``holes``, ...), in this
            object's component frame. A dict measured on a reversed-frame json is mirrored
            first (``mpds.mirror_liquidus_coverage``).
        liquidus_holes (list): Read-only view of ``liq_coverage['holes']``.
        undigitized_comp_ranges (list): ``[lo, hi]`` intervals of the whole composition axis
            with no digitized liquidus: outside the digitized endpoints, or inside a hole.
            ``assessed_crange_floor`` gates on their complement.
        unsupported_comp_ranges (list): ``[lo, hi]`` composition intervals whose conjugate
            solid has no evaluable free energy. ``fit_parameters`` fills it from
            ``coverage_report.unsupported_ranges`` and its own invariant walk.
        fit_unsupported_range_mode (str | None): The ``unsupported_range_mode`` of the last
            ``fit_parameters`` call, ``None`` before one. ``BLPlotter`` marks
            ``unsupported_comp_ranges`` on the assessed liquidus unless it is ``'score'``, whose
            objective treats those ranges like any other composition.
        expected_gap_ranges (list): Read-only; composition ranges where the assessed diagram
            allows two liquids.
        dft_type (str): Functional used for DFT calculations.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.
        removed_entry_ids (list): Canonical ids of the DFT entries a ConvexHullEditor edit
            removed from ``dft_ch``.
        phases (list): List of phase data.
        ss_models (dict): Solid-solution models by phase name.
        ss_in_hull (bool): Whether solid-solution phases enter the hull; the
            ``fit_parameters`` ``ignore_ss`` option sets it False. Re-enable with
            ``bl.ss_in_hull = True; bl.update_phase_points()``.
        xs_mix (RKPolyExp): The liquid's excess-mixing model: parameter values,
            formalism ('linear'/'combined'/'comb-exp'), and fitting metadata. Pass a prebuilt
            one as ``xs_mix``, or ``param_format``/``params``/``tau``.
        param_format (str): The mixing model's formalism name (read-only view).
        tau (float): exp(-T/tau) decay constant of the 'combined'/'comb-exp' forms.
        eqs (dict): Dictionary of thermodynamic Sympy expressions.
        invariants (list): Identified invariant points.
        low_t_exp_phases (list): Low-temperature phases from MPDS JSON.
        coverage_report (SolidCoverageReport | None): Result of the solid-energy coverage
            assessment, populated by ``assess_solid_coverage``.
        guess_symbols (list): Sympy symbols for corresponding to guessed parameters.
        constraints (list): Sympy equations used to store parameter constraints.
        init_triangle (np.ndarray): Initial simplex for Nelder-Mead optimization.
        nmpath (np.ndarray): Nelder-Mead optimization path, stored after running Nelder-Mead for plotting purposes.
        hsx (HSX): HSX object for phase diagram calculations.
        dropped_candidates (list): Names of the invariant candidates the last fit dropped:
            their solve left a symbol undetermined, or their initial objective was infinite.
        pruned_candidates (list): ``[name, initial objective]`` of each candidate the last
            fit's ``init_objective_cutoff`` kept from optimising.
    """

    @config._removed_keywords(
        {
            "ignored_comp_ranges": "unsupported_comp_ranges",
            "comp_range_fit_lim": "assessed_crange_floor",
        }
    )
    def __init__(
        self,
        sys_name: str,
        components: list,
        init_error=False,
        *,
        skip_reason=None,
        component_data=None,
        pd_ind=None,
        mpds_json=None,
        digitized_liq=None,
        temp_range=None,
        assessed_crange_floor=None,
        undigitized_comp_ranges=None,
        liq_coverage=None,
        unsupported_comp_ranges=None,
        dft_type="GGA",
        dft_ch=None,
        removed_entry_ids=None,
        phases=None,
        ss_models=None,
        xs_mix=None,
        param_format="comb-exp",
        params=(),
        tau=DEFAULT_TAU,
        eqs=None,
        invariants=None,
    ):
        self.init_error = init_error
        self.skip_reason = skip_reason
        self.full_comp_ss = False
        self.coverage_report = None
        self.sys_name = sys_name
        self.components = components
        self.component_data = {} if component_data is None else component_data
        self.mean_elt_tm = np.mean([self.component_data[comp].t_fusion for comp in self.components])
        self.pd_ind = pd_ind
        self.mpds_json = {} if mpds_json is None else mpds_json
        self.digitized_liq = [] if digitized_liq is None else digitized_liq
        self.max_liq_temp = (
            max(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        )
        self.min_liq_temp = (
            min(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        )
        self.temp_range = [] if temp_range is None else temp_range
        self.assessed_crange_floor = (
            config.assessed_crange_floor if assessed_crange_floor is None else assessed_crange_floor
        )
        self.undigitized_comp_ranges = undigitized_comp_ranges or []
        self.liq_coverage = liq_coverage
        self.unsupported_comp_ranges = (
            [] if unsupported_comp_ranges is None else unsupported_comp_ranges
        )
        self.fit_unsupported_range_mode = None
        self.dft_type = dft_type
        self.dft_ch = dft_ch
        self.removed_entry_ids = list(removed_entry_ids or [])
        self.phases = [] if phases is None else phases
        self.ss_models = ss_models or {}
        self.ss_in_hull = True
        if self.ss_models:
            self._ensure_solid_solution_phases()
        self.xs_mix = xs_mix if xs_mix is not None else RKPolyExp(param_format, params, tau=tau)
        self.tau = self.xs_mix.tau
        self.init_triangle = build_init_triangle(
            self.xs_mix.format.guess_params, self.dft_ch, tau=self.tau, fmt=self.xs_mix.format
        )
        self.eqs = eqs if eqs is not None else self._build_eqs()
        self.invariants = invariants
        self.low_t_exp_phases = None
        self.guess_symbols = None
        self.constraints = None
        self._ref_params = None
        self.nmpath = None
        self.hsx = None
        self.liquid_tstar = None

    def __str__(self):
        return f"BinaryLiquid({self.sys_name})"

    def __repr__(self):
        return (
            f"BinaryLiquid(sys_name='{self.sys_name}', components={self.components}, "
            f"params={self.xs_mix.values}, param_format='{self.param_format}', "
            f"dft_type='{self.dft_type}')"
        )

    @property
    def param_format(self) -> str:
        """The mixing model's formalism name (e.g. 'linear', 'combined', 'comb-exp')."""
        return self.xs_mix.format.name

    @property
    def liquidus_holes(self) -> list[list[float]]:
        """``[x_lo, x_hi]`` intervals where NO liquidus was digitized, in this frame.

        The undigitized stretches between disjoint 'L' shapes that
        ``mpds.extract_digitized_liquidus`` refuses to bridge (see
        ``mpds.liquidus_coverage``). Empty for a contiguous liquid field, and empty for any
        object built without a ``liq_coverage`` — a missing measurement masks nothing.
        """
        return list((self.liq_coverage or {}).get("holes") or [])

    @property
    def expected_gap_ranges(self) -> list[list[float]]:
        """Composition ranges where the assessed diagram allows two liquids.

        One ``[lo, hi]`` per ``mig`` invariant whose feet lie on the digitized liquidus
        within ``config.coverage_mig_foot_tol``; a ``mig`` whose feet do not contributes
        ``[0, 1]``, its extent being unknown. Merged, in composition order; empty when no
        invariant records a gap.
        """
        ranges = [[lo, hi] for lo, hi, _ in self._expected_gaps()]
        return mpds.merge_ranges(ranges) if ranges else []

    def _expected_gaps(self) -> list[tuple[float, float, float | None]]:
        """``(lo, hi, T_iso)`` per ``mig`` invariant: the feet and their isotherm in K when
        the feet rest on the digitized liquidus, else ``(0, 1, None)``."""
        gaps = []
        for inv in self.invariants or []:
            if inv.get("type") != "mig":
                continue
            if mpds.mig_feet_on_liquidus(inv, self.digitized_liq, config.coverage_mig_foot_tol):
                lo, hi = mpds.mig_feet([inv])[0]
                t_iso = (float(inv["cbounds"][0][1]) + float(inv["cbounds"][1][1])) / 2
                gaps.append((lo, hi, t_iso))
            else:
                gaps.append((0.0, 1.0, None))
        return gaps

    def unexpected_gap_width(self, match_tol: float | None = None) -> float:
        """Composition width of the generated two-liquid region the expected gaps do not cover.

        Reads the diagram of the last ``update_phase_points``. Its two-liquid ties are replaced
        by those of the liquid's exact Gibbs energy at their temperatures
        (``_exact_two_liquid_ties``). For each expected gap with a known isotherm the tie line at
        that temperature is measured; a tie overlapping no expected gap counts in full.

        Args:
            match_tol: composition slack added to each side of every expected range;
                ``config.gap_match_tol`` when None.

        Returns:
            float: summed width; zero when the generated diagram ties no two liquids or
            every measured tie lies inside an expected range.
        """
        if self.hsx is None:
            return 0.0
        tol = config.gap_match_tol if match_tol is None else float(match_tol)
        ties = self.hsx.two_liquid_ties()
        if not ties:
            return 0.0
        expected = self._expected_gaps()
        ties = self._exact_two_liquid_ties(ties, expected)
        if not ties:
            return 0.0

        def overlaps(lo, hi, a, b):
            return min(hi, b + tol) - max(lo, a - tol) > 0

        orphans = [
            [lo, hi]
            for lo, hi, _ in ties
            if not any(overlaps(lo, hi, a, b) for a, b, _ in expected)
        ]
        width = sum(hi - lo for lo, hi in mpds.merge_ranges(orphans)) if orphans else 0.0
        for a, b, t_iso in expected:
            if t_iso is None:
                continue
            span = _binodal_at([tie for tie in ties if overlaps(tie[0], tie[1], a, b)], t_iso)
            if span is None:
                continue
            width += max(a - tol - span[0], 0.0) + max(span[1] - b - tol, 0.0)
        return float(width)

    def _exact_two_liquid_ties(self, hull_ties: list, expected: list) -> list[list[float]]:
        """Two-liquid ties of the exact liquid, H and S taken at each temperature, with the solids.

        Temperatures are the hull ties' (rounded to 1 K; at most 12 of them, evenly spread with
        the lowest and highest kept) and each expected gap's isotherm. A tie joins two liquids
        more than 0.012 apart in composition.

        Args:
            hull_ties: ``[lo, hi, T]`` two-liquid ties of the hull.
            expected: ``[lo, hi, T or None]`` expected gaps (``_expected_gaps``).

        Returns:
            list: ``[lo, hi, T]`` per tie of the exact lower hull.
        """
        data = self.to_hsx()
        grid = np.asarray(_x_vals, dtype=float)
        n = len(grid)
        hull_t = np.unique(np.round([float(t) for _, _, t in hull_ties], 0))
        if len(hull_t) > 12:
            keep = np.unique(np.linspace(0, len(hull_t) - 1, 12).round().astype(int))
            hull_t = hull_t[keep]
        temps = set(hull_t.tolist()) | {float(t) for _, _, t in expected if t is not None}
        temps = np.asarray(sorted(temps), dtype=float)
        xi = grid[1:-1]
        xx = np.tile(xi, len(temps))
        tt = np.repeat(temps, len(xi))
        params = self.xs_mix.values
        h = np.broadcast_to(
            np.asarray(self.eqs["h_liq_lambdified"](xx, tt, *params), dtype=float), xx.shape
        )
        s = np.broadcast_to(
            np.asarray(self.eqs["s_liq_lambdified"](xx, tt, *params), dtype=float), xx.shape
        )
        big_h = np.asarray(data["H"], dtype=float)
        big_s = np.asarray(data["S"], dtype=float)
        big_x = np.asarray(data["X"], dtype=float)
        solid = np.arange(len(big_x)) >= n
        g_liq = np.empty((len(temps), n))
        g_liq[:, 0] = big_h[0] - temps * big_s[0]
        g_liq[:, -1] = big_h[n - 1] - temps * big_s[n - 1]
        g_liq[:, 1:-1] = h.reshape(len(temps), len(xi)) - temps[:, None] * s.reshape(
            len(temps), len(xi)
        )
        xs = np.concatenate([grid, big_x[solid]])
        ties = []
        for k, temp in enumerate(temps):
            gs = np.concatenate([g_liq[k], big_h[solid] - temp * big_s[solid]])
            verts = _lower_hull_vertices(xs, gs)
            for a, b in zip(verts[:-1], verts[1:]):
                if a < n and b < n and xs[b] - xs[a] > 0.012:
                    ties.append([float(xs[a]), float(xs[b]), float(temp)])
        return ties

    def _flag_skip(self, reason: str) -> None:
        """Set ``init_error`` and record ``reason`` unless a skip reason is already set."""
        self.init_error = True
        if self.skip_reason is None:
            self.skip_reason = reason

    def _phase_x(self, phase: Phase) -> float:
        """A fixed phase's binary evaluation-axis fraction (x of components[1])."""
        return phase.fraction_in(self.components)[0]

    def _find_nearest_phase(self, composition, tol=0.02):
        sorted_phases = sorted(
            [p for p in self.phases if p.composition is not None],
            key=lambda p: abs(self._phase_x(p) - composition),
        )
        if not sorted_phases:
            return None, float("inf")
        nearest = sorted_phases[0]
        deviation = abs(self._phase_x(nearest) - composition)
        if deviation > tol:
            return None, deviation
        return nearest, deviation

    def _build_eqs(self, ga_expr: sp.Expr = 0 * t_sym, gb_expr: sp.Expr = 0 * t_sym) -> dict:
        """The liquid eqs dict from the instance's mixing model and reference exprs."""
        return SolutionModel(
            tuple(self.components), (ga_expr, gb_expr), {(0, 1): self.xs_mix}
        ).binary_eqs()

    def rebuild_hull_interp(self, phases: list[Phase] | None = None) -> None:
        """Recompute ``self.eqs['h_hull_interp']`` from ``phases``.

        The value is the linear interpolation, on the interior points of the composition
        grid, of the fixed-composition phases' ``(x, enthalpy)`` points sorted by x. Solution
        phases carry no fixed composition and are excluded. No other entry of ``eqs`` is
        touched.

        Args:
            phases: the phase list to interpolate, in phase-list order; ``self.phases`` when
                omitted.

        Raises:
            ValueError: No fixed-composition phase is left; ``eqs`` is left unchanged.
        """
        phases = self.phases if phases is None else phases
        hull_points = np.array(
            sorted(
                ([self._phase_x(p), p.enthalpy] for p in phases if p.composition is not None),
                key=lambda pt: pt[0],
            )
        )
        if hull_points.size == 0:
            raise ValueError(
                f"{self.sys_name} has no fixed-composition phases left to build a hull from."
            )
        self.eqs["h_hull_interp"] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])

    def reset_fit_state(self) -> None:
        """Clear every field ``fit_parameters`` populates so the next fit recomputes from scratch.

        Sets ``invariants``, ``low_t_exp_phases``, ``coverage_report``, ``guess_symbols``,
        ``constraints``, ``_ref_params``, ``nmpath``, ``hsx``, ``skip_reason`` and
        ``fit_unsupported_range_mode`` to ``None``;
        ``unsupported_comp_ranges`` to ``[]``; ``full_comp_ss`` and ``init_error`` to
        ``False``; the mixing parameters to the format's zero vector (``xs_mix.update([])``);
        and every phase's ``points`` to ``[]``.
        """
        self.invariants = None
        self.low_t_exp_phases = None
        self.unsupported_comp_ranges = []
        self.fit_unsupported_range_mode = None
        self.coverage_report = None
        self.full_comp_ss = False
        self.guess_symbols = None
        self.constraints = None
        self._ref_params = None
        self.nmpath = None
        self.hsx = None
        self.init_error = False
        self.skip_reason = None
        self.xs_mix.update([])
        for phase in self.phases:
            phase.points = []

    def _rebuild_thermodynamic_expressions(self) -> None:
        """Rebuild thermodynamic expressions and lambdified callables after unpickling."""
        if self.component_data and self.components:
            eqs = self._build_eqs(
                ga_expr=self.component_data[self.components[0]].gibbs_ref_expr(t_sym),
                gb_expr=self.component_data[self.components[1]].gibbs_ref_expr(t_sym),
            )
        else:
            eqs = self._build_eqs()

        hull_points = np.array(
            [[0, 0]]
            + [[self._phase_x(p), p.enthalpy] for p in self.phases if p.composition is not None]
            + [[1, 0]]
        )
        eqs["h_hull_interp"] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])
        self.eqs = eqs

    def __getstate__(self):
        """Pickle-safe state: lambdified callables and ``hsx`` are dropped and rebuilt on load."""
        state = self.__dict__.copy()
        state["hsx"] = None
        state["eqs"] = {
            key: value for key, value in state.get("eqs", {}).items() if "lambdified" not in key
        }
        return state

    def __setstate__(self, state):
        """Restore pickle state and regenerate thermodynamic callables."""
        self.__dict__.update(state)
        self.hsx = None
        self._rebuild_thermodynamic_expressions()

    @classmethod
    @config._removed_keywords({"input": "system", "comp_range_fit_lim": "assessed_crange_floor"})
    def from_cache(
        cls,
        system,
        dft_type="GGA",
        pd_ind=None,
        params=(),
        param_format="comb-exp",
        assessed_crange_floor=None,
        include_imputed=False,
        solid_solutions: bool | None = None,
        ss_kwargs=None,
        tau=DEFAULT_TAU,
        liq_max_gap=None,
        liq_min_coverage=None,
        hull: PhaseDiagram | None = None,
    ) -> BinaryLiquid:
        """
        Initializes a BinaryLiquid object from cached data.

        Args:
            system (any): Binary system - can be either a list or hyphenated string
            dft_type (str): Type of DFT calculation.
            pd_ind (int | None): Index of the MPDS binary phase diagram in the cache.
            params (list): Initial fitting parameters.
            param_format (str): Format of the excess mixing energy params: 'comb-exp' (default),
                'linear' or 'combined'.
            assessed_crange_floor (float | None): Per-call override of
                ``config.assessed_crange_floor``, the smallest fraction of the composition
                axis that must carry digitized liquidus (outside ``undigitized_comp_ranges``).
                Below it, ``init_error`` is flagged with reason ``SKIP_NARROW_SPAN``.
            include_imputed (bool): Passed to ``api.get_dft_convexhull``. True also admits
                entries cached by ``api.cache_imputed_entries``.
            liq_max_gap (float | None): Per-call override of ``config.liquidus_max_gap`` for
                the interior-coverage gate: flag ``init_error`` when the widest composition
                gap between consecutive PRE-fill digitized liquidus points exceeds this.
            liq_min_coverage (float | None): Per-call override of
                ``config.liquidus_min_coverage``: flag ``init_error`` when less than this
                fraction of the stitched liquidus span is locally sampled
                (``mpds.liquidus_coverage`` with ``config.liquidus_gap_tol``).
            solid_solutions (bool | None): None (default) defers to
                ``config.solid_solutions``. When resolved True, loads per-phase solid-solution
                models (``solution.load_solid_solution_models``), reconciles each element's
                liquid reference against the model's solid ladder, and represents the covered
                structures as continuous solid-solution phases instead of endpoint line
                compounds. A system the omegas file does not cover loads no models.
            ss_kwargs (dict | None): Passed through to load_solid_solution_models
                (e.g. ref_mode, omegas_path, entries).
            tau (float): exp(-T/tau) decay constant for 'combined'/'comb-exp' mixing
                expressions (default 8000 K).
            hull (PhaseDiagram | None): The solid hull to build the phases from, in place of
                the stored DFT hull, which is then not read; ``include_imputed`` has no
                effect. None (default) reads the stored hull.

        Returns:
            BinaryLiquid: Initialized BinaryLiquid object.
        """
        components, sys_name, order_changed = validate_and_format_system(system)
        if len(components) != 2:
            raise ValueError(f"BinaryLiquid needs exactly 2 components, got {components}.")
        xs_mix = RKPolyExp(param_format, params, tau=tau)
        if order_changed:  # Odd L orders flip sign when the component order changes
            xs_mix = xs_mix.swapped()

        if hull is None:
            hull, _ = api.get_dft_convexhull(components, dft_type, include_imputed=include_imputed)
        ch = hull
        component_data = UNARY.component_data(components)
        for comp, ref in component_data.items():
            logger.info(
                f"{comp}: H_liq = {ref.h_liq} J/mol, S_liq = {ref.s_liq:.4f} J/(mol·K), "
                f"T_fusion = {ref.t_fusion} K, polymorphs = {len(ref.polymorphs)}"
            )
        mpds_json, (digitized_liq, is_partial) = mpds.load_mpds_data(components, pd_ind=pd_ind)
        # mpds_json stays in its own frame; the artifacts derived from it are mirrored.
        mpds_frame_reversed = not mpds.mpds_frame_matches(mpds_json, components)
        if digitized_liq and mpds_frame_reversed:
            digitized_liq = mpds.mirror_liquidus(digitized_liq)

        ss_models = {}
        resolved_solid_solutions = (
            config.solid_solutions if solid_solutions is None else solid_solutions
        )
        if resolved_solid_solutions:
            ss_kwargs = dict(ss_kwargs or {})
            ss_kwargs.setdefault("ref_mode", config.ss_ref_mode)
            if ss_kwargs.get("ref_mode") == "from_dft_entries" and "entries" not in ss_kwargs:
                ss_kwargs["entries"] = api.get_dft_structure_entries(components, dft_type)
            # Reconciles the liquid references inside component_data in place.
            ss_models = solution.load_solid_solution_models(components, component_data, **ss_kwargs)

        phases = build_phases_from_chull(
            ch,
            components,
            component_data,
            exclude_spacegroups={solution.SS_SPACEGROUPS[p] for p in ss_models} or None,
        )

        if "temp" in mpds_json:
            temp_range = [mpds_json["temp"][0] + 273.15, mpds_json["temp"][1] + 273.15]
        else:
            comp_tms = [component_data[comp].t_fusion for comp in components]
            temp_range = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        eqs = SolutionModel(
            tuple(components),
            (
                component_data[components[0]].gibbs_ref_expr(t_sym),
                component_data[components[1]].gibbs_ref_expr(t_sym),
            ),
            {(0, 1): xs_mix},
        ).binary_eqs()

        comp_range = mpds_json.get("comp_range", [0, 100])
        if mpds_frame_reversed:
            comp_range = [100 - comp_range[1], 100 - comp_range[0]]
        comp_range = (
            [
                max(min(digitized_liq, key=lambda x: x[0])[0], comp_range[0] / 100.0),
                min(max(digitized_liq, key=lambda x: x[0])[0], comp_range[1] / 100.0),
            ]
            if digitized_liq
            else comp_range
        )
        liq_coverage = mpds.liquidus_coverage(mpds_json) if digitized_liq else None
        if liq_coverage is not None and mpds_frame_reversed:
            liq_coverage = mpds.mirror_liquidus_coverage(liq_coverage)

        assessed_floor = (
            config.assessed_crange_floor
            if assessed_crange_floor is None
            else float(assessed_crange_floor)
        )
        ignored_ranges = mpds.ignored_comp_ranges(
            comp_range if digitized_liq else None, liq_coverage
        )
        assessed_crange = 1.0 - sum(b - a for a, b in ignored_ranges)
        init_error = not bool(digitized_liq) or bool(assessed_crange < assessed_floor)
        skip_reason = None
        if init_error:
            skip_reason = SKIP_NO_LIQUIDUS if not digitized_liq else SKIP_NARROW_SPAN

        if not init_error and liq_coverage is not None:
            max_gap_lim = config.liquidus_max_gap if liq_max_gap is None else float(liq_max_gap)
            min_coverage = (
                config.liquidus_min_coverage
                if liq_min_coverage is None
                else float(liq_min_coverage)
            )
            if (
                liq_coverage["max_gap"] > max_gap_lim
                or liq_coverage["covered_fraction"] < min_coverage
            ):
                logger.warning(
                    f"Digitized liquidus is interior-sparse [{sys_name}]: max gap "
                    f"{liq_coverage['max_gap']:.2f}, covered {liq_coverage['covered_fraction']:.0%} "
                    f"of span {liq_coverage['span']:.2f} (limits: max gap <= {max_gap_lim:.2f}, "
                    f"covered >= {min_coverage:.0%}); flagging init_error."
                )
                init_error = True
                skip_reason = SKIP_SPARSE_LIQUIDUS

        kwargs = {
            "mpds_json": mpds_json,
            "component_data": component_data,
            "digitized_liq": digitized_liq,
            "temp_range": temp_range,
            "dft_type": dft_type,
            "dft_ch": ch,
            "phases": phases,
            "xs_mix": xs_mix,
            "eqs": eqs,
            "pd_ind": pd_ind,
            "assessed_crange_floor": assessed_crange_floor,
            "undigitized_comp_ranges": ignored_ranges,
            "liq_coverage": liq_coverage,
            "skip_reason": skip_reason,
            "ss_models": ss_models,
        }
        bl = cls(sys_name, components, init_error, **kwargs)
        bl.rebuild_hull_interp()
        return bl

    def _ensure_solid_solution_phases(self) -> None:
        """Insert a continuous solid-solution Phase (model attached) per ss_model before 'L'."""
        insert_idx = max(len(self.phases) - 1, 0)
        for ss_name, ss_model in self.ss_models.items():
            if any(phase.name == ss_name for phase in self.phases):
                continue
            self.phases.insert(
                insert_idx,
                Phase(
                    phase_type="solid",
                    name=ss_name,
                    model=SolutionModel.from_ss_model(self.components, ss_model),
                ),
            )
            insert_idx += 1

    def with_component_order(self, order) -> BinaryLiquid:
        """A copy of this system re-framed onto ``order`` (any spec resolve_component_order
        accepts). Returns ``self`` when the order already matches.

        xs_mix swaps odd RK orders, the digitized liquidus and every composition range
        mirror (``fit_unsupported_range_mode`` travels with them), and the hull rebuilds with
        reordered elements. Fitting state (nmpath, invariants, constraints) does not transfer;
        fit in the construction frame.
        """
        order = resolve_component_order(order, self.components)
        if list(order) == list(self.components):
            return self
        mirrored_liq = mpds.mirror_liquidus(self.digitized_liq) if self.digitized_liq else []
        ch = self.dft_ch
        if ch is not None:
            ch = PhaseDiagram(ch.all_entries, elements=[Element(c) for c in order])
        clone = BinaryLiquid(
            "-".join(order),
            list(order),
            init_error=self.init_error,
            skip_reason=self.skip_reason,
            component_data={c: ref.copy() for c, ref in self.component_data.items()},
            mpds_json=self.mpds_json,
            digitized_liq=mirrored_liq,
            temp_range=list(self.temp_range),
            assessed_crange_floor=self.assessed_crange_floor,
            undigitized_comp_ranges=[
                [1 - b, 1 - a] for a, b in reversed(self.undigitized_comp_ranges or [])
            ],
            unsupported_comp_ranges=[
                [1 - hi, 1 - lo] for lo, hi in reversed(self.unsupported_comp_ranges or [])
            ],
            liq_coverage=(
                mpds.mirror_liquidus_coverage(self.liq_coverage)
                if self.liq_coverage
                else self.liq_coverage
            ),
            dft_type=self.dft_type,
            dft_ch=ch,
            removed_entry_ids=self.removed_entry_ids,
            # the constructor re-creates the SS phases in the new frame
            phases=[replace(p, points=[]) for p in self.phases if p.name not in self.ss_models],
            ss_models=self.ss_models,
            xs_mix=self.xs_mix.swapped(),
            pd_ind=self.pd_ind,
        )
        clone._rebuild_thermodynamic_expressions()
        clone.fit_unsupported_range_mode = self.fit_unsupported_range_mode
        return clone

    def solid_solution_h_s(
        self, ss_name: str, x_vals: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """H(x) and S(x) of a solid-solution phase across the composition grid, from the
        phase's T-independent regular-solution SolutionModel."""
        ss_model = self.ss_models[ss_name]  # unknown phase names raise KeyError here
        phase = next((p for p in self.phases if p.name == ss_name and p.model is not None), None)
        model = (
            phase.model
            if phase is not None
            else SolutionModel.from_ss_model(self.components, ss_model)
        )
        x_arr = np.asarray(_x_vals if x_vals is None else x_vals, dtype=float)
        return model.h_s_grid((x_arr,), self.mean_elt_tm)

    def solid_solution_gibbs(self, ss_name: str, x_vals: np.ndarray, temp_k: float) -> np.ndarray:
        """G(x) = H(x) - T*S(x) of a solid-solution phase at ``temp_k``."""
        h_vals, s_vals = self.solid_solution_h_s(ss_name, x_vals=x_vals)
        return h_vals - float(temp_k) * s_vals

    def _solution_phase_h_s(
        self, name: str, x_vals: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Grid H/S for a continuous solid phase, or None if ``name`` carries no SS model."""
        if name in self.ss_models:
            return self.solid_solution_h_s(name, x_vals=x_vals)
        return None

    def _build_hsx_data(
        self,
        h_a: float,
        h_b: float,
        s_a: float,
        s_b: float,
        liq_h_vals: list,
        liq_s_vals: list,
        x_vals: np.ndarray | None = None,
        ss_x_vals: np.ndarray | None = None,
    ) -> dict:
        """
        Assemble the HSX data dictionary from endpoint and interior liquid H/S values.

        Args:
            h_a (float): Enthalpy of pure component A liquid (J/mol).
            h_b (float): Enthalpy of pure component B liquid (J/mol).
            s_a (float): Entropy of pure component A liquid (J/mol/K), 0 if H_liq is 0 (reference state).
            s_b (float): Entropy of pure component B liquid (J/mol/K), 0 if H_liq is 0 (reference state).
            liq_h_vals (list): Enthalpy values for interior liquid compositions.
            liq_s_vals (list): Entropy values for interior liquid compositions.
            x_vals (np.ndarray | None): Composition grid the liquid rows sit on. Defaults
                to the module grid (``solution.x_vals``).
            ss_x_vals (np.ndarray | None): Composition grid for the continuous
                solid-solution rows. Defaults to ``x_vals``.

        Returns:
            dict: Data dictionary in HSX format with keys 'X', 'S', 'H', 'Phase Name'.
        """
        grid = _x_vals if x_vals is None else np.asarray(x_vals, dtype=float)
        ss_grid = grid if ss_x_vals is None else np.asarray(ss_x_vals, dtype=float)
        data = {
            "X": [float(x) for x in grid],
            "S": [s_a] + list(liq_s_vals) + [s_b],
            "H": [h_a] + list(liq_h_vals) + [h_b],
            "Phase Name": ["L"] * len(grid),
        }
        for phase in self.phases:
            if phase.is_solution:
                # one row per grid point for a modelled solid solution; none for 'L'
                if phase.name != "L" and getattr(self, "ss_in_hull", True):
                    solution_h_s = self._solution_phase_h_s(phase.name, x_vals=ss_grid)
                    if solution_h_s is not None:
                        ss_h_vals, ss_s_vals = solution_h_s
                        data["X"].extend(float(x) for x in ss_grid)
                        data["S"].extend(float(s) for s in ss_s_vals)
                        data["H"].extend(float(h) for h in ss_h_vals)
                        data["Phase Name"].extend([phase.name] * len(ss_grid))
                continue
            data["H"].append(phase.enthalpy)
            data["S"].append(phase.entropy)
            # exact composition: rounding onto the grid can lift a phase above its tie line
            data["X"].append(float(self._phase_x(phase)))
            data["Phase Name"].append(phase.name)
        return data

    def to_hsx(
        self, fmt="dict", x_vals: np.ndarray | None = None, ss_x_vals: np.ndarray | None = None
    ) -> dict | pd.DataFrame:
        """
        Converts phase data into HSX format.

        Each liquid row's H and S are evaluated at its linearization temperature:
        ``liquid_tstar`` (interpolated onto ``x_vals``), or ``linear_tstar`` before the first
        ``update_phase_points``. The 'combined' and 'comb-exp' forms depend on it, 'linear' does
        not.

        Args:
            fmt (str): Output format ('dict' or 'dataframe').
            x_vals (np.ndarray | None): Composition grid the liquid rows are evaluated on.
                Defaults to the module grid (``solution.x_vals``).
            ss_x_vals (np.ndarray | None): Composition grid for the continuous
                solid-solution phases. Defaults to ``x_vals``.

        Returns:
            dict | pd.DataFrame: Data in HSX format.
        """
        grid = _x_vals if x_vals is None else np.asarray(x_vals, dtype=float)
        x_inner = grid[1:-1]
        params = self.xs_mix.values

        # endpoint H and S: pure-element fusion enthalpy and entropy
        comp_a = self.component_data[self.components[0]]
        comp_b = self.component_data[self.components[1]]
        s_a = 0 if comp_a.h_liq == 0 else comp_a.s_liq
        s_b = 0 if comp_b.h_liq == 0 else comp_b.s_liq
        h_a = comp_a.h_liq
        h_b = comp_b.h_liq

        if self.liquid_tstar is None:
            t_star = self.linear_tstar(x_inner)
        else:
            t_star = np.interp(x_inner, _x_vals[1:-1], self.liquid_tstar)
        liq_h_vals = np.ravel(self.eqs["h_liq_lambdified"](x_inner, t_star, *params)).tolist()
        liq_s_vals = np.ravel(self.eqs["s_liq_lambdified"](x_inner, t_star, *params)).tolist()

        data = self._build_hsx_data(
            h_a, h_b, s_a, s_b, liq_h_vals, liq_s_vals, x_vals=grid, ss_x_vals=ss_x_vals
        )

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            return pd.DataFrame(data)
        else:
            raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")

    def update_phase_points(self) -> dict:
        """
        Calculates every phase's points for the current parameters with ``self.hsx``.

        The liquid's H and S are evaluated row by row at a linearization temperature T*(x). The
        first hull takes ``linear_tstar``. Each further hull re-linearizes every row at the
        previous hull's liquidus, the step halved where it grew since the last hull, until
        ``liquid_tstar_error`` is at most ``DEFAULT_TSTAR_TOL`` and ``liquidus_shift_estimate``
        at most ``DEFAULT_TSTAR_LIQ_TOL`` on every row, or ``DEFAULT_TSTAR_MAX_HULLS`` hulls are
        built. ``liquid_tstar`` records the last hull's.

        ``self.hsx`` is updated in place, and rebuilt only when absent or when the hull's
        phase list changed.

        Returns:
            data (dict): A dictionary containing the phase data in HSX format, including phase names and components.
        """
        phase_names = self._hsx_phase_names()
        conds = [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15]
        grid = _x_vals[1:-1]
        self.liquid_tstar = self.linear_tstar(grid)
        prev_step = None
        for k in range(DEFAULT_TSTAR_MAX_HULLS):
            data = self.to_hsx()
            if self.hsx is None or self.hsx.phases != phase_names:
                self.hsx = HSX(
                    {"data": data, "phases": phase_names, "comps": self.components}, conds
                )
            else:
                self.hsx.conds = conds
                self.hsx.set_data(data)
            phase_points = self.hsx.get_phase_points()
            if k + 1 == DEFAULT_TSTAR_MAX_HULLS:
                break
            liq = _grid_liquidus(phase_points.get("L"), grid)
            errors = self.liquid_tstar_row_errors(liq)
            if (
                errors.max() <= DEFAULT_TSTAR_TOL
                and self.liquidus_shift_estimate(data, liq, errors).max() <= DEFAULT_TSTAR_LIQ_TOL
            ):
                break
            step = np.where(np.isfinite(liq), liq - self.liquid_tstar, 0.0)
            if prev_step is not None:
                step = np.where(np.abs(step) > np.abs(prev_step), 0.5 * step, step)
            self.liquid_tstar = self.liquid_tstar + step
            prev_step = step
        for phase in self.phases:
            phase.points = phase_points.get(phase.name, [])
        return data

    def linear_tstar(self, x) -> np.ndarray:
        """The x-weighted component melting point (K) at fractions ``x`` of ``components[1]``."""
        x = np.asarray(x, dtype=float)
        t_a, t_b = (float(self.component_data[c].t_fusion) for c in self.components)
        return (1 - x) * t_a + x * t_b

    def liquid_tstar_error(self, liq: np.ndarray) -> float:
        """Largest |G_lin(T_liq; T*) - G(T_liq)| (J/mol) of the liquid over the interior rows.

        The largest of ``liquid_tstar_row_errors``.

        Args:
            liq: liquidus temperature (K) of each interior grid row; NaN where there is none.

        Returns:
            float: the largest difference; 0 when no row counts.
        """
        return float(self.liquid_tstar_row_errors(liq).max())

    def liquid_tstar_row_errors(self, liq: np.ndarray) -> np.ndarray:
        """|G_lin(T_liq; T*) - G(T_liq)| (J/mol) of the liquid on each interior row.

        ``G_lin`` takes H and S at ``liquid_tstar``, ``G`` at ``liq``. Rows outside the
        digitized liquidus's composition span, and rows where either temperature is not finite,
        report 0.

        Args:
            liq: liquidus temperature (K) of each interior grid row; NaN where there is none.

        Returns:
            np.ndarray: one difference per interior grid row.
        """
        grid = _x_vals[1:-1]
        t_star = np.asarray(self.liquid_tstar, dtype=float)
        liq = np.asarray(liq, dtype=float)
        errors = np.zeros(len(grid))
        ok = np.isfinite(liq) & np.isfinite(t_star)
        dig = np.asarray(self.digitized_liq or [], dtype=float).reshape(-1, 2)
        if len(dig):
            ok &= (grid >= dig[:, 0].min()) & (grid <= dig[:, 0].max())
        if not ok.any():
            return errors
        xs, ts, tl = grid[ok], t_star[ok], liq[ok]
        params = self.xs_mix.values
        h_fn, s_fn = self.eqs["h_liq_lambdified"], self.eqs["s_liq_lambdified"]
        shape = xs.shape
        g_lin = np.broadcast_to(np.asarray(h_fn(xs, ts, *params), float), shape) - tl * (
            np.broadcast_to(np.asarray(s_fn(xs, ts, *params), float), shape)
        )
        g_ex = np.broadcast_to(np.asarray(h_fn(xs, tl, *params), float), shape) - tl * (
            np.broadcast_to(np.asarray(s_fn(xs, tl, *params), float), shape)
        )
        errors[ok] = np.abs(g_lin - g_ex)
        return errors

    def liquidus_shift_estimate(
        self, data: dict, liq: np.ndarray, errors: np.ndarray, min_error: float = 1e-3
    ) -> np.ndarray:
        """Estimated liquidus shift (K) of each interior row when it is re-linearized at ``liq``.

        A row's ``errors`` entry over the entropy step between the row and the hull chord it
        leaves at its liquidus. The chord joins the lower-hull vertices either side of the row
        just below ``liq``, from ``data``'s H and S.

        Args:
            data: the HSX data the hull giving ``liq`` was built from (``to_hsx``).
            liq: liquidus temperature (K) of each interior grid row; NaN where there is none.
            errors: ``liquid_tstar_row_errors(liq)``.
            min_error: rows whose error is at most this (J/mol) report 0.

        Returns:
            np.ndarray: one shift per interior grid row.
        """
        shifts = np.zeros(len(errors))
        xs = np.asarray(data["X"], dtype=float)
        hs = np.asarray(data["H"], dtype=float)
        ss = np.asarray(data["S"], dtype=float)
        for i in np.flatnonzero((errors > min_error) & np.isfinite(liq)):
            row = i + 1
            gs = hs - (liq[i] - 0.01) * ss
            others = np.arange(len(xs)) != row
            left = np.flatnonzero(others & (xs <= xs[row]))
            right = np.flatnonzero(others & (xs >= xs[row]))
            if not len(left) or not len(right):
                continue
            span = xs[right][None, :] - xs[left][:, None]
            frac = np.divide(
                xs[row] - xs[left][:, None], span, out=np.zeros(span.shape), where=span > 0
            )
            chord = gs[left][:, None] + frac * (gs[right][None, :] - gs[left][:, None])
            a, b = np.unravel_index(np.argmin(chord), chord.shape)
            s_chord = (1 - frac[a, b]) * ss[left[a]] + frac[a, b] * ss[right[b]]
            shifts[i] = errors[i] / max(abs(ss[row] - s_chord), 1e-6)
        return shifts

    def _hsx_phase_names(self) -> list[str]:
        """Phase names entering the hull; SS names drop when ``ss_in_hull`` is off."""
        ss_in_hull = getattr(self, "ss_in_hull", True)
        return [
            phase.name
            for phase in self.phases
            if ss_in_hull or not (phase.is_solution and phase.name != "L")
        ]

    def refined_hsx(self, factor: int = 5) -> HSX:
        """A fresh presentation hull with the solid-solution phases on a grid refined ``factor``
        times; the liquid keeps the coarse grid. ``self.hsx`` and ``phase.points`` are not
        touched, and ``factor=1`` reproduces the coarse hull."""
        data = self.to_hsx(ss_x_vals=solution.refined_x_vals(factor))
        conds = [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15]
        return HSX(
            {"data": data, "phases": self._hsx_phase_names(), "comps": self.components}, conds
        )

    def get_L0_a(self) -> int | float:
        return self.xs_mix["L0_a"]

    def get_L0_b(self) -> int | float:
        return self.xs_mix["L0_b"]

    def get_L1_a(self) -> int | float:
        return self.xs_mix["L1_a"]

    def get_L1_b(self) -> int | float:
        return self.xs_mix["L1_b"]

    def get_params(self) -> list[int | float]:
        """
        A copy of the current parameters.

        Returns:
            list: The mixing model's parameter values in its flat layout
            (``xs_mix.format.param_names``, e.g. [L0_a, L0_b, L1_a, L1_b]).
        """
        return self.xs_mix.values

    @config._removed_keywords({"input": "params"})
    def update_params(self, params) -> None:
        """
        Update the non-ideal mixing parameters with validity checks,
        then recalculate phase boundaries for the new parameters.

        Args:
            params (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

        Returns:
            None
        """
        self.xs_mix.update(params)
        self.update_phase_points()

    def find_invariant_points(
        self, verbose=False, check_full_ss=True, t_tol=15
    ) -> tuple[list[dict], list[dict]]:
        """
        Identifies invariant points in the MPDS data using the provided MPDS JSON and liquidus data.

        Wraps :func:`gliquid.mpds.identify_invariant_points` and assigns ``self.invariants``
        and ``self.low_t_exp_phases``. A full-composition solid solution sets
        ``self.full_comp_ss``, not ``init_error``. DFT phases are not read.

        Args:
            verbose (bool): If True, outputs additional debugging information.
            t_tol (int): Temperature tolerance for invariant point identification.
            check_full_ss (bool): If True, identification stops early when the MPDS labels
                report a full-composition solid solution.

        Returns:
            tuple: A tuple containing two lists:
                - List of identified invariant points.
                - List of low-temperature phases from the MPDS JSON.
        """
        if self.mpds_json["reference"] is None:
            logger.warning("System JSON does not contain any data!")
            return [], []

        invariants, low_t_exp_phases, full_comp_ss = mpds.identify_invariant_points(
            self.mpds_json,
            self.components,
            self.digitized_liq,
            self.component_data,
            self.temp_range,
            verbose=verbose,
            check_full_ss=check_full_ss,
            t_tol=t_tol,
            liquidus_holes=self.liquidus_holes,
        )
        self.invariants = invariants
        self.low_t_exp_phases = low_t_exp_phases
        if full_comp_ss:
            self.full_comp_ss = True
        return self.invariants, self.low_t_exp_phases

    def coverage_phase_inputs(
        self, polymorph_pair_tol: float | None = None
    ) -> tuple[list[dict], frozenset[tuple[str, float]]]:
        """The MPDS phase records and unpaired-ht keys the coverage rules read, in the
        instance's component frame.

        Args:
            polymorph_pair_tol: Passed to :func:`gliquid.mpds.mark_unpaired_ht_forms`;
                ``None`` defers to ``config``.

        Returns:
            tuple: the phase records and the unpaired high-temperature keys.
        """
        phases = mpds.identify_mpds_phases(
            self.mpds_json, with_structure=True, with_polymorph=True, elements=self.components
        )
        # Read off the raw shapes in the json's frame, then mirrored alongside them.
        unpaired_ht = mpds.mark_unpaired_ht_forms(
            phases,
            self.components,
            pair_tol=polymorph_pair_tol,
            mpds_json=self.mpds_json,
        )
        if not mpds.mpds_frame_matches(self.mpds_json, self.components):
            phases = mpds.mirror_mpds_phases(phases)
            unpaired_ht = frozenset((base, 1 - comp) for base, comp in unpaired_ht)
        return phases, unpaired_ht

    def _interior_dft_comps(self) -> list[float]:
        """Interior (0 < x < 1) compositions of the fixed-composition DFT phases."""
        return [
            self._phase_x(p)
            for p in self.phases
            if not p.is_solution and 0.0 < round(self._phase_x(p), 6) < 1.0
        ]

    def dft_spacegroups(self) -> dict[float, int | None]:
        """Space-group number of each interior DFT hull phase, keyed by its composition.

        Read from the stable entry's structure at a symmetry precision of 0.1 angstrom. An
        entry with no structure, or one :func:`gliquid.cache.non_dft_entry_kind` reports as
        imputed or user-defined, maps to None. Cached per entry id.

        Returns:
            dict: fraction of ``components[1]`` to space-group number or None.
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        if getattr(self, "dft_ch", None) is None:
            return {}
        known = getattr(self, "_dft_spacegroup_cache", None)
        if known is None:
            known = self._dft_spacegroup_cache = {}
        out = {}
        for entry in self.dft_ch.stable_entries:
            x = float(api.entry_frac_along(self.dft_ch, entry, self.components)[0])
            if not 0.0 < round(x, 6) < 1.0:
                continue
            entry_id = getattr(api.entry_original(entry), "entry_id", None)
            if entry_id is None:
                out[x] = None
                continue
            key = str(entry_id)
            if key not in known:
                structure = getattr(entry, "structure", None)
                if non_dft_entry_kind(entry) is not None:
                    structure = None
                try:
                    known[key] = (
                        None
                        if structure is None
                        else int(
                            SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
                        )
                    )
                except Exception:  # noqa: BLE001 - an unanalysable structure has no known group
                    known[key] = None
            out[x] = known[key]
        return out

    def dft_assignment(self) -> tuple[list[dict], dict[int, solid_coverage.DftMatch]]:
        """This system's MPDS phases and the DFT hull phases :func:`solid_coverage.assign_dft_phases`
        matches to them, under the ``config`` tolerances.

        Returns:
            tuple: :meth:`coverage_phase_inputs`' phase records, and phase index to
            :class:`gliquid.solid_coverage.DftMatch`.
        """
        phases, unpaired_ht = self.coverage_phase_inputs()
        assigned = solid_coverage.assign_dft_phases(
            phases,
            self.components,
            self._interior_dft_comps(),
            self.dft_spacegroups(),
            unpaired_ht=unpaired_ht,
            rescuable=solid_coverage.rescuable_ss_indices(phases, self.components),
        )
        return phases, assigned

    def assess_solid_coverage(self, **overrides) -> solid_coverage.SolidCoverageReport:
        """Measure how much of this system's liquidus has no solid free-energy reference.

        Wraps :func:`gliquid.solid_coverage.assess_solid_coverage` with this system's DFT hull
        compositions and solid-solution models; assigns and returns ``self.coverage_report``.
        Call :meth:`find_invariant_points` first (``fit_parameters`` does): without
        invariants the compound-field masking falls back to the nearest local minimum of
        ``self.digitized_liq``, then to the nearest supported anchor.

        Args:
            **overrides: Passed through (``liquidus``, ``dft_spacegroups``, ``unpaired_ht``,
                ``thresholds``, ``elements``). ``liquidus`` defaults to ``self.digitized_liq``,
                ``elements`` to ``self.components``, ``dft_spacegroups`` to
                :meth:`dft_spacegroups`, and ``unpaired_ht`` to
                :func:`gliquid.mpds.mark_unpaired_ht_forms` over this system's shapes under
                the ``polymorph_pair_tol`` in force.

        Raises:
            config.CacheModeError: when ``self.mpds_json`` is a LEAN record (see
                :func:`gliquid.mpds.require_full_record`).
        """
        mpds.require_full_record(self.mpds_json, "BinaryLiquid.assess_solid_coverage")
        settings = config.coverage_settings(overrides.get("thresholds"))
        phases, unpaired_ht = self.coverage_phase_inputs(
            polymorph_pair_tol=settings["polymorph_pair_tol"]
        )
        span = (
            [self.digitized_liq[0][0], self.digitized_liq[-1][0]]
            if self.digitized_liq
            else [0.0, 1.0]
        )
        overrides.setdefault("liquidus", self.digitized_liq)
        overrides.setdefault("elements", self.components)
        if overrides.get("dft_spacegroups") is None:
            overrides["dft_spacegroups"] = self.dft_spacegroups()
        overrides.setdefault("unpaired_ht", unpaired_ht)
        overrides.setdefault(
            "printed_gap_comps", mpds.two_liquid_label_comps(self.mpds_json, self.components)
        )
        overrides.setdefault(
            "ignored_ranges",
            self.undigitized_comp_ranges
            or mpds.ignored_comp_ranges(span if self.digitized_liq else None, self.liq_coverage),
        )
        self.coverage_report = solid_coverage.assess_solid_coverage(
            phases, self.invariants, span, self._interior_dft_comps(), self.ss_models, **overrides
        )
        return self.coverage_report

    def solve_params_from_constraints(self, guessed_vals: dict) -> None:
        """
        Updates the parameters of the object based on guessed values and constraints.

        Args:
            guessed_vals (dict): A dictionary containing guessed values for the parameters.
        """
        for ind, symbol in enumerate(self.xs_mix.format.symbols()):
            try:
                if symbol in guessed_vals:
                    self.xs_mix[ind] = float(guessed_vals[symbol])
                elif self.constraints:
                    self.xs_mix[ind] = float(self.constraints[symbol].subs(guessed_vals))
            except TypeError as exc:
                raise RuntimeError("Error in constraint equations!") from exc

    def lupis_elliott_penalty(self, penalty_cfg: dict | None = None) -> float:
        """
        Assigns a penalty which scales with degree of violation for Lupis-Elliott sign constraints.

        Active penalty term for each violating component uses:
            P = 1 + 2*A * sqrt(d) * |x*y| / (|x| + |y|)
        where d = x^2 + y^2, A is 'strength', x is normalized enthalpy, and y is entropy.

        Args:
            penalty_cfg (dict | None): Optional Lupis-Elliott penalty configuration.
                Supports shared defaults and optional per-term overrides:
                - 'strength' (float): global default strength, applied to L0. Default: 7.5E-3.
                - Per-term keys: 'l0', 'l1' each containing {'strength'}.
                  L1 defaults to 0 (not penalized) unless 'l1' is explicitly set.

        Returns:
            float: A penalty greater than 1.0 if the parameters violate the Lupis-Elliott sign constraints,
                   otherwise returns 1.0.

        Raises:
            TypeError: when the cfg or a per-term cfg carries the removed key 'scale'.
        """
        penalty_cfg = penalty_cfg or {}
        if "scale" in penalty_cfg or any(
            isinstance(term, dict) and "scale" in term for term in penalty_cfg.values()
        ):
            raise TypeError("lupis_elliott_cfg key 'scale' was removed in 0.3.0; use 'strength'.")
        global_strength = float(penalty_cfg.get("strength", 7.5e-3))
        entropy_scale = float(penalty_cfg.get("hs_ratio", getattr(self, "tau", DEFAULT_TAU)))

        def resolve_term_cfg(term: str, default: float | None = None) -> float:
            if default is None:
                default = global_strength
            term_cfg = penalty_cfg.get(term, {})
            if not isinstance(term_cfg, dict):
                term_cfg = {}
            return float(term_cfg.get("strength", default))

        def calculate_penalty(x, y, strength=0.005):
            if strength <= 0:
                return 0.0
            # x*y < 0 is a quick way to check for opposite signs
            if x * y < 0:
                abs_sum = abs(x) + abs(y)
                if abs_sum <= 1e-16:
                    return 0.0
                d = x**2 + y**2
                if d <= 0:
                    return 0.0
                alignment_factor = (2.0 * abs(x * y)) / (abs_sum**2)
                return float(strength * np.sqrt(d) * alignment_factor * abs_sum)
            return 0.0

        total = 0.0
        for order in self.xs_mix.format.lupis_orders:
            strength = resolve_term_cfg(f"l{order}", default=None if order == 0 else 0.0)
            args = [0, *self.xs_mix.order_values(order)]
            h_k = self.eqs[f"h_l{order}_lambdified"](*args)
            s_k = self.eqs[f"s_l{order}_lambdified"](*args)
            total += calculate_penalty(h_k / entropy_scale, s_k, strength)
        return float(1.0 + total)

    def lxb_penalty(self, penalty_cfg: dict | None = None) -> float:
        """
        Computes a multiplicative distribution-aware penalty on L0_b/L1_b.

        Each active term contributes to a shared factor of the form:

            1 + w * [log(1 + |x - median| / MAD)]^exponent

        The contributions are additive inside a single multiplicative factor:

            penalty = 1 + term(L0_b) + term(L1_b)

        Only L0_b is penalized unless ``apply_l1`` is True.

        Args:
            penalty_cfg (dict | None): Dictionary with optional keys:
                - 'l0': {'weight', 'median', 'mad', 'exponent'} — L0_b term config.
                  Defaults: weight=0.10, exponent=2.5. 'median' and 'mad' must be provided.
                - 'l1': {'weight', 'median', 'mad', 'exponent'} — L1_b term config.
                  Applied only when 'apply_l1' is True.
                - 'apply_l1' (bool): Force-enable/disable L1 term. Default: False.

        Returns:
            float: Multiplicative penalty factor >= 1.0.
        """
        if not penalty_cfg:
            return 1.0

        def _term(x_val: float, cfg: dict | None) -> float:
            if not cfg:
                return 0.0
            weight = float(cfg.get("weight", 0.10))
            median = float(cfg.get("median", 0.0))
            mad = float(cfg.get("mad", 0.0))
            exponent = float(cfg.get("exponent", 2.5))
            if weight <= 0 or mad <= 0:
                return 0.0
            log_term = math.log(1.0 + abs(x_val - median) / mad)
            return weight * (log_term**exponent)

        total = _term(self.get_L0_b(), penalty_cfg.get("l0"))
        if penalty_cfg.get("apply_l1", False):
            total += _term(self.get_L1_b(), penalty_cfg.get("l1"))
        return float(1.0 + total)

    def _stable_solid_gibbs_at_T(self, comp_x: float, temp_K: float) -> float:
        """
        Gibbs energy of the stable solid at ``(comp_x, temp_K)``.

        At a pure-element endpoint (comp_x 0 or 1) this is the lowest Gibbs energy among the
        ground state (0) and the polymorphs stable at ``temp_K``. Elsewhere it is the
        enthalpy of the nearest fixed-composition phase.

        Args:
            comp_x (float): Composition fraction of component B (0 or 1 for pure elements).
            temp_K (float): Temperature in Kelvin at which to evaluate stability.

        Returns:
            float: Gibbs energy (J/mol) of the stable solid at (comp_x, temp_K).
        """
        if comp_x not in (0.0, 1.0):
            nearest = min(
                (p for p in self.phases if not p.is_solution),
                key=lambda p: abs(self._phase_x(p) - comp_x),
                default=None,
            )
            return nearest.enthalpy if nearest else 0.0

        comp_name = self.components[int(comp_x)]
        polymorphs = self.component_data[comp_name].polymorphs

        best_g = 0.0

        for poly in polymorphs:
            if poly.t_transition is None or temp_K < poly.t_transition:
                continue  # This polymorph is not yet stable
            g_poly = poly.gibbs(temp_K)
            if g_poly < best_g:
                best_g = g_poly

        return best_g

    def h0_below_ch(self, tol=1e-6) -> bool:
        """
        Checks if the liquid enthalpy curve at T=0K falls below the solid convex hull.

        Args:
            tol (float): Minimum distance that liquid enthalpy curve must be above the solid convex hull.

        Returns:
            bool: True if any part of the liquid enthalpy curve falls below the solid
                  convex hull, False otherwise.
        """
        # Calculate the liquid enthalpy at T=0K.
        lambda_args_vals = [
            _x_vals[1:-1],
            0,
            self.get_L0_a(),
            self.get_L0_b(),
            self.get_L1_a(),
            self.get_L1_b(),
        ]
        h_vals_0k = self.eqs["h_liq_lambdified"](*lambda_args_vals)
        return np.any(h_vals_0k < self.eqs["h_hull_interp"] + tol)

    def liquidus_is_continuous(self, tol=2 * _x_step) -> bool:
        """
        Checks if the liquidus line is continuous within a given tolerance.

        Args:
            tol (float): Tolerance for liquidus continuity. Default is twice the step size.

        Returns:
            bool: True if the generated liquidus line is compositionally-continuous, False otherwise.
        """
        last_coords = None
        for coords in self.phases[-1].points:
            # the slack absorbs float error in grid differences; without it one missing row fails
            if last_coords and coords[0] - last_coords[0] > tol + 1e-9:
                return False
            last_coords = coords
        return True

    def calculate_deviation_metrics(
        self, num_points=30, **kwargs
    ) -> tuple[float, float, float, float]:
        """
        Calculates the deviation metrics between the digitized (measured) liquidus and the generated liquidus.

        ``self.liquidus_holes`` is always dropped from the mesh and from both curves;
        ``self.unsupported_comp_ranges`` is treated per ``unsupported_range_mode`` (see
        ``UNSUPPORTED_RANGE_MODES``). Reports ``inf`` when ``init_error`` is set, when masking
        leaves no liquidus, when the compared span is below ``min_comp_span``, when the mesh
        has fewer than 10 points, or when every mesh point is capped.

        Args:
            num_points (int): Number of points to sample for deviation metrics. Default is 30.
            **kwargs: Additional keyword arguments:
                - unsupported_range_mode (str): 'ignore', 'cap' or 'score' (default).
                - allow_sparse_data (bool): If True, waives the 10-point floor; an empty mesh
                    still reports ``inf``.
                - min_comp_span (float | None): Minimum composition span the comparison
                    must cover; None uses ``_OBJECTIVE_MIN_SPAN``. Pass 0 with
                    ``allow_sparse_data`` to score a few measured points over any span.

        Returns:
            tuple: MAE, RMSE, MAPE, RMSPE of the liquidus in temperature units (K) and percentage (%).

        Raises:
            ValueError: for an unrecognized ``unsupported_range_mode``, on every call.
        """
        mode = resolve_unsupported_range_mode(kwargs)

        if self.init_error or not self.digitized_liq:
            return (float("inf"),) * 4
        digitized = np.array(self.digitized_liq)
        if not self.phases[-1].points:
            self.update_phase_points()
        generated = np.array(self.phases[-1].points)

        # Filter out endpoint compositions (x=0 and x=1)
        mask = (digitized[:, 0] != 0) & (digitized[:, 0] != 1)
        if not np.any(mask):
            return (float("inf"),) * 4
        digitized = digitized[mask]

        digitized_liq_lims = [digitized[0, 0], digitized[-1, 0]]
        capped_ranges = [list(r) for r in self.unsupported_comp_ranges] if mode == "cap" else []
        masked_ranges = list(self.unsupported_comp_ranges) if mode == "ignore" else []
        masked_ranges += self.liquidus_holes
        if masked_ranges:
            dig_mask = np.ones_like(digitized[:, 0], dtype=bool)
            gen_mask = np.ones_like(generated[:, 0], dtype=bool)
            for lower, upper in masked_ranges:
                gen_mask = gen_mask & ((generated[:, 0] < lower) | (generated[:, 0] > upper))
                dig_mask = dig_mask & ((digitized[:, 0] < lower) | (digitized[:, 0] > upper))
            if not np.any(gen_mask) or not np.any(dig_mask):
                logger.error(f"Error: Masking leaves no liquidus to compare [{self.sys_name}].")
                return (float("inf"),) * 4
            generated = generated[gen_mask]
            digitized = digitized[dig_mask]
            masked_comp_lims = [generated[0, 0], generated[-1, 0]]
        else:
            masked_comp_lims = [0, 1]

        x_min = max(_x_step, digitized_liq_lims[0], masked_comp_lims[0])
        x_max = min(1 - _x_step, digitized_liq_lims[1], masked_comp_lims[1])
        min_span = kwargs.get("min_comp_span")
        if min_span is None:
            min_span = _OBJECTIVE_MIN_SPAN
        if x_max - x_min < min_span:
            logger.error(
                f"Error: Large composition range not considered [{self.sys_name}] (remaining range = {[float(x_min), float(x_max)]})"
            )
            return (float("inf"),) * 4

        x_mesh = np.linspace(x_min, x_max, num_points)
        mesh_mask = np.ones_like(x_mesh, dtype=bool)
        for lower, upper in masked_ranges:
            mesh_mask = mesh_mask & ((x_mesh < lower) | (x_mesh > upper))
        x_mesh = x_mesh[mesh_mask]

        if len(digitized) < len(x_mesh):
            x_mesh = np.array([pt[0] for pt in digitized])

        if len(x_mesh) == 0 or (not kwargs.get("allow_sparse_data", False) and len(x_mesh) < 10):
            logger.error(
                f"Error: Not enough comparison points for accurate deviation metrics calculation [{self.sys_name}]."
            )
            return (float("inf"),) * 4

        capped = np.zeros_like(x_mesh, dtype=bool)
        for lower, upper in capped_ranges:
            capped = capped | ((x_mesh >= lower) & (x_mesh <= upper))
        if capped_ranges and np.all(capped):
            logger.error(
                f"Error: Every comparison point is one-sided [{self.sys_name}] (unsupported ranges cover the mesh)."
            )
            return (float("inf"),) * 4

        # Find closest temperature values for each evaluation point
        Y1 = np.array([digitized[np.argmin(np.abs(digitized[:, 0] - x)), 1] for x in x_mesh])
        Y2 = np.array([generated[np.argmin(np.abs(generated[:, 0] - x)), 1] for x in x_mesh])
        # capped points count only a model above the data
        signed = Y2 - Y1
        diffs = np.where(capped, np.maximum(signed, 0.0), np.abs(signed))
        pdiffs = diffs / Y1 * 100
        return (
            float(np.mean(diffs)),
            float(np.sqrt(np.mean(diffs**2))),
            float(np.mean(pdiffs)),
            float(np.sqrt(np.mean(pdiffs**2))),
        )

    def f(self, guess: list | tuple, **kwargs) -> float:
        """
        Objective function for parameter fitting.

        Args:
            guess (list | tuple): Guessed parameter values to evaluate.
            **kwargs (dict): The fit options ``fit_parameters`` forwards (``FIT_KWARGS``).

        Returns:
            float: The liquidus RMSE times the Lupis-Elliott, lxb and unexpected-gap penalty
            factors; ``inf`` when the liquid falls below the hull at 0 K, the hull fails, or
            the generated liquidus is discontinuous.
        """
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, guess)}
        self.solve_params_from_constraints(guess_dict)

        if self.h0_below_ch():
            return float("inf")

        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            logger.error(e)
            return float("inf")

        if not self.liquidus_is_continuous():
            return float("inf")

        gap_factor = 1.0
        if kwargs.get("suppress_unexpected_migs", True):
            gap_factor = (1.0 + self.unexpected_gap_width() / GAP_WIDTH_SCALE) ** GAP_EXPONENT

        _, f_val, _, _ = self.calculate_deviation_metrics(**kwargs)
        if kwargs.get("check_lupis_elliott", True):
            f_val = f_val * self.lupis_elliott_penalty(kwargs.get("lupis_elliott_cfg"))

        if kwargs.get("use_lxb_penalty", self.xs_mix.format.lxb_default):
            f_val *= self.lxb_penalty(kwargs.get("lxb_penalty_cfg"))

        return f_val * gap_factor

    def nelder_mead(
        self, max_iter=64, tol=0.05, verbose=False, initial_guesses=None, **kwargs
    ) -> tuple[float, dict, np.ndarray]:
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Reflection is accepted when the reflected objective lies between the best and the
        second-worst vertex; expansion, outside and inside contraction and shrink follow with
        coefficients 1, 2, 1/2 and 1/2, a shrink moving every vertex but the best halfway to it.
        A vertex whose objective is infinite ranks worst; infinite vertices of the INITIAL
        simplex are first moved onto the feasible side of the constraint that rejected them.
        Convergence is ``max|x - centroid| < tol`` over the simplex, the centroid taken over
        every vertex but the worst; at ``max_iter`` iterations the best vertex is returned.

        Args:
            max_iter (int): Maximum number of iterations. Default is 64.
            tol (float): Tolerance for algorithm convergence. Default is 0.05.
            verbose (bool): If True, print updates during optimization. Default is False.
            initial_guesses (list): Reasonable initial values for guessed parameters, determined by self.guess_symbols.
            **kwargs (dict): Additional keyworded arguments passed to f().

        Returns:
            tuple: ``(objective_value, metrics, nmpath)``; ``objective_value`` is ``f`` at the
            best vertex. ``metrics`` holds eight numbers:
            ``mae``/``rmse``/``mape``/``rmspe`` scored over the FULL digitized liquidus, and
            ``mae_window``/``rmse_window``/``mape_window``/``rmspe_window`` scored over the
            window the optimizer actually saw.
        """
        if not initial_guesses:
            initial_guesses = self.init_triangle
        if self.guess_symbols is not None and len(self.guess_symbols) != 2:
            raise NotImplementedError(
                f"nelder_mead optimizes a 2-D simplex; got {len(self.guess_symbols)} guess "
                f"symbols. Formats with a different free-parameter count need a "
                f"generalized optimizer."
            )

        x0 = np.array(initial_guesses, dtype=float)
        n_params = self.xs_mix.n_params
        self.nmpath = np.empty((3, n_params + 1, max_iter), dtype=float)
        initial_time = time.time()

        if verbose:
            logger.info("--- Beginning Nelder-Mead optimization ---")

        def evaluate(x):
            return float(self.f(x, **kwargs)), np.array(self.get_params(), dtype=float)

        f_vals = np.empty(x0.shape[0])
        param_vals = np.empty((x0.shape[0], n_params))
        for idx, x in enumerate(x0):
            f_vals[idx], param_vals[idx] = evaluate(x)
        if not np.isfinite(f_vals).any():
            raise RuntimeError(
                "Nelder-Mead initialization has produced a simplex with invalid vertices."
            )
        _restore_infeasible_vertices(evaluate, x0, f_vals, param_vals, DEFAULT_RESTORE_MAX_HALVINGS)

        for i in range(max_iter):
            start_time = time.time()
            if verbose:
                logger.info(f"Iteration # {i}")

            self.nmpath[:, :-1, i] = param_vals
            self.nmpath[:, -1, i] = f_vals

            if not np.isfinite(f_vals).any():
                self.nmpath = self.nmpath[:, :, :i]
                raise RuntimeError(
                    "Nelder-Mead algorithm has produced a simplex with invalid vertices."
                )

            order = np.argsort(f_vals, kind="stable")
            ibest, isecond, iworst = int(order[0]), int(order[-2]), int(order[-1])
            centroid = np.mean(x0[order[:-1]], axis=0)

            if np.max(np.abs(x0 - centroid)) < tol or i == max_iter - 1:
                f_val = self.f(x0[ibest, :], **kwargs)
                window = self.calculate_deviation_metrics(**kwargs)
                kwargs["unsupported_range_mode"] = "score"
                mae, rmse, mape, rmspe = self.calculate_deviation_metrics(**kwargs)
                metrics = {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "rmspe": rmspe,
                    "mae_window": window[0],
                    "rmse_window": window[1],
                    "mape_window": window[2],
                    "rmspe_window": window[3],
                }
                if verbose:
                    logger.info(
                        "--- Nelder-Mead converged in %s seconds ---" % (time.time() - initial_time)
                    )
                    logger.info(
                        f"Mean temperature deviation per point between liquidus curves = {mae}"
                    )
                self.nmpath = self.nmpath[:, :, : i + 1]
                return f_val, metrics, self.nmpath

            xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
            f_xreflect, p_xreflect = evaluate(xreflect)

            # Simplex reflection step
            if f_vals[ibest] <= f_xreflect < f_vals[isecond]:
                x0[iworst, :], f_vals[iworst], param_vals[iworst] = (
                    xreflect,
                    f_xreflect,
                    p_xreflect,
                )
            # Simplex expansion step
            elif f_xreflect < f_vals[ibest]:
                xexp = centroid + 2.0 * (xreflect - centroid)
                f_xexp, p_xexp = evaluate(xexp)
                if f_xexp < f_xreflect:
                    x0[iworst, :], f_vals[iworst], param_vals[iworst] = xexp, f_xexp, p_xexp
                else:
                    x0[iworst, :], f_vals[iworst], param_vals[iworst] = (
                        xreflect,
                        f_xreflect,
                        p_xreflect,
                    )
            # Simplex contraction step, and a shrink where the contraction is no better
            else:
                if f_xreflect < f_vals[iworst]:
                    xcontract = centroid + 0.5 * (xreflect - centroid)
                    f_xcontract, p_xcontract = evaluate(xcontract)
                    accept = f_xcontract <= f_xreflect
                else:
                    xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                    f_xcontract, p_xcontract = evaluate(xcontract)
                    accept = f_xcontract < f_vals[iworst]
                if accept:
                    x0[iworst, :], f_vals[iworst], param_vals[iworst] = (
                        xcontract,
                        f_xcontract,
                        p_xcontract,
                    )
                else:
                    for j in range(x0.shape[0]):
                        if j == ibest:
                            continue
                        x0[j, :] = x0[ibest, :] + 0.5 * (x0[j, :] - x0[ibest, :])
                        f_vals[j], param_vals[j] = evaluate(x0[j, :])

            if verbose:
                guess_dict = {
                    symbol: float(guess) for symbol, guess in zip(self.guess_symbols, x0[ibest, :])
                }
                logger.info(f"Best guess: {guess_dict} f={f_vals[ibest]}")
                logger.info(f"Height of triangle = {2 * np.max(np.abs(x0 - centroid))}")
                logger.info("--- %s seconds elapsed ---" % (time.time() - start_time))

        raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")

    def _identity_entries(self) -> list[list]:
        """Constraint entries pinning each of the format's pinned parameters at zero.

        Returns:
            list: ``[name, order, temperature, sympy.Eq]`` entries at infinite temperature.
        """
        fmt = self.xs_mix.format
        return [
            [f"no1S - {p} = 0", "0th order", float("inf"), eq]
            for p, eq in zip(fmt.pinned_params, fmt.identity_constraints())
        ]

    def _mig_constraint_equations(self, inv: dict, foot_tol: float | None = None) -> list[list]:
        """Constraint equations a miscibility-gap invariant contributes.

        The top-of-dome equation sets the liquid free-energy curvature to zero at the apex
        and is emitted only when ``mpds.mig_apex`` knows one. The bottom-of-dome equation
        equates the liquid free-energy slope at the two feet on the monotectic isotherm and
        is emitted only when the feet share a temperature within ``foot_tol`` and both lie
        on the digitized liquidus within it.

        Args:
            inv: a ``mig`` invariant record.
            foot_tol: kelvin a foot may sit off the liquidus; ``config.coverage_mig_foot_tol``
                when None, ``math.inf`` to skip the check.

        Returns:
            list: ``[name, order, temperature, sympy.Eq]`` entries, apex equation first.
        """
        foot_tol = config.coverage_default("mig_foot_tol", foot_tol)
        entries = []
        apex = mpds.mig_apex(inv)
        if apex is not None:
            x2, t2 = apex
            eqn = sp.Eq(self.eqs["g_double_prime"].subs({xb_sym: x2, t_sym: t2}), 0)
            entries.append([f"mig - {round(x2, 2)} 2nd", "2nd order", t2, eqn])
        cbounds = inv.get("cbounds") or []
        feet_ok = (
            len(cbounds) >= 2
            and abs(float(cbounds[0][1]) - float(cbounds[1][1])) <= foot_tol
            and (
                math.isinf(foot_tol) or mpds.mig_feet_on_liquidus(inv, self.digitized_liq, foot_tol)
            )
        )
        if feet_ok:
            (x1, t1), (x3, t3) = cbounds[0], cbounds[1]
            eqn = sp.Eq(
                self.eqs["g_prime"].subs({xb_sym: x1, t_sym: t1}),
                self.eqs["g_prime"].subs({xb_sym: x3, t_sym: t3}),
            )
            entries.append([f"mig - {round(x1, 2)}-{round(x3, 2)} 1st", "1st order", t1, eqn])
        return entries

    def _solid_gibbs_at_T(self, phase: Phase, temp_K: float) -> float:
        """A fixed phase's Gibbs energy at ``temp_K``: the stable polymorph at an endpoint,
        the enthalpy of a line compound elsewhere."""
        x = self._phase_x(phase)
        return self._stable_solid_gibbs_at_T(x, temp_K) if x in (0.0, 1.0) else phase.enthalpy

    def _eut_constraint_equations(
        self, inv: dict, lhs_phase: Phase, rhs_phase: Phase
    ) -> list[list]:
        """Constraint equations a eutectic invariant contributes.

        The ``1st`` equation equates the liquid free-energy slope at the eutectic to the
        slope of the line through the two solids. The ``0th`` equation places the lower-
        energy solid on the tangent to the liquid free energy at the eutectic.

        Args:
            inv: a ``eut`` invariant record.
            lhs_phase: the hull phase matched to ``inv["phase_comps"][0]``.
            rhs_phase: the hull phase matched to ``inv["phase_comps"][1]``.

        Returns:
            list: ``[name, order, temperature, sympy.Eq]`` entries, ``1st`` first.
        """
        x1, x3 = self._phase_x(lhs_phase), self._phase_x(rhs_phase)
        x2, t2 = inv["comp"], inv["temp"]
        g1, g3 = self._solid_gibbs_at_T(lhs_phase, t2), self._solid_gibbs_at_T(rhs_phase, t2)
        g_liq = self.eqs["g_liquid"].subs({xb_sym: x2, t_sym: t2})
        g_prime = self.eqs["g_prime"].subs({xb_sym: x2, t_sym: t2})
        entries = [
            [f"eut - {round(x2, 2)} 1st", "1st order", t2, sp.Eq(g_prime, (g3 - g1) / (x3 - x1))]
        ]
        if g1 <= g3:
            eqn = sp.Eq(g_liq + g_prime * (x1 - x2), g1)
            entries.append([f"eut - {round(x2, 2)} 0th", "0th order lhs", t2, eqn])
        else:
            eqn = sp.Eq(g_liq + g_prime * (x3 - x2), g3)
            entries.append([f"eut - {round(x2, 2)} 0th", "0th order rhs", t2, eqn])
        return entries

    def _reacting_solid(
        self,
        x_liq: float,
        per_phase: Phase,
        tol: float = 0.04,
        x_beta_ref: float | None = None,
    ) -> dict | None:
        """The solid the liquid reacts with at a peritectic, read off the hull.

        It is the fixed hull phase nearest the forming compound on the side away from the
        liquid, the pure component when nothing lies between. The digitized diagram's own
        neighbour on that side is reported alongside so a substitution is visible.

        Args:
            x_liq: the invariant's liquid composition.
            per_phase: the forming compound's hull phase.
            tol: composition within which the hull and digitized neighbours count as the same.
            x_beta_ref: the forming compound's digitized composition, which fixes the side the
                reacting solid lies on; the hull phase's own composition when None.

        Returns:
            dict | None: ``{"phase", "x", "x_mpds", "matches_mpds"}``, or None when the
            compound has nothing beyond it on the hull.
        """
        x_beta = self._phase_x(per_phase)
        x_ref = x_beta if x_beta_ref is None else float(x_beta_ref)
        end = 1.0 if x_liq < x_ref else 0.0

        def beyond(x: float, origin: float) -> bool:
            return x > origin + 1e-6 if end == 1.0 else x < origin - 1e-6

        candidates = [
            p
            for p in self.phases
            if p.composition is not None and not p.is_solution and beyond(self._phase_x(p), x_beta)
        ]
        if not candidates:
            return None
        alpha = min(candidates, key=lambda p: abs(self._phase_x(p) - x_beta))
        x_alpha = self._phase_x(alpha)
        digitized = [
            float(rec["comp"])
            for rec in (self.low_t_exp_phases or [])
            if rec.get("comp") is not None
            and beyond(float(rec["comp"]), x_ref)
            and abs(float(rec["comp"]) - x_ref) > tol
        ]
        x_mpds = min(digitized, key=lambda x: abs(x - x_ref)) if digitized else end
        return {
            "phase": alpha,
            "x": x_alpha,
            "x_mpds": x_mpds,
            "matches_mpds": abs(x_alpha - x_mpds) <= tol,
        }

    def _per_constraint_equation(self, inv: dict, per_phase: Phase, t_tol: float = 15) -> list:
        """The constraint equation a peritectic invariant contributes.

        The forming compound is placed on the tangent to the liquid free energy at the
        invariant liquid composition; when the compound sits on the liquidus within ``t_tol``
        the tangent term is dropped and the compound is treated as a congruent point.

        Args:
            inv: a ``per`` invariant record.
            per_phase: the hull phase matched to ``inv["phase_comps"][0]``.
            t_tol: kelvin within which the compound counts as on the liquidus.

        Returns:
            list: ``[name, order, temperature, sympy.Eq]``.
        """
        x1, t1 = inv["comp"], inv["temp"]
        x2 = self._phase_x(per_phase)
        g2 = self._solid_gibbs_at_T(per_phase, t1)
        g_liq = self.eqs["g_liquid"].subs({xb_sym: x1, t_sym: t1})
        liq_point_at_phase = min(self.digitized_liq, key=lambda x: abs(x[0] - x2))
        if liq_point_at_phase[1] - t1 > t_tol:
            g_prime = self.eqs["g_prime"].subs({xb_sym: x1, t_sym: t1})
            eqn = sp.Eq(g_liq + g_prime * (x2 - x1), g2)
            return [f"per - {round(x1, 2)} 0th", "0th order", t1, eqn]
        return [f"per - {round(x1, 2)} 0th", "H-S partition 0th order", t1, sp.Eq(g_liq, g2)]

    def _invariant_constraint_ics(self, eqs: list[list], **kwargs) -> list[dict]:
        """Nelder-Mead initial conditions from invariant-derived constraint equations.

        A format consuming two invariant equations per candidate pairs the hottest equation
        with each other one; a format consuming one solves every equation together with the
        format's identity constraints. A candidate is dropped when solving its equations
        leaves any of the format's solve symbols undetermined, and otherwise scored at the
        initial simplex and dropped when that score is infinite; the dropped names are left
        on ``self.dropped_candidates``.

        Args:
            eqs: ``[name, order, temperature, sympy.Eq]`` entries; ``sympy.false`` entries
                are ignored.
            **kwargs: forwarded to ``f``.

        Returns:
            list[dict]: ``{"f", "constrs", "init_tri"}`` per surviving candidate, unsorted.
        """
        fmt = self.xs_mix.format
        identity_entries = self._identity_entries()
        solve_syms = tuple(self.xs_mix.solve_symbols)
        # identity against sp.false, NOT `if eq[3]`: bool() on a symbolic Eq raises TypeError
        eqs = [eq for eq in eqs if eq[3] is not sp.false]
        ics = []
        self.dropped_candidates = []
        if not eqs:
            return ics
        if fmt.n_invariant_constraints == 2:
            highest_tm_eq = eqs.pop(eqs.index(max(eqs, key=lambda x: x[2])))
        for eq in eqs:
            try:
                self.guess_symbols = self.xs_mix.guess_symbols
                if fmt.n_invariant_constraints == 2:
                    constr_entries = [eq, highest_tm_eq]
                else:
                    constr_entries = [eq, *identity_entries]
                self.constraints = sp.solve(
                    [entry[3] for entry in constr_entries],
                    solve_syms,
                    rational=False,
                    simplify=False,
                )
                solved = self.constraints if isinstance(self.constraints, dict) else {}
                if any(sym not in solved for sym in solve_syms):
                    self.dropped_candidates.append(eq[0])
                    continue
                init_f = min(self.f(v, **kwargs) for v in self.init_triangle)
                if init_f == float("inf"):
                    self.dropped_candidates.append(eq[0])
                    continue
                ics.append({"f": init_f, "constrs": constr_entries, "init_tri": self.init_triangle})
            except RuntimeError as e:
                logger.error(f"Error while evaluating invariant constraints [{self.sys_name}]: {e}")
                continue
        return ics

    @config._removed_keywords(
        {
            **{
                f"coverage_{name}": f"coverage_thresholds={{'{name}': ...}}"
                for name in ("ss_narrow_tol", "dft_cover_tol", "ss_rescue_max_width")
            },
            "check_phase_mismatch": "check_solid_coverage; it has gated nothing since 0.1.0",
            "ignored_ranges": "unsupported_range_mode='ignore' or 'score'",
            "num_points": "calculate_deviation_metrics(num_points=...) after the fit",
        }
    )
    def fit_parameters(
        self, verbose=False, n_opts=1, t_tol=15, enable_multi_threading=False, **kwargs
    ) -> list[dict]:
        """
        Fit the liquidus non-ideal mixing parameters with Nelder-Mead.

        Invariant constraints read solid-solution phases as absent and line compounds at their
        T-independent enthalpy.

        Args:
            verbose (bool): If True, prints detailed progress and results.
            n_opts (int): Number of candidates optimised. The object takes the parameters of
                the fit with the lowest objective.
            t_tol (float): Temperature tolerance for invariant point identification.
            enable_multi_threading (bool): If True, optimises the candidates in parallel threads.
            **kwargs: The names in ``FIT_KWARGS``; any other raises ``TypeError``. Each is
                documented where it takes effect:
                - max_iter: :meth:`nelder_mead`.
                - ignore_ss (bool): sets ``ss_in_hull`` False (class Attributes).
                - disable_inv_constrs (bool): If True, fit without invariant constraints.
                - check_full_ss: :meth:`find_invariant_points`.
                - check_solid_coverage (bool): If True (default), skip the system, flagging
                    ``SKIP_SOLID_COVERAGE`` or ``SKIP_UNION_COVERAGE``, when the coverage
                    report's ranges, alone or merged with the invariant-derived ones, leave
                    less constraining liquidus than the coverage floor
                    (:meth:`gliquid.solid_coverage.SolidCoverageReport.is_insufficient`). The
                    verdict depends on the loaded solid-solution energies.
                - coverage_thresholds (dict | None): Per-call coverage knobs,
                    :func:`gliquid.config.coverage_settings`; its ``dft_cover_tol`` also sets
                    the invariant-derived masking's DFT cover tolerance.
                - init_objective_cutoff: ``DEFAULT_INIT_OBJECTIVE_CUTOFF``; the best initial
                    objective is floored at 1.
                - check_lupis_elliott (bool), lupis_elliott_cfg: :meth:`lupis_elliott_penalty`.
                - use_lxb_penalty (bool), lxb_penalty_cfg: :meth:`lxb_penalty`; on by default
                    for a format whose ``lxb_default`` is True ('comb-exp').
                - suppress_unexpected_migs (bool): If True (default), scales the objective by
                    :meth:`unexpected_gap_width` (``GAP_WIDTH_SCALE``).
                - unsupported_range_mode: ``UNSUPPORTED_RANGE_MODES``; stored as
                    ``fit_unsupported_range_mode``, and ``BLPlotter`` marks the unsupported
                    ranges on the assessed liquidus for ``'ignore'`` and ``'cap'`` only.
                - allow_sparse_data, min_comp_span: :meth:`calculate_deviation_metrics`.

        Returns:
            list[dict]: Parameter fitting data containing results of all optimization attempts.

        Raises:
            TypeError: on an unrecognized keyword argument.
            ValueError: on an unrecognized ``unsupported_range_mode``.
            config.CacheModeError: when ``self.mpds_json`` is a LEAN record (see
                ``mpds.record_mode``) and the two checks it cannot support have not both
                been switched off explicitly.
        """
        unknown = sorted(set(kwargs) - FIT_KWARGS)
        if unknown:
            raise TypeError(
                f"fit_parameters() got "
                f"{'unexpected keyword arguments' if len(unknown) > 1 else 'an unexpected keyword argument'}: "
                f"{', '.join(repr(key) for key in unknown)}. Accepted: "
                f"{', '.join(sorted(FIT_KWARGS))}."
            )

        kwargs["unsupported_range_mode"] = resolve_unsupported_range_mode(kwargs)
        dft_cover_tol = config.coverage_settings(kwargs.get("coverage_thresholds"))["dft_cover_tol"]

        if mpds.record_mode(getattr(self, "mpds_json", None)) == "lean" and not (
            kwargs.get("disable_inv_constrs", False)
            and not kwargs.get("check_solid_coverage", True)
        ):
            raise config.CacheModeError(
                f"[{self.sys_name}] cannot be fit from a LEAN MPDS record: it carries the "
                f"digitized liquidus but not the 'shapes' block, so neither the invariant "
                f"constraints nor the solid free-energy coverage gate can be evaluated. "
                f"Both would come back EMPTY, which the coverage gate reads as 'nothing "
                f"unsupported' and passes — a silently unconstrained fit. The real fix is "
                f"a full MPDS store: rebuild it with `python -m gliquid.cache migrate "
                f"--mpds-mode full`, or point gliquid at the directory corpus. To fit "
                f"anyway, on a liquidus alone and with both checks knowingly off, pass "
                f"fit_parameters(disable_inv_constrs=True, check_solid_coverage=False) — "
                f"that is a different fit, not the same one."
            )

        self.fit_unsupported_range_mode = kwargs["unsupported_range_mode"]

        if kwargs.get("ignore_ss", False):
            self.ss_in_hull = False

        if self.digitized_liq is None:
            logger.warning(
                "System missing liquidus data! Ensure that 'BinaryLiquid.digitized_liq' is not empty!"
            )
            return []

        if (
            self.invariants is None
            and self.low_t_exp_phases is None
            and not kwargs.get("disable_inv_constrs", False)
        ):
            self.invariants, self.low_t_exp_phases = self.find_invariant_points(
                verbose=verbose, t_tol=t_tol, check_full_ss=kwargs.get("check_full_ss", True)
            )
            if verbose:
                mpds.print_phase_mismatch_chart(self.low_t_exp_phases, self._interior_dft_comps())

        if kwargs.get("check_solid_coverage", True):
            coverage = self.assess_solid_coverage(thresholds=kwargs.get("coverage_thresholds"))
            insufficient, reason = coverage.is_insufficient()
            if verbose:
                logger.info(f"Solid-energy coverage [{self.sys_name}]: {coverage.summary_line}")
            if insufficient:
                logger.warning(
                    f"Insufficient solid-phase energy information [{self.sys_name}]: "
                    f"{reason}; skipping fit. {coverage.summary_line}"
                )
                self._flag_skip(SKIP_SOLID_COVERAGE)
                return []

        if self.init_error:
            return []

        # Compare invariant points to self.phases to assess solving conditions
        eqs = []
        auto_unsupported_ranges = not bool(self.unsupported_comp_ranges)

        # coverage ranges join after the mask-fraction cap; a caller-supplied list skips both
        coverage_ranges = (
            [list(r) for r in self.coverage_report.unsupported_ranges]
            if auto_unsupported_ranges and self.coverage_report is not None
            else []
        )

        mask_fraction_cap = MASK_FRACTION_CAP
        _inv_comps = mpds.bounding_invariant_comps(self.invariants)
        _dft_comps = self._interior_dft_comps()

        gap_comps = mpds.two_liquid_label_comps(self.mpds_json, self.components)

        if not kwargs.get("disable_inv_constrs", False):
            for inv in self.invariants:
                if inv["type"] == "mig":
                    eqs.extend(self._mig_constraint_equations(inv))

                if inv["type"] == "cmp":
                    if "(" in inv["phases"][0] and auto_unsupported_ranges:
                        if not _dft_comp_within(inv["comp"], _dft_comps, dft_cover_tol):
                            if inv["comp"] < 0.5:
                                lo = mpds.flank_bound(
                                    inv["comp"],
                                    "left",
                                    ((_inv_comps, None),),
                                    tol=_INV_FLANK_TOL,
                                    default=(0, None),
                                )[0]
                                _mask_walked(
                                    self.unsupported_comp_ranges, lo, inv["comp"], gap_comps
                                )
                            elif inv["comp"] > 0.5:
                                hi = mpds.flank_bound(
                                    inv["comp"],
                                    "right",
                                    ((_inv_comps, None),),
                                    tol=_INV_FLANK_TOL,
                                    default=(1, None),
                                )[0]
                                _mask_walked(
                                    self.unsupported_comp_ranges, inv["comp"], hi, gap_comps
                                )
                        continue

                    nearest_phase, _ = self._find_nearest_phase(inv["comp"])
                    if not nearest_phase:
                        continue

                    x1, t1 = self._phase_x(nearest_phase), inv["temp"]
                    eqn = sp.Eq(
                        self.eqs["g_liquid"].subs({xb_sym: x1, t_sym: t1}), nearest_phase.enthalpy
                    )
                    eqs.append([f"cmp - {round(x1, 2)} 0th", "0th order", t1, eqn])

                if inv["type"] == "per":
                    if "(" in inv["phases"][0] and auto_unsupported_ranges:
                        if not _dft_comp_within(inv["comp"], _dft_comps, dft_cover_tol):
                            if inv["phase_comps"][0] < inv["comp"]:
                                lo = mpds.flank_bound(
                                    inv["comp"],
                                    "left",
                                    ((_inv_comps, None),),
                                    tol=_INV_FLANK_TOL,
                                    default=(0, None),
                                )[0]
                                _mask_walked(
                                    self.unsupported_comp_ranges, lo, inv["comp"], gap_comps
                                )
                            elif inv["phase_comps"][0] > inv["comp"]:
                                hi = mpds.flank_bound(
                                    inv["comp"],
                                    "right",
                                    ((_inv_comps, None),),
                                    tol=_INV_FLANK_TOL,
                                    default=(1, None),
                                )[0]
                                _mask_walked(
                                    self.unsupported_comp_ranges, inv["comp"], hi, gap_comps
                                )
                        continue

                    per_phase, _ = self._find_nearest_phase(inv["phase_comps"][0], tol=0.04)
                    if not per_phase:
                        continue

                    eqs.append(self._per_constraint_equation(inv, per_phase, t_tol=t_tol))

                if inv["type"] == "eut":
                    if None in inv["phase_comps"]:
                        continue

                    lhs_phase, _ = self._find_nearest_phase(inv["phase_comps"][0], tol=0.04)
                    rhs_phase, _ = self._find_nearest_phase(inv["phase_comps"][1], tol=0.04)

                    invalid_eut = False
                    if not lhs_phase or self._phase_x(lhs_phase) > inv["comp"]:
                        if auto_unsupported_ranges and not _dft_comp_within(
                            inv["phase_comps"][0], _dft_comps, dft_cover_tol
                        ):
                            lo = mpds.flank_bound(
                                inv["comp"],
                                "left",
                                ((_inv_comps, None),),
                                tol=_INV_FLANK_TOL,
                                default=(0, None),
                            )[0]
                            _mask_walked(self.unsupported_comp_ranges, lo, inv["comp"], gap_comps)
                        invalid_eut = True
                    elif "(" in inv["phases"][0] and inv["phase_comps"][0] > 0.05:
                        invalid_eut = True
                    if not rhs_phase or self._phase_x(rhs_phase) < inv["comp"]:
                        if auto_unsupported_ranges and not _dft_comp_within(
                            inv["phase_comps"][1], _dft_comps, dft_cover_tol
                        ):
                            hi = mpds.flank_bound(
                                inv["comp"],
                                "right",
                                ((_inv_comps, None),),
                                tol=_INV_FLANK_TOL,
                                default=(1, None),
                            )[0]
                            _mask_walked(self.unsupported_comp_ranges, inv["comp"], hi, gap_comps)
                        invalid_eut = True
                    elif "(" in inv["phases"][1] and inv["phase_comps"][1] < 0.95:
                        invalid_eut = True
                    if invalid_eut:
                        continue

                    eqs.extend(self._eut_constraint_equations(inv, lhs_phase, rhs_phase))

        if auto_unsupported_ranges and self.unsupported_comp_ranges:
            self.unsupported_comp_ranges = mpds.merge_ranges(self.unsupported_comp_ranges)
            liq_lo, liq_hi = self.digitized_liq[0][0], self.digitized_liq[-1][0]
            span = max(liq_hi - liq_lo, 1e-9)
            masked = sum(
                min(hi, liq_hi) - max(lo, liq_lo)
                for lo, hi in self.unsupported_comp_ranges
                if min(hi, liq_hi) > max(lo, liq_lo)
            )
            if masked / span > mask_fraction_cap:
                if verbose:
                    logger.info(
                        f"{masked / span:.0%} of the liquidus would be "
                        f"masked (> {mask_fraction_cap:.0%} cap); flagging system as unreliable "
                        f"(DFT too incomplete) and skipping."
                    )
                self._flag_skip(SKIP_MASK_FRACTION)
                return []

        if coverage_ranges:
            self.unsupported_comp_ranges = mpds.merge_ranges(
                list(self.unsupported_comp_ranges) + coverage_ranges
            )

        # the coverage floor again on the merged mask; self.coverage_report is not changed
        if (
            auto_unsupported_ranges
            and kwargs.get("check_solid_coverage", True)
            and self.coverage_report is not None
        ):
            union_coverage = replace(
                self.coverage_report,
                unsupported_ranges=[list(r) for r in self.unsupported_comp_ranges],
            )
            insufficient, reason = union_coverage.is_insufficient()
            if insufficient:
                logger.warning(
                    f"Insufficient solid-phase energy information across the merged mask "
                    f"[{self.sys_name}]: {reason}; skipping fit."
                )
                self._flag_skip(SKIP_UNION_COVERAGE)
                return []

        fmt = self.xs_mix.format
        identity_entries = self._identity_entries()
        solve_syms = tuple(self.xs_mix.solve_symbols)
        nelder_mead_ics = self._invariant_constraint_ics(eqs, **kwargs)

        # Derive H-S partition constraints using Nelder-Mead for the enthalpy of mixing
        mean_liq_temp = np.mean([point[1] for point in self.digitized_liq])
        if verbose:
            logger.info(
                f"Maximum composition range fitted: {[self.digitized_liq[0][0], self.digitized_liq[-1][0]]}"
            )
            logger.info(f"Unsupported composition ranges: {self.unsupported_comp_ranges}")

        try:
            L0_a_sym, L0_b_sym = l_sym("L0_a"), l_sym("L0_b")
            L1_a_sym, L1_b_sym = l_sym("L1_a"), l_sym("L1_b")
            no_L0Sxs_eq = sp.Eq(
                sp.diff(self.eqs["l0"], t_sym).subs({t_sym: 0}), 0
            )  # enforce S0xs @0K = 0 -> L0_b != 0 (comb)
            no_L1Sxs_eq = sp.Eq(
                sp.diff(self.eqs["l1"], t_sym).subs({t_sym: 0}), 0
            )  # enforce S1xs @0K = 0 -> L1_b != 0 (comb)
            self.update_params([])  # Restore parameter defaults

            self.guess_symbols = [L0_a_sym, L1_a_sym]
            self.constraints = sp.solve(
                [no_L0Sxs_eq, no_L1Sxs_eq], (L0_b_sym, L1_b_sym), rational=False, simplify=False
            )
            hs_partition_triangle = build_init_triangle(("L0_a", "L1_a"), self.dft_ch)
            if verbose:
                logger.info(
                    f"Initial triangle for H-S partition constraints: {hs_partition_triangle}"
                )
            init_f, _, _ = self.nelder_mead(
                tol=10, verbose=verbose, initial_guesses=hs_partition_triangle, **kwargs
            )

            # Store reference parameter values from the orthogonal H-S partition constraint pass
            self._ref_params = {
                "L0_a": self.get_L0_a(),
                "L0_b": self.get_L0_b(),
                "L1_a": self.get_L1_a(),
                "L1_b": self.get_L1_b(),
            }
            eq1 = sp.Eq(
                self.eqs["l0"].subs({t_sym: mean_liq_temp}),
                self.eqs["l0"].subs(
                    {t_sym: mean_liq_temp, L0_a_sym: self.get_L0_a(), L0_b_sym: self.get_L0_b()}
                ),
            )

            if fmt.n_invariant_constraints == 2:
                eq2 = sp.Eq(
                    self.eqs["l1"].subs({t_sym: mean_liq_temp}),
                    self.eqs["l1"].subs(
                        {t_sym: mean_liq_temp, L1_a_sym: self.get_L1_a(), L1_b_sym: self.get_L1_b()}
                    ),
                )
                init_tri = [
                    [self.get_L0_b(), self.get_L1_b()],
                    [
                        _offset_vertex_value(
                            self.get_L0_b(), DEFAULT_B_SPREAD_MIN, self.get_L0_a()
                        ),
                        self.get_L1_b(),
                    ],
                    [
                        self.get_L0_b(),
                        _offset_vertex_value(
                            self.get_L1_b(), DEFAULT_B_SPREAD_MIN, self.get_L1_a()
                        ),
                    ],
                ]
                self.guess_symbols = self.xs_mix.guess_symbols
                self.constraints = sp.solve([eq1, eq2], solve_syms, rational=False, simplify=False)
                hs_init_f = min(self.f(v, **kwargs) for v in init_tri)
                nelder_mead_ics.append(
                    {
                        "f": hs_init_f,
                        "constrs": [
                            ["hs_partition", "0th order", mean_liq_temp, e] for e in [eq1, eq2]
                        ],
                        "init_tri": init_tri,
                    }
                )
            else:
                init_tri = [
                    [self.get_L0_b(), self.get_L1_a()],
                    [
                        _offset_vertex_value(
                            self.get_L0_b(), DEFAULT_B_SPREAD_MIN, self.get_L0_a()
                        ),
                        self.get_L1_a(),
                    ],
                    [self.get_L0_b(), self.get_L1_a() * 0.8],
                ]
                self.guess_symbols = self.xs_mix.guess_symbols
                self.constraints = sp.solve(
                    [eq1, *[e[3] for e in identity_entries]],
                    solve_syms,
                    rational=False,
                    simplify=False,
                )
                hs_init_f = min(self.f(v, **kwargs) for v in init_tri)
                nelder_mead_ics.append(
                    {
                        "f": hs_init_f,
                        "constrs": [
                            ["hs_partition", "0th order", mean_liq_temp, eq1],
                            *identity_entries,
                        ],
                        "init_tri": init_tri,
                    }
                )
        except RuntimeError as e:
            logger.error(
                f"Nelder-Mead process encountered a fatal error while deriving H-S partition constraints [{self.sys_name}]: {e}"
            )

        # lowest initial objective first
        nelder_mead_ics.sort(key=lambda x: x["f"])
        cutoff = kwargs.get("init_objective_cutoff", DEFAULT_INIT_OBJECTIVE_CUTOFF)
        self.pruned_candidates = []
        if cutoff is not None and nelder_mead_ics:
            ceiling = max(nelder_mead_ics[0]["f"], 1.0) * float(cutoff)
            kept = [ic for ic in nelder_mead_ics if ic["f"] <= ceiling]
            self.pruned_candidates = [
                [ic["constrs"][0][0], float(ic["f"])] for ic in nelder_mead_ics[len(kept) :]
            ]
            nelder_mead_ics = kept
        self.guess_symbols = self.xs_mix.guess_symbols
        solve_symbols = self.xs_mix.solve_symbols

        def run(task_idx, task_ics):
            try:
                return _run_single_optimization_worker(
                    task_idx, task_ics, self, kwargs, verbose, solve_symbols, mean_liq_temp
                )
            except Exception as e:
                logger.error(
                    f"Optimization attempt #{task_idx + 1} failed with exception [{self.sys_name}]: {e}"
                )
                return None

        tasks = list(enumerate(nelder_mead_ics[:n_opts]))
        if enable_multi_threading and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                results = list(executor.map(lambda task: run(*task), tasks))
        else:
            results = [run(*task) for task in tasks]
        fitting_data = [result for result in results if result is not None]

        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x["f"])
            self.xs_mix.update([best_fit[name] for name in self.xs_mix.format.param_names])
            self.nmpath = best_fit["nmpath"]
            self.update_phase_points()
        return fitting_data


def _run_single_optimization_worker(
    task_index, selected_ics, bl_template, run_kwargs, verbose, solve_symbols, mean_liq_temp
):
    """
    One Nelder-Mead optimisation of a deep copy of ``bl_template``.

    Args:
        task_index (int): Index of the optimization attempt.
        selected_ics (dict): Initial conditions for this optimization attempt.
        bl_template (BinaryLiquid): Template BinaryLiquid object to copy.
        run_kwargs (dict): kwargs forwarded to Nelder-Mead and the objective function.
        verbose (bool): Print progress information.
        solve_symbols (list[sp.Symbol]): Symbols solved from constraints.
        mean_liq_temp (float): Mean liquidus temperature for reporting L0/L1.

    Returns:
        dict | None: Fitting result dictionary, or None if optimization failed.
    """
    bl_copy: BinaryLiquid = copy.deepcopy(bl_template)
    bl_copy.guess_symbols = bl_copy.xs_mix.guess_symbols

    constrs_str = "/".join([c[0] for c in selected_ics["constrs"]])
    constr_algo = "hs_partition_constr" if constrs_str.startswith("hs_partition") else "inv_constr"

    if verbose:
        logger.info(
            f"--- Nelder-Mead ICs Attempt #{task_index + 1} (initial f = {round(selected_ics['f'], 2)}) ---"
        )
        for source, order, temp, eq in selected_ics["constrs"]:
            logger.info(
                f"Source: {source}, Order: {order}, Temperature: {round(temp, 1)}, Equation: {eq}"
            )
        logger.info(f"Initial triangle: {selected_ics['init_tri']}")

    selected_eqs = [eq[3] for eq in selected_ics["constrs"]]
    bl_copy.constraints = sp.solve(selected_eqs, solve_symbols, rational=False, simplify=False)
    try:
        f, metrics, path = bl_copy.nelder_mead(
            verbose=verbose, tol=5, initial_guesses=selected_ics["init_tri"], **run_kwargs
        )
    except RuntimeError as e:
        logger.error(f"Nelder-Mead process encountered a fatal error [{bl_copy.sys_name}]: {e}")
        return None

    param_subs = bl_copy.xs_mix.subs_map()
    l0 = float(bl_copy.eqs["l0"].subs({t_sym: mean_liq_temp, **param_subs}))
    l1 = float(bl_copy.eqs["l1"].subs({t_sym: mean_liq_temp, **param_subs}))
    fit_invs = bl_copy.hsx.liquidus_invariants()[0]
    result = {
        "f": f,
        **metrics,
        "constrs": constrs_str,
        "algo": constr_algo,
        "constr_temps": [float(c[2]) for c in selected_ics["constrs"]],
        "n_iters": path.shape[2],
        "nmpath": path,
        "L0": l0,
        "L1": l1,
        "euts": fit_invs["Eutectics"],
        "pers": fit_invs["Peritectics"],
        "cmps": fit_invs["Congruent Melting"],
        "migs": fit_invs["Misc Gaps"],
    }
    result.update({name: bl_copy.xs_mix[name] for name in bl_copy.xs_mix.format.param_names})
    return result


# Plot stack names re-exported from gliquid.plotting.
from gliquid.plotting.binary_tx import (  # noqa: F401
    _plot_tx_inputs,
    build_polymorph_transitions,
    plot_tx,
    record_tie_lines,
)
from gliquid.plotting.style import (  # noqa: F401
    SS_FIXED_COLORS,
    build_phase_color_map,
    format_phase_display_name,
    non_dft_legend_label,
)

# Grid refinement of the SS presentation hull; other values drop invariants or open field ends.
_SS_PLOT_REFINE = 5


class BLPlotter:
    """Matplotlib and Plotly figures of a BinaryLiquid; see :meth:`get_plot` for the types."""

    def __init__(self, binaryliquid: BinaryLiquid, order="alphabetical", **plotkwargs):
        """
        Args:
            binaryliquid (BinaryLiquid): BinaryLiquid object containing the system data.
            order: Presentation component order — 'alphabetical' (default), 'given'
                (the system's construction order), or any permutation spec accepted by
                ``gliquid.phase.resolve_component_order``. A non-matching order plots a
                re-framed copy (``BinaryLiquid.with_component_order``).
            plotkwargs (dict): Optional keyword arguments for plot customization (e.g., axis margins).
        """
        self._bl_raw = binaryliquid
        self._bl = binaryliquid.with_component_order(order)
        self.plotkwargs = plotkwargs or {"axes": {"xmargin": 0.005, "ymargin": 0}}

    def get_plot(self, plot_type: str, **kwargs) -> go.Figure | plt.Axes:
        """
        Generates the specified plot for the BinaryLiquid object.

        Args:
            plot_type (str): One of:
                - 'pc': low-temperature phase comparison, MPDS against MP
                - 'ch', 'ch+g': T=0K DFT convex hull; 'ch+g' overlays solution Gibbs curves
                - 'fit', 'fit+liq', 'pred', 'pred+liq': the generated phase diagram; '+liq'
                  overlays the digitized liquidus, 'pred' styles it as a prediction
                - 'nmp': Nelder-Mead path, in the construction frame only (``order='given'``)
                - 'hsx': HSX hull diagnostic
                - 'scatter': raw T-x points of the HSX computation
            kwargs: Keyword arguments for the 'ch', 'ch+g', 'nmp', 'hsx' and
                'scatter' renderers; 'pc', 'fit', 'fit+liq', 'pred' and 'pred+liq' ignore them.

        Returns:
            go.Figure | plt.Axes: The generated plot object (Plotly or Matplotlib).

        Raises:
            ValueError: for an unknown ``plot_type``, or 'nmp' on a re-ordered plotter.
        """
        valid_plot_types = [
            "pc",
            "ch",
            "ch+g",
            "fit",
            "fit+liq",
            "pred",
            "pred+liq",
            "nmp",
            "hsx",
            "scatter",
        ]
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}"
            )

        fig = None
        if plot_type == "pc":
            fig = self._generate_phase_comparison_plot()
        elif plot_type in ["ch", "ch+g"]:
            fig = self._generate_convex_hull_plot(plot_type, **kwargs)
        elif plot_type in ["fit", "fit+liq", "pred", "pred+liq"]:
            fig = self._generate_liquidus_fit_plot(plot_type)
        elif plot_type == "nmp":
            if self._bl is not self._bl_raw:
                raise ValueError(
                    "'nmp' renders in the fitting (construction) frame; "
                    "construct BLPlotter with order='given'."
                )
            fig = self._generate_nelder_mead_path_plot(**kwargs)
        elif plot_type == "hsx":
            fig = self._generate_hsx_plot(**kwargs)
        elif plot_type == "scatter":
            fig = self._generate_tx_scatter_plot(**kwargs)

        return fig

    def _phase_color_map(self) -> dict[str, str]:
        """Deterministic phase -> color map, cached on the plotter until the model's phases,
        ``hsx`` phases, solid-solution models or hull change; SS phases use reserved fixed
        colors."""
        bl = self._bl
        key = (
            tuple(bl.hsx.phases),
            tuple((p.name, non_dft_kind(p)) for p in bl.phases),
            tuple(bl.ss_models),
        )
        cached = getattr(self, "_ss_phase_color_cache", None)
        if cached is not None and cached[0] == key and cached[1] is bl.dft_ch:
            return cached[2]

        polymorph_names = {
            p.name
            for p in bl.phases
            if not p.is_solution and p.fraction_in(bl.components)[0] in (0, 1)
        }
        non_dft_names = {p.name for p in bl.phases if non_dft_kind(p)}
        color_map = build_phase_color_map(
            self._palette_phase_names(bl.hsx.phases),
            ss_names=list(bl.ss_models),
            polymorph_names=polymorph_names,
            imputed_names=non_dft_names,
        )
        self._ss_phase_color_cache = (key, bl.dft_ch, color_map)
        return color_map

    def _palette_phase_names(self, phase_names) -> list[str]:
        """``phase_names`` with the DFT compounds a non-DFT phase pushed off the hull put back.

        Each displaced compound is inserted at its composition, so every other phase takes
        the palette slot it has on the system's DFT-only hull. ``phase_names`` comes back
        unchanged when the system carries no imputed or user-defined phase, or none displaced
        a compound.

        Args:
            phase_names: the phase names a figure draws, in hull order.

        Returns:
            list[str]: the names to assign palette slots from.
        """
        bl = self._bl
        if not any(non_dft_kind(p) for p in bl.phases):
            return list(phase_names)
        dft_hull = PhaseDiagram(
            [e for e in bl.dft_ch.all_entries if non_dft_entry_kind(e) is None],
            elements=list(bl.dft_ch.elements),
        )
        current = {p.name for p in bl.phases}
        displaced = [
            p
            for p in build_phases_from_chull(dft_hull, bl.components, bl.component_data)
            if p.name not in current and not p.is_solution and 0 < bl._phase_x(p) < 1
        ]
        if not displaced:
            return list(phase_names)
        x_of = {p.name: bl._phase_x(p) for p in [*bl.phases, *displaced] if not p.is_solution}
        line = [n for n in phase_names if n in x_of] + [p.name for p in displaced]
        return sorted(line, key=x_of.get) + [n for n in phase_names if n not in x_of]

    def _phase_color(self, phase_label: str) -> str:
        return self._phase_color_map().get(phase_label, "#555555")

    def _generate_tx_scatter_plot(self, **kwargs) -> go.Figure:
        """Diagnostic TX scatter using raw points from HSX compute_tx (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        if self._bl.hsx is None or not self._bl.phases[-1].points:
            self._bl.update_phase_points()  # hsx must exist before the color map reads its phases
        return binary_figs.render_tx_scatter(self._bl, self._phase_color_map(), **kwargs)

    def _generate_hsx_plot(self, **kwargs) -> go.Figure:
        """HSX hull debug plot with solid-solution phase blocks (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        if self._bl.hsx is None or not self._bl.phases[-1].points:
            self._bl.update_phase_points()  # hsx must exist before the color map reads its phases
        return binary_figs.render_hsx_diagnostic(self._bl, self._phase_color_map(), **kwargs)

    def show(self, plot_type: str, **kwargs) -> None:
        """
        Displays the generated plot.

        Args:
            plot_type (str): Any type :meth:`get_plot` accepts.
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)
        plot_export.show_figure(fig)

    def write_image(
        self,
        plot_type: str,
        stream: str | StringIO,
        image_format: str = "svg",
        export_timeout_s: float = 120.0,
        **kwargs,
    ) -> None:
        """
        Saves the generated plot as an image.

        Args:
            plot_type (str): Any type :meth:`get_plot` accepts.
            stream (str | StringIO): The file path or stream to save the image.
            image_format (str): The format of the image (default is 'svg').
            export_timeout_s (float): Maximum time allowed for Plotly image export.
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)
        write_kwargs = {}
        if plot_type in ["ch", "ch+g"]:
            write_kwargs.update({"width": 480 * 1.8, "height": 300 * 1.7})
        plot_export.save_figure(
            fig,
            stream,
            image_format=image_format,
            export_timeout_s=export_timeout_s,
            label=plot_type,
            **write_kwargs,
        )

    def _generate_phase_comparison_plot(self) -> plt.Figure:
        """MPDS-vs-MP low-temperature phase comparison (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        return binary_figs.render_phase_comparison(self._bl)

    def _generate_convex_hull_plot(self, plot_type: str, **kwargs) -> go.Figure:
        """T=0K DFT convex hull / Gibbs-overlay figure (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        return binary_figs.render_dft_hull(self._bl, plot_type, **kwargs)

    def _generate_liquidus_fit_plot(self, plot_type: str) -> go.Figure:
        """
        Generates liquidus fitting and prediction plots.

        The assessed liquidus is drawn in the unsupported style inside
        ``unsupported_comp_ranges``, except when the parameters were fit with
        ``unsupported_range_mode='score'`` (``fit_unsupported_range_mode``).

        Args:
            plot_type (str): The type of liquidus plot to generate ('fit', 'fit+liq', 'pred', 'pred+liq').

        Returns:
            go.Figure: The generated plot object.
        """
        if plot_type in ["fit+liq", "pred+liq"] and not self._bl.digitized_liq:
            logger.warning(
                "Digitized_liquidus is not initialized! Returning plot without digitized liquidus"
            )

        pred_pd = bool(plot_type in ["pred", "pred+liq"])

        if self._bl.hsx is None:
            self._bl.update_phase_points()

        hsx, shared = _plot_tx_inputs(self._bl, refine=_SS_PLOT_REFINE)
        unsupported = (
            None
            if self._bl.fit_unsupported_range_mode == "score"
            else self._bl.unsupported_comp_ranges
        )
        return plot_tx(
            hsx,
            digitized_liquidus=self._bl.digitized_liq
            if plot_type in ["fit+liq", "pred+liq"]
            else None,
            pred=pred_pd,
            palette_phases=self._palette_phase_names(hsx.phases),
            unsupported_ranges=unsupported,
            components=self._bl.components,
            mpds_temp_range=(
                (self._bl.temp_range[0] - 273.15, self._bl.temp_range[-1] - 273.15)
                if self._bl.temp_range
                else None
            ),
            **shared,
        )

    def _generate_nelder_mead_path_plot(self, **kwargs) -> plt.Figure:
        """Nelder-Mead optimization-path figure (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        return binary_figs.render_nelder_mead_path(self._bl, **kwargs)
