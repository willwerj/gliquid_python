"""
Authors: Joshua Willwerth, Shibo Tan, Abrar Rauf
Last Modified: June 16 2026
Description: This script is designed for the thermodynamic modeling of two-component systems.
It provides tools for fitting the non-ideal mixing parameters of the liquid phase from T=0K DFT-calculated phases and
digitized equilibrium phase boundary data. The data stored and produced may be visualized using the BLPlotter class
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import copy

# import json
import logging
import math

# import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from io import StringIO

import matplotlib.pyplot as plt

# import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element

# from collections import defaultdict
import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds
import gliquid.plotting.export as plot_export
import gliquid.solution as solution
from gliquid.hsx import HSX
from gliquid.phase import (
    EV_ATOM_TO_J_MOL,
    UNARY,
    ComponentRef,
    Phase,
    resolve_component_order,
    validate_and_format_system,
)
from gliquid.solution import DEFAULT_TAU, RKPolyExp, SolutionModel, comp_symbols, l_sym, t_sym

logger = logging.getLogger(__name__)

# Composition grid and base thermodynamic symbols live in gliquid.solution; the binary
# module keeps short local aliases for its own dense internal use.
_x_step = solution.x_step
_x_prec = solution.x_prec
_x_vals = solution.x_vals
xb_sym = comp_symbols(2)[0]

# Machine-readable causes of a BinaryLiquid ``init_error``. The bool stays the decision;
# ``BinaryLiquid.skip_reason`` records WHICH of these set it, so downstream consumers
# (campaign workbook, matrix colouring) can tell "no usable liquidus" from "not enough
# liquidus" from "not enough solid free energy" instead of collapsing all five into one
# string. Purely descriptive: adding or reading a reason never changes a fit outcome.
SKIP_NO_LIQUIDUS = "no_liquidus"  # from_cache: no digitized liquidus at all
SKIP_NARROW_SPAN = "narrow_liquidus_span"  # from_cache: endpoint span < comp_range_fit_lim
SKIP_SPARSE_LIQUIDUS = "sparse_liquidus_interior"  # from_cache: max_gap / covered_fraction gate
SKIP_SOLID_COVERAGE = "insufficient_solid_coverage"  # fit_parameters: solid-energy coverage gate
SKIP_MASK_FRACTION = "mask_fraction_exceeded"  # fit_parameters: ignored-range mask cap
SKIP_REASONS = (
    SKIP_NO_LIQUIDUS,
    SKIP_NARROW_SPAN,
    SKIP_SPARSE_LIQUIDUS,
    SKIP_SOLID_COVERAGE,
    SKIP_MASK_FRACTION,
)

# Every keyword ``BinaryLiquid.fit_parameters`` accepts through its ``**kwargs``: the options
# it reads itself, plus everything it forwards to nelder_mead / f / calculate_deviation_metrics.
# Anything outside this set and RETIRED_FIT_KWARGS raises, because a swallowed keyword reads
# to the caller as a setting that took effect when it did not.
FIT_KWARGS = frozenset(
    {
        # read by fit_parameters itself
        "ignore_ss",
        "disable_inv_constrs",
        "check_full_ss",
        "check_solid_coverage",
        "coverage_thresholds",
        "coverage_ss_narrow_tol",
        "coverage_dft_cover_tol",
        "coverage_ss_rescue_max_width",
        "enable_multi_threading",
        # forwarded to nelder_mead (tol/initial_guesses are set by fit_parameters, not the caller)
        "max_iter",
        # forwarded to f
        "check_lupis_elliott",
        "lupis_elliott_cfg",
        "use_lxb_penalty",
        "lxb_penalty_cfg",
        # forwarded through f to calculate_deviation_metrics
        "num_points",
        "ignored_ranges",
        "allow_sparse_data",
    }
)

# Removed keywords that stay recognized: callers passed them to switch a gate, so they warn
# and the run continues under the replacement behaviour rather than failing outright.
RETIRED_FIT_KWARGS = frozenset({"check_phase_mismatch"})

# Golden-ratio conjugate, (sqrt(5) - 1) / 2. A module-level singleton only because a call
# in a default argument is evaluated once at def time anyway; the value is unchanged.
DEFAULT_ASYM_SCALE = float((np.sqrt(5) - 1) / 2.0)


def get_hull_rel_enth_skew(dft_ch: PhaseDiagram) -> float:
    """
    Calculate the enthalpy skew of the DFT T=0K convex hull, relative to the ideal liquid enthalpy.
    """
    comps = api.pd_components(dft_ch)
    hull_skew = (UNARY[comps[0]].h_liq - UNARY[comps[1]].h_liq) / 4.0
    for e in dft_ch.stable_entries:
        xb_comp = e.composition.fractional_composition.as_dict().get(comps[1], 0)
        form_energy_kj = dft_ch.get_form_energy_per_atom(e) * 96485
        hull_skew += (xb_comp - 0.5) * form_energy_kj
    return float(hull_skew)


def get_hull_rel_mid_depth(dft_ch: PhaseDiagram) -> float:
    """
    Calculate the depth at the middle composition of the DFT T=0K convex hull, relative to the ideal liquid enthalpy.
    """
    comps = api.pd_components(dft_ch)
    lhs_ref = dft_ch.get_hull_energy_per_atom(Composition({comps[0]: 1}))
    rhs_ref = dft_ch.get_hull_energy_per_atom(Composition({comps[1]: 1}))
    hull_mid_ref = dft_ch.get_hull_energy_per_atom(Composition({c: 0.5 for c in comps}))
    e_depth = (
        hull_mid_ref * 96485
        - (lhs_ref * 96485 + UNARY[comps[0]].h_liq + rhs_ref * 96485 + UNARY[comps[1]].h_liq) / 2
    )
    return float(e_depth)


def build_init_triangle(
    guess_keys,
    dft_ch: PhaseDiagram = None,
    asym_scale=DEFAULT_ASYM_SCALE,
    invert_scale=-1.0321,
    tau: float = DEFAULT_TAU,
) -> list[list[float]]:
    """Initial Nelder-Mead simplex from DFT convex-hull statistics.

    ``guess_keys`` names the two guessed parameters (normally the format's
    ``guess_params``; the fitting code's H-S-partition pass passes
    ``('L0_a', 'L1_a')`` explicitly).
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

    pg1 = param_guesses[guess_keys[0]]
    pg2 = param_guesses[guess_keys[1]]

    return [[pg1[0], pg2[0]], [pg1[0], pg2[1]], [pg1[1], pg2[1]]]


# The binary liquid expressions are built by gliquid.solution.SolutionModel.binary_eqs()
# from the instance's RKPolyExp mixing model, with tau as a first-class kwarg —
# the retired module-global _L0_*/_L1_* expression seam has no replacement by design.


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
            this set are skipped — used when a continuous solid-solution phase covers that
            structure, so it does not also appear as a fixed-composition line compound.

    Returns:
        list[Phase]: Elemental polymorphs and line compounds sorted by composition,
        with the 'L' liquid sentinel appended last.
    """
    phases = []
    for comp in components:
        for polymorph in component_data[comp].polymorphs:
            if exclude_spacegroups and polymorph.spacegroup_number in exclude_spacegroups:
                continue
            # Fresh registry-safe copy; the insert(-1) ordering must be preserved.
            phases.insert(-1, replace(polymorph, points=[]))
    for entry in ch.stable_entries:
        composition = api.entry_frac_along(ch, entry, components)[0]
        if composition in [
            p.fraction_in(components)[0] for p in phases
        ]:  # Skip if a polymorph of the pure element is already added
            continue
        original = api.entry_original(entry)
        entry_data = getattr(original, "data", None) or {}
        is_imputed = bool(entry_data.get("imputed"))
        phases.insert(
            -1,
            Phase(
                phase_type="solid",
                name=entry_data.get("label", original.name) if is_imputed else original.name,
                composition=original.composition,
                enthalpy=EV_ATOM_TO_J_MOL * ch.get_form_energy_per_atom(entry),
                entropy=0,
                imputed=is_imputed,
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
        skip_reason (str | None): Which of the ``SKIP_*`` causes set ``init_error`` — the
            machine-readable refinement of that bool, ``None`` while it is False. The first
            cause seen wins (``_flag_skip``), so a system carries exactly one reason.
        sys_name (str): Binary system name.
        components (list): List of component names.
        component_data (dict): Thermodynamic data for components.
        mean_elt_tm (float): Mean elemental melting temperature.
        pd_ind (int | None): Index of the MPDS phase diagram used.
        mpds_json (dict): MPDS phase equilibrium data for the system.
        digitized_liq (list): Digitized liquidus data points.
        max_liq_temp (float | None): Maximum temperature on the liquidus line.
        min_liq_temp (float | None): Minimum temperature on the liquidus line.
        temp_range (list): Temperature range for calculations.
        comp_range_fit_lim (float): Composition range limit for fitting.
        liq_coverage (dict | None): ``mpds.liquidus_coverage`` metrics of the pre-fill
            digitized liquidus (``max_gap``, ``covered_fraction``, ``holes``, ...); the
            basis of ``from_cache``'s interior-coverage ``init_error``. Held in THIS
            object's component frame, not the raw json's — ``from_cache`` and
            ``with_component_order`` mirror it, because ``holes`` are positions.
        liquidus_holes (list): Read-only view of ``liq_coverage['holes']`` — the
            composition intervals where no liquidus was digitized at all.
        ignored_comp_ranges (list): Ignored composition ranges.
        dft_type (str): Functional used for DFT calculations.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.
        phases (list): List of phase data.
        xs_mix (RKPolyExp): The liquid's excess-mixing model — parameter values,
            formalism ('linear'/'combined'/'comb-exp'), and fitting metadata.
        param_format (str): The mixing model's formalism name (read-only view).
        eqs (dict): Dictionary of thermodynamic Sympy expressions.
        invariants (list): Identified invariant points.
        low_t_exp_phases (list): Low-temperature phases from MPDS JSON.
        coverage_report (SolidCoverageReport | None): Result of the solid-energy coverage
            assessment, populated by ``assess_solid_coverage`` (which ``fit_parameters``
            calls). Records which digitized solid phases have no free energy behind them and
            what fraction of the liquidus that leaves unconstrained.
        guess_symbols (list): Sympy symbols for corresponding to guessed parameters.
        constraints (list): Sympy equations used to store parameter constraints.
        init_triangle (np.ndarray): Initial simplex for Nelder-Mead optimization.
        nmpath (np.ndarray): Nelder-Mead optimization path, stored after running Nelder-Mead for plotting purposes.
        hsx (HSX): HSX object for phase diagram calculations.
    """

    def __init__(self, sys_name: str, components: list, init_error=False, **kwargs):
        self.init_error = init_error
        # Machine-readable refinement of init_error (one of the module's SKIP_* values).
        # Descriptive only — nothing branches on it inside the package.
        self.skip_reason = kwargs.get("skip_reason", None)
        # Set by find_invariant_points when MPDS labels report a full-composition solid
        # solution. Informational only — it records that MPDS spelled the field '(A, B)'.
        # The fit/skip decision belongs to the measured coverage report below, which catches
        # the wide fields this label match misses (Lu-Nd's spans all x but is labelled '(Lu)').
        self.full_comp_ss = False
        # Populated by assess_solid_coverage(); the basis of the fit/skip decision.
        self.coverage_report = None
        self.sys_name = sys_name
        self.components = components
        self.component_data = kwargs.get("component_data", {})
        self.mean_elt_tm = np.mean([self.component_data[comp].t_fusion for comp in self.components])
        self.pd_ind = kwargs.get("pd_ind", None)
        self.mpds_json = kwargs.get("mpds_json", {})
        self.digitized_liq = kwargs.get("digitized_liq", [])
        self.max_liq_temp = (
            max(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        )
        self.min_liq_temp = (
            min(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        )
        self.temp_range = kwargs.get("temp_range", [])
        self.comp_range_fit_lim = kwargs.get("comp_range_fit_lim", 0.7)
        # mpds.liquidus_coverage metrics of the PRE-fill digitized liquidus (None when no
        # liquidus was extracted); the basis of from_cache's interior-coverage init_error.
        # Expected in THIS object's component frame: 'holes' are positions, so a caller
        # passing a dict measured on a reversed-frame json must mirror it first
        # (mpds.mirror_liquidus_coverage), exactly as it must mirror digitized_liq.
        self.liq_coverage = kwargs.get("liq_coverage", None)
        self.ignored_comp_ranges = kwargs.get("ignored_comp_ranges", [])
        self.dft_type = kwargs.get("dft_type", "GGA")
        self.dft_ch = kwargs.get("dft_ch", None)
        self.phases = kwargs.get("phases", [])
        self.ss_models = kwargs.get("ss_models") or {}
        # Hull-inclusion switch for solid-solution phases (the fit_parameters ignore_ss
        # option flips it False). ss_models stay loaded either way, so SS plot overlays
        # keep working; re-enable with `bl.ss_in_hull = True; bl.update_phase_points()`.
        self.ss_in_hull = True
        if self.ss_models:
            self._ensure_solid_solution_phases()
        # exp(-T/tau) decay constant for the 'combined'/'comb-exp' mixing forms. Feeds the
        # L expressions, the initial-simplex scaling, and the Lupis-Elliott hs_ratio default
        # — matching the retired dev-script set_tau() that kept binary.tau in sync.
        # The liquid's excess model: pass a prebuilt RKPolyExp via 'xs_mix' (from_cache
        # does), or 'param_format'/'params'/'tau' for direct construction.
        xs_mix = kwargs.get("xs_mix")
        if xs_mix is not None:
            self.xs_mix = xs_mix
        else:
            self.xs_mix = RKPolyExp(
                kwargs.get("param_format", "linear"),
                kwargs.get("params", ()),
                tau=kwargs.get("tau", DEFAULT_TAU),
            )
        self.tau = self.xs_mix.tau
        self.init_triangle = kwargs.get(
            "init_triangle",
            build_init_triangle(self.xs_mix.format.guess_params, self.dft_ch, tau=self.tau),
        )
        self.eqs = kwargs.get("eqs", self._build_eqs())
        self.invariants = kwargs.get("invariants", None)
        self.low_t_exp_phases = None
        self.guess_symbols = None
        self.constraints = None
        self._ref_params = None
        self.nmpath = None
        self.hsx = None

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

    def _flag_skip(self, reason: str) -> None:
        """Set ``init_error`` and record WHY, keeping the first cause seen.

        The gates fire in pipeline order (liquidus first in ``from_cache``, then the
        solid-energy and mask gates inside ``fit_parameters``), so first-wins means the
        reason names the earliest thing that made the system unfittable — and every
        flagged system carries exactly one. Purely additive: the bool is what every
        existing consumer reads, and it is set here exactly as before.
        """
        self.init_error = True
        if self.skip_reason is None:
            self.skip_reason = reason

    def _phase_x(self, phase: Phase) -> float:
        """A fixed phase's binary evaluation-axis fraction (x of components[1])."""
        return phase.fraction_in(self.components)[0]

    def _build_eqs(self, ga_expr: sp.Expr = 0 * t_sym, gb_expr: sp.Expr = 0 * t_sym) -> dict:
        """The liquid eqs dict from the instance's mixing model and reference exprs."""
        return SolutionModel(
            tuple(self.components), (ga_expr, gb_expr), {(0, 1): self.xs_mix}
        ).binary_eqs()

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
        """
        Return a pickle-safe state for multiprocessing.

        Lambdified callables and HSX state are intentionally dropped and rebuilt on load.
        """
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
    def from_cache(
        cls,
        input,
        dft_type="GGA",
        pd_ind=None,
        params=(),
        param_format="linear",
        comp_range_fit_lim=0.7,
        include_imputed=False,
        solid_solutions: bool | None = None,
        ss_kwargs=None,
        tau=DEFAULT_TAU,
        liq_max_gap=None,
        liq_min_coverage=None,
    ) -> BinaryLiquid:
        """
        Initializes a BinaryLiquid object from cached data.

        Args:
            input (any): Binary system - can be either a list or hyphenated string
            dft_type (str): Type of DFT calculation.
            pd_ind (int | None): Index of MPDS binary phase diagram data in cache or from API call downloads
            params (list): Initial fitting parameters.
            param_format (str): Format of the excess mixing energy params, either 'linear', 'combined', or 'comb-exp'.
            comp_range_fit_lim (float): Minimum digitized-liquidus endpoint span, as a
                mole fraction, for the system to be considered fittable: a narrower span
                flags ``init_error`` with reason ``SKIP_NARROW_SPAN``.
            include_imputed (bool): Passed to ``api.get_dft_convexhull``. False (default)
                builds the hull from DFT entries only; True also admits entries cached by
                ``api.cache_imputed_entries`` (phase-energy imputation workflow).
            liq_max_gap (float | None): Per-call override of ``config.liquidus_max_gap`` for
                the interior-coverage gate: flag ``init_error`` when the widest composition
                gap between consecutive PRE-fill digitized liquidus points exceeds this.
            liq_min_coverage (float | None): Per-call override of
                ``config.liquidus_min_coverage``: flag ``init_error`` when less than this
                fraction of the stitched liquidus span is locally sampled
                (``mpds.liquidus_coverage`` with ``config.liquidus_gap_tol``).
            solid_solutions (bool | None): Tri-state SS switch. None (default) defers to
                the package-wide ``config.solid_solutions`` flag; an explicit True/False
                overrides config outright. When resolved truthy, load per-phase
                solid-solution models (solution.load_solid_solution_models), reconcile each
                element's liquid reference against the model's solid ladder, and represent
                the covered structures as continuous solid-solution phases instead of
                endpoint line compounds. Resolved False is zero behavior change; a system
                the omegas file does not cover also degrades to zero behavior change (the
                coverage gate in load_solid_solution_models returns {} without touching
                component_data).
            ss_kwargs (dict | None): Passed through to load_solid_solution_models
                (e.g. ref_mode, omegas_path, entries).
            tau (float): exp(-T/tau) decay constant for 'combined'/'comb-exp' mixing
                expressions (default 8000 K) — pass this instead of monkeypatching the
                retired binary._L0_*/_L1_* module globals.

        Returns:
            BinaryLiquid: Initialized BinaryLiquid object.
        """
        components, sys_name, order_changed = validate_and_format_system(input)
        if len(components) != 2:
            raise ValueError(f"BinaryLiquid needs exactly 2 components, got {components}.")
        xs_mix = RKPolyExp(param_format, params, tau=tau)
        if order_changed:  # Odd L orders flip sign when the component order changes
            xs_mix = xs_mix.swapped()

        ch, _ = api.get_dft_convexhull(components, dft_type, include_imputed=include_imputed)
        component_data = UNARY.component_data(components)
        for comp, ref in component_data.items():
            logger.info(
                f"{comp}: H_liq = {ref.h_liq} J/mol, S_liq = {ref.s_liq:.4f} J/(mol·K), "
                f"T_fusion = {ref.t_fusion} K, polymorphs = {len(ref.polymorphs)}"
            )
        mpds_json, (digitized_liq, is_partial) = mpds.load_mpds_data(components, pd_ind=pd_ind)
        # The raw json (and thus the liquidus) is digitized in its own alphabetical frame;
        # mirror the derived artifacts into the construction frame. mpds_json itself is
        # kept as loaded — frame-sensitive consumers convert at use (identify_mpds_phases
        # via mirror_mpds_phases, get_low_temp_phase_data internally).
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
            # Reconciles the liquid references inside component_data in place, so the eqs
            # built below automatically share the solid-solution reference frame — but only
            # for systems the omegas file actually covers. An uncovered system returns {}
            # here with component_data untouched (coverage gate in load_solid_solution_models),
            # so resolved-True-but-uncovered is byte-identical to the SS-off path.
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

        hull_points = np.array(
            [
                [p.fraction_in(components)[0], p.enthalpy]
                for p in phases
                if p.composition is not None
            ]
        )
        eqs["h_hull_interp"] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])

        comp_range = mpds_json.get(
            "comp_range", [0, 100]
        )  # Need to transform liquidus x when comp_range is partial?
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
        init_error = not bool(digitized_liq) or bool(
            (comp_range[1] - comp_range[0]) < comp_range_fit_lim
        )
        # Which of the two disjuncts above fired. Read off the same predicates, never
        # re-deciding them: the bool is authoritative, the reason only names its cause.
        skip_reason = None
        if init_error:
            skip_reason = SKIP_NO_LIQUIDUS if not digitized_liq else SKIP_NARROW_SPAN

        # Endpoint span alone cannot see interior holes: extract_digitized_liquidus
        # linearly fills every gap wider than 0.06 before digitized_liq reaches this
        # frame, so a liquidus digitized only near the pure ends (Bi-Si class) arrives
        # span-complete with most of its interior fabricated. Measure the PRE-fill curve.
        # The gate below reads only mirror-invariant scalars, so the raw-frame json is
        # fine to measure -- but liq_coverage also carries 'holes', which are POSITIONS,
        # so mirror the dict onto the construction frame exactly as digitized_liq was
        # above. An unmirrored hole would mask the wrong end of the diagram, silently.
        liq_coverage = mpds.liquidus_coverage(mpds_json) if digitized_liq else None
        if liq_coverage is not None and mpds_frame_reversed:
            liq_coverage = mpds.mirror_liquidus_coverage(liq_coverage)
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
            "comp_range_fit_lim": comp_range_fit_lim,
            "liq_coverage": liq_coverage,
            "skip_reason": skip_reason,
            "ss_models": ss_models,
        }
        return cls(sys_name, components, init_error, **kwargs)

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

        The presentation frame is complete: xs_mix swaps odd RK orders, the digitized
        liquidus mirrors, phases/ss_models re-derive from their order-independent
        schemas, and the hull rebuilds with reordered elements (frame-aware MPDS
        consumers then normalize to it). Fitting state (nmpath, invariants,
        constraints) deliberately does NOT transfer — fit in the construction frame.
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
            comp_range_fit_lim=self.comp_range_fit_lim,
            # Scalars are mirror-invariant, but 'holes' are positions: re-frame them with
            # digitized_liq or the deviation metrics mask the opposite end of the diagram.
            liq_coverage=(
                mpds.mirror_liquidus_coverage(self.liq_coverage)
                if self.liq_coverage
                else self.liq_coverage
            ),
            dft_type=self.dft_type,
            dft_ch=ch,
            # SS phases are dropped here and re-created by the constructor in the new
            # frame (their attached SolutionModels are frame-sensitive).
            phases=[replace(p, points=[]) for p in self.phases if p.name not in self.ss_models],
            ss_models=self.ss_models,
            xs_mix=self.xs_mix.swapped(),
            pd_ind=self.pd_ind,
        )
        clone._rebuild_thermodynamic_expressions()
        return clone

    def solid_solution_h_s(
        self, ss_name: str, x_vals: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """H(x) and S(x) of a solid-solution phase across the composition grid.

        Evaluated through the phase's SolutionModel (regular solution: ideal
        configurational entropy plus per-element enthalpy/entropy offsets plus the
        single interaction term Omega*x_a*x_b) — H and S are analytically
        T-independent for this model, so the evaluation temperature is arbitrary.
        """
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
        # Solid solutions may be sampled finer than the liquid: the presentation hull
        # refines them to close their field termini, while the liquid stays on the coarse
        # grid so the liquidus -- and every non-SS figure -- is untouched.
        ss_grid = grid if ss_x_vals is None else np.asarray(ss_x_vals, dtype=float)
        data = {
            "X": [float(x) for x in grid],
            "S": [s_a] + list(liq_s_vals) + [s_b],
            "H": [h_a] + list(liq_h_vals) + [h_b],
            "Phase Name": ["L"] * len(grid),
        }
        for phase in self.phases:
            if phase.is_solution:
                # A continuous solid phase contributes a full composition grid of rows, sharing
                # the liquid's grid-evaluation shape; the 'L' sentinel (and any solution phase
                # without a model) contributes nothing here. ss_in_hull=False (the
                # fit_parameters ignore_ss option) drops the SS rows entirely.
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
            data["X"].append(round(self._phase_x(phase), _x_prec))
            data["Phase Name"].append(phase.name)
        return data

    def to_HSX(
        self, fmt="dict", x_vals: np.ndarray | None = None, ss_x_vals: np.ndarray | None = None
    ) -> dict | pd.DataFrame:
        """
        Converts phase data into HSX format for further calculations.

        The liquid H(x)/S(x) rows are evaluated at ONE temperature, the mean
        elemental melting point (``mean_elt_tm``). For the 'linear' model they are
        analytically T-independent; for the exp(−T/τ)-enveloped models ('combined',
        'comb-exp') this fixes the envelope at a representative temperature — the
        pinned behavior the hull goldens freeze.

        Args:
            fmt (str): Output format ('dict' or 'dataframe').
            x_vals (np.ndarray | None): Composition grid the liquid rows are evaluated on.
                Defaults to the module grid (``solution.x_vals``).
            ss_x_vals (np.ndarray | None): Separate composition grid for the continuous
                solid-solution phases, so they can be sampled finer than the liquid
                (``refined_hsx``). Defaults to ``x_vals``.

        Returns:
            dict | pd.DataFrame: Data in HSX format.
        """
        grid = _x_vals if x_vals is None else np.asarray(x_vals, dtype=float)
        x_inner = grid[1:-1]
        params = self.xs_mix.values

        # Endpoint H and S (always T-independent: pure-element fusion enthalpy/entropy)
        comp_a = self.component_data[self.components[0]]
        comp_b = self.component_data[self.components[1]]
        s_a = 0 if comp_a.h_liq == 0 else comp_a.s_liq
        s_b = 0 if comp_b.h_liq == 0 else comp_b.s_liq
        h_a = comp_a.h_liq
        h_b = comp_b.h_liq

        # For the linear model, liquid H and S are analytically T-independent,
        liq_h_vals = (
            self.eqs["h_liq_lambdified"](x_inner, self.mean_elt_tm, *params).flatten().tolist()
        )
        liq_s_vals = (
            self.eqs["s_liq_lambdified"](x_inner, self.mean_elt_tm, *params).flatten().tolist()
        )

        # Build output data
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
        Calculates the phase points for given parameter values using the HSX class.

        This method converts phase data into the HSX form and uses HSX code to calculate the liquidus
        and low-temperature DFT phase boundaries. The instance's HSX object is updated in
        place across parameter updates (the phase set is constant there); it is only
        rebuilt when absent or when the phase list itself changed.

        Returns:
            data (dict): A dictionary containing the phase data in HSX format, including phase names and components.
        """
        data = self.to_HSX()
        # ss_in_hull=False (the ignore_ss fitting option) leaves solid-solution phases
        # out of the hull: their names drop from the HSX phase list (which also forces
        # the rebuild below when the flag flips) and their points empty.
        phase_names = self._hsx_phase_names()
        conds = [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15]
        if self.hsx is None or self.hsx.phases != phase_names:
            self.hsx = HSX({"data": data, "phases": phase_names, "comps": self.components}, conds)
        else:
            self.hsx.conds = conds
            self.hsx.set_data(data)
        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase.points = phase_points.get(phase.name, [])
        return data

    def _hsx_phase_names(self) -> list[str]:
        """Phase names entering the hull; SS names drop when ``ss_in_hull`` is off."""
        ss_in_hull = getattr(self, "ss_in_hull", True)
        return [
            phase.name
            for phase in self.phases
            if ss_in_hull or not (phase.is_solution and phase.name != "L")
        ]

    def refined_hsx(self, factor: int = 5) -> HSX:
        """A PRESENTATION-ONLY hull with solution phases on an interior-refined grid.

        Returns a fresh HSX; ``self.hsx`` and ``phase.points`` are left exactly as the
        fitting path produced them, so ``fit_parameters``, its objective, and the pinned
        hull goldens are untouched. The cost is small -- Hf-Y at ``factor=5`` is ~2000 hull
        points and ~0.1 s -- and it is what lets a solid-solution field terminate in a
        single apex instead of a vertical face straddling several equilibria.

        Only the SOLID solutions are refined. The liquid keeps the coarse grid, so the
        liquidus -- and therefore every diagram of a system with no SS models -- is
        bit-identical to the fitted hull's. ``factor=1`` reproduces the coarse hull exactly.
        """
        data = self.to_HSX(ss_x_vals=solution.refined_x_vals(factor))
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
        Get a copy of the current parameters such that the BinaryLiquid object will not be modified accidentally.

        Returns:
            list: The mixing model's parameter values in its flat layout
            (``xs_mix.format.param_names``, e.g. [L0_a, L0_b, L1_a, L1_b]).
        """
        return self.xs_mix.values

    def update_params(self, input) -> None:
        """
        Update the non-ideal mixing parameters with validity checks,
        then recalculate phase boundaries for the new parameters.

        Args:
            input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

        Returns:
            None
        """
        self.xs_mix.update(input)
        self.update_phase_points()

    def find_invariant_points(
        self, verbose=False, check_full_ss=True, t_tol=15
    ) -> tuple[list[dict], list[dict]]:
        """
        Identifies invariant points in the MPDS data using the provided MPDS JSON and liquidus data.

        Thin wrapper over :func:`gliquid.mpds.identify_invariant_points`, where the
        algorithm lives with the rest of the MPDS phase policy. Assigns
        ``self.invariants`` / ``self.low_t_exp_phases``. When the system is a
        full-composition solid solution this NOTIFIES and sets ``self.full_comp_ss``
        (not ``init_error``); ``fit_parameters`` then skips fitting only if the object
        lacks solid-solution phase energies (``ss_models``).

        This function does not consider DFT phases, which may differ in composition from the MPDS data. It requires both
        complete liquidus and JSON data for a binary system. (The DFT-vs-MPDS mismatch
        chart lives in ``mpds.print_phase_mismatch_chart``, called from ``fit_parameters``.)

        Args:
            verbose (bool): If True, outputs additional debugging information.
            t_tol (int): Temperature tolerance for invariant point identification.
            check_full_ss (bool): If True, checks for full composition range solid solutions.

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
        )
        self.invariants = invariants
        self.low_t_exp_phases = low_t_exp_phases
        if full_comp_ss:
            self.full_comp_ss = True
        return self.invariants, self.low_t_exp_phases

    def _interior_dft_comps(self) -> list[float]:
        """Interior (0 < x < 1) fixed-composition DFT phase compositions.

        Solid-solution phases are excluded: they carry no composition (``is_solution``), so
        they cannot serve as a point solid reference. Single definition shared by the coverage
        gate and the ignored-range masking so the two cannot drift apart.
        """
        return [
            self._phase_x(p)
            for p in self.phases
            if not p.is_solution and 0.0 < round(self._phase_x(p), 6) < 1.0
        ]

    def assess_solid_coverage(self, **overrides) -> mpds.SolidCoverageReport:
        """Measure how much of this system's liquidus has no solid free-energy reference.

        Thin wrapper over :func:`gliquid.mpds.assess_solid_coverage` -- the phase policy lives
        with the rest of the MPDS phase handling; this method only supplies the DFT hull
        compositions and the loaded solid-solution models. Assigns and returns
        ``self.coverage_report``.

        Requires ``self.invariants`` to be populated for the compound-field masking to be
        tight; call :meth:`find_invariant_points` first (``fit_parameters`` does). With no
        invariants the masking falls back to the nearest supported anchors, which is
        conservative rather than wrong.

        Args:
            **overrides: Passed through (``ss_narrow_tol``, ``dft_cover_tol``,
                ``ss_rescue_max_width``, ``thresholds``); ``None`` values defer to ``config``.

        Raises:
            config.CacheModeError: when ``self.mpds_json`` is a LEAN record. THE
                highest-value guard in the lean-mode contract: a lean record has no
                ``shapes``, an empty phase list scores as "zero reported compounds, nothing
                unsupported", and this gate would PASS — a silent wrong answer of exactly
                the class it was built to prevent. Guarded here, before the call, so the
                message names this method rather than ``identify_mpds_phases``.
        """
        mpds._require_full_record(self.mpds_json, "BinaryLiquid.assess_solid_coverage")
        # Pass components explicitly: '(X)' component-phase recognition needs the element
        # symbols, and ~13% of cached jsons carry no 'chemical_elements' block. Only the SET
        # is read, so supplying them before the frame mirror below is safe.
        phases = mpds.identify_mpds_phases(
            self.mpds_json, with_structure=True, elements=self.components
        )
        if not mpds.mpds_frame_matches(self.mpds_json, self.components):
            phases = mpds.mirror_mpds_phases(phases)
        span = (
            [self.digitized_liq[0][0], self.digitized_liq[-1][0]]
            if self.digitized_liq
            else [0.0, 1.0]
        )
        self.coverage_report = mpds.assess_solid_coverage(
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
                - Backward compatibility: 'scale' is accepted as an alias for 'strength'

        Returns:
            float: A penalty greater than 1.0 if the parameters violate the Lupis-Elliott sign constraints,
                   otherwise returns 1.0.
        """
        penalty_cfg = penalty_cfg or {}
        global_strength = float(penalty_cfg.get("strength", penalty_cfg.get("scale", 7.5e-3)))
        entropy_scale = float(penalty_cfg.get("hs_ratio", getattr(self, "tau", DEFAULT_TAU)))

        def resolve_term_cfg(term: str, default: float | None = None) -> float:
            if default is None:
                default = global_strength
            term_cfg = penalty_cfg.get(term, {})
            if not isinstance(term_cfg, dict):
                term_cfg = {}
            return float(term_cfg.get("strength", term_cfg.get("scale", default)))

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

        # Which RK orders the sign constraint inspects is format metadata: comb-exp
        # penalizes L0 only; linear/combined inspect L0 and L1 (L1 strength defaults 0).
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

        By default, only L0_b is penalized. L1_b is never penalized unless
        ``apply_l1`` is True or an explicit ``l1`` config is provided.

        Args:
            penalty_cfg (dict | None): Dictionary with optional keys:
                - 'l0': {'weight', 'median', 'mad', 'exponent'} — L0_b term config.
                  Defaults: weight=0.10, exponent=2.5. 'median' and 'mad' must be provided.
                - 'l1': {'weight', 'median', 'mad', 'exponent'} — L1_b term config.
                  Not applied by default; requires 'apply_l1': True or explicit 'l1' config.
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

        use_l1_default = False
        use_l1 = bool(penalty_cfg.get("apply_l1", use_l1_default))

        total = _term(self.get_L0_b(), penalty_cfg.get("l0"))
        if use_l1:
            total += _term(self.get_L1_b(), penalty_cfg.get("l1"))
        return float(1.0 + total)

    def _stable_solid_gibbs_at_T(self, comp_x: float, temp_K: float) -> float:
        """
        Returns the Gibbs energy of the thermodynamically stable solid polymorph
        at a given temperature for a pure elemental endpoint.

        For pure elements (comp_x == 0 or 1), the ground state has G = H - T*S = 0 at all T.
        Higher-temperature polymorphs stored in component_data['polymorphs'] may become
        stable above their transition temperature.  This method selects the polymorph
        with the lowest Gibbs energy at *temp_K* and returns that value.

        For non-elemental compositions this returns the enthalpy of the nearest
        DFT phase unchanged (polymorphs are only tracked for pure elements).

        Args:
            comp_x (float): Composition fraction of component B (0 or 1 for pure elements).
            temp_K (float): Temperature in Kelvin at which to evaluate stability.

        Returns:
            float: Gibbs energy (J/mol) of the stable solid at (comp_x, temp_K).
        """
        # Only apply polymorph correction for the pure-element endpoints
        if comp_x not in (0.0, 1.0):
            nearest = min(
                (p for p in self.phases if not p.is_solution),
                key=lambda p: abs(self._phase_x(p) - comp_x),
                default=None,
            )
            return nearest.enthalpy if nearest else 0.0

        comp_name = self.components[int(comp_x)]
        polymorphs = self.component_data[comp_name].polymorphs

        # Ground state: H=0, S=0 → G=0 at all T
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
            if last_coords and coords[0] - last_coords[0] > tol:
                return False
            last_coords = coords
        return True

    def calculate_deviation_metrics(
        self, num_points=30, **kwargs
    ) -> tuple[float, float, float, float]:
        """
        Calculates the deviation metrics between the digitized (measured) liquidus and the generated liquidus.

        The fit is graded only where a digitized liquidus actually exists. Two kinds of
        composition interval are excluded from the mesh and from both curves:
        ``self.ignored_comp_ranges`` (un-modellable, and switchable off with
        ``ignored_ranges=False``) and ``self.liquidus_holes`` (undigitized, and never
        switchable — see the comment on the mask below). Masking costs mesh points, so an
        extensively holed system can fall under the 10-point floor and report ``inf``
        rather than a flattering score over a handful of survivors.

        Args:
            num_points (int): Number of points to sample for deviation metrics. Default is 30.
            **kwargs: Additional keyword arguments:
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - allow_sparse_data (bool): If True, allows for small composition ranges in the generated points.

        Returns:
            tuple: MAE, RMSE, MAPE, RMSPE of the liquidus in temperature units (K) and percentage (%).
        """
        # Convert liquidus data to numpy arrays
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
        # Composition intervals this comparison must not see, from two sources but through
        # ONE mask -- the digitized array, the generated array and the evaluation mesh have
        # to stay consistent with each other, so there is exactly one masking path:
        #   * self.ignored_comp_ranges -- stretches fit_parameters judged un-modellable
        #     (invariants it could not resolve, compounds with no DFT energy). Optional:
        #     ignored_ranges=False asks for the metric without them.
        #   * self.liquidus_holes -- stretches where MPDS digitized no liquidus at all,
        #     between disjoint 'L' shapes. NOT optional, and not a judgement call: the
        #     nearest-neighbour lookup below has nothing to return inside one but the
        #     temperature of whichever region edge happens to be closer, a fabricated flat
        #     step. Grading a fit against that measures the digitizer, not the model.
        masked_ranges = list(self.ignored_comp_ranges) if kwargs.get("ignored_ranges", True) else []
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
            ignored_comp_lims = [generated[0, 0], generated[-1, 0]]
        else:
            ignored_comp_lims = [0, 1]

        x_min = max(_x_step, digitized_liq_lims[0], ignored_comp_lims[0])
        x_max = min(1 - _x_step, digitized_liq_lims[1], ignored_comp_lims[1])
        if x_max - x_min < self.comp_range_fit_lim:
            logger.error(
                f"Error: Large composition range not considered [{self.sys_name}] (remaining range = {[float(x_min), float(x_max)]})"
            )
            return (float("inf"),) * 4

        # The mesh spans the full endpoint range and then drops the masked intervals, so a
        # hole costs its share of the points rather than being resampled around -- the same
        # bargain ignored_comp_ranges has always made.
        x_mesh = np.linspace(x_min, x_max, num_points)
        mesh_mask = np.ones_like(x_mesh, dtype=bool)
        for lower, upper in masked_ranges:
            mesh_mask = mesh_mask & ((x_mesh < lower) | (x_mesh > upper))
        x_mesh = x_mesh[mesh_mask]

        if len(digitized) < len(x_mesh):
            x_mesh = np.array([pt[0] for pt in digitized])

        # allow_sparse_data waives the 10-point floor but not an EMPTY mesh: the means
        # below would be nan, which reads as a number downstream instead of a failure.
        if len(x_mesh) == 0 or (not kwargs.get("allow_sparse_data", False) and len(x_mesh) < 10):
            logger.error(
                f"Error: Not enough comparison points for accurate deviation metrics calculation [{self.sys_name}]."
            )
            return (float("inf"),) * 4

        # Find closest temperature values for each evaluation point
        Y1 = np.array([digitized[np.argmin(np.abs(digitized[:, 0] - x)), 1] for x in x_mesh])
        Y2 = np.array([generated[np.argmin(np.abs(generated[:, 0] - x)), 1] for x in x_mesh])
        diffs = np.abs(Y1 - Y2)
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
            **kwargs (dict): Additional keyworded arguments to dictate which constraints are applied.

        Returns:
            float: Generated liquidus mean absolute error (MAE) for the given parameter values.
        """
        verbose = kwargs.get("verbose", False)
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, guess)}
        self.solve_params_from_constraints(guess_dict)

        if self.h0_below_ch():
            if verbose:
                logger.info(f"T=0K enthalpy constraint violated for params {self.get_params()}")
            return float("inf")

        # Update HSX object and generate new phase points
        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            logger.error(e)
            return float("inf")

        # Check if generated liquidus is continuous
        if not self.liquidus_is_continuous():
            if verbose:
                logger.info(
                    f"Liquidus continuity constraint violated for guess {self.get_params()}"
                )
            return float("inf")

        # Evaluate the liquidus temperature deviation metrics
        _, f_val, _, _ = self.calculate_deviation_metrics(**kwargs)
        if kwargs.get("check_lupis_elliott", True):
            f_val = f_val * self.lupis_elliott_penalty(kwargs.get("lupis_elliott_cfg"))

        # Apply lxb penalty using distribution priors on L0_b/L1_b.
        if kwargs.get("use_lxb_penalty", self.xs_mix.format.lxb_default):
            f_val *= self.lxb_penalty(kwargs.get("lxb_penalty_cfg"))

        return f_val

    def nelder_mead(
        self, max_iter=64, tol=0.05, verbose=False, initial_guesses=None, **kwargs
    ) -> tuple[float, float, np.ndarray]:
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter (int): Maximum number of iterations. Default is 64.
            tol (float): Tolerance for algorithm convergence. Default is 0.05.
            verbose (bool): If True, print updates during optimization. Default is False.
            initial_guesses (list): Reasonable initial values for guessed parameters, determined by self.guess_symbols.
            **kwargs (dict): Additional keyworded arguments passed to f().

        Returns:
            tuple: MAE, RMSE, and Nelder-Mead optimization path.
        """
        if not initial_guesses:
            initial_guesses = self.init_triangle
        if self.guess_symbols is not None and len(self.guess_symbols) != 2:
            raise NotImplementedError(
                f"nelder_mead optimizes a 2-D simplex; got {len(self.guess_symbols)} guess "
                f"symbols. Formats with a different free-parameter count need a "
                f"generalized optimizer."
            )

        # Initial guesses for parameters
        x0 = np.array(initial_guesses, dtype=float)
        n_params = self.xs_mix.n_params
        self.nmpath = np.empty((3, n_params + 1, max_iter), dtype=float)
        initial_time = time.time()

        if verbose:
            logger.info("--- Beginning Nelder-Mead optimization ---")

        for i in range(max_iter):
            start_time = time.time()
            if verbose:
                logger.info(f"Iteration # {i}")

            f_vals = np.empty(x0.shape[0])
            param_vals = np.empty((x0.shape[0], n_params))
            for idx, x in enumerate(x0):
                f_vals[idx] = self.f(x, **kwargs)  # 3 f() calls
                param_vals[idx] = self.get_params()
            self.nmpath[:, :-1, i] = param_vals
            self.nmpath[:, -1, i] = f_vals
            iworst = np.argmax(f_vals)
            ibest = np.argmin(f_vals)

            # Check if all current simplex vertices are invalid
            if iworst == ibest:
                self.nmpath = self.nmpath[:, :, :i]
                if i == 0:
                    raise RuntimeError(
                        "Nelder-Mead initialization has produced a simplex with invalid vertices."
                    )
                else:
                    raise RuntimeError(
                        "Nelder-Mead algorithm has produced a simplex with invalid vertices."
                    )

            centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
            xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
            f_xreflect = self.f(xreflect, **kwargs)  # 1 f() call

            # Simplex reflection step
            if f_vals[iworst] <= f_xreflect < f_vals[2]:
                x0[iworst, :] = xreflect
            # Simplex expansion step
            elif f_xreflect < f_vals[ibest]:
                xexp = centroid + 2.0 * (xreflect - centroid)
                if self.f(xexp, **kwargs) < f_xreflect:  # 1 f() call
                    x0[iworst, :] = xexp
                else:
                    x0[iworst, :] = xreflect
            # Simplex contraction step
            else:
                if f_xreflect < f_vals[2]:
                    xcontract = centroid + 0.5 * (xreflect - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs):  # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                else:
                    xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs):  # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

            if verbose:
                guess_dict = {
                    symbol: float(guess) for symbol, guess in zip(self.guess_symbols, x0[ibest, :])
                }
                logger.info(f"Best guess: {guess_dict} f={f_vals[ibest]}")
                logger.info(f"Height of triangle = {2 * np.max(np.abs(x0 - centroid))}")
                logger.info("--- %s seconds elapsed ---" % (time.time() - start_time))

            # Convergence check
            if np.max(np.abs(x0 - centroid)) < tol:
                f_val = self.f(x0[ibest, :], **kwargs)
                kwargs["ignored_ranges"] = False  # Re-include all points for final metrics
                mae, rmse, mape, rmspe = self.calculate_deviation_metrics(**kwargs)
                if verbose:
                    logger.info(
                        "--- Nelder-Mead converged in %s seconds ---" % (time.time() - initial_time)
                    )
                    logger.info(
                        f"Mean temperature deviation per point between liquidus curves = {mae}"
                    )
                self.nmpath = self.nmpath[:, :, : i + 1]
                return f_val, (mae, rmse, mape, rmspe), self.nmpath

        raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")

    def fit_parameters(
        self, verbose=False, n_opts=1, t_tol=15, enable_multi_threading=False, **kwargs
    ) -> list[dict]:
        """
        Fit the liquidus non-ideal mixing parameters for a binary system.

        This function utilizes the Nelder-Mead algorithm to minimize the temperature deviation in the liquidus.

        Solid-solution caveat: continuous solid-solution phases (``is_solution=True`` entries
        backed by ``ss_models``) are excluded from invariant-point matching like the liquid, and
        interior solid Gibbs energies in the invariant constraints remain T-independent line-
        compound enthalpies. SS-aware invariant constraints (a T- and x-dependent solid G) are
        explicitly future work.

        Args:
            verbose (bool): If True, prints detailed progress and results.
            n_opts (int): Number of optimization attempts. Updates the BinaryLiquid object to reflect the lowest MAE fit
            t_tol (float): Temperature tolerance for invariant point identification.
            enable_multi_threading (bool): If True, uses ThreadPoolExecutor for parallel
                multi-attempt optimization. Defaults to False.
            **kwargs: Additional keyword arguments for fitting options or arguments passed to nelder-mead.
                Restricted to the names below (module constant ``FIT_KWARGS``) plus the retired
                names in ``RETIRED_FIT_KWARGS``; anything else raises ``TypeError``.
                - max_iter (int): Maximum number of iterations for the Nelder-Mead algorithm. Default is 64.
                - ignore_ss (bool): If True, exclude solid-solution phases from the hull for the
                    duration of the fit (sets ``self.ss_in_hull = False``; SS inclusion in fitting
                    is not yet benchmarked). ``ss_models`` stay loaded, so SS plot overlays keep
                    working. Re-enable with ``bl.ss_in_hull = True; bl.update_phase_points()``.
                - disable_inv_constrs (bool): If True, does not use invariant points as constraints.
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - check_full_ss (bool): If True, emits the full-composition-solid-solution
                    notice from invariant identification. Informational; see check_solid_coverage
                    for the fit/skip decision.
                - check_solid_coverage (bool): If True (default), skip the system when too much
                    of its liquidus has no solid free-energy reference behind it — either a wide
                    solid-solution field with no loaded solid-solution model, or too many
                    reported compounds absent from the DFT hull. Note the verdict depends on
                    which energies were loaded, i.e. on ``config.solid_solutions`` /
                    ``ss_ref_mode``: the same system can pass with solid solutions on and fail
                    with them off. The full assessment lands on ``self.coverage_report``.
                - coverage_thresholds (dict | None): Override the config decision thresholds
                    ('skip_frac', 'min_missing', 'missing_frac') for this call.
                - coverage_ss_narrow_tol / coverage_dft_cover_tol /
                    coverage_ss_rescue_max_width (float | None): Per-call overrides of the
                    corresponding config values. None defers to config.
                - check_lupis_elliott (bool): If True, applies Lupis-Elliott sign constraints as a penalty.
                - lupis_elliott_cfg (dict): Optional Lupis-Elliott penalty config.
                    Supported keys:
                        * 'strength' (float): global default penalty strength. Default: 7.5E-3.
                        * 'l0' (dict): optional per-term override with {'strength'}.
                        * 'l1' (dict): optional per-term override with {'strength'}.
                          L1 is not penalized unless this key is explicitly provided.
                - use_lxb_penalty (bool): If True, applies the distribution-aware lxb penalty.
                    Defaults to True for 'comb-exp' models, False for 'combined'/'linear'.
                - lxb_penalty_cfg (dict): Distribution prior configuration dictionary.
                    Supported keys:
                        * 'l0': {'weight', 'median', 'mad', 'exponent'}
                          Defaults: weight=0.10, exponent=2.5. 'median'/'mad' must be provided.
                        * 'l1': {'weight', 'median', 'mad', 'exponent'}
                          Not applied by default; use 'apply_l1': True to enable.
                        * 'apply_l1' (bool): force-enable/disable L1 term. Default: False.

        Returns:
            list[dict]: Parameter fitting data containing results of all optimization attempts.

        Raises:
            TypeError: on an unrecognized keyword argument.
            config.CacheModeError: when ``self.mpds_json`` is a LEAN record (see
                ``mpds.record_mode``) and the two checks it cannot support have not both
                been switched off explicitly.
        """

        # Unrecognized keywords are a caller error, not a no-op: **kwargs used to swallow a
        # misspelled or removed option and run on as if it had been applied. Checked before
        # anything else so nothing is mutated on the way to the raise.
        unknown = sorted(set(kwargs) - FIT_KWARGS - RETIRED_FIT_KWARGS)
        if unknown:
            raise TypeError(
                f"fit_parameters() got "
                f"{'unexpected keyword arguments' if len(unknown) > 1 else 'an unexpected keyword argument'}: "
                f"{', '.join(repr(key) for key in unknown)}. Accepted: "
                f"{', '.join(sorted(FIT_KWARGS))}."
            )

        # A LEAN MPDS record (liquidus only, no digitized 'shapes') cannot support either
        # invariant constraints or the solid-coverage gate. Checked UP FRONT, before
        # anything is mutated and before any optimization runs, because the dangerous
        # outcome is not a crash: without shapes the invariant list and the phase list both
        # come back empty, the coverage gate reads "nothing unsupported" and PASSES, and the
        # caller gets a plausible-looking fit that was never constrained. There is no
        # auto-degrade — a fit without invariant constraints and without the coverage gate
        # is a DIFFERENT fit, and returning it silently under the same call is the failure
        # mode. Asking for it explicitly is fine, which is what the two kwargs below are.
        # getattr, not self.mpds_json: this check is deliberately the FIRST thing after the
        # kwarg gate, which puts it ahead of the retired-kwarg notice and of every other
        # diagnostic. A bare BinaryLiquid.__new__ instance has no attributes at all, and an
        # AttributeError raised from here would swallow those diagnostics instead of the
        # object's own later failure reporting them (tests/test_notebook_imports.py pins
        # exactly that). record_mode(None) is 'empty', so an unbuilt object falls through.
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

        if kwargs.get("ignore_ss", False):
            self.ss_in_hull = False  # persists (deepcopied into workers; final state consistent)

        if "check_phase_mismatch" in kwargs:
            # Retired kwargs warn rather than silently no-opping: callers passed
            # check_phase_mismatch=False, so a quiet no-op would leave them believing a
            # gate is active that is not.
            logger.warning(
                "'check_phase_mismatch' is retired and ignored; the DFT phase-count "
                "check is now part of the solid-energy coverage gate. Use "
                "'check_solid_coverage' (and 'coverage_thresholds') instead."
            )

        if self.digitized_liq is None:
            logger.warning(
                "System missing liquidus data! Ensure that 'BinaryLiquid.digitized_liq' is not empty!"
            )
            return []

        def find_nearest_phase(composition, tol=0.02):
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

        def merge_comp_ranges(ranges):
            """Merge overlapping/adjacent [lo, hi] composition ranges into a minimal set."""
            if not ranges:
                return []
            ordered = sorted([sorted(r) for r in ranges])
            merged = [list(ordered[0])]
            for lo, hi in ordered[1:]:
                if lo <= merged[-1][1] + 1e-9:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            return merged

        # Find invariant points. A full-composition solid solution sets self.full_comp_ss,
        # which is now informational only — the fit/skip decision is the measured coverage
        # assessment below.
        if (
            self.invariants is None
            and self.low_t_exp_phases is None
            and not kwargs.get("disable_inv_constrs", False)
        ):
            self.invariants, self.low_t_exp_phases = self.find_invariant_points(
                verbose=verbose, t_tol=t_tol, check_full_ss=kwargs.get("check_full_ss", True)
            )
            if verbose:
                # DFT-vs-MPDS low-T phase mismatch chart (moved here from
                # identify_invariant_points: only fit_parameters has DFT phases in scope).
                mpds.print_phase_mismatch_chart(self.low_t_exp_phases, self._interior_dft_comps())

        # Single measured admission criterion: how much of the liquidus is conjugate to a
        # solid whose free energy we cannot evaluate. Supersedes the two gates it replaces —
        # the count of near-liquidus MPDS compounds missing from DFT, and the '(A, B)'
        # label-string full-composition-SS check — which disagreed with each other because
        # they measured incommensurate things.
        if kwargs.get("check_solid_coverage", True):
            coverage = self.assess_solid_coverage(
                ss_narrow_tol=kwargs.get("coverage_ss_narrow_tol"),
                dft_cover_tol=kwargs.get("coverage_dft_cover_tol"),
                ss_rescue_max_width=kwargs.get("coverage_ss_rescue_max_width"),
                thresholds=kwargs.get("coverage_thresholds"),
            )
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
        auto_ignored_ranges = not bool(self.ignored_comp_ranges)

        # Conservative masking: bound auto-detected ignored ranges to the local missing-phase
        # field (nearest flanking invariant) instead of spanning to a pure-element endpoint, and
        # flag-and-skip if the un-modellable masked fraction of the liquidus exceeds a cap.
        mask_fraction_cap = 0.60
        # Same tolerance the coverage gate uses for "a DFT phase covers this composition",
        # sourced from config so the two cannot drift apart.
        dft_cover_tol = config.coverage_dft_cover_tol
        _inv_comps = sorted(iv["comp"] for iv in (self.invariants or []) if "comp" in iv)
        _dft_comps = self._interior_dft_comps()

        def nearest_inv_left(comp, floor=0):
            cands = [x for x in _inv_comps if x < comp - 1e-6]
            return max(cands) if cands else floor

        def nearest_inv_right(comp, ceil=1):
            cands = [x for x in _inv_comps if x > comp + 1e-6]
            return min(cands) if cands else ceil

        def dft_covers(comp):
            """True if a stable interior DFT compound sits within dft_cover_tol of `comp`.
            When DFT has a (possibly off-stoichiometry) compound near a missing experimental
            phase, the convex hull still provides a valid solid reference for the liquidus
            there, so masking that region is unwarranted (e.g. Au-Ca: DFT compounds are
            shifted from the measured ones but coverage is adequate)."""
            return any(abs(c - comp) <= dft_cover_tol for c in _dft_comps)

        # A compound whose melting form is a DISTINCT high-temperature polymorph carries no
        # usable solid reference: the DFT hull is a 0 K ground-state hull, so it holds the
        # low-temperature form, whose enthalpy is not the melting form's. Composition is no
        # defence here (both forms sit at the same x), so dft_covers cannot be consulted --
        # C-La's hull has LaC2 and the liquidus from the C+LaC2 eutectic to the La2C3
        # peritectic is nonetheless conjugate to LaC2 ht (cF36) the whole way across.
        ht_polymorphs = {
            phase["name"]: phase
            for phase in (self.low_t_exp_phases or [])
            if phase.get("distinct_melting_polymorph")
        }

        def ht_polymorph_governed(inv):
            """True when this invariant's solid is the high-temperature form, so any
            constraint built from the hull's low-temperature enthalpy would be wrong."""
            for name in inv.get("phases") or []:
                phase = ht_polymorphs.get(name)
                if phase is None:
                    continue
                t_transition = phase.get("polymorph_transition_temp")
                if t_transition is None or inv.get("temp", 0) > t_transition - t_tol:
                    return True
            return False

        # Mask the liquidus flanking each such compound, bounded by its neighbouring
        # invariants -- the same conservative local bound the missing-phase cases below use.
        if auto_ignored_ranges:
            for phase in ht_polymorphs.values():
                lo, hi = nearest_inv_left(phase["comp"]), nearest_inv_right(phase["comp"])
                t_transition = phase.get("polymorph_transition_temp")
                # Where the low-temperature form's own ceiling is known, only the liquidus
                # ABOVE it is conjugate to the hotter form; below it the hull's ground state
                # is the right reference and the branch stays fittable. Masking the whole
                # flanking interval on the strength of a narrow window at the top costs real
                # systems: Cr-Zr's ZrCr2 is C15 up to 1864 K and melts at 1945 K, so 81 K of
                # ht stability would otherwise mask 60% of the liquidus and skip the system.
                # A ceiling that reaches the melting point carries no information (C-Ce draws
                # both CeC2 lines full height), so there the whole interval stays masked.
                if t_transition is not None and t_transition < phase["tbounds"][1][1] - t_tol:
                    hot = [x for x, t in self.digitized_liq if lo <= x <= hi and t > t_transition]
                    if not hot:
                        continue
                    lo, hi = min(hot), max(hot)
                self.ignored_comp_ranges.append([lo, hi])

        if not kwargs.get("disable_inv_constrs", False):
            for inv in self.invariants:
                if inv["type"] == "mig":
                    x1, t1 = inv["cbounds"][0]  # Bottom left of dome
                    x2, t2 = inv["tbounds"][1]  # Top of dome
                    x3, t3 = inv["cbounds"][1]  # Bottom right of dome

                    eqn1 = sp.Eq(self.eqs["g_double_prime"].subs({xb_sym: x2, t_sym: t2}), 0)
                    eqn4 = sp.Eq(
                        self.eqs["g_prime"].subs({xb_sym: x1, t_sym: t1}),
                        self.eqs["g_prime"].subs({xb_sym: x3, t_sym: t3}),
                    )

                    eqs.append([f"mig - {round(x2, 2)} 2nd", "2nd order", t2, eqn1])
                    eqs.append([f"mig - {round(x1, 2)}-{round(x3, 2)} 1st", "1st order", t1, eqn4])

                if inv["type"] == "cmp":
                    if ht_polymorph_governed(inv):
                        continue
                    if "(" in inv["phases"][0] and auto_ignored_ranges:
                        if not dft_covers(inv["comp"]):
                            if inv["comp"] < 0.5:
                                lo = nearest_inv_left(inv["comp"])
                                self.ignored_comp_ranges.append([lo, inv["comp"]])
                            elif inv["comp"] > 0.5:
                                hi = nearest_inv_right(inv["comp"])
                                self.ignored_comp_ranges.append([inv["comp"], hi])
                        continue

                    nearest_phase, _ = find_nearest_phase(inv["comp"])
                    if not nearest_phase:
                        continue

                    x1, t1 = self._phase_x(nearest_phase), inv["temp"]
                    eqn = sp.Eq(
                        self.eqs["g_liquid"].subs({xb_sym: x1, t_sym: t1}), nearest_phase.enthalpy
                    )
                    eqs.append([f"cmp - {round(x1, 2)} 0th", "0th order", t1, eqn])

                if inv["type"] == "per":
                    if ht_polymorph_governed(inv):
                        continue
                    if "(" in inv["phases"][0] and auto_ignored_ranges:
                        if not dft_covers(inv["comp"]):
                            if inv["phase_comps"][0] < inv["comp"]:
                                lo = nearest_inv_left(inv["comp"])
                                self.ignored_comp_ranges.append([lo, inv["comp"]])
                            elif inv["phase_comps"][0] > inv["comp"]:
                                hi = nearest_inv_right(inv["comp"])
                                self.ignored_comp_ranges.append([inv["comp"], hi])
                        continue

                    per_phase, _ = find_nearest_phase(inv["phase_comps"][0], tol=0.04)
                    if not per_phase:
                        continue

                    x1, t1 = inv["comp"], inv["temp"]
                    x2 = self._phase_x(per_phase)
                    g2 = (
                        self._stable_solid_gibbs_at_T(x2, t1)
                        if x2 in (0.0, 1.0)
                        else per_phase.enthalpy
                    )

                    eqn1 = sp.Eq(
                        self.eqs["g_liquid"].subs({xb_sym: x1, t_sym: t1})
                        + self.eqs["g_prime"].subs({xb_sym: x1, t_sym: t1}) * (x2 - x1),
                        g2,
                    )
                    eqn2 = sp.Eq(self.eqs["g_liquid"].subs({xb_sym: x1, t_sym: t1}), g2)

                    liq_point_at_phase = min(self.digitized_liq, key=lambda x: abs(x[0] - x2))
                    temp_below_liq = liq_point_at_phase[1] - t1

                    if temp_below_liq > t_tol:
                        eqs.append([f"per - {round(x1, 2)} 0th", "0th order", t1, eqn1])
                    else:
                        eqs.append(
                            [f"per - {round(x1, 2)} 0th", "H-S partition 0th order", t1, eqn2]
                        )

                if inv["type"] == "eut":
                    if None in inv["phase_comps"] or ht_polymorph_governed(inv):
                        continue

                    lhs_phase, _ = find_nearest_phase(inv["phase_comps"][0], tol=0.04)
                    rhs_phase, _ = find_nearest_phase(inv["phase_comps"][1], tol=0.04)

                    invalid_eut = False
                    if not lhs_phase or self._phase_x(lhs_phase) > inv["comp"]:
                        if auto_ignored_ranges and not dft_covers(inv["phase_comps"][0]):
                            lo = nearest_inv_left(inv["comp"])
                            self.ignored_comp_ranges.append([lo, inv["comp"]])
                        invalid_eut = True
                    elif "(" in inv["phases"][0] and inv["phase_comps"][0] > 0.05:
                        invalid_eut = True
                    if not rhs_phase or self._phase_x(rhs_phase) < inv["comp"]:
                        if auto_ignored_ranges and not dft_covers(inv["phase_comps"][1]):
                            hi = nearest_inv_right(inv["comp"])
                            self.ignored_comp_ranges.append([inv["comp"], hi])
                        invalid_eut = True
                    elif "(" in inv["phases"][1] and inv["phase_comps"][1] < 0.95:
                        invalid_eut = True
                    if invalid_eut:
                        continue

                    x1 = self._phase_x(lhs_phase)
                    x2, t2 = inv["comp"], inv["temp"]
                    x3 = self._phase_x(rhs_phase)
                    g1 = (
                        self._stable_solid_gibbs_at_T(x1, t2)
                        if x1 in (0.0, 1.0)
                        else lhs_phase.enthalpy
                    )
                    g3 = (
                        self._stable_solid_gibbs_at_T(x3, t2)
                        if x3 in (0.0, 1.0)
                        else rhs_phase.enthalpy
                    )

                    eqn1 = sp.Eq(
                        self.eqs["g_prime"].subs({xb_sym: x2, t_sym: t2}), (g3 - g1) / (x3 - x1)
                    )
                    eqn2 = sp.Eq(
                        self.eqs["g_liquid"].subs({xb_sym: x2, t_sym: t2})
                        + self.eqs["g_liquid"].subs({xb_sym: x2, t_sym: t2}) * (x1 - x2),
                        g1,
                    )
                    eqn3 = sp.Eq(
                        self.eqs["g_liquid"].subs({xb_sym: x2, t_sym: t2})
                        + self.eqs["g_liquid"].subs({xb_sym: x2, t_sym: t2}) * (x3 - x2),
                        g3,
                    )

                    eqs.append([f"eut - {round(x2, 2)} 1st", "1st order", t2, eqn1])
                    if g1 <= g3:
                        eqs.append([f"eut - {round(x2, 2)} 0th", "0th order lhs", t2, eqn2])
                    else:
                        eqs.append([f"eut - {round(x2, 2)} 0th", "0th order rhs", t2, eqn3])

        # Conservative masking: merge the auto-detected ignored ranges and, if the
        # un-modellable (masked) fraction of the digitized liquidus exceeds the cap, flag the
        # system as unreliable and skip it rather than fitting a data-starved liquidus.
        if auto_ignored_ranges and self.ignored_comp_ranges:
            self.ignored_comp_ranges = merge_comp_ranges(self.ignored_comp_ranges)
            liq_lo, liq_hi = self.digitized_liq[0][0], self.digitized_liq[-1][0]
            span = max(liq_hi - liq_lo, 1e-9)
            masked = sum(
                min(hi, liq_hi) - max(lo, liq_lo)
                for lo, hi in self.ignored_comp_ranges
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

        # Test invariant-derived equations for validity as constraints. The format's
        # topology decides how many invariant equations each candidate consumes: the
        # remainder of the solve system comes from its identity constraints (pinned
        # parameters, e.g. comb-exp's L1_b = 0).
        fmt = self.xs_mix.format
        identity_entries = [
            [f"no1S - {p} = 0", "0th order", float("inf"), eq]
            for p, eq in zip(fmt.pinned_params, fmt.identity_constraints())
        ]
        solve_syms = tuple(self.xs_mix.solve_symbols)
        nelder_mead_ics = []
        # Drop invariant equations sympy already collapsed to False. Identity against
        # sp.false, NOT `if eq[3]`: every entry is a sympy object, and bool() on a symbolic
        # Eq raises TypeError('cannot determine truth value of Relational').
        eqs = [eq for eq in eqs if eq[3] is not sp.false]

        if eqs:  # 1 or more valid invariant-derived constraint equations
            highest_tm_eq = eqs.pop(eqs.index(max(eqs, key=lambda x: x[2])))
            for eq in eqs:  # 2 or more valid invariant-derived constraint equations
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
                    init_f = min(self.f(v, **kwargs) for v in self.init_triangle)
                    if init_f == float("inf"):
                        continue
                    nelder_mead_ics.append(
                        {"f": init_f, "constrs": constr_entries, "init_tri": self.init_triangle}
                    )
                except RuntimeError as e:
                    logger.error(
                        f"Error while evaluating invariant constraints [{self.sys_name}]: {e}"
                    )
                    continue

        # Derive H-S partition constraints using Nelder-Mead for the enthalpy of mixing
        mean_liq_temp = np.mean([point[1] for point in self.digitized_liq])
        if verbose:
            logger.info(
                f"Maximum composition range fitted: {[self.digitized_liq[0][0], self.digitized_liq[-1][0]]}"
            )
            logger.info(f"Ignored composition ranges: {self.ignored_comp_ranges}")

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
                    [self.get_L0_b() * 0.8, self.get_L1_b()],
                    [self.get_L0_b(), self.get_L1_b() * 0.8],
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
                    [self.get_L0_b() * 0.8, self.get_L1_a()],
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

        # Sort by ascending initial MAE such that the 'best' constraints are used first (if limited on number of attempts)
        nelder_mead_ics.sort(key=lambda x: x["f"])
        self.guess_symbols = self.xs_mix.guess_symbols
        solve_symbols = self.xs_mix.solve_symbols
        fitting_data = []
        enable_multi_threading = bool(
            enable_multi_threading or kwargs.get("enable_multi_threading", False)
        )

        # Prepare optimization tasks (up to n_opts, limited by available ICs)
        optimization_tasks = []
        for i in range(min(n_opts, len(nelder_mead_ics))):
            optimization_tasks.append((i, nelder_mead_ics[i]))

        if len(optimization_tasks) == 1:
            # Single optimization: run directly without executor overhead
            task_idx, task_ics = optimization_tasks[0]
            result = _run_single_optimization_worker(
                task_idx, task_ics, self, kwargs, verbose, solve_symbols, mean_liq_temp
            )
            if result is not None:
                fitting_data.append(result)
        elif optimization_tasks:
            if enable_multi_threading:
                max_workers = min(len(optimization_tasks), n_opts)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_single_optimization_worker,
                            task_idx,
                            task_ics,
                            self,
                            kwargs,
                            verbose,
                            solve_symbols,
                            mean_liq_temp,
                        ): task_idx
                        for task_idx, task_ics in optimization_tasks
                    }
                    for future in as_completed(futures):
                        task_idx = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                fitting_data.append(result)
                        except Exception as e:
                            logger.error(
                                f"Optimization attempt #{task_idx + 1} failed with exception [{self.sys_name}]: {e}"
                            )
            else:
                for task_idx, task_ics in optimization_tasks:
                    try:
                        result = _run_single_optimization_worker(
                            task_idx, task_ics, self, kwargs, verbose, solve_symbols, mean_liq_temp
                        )
                        if result is not None:
                            fitting_data.append(result)
                    except Exception as e:
                        logger.error(
                            f"Optimization attempt #{task_idx + 1} failed with exception [{self.sys_name}]: {e}"
                        )

        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x["f"])
            self.xs_mix.update([best_fit[name] for name in self.xs_mix.format.param_names])
            self.nmpath = best_fit["nmpath"]
            self.update_phase_points()
        return fitting_data


def _run_single_optimization_worker(
    task_index, selected_ics, bl_template, run_kwargs_base, verbose, solve_symbols, mean_liq_temp
):
    """
    Module-level worker for threaded optimization execution.

    Args:
        task_index (int): Index of the optimization attempt.
        selected_ics (dict): Initial conditions for this optimization attempt.
        bl_template (BinaryLiquid): Template BinaryLiquid object to copy.
        run_kwargs_base (dict): Base kwargs forwarded to Nelder-Mead and objective function.
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

    run_kwargs = dict(run_kwargs_base)
    run_kwargs["check_lupis_elliott"] = run_kwargs_base.get("check_lupis_elliott", True)
    run_kwargs["use_lxb_penalty"] = run_kwargs_base.get(
        "use_lxb_penalty", bl_template.xs_mix.format.lxb_default
    )
    run_kwargs["lxb_penalty_cfg"] = copy.deepcopy(run_kwargs_base.get("lxb_penalty_cfg"))
    run_kwargs["lupis_elliott_cfg"] = copy.deepcopy(run_kwargs_base.get("lupis_elliott_cfg"))

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
        f, (mae, rmse, mape, rmspe), path = bl_copy.nelder_mead(
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
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "rmspe": rmspe,
        "constrs": constrs_str,
        "algo": constr_algo,
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


# --- Binary plot stack: implemented in gliquid.plotting (binary_tx + style). ---
# Re-exported here so existing `from gliquid.binary import ...` consumers keep
# working; BLPlotter below consumes the same names.
from gliquid.plotting.binary_tx import (  # noqa: F401
    _BOTTOM_PAD_FRAC,
    _CHAR_W_FACTOR,
    _FIG_H_PX,
    _FIG_W_PX,
    _GREEK_MAP,
    _LINE_H_FACTOR,
    _MARGIN_B_PX,
    _MARGIN_L_PX,
    _MARGIN_R_PX,
    _MARGIN_T_PX,
    _PLOT_H_PX,
    _PLOT_W_PX,
    _STRUCT_ABBREV,
    _SUB_TAG_RE,
    _SUB_W_FACTOR,
    _TAG_PHASE_TAG_RE,
    _abbrev_structure,
    _abbreviate_phase_name,
    _apex_from_invariant,
    _assemblage_across,
    _assign_compound_sides,
    _box_overlap,
    _curve_crossings_at_temp,
    _detect_tie_lines,
    _edge_inside_ss_field,
    _estimate_label_box,
    _facet_assemblages,
    _liquidus_top_fn,
    _match_edge_interval,
    _merge_close_values,
    _pack_labels,
    _parse_elemental_phase,
    _place_compound_y,
    _place_legend,
    _place_liquid_label,
    _resolve_label_collisions,
    _split_indices,
    _split_segments,
    _ss_boundary_crossings,
    _ss_family_maxima,
    _ss_minimum_anchor,
    _ss_minimum_tie_allowed,
    _ss_regions,
    _ss_solid_pair_phase,
    _ss_tie_allowed,
    _subscript_formula,
    _text_glyph_width,
    build_polymorph_transitions,
    plot_tx,
    record_tie_lines,
)
from gliquid.plotting.style import (  # noqa: F401
    SS_FIXED_COLORS,
    build_phase_color_map,
    format_phase_display_name,
)

# Composition-grid subdivision for the solid-solution presentation hull. 5 closes the
# field termini that matter (Hf-Y and Ru-Y HCP reach a true apex; Cr-W and Hf-W BCC
# converge) while keeping every three-phase invariant the coarse hull found -- 10 drops
# Hf-Y's L+HCP+Y peritectic entirely.
_SS_PLOT_REFINE = 5


class BLPlotter:
    """
    A plotting class for BinaryLiquid objects.

    This class contains methods to create various subfigures and visualizations for analyzing
    BinaryLiquid system data. It uses both static matplotlib and interactive Plotly plots.
    """

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
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments for customization.

        Returns:
            go.Figure | plt.Axes: The generated plot object (Plotly or Matplotlib).
        """
        valid_plot_types = [
            "pc",
            "ch",
            "ch+g",
            "vch",
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

        # Phase comparison plot
        if plot_type == "pc":
            fig = self._generate_phase_comparison_plot()

        # Convex hull plots (ch+g overlays solution Gibbs curves when SS models exist)
        elif plot_type in ["ch", "ch+g", "vch"]:
            fig = self._generate_convex_hull_plot(plot_type, **kwargs)

        # Liquidus fitting and prediction plots (solution bands render when SS models exist)
        elif plot_type in ["fit", "fit+liq", "pred", "pred+liq"]:
            fig = self._generate_liquidus_fit_plot(plot_type, **kwargs)

        # Nelder-Mead path visualization
        elif plot_type == "nmp":
            if self._bl is not self._bl_raw:
                raise ValueError(
                    "'nmp' renders in the fitting (construction) frame; "
                    "construct BLPlotter with order='given'."
                )
            fig = self._generate_nelder_mead_path_plot(**kwargs)

        # HSX hull diagnostics / raw TX scatter (meaningful with or without SS phases)
        elif plot_type == "hsx":
            fig = self._generate_hsx_plot(**kwargs)

        elif plot_type == "scatter":
            fig = self._generate_tx_scatter_plot(**kwargs)

        return fig

    def _phase_display_name(self, phase_label: str) -> str:
        """Legend name: SS phases carry their component pair, others their raw label."""
        return format_phase_display_name(phase_label, self._bl.ss_models, self._bl.components)

    def _phase_color_map(self) -> dict[str, str]:
        """Deterministic phase -> color map; SS phases use reserved fixed colors.

        Cached on the plotter (not the model) so BinaryLiquid pickles stay clean.
        """
        cached = getattr(self, "_ss_phase_color_cache", None)
        if cached is not None:
            return cached

        self._ss_phase_color_cache = build_phase_color_map(
            self._bl.hsx.phases, ss_names=list(self._bl.ss_models)
        )
        return self._ss_phase_color_cache

    def _phase_color(self, phase_label: str) -> str:
        return self._phase_color_map().get(phase_label, "#555555")

    def _generate_tx_scatter_plot(self, **kwargs) -> go.Figure:
        """Diagnostic TX scatter using raw points from HSX compute_tx (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        if not self._bl.phases[-1].points:
            self._bl.update_phase_points()  # hsx must exist before the color map reads its phases
        return binary_figs.render_tx_scatter(self._bl, self._phase_color_map(), **kwargs)

    def _generate_hsx_plot(self, **kwargs) -> go.Figure:
        """HSX hull debug plot with solid-solution phase blocks (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        if not self._bl.phases[-1].points:
            self._bl.update_phase_points()  # hsx must exist before the color map reads its phases
        return binary_figs.render_hsx_diagnostic(self._bl, self._phase_color_map(), **kwargs)

    def show(self, plot_type: str, **kwargs) -> None:
        """
        Displays the generated plot.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
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
            plot_type (str): The type of plot to save.
            stream (str | StringIO): The file path or stream to save the image.
            image_format (str): The format of the image (default is 'svg').
            export_timeout_s (float): Maximum time allowed for Plotly image export.
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)
        write_kwargs = {}
        if plot_type in ["ch", "ch+g", "vch"]:
            write_kwargs.update({"width": 480 * 1.8, "height": 300 * 1.7})  # 960, 700?
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

        Args:
            plot_type (str): The type of liquidus plot to generate ('fit', 'fit+liq', 'pred', 'pred+liq').

        Returns:
            go.Figure: The generated plot object.
        """

        # Check if the plot type includes the MPDS liquidus
        if plot_type in ["fit+liq", "pred+liq"] and not self._bl.digitized_liq:
            logger.warning(
                "Digitized_liquidus is not initialized! Returning plot without digitized liquidus"
            )

        # Determine if prediction is required
        pred_pd = bool(plot_type in ["pred", "pred+liq"])

        # Ensure phase points are updated if not already done
        if self._bl.hsx is None:
            self._bl.update_phase_points()

        # Build polymorph transition data for tie lines and labels (shared annotation builder)
        polymorph_transitions = build_polymorph_transitions(self._bl)

        # Imputed (phase-energy-imputation) phases are rendered with dashed lines.
        imputed_phases = {p.name for p in self._bl.phases if p.imputed}

        # Solid-solution systems render from a presentation hull whose SS phases are
        # sampled finer, so their fields terminate in a point rather than a vertical face
        # straddling a dozen equilibria. Systems without SS models keep the fitted HSX
        # object itself -- same hull, same in-place conds mutation, same figure bytes.
        hsx = self._bl.refined_hsx(_SS_PLOT_REFINE) if self._bl.ss_models else self._bl.hsx

        # Generate the plot using the migrated module-level plot function
        fig = plot_tx(
            hsx,
            digitized_liquidus=self._bl.digitized_liq
            if plot_type in ["fit+liq", "pred+liq"]
            else None,
            pred=pred_pd,  # Determines generated liquidus color and temperature axis scaling
            polymorph_transitions=polymorph_transitions,
            imputed_phases=imputed_phases,
            ss_phases=set(self._bl.ss_models),  # empty set leaves the non-SS path untouched
        )

        return fig

    def _generate_nelder_mead_path_plot(self, **kwargs) -> plt.Figure:
        """Nelder-Mead optimization-path figure (body in gliquid.plotting.binary_figs)."""
        from gliquid.plotting import binary_figs

        return binary_figs.render_nelder_mead_path(self._bl, **kwargs)
