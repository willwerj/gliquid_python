"""
Phase-ablation overfitting benchmark library for the gliquid binary liquid model.

When the DFT convex hull is missing a solid phase that exists in the experimental
(MPDS) phase diagram, fitting the four Redlich-Kister liquid parameters
[L0_a, L0_b, L1_a, L1_b] can overfit: the liquidus still looks roughly right but the
parameters absorb the error, polluting the downstream ML dataset.

This module provides the primitives for studying that effect:
  1. ablate one-to-several DFT phases from a good-fit baseline and re-fit
     (``ablate_phases``, ``fit_ablated``),
  2. predict a priori — from the baseline fit alone — whether ablating a phase will
     cause large parameter drift (``predict_overfit_risk``),
  3. benchmark mitigation strategies head-to-head through a pluggable registry
     (``register_strategy``, ``available_strategies``, ``strategy_kwargs``).

The reusable analysis library deliberately contains no plotting or file I/O; the
driver script ``Benchmark_Phase_Ablation.py`` handles iteration and output.

Design notes grounded in src/gliquid/binary.py:
  * ``bl.phases`` lists solid phase dicts plus a trailing liquid solution phase.
    Pure-element polymorph endpoints (comp 0.0/1.0) anchor the hull and are never
    ablatable; only interior intermetallics are.
  * ``bl.eqs['h_hull_interp']`` is a linear interpolation of the solid hull used by
    the hard T=0K constraint ``h0_below_ch``. Removing a phase REQUIRES rebuilding
    this interp or the constraint becomes inconsistent with the ablated hull.
  * ``fit_parameters`` aborts (init_error) if more than half the critical phases lack
    a DFT match, so ablated fits must pass ``check_phase_mismatch=False``.
  * Pre-populating ``bl.ignored_comp_ranges`` disables the auto-detection heuristic;
    leaving it ``[]`` enables it. ``ignored_ranges`` (a fit kwarg) controls whether
    those ranges are masked out of the objective.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from itertools import combinations
from typing import Callable

import numpy as np

from gliquid.binary import (
    BinaryLiquid,
    _x_vals,
    a_sym,
    b_sym,
    c_sym,
    d_sym,
    t_sym,
)
from gliquid.fisher_information import (
    FIMResult,
    compute_fim,
    find_optimal_next_measurement,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREE_PARAM_INDICES = [0, 2]          # L0_a, L1_a (mirrors the production FIM convention)
PARAM_NAMES = ['L0_a', 'L0_b', 'L1_a', 'L1_b']
DEFAULT_FIM_SIGMA = 5.0
ENDPOINT_COMPS = (0.0, 1.0)

# Real FIM/Tikhonov prior (the `fim_prior` strategy). The baseline FIM precision is
# normalized per measurement point, so `maha` is the mean squared standardized liquidus
# shift and these weights are portable across systems. Tunable post hoc.
FIM_PRIOR_WEIGHT = 0.5
FIM_PRIOR_EXPONENT = 1.0     # 1.0 => principled Gaussian/Tikhonov shape

# Production prior configuration (mirrors dev/scripts/Fit_Binary_Systems.py).
L0B_PRIOR_MEDIAN = -12.27
L0B_PRIOR_MAD = 6.41
PRODUCTION_LE_CFG = {'strength': 7.5e-3}
PRODUCTION_LXB_CFG = {
    'l0': {'weight': 0.10, 'median': L0B_PRIOR_MEDIAN, 'mad': L0B_PRIOR_MAD, 'exponent': 2.5},
    'apply_l1': False,
}
PRODUCTION_FIT_KWARGS = {
    'n_opts': 1,
    'max_iter': 64,
    'check_lupis_elliott': True,
    'lupis_elliott_cfg': PRODUCTION_LE_CFG,
    'use_lxb_penalty': True,
    'lxb_penalty_cfg': PRODUCTION_LXB_CFG,
}

# Default weights for the a-priori risk score (transparent, tunable post hoc).
DEFAULT_RISK_WEIGHTS = {
    'info_at_target': 1.0,
    'baseline_cond': 1.0,
    'baseline_sigma_free': 1.0,
    'decomp_near_liq': 1.0,
    'nn_distance': 1.0,
    'ignored_span_fraction': 1.0,
    'n_phases_removed': 0.5,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BaselineFit:
    """Reference state: a system fitted against its full (un-ablated) DFT hull."""
    system: str
    bl: BinaryLiquid                  # fitted full-hull snapshot
    params: list                      # [L0_a, L0_b, L1_a, L1_b]
    fim: FIMResult | None
    L0_T: float                       # combined L0 evaluated at mean liquidus T
    L1_T: float                       # combined L1 evaluated at mean liquidus T
    mean_liq_temp: float
    fit_dict: dict                    # best fit dict from fit_parameters (empty if unfittable)
    ablatable: list                   # list of ablatable phase dicts (interior intermetallics)

    @property
    def ok(self) -> bool:
        return bool(self.fit_dict) and not self.bl.init_error


@dataclasses.dataclass
class AblationResult:
    """Outcome of re-fitting after ablating one or more DFT phases under a strategy."""
    system: str
    ablated_phases: list              # phase names removed
    ablated_comps: list               # B-mole-fraction comps removed
    strategy: str
    fit_ok: bool
    params: list                      # fitted [L0_a, L0_b, L1_a, L1_b]
    drift_raw: dict                   # signed delta per raw param (abl - base)
    drift_free: float                 # normalized drift in the [L0_a, L1_a] subspace
    drift_L0_T: float                 # |L0(T)_abl - L0(T)_base| at mean liquidus T (headline)
    drift_L1_T: float                 # |L1(T)_abl - L1(T)_base| at mean liquidus T (headline)
    mae_full: float                   # liquidus MAE incl. ablated region (ignored_ranges=False)
    mae_ignored: float                # liquidus MAE heuristic-masked (ignored_ranges=True)
    mae_ablated_region: float         # MAE restricted to the ablated comp window
    fim_det: float
    fim_cond: float
    sigma_L0_a: float
    sigma_L1_a: float
    heuristic_fired: bool
    ignored_comp_ranges: list
    sum_ignored_range: float
    fit_meta: dict                    # constrs, algo, n_iters, f


@dataclasses.dataclass
class RiskScore:
    """A-priori overfit-risk estimate for ablating a target phase set."""
    system: str
    target_phases: list
    score: float                      # higher => more drift expected
    features: dict                    # raw feature values (re-weightable post hoc)


# ---------------------------------------------------------------------------
# Ablation primitive
# ---------------------------------------------------------------------------

def list_ablatable_phases(bl: BinaryLiquid) -> list[dict]:
    """
    Return the interior solid phase dicts that may be ablated.

    Pure-element polymorph endpoints (comp 0.0 / 1.0) anchor the hull and the liquid
    endpoints and are never ablatable; the liquid solution phase is excluded too.

    Args:
        bl (BinaryLiquid): System whose phases to inspect.

    Returns:
        list[dict]: References into ``bl.phases`` (read-only use intended).
    """
    return [
        p for p in bl.phases
        if not p.get('is_solution', False)
        and 'comp' in p
        and round(float(p['comp']), 6) not in ENDPOINT_COMPS
    ]


def _resolve_targets(bl: BinaryLiquid, targets, tol: float = 0.02) -> list[dict]:
    """
    Resolve target specifiers to concrete ablatable phase dicts in ``bl.phases``.

    Args:
        bl (BinaryLiquid): System to resolve against.
        targets: A single target or an iterable of targets, each either a phase name
            (str) or a B-mole-fraction composition (float, matched to the nearest
            ablatable phase within ``tol``).
        tol (float): Composition match tolerance for float targets.

    Returns:
        list[dict]: The matched ablatable phase dicts (object identity preserved).

    Raises:
        ValueError: If a target is an endpoint/solution phase or matches nothing.
    """
    if isinstance(targets, (str, float, int)):
        targets = [targets]

    ablatable = list_ablatable_phases(bl)
    resolved: list[dict] = []
    for target in targets:
        if isinstance(target, str):
            matches = [p for p in ablatable if p['name'] == target]
            if not matches:
                raise ValueError(
                    f"Target phase '{target}' is not an ablatable interior phase of "
                    f"{bl.sys_name}. Ablatable: {[p['name'] for p in ablatable]}"
                )
            resolved.append(matches[0])
        elif isinstance(target, (float, int)) and not isinstance(target, bool):
            comp = round(float(target), 6)
            if comp in ENDPOINT_COMPS:
                raise ValueError(f"Endpoint composition {comp} is not ablatable.")
            if not ablatable:
                raise ValueError(f"{bl.sys_name} has no ablatable interior phases.")
            nearest = min(ablatable, key=lambda p: abs(p['comp'] - comp))
            if abs(nearest['comp'] - comp) > tol:
                raise ValueError(
                    f"No ablatable phase within tol={tol} of composition {comp} in "
                    f"{bl.sys_name} (nearest at {nearest['comp']})."
                )
            resolved.append(nearest)
        else:
            raise ValueError(f"Target must be a phase name (str) or composition (float); got {target!r}")
    return resolved


def _rebuild_hull_interp(bl: BinaryLiquid) -> None:
    """
    Recompute only ``bl.eqs['h_hull_interp']`` from the current ``bl.phases``.

    Mirrors the interpolation built in ``BinaryLiquid.from_cache`` so the hard T=0K
    constraint ``h0_below_ch`` stays consistent with the (ablated) solid hull. Does not
    re-lambdify any expressions.

    Args:
        bl (BinaryLiquid): System whose hull interpolation to refresh in place.
    """
    hull_points = np.array(
        sorted(([p['comp'], p['enthalpy']] for p in bl.phases if 'comp' in p), key=lambda pt: pt[0])
    )
    bl.eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])


def ablate_phases(bl: BinaryLiquid, targets, in_place: bool = False, tol: float = 0.02) -> BinaryLiquid:
    """
    Remove one or more interior DFT phases and reset all DFT-dependent state for a
    clean re-fit.

    Args:
        bl (BinaryLiquid): Source system (a fitted full-hull baseline is typical).
        targets: Phase name(s) (str) or composition(s) (float) to remove. See
            ``_resolve_targets``.
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

    _rebuild_hull_interp(work)

    # Reset every cache that fit_parameters populates so the next fit recomputes cleanly.
    work.invariants = None
    work.low_t_exp_phases = None
    work.ignored_comp_ranges = []
    work.guess_symbols = None
    work.constraints = None
    work._ref_params = None
    work.nmpath = None
    work.hsx = None
    work.init_error = False
    work._params = [0, 0, 0, 0]
    return work


# ---------------------------------------------------------------------------
# Parameter / liquidus metrics
# ---------------------------------------------------------------------------

def _combined_L_at_T(bl: BinaryLiquid, params, temp: float) -> tuple[float, float]:
    """
    Evaluate the T-dependent combined L0(T), L1(T) from the model expressions.

    Mirrors the L0/L1 reporting in ``_run_single_optimization_worker``.

    Args:
        bl (BinaryLiquid): Provides the symbolic ``eqs['l0']`` / ``eqs['l1']``.
        params: Raw parameters [L0_a, L0_b, L1_a, L1_b].
        temp (float): Temperature (K) at which to evaluate.

    Returns:
        tuple[float, float]: (L0(temp), L1(temp)).
    """
    l0 = float(bl.eqs['l0'].subs({t_sym: temp, a_sym: params[0], b_sym: params[1]}))
    l1 = float(bl.eqs['l1'].subs({t_sym: temp, c_sym: params[2], d_sym: params[3]}))
    return l0, l1


def parameter_drift(base_params, abl_params, scale=None) -> tuple[float, dict]:
    """
    Quantify parameter change between a baseline and an ablated fit.

    Args:
        base_params: Baseline raw parameters [L0_a, L0_b, L1_a, L1_b].
        abl_params: Ablated-fit raw parameters [L0_a, L0_b, L1_a, L1_b].
        scale: Optional per-index normalization (indexable by 0..3). Defaults to
            ``max(|base_i|, 1.0)`` so drift is comparable across systems with very
            different parameter magnitudes.

    Returns:
        tuple[float, dict]:
            - drift_free: sqrt of the summed squared normalized deltas over the free
              subspace ``FREE_PARAM_INDICES`` ([L0_a, L1_a]).
            - drift_raw: signed raw delta (abl - base) for all four parameters.
    """
    drift_raw = {PARAM_NAMES[i]: float(abl_params[i] - base_params[i]) for i in range(4)}
    total = 0.0
    for i in FREE_PARAM_INDICES:
        s = scale[i] if scale is not None else max(abs(base_params[i]), 1.0)
        total += ((abl_params[i] - base_params[i]) / s) ** 2
    return float(math.sqrt(total)), drift_raw


def _liquid_phase(bl: BinaryLiquid) -> dict | None:
    """Return the liquid solution phase dict, or None if not present."""
    return next((p for p in bl.phases if p.get('is_solution', False)), None)


def _region_mae(bl: BinaryLiquid, comp_window, num_points: int = 20) -> float:
    """
    MAE between the digitized and generated liquidus restricted to a composition window.

    Uses direct interpolation of both curves on a grid inside the window, sidestepping
    the <10-point inf floor in ``calculate_deviation_metrics`` and avoiding state
    mutation.

    Args:
        bl (BinaryLiquid): System whose generated liquidus (``phases[-1]['points']``)
            and ``digitized_liq`` are compared.
        comp_window (tuple[float, float]): (lo, hi) composition bounds.
        num_points (int): Grid resolution inside the window.

    Returns:
        float: Mean absolute liquidus temperature deviation (K) in the window, or inf
        if either curve has no points in range.
    """
    liq = _liquid_phase(bl)
    if liq is None or not liq['points'] or not bl.digitized_liq:
        return float('inf')

    dig = np.asarray(bl.digitized_liq, dtype=float)
    gen = np.asarray(liq['points'], dtype=float)
    lo, hi = comp_window
    x_lo = max(lo, dig[:, 0].min(), gen[:, 0].min())
    x_hi = min(hi, dig[:, 0].max(), gen[:, 0].max())
    if x_hi - x_lo <= 0:
        return float('inf')

    grid = np.linspace(x_lo, x_hi, num_points)
    dig = dig[np.argsort(dig[:, 0])]
    gen = gen[np.argsort(gen[:, 0])]
    y_dig = np.interp(grid, dig[:, 0], dig[:, 1])
    y_gen = np.interp(grid, gen[:, 0], gen[:, 1])
    return float(np.mean(np.abs(y_dig - y_gen)))


def _fim_metrics(bl: BinaryLiquid, fim_sigma: float = DEFAULT_FIM_SIGMA) -> dict:
    """
    Compute FIM identifiability metrics for a fitted system; nan on failure.

    Returns a dict with keys: fim_det, fim_cond, sigma_L0_a, sigma_L1_a.
    """
    out = {'fim_det': np.nan, 'fim_cond': np.nan, 'sigma_L0_a': np.nan, 'sigma_L1_a': np.nan}
    try:
        fim = compute_fim(bl, sigma=fim_sigma, free_param_indices=FREE_PARAM_INDICES)
    except Exception:
        return out
    sigma_lookup = {
        name: float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
        for name, var in zip(fim.param_names, fim.param_variances)
    }
    out['fim_det'] = float(fim.det_fim) if np.isfinite(fim.det_fim) else np.nan
    out['fim_cond'] = float(fim.condition_number) if np.isfinite(fim.condition_number) else np.nan
    out['sigma_L0_a'] = sigma_lookup.get('L0_a', np.nan)
    out['sigma_L1_a'] = sigma_lookup.get('L1_a', np.nan)
    return out


# ---------------------------------------------------------------------------
# Pluggable mitigation-strategy registry
# ---------------------------------------------------------------------------

StrategyFn = Callable[["BaselineFit"], dict]
STRATEGY_REGISTRY: dict[str, StrategyFn] = {}


def register_strategy(name: str) -> Callable[[StrategyFn], StrategyFn]:
    """
    Decorator registering a mitigation strategy under ``name``.

    A strategy is a callable ``f(baseline) -> dict`` returning the ``fit_parameters``
    kwargs that define the approach (excluding ``check_phase_mismatch``, which
    ``fit_ablated`` always forces to False). Registering a new approach is all that is
    needed for it to be benchmarked by the driver.

    Args:
        name (str): Unique strategy label.

    Returns:
        Callable: The decorator.
    """
    def _decorator(fn: StrategyFn) -> StrategyFn:
        STRATEGY_REGISTRY[name] = fn
        return fn
    return _decorator


def available_strategies() -> list[str]:
    """Return the registered strategy names."""
    return list(STRATEGY_REGISTRY)


def strategy_kwargs(name: str, baseline: BaselineFit) -> dict:
    """
    Resolve a strategy name to its ``fit_parameters`` kwargs.

    Args:
        name (str): Registered strategy label.
        baseline (BaselineFit): Baseline reference (used by data-dependent strategies).

    Returns:
        dict: Fit kwargs for the strategy.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {available_strategies()}")
    return dict(STRATEGY_REGISTRY[name](baseline))


@register_strategy('naive')
def _strategy_naive(baseline: BaselineFit) -> dict:
    """No mitigation: optimize against the full liquidus, no penalties (worst case)."""
    return {
        'ignored_ranges': False,
        'check_lupis_elliott': False,
        'use_lxb_penalty': False,
    }


@register_strategy('heuristic')
def _strategy_heuristic(baseline: BaselineFit) -> dict:
    """Current approach: auto-detect and mask comp ranges adjacent to missing phases."""
    return {
        'ignored_ranges': True,
        'check_lupis_elliott': False,
        'use_lxb_penalty': False,
    }


@register_strategy('penalty')
def _strategy_penalty(baseline: BaselineFit) -> dict:
    """Shipped priors: Lupis-Elliott sign penalty + distribution prior on L0_b."""
    return {
        'ignored_ranges': True,
        'check_lupis_elliott': True,
        'lupis_elliott_cfg': dict(PRODUCTION_LE_CFG),
        'use_lxb_penalty': True,
        'lxb_penalty_cfg': copy.deepcopy(PRODUCTION_LXB_CFG),
    }


def _baseline_fim_precision(baseline: BaselineFit) -> np.ndarray | None:
    """
    Return the baseline FIM precision over the free subspace, normalized per measurement
    point (so the resulting Mahalanobis distance is the mean squared standardized liquidus
    shift and is comparable across systems). None if no usable FIM is available.
    """
    fim = baseline.fim
    if fim is None:
        return None
    P = np.asarray(fim.fim, dtype=float)
    if P.size == 0 or not np.all(np.isfinite(P)):
        return None
    n_meas = int(np.asarray(fim.x_used).size)
    if n_meas <= 0:
        n_meas = max(int(np.asarray(fim.jacobian).shape[0]), 1)
    return P / float(max(n_meas, 1))


@register_strategy('fim_prior')
def _strategy_fim_prior(baseline: BaselineFit) -> dict:
    """
    Real FIM/Tikhonov prior on top of the heuristic: anchor the free parameters
    [L0_a, L1_a] at their well-identified baseline values, using the baseline Fisher
    Information Matrix (J^T J / sigma^2, normalized per measurement point) as the
    precision metric. Well-identified directions are held near the baseline; the
    poorly-identified directions that absorb the missing-phase error stay free.

    This is the genuine Tikhonov prior wired into BinaryLiquid.f() via
    ``use_fim_prior`` / ``fim_prior_cfg`` (the ``fim_tikhonov_penalty`` method),
    replacing the earlier lxb-based approximation. If the baseline FIM is unavailable
    the strategy degrades gracefully to the plain heuristic.
    """
    kwargs = {
        'ignored_ranges': True,
        'check_lupis_elliott': False,
        'use_lxb_penalty': False,
        'use_fim_prior': False,
    }
    precision = _baseline_fim_precision(baseline)
    if precision is not None:
        kwargs['use_fim_prior'] = True
        kwargs['fim_prior_cfg'] = {
            'precision': precision,
            'ref_params': list(baseline.params),
            'free_indices': list(FREE_PARAM_INDICES),
            'weight': FIM_PRIOR_WEIGHT,
            'exponent': FIM_PRIOR_EXPONENT,
        }
    return kwargs


@register_strategy('production')
def _strategy_production(baseline: BaselineFit) -> dict:
    """Reference point: the exact shipped production config (heuristic + both penalties)."""
    kwargs = dict(PRODUCTION_FIT_KWARGS)
    kwargs['lupis_elliott_cfg'] = dict(PRODUCTION_LE_CFG)
    kwargs['lxb_penalty_cfg'] = copy.deepcopy(PRODUCTION_LXB_CFG)
    kwargs['ignored_ranges'] = True
    # n_opts/max_iter are supplied by the caller; drop them so they are not double-set.
    kwargs.pop('n_opts', None)
    kwargs.pop('max_iter', None)
    return kwargs


# ---------------------------------------------------------------------------
# Baseline fit + ablated fit
# ---------------------------------------------------------------------------

def baseline_fit(
    system,
    param_format: str = 'comb-exp',
    fit_kwargs: dict | None = None,
    fim_sigma: float = DEFAULT_FIM_SIGMA,
    **from_cache_kwargs,
) -> BaselineFit:
    """
    Load a system and fit it against its full (un-ablated) DFT hull.

    Args:
        system: System specifier accepted by ``BinaryLiquid.from_cache`` (e.g. 'Cu-Mg').
        param_format (str): Mixing-parameter format (default 'comb-exp').
        fit_kwargs (dict | None): Overrides for ``fit_parameters`` (defaults to
            ``PRODUCTION_FIT_KWARGS``).
        fim_sigma (float): Measurement uncertainty (K) for the baseline FIM.
        **from_cache_kwargs: Extra kwargs forwarded to ``from_cache`` (e.g. ``pd_ind``).

    Returns:
        BaselineFit: Reference state. ``.ok`` is False if the system could not be fit
        (empty fit or init error); callers should skip such systems.
    """
    bl = BinaryLiquid.from_cache(system, param_format=param_format, **from_cache_kwargs)
    sys_name = bl.sys_name

    if bl.init_error or not bl.digitized_liq:
        return BaselineFit(sys_name, bl, list(bl.get_params()), None,
                           float('nan'), float('nan'), float('nan'), {}, [])

    kwargs = copy.deepcopy(PRODUCTION_FIT_KWARGS) if fit_kwargs is None else dict(fit_kwargs)
    fits = bl.fit_parameters(**kwargs)
    if not fits:
        return BaselineFit(sys_name, bl, list(bl.get_params()), None,
                           float('nan'), float('nan'), float('nan'), {}, list_ablatable_phases(bl))

    best = min(fits, key=lambda f: f['f'])
    params = list(bl.get_params())  # fit_parameters mutated bl to the best fit
    mean_liq_temp = float(np.mean([pt[1] for pt in bl.digitized_liq]))
    l0_t, l1_t = _combined_L_at_T(bl, params, mean_liq_temp)

    fim = None
    try:
        fim = compute_fim(bl, sigma=fim_sigma, free_param_indices=FREE_PARAM_INDICES)
    except Exception:
        fim = None

    return BaselineFit(
        system=sys_name,
        bl=bl,
        params=params,
        fim=fim,
        L0_T=l0_t,
        L1_T=l1_t,
        mean_liq_temp=mean_liq_temp,
        fit_dict=best,
        ablatable=list_ablatable_phases(bl),
    )


def fit_ablated(
    baseline: BaselineFit,
    targets,
    strategy: str,
    param_format: str = 'comb-exp',
    fit_kwargs_extra: dict | None = None,
    fim_sigma: float = DEFAULT_FIM_SIGMA,
    n_opts: int = 1,
    max_iter: int = 64,
) -> AblationResult:
    """
    Ablate ``targets`` from a fresh copy of the baseline system, re-fit under
    ``strategy``, and measure parameter drift and liquidus error versus the baseline.

    ``check_phase_mismatch`` is always forced to False because the ablated hull is meant
    to mismatch the experimental phases.

    Args:
        baseline (BaselineFit): Full-hull reference (must be ``.ok``).
        targets: Phase name(s)/composition(s) to ablate.
        strategy (str): Registered mitigation strategy.
        param_format (str): Mixing-parameter format (kept for symmetry; the copied
            baseline already carries its format).
        fit_kwargs_extra (dict | None): Extra ``fit_parameters`` overrides applied last.
        fim_sigma (float): Measurement uncertainty (K) for the ablated FIM.
        n_opts (int): Optimization attempts for the ablated fit.
        max_iter (int): Nelder-Mead iteration cap.

    Returns:
        AblationResult: Drift and error metrics. ``fit_ok`` is False (drift 0, MAEs inf)
        if the ablated fit produced no result.
    """
    resolved = _resolve_targets(baseline.bl, targets)
    names = [p['name'] for p in resolved]
    comps = [float(p['comp']) for p in resolved]
    comp_window = (min(comps) - 0.05, max(comps) + 0.05)

    abl_bl = ablate_phases(baseline.bl, names, in_place=False)

    eff = {'n_opts': n_opts, 'max_iter': max_iter}
    eff.update(strategy_kwargs(strategy, baseline))
    eff['check_phase_mismatch'] = False
    if fit_kwargs_extra:
        eff.update(fit_kwargs_extra)

    fits = abl_bl.fit_parameters(**eff)

    ignored = list(abl_bl.ignored_comp_ranges or [])
    sum_ignored = float(sum(abs(u - l) for l, u in ignored))

    if not fits:
        return AblationResult(
            system=baseline.system, ablated_phases=names, ablated_comps=comps,
            strategy=strategy, fit_ok=False, params=list(baseline.params),
            drift_raw={name: 0.0 for name in PARAM_NAMES}, drift_free=0.0,
            drift_L0_T=0.0, drift_L1_T=0.0,
            mae_full=float('inf'), mae_ignored=float('inf'), mae_ablated_region=float('inf'),
            fim_det=np.nan, fim_cond=np.nan, sigma_L0_a=np.nan, sigma_L1_a=np.nan,
            heuristic_fired=bool(ignored), ignored_comp_ranges=ignored, sum_ignored_range=sum_ignored,
            fit_meta={},
        )

    best = min(fits, key=lambda f: f['f'])
    params = list(abl_bl.get_params())  # mutated to best fit

    drift_free, drift_raw = parameter_drift(baseline.params, params)
    l0_t, l1_t = _combined_L_at_T(abl_bl, params, baseline.mean_liq_temp)
    drift_L0_T = abs(l0_t - baseline.L0_T)
    drift_L1_T = abs(l1_t - baseline.L1_T)

    mae_full = abl_bl.calculate_deviation_metrics(ignored_ranges=False)[0]
    mae_ignored = abl_bl.calculate_deviation_metrics(ignored_ranges=True)[0]
    mae_region = _region_mae(abl_bl, comp_window)

    fim = _fim_metrics(abl_bl, fim_sigma=fim_sigma)

    return AblationResult(
        system=baseline.system, ablated_phases=names, ablated_comps=comps,
        strategy=strategy, fit_ok=True, params=params,
        drift_raw=drift_raw, drift_free=drift_free,
        drift_L0_T=float(drift_L0_T), drift_L1_T=float(drift_L1_T),
        mae_full=float(mae_full), mae_ignored=float(mae_ignored), mae_ablated_region=float(mae_region),
        fim_det=fim['fim_det'], fim_cond=fim['fim_cond'],
        sigma_L0_a=fim['sigma_L0_a'], sigma_L1_a=fim['sigma_L1_a'],
        heuristic_fired=bool(ignored), ignored_comp_ranges=ignored, sum_ignored_range=sum_ignored,
        fit_meta={'constrs': best.get('constrs'), 'algo': best.get('algo'),
                  'n_iters': best.get('n_iters'), 'f': best.get('f')},
    )


# ---------------------------------------------------------------------------
# Target enumeration + a-priori predictor
# ---------------------------------------------------------------------------

def enumerate_targets(baseline: BaselineFit, max_k: int = 2) -> list[list[str]]:
    """
    Enumerate ablation targets: all single phases, then all size-2..``max_k`` combos.

    Never enumerates a combination that would remove every interior phase.

    Args:
        baseline (BaselineFit): Provides the ablatable phase set.
        max_k (int): Maximum number of phases removed together.

    Returns:
        list[list[str]]: Each entry is a list of phase names to ablate together.
    """
    names = [p['name'] for p in baseline.ablatable]
    targets: list[list[str]] = [[n] for n in names]
    for k in range(2, max_k + 1):
        if k >= len(names):
            break
        targets.extend([list(combo) for combo in combinations(names, k)])
    return targets


def _phase_decomp_near_liq(digitized_liq, comp: float, temp: float, tol: float = 10.0) -> bool:
    """
    Whether a phase decomposing at (comp, temp) sits within ``tol`` K of the digitized
    liquidus. Replicates the nested ``phase_decomp_near_liq`` helper in fit_parameters.
    """
    for i in range(len(digitized_liq) - 1):
        x_i, x_j = digitized_liq[i][0], digitized_liq[i + 1][0]
        if x_i == comp:
            return abs(digitized_liq[i][1] - temp) < tol
        if x_i < comp < x_j:
            mid_t = (digitized_liq[i][1] + digitized_liq[i + 1][1]) / 2
            return abs(mid_t - temp) < tol
    return False


def _liquidus_span(bl: BinaryLiquid) -> float:
    """Composition span covered by the digitized liquidus (excluding endpoints)."""
    if not bl.digitized_liq:
        return 1.0
    xs = [pt[0] for pt in bl.digitized_liq if 0.0 < pt[0] < 1.0]
    if len(xs) < 2:
        return 1.0
    return max(xs) - min(xs)


def predict_overfit_risk(
    baseline: BaselineFit,
    targets,
    fim_sigma: float = DEFAULT_FIM_SIGMA,
    weights: dict | None = None,
) -> RiskScore:
    """
    Estimate, a priori, the overfitting risk of ablating ``targets`` — using only the
    baseline (full-hull) fit, without running the ablated fit.

    Features (each scaled so larger generally implies more expected drift):
      - info_at_target: max D-optimal information score in the target comp window
        (high identifiability there means removing it swings the parameters).
      - baseline_cond: baseline FIM condition number (already ill-conditioned => fragile).
      - baseline_sigma_free: max free-parameter std-dev (more prior slack => more drift).
      - decomp_near_liq: 1.0 if any removed phase decomposes near the liquidus.
      - nn_distance: comp gap from the removed set to the nearest remaining phase.
      - ignored_span_fraction: fraction of the liquidus comp span the removed set
        influences (proxy for how much data the heuristic would discard).
      - n_phases_removed: number of phases ablated together.

    Args:
        baseline (BaselineFit): Full-hull reference (must be ``.ok``).
        targets: Phase name(s)/composition(s) under consideration.
        fim_sigma (float): Measurement uncertainty (K) for the info computation.
        weights (dict | None): Per-feature weights (defaults to ``DEFAULT_RISK_WEIGHTS``).

    Returns:
        RiskScore: Scalar ``score`` plus the raw ``features`` (re-weightable post hoc).
    """
    resolved = _resolve_targets(baseline.bl, targets)
    names = [p['name'] for p in resolved]
    comps = [float(p['comp']) for p in resolved]
    c_lo, c_hi = min(comps), max(comps)

    bl = baseline.bl

    # info_at_target: D-optimal score in [c_lo, c_hi] from the baseline FIM.
    info_at_target = 1.0
    if baseline.fim is not None:
        xs = [pt[0] for pt in bl.digitized_liq if 0.0 < pt[0] < 1.0]
        if xs:
            lo = max(c_lo, min(xs))
            hi = min(c_hi, max(xs))
            if hi <= lo:
                lo, hi = max(min(xs), c_lo - 0.05), min(max(xs), c_hi + 0.05)
            grid = np.linspace(lo, hi, 11) if hi > lo else np.array([np.clip(c_lo, min(xs), max(xs))])
            try:
                opt = find_optimal_next_measurement(baseline.fim, bl, candidate_x=grid)
                if opt.d_optimal_scores.size:
                    info_at_target = float(np.nanmax(opt.d_optimal_scores))
            except Exception:
                info_at_target = 1.0

    baseline_cond = float(baseline.fim.condition_number) if baseline.fim is not None else np.nan
    if baseline.fim is not None:
        var = baseline.fim.param_variances
        finite = var[np.isfinite(var) & (var >= 0)]
        baseline_sigma_free = float(np.sqrt(np.max(finite))) if finite.size else np.nan
    else:
        baseline_sigma_free = np.nan

    # decomp_near_liq: does any removed phase touch the experimental liquidus?
    decomp = 0.0
    low_t = baseline.bl.low_t_exp_phases or []
    for comp in comps:
        nearest = min(low_t, key=lambda ph: abs(ph['comp'] - comp), default=None)
        if nearest is not None and abs(nearest['comp'] - comp) <= 0.05:
            t_decomp = nearest['tbounds'][1][1]
            if _phase_decomp_near_liq(bl.digitized_liq, nearest['comp'], t_decomp,
                                      tol=(bl.temp_range[1] - bl.temp_range[0]) * 0.05):
                decomp = 1.0
                break

    # nn_distance: gap from the removed set to the nearest remaining phase (incl. endpoints).
    remaining = [0.0, 1.0] + [p['comp'] for p in baseline.ablatable if p['name'] not in names]
    nn_distance = 0.0
    for comp in comps:
        nn = min(abs(comp - r) for r in remaining)
        nn_distance = max(nn_distance, nn)

    # ignored_span_fraction: comp influence of the removed set as a fraction of the span.
    span = _liquidus_span(bl)
    left_gap = min((abs(c_lo - r) for r in remaining if r <= c_lo), default=c_lo)
    right_gap = min((abs(r - c_hi) for r in remaining if r >= c_hi), default=1.0 - c_hi)
    influence = (c_hi - c_lo) + left_gap + right_gap
    ignored_span_fraction = float(min(1.0, influence / span)) if span > 0 else 0.0

    features = {
        'info_at_target': info_at_target,
        'baseline_cond': baseline_cond,
        'baseline_sigma_free': baseline_sigma_free,
        'decomp_near_liq': decomp,
        'nn_distance': float(nn_distance),
        'ignored_span_fraction': ignored_span_fraction,
        'n_phases_removed': float(len(names)),
    }
    score = score_from_features(features, weights=weights)
    return RiskScore(system=baseline.system, target_phases=names, score=score, features=features)


def _transform_features(features: dict) -> dict:
    """Apply transparent monotone transforms so features are on comparable scales."""
    def safe(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default
    return {
        'info_at_target': math.log1p(max(0.0, safe(features.get('info_at_target'), 1.0) - 1.0)),
        'baseline_cond': math.log10(max(1.0, safe(features.get('baseline_cond'), 1.0))),
        'baseline_sigma_free': math.log1p(max(0.0, safe(features.get('baseline_sigma_free')))),
        'decomp_near_liq': safe(features.get('decomp_near_liq')),
        'nn_distance': safe(features.get('nn_distance')),
        'ignored_span_fraction': safe(features.get('ignored_span_fraction')),
        'n_phases_removed': max(0.0, safe(features.get('n_phases_removed'), 1.0) - 1.0),
    }


def score_from_features(features: dict, weights: dict | None = None) -> float:
    """
    Combine raw predictor features into a single standalone risk score.

    The transforms (log/clip) keep features on comparable scales without needing
    cross-set statistics, so the score is meaningful for a single target set. The raw
    features are preserved on the ``RiskScore`` for post-hoc re-weighting or z-scoring.

    Args:
        features (dict): Raw feature values from ``predict_overfit_risk``.
        weights (dict | None): Per-feature weights (defaults to ``DEFAULT_RISK_WEIGHTS``).

    Returns:
        float: Weighted-sum risk score.
    """
    w = DEFAULT_RISK_WEIGHTS if weights is None else weights
    transformed = _transform_features(features)
    return float(sum(w.get(name, 0.0) * value for name, value in transformed.items()))


def correlate_predictor(results, scores, drift_key: str = 'drift_L0_T') -> dict:
    """
    Validate the predictor by correlating a-priori scores with observed drift.

    Args:
        results: Iterable of ``AblationResult`` (observed drift).
        scores: Iterable of ``RiskScore`` (a-priori predictions).
        drift_key (str): Which observed drift attribute to use ('drift_L0_T',
            'drift_L1_T', or 'drift_free'). Defaults to the headline T-evaluated drift.

    Returns:
        dict: {'pearson', 'spearman', 'kendall', 'n'} (nan when fewer than 2 aligned
        pairs or zero variance). The predictor succeeds when these are positive.
    """
    score_lookup = {(s.system, tuple(sorted(s.target_phases))): s.score for s in scores}
    xs, ys = [], []
    for r in results:
        key = (r.system, tuple(sorted(r.ablated_phases)))
        if key in score_lookup and r.fit_ok:
            obs = getattr(r, drift_key)
            if np.isfinite(obs):
                xs.append(score_lookup[key])
                ys.append(obs)

    out = {'pearson': np.nan, 'spearman': np.nan, 'kendall': np.nan, 'n': len(xs)}
    if len(xs) < 2:
        return out

    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.std(xs_a) == 0 or np.std(ys_a) == 0:
        return out

    out['pearson'] = float(np.corrcoef(xs_a, ys_a)[0, 1])
    try:
        from scipy.stats import kendalltau, spearmanr
        out['spearman'] = float(spearmanr(xs_a, ys_a).correlation)
        out['kendall'] = float(kendalltau(xs_a, ys_a).correlation)
    except Exception:
        # Spearman via rank-transformed Pearson if scipy is unavailable.
        rx = np.argsort(np.argsort(xs_a))
        ry = np.argsort(np.argsort(ys_a))
        out['spearman'] = float(np.corrcoef(rx, ry)[0, 1])
    return out
