"""
Fisher Information Matrix analysis for the gliquid binary liquid model.

Provides tools to quantify how much information the liquidus curve data contains
about each Redlich-Kister parameter, compare the effect of different constraint
sets, and find the optimal composition for a next liquidus measurement.

Mathematical basis:
    For Gaussian noise T_i ~ N(T_model(x_i; theta), sigma^2), the Fisher
    Information Matrix is:

        FIM = (1/sigma^2) * J^T * J

    where J[i,j] = dT_model(x_i) / d(theta_j) is the Jacobian of predicted
    liquidus temperature at each composition w.r.t. each free parameter.

    Parameter covariance ~ FIM^{-1}.

    D-optimal next measurement at x* maximizes:
        det(FIM_new) / det(FIM_old) = 1 + j(x*)^T FIM^{-1} j(x*) / sigma^2

Usage example:
    bl = BinaryLiquid.from_cache('Cu-Mg')
    bl.fit_parameters(n_opts=1)

    result = compute_fim(bl)
    print(result.param_names, result.param_variances)

    opt = find_optimal_next_measurement(result, bl)
    print("Best composition to measure next:", opt.ranked_x[0])

Notes on constraint state:
    After fit_parameters(), bl.guess_symbols reflects which parameters were
    free during the last fitting pass, but bl.constraints may be in a stale
    intermediate state. All FIM functions here bypass bl.constraints entirely
    and use direct parameter perturbation (setting bl._params directly) so that
    any post-fit object works without modification.
"""
from __future__ import annotations

import copy
import dataclasses
import warnings

import numpy as np

from gliquid.binary import a_sym, b_sym, c_sym, d_sym

_SYMBOL_TO_INDEX: dict = {a_sym: 0, b_sym: 1, c_sym: 2, d_sym: 3}
_SYMBOL_TO_NAME: dict = {a_sym: 'L0_a', b_sym: 'L0_b', c_sym: 'L1_a', d_sym: 'L1_b'}
_ALL_PARAM_NAMES = ['L0_a', 'L0_b', 'L1_a', 'L1_b']


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FIMResult:
    """Result of Fisher Information Matrix computation."""
    fim: np.ndarray
    fim_inv: np.ndarray
    jacobian: np.ndarray
    x_used: np.ndarray
    x_rejected: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    det_fim: float
    param_variances: np.ndarray
    param_names: list[str]
    is_singular: bool
    sigma: float


@dataclasses.dataclass
class OptimalMeasurementResult:
    """Result of D-optimal next-measurement search."""
    ranked_x: np.ndarray
    d_optimal_scores: np.ndarray
    jacobian_at_candidates: np.ndarray
    n_out_of_range: int


@dataclasses.dataclass
class NMSensitivityResult:
    """Sensitivity analysis extracted from the Nelder-Mead optimization path."""
    per_param_sensitivity: np.ndarray
    param_names: list[str]
    delta_f_per_iter: np.ndarray
    delta_params_per_iter: np.ndarray
    n_iters: int


@dataclasses.dataclass
class ConstraintComparisonResult:
    """Comparison of FIMs computed under different constraint configurations."""
    labels: list[str]
    fim_results: list[FIMResult]
    det_fim_values: np.ndarray
    condition_numbers: np.ndarray
    eigenvalue_table: np.ndarray
    param_variance_table: dict[str, list]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _working_copy(bl):
    """Return a deepcopy of bl so finite-difference evaluations never mutate the original."""
    return copy.deepcopy(bl)


def _interpolate_liquidus_at_x(phase_points_L: list, x: float) -> float:
    """
    Linearly interpolate liquidus temperature at composition x.
    Returns np.nan if x is outside the liquidus range or phase_points_L is empty.
    """
    if not phase_points_L:
        return np.nan
    arr = np.asarray(phase_points_L, dtype=float)
    order = np.argsort(arr[:, 0])
    xs, ts = arr[order, 0], arr[order, 1]
    return float(np.interp(x, xs, ts, left=np.nan, right=np.nan))


def _eval_liquidus_at_params(bl, params_4: list, x_query: np.ndarray) -> np.ndarray:
    """
    Evaluate T_model at compositions x_query by directly setting all 4 parameters.

    Does NOT mutate bl. Returns an array of length len(x_query); entries are
    np.nan when the composition is outside the computed liquidus range or when
    the phase diagram calculation fails.

    This function intentionally bypasses bl.constraints / bl.guess_symbols so
    that it works correctly after fit_parameters() (which may leave the constraint
    dict in a stale intermediate state).

    Parameters
    ----------
    bl : BinaryLiquid
    params_4 : list of 4 floats — [L0_a, L0_b, L1_a, L1_b]
    x_query : np.ndarray of compositions
    """
    nan_result = np.full(len(x_query), np.nan)
    wc = _working_copy(bl)
    try:
        wc.update_params(params_4)  # calls update_phase_points() internally
    except (ValueError, TypeError, RuntimeError):
        return nan_result
    try:
        phase_L = wc.hsx.get_phase_points().get('L', [])
    except Exception:
        return nan_result
    if not phase_L:
        return nan_result
    return np.array([_interpolate_liquidus_at_x(phase_L, x) for x in x_query])


def _free_param_indices(bl) -> list[int]:
    """
    Return the indices (into bl._params) of the free parameters.

    Uses bl.guess_symbols if set and non-None, otherwise returns all 4.
    """
    if bl.guess_symbols:
        return [_SYMBOL_TO_INDEX[sym] for sym in bl.guess_symbols]
    return [0, 1, 2, 3]


def _param_names_from_indices(indices: list[int]) -> list[str]:
    """Return human-readable parameter names for the given _params indices."""
    return [_ALL_PARAM_NAMES[i] for i in indices]


def _theta0_from_bl(bl) -> list[float]:
    """Return current parameter values for the free (guessed) parameters of bl."""
    return [bl._params[i] for i in _free_param_indices(bl)]


def build_nm_path_parameter_precision(
    bl,
    floor: float = 1e-3,
    power: float = 1.0,
) -> np.ndarray | None:
    """
    Build a diagonal precision prior from the Nelder-Mead optimization path.

    The path sensitivity is converted into a nonnegative per-parameter weight.
    Larger sensitivity means the objective changes more strongly with that
    parameter, so the corresponding prior precision is higher.
    """
    if bl.nmpath is None or bl.nmpath.ndim != 3:
        return None

    nm_sensitivity = compute_nm_path_sensitivity(bl)
    sensitivity = np.asarray(nm_sensitivity.per_param_sensitivity, dtype=float)
    if sensitivity.size == 0 or not np.any(np.isfinite(sensitivity)):
        return None

    sensitivity = np.clip(sensitivity, 0.0, None)
    total = float(np.sum(sensitivity))
    if total <= 0.0:
        weights = np.ones_like(sensitivity) / max(len(sensitivity), 1)
    else:
        weights = sensitivity / total

    if power != 1.0:
        weights = np.power(weights, power)

    max_weight = float(np.max(weights)) if np.any(weights > 0.0) else 1.0
    precision = floor + weights / max(max_weight, 1e-12)
    return precision


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_jacobian(
    bl,
    x_compositions: np.ndarray,
    free_param_indices: list[int] | None = None,
    h_rel: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute J[i, j] = dT_model(x_i) / d(theta_j) using central finite differences.

    Parameters
    ----------
    bl : BinaryLiquid
        Must have bl._params set to the operating-point parameter values.
    x_compositions : np.ndarray
        Compositions at which to evaluate the liquidus, shape (n_x,).
    free_param_indices : list of int or None
        Indices into bl._params to treat as free (perturbed) parameters.
        If None, uses bl.guess_symbols if set, otherwise all 4 parameters.
    h_rel : float
        Relative step size for finite differences. Absolute step = h_rel * max(|theta_j|, 1).

    Returns
    -------
    J : np.ndarray, shape (n_valid, n_free_params)
        Jacobian rows correspond to compositions where T_model is finite.
    valid_mask : np.ndarray of bool, shape (n_x,)
        True for compositions included in J.
    """
    x_compositions = np.asarray(x_compositions, dtype=float)
    params_4 = list(bl._params)
    indices = free_param_indices if free_param_indices is not None else _free_param_indices(bl)
    n_params = len(indices)

    T_nom = _eval_liquidus_at_params(bl, params_4, x_compositions)
    valid_mask = np.isfinite(T_nom)
    n_rejected = int(np.sum(~valid_mask))
    if n_rejected > 0:
        warnings.warn(
            f"compute_jacobian: {n_rejected}/{len(x_compositions)} composition(s) "
            "out of liquidus range — excluded from Jacobian.",
            stacklevel=2,
        )

    x_valid = x_compositions[valid_mask]
    n_valid = len(x_valid)
    J = np.zeros((n_valid, n_params))

    for j, param_idx in enumerate(indices):
        h = h_rel * max(abs(params_4[param_idx]), 1.0)

        p_plus = list(params_4)
        p_plus[param_idx] += h
        p_minus = list(params_4)
        p_minus[param_idx] -= h

        T_plus = _eval_liquidus_at_params(bl, p_plus, x_valid)
        T_minus = _eval_liquidus_at_params(bl, p_minus, x_valid)

        for row in range(n_valid):
            tp, tm = T_plus[row], T_minus[row]
            if np.isfinite(tp) and np.isfinite(tm):
                J[row, j] = (tp - tm) / (2.0 * h)
            elif np.isfinite(tp):
                T_nom_row = _eval_liquidus_at_params(bl, params_4, x_valid[row:row+1])[0]
                J[row, j] = (tp - T_nom_row) / h if np.isfinite(T_nom_row) else 0.0
            elif np.isfinite(tm):
                T_nom_row = _eval_liquidus_at_params(bl, params_4, x_valid[row:row+1])[0]
                J[row, j] = (T_nom_row - tm) / h if np.isfinite(T_nom_row) else 0.0
            else:
                J[row, j] = 0.0

    return J, valid_mask


def compute_fim(
    bl,
    x_compositions: np.ndarray | None = None,
    sigma: float = 1.0,
    free_param_indices: list[int] | None = None,
    h_rel: float = 1e-3,
    prior_lambda: float = 0.0,
    parameter_prior_precision: np.ndarray | None = None,
    parameter_prior_strength: float = 0.0,
) -> FIMResult:
    """
    Compute the Fisher Information Matrix at the current parameter values of bl.

    Parameters
    ----------
    bl : BinaryLiquid
        Must have bl._params set. bl.digitized_liq is used when x_compositions is None.
    x_compositions : array-like or None
        Compositions at which to evaluate. If None, uses x-values from
        bl.digitized_liq (endpoints x=0 and x=1 excluded). Pass an empty
        array ``np.array([])`` together with ``prior_lambda > 0`` for the
        cold-start case (no measurements yet).
    sigma : float
        Temperature measurement uncertainty (K). FIM = J^T J / sigma^2.
        Absolute value does not affect relative comparisons; default 1.0.
    free_param_indices : list of int or None
        Indices of bl._params to treat as free parameters.
        If None, uses bl.guess_symbols if set, otherwise all 4.
    h_rel : float
        Relative step size for finite differences.
    prior_lambda : float
        Regularization added to FIM diagonal: FIM += prior_lambda * I.
        Required when x_compositions is empty (cold start). With no data the
        score for each candidate becomes 1 + ||j(x)||² / prior_lambda, which
        ranks compositions by Jacobian magnitude — sensible for a first
        measurement. Default 0.0 (no regularization).
    parameter_prior_precision : np.ndarray or None
        Optional per-parameter diagonal precision prior aligned with the free
        parameter indices. Typically derived from the Nelder-Mead path via
        ``build_nm_path_parameter_precision()``.
    parameter_prior_strength : float
        Scalar multiplier applied to ``parameter_prior_precision`` before it is
        added to the FIM diagonal.

    Returns
    -------
    FIMResult
    """
    if x_compositions is None:
        if not bl.digitized_liq:
            if prior_lambda <= 0.0:
                raise ValueError(
                    "bl.digitized_liq is empty; provide x_compositions explicitly "
                    "or set prior_lambda > 0 for cold-start mode."
                )
            x_compositions = np.array([], dtype=float)
        else:
            xs = np.array([pt[0] for pt in bl.digitized_liq], dtype=float)
            x_compositions = np.unique(xs[(xs > 0.0) & (xs < 1.0)])
    else:
        x_compositions = np.asarray(x_compositions, dtype=float)
        if len(x_compositions) == 0 and prior_lambda <= 0.0:
            raise ValueError(
                "x_compositions is empty; set prior_lambda > 0 for cold-start mode."
            )

    indices = free_param_indices if free_param_indices is not None else _free_param_indices(bl)
    J, valid_mask = compute_jacobian(bl, x_compositions, free_param_indices=indices, h_rel=h_rel)
    x_used = x_compositions[valid_mask] if len(x_compositions) > 0 else np.array([], dtype=float)
    x_rejected = x_compositions[~valid_mask] if len(x_compositions) > 0 else np.array([], dtype=float)

    fim = J.T @ J / sigma**2

    if prior_lambda > 0.0:
        fim = fim + prior_lambda * np.eye(len(indices))

    if parameter_prior_precision is not None:
        parameter_prior_precision = np.asarray(parameter_prior_precision, dtype=float)
        if parameter_prior_precision.shape != (len(indices),):
            raise ValueError(
                "parameter_prior_precision must have shape (n_free_params,)"
            )
        if parameter_prior_strength != 0.0:
            fim = fim + parameter_prior_strength * np.diag(parameter_prior_precision)

    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # numerical negatives near zero → 0

    cond = float(eigenvalues[-1] / max(float(eigenvalues[0]), 1e-300))
    det_fim = float(np.linalg.det(fim))

    try:
        fim_inv = np.linalg.inv(fim)
        is_singular = False
    except np.linalg.LinAlgError:
        fim_inv = np.linalg.pinv(fim)
        is_singular = True

    if np.isnan(fim_inv).any() or np.isinf(fim_inv).any():
        fim_inv = np.linalg.pinv(fim)
        is_singular = True

    param_variances = np.diag(fim_inv)
    param_names = _param_names_from_indices(indices)

    return FIMResult(
        fim=fim,
        fim_inv=fim_inv,
        jacobian=J,
        x_used=x_used,
        x_rejected=x_rejected,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=cond,
        det_fim=det_fim,
        param_variances=param_variances,
        param_names=param_names,
        is_singular=is_singular,
        sigma=sigma,
    )


def find_optimal_next_measurement(
    fim_result: FIMResult,
    bl,
    candidate_x: np.ndarray | None = None,
    h_rel: float = 1e-3,
) -> OptimalMeasurementResult:
    """
    D-optimal experimental design: rank candidate compositions by the gain in
    det(FIM) from adding one hypothetical liquidus measurement there.

    Uses the matrix determinant lemma:
        det(FIM + j j^T / sigma^2) / det(FIM) = 1 + j^T FIM^{-1} j / sigma^2

    Parameters
    ----------
    fim_result : FIMResult
        Result from compute_fim().
    bl : BinaryLiquid
        Used to evaluate the Jacobian at candidate compositions.
    candidate_x : np.ndarray or None
        Compositions to evaluate. If None, 50 evenly-spaced points in
        [x_used.min(), x_used.max()] excluding already-measured compositions.
    h_rel : float
        Step size for finite differences.

    Returns
    -------
    OptimalMeasurementResult
        ranked_x and d_optimal_scores are sorted in descending order of score.
        If every candidate is out of range for the current liquidus evaluation,
        the ranking falls back to a boundary-expansion heuristic so the result
        is deterministic instead of depending on array order.
    """
    x_used = fim_result.x_used
    if candidate_x is None:
        lo, hi = float(x_used.min()), float(x_used.max())
        grid = np.linspace(lo, hi, 52)[1:-1]  # 50 interior points
        candidate_x = np.array([x for x in grid
                                 if not np.any(np.abs(x_used - x) < 1e-3)])
        if len(candidate_x) == 0:
            candidate_x = grid

    candidate_x = np.asarray(candidate_x, dtype=float)

    # Use the same free_param_indices that generated fim_result
    param_names = fim_result.param_names
    indices = [_ALL_PARAM_NAMES.index(name) for name in param_names]
    J_new, _ = compute_jacobian(bl, candidate_x, free_param_indices=indices, h_rel=h_rel)

    scores = np.ones(len(candidate_x))
    n_out_of_range = 0
    for i, j_row in enumerate(J_new):
        if np.all(j_row == 0.0):
            n_out_of_range += 1
            scores[i] = 1.0
        else:
            scores[i] = 1.0 + float(j_row @ fim_result.fim_inv @ j_row) / fim_result.sigma**2

    if len(candidate_x) > 0 and n_out_of_range == len(candidate_x):
        # When the model cannot evaluate any candidate, prefer sampling near the
        # current observed window edges instead of relying on arbitrary tie-breaking.
        if len(x_used) > 0:
            edge_dist = np.minimum(np.abs(candidate_x - float(np.min(x_used))),
                                   np.abs(candidate_x - float(np.max(x_used))))
        else:
            center = float(np.mean(candidate_x))
            edge_dist = np.abs(candidate_x - center)
        scores = 1.0 + 1.0 / (1.0 + edge_dist)

    order = np.argsort(scores)[::-1]
    return OptimalMeasurementResult(
        ranked_x=candidate_x[order],
        d_optimal_scores=scores[order],
        jacobian_at_candidates=J_new[order],
        n_out_of_range=n_out_of_range,
    )


def compute_nm_path_sensitivity(bl) -> NMSensitivityResult:
    """
    Analyze the Nelder-Mead optimization path to identify which free-parameter
    directions drive the largest changes in the objective (MAE).

    Requires bl.nmpath to be populated (i.e., fit_parameters() has been run).

    The per-parameter sensitivity is a normalized score: sensitivity[j] reflects
    how much |Delta_f| accumulates per unit change in parameter j across the
    entire NM trajectory. It sums to 1 for easy comparison.

    Returns
    -------
    NMSensitivityResult
    """
    if bl.nmpath is None or bl.nmpath.ndim != 3:
        raise ValueError(
            "bl.nmpath is not populated. Run bl.fit_parameters() or bl.nelder_mead() first."
        )

    indices = _free_param_indices(bl)
    n_free = len(indices)
    n_iters = bl.nmpath.shape[2]

    # nmpath shape: (3 vertices, 5 values [4 params + f], n_iters)
    params_path = bl.nmpath[:, indices, :]  # (3, n_free, n_iters)
    f_path = bl.nmpath[:, 4, :]            # (3, n_iters)

    sensitivity = np.zeros(n_free)
    delta_f_list = []
    delta_params_list = []
    eps = 1e-12

    for i in range(1, n_iters):
        ibest_prev = int(np.argmin(f_path[:, i - 1]))
        ibest_curr = int(np.argmin(f_path[:, i]))
        delta_params = params_path[ibest_curr, :, i] - params_path[ibest_prev, :, i - 1]
        delta_f = abs(float(f_path[ibest_curr, i]) - float(f_path[ibest_prev, i - 1]))
        sensitivity += delta_f / (np.abs(delta_params) + eps)
        delta_f_list.append(delta_f)
        delta_params_list.append(delta_params.copy())

    total = sensitivity.sum()
    if total > 0:
        sensitivity /= total

    return NMSensitivityResult(
        per_param_sensitivity=sensitivity,
        param_names=_param_names_from_indices(indices),
        delta_f_per_iter=np.array(delta_f_list),
        delta_params_per_iter=np.array(delta_params_list),
        n_iters=n_iters,
    )


def compare_constraint_sets(
    bl_list: list,
    labels: list[str],
    x_compositions: np.ndarray | None = None,
    sigma: float = 1.0,
    free_param_indices: list[int] | None = None,
) -> ConstraintComparisonResult:
    """
    Compute FIMs for multiple BinaryLiquid objects (each with a different
    constraint configuration) and return a comparative summary.

    Each BinaryLiquid in bl_list should reflect the desired constraint configuration
    via bl.guess_symbols (which determines which parameters are treated as free).
    FIMs can have different dimensions when constraint sets differ; comparison is
    provided via:
      - det(FIM) — overall information content
      - Condition number — parameter identifiability
      - Eigenvalue spectra (nan-padded to 4 columns)
      - Per-parameter variances (nan where a parameter is not free)

    Parameters
    ----------
    bl_list : list of BinaryLiquid
    labels : list of str
        Human-readable label for each configuration.
    x_compositions : array-like or None
        If provided, used for all FIM computations (makes comparisons fair).
        If None, each bl uses its own digitized_liq compositions.
    sigma : float
        Temperature measurement uncertainty (K).
    free_param_indices : list of int or None
        Override which parameter indices to treat as free for every FIM
        computation. If None, each bl's own guess_symbols is used.
        Example: [0, 2] to use L0_a and L1_a regardless of guess_symbols.

    Returns
    -------
    ConstraintComparisonResult
    """
    if len(bl_list) != len(labels):
        raise ValueError("bl_list and labels must have the same length.")

    results = [compute_fim(bl, x_compositions, sigma,
                           free_param_indices=free_param_indices)
               for bl in bl_list]

    det_fim_values = np.array([r.det_fim for r in results])
    condition_numbers = np.array([r.condition_number for r in results])

    eigenvalue_table = np.full((len(results), 4), np.nan)
    for i, r in enumerate(results):
        n = len(r.eigenvalues)
        eigenvalue_table[i, :n] = r.eigenvalues

    param_variance_table: dict[str, list] = {name: [] for name in _ALL_PARAM_NAMES}
    for r in results:
        present = {name: var for name, var in zip(r.param_names, r.param_variances)}
        for name in _ALL_PARAM_NAMES:
            param_variance_table[name].append(present.get(name, np.nan))

    return ConstraintComparisonResult(
        labels=labels,
        fim_results=results,
        det_fim_values=det_fim_values,
        condition_numbers=condition_numbers,
        eigenvalue_table=eigenvalue_table,
        param_variance_table=param_variance_table,
    )
