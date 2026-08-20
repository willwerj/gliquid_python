"""
Author: Joshua Willwerth
Description: The shared solution-phase thermodynamics module of gliquid — the single home for
the symbolic expression builders (N-component core plus the binary and ternary wrappers), the
thermodynamic symbols and composition grid, the interpolation geometry (linear/Muggianu/Kohler),
and the solid-solution reference resolvers. Any phase type plugs into the builders through its
per-component reference G(T) expressions (liquid H−TS references, solid-solution ΔH−TΔS
references, or a future gas model).

The resolver half builds per-phase sub-regular solid-solution models (a single regular-solution
interaction Omega per phase plus per-element enthalpy/entropy offsets relative to each element's
DFT ground state) from one of three reference sources, and reconciles each element's liquid
reference against the solid-phase ladder the model carries so the solid-solution and liquid free
energies share one reference frame.

The resolvers are written against the ComponentRef/Phase API in gliquid.phase. Numeric
behavior is pinned by tests/test_binary_solution.py and tests/test_solution.py against
fixtures/ss_characterization_pins.json.

Model dict schema (one per solid-solution phase name, e.g. "BCC"; any n >= 2 components):
    {
      "refs": {<el>: {source, ground_material_id, ground_spacegroup, ground_symbol,
                      ground_energy_ev_per_atom, material_id, spacegroup, symbol,
                      energy_ev_per_atom, delta_h_jmol, delta_s_jmol_k}},
      "ref_mode": str,                     # resolver that produced the refs
      "omega":   {"<A>-<B>": float},       # J/mol regular-solution interaction per
                                           #   alphabetized pair key, all C(n,2) pairs
      "delta_h": {<el>: float},            # J/mol per-element enthalpy offsets
      "delta_s": {<el>: float},            # J/(mol*K) per-element entropy offsets
    }
All values are plain JSON-able scalars so models pickle cleanly through multiprocessing.

REFERENCE FRAME (the load-bearing contract every resolver must honor): ``delta_h_jmol`` and
``delta_s_jmol_k`` are CUMULATIVE above the element's ground state — the enthalpy/entropy of the
whole ladder up to that phase, i.e. ``Phase.enthalpy``/``Phase.entropy``, NOT the single-transition
steps ``Phase.delta_h``/``Phase.delta_s``. They must be on the same zero as the elemental polymorph
line compounds, which carry cumulative values (``binary.build_phases_from_chull``), because the
hull compares the two directly and an SS phase replaces the polymorph it covers. Publishing steps
here silently moves the pure-element melting point for any element with more than one solid
transition below melting (Ti was off by 458 K); a single-transition element cannot reveal it,
since there step == cumulative. The unary DB's ``lattice_stabilities`` block shares this frame:
its values are already cumulative above the element anchor (no transition temperature), so
``_resolve_refs_db`` publishes them directly. Their entropy is 0 by the recalculation convention
-- no transition temperature to divide by -- except for the builder's scoped metastable-entropy
exception, which carries a NEGATIVE SGTE entropy on a few HCP entries (Au, Li, Ba, Ca, Cu, Ag) so
their free energy climbs with temperature instead of staying pinned at the 0 K enthalpy. Those
entries still have no transition temperature, so they stay out of ``_ordered_solid_steps`` and the
liquid reconciliation; the sign is what keeps the element's own melting point exact.
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import json
import logging
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import gliquid.config as config
from gliquid.phase import (
    EV_ATOM_TO_J_MOL,
    SS_SPACEGROUPS,
    SS_SYMBOLS,
    ComponentRef,
    Phase,
)

logger = logging.getLogger(__name__)

ALL_SS_PHASES = list(SS_SPACEGROUPS)
DEFAULT_REF_MODE = (
    "from_unary_db"  # Options: "from_dft_entries", "from_omegas_file", or "from_unary_db"
)


# --------------------------------------------------------------------------------------
# Shared thermodynamic symbols, composition grid, and mixing-expression helpers — the
# single home for the definitions binary.py and ternary.py both use.
# --------------------------------------------------------------------------------------

R = 8.314  # J/(mol*K), universal gas constant
DEFAULT_TAU = 8000  # K; decay constant of the exp(-T/tau) excess-enthalpy envelope

t_sym = sp.Symbol("t")

x_step = 0.01  # Sets composition grid precision; has not been tested for values other than 0.01
x_prec = len(str(x_step).split(".")[-1])
x_vals = np.arange(0, 1 + x_step, x_step)


def refined_x_vals(factor: int = 5, step: float = x_step) -> np.ndarray:
    """Composition grid with the INTERIOR subdivided ``factor`` times.

    Used by the presentation hull only (``BinaryLiquid.refined_hsx``). A solution phase's
    field terminates between grid nodes, so the coarse grid collapses a range of true
    compositions onto one node and the plotted field ends in a vertical face that is not a
    phase boundary -- Hf-Y's BCC has 14 distinct equilibria (1298-1629 C) all sitting at
    x=0.21. Subdividing converges that face: Hf-Y's HCP termini close to a single apex at
    ``factor=5``, the Y-side one landing on the eutectic temperature to 0.1 K.

    The two OUTERMOST intervals deliberately keep their coarse spacing. Ideal-mixing
    entropy drives a solvus asymptotically to T=0 as x->0, so sampling nearer an axis walks
    the terminal invariant downward without ever converging (Hf-Y's Hf+HCP+Y peritectic
    runs 733 -> 627 -> 515 -> 446 C as the step goes 0.01 -> 0.001, eventually dropping off
    the plot floor). Refining there would move real tie lines for no geometric gain; the
    axis corners come from the component transition temperatures instead.
    """
    if factor < 1:
        raise ValueError(f"'factor' must be a positive integer, got {factor!r}.")
    if factor == 1 and step == x_step:
        return x_vals  # the module grid itself, so factor=1 is bit-identical, not merely close
    n = int(round(1.0 / step)) * int(factor)
    return np.arange(n + 1, dtype=float) / n


def comp_symbols(n: int) -> tuple[sp.Symbol, ...]:
    """The n-1 independent composition symbols of an n-component solution phase.

    ``x_0 = 1 - sum(x_i)`` is always the derived fraction. Legacy names are preserved so
    expressions print and substitute identically to the pre-refactor modules:
    n=2 -> (x,), n=3 -> (x1, x2), n=4 -> (x1, x2, x3), ...
    """
    if n < 2:
        raise ValueError("A solution phase needs at least 2 components.")
    if n == 2:
        return (sp.Symbol("x"),)
    return tuple(sp.Symbol(f"x{i}") for i in range(1, n))


xb_sym = comp_symbols(2)[0]  # the binary composition symbol (fraction of component B)


# --------------------------------------------------------------------------------------
# Redlich-Kister parameter formats — the single, centralized registry of the L-term
# formalisms. A ParamFormat declares, per RK order L0..L3: the subterm roles
# ('a', 'b', ...), the functional form L_k(T), which parameters are pinned to zero
# during fitting, and the fitting topology (Nelder-Mead guess parameters, how many
# invariant-derived constraints the format consumes, penalty policy). Consumers across
# binary.py/ternary.py read this metadata rather than branching on the raw param_format
# string.
# --------------------------------------------------------------------------------------

RK_MAX_ORDER = 3  # Highest supported Redlich-Kister order (L0..L3)


def l_sym(name: str) -> sp.Symbol:
    """The sympy symbol for one Redlich-Kister parameter name (e.g. ``'L0_a'``)."""
    return sp.Symbol(name)


def linear_term(syms, tau: float = DEFAULT_TAU) -> sp.Expr:
    """``L_k(T) = a + b*T``."""
    a, b = syms
    return a + b * t_sym


def combined_term(syms, tau: float = DEFAULT_TAU) -> sp.Expr:
    """``L_k(T) = (a + b*T) * exp(-T/tau)`` — the excess-enthalpy decay envelope."""
    a, b = syms
    return (a + b * t_sym) * sp.exp(-t_sym / sp.Integer(tau))


def constant_term(syms, tau: float = DEFAULT_TAU) -> sp.Expr:
    """``L_k(T) = a`` (temperature-independent, e.g. a regular-solution omega)."""
    (a,) = syms
    return a


@dataclass(frozen=True)
class ParamFormat:
    """One Redlich-Kister parameter formalism.

    Attributes:
        name: Registry key (e.g. ``'linear'``).
        orders: RK orders the format carries, ascending subset of 0..3.
        order_roles: Per-order subterm roles, aligned with ``orders`` — e.g.
            ``(('a', 'b'), ('a', 'b'))`` names parameters L0_a, L0_b, L1_a, L1_b.
        order_exprs: Per-order term builders, aligned with ``orders``; each maps
            (subterm symbols, tau) to the symbolic ``L_k(T)``.
        pinned_params: Parameters structurally present but held at zero by the
            FITTING topology (identity constraints) — e.g. ``('L1_b',)`` for
            'comb-exp'. Manual parameter updates may still set them.
        guess_params: The Nelder-Mead free parameters (the optimizer is 2-D).
        n_invariant_constraints: How many invariant-derived constraint equations
            the fit consumes per candidate (the rest come from identity constraints).
        lupis_orders: RK orders the Lupis-Elliott sign penalty inspects.
        lxb_default: Whether the L*_b distribution penalty defaults on.
    """

    name: str
    orders: tuple[int, ...]
    order_roles: tuple[tuple[str, ...], ...]
    order_exprs: tuple[Callable, ...]
    pinned_params: tuple[str, ...] = ()
    guess_params: tuple[str, ...] = ()
    n_invariant_constraints: int = 2
    lupis_orders: tuple[int, ...] = (0,)
    lxb_default: bool = False

    def __post_init__(self):
        if not (len(self.orders) == len(self.order_roles) == len(self.order_exprs)):
            raise ValueError(
                f"Format '{self.name}': orders, order_roles and order_exprs "
                f"must align (got {len(self.orders)}/{len(self.order_roles)}"
                f"/{len(self.order_exprs)})."
            )
        if any(k < 0 or k > RK_MAX_ORDER for k in self.orders):
            raise ValueError(
                f"Format '{self.name}': orders must lie in 0..{RK_MAX_ORDER}, got {self.orders}."
            )
        if tuple(sorted(set(self.orders))) != self.orders:
            raise ValueError(
                f"Format '{self.name}': orders must be strictly ascending, got {self.orders}."
            )
        unknown = [
            p for p in (*self.pinned_params, *self.guess_params) if p not in self.param_names
        ]
        if unknown:
            raise ValueError(
                f"Format '{self.name}': {unknown} not among parameters {self.param_names}."
            )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Flat canonical parameter names, order-major: ('L0_a', 'L0_b', 'L1_a', ...)."""
        return tuple(f"L{k}_{r}" for k, roles in zip(self.orders, self.order_roles) for r in roles)

    def symbols(self) -> tuple[sp.Symbol, ...]:
        """Sympy symbols aligned with ``param_names`` (the flat parameter layout)."""
        return tuple(l_sym(n) for n in self.param_names)

    @property
    def solve_params(self) -> tuple[str, ...]:
        """Parameters the fit solves from constraints (everything not guessed)."""
        return tuple(n for n in self.param_names if n not in self.guess_params)

    def identity_constraints(self) -> list[sp.Eq]:
        """``Eq(param, 0)`` for each pinned parameter — the fit's fixed equations."""
        return [sp.Eq(l_sym(p), 0) for p in self.pinned_params]

    def order_expr(self, order: int, tau: float = DEFAULT_TAU) -> sp.Expr:
        """Symbolic ``L_k(T)`` for one order, in that order's own parameter symbols."""
        idx = self.orders.index(order)
        syms = tuple(l_sym(f"L{order}_{r}") for r in self.order_roles[idx])
        return self.order_exprs[idx](syms, tau)


PARAM_FORMATS: dict[str, ParamFormat] = {
    "linear": ParamFormat(
        name="linear",
        orders=(0, 1),
        order_roles=(("a", "b"), ("a", "b")),
        order_exprs=(linear_term, linear_term),
        guess_params=("L0_b", "L1_b"),
        n_invariant_constraints=2,
        lupis_orders=(0, 1),
    ),
    "combined": ParamFormat(
        name="combined",
        orders=(0, 1),
        order_roles=(("a", "b"), ("a", "b")),
        order_exprs=(combined_term, combined_term),
        guess_params=("L0_b", "L1_b"),
        n_invariant_constraints=2,
        lupis_orders=(0, 1),
    ),
    "comb-exp": ParamFormat(
        name="comb-exp",
        orders=(0, 1),
        order_roles=(("a", "b"), ("a", "b")),
        order_exprs=(combined_term, combined_term),
        pinned_params=("L1_b",),
        guess_params=("L0_b", "L1_a"),
        n_invariant_constraints=1,
        lupis_orders=(0,),
        lxb_default=True,
    ),
    "regular": ParamFormat(
        name="regular",
        orders=(0,),
        order_roles=(("a",),),
        order_exprs=(constant_term,),
        n_invariant_constraints=0,
    ),
}


def get_param_format(name: str) -> ParamFormat:
    """Resolve a registry key to its ParamFormat; unknown keys raise ValueError."""
    try:
        return PARAM_FORMATS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"'param_format' must be one of {sorted(PARAM_FORMATS)}, got {name!r}."
        ) from None


class RKPolyExp:
    """Excess-interaction model of one component pair.

    Contributes ``x_i * x_j * sum_k L_k(T) * diff^k`` to a solution phase's excess
    Gibbs energy, where the orders k (up to L3), each L_k(T)'s functional form, its
    a/b/... subterms, and the fitting topology all come from the ``ParamFormat``.
    Parameter values live in the format's flat order-major layout
    (``format.param_names``, e.g. ``[L0_a, L0_b, L1_a, L1_b]``).

    Args:
        param_format: Registry key (``'linear'``/``'combined'``/``'comb-exp'``/
            ``'regular'``) or a ``ParamFormat`` instance (for unregistered formats).
        params: Flat parameter values (empty means all-zero defaults).
        tau: exp(-T/tau) decay constant consumed by the 'combined'-family forms.
    """

    def __init__(
        self, param_format: str | ParamFormat = "linear", params=(), tau: float = DEFAULT_TAU
    ):
        self.format = (
            param_format
            if isinstance(param_format, ParamFormat)
            else get_param_format(param_format)
        )
        self.tau = tau
        self._params = self._validate(params)

    def _validate(self, params) -> list[float]:
        names = self.format.param_names
        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                return [0.0] * len(names)
            if len(params) == len(names) and all(
                isinstance(v, numbers.Number) and not isinstance(v, bool) for v in params
            ):
                return [float(v) for v in params]
        raise ValueError(
            f"Parameters for format '{self.format.name}' must be a list or "
            f"tuple of {len(names)} numbers in the order {list(names)}."
        )

    # ------------------------------------------------------------------ values
    @property
    def values(self) -> list[float]:
        """Flat parameter values (a copy), aligned with ``format.param_names``."""
        return list(self._params)

    @property
    def n_params(self) -> int:
        return len(self._params)

    def update(self, params) -> None:
        """Replace the full parameter vector (validated)."""
        self._params = self._validate(params)

    def _index(self, key) -> int:
        if isinstance(key, str):
            try:
                return self.format.param_names.index(key)
            except ValueError:
                raise KeyError(
                    f"Unknown parameter {key!r}; format '{self.format.name}' "
                    f"has {list(self.format.param_names)}."
                ) from None
        return int(key)

    def order_values(self, order: int) -> list[float]:
        """The parameter values of one RK order, in its subterm-role order."""
        pos = 0
        for k, roles in zip(self.format.orders, self.format.order_roles):
            if k == order:
                return self._params[pos : pos + len(roles)]
            pos += len(roles)
        raise KeyError(f"Format '{self.format.name}' carries no L{order} term.")

    def __getitem__(self, key) -> float:
        return self._params[self._index(key)]

    def __setitem__(self, key, value) -> None:
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise ValueError(f"Parameter value must be numeric, got {value!r}.")
        self._params[self._index(key)] = float(value)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, RKPolyExp)
            and self.format == other.format
            and self._params == other._params
            and self.tau == other.tau
        )

    def __repr__(self) -> str:
        return (
            f"RKPolyExp(param_format='{self.format.name}', params={self._params}, tau={self.tau})"
        )

    # ------------------------------------------------------------------ symbolic
    def exprs(self) -> tuple[sp.Expr, ...]:
        """Per-order symbolic ``L_k(T)`` in the format's parameter symbols."""
        return tuple(self.format.order_expr(k, self.tau) for k in self.format.orders)

    def subs_map(self) -> dict[sp.Symbol, float]:
        """``{parameter symbol: value}`` over the full flat layout."""
        return dict(zip(self.format.symbols(), self._params))

    def numeric_exprs(self) -> tuple[sp.Expr, ...]:
        """Per-order ``L_k(T)`` with this instance's parameter values substituted."""
        subs = self.subs_map()
        return tuple(expr.subs(subs) for expr in self.exprs())

    def order_h_s(self, order: int) -> tuple[sp.Expr, sp.Expr]:
        """Symbolic (H_k, S_k) decomposition of one order: S = -dL/dT, H = L + T*S."""
        l_expr = self.format.order_expr(order, self.tau)
        s_expr = -sp.diff(l_expr, t_sym)
        return l_expr + t_sym * s_expr, s_expr

    # ------------------------------------------------------------------ swap
    def swapped(self) -> RKPolyExp:
        """This model with the component order reversed.

        ``L_k`` multiplies ``(x_i - x_j)^k``, so odd orders are odd under an i/j
        swap: every subterm of orders 1 and 3 negates (generalizes the retired
        ``flip_binary_l1``, which negated ``params[2:]`` of the 4-vector layout).
        """
        flipped = list(self._params)
        pos = 0
        for order, roles in zip(self.format.orders, self.format.order_roles):
            for _ in roles:
                if order % 2 == 1:
                    flipped[pos] = -flipped[pos]
                pos += 1
        return RKPolyExp(self.format, flipped, tau=self.tau)

    # ------------------------------------------------------------------ fitting metadata
    @property
    def guess_symbols(self) -> list[sp.Symbol]:
        """The Nelder-Mead free-parameter symbols (the 2-D optimizer's axes)."""
        return [l_sym(p) for p in self.format.guess_params]

    @property
    def solve_symbols(self) -> list[sp.Symbol]:
        """Symbols the fit solves from constraint equations (everything not guessed)."""
        return [l_sym(p) for p in self.format.solve_params]

    def identity_constraints(self) -> list[sp.Eq]:
        return self.format.identity_constraints()

    @property
    def guess_param_indices(self) -> tuple[int, ...]:
        """Flat-layout indices of the guess parameters (nmpath plotting axes)."""
        names = self.format.param_names
        return tuple(names.index(p) for p in self.format.guess_params)


class SolutionModel:
    """One N-component solution phase: references + per-pair Redlich-Kister excess.

    The single implementation behind the binary liquid ``eqs`` dict, the ternary
    liquid surface, and every solid-solution phase — replacing the former split
    between a symbolic ternary path and a hand-coded numpy binary-SS path.

    Args:
        components: Component symbols in model order (defines pair-index meaning).
        g_ref_exprs: Per-component reference G(T) sympy expressions, aligned with
            ``components`` (liquid H-TS references, SS dH-T*dS references, or a
            future gas model).
        pair_models: ``{key: RKPolyExp}`` interaction models. A PAIR key — an
            orientation-sensitive ``(i, j)`` index tuple or ``'A-B'`` name resolved
            against ``components`` — contributes the Redlich-Kister excess (odd RK
            orders multiply ``x_i - x_j``). A TRIPLET key — ``(i, j, k)`` or
            ``'A-B-C'`` — contributes ``x_i * x_j * x_k * L0(T)`` (permutation-
            symmetric; the model must carry ONLY order 0, e.g. the 'regular'
            format), the standard ternary interaction term.
        interp_scheme: 'linear' | 'muggianu' | 'kohler' pair geometry (n >= 3;
            a binary's single pair always reduces to the plain difference).
        ideal: Default ideal-term form for ``expressions()``/``h_s_grid()`` —
            'plain' (differentiable, the binary fitting form) or 'safe'
            (Piecewise-guarded, exact on edge/corner grid points).

    The dict-based ``ss_models`` schema (see the module docstring) remains the
    serialization format; ``from_ss_model`` lifts one phase's dict into a model.
    Lambdified grid callables are cached and dropped on pickle.
    """

    def __init__(
        self,
        components,
        g_ref_exprs,
        pair_models,
        *,
        interp_scheme: str = "linear",
        ideal: str = "plain",
    ):
        self.components = tuple(components)
        n = len(self.components)
        if n < 2:
            raise ValueError("A solution phase needs at least 2 components.")
        if len(g_ref_exprs) != n:
            raise ValueError(f"Expected {n} reference expressions, got {len(g_ref_exprs)}.")
        self.g_ref_exprs = tuple(g_ref_exprs)
        self.pair_models: dict[tuple[int, ...], RKPolyExp] = {
            self._interaction_key(key): model for key, model in pair_models.items()
        }
        for key, rk in self.pair_models.items():
            if len(key) == 3 and rk.format.orders != (0,):
                raise ValueError(
                    f"Ternary interaction {key} carries orders {rk.format.orders}; "
                    "triplet terms support single order-0 formats only (e.g. 'regular')."
                )
        self.interp_scheme = interp_scheme
        self.ideal = ideal
        self._grid_fns: dict = {}
        # (dh_a, dh_b, ds_a, ds_b, omega) for the binary pure-regular case — set by
        # from_ss_model so h_s_grid can reproduce the legacy arithmetic bit-exactly
        # (the hull goldens pin exact simplex sets; the sympy-lambdified path differs
        # at ~1e-16, enough to flip a degenerate facet).
        self._binary_regular: tuple | None = None

    def _interaction_key(self, key) -> tuple[int, ...]:
        """Normalize an ``'A-B'``/``'A-B-C'`` name key or index tuple to component indices."""
        if isinstance(key, str):
            names = key.split("-")
            if len(names) not in (2, 3) or not all(nm in self.components for nm in names):
                raise ValueError(
                    f"Interaction key {key!r} does not name two or three of {self.components}."
                )
            return tuple(self.components.index(nm) for nm in names)
        indices = tuple(int(i) for i in key)
        if len(indices) not in (2, 3):
            raise ValueError(f"Interaction key {key!r} must have 2 or 3 indices.")
        return indices

    @classmethod
    def from_ss_model(
        cls, components, ss_model: dict, interp_scheme: str = "linear"
    ) -> SolutionModel:
        """Lift one solid-solution phase's keyed model dict into a SolutionModel.

        ``g_ref_i = dH_i - T*dS_i`` and each pair's omega becomes a 'regular'
        RKPolyExp (single constant L0_a). The model follows the GIVEN component
        order (the owning system's construction order); the resolver schema itself is
        order-independent — omega keys are alphabetized pair strings, delta_h/delta_s
        are keyed by symbol.
        """
        els = list(components)
        dh, ds, omega = ss_model["delta_h"], ss_model["delta_s"], ss_model["omega"]
        pair_models = {}
        for i in range(len(els)):
            for j in range(i + 1, len(els)):
                key = "-".join(sorted((els[i], els[j])))
                pair_models[(i, j)] = RKPolyExp("regular", [float(omega[key])])
        model = cls(
            els,
            [dh[el] - t_sym * ds[el] for el in els],
            pair_models,
            interp_scheme=interp_scheme,
            ideal="safe",
        )
        if len(els) == 2:
            model._binary_regular = (
                float(dh[els[0]]),
                float(dh[els[1]]),
                float(ds[els[0]]),
                float(ds[els[1]]),
                float(omega["-".join(sorted(els))]),
            )
        return model

    # ------------------------------------------------------------------ symbolic
    def expressions(
        self, x_syms=None, t: sp.Symbol = t_sym, ideal: str | None = None, numeric: bool = True
    ) -> dict[str, sp.Expr]:
        """Symbolic G/H/S dict for this phase (``build_solution_expressions`` keys).

        ``numeric=True`` substitutes each pair's parameter values (evaluation
        surfaces); ``numeric=False`` leaves the RK parameter symbols free (the
        binary fitting path solves for them).
        """
        n = len(self.components)
        if x_syms is None:
            x_syms = comp_symbols(n)
        x_syms = tuple(x_syms)
        pairs = {k: rk for k, rk in self.pair_models.items() if len(k) == 2}
        triplets = {k: rk for k, rk in self.pair_models.items() if len(k) == 3}
        geometry = interp_geometry(self.interp_scheme, x_syms, pair_order=list(pairs))
        pair_terms, weights, diffs = {}, {}, {}
        for pair, rk in pairs.items():
            pair_terms[pair] = rk.numeric_exprs() if numeric else rk.exprs()
            weights[pair], diffs[pair] = geometry[pair]
        # Triplet interaction terms: x_i*x_j*x_k * L0(T), composed OUTSIDE the pair
        # geometry (permutation-symmetric, so orientation never flips them).
        fracs = [1 - sum(x_syms)] + list(x_syms)
        higher_order = sp.S.Zero
        for key, rk in triplets.items():
            term = (rk.numeric_exprs() if numeric else rk.exprs())[0]
            i, j, k = key
            higher_order = higher_order + fracs[i] * fracs[j] * fracs[k] * term
        return build_solution_expressions(
            self.g_ref_exprs,
            pair_terms,
            x_syms=x_syms,
            t=t,
            pair_weights=weights,
            pair_diffs=diffs,
            higher_order_expr=higher_order,
            ideal=self.ideal if ideal is None else ideal,
        )

    def binary_eqs(self) -> dict:
        """The BinaryLiquid ``eqs`` dict — unchanged key names and argument orders.

        Per-order entries ``l{k}``/``h_l{k}``/``s_l{k}`` (+ ``*_lambdified`` taking
        ``(t, L{k}_a, L{k}_b, ...)``), the liquid surface entries, the lambdified
        H/S callables taking ``(x, t, *flat_params)``, and the fitting derivatives
        ``g_prime``/``g_double_prime``.
        """
        if len(self.components) != 2:
            raise ValueError(
                "binary_eqs is defined for 2-component models only; "
                "use expressions() for higher dimensions."
            )
        ((pair, rk),) = self.pair_models.items()
        xb = comp_symbols(2)[0]
        core = self.expressions(x_syms=(xb,), ideal="plain", numeric=False)
        g_liquid, s_liquid, h_liquid = core["g_liquid"], core["s_liquid"], core["h_liquid"]

        eqs: dict = {"ga": self.g_ref_exprs[0], "gb": self.g_ref_exprs[1]}
        for order, l_expr in zip(rk.format.orders, rk.exprs()):
            h_k, s_k = rk.order_h_s(order)
            order_syms = [
                l_sym(f"L{order}_{role}")
                for role in rk.format.order_roles[rk.format.orders.index(order)]
            ]
            eqs[f"l{order}"] = l_expr
            eqs[f"h_l{order}"] = h_k
            eqs[f"s_l{order}"] = s_k
            eqs[f"h_l{order}_lambdified"] = sp.lambdify([t_sym, *order_syms], h_k, modules="numpy")
            eqs[f"s_l{order}_lambdified"] = sp.lambdify([t_sym, *order_syms], s_k, modules="numpy")

        flat_syms = rk.format.symbols()
        g_prime = sp.diff(g_liquid, xb)
        eqs.update(
            {
                "g_ideal": core["g_ideal"],
                "g_xs": core["g_xs"],
                "g_liquid": g_liquid,
                "h_liquid": h_liquid,
                "s_liquid": s_liquid,
                "h_liq_lambdified": sp.lambdify([xb, t_sym, *flat_syms], h_liquid, modules="numpy"),
                "s_liq_lambdified": sp.lambdify([xb, t_sym, *flat_syms], s_liquid, modules="numpy"),
                "g_prime": g_prime,
                "g_double_prime": sp.diff(g_prime, xb),
            }
        )
        return eqs

    # ------------------------------------------------------------------ numeric
    def h_s_grid(self, x_arrays, temp: float) -> tuple[np.ndarray, np.ndarray]:
        """H and S evaluated over composition arrays at one temperature.

        ``x_arrays`` is the length-(n-1) sequence of independent-fraction arrays.
        Uses the 'safe' ideal form so edge/corner grid points are exact. The
        lambdified callables are cached per instance (dropped on pickle).

        The binary pure-regular case evaluates through the legacy numpy arithmetic
        (bit-exact with the retired BinaryLiquid.solid_solution_h_s — the hull
        goldens pin exact simplex sets, which a ~1e-16 float-reassociation can flip).
        """
        if self._binary_regular is not None:
            dh_a, dh_b, ds_a, ds_b, omega = self._binary_regular
            x_arr = np.asarray(x_arrays[0], dtype=float)
            conf_term = np.zeros_like(x_arr)
            interior = (x_arr > 0.0) & (x_arr < 1.0)
            xa = 1 - x_arr
            conf_term[interior] = xa[interior] * np.log(xa[interior]) + x_arr[interior] * np.log(
                x_arr[interior]
            )
            s_conf = -R * conf_term
            s_total = (ds_a * xa + ds_b * x_arr) + s_conf
            h_total = (dh_a * xa + dh_b * x_arr) + omega * xa * x_arr
            return h_total, s_total
        fns = self._grid_fns.get("safe")
        if fns is None:
            x_syms = comp_symbols(len(self.components))
            exprs = self.expressions(x_syms=x_syms, ideal="safe", numeric=True)
            fns = (
                sp.lambdify([*x_syms, t_sym], exprs["h_liquid"], modules="numpy"),
                sp.lambdify([*x_syms, t_sym], exprs["s_liquid"], modules="numpy"),
            )
            self._grid_fns["safe"] = fns
        h_fn, s_fn = fns
        shape = np.shape(x_arrays[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.broadcast_to(h_fn(*x_arrays, temp), shape).astype(float)
            s = np.broadcast_to(s_fn(*x_arrays, temp), shape).astype(float)
        return h, s

    # ------------------------------------------------------------------ plumbing
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_grid_fns"] = {}
        return state

    def __repr__(self) -> str:
        pairs = {
            "-".join(self.components[i] for i in key): rk.format.name
            for key, rk in self.pair_models.items()
        }
        return (
            f"SolutionModel(components={list(self.components)}, pairs={pairs}, "
            f"interp_scheme='{self.interp_scheme}')"
        )


# --------------------------------------------------------------------------------------
# N-component solution-phase expression builder. build_solution_expressions is the one
# core every phase type evaluates through; SolutionModel (below) is its object form.
# --------------------------------------------------------------------------------------


def build_solution_expressions(
    g_ref_exprs,
    pair_terms,
    *,
    x_syms=None,
    t=t_sym,
    pair_weights=None,
    pair_diffs=None,
    higher_order_expr=0,
    ideal="plain",
) -> dict[str, sp.Expr]:
    """Symbolic G/H/S of an n-component solution phase — the shared core under the binary
    and ternary builders (liquid, solid-solution, and any future phase type: the
    per-component reference expressions in ``g_ref_exprs`` are arbitrary G_i(T)).

    Args:
        g_ref_exprs: Length-n sequence of per-component reference G(T) expressions.
        pair_terms: ``{(i, j): (L0_expr, L1_expr, ...)}`` over component-index pairs —
            each value is the pair's per-order Redlich-Kister expression sequence
            (up to L3); order k multiplies ``diff_ij**k``, so the pair orientation
            matters for odd orders. Orders given as exactly ``0`` vanish symbolically.
        x_syms: The n-1 independent composition symbols (default ``comp_symbols(n)``);
            ``x_0 = 1 - sum(x_syms)`` is the derived fraction.
        t: Temperature symbol.
        pair_weights: ``{(i, j): weight}`` per pair (default 1).
        pair_diffs: ``{(i, j): expr}`` composition difference multiplying L1
            (default ``x_i - x_j``; interpolation schemes supply theirs via
            ``interp_geometry``).
        higher_order_expr: Optional >2-body interaction added to g_xs (e.g. the ternary
            ``L_abc * x_a * x_b * x_c``), built by the caller from the same symbols.
        ideal: ``'plain'`` — ``R*t*sum(x_i*log(x_i))``, differentiable everywhere in the
            open simplex (the binary fitting path's g_prime/g_double_prime feed sp.solve);
            ``'safe'`` — per-term ``Piecewise((0, Eq(x_i, 0)), (x_i*log(x_i)))``, exact on
            edge/corner grid points of the closed simplex (the >=3-component surfaces are
            evaluated there). The per-term form scales O(n), unlike a per-region Piecewise.

    Returns:
        ``{'g_ref', 'g_ideal', 'g_xs', 'g_liquid', 's_liquid', 'h_liquid'}`` — symbolic
        only; lambdification stays with the callers.
    """
    n = len(g_ref_exprs)
    if x_syms is None:
        x_syms = comp_symbols(n)
    x_syms = tuple(x_syms)
    if len(x_syms) != n - 1:
        raise ValueError(
            f"Expected {n - 1} composition symbols for {n} components, got {len(x_syms)}."
        )
    fracs = [1 - sum(x_syms)] + list(x_syms)

    g_ref = sum(g * x for g, x in zip(g_ref_exprs, fracs))

    if ideal == "plain":
        g_ideal = R * t * sum(x * sp.log(x) for x in fracs)
    elif ideal == "safe":
        g_ideal = R * t * sum(sp.Piecewise((0, sp.Eq(x, 0)), (x * sp.log(x), True)) for x in fracs)
    else:
        raise ValueError("ideal must be 'plain' or 'safe'.")

    pair_weights = dict(pair_weights or {})
    pair_diffs = dict(pair_diffs or {})
    g_xs = sp.S.Zero
    for (i, j), term_seq in pair_terms.items():
        if len(term_seq) > RK_MAX_ORDER + 1:
            raise ValueError(
                f"Pair ({i}, {j}) carries {len(term_seq)} RK orders; "
                f"the supported maximum is L{RK_MAX_ORDER}."
            )
        weight = pair_weights.get((i, j), 1)
        diff = pair_diffs.get((i, j))
        if diff is None:
            diff = fracs[i] - fracs[j]
        rk_sum = sp.S.Zero
        for order, l_expr in enumerate(term_seq):
            if l_expr == 0:
                continue
            rk_sum = rk_sum + (l_expr * diff**order if order else l_expr)
        g_xs = g_xs + fracs[i] * fracs[j] * weight * rk_sum
    g_xs = g_xs + higher_order_expr

    g_liquid = g_ref + g_ideal + g_xs
    s_liquid = -sp.diff(g_liquid, t)
    h_liquid = g_liquid + t * s_liquid
    return {
        "g_ref": g_ref,
        "g_ideal": g_ideal,
        "g_xs": g_xs,
        "g_liquid": g_liquid,
        "s_liquid": s_liquid,
        "h_liquid": h_liquid,
    }


def interp_geometry(scheme: str, x_syms, pair_order=None) -> dict:
    """Pairwise interpolation weights and composition-difference expressions.

    One home for the linear/Muggianu/Kohler extrapolation geometry over the C(n,2)
    component pairs (previously ternary-only). Muggianu splits the other components'
    total evenly onto the pair's endpoints (which cancels in the difference, so it equals
    linear for this Redlich-Kister order); Kohler renormalizes onto the pair's own
    sub-simplex, guarded against a vanishing pair sum.

    Args:
        scheme: 'linear' | 'muggianu' | 'kohler'.
        x_syms: The n-1 independent composition symbols (``x_0 = 1 - sum`` derived).
        pair_order: Iterable of orientation-sensitive (i, j) component-index pairs
            (``diff = x_i_eff - x_j_eff``). Defaults to all i < j pairs; the ternary
            convention is ``((0, 1), (1, 2), (2, 0))``.

    Returns:
        ``{(i, j): (weight_expr, diff_expr)}``
    """
    x_syms = tuple(x_syms)
    n = len(x_syms) + 1
    fracs = [1 - sum(x_syms)] + list(x_syms)
    if pair_order is None:
        pair_order = [(i, j) for i in range(n) for j in range(i + 1, n)]
    scheme_l = str(scheme).lower()
    geometry = {}
    for i, j in pair_order:
        if scheme_l == "linear":
            diff = fracs[i] - fracs[j]
        elif scheme_l == "muggianu":
            rest_half = (1 - fracs[i] - fracs[j]) / 2
            diff = (fracs[i] + rest_half) - (fracs[j] + rest_half)
        elif scheme_l == "kohler":
            pair_sum = fracs[i] + fracs[j]
            x_i_eff = sp.Piecewise(
                (sp.Rational(1, 2), sp.Eq(pair_sum, 0)), (fracs[i] / pair_sum, True)
            )
            x_j_eff = sp.Piecewise(
                (sp.Rational(1, 2), sp.Eq(pair_sum, 0)), (fracs[j] / pair_sum, True)
            )
            diff = x_i_eff - x_j_eff
        else:
            raise ValueError(
                f"Unsupported interp_type '{scheme}'. Supported: linear, muggianu, kohler"
            )
        geometry[(i, j)] = (sp.S.One, diff)
    return geometry


def _safe_spacegroup(structure) -> tuple[int | None, str | None]:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        return int(sga.get_space_group_number()), str(sga.get_space_group_symbol())
    except Exception as e:
        logger.warning(e)
        return None, None


def _find_gs_polymorph(ref: ComponentRef) -> Phase | None:
    """The DFT ground-state polymorph (step enthalpy of zero)."""
    return next((p for p in ref.polymorphs if p.delta_h == 0), None)


def _find_matching_polymorph(ss_phase: str, ref: ComponentRef) -> Phase | None:
    """The element's polymorph matching an SS phase by spacegroup number.

    An element may carry SEVERAL polymorphs of one spacegroup (Fe has alpha-Fe and delta-Fe, both
    Im-3m/229). Pick the one stable immediately below melting -- the highest transition temperature
    under ``t_fusion`` -- because that is the solid the liquid is actually in equilibrium with, and
    ``build_phases_from_chull`` deletes EVERY polymorph of a covered spacegroup from the line
    compounds. Taking the first match instead referenced Fe's BCC solution to alpha-Fe (dH = 0) and
    dropped delta-Fe from the hull entirely.
    """
    matches = [p for p in ref.polymorphs if p.spacegroup_number == SS_SPACEGROUPS[ss_phase]]
    if not matches:
        return None
    t_melt = ref.t_fusion
    below = [p for p in matches if (p.t_transition or 0) < t_melt] if t_melt else []
    best = max(below or matches, key=lambda p: p.t_transition or 0)
    if best.spacegroup_symbol != SS_SYMBOLS[ss_phase]:
        logger.warning(
            f"spacegroup symbol does not match expected for phase "
            f"'{ss_phase}'. Check for inconsistencies in the omegas file."
        )
    return best


def _make_ground_ref(
    source: str, material_id: str, spacegroup: int, symbol: str, energy_ev_per_atom: float
) -> dict:
    return {
        "source": source,
        "ground_material_id": material_id,
        "ground_spacegroup": spacegroup,
        "ground_symbol": symbol,
        "ground_energy_ev_per_atom": energy_ev_per_atom,
    }


def _make_phase_ref(
    ss_phase: str,
    material_id: str,
    energy_ev_per_atom: float,
    delta_h_jmol: float,
    delta_s_jmol_k: float,
    spacegroup: int | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> dict:
    """One phase reference. ``delta_h_jmol``/``delta_s_jmol_k`` are CUMULATIVE above the element's
    ground state, never single-transition steps -- see the module docstring's REFERENCE FRAME note.
    ``source`` marks a non-primary provenance (e.g. the omegas-file fallback) when set.
    """
    ref = {
        "material_id": material_id,
        "spacegroup": spacegroup if spacegroup is not None else SS_SPACEGROUPS.get(ss_phase, -1),
        "symbol": symbol if symbol is not None else SS_SYMBOLS.get(ss_phase, "unknown"),
        "energy_ev_per_atom": energy_ev_per_atom,
        "delta_h_jmol": delta_h_jmol,
        "delta_s_jmol_k": delta_s_jmol_k,
    }
    if source is not None:
        ref["source"] = source
    return ref


def _compute_solid_ss_entropy(
    el: str,
    ss_phase: str,
    delta_h_jmol: float,
    phase_refs_all: dict[str, dict[str, dict]],
    ref: ComponentRef,
) -> float:
    """Cumulative entropy for a solid SS phase via the stepwise S = sum(dH_i / T_i) formula.

    ``phase_refs_all`` must hold EVERY phase resolved for ``el``, not just the ones resolved so
    far: the sum runs over the transitions BELOW this phase, so a lower-temperature phase that
    has not been reached yet would silently drop out. ``ALL_SS_PHASES`` iterates BCC before HCP,
    so with a partially-filled map Ti's BCC entropy missed its 180 K HCP step entirely and left
    the Ti edge 420 K low under ``from_dft_entries``. Callers resolve all enthalpies first, then
    call this. Transition temperatures come from the element's own ladder.
    """
    phase = _find_matching_polymorph(ss_phase, ref)
    t_this = phase.t_transition if phase else None
    if t_this is None or t_this <= 0:
        return 0.0

    candidates: list[tuple[float, float]] = []
    for other_phase, refs in phase_refs_all.items():
        if el not in refs or other_phase == ss_phase:
            continue
        p = _find_matching_polymorph(other_phase, ref)
        t_p = p.t_transition if p else None
        if t_p is not None and 0 < t_p <= t_this:
            candidates.append((refs[el]["delta_h_jmol"], t_p))

    candidates.append((delta_h_jmol, t_this))
    candidates.sort(key=lambda x: x[1])

    s_accum = 0.0
    prev_h = 0.0
    for dh, t in candidates:
        s_accum += (dh - prev_h) / t
        prev_h = dh
    return s_accum


def _build_ss_models_from_refs(
    components: list[str], ground_refs: dict[str, dict], phase_refs: dict[str, dict[str, dict]]
) -> dict[str, dict]:
    """Merge ground_refs and phase_refs into the ss_models skeleton."""
    ss_models: dict[str, dict] = {}
    for ss_phase, el_refs in phase_refs.items():
        refs = {el: dict(ground_refs[el]) for el in components}
        for el, pr in el_refs.items():
            refs[el].update(pr)
        ss_models[ss_phase] = {"refs": refs}
    return ss_models


def pair_keys(components: list[str]) -> list[str]:
    """Alphabetized pair keys ('A-B') for all C(n,2) component pairs."""
    comps = list(components)
    return [
        "-".join(sorted((comps[i], comps[j])))
        for i in range(len(comps))
        for j in range(i + 1, len(comps))
    ]


def _package_ss_models(
    ss_models: dict[str, dict],
    components: list[str],
    omega_data: dict[str, dict[str, float]],
    ref_mode: str,
) -> dict[str, dict]:
    """Attach keyed omega and delta H/S maps to every phase in ss_models in-place.

    A phase needs a reference for EVERY component and an omega for EVERY pair; a phase
    missing either is skipped with a warning and stays edge-only, regardless of component
    count (n == 2 included) — the semantics of the retired three-edge merge
    (build_ternary_ss_models), now unified across component counts. A genuinely uncovered
    SYSTEM (every candidate phase dropped) surfaces as an empty dict from
    load_solid_solution_models, never an exception.
    """
    keys = pair_keys(components)
    sys_name = "-".join(components)
    uncovered = []
    for ss_phase, model_dict in ss_models.items():
        if "refs" not in model_dict:
            continue
        omega_block = omega_data.get(ss_phase, {})
        missing_pairs = [pk for pk in keys if pk not in omega_block]
        refs = model_dict["refs"]
        if not all("delta_h_jmol" in refs[el] for el in components):
            # The resolver found this phase for only some endpoints (e.g. an element with
            # no matching polymorph) — a continuous solution needs every reference.
            missing = [el for el in components if "delta_h_jmol" not in refs[el]]
            logger.warning(
                f"SS phase '{ss_phase}' has no reference for {missing} in "
                f"'{sys_name}'; skipping this phase."
            )
            uncovered.append(ss_phase)
            continue
        if missing_pairs:
            logger.warning(
                f"SS phase '{ss_phase}' lacks an omega for pair(s) "
                f"{missing_pairs} of {sys_name}; it stays edge-only."
            )
            uncovered.append(ss_phase)
            continue
        model_dict.update(
            {
                "ref_mode": ref_mode,
                "omega": {pk: float(omega_block[pk]) * EV_ATOM_TO_J_MOL for pk in keys},
                "delta_h": {el: float(refs[el]["delta_h_jmol"]) for el in components},
                "delta_s": {el: float(refs[el]["delta_s_jmol_k"]) for el in components},
            }
        )
    for ss_phase in uncovered:
        ss_models.pop(ss_phase)
    return ss_models


def _compound_components(components) -> list[str]:
    """The multi-element (compound) formulas among ``components`` (parse failures skip)."""
    found = []
    for c in components:
        try:
            if len(Composition(c).elements) > 1:
                found.append(c)
        except Exception:
            continue
    return found


def _resolve_refs_legacy(
    data: dict, components: list[str], component_data: dict[str, ComponentRef]
) -> tuple[dict, dict]:
    """Ground/phase refs from the flat omegas file's ``elements`` block (legacy format).

    Entropy is recomputed stepwise from the element's transition temperatures.
    """
    compounds = _compound_components(components)
    if compounds:
        raise ValueError(
            f"The omegas-file reference source only supports elemental components (got "
            f"{compounds}); it will never carry compound reference states."
        )
    element_blocks: dict[str, dict[str, float]] = data["elements"]

    stable_pure: dict[str, float] = {}
    for el in components:
        candidates = [float(block[el]) for block in element_blocks.values() if el in block]
        if not candidates:
            raise KeyError(f"Could not find pure-element references for '{el}' in omegas file.")
        stable_pure[el] = min(candidates)

    ground_refs = {
        el: _make_ground_ref(
            source="from_omegas_file",
            material_id="legacy",
            spacegroup=-1,
            symbol="legacy",
            energy_ev_per_atom=stable_pure[el],
        )
        for el in components
    }

    # Pass 1 -- every enthalpy. Pass 2 -- entropies, which sum over the transitions BELOW each
    # phase and so need the complete map (see _compute_solid_ss_entropy).
    phase_refs: dict[str, dict[str, dict]] = {}
    for ss_phase, phase_block in element_blocks.items():
        if not all(el in phase_block for el in components):
            continue
        phase_refs[ss_phase] = {}
        for el in components:
            ss_e = float(phase_block[el])
            phase_refs[ss_phase][el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id="legacy",
                energy_ev_per_atom=ss_e,
                delta_h_jmol=(ss_e - stable_pure[el]) * EV_ATOM_TO_J_MOL,
                delta_s_jmol_k=0.0,
            )

    for ss_phase, refs in phase_refs.items():
        for el in components:
            ref = component_data.get(el)
            if ref is not None and _find_matching_polymorph(ss_phase, ref) is not None:
                refs[el]["delta_s_jmol_k"] = _compute_solid_ss_entropy(
                    el, ss_phase, refs[el]["delta_h_jmol"], phase_refs, ref
                )

    return ground_refs, phase_refs


def _resolve_refs_cache(
    components: list[str],
    entries: list[ComputedStructureEntry],
    component_data: dict[str, ComponentRef] | None = None,
) -> tuple[dict, dict]:
    """Ground/phase refs from pymatgen ComputedStructureEntries.

    Enthalpies come from DFT (cumulative above the lowest-energy pure entry). Entropies come from
    the element's own transition ladder via ``_compute_solid_ss_entropy``, exactly as the
    omegas-file resolver does -- a hard-coded ``delta_s = 0`` left the SS endpoints flat in T while
    the liquid reference still carried ``sum(dH_i / T_i)``, which depressed BOTH endpoints of every
    element with any solid transition (Hf-Ti came out 2292 / 1338 K against 2506 / 1943 K).
    ``component_data`` is optional only so the retired two-argument call sites keep working; without
    it the entropies fall back to zero and that defect returns.
    """
    compounds = _compound_components(components)
    if compounds:
        raise NotImplementedError(
            f"The DFT-entries reference source does not yet support compound components "
            f"(got {compounds}); support is planned for a future release."
        )
    ground_refs: dict[str, dict] = {}
    phase_refs: dict[str, dict[str, dict]] = {}

    for el in components:
        pure_entries = [e for e in entries if e.composition.reduced_formula == el]
        if not pure_entries:
            raise RuntimeError(f"No pure-element entries found for '{el}'.")

        ground = min(pure_entries, key=lambda e: float(e.energy_per_atom))
        ground_sg, ground_symbol = _safe_spacegroup(getattr(ground, "structure", None))

        ground_refs[el] = _make_ground_ref(
            source="from_dft_entries",
            material_id=str(getattr(ground, "entry_id", "unknown")),
            spacegroup=int(ground_sg) if ground_sg is not None else -1,
            symbol=ground_symbol or "unknown",
            energy_ev_per_atom=float(ground.energy_per_atom),
        )

        for ss_phase in ALL_SS_PHASES:
            phase_entries = [
                e
                for e in pure_entries
                if _safe_spacegroup(getattr(e, "structure", None))[0] == SS_SPACEGROUPS[ss_phase]
            ]
            if not phase_entries:
                logger.warning(
                    f"No {ss_phase} (spacegroup {SS_SPACEGROUPS[ss_phase]}) pure entry "
                    f"found for '{el}' in local cache."
                )
                continue

            best = min(phase_entries, key=lambda e: float(e.energy_per_atom))
            sg, symbol = _safe_spacegroup(getattr(best, "structure", None))
            delta_h_jmol = (
                float(best.energy_per_atom) - float(ground.energy_per_atom)
            ) * EV_ATOM_TO_J_MOL

            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id=str(getattr(best, "entry_id", "unknown")),
                energy_ev_per_atom=float(best.energy_per_atom),
                delta_h_jmol=delta_h_jmol,
                delta_s_jmol_k=0.0,
                spacegroup=int(sg) if sg is not None else SS_SPACEGROUPS[ss_phase],
                symbol=symbol or SS_SYMBOLS[ss_phase],
            )

    # Second pass: cumulative transition entropy off the element's own ladder, same as the
    # omegas-file resolver. Deferred until every enthalpy is in hand because the sum runs over
    # the transitions BELOW each phase; zero stands only where the element has no ladder to read.
    for ss_phase, refs in phase_refs.items():
        for el, phase_ref in refs.items():
            ref = (component_data or {}).get(el)
            if ref is not None and _find_matching_polymorph(ss_phase, ref) is not None:
                phase_ref["delta_s_jmol_k"] = _compute_solid_ss_entropy(
                    el, ss_phase, phase_ref["delta_h_jmol"], phase_refs, ref
                )

    return ground_refs, phase_refs


def _resolve_refs_db(
    components: list[str], component_data: dict[str, ComponentRef]
) -> tuple[dict, dict]:
    """Ground/phase refs straight from the unary element database (ComponentRef ladders)."""
    compounds = _compound_components(components)
    if compounds:
        raise ValueError(
            f"The unary-db reference source only supports elemental components (got "
            f"{compounds}); it will never carry compound reference states."
        )
    ground_refs: dict[str, dict] = {}
    phase_refs: dict[str, dict[str, dict]] = {}

    for el in components:
        ref = component_data[el]
        p_ground = _find_gs_polymorph(ref)

        ground_refs[el] = _make_ground_ref(
            source="from_unary_db",
            material_id=(p_ground.material_id or "unknown") if p_ground else "unknown",
            spacegroup=(
                p_ground.spacegroup_number if p_ground.spacegroup_number is not None else -1
            )
            if p_ground
            else -1,
            symbol=(p_ground.spacegroup_symbol or "unknown") if p_ground else "unknown",
            energy_ev_per_atom=(
                (p_ground.enthalpy if p_ground.enthalpy is not None else -EV_ATOM_TO_J_MOL)
                / EV_ATOM_TO_J_MOL
            )
            if p_ground
            else -1.0,
        )

        for ss_phase in ALL_SS_PHASES:
            p_poly = _find_matching_polymorph(ss_phase, ref)
            if p_poly is None:
                continue
            # CUMULATIVE above the ground state (Phase.enthalpy/entropy), not the single-transition
            # steps (Phase.delta_h/delta_s) -- see the module docstring's REFERENCE FRAME note. The
            # elemental polymorph line compounds this phase replaces carry cumulative values, and
            # the hull compares the two directly.
            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id=p_poly.material_id or "unknown",
                energy_ev_per_atom=(
                    p_poly.enthalpy if p_poly.enthalpy is not None else -EV_ATOM_TO_J_MOL
                )
                / EV_ATOM_TO_J_MOL,
                delta_h_jmol=p_poly.enthalpy if p_poly.enthalpy is not None else -1.0,
                delta_s_jmol_k=p_poly.entropy if p_poly.entropy is not None else 0.0,
                spacegroup=(
                    p_poly.spacegroup_number
                    if p_poly.spacegroup_number is not None
                    else SS_SPACEGROUPS[ss_phase]
                ),
                symbol=p_poly.spacegroup_symbol or SS_SYMBOLS[ss_phase],
            )

        # Tier B: structures the ladder lacks come from the builder-baked
        # lattice_stabilities block (cumulative above the same anchor) — this
        # supersedes the runtime omegas fallback for from_unary_db, whose
        # conflict guard now lives in the BUILDER as a hard error.
        for ss_phase in ALL_SS_PHASES:
            if el in phase_refs.get(ss_phase, {}):
                continue
            p_ls = next(
                (
                    p
                    for p in ref.lattice_stabilities
                    if p.spacegroup_number == SS_SPACEGROUPS[ss_phase]
                ),
                None,
            )
            if p_ls is None or p_ls.enthalpy is None:
                continue
            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id=p_ls.material_id or "lattice_stability",
                # Raw DFT energy when the builder recorded one (the runtime
                # fallback published the raw omegas eV/atom; pins froze that).
                energy_ev_per_atom=(
                    p_ls.energy_per_atom_ev
                    if p_ls.energy_per_atom_ev is not None
                    else p_ls.enthalpy / EV_ATOM_TO_J_MOL
                ),
                delta_h_jmol=p_ls.enthalpy,
                # Normally 0.0 — a structure with no stability field has no
                # transition temperature, so S = sum(dH_i / T_i) is undefined and
                # the convention returns 0.0. The builder's scoped metastable-
                # entropy exception emits a NEGATIVE SGTE value on a few HCP
                # entries; publish what it wrote rather than re-flattening it.
                # A negative delta_s only enlarges the (s_liq - dS) denominator of
                # the endpoint melting reconstruction, so the element's own melting
                # point is preserved by construction. These phases still have no
                # transition temperature, so _ordered_solid_steps skips them and
                # the liquid reconciliation is untouched (which from_unary_db does
                # not run anyway — see load_solid_solution_models).
                delta_s_jmol_k=p_ls.entropy if p_ls.entropy is not None else 0.0,
                spacegroup=p_ls.spacegroup_number,
                symbol=p_ls.spacegroup_symbol or SS_SYMBOLS[ss_phase],
                source=f"from_unary_db:lattice_stability ({p_ls.source})",
            )

    return ground_refs, phase_refs


def _reconciled_liquid_ref(
    ref: ComponentRef, ordered_cum_steps: list[tuple[float, float]]
) -> tuple[float, float]:
    """(h_liq, s_liq) from resolver-produced cumulative solid enthalpies plus fusion.

    Builds a synthetic ladder whose step enthalpies are the increments between consecutive
    resolver cumulative values and reuses ``ComponentRef.liquid_ref_from_solids`` on it, so the
    stepwise S = sum(dH_i / T_i) + H_fus / T_melt formula lives in exactly one place.
    """
    liquid = ref.liquid
    if liquid is None or liquid.delta_h is None or not liquid.t_transition:
        logger.warning(
            f"Missing fusion enthalpy or melt temperature for '{ref.symbol}'. H_liq/S_liq set to 0."
        )
        return 0.0, 0.0

    from dataclasses import replace

    solids = []
    prev = 0.0
    for cum_dh, t_trans in ordered_cum_steps:
        solids.append(Phase(phase_type="solid", delta_h=cum_dh - prev, t_transition=t_trans))
        prev = cum_dh
    synthetic = ComponentRef(ref.symbol, solids + [replace(liquid, enthalpy=prev + liquid.delta_h)])
    return synthetic.liquid_ref_from_solids(solids)


def _ordered_solid_steps(
    ref: ComponentRef, el: str, phase_refs: dict[str, dict[str, dict]]
) -> list[tuple[float, float]]:
    """(cumulative delta_h_jmol, T_transition) per SS phase below the melt, by increasing T."""
    t_melt = ref.t_fusion
    candidates: list[tuple[float, float]] = []
    for ss_phase, refs in phase_refs.items():
        if el not in refs:
            continue
        # _find_matching_polymorph, not ComponentRef.solid_phase: an element may carry
        # SEVERAL polymorphs of one spacegroup (alpha-Fe and delta-Fe are both Im-3m/229),
        # and solid_phase returns the FIRST. For Fe that is alpha-Fe at T_tr = 0, which the
        # filter below then discards -- so delta-Fe, the phase actually stable just under
        # the melt, vanished from the reconciled ladder entirely. Every other SS site in
        # this module already resolves through _find_matching_polymorph, whose docstring
        # names this exact case; this was the last first-match holdout.
        p = _find_matching_polymorph(ss_phase, ref) if ss_phase in SS_SPACEGROUPS else None
        t_trans = p.t_transition if p else None
        if t_trans is None or t_trans <= 0 or t_trans >= t_melt:
            continue
        candidates.append((refs[el]["delta_h_jmol"], t_trans))
    return sorted(candidates, key=lambda x: x[1])


def _apply_omegas_fallback(
    components: list[str],
    phase_refs: dict[str, dict[str, dict]],
    omegas_elements: dict[str, dict[str, float]],
    ref_mode: str,
) -> None:
    """Fill phases the primary reference source lacks from the omegas file's ``elements`` block.

    An element that never *exhibits* a structure has no polymorph entry for it -- Cr is never FCC,
    Ni never BCC -- so the primary resolver produces no reference and ``_package_ss_models`` drops
    the whole phase, blocking a solid solution the omegas file could support. The omegas
    ``elements`` block carries a DFT energy for those metastable structures; this ports it in.

    Placement (``phase_refs`` is updated in place):
      * Anchored -- if the element already has a phase that the omegas block also lists, take the
        lowest-omegas-energy such phase as the anchor and shift the omegas ladder onto the primary
        source's zero. Keeps primary-source phases authoritative.
      * Legacy convention -- if there is no shared phase (As, Sb, Bi have no BCC/FCC/HCP
        polymorph at all in these modes), fall back to ``_resolve_refs_legacy``'s own zero: the
        lowest omegas SS energy. Required, not optional -- refusing here would leave Bi-Sb
        and As-Sb with no solid solution. (``from_unary_db`` no longer calls this function at
        all: the tiered unary DB bakes the same energies into its ``lattice_stabilities`` block
        at build time, where Ta/Ag/Co are anchored on their corrected experimental ground
        states and the negative-value conflict guard is a builder hard error.)
      * Entropy is 0.0, as ``_resolve_refs_legacy`` does for a structure the element never
        exhibits. Such a phase has no transition temperature, so ``_ordered_solid_steps`` skips it
        and the liquid reconciliation is untouched.
      * A negative anchored enthalpy is REFUSED, not clamped: it means the omegas DFT ranks this
        structure BELOW the primary source's ground state, i.e. the two sources disagree about the
        ground state (Ag, Co, Mn). Inventing a value there moves the element's melting point --
        Co by 182 K -- so the phase stays edge-only and the conflict is logged.
    """
    if not omegas_elements:
        return
    for el in components:
        available = [p for p in ALL_SS_PHASES if el in omegas_elements.get(p, {})]
        missing = [p for p in available if el not in phase_refs.get(p, {})]
        if not missing:
            continue
        shared = [p for p in available if el in phase_refs.get(p, {})]
        if shared:
            anchor = min(shared, key=lambda p: omegas_elements[p][el])
            offset = (
                phase_refs[anchor][el]["delta_h_jmol"]
                - omegas_elements[anchor][el] * EV_ATOM_TO_J_MOL
            )
            basis = f"anchored on {anchor}"
        else:
            offset = -min(omegas_elements[p][el] for p in available) * EV_ATOM_TO_J_MOL
            basis = "omegas-file zero (no shared phase)"
        for ss_phase in missing:
            energy_ev = float(omegas_elements[ss_phase][el])
            delta_h_jmol = energy_ev * EV_ATOM_TO_J_MOL + offset
            if delta_h_jmol < -1.0:
                logger.warning(
                    f"omegas fallback for '{el}' {ss_phase} would sit "
                    f"{delta_h_jmol:.1f} J/mol BELOW the {ref_mode} ground state -- the two "
                    f"sources disagree about {el}'s ground state. Leaving {ss_phase} edge-only; "
                    f"check the {el} entry in phase_transitions.json."
                )
                continue
            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id="omegas_fallback",
                energy_ev_per_atom=energy_ev,
                delta_h_jmol=delta_h_jmol,
                delta_s_jmol_k=0.0,
                source=f"{ref_mode}+omegas_fallback ({basis})",
            )


def _reconcile_liquid_refs(
    components: list[str],
    component_data: dict[str, ComponentRef],
    phase_refs: dict[str, dict[str, dict]],
) -> None:
    """Replace each element's liquid reference so it derives from the SS solid ladder.

    Updates ``component_data`` in place with ``with_liquid_ref`` copies -- shared registry
    ``Phase`` instances are never mutated.

    Only meaningful when the resolver's zero DIFFERS from the stored ladder's (``from_omegas_file``
    anchors on the lowest omegas SS energy; ``from_dft_entries`` on the lowest cached DFT entry).
    ``from_unary_db`` refs are the stored ladder, so re-deriving from a SUBSET of it can only lose
    information -- see the caller in ``load_solid_solution_models``.
    """
    for el in components:
        ref = component_data[el]
        if ref.liquid is None or not ref.t_fusion:
            logger.warning(f"No melt temperature found for '{el}'. Skipping reconciliation.")
            continue
        steps = _ordered_solid_steps(ref, el, phase_refs)
        h_liq, s_liq = _reconciled_liquid_ref(ref, steps)
        component_data[el] = ref.with_liquid_ref(h_liq, s_liq)


def load_solid_solution_models(
    components: list[str],
    component_data: dict[str, ComponentRef],
    entries: list[ComputedStructureEntry] | None = None,
    omegas_path: Path | str | None = None,
    ref_mode: str = DEFAULT_REF_MODE,
) -> dict[str, dict]:
    """Build per-phase solid-solution models for an n-component system (n >= 2).

    One resolution pass covers any component count: the resolvers are per-element and
    the omegas file is looked up per pair, so the retired three-edge merge
    (``build_ternary_ss_models``) is subsumed — cross-edge consistency is automatic
    because every element is resolved exactly once.

    Args:
        components: Alphabetized component list (two or more elements).
        component_data: ``{symbol: ComponentRef}`` per-system copies (e.g. from
            ``phase.UNARY.component_data``). Reconciled liquid references are installed
            in place via registry-safe copies.
        entries: ComputedStructureEntry list (required for ``ref_mode='from_dft_entries'``).
        omegas_path: Omegas JSON path; defaults to ``config.omegas_file``.
        ref_mode: 'from_omegas_file' | 'from_dft_entries' | 'from_unary_db'.

    Returns:
        ``{phase_name: model_dict}`` -- see the module docstring for the keyed schema.
    """
    omegas_path = Path(omegas_path) if omegas_path is not None else Path(config.omegas_file)
    data = json.loads(omegas_path.read_text(encoding="utf-8"))
    keys = pair_keys(components)
    omega_data: dict[str, dict[str, float]] = data.get("omegas", {})

    if ref_mode == "from_omegas_file":
        available_phases = {
            phase
            for phase, block in omega_data.items()
            if all(pk in block for pk in keys)
            and phase in data.get("elements", {})
            and all(el in data["elements"][phase] for el in components)
        }
        if not available_phases:
            # Coverage gate: nothing usable in the omegas file for this system — behave
            # exactly like SS-off. Returning here skips _resolve_refs_legacy and
            # _reconcile_liquid_refs, so component_data is never touched.
            return {}
        ground_refs, phase_refs = _resolve_refs_legacy(data, components, component_data)
        phase_refs = {k: v for k, v in phase_refs.items() if k in available_phases}

    elif ref_mode == "from_dft_entries":
        ground_refs, phase_refs = _resolve_refs_cache(components, entries or [], component_data)
        _apply_omegas_fallback(components, phase_refs, data.get("elements", {}), ref_mode)

    elif ref_mode == "from_unary_db":
        # No runtime omegas fallback here: the builder bakes the same energies
        # (with the same anchoring math) into the unary DB's lattice_stabilities
        # block, and its negative-value guard is a build-time hard error.
        ground_refs, phase_refs = _resolve_refs_db(components, component_data)

    else:
        raise ValueError(
            "ref_mode must be one of: from_dft_entries, from_omegas_file, from_unary_db."
        )

    ss_models = _build_ss_models_from_refs(components, ground_refs, phase_refs)
    ss_models = _package_ss_models(ss_models, components, omega_data, ref_mode)
    if not ss_models:
        # Coverage gate: no SS phase survived packaging (e.g. the omegas file carries no
        # interaction parameter for this system's pair) — skip reconciliation entirely so
        # component_data is byte-identical to the SS-off path. Same predicate for every ref_mode.
        return {}

    # Reconcile ONLY when the resolver's zero differs from the stored ladder's. 'from_unary_db'
    # refs ARE the stored ladder (cumulative, same ground state), so the liquid reference already
    # shares their frame; re-deriving it from the SS-spacegroup SUBSET would drop every polymorph
    # whose spacegroup is not BCC/FCC/HCP -- beta-Mn (sg 213) is such a step, and losing it left
    # Mn's endpoint ~118 K high. The other two frames need the re-derivation and get it.
    #
    # _package_ss_models never mutates phase_refs/component_data, so for a covered system this
    # always runs with the full, unfiltered phase_refs.
    if ref_mode != "from_unary_db":
        _reconcile_liquid_refs(components, component_data, phase_refs)
    return ss_models
