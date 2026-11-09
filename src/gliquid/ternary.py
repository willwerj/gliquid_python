"""
Authors: Abrar Rauf, Joshua Wilwerth
Description: This module contains the classes for ternary interpolation and ternary phase diagram plotting.
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""

import functools
import itertools
import logging
import operator
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Element
from scipy.spatial import Delaunay

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as cfg
import gliquid.imputation.core as imp_core
import gliquid.mpds as mpds
import gliquid.plotting.export as plot_export
import gliquid.plotting.liquidus_fields as liquidus_fields
import gliquid.plotting.ternary_surface as ternary_surface
import gliquid.solution as solution
from gliquid.binary import BinaryLiquid, hull_phase_fields
from gliquid.hsx import HSX, lower_convex_hull
from gliquid.liquidus_sheets import LiquidusSheets, TstarField
from gliquid.phase import (
    EV_ATOM_TO_J_MOL,
    UNARY,
    polymorph_ladder,
    resolve_component_order,
    validate_and_format_system,
)
from gliquid.plotting.binary_tx import _plot_tx_inputs, add_corner_note, plot_tx
from gliquid.plotting.style import build_phase_color_map, mape_color
from gliquid.solution import DEFAULT_TAU, RKPolyExp, SolutionModel, t_sym

logger = logging.getLogger(__name__)

#: Rungs of headroom every window ceiling carries above the temperature it was sized from.
WINDOW_MARGIN_SLICES = 1

#: Bracket and resolution of the liquidus pre-screen, in kelvin.
PRESCREEN_CEILING_K = 20000.0
PRESCREEN_TOL_K = 0.05

#: Temperature a predicted binary edge's window starts at, below its lower melting point.
PREDICTED_EDGE_FLOOR_MARGIN_K = 50.0

#: Temperature-grid increment, K, per hull_mode when T_incr is None.
T_INCR_BY_MODE = {"gtx": 25, "hsx": 10}
#: Most, K, a node's hull liquidus may lie above the liquidus sheets before the liquidus is
#: drawn without phase fields.
SHEET_GUARD_K = 1.0


def ordered_binary_systems(elements):
    # given a ternary system, returns the ordered binary systems
    binary_pairs = []
    for i in range(len(elements)):
        next_element = elements[(i + 1) % len(elements)]
        binary_pairs.append(f"{elements[i]}-{next_element}")

    return binary_pairs


def invert_substrings(input_string):
    substring1, substring2 = input_string.split("-")
    inverted_string = f"{substring2}-{substring1}"
    return inverted_string


def orient_to_edges(mapping: dict, edges: list[str], flip=None) -> dict:
    """``mapping`` with each key re-keyed onto the names ``edges`` use.

    A pair named in the inverse order (``'Al-Si'`` for the edge ``'Si-Al'``) is re-keyed to the
    edge and its value passed through ``flip`` when one is given. A triplet key naming the edges'
    three elements in any order is re-keyed to them in the edges' order. Keys that are already
    edges and keys naming other elements are kept as given.

    Raises:
        ValueError: an interaction is named in two orders.
    """
    elements = [edge.split("-")[0] for edge in edges]
    oriented = {}
    for key, value in mapping.items():
        if key.count("-") == 1 and key not in edges and invert_substrings(key) in edges:
            key = invert_substrings(key)
            value = flip(value) if flip is not None else value
        elif key.count("-") == 2 and sorted(key.split("-")) == sorted(elements):
            key = "-".join(elements)
        if key in oriented:
            raise ValueError(f"The interaction {key} is given in both orders; give each once.")
        oriented[key] = value
    return oriented


def cartesian_to_ternary(df):
    xs = df.iloc[:, 0].values
    ys = df.iloc[:, 1].values
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
        trans_coord = np.dot(np.array([x, y]), unitvec)
        new_xs.append(trans_coord[0])
        new_ys.append(trans_coord[1])

    df.iloc[:, 0] = new_xs
    df.iloc[:, 1] = new_ys

    return df


def ternary_to_cartesian(x_A, x_B):
    x = x_A + 0.5 * x_B
    y = np.sqrt(3) / 2 * x_B
    return x, y


@dataclass(frozen=True)
class _EdgeRows:
    """Rows of liquid nodes near each side of the composition triangle."""

    #: Rows :class:`TernaryLiquidInterpolation` adds at ``delta / ratio**k`` (k = 1..rows) from
    #: each side.
    rows: int = 3
    #: Ratio of the distances of successive rows from their side (:func:`generate_comp_grid`).
    ratio: float = 3.0
    #: Coarsest grid step, mole fraction, that gets the rows; a grid with solid solutions gets none.
    max_delta: float = 0.05


EDGE_ROW = _EdgeRows()


def generate_comp_grid(delta=0.025, atol=1e-6, edge_rows=0):
    """The ternary composition grid of step ``delta`` as ``{"A", "B", "C"}`` fraction arrays.

    With ``edge_rows`` > 0, each side also gets rows at ``delta / EDGE_ROW.ratio**k`` from it
    (k = 1..edge_rows). A row holds its two ends, the midpoints between the grid lines it crosses
    and its crossings with the other sides' rows, so the grid is the same under every order of
    the components.
    """
    incr = np.arange(0, 1 + delta, delta)
    A, B, C = np.meshgrid(incr, incr, incr)
    x_A = A.flatten()
    x_B = B.flatten()
    x_C = C.flatten()
    valid_indices = np.where(np.isclose(x_A + x_B + x_C, 1, atol=atol))
    x_A = x_A[valid_indices]
    x_B = x_B[valid_indices]
    x_C = x_C[valid_indices]
    decimal_places = max(2, -int(np.log10(atol)))
    x_A = np.round(x_A, decimal_places)
    x_B = np.round(x_B, decimal_places)
    x_C = np.round(x_C, decimal_places)
    if not edge_rows:
        return {"A": x_A, "B": x_B, "C": x_C}
    gaps = delta / EDGE_ROW.ratio ** np.arange(1, edge_rows + 1)
    mids = np.round(incr[incr <= 1 - delta / 2] + delta / 2, 12)
    rows = [np.column_stack([x_A, x_B, x_C])]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for gap in gaps:
            ends = [0.0, 1 - gap]
            along = np.unique(np.concatenate([ends, mids - gap / 2, gaps, 1 - gap - gaps]))
            along = along[(along >= 0) & (along <= 1 - gap)]
            row = np.zeros((len(along), 3))
            row[:, i] = gap
            row[:, j] = along
            row[:, k] = 1 - gap - along
            rows.append(row)
    points = np.vstack(rows)
    points = points / points.sum(axis=1, keepdims=True)
    _, first = np.unique(np.round(points, 13), axis=0, return_index=True)
    points = points[np.sort(first)]
    return {"A": points[:, 0], "B": points[:, 1], "C": points[:, 2]}


def _user_entropies(bl) -> dict[tuple[str, float], float]:
    """The entropies, J/mol-atom/K, of ``bl``'s user-defined solid phases that carry one, keyed
    by each phase's name and its fraction of ``bl.components[1]`` rounded to 4 places."""
    return {
        (p.name, round(p.fraction_in(bl.components)[0], 4)): float(p.entropy)
        for p in getattr(bl, "phases", None) or []
        if p.non_dft == "user" and p.composition is not None and p.entropy
    }


def _edge_user_entries(bl) -> list[dict]:
    """The user-defined entries on ``bl.dft_ch`` as tagged dicts (see :func:`_user_entry_dict`),
    each with the entropy of the solid phase ``bl`` holds under its name at its composition."""
    entropies = _user_entropies(bl)
    return [
        _user_entry_dict(
            bl.dft_ch,
            e,
            entropies.get(
                (
                    hull_phase_fields(e)["name"],
                    round(api.entry_frac_along(bl.dft_ch, e, bl.components)[0], 4),
                ),
                0.0,
            ),
        )
        for e in bl.dft_ch.all_entries
        if cache.non_dft_entry_kind(e) == "user"
    ]


def _entry_entropy(entry) -> float:
    """``entry``'s ``data[cache.ENTROPY_KEY]``, J/mol-atom/K; 0 when it has none."""
    return float((getattr(entry, "data", None) or {}).get(cache.ENTROPY_KEY, 0.0))


def _user_entry_dict(hull: PhaseDiagram, entry, entropy: float = 0.0) -> dict:
    """``entry``, a user-defined entry on ``hull``, as a tagged dict whose
    ``data['h_form_j_per_mol']`` is its formation enthalpy on ``hull``.

    A nonzero ``entropy`` (J/mol-atom/K) is stored as ``data[cache.ENTROPY_KEY]``.
    """
    original = api.entry_original(entry)
    data = getattr(original, "data", None) or {}
    extra = (
        {imp_core.REPLACES_KEY: list(data[imp_core.REPLACES_KEY])}
        if data.get(imp_core.REPLACES_KEY)
        else {}
    )
    if entropy:
        extra[cache.ENTROPY_KEY] = float(entropy)
    entry_id = getattr(original, "entry_id", None)
    return api.user_entry(
        original.composition,
        original.energy,
        cache.non_dft_entry_label(original),
        entry_id=None if entry_id is None else str(entry_id),
        h_form_j_per_mol=float(hull.get_form_energy_per_atom(entry)) * EV_ATOM_TO_J_MOL,
        **extra,
    ).as_dict()


#: Least |determinant| of a triangle's edge vectors, mole fraction squared, that places a point
#: on it.
FLAT_TRIANGLE = 1e-12


def _barycentric(corners, points) -> np.ndarray:
    """(..., 3) weights placing ``points`` (..., 2) on the triangles ``corners`` (..., 3, 2),
    broadcast together; NaN where a triangle is flat (:data:`FLAT_TRIANGLE`)."""
    origin = corners[..., 2, :]
    u, v, r = corners[..., 0, :] - origin, corners[..., 1, :] - origin, points - origin
    det = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    det = np.where(np.abs(det) > FLAT_TRIANGLE, det, np.nan)
    wa = (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]) / det
    wb = (u[..., 0] * r[..., 1] - u[..., 1] * r[..., 0]) / det
    return np.stack([wa, wb, 1 - wa - wb], axis=-1)


def _containing(corners, points) -> np.ndarray:
    """Per point of ``points`` (m, 2), the index of the triangle of ``corners`` (n, 3, 2) holding
    it whose least barycentric weight is largest; -1 where none holds it."""
    if not len(corners):
        return np.full(len(points), -1)
    least = _barycentric(corners[None], points[:, None]).min(axis=2)
    least = np.where(np.isnan(least), -np.inf, least)
    best = least.argmax(axis=1)
    return np.where(least[np.arange(len(points)), best] >= -1e-9, best, -1)


def _spans_plane(points) -> bool:
    """Whether ``points`` (n, 2) hold three that do not lie on one line."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return len(points) >= 3 and np.linalg.matrix_rank(points[1:] - points[0]) == 2


@functools.cache
def _triples(count: int) -> np.ndarray:
    """(t, 3) indices of every choice of three of ``count`` items."""
    return np.array(list(itertools.combinations(range(count), 3)), dtype=int).reshape(-1, 3)


def _plane_gap(ti, node_rows, plane_node, plane_rows, plane_weights):
    """``gap(temps, nodes)``: the Gibbs energy of ``node_rows[nodes]`` at ``temps`` (K) minus the
    lowest of their planes there, every Gibbs energy from ``ti._row_gibbs``.

    Args:
        ti: the :class:`TernaryLiquidInterpolation` whose ``proc_df`` rows these are.
        node_rows: (n,) ``proc_df`` rows of the nodes.
        plane_node: (p,) index into ``node_rows`` of the node each plane belongs to.
        plane_rows: (p, 3) ``proc_df`` rows each plane passes through.
        plane_weights: (p, 3) weights placing its node on each plane.
    """
    pairs, corner = np.unique(
        np.column_stack([np.repeat(plane_node, 3), np.ravel(plane_rows)]),
        axis=0,
        return_inverse=True,
    )
    corner = corner.reshape(-1, 3)

    def gap(temps, nodes):
        at = np.zeros(len(node_rows))
        at[nodes] = temps
        wanted = np.zeros(len(node_rows), dtype=bool)
        wanted[nodes] = True
        live = wanted[pairs[:, 0]]
        gibbs = np.zeros(len(pairs))
        gibbs[live] = ti._row_gibbs(at[pairs[live, 0]], pairs[live, 1])
        used = wanted[plane_node]
        heights = (plane_weights[used] * gibbs[corner[used]]).sum(axis=1)
        lowest = np.full(len(node_rows), np.inf)
        np.minimum.at(lowest, plane_node[used], heights)
        return ti._row_gibbs(temps, node_rows[nodes]) - lowest[nodes]

    return gap


@dataclass(frozen=True)
class _NodeLiquidus:
    """Solution of a node's liquidus on the lower hulls."""

    #: Temperatures, K, a node liquidus is searched between.
    search_k: tuple[float, float] = (1.0, 1.0e5)
    #: Width, K, of the bracket a node liquidus is bisected to against planes through other points.
    tol_k: float = 1e-3
    #: Width, K, of the bracket a node liquidus is bisected to on whole lower hulls, and how far a
    #: checked node liquidus may sit from the hulls of the check.
    hull_tol_k: float = 0.5
    #: Spacing, K, of the lower hulls every hull_mode 'hsx' node liquidus is checked against.
    check_k: float = 25.0
    #: Rungs of the lower-hull scan that solves a node liquidus with no facet to solve it on.
    scan_rungs: int = 48


NODE_LIQUIDUS = _NodeLiquidus()


def _bisect(gap, nodes, low, high) -> np.ndarray:
    """Per node of ``nodes``, the temperature (K) between ``low`` and ``high`` where
    ``gap(temps, nodes)`` changes sign: bisected to ``NODE_LIQUIDUS.tol_k``, then the zero of the
    line through the gap at the two ends (the midpoint where that zero is not between them)."""
    low, high = np.array(low, dtype=float), np.array(high, dtype=float)
    at_low, at_high = gap(low, nodes), gap(high, nodes)
    while True:
        live = np.flatnonzero(high - low > NODE_LIQUIDUS.tol_k)
        if not len(live):
            break
        mid = 0.5 * (low[live] + high[live])
        at_mid = gap(mid, nodes[live])
        same = np.sign(at_mid) == np.sign(at_low[live])
        low[live[same]], at_low[live[same]] = mid[same], at_mid[same]
        high[live[~same]], at_high[live[~same]] = mid[~same], at_mid[~same]
    with np.errstate(divide="ignore", invalid="ignore"):
        zero = low + at_low / (at_low - at_high) * (high - low)
    return np.where((zero >= low) & (zero <= high), zero, 0.5 * (low + high))


class TernaryLiquidInterpolation:
    """Ternary liquid surface interpolated from three binary-edge mixing models.

    Fields mirror ``BinaryLiquid`` where they carry the same information:
    ``components`` (three elements, sorted), ``xs_mix`` (per-edge
    ``RKPolyExp`` models — the ternary counterpart of the binary's single
    model), ``param_format``, ``tau``, ``interp_scheme`` (linear/Muggianu/Kohler
    pair geometry). Construct either with an explicit per-edge ``xs_mix`` dict or
    from three fitted ``BinaryLiquid`` systems via ``from_binaries``.

    ``imputed_entries`` holds tagged imputed ``ComputedEntry`` dicts that join the solid side
    of the ternary hull; ``from_binaries(..., include_imputed=True)`` collects them off the
    three edges. ``include_imputed=True`` also places the imputed entries the store holds for
    this ternary system and for its three binary edges. :meth:`impute_phase` adds one from a
    composition and a temperature.

    ``user_entries`` holds user-defined ``ComputedEntry`` dicts (see
    :func:`gliquid.api.user_entry`), each carrying ``data['h_form_j_per_mol']`` and optionally
    an entropy in ``data[cache.ENTROPY_KEY]``. They join the solid side of every hull this
    object builds, re-anchored onto its elemental references, in place of the DFT entries they
    list under ``data['replaces']``. An entry with an entropy is placed even where it sits above
    the 0 K hull, since the entropy can bring it onto the hull at temperature. ``from_binaries``
    collects them off the three edges.

    ``excluded_entries`` holds DFT entry ids left off the solid side of every hull this object
    builds, compared as :func:`gliquid.cache.canonical_entry_id` spells them.
    ``from_binaries`` collects the three edges' ``removed_entry_ids``.
    """

    HULL_COMP_DECIMALS = 12
    """Decimals the hull rounds every composition to (see :meth:`hull_table`)."""

    _DERIVED_STATE = (
        "hsx_df",
        "surface_df",
        "tern_mp_df",
        "proc_df",
        "_proc_arrays",
        "liquid_tstar",
        "df_Tgroups",
        "equil_df_list",
        "T_grid",
        "conds",
        "phase_names",
        "color_map",
        "sys_name",
        "ref_data",
    )

    @cfg._removed_keywords({"ternary_l0": "an 'A-B-C' entry in xs_mix"})
    def __init__(
        self,
        components: list[str],
        data_dir: cache.CacheBackend | Path | str | None = None,
        *,
        delta: float = 0.025,
        interp_scheme: str = "linear",
        param_format: str = "comb-exp",
        tau: float = DEFAULT_TAU,
        xs_mix: dict | None = None,
        fit_or_pred: dict | None = None,
        solid_solutions: bool | None = None,
        ss_kwargs: dict | None = None,
        temp_slider=(0, 0),
        T_incr: int | None = None,
        hull_mode: str = "hsx",
        order: str | list | None = None,
        imputed_entries: list[dict] | None = None,
        temp_range: list | tuple | None = None,
        include_imputed: bool = False,
        user_entries: list[dict] | None = None,
        excluded_entries: list[str] | None = None,
    ):
        validate_and_format_system(list(components))  # compound components raise NIE here
        # Components kept as given (order=None == 'given'); mixing keys follow the resolved order's
        # cyclic edges.
        self.components = resolve_component_order(
            order if order is not None else "given", components
        )
        self.binary_systems = ordered_binary_systems(self.components)
        # A cache backend, threaded to ``api.get_dft_convexhull``; the ternary cache is flat inside
        # it.
        self.data_dir = cache.resolve_backend(data_dir if data_dir is not None else cfg.cache_dir)
        self.delta = delta
        self.interp_scheme = interp_scheme
        self.param_format = param_format
        # exp(-T/tau) decay constant for the 'combined'/'comb-exp' mixing forms.
        self.tau = tau
        # 'fit' or 'pred' per binary system, keyed by the cyclic ``binary_systems`` names
        self.fit_or_pred = orient_to_edges(fit_or_pred or {}, self.binary_systems)

        # Excess models keyed by the cyclic pair names, plus an optional 'A-B-C' triplet in
        # component order. Plain lists coerce via (param_format, tau) for pairs and 'regular' for
        # the triplet; an inverse-order pair is re-keyed with odd orders flipped.
        def _coerce(key, val):
            if isinstance(val, RKPolyExp):
                return val
            if key.count("-") == 2:
                return RKPolyExp("regular", list(val))
            return RKPolyExp(self.param_format, list(val), tau=self.tau)

        self.xs_mix: dict[str, RKPolyExp] = orient_to_edges(
            {key: _coerce(key, val) for key, val in (xs_mix or {}).items()},
            self.binary_systems,
            flip=RKPolyExp.swapped,
        )
        self.ternary_meta = {}
        # Tagged imputed entry dicts, re-anchored onto the ternary hull by get_ternary_form_en.
        self.imputed_entries = list(imputed_entries or [])
        # Also place the store's imputed entries for the ternary and its edges (see
        # all_imputed_entries).
        self.include_imputed = bool(include_imputed)
        # User-defined entry dicts placed on the solid side of every hull (see _solid_hull).
        self.user_entries = list(user_entries or [])
        # Canonical DFT entry ids left off the solid side of every hull (see _solid_hull).
        self.excluded_entries = sorted(
            {cache.canonical_entry_id(i) for i in excluded_entries or ()} - {None}
        )
        self._liquid_model_cache = None
        self._dft_hulls: dict[tuple, PhaseDiagram] = {}  # DFT hull per component tuple
        self._edge_entry_energies: dict[str, float] = {}  # binary edges' energies, by entry id
        # Solid-solution support (None follows config.solid_solutions): one n-ary
        # load_solid_solution_models pass, then a SolutionModel surface per phase.
        self.solid_solutions = bool(
            cfg.solid_solutions if solid_solutions is None else solid_solutions
        )
        edge_rows = (
            EDGE_ROW.rows if self.delta <= EDGE_ROW.max_delta and not self.solid_solutions else 0
        )
        self.comp_grid = generate_comp_grid(self.delta, edge_rows=edge_rows)
        self.ss_kwargs = ss_kwargs
        self.ss_models = {}
        # Hull evaluation: extra temperature margin, temperature-grid increment and hull algorithm.
        self.temp_slider = list(temp_slider)
        # The window T_grid spans, in kelvin, filled by _init_sys. A pinned temp_range end is kept;
        # a None end is derived.
        self._temp_range_arg = list(temp_range) if temp_range is not None else None
        self.temp_range: list[float] = []
        self._window: tuple[float, float] | None = None
        # 'hsx' (default): one (x0, x1, S, H) lower hull with analytic facet temperatures.
        # 'gtx': per-temperature (x0, x1, G) lower hulls on the T_incr grid.
        self.hull_mode = hull_mode
        if self.hull_mode not in ("gtx", "hsx"):
            raise ValueError(
                "hull_mode must be 'gtx' (per-temperature slicing) or "
                "'hsx' (one N-D hull through gliquid.hsx.HSX)."
            )
        self.T_incr = T_INCR_BY_MODE[hull_mode] if T_incr is None else T_incr
        self._n_simplex_vertices = 3  # gtx facets are triangles; the hsx path sets 4

    @classmethod
    @cfg._removed_keywords({"components": "order=[...]"})
    def from_binaries(
        cls, binaries, *, include_imputed: bool = False, **kwargs
    ) -> "TernaryLiquidInterpolation":
        """Construct from three parameterized ``BinaryLiquid`` edge systems.

        Derives the three components (alphabetical; ``order=`` takes any spec
        ``resolve_component_order`` accepts), re-orients each edge's mixing model onto the cyclic
        ``binary_systems`` convention (odd RK orders flip when a stored pair maps onto
        an inverted cyclic edge), and inherits ``param_format``/``tau`` (which must
        agree across the three systems). ``solid_solutions`` defaults on when any
        binary carries ss_models; further constructor kwargs pass through.

        ``include_imputed`` collects every tagged imputed entry off the three edges'
        ``dft_ch`` hulls into ``imputed_entries`` (an explicit ``imputed_entries=`` wins) and
        sets the constructor's ``include_imputed``, which places the store's imputed entries
        for the ternary and its edges too.

        Every user-defined entry (:func:`gliquid.cache.non_dft_entry_kind`) on the three edges'
        ``dft_ch`` hulls is collected into ``user_entries`` (an explicit ``user_entries=``
        wins), carrying its formation enthalpy on that edge hull, any ``replaces`` it holds, and
        the entropy its edge's matching solid phase holds in ``phases`` (see
        :data:`gliquid.cache.ENTROPY_KEY`). The DFT entry ids each edge lists in
        ``removed_entry_ids`` are collected into ``excluded_entries`` (an explicit
        ``excluded_entries=`` wins).

        ``fit_or_pred`` names each edge ``'fit'`` when its binary carries a digitized liquidus
        (``digitized_liq``) and ``'pred'`` otherwise; an explicit ``fit_or_pred=`` wins.
        """
        binaries = list(binaries)
        if len(binaries) != 3:
            raise ValueError(
                f"from_binaries needs exactly 3 BinaryLiquid systems, got {len(binaries)}."
            )
        spanned = {el for bl in binaries for el in bl.components}
        if len(spanned) != 3:
            raise ValueError(
                f"The three binaries must span exactly 3 elements, got {sorted(spanned)}."
            )
        elements = sorted(spanned)
        by_pair = {frozenset(bl.components): bl for bl in binaries}
        if len(by_pair) != 3:
            raise ValueError("The three binaries must cover three distinct element pairs.")
        formats = {bl.param_format for bl in binaries}
        if len(formats) != 1:
            raise ValueError(
                f"Binaries carry mixed param formats {sorted(formats)}; "
                f"re-fit or convert to one formalism first."
            )
        taus = {bl.tau for bl in binaries}
        if len(taus) != 1:
            raise ValueError(f"Binaries carry mixed tau values {sorted(taus)}.")

        xs_mix = {}
        for edge in ordered_binary_systems(elements):
            a, b = edge.split("-")
            bl = by_pair[frozenset((a, b))]
            rk = RKPolyExp(bl.xs_mix.format, bl.xs_mix.values, tau=bl.xs_mix.tau)
            if list(bl.components) != [a, b]:  # inverted cyclic edge: odd orders flip
                rk = rk.swapped()
            xs_mix[edge] = rk
        kwargs.setdefault("param_format", next(iter(formats)))
        kwargs.setdefault("tau", next(iter(taus)))
        kwargs.setdefault(
            "fit_or_pred",
            {
                edge: "fit"
                if getattr(by_pair[frozenset(edge.split("-"))], "digitized_liq", None)
                else "pred"
                for edge in xs_mix
            },
        )
        kwargs.setdefault("solid_solutions", any(bl.ss_models for bl in binaries))
        if include_imputed:
            collected = []
            for bl in binaries:
                for entry in bl.dft_ch.all_entries:
                    as_dict = entry.as_dict()
                    if cache.is_imputed_entry_dict(as_dict):
                        collected.append(as_dict)
            kwargs.setdefault("imputed_entries", collected)
            kwargs.setdefault("include_imputed", True)
        kwargs.setdefault("user_entries", [d for bl in binaries for d in _edge_user_entries(bl)])
        kwargs.setdefault(
            "excluded_entries",
            [i for bl in binaries for i in getattr(bl, "removed_entry_ids", None) or ()],
        )
        ternary = cls(elements, xs_mix=xs_mix, **kwargs)
        ternary._edge_entry_energies = cache.entry_energies_by_id(
            entry for bl in binaries for entry in bl.dft_ch.all_entries
        )
        return ternary

    def with_component_order(self, order) -> "TernaryLiquidInterpolation":
        """A copy of this system re-framed onto ``order`` (any spec
        ``resolve_component_order`` accepts). Returns ``self`` when the order already matches.

        Mirrors ``BinaryLiquid.with_component_order``: copies of the mixing models and
        ``fit_or_pred`` are re-keyed onto the new cyclic ``binary_systems`` as the constructor
        re-keys them (see :func:`orient_to_edges`); every other setting, the entry lists and the
        edge entry energies carry over. Interpolated/derived state (hsx_df, ref_data, ss_models,
        ternary_meta) is NOT carried over — it re-derives on interpolate().
        """
        order = resolve_component_order(order, self.components)
        if list(order) == list(self.components):
            return self
        reordered = TernaryLiquidInterpolation(
            list(order),
            self.data_dir,
            delta=self.delta,
            interp_scheme=self.interp_scheme,
            param_format=self.param_format,
            tau=self.tau,
            xs_mix={
                key: RKPolyExp(rk.format, rk.values, tau=rk.tau) for key, rk in self.xs_mix.items()
            },
            fit_or_pred=self.fit_or_pred,
            solid_solutions=self.solid_solutions,
            ss_kwargs=self.ss_kwargs,
            temp_slider=self.temp_slider,
            T_incr=self.T_incr,
            hull_mode=self.hull_mode,
            order="given",
            imputed_entries=self.imputed_entries,
            temp_range=self._temp_range_arg,
            include_imputed=self.include_imputed,
            user_entries=self.user_entries,
            excluded_entries=self.excluded_entries,
        )
        reordered._edge_entry_energies = dict(self._edge_entry_energies)
        return reordered

    def init_ref_data(self):
        # initialize reference data for fusion enthalpies and entropies
        tern_enthalpy = np.array([UNARY[el].h_liq for el in self.components])
        tern_entropy = np.array([UNARY[el].s_liq for el in self.components])
        tern_temp = np.array([UNARY[el].t_fusion for el in self.components])
        self.ref_data = {"H": tern_enthalpy, "S": tern_entropy, "T": tern_temp}

    def liquid_model(self) -> SolutionModel:
        """The liquid's ``SolutionModel``: the three cyclic edge pairs plus the optional
        'A-B-C' triplet, unary references from the registry, and ``interp_scheme``'s geometry.
        Built once per interpolation.

        Raises:
            ValueError: ``xs_mix`` does not cover all three edges.
        """
        if self._liquid_model_cache is not None:
            return self._liquid_model_cache
        self.init_ref_data()

        if not all(sys in self.xs_mix for sys in self.binary_systems):
            raise ValueError(
                "Mixing models for the binary systems are incomplete. Provide a "
                "per-edge 'xs_mix' dict or construct via from_binaries()."
            )

        refs = [h - t_sym * s for h, s in zip(self.ref_data["H"], self.ref_data["S"])]
        interactions = {sys: self.xs_mix[sys] for sys in self.binary_systems}
        trip_key = "-".join(self.components)
        if trip_key in self.xs_mix:
            interactions[trip_key] = self.xs_mix[trip_key]
        self._liquid_model_cache = SolutionModel(
            tuple(self.components),
            refs,
            interactions,
            interp_scheme=self.interp_scheme,
            ideal="safe",
        )
        return self._liquid_model_cache

    def linear_tstar(self, x0, x1) -> np.ndarray:
        """The x-weighted component melting point (K) at independent fractions ``x0``, ``x1`` of
        ``components[1]`` and ``components[2]``: the liquidus sheets' T* where no liquid node has
        a liquidus."""
        if not hasattr(self, "ref_data"):
            self.init_ref_data()
        tm = np.asarray(self.ref_data["T"], dtype=float)
        x0, x1 = np.asarray(x0, dtype=float), np.asarray(x1, dtype=float)
        return (1 - x0 - x1) * tm[0] + x0 * tm[1] + x1 * tm[2]

    def liquid_tstar_field(self) -> TstarField:
        """T*(x) over all compositions: ``linear_tstar``, brought to ``liquid_tstar`` at the
        liquid nodes once :meth:`process_data` has found their liquidus."""
        if not hasattr(self, "ref_data"):
            self.init_ref_data()
        nodes = getattr(self, "liquid_tstar", None)
        if nodes is None:
            return TstarField(self.ref_data["T"])
        return TstarField(
            self.ref_data["T"], nodes[["x0", "x1"]].to_numpy(float), nodes["tstar"].to_numpy(float)
        )

    def liquid_gibbs(self, x, temp: float):
        """Liquid Gibbs energy (J/mol-atom) at ``x`` and ``temp``, H and S taken at ``temp``: the
        energy every node liquidus of the processed hull is solved with.

        Args:
            x: independent fractions of ``components[1:]`` -- one ``(2,)`` point or ``(m, 2)``
                points.
            temp: temperature, K.

        Returns:
            float for one point, else an ``(m,)`` array.
        """
        points = np.atleast_2d(np.asarray(x, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            h, s = self.liquid_model().h_s_grid((points[:, 0], points[:, 1]), float(temp))
        g = np.asarray(h, dtype=float) - float(temp) * np.asarray(s, dtype=float)
        return float(g[0]) if np.ndim(x) == 1 else g

    def imputation_params(self) -> dict:
        """The liquid model an imputed energy is read off, JSON-serializable.

        Returns:
            dict: ``param_format``, ``tau``, ``interp_scheme``, ``delta``, ``solid_solutions``,
            ``edges`` (each edge's parameters keyed by its alphabetical pair, odd orders
            oriented to that pair), ``triplet`` (the 'A-B-C' parameters, or ``None``) and
            ``liquid_gibbs`` (``'exact'`` when an edge model is nonlinear in T, so that
            ``liquid_gibbs`` takes H and S at the temperature asked for, else ``None``).
        """
        edges = {}
        for key, rk in self.xs_mix.items():
            pair = key.split("-")
            if len(pair) != 2:
                continue
            ordered = sorted(pair)
            oriented = rk if pair == ordered else rk.swapped()
            edges["-".join(ordered)] = [float(v) for v in oriented.values]
        triplet = self.xs_mix.get("-".join(self.components))
        return {
            "param_format": str(self.param_format),
            "tau": float(self.tau),
            "interp_scheme": str(self.interp_scheme),
            "delta": float(self.delta),
            "solid_solutions": bool(self.solid_solutions),
            "edges": dict(sorted(edges.items())),
            "triplet": None if triplet is None else [float(v) for v in triplet.values],
            "liquid_gibbs": None
            if all(rk.is_linear_in_t() for key, rk in self.xs_mix.items() if key.count("-") == 1)
            else "exact",
        }

    def interpolate_liquid_surface(self):
        """Fill ``hsx_df`` with the liquid's H and S on the composition grid, taken at the
        components' mean melting point."""
        x_B, x_C = self.comp_grid["B"], self.comp_grid["C"]

        self._liquid_model_cache = None
        model = self.liquid_model()

        with np.errstate(divide="ignore", invalid="ignore"):
            h_vals_mesh, s_vals_mesh = model.h_s_grid((x_B, x_C), np.mean(self.ref_data["T"]))

        # Replace inf and nan values with finite values if needed
        H = np.where(np.isfinite(h_vals_mesh), h_vals_mesh, 0).flatten()
        S = np.where(np.isfinite(s_vals_mesh), s_vals_mesh, 0).flatten()

        logger.info(f"Composition map: x0: {self.components[1]}, x1: {self.components[2]}")
        self.hsx_df = pd.DataFrame({"x0": x_B, "x1": x_C, "S": S, "H": H})
        self.hsx_df["Phase Name"] = "L"

    def append_solid_solution_surfaces(self):
        """Append per-phase solid-solution Gibbs surfaces to the ternary HSX grid.

        Evaluates each ternary SS phase's H(x)/S(x) on the same composition grid the liquid
        uses (one ``SolutionModel.from_ss_model`` surface per phase), so continuous solid
        phases enter the ternary lower hull exactly like the liquid does (dense rows
        sharing one phase name). Covers the full simplex including the binary edges. A phase's
        H and S do not depend on temperature, so a row's ``H - T*S`` is its Gibbs energy at
        every temperature.

        Models come from ONE n-ary ``load_solid_solution_models`` pass over the three
        elements, with ``ss_kwargs`` (``ref_mode`` defaults to ``config.ss_ref_mode``; for
        ``ref_mode='from_dft_entries'`` the caller must supply ``entries``).
        """
        if not self.ss_models:
            self.ss_models = solution.load_solid_solution_models(
                self.components,
                UNARY.component_data(self.components),
                **{"ref_mode": cfg.ss_ref_mode, **(self.ss_kwargs or {})},
            )

        x_B, x_C = self.comp_grid["B"], self.comp_grid["C"]
        tm_mean = np.mean(self.ref_data["T"])

        ss_frames = []
        for phase_name, ss_model in self.ss_models.items():
            surface = SolutionModel.from_ss_model(self.components, ss_model, self.interp_scheme)
            with np.errstate(divide="ignore", invalid="ignore"):
                h_vals, s_vals = surface.h_s_grid((x_B, x_C), tm_mean)
            frame = pd.DataFrame(
                {
                    "x0": x_B,
                    "x1": x_C,
                    "S": np.where(np.isfinite(s_vals), s_vals, 0),
                    "H": np.where(np.isfinite(h_vals), h_vals, 0),
                }
            )
            frame["Phase Name"] = phase_name
            ss_frames.append(frame)

        if ss_frames:
            self.hsx_df = pd.concat([self.hsx_df, *ss_frames], ignore_index=True)

    def all_imputed_entries(self) -> list[dict]:
        """``imputed_entries``, plus -- with ``include_imputed`` -- the imputed entries this
        object's store holds for the ternary system and for each of its three binary edges.

        One entry per ``entry_id``: an explicit entry wins over a stored one, and the ternary
        record over an edge record.
        """
        entries = list(self.imputed_entries)
        if not self.include_imputed:
            return entries
        seen = {e.get("entry_id") for e in entries}
        systems = [list(self.components)] + [pair.split("-") for pair in self.binary_systems]
        for system in systems:
            for entry in api.read_imputed_entries(
                system, api.DEFAULT_DFT_TYPE, data_dir=self.data_dir
            ):
                if entry.get("entry_id") not in seen:
                    seen.add(entry.get("entry_id"))
                    entries.append(entry)
        return entries

    def _warn_on_edge_energy_drift(self, pdia: PhaseDiagram) -> None:
        """Log a warning when an entry the binary edges and ``pdia`` share carries two energies."""
        drift = cache.entry_energy_drift(
            self._edge_entry_energies, cache.entry_energies_by_id(pdia.all_entries)
        )
        if drift:
            entry_id, edge_energy, hull_energy = drift[0]
            logger.warning(
                "%d entries shared by the binary edges and the %s DFT hull carry different "
                "energies (largest: %s, %+.0f J/mol-atom in the hull); the ternary edges will "
                "not follow the binary fits.",
                len(drift),
                "-".join(self.components),
                entry_id,
                (hull_energy - edge_energy) * EV_ATOM_TO_J_MOL,
            )

    def _solid_hull(self, components, imputed_entries, exclude=()) -> tuple[PhaseDiagram, list]:
        """The DFT hull with ``user_entries`` and ``imputed_entries`` re-anchored onto it, and
        the injected imputed entries.

        Entries whose ids are in ``exclude`` or ``excluded_entries``, or that a user-defined or
        imputed entry replaces (``data['replaces']``), are left out. Ids compare as
        :func:`gliquid.cache.canonical_entry_id` spells them, so a classic and a new-API id of
        one entry match.
        """
        # The instance's own data_dir is used, flat inside it.
        key = tuple(components)
        if key not in self._dft_hulls:
            self._dft_hulls[key], _ = api.get_dft_convexhull(
                components, api.DEFAULT_DFT_TYPE, data_dir=self.data_dir
            )
            self._warn_on_edge_energy_drift(self._dft_hulls[key])
        pdia = self._dft_hulls[key]
        excluded = {cache.canonical_entry_id(i) for i in exclude} | set(self.excluded_entries)
        excluded |= imp_core.replaced_entry_ids(imputed_entries)
        if self.user_entries:
            excluded |= imp_core.replaced_entry_ids(self.user_entries)
            pdia, _ = imp_core.hull_with_reanchored_entries(pdia, self.user_entries)
        if excluded:
            pdia = PhaseDiagram(
                [
                    e
                    for e in pdia.all_entries
                    if cache.canonical_entry_id(e.entry_id) not in excluded
                ],
                elements=list(pdia.elements),
            )
        if not imputed_entries:
            return pdia, []
        return imp_core.hull_with_reanchored_entries(pdia, imputed_entries)

    def solid_hull(self, imputed_entries: list[dict] | None = None, exclude=()) -> PhaseDiagram:
        """The 0 K solid hull: the DFT entries plus imputed entries re-anchored onto them.

        Args:
            imputed_entries: the imputed entries to place. ``None`` places
                :meth:`all_imputed_entries`.
            exclude: DFT entry ids to leave out, beside those an imputed entry replaces.
        """
        entries = self.all_imputed_entries() if imputed_entries is None else imputed_entries
        return self._solid_hull(list(self.components), list(entries), exclude)[0]

    def edge_hull(self, edge) -> PhaseDiagram:
        """The solid hull on one binary edge: the entries of :meth:`solid_hull` made of that
        edge's two elements only.

        Args:
            edge: Two of this ternary's components, as ``'A-B'`` or a pair, in either order.

        Raises:
            ValueError: ``edge`` does not name two distinct components of this ternary.
        """
        pair = edge.split("-") if isinstance(edge, str) else list(edge)
        if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(self.components):
            raise ValueError(
                f"edge must name two distinct components of {self.components}, got {edge!r}."
            )
        hull = self.solid_hull()
        return PhaseDiagram(
            [
                e
                for e in hull.all_entries
                if {str(el) for el in e.composition.elements} <= set(pair)
            ],
            elements=[Element(el) for el in pair],
        )

    def get_ternary_form_en(self, sys, imputed_entries: list[dict] | None = None, exclude=()):
        """Solid side of the ternary hull: DFT stable entries PLUS the elemental polymorph ladders.

        Mirrors :func:`binary.build_phases_from_chull` — a stable hull entry sitting at a
        pure-element vertex is REPLACED by that element's polymorph ladder, whose ground state
        is the same point (H = 0, S = 0) and whose higher-temperature polymorphs then compete
        on the hull. Without them the liquid crosses the *extrapolated* ground state, so a pure
        corner melts at ``h_liq / s_liq`` instead of ``t_fusion``.

        Polymorphs that are the end member of a modelled solid-solution surface
        (``ComponentRef.ss_polymorphs``) are excluded (the ternary counterpart of the binary's
        ``exclude_spacegroups``), so a structure never appears as both an SS surface and a
        fixed-composition line compound.

        ``imputed_entries`` are re-anchored onto this hull and compete on it exactly as a DFT
        phase does; ``ternary_meta['n_imputed_entries']`` and ``['n_imputed_stable']`` record
        how many were injected and how many reached the hull, and
        ``ternary_meta['imputed_phase_names']`` the phase names of those that reached it.
        ``ternary_meta['user_phase_names']`` names the ``user_entries`` placed on the solid side:
        those on the 0 K hull and those carrying an entropy (``data[cache.ENTROPY_KEY]``), which
        are placed even above it. A row's ``S`` is that entropy, J/mol-atom/K, and 0 for every
        other phase.

        ``imputed_entries`` places those entries instead of :meth:`all_imputed_entries`, and
        ``exclude`` leaves those DFT entry ids out; either leaves ``ternary_meta`` untouched.

        Raises:
            ValueError: an injected entry dict carries no ``data['h_form_j_per_mol']``.
        """
        components = list(sys)
        entries = self.all_imputed_entries() if imputed_entries is None else imputed_entries
        pdia, injected = self._solid_hull(components, list(entries), exclude)
        meta = {}
        stable_ids = {str(e.entry_id) for e in pdia.stable_entries}
        meta["n_imputed_entries"] = len(injected)
        meta["n_imputed_stable"] = sum(1 for e in injected if str(e.entry_id) in stable_ids)
        meta["imputed_phase_names"] = sorted(
            {api.hull_entry_name(e) for e in injected if str(e.entry_id) in stable_ids}
        )
        user_ids = {str(e.get("entry_id")) for e in self.user_entries}
        entropy_bearing = [
            e
            for e in pdia.all_entries
            if str(e.entry_id) in user_ids
            and str(e.entry_id) not in stable_ids
            and _entry_entropy(e)
        ]
        meta["user_phase_names"] = sorted(
            {api.hull_entry_name(e) for e in pdia.stable_entries if str(e.entry_id) in user_ids}
            | {api.hull_entry_name(e) for e in entropy_bearing}
        )

        meta["n_ternary_compounds"] = sum(
            1 for e in pdia.stable_entries if len(api.entry_original(e).composition.elements) == 3
        )

        # ss_models is filled by append_solid_solution_surfaces; empty when solid_solutions is off.
        exclude_spacegroups = {solution.SS_SPACEGROUPS[p] for p in self.ss_models}
        component_data = UNARY.component_data(components)

        # Corner ladders first, so the hull-entry loop below knows which vertices they own.
        ladder_rows = []
        claimed_corners = {}  # element -> its (x0, x1) vertex on the components[1:] axes
        for i, comp in enumerate(components):
            polymorphs = polymorph_ladder(component_data[comp], exclude_spacegroups)
            if not polymorphs:
                # Every polymorph of this element is covered by an SS surface: the DFT elemental
                # entry stays.
                continue
            corner = tuple(1.0 if j == i else 0.0 for j in range(1, len(components)))
            claimed_corners[comp] = corner
            for p in polymorphs:
                ladder_rows.append(
                    {
                        "x0": corner[0],
                        "x1": corner[1],
                        "S": p.entropy or 0.0,
                        "H": p.enthalpy or 0.0,
                        # One name per polymorph; the groupby below keeps the lowest H per name.
                        "Phase Name": p.name or f"{comp} (sg {p.spacegroup_number})",
                    }
                )

        hull_rows = []
        deepest = []
        for entry in list(pdia.stable_entries) + entropy_bearing:
            form_en = pdia.get_form_energy_per_atom(entry) * EV_ATOM_TO_J_MOL
            if str(entry.entry_id) in stable_ids:
                deepest.append(form_en)
            frac = tuple(api.entry_frac_along(pdia, entry, components))
            if any(np.allclose(frac, corner, atol=1e-9) for corner in claimed_corners.values()):
                continue  # a polymorph ladder owns this vertex
            hull_rows.append(
                {
                    "x0": frac[0],
                    "x1": frac[1],
                    "S": _entry_entropy(entry),
                    "H": form_en,
                    "Phase Name": api.hull_entry_name(entry),
                }
            )

        # Over all stable entries, so the depth is the DFT hull's.
        meta["deepest_formation_energy"] = min(deepest)

        tern_mp_df = pd.DataFrame(
            hull_rows + ladder_rows, columns=["x0", "x1", "S", "H", "Phase Name"]
        )
        tern_mp_df = tern_mp_df.loc[tern_mp_df.groupby("Phase Name")["H"].idxmin()]

        if imputed_entries is None and not exclude:
            self.ternary_meta.update(meta)
        return tern_mp_df

    def hull_table(self, imputed_entries: list[dict] | None = None, exclude=()) -> pd.DataFrame:
        """The H-S-x rows the hull is built from, ``x0``/``x1`` rounded to
        :attr:`HULL_COMP_DECIMALS`. Interpolates first when needed.

        Args:
            imputed_entries: place these imputed entries on the solid side instead of the ones
                ``hsx_df`` was built with. ``None`` with no ``exclude`` returns a copy of
                ``hsx_df``.
            exclude: DFT entry ids to leave off the solid side.
        """
        if not hasattr(self, "hsx_df"):
            self.interpolate()
        if imputed_entries is None and not exclude:
            table = self.hsx_df.copy()
        else:
            entries = self.all_imputed_entries() if imputed_entries is None else imputed_entries
            solids = self.get_ternary_form_en(
                self.components, imputed_entries=list(entries), exclude=exclude
            )
            table = pd.concat([self.surface_df, solids], ignore_index=True)
            table = table.drop_duplicates().reset_index(drop=True)
        table["x0"] = table["x0"].round(self.HULL_COMP_DECIMALS)
        table["x1"] = table["x1"].round(self.HULL_COMP_DECIMALS)
        return table

    def imputed_phase_names(self) -> list[str]:
        """Names of the imputed phases on the solid hull, sorted. Interpolates first when
        needed."""
        if not hasattr(self, "hsx_df"):
            self.interpolate()
        return list(self.ternary_meta.get("imputed_phase_names", []))

    def user_phase_names(self) -> list[str]:
        """Names of the user-defined phases on the solid hull, sorted. Interpolates first when
        needed."""
        if not hasattr(self, "hsx_df"):
            self.interpolate()
        return list(self.ternary_meta.get("user_phase_names", []))

    def reset_derived_state(self) -> None:
        """Drop everything interpolation and processing derived from the model's inputs, so
        the next ``interpolate`` / ``process_data`` rebuilds it."""
        for name in self._DERIVED_STATE:
            self.__dict__.pop(name, None)
        self.temp_range = []
        self._window = None
        self.ternary_meta = {}
        self._liquid_model_cache = None

    def impute_phase(self, composition, temperature: float, **kwargs):
        """Impute a solid phase that leaves this system's hull at ``temperature`` (K).

        See :func:`gliquid.imputation.ternary_imputation.impute_ternary_phase` for the arguments.
        """
        from gliquid.imputation.ternary_imputation import impute_ternary_phase

        return impute_ternary_phase(self, composition, temperature, **kwargs)

    def adjust_phase_temperature(self, phase: str, temperature: float, **kwargs):
        """Re-impute an existing solid phase so it leaves this system's hull at ``temperature``
        (K).

        See :func:`gliquid.imputation.ternary_imputation.adjust_ternary_phase_temperature` for the
        arguments.
        """
        from gliquid.imputation.ternary_imputation import adjust_ternary_phase_temperature

        return adjust_ternary_phase_temperature(self, phase, temperature, **kwargs)

    def interpolate(self):
        # create the hsx dataframe for the ternary system
        self.interpolate_liquid_surface()  # populates self.hsx_df with ternary liquid phase data
        if self.solid_solutions:
            self.append_solid_solution_surfaces()  # appends per-phase SS Gibbs surfaces
        self.surface_df = self.hsx_df.copy()
        self.tern_mp_df = self.get_ternary_form_en(self.components)
        self.hsx_df = pd.concat([self.hsx_df, self.tern_mp_df], ignore_index=True)
        self.hsx_df = self.hsx_df.drop_duplicates()
        self.hsx_df = self.hsx_df.reset_index(drop=True)

    # -- Hull-evaluation pipeline, callable on the model directly --

    def _row_arrays(self) -> tuple[np.ndarray, ...]:
        """``proc_df``'s ``(x0, x1)``, H, S and liquid mask as arrays, read once per ``proc_df``
        object."""
        df = self.proc_df
        cached = getattr(self, "_proc_arrays", None)
        if cached is None or cached[0] is not df:
            liquid = (df["Phase"] == "L").to_numpy()
            xy, enthalpy, entropy = (df[c].to_numpy(float) for c in (["x0", "x1"], "H", "S"))
            cached = self._proc_arrays = (df, xy, enthalpy, entropy, liquid)
        return cached[1:]

    def _row_gibbs(self, temps, rows=None) -> np.ndarray:
        """Gibbs energy, J/mol-atom, of ``proc_df`` rows at ``temps`` (K): the liquid's with its
        H and S taken at that temperature (non-finite ones read 0), every other row's
        ``H - T*S``.

        Args:
            temps: one temperature, or one per row of ``rows``.
            rows: ``proc_df`` row positions; every row when ``None``.
        """
        xy, enthalpy, entropy, is_liquid = self._row_arrays()
        rows = np.arange(len(xy)) if rows is None else np.asarray(rows, dtype=int)
        temps = np.broadcast_to(np.asarray(temps, dtype=float), rows.shape)
        gibbs = enthalpy[rows] - temps * entropy[rows]
        liquid = is_liquid[rows]
        if liquid.any():
            x = xy[rows[liquid]]
            t = temps[liquid]
            with np.errstate(divide="ignore", invalid="ignore"):
                h, s = self.liquid_model().h_s_grid((x[:, 0], x[:, 1]), t)
            gibbs[liquid] = np.where(np.isfinite(h), h, 0.0) - t * np.where(np.isfinite(s), s, 0.0)
        return gibbs

    def _exact_on_hull(self, temp: float) -> np.ndarray:
        """Which ``proc_df`` rows are vertices of the lower hull at ``temp`` (K), every row at its
        :meth:`_row_gibbs`."""
        points = np.column_stack([self._row_arrays()[0], self._row_gibbs(temp)])
        on_hull = np.zeros(len(points), dtype=bool)
        on_hull[np.asarray(lower_convex_hull(points, vertical_simplices=False), dtype=int)] = True
        return on_hull

    def _liquid_nodes_on_hull(self, temp: float) -> int:
        """Count the grid compositions carrying liquid on the lower hull at ``temp`` (K).

        Applies the same test ``_eval_hull`` does at one grid rung: a composition reports a
        liquidus when its liquid point is a VERTEX of the lower hull over solids and liquid
        together. Reads ``proc_df``, so it is callable before the temperature grid exists.
        """
        return int(np.count_nonzero(self._exact_on_hull(temp) & self._row_arrays()[3]))

    def _liquidus_bracket(self) -> tuple[float | None, float | None]:
        """Temperatures where the FIRST and LAST grid composition turn liquid, in kelvin.

        The liquid node count rises monotonically with temperature, so each end is found by
        bisection to ``PRESCREEN_TOL_K``. Costs one lower hull per bisection step against one
        per grid rung. Returns ``(None, None)`` when no composition is liquid anywhere below
        ``PRESCREEN_CEILING_K``.
        """
        n_liquid = int((self.proc_df["Phase"] == "L").sum())
        if not n_liquid or self._liquid_nodes_on_hull(PRESCREEN_CEILING_K) < n_liquid:
            return None, None

        def first_reaching(target: int) -> float:
            low, high = 0.0, PRESCREEN_CEILING_K
            if self._liquid_nodes_on_hull(low) >= target:
                return low
            while high - low > PRESCREEN_TOL_K:
                mid = 0.5 * (low + high)
                if self._liquid_nodes_on_hull(mid) >= target:
                    high = mid
                else:
                    low = mid
            return float(high)

        return first_reaching(1), first_reaching(n_liquid)

    def _require_consistent_store(self, purpose: str) -> None:
        """Refuse a read of the global store that this object's own store would not serve.

        Returns at once while the global store is explicitly configured. While it is the
        implicit ``<cwd>/gliquid_cache`` default, returns only when this object's store is
        at that same location; a store whose location cannot be derived counts as
        elsewhere.

        Args:
            purpose: what reads the global store, named in the message.

        Raises:
            ConfigError: the global store is the implicit default and this object's store
                is somewhere else. The message names both locations.
        """
        if not cfg._cache_dir_is_implicit():
            return
        default = Path(cfg.cache_dir).resolve()
        location = cache._store_location(self.data_dir)
        if location is not None:
            location = location.resolve()
            if os.path.normcase(str(location)) == os.path.normcase(str(default)):
                return
            where = f"at {location}"
            remedy = f"gliquid.config.set_cache_dir({str(location)!r})"
        else:
            where = f"{self.data_dir!r}, whose location cannot be determined"
            remedy = "gliquid.config.set_cache_dir(<the ternary's store>)"
        raise cfg.ConfigError(
            f"{purpose} reads the global gliquid cache store, which is the implicit default "
            f"store at {default}, but this ternary's own store is {where}. The two stores "
            f"can hold different records for the same system. Point the global store at the "
            f"ternary's store with {remedy}."
        )

    def _reference_floor(self) -> float | None:
        """Lowest temperature the three edges' reference frames reach, in kelvin.

        Each edge contributes the bottom of the MPDS diagram cached for it, or
        ``min(t_fusion) - PREDICTED_EDGE_FLOOR_MARGIN_K`` where the edge is predicted -- the
        rule ``BinaryLiquid.from_cache`` applies to one binary, taken over the three edges.

        Consults the cache only: an edge with no cached diagram takes the predicted floor
        rather than reaching for the network.

        Raises:
            ConfigError: the global store is the implicit ``<cwd>/gliquid_cache`` default and
                this object's store is at a different location.
        """
        self._require_consistent_store("Deriving the reference floor of the temperature window")
        backend = cache.resolve_backend(None)
        floors = []
        for pair in self.binary_systems:
            elements = pair.split("-")
            mpds_json = {}
            try:
                if backend.variants("-".join(sorted(elements)), cache.KIND_MPDS):
                    mpds_json, _ = mpds.load_mpds_data(sorted(elements))
            except Exception:  # noqa: BLE001
                mpds_json = {}
            if mpds_json.get("temp"):
                floors.append(float(mpds_json["temp"][0]) + 273.15)
            else:
                t_fusion = [float(UNARY[element].t_fusion) for element in elements]
                floors.append(min(t_fusion) - PREDICTED_EDGE_FLOOR_MARGIN_K)
        return min(floors) if floors else None

    @staticmethod
    def _liquidus_span(temp_df) -> tuple[float, float] | None:
        """Coldest and hottest liquidus temperature in a facet frame, in kelvin.

        The liquidus at a composition is the COLDEST facet carrying liquid there, so rows are
        reduced per composition before either extreme is taken. Returns ``None`` when the
        frame carries no liquid.

        Reducing first is not cosmetic: a raw maximum over liquid rows lands on a
        near-degenerate facet, whose hyperplane normal yields an enormous finite temperature.
        """
        liquid = temp_df.loc[temp_df["Phase"] == "L", ["x0", "x1", "T"]]
        if liquid.empty:
            return None
        per_composition = liquid.groupby(["x0", "x1"])["T"].min()
        return float(per_composition.min()), float(per_composition.max())

    def _pad_ceiling(self, value: float) -> float:
        """``value`` raised onto the T_incr lattice with the window's standard headroom."""
        return (int(np.ceil(value / self.T_incr)) + WINDOW_MARGIN_SLICES) * self.T_incr

    def _apply_window(self, low: float, high: float) -> None:
        """Lay ``T_grid``, ``conds`` and ``temp_range`` over one window, in kelvin.

        All three come from the same two numbers so none can drift from the others.
        ``temp_slider`` widens the window here: ``[0]`` downward against the absolute-zero
        clamp, ``[1]`` upward.
        """
        floor = max(0.0, low - self.temp_slider[0])
        floor = np.floor(floor / self.T_incr) * self.T_incr
        self.T_grid = np.arange(floor, high + self.temp_slider[1] + self.T_incr, self.T_incr)
        self.conds = [float(self.T_grid[0]) - 273.15, float(self.T_grid[-1]) - 273.15]
        self.temp_range = [float(self.T_grid[0]), float(self.T_grid[-1])]

    def _resolve_window(self, liquidus_floor, liquidus_ceiling) -> tuple[float, float]:
        """The temperature window before the grid lattice is applied, in kelvin.

        Either end of ``temp_range=`` wins where it is given; a ``None`` end is derived. The
        derived floor is the lowest of the liquidus, the frame the three edges' references
        are drawn in and the ceiling, clamped at absolute zero; the derived ceiling is the
        screened liquidus top, or the highest melting point where there is no screen, both
        carrying the same headroom through ``_pad_ceiling``, and at least a pinned floor with
        that headroom.

        Raises:
            ValueError: both ends are pinned and the low end is above the high end.
        """
        low, high = self._temp_range_arg or (None, None)
        if low is not None and high is not None and float(low) > float(high):
            raise ValueError(
                f"temp_range {self._temp_range_arg} has its low end above its high end."
            )
        if high is None:
            # Only a DERIVED ceiling carries headroom; a pinned one is honoured exactly.
            high = self._pad_ceiling(
                float(np.max(self.ref_data["T"])) if liquidus_ceiling is None else liquidus_ceiling
            )
            if low is not None and float(low) >= high:
                high = self._pad_ceiling(float(low))
        if low is None:
            candidates = [high] if liquidus_floor is None else [high, float(liquidus_floor)]
            reference = self._reference_floor()
            if reference is not None:
                candidates.append(float(reference))
            low = min(candidates)
        return max(0.0, float(low)), float(high)

    def _init_sys(self, color_map=None):
        """Prepare hull-evaluation inputs on a processed copy (``proc_df``).

        Interpolates lazily, then rounds/relabels/colors a COPY of ``hsx_df`` (the
        pristine interpolated frame is never mutated), computes the temperature grid,
        and (gtx mode) slices per-T Gibbs frames (:meth:`_row_gibbs`) into ``df_Tgroups``.
        ``color_map`` overrides the palette; the default is the shared build_phase_color_map
        scheme.
        """
        self.proc_df = self.hull_table()
        self.sys_name = "-".join(sorted(self.components))
        self.phase_names = self.proc_df["Phase Name"].unique().tolist()

        if color_map is not None:
            self.color_map = dict(color_map)
        else:
            # Elemental polymorph ladder entries, as in get_ternary_form_en.
            exclude_spacegroups = {solution.SS_SPACEGROUPS[p] for p in self.ss_models}
            component_data = UNARY.component_data(self.components)
            polymorph_names = {
                p.name or f"{comp} (sg {p.spacegroup_number})"
                for comp in self.components
                for p in polymorph_ladder(component_data[comp], exclude_spacegroups)
            } & set(self.phase_names)
            # The binary plots' color builder.
            self.color_map = build_phase_color_map(
                self.phase_names,
                ss_names=list(self.ss_models),
                polymorph_names=polymorph_names,
                imputed_names=self.imputed_phase_names() + self.user_phase_names(),
            )

        self.proc_df = self.proc_df.rename(columns={"Phase Name": "Phase"})
        self.proc_df["Colors"] = self.proc_df["Phase"].map(self.color_map)

        # One temperature window as T_grid (K), temp_range (its ends) and conds (its ends in C). gtx
        # screens the ceiling against the liquidus; hsx widens it in _process_data_hsx. temp_slider
        # widens it ([0] down, [1] up) and the floor is clamped at 0 K.
        liquidus_floor, liquidus_ceiling = (
            self._liquidus_bracket() if self.hull_mode == "gtx" else (None, None)
        )
        self._window = self._resolve_window(liquidus_floor, liquidus_ceiling)
        self._apply_window(*self._window)

        self.df_Tgroups = {}
        if self.hull_mode == "gtx":
            n = len(self.proc_df)
            gibbs = self._row_gibbs(
                np.repeat(self.T_grid, n), np.tile(np.arange(n), len(self.T_grid))
            )
            frame = self.proc_df[["x0", "x1", "Phase", "Colors"]]
            for T, g in zip(self.T_grid, gibbs.reshape(-1, n)):
                self.df_Tgroups[T] = frame.assign(G=g)[["x0", "x1", "G", "Phase", "Colors"]]

    def _liquid_hull(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """One HSX hull over ``proc_df``: ``compute_tx``'s frame (``t`` in K), its facets as
        ``proc_df`` row positions, and each facet's temperature (K)."""
        table = self.proc_df[["x0", "x1", "S", "H", "Phase"]]
        hsx_obj = HSX({"data": table, "phases": self.phase_names, "comps": self.components}, [0, 0])
        df_tx, _, simplices, temps = hsx_obj.compute_tx()
        return df_tx, np.asarray(simplices, dtype=int).reshape(-1, 4), np.asarray(temps, float)

    def _row_facets(self, simplices, temps, rows, hottest=False) -> tuple[np.ndarray, np.ndarray]:
        """Each of ``rows``' lowest temperature of a facet it is a vertex of (its highest where
        ``hottest``; NaN where none), and the index of that facet (-1 where none)."""
        liq = np.full(len(rows), np.nan)
        facet = np.full(len(rows), -1)
        if not len(simplices):
            return liq, facet
        where = np.full(len(self.proc_df), -1)
        where[rows] = np.arange(len(rows))
        member = where[simplices].ravel()
        facets = np.repeat(np.arange(len(simplices)), simplices.shape[1])
        keep = member >= 0
        member, facets = member[keep], facets[keep]
        order = np.lexsort((-temps[facets] if hottest else temps[facets], member))
        member, facets = member[order], facets[order]
        first = np.r_[True, member[1:] != member[:-1]]
        liq[member[first]] = temps[facets[first]]
        facet[member[first]] = facets[first]
        return liq, facet

    @staticmethod
    def _segment_weights(points, x) -> np.ndarray | None:
        """Weights on collinear ``points`` (3, 2) placing ``x`` between two of them, or None."""
        for p, q in ((0, 1), (0, 2), (1, 2)):
            d = points[q] - points[p]
            length2 = float(d @ d)
            if length2 <= 0:
                continue
            u = float((x - points[p]) @ d) / length2
            if -1e-9 <= u <= 1 + 1e-9 and np.linalg.norm(points[p] + u * d - x) <= 1e-9:
                weights = np.zeros(3)
                weights[p], weights[q] = 1 - u, u
                return weights
        return None

    def _hull_brackets(self, rows, temps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each of ``rows`` on or off the lower hull (:meth:`_exact_on_hull`) at each of ``temps``
        (K, ascending), and the two of ``temps`` around its last move onto it.

        Returns:
            tuple: the (len(temps), len(rows)) on-hull matrix, and per row the rung it is last
            off the hull at and the rung after it; NaN where the row is never off the hull or is
            still off it at the last rung.
        """
        on = np.array([self._exact_on_hull(t)[rows] for t in temps]).reshape(len(temps), -1)
        last_off = len(temps) - 1 - np.argmax(~on[::-1], axis=0)
        moved = (~on).any(axis=0) & (last_off < len(temps) - 1)
        low = np.where(moved, temps[np.minimum(last_off, len(temps) - 1)], np.nan)
        high = np.where(moved, temps[np.minimum(last_off + 1, len(temps) - 1)], np.nan)
        return on, low, high

    def _hull_bisect(self, rows, low, high) -> np.ndarray:
        """Each of ``rows``' liquidus (K) between ``low``, where it is off the lower hull, and
        ``high``, where it is on it, bisected on lower hulls (:meth:`_exact_on_hull`) to
        ``NODE_LIQUIDUS.hull_tol_k``, one hull per distinct midpoint; NaN where a bound is."""
        low, high = np.array(low, dtype=float), np.array(high, dtype=float)
        while True:
            live = np.flatnonzero(high - low > NODE_LIQUIDUS.hull_tol_k)
            if not len(live):
                return 0.5 * (low + high)
            mid = 0.5 * (low[live] + high[live])
            for temp in np.unique(mid):
                at = live[mid == temp]
                on = self._exact_on_hull(temp)[rows[at]]
                high[at[on]] = temp
                low[at[~on]] = temp

    def _scanned_liquidus(self, rows, top: float) -> np.ndarray:
        """Each of ``rows``' liquidus (K) from ``NODE_LIQUIDUS.scan_rungs`` lower hulls spread from
        the bottom of ``NODE_LIQUIDUS.search_k`` to ``top``, ``top`` first raised by half while a
        row is off the hull there, then bisected (:meth:`_hull_bisect`); NaN where a row is never
        off the hull or is still off it at the top."""
        bottom, ceiling = NODE_LIQUIDUS.search_k
        top = max(float(top), 2 * bottom)
        while top < ceiling and not self._exact_on_hull(top)[rows].all():
            top = min(ceiling, 1.5 * top)
        _, low, high = self._hull_brackets(rows, np.linspace(bottom, top, NODE_LIQUIDUS.scan_rungs))
        return self._hull_bisect(rows, low, high)

    def _checked_liquidus(self, rows, liquidus) -> np.ndarray:
        """``liquidus`` (K) of ``rows`` checked against lower hulls (:meth:`_exact_on_hull`) every
        ``NODE_LIQUIDUS.check_k``, from a rung below the coldest value to a rung above the hottest.

        A row on the hull at a rung more than ``NODE_LIQUIDUS.hull_tol_k`` below its value, or off
        it at a rung more than that above, is solved again: bisected between the rungs around its
        last move onto the hull (:meth:`_hull_bisect`), or scanned (:meth:`_scanned_liquidus`)
        where the check holds no such rungs, keeping its value where the scan finds none.
        """
        out = np.array(liquidus, dtype=float)
        known = np.isfinite(out)
        if not known.any():
            return out
        step = NODE_LIQUIDUS.check_k
        bottom = max(NODE_LIQUIDUS.search_k[0], np.floor(out[known].min() / step) * step - step)
        temps = np.arange(bottom, np.ceil(out[known].max() / step) * step + 1.5 * step, step)
        on, low, high = self._hull_brackets(rows, temps)
        rung = temps[:, None]
        wrong = known & (
            ((rung < out - NODE_LIQUIDUS.hull_tol_k) & on).any(axis=0)
            | ((rung > out + NODE_LIQUIDUS.hull_tol_k) & ~on).any(axis=0)
        )
        inside = np.flatnonzero(wrong & np.isfinite(low))
        outside = np.flatnonzero(wrong & ~np.isfinite(low))
        out[inside] = self._hull_bisect(rows[inside], low[inside], high[inside])
        if len(outside):
            found = self._scanned_liquidus(rows[outside], 1.05 * out[outside].max())
            out[outside] = np.where(np.isfinite(found), found, out[outside])
        return out

    def _facet_crossing(self, simplices, facet, start, rows, rising=False) -> np.ndarray:
        """Each of ``rows``' temperature (K) on its HSX ``facet`` where its Gibbs energy
        (:meth:`_row_gibbs`) crosses the plane through the facet's other three vertices.

        The crossing is looked for from ``start`` in steps of 1, 2, 4, ... K inside
        ``NODE_LIQUIDUS.search_k``, upward where the row lies above the plane, then bisected
        (:func:`_bisect`). With ``rising`` the row's height over the plane rises with temperature
        and the search runs the other way. NaN where the row has no facet, the other three
        vertices place no plane under it, or no crossing is found.
        """
        out = np.full(len(rows), np.nan)
        nodes = np.flatnonzero((facet >= 0) & np.isfinite(start))
        vertices = simplices[facet[nodes]]
        others = vertices != rows[nodes, None]
        three = others.sum(axis=1) == 3
        nodes, others = nodes[three], vertices[three][others[three]].reshape(-1, 3)
        xy = self.proc_df[["x0", "x1"]].to_numpy(float)
        weights = _barycentric(xy[others], xy[rows[nodes]])
        for i in np.flatnonzero(np.isnan(weights).any(axis=1)):
            on_segment = self._segment_weights(xy[others[i]], xy[rows[nodes[i]]])
            weights[i] = np.nan if on_segment is None else on_segment
        held = np.isfinite(weights).all(axis=1)
        nodes, others, weights = nodes[held], others[held], weights[held]
        if not len(nodes):
            return out
        gap = _plane_gap(self, rows, nodes, others, weights)
        bottom, ceiling = NODE_LIQUIDUS.search_k
        start = np.clip(start[nodes], bottom, ceiling)
        at_start = gap(start, nodes)
        toward = np.where((at_start > 0) != rising, 1.0, -1.0)
        low, high = start.copy(), np.where(at_start == 0, start, np.nan)
        searching = np.isfinite(at_start) & (at_start != 0)
        step = 1.0
        while searching.any():
            live = np.flatnonzero(searching)
            temp = np.clip(start[live] + toward[live] * step, bottom, ceiling)
            at = gap(temp, nodes[live])
            crossed = np.isfinite(at) & (np.sign(at) != np.sign(at_start[live]))
            high[live[crossed]] = temp[crossed]
            low[live[~crossed]] = temp[~crossed]
            searching[live[crossed | (temp <= bottom) | (temp >= ceiling)]] = False
            step *= 2
        value = np.where(np.isfinite(high), start, np.nan)
        span = np.flatnonzero(np.isfinite(high) & (high != low))
        value[span] = _bisect(
            gap, nodes[span], np.minimum(low, high)[span], np.maximum(low, high)[span]
        )
        out[nodes] = value
        return out

    def _hsx_liquidus(self, simplices, temps, rows) -> tuple[np.ndarray, np.ndarray]:
        """Each of the liquid ``rows``' liquidus (K) on one HSX hull, and its liquidus facet (-1
        where it has none).

        The facet temperature (:meth:`_row_facets`) is solved again on that facet
        (:meth:`_facet_crossing`). A row that solve leaves without a value is scanned
        (:meth:`_scanned_liquidus`) and keeps its facet temperature where the scan finds none.
        Every value is then checked (:meth:`_checked_liquidus`).
        """
        liquidus, facet = self._row_facets(simplices, temps, rows)
        solved = self._facet_crossing(simplices, facet, liquidus, rows)
        unsolved = np.flatnonzero(~np.isfinite(solved))
        if len(unsolved):
            hint = liquidus[unsolved]
            if np.isnan(hint).any() and np.isfinite(liquidus).any():
                hint = np.append(hint, np.nanmax(liquidus))
            top = 1.05 * np.nanmax(hint) if np.isfinite(hint).any() else np.max(self.ref_data["T"])
            found = self._scanned_liquidus(rows[unsolved], top)
            solved[unsolved] = np.where(np.isfinite(found), found, liquidus[unsolved])
        return self._checked_liquidus(rows, solved), facet

    def _solid_tops(self, simplices, temps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The solid ``proc_df`` rows of the HSX hull, the highest temperature (K) each is on it,
        and the facet that temperature belongs to.

        Where that facet carries liquid, the temperature is solved again on it with the liquid's
        Gibbs energy at temperature (:meth:`_facet_crossing`); elsewhere, or where that solve
        finds none, it is the facet's temperature.
        """
        liquid = (self.proc_df["Phase"] == "L").to_numpy()
        rows = np.unique(simplices[~liquid[simplices]])
        top, facet = self._row_facets(simplices, temps, rows, hottest=True)
        melting = np.flatnonzero(liquid[simplices[facet]].any(axis=1))
        solved = self._facet_crossing(
            simplices, facet[melting], top[melting], rows[melting], rising=True
        )
        top[melting] = np.where(np.isfinite(solved), solved, top[melting])
        return rows, top, facet

    def _placed(self, df_tx, simplices, facet, rows, values, bound) -> pd.DataFrame:
        """``df_tx`` with each of ``rows``' vertex rows bounded by its ``values`` (K) through
        ``bound`` (``np.fmax`` or ``np.fmin``) and the vertex row of its ``facet`` set to it."""
        node = np.full(len(self.proc_df), -1)
        node[rows] = np.arange(len(rows))
        vertex = node[simplices.ravel()]
        t = df_tx["t"].to_numpy(float).copy()
        at_node = vertex >= 0
        t[at_node] = bound(t[at_node], values[vertex[at_node]])
        placed = np.flatnonzero((facet >= 0) & np.isfinite(values))
        corner = np.argmax(simplices[facet[placed]] == rows[placed, None], axis=1)
        t[simplices.shape[1] * facet[placed] + corner] = values[placed]
        return df_tx.assign(t=t)

    def _with_node_liquidus(self, df_tx, simplices, facet, rows, liquidus) -> pd.DataFrame:
        """``df_tx`` with each liquid row's vertex rows raised to at least its ``liquidus`` (K) and
        the vertex row of its liquidus ``facet`` set to it; a row with a liquidus and no facet
        adds a facet of its own, four liquid rows at its liquidus."""
        df_tx = self._placed(df_tx, simplices, facet, rows, liquidus, np.fmax)
        alone = np.flatnonzero((facet < 0) & np.isfinite(liquidus))
        if not len(alone):
            return df_tx
        xy = self.proc_df[["x0", "x1"]].to_numpy(float)[np.repeat(rows[alone], 4)]
        added = pd.DataFrame(
            {"x0": xy[:, 0], "x1": xy[:, 1], "t": np.repeat(liquidus[alone], 4), "label": "L"}
        )
        return pd.concat([df_tx, added[df_tx.columns]], ignore_index=True)

    def _set_liquid_tstar(self, rows, liquidus) -> None:
        """Set ``liquid_tstar`` to the liquid ``rows``' ``liquidus`` (K), ``linear_tstar`` where
        a row has none, kept inside ``NODE_LIQUIDUS.search_k``."""
        x0, x1 = (self.proc_df[column].to_numpy(float)[rows] for column in ("x0", "x1"))
        tstar = np.clip(
            np.where(np.isfinite(liquidus), liquidus, self.linear_tstar(x0, x1)),
            *NODE_LIQUIDUS.search_k,
        )
        self.liquid_tstar = pd.DataFrame({"x0": x0, "x1": x1, "tstar": tstar})

    def _set_sheets_tstar(self, rows, liquidus) -> None:
        """Set ``liquid_tstar`` to the liquid ``rows``' liquidus on the liquidus sheets built with
        T* at their node ``liquidus`` (K) (:meth:`_set_liquid_tstar`), keeping that T* at a row
        where the sheets' liquidus is not finite."""
        self._set_liquid_tstar(rows, liquidus)
        nodes = self.liquid_tstar[["x0", "x1"]].to_numpy(float)
        on_sheets = LiquidusSheets.from_interpolation(self).liquidus(nodes)
        self._set_liquid_tstar(
            rows, np.where(np.isfinite(on_sheets), on_sheets, self.liquid_tstar["tstar"])
        )

    def _process_data_hsx(self):
        """One (x0, x1, S, H) lower hull with analytic facet temperatures (hull_mode='hsx').

        The generalized HSX replaces the per-temperature G-slicing: coexistence
        temperatures come from facet hyperplane normals, so they are continuous instead
        of T-grid multiples. Each liquid node's liquidus, and each solid's highest temperature on
        the hull, is then solved with the liquid's Gibbs energy at temperature
        (:meth:`_hsx_liquidus`, :meth:`_solid_tops`) and placed in the frame, and
        ``liquid_tstar`` takes the liquidus sheets' own liquidus at the nodes
        (:meth:`_set_sheets_tstar`). Output schema matches the gtx path (equil_df_list rows of
        [x0, x1, T(C), Phase, Colors, simplex_id] in ternary Cartesian coordinates).

        Re-lays ``temp_range`` (and with it ``T_grid`` and ``conds``) on the liquid facet
        temperatures, so on this path the window is final here rather than in ``_init_sys``.
        """
        start_time = time.time()
        df_tx, simplices, temps = self._liquid_hull()
        solids, tops, top_facet = self._solid_tops(simplices, temps)
        df_tx = self._placed(df_tx, simplices, top_facet, solids, tops, np.fmin)
        rows = np.flatnonzero((self.proc_df["Phase"] == "L").to_numpy())
        liquidus, facet = self._hsx_liquidus(simplices, temps, rows)
        df_tx = self._with_node_liquidus(df_tx, simplices, facet, rows, liquidus)
        self._set_sheets_tstar(rows, liquidus)

        temp_df = df_tx.rename(columns={"t": "T", "label": "Phase"})
        temp_df["Colors"] = temp_df["Phase"].map(self.color_map)  # keep this system's palette
        temp_df["simplex_id"] = np.repeat(np.arange(len(temp_df) // 4), 4)
        temp_df = temp_df[["x0", "x1", "T", "Phase", "Colors", "simplex_id"]]

        # Widens the window to the liquid facet temperatures; ends pinned through temp_range= stay.
        span = self._liquidus_span(temp_df)
        if span is not None:
            pinned_low, pinned_high = self._temp_range_arg or (None, None)
            low, high = self._window
            if pinned_low is None:
                low = min(low, span[0])
            if pinned_high is None:
                high = max(high, self._pad_ceiling(span[1]))
            self._window = (low, high)
            self._apply_window(low, high)

        temp_df["x0_orig"] = temp_df["x0"].copy()
        temp_df["x1_orig"] = temp_df["x1"].copy()
        temp_df = cartesian_to_ternary(temp_df)
        temp_df["T"] = temp_df["T"] - 273.15

        self.equil_df_list = [temp_df]
        self._n_simplex_vertices = 4
        logger.info(
            f"HSX lower hull evaluation and post processing time:: "
            f"{time.time() - start_time} seconds (single hull, analytic temperatures)"
        )

    def _rung_liquidus(self, rungs, rows) -> np.ndarray:
        """Each of the liquid ``rows``' liquidus (K) from the lower hulls ``rungs``, one per
        ``T_grid`` temperature, written into every frame of ``equil_df_list`` up to the one it is
        found at.

        A row's liquidus rung is the rung after the last one it is off the hull at, or the first
        one it is on at where it is off the hull at the last rung. A row whose liquidus rung is
        k > 0 is solved between rungs k-1 and k (:func:`_bisect`) where its Gibbs energy
        (:meth:`_row_gibbs`) meets the lowest plane holding it through three points of the rung
        k-1 facet holding it and of its neighbours at rung k; it takes rung k-1 where it is
        already below that plane there and rung k where it is still above it there. A row on at
        every rung takes rung 0, a row never on is NaN.
        """
        T_grid = np.asarray(self.T_grid, dtype=float)
        xy = self.proc_df[["x0", "x1"]].to_numpy(float)
        node = np.full(len(xy), -1)
        node[rows] = np.arange(len(rows))
        on = np.zeros((len(rungs), len(rows)), dtype=bool)
        for k, simplices in enumerate(rungs):
            joined = node[np.unique(simplices)]
            on[k, joined[joined >= 0]] = True
        last_off = np.where(on, -1, np.arange(len(rungs))[:, None]).max(axis=0, initial=-1)
        rung = np.where(on[-1], last_off + 1, np.where(on.any(axis=0), on.argmax(axis=0), -1))
        liquidus = np.where(rung >= 0, T_grid[np.maximum(rung, 0)], np.nan)
        plane_node, plane_rows, plane_weights = [], [], []
        for k in np.unique(rung[rung > 0]):
            below, above = rungs[k - 1], rungs[k]
            at_k = np.flatnonzero(rung == k)
            holding = _containing(xy[below], xy[rows[at_k]])
            for i, facet in zip(at_k, holding):
                if facet < 0:
                    continue
                ring = above[(above == rows[i]).any(axis=1)].ravel()
                points = np.setdiff1d(np.concatenate([below[facet], ring]), rows[i])
                corners = points[_triples(len(points))]
                weights = _barycentric(xy[corners], xy[rows[i]])
                holds = np.isfinite(weights).all(axis=1) & (weights.min(axis=1) >= -1e-9)
                weights = np.clip(weights[holds], 0.0, None)
                plane_node.append(np.full(len(weights), i))
                plane_rows.append(corners[holds])
                plane_weights.append(weights / weights.sum(axis=1, keepdims=True))
        if plane_node:
            plane_node = np.concatenate(plane_node)
            nodes = np.unique(plane_node)
            gap = _plane_gap(
                self, rows, plane_node, np.concatenate(plane_rows), np.concatenate(plane_weights)
            )
            low, high = T_grid[rung[nodes] - 1], T_grid[rung[nodes]]
            at_low, at_high = gap(low, nodes), gap(high, nodes)
            value = np.where(at_low < 0, low, high)
            span = np.flatnonzero((at_low >= 0) & (at_high <= 0))
            value[span] = _bisect(gap, nodes[span], low[span], high[span])
            liquidus[nodes] = value
        for k in range(int(rung.max(initial=-1)) + 1):
            at = node[rungs[k].ravel()]
            lift = (at >= 0) & (rung[at] >= k)
            frame = self.equil_df_list[k]
            celsius = frame["T"].to_numpy(float).copy()
            celsius[lift] = liquidus[at[lift]] - 273.15
            frame["T"] = celsius
        return liquidus

    def _eval_hull(self):
        """Per-temperature lower hulls over ``df_Tgroups`` (hull_mode='gtx'), one frame per rung.

        Each liquid node's liquidus is refined below the rung of its last move onto the hull
        (:meth:`_rung_liquidus`), and ``liquid_tstar`` takes the liquidus sheets' own liquidus
        at the nodes (:meth:`_set_sheets_tstar`).
        """
        start_time = time.time()
        phases = self.proc_df["Phase"].to_numpy(object)
        colors = self.proc_df["Colors"].to_numpy(object)
        self.equil_df_list, rungs, shifter = [], [], 0
        for T in self.T_grid:
            points = self.df_Tgroups[T][["x0", "x1", "G"]].to_numpy(float)
            simplices = lower_convex_hull(points, vertical_simplices=False)
            simplices = np.asarray(simplices, dtype=int).reshape(-1, 3)
            vertices = simplices.ravel()
            x0, x1 = points[vertices, 0], points[vertices, 1]
            rungs.append(simplices)
            self.equil_df_list.append(
                pd.DataFrame(
                    {
                        "x0": x0 + 0.5 * x1,
                        "x1": x1 * (np.sqrt(3) / 2),
                        "T": np.full(len(vertices), float(T) - 273.15),
                        "Phase": phases[vertices],
                        "Colors": colors[vertices],
                        "simplex_id": shifter + np.repeat(np.arange(len(simplices)), 3),
                        "x0_orig": x0,
                        "x1_orig": x1,
                    }
                )
            )
            shifter += max(len(simplices), 1)
        rows = np.flatnonzero((self.proc_df["Phase"] == "L").to_numpy())
        self._set_sheets_tstar(rows, self._rung_liquidus(rungs, rows))
        logger.info(
            f"Lower hull evaluation and post processing time:: {time.time() - start_time} "
            f"seconds for temperature increment of {self.T_incr}"
        )

    def process_data(self, color_map=None):
        """The full hull pipeline on the model -- interpolate, prepare, evaluate -- with
        no plotter involved. ``color_map`` is forwarded to :meth:`_init_sys`."""
        self._init_sys(color_map=color_map)
        if self.hull_mode == "hsx":
            self._process_data_hsx()
            return
        self._eval_hull()

    def _gibbs_slice(self, T_kelvin: float) -> pd.DataFrame:
        """The ``[x0, x1, G, Phase, Colors]`` hull frame at ``T_kelvin``, indexed from 0.

        Read from ``df_Tgroups`` when the gtx pipeline already built that slice, otherwise
        evaluated from ``proc_df`` by :meth:`_row_gibbs`.
        """
        if T_kelvin in self.df_Tgroups:
            return self.df_Tgroups[T_kelvin].copy().reset_index(drop=True)
        frame = self.proc_df[["x0", "x1", "Phase", "Colors"]].copy()
        frame.insert(2, "G", self._row_gibbs(T_kelvin))
        return frame.reset_index(drop=True)

    def get_convex_hull(self, T_celsius: float) -> dict:
        """Extract a single G-x0-x1 lower convex-hull slice at the nearest grid temperature.

        Returns the hull DATA -- points, simplices, per-simplex and transformed-point
        frames, and the exact grid temperature the request snapped to. Works in either
        ``hull_mode``.
        """
        if not hasattr(self, "df_Tgroups") or not hasattr(self, "T_grid"):
            self._init_sys()

        T_kelvin_request = float(T_celsius) + 273.15
        if len(self.T_grid) == 0:
            raise ValueError(
                "Temperature grid is empty. Run initialization before extracting a hull slice."
            )

        nearest_index = int(np.argmin(np.abs(self.T_grid - T_kelvin_request)))
        T_kelvin = float(self.T_grid[nearest_index])
        T_celsius_exact = T_kelvin - 273.15

        slice_df = self._gibbs_slice(T_kelvin)
        points = np.array(slice_df[["x0", "x1", "G"]])
        simplices = np.asarray(lower_convex_hull(points, vertical_simplices=False))

        if simplices.size == 0:
            raise ValueError(f"No lower-hull simplices found at T={T_celsius_exact:.6g} C.")

        simplex_rows = []
        for simplex_id, simplex in enumerate(simplices):
            for vertex in simplex:
                label = slice_df.loc[vertex, "Phase"]
                simplex_rows.append(
                    [
                        points[vertex][0],
                        points[vertex][1],
                        points[vertex][2],
                        label,
                        self.color_map[label],
                        simplex_id,
                    ]
                )

        simplex_df = pd.DataFrame(
            simplex_rows, columns=["x0", "x1", "G", "Phase", "Colors", "simplex_id"]
        )
        simplex_df["x0_orig"] = simplex_df["x0"].copy()
        simplex_df["x1_orig"] = simplex_df["x1"].copy()
        simplex_df = cartesian_to_ternary(simplex_df)

        transformed_points_df = slice_df[["x0", "x1"]].copy()
        transformed_points_df["x0_orig"] = transformed_points_df["x0"].copy()
        transformed_points_df["x1_orig"] = transformed_points_df["x1"].copy()
        transformed_points_df = cartesian_to_ternary(transformed_points_df)

        return {
            "requested_temperature_c": float(T_celsius),
            "requested_temperature_k": T_kelvin_request,
            "temperature_c": T_celsius_exact,
            "temperature_k": T_kelvin,
            "temperature_offset_c": T_celsius_exact - float(T_celsius),
            "raw_slice_df": slice_df,
            "hull_points": points,
            "hull_simplices": simplices,
            "transformed_points_df": transformed_points_df,
            "simplex_df": simplex_df,
        }

    def get_inter_melting_temps(self, interphases_for_melting: list[str]):
        """Max stability temperature per requested phase over the computed hull slices."""
        if not hasattr(self, "equil_df_list"):
            raise Exception(
                "You must run the interpolate() and process_data() methods before getting melting temperatures."
            )

        df_list = self.equil_df_list
        concat_df = pd.concat(df_list, ignore_index=True)
        melting_temps = {}
        for phase in interphases_for_melting:
            if phase not in self.phase_names:
                raise ValueError(
                    f"Phase '{phase}' not found in the system phases: {self.phase_names}"
                )
            sub_df = concat_df[concat_df["Phase"] == phase]
            if sub_df.empty:
                logger.warning(f"No data found for phase '{phase}'. Skipping.")
                continue
            sub_df = sub_df.sort_values(by="T", ascending=False)
            sub_df = sub_df.iloc[0]
            temp = sub_df["T"]
            melting_temps[phase] = temp

        return melting_temps


def _edge_mape(sys: "BinaryLiquid") -> float:
    """The liquidus MAPE of an edge as drawn, by the deviation call the matrix scores with.

    ``init_error`` is cleared first, as the matrix does; invariant points are not inserted into
    the digitized liquidus, since a lean MPDS record cannot supply them. Returns ``inf`` when the
    metric cannot be computed.
    """
    sys.init_error = False
    try:
        mape = sys.calculate_deviation_metrics(
            allow_sparse_data=True, unsupported_range_mode="score"
        )[2]
    except Exception:
        logger.warning("No liquidus MAPE for the %s edge", sys.sys_name, exc_info=True)
        return float("inf")
    return float(mape)


def build_binary_edge_figures(
    ti: "TernaryLiquidInterpolation",
    ternary_color_map=None,
    edge_mape: bool = False,
    pred_liquidus: bool = False,
    progress=None,
):
    """Build one TX figure per parameterized binary edge of ``ti``.

    ``TLIPlotter._init_sys`` collects the result as ``bin_fig_list``.

    Each edge's liquid is converged on every row (``update_phase_points(every_row=True)``).
    Each edge's phases come from :meth:`TernaryLiquidInterpolation.edge_hull`, the hull ``ti``
    places on that edge: its user-defined and imputed phases are drawn dashed under the legend
    entry that names their kind, and a DFT entry one of them replaces is left out.

    A fitted edge draws its digitized MPDS liquidus; with ``pred_liquidus``, so does a predicted
    edge that has one. With ``edge_mape``, each edge that draws a digitized liquidus and whose
    liquidus MAPE against it is finite gets a ``"MAPE x.x%"`` chip
    (:func:`gliquid.plotting.binary_tx.add_corner_note`) colored by
    :func:`gliquid.plotting.style.mape_color`.

    Args:
        ti: the ternary interpolation whose edges are drawn.
        ternary_color_map: phase colours shared with the ternary figure.
        edge_mape: add the MAPE chip.
        pred_liquidus: draw the digitized MPDS liquidus on predicted edges too.
        progress: called as ``progress("edges", done, total)`` after each edge figure.

    Raises:
        ConfigError: the global store is the implicit ``<cwd>/gliquid_cache`` default and
            ``ti``'s store is at a different location.
    """
    ti._require_consistent_store("Building the binary edge figures")
    bin_fig_list = []
    edges = [sys_name for sys_name in ti.binary_systems if sys_name in ti.xs_mix]

    def process_system(sys_name):
        rk = ti.xs_mix[sys_name]
        alphabetical_order = "-".join(sorted(sys_name.split("-")))
        if sys_name != alphabetical_order:  # odd L orders flip under the swap
            rk = rk.swapped()
        sys = BinaryLiquid.from_cache(
            alphabetical_order,
            params=rk.values,
            param_format=ti.param_format,
            pd_ind=0,
            solid_solutions=ti.solid_solutions,
            ss_kwargs=ti.ss_kwargs,
            tau=ti.tau,
            hull=ti.edge_hull(alphabetical_order),
        )
        sys.update_phase_points(every_row=True)
        fit = ti.fit_or_pred.get(sys_name, "pred") == "fit"
        with_liquidus = fit or (pred_liquidus and bool(sys.digitized_liq))
        hsx, shared = _plot_tx_inputs(sys)
        fig = plot_tx(
            hsx,
            pred=not fit,
            digitized_liquidus=sys.digitized_liq if with_liquidus else None,
            ternary_color_map=ternary_color_map,
            **shared,
        )
        if edge_mape and with_liquidus:
            mape = _edge_mape(sys)
            if np.isfinite(mape):
                add_corner_note(fig, f"MAPE {mape:.1f}%", mape_color(mape))
        bin_fig_list.append(fig)

    for done, sys_name in enumerate(edges, start=1):
        process_system(sys_name)
        if progress is not None:
            progress("edges", done, len(edges))

    return bin_fig_list


class TLIPlotter:
    """Plotter over a :class:`TernaryLiquidInterpolation`'s liquid + solid surfaces.

    Wraps a ternary interpolation the way :class:`~gliquid.binary.BLPlotter` wraps a
    ``BinaryLiquid``: the model is HELD (``self._ti``) rather than inherited, and re-framed
    onto the presentation ``order`` (default 'alphabetical'). Build from an existing
    interpolation with ``TLIPlotter(ti, ...)``, or straight from parameters with
    ``TLIPlotter.from_components(...)``.

    The hull-evaluation settings (``temp_slider``, ``T_incr``, ``hull_mode``) are the held
    model's. ``edge_mape`` puts a MAPE chip on each binary edge figure that draws a digitized
    liquidus, and ``pred_liquidus`` draws the digitized liquidus on predicted edges too (see
    :func:`build_binary_edge_figures`).
    """

    @cfg._removed_keywords(
        dict.fromkeys(("temp_slider", "T_incr", "hull_mode"), "TernaryLiquidInterpolation(...)")
    )
    def __init__(
        self,
        ternary_interp: "TernaryLiquidInterpolation",
        order="alphabetical",
        *,
        edge_mape: bool = False,
        pred_liquidus: bool = False,
    ):
        self.edge_mape = bool(edge_mape)
        self.pred_liquidus = bool(pred_liquidus)
        self._ti_raw = ternary_interp
        # Re-frames the held model onto ``order``.
        self._ti = ternary_interp.with_component_order(order)
        self._partition_cache = (None, None, None)

    @classmethod
    def from_components(
        cls,
        components,
        data_dir=None,
        *,
        order="alphabetical",
        edge_mape: bool = False,
        pred_liquidus: bool = False,
        **tli_kwargs,
    ) -> "TLIPlotter":
        """Build the :class:`TernaryLiquidInterpolation` and wrap it in one call.

        Convenience mirror of the ``BinaryLiquid.from_cache`` -> ``BLPlotter`` path, for going
        straight from parameters to a plot. ``tli_kwargs`` (xs_mix, interp_scheme, param_format,
        delta, tau, fit_or_pred, solid_solutions, ss_kwargs, temp_slider, T_incr, hull_mode,
        imputed_entries, include_imputed, user_entries, excluded_entries, temp_range)
        go to the interpolation, built directly in ``order``; it is left un-interpolated
        (``process_data`` / ``get_plot`` interpolate lazily). ``edge_mape`` and ``pred_liquidus``
        go to the plotter.
        """
        ti = TernaryLiquidInterpolation(components, data_dir, order=order, **tli_kwargs)
        return cls(ti, order=order, edge_mape=edge_mape, pred_liquidus=pred_liquidus)

    # Read-only views of the held model: ``plotter.<name>`` reads ``plotter._ti.<name>`` for every
    # name in _MODEL_VIEWS.
    _MODEL_VIEWS = (
        "components",
        "binary_systems",
        "solid_solutions",
        "ss_kwargs",
        "ss_models",
        "temp_slider",
        "T_incr",
        "hull_mode",
        "_n_simplex_vertices",
        "sys_name",
        "phase_names",
        "color_map",
        "conds",
        "T_grid",
        "temp_range",
        "df_Tgroups",
        "equil_df_list",
    )

    @property
    def hsx_df(self):
        """The held model's processed hull table (``proc_df``)."""
        return self._ti.proc_df

    def _init_sys(self, progress=None):
        """Delegate data-prep to the model, then collect the binary edge figures.

        Fixed order, which the plotter's output depends on: model init, then
        ``bin_fig_list`` (when the edges are parameterized), then the completion print.
        ``progress`` is called as in :meth:`process_data`.
        """
        self._ti._init_sys()
        if progress is not None:
            progress("model", 1, 1)
        self.bin_fig_list = (
            build_binary_edge_figures(
                self._ti,
                ternary_color_map=self.color_map,
                edge_mape=self.edge_mape,
                pred_liquidus=self.pred_liquidus,
                progress=progress,
            )
            if self._ti.fit_or_pred and self._ti.xs_mix
            else []
        )
        logger.info("Initialization complete")

    def impute_phase(self, composition, temperature: float, **kwargs):
        """:meth:`TernaryLiquidInterpolation.impute_phase` on the plotted model."""
        return self._ti.impute_phase(composition, temperature, **kwargs)

    def adjust_phase_temperature(self, phase: str, temperature: float, **kwargs):
        """:meth:`TernaryLiquidInterpolation.adjust_phase_temperature` on the plotted model."""
        return self._ti.adjust_phase_temperature(phase, temperature, **kwargs)

    def process_data(self, progress=None):
        """Run the model's hull pipeline (see :meth:`TernaryLiquidInterpolation.process_data`),
        with the plotter's `_init_sys` supplying the binary edge figures.

        Args:
            progress: called as ``progress(stage, done, total)``: ``("model", 1, 1)`` once the
                model is prepared, ``("edges", k, n)`` after each binary edge figure, and
                ``("hull", 1, 1)`` once the hull is evaluated.
        """
        self._init_sys(progress=progress)
        if self.hull_mode == "hsx":
            self._ti._process_data_hsx()
        else:
            self._ti._eval_hull()
        if progress is not None:
            progress("hull", 1, 1)

    def get_plot(
        self,
        plot_type: str = "tx",
        *,
        highlight_phase: str | None = None,
        interactive_legend: bool | None = None,
        **kwargs,
    ) -> go.Figure:
        """Generate a ternary plot (mirrors ``BLPlotter.get_plot``).

        Args:
            plot_type: 'tx' (default) — the ternary liquidus (T-x) phase-diagram surface;
                'ch' — a single-temperature lower convex-hull slice (requires ``T_celsius=``).
            highlight_phase: 'tx' only. A solid phase of the legend whose primary-crystallisation
                field (the part of the liquidus where it is the first solid to form) is drawn in
                full colour and outlined, with the rest of the liquidus, the other legend entries
                and the other phases' lines dimmed.
            interactive_legend: 'tx' only, on unless ``False``. Embed the highlighted view of
                every legend phase in the figure, so that clicking a legend entry in a browser
                switches to it.
            kwargs: forwarded to the underlying builder (``T_celsius`` for 'ch').

        Returns:
            go.Figure: the generated Plotly figure. A 'tx' figure is a
            :class:`gliquid.plotting.ternary_surface.LiquidusFigure`: ``show()``, ``write_html``
            and a bare ``fig`` in a notebook cell run its browser script, which handles legend
            clicks and turns the flat labels of the base triangle with the camera so they read
            upright from the edge lowest on screen.
            :func:`gliquid.plotting.ternary_labels.orient_labels` re-fits them to a camera set in
            Python. Where fewer than three liquidus nodes off one line lie at or below the top of
            the temperature window, a WARNING says so; the figure is still drawn, and without a
            liquidus mesh where the nodes the hull returned do not span the triangle.

        Raises:
            ValueError: an unknown ``plot_type`` or ``highlight_phase``, a highlight option with
                'ch', or 'ch' without ``T_celsius``.
            TypeError: an unknown keyword for 'tx'.
        """
        valid_plot_types = ["tx", "ch"]
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}"
            )
        if plot_type == "tx":
            if kwargs:
                raise TypeError(f"get_plot('tx') got unexpected keywords: {sorted(kwargs)}")
            if not hasattr(self, "equil_df_list"):
                self.process_data()
            return self._plot_tx(
                highlight_phase=highlight_phase, interactive_legend=interactive_legend
            )
        if highlight_phase is not None or interactive_legend:
            raise ValueError("highlight_phase and interactive_legend apply to get_plot('tx') only.")
        # 'ch' — single-temperature lower-hull slice
        if "T_celsius" not in kwargs:
            raise ValueError("get_plot('ch', ...) requires a T_celsius=<float> keyword.")
        return self._plot_convex_hull(kwargs["T_celsius"])

    def show(self, plot_type: str = "tx", **kwargs) -> None:
        """Display the generated plot (see :meth:`get_plot` for plot_type / kwargs)."""
        self.get_plot(plot_type, **kwargs).show()

    def write_html(self, plot_type: str, file, **kwargs) -> None:
        """Save the generated plot as a standalone interactive HTML file; ``kwargs`` are
        forwarded to :meth:`get_plot`."""
        self.get_plot(plot_type, **kwargs).write_html(file)

    def write_image(
        self,
        plot_type: str,
        stream,
        image_format: str = "svg",
        export_timeout_s: float = 120.0,
        **kwargs,
    ) -> None:
        """Save the generated plot as an image (mirrors ``BLPlotter.write_image``).

        ``kwargs`` are forwarded to :meth:`get_plot` (e.g. ``T_celsius=`` for 'ch'); the format
        is inferred from a path ``stream`` that carries an extension.
        """
        fig = self.get_plot(plot_type, **kwargs)
        if isinstance(stream, str) and "." in stream:
            image_format = stream.split(".")[-1]
        plot_export.write_image_with_timeout(
            fig, stream, timeout_s=export_timeout_s, format=image_format
        )

    def get_convex_hull(self, T_celsius: float) -> dict:
        """Extract a single G-x0-x1 lower convex-hull slice at the nearest grid temperature.

        Returns the hull DATA -- points, simplices, per-simplex and transformed-point frames,
        and the exact grid temperature the request snapped to. For the rendered 3-D slice use
        ``get_plot('ch', T_celsius=...)``; the computation lives on the model
        (:meth:`TernaryLiquidInterpolation.get_convex_hull`).
        """
        if not hasattr(self._ti, "df_Tgroups") or not hasattr(self._ti, "T_grid"):
            self._init_sys()  # the plotter path also builds bin_fig_list
        return self._ti.get_convex_hull(T_celsius)

    def _plot_convex_hull(self, T_celsius: float) -> go.Figure:
        """The 3-D single-slice lower-hull figure for ``get_plot('ch', T_celsius=...)``
        (draw body in gliquid.plotting.ternary_surface)."""
        return ternary_surface.render_hull_slice(self.get_convex_hull(T_celsius), self.components)

    def _plot_tx(self, highlight_phase=None, interactive_legend=None):
        self.plotting_df = pd.concat(self.equil_df_list)

        # Each facet's solid assemblage keyed by the globally unique simplex_id, from the raw
        # frames.
        assemblage_by_simplex = (
            self.plotting_df[self.plotting_df["Phase"] != "L"]
            .groupby("simplex_id")["Phase"]
            .apply(lambda phases: frozenset(str(phase) for phase in phases))
            .to_dict()
        )

        self.liq_plotting_df = self.plotting_df[self.plotting_df["Phase"] == "L"]
        self.solid_plotting_df = self.plotting_df[self.plotting_df["Phase"] != "L"]
        # Deduped per phase, so every polymorph at an element vertex keeps its row.
        self.solid_plotting_df = self.solid_plotting_df.sort_values("T").drop_duplicates(
            subset=["x0", "x1", "Phase"], keep="last"
        )
        # Phases whose stability ends below the plotted window are dropped.
        self.solid_plotting_df = self.solid_plotting_df[
            self.solid_plotting_df["T"] >= self.conds[0]
        ]
        solids = set(self.solid_plotting_df["Phase"].tolist())
        solids = [str(x) for x in solids]

        # Each phase is a vertical segment up to its stability ceiling, starting at the plot floor
        # or at the ceiling of the phase below it.
        floor_rows = []
        for (x0, x1), grp in self.solid_plotting_df.groupby(["x0", "x1"], sort=False):
            # conds[0] is the bottom of the temperature grid in Celsius.
            base = self.conds[0]
            for _, row in grp.sort_values("T").iterrows():
                floor_rows.append(
                    {"x0": x0, "x1": x1, "T": base, "Phase": row["Phase"], "Colors": row["Colors"]}
                )
                base = row["T"]
        if floor_rows:
            self.solid_plotting_df = pd.concat(
                [self.solid_plotting_df, pd.DataFrame(floor_rows)], ignore_index=True
            )

        # The liquidus row at a composition is the coldest there; its assemblage is the union over
        # every facet meeting that vertex at that temperature.
        liquid = self.liq_plotting_df
        at_liquidus = liquid[liquid["T"] == liquid.groupby(["x0", "x1"])["T"].transform("min")]
        node_assemblage = (
            at_liquidus.groupby(["x0", "x1"])["simplex_id"]
            .apply(
                lambda ids: frozenset().union(
                    *(assemblage_by_simplex.get(simplex, frozenset()) for simplex in ids)
                )
            )
            .to_dict()
        )
        self.liq_plotting_df = (
            at_liquidus.assign(
                _no_assemblage=~at_liquidus["simplex_id"].isin(set(assemblage_by_simplex))
            )
            .sort_values(["T", "_no_assemblage"])
            .drop_duplicates(subset=["x0", "x1"], keep="first")
            .drop(columns="_no_assemblage")
        )
        self.liq_plotting_df["coexistent_phases"] = [
            ", ".join(sorted(node_assemblage[node]))
            for node in zip(self.liq_plotting_df["x0"], self.liq_plotting_df["x1"])
        ]

        nodes = self.liq_plotting_df[["x0", "x1"]].to_numpy(float)
        if _spans_plane(nodes):
            self.triangulation = Delaunay(nodes)
            triangles = self.triangulation.simplices
        else:
            self.triangulation = None
            triangles = np.empty((0, 3), dtype=int)

        self.plotting_df = pd.concat([self.solid_plotting_df, self.liq_plotting_df])
        partition = self._liquidus_partition(triangles) if len(triangles) else None
        in_window = self.liq_plotting_df["T"].to_numpy(float) <= self.conds[1]
        if not _spans_plane(nodes[in_window]):
            logger.warning(
                f"{self.sys_name}: {int(in_window.sum())} liquidus nodes lie at or below the top "
                f"of the temperature window ({self.conds[1]:.1f} °C), too few to span the "
                "composition triangle; raise the top of temp_range to show the liquidus."
            )
        elif partition is None and (highlight_phase is not None or interactive_legend):
            logger.warning(f"{self.sys_name}: the liquidus has no phase fields to highlight.")

        # The drawn window reaches the partition's hottest vertex unless temp_range pins its top.
        conds = list(self.conds)
        pinned_high = (self._ti._temp_range_arg or (None, None))[1]
        if partition is not None and pinned_high is None:
            top = float(np.max(partition.surface.T))
            if top > conds[1]:
                conds[1] = self._ti._pad_ceiling(top + 273.15) - 273.15

        return ternary_surface.render_tx_surface(
            self.solid_plotting_df,
            self.liq_plotting_df,
            triangles,
            self.color_map,
            self.components,
            conds,
            imputed_phases=self._ti.imputed_phase_names(),
            user_phases=self._ti.user_phase_names(),
            partition=partition,
            highlight_phase=highlight_phase,
            interactive_legend=interactive_legend is not False,
        )

    def _liquidus_partition(self, triangles) -> liquidus_fields.Partition | None:
        """The liquidus surface split into primary-phase fields by the liquidus sheets, with a
        liquid miscibility gap as the two-liquid field, or ``None``, with a WARNING naming the
        system and the share of nodes rejected, where the sheets do not describe this hull's
        liquidus.

        The sheets are rejected when one is not finite at a node, or when a node's hull liquidus
        lies above them, or above the bottom of the temperature window where they lie below it,
        by more than :data:`SHEET_GUARD_K`, or when the partition's surface is not
        finite at one of its vertices.

        The result is cached on the plotter until the model's processed output
        (``equil_df_list``), ``triangles``, the temperature window or the hull mode changes.
        """
        key = (np.asarray(triangles).tobytes(), tuple(self.conds), self.hull_mode)
        processed, cached_key, partition = self._partition_cache
        if processed is not self.equil_df_list or cached_key != key:
            partition = self._build_partition(triangles)
            self._partition_cache = (self.equil_df_list, key, partition)
        return partition

    def _build_partition(self, triangles) -> liquidus_fields.Partition | None:
        liq = self.liq_plotting_df
        sheets = LiquidusSheets.from_interpolation(self._ti)
        comp = liq[["x0_orig", "x1_orig"]].to_numpy(float)
        liquidus = sheets.liquidus(comp) - 273.15
        bound = np.maximum(liquidus, self.conds[0])
        rejected = ~np.isfinite(liquidus) | (liq["T"].to_numpy(float) > bound + SHEET_GUARD_K)
        if rejected.any():
            logger.warning(
                f"{self.sys_name}: the liquidus sheets disagree with the hull at "
                f"{int(rejected.sum())} of {len(liq)} nodes ({rejected.mean():.1%}); the liquidus "
                "is drawn without phase fields."
            )
            return None
        partition = liquidus_fields.partition_fields(
            liq[["x0", "x1"]].to_numpy(float),
            comp,
            triangles,
            lambda points: sheets(points) - 273.15,
            sheets.phases,
            sheet_columns=lambda points, which: sheets.columns(points, which) - 273.15,
        )
        unbounded = ~np.isfinite(partition.surface.T)
        if unbounded.any():
            logger.warning(
                f"{self.sys_name}: the liquidus sheets are not finite at {int(unbounded.sum())} of "
                f"{len(unbounded)} surface vertices; the liquidus is drawn without phase fields."
            )
            return None
        return partition

    def get_inter_melting_temps(self, interphases_for_melting: list[str]):
        """Max stability temperature per requested phase (computation on the model)."""
        return self._ti.get_inter_melting_temps(interphases_for_melting)


for _name in TLIPlotter._MODEL_VIEWS:
    setattr(TLIPlotter, _name, property(operator.attrgetter(f"_ti.{_name}")))
