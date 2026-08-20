"""
Authors: Abrar Rauf, Joshua Wilwerth
Last Modified: March 16 2026
Description: This module contains the classes for ternary interpolation and ternary phase diagram plotting.
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""

import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from tqdm import tqdm

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as cfg
import gliquid.plotting.export as plot_export
import gliquid.plotting.ternary_surface as ternary_surface
import gliquid.solution as solution
from gliquid.binary import BinaryLiquid
from gliquid.hsx import HSX, lower_convex_hull
from gliquid.phase import UNARY, resolve_component_order, validate_and_format_system
from gliquid.plotting.binary_tx import build_polymorph_transitions, plot_tx
from gliquid.plotting.style import build_phase_color_map
from gliquid.solution import DEFAULT_TAU, RKPolyExp, SolutionModel, t_sym

logger = logging.getLogger(__name__)


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


def generate_comp_grid(delta=0.025, atol=1e-6):
    # generate composition grid for ternary system
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
    return {"A": x_A, "B": x_B, "C": x_C}


class TernaryLiquidInterpolation:
    """Ternary liquid surface interpolated from three binary-edge mixing models.

    Fields mirror ``BinaryLiquid`` where they carry the same information:
    ``components`` (three elements, sorted), ``xs_mix`` (per-edge
    ``RKPolyExp`` models — the ternary counterpart of the binary's single
    model), ``param_format``, ``tau``, ``interp_scheme`` (linear/Muggianu/Kohler
    pair geometry). Construct either with an explicit per-edge ``xs_mix`` dict or
    from three fitted ``BinaryLiquid`` systems via ``from_binaries``.
    """

    def __init__(
        self,
        components: list[str],
        data_dir: cache.CacheBackend | Path | str | None = None,
        *,
        delta: float = 0.025,
        interp_scheme: str = "linear",
        param_format: str = "linear",
        tau: float = DEFAULT_TAU,
        xs_mix: dict | None = None,
        ternary_l0: float = 0.0,
        fit_or_pred: dict | None = None,
        solid_solutions: bool = False,
        ss_kwargs: dict | None = None,
        temp_slider=(0, 0),
        T_incr: int = 10,
        hull_mode: str = "gtx",
        order: str | list | None = None,
    ):
        validate_and_format_system(list(components))  # compound components raise NIE here
        # Construction order is authoritative (matches BinaryLiquid): this class keeps
        # the components exactly as given (order=None == 'given'). The GTX plotter
        # subclass defaults order='alphabetical' instead — presentation classes
        # alphabetize unless told otherwise; mixing keys must match the RESOLVED
        # order's cyclic edges.
        self.components = resolve_component_order(
            order if order is not None else "given", components
        )
        self.binary_systems = ordered_binary_systems(self.components)
        # A cache BACKEND, not a path: a single-file store cannot be named by a directory,
        # and this attribute is threaded straight through to ``api.get_dft_convexhull``.
        # ``cfg.cache_dir`` is passed positionally when the caller gave nothing, which keeps
        # the historical semantics exactly — the ternary cache has always been FLAT inside
        # whatever directory it was handed, even when config.dir_structure is 'nested'.
        self.data_dir = cache.resolve_backend(data_dir if data_dir is not None else cfg.cache_dir)
        self.delta = delta
        self.comp_grid = generate_comp_grid(self.delta)
        self.interp_scheme = interp_scheme
        self.param_format = param_format
        # exp(-T/tau) decay constant for the 'combined'/'comb-exp' mixing forms — an
        # instance attribute mirroring BinaryLiquid.tau.
        self.tau = tau
        self.fit_or_pred = fit_or_pred or {}  # dict of 'fit' or 'pred' per binary system

        # Excess-mixing models keyed by the cyclic ``binary_systems`` pair names, plus an
        # optional 'A-B-C' TRIPLET entry (the ternary interaction term, part of the
        # solution model — not a separate field). Plain parameter lists coerce through
        # (param_format, tau) for pairs and the 'regular' format for the triplet.
        def _coerce(key, val):
            if isinstance(val, RKPolyExp):
                return val
            if key.count("-") == 2:
                return RKPolyExp("regular", list(val))
            return RKPolyExp(self.param_format, list(val), tau=self.tau)

        self.xs_mix: dict[str, RKPolyExp] = {
            key: _coerce(key, val) for key, val in (xs_mix or {}).items()
        }
        # ternary_l0= is folding sugar for the triplet entry (an explicit mixing entry wins)
        if ternary_l0:
            self.xs_mix.setdefault(
                "-".join(self.components), RKPolyExp("regular", [float(ternary_l0)])
            )
        self.ternary_meta = {}
        # Solid-solution support (opt-in): one n-ary load_solid_solution_models pass,
        # then a per-phase SolutionModel surface on the same composition grid.
        self.solid_solutions = solid_solutions
        self.ss_kwargs = ss_kwargs
        self.ss_models = {}
        # Hull-evaluation config (computation-relevant, not cosmetic): the extra
        # temperature margin, the gtx grid increment, and the hull algorithm choice.
        self.temp_slider = list(temp_slider)
        self.T_incr = T_incr
        # 'gtx' (default): per-temperature (x0, x1, G) lower hulls -- the pinned behavior.
        # 'hsx': ONE (x0, x1, S, H) lower hull via the generalized HSX class, with facet
        #        temperatures extracted analytically (no T-grid discretization).
        self.hull_mode = hull_mode
        if self.hull_mode not in ("gtx", "hsx"):
            raise ValueError(
                "hull_mode must be 'gtx' (per-temperature slicing) or "
                "'hsx' (one N-D hull through gliquid.hsx.HSX)."
            )
        self._n_simplex_vertices = 3  # gtx facets are triangles; the hsx path sets 4

    @classmethod
    def from_binaries(cls, binaries, *, components=None, **kwargs) -> "TernaryLiquidInterpolation":
        """Construct from three parameterized ``BinaryLiquid`` edge systems.

        Derives the three components (alphabetical by default; pass ``components=`` to
        choose the construction order — it must be a permutation of the spanned
        elements), re-orients each edge's mixing model onto the cyclic
        ``binary_systems`` convention (odd RK orders flip when a stored pair maps onto
        an inverted cyclic edge), and inherits ``param_format``/``tau`` (which must
        agree across the three systems). ``solid_solutions`` defaults on when any
        binary carries ss_models; further constructor kwargs pass through.
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
        if components is not None:
            elements = list(components)
            if len(elements) != 3 or set(elements) != spanned:
                raise ValueError(
                    f"components= must be a permutation of the spanned elements "
                    f"{sorted(spanned)}, got {elements}."
                )
            # honor the override even on plotter subclasses whose order defaults to
            # 'alphabetical' (an explicit order= kwarg still wins)
            kwargs.setdefault("order", "given")
        else:
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
        kwargs.setdefault("solid_solutions", any(bl.ss_models for bl in binaries))
        return cls(elements, xs_mix=xs_mix, **kwargs)

    def with_component_order(self, order) -> "TernaryLiquidInterpolation":
        """A copy of this system re-framed onto ``order`` (any spec
        ``resolve_component_order`` accepts). Returns ``self`` when the order already matches.

        Mirrors ``BinaryLiquid.with_component_order``: the per-edge mixing models reorient
        onto the new cyclic ``binary_systems`` convention (odd RK orders flip when a stored
        pair maps onto an inverted cyclic edge), the optional 'A-B-C' triplet interaction key
        re-joins (that term is symmetric, so its RK is unchanged), and ``fit_or_pred`` re-keys
        by element pair. Interpolated/derived state (hsx_df, ref_data, ss_models,
        ternary_meta) is NOT carried over — it re-derives on interpolate().
        """
        order = resolve_component_order(order, self.components)
        if list(order) == list(self.components):
            return self

        old_by_pair = {
            frozenset(k.split("-")): (k, rk) for k, rk in self.xs_mix.items() if k.count("-") == 1
        }
        new_xs_mix = {}
        for edge in ordered_binary_systems(order):
            a, b = edge.split("-")
            old_key, rk = old_by_pair[frozenset((a, b))]
            rk = RKPolyExp(rk.format, rk.values, tau=rk.tau)
            if old_key != edge:  # stored under the inverse cyclic orientation -> flip odd orders
                rk = rk.swapped()
            new_xs_mix[edge] = rk
        old_trip = "-".join(self.components)
        if old_trip in self.xs_mix:  # symmetric triplet term: same RK, new join key
            trip = self.xs_mix[old_trip]
            new_xs_mix["-".join(order)] = RKPolyExp(trip.format, trip.values, tau=trip.tau)

        new_fit_or_pred = {}
        for edge in ordered_binary_systems(order):
            pair = frozenset(edge.split("-"))
            for old_edge, val in self.fit_or_pred.items():
                if frozenset(old_edge.split("-")) == pair:
                    new_fit_or_pred[edge] = val

        return TernaryLiquidInterpolation(
            list(order),
            self.data_dir,
            delta=self.delta,
            interp_scheme=self.interp_scheme,
            param_format=self.param_format,
            tau=self.tau,
            xs_mix=new_xs_mix,
            fit_or_pred=new_fit_or_pred,
            solid_solutions=self.solid_solutions,
            ss_kwargs=self.ss_kwargs,
            temp_slider=self.temp_slider,
            T_incr=self.T_incr,
            hull_mode=self.hull_mode,
            order="given",
        )

    def init_ref_data(self):
        # initialize reference data for fusion enthalpies and entropies
        tern_enthalpy = np.array([UNARY[el].h_liq for el in self.components])
        tern_entropy = np.array([UNARY[el].s_liq for el in self.components])
        tern_temp = np.array([UNARY[el].t_fusion for el in self.components])
        self.ref_data = {"H": tern_enthalpy, "S": tern_entropy, "T": tern_temp}

    def interpolate_liquid_surface(self):
        # interpolate the ternary system using the binary-edge mixing models
        x_B, x_C = self.comp_grid["B"], self.comp_grid["C"]

        self.init_ref_data()

        if not all(sys in self.xs_mix for sys in self.binary_systems):
            raise ValueError(
                "Mixing models for the binary systems are incomplete. Provide a "
                "per-edge 'xs_mix' dict or construct via from_binaries()."
            )

        # One SolutionModel over the three cyclic edge pairs (+ the optional 'A-B-C'
        # triplet interaction entry): per-edge RKPolyExp models, reference G(T)
        # from the unary registry, and the interpolation geometry resolved by scheme.
        refs = [h - t_sym * s for h, s in zip(self.ref_data["H"], self.ref_data["S"])]
        interactions = {sys: self.xs_mix[sys] for sys in self.binary_systems}
        trip_key = "-".join(self.components)
        if trip_key in self.xs_mix:
            interactions[trip_key] = self.xs_mix[trip_key]
        model = SolutionModel(
            tuple(self.components),
            refs,
            interactions,
            interp_scheme=self.interp_scheme,
            ideal="safe",
        )

        tm_mean = np.mean(
            self.ref_data["T"]
        )  # mean melting point in ternary - used for t-dependent H and S forms
        with np.errstate(divide="ignore", invalid="ignore"):
            h_vals_mesh, s_vals_mesh = model.h_s_grid((x_B, x_C), tm_mean)

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
        sharing one phase name). Covers the full simplex including the binary edges.

        Models come from ONE n-ary ``load_solid_solution_models`` pass over the three
        elements (the retired per-edge load + merge produced identical values; for
        ``ref_mode='from_dft_entries'`` the caller must supply ``entries`` via ``ss_kwargs``).
        """
        if not self.ss_models:
            self.ss_models = solution.load_solid_solution_models(
                self.components, UNARY.component_data(self.components), **(self.ss_kwargs or {})
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

    def get_ternary_form_en(self, sys):
        # get the formation energies of the stable phases in the ternary system
        tern_mp_dict = {}
        # Shared n-component DFT cache/loader (api.py); the instance's own data_dir keeps
        # winning (flat layout inside it — the historical ternary cache convention).
        pdia, _ = api.get_dft_convexhull(list(sys), "GGA", data_dir=self.data_dir)
        self.ternary_meta["n_ternary_compounds"] = sum(
            1 for e in pdia.stable_entries if len(api.entry_original(e).composition.elements) == 3
        )
        entries = pdia.stable_entries
        all_atm_fracs = []
        all_form_ens = []
        phases = []
        for entry in entries:
            form_en = pdia.get_form_energy_per_atom(entry)
            all_form_ens.append(form_en * 96485)
            all_atm_fracs.append(list(api.entry_frac_along(pdia, entry, list(sys))))
            phases.append(api.entry_display_name(entry))

        all_atm_fracs_arr = np.array(all_atm_fracs)

        for i, arr in enumerate(all_atm_fracs_arr.T):
            tern_mp_dict[f"x{i}"] = arr

        self.ternary_meta["deepest_formation_energy"] = min(all_form_ens)
        tern_mp_dict["H"] = all_form_ens
        tern_mp_dict["Phase Name"] = phases

        entropy = [0] * len(all_form_ens)
        tern_mp_dict["S"] = entropy

        tern_mp_df = pd.DataFrame(tern_mp_dict)
        tern_mp_df = tern_mp_df[["x0", "x1", "S", "H", "Phase Name"]]
        tern_mp_df = tern_mp_df.loc[tern_mp_df.groupby("Phase Name")["H"].idxmin()]

        return tern_mp_df

    def interpolate(self):
        # create the hsx dataframe for the ternary system
        self.interpolate_liquid_surface()  # populates self.hsx_df with ternary liquid phase data
        if self.solid_solutions:
            self.append_solid_solution_surfaces()  # appends per-phase SS Gibbs surfaces
        # self.bin_fig_list = self._add_binary_data()
        self.tern_mp_df = self.get_ternary_form_en(self.components)
        self.hsx_df = pd.concat([self.hsx_df, self.tern_mp_df], ignore_index=True)
        self.hsx_df = self.hsx_df.drop_duplicates()
        self.hsx_df = self.hsx_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Hull-evaluation pipeline: every step below is plotter-free -- callable on the
    # model directly.
    # ------------------------------------------------------------------

    def _init_sys(self, color_map=None):
        """Prepare hull-evaluation inputs on a processed copy (``proc_df``).

        Interpolates lazily, then rounds/relabels/colors a COPY of ``hsx_df`` (the
        pristine interpolated frame is never mutated), computes the temperature grid,
        and (gtx mode) slices per-T Gibbs frames into ``df_Tgroups``. ``color_map``
        overrides the palette; the default is the shared build_phase_color_map scheme.
        """
        if not hasattr(self, "hsx_df"):
            self.interpolate()
        self.proc_df = self.hsx_df.copy()
        self.sys_name = "-".join(sorted(self.components))
        self.phase_names = self.proc_df["Phase Name"].unique().tolist()

        if color_map is not None:
            self.color_map = dict(color_map)
        else:
            # The same builder the binary plots use -- PHASE_PALETTE line phases, reserved
            # SS colors, 'L' cornflowerblue -- so binary and ternary figures agree on color.
            self.color_map = build_phase_color_map(self.phase_names, ss_names=list(self.ss_models))

        tern_temp = self.ref_data["T"]
        max_temp = round(np.max(tern_temp) + 200)
        min_temp = round(np.min(tern_temp))
        self.conds = [
            np.min(np.array([0, min_temp - 200])) - self.temp_slider[0],
            max_temp + self.temp_slider[1],
        ]
        self.T_grid = np.arange(self.conds[0], self.conds[1] + self.T_incr, self.T_incr)
        self.proc_df["x0"] = self.proc_df["x0"].round(4)
        self.proc_df["x1"] = self.proc_df["x1"].round(4)
        self.proc_df = self.proc_df.rename(columns={"Phase Name": "Phase"})
        self.proc_df["Colors"] = self.proc_df["Phase"].map(self.color_map)

        self.df_Tgroups = {}
        if self.hull_mode == "gtx":
            for T in self.T_grid:
                self.proc_df["G"] = self.proc_df["H"] - T * self.proc_df["S"]
                self.df_Tgroups[T] = self.proc_df[["x0", "x1", "G", "Phase", "Colors"]].copy()

    def _process_data_hsx(self):
        """One (x0, x1, S, H) lower hull with analytic facet temperatures (hull_mode='hsx').

        The generalized HSX replaces the per-temperature G-slicing: coexistence
        temperatures come from facet hyperplane normals, so they are continuous instead
        of T-grid multiples. Output schema matches the gtx path (equil_df_list rows of
        [x0, x1, T(C), Phase, Colors, simplex_id] in ternary Cartesian coordinates).
        """
        start_time = time.time()
        table = self.proc_df[["x0", "x1", "S", "H", "Phase"]]
        hsx_obj = HSX(
            {"data": table, "phases": self.phase_names, "comps": self.components},
            conds=list(self.conds),
        )
        df_tx = hsx_obj.compute_tx()[0]

        temp_df = df_tx.rename(columns={"t": "T", "label": "Phase"})
        temp_df["Colors"] = temp_df["Phase"].map(self.color_map)  # keep this system's palette
        temp_df["simplex_id"] = np.repeat(np.arange(len(temp_df) // 4), 4)
        temp_df = temp_df[["x0", "x1", "T", "Phase", "Colors", "simplex_id"]]

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

    def _eval_hull(self):
        """Per-temperature lower-hull evaluation over ``df_Tgroups`` (hull_mode='gtx')."""
        start_time = time.time()
        self.equil_df_list = []
        shifter = 0
        for T in tqdm(self.T_grid, desc="Evaluating lower hull over temperature intervals"):
            if T < self.conds[0]:
                continue
            points = np.array(self.df_Tgroups[T][["x0", "x1", "G"]])
            simplices = lower_convex_hull(points, vertical_simplices=False)
            simplex_vertices = []
            for simplex in simplices:
                simplex_vertices.append(points[simplex])

            final_phases = []
            for simplex in simplices:
                phase1 = self.df_Tgroups[T].loc[simplex[0], "Phase"]
                phase2 = self.df_Tgroups[T].loc[simplex[1], "Phase"]
                phase3 = self.df_Tgroups[T].loc[simplex[2], "Phase"]

                phase_arr = np.array([phase1, phase2, phase3])
                final_phases.append(phase_arr)

            data = []
            last_val = 0
            for i, simplex in enumerate(simplices):
                labels = final_phases[i]
                if len(set(labels)) == 0:
                    continue
                else:
                    x0_coords = [points[vertex][0] for vertex in simplex]
                    x1_coords = [points[vertex][1] for vertex in simplex]
                    t_val = T

                j = 0
                for x0, x1 in zip(x0_coords, x1_coords):
                    label = labels[j]
                    color = self.color_map[label]
                    data.append([x0, x1, t_val, label, color, shifter + i])
                    j += 1

                last_val = i

            shifter += last_val + 1

            temp_df = pd.DataFrame(data, columns=["x0", "x1", "T", "Phase", "Colors", "simplex_id"])

            # Store original coordinates before transformation
            temp_df["x0_orig"] = temp_df["x0"].copy()
            temp_df["x1_orig"] = temp_df["x1"].copy()

            temp_df = cartesian_to_ternary(temp_df)
            temp_df["T"] = temp_df["T"] - 273.15

            self.equil_df_list.append(temp_df)

        end_time = time.time()
        logger.info(
            f"Lower hull evaluation and post processing time:: {end_time - start_time} seconds for temperature increment of {self.T_incr}"
        )

    def process_data(self, color_map=None):
        """The full hull pipeline on the model -- interpolate, prepare, evaluate -- with
        no plotter involved. ``color_map`` is forwarded to :meth:`_init_sys`."""
        self._init_sys(color_map=color_map)
        if self.hull_mode == "hsx":
            self._process_data_hsx()
            return
        self._eval_hull()

    def get_convex_hull(self, T_celsius: float) -> dict:
        """Extract a single G-x0-x1 lower convex-hull slice at the nearest grid temperature.

        Returns the hull DATA -- points, simplices, per-simplex and transformed-point
        frames, and the exact grid temperature the request snapped to.
        """
        if self.hull_mode == "hsx":
            raise ValueError(
                "get_convex_hull is a GTX-mode diagnostic (per-T slices); "
                "construct the plotter with hull_mode='gtx'."
            )
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

        slice_df = self.df_Tgroups[T_kelvin].copy().reset_index(drop=True)
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


def build_binary_edge_figures(ti: "TernaryLiquidInterpolation", ternary_color_map=None):
    """Build one TX figure per parameterized binary edge of ``ti`` (ex-TLI._add_binary_data).

    A model-layer act (constructs BinaryLiquid edge systems), so it lives here rather
    than in gliquid.plotting; ``TLIPlotter._init_sys`` collects the result as
    ``bin_fig_list``.
    """
    bin_fig_list = []

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
        )
        sys.update_phase_points()
        fit_type = ti.fit_or_pred.get(sys_name, "pred")
        polymorph_transitions = build_polymorph_transitions(sys)
        if fit_type == "fit":
            figr = plot_tx(
                sys.hsx,
                digitized_liquidus=sys.digitized_liq,
                polymorph_transitions=polymorph_transitions,
                ternary_color_map=ternary_color_map,
                ss_phases=set(sys.ss_models),
            )
        else:
            figr = plot_tx(
                sys.hsx,
                pred=True,
                polymorph_transitions=polymorph_transitions,
                ternary_color_map=ternary_color_map,
                ss_phases=set(sys.ss_models),
            )
        bin_fig_list.append(figr)

    for sys_name in ti.binary_systems:
        if sys_name in ti.xs_mix:
            process_system(sys_name)

    return bin_fig_list


class TLIPlotter:
    """Plotter over a :class:`TernaryLiquidInterpolation`'s liquid + solid surfaces.

    Wraps a ternary interpolation the way :class:`~gliquid.binary.BLPlotter` wraps a
    ``BinaryLiquid``: the model is HELD (``self._ti``) rather than inherited, and re-framed
    onto the presentation ``order`` (default 'alphabetical'). Build from an existing
    interpolation with ``TLIPlotter(ti, ...)``, or straight from parameters with
    ``TLIPlotter.from_components(...)``.
    """

    def __init__(
        self,
        ternary_interp: "TernaryLiquidInterpolation",
        order="alphabetical",
        *,
        temp_slider=(0, 0),
        T_incr: int = 10,
        hull_mode: str = "gtx",
    ):
        self._ti_raw = ternary_interp
        # Presentation frame: re-frame the held model onto `order` (a no-op when it already
        # matches), mirroring BLPlotter's self._bl = bl.with_component_order(order).
        self._ti = ternary_interp.with_component_order(order)
        # Hull-evaluation config lives on the model. Explicit plotter args win over
        # whatever the model carries, and _n_simplex_vertices is reset fresh per plotter.
        if hull_mode not in ("gtx", "hsx"):
            raise ValueError(
                "hull_mode must be 'gtx' (per-temperature slicing) or "
                "'hsx' (one N-D hull through gliquid.hsx.HSX)."
            )
        self._ti.temp_slider = list(temp_slider)
        self._ti.T_incr = T_incr
        self._ti.hull_mode = hull_mode
        self._ti._n_simplex_vertices = 3

    @classmethod
    def from_components(
        cls,
        components,
        data_dir=None,
        *,
        order="alphabetical",
        temp_slider=(0, 0),
        T_incr: int = 10,
        hull_mode: str = "gtx",
        **tli_kwargs,
    ) -> "TLIPlotter":
        """Build the :class:`TernaryLiquidInterpolation` and wrap it in one call.

        Convenience mirror of the ``BinaryLiquid.from_cache`` -> ``BLPlotter`` path, for going
        straight from parameters to a plot. ``tli_kwargs`` (xs_mix, interp_scheme, param_format,
        delta, tau, fit_or_pred, solid_solutions, ss_kwargs, ternary_l0) go to the interpolation,
        built directly in ``order`` (so its cyclic edge keys match); it is left un-interpolated
        (``process_data`` / ``get_plot`` interpolate lazily).
        """
        ti = TernaryLiquidInterpolation(components, data_dir, order=order, **tli_kwargs)
        return cls(ti, order=order, temp_slider=temp_slider, T_incr=T_incr, hull_mode=hull_mode)

    # Read-only views onto the wrapped interpolation. BLPlotter reads self._bl.X explicitly;
    # the ternary plotter's methods and external callers read these off the plotter, so expose
    # them as passthrough properties (the held model owns the state).
    @property
    def components(self):
        return self._ti.components

    @property
    def binary_systems(self):
        return self._ti.binary_systems

    @property
    def solid_solutions(self):
        return self._ti.solid_solutions

    @property
    def ss_kwargs(self):
        return self._ti.ss_kwargs

    @property
    def ss_models(self):
        return self._ti.ss_models

    @property
    def temp_slider(self):
        return self._ti.temp_slider

    @property
    def T_incr(self):
        return self._ti.T_incr

    @property
    def hull_mode(self):
        return self._ti.hull_mode

    @property
    def _n_simplex_vertices(self):
        return self._ti._n_simplex_vertices

    # Processing state lives on the model; these mirror it on the plotter so pinned
    # tests and downstream callers keep reading plotter.<attr>. The plotter's ``hsx_df`` is the
    # model's PROCESSED copy (``proc_df``); the pristine interpolated frame stays on the
    # model as ``hsx_df``.
    @property
    def hsx_df(self):
        return self._ti.proc_df

    @property
    def sys_name(self):
        return self._ti.sys_name

    @property
    def phase_names(self):
        return self._ti.phase_names

    @property
    def color_map(self):
        return self._ti.color_map

    @property
    def conds(self):
        return self._ti.conds

    @property
    def T_grid(self):
        return self._ti.T_grid

    @property
    def df_Tgroups(self):
        return self._ti.df_Tgroups

    @property
    def equil_df_list(self):
        return self._ti.equil_df_list

    def _init_sys(self):
        """Delegate data-prep to the model, then collect the binary edge figures.

        Fixed order, which the plotter's output depends on: model init, then
        ``bin_fig_list`` (when the edges are parameterized), then the completion print.
        """
        self._ti._init_sys()
        self.bin_fig_list = (
            build_binary_edge_figures(self._ti, ternary_color_map=self.color_map)
            if self._ti.fit_or_pred and self._ti.xs_mix
            else []
        )
        logger.info("Initialization complete")

    def process_data(self):
        """Run the model's hull pipeline (see :meth:`TernaryLiquidInterpolation.process_data`),
        with the plotter's `_init_sys` supplying the binary edge figures."""
        self._init_sys()
        if self.hull_mode == "hsx":
            self._ti._process_data_hsx()
            return
        self._ti._eval_hull()

    def get_plot(self, plot_type: str = "tx", **kwargs) -> go.Figure:
        """Generate a ternary plot (mirrors ``BLPlotter.get_plot``).

        Args:
            plot_type: 'tx' (default) — the ternary liquidus (T-x) phase-diagram surface;
                'ch' — a single-temperature lower convex-hull slice (requires ``T_celsius=``).
            kwargs: forwarded to the underlying builder (``T_celsius`` for 'ch').

        Returns:
            go.Figure: the generated Plotly figure.
        """
        valid_plot_types = ["tx", "ch"]
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}"
            )
        if plot_type == "tx":
            if not hasattr(self, "equil_df_list"):
                self.process_data()
            return self._plot_tx()
        # 'ch' — single-temperature lower-hull slice
        if "T_celsius" not in kwargs:
            raise ValueError("get_plot('ch', ...) requires a T_celsius=<float> keyword.")
        return self._plot_convex_hull(kwargs["T_celsius"])

    def show(self, plot_type: str = "tx", **kwargs) -> None:
        """Display the generated plot (see :meth:`get_plot` for plot_type / kwargs)."""
        self.get_plot(plot_type, **kwargs).show()

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
        if self.hull_mode == "hsx":
            raise ValueError(
                "get_convex_hull is a GTX-mode diagnostic (per-T slices); "
                "construct the plotter with hull_mode='gtx'."
            )
        if not hasattr(self._ti, "df_Tgroups") or not hasattr(self._ti, "T_grid"):
            self._init_sys()  # the plotter path also builds bin_fig_list
        return self._ti.get_convex_hull(T_celsius)

    def _plot_convex_hull(self, T_celsius: float) -> go.Figure:
        """The 3-D single-slice lower-hull figure for ``get_plot('ch', T_celsius=...)``
        (draw body in gliquid.plotting.ternary_surface)."""
        return ternary_surface.render_hull_slice(self.get_convex_hull(T_celsius), self.components)

    def _plot_tx(self):
        self.plotting_df = pd.concat(self.equil_df_list)
        simplex_df = deepcopy(self.plotting_df)

        liq_simplex_df = simplex_df[simplex_df["Phase"] == "L"]
        solid_simplex_df = simplex_df[simplex_df["Phase"] != "L"]
        liq_simplex_df = liq_simplex_df.sort_values("T").drop_duplicates(
            subset=["x0", "x1"], keep="first"
        )
        simplex_df = pd.concat([solid_simplex_df, liq_simplex_df])

        id_counts = simplex_df["simplex_id"].value_counts()
        valid_ids = id_counts[id_counts == self._n_simplex_vertices].index
        simplex_df = simplex_df[simplex_df["simplex_id"].isin(valid_ids)].copy()
        simplex_df = simplex_df.sort_values(by="simplex_id").reset_index(drop=True)

        self.liq_plotting_df = self.plotting_df[self.plotting_df["Phase"] == "L"]
        self.solid_plotting_df = self.plotting_df[self.plotting_df["Phase"] != "L"]
        self.solid_plotting_df = self.solid_plotting_df.sort_values("T").drop_duplicates(
            subset=["x0", "x1"], keep="last"
        )
        solids = set(self.solid_plotting_df["Phase"].tolist())
        solids = [str(x) for x in solids]

        for _, row in self.solid_plotting_df.iterrows():
            x0 = row["x0"]
            x1 = row["x1"]
            label = row["Phase"]
            color = row["Colors"]
            new_row = {"x0": x0, "x1": x1, "T": self.conds[0], "Phase": label, "Colors": color}
            new_row_df = pd.DataFrame([new_row])
            self.solid_plotting_df = pd.concat([self.solid_plotting_df, new_row_df])

        self.liq_plotting_df = self.liq_plotting_df.sort_values("T").drop_duplicates(
            subset=["x0", "x1"], keep="first"
        )

        # Create coexistent phases information for liquid points
        def get_coexistent_phases(simplex_id):
            """Get coexistent solid phases for a given simplex_id"""
            coexistent = simplex_df[
                (simplex_df["simplex_id"] == simplex_id) & (simplex_df["Phase"] != "L")
            ]
            # coexistent = simplex_df[(simplex_df['simplex_id'] == simplex_id)]
            if len(coexistent) > 0:
                phases = coexistent["Phase"].unique()
                return ", ".join(sorted(phases))
            return ""

        # Add coexistent phases to liquid plotting dataframe
        self.liq_plotting_df["coexistent_phases"] = self.liq_plotting_df["simplex_id"].apply(
            get_coexistent_phases
        )

        liq_points = np.array(
            list(
                zip(
                    self.liq_plotting_df["x0"],
                    self.liq_plotting_df["x1"],
                    self.liq_plotting_df["T"],
                )
            )
        )
        cart_liq_points = [ternary_to_cartesian(point[0], point[1]) for point in liq_points]
        self.triangulation = Delaunay(cart_liq_points)
        triangles = self.triangulation.simplices

        self.plotting_df = pd.concat([self.solid_plotting_df, self.liq_plotting_df])

        return ternary_surface.render_tx_surface(
            self.solid_plotting_df,
            self.liq_plotting_df,
            liq_points,
            triangles,
            self.color_map,
            self.components,
            self.conds,
            self.temp_slider,
        )

    def get_inter_melting_temps(self, interphases_for_melting: list[str]):
        """Max stability temperature per requested phase (computation on the model)."""
        return self._ti.get_inter_melting_temps(interphases_for_melting)
