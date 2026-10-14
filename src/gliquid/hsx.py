"""
Authors: Abrar Rauf, Joshua Willwerth
Description: This script takes the phase energy data in the form of enthalpy (H), entropy (S) and composition (X)
and performs transformations to composition-temperature (TX) phase diagrams with well-defined coexistence boundaries
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def _hull_from_fictitious(real_points, fict_points, second_filter=None):
    """Shared lower-hull core: ConvexHull('Qt i') over real+fictitious points, drop every
    simplex touching a fictitious vertex (index >= len(real_points)), then apply an
    optional ``second_filter(real_simplices)`` mask.

    The two fictitious-point strategies (sub-hull-centroid in ``lower_convex_hull``,
    corner-point + lifted-liquid in ``HSX.hull``) stay at their call sites and are
    separately pinned; only this hull-and-filter computation is common to both.
    """
    n_real = len(real_points)
    new_points = np.vstack((real_points, fict_points))
    new_hull = ConvexHull(new_points, qhull_options="Qt i")
    all_simplices = new_hull.simplices

    # Filter 1: discard simplices with any fictitious vertex (index >= n_real)
    mask_no_fict = np.all(all_simplices < n_real, axis=1)
    real_simplices = all_simplices[mask_no_fict]

    if second_filter is not None:
        real_simplices = second_filter(real_simplices)
    return real_simplices


def lower_convex_hull(points, vertical_simplices=False):
    # Lower convex hull of an N-dimensional Xi-E space (energy-like axis last) by the
    # sub-hull-centroid fictitious-point strategy; returns the lower-hull simplices.

    sub_points = points[:, :-1]
    sub_hull = ConvexHull(sub_points)
    sub_hull_points = sub_points[sub_hull.vertices]
    center = np.mean(sub_hull_points, axis=0)
    sub_hull_points = np.vstack((sub_hull_points, center))

    # Compute upper bound for fictitious points using single-pass ptp
    h_data = points[:, -1]
    h_max = np.max(h_data)
    upper_bound = h_max + 10 * np.ptp(h_data)

    # Create fictitious points with upper_bound H values
    upper_bound_col = np.full((len(sub_hull_points), 1), upper_bound)
    fake_points = np.hstack((sub_hull_points, upper_bound_col))
    fake_points[-1, -1] += 0.5 * upper_bound  # Offset center point higher

    def drop_vertical(real_simplices):
        # Filter 2: drop simplices whose x coordinates span no volume.
        x_coords = points[real_simplices][:, :, :-1]
        volume = np.abs(np.linalg.det(x_coords[:, 1:, :] - x_coords[:, :1, :]))
        scale = np.ptp(sub_points, axis=0).max() ** sub_points.shape[1]
        return real_simplices[volume > 1e-12 * scale]

    return _hull_from_fictitious(points, fake_points, None if vertical_simplices else drop_vertical)


def two_liquid_spans_from_facets(xs, labels, min_gap: float = 0.012) -> list[list[float]]:
    """Merged composition spans tied by two-liquid facets.

    A facet counts when exactly two of its vertices are liquid and their compositions
    differ by more than ``min_gap``; its span is the interval between those two vertices.

    Args:
        xs: ``(n_facets, n_vertices)`` vertex compositions.
        labels: ``(n_facets, n_vertices)`` vertex phase labels.
        min_gap: smallest liquid-liquid composition difference that counts as a tie.

    Returns:
        list[list[float]]: ``[lo, hi]`` spans, merged and in composition order.
    """
    xs = np.asarray(xs, dtype=float)
    liquid = np.asarray(labels) == "L"
    if xs.size == 0:
        return []
    two = liquid.sum(axis=1) == 2
    if not np.any(two):
        return []
    liquid_x = np.where(liquid[two], xs[two], np.nan)
    lo = np.nanmin(liquid_x, axis=1)
    hi = np.nanmax(liquid_x, axis=1)
    keep = hi - lo > min_gap
    merged: list[list[float]] = []
    for a, b in sorted(zip(lo[keep].tolist(), hi[keep].tolist())):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def two_liquid_ties_from_facets(xs, labels, temps, min_gap: float = 0.012) -> list[list[float]]:
    """Two-liquid tie lines of a set of facets, with their temperatures.

    A facet counts when at least two of its vertices are liquid and the widest gap between
    consecutive liquid compositions exceeds ``min_gap``, which must exceed the composition
    grid step; its tie runs between the outermost liquid vertices.

    Args:
        xs: ``(n_facets, n_vertices)`` vertex compositions.
        labels: ``(n_facets, n_vertices)`` vertex phase labels.
        temps: ``(n_facets,)`` facet temperatures.
        min_gap: smallest liquid-liquid composition difference that counts as a tie.

    Returns:
        list[list[float]]: ``[lo, hi, T]`` per tie, in temperature order.
    """
    xs = np.asarray(xs, dtype=float)
    liquid = np.asarray(labels) == "L"
    temps = np.asarray(temps, dtype=float)
    if xs.size == 0:
        return []
    many = liquid.sum(axis=1) >= 2
    if not np.any(many):
        return []
    liquid_x = np.where(liquid[many], xs[many], np.nan)
    gaps = np.nanmax(np.diff(np.sort(liquid_x, axis=1), axis=1), axis=1)
    keep = gaps > min_gap
    if not np.any(keep):
        return []
    lo = np.nanmin(liquid_x[keep], axis=1)
    hi = np.nanmax(liquid_x[keep], axis=1)
    tk = temps[many][keep]
    order = np.argsort(tk, kind="stable")
    return [[float(lo[i]), float(hi[i]), float(tk[i])] for i in order]


class HSX:
    """Handles enthalpy (H), entropy (S), and composition (X) transformations for TX phase diagrams."""

    def __init__(self, data_dict: dict, conds: list[float], use_filter_2=False):
        """Initializes the HSX instance with provided phase data and conditions.

        ``len(data_dict['comps'])`` sets the dimensionality: the data table must carry
        n-1 composition columns followed by S, H and the phase name (positional — the
        binary ``to_hsx`` dict and the ternary ``hsx_df[['x0','x1','S','H',...]]`` order).
        """
        self.phases = data_dict["phases"]
        self.comps = data_dict["comps"]
        self.conds = conds
        self.use_filter_2 = use_filter_2

        self._load_data(data_dict["data"])

    def _load_data(self, data) -> None:
        """(Re)build the data table and every data-derived cache from raw H/S/X rows,
        leaving phases/comps/conds untouched."""
        self.df = pd.DataFrame(data)
        self.simplices = []
        self.final_phases = []
        self.df_tx = pd.DataFrame()
        self.facet_temps = np.array([])

        # Data scaling
        s_scaler = 100
        h_scaler = 10000
        n_comp_cols = len(self.comps) - 1
        if self.df.shape[1] != n_comp_cols + 3:
            raise ValueError(
                f"HSX data for a {len(self.comps)}-component system must carry "
                f"{n_comp_cols + 3} positional columns ({n_comp_cols} composition + S + H "
                f"+ phase name); got {self.df.shape[1]}."
            )
        # Binary column names are pinned (consumers read 'X [Fraction]' literally).
        self.x_cols = (
            ["X [Fraction]"]
            if n_comp_cols == 1
            else [f"X{i} [Fraction]" for i in range(n_comp_cols)]
        )
        self.df.columns = self.x_cols + ["S [J/mol/K]", "H [J/mol]", "Phase"]
        self.df["S [J/mol/K]"] /= s_scaler
        self.df["H [J/mol]"] /= h_scaler

        # Data extraction for convex hull calculation
        value_cols = self.x_cols + ["S [J/mol/K]", "H [J/mol]"]
        df_inter = self.df[self.df["Phase"] != "L"]
        df_liq = self.df[self.df["Phase"] == "L"]
        self.liq_points = df_liq[value_cols].to_numpy()
        self.inter_points = df_inter[value_cols].to_numpy()
        self.points = self.df[value_cols].to_numpy()
        self.scaler = h_scaler / s_scaler

    def set_data(self, data) -> None:
        """Swap in new H/S/X rows in place, keeping the instance's phases/comps/conds.

        ``data`` must carry the same positional layout as ``data_dict['data']`` at
        construction and describe the same phase set. Every data-derived cache
        (df, points, simplices, df_tx, ...) is rebuilt, so the next ``compute_tx()`` /
        ``get_phase_points()`` reflects the new data exactly as a fresh instance would.
        """
        self._load_data(data)

    def hull(self) -> np.ndarray:
        """Computes the lower convex hull of an N-dimensional Xi-S-H space."""
        dim = self.points.shape[1]

        # Initialize bounds for Xi
        x_list = [[1 if j == i - 1 else 0 for j in range(dim - 2)] for i in range(dim - 1)]
        x_list[0] = [0] * (dim - 2)

        # Compute S and H bounds
        s_min, s_extr = (
            np.min(self.points[:, -2]),
            np.max([self.liq_points[0, -2], self.liq_points[-1, -2]]),
        )
        h_max = np.max(self.points[:, -1])
        upper_bound = 20 * h_max

        # Generate fictitious points (liquid coordinates with H lifted to the upper bound)
        liq_fict_coords = np.column_stack(
            (self.liq_points[:, :-1], np.full(len(self.liq_points), upper_bound))
        )
        fict_coords = np.vstack(
            [np.append(x_list[i], [s_min, upper_bound]) for i in range(dim - 1)]
            + [np.append(x_list[i], [s_extr, upper_bound]) for i in range(dim - 1)]
        )

        fict_points = np.vstack((fict_coords, liq_fict_coords))

        def drop_all_intermetallic(real_simplices):
            # Filter 2: discard simplices whose vertices are all intermetallic (non-liquid)
            is_inter = (self.df["Phase"] != "L").values
            inter_counts = np.sum(is_inter[real_simplices], axis=1)
            return real_simplices[inter_counts < self.points.shape[1]]

        self.simplices = _hull_from_fictitious(
            self.points, fict_points, drop_all_intermetallic if self.use_filter_2 else None
        )
        return self.simplices

    def compute_tx(self) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
        """Computes the TX phase diagram transformation.

        Each lower-hull facet's hyperplane normal yields its coexistence temperature
        ``T = (-n_S / n_H) * scaler``. The binary (dim == 3) branch uses the cross-product
        formula (its 1e-12 degeneracy threshold acts on the
        UNNORMALIZED normal and is pinned); higher dimensions use an SVD null-space
        normal, whose unit scaling makes the same threshold scale-invariant.
        """
        self.hull()
        dim = self.points.shape[1]
        temps, valid_simplices, new_phases = [], [], []
        for simplex in self.simplices:
            verts = self.points[simplex]
            if dim == 3:
                A, B, C = verts
                n = np.cross(B - A, C - A).astype(float)
                # Degenerate / near-vertical facets yield n[2] ~ 0 and non-physical infinite
                # temperatures.
                if np.isclose(n[2], 0.0, atol=1e-12):
                    continue
                T = (-n[1] / n[2]) * self.scaler
            else:
                edges = (verts[1:] - verts[0]).astype(float)
                n = np.linalg.svd(edges)[2][-1]  # unit normal spanning the facet's null space
                if np.isclose(n[-1], 0.0, atol=1e-12):
                    continue
                T = (-n[-2] / n[-1]) * self.scaler
            if np.isfinite(T):
                temps.append(T)
                valid_simplices.append(simplex)
                new_phases.append([self.df.loc[simplex[i], "Phase"] for i in range(len(simplex))])

        temps = np.array(temps)
        self.facet_temps = temps.copy()
        self.final_phases = np.array(new_phases)

        n_x = dim - 2
        data = [
            [*self.points[vertex][:n_x], temps[i], labels[j]]
            for i, simplex in enumerate(valid_simplices)
            for j, vertex in enumerate(simplex)
            for labels in [self.final_phases[i]]  # Extract labels once per simplex
        ]

        x_cols = ["x"] if n_x == 1 else [f"x{k}" for k in range(n_x)]
        self.df_tx = pd.DataFrame(data, columns=x_cols + ["t", "label"])

        return self.df_tx, self.final_phases, np.array(valid_simplices), temps

    @staticmethod
    def _collapse_gap_runs(entries: list, x_tol: float = 0.025) -> list:
        """Reduce each continuous two-phase boundary in 'Misc Gaps' to its two ends.

        A miscibility gap or solvus is a CONTINUOUS field, not a set of invariants, but it
        contributes one collapsed-triangle facet per composition grid step, walking the same
        boundary; those facets are not distinct invariants.

        Entries are grouped by phase-name multiset, then chained into runs: consecutive
        (in temperature) entries belong to the same boundary when BOTH ends of their tie
        span moved by no more than ``x_tol``. A run of three or more collapses to its
        coldest and hottest member -- the points where the boundary terminates on a real
        invariant or the temperature frame. Runs of one or two are already extremal and
        pass through untouched, which is what preserves genuine three-phase horizontals
        that happen to share a phase multiset with a long solvus run.
        """
        buckets = defaultdict(list)
        for entry in entries:
            buckets[tuple(sorted(entry[3]))].append(entry)

        collapsed = []
        for group in buckets.values():
            group.sort(key=lambda e: e[0])
            run = [group[0]]
            for prev, cur in zip(group, group[1:]):
                same_boundary = (
                    abs(min(cur[2]) - min(prev[2])) <= x_tol
                    and abs(max(cur[2]) - max(prev[2])) <= x_tol
                )
                if same_boundary:
                    run.append(cur)
                else:
                    collapsed.extend(run if len(run) <= 2 else [run[0], run[-1]])
                    run = [cur]
            collapsed.extend(run if len(run) <= 2 else [run[0], run[-1]])
        return collapsed

    def liquidus_invariants(self) -> tuple[dict, list, dict]:
        """Extracts eutectic, peritectic, and congruent melting points from the computed TX phase diagram."""
        if len(self.comps) != 2:
            raise NotImplementedError(
                "liquidus_invariants classifies binary 3-vertex simplices; n-component "
                "invariant classification is not implemented."
            )
        self.df_tx, self.final_phases, final_simplices, final_temps = self.compute_tx()
        self.df_tx["t"] -= 273.15
        final_temps -= 273.15

        compositions = np.array(
            [[vertex[0] for vertex in self.points[simplex]] for simplex in final_simplices]
        )

        combined_list = []
        for i in range(len(compositions)):
            row_dict = {}
            for j in range(len(compositions[i])):
                row_dict[compositions[i][j]] = self.final_phases[i][j]
            if len(row_dict) == 2:
                for key in row_dict.keys():
                    if key == 0.0:
                        row_dict[key] = self.comps[0]
                    elif key == 1.0:
                        row_dict[key] = self.comps[1]
            combined_list.append([final_temps[i], row_dict])

        int_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], "L"]]

        inv_points = {
            "Eutectics": [],
            "Peritectics": [],
            "Congruent Melting": [],
            "Misc Gaps": [],
            "Solid Ties": [],
        }
        peritectic_phases, non_triples = [], []

        for temp, comb_dict in combined_list:
            sorted_dict = dict(sorted(comb_dict.items()))
            comp, phase = list(sorted_dict.keys()), list(sorted_dict.values())

            if len(comp) == 3:
                if len(set(phase)) == 3:
                    if phase[1] == "L":
                        inv_points["Eutectics"].append([temp, comp[1], comp, phase])
                    else:
                        inv_points["Peritectics"].append([temp, comp[1], comp, phase])
                        peritectic_phases.append(phase[1])
                else:
                    non_triples.append([temp, comp, phase])

        congruents_init = []
        for temp, comp, phase in non_triples:
            if phase[0] == "L" and phase[2] != "L":
                comp_diff = abs(comp[0] - comp[1])
                if comp_diff > 0.012:
                    inv_points["Misc Gaps"].append([temp, comp[1], comp, phase])
            elif phase[0] != "L":
                comp_diff = abs(comp[1] - comp[2])
                if comp_diff > 0.012:
                    inv_points["Misc Gaps"].append([temp, comp[1], comp, phase])
            phase = [p for p in phase if p != "L"]
            if phase and phase[0] in int_phases and phase[0] not in peritectic_phases:
                congruents_init.append([temp, comp[0], comp, phase])

        grouped_data = defaultdict(list)
        for entry in congruents_init:
            grouped_data[entry[3][0]].append(entry)

        inv_points["Congruent Melting"] = [
            max(entries, key=lambda x: x[0]) for entries in grouped_data.values()
        ]

        # Solid-solid ties from 2-vertex simplices: the hottest tie per solid pair, excluding any
        # that restates a detected invariant or bridges the pure elements.
        solid_pair_ties = {}  # frozenset(phaseA, phaseB) -> [temp, comp_mid, comps, phases]
        for temp, comb_dict in combined_list:
            if len(comb_dict) != 2:
                continue
            (cA, pA), (cB, pB) = sorted(comb_dict.items())
            if pA == pB or pA == "L" or pB == "L":
                continue
            if abs(cB - cA) < 0.012:  # negligible composition gap -> not a real tie
                continue
            # An element-to-element span is kept: the top of the terminal (A)+(B) two-phase field.
            key = frozenset((pA, pB))
            if key not in solid_pair_ties or temp > solid_pair_ties[key][0]:
                solid_pair_ties[key] = [temp, (cA + cB) / 2, [cA, cB], [pA, pB]]

        existing_spans = [
            (min(c), max(c))
            for k in ("Eutectics", "Peritectics", "Misc Gaps")
            for _, _, c, _ in inv_points[k]
        ]
        for entry in solid_pair_ties.values():
            lo, hi = min(entry[2]), max(entry[2])
            if any(abs(elo - lo) < 0.02 and abs(ehi - hi) < 0.02 for elo, ehi in existing_spans):
                continue  # already drawn as a eutectic/peritectic
            inv_points["Solid Ties"].append(entry)

        # Continuous-boundary runs collapse after the Solid Ties dedup, which reads the full span
        # list.
        inv_points["Misc Gaps"] = self._collapse_gap_runs(inv_points["Misc Gaps"])

        # Normalize invariant-point numeric fields to built-in Python floats.
        for inv_type, entries in inv_points.items():
            inv_points[inv_type] = [
                [float(temp), float(comp_mid), [float(c) for c in comp], [str(p) for p in phase]]
                for temp, comp_mid, comp, phase in entries
            ]

        count_dict = {key: len(value) for key, value in inv_points.items()}

        return inv_points, combined_list, count_dict

    def get_phase_points(self) -> dict:
        """Extracts phase boundary points from the HSX object and converts to a list of dictionaries for BinaryLiquid"""
        if len(self.comps) != 2:
            raise NotImplementedError(
                "get_phase_points emits the binary per-phase [x, t] contract; use "
                "compute_tx()'s df_tx directly for n-component systems."
            )
        df_tx = self.compute_tx()[0]
        phase_points = {
            phase: df_tx[df_tx["label"] == phase][["x", "t"]].values.tolist()
            for phase in self.phases
        }
        liq_df = (
            df_tx[df_tx["label"] == "L"]
            .sort_values(["x", "t"])
            .drop_duplicates(subset="x", keep="first")
        )
        phase_points["L"] = liq_df[liq_df["t"] >= -273.15][["x", "t"]].values.tolist()
        return phase_points

    def two_liquid_spans(self, min_gap: float = 0.012) -> list[list[float]]:
        """Composition spans tied by a two-liquid facet of the last computed diagram.

        Reads the facets the last ``compute_tx`` stored, so call ``get_phase_points`` or
        ``compute_tx`` first. Binary systems only.

        Args:
            min_gap: smallest liquid-liquid composition difference that counts as a tie.

        Returns:
            list[list[float]]: merged ``[lo, hi]`` spans; empty when no facet ties two liquids.

        Raises:
            NotImplementedError: for an n-component system.
        """
        if len(self.comps) != 2:
            raise NotImplementedError(
                "two_liquid_spans reads binary 3-vertex facets; n-component ties are not "
                "classified."
            )
        if self.df_tx is None or len(self.df_tx) == 0:
            return []
        xs = self.df_tx["x"].to_numpy(dtype=float).reshape(-1, 3)
        labels = np.asarray(self.final_phases).reshape(-1, 3)
        return two_liquid_spans_from_facets(xs, labels, min_gap)

    def two_liquid_ties(self, min_gap: float = 0.012) -> list[list[float]]:
        """Two-liquid tie lines of the last computed diagram, with their temperatures in K.

        Reads the facets the last ``compute_tx`` stored, so call ``get_phase_points`` or
        ``compute_tx`` first. Binary systems only.

        Args:
            min_gap: smallest liquid-liquid composition difference that counts as a tie.

        Returns:
            list[list[float]]: ``[lo, hi, T]`` per tie in temperature order; empty when no
            facet ties two liquids.

        Raises:
            NotImplementedError: for an n-component system.
        """
        if len(self.comps) != 2:
            raise NotImplementedError(
                "two_liquid_ties reads binary 3-vertex facets; n-component ties are not classified."
            )
        if self.df_tx is None or len(self.df_tx) == 0:
            return []
        xs = self.df_tx["x"].to_numpy(dtype=float).reshape(-1, 3)
        labels = np.asarray(self.final_phases).reshape(-1, 3)
        return two_liquid_ties_from_facets(xs, labels, self.facet_temps, min_gap)
