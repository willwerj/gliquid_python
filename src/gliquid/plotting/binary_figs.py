"""Diagnostic/analysis figure builders over a BinaryLiquid-like model (ex-BLPlotter bodies).

Extracted verbatim from BLPlotter: raw TX scatter, HSX hull debug, MPDS-vs-MP low-T
phase comparison, T=0K DFT convex hull (+ Gibbs overlays), and the Nelder-Mead path
figure. Each function takes the model object (duck-typed ``bl``) and,
where colors are drawn, an explicit phase->color map (BLPlotter passes its cached
build_phase_color_map result). No gliquid.binary import -- import order:
{api, mpds, solution, phase, plotting.style, plotting.binary_tx} <- plotting.binary_figs
<- binary. (binary_tx is presentation-only and imports nothing from here, so the assessed-
liquidus segment splitter is shared from there rather than duplicated.)
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sympy as sp
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from pymatgen.analysis.phase_diagram import (
    PDEntry,
    PDPlotter,
    PhaseDiagram,
)  # The PMG PDPlotter source code is modified here
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.mpds as mpds
import gliquid.solution as solution
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.plotting.binary_tx import _assessed_liquidus_segments
from gliquid.plotting.style import ASSESSED_LIQUIDUS_COLOR, format_phase_display_name
from gliquid.solution import comp_symbols, t_sym

logger = logging.getLogger(__name__)

_x_vals = solution.x_vals
xb_sym = comp_symbols(2)[0]


def _display_name(bl, label: str) -> str:
    """Legend name via the shared style rule (SS phases carry their component pair)."""
    return format_phase_display_name(label, bl.ss_models, bl.components)


def render_tx_scatter(bl, color_map, **kwargs) -> go.Figure:
    """Diagnostic TX scatter using raw points from HSX compute_tx (no envelope post-processing)."""
    if not bl.phases[-1].points:
        bl.update_phase_points()

    include_digitized_liquidus = kwargs.get("include_digitized_liquidus", True)

    df_tx, _, _, _ = bl.hsx.compute_tx()
    df = df_tx.copy()
    df["x_at"] = df["x"].astype(float) * 100.0
    df["t_c"] = df["t"].astype(float) - 273.15
    df["label_display"] = [_display_name(bl, str(label)) for label in df["label"]]

    display_color_map = {
        _display_name(bl, phase): color_map.get(phase, "#555555") for phase in bl.hsx.phases
    }

    fig = px.scatter(
        df,
        x="x_at",
        y="t_c",
        color="label_display",
        color_discrete_map=display_color_map,
        title=f"{bl.sys_name} TX Scatter (Raw HSX compute_tx Points)",
        width=920,
        height=700,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.85})

    if include_digitized_liquidus and bl.digitized_liq:
        # Split at undigitized holes so no dashed line crosses a gap between disjoint
        # 'L' regions; one legend entry for the whole (possibly multi-segment) curve.
        assessed = [
            [float(point[0] * 100.0), float(point[1] - 273.15)] for point in bl.digitized_liq
        ]
        for shown, segment in enumerate(_assessed_liquidus_segments(assessed)):
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in segment],
                    y=[p[1] for p in segment],
                    mode="lines",
                    line={"color": ASSESSED_LIQUIDUS_COLOR, "width": 2.0, "dash": "dash"},
                    name="Assessed Liquidus",
                    showlegend=not shown,
                )
            )

    y_lo = float(bl.temp_range[0] - 273.15)
    y_hi = float(bl.temp_range[-1] - 273.15) + 100.0
    fig.update_layout(
        xaxis={"range": [0, 100], "title": f"X_{bl.components[1]} (at. %)"},
        yaxis={"range": [y_lo, y_hi], "title": "T [C]"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        font={"size": 18},
    )
    fig.update_xaxes(
        mirror=True, ticks="inside", showline=True, linecolor="gray", linewidth=2, tickcolor="gray"
    )
    fig.update_yaxes(
        mirror=True, ticks="inside", showline=True, linecolor="gray", linewidth=2, tickcolor="gray"
    )
    return fig


def render_hsx_diagnostic(bl, color_map, **kwargs) -> go.Figure:
    """HSX diagnostic plot with solid-solution phases rendered as phase-level blocks.

    Mirrors the original HSX plot style (scatter + lower-hull simplices) and adds one
    mesh block per solid-solution phase so each continuous solution is visually grouped.
    """
    system = bl
    if not system.phases[-1].points:
        system.update_phase_points()

    show_hull_simplices = kwargs.get("show_hull_simplices", True)
    simplex_color = kwargs.get("simplex_color", "cyan")
    simplex_opacity = kwargs.get("simplex_opacity", 0.28)
    ss_block_opacity = kwargs.get("ss_block_opacity", 0.30)

    hsx_obj = system.hsx
    simplices = hsx_obj.hull()
    df = hsx_obj.df
    points = hsx_obj.points
    scatter_colors = [color_map.get(str(phase), "#555555") for phase in df["Phase"]]

    fig = go.Figure()

    # Base scatter for all HSX points.
    fig.add_trace(
        go.Scatter3d(
            x=df["X [Fraction]"],
            y=df["S [J/mol/K]"],
            z=df["H [J/mol]"],
            mode="markers",
            marker={"size": 4, "opacity": 0.55, "color": scatter_colors},
            name="HSX points",
            showlegend=False,
            hovertemplate=(
                "Phase: %{customdata}<br>X: %{x:.4f}<br>S: %{y:.4f}<br>H: %{z:.4f}<extra></extra>"
            ),
            customdata=df["Phase"],
        )
    )

    # Overlay lower-hull simplices used for TX construction.
    if show_hull_simplices:
        for simplex in simplices:
            fig.add_trace(
                go.Mesh3d(
                    x=points[simplex, 0],
                    y=points[simplex, 1],
                    z=points[simplex, 2],
                    i=np.array([0]),
                    j=np.array([1]),
                    k=np.array([2]),
                    opacity=float(simplex_opacity),
                    color=simplex_color,
                    name="Hull simplex",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Render each SS phase as a single colored mesh block.
    for ss_name in system.ss_models:
        ss_df = df[df["Phase"] == ss_name]
        if ss_df.empty:
            continue
        fig.add_trace(
            go.Mesh3d(
                x=ss_df["X [Fraction]"].to_numpy(dtype=float),
                y=ss_df["S [J/mol/K]"].to_numpy(dtype=float),
                z=ss_df["H [J/mol]"].to_numpy(dtype=float),
                alphahull=5,
                opacity=float(ss_block_opacity),
                color=color_map.get(ss_name, "#555555"),
                name=f"{ss_name} block",
                hovertemplate=(
                    f"{ss_name} block<br>X: %{{x:.4f}}<br>"
                    "S: %{y:.4f}<br>H: %{z:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Clean legend entries for all non-liquid phases and liquid.
    legend_labels = [p for p in hsx_obj.phases if p != "L"] + ["L"]
    for label in legend_labels:
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 8, "color": color_map.get(label, "#555555")},
                name=_display_name(bl, label),
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"<b>{system.sys_name} HSX Convex Hull Debug (Solid-Solution Blocks)</b>",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "S [scaled J/mol/K]",
            "zaxis_title": "H [scaled J/mol]",
        },
        legend={
            "itemsizing": "constant",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.02,
        },
        font={"size": 14},
        width=980,
        height=760,
    )
    return fig


def render_phase_comparison(bl) -> plt.Figure:
    """
    Generates a phase comparison plot showing congruent and incongruent phases
    from MPDS and MP data. The plot consists of two subplots displaying phases
    in different temperature and magnitude ranges.

    Returns:
        plt.Figure: The generated phase comparison plot.
    """
    # Extract low-temperature phase data from MPDS and MP
    (
        [mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp],
        [mp_phases, mp_phases_ebelow, min_form_e],
    ) = mpds.get_low_temp_phase_data(bl.mpds_json, bl.dft_ch)

    # Filter out phases containing parentheses
    mpds_congruent_phases = {
        key: value for key, value in mpds_congruent_phases.items() if "(" not in key
    }
    mpds_incongruent_phases = {
        key: value for key, value in mpds_incongruent_phases.items() if "(" not in key
    }

    # Create subplots with specific layout
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 2),
        gridspec_kw={"hspace": 0, "left": 0.117, "bottom": 0.13, "right": 0.909},
    )

    def plot_phases(ax, source, color, alpha=0.5):
        """
        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            source (dict): Phase data with keys as phase names and values as bounds/magnitudes.
            color (str): Color for the phase fill.
            alpha (float): Transparency for the fill.
        """
        for _, ((lb, ub), mag) in source.items():
            # Ensure a minimum width for labeling
            if ub - lb < 0.026:
                ave = (ub + lb) / 2
                lb = ave - 0.013
                ub = ave + 0.013
            ax.fill_betweenx([min(0, mag), max(0, mag)], lb, ub, color=color, alpha=alpha)
            ax.set_xlim(0, 1)
            ax.margins(x=0, y=0)

    # Plot phases for both subplots
    plot_phases(ax1, mpds_congruent_phases, "blue")
    plot_phases(ax1, mpds_incongruent_phases, "purple")
    plot_phases(ax2, mp_phases, "orange")
    plot_phases(ax2, mp_phases_ebelow, "red")

    # Check if MPDS phases exist
    mpds_phases = bool(mpds_congruent_phases or mpds_incongruent_phases)

    # Configure y-axis for the first subplot
    if mpds_phases:
        tick_range = np.linspace(0, max_phase_temp, 4)[1:]
        ax1.set_yticks(tick_range)
        ax1.set_yticklabels([format(tick, ".1e") for tick in tick_range])
        ax1.set_ylim(0, 1.1 * max_phase_temp)
    else:
        ax1.set_yticks([])

    ax1.set_ylabel("MPDS", fontsize=11, rotation=90, labelpad=5, fontweight="semibold")
    ax1.yaxis.set_label_position("right")
    ax1.set_xticks([])

    # Configure y-axis for the second subplot
    if mp_phases:
        tick_range = np.linspace(0, min_form_e, 4)
        ax2.set_yticks(tick_range)
        ax2.set_yticklabels([format(tick, ".1e") for tick in tick_range])
        ax2.set_ylim(1.1 * min_form_e, 0)
    elif mpds_phases:
        ax2.set_yticks([0])
        ax2.set_yticklabels([format(0, ".1e")])
        ax2.set_ylim(-1, 0)
    else:
        ax2.set_yticks([])

    ax2.set_ylabel("MP", fontsize=11, rotation=90, labelpad=5, fontweight="semibold")
    ax2.yaxis.set_label_position("right")
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.set_xticklabels([0, 20, 40, 60, 80, 100])

    # Add a title to the figure
    fig.suptitle("Low Temperature Phase Comparison", fontweight="semibold")

    return fig


def render_dft_hull(bl, plot_type: str, **kwargs) -> go.Figure:
    """
    Generates a convex hull plot or phase diagram visualization.

    Args:
        bl: The model to plot (duck-typed ``BinaryLiquid``); its ``dft_ch``, ``components``,
            ``component_data``, ``eqs``, and ``sys_name`` are read.
        plot_type (str): The type of plot to generate ('ch', 'vch', 'ch+g').
        kwargs: Additional arguments for customization, such as 't_vals' or 't_units' for temperature values.

    Returns:
        go.Figure: The generated Plotly figure.
    """
    if any(len(Composition(comp).as_dict()) > 1 for comp in bl.components):
        raise NotImplementedError("This feature is not presently supported for compound components")

    if not bl.component_data:
        logger.warning(
            "BinaryLiquid object phase diagram not initialized! Returning plot without liquid energy"
        )
        plot_type = "ch"  # Override to avoid errors

    if plot_type == "vch":
        # Generate volume-referenced convex hull
        ch, atomic_vols = api.get_dft_convexhull(bl.sys_name, bl.dft_type, inc_structure_data=True)

        new_entries = [
            PDEntry(
                composition=e.composition,
                energy=atomic_vols[e.composition.reduced_formula] * e.composition.num_atoms,
            )
            for e in ch.stable_entries
        ]

        vch = PhaseDiagram(new_entries)
        pdp = PDPlotter(vch)
        fig = pdp.get_plot()
        fig.update_yaxes(title={"text": "Referenced Atomic Volume (Å^3/atom)"})
    else:
        pdp = PDPlotter(bl.dft_ch)

    if plot_type == "ch":
        # Use the standard convex hull
        fig = pdp.get_plot()

    else:  # plot_type == 'ch+g'
        # Generate convex hull plot with liquidus curves overlaid
        t_vals = kwargs.get("t_vals", [])
        if not isinstance(t_vals, list) or not all(isinstance(t, (int, float)) for t in t_vals):
            raise ValueError(
                "kwarg 't_vals' must be a list of valid temperatures, either as ints or floats!"
            )
        t_units = kwargs.get("t_units", "C")
        if not t_units or not isinstance(t_units, str) or t_units not in ["C", "K"]:
            raise ValueError(
                "kwarg 't_units' must be a string, either 'C' for Celsius or 'K' for Kelvin"
            )
        if t_units and not t_vals:
            logger.info("No arguments specified for 't_vals', setting 't_units' to 'K'")

        if (
            not t_vals
        ):  # Determine max phase temp if there are compounds, else maximum or minimum liquidus temp
            if not bl.phases[-1].points:
                bl.update_phase_points()
            solid_phases = [
                p for p in bl.phases if (p.name not in bl.components + ["L"]) and len(p.points) > 0
            ]
            if solid_phases:
                max_phase_temp = max([max(p.points, key=lambda x: x[1])[1] for p in solid_phases])
            elif (
                max(bl.phases[-1].points, key=lambda x: x[1])[1]
                > max(bl.component_data.values(), key=lambda x: x.t_fusion).t_fusion
            ):
                max_phase_temp = max(bl.phases[-1].points, key=lambda x: x[1])[1]  # liquid misc gap
            else:
                max_phase_temp = min(bl.phases[-1].points, key=lambda x: x[1])[
                    1
                ]  # eutectic or azeotrope
            t_units = "K"
            t_vals = [0, max_phase_temp]
        if t_units == "C":
            t_vals = [t + 273.15 for t in t_vals if t >= 0]
        else:
            t_vals = [t for t in t_vals if t >= 0]

        def get_g_curve(A=0, B=0, C=0, D=0, T=0, name="") -> go.Scatter:
            """
            Args:
                A, B, C, D (float): Non-ideal mixing parameters.
                T (float): Temperature in Kelvin.
                name (str): Trace name. Empty (default) auto-labels the curve from the
                    parameters and ``T`` in the enclosing figure's temperature units.

            Returns:
                go.Scatter: Plotly scatter trace for the Gibbs free energy curve.
            """
            g = bl.eqs["g_liquid"].subs(
                {t_sym: T, **dict(zip(bl.xs_mix.format.symbols(), (A, B, C, D)))}
            )
            gliq_vals = (
                sp.lambdify(xb_sym, g, "numpy")(_x_vals[1:-1])
                if g.has(xb_sym)
                else [0] * len(_x_vals[1:-1])
            )
            ga = np.float64(bl.eqs["ga"].subs({t_sym: T}) / 96485)
            gb = np.float64(bl.eqs["gb"].subs({t_sym: T}) / 96485)
            if name == "":
                name += "Ideal " if A == 0 and C == 0 else ""
                name += (
                    "Liquid T=" + str(int(T)) + "K"
                    if t_units == "K"
                    else str(int(T - 273.15)) + "C"
                )

            return go.Scatter(
                x=_x_vals, y=[ga] + [g / 96485 for g in gliq_vals] + [gb], mode="lines", name=name
            )

        # Build the G curves first, then construct a new figure whose data list
        # starts with the G curves followed by the hull traces — identical
        # ordering and legend position to passing them via the data= argument.
        params = bl.get_params()
        g_curves = [
            get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp)
            for temp in reversed(t_vals)
        ]
        # Solid-solution overlays: one dashed Gibbs curve per stable SS phase per
        # temperature (show_unstable=True draws them regardless of hull stability).
        ss_names = list(bl.ss_models)
        if ss_names:
            if not kwargs.get("show_unstable", False):
                if not bl.phases[-1].points:
                    bl.update_phase_points()
                _, final_phases, _, _ = bl.hsx.compute_tx()
                stable_labels = {str(label) for phase_list in final_phases for label in phase_list}
                ss_names = [name for name in ss_names if name in stable_labels]
            x_ss = np.asarray(_x_vals, dtype=float)
            for temp in reversed(t_vals):
                for ss_name in ss_names:
                    y_ss = bl.solid_solution_gibbs(ss_name, x_ss, temp) / EV_ATOM_TO_J_MOL
                    g_curves.append(
                        go.Scatter(
                            x=x_ss,
                            y=y_ss,
                            mode="lines",
                            line={"width": 2.5, "dash": "dash"},
                            name=f"{ss_name} {int(round(temp))} K",
                        )
                    )
        hull_fig = pdp.get_plot()
        fig = go.Figure(data=g_curves + list(hull_fig.data), layout=hull_fig.layout)

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title=dict(text="Composition (fraction)", font=dict(size=18)),
            tickfont=dict(color="black"),
        ),
        yaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(color="black")),
        font=dict(color="black", size=15),  # Sets default font color for all text elements
        width=750,
        height=600,
    )
    return fig


def render_nelder_mead_path(bl, **kwargs) -> plt.Figure:
    """
    Generates a visualization of the Nelder-Mead optimization path.

    This method plots the progression of the Nelder-Mead optimization algorithm in the parameter
    space, using triangles to represent each iteration and color coding for iterations and errors.
    To use, BinaryLiquid object field 'nmpath' must be initialized.

    Returns:
        plt.Figure: The generated plot figure.
    """
    if bl.nmpath is None:
        raise ValueError(
            "Underlying BinaryLiquid object has no Nelder-Mead path! Generate using `fit_parameters`"
        )
    plot_a_params = kwargs.get("plot_a_params", False)
    # nmpath columns to plot: the constant (a) subterm of each order, or the
    # format's Nelder-Mead guess parameters (one home for the former 3 branches).
    fmt = bl.xs_mix.format
    if plot_a_params:
        axes_labels = [f"L{k}_a" for k in fmt.orders[:2]]
        axis_idx = [fmt.param_names.index(n) for n in axes_labels]
    else:
        axes_labels = list(fmt.guess_params)
        axis_idx = list(bl.xs_mix.guess_param_indices)
    fig, ax = plt.subplots(figsize=(8, 5))
    num_iters = bl.nmpath.shape[2]
    total_iters_for_scale = int(kwargs.get("nmp_total_iters", num_iters))
    if total_iters_for_scale < 1:
        total_iters_for_scale = num_iters if num_iters > 0 else 1
    total_iters_for_scale = max(total_iters_for_scale, num_iters)

    # Determine the range of temperature deviations (tdev_range)
    tdev_range = [None, None]
    for i in range(num_iters):
        path_i = bl.nmpath[:, [*axis_idx, -1], i]

        t_devs = [float(num) for num in path_i[:, -1] if num != float("inf")]
        if t_devs:
            tdev_range[0] = (
                min(t_devs) if tdev_range[0] is None else min(tdev_range[0], min(t_devs))
            )
            tdev_range[1] = (
                max(t_devs) if tdev_range[1] is None else max(tdev_range[1], max(t_devs))
            )

    override_obj_range = kwargs.get("objective_range", None)
    if override_obj_range is not None:
        if not (isinstance(override_obj_range, (list, tuple)) and len(override_obj_range) == 2):
            raise ValueError(
                "kwarg 'objective_range' must be a 2-item tuple/list: (min_obj, max_obj)"
            )
        tdev_range = [float(override_obj_range[0]), float(override_obj_range[1])]

    if tdev_range[0] is None or tdev_range[1] is None:
        tdev_range = [0.0, 1.0]
    elif tdev_range[0] == tdev_range[1]:
        eps = max(1e-9, abs(tdev_range[0]) * 1e-6)
        tdev_range = [tdev_range[0] - eps, tdev_range[1] + eps]

    # Triangle color mapping (iteration-based)
    sm1 = cm.ScalarMappable(cmap="winter", norm=LogNorm(vmin=1, vmax=total_iters_for_scale))
    triangle_colors = sm1.to_rgba(np.arange(1, num_iters + 1, 1))
    max_tick_exp = int(math.floor(np.log2(total_iters_for_scale)))
    ticks = [2**exp for exp in range(max_tick_exp + 1)]
    top_tick = total_iters_for_scale
    lower_tick = ticks[-1]
    if top_tick > lower_tick:
        log_gap_fraction = math.log(top_tick / lower_tick, 2)
        if log_gap_fraction >= 0.18:
            ticks.append(top_tick)
    cbar1 = fig.colorbar(sm1, ax=ax, aspect=14)
    cbar1.minorticks_off()
    cbar1.set_ticks(ticks)
    cbar1.set_ticklabels(ticks)
    cbar1.set_label("Nelder-Mead Iteration", style="italic", labelpad=8, fontsize=12)

    # Marker color mapping (temperature deviation-based)
    sm2 = cm.ScalarMappable(cmap="autumn", norm=plt.Normalize(tdev_range[0], tdev_range[1]))
    cbar2 = fig.colorbar(sm2, ax=ax, aspect=14)
    cbar2.set_label("Objective Function Value", style="italic", labelpad=10, fontsize=12)

    plotted_points = []

    for i in range(num_iters):
        path_i = bl.nmpath[:, [*axis_idx, -1], i]

        triangle = path_i[:, :-1]  # Extract triangle vertices
        t_devs = path_i[:, -1]  # Extract objective values

        # Plot triangles connecting vertices
        coordinates = [triangle[j, :] for j in range(triangle.shape[0])]
        pair_combinations = list(combinations(coordinates, 2))
        for combo in pair_combinations:
            line = np.array(combo)
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=triangle_colors[i],
                linewidth=(2 - 1.7 * (i / num_iters)),
                zorder=0,
            )

        # Plot markers at triangle vertices
        for point, t_dev in zip(triangle, t_devs):
            if list(point) in plotted_points:
                continue
            if t_dev != float("inf"):
                marker_color = sm2.to_rgba(float(t_dev))
                ax.scatter(
                    point[0],
                    point[1],
                    s=(55 - 54.7 * (i / num_iters)),
                    color=marker_color,
                    marker="^",
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=1,
                )
            else:
                ax.scatter(
                    point[0],
                    point[1],
                    s=(45 - 44.7 * (i / num_iters)),
                    color="black",
                    label="Incalculable MAE",
                    marker="^",
                    zorder=1,
                )
            plotted_points.append(list(point))

    # Add legend and adjust axis labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())

    # Adjust axis limits for better scaling
    x_axis_range = kwargs.get("x_axis_range", None)
    y_axis_range = kwargs.get("y_axis_range", None)
    if x_axis_range is not None:
        if not (isinstance(x_axis_range, (list, tuple)) and len(x_axis_range) == 2):
            raise ValueError("kwarg 'x_axis_range' must be a 2-item tuple/list: (xmin, xmax)")
        ax.set_xlim(float(x_axis_range[0]), float(x_axis_range[1]))
    if y_axis_range is not None:
        if not (isinstance(y_axis_range, (list, tuple)) and len(y_axis_range) == 2):
            raise ValueError("kwarg 'y_axis_range' must be a 2-item tuple/list: (ymin, ymax)")
        ax.set_ylim(float(y_axis_range[0]), float(y_axis_range[1]))
    if x_axis_range is None or y_axis_range is None:
        ax.autoscale()
        if y_axis_range is None:
            ly, uy = ax.get_ylim()
            ax.set_ylim((uy + ly) / 2 - (uy - ly) / 2 * 1.1, (uy + ly) / 2 + (uy - ly) / 2 * 1.1)
        if x_axis_range is None:
            lx, ux = ax.get_xlim()
            ax.set_xlim((ux + lx) / 2 - (ux - lx) / 2 * 1.1, (ux + lx) / 2 + (ux - lx) / 2 * 1.1)
    ax.set_xlabel(axes_labels[0], fontweight="semibold", fontsize=12)
    ax.set_ylabel(axes_labels[1], fontweight="semibold", fontsize=12)
    fig.tight_layout(pad=1.5)

    # Use scientific notation for tick labels
    def set_sci_notation(axis, which="both"):
        # ticks = axis.get_majorticklocs()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        if which in ("x", "both"):
            ax.xaxis.set_major_formatter(formatter)
        if which in ("y", "both"):
            ax.yaxis.set_major_formatter(formatter)
        fig.canvas.draw_idle()

    set_sci_notation(ax.xaxis, which="x")
    set_sci_notation(ax.yaxis, which="y")

    return fig
