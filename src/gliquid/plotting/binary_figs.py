"""Diagnostic/analysis figure builders over a BinaryLiquid-like model.

Raw TX scatter, HSX hull debug, MPDS-vs-MP low-T phase comparison, T=0K DFT convex hull
(+ Gibbs overlays), and the Nelder-Mead path figure. Each function takes the model object
(duck-typed ``bl``) and,
where colors are drawn, an explicit phase->color map (BLPlotter passes its cached
build_phase_color_map result). No gliquid.binary import -- import order:
{api, mpds, solution, phase, plotting.style, plotting.binary_tx} <- plotting.binary_figs
<- binary. (binary_tx is presentation-only and imports nothing from here, so the assessed-
liquidus segment splitter is shared from there.)
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sympy as sp
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
from pymatgen.analysis.phase_diagram import (
    PDPlotter,  # The PMG PDPlotter source code is modified here
)
from pymatgen.core import Composition
from pymatgen.util.string import htmlify

import gliquid.mpds as mpds
import gliquid.solution as solution
from gliquid.cache import is_imputed_entry_dict, non_dft_entry_kind, non_dft_entry_label
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.plotting.binary_tx import _assessed_liquidus_segments
from gliquid.plotting.style import (
    ASSESSED_LIQUIDUS_COLOR,
    IMPUTED_BORDER_DASHES,
    IMPUTED_BORDER_WIDTH,
    IMPUTED_EDGE_COLOR,
    IMPUTED_LEGEND_HANDLE_SIZE,
    IMPUTED_LEGEND_HANDLE_TEXT_PAD,
    IMPUTED_MARKER_COLOR,
    format_phase_display_name,
    imputed_band_alphas,
    non_dft_legend_label,
    standard_formula,
)
from gliquid.solution import comp_symbols, t_sym

#: Fill colour of a thermal-form segment on the phase-comparison MPDS panel, by decomposition
#: type.
THERMAL_FORM_COLORS = {"congruent": "blue", "incongruent": "purple", "transition": "purple"}

#: Colour and width of the divider drawn between two stacked thermal-form segments.
THERMAL_FORM_DIVIDER_COLOR = "white"
THERMAL_FORM_DIVIDER_WIDTH = 1.2
# Shared by the MPDS-panel fills and dividers; they paint in call order within it.
THERMAL_FORM_ZORDER = 2

logger = logging.getLogger(__name__)

_x_vals = solution.x_vals
xb_sym = comp_symbols(2)[0]


def _display_name(bl, label: str) -> str:
    """Legend name via the shared style rule (SS phases carry their component pair)."""
    return format_phase_display_name(label, bl.ss_models, bl.components)


def imputed_hull_entries(dft_ch) -> list:
    """Stable entries on ``dft_ch`` that phase-energy imputation synthesized.

    Identified from the entry TAG, never by matching names against ``bl.phases``: an imputed
    entry is built at a FRACTIONAL composition, which pymatgen reduces to a formula that need
    not equal the MPDS label the phase carries. ``get_low_temp_phase_data`` keys its MP panel
    on ``entry.name``, so reading the names off the entries here is what makes the two agree.

    Args:
        dft_ch: a pymatgen ``PhaseDiagram``.

    Returns:
        list: the tagged entries, in hull order. Empty for a hull with none.
    """
    return [e for e in dft_ch.stable_entries if is_imputed_entry_dict(e.as_dict())]


def non_dft_hull_entries(dft_ch) -> list:
    """Stable entries on ``dft_ch`` that stand in for DFT compounds, imputed or user-defined.

    Identified by :func:`gliquid.cache.non_dft_entry_kind`, so every elemental entry is left
    out.

    Args:
        dft_ch: a pymatgen ``PhaseDiagram``.

    Returns:
        list: the entries, in hull order. Empty for a hull with none.
    """
    return [e for e in dft_ch.stable_entries if non_dft_entry_kind(e) is not None]


def standardized_vertex_label(entry, components) -> str:
    """Compound label for one hull vertex, in the system's own component order.

    An imputed entry sits at a digitized FRACTIONAL composition, whose reduced formula has no
    small integer ratio, so its MPDS label supplies the stoichiometry; a user-defined entry's
    label is read the same way. Every other vertex -- and any non-DFT one whose label will
    not parse -- is written from the entry composition.

    Args:
        entry: a hull entry; its ``composition``, ``entry_id`` and ``data`` are read.
        components: the system's ordered component list.

    Returns:
        str: the label, with no HTML markup.
    """
    data = getattr(entry, "data", None) or {}
    label = data.get("label") if non_dft_entry_kind(entry) is not None else None
    if label:
        try:
            return standard_formula(label, components)
        except ValueError:
            pass
    return standard_formula(entry.composition, components)


def _standardize_vertex_labels(fig: go.Figure, entries: list, components) -> go.Figure:
    """Write every stable vertex under the one compound rule instead of pymatgen's formula.

    Rewrites the ``"Labels (stable)"`` text entries and the ``"Stable"`` hovertext strings of a
    ``PDPlotter`` figure in place, matched on the reduced formula the traces already carry;
    vertex positions are untouched. Pure endpoints keep their element symbol.

    Args:
        fig: the figure ``PDPlotter.get_plot`` returned, or one built from its traces.
        entries: the hull's stable entries.
        components: the system's ordered component list.

    Returns:
        go.Figure: ``fig``.
    """
    swaps = {}
    for entry in entries:
        formula = htmlify(entry.composition.reduced_formula)
        label = htmlify(standardized_vertex_label(entry, components))
        if label != formula:
            swaps[formula] = label
    if not swaps:
        return fig
    # Longest first, so a shorter formula is not replaced inside a longer one.
    ordered = sorted(swaps, key=len, reverse=True)
    for trace in fig.data:
        if trace.name == "Labels (stable)" and isinstance(trace.text, (list, tuple)):
            trace.text = [swaps.get(text, text) for text in trace.text]
        elif trace.name == "Stable" and isinstance(trace.hovertext, (list, tuple)):
            hover = list(trace.hovertext)
            for formula in ordered:
                hover = [text.replace(formula, swaps[formula]) for text in hover]
            trace.hovertext = hover
    return fig


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
        # Split at undigitized holes; one legend entry for the whole curve.
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


# Panel labels and the imputed legend entry of the phase-comparison figure share one size.
PHASE_COMPARISON_FONTSIZE = 11

# Bottom corners the imputed key may occupy, in the order they are tried.
_IMPUTED_LEGEND_CORNERS = ("lower left", "lower right")


def _band_pixel_boxes(ax, bands) -> list[Bbox]:
    """Pixel bounding boxes of data-space rectangles ``(lb, ub, bottom, top)`` on ``ax``."""
    boxes = []
    for lb, ub, bottom, top in bands:
        (x0, y0), (x1, y1) = ax.transData.transform([(lb, bottom), (ub, top)])
        boxes.append(Bbox.from_extents(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return boxes


def _overlap_area(box: Bbox, boxes) -> float:
    """Total area, in square pixels, by which ``box`` overlaps each of ``boxes``."""
    total = 0.0
    for other in boxes:
        intersection = Bbox.intersection(box, other)
        if intersection is not None:
            total += intersection.width * intersection.height
    return total


def _place_legend_clear_of(fig, ax, bands) -> None:
    """Move ``ax``'s legend to whichever bottom corner it overlaps ``bands`` least.

    The first corner is kept unless it overlaps; otherwise the corner with the smaller total
    overlap area wins and a tie goes to the later one. Call once ``ax``'s limits and ticks are
    final: the decision is read in pixels off the current data transform.

    Args:
        fig: the figure ``ax`` belongs to.
        ax: the axes carrying the legend; a no-op when it carries none.
        bands: data-space rectangles ``(lb, ub, bottom, top)`` drawn on ``ax``.
    """
    legend = ax.get_legend()
    if legend is None or not bands:
        return
    boxes = _band_pixel_boxes(ax, bands)
    areas = []
    for corner in _IMPUTED_LEGEND_CORNERS:
        legend.set_loc(corner)
        fig.draw_without_rendering()
        areas.append(_overlap_area(legend.get_window_extent(), boxes))
        if areas[0] == 0.0:
            break
    best = min(range(len(areas)), key=lambda i: (areas[i], -i))
    legend.set_loc(_IMPUTED_LEGEND_CORNERS[best])


def render_phase_comparison(bl) -> plt.Figure:
    """
    Generates a phase comparison plot showing congruent and incongruent phases
    from MPDS and MP data. The plot consists of two subplots displaying phases
    in different temperature and magnitude ranges. On the MPDS panel a compound is a
    stack of one fill per thermal form, coloured by its own decomposition type
    (:data:`THERMAL_FORM_COLORS`) and all sharing the compound's own composition range;
    a thin divider (:data:`THERMAL_FORM_DIVIDER_COLOR`) marks each internal transition.

    Returns:
        plt.Figure: The generated phase comparison plot.
    """
    # Extract low-temperature phase data from MPDS and MP
    (
        [mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp, mpds_forms],
        [mp_phases, mp_phases_ebelow, min_form_e],
    ) = mpds.get_low_temp_phase_data(bl.mpds_json, bl.dft_ch)

    # Filter out phases containing parentheses
    mpds_congruent_phases = {
        key: value for key, value in mpds_congruent_phases.items() if "(" not in key
    }
    mpds_incongruent_phases = {
        key: value for key, value in mpds_incongruent_phases.items() if "(" not in key
    }
    mpds_forms = {key: value for key, value in mpds_forms.items() if "(" not in key}

    # Create subplots with specific layout
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 2),
        gridspec_kw={"hspace": 0, "left": 0.117, "bottom": 0.13, "right": 0.909},
    )

    # Data-space rectangles (lb, ub, bottom, top) plot_phases has drawn, per axes.
    drawn_bands = {}

    def plot_phases(
        ax,
        source,
        color,
        alpha=0.5,
        imputed_names=(),
        imputed_alpha=None,
        imputed_outline=False,
    ):
        """
        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            source (dict): Phase data with keys as phase names and values as bounds/magnitudes.
            color (str): Color for the phase fill.
            alpha (float): Transparency for the fill.
            imputed_names (set): Names in ``source`` whose energy was imputed rather than
                computed. Marked so the MP panel does not present them as DFT data.
            imputed_alpha (float | None): Fill transparency for an imputed band; ``alpha`` when
                None.
            imputed_outline (bool): Draw a dotted border inside an imputed band's own bounds.
        """
        for name, ((lb, ub), mag) in source.items():
            # Ensure a minimum width for labeling
            if ub - lb < 0.026:
                ave = (ub + lb) / 2
                lb = ave - 0.013
                ub = ave + 0.013
            is_imputed = name in imputed_names
            bottom, top = min(0, mag), max(0, mag)
            drawn_bands.setdefault(ax, []).append((lb, ub, bottom, top))
            fill_alpha = imputed_alpha if is_imputed and imputed_alpha is not None else alpha
            ax.fill_betweenx(
                [bottom, top],
                lb,
                ub,
                color=color,
                alpha=fill_alpha,
                linewidth=0.0,
            )
            if is_imputed and imputed_outline:
                border = mpatches.Rectangle(
                    (lb, bottom),
                    ub - lb,
                    top - bottom,
                    facecolor="none",
                    edgecolor=IMPUTED_EDGE_COLOR,
                    linestyle=IMPUTED_BORDER_DASHES,
                    linewidth=IMPUTED_BORDER_WIDTH,
                )
                ax.add_patch(border)
                # Clipped to its own path so the stroke stays inside the band.
                border.set_clip_path(border)
            ax.set_xlim(0, 1)
            ax.margins(x=0, y=0)

    def plot_form_segments(ax, forms, alpha=0.5):
        """One fill per thermal-form record, coloured by the record's decomposition type.

        Every record of one compound shares one ``comp_range`` (``get_low_temp_phase_data``'s
        contract), so the width is read once per compound. A thin divider is drawn at each
        internal boundary -- the lower record's own ``t_hi`` -- so two same-coloured stacked
        forms stay visually distinct. Compounds are painted widest first, each compound's
        dividers directly over its own fills, so a narrower compound sits in front of any
        wider one it overlaps.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            forms (dict): Compound name -> list of form records, as ``get_low_temp_phase_data``
                returns them.
            alpha (float): Transparency for the fill.
        """
        spans = []
        for records in forms.values():
            lb, ub = records[0]["comp_range"]
            # Ensure a minimum width for labeling
            if ub - lb < 0.026:
                ave = (ub + lb) / 2
                lb = ave - 0.013
                ub = ave + 0.013
            spans.append((lb, ub, records))
        for lb, ub, records in sorted(spans, key=lambda span: span[0] - span[1]):
            for record in records:
                bottom, top = record["t_lo"], record["t_hi"]
                drawn_bands.setdefault(ax, []).append((lb, ub, bottom, top))
                ax.fill_betweenx(
                    [bottom, top],
                    lb,
                    ub,
                    color=THERMAL_FORM_COLORS[record["decomp"]],
                    alpha=alpha,
                    linewidth=0.0,
                    zorder=THERMAL_FORM_ZORDER,
                )
            for record in records[:-1]:
                ax.plot(
                    [lb, ub],
                    [record["t_hi"], record["t_hi"]],
                    color=THERMAL_FORM_DIVIDER_COLOR,
                    linewidth=THERMAL_FORM_DIVIDER_WIDTH,
                    zorder=THERMAL_FORM_ZORDER,
                )
            ax.set_xlim(0, 1)
            ax.margins(x=0, y=0)

    # Imputed and user-defined entries in bl.dft_ch are marked on the MP panel.
    non_dft_entries = non_dft_hull_entries(bl.dft_ch)
    imputed_names = {e.name for e in non_dft_entries}

    # Plot phases for both subplots
    plot_form_segments(ax1, mpds_forms)
    # Every MP band is drawn twice; these two alphas hold the overlap's hue.
    under, over = imputed_band_alphas(0.5)
    plot_phases(
        ax2,
        mp_phases,
        "orange",
        imputed_names=imputed_names,
        imputed_alpha=under,
        imputed_outline=True,
    )
    plot_phases(ax2, mp_phases_ebelow, "red", imputed_names=imputed_names, imputed_alpha=over)
    if imputed_names:
        ax2.legend(
            handles=[
                mpatches.Patch(
                    facecolor="white",
                    edgecolor=IMPUTED_EDGE_COLOR,
                    linestyle=IMPUTED_BORDER_DASHES,
                    linewidth=IMPUTED_BORDER_WIDTH,
                    label=non_dft_legend_label(non_dft_entry_kind(e) for e in non_dft_entries),
                )
            ],
            loc=_IMPUTED_LEGEND_CORNERS[0],
            fontsize=PHASE_COMPARISON_FONTSIZE,
            frameon=False,
            handlelength=IMPUTED_LEGEND_HANDLE_SIZE,
            handleheight=IMPUTED_LEGEND_HANDLE_SIZE,
            handletextpad=IMPUTED_LEGEND_HANDLE_TEXT_PAD,
        )

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

    ax1.set_ylabel(
        "MPDS",
        fontsize=PHASE_COMPARISON_FONTSIZE,
        rotation=90,
        labelpad=5,
        fontweight="semibold",
    )
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

    ax2.set_ylabel(
        "MP",
        fontsize=PHASE_COMPARISON_FONTSIZE,
        rotation=90,
        labelpad=5,
        fontweight="semibold",
    )
    ax2.yaxis.set_label_position("right")
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.set_xticks(np.linspace(0, 1, 6), labels=[0, 20, 40, 60, 80, 100])

    # Add a title to the figure
    fig.suptitle("Low Temperature Phase Comparison", fontweight="semibold")

    # The corner is decided in pixels from ax2's final limits and ticks.
    _place_legend_clear_of(fig, ax2, drawn_bands.get(ax2, ()))

    return fig


def render_dft_hull(bl, plot_type: str, **kwargs) -> go.Figure:
    """
    Generates a convex hull plot or phase diagram visualization.

    Args:
        bl: The model to plot (duck-typed ``BinaryLiquid``); its ``dft_ch``, ``components``,
            ``component_data`` and ``eqs`` are read.
        plot_type (str): The type of plot to generate ('ch' or 'ch+g').
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
            ga = np.float64(bl.eqs["ga"].subs({t_sym: T}) / EV_ATOM_TO_J_MOL)
            gb = np.float64(bl.eqs["gb"].subs({t_sym: T}) / EV_ATOM_TO_J_MOL)
            if name == "":
                name += "Ideal " if A == 0 and C == 0 else ""
                name += (
                    "Liquid T=" + str(int(T)) + "K"
                    if t_units == "K"
                    else str(int(T - 273.15)) + "C"
                )

            return go.Scatter(
                x=_x_vals,
                y=[ga] + [g / EV_ATOM_TO_J_MOL for g in gliq_vals] + [gb],
                mode="lines",
                name=name,
            )

        # The G curves first, then the hull traces.
        params = bl.get_params()
        g_curves = [
            get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp)
            for temp in reversed(t_vals)
        ]
        # One dashed Gibbs curve per solid-solution phase per temperature (all of them with
        # show_unstable=True).
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

    # Imputed and user-defined vertices are overlaid on PDPlotter's hull.
    _standardize_vertex_labels(fig, bl.dft_ch.stable_entries, bl.components)
    imputed = non_dft_hull_entries(bl.dft_ch)
    if imputed:
        b_element = str(Composition(bl.components[1]).elements[0])
        fig.add_trace(
            go.Scatter(
                x=[
                    e.composition.fractional_composition.as_dict().get(b_element, 0.0)
                    for e in imputed
                ],
                y=[bl.dft_ch.get_form_energy_per_atom(e) for e in imputed],
                mode="markers",
                # an -open symbol strokes in marker.color; marker.line.color is ignored.
                marker=dict(
                    symbol="circle-open",
                    size=14,
                    color=IMPUTED_MARKER_COLOR,
                    line=dict(width=3),
                ),
                name=non_dft_legend_label(non_dft_entry_kind(e) for e in imputed),
                hovertext=[non_dft_entry_label(e) for e in imputed],
            )
        )

    # PDPlotter's vertex-label trace is kept, without its legend entry.
    for trace in fig.data:
        if trace.name == "Labels (stable)":
            trace.showlegend = False

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
    # nmpath columns: the constant (a) subterm of each order, or the format's guess parameters.
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
    # Plain 5+ digit tick labels push this colorbar's label onto the iteration colorbar.
    obj_formatter = ScalarFormatter(useMathText=True)
    obj_formatter.set_powerlimits((-2, 4))
    cbar2 = fig.colorbar(sm2, ax=ax, aspect=14, format=obj_formatter)
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
