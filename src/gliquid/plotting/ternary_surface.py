"""Ternary 3-D figure rendering over TLI-computed data (ex-TLIPlotter draw bodies).

Extracted from TLIPlotter: the liquidus-surface figure, the single-slice hull figure,
and the isothermal contour machinery. Pure presentation --
numpy/plotly plus the leaf display module gliquid.plotting.style -- no MODEL imports;
TLIPlotter passes model-computed dataframes,
hull data, colors, and axis conditions in.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# The SAME formatter the binary field labels use, so a phase reads identically in both
# figures: greek word -> symbol, structure abbreviated inside the parentheses, formula
# digits subscripted. binary_tx imports only from plotting.style, and ternary.py already
# imports it, so this adds no cycle and no load.
from gliquid.plotting.binary_tx import _abbreviate_phase_name

# Plotly CLIPS 3-D geometry at the axis bounds, and the base triangle is drawn on the floor.
# Placed exactly at ``conds[0]`` its two back edges land on the clip plane and disappear, so it
# is lifted by a hair to sit strictly inside the range -- 0.1% of the temperature span, about
# 3 C on a 2800 C diagram and invisible at plot scale. This is what lets the z-axis stop at
# ``conds[0]``, which is absolute zero, instead of borrowing drawing room below it: the old code
# put the labels 150 C BELOW the floor and then opened the axis 200 C below it to fit them,
# which only looked harmless while ``conds[0]`` was 0 K misread as 0 C. The solid-phase segments
# are NOT lifted -- they still start at the true grid floor.
FLOOR_INSET_FRAC = 1e-3


def render_hull_slice(hull_data, components) -> go.Figure:
    """The 3-D single-slice lower-hull figure over ``get_convex_hull`` output."""
    slice_df = hull_data["raw_slice_df"]
    points = hull_data["hull_points"]
    simplices = hull_data["hull_simplices"]
    transformed_points_df = hull_data["transformed_points_df"]
    T_celsius_exact = hull_data["temperature_c"]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=transformed_points_df["x0"],
            y=transformed_points_df["x1"],
            z=points[:, 2],
            i=simplices[:, 0],
            j=simplices[:, 1],
            k=simplices[:, 2],
            opacity=0.55,
            colorscale="Viridis",
            intensity=points[:, 2],
            showscale=True,
            colorbar=dict(title="G"),
            customdata=np.column_stack((slice_df["x0"], slice_df["x1"], slice_df["Phase"])),
            hovertemplate=(
                f"x_{components[1]}: %{{customdata[0]:.3f}}<br>"
                + f"x_{components[2]}: %{{customdata[1]:.3f}}<br>"
                + "Phase: %{customdata[2]}<br>"
                + "G: %{z:.4f}<extra></extra>"
            ),
        )
    )

    for phase, group in slice_df.groupby("Phase"):
        phase_points = transformed_points_df.loc[group.index]
        fig.add_trace(
            go.Scatter3d(
                x=phase_points["x0"],
                y=phase_points["x1"],
                z=points[group.index, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=group["Colors"].iloc[0],
                    opacity=0.95,
                    line=dict(color="black", width=0.4),
                ),
                name=phase,
                customdata=np.column_stack((group["x0"], group["x1"], group["G"])),
                hovertemplate=(
                    f"<b>{phase}</b><br>"
                    + f"x_{components[1]}: %{{customdata[0]:.3f}}<br>"
                    + f"x_{components[2]}: %{{customdata[1]:.3f}}<br>"
                    + "G: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

    g_floor = float(np.min(points[:, 2]))
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3) / 2, 0, 0],
            z=[g_floor, g_floor, g_floor, g_floor],
            mode="lines",
            line=dict(color="black", width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"Single-slice lower hull at T = {T_celsius_exact:.2f} C",
        scene=dict(
            # A 3 % left inset inside the plot area: the z tick labels are drawn INSIDE the
            # gl canvas, hanging left of the axis line, so with the scene flush against the
            # canvas edge the leading digit of "1500" was clipped. Insetting the SCENE (not
            # the margin) is what gives them room, since widening the margin just moves the
            # canvas and takes the axis with it.
            domain=dict(x=[0.09, 1.0], y=[0.0, 1.0]),
            xaxis=dict(title=" ", showticklabels=False, showaxeslabels=False, showgrid=False),
            yaxis=dict(title=" ", showticklabels=False, showaxeslabels=False, showgrid=False),
            zaxis=dict(title="G"),
            bgcolor="white",
            camera=dict(projection=dict(type="orthographic")),
        ),
        # Same anchoring as render_tx_surface: anchored 'left' at x=0.95 the legend runs off
        # the paper and plotly reserves its width out of the 3D scene. See the note there.
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.72)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
        ),
    )

    return fig


def compute_isotherm_contours(liq_points, triangles):
    """Iso-temperature contour polylines across the liquidus mesh.

    The delta-T ladder is a display heuristic (denser contours on flatter surfaces);
    contours are returned in ascending-temperature order, each a list of (x0, x1, T)
    points, exactly as the retired TLIPlotter._add_isothermal_lines computed them.
    """
    # Get temperature range
    temps = liq_points[:, 2]
    temp_min, temp_max = np.min(temps), np.max(temps)
    temp_range = temp_max - temp_min

    # Choose appropriate delta_T based on range
    if temp_range <= 10:
        delta_T = 1.0
    elif temp_range <= 25:
        delta_T = 2.0
    elif temp_range <= 50:
        delta_T = 2.5
    elif temp_range <= 100:
        delta_T = 5
    elif temp_range <= 200:
        delta_T = 10
    else:
        delta_T = max(10, temp_range / 20)

    # Generate iso-temperature values
    iso_temps = np.arange(temp_min + delta_T, temp_max, delta_T)

    # For each iso-temperature, find intersection lines with triangles
    contours = []
    for iso_temp in iso_temps:
        line_segments = []

        for triangle in triangles:
            # Get the three vertices of the triangle
            v1 = liq_points[triangle[0]]
            v2 = liq_points[triangle[1]]
            v3 = liq_points[triangle[2]]

            # Find intersections of the iso-temperature plane with triangle edges
            intersections = []

            # Check each edge of the triangle
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            for p1, p2 in edges:
                # Check if iso_temp is between the temperatures of the edge endpoints
                t1, t2 = p1[2], p2[2]
                if (t1 <= iso_temp <= t2) or (t2 <= iso_temp <= t1):
                    if abs(t2 - t1) > 1e-8:  # More strict tolerance for flatter surfaces
                        # Linear interpolation to find intersection point
                        alpha = (iso_temp - t1) / (t2 - t1)
                        intersection = p1 + alpha * (p2 - p1)
                        intersection[2] = iso_temp  # Ensure exact temperature
                        intersections.append(intersection)

            # If we have exactly 2 intersections, we have a line segment
            if len(intersections) == 2:
                line_segments.append(intersections)

        # Connect line segments into continuous contours
        if line_segments:
            contours.extend(connect_line_segments(line_segments))

    return contours


def draw_isotherms(fig, contours) -> None:
    """Add computed isotherm contours to ``fig`` as white line traces."""
    for contour in contours:
        if len(contour) >= 2:  # Only plot if we have at least 2 points
            x_coords = [point[0] for point in contour]
            y_coords = [point[1] for point in contour]
            z_coords = [point[2] for point in contour]

            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


def connect_line_segments(segments, tolerance=1e-4):
    """
    Connect line segments into continuous contours.

    Args:
        segments: List of line segments, each segment is [point1, point2]
        tolerance: Distance tolerance for connecting endpoints

    Returns:
        List of connected contours, each contour is a list of points
    """
    if not segments:
        return []

    contours = []
    remaining_segments = segments.copy()

    while remaining_segments:
        # Start a new contour with the first remaining segment
        current_contour = list(remaining_segments.pop(0))

        # Keep trying to extend the contour
        extended = True
        while extended and remaining_segments:
            extended = False

            # Try to find a segment that connects to either end of current contour
            for i, segment in enumerate(remaining_segments):
                p1, p2 = segment[0], segment[1]

                # Check if segment connects to the end of current contour
                end_point = current_contour[-1]
                if np.linalg.norm(p1[:2] - end_point[:2]) < tolerance:
                    current_contour.append(p2)
                    remaining_segments.pop(i)
                    extended = True
                    break
                elif np.linalg.norm(p2[:2] - end_point[:2]) < tolerance:
                    current_contour.append(p1)
                    remaining_segments.pop(i)
                    extended = True
                    break

                # Check if segment connects to the beginning of current contour
                start_point = current_contour[0]
                if np.linalg.norm(p1[:2] - start_point[:2]) < tolerance:
                    current_contour.insert(0, p2)
                    remaining_segments.pop(i)
                    extended = True
                    break
                elif np.linalg.norm(p2[:2] - start_point[:2]) < tolerance:
                    current_contour.insert(0, p1)
                    remaining_segments.pop(i)
                    extended = True
                    break

        contours.append(current_contour)

    return contours


def render_tx_surface(
    solid_df, liq_df, liq_points, triangles, color_map, components, conds
) -> go.Figure:
    """The 3-D ternary liquidus figure: solid-phase lines, liquidus mesh, isotherms,
    legend markers, and the base-triangle axes/labels (ex-TLIPlotter._plot_tx draw tail).

    ``conds`` is the plotted temperature window in CELSIUS -- the same frame as the ``T``
    columns of ``solid_df``/``liq_df`` -- and it is the z-axis range verbatim. It already
    carries any ``temp_slider`` margin, which is folded into the temperature grid it is
    derived from (:meth:`gliquid.ternary.TernaryLiquidInterpolation._init_sys`), so this
    function takes no slider of its own; the retired one was applied a second time here.
    """
    fig = go.Figure()

    # _abbreviate_phase_name drops a greek prefix when the element has only ONE phase in
    # the figure (nothing to tell apart), so it needs the full name list, not just the one
    # being labelled. color_map's keys are exactly the phases this figure draws.
    phase_names = list(color_map)

    for label, group in solid_df.groupby("Phase"):
        fig.add_trace(
            go.Scatter3d(
                x=group["x0"],
                y=group["x1"],
                z=group["T"],
                mode="lines",
                line=dict(color=group["Colors"], width=10),
                showlegend=False,
                opacity=1,
                hovertemplate=f"<b>Phase: {_abbreviate_phase_name(label, phase_names)}</b>"
                + "<br><extra></extra>",
            )
        )

    fig.add_trace(
        go.Mesh3d(
            x=liq_df["x0"],
            y=liq_df["x1"],
            z=liq_df["T"],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.6,
            colorscale="Viridis",
            intensity=liq_df["T"],
            showscale=False,
            hovertemplate="<b>Liquidus Surface</b><br>"
            + f"x_{components[1]}: %{{customdata[0]:.3f}}<br>"
            + f"x_{components[2]}: %{{customdata[1]:.3f}}<br>"
            + "T: %{z:.1f}°C<br>"
            +
            #   'Coexistent Phases: %{customdata[2]}<br>' +
            "<extra></extra>",
            customdata=np.column_stack(
                (liq_df["x0_orig"], liq_df["x1_orig"], liq_df["coexistent_phases"])
            ),
        )
    )

    # Add iso-temperature lines
    draw_isotherms(fig, compute_isotherm_contours(liq_points, triangles))

    for phase, color in color_map.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(color=color, size=10, opacity=1.0),
                name=_abbreviate_phase_name(phase, phase_names),
                textfont=dict(size=8),
                showlegend=True,
            )
        )

    # The drawn floor: conds[0] lifted clear of the clip plane (see FLOOR_INSET_FRAC).
    floor_z = conds[0] + FLOOR_INSET_FRAC * (conds[1] - conds[0])

    fig.add_trace(
        go.Scatter3d(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3) / 2, 0, 0],
            z=[floor_z] * 4,
            mode="lines",
            line=dict(color="black", width=5),
            name="axes",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[-0.02, 0.48, 0.98, -0.02],
            y=[0.02, np.sqrt(3) / 2 + 0.02, 0.02, 0.02],
            # Anchored ON the base triangle, with the drop to below it taken in SCREEN space
            # (textposition) rather than as a temperature. conds[0] is absolute zero, so there
            # is no room under the floor to borrow: a z coordinate below it would be an
            # impossible temperature, and plotly clips out-of-range 3-D text, so the labels
            # would simply vanish -- which is what the old ``conds[0] - 150`` anchor and the
            # matching ``- 200`` on the z-axis range were paying for.
            z=[floor_z] * 4,
            mode="text",
            text=[f"<b>{components[0]}</b>", f"<b>{components[2]}</b>", f"<b>{components[1]}</b>"],
            textposition="bottom center",
            showlegend=False,
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        # The legend is anchored INSIDE the paper (xanchor='right' at x=0.99), not started
        # at x=0.95 and left to run off the edge. Anchored 'left' there, plotly treats the
        # box as sitting outside the plot area and reserves its full width -- measured at
        # 178 px of a 700 px figure -- which it takes off the 3D scene. Together with the
        # margins below that left the WebGL canvas at 484x800 in a 700x900 figure, 61% of
        # the area, and it is that canvas (not the figure) that clips the view when the
        # user zooms. Anchored right and overlaid, the same legend costs nothing: 700x860,
        # 96%. `scene.domain` and `aspectmode` are NOT involved -- domain already spans
        # [0,1]x[0,1] and aspectratio only moves the cube around inside the canvas; both
        # were measured to change nothing (dev/scripts/_generated/probe_ternary_scene_sizing.py).
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.72)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
        ),
        autosize=True,
        # The z-axis title, rotated so it reads upwards and the unit lands at the top of
        # the axis, matching the binary figures. textangle=-90 is a screen-space rotation:
        # it does NOT follow the axis if the user orbits the scene, which is the cost of
        # plotly not exposing an angle on the built-in title.
        annotations=[
            dict(
                text="Temperature (°C)",
                textangle=-90,
                # NEGATIVE paper x: paper coordinates span the area INSIDE the margins,
                # so x=0 still sits over the scene's own tick labels. -0.035 of a ~660 px
                # plot area is ~23 px, which lands the text in the left margin beside them.
                xref="paper", yref="paper",
                x=-0.03, y=0.5,
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=17, color="black"),
            )
        ],
        # A 3D scene draws its own ticks INSIDE the canvas, so r/b margins buy nothing; t
        # leaves room for the figure title. l is the ONE exception: the z-axis title is a
        # paper annotation now (see below) and at l=0 it lands on top of the scene's own
        # tick labels. 32 px is what clears them, and costs 5 points of the 96% area.
        margin=dict(l=30, r=0, b=0, t=40),
        scene=dict(
            zaxis=dict(
                # Exactly the plotted window: conds IS the temperature grid in Celsius, so the
                # axis runs from absolute zero to the top of the grid and nothing drawn can
                # fall outside it. ``temp_slider`` is NOT re-applied here -- it is already
                # folded into the grid, hence into conds, by _init_sys; the retired
                # ``- 200 - temp_slider[0]`` / ``- 200 + temp_slider[1]`` counted it twice.
                range=[conds[0], conds[1]],
                # Blank, and drawn as a paper annotation below instead. plotly renders a
                # gl3d axis title inside the WebGL layer and its schema exposes only
                # `text` and `font` for it -- there is no angle attribute (checked against
                # plotly 2.35.2), and it comes out reading downwards, putting the unit at
                # the BOTTOM of the axis. An annotation is the only way to set the angle.
                title="",
            ),
            xaxis=dict(
                title=" ",
                showticklabels=False,
                showaxeslabels=False,
                showgrid=False,
            ),
            yaxis=dict(
                title=" ",
                showticklabels=False,
                showaxeslabels=False,
                showgrid=False,
            ),
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            bgcolor="white",
            camera=dict(
                projection=dict(type="orthographic"),
            ),
        ),
    )

    return fig
