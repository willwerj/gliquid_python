"""Ternary 3-D figure rendering over TLI-computed data.

The liquidus-surface figure, the single-slice hull figure, the isothermal contour
machinery, and the JavaScript the liquidus figure runs in a browser (:func:`post_script`,
:func:`dash_clientside_function`, :class:`LiquidusFigure`). Pure presentation -- numpy/plotly
plus the gliquid.plotting modules style, export, liquidus_fields, phase_highlight and
ternary_labels -- no MODEL imports; TLIPlotter passes model-computed dataframes, hull data, colors,
and axis conditions in.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# The binary field-label formatter.
from gliquid.plotting import liquidus_fields, phase_highlight, ternary_labels
from gliquid.plotting.export import side_by_side_html
from gliquid.plotting.style import (
    FIELD_OUTLINE_COLOR,
    FIELD_OUTLINE_WIDTH,
    GAP_OUTLINE_COLOR,
    GAP_OUTLINE_DASH,
    IMPUTED_LEGEND_LABEL,
    USER_DEFINED_LEGEND_LABEL,
    abbreviate_phase_name,
)

# Lift of the base triangle off ``conds[0]``, as a fraction of the temperature span; solid-phase
# segments start at the grid floor.
FLOOR_INSET_FRAC = 1e-3

# Decimal places of the coordinates, and of the temperatures, of a highlighted figure's meshes.
FIELD_COORD_DECIMALS = 6
FIELD_T_DECIMALS = 3

# A 3-D legend draws every line key solid, so an imputed phase is named rather than keyed.
IMPUTED_SUFFIX = f" ({IMPUTED_LEGEND_LABEL.lower()})"
USER_DEFINED_SUFFIX = f" ({USER_DEFINED_LEGEND_LABEL.lower()})"

# Height in pixels of a notebook cell's figure without a ``layout.height``.
NOTEBOOK_HEIGHT = 600

# Margin added around everything drawn in the plane when setting the x and y axis ranges.
PLANE_PAD = 0.02

# The legend box, at the top right corner and anchored inside the paper.
_LEGEND = dict(
    x=0.99,
    y=0.99,
    xanchor="right",
    yanchor="top",
    bgcolor="rgba(255,255,255,0.72)",
    bordercolor="rgba(0,0,0,0.15)",
    borderwidth=1,
)

# The outline of the base triangle, closed, as plot x and y.
_TRIANGLE_X = [0, 0.5, 1, 0]
_TRIANGLE_Y = [0, ternary_labels._H, 0, 0]


def composition_at_pct(comp, components) -> np.ndarray:
    """The (n, 3) at. % of each element of ``components``, in alphabetical order (the order of a
    plotter's ``sys_name``), to one decimal place.

    ``comp`` is the (n, 2) mole fractions of ``components[1]`` and ``components[2]``; the rest is
    ``components[0]``.
    """
    comp = np.asarray(comp, dtype=float)
    fractions = dict(zip(components, (1.0 - comp[:, 0] - comp[:, 1], comp[:, 0], comp[:, 1])))
    return np.round(np.column_stack([100.0 * fractions[e] for e in sorted(components)]), 1)


def composition_hover(components) -> str:
    """Hover lines ``<element>: <x> at. %`` for customdata columns 0-2 as filled by
    :func:`composition_at_pct`."""
    return "".join(
        f"{element}: %{{customdata[{n}]:.1f}} at. %<br>"
        for n, element in enumerate(sorted(components))
    )


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
            customdata=np.column_stack(
                (composition_at_pct(slice_df[["x0", "x1"]], components), slice_df["Phase"])
            ),
            hovertemplate=(
                composition_hover(components)
                + "Phase: %{customdata[3]}<br>"
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
                customdata=np.column_stack(
                    (composition_at_pct(group[["x0", "x1"]], components), group["G"])
                ),
                hovertemplate=(
                    f"<b>{phase}</b><br>"
                    + composition_hover(components)
                    + "G: %{customdata[3]:.4f}<extra></extra>"
                ),
            )
        )

    g_floor = float(np.min(points[:, 2]))
    fig.add_trace(
        go.Scatter3d(
            x=_TRIANGLE_X,
            y=_TRIANGLE_Y,
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
            # Left inset of the scene, which holds the z tick labels drawn inside the gl canvas.
            domain=dict(x=[0.09, 1.0], y=[0.0, 1.0]),
            xaxis=dict(title=" ", showticklabels=False, showaxeslabels=False, showgrid=False),
            yaxis=dict(title=" ", showticklabels=False, showaxeslabels=False, showgrid=False),
            zaxis=dict(title="G"),
            bgcolor="white",
            camera=dict(projection=dict(type="orthographic")),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=_LEGEND,
    )

    return fig


def compute_isotherm_contours(liq_points, triangles):
    """Iso-temperature contour polylines across the liquidus mesh.

    The delta-T ladder is a display heuristic (denser contours on flatter surfaces);
    contours are returned in ascending-temperature order, each a (k, 3) array of
    (x0, x1, T) points. A contour crosses each mesh edge whose ends lie on either side of
    its temperature (a vertex at the temperature counts as above it), and its pieces are
    joined through the edges they share.
    """
    liq_points = np.asarray(liq_points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
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

    edges, edge_of = np.unique(liquidus_fields._face_edges(triangles), axis=0, return_inverse=True)
    edge_of = edge_of.reshape(3, -1).T
    start, end = liq_points[edges[:, 0]], liq_points[edges[:, 1]]

    contours = []
    for iso_temp in np.arange(temp_min + delta_T, temp_max, delta_T):
        above = temps >= iso_temp
        crossed = above[edges[:, 0]] != above[edges[:, 1]]
        hit = crossed[edge_of]
        two = hit.sum(axis=1) == 2
        segments = edge_of[two][hit[two]].reshape(-1, 2)
        if not len(segments):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = (iso_temp - start[:, 2]) / (end[:, 2] - start[:, 2])
            points = start + alpha[:, None] * (end - start)
        points[:, 2] = iso_temp
        contours.extend(
            points[line] for line in liquidus_fields.chain_segments(map(tuple, segments))
        )
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


def render_tx_surface(
    solid_df,
    liq_df,
    triangles,
    color_map,
    components,
    conds,
    imputed_phases=(),
    user_phases=(),
    partition=None,
    highlight_phase=None,
    interactive_legend=False,
) -> LiquidusFigure:
    """The 3-D ternary liquidus figure: solid-phase lines, liquidus mesh, isotherms,
    legend markers, and the base triangle with its composition ticks and flat labels
    (:mod:`gliquid.plotting.ternary_labels`), the labels in the view of the figure's camera.

    ``conds`` is the plotted temperature window in CELSIUS -- the same frame as the ``T``
    columns of ``solid_df``/``liq_df`` -- and it is the z-axis range verbatim. It already
    carries any ``temp_slider`` margin, which is folded into the temperature grid it is
    derived from (:meth:`gliquid.ternary.TernaryLiquidInterpolation._init_sys`), so this
    function takes no slider of its own.

    ``imputed_phases`` names phases whose energy was imputed; their lines are drawn dashed
    and their legend entries and hover text carry an ``(imputed)`` suffix. ``user_phases``
    names user-defined phases, drawn the same way with a ``(user-defined)`` suffix.

    ``partition`` (a :class:`gliquid.plotting.liquidus_fields.Partition`) replaces the mesh of
    ``liq_df`` nodes as the drawn surface, and its fields are the phases' liquidus fields. A field
    of a phase with no legend entry (the two-liquid field) is drawn with the rest of the surface,
    and a dotted grey line, shown in every view, marks where it borders other fields; where a
    highlighted field borders it, that field's outline takes the line's place. Without it the
    surface is the node mesh and has no fields. ``highlight_phase`` draws that solid phase's field
    in full colour, outlined, with the rest of the surface, the other legend entries and the other
    phases' lines dimmed; ``interactive_legend`` embeds every such view for a browser-side legend
    handler (see :mod:`gliquid.plotting.phase_highlight`). Without a ``partition`` neither draws
    anything.

    Raises:
        ValueError: ``highlight_phase`` is not a solid phase of the figure.
    """
    legend_phases = [phase for phase in color_map if phase != "L"]
    if highlight_phase is not None and highlight_phase not in legend_phases:
        raise ValueError(
            f"highlight_phase {highlight_phase!r} is not a solid phase of this figure; "
            f"choose one of {legend_phases}."
        )
    highlight = partition is not None and (highlight_phase is not None or interactive_legend)
    tags = phase_highlight.HighlightTraces(liquidus="gliquid-liquidus") if highlight else None

    def tagged(key, role, phase=None):
        if not highlight:
            return {}
        tag = {"meta": _meta(key, role, phase)}
        return {**tag, "visible": True} if role in ("liquidus", "legend") else tag

    fig = LiquidusFigure()

    # abbreviate_phase_name takes the full name list of the figure.
    phase_names = list(color_map)
    imputed_phases = set(imputed_phases)
    user_phases = set(user_phases)

    def display_name(phase):
        name = abbreviate_phase_name(phase, phase_names)
        if phase in imputed_phases:
            return f"{name}{IMPUTED_SUFFIX}"
        if phase in user_phases:
            return f"{name}{USER_DEFINED_SUFFIX}"
        return name

    for n, (label, group) in enumerate(solid_df.groupby("Phase")):
        # One vertical segment per composition, broken from the next by None.
        x, y, z, colors = [], [], [], []
        for _, segment in group.sort_values("T").groupby(["x0", "x1"], sort=False):
            x += [*segment["x0"], None]
            y += [*segment["x1"], None]
            z += [*segment["T"], None]
            colors += [*segment["Colors"], segment["Colors"].iloc[-1]]
        x, y, z, colors = x[:-1], y[:-1], z[:-1], colors[:-1]
        line = dict(color=colors[0] if len(set(colors)) == 1 else colors, width=10)
        if highlight:
            tags.solids[str(label)] = (f"gliquid-solid-{n}", line["color"])
        if label in imputed_phases or label in user_phases:
            line["dash"] = "dash"
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=line,
                showlegend=False,
                opacity=1,
                hovertemplate=f"<b>Phase: {display_name(label)}</b>" + "<br><extra></extra>",
                **tagged(f"gliquid-solid-{n}", "solid", str(label)),
            )
        )

    liquidus_hover = (
        "<b>Liquidus Surface</b><br>"
        + composition_hover(components)
        + "T: %{z:.1f}°C<br>"
        + "Coexistent Phases: %{customdata[3]}<br>"
        + "<extra></extra>"
    )

    if partition is not None:
        surface = partition.surface
        coexistent = np.array(
            [", ".join(phases) for phases in partition.vertex_phases], dtype=object
        )
        unlisted = [phase for phase in partition.fields if phase not in legend_phases]
    else:
        unlisted = []
        surface = liquidus_fields.SurfacePiece(
            xy=liq_df[["x0", "x1"]].to_numpy(float),
            T=liq_df["T"].to_numpy(float),
            comp=liq_df[["x0_orig", "x1_orig"]].to_numpy(float),
            parent=np.arange(len(liq_df)),
            faces=np.asarray(triangles),
        )
        coexistent = liq_df["coexistent_phases"].to_numpy(dtype=object)

    fig.add_trace(
        _field_mesh(
            surface,
            "Viridis",
            components,
            coexistent,
            liquidus_hover,
            name="liquidus",
            **tagged(tags.liquidus if highlight else None, "liquidus"),
        )
    )
    boundaries = [
        (neighbour, lines)
        for phase in unlisted
        for neighbour, lines in liquidus_fields.field_boundaries(partition, phase).items()
        if neighbour is not None
    ]
    for n, (neighbour, lines) in enumerate(boundaries):
        key = f"gliquid-boundary-{n}"
        if highlight and neighbour in legend_phases:
            tags.boundaries.setdefault(neighbour, []).append(key)
        fig.add_trace(
            _outline_trace(
                lines,
                color=GAP_OUTLINE_COLOR,
                dash=GAP_OUTLINE_DASH,
                visible=True,
                **tagged(key, "boundary", neighbour),
            )
        )

    # Add iso-temperature lines
    draw_isotherms(
        fig, compute_isotherm_contours(np.column_stack([surface.xy, surface.T]), surface.faces)
    )

    for n, phase in enumerate(legend_phases):
        if highlight:
            tags.legend[str(phase)] = f"gliquid-legend-{n}"
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(color=color_map[phase], size=10, opacity=1.0),
                name=display_name(phase),
                textfont=dict(size=8),
                showlegend=True,
                **tagged(f"gliquid-legend-{n}", "legend", str(phase)),
            )
        )

    # The drawn floor: conds[0] lifted clear of the clip plane (see FLOOR_INSET_FRAC).
    floor_z = conds[0] + FLOOR_INSET_FRAC * (conds[1] - conds[0])

    fig.add_trace(
        go.Scatter3d(
            x=_TRIANGLE_X,
            y=_TRIANGLE_Y,
            z=[floor_z] * 4,
            mode="lines",
            line=dict(color="black", width=5),
            name="axes",
            showlegend=False,
        )
    )

    ternary_labels.draw_composition_axes(fig, components, floor_z)
    x_range, y_range = _plane_ranges(fig, PLANE_PAD)

    fig.update_layout(
        legend=_LEGEND,
        autosize=True,
        # The z-axis title, a screen-space annotation.
        annotations=[
            dict(
                text="Temperature (°C)",
                textangle=-90,
                # Negative paper x places the text in the left margin, beside the scene's tick
                # labels.
                xref="paper",
                yref="paper",
                x=-0.03,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=17, color="black"),
            )
        ],
        # t leaves room for the figure title; l holds the z-axis title.
        margin=dict(l=30, r=0, b=0, t=40),
        scene=dict(
            zaxis=dict(
                # The plotted window: conds is the temperature grid in Celsius, temp_slider already
                # folded in.
                range=[conds[0], conds[1]],
                # Blank; the title is drawn as a paper annotation.
                title="",
            ),
            xaxis=dict(
                title=" ",
                showticklabels=False,
                showaxeslabels=False,
                showgrid=False,
                range=x_range,
            ),
            yaxis=dict(
                title=" ",
                showticklabels=False,
                showaxeslabels=False,
                showgrid=False,
                range=y_range,
            ),
            # x and y share one scale, so the triangle and the flat labels keep their shapes.
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=(y_range[1] - y_range[0]) / (x_range[1] - x_range[0]), z=1.0),
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            bgcolor="white",
            camera=dict(
                projection=dict(type="orthographic"),
            ),
        ),
    )
    ternary_labels.orient_labels(fig)

    if highlight:
        _add_phase_fields(
            fig,
            tags,
            surface,
            components,
            coexistent,
            partition.fields,
            legend_phases,
            liquidus_hover,
        )
        states, reset = phase_highlight.build_states(tags)
        if interactive_legend:
            phase_highlight.embed_spec(fig, tags, states, reset)
        if highlight_phase is not None:
            phase_highlight.apply_patches(fig, states[str(highlight_phase)])

    return fig


_ATTACH_ALL = "window.gliquidAttachLabelViews(gd); window.gliquidAttachPhaseHighlight(gd);"


def post_script() -> str:
    """JavaScript run on a :func:`render_tx_surface` figure once a browser has drawn it, with
    ``{plot_id}`` standing for its div id. A :class:`LiquidusFigure` adds it itself; pass it to
    anything else that draws the figure's JSON.

    It shows the view of the base triangle's labels (:mod:`gliquid.plotting.ternary_labels`)
    whose edge is lowest on screen as the camera turns, and on a figure built with
    ``interactive_legend`` it handles legend clicks (:mod:`gliquid.plotting.phase_highlight`).
    """
    return (
        ternary_labels.ATTACH_JS
        + phase_highlight.ATTACH_JS
        + "(function (gd) { "
        + _ATTACH_ALL
        + " })(document.getElementById('{plot_id}'));\n"
    )


def dash_clientside_function() -> str:
    """A Dash clientside-callback function attaching the :func:`post_script` handlers to a
    ``dcc.Graph``.

    Its inputs are the graph's ``figure`` and ``id``; it returns ``no_update``. For a
    ``dcc.Graph`` with id ``graph_id`` and a ``dcc.Store(id=<store_id>)`` in the layout, register
    it as::

        app.clientside_callback(
            dash_clientside_function(),
            Output(<store_id>, "data"),
            Input(graph_id, "figure"),
            State(graph_id, "id"),
        )
    """
    return (
        "function (figure, graphId) {\n"
        + ternary_labels.ATTACH_JS
        + phase_highlight.ATTACH_JS
        + """
  var attempts = 0;
  (function attach() {
    var root = document.getElementById(graphId);
    var gd = root && (root.classList.contains("js-plotly-plot")
      ? root : root.querySelector(".js-plotly-plot"));
    if (gd && gd.on) { __ATTACH_ALL__ return; }
    if (attempts++ < 100) { setTimeout(attach, 100); }
  })();
  return window.dash_clientside.no_update;
}
""".replace("__ATTACH_ALL__", _ATTACH_ALL)
    )


class LiquidusFigure(go.Figure):
    """A :func:`render_tx_surface` figure that runs :func:`post_script` wherever it is drawn.

    In a notebook kernel, ``show()`` and a bare ``fig`` as the last line of a cell draw it from
    :meth:`notebook_html`; elsewhere, and with an explicit ``renderer``, ``show`` hands plotly the
    script. ``to_html`` and ``write_html`` run it ahead of any ``post_script`` passed in.
    """

    def show(self, *args, **kwargs):
        if args or "renderer" in kwargs or not _in_notebook_kernel():
            return super().show(*args, **_with_post_script(kwargs))
        from IPython.display import HTML, display

        display(HTML(self.notebook_html(config=kwargs.get("config"))))

    def to_html(self, *args, **kwargs):
        return super().to_html(*args, **_with_post_script(kwargs))

    def write_html(self, *args, **kwargs):
        return super().write_html(*args, **_with_post_script(kwargs))

    def notebook_html(self, config=None) -> str:
        """The figure as :func:`gliquid.plotting.export.side_by_side_html` draws it, for
        ``IPython.display.HTML``: ``layout.height`` pixels tall, or :data:`NOTEBOOK_HEIGHT`
        without one; ``config`` is a plotly config dict."""
        height = self.layout.height or NOTEBOOK_HEIGHT
        return side_by_side_html([self], configs=[config], height=f"{height}px")

    def _ipython_display_(self):
        self.show()


def _in_notebook_kernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return hasattr(get_ipython(), "kernel")


def _with_post_script(kwargs) -> dict:
    extra = kwargs.get("post_script")
    extra = [] if extra is None else [extra] if isinstance(extra, str) else list(extra)
    return {**kwargs, "post_script": [post_script(), *extra]}


def _plane_ranges(fig, pad) -> tuple[list[float], list[float]]:
    """The x and y extents of every trace in ``fig`` and of every view of its labels, each
    widened by ``pad`` on both sides."""
    views = [
        xy
        for t in fig.data
        if isinstance(t.meta, dict)
        for xy in t.meta.get(ternary_labels.XY_KEY, {}).values()
    ]
    xs = np.concatenate(
        [np.asarray(t.x, dtype=float) for t in fig.data if t.x is not None]
        + [np.asarray(x, dtype=float) for x, _ in views]
    )
    ys = np.concatenate(
        [np.asarray(t.y, dtype=float) for t in fig.data if t.y is not None]
        + [np.asarray(y, dtype=float) for _, y in views]
    )
    return (
        [float(np.nanmin(xs)) - pad, float(np.nanmax(xs)) + pad],
        [float(np.nanmin(ys)) - pad, float(np.nanmax(ys)) + pad],
    )


def _meta(key, role, phase=None) -> dict:
    """The trace ``meta`` :mod:`gliquid.plotting.phase_highlight` reads: id, role and phase."""
    return {"gliquid_id": key, "gliquid_role": role, "gliquid_phase": phase}


def _field_mesh(
    piece, colorscale, components, coexistent, hovertemplate, all_T=None, **kwargs
) -> go.Mesh3d:
    """A piece of the liquidus as a Mesh3d, coloured by T, hovering the at. % of ``components``
    (:func:`composition_hover`) and the vertex's coexistent phases.

    ``coexistent`` is the hover text of each surface vertex, which ``piece.parent`` indexes;
    ``kwargs`` go to the trace. With ``all_T``, the coordinates and temperatures are rounded to
    :data:`FIELD_COORD_DECIMALS` and :data:`FIELD_T_DECIMALS` places and coloured on the scale
    of ``all_T``.
    """
    xy, temps = piece.xy, piece.T
    if all_T is not None:
        xy = np.round(xy, FIELD_COORD_DECIMALS)
        temps = np.round(temps, FIELD_T_DECIMALS)
        kwargs.update(cmin=float(all_T.min()), cmax=float(all_T.max()), cauto=False)
    return go.Mesh3d(
        x=xy[:, 0],
        y=xy[:, 1],
        z=temps,
        i=piece.faces[:, 0],
        j=piece.faces[:, 1],
        k=piece.faces[:, 2],
        opacity=0.6,
        colorscale=colorscale,
        intensity=temps,
        showscale=False,
        hovertemplate=hovertemplate,
        customdata=np.column_stack(
            (composition_at_pct(piece.comp, components), coexistent[piece.parent])
        ),
        **kwargs,
    )


def _outline_trace(outline, color=FIELD_OUTLINE_COLOR, dash=None, **kwargs) -> go.Scatter3d:
    """A field outline, polylines of (k, 3) x, y, T arrays, as one line trace in ``color``,
    ``dash`` styled; ``kwargs`` go to the trace."""
    x, y, z = [], [], []
    for line in outline:
        x += [*line[:, 0], None]
        y += [*line[:, 1], None]
        z += [*line[:, 2], None]
    return go.Scatter3d(
        x=x[:-1],
        y=y[:-1],
        z=z[:-1],
        mode="lines",
        line=dict(color=color, width=FIELD_OUTLINE_WIDTH, dash=dash),
        showlegend=False,
        hoverinfo="skip",
        **kwargs,
    )


def _add_phase_fields(
    fig, tags, surface, components, coexistent, fields, phases, hovertemplate
) -> None:
    """Add every legend phase's field, the dimmed rest of the surface and the field outline,
    all hidden, and record their ids in ``tags``.

    ``coexistent`` is the hover text of each ``surface`` vertex, which the ``parent`` of every
    field piece indexes.
    """
    viridis = go.Mesh3d(colorscale="Viridis").colorscale
    dimmed = phase_highlight.dimmed_colorscale(viridis)

    def mesh(piece, colorscale, key, role, phase):
        return _field_mesh(
            piece,
            colorscale,
            components,
            coexistent,
            hovertemplate,
            surface.T,
            visible=False,
            meta=_meta(key, role, phase),
        )

    for n, phase in enumerate(str(phase) for phase in phases):
        field = fields.get(phase)
        if field is None or field.inside is None:
            continue
        tags.fields[phase] = f"gliquid-field-{n}"
        fig.add_trace(mesh(field.inside, viridis, tags.fields[phase], "field", phase))
        if field.outside is not None:
            tags.rests[phase] = f"gliquid-rest-{n}"
            fig.add_trace(mesh(field.outside, dimmed, tags.rests[phase], "rest", phase))
        tags.outlines[phase] = f"gliquid-outline-{n}"
        fig.add_trace(
            _outline_trace(
                field.outline,
                visible=False,
                meta=_meta(tags.outlines[phase], "outline", phase),
            )
        )

    if len(tags.fields) < len(tags.legend):
        tags.dimmed_liquidus = "gliquid-liquidus-dimmed"
        fig.add_trace(mesh(surface, dimmed, tags.dimmed_liquidus, "dimmed_liquidus", None))
