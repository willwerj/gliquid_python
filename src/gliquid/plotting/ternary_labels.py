"""The composition ticks and flat labels of the ternary liquidus figure's base triangle.

Ticks mark every tenth of each edge's element. Their numbers, an ``at. % <element>`` title per
edge and each corner's element name are set flat in the triangle's plane as filled glyph meshes
(:func:`glyph_mesh`) cut from matplotlib's font outlines. The labels come in one view per edge
(:data:`VIEWS`): in a view, every label reads upright from that edge's side of the triangle.
Each kind of label is one mesh that carries every view's vertices and draws the view whose edge
is lowest on screen (:func:`label_view`, :func:`orient_labels`); in a browser, :data:`ATTACH_JS`
redraws it in the view of the turning camera.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import plotly.graph_objects as go

# Composition ticks: spacing (mole fraction) and length, in plot units.
TICK_STEP = 0.1
TICK_LENGTH = 0.025

# Label font sizes and clearances in plot units: tick end to number, numbers to edge title, corner
# to name, and corner line to name.
TICK_LABEL_SIZE = 0.026
TICK_TITLE_SIZE = 0.031
CORNER_LABEL_SIZE = 0.04
TICK_LABEL_GAP = 0.008
TICK_TITLE_GAP = 0.012
CORNER_LABEL_GAP = 0.03
CORNER_LINE_GAP = 0.025

TICK_COLOR = "#3d3c39"
CORNER_LABEL_COLOR = "black"
LABEL_FONT = "DejaVu Sans"

# Per base-triangle edge, counterclockwise from the lower left: the ``components`` index of its
# element, its start and end corners, the tick direction into the triangle and the outward normal.
_H = np.sqrt(3) / 2
_EDGES = {
    "bottom": (1, (0.0, 0.0), (1.0, 0.0), (0.5, _H), (0.0, -1.0)),
    "right": (2, (1.0, 0.0), (0.5, _H), (-1.0, 0.0), (_H, 0.5)),
    "left": (0, (0.5, _H), (0.0, 0.0), (0.5, -_H), (-_H, 0.5)),
}
# The corner of each element of ``components``, in order, and the triangle's centroid.
_CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.5, _H))
_CENTROID = (0.5, _H / 3)

VIEWS = tuple(_EDGES)

# Trace ``meta`` keys of a label mesh's view and of every view's vertices, and the
# ``layout.meta`` key of the spec :data:`ATTACH_JS` reads.
VIEW_KEY = "gliquid_label_view"
XY_KEY = "gliquid_label_xy"
SPEC_KEY = "gliquid_label_views"
SPEC_VERSION = 2

# plotly's camera where a figure leaves it unset, in its normalised scene coordinates.
_DEFAULT_CAMERA = {"eye": (1.25, 1.25, 1.25), "center": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)}
# The scene dragmodes plotly takes from ``layout.dragmode`` in a 3-D-only figure, and those
# trace types.
_SCENE_DRAGMODES = ("orbit", "turntable", "zoom", "pan", False)
_TRACES_3D = ("scatter3d", "mesh3d", "surface", "cone", "streamtube", "volume", "isosurface")

_LABEL_TRACES = (
    ("tick", "composition tick labels", TICK_COLOR),
    ("title", "composition axis titles", TICK_COLOR),
    ("corner", "corner labels", CORNER_LABEL_COLOR),
)


def composition_ticks(components) -> list[dict]:
    """The composition ticks of the base triangle, whose corners are ``components[0]`` (lower
    left), ``components[1]`` (lower right) and ``components[2]`` (apex).

    Each edge carries the mole fraction of the element it runs towards counterclockwise: the
    bottom edge ``components[1]``, the right edge ``components[2]``, the left edge
    ``components[0]``. A tick lies on the line of constant composition through its point.

    Returns:
        list[dict]: one per tick, with ``edge`` ('bottom', 'right' or 'left'), ``element``,
        ``value`` (mole fraction), ``point`` (plot x, y on the edge), ``direction`` (unit vector
        along the constant-composition line, into the triangle) and ``normal`` (the edge's
        outward unit normal).
    """
    ticks = []
    for k in range(1, int(round(1.0 / TICK_STEP))):
        v = round(k * TICK_STEP, 6)
        for edge, (index, start, end, direction, normal) in _EDGES.items():
            ticks.append(
                {
                    "edge": edge,
                    "element": components[index],
                    "value": v,
                    "point": tuple(float(s + v * (e - s)) for s, e in zip(start, end)),
                    "direction": direction,
                    "normal": normal,
                }
            )
    return ticks


def _camera_dict(camera) -> dict:
    """``camera`` (a plotly scene camera, a dict or None) as a plain dict."""
    if hasattr(camera, "to_plotly_json"):
        camera = camera.to_plotly_json()
    return dict(camera or {})


def label_view(camera=None) -> str:
    """The view whose edge of the base triangle is lowest on screen under ``camera``: the edge
    whose outward normal points most nearly down the screen. With ``up`` along z and the camera
    looking down on the floor, this is the edge whose midpoint is nearest the camera.

    Args:
        camera: a plotly ``layout.scene.camera``, as a dict or a plotly object; any part left
            unset takes plotly's default (``eye`` (1.25, 1.25, 1.25), ``center`` the origin,
            ``up`` along z). ``up`` is read as given; :func:`orient_labels` holds it on z where
            plotly turns the scene as a turntable.

    Returns:
        str: one of :data:`VIEWS`; 'bottom' when the camera looks along the floor.
    """
    camera = _camera_dict(camera)
    eye, center, up = (
        np.array(
            [
                d if (camera.get(name) or {}).get(a) is None else camera[name][a]
                for a, d in zip("xyz", default)
            ],
            dtype=float,
        )
        for name, default in _DEFAULT_CAMERA.items()
    )
    forward = center - eye
    if not forward @ forward > 0:
        return VIEWS[0]
    down = (up @ forward) / (forward @ forward) * forward - up
    scores = {view: float(down[:2] @ edge[4]) for view, edge in _EDGES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 1e-12 else VIEWS[0]


@cache
def glyph_mesh(text: str, weight: str = "normal") -> tuple[np.ndarray, np.ndarray]:
    """``text`` set in ``LABEL_FONT`` at font size 1, as filled triangles.

    The text is centred on the middle of its ink width and of the font's capital height.

    Args:
        text: the text to set.
        weight: a matplotlib font weight, e.g. 'normal' or 'bold'.

    Returns:
        tuple: ``(vertices, faces)`` -- read-only (n, 2) plot coordinates and (m, 3) vertex
        indices.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    from scipy.spatial import Delaunay

    from gliquid.plotting.liquidus_fields import _face_edges

    font = FontProperties(family=LABEL_FONT, weight=weight)
    rings = [
        np.asarray(ring, dtype=float)[:-1]
        for ring in TextPath((0.0, 0.0), text, size=1.0, prop=font).to_polygons()
    ]
    rings = [ring for ring in rings if len(ring) >= 3]
    points = [tuple(p) for ring in rings for p in ring]
    segments, n = [], 0
    for ring in rings:
        segments += [(n + k, n + (k + 1) % len(ring)) for k in range(len(ring))]
        n += len(ring)
    # Split outline segments until each triangle lies wholly inside or outside the glyphs.
    for _ in range(50):
        simplices = Delaunay(np.asarray(points)).simplices
        edges = {frozenset(e) for e in _face_edges(simplices).tolist()}
        missing = [s for s in segments if frozenset(s) not in edges]
        if not missing:
            break
        segments = [s for s in segments if frozenset(s) in edges]
        for a, b in missing:
            points.append(tuple((np.asarray(points[a]) + np.asarray(points[b])) / 2))
            segments += [(a, len(points) - 1), (len(points) - 1, b)]
    vertices = np.asarray(points)
    centroids = vertices[simplices].mean(axis=1)
    crossings = sum(Path(ring).contains_points(centroids).astype(int) for ring in rings)
    faces = simplices[crossings % 2 == 1]
    cap_height = TextPath((0.0, 0.0), "M", size=1.0, prop=font).get_extents().ymax
    vertices = vertices - [(vertices[:, 0].min() + vertices[:, 0].max()) / 2, cap_height / 2]
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


def label_outline(label) -> np.ndarray:
    """The glyph vertices of ``label`` (an entry of :func:`composition_labels`) at its size and
    angle, relative to its ``center``, as an (n, 2) array in plot units."""
    vertices, _ = glyph_mesh(label["text"], label["weight"])
    c, s = np.cos(label["angle"]), np.sin(label["angle"])
    return label["size"] * vertices @ np.array([[c, s], [-s, c]])


def _set_off(label, anchor, outward, clearance) -> dict:
    """``label`` centred on the line from ``anchor`` along the unit vector ``outward``, with its
    nearest ink ``clearance`` beyond ``anchor``."""
    outward = np.asarray(outward, dtype=float)
    reach = float(np.max(label_outline(label) @ -outward))
    center = np.asarray(anchor, dtype=float) + (clearance + reach) * outward
    return {**label, "center": (float(center[0]), float(center[1]))}


def _turned(vector, degrees) -> np.ndarray:
    """``vector`` turned counterclockwise by ``degrees``."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _beside_line(label, corner, along, behind) -> dict:
    """``label`` set off ``corner`` along the unit vector ``along``, more than 60 degrees from the
    unit vector ``behind``: its nearest ink at least ``CORNER_LABEL_GAP`` beyond ``corner``, and
    all of it ``CORNER_LINE_GAP`` clear of every ray from ``corner`` within 60 degrees of
    ``behind``."""
    along, behind = np.asarray(along, dtype=float), np.asarray(behind, dtype=float)
    side = np.sign(behind[0] * along[1] - behind[1] * along[0])
    # Unit normal, on ``along``'s side, of the ray 60 degrees from ``behind`` towards ``along``.
    clear = _turned(_turned(behind, side * 60.0), side * 90.0)
    outline = label_outline(label)
    distance = max(
        CORNER_LABEL_GAP + float(np.max(outline @ -along)),
        (CORNER_LINE_GAP - float(np.min(outline @ clear))) / float(along @ clear),
    )
    center = np.asarray(corner, dtype=float) + distance * along
    return {**label, "center": (float(center[0]), float(center[1]))}


def composition_labels(components, view: str = "bottom") -> list[dict]:
    """The labels of the base triangle in ``view``, laid flat in its plane.

    A number beside each tick of :func:`composition_ticks` (its value in at.%), an
    ``at. % <element>`` title beside the middle of each edge, beyond its numbers, and each
    element's name, in bold, beyond its corner. Every label shares one angle: its baseline runs
    along the ``view`` edge in the direction that edge's value rises, so it reads upright from
    that edge's side. Numbers and titles are set off their edge along its outward normal,
    with their nearest ink ``TICK_LABEL_GAP`` beyond the outer end of the tick or
    ``TICK_TITLE_GAP`` beyond the edge's numbers. The names of the ``view`` edge's corners are
    set off their corner directly away from the centroid, and the far corner's name beside it
    along the baseline; each has its nearest ink at least ``CORNER_LABEL_GAP`` from its corner
    and lies ``CORNER_LINE_GAP`` clear of where the vertical line above its corner falls on
    screen under any orthographic camera that shows ``view`` (:func:`label_view`) with ``up``
    along z.

    Args:
        components: the three elements, as in :func:`composition_ticks`.
        view: one of :data:`VIEWS`.

    Returns:
        list[dict]: one per label, with ``kind`` ('tick', 'title' or 'corner'), ``edge`` (the
        edge of a number or title, None for a name), ``element``, ``text``, ``weight``,
        ``center`` (plot x, y of the middle of the text), ``angle`` (counterclockwise rotation of
        the baseline from the x axis, radians) and ``size`` (font size, plot units).

    Raises:
        ValueError: ``view`` is not one of :data:`VIEWS`.
    """
    if view not in _EDGES:
        raise ValueError(f"view {view!r} is not one of {VIEWS}.")
    _, view_start, view_end, _, view_normal = _EDGES[view]
    baseline = np.subtract(view_end, view_start)
    angle = float(np.arctan2(baseline[1], baseline[0]))

    def label(kind, edge, element, text, size, weight="normal"):
        return {
            "kind": kind,
            "edge": edge,
            "element": element,
            "text": text,
            "weight": weight,
            "center": (0.0, 0.0),
            "angle": angle,
            "size": size,
        }

    labels, reach = [], dict.fromkeys(_EDGES, 0.0)
    for tick in composition_ticks(components):
        point, normal = np.asarray(tick["point"]), np.asarray(tick["normal"])
        tick_end = -TICK_LENGTH * float(np.dot(tick["direction"], normal))
        number = _set_off(
            label(
                "tick",
                tick["edge"],
                tick["element"],
                f"{round(100 * tick['value']):d}",
                TICK_LABEL_SIZE,
            ),
            point,
            normal,
            tick_end + TICK_LABEL_GAP,
        )
        far = float(np.max((number["center"] - point + label_outline(number)) @ normal))
        reach[tick["edge"]] = max(reach[tick["edge"]], far)
        labels.append(number)
    for edge, (index, start, end, _, normal) in _EDGES.items():
        midpoint = (np.asarray(start) + np.asarray(end)) / 2
        title = label(
            "title", edge, components[index], f"at. % {components[index]}", TICK_TITLE_SIZE
        )
        labels.append(_set_off(title, midpoint, normal, reach[edge] + TICK_TITLE_GAP))
    behind = -np.asarray(view_normal)
    for element, corner in zip(components, _CORNERS):
        name = label("corner", None, element, element, CORNER_LABEL_SIZE, weight="bold")
        along = baseline
        if corner in (view_start, view_end):
            along = (np.asarray(corner) - _CENTROID) / np.linalg.norm(
                np.subtract(corner, _CENTROID)
            )
        labels.append(_beside_line(name, corner, along, behind))
    return labels


def _glyph_triangles(labels) -> tuple[np.ndarray, np.ndarray]:
    """``labels`` (entries of :func:`composition_labels`) as one set of filled triangles: the
    (n, 2) plot x, y of their vertices, rounded to 5 places, and (m, 3) vertex indices."""
    xy, faces, n = [], [], 0
    for label in labels:
        _, triangles = glyph_mesh(label["text"], label["weight"])
        xy.append(label_outline(label) + label["center"])
        faces.append(triangles + n)
        n += len(xy[-1])
    return np.round(np.concatenate(xy), 5), np.concatenate(faces)


def draw_composition_axes(fig, components, z) -> None:
    """Add the composition ticks and the labels to ``fig``, flat at height ``z``, and store the
    spec :data:`ATTACH_JS` reads in ``layout.meta[SPEC_KEY]``.

    The ticks are one Scatter3d, 'composition ticks', outside the base triangle. The labels are
    three Mesh3d traces -- 'composition tick labels', 'composition axis titles' and 'corner
    labels' -- whose triangles are the same in every view. Each carries every view's vertices,
    ``[x, y]`` by view, in ``meta[XY_KEY]``, and draws the view named in ``meta[VIEW_KEY]``: the
    first of :data:`VIEWS` until :func:`orient_labels` sets another.
    """
    marks_x, marks_y, marks_z = [], [], []
    for tick in composition_ticks(components):
        (px, py), (dx, dy) = tick["point"], tick["direction"]
        marks_x += [px, px - TICK_LENGTH * dx, None]
        marks_y += [py, py - TICK_LENGTH * dy, None]
        marks_z += [z, z, None]
    fig.add_trace(
        go.Scatter3d(
            x=marks_x,
            y=marks_y,
            z=marks_z,
            mode="lines",
            line=dict(color=TICK_COLOR, width=3),
            name="composition ticks",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    views = {view: composition_labels(components, view) for view in VIEWS}
    for kind, name, color in _LABEL_TRACES:
        xy = {}
        for view, labels in views.items():
            vertices, faces = _glyph_triangles([lab for lab in labels if lab["kind"] == kind])
            xy[view] = [vertices[:, 0].tolist(), vertices[:, 1].tolist()]
        x, y = xy[VIEWS[0]]
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=np.full(len(x), z),
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
                name=name,
                showlegend=False,
                hoverinfo="skip",
                meta={VIEW_KEY: VIEWS[0], XY_KEY: xy},
            )
        )
    meta = fig.layout.meta
    spec = {
        "version": SPEC_VERSION,
        "normals": {view: [float(c) for c in edge[4]] for view, edge in _EDGES.items()},
    }
    fig.update_layout(meta={**(meta if isinstance(meta, dict) else {}), SPEC_KEY: spec})


def _turntable(fig, camera) -> bool:
    """Whether plotly turns ``fig``'s scene as a turntable under ``camera``, holding its ``up``
    on z: the scene's dragmode (set or from the template), else, in a figure of 3-D traces only,
    a scene dragmode set on the layout (or its template), else plotly's default for
    ``camera``'s ``up``."""
    template = fig.layout.template.layout
    dragmode = fig.layout.scene.dragmode
    if dragmode is None:
        dragmode = template.scene.dragmode
    layout_dragmode = fig.layout.dragmode
    if layout_dragmode is None:
        layout_dragmode = template.dragmode
    only_3d = all(trace.type in _TRACES_3D for trace in fig.data)
    if dragmode is None and only_3d and layout_dragmode in _SCENE_DRAGMODES:
        dragmode = layout_dragmode
    if dragmode is not None:
        return dragmode == "turntable"
    up = camera.get("up")
    if not up:
        return True
    x, y, z = (up.get(a) for a in "xyz")
    if z == 0:
        return False
    if x and y and z:
        return z / np.sqrt(x * x + y * y + z * z) > 0.999
    return True


def orient_labels(fig, camera=None) -> str:
    """Draw ``fig``'s labels in its view under ``camera`` (:func:`label_view`). Where plotly
    turns the scene as a turntable, the view is that of ``camera`` with its ``up`` along z, as
    plotly draws it.

    Args:
        fig: a figure drawn by :func:`draw_composition_axes`.
        camera: a plotly scene camera; defaults to ``fig.layout.scene.camera``.

    Returns:
        str: the view drawn.
    """
    camera = _camera_dict(fig.layout.scene.camera if camera is None else camera)
    if _turntable(fig, camera):
        camera["up"] = {"x": 0.0, "y": 0.0, "z": 1.0}
    view = label_view(camera)
    for trace in fig.data:
        meta = trace.meta
        if isinstance(meta, dict) and XY_KEY in meta:
            trace.x, trace.y = meta[XY_KEY][view]
            trace.meta = {**meta, VIEW_KEY: view}
    return view


ATTACH_JS = (
    """
(function () {
  if (window.gliquidAttachLabelViews) { return; }
  var VERSION = __SPEC_VERSION__, KEY = "__SPEC_KEY__", TAG = "__VIEW_KEY__", XY = "__XY_KEY__";
  function viewOf(spec, camera) {
    var eye = camera.eye, center = camera.center || {x: 0, y: 0, z: 0};
    var up = camera.up || {x: 0, y: 0, z: 1};
    var f = [center.x - eye.x, center.y - eye.y, center.z - eye.z];
    var ff = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
    if (!(ff > 0)) { return null; }
    var k = (up.x * f[0] + up.y * f[1] + up.z * f[2]) / ff;
    var down = [k * f[0] - up.x, k * f[1] - up.y];
    var best = null, score = 1e-12;
    Object.keys(spec.normals).forEach(function (view) {
      var n = spec.normals[view], s = n[0] * down[0] + n[1] * down[1];
      if (s > score) { best = view; score = s; }
    });
    return best;
  }
  // Moves every label mesh drawn in another view to the live camera's view, or none of them when
  // plotly does not draw one of them.
  function show(gd) {
    var meta = gd.layout && gd.layout.meta, spec = meta && meta[KEY];
    var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    if (!spec || spec.version !== VERSION || !scene || !scene.getCamera || !scene.traces) {
      return;
    }
    var view = viewOf(spec, scene.getCamera());
    if (view === null) { return; }
    var full = {}, moves = [];
    (gd._fullData || []).forEach(function (trace) { full[trace.index] = trace; });
    for (var i = 0; i < gd.data.length; i++) {
      var trace = gd.data[i], tag = trace.meta && trace.meta[TAG];
      if (typeof tag !== "string" || tag === view) { continue; }
      var xy = trace.meta[XY] && trace.meta[XY][view];
      var mesh = full[i] && full[i].visible === true && scene.traces[full[i].uid];
      if (!xy || !mesh || typeof mesh.update !== "function") { return; }
      moves.push({trace: trace, full: full[i], mesh: mesh, xy: xy});
    }
    // Not Plotly.restyle: a redraw replots every trace and snaps a dragged camera back.
    moves.forEach(function (move) {
      move.trace.x = move.full.x = move.xy[0];
      move.trace.y = move.full.y = move.xy[1];
      move.trace.meta[TAG] = view;
      move.mesh.update(move.full);
    });
  }
  window.gliquidAttachLabelViews = function (gd) {
    if (!gd || !gd.on) { return; }
    var handler = gd._gliquidLabelViews;
    // plotly.newPlot drops the listeners of a div it plots into again.
    var listening = Boolean(handler) && (!gd._ev || typeof gd._ev.listeners !== "function" ||
      gd._ev.listeners("plotly_relayout").indexOf(handler) !== -1);
    if (!listening) {
      handler = gd._gliquidLabelViews = function () { show(gd); };
      gd.on("plotly_relayouting", handler);
      gd.on("plotly_relayout", handler);
      gd.on("plotly_afterplot", handler);
    }
    if (!gd._gliquidLabelGestures) {
      gd._gliquidLabelGestures = true;
      ["touchmove", "touchend", "wheel"].forEach(function (type) {
        gd.addEventListener(type, function () { show(gd); }, {passive: true});
      });
    }
    show(gd);
  };
})();
""".replace("__SPEC_VERSION__", str(SPEC_VERSION))
    .replace("__SPEC_KEY__", SPEC_KEY)
    .replace("__VIEW_KEY__", VIEW_KEY)
    .replace("__XY_KEY__", XY_KEY)
)
