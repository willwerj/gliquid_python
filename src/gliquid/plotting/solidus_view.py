"""The solidus triangles of the ternary liquidus figure.

:func:`traces` draws each triangle of three solid phases as an outline and a fill of the three
phases' colours that is fully transparent. In a browser, :data:`ATTACH_JS` draws only the parts
of the outlines the liquidus surface does not cover from the camera, and fills the triangle
under the pointer; it is attached by :func:`gliquid.plotting.ternary_surface.post_script` (a
standalone HTML file) or :func:`gliquid.plotting.ternary_surface.dash_clientside_function` (a
Dash ``dcc.Graph``). Without it every outline is drawn whole and no triangle is filled.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb

from gliquid.plotting.style import (
    SOLIDUS_FILL_OPACITY,
    SOLIDUS_OUTLINE_COLOR,
    SOLIDUS_OUTLINE_WIDTH,
)

#: Trace ``meta`` key of the two solidus traces.
META_KEY = "gliquid_solidus"
#: Name of the liquidus surface trace :data:`ATTACH_JS` reads the covering surface from.
SURFACE_NAME = "liquidus"


def _channels(colour) -> str:
    """``colour``, a hex, named or ``rgb(...)`` colour, as ``"r,g,b"`` with 0-255 channels."""
    colour = str(colour)
    if colour.startswith("rgb"):
        parts = colour[colour.index("(") + 1 : colour.index(")")].split(",")[:3]
        channels = [float(part) for part in parts]
    else:
        channels = [255 * channel for channel in to_rgb(colour)]
    return ",".join(f"{channel:.0f}" for channel in channels)


def traces(solidus, color_map, display_name) -> list:
    """The fill and the outlines of the ``solidus`` triangles of
    :func:`gliquid.plotting.ternary_surface.render_tx_surface`.

    Returns:
        list: a mesh with one face per triangle, each corner in its phase's colour at zero
        opacity, and a line trace of the outlines; both name the three phases and the
        temperature on hover. ``meta[META_KEY]`` of the mesh holds its ``role`` (``"fill"``),
        the ``colors`` of its corners as ``"r,g,b"`` and the ``opacity`` of a shown fill; that
        of the lines holds its ``role`` (``"outline"``) and, per triangle, the ``xy`` of the
        corners, the temperature ``T`` and the hover ``text``.
    """
    corners = np.concatenate([np.asarray(xy, dtype=float) for _, xy, _ in solidus])
    temps = np.repeat([float(temp) for _, _, temp in solidus], 3)
    first = 3 * np.arange(len(solidus))
    colors = [_channels(color_map[phase]) for phases, _, _ in solidus for phase in phases]
    labels = [
        "<b>Solidus</b><br>"
        + " + ".join(display_name(phase) for phase in phases)
        + f"<br>T: {temp:.1f}°C"
        for phases, _, temp in solidus
    ]
    fill = go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=temps,
        i=first,
        j=first + 1,
        k=first + 2,
        vertexcolor=[f"rgba({color},0)" for color in colors],
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
        text=np.repeat(labels, 3),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="solidus fill",
        meta={META_KEY: {"role": "fill", "colors": colors, "opacity": SOLIDUS_FILL_OPACITY}},
    )
    x, y, z, text, triangles = [], [], [], [], []
    for (_, xy, temp), label in zip(solidus, labels):
        xy = np.asarray(xy, dtype=float)
        triangles.append({"xy": xy.tolist(), "T": float(temp), "text": label})
        closed = xy[[0, 1, 2, 0]]
        x += [*closed[:, 0], None]
        y += [*closed[:, 1], None]
        z += [float(temp)] * 4 + [None]
        text += [label] * 4 + [None]
    outline = go.Scatter3d(
        x=x[:-1],
        y=y[:-1],
        z=z[:-1],
        mode="lines",
        line=dict(color=SOLIDUS_OUTLINE_COLOR, width=SOLIDUS_OUTLINE_WIDTH),
        text=text[:-1],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="solidus",
        meta={META_KEY: {"role": "outline", "triangles": triangles}},
    )
    return [fill, outline]


ATTACH_JS = """
(function () {
  if (window.gliquidAttachSolidusView) { return; }
  var KEY = "__META_KEY__", SURFACE = "__SURFACE_NAME__";
  var GRID = 256, SAMPLES = 48, BISECTIONS = 6, CLEAR = 0.004;
  function sceneOf(gd) {
    return gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
  }
  // The drawn solidus traces and the liquidus surface, each as {trace, full, object}.
  function parts(gd) {
    var scene = sceneOf(gd), found = {}, full = {};
    if (!scene || !scene.traces) { return found; }
    (gd._fullData || []).forEach(function (trace) { full[trace.index] = trace; });
    gd.data.forEach(function (trace, i) {
      var spec = trace.meta && trace.meta[KEY];
      var role = spec ? spec.role : trace.name === SURFACE && trace.type === "mesh3d" ? "surface" : null;
      if (!role || !full[i]) { return; }
      found[role] = {trace: trace, full: full[i], object: scene.traces[full[i].uid], spec: spec};
    });
    return found;
  }
  // The heights of a surface mesh at the centres of a GRID x GRID raster of its bounding box in
  // the plane; NaN where no face covers a centre.
  function raster(full) {
    var x = full.x, y = full.y, z = full.z, fi = full.i, fj = full.j, fk = full.k;
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (var n = 0; n < x.length; n++) {
      if (x[n] < x0) { x0 = x[n]; } if (x[n] > x1) { x1 = x[n]; }
      if (y[n] < y0) { y0 = y[n]; } if (y[n] > y1) { y1 = y[n]; }
      if (z[n] < z0) { z0 = z[n]; } if (z[n] > z1) { z1 = z[n]; }
    }
    var dx = (x1 - x0) / GRID, dy = (y1 - y0) / GRID;
    var heights = new Float32Array(GRID * GRID).fill(NaN);
    for (var f = 0; f < fi.length; f++) {
      var a = fi[f], b = fj[f], c = fk[f];
      var det = (y[b] - y[c]) * (x[a] - x[c]) + (x[c] - x[b]) * (y[a] - y[c]);
      if (!(Math.abs(det) > 0)) { continue; }
      var c0 = Math.max(0, Math.floor((Math.min(x[a], x[b], x[c]) - x0) / dx - 0.5));
      var c1 = Math.min(GRID - 1, Math.ceil((Math.max(x[a], x[b], x[c]) - x0) / dx - 0.5));
      var r0 = Math.max(0, Math.floor((Math.min(y[a], y[b], y[c]) - y0) / dy - 0.5));
      var r1 = Math.min(GRID - 1, Math.ceil((Math.max(y[a], y[b], y[c]) - y0) / dy - 0.5));
      for (var r = r0; r <= r1; r++) {
        var py = y0 + (r + 0.5) * dy;
        for (var col = c0; col <= c1; col++) {
          var px = x0 + (col + 0.5) * dx;
          var u = ((y[b] - y[c]) * (px - x[c]) + (x[c] - x[b]) * (py - y[c])) / det;
          var v = ((y[c] - y[a]) * (px - x[c]) + (x[a] - x[c]) * (py - y[c])) / det;
          if (u < -1e-9 || v < -1e-9 || u + v > 1 + 1e-9) { continue; }
          heights[r * GRID + col] = u * z[a] + v * z[b] + (1 - u - v) * z[c];
        }
      }
    }
    return {x0: x0, y0: y0, dx: dx, dy: dy, heights: heights, span: z1 - z0};
  }
  function height(grid, x, y) {
    var col = Math.floor((x - grid.x0) / grid.dx), row = Math.floor((y - grid.y0) / grid.dy);
    if (col < 0 || row < 0 || col >= GRID || row >= GRID) { return NaN; }
    return grid.heights[row * GRID + col];
  }
  // The direction from the scene toward an orthographic camera, in data units; null otherwise.
  function toCamera(gd) {
    var scene = sceneOf(gd), layout = gd._fullLayout.scene;
    if (!scene || !scene.getCamera || !scene.glplot) { return null; }
    var camera = scene.getCamera();
    if (!camera.projection || camera.projection.type !== "orthographic") { return null; }
    var center = camera.center || {x: 0, y: 0, z: 0}, aspect = scene.glplot.getAspectratio();
    var out = {};
    ["x", "y", "z"].forEach(function (axis) {
      var range = layout[axis + "axis"].range;
      out[axis] = (camera.eye[axis] - center[axis]) * (range[1] - range[0]) / aspect[axis];
    });
    return out;
  }
  // Whether the surface lies between the point and the camera.
  function covered(grid, dir, x, y, z) {
    var along = Math.hypot(dir.x / grid.dx, dir.y / grid.dy);
    if (!(along > 1e-9)) { return dir.z > 0; }
    var step = 0.5 / along, clear = CLEAR * grid.span, inside = false;
    for (var n = 1; n <= 4 * GRID; n++) {
      var h = height(grid, x + n * step * dir.x, y + n * step * dir.y);
      if (isNaN(h)) {
        if (inside || n > 4) { return false; }
        continue;
      }
      inside = true;
      if (z + n * step * dir.z > h + clear) { return true; }
    }
    return false;
  }
  // The outlines as lines of the parts the surface does not cover.
  function visibleLines(triangles, grid, dir) {
    var x = [], y = [], z = [], text = [];
    function at(a, b, t) { return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]; }
    triangles.forEach(function (triangle) {
      function hidden(a, b, t) {
        var p = at(a, b, t);
        return covered(grid, dir, p[0], p[1], triangle.T);
      }
      function edge(a, b, shown, off) {
        for (var n = 0; n < BISECTIONS; n++) {
          var mid = 0.5 * (shown + off);
          if (hidden(a, b, mid)) { off = mid; } else { shown = mid; }
        }
        return shown;
      }
      for (var e = 0; e < 3; e++) {
        var a = triangle.xy[e], b = triangle.xy[(e + 1) % 3], start = null;
        for (var s = 0; s <= SAMPLES + 1; s++) {
          var off = s > SAMPLES || hidden(a, b, s / SAMPLES);
          if (!off && start === null) {
            start = s === 0 ? 0 : edge(a, b, s / SAMPLES, (s - 1) / SAMPLES);
          } else if (off && start !== null) {
            var end = s > SAMPLES ? 1 : edge(a, b, (s - 1) / SAMPLES, s / SAMPLES);
            var p = at(a, b, start), q = at(a, b, end);
            x.push(p[0], q[0], null); y.push(p[1], q[1], null);
            z.push(triangle.T, triangle.T, null); text.push(triangle.text, triangle.text, null);
            start = null;
          }
        }
      }
    });
    return {x: x, y: y, z: z, text: text};
  }
  function set(part, values) {
    Object.keys(values).forEach(function (key) {
      part.trace[key] = part.full[key] = values[key];
    });
    part.object.update(part.full);
  }
  function cull(gd) {
    var found = parts(gd), outline = found.outline, surface = found.surface;
    if (!outline || !outline.object || typeof outline.object.update !== "function") { return; }
    var dir = surface && surface.full.x && surface.full.i ? toCamera(gd) : null;
    var state = gd._gliquidSolidus = gd._gliquidSolidus || {};
    var key = dir ? [dir.x, dir.y, dir.z].join() : "all";
    if (state.spec === outline.spec && state.key === key) { return; }
    if (dir && state.surface !== surface.full.x) {
      state.surface = surface.full.x;
      state.grid = raster(surface.full);
    }
    var lines;
    if (dir) {
      lines = visibleLines(outline.spec.triangles, state.grid, dir);
    } else {
      lines = {x: [], y: [], z: [], text: []};
      outline.spec.triangles.forEach(function (triangle) {
        [0, 1, 2, 0].forEach(function (corner) {
          lines.x.push(triangle.xy[corner][0]); lines.y.push(triangle.xy[corner][1]);
          lines.z.push(triangle.T); lines.text.push(triangle.text);
        });
        lines.x.push(null); lines.y.push(null); lines.z.push(null); lines.text.push(null);
      });
    }
    if (!lines.x.length) { lines = {x: [null], y: [null], z: [null], text: [null]}; }
    set(outline, lines);
    state.spec = outline.spec;
    state.key = key;
  }
  // The triangle a hovered point of a solidus trace lies on; -1 for any other point.
  function hovered(gd, event) {
    var point = event && event.points && event.points[0];
    var trace = point && gd.data[point.curveNumber];
    if (!trace || !trace.meta || !trace.meta[KEY]) { return -1; }
    var outline = parts(gd).outline, best = -1, least = Infinity;
    if (!outline) { return -1; }
    outline.spec.triangles.forEach(function (triangle, n) {
      var a = triangle.xy[0], b = triangle.xy[1], c = triangle.xy[2];
      var det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
      var u = ((b[1] - c[1]) * (point.x - c[0]) + (c[0] - b[0]) * (point.y - c[1])) / det;
      var v = ((c[1] - a[1]) * (point.x - c[0]) + (a[0] - c[0]) * (point.y - c[1])) / det;
      var off = Math.abs(triangle.T - point.z);
      if (u > -1e-6 && v > -1e-6 && u + v < 1 + 1e-6 && off < least) { best = n; least = off; }
    });
    return best;
  }
  function fill(gd, shown) {
    var part = parts(gd).fill, state = gd._gliquidSolidus = gd._gliquidSolidus || {};
    if (!part || !part.object || typeof part.object.update !== "function") { return; }
    if (state.fill === part.spec && state.shown === shown) { return; }
    set(part, {vertexcolor: part.spec.colors.map(function (color, n) {
      return "rgba(" + color + "," + (Math.floor(n / 3) === shown ? part.spec.opacity : 0) + ")";
    })});
    state.fill = part.spec;
    state.shown = shown;
  }
  window.gliquidAttachSolidusView = function (gd) {
    if (!gd || !gd.on) { return; }
    var handler = gd._gliquidSolidusView;
    var listening = Boolean(handler) && (!gd._ev || typeof gd._ev.listeners !== "function" ||
      gd._ev.listeners("plotly_relayout").indexOf(handler) !== -1);
    if (!listening) {
      handler = gd._gliquidSolidusView = function () { cull(gd); };
      gd.on("plotly_relayouting", handler);
      gd.on("plotly_relayout", handler);
      gd.on("plotly_afterplot", handler);
      gd.on("plotly_hover", function (event) { fill(gd, hovered(gd, event)); });
      gd.on("plotly_unhover", function () { fill(gd, -1); });
    }
    if (!gd._gliquidSolidusGestures) {
      gd._gliquidSolidusGestures = true;
      ["touchmove", "touchend", "wheel"].forEach(function (type) {
        gd.addEventListener(type, function () { cull(gd); }, {passive: true});
      });
    }
    cull(gd);
  };
})();
""".replace("__META_KEY__", META_KEY).replace("__SURFACE_NAME__", SURFACE_NAME)
