"""The binary solidus lines of the ternary liquidus figure.

:func:`traces` draws the tie lines of the three binary sides, each on its side of the figure at
its temperature. In a browser, :data:`ATTACH_JS` draws only the parts of the lines the liquidus
surface does not cover from the camera; it is attached by
:func:`gliquid.plotting.ternary_surface.post_script` (a standalone HTML file) or
:func:`gliquid.plotting.ternary_surface.dash_clientside_function` (a Dash ``dcc.Graph``). Without
it every line is drawn whole.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gliquid.plotting.style import SOLIDUS_LINE_COLOR, SOLIDUS_LINE_WIDTH

#: Trace ``meta`` key of the solidus trace.
META_KEY = "gliquid_solidus"
#: Name of the liquidus surface trace :data:`ATTACH_JS` reads the covering surface from.
SURFACE_NAME = "liquidus"


def traces(solidus) -> list:
    """The ``solidus`` lines of :func:`gliquid.plotting.ternary_surface.render_tx_surface`.

    Returns:
        list: one line trace, each line naming its temperature on hover. Its ``meta[META_KEY]``
        holds, per line, the ``xy`` of its two ends, the temperature ``T`` and the hover
        ``text``.
    """
    x, y, z, text, lines = [], [], [], [], []
    for xy, temp in solidus:
        xy = np.asarray(xy, dtype=float)
        label = f"<b>Binary solidus</b><br>T: {temp:.1f}°C"
        lines.append({"xy": xy.tolist(), "T": float(temp), "text": label})
        x += [*xy[:, 0], None]
        y += [*xy[:, 1], None]
        z += [float(temp)] * 2 + [None]
        text += [label] * 2 + [None]
    return [
        go.Scatter3d(
            x=x[:-1],
            y=y[:-1],
            z=z[:-1],
            mode="lines",
            line=dict(color=SOLIDUS_LINE_COLOR, width=SOLIDUS_LINE_WIDTH),
            text=text[:-1],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="solidus",
            meta={META_KEY: {"lines": lines}},
        )
    ]


ATTACH_JS = """
(function () {
  if (window.gliquidAttachSolidusView) { return; }
  var KEY = "__META_KEY__", SURFACE = "__SURFACE_NAME__";
  var GRID = 256, SAMPLES = 48, BISECTIONS = 6, CLEAR = 0.004;
  function sceneOf(gd) {
    return gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
  }
  // The drawn solidus trace and the liquidus surface, each as {trace, full, object}.
  function parts(gd) {
    var scene = sceneOf(gd), found = {}, full = {};
    if (!scene || !scene.traces) { return found; }
    (gd._fullData || []).forEach(function (trace) { full[trace.index] = trace; });
    gd.data.forEach(function (trace, i) {
      var spec = trace.meta && trace.meta[KEY];
      var role = spec ? "lines" : null;
      if (trace.name === SURFACE && trace.type === "mesh3d") { role = "surface"; }
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
  // The lines as the parts the surface does not cover.
  function visibleLines(lines, grid, dir) {
    var x = [], y = [], z = [], text = [];
    function at(a, b, t) { return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]; }
    lines.forEach(function (line) {
      var a = line.xy[0], b = line.xy[1], start = null;
      function hidden(t) {
        var p = at(a, b, t);
        return covered(grid, dir, p[0], p[1], line.T);
      }
      function edge(shown, off) {
        for (var n = 0; n < BISECTIONS; n++) {
          var mid = 0.5 * (shown + off);
          if (hidden(mid)) { off = mid; } else { shown = mid; }
        }
        return shown;
      }
      for (var s = 0; s <= SAMPLES + 1; s++) {
        var off = s > SAMPLES || hidden(s / SAMPLES);
        if (!off && start === null) {
          start = s === 0 ? 0 : edge(s / SAMPLES, (s - 1) / SAMPLES);
        } else if (off && start !== null) {
          var end = s > SAMPLES ? 1 : edge((s - 1) / SAMPLES, s / SAMPLES);
          var p = at(a, b, start), q = at(a, b, end);
          x.push(p[0], q[0], null); y.push(p[1], q[1], null);
          z.push(line.T, line.T, null); text.push(line.text, line.text, null);
          start = null;
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
    var found = parts(gd), drawn = found.lines, surface = found.surface;
    if (!drawn || !drawn.object || typeof drawn.object.update !== "function") { return; }
    var dir = surface && surface.full.x && surface.full.i ? toCamera(gd) : null;
    var state = gd._gliquidSolidus = gd._gliquidSolidus || {};
    var key = dir ? [dir.x, dir.y, dir.z].join() : "all";
    if (state.spec === drawn.spec && state.key === key) { return; }
    if (dir && state.surface !== surface.full.x) {
      state.surface = surface.full.x;
      state.grid = raster(surface.full);
    }
    var lines;
    if (dir) {
      lines = visibleLines(drawn.spec.lines, state.grid, dir);
    } else {
      lines = {x: [], y: [], z: [], text: []};
      drawn.spec.lines.forEach(function (line) {
        lines.x.push(line.xy[0][0], line.xy[1][0], null);
        lines.y.push(line.xy[0][1], line.xy[1][1], null);
        lines.z.push(line.T, line.T, null); lines.text.push(line.text, line.text, null);
      });
    }
    if (!lines.x.length) { lines = {x: [null], y: [null], z: [null], text: [null]}; }
    set(drawn, lines);
    state.spec = drawn.spec;
    state.key = key;
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
