"""Views of the ternary liquidus figure that highlight one solid phase.

A highlighted view shows one phase's liquidus field in full colour and dims everything else.
Each view is a set of per-trace patches keyed by the ``gliquid_id`` in each trace's ``meta``,
built by :func:`build_states`
and applied either in Python (:func:`apply_patches`, behind ``get_plot('tx',
highlight_phase=...)``) or in the browser from the spec :func:`embed_spec` stores in
``layout.meta`` (``get_plot('tx', interactive_legend=True)``). In the browser, a legend click
on a phase shows its view, a second click on it restores the full liquidus, and a click on
another phase switches to that one; none of them moves the camera or the zoom. The click
handling is the plain JavaScript in :data:`ATTACH_JS`, attached to a figure by
:func:`gliquid.plotting.ternary_surface.post_script` (a standalone HTML file) or
:func:`gliquid.plotting.ternary_surface.dash_clientside_function` (a Dash ``dcc.Graph``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from gliquid.plotting.style import DIMMED_PHASE_COLOR, DIMMED_SURFACE_DESATURATION

SPEC_KEY = "gliquid_phase_highlight"
SPEC_VERSION = 1


@dataclass
class HighlightTraces:
    """The id (``meta["gliquid_id"]``) of every trace a highlighted view changes.

    Attributes:
        liquidus: the full liquidus surface.
        dimmed_liquidus: the full surface in dimmed colours, for a phase with no field.
        legend: phase -> its legend marker.
        solids: phase -> (its solid-phase line, the line colour as drawn).
        fields: phase -> its field.
        rests: phase -> the rest of the surface, in dimmed colours.
        outlines: phase -> the outline of its field.
        boundaries: phase -> the boundary lines it shares with a field that has no legend entry,
            hidden while its own outline is shown.
    """

    liquidus: str
    dimmed_liquidus: str | None = None
    legend: dict[str, str] = field(default_factory=dict)
    solids: dict[str, tuple[str, object]] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    rests: dict[str, str] = field(default_factory=dict)
    outlines: dict[str, str] = field(default_factory=dict)
    boundaries: dict[str, list[str]] = field(default_factory=dict)


def dimmed_colorscale(colorscale, share=DIMMED_SURFACE_DESATURATION) -> list[list]:
    """``colorscale`` with every stop blended ``share`` of the way toward its own grey.

    Args:
        colorscale: a validated plotly colorscale, a sequence of (position, colour) pairs with
            hex or ``rgb(...)`` colours.
        share: 0 keeps the colours, 1 turns them fully grey.
    """
    dimmed = []
    for position, colour in colorscale:
        if colour.startswith("#"):
            r, g, b = hex_to_rgb(colour)
        else:
            r, g, b = (float(c) for c in colour[colour.index("(") + 1 : -1].split(",")[:3])
        grey = 0.299 * r + 0.587 * g + 0.114 * b
        mixed = [round(c + share * (grey - c)) for c in (r, g, b)]
        dimmed.append([position, f"rgb({mixed[0]}, {mixed[1]}, {mixed[2]})"])
    return dimmed


def build_states(traces: HighlightTraces) -> tuple[dict[str, dict], dict]:
    """The highlighted view of every legend phase, and the full view.

    Returns:
        tuple: (phase -> patches, reset patches), where patches map a trace id to the
        properties to set on it.
    """
    states = {}
    for phase in traces.legend:
        has_field = phase in traces.fields
        patches = {traces.liquidus: {"visible": False}}
        if traces.dimmed_liquidus is not None:
            patches[traces.dimmed_liquidus] = {"visible": not has_field}
        for group in (traces.fields, traces.rests, traces.outlines):
            for other, trace_id in group.items():
                patches[trace_id] = {"visible": other == phase}
        for other, trace_ids in traces.boundaries.items():
            for trace_id in trace_ids:
                patches[trace_id] = {"visible": other != phase}
        for other, trace_id in traces.legend.items():
            patches[trace_id] = {"visible": True if other == phase else "legendonly"}
        for other, (trace_id, colour) in traces.solids.items():
            patches[trace_id] = {
                "line": {"color": colour if other == phase else DIMMED_PHASE_COLOR}
            }
        states[phase] = patches

    reset = {traces.liquidus: {"visible": True}}
    if traces.dimmed_liquidus is not None:
        reset[traces.dimmed_liquidus] = {"visible": False}
    for group in (traces.fields, traces.rests, traces.outlines):
        for trace_id in group.values():
            reset[trace_id] = {"visible": False}
    for trace_ids in traces.boundaries.values():
        for trace_id in trace_ids:
            reset[trace_id] = {"visible": True}
    for trace_id in traces.legend.values():
        reset[trace_id] = {"visible": True}
    for trace_id, colour in traces.solids.values():
        reset[trace_id] = {"line": {"color": colour}}
    return states, reset


def trace_index(fig: go.Figure) -> dict[str, int]:
    """Trace id -> position in ``fig.data``, for the traces that carry an id."""
    return {
        trace.meta["gliquid_id"]: i
        for i, trace in enumerate(fig.data)
        if isinstance(trace.meta, dict) and "gliquid_id" in trace.meta
    }


def apply_patches(fig: go.Figure, patches: dict) -> None:
    """Set each patch on the trace of ``fig`` with that id."""
    index = trace_index(fig)
    for key, patch in patches.items():
        fig.data[index[key]].update(patch)


def _flatten(patch, prefix=""):
    flat = {}
    for key, value in patch.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def restyle_calls(patches: dict) -> list[dict]:
    """``patches`` as ``Plotly.restyle`` calls, one per set of properties.

    Each call is ``{"data": {attr: [value per trace]}, "ids": [...]}``; ``data`` is the
    restyle update for the traces listed in ``ids``, in that order, and works as given with
    ``Plotly.restyle(gd, data, indices)`` and ``fig.plotly_restyle(data, indices)``.
    """
    calls = {}
    for trace_id, patch in patches.items():
        flat = _flatten(patch)
        keys = tuple(sorted(flat))
        call = calls.setdefault(keys, {"data": {key: [] for key in keys}, "ids": []})
        call["ids"].append(trace_id)
        for key in keys:
            call["data"][key].append(flat[key])
    return list(calls.values())


def embed_spec(fig: go.Figure, traces: HighlightTraces, states: dict, reset: dict) -> None:
    """Store every view in ``fig.layout.meta`` for the browser-side legend handler.

    Also turns off plotly's own legend click and double-click, which would hide traces.
    """
    spec = {
        "version": SPEC_VERSION,
        "liquidus": traces.liquidus,
        "legend": dict(traces.legend),
        "states": {phase: restyle_calls(patches) for phase, patches in states.items()},
        "reset": restyle_calls(reset),
    }
    meta = fig.layout.meta
    fig.update_layout(
        meta={**(meta if isinstance(meta, dict) else {}), SPEC_KEY: spec},
        legend=dict(itemclick=False, itemdoubleclick=False),
    )


ATTACH_JS = """
(function () {
  if (window.gliquidAttachPhaseHighlight) { return; }
  var VERSION = __SPEC_VERSION__, KEY = "__SPEC_KEY__";
  function idIndex(gd) {
    var index = {};
    gd.data.forEach(function (trace, i) {
      if (trace.meta && trace.meta.gliquid_id) { index[trace.meta.gliquid_id] = i; }
    });
    return index;
  }
  function selectedPhase(gd, spec, index) {
    var liquidus = gd.data[index[spec.liquidus]];
    if (!liquidus || liquidus.visible !== false) { return null; }
    var phases = Object.keys(spec.legend);
    for (var i = 0; i < phases.length; i++) {
      var marker = gd.data[index[spec.legend[phases[i]]]];
      if (marker && marker.visible !== "legendonly" && marker.visible !== false) {
        return phases[i];
      }
    }
    return null;
  }
  // The camera and zoom on screen; a redraw without them resets the view to the layout's, which lag.
  function liveView(gd) {
    var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    if (!scene || typeof scene.getCamera !== "function" || !scene.glplot) { return null; }
    var aspect = scene.glplot.getAspectratio();
    return {
      "scene.camera": scene.getCamera(),
      "scene.aspectratio": {x: aspect.x, y: aspect.y, z: aspect.z}
    };
  }
  function apply(gd, calls, index) {
    var view = liveView(gd);
    calls.forEach(function (call) {
      var data = {}, traces = [];
      call.ids.forEach(function (id, j) {
        if (!(id in index)) { return; }
        traces.push(index[id]);
        Object.keys(call.data).forEach(function (key) {
          (data[key] = data[key] || []).push(call.data[key][j]);
        });
      });
      if (!traces.length) { return; }
      if (view) { Plotly.update(gd, data, view, traces); } else { Plotly.restyle(gd, data, traces); }
    });
  }
  // Whether ``listener`` still listens to ``name`` on ``gd``; plotly.newPlot drops listeners.
  function listening(gd, name, listener) {
    return Boolean(listener) && (!gd._ev || typeof gd._ev.listeners !== "function" ||
      gd._ev.listeners(name).indexOf(listener) !== -1);
  }
  window.gliquidAttachPhaseHighlight = function (gd) {
    if (!gd || !gd.on || listening(gd, "plotly_legendclick", gd._gliquidPhaseHighlight)) {
      return;
    }
    var last = {curve: -1, stamp: -Infinity};
    gd._gliquidPhaseHighlight = function (event) {
      var meta = gd.layout && gd.layout.meta, spec = meta && meta[KEY];
      if (!spec || spec.version !== VERSION) { return; }
      var trace = gd.data[event.curveNumber];
      var phase = trace && trace.meta && trace.meta.gliquid_phase;
      if (typeof phase !== "string" || !(phase in spec.states)) { return false; }
      // The mouse event's own time: a restyle can delay handling the second click of a pair.
      var stamp = event.event && event.event.timeStamp ? event.event.timeStamp : performance.now();
      var delay = (gd._context && gd._context.doubleClickDelay) || 300;
      if (event.curveNumber === last.curve && stamp - last.stamp < delay) { return false; }
      last = {curve: event.curveNumber, stamp: stamp};
      var index = idIndex(gd);
      var showing = selectedPhase(gd, spec, index);
      apply(gd, showing === phase ? spec.reset : spec.states[phase], index);
      return false;
    };
    gd.on("plotly_legendclick", gd._gliquidPhaseHighlight);
    gd.on("plotly_legenddoubleclick", function () {
      var meta = gd.layout && gd.layout.meta, spec = meta && meta[KEY];
      if (spec && spec.version === VERSION) { return false; }
    });
  };
})();
""".replace("__SPEC_VERSION__", str(SPEC_VERSION)).replace("__SPEC_KEY__", SPEC_KEY)
