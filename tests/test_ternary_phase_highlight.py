"""get_plot('tx', highlight_phase=..., interactive_legend=...) -- one phase's liquidus field.

A highlighted view draws one solid phase's field in colour and dims the rest; every view is a
set of per-trace patches, applied in Python for ``highlight_phase`` and embedded as restyle
calls for ``interactive_legend``. The fields are the primary-phase partition of the liquidus
sheets; where the sheets do not describe the hull's liquidus, the figure has no fields.
Runs offline (cached Al-Mg-Si MP entries), in both hull modes.
"""

import json
import logging

import numpy as np
import plotly.graph_objects as go
import pytest

import gliquid.ternary as tmod
from gliquid.liquidus_sheets import TWO_LIQUIDS, LiquidusSheets
from gliquid.plotting import liquidus_fields as lf
from gliquid.plotting import phase_highlight as ph
from gliquid.plotting.style import (
    DIMMED_PHASE_COLOR,
    FIELD_OUTLINE_COLOR,
    GAP_OUTLINE_COLOR,
    GAP_OUTLINE_DASH,
)
from gliquid.plotting.ternary_surface import FIELD_COORD_DECIMALS, composition_at_pct

AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}


def _build(hull_mode, xs_mix=AMS_LD):
    p = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in xs_mix.items()},
        hull_mode=hull_mode,
        T_incr=10,
        param_format="linear",
    )
    p.process_data()
    return p


@pytest.fixture(scope="module", params=["gtx", "hsx"])
def plotter(request):
    return _build(request.param)


@pytest.fixture(scope="module")
def tx():
    """``tx(p, **kwargs)``: ``p.get_plot("tx", **kwargs)``, built once per module for each
    plotter and arguments. The figure is shared: read it, never restyle or change it."""
    figures = {}

    def get(p, **kwargs):
        key = (id(p), tuple(sorted(kwargs.items())))
        if key not in figures:
            figures[key] = (p, p.get_plot("tx", **kwargs))
        return figures[key][1]

    return get


def _legend_phases(plotter):
    return [phase for phase in plotter.color_map if phase != "L"]


def _by_id(fig):
    return {key: fig.data[i] for key, i in ph.trace_index(fig).items()}


def _primary_nodes(plotter, phase):
    sheets = LiquidusSheets.from_interpolation(plotter._ti)
    liq = plotter.liq_plotting_df
    temps = sheets(liq[["x0_orig", "x1_orig"]].to_numpy(float))
    primary = np.array(sheets.phases)[temps.argmax(axis=1)]
    return liq.loc[primary == phase, ["x0", "x1"]].to_numpy()


def _points(x, y):
    return {tuple(p) for p in np.round(np.column_stack([x, y]), FIELD_COORD_DECIMALS)}


def _mesh_area(trace):
    xy = np.column_stack([trace.x, trace.y])
    return float(lf._triangle_areas(xy, np.column_stack([trace.i, trace.j, trace.k])).sum())


def test_legend_lists_the_solid_phases_only(plotter, tx):
    fig = tx(plotter)
    legend = [trace for trace in fig.data if trace.showlegend]
    assert len(legend) == len(_legend_phases(plotter))
    assert "L" not in {trace.name for trace in legend}


def test_a_figure_without_the_legend_carries_no_highlight(plotter, tx):
    fig = tx(plotter, interactive_legend=False)
    assert not ph.trace_index(fig)
    assert ph.SPEC_KEY not in fig.layout.meta
    assert fig.layout.legend.itemclick is None


@pytest.mark.parametrize("phase", ["Mg2Si", "Mg (hcp)"])
def test_highlighted_view(plotter, tx, phase):
    fig = tx(plotter, highlight_phase=phase)
    traces = _by_id(fig)
    phases = _legend_phases(plotter)
    n = phases.index(phase)

    assert traces["gliquid-liquidus"].visible is False
    field = traces[f"gliquid-field-{n}"]
    assert field.visible is True
    assert field.colorscale == go.Mesh3d(colorscale="Viridis").colorscale
    assert traces[f"gliquid-rest-{n}"].visible is True
    assert traces[f"gliquid-outline-{n}"].visible is True
    shown = {key for key, trace in traces.items() if trace.visible is True}
    assert {key for key in shown if key.startswith(("gliquid-field", "gliquid-rest"))} == {
        f"gliquid-field-{n}",
        f"gliquid-rest-{n}",
    }

    for i, other in enumerate(phases):
        marker = traces[f"gliquid-legend-{i}"]
        assert marker.visible == (True if other == phase else "legendonly")
    for trace in fig.data:
        if trace.meta and trace.meta.get("gliquid_role") == "solid":
            expected = plotter.color_map[phase] if trace.meta["gliquid_phase"] == phase else None
            assert trace.line.color == (expected or DIMMED_PHASE_COLOR)

    primary = _primary_nodes(plotter, phase)
    assert len(primary) > 0
    assert _points(*primary.T) <= _points(field.x, field.y)


def test_fields_tile_the_drawn_surface(plotter, tx):
    traces = _by_id(tx(plotter, interactive_legend=True))
    whole = _mesh_area(traces["gliquid-liquidus"])
    fields = [key for key in traces if key.startswith("gliquid-field-")]
    assert sum(_mesh_area(traces[key]) for key in fields) == pytest.approx(whole, rel=1e-6)
    for key in fields:
        rest = traces[key.replace("field", "rest")]
        assert _mesh_area(traces[key]) + _mesh_area(rest) == pytest.approx(whole, rel=1e-6)


def test_surface_is_the_liquidus_of_the_sheets(plotter, tx):
    (surface,) = [trace for trace in tx(plotter).data if trace.name == "liquidus"]
    sheets = LiquidusSheets.from_interpolation(plotter._ti)
    comp = plotter._liquidus_partition(plotter.triangulation.simplices).surface.comp
    assert np.asarray(surface.z) == pytest.approx(sheets.liquidus(comp) - 273.15, abs=1e-6)
    resolution = 1.0 if plotter.hull_mode == "hsx" else plotter.T_incr + 1.0
    liq = plotter.liq_plotting_df
    node_liquidus = sheets.liquidus(liq[["x0_orig", "x1_orig"]].to_numpy(float)) - 273.15
    assert (liq["T"].to_numpy() <= node_liquidus + resolution).all()


def test_junctions_and_field_vertices_sit_on_their_own_sheets(plotter, tx):
    tx(plotter)
    part = plotter._liquidus_partition(plotter.triangulation.simplices)
    sheets = LiquidusSheets.from_interpolation(plotter._ti)
    temps = sheets(part.surface.comp) - 273.15
    triples = [v for v, phases in enumerate(part.vertex_phases) if len(phases) == 3]
    assert triples
    for v in triples:
        assert np.ptp(temps[v, [part.phases.index(p) for p in part.vertex_phases[v]]]) < 1.0
    for k in np.unique(part.owner):
        vertices = np.unique(part.surface.faces[part.owner == k])
        assert (part.surface.T[vertices] - temps[vertices, k]).max() < 1.0


def test_liquidus_hover_names_the_phases_meeting_at_each_vertex(plotter, tx):
    (surface,) = [trace for trace in tx(plotter).data if trace.name == "liquidus"]
    part = plotter._liquidus_partition(plotter.triangulation.simplices)
    assert list(np.asarray(surface.customdata)[:, 3]) == [
        ", ".join(phases) for phases in part.vertex_phases
    ]


def test_liquidus_hover_gives_the_at_pct_of_each_element_in_sys_name_order(plotter, tx):
    fig = tx(plotter, interactive_legend=True)
    part = plotter._liquidus_partition(plotter.triangulation.simplices)
    roles = ("liquidus", "field", "rest")
    meshes = [t for t in fig.data if (t.meta or {}).get("gliquid_role") in roles]
    assert len(meshes) > 1, "the field meshes share the liquidus hover"
    for mesh in meshes:
        assert mesh.hovertemplate.count(" at. %<br>") == 3
        lines = mesh.hovertemplate.split("<br>")
        assert [line.split(":")[0] for line in lines[1:4]] == plotter.sys_name.split("-")
        assert "x_" not in mesh.hovertemplate
    (surface,) = [t for t in meshes if t.name == "liquidus"]
    pct = np.asarray(surface.customdata[:, :3], dtype=float)
    fractions = dict(
        zip(plotter.components, (1 - part.surface.comp.sum(axis=1), *part.surface.comp.T))
    )
    expected = np.column_stack([100 * fractions[e] for e in plotter.sys_name.split("-")])
    assert pct == pytest.approx(expected, abs=0.0501)
    assert pct.sum(axis=1) == pytest.approx(100.0, abs=0.15)


def test_composition_at_pct_orders_columns_alphabetically():
    pct = composition_at_pct(np.array([[0.2, 0.3]]), ["Si", "Al", "Mg"])
    assert pct.tolist() == [[20.0, 30.0, 50.0]]


def _segment_distance(points, a, b):
    """(p, e) distance from each of ``points`` to each segment from ``a`` to ``b``."""
    ab = b - a
    t = np.einsum("pek,ek->pe", points[:, None, :] - a[None], ab) / np.einsum("ek,ek->e", ab, ab)
    nearest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - nearest, axis=2)


def test_isotherms_run_flat_along_the_drawn_surface(plotter, tx):
    fig = tx(plotter)
    (surface,) = [trace for trace in fig.data if trace.name == "liquidus"]
    vertices = np.column_stack([surface.x, surface.y, surface.z])
    faces = np.column_stack([surface.i, surface.j, surface.k])
    edges = np.unique(
        np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0,
    )
    isotherms = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d"
        and isinstance(trace.line.color, str)
        and trace.line.color == "white"
    ]
    assert isotherms
    for trace in isotherms:
        z = np.array([v for v in trace.z if v is not None], dtype=float)
        assert np.ptp(z) == 0
        points = np.column_stack(
            [[v for v in trace.x if v is not None], [v for v in trace.y if v is not None], z]
        ).astype(float)
        near = _segment_distance(points, vertices[edges[:, 0]], vertices[edges[:, 1]])
        assert near.min(axis=1).max() < 1e-6


def test_a_node_lists_every_phase_of_the_facets_meeting_it_at_its_liquidus(plotter, tx):
    import pandas as pd

    tx(plotter)
    rows = pd.concat(plotter.equil_df_list)
    solids = rows[rows["Phase"] != "L"].groupby("simplex_id")["Phase"].apply(set).to_dict()
    liquid = rows[rows["Phase"] == "L"]
    coldest = liquid[liquid["T"] == liquid.groupby(["x0", "x1"])["T"].transform("min")]
    expected = coldest.groupby(["x0", "x1"])["simplex_id"].apply(
        lambda ids: ", ".join(sorted(set().union(*(solids.get(i, set()) for i in ids))))
    )
    shown = plotter.liq_plotting_df.set_index(["x0", "x1"])["coexistent_phases"]
    assert shown.sort_index().to_dict() == expected.sort_index().to_dict()


def test_a_raised_window_floor_keeps_the_sheet_fields():
    p = tmod.TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        hull_mode="gtx",
        T_incr=10,
        temp_range=(1000.0, None),
        param_format="linear",
    )
    p.process_data()
    p.get_plot("tx")
    assert (p.liq_plotting_df["T"] == p.conds[0]).any()
    assert p._liquidus_partition(p.triangulation.simplices) is not None


GAP_LD = {**AMS_LD, "Al-Mg": [25000.0, 1.0, 2000.0, -0.5]}


@pytest.fixture(scope="module", params=["gtx", "hsx"])
def gap(request):
    return _build(request.param, GAP_LD)


def _lines(fig, colour):
    return [
        t
        for t in fig.data
        if t.type == "scatter3d"
        and isinstance(t.line.color, str)
        and t.line.color == colour
        and t.visible is not False
    ]


def _line_points(traces):
    return set().union(
        *(_points(*np.array([(x, y) for x, y in zip(t.x, t.y) if x is not None]).T) for t in traces)
    )


def _polyline_points(lines):
    return set().union(*(_points(line[:, 0], line[:, 1]) for line in lines))


def test_a_liquid_miscibility_gap_is_coloured_with_the_surface_and_dotted_where_it_borders(gap, tx):
    fig = tx(gap)
    part = gap._liquidus_partition(gap.triangulation.simplices)
    assert part is not None and TWO_LIQUIDS in part.fields
    (base,) = [t for t in fig.data if t.name == "liquidus"]
    whole = float(lf._triangle_areas(part.surface.xy, part.surface.faces).sum())
    assert _mesh_area(base) == pytest.approx(whole, rel=1e-6)
    assert set(np.asarray(base.customdata)[:, 3]) >= {TWO_LIQUIDS}
    assert not _lines(fig, FIELD_OUTLINE_COLOR)
    dotted = _lines(fig, GAP_OUTLINE_COLOR)
    assert {t.line.dash for t in dotted} == {GAP_OUTLINE_DASH}
    shared = lf.field_boundaries(part, TWO_LIQUIDS)
    assert None in shared
    inner = set().union(*(_polyline_points(lines) for phase, lines in shared.items() if phase))
    assert _line_points(dotted) == inner
    assert TWO_LIQUIDS not in {t.name for t in fig.data if t.showlegend}


def test_a_highlighted_neighbour_s_outline_replaces_their_shared_dotted_boundary(gap, tx):
    tx(gap)
    part = gap._liquidus_partition(gap.triangulation.simplices)
    shared = lf.field_boundaries(part, TWO_LIQUIDS)
    neighbour = next(phase for phase in shared if phase in _legend_phases(gap))
    fig = gap.get_plot("tx", highlight_phase=neighbour)
    rest = _by_id(fig)[f"gliquid-rest-{_legend_phases(gap).index(neighbour)}"]
    assert rest.visible is True
    assert _mesh_area(rest) + _mesh_area(
        _by_id(fig)[f"gliquid-field-{_legend_phases(gap).index(neighbour)}"]
    ) == pytest.approx(float(lf._triangle_areas(part.surface.xy, part.surface.faces).sum()))
    (outline,) = _lines(fig, FIELD_OUTLINE_COLOR)
    assert _polyline_points(shared[neighbour]) <= _line_points([outline])
    others = [lines for phase, lines in shared.items() if phase not in (neighbour, None)]
    assert _line_points(_lines(fig, GAP_OUTLINE_COLOR)) == set().union(
        *(_polyline_points(lines) for lines in others)
    )
    base = tx(gap, interactive_legend=True)
    _restyle(fig, base.layout.meta[ph.SPEC_KEY]["reset"])
    assert _line_points(_lines(fig, GAP_OUTLINE_COLOR)) == set().union(
        *(_polyline_points(lines) for phase, lines in shared.items() if phase)
    )


def test_sheets_that_miss_the_hull_draw_the_node_mesh_without_fields(plotter, monkeypatch, caplog):
    real = LiquidusSheets.liquidus
    monkeypatch.setattr(LiquidusSheets, "liquidus", lambda self, comp: real(self, comp) - 200.0)
    monkeypatch.setattr(plotter, "_partition_cache", (None, None, None))
    with caplog.at_level(logging.WARNING, logger="gliquid.ternary"):
        fig = plotter.get_plot("tx", highlight_phase="Mg2Si")
    assert plotter._liquidus_partition(plotter.triangulation.simplices) is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    rejected = [m for m in warnings if "disagree with the hull" in m]
    assert len(rejected) == 1
    assert rejected[0].startswith(plotter.sys_name) and "%)" in rejected[0]
    assert any("no phase fields to highlight" in m for m in warnings)
    (liquidus,) = [t for t in fig.data if t.name == "liquidus"]
    assert len(liquidus.x) == len(plotter.liq_plotting_df)
    assert ph.trace_index(fig) == {}


def test_the_colab_two_view_render_runs():
    from gliquid.plotting.export import side_by_side_html
    from gliquid.plotting.ternary_labels import orient_labels
    from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter

    ternary = TernaryLiquidInterpolation(
        ["Al", "Mg", "Si"],
        delta=0.1,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        param_format="linear",
    )
    plotter = TLIPlotter(ternary)
    rotatable = plotter.get_plot("tx", highlight_phase="Mg2Si")
    above = plotter.get_plot("tx", highlight_phase="Mg2Si", interactive_legend=False)
    above.layout.annotations = ()
    aspect = above.layout.scene.aspectratio
    above.update_layout(
        showlegend=False,
        scene=dict(
            dragmode=False,
            aspectratio=dict(x=1.5 * aspect.x, y=1.5 * aspect.y, z=aspect.z),
            camera=dict(eye=dict(x=0, y=0, z=1), up=dict(x=0, y=1, z=0)),
        ),
    )
    orient_labels(above)
    html = side_by_side_html(
        [rotatable, above],
        configs=[None, {"displayModeBar": False, "scrollZoom": False}],
    )
    assert html.count("<iframe") == 2
    assert html.count("window.gliquidAttachPhaseHighlight(gd)") == 2


def test_every_view_is_a_valid_figure(plotter, tx):
    for phase in _legend_phases(plotter):
        fig = tx(plotter, highlight_phase=phase)
        z_lo, z_hi = fig.layout.scene.zaxis.range
        for trace in fig.data:
            z = [v for v in (trace.z if trace.z is not None else []) if v is not None]
            assert all(z_lo <= v <= z_hi for v in z), trace.meta
        assert sum(trace.name == "corner labels" for trace in fig.data) == 1
        assert sum(trace.name == "axes" for trace in fig.data) == 1
        assert not any(
            trace.type == "scatter3d" and trace.line and trace.line.dash for trace in fig.data
        )
        go.Figure(fig.to_dict())


def _restyle(fig, calls):
    index = ph.trace_index(fig)
    for call in calls:
        fig.plotly_restyle(call["data"], [index[key] for key in call["ids"]])


def _data(fig):
    return json.loads(fig.to_json())["data"]


def test_interactive_views_match_the_static_ones(plotter, tx):
    spec = tx(plotter, interactive_legend=True).layout.meta[ph.SPEC_KEY]
    assert spec["version"] == ph.SPEC_VERSION
    assert set(spec["states"]) == set(_legend_phases(plotter))
    for phase in _legend_phases(plotter):
        fig = plotter.get_plot("tx", interactive_legend=True)
        _restyle(fig, spec["states"][phase])
        assert _data(fig) == _data(tx(plotter, highlight_phase=phase))


def test_reset_restores_the_full_view(plotter, tx):
    base = tx(plotter, interactive_legend=True)
    spec = base.layout.meta[ph.SPEC_KEY]
    fig = plotter.get_plot("tx", interactive_legend=True, highlight_phase="Mg2Si")
    _restyle(fig, spec["reset"])
    assert _data(fig) == _data(base)


def test_interactive_legend_turns_off_plotly_toggling(plotter, tx):
    fig = tx(plotter, interactive_legend=True)
    assert fig.layout.legend.itemclick is False
    assert fig.layout.legend.itemdoubleclick is False


@pytest.mark.parametrize("phase", ["L", "Nope"])
def test_unknown_phase_is_rejected(plotter, phase):
    with pytest.raises(ValueError, match="not a solid phase"):
        plotter.get_plot("tx", highlight_phase=phase)


def test_unknown_tx_keyword_is_rejected(plotter):
    with pytest.raises(TypeError, match="unexpected keywords"):
        plotter.get_plot("tx", highlight="Mg2Si")


def test_highlight_options_are_tx_only(plotter):
    with pytest.raises(ValueError, match="tx"):
        plotter.get_plot("ch", T_celsius=700.0, highlight_phase="Mg2Si")


class TestStates:
    """phase_highlight on its own: states, restyle calls, colours, scripts."""

    @pytest.fixture
    def traces(self):
        return ph.HighlightTraces(
            liquidus="liq",
            dimmed_liquidus="liq-dim",
            legend={"A": "leg-A", "B": "leg-B"},
            solids={"A": ("sol-A", "#ff0000"), "B": ("sol-B", "#00ff00")},
            fields={"A": "field-A"},
            rests={"A": "rest-A"},
            outlines={"A": "out-A"},
        )

    def test_a_phase_without_a_field_dims_the_whole_liquidus(self, traces):
        states, reset = ph.build_states(traces)
        assert states["B"]["liq-dim"] == {"visible": True}
        assert states["B"]["field-A"] == {"visible": False}
        assert states["A"]["liq-dim"] == {"visible": False}
        assert states["A"]["field-A"] == {"visible": True}
        assert reset["liq"] == {"visible": True}
        assert reset["liq-dim"] == {"visible": False}
        assert states["A"]["sol-B"] == {"line": {"color": DIMMED_PHASE_COLOR}}
        assert reset["sol-B"] == {"line": {"color": "#00ff00"}}

    def test_restyle_calls_group_by_property(self, traces):
        states, _ = ph.build_states(traces)
        calls = ph.restyle_calls(states["A"])
        by_keys = {tuple(call["data"]): call for call in calls}
        assert set(by_keys) == {("visible",), ("line.color",)}
        colors = by_keys[("line.color",)]
        assert dict(zip(colors["ids"], colors["data"]["line.color"])) == {
            "sol-A": "#ff0000",
            "sol-B": DIMMED_PHASE_COLOR,
        }

    def test_dimmed_colorscale(self):
        viridis = go.Mesh3d(colorscale="Viridis").colorscale
        grey = ph.dimmed_colorscale(viridis, share=1.0)
        for _, colour in grey:
            r, g, b = (int(c) for c in colour[4:-1].split(","))
            assert r == g == b
        same = ph.dimmed_colorscale(viridis, share=0.0)
        assert [pos for pos, _ in same] == [pos for pos, _ in viridis]

    def test_script(self):
        assert "window.gliquidAttachPhaseHighlight = function (gd)" in ph.ATTACH_JS
        assert ph.SPEC_KEY in ph.ATTACH_JS

    def test_script_sends_the_camera_and_zoom_on_screen_with_every_update(self):
        assert '"scene.camera": scene.getCamera()' in ph.ATTACH_JS
        assert '"scene.aspectratio": {x: aspect.x, y: aspect.y, z: aspect.z}' in ph.ATTACH_JS
        assert "Plotly.update(gd, data, view, traces)" in ph.ATTACH_JS
        assert "Plotly.restyle(gd, data, traces)" in ph.ATTACH_JS
