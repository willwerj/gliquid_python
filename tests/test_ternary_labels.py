"""Composition ticks and flat labels on the base triangle of the ternary liquidus (tx) figure.

The triangle's corners are ``components[0]`` (lower left), ``components[1]`` (lower right) and
``components[2]`` (apex), at the cartesian points of :func:`gliquid.ternary.ternary_to_cartesian`.
Each edge is ticked every 10 at.% of the element it runs towards counterclockwise, with the tick
on the line of constant composition, outside the triangle, numbered, and titled
``at. % <element>``; each corner carries its element's name. The labels are filled glyphs lying
flat on the floor, in three views: in the view of an edge every label reads upright from that
edge's side. Each kind of label is one mesh carrying every view, and the figure draws the view
of the edge lowest on screen under its camera. Runs offline (cached Al-Mg-Si MP entries).
"""

import re
import warnings
from html import unescape

import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from gliquid.plotting import phase_highlight
from gliquid.plotting import ternary_labels as tl
from gliquid.plotting import ternary_surface as ts
from gliquid.ternary import TLIPlotter

H = np.sqrt(3) / 2
CENTROID = np.array([0.5, H / 3])
COMPONENTS = ["A", "B", "C"]
CORNERS = [(0.0, 0.0), (1.0, 0.0), (0.5, H)]
EDGE_ELEMENT = {"bottom": 1, "right": 2, "left": 0}
EDGE_NORMAL = {"bottom": (0.0, -1.0), "right": (H, 0.5), "left": (-H, 0.5)}
EDGE_CORNERS = {"bottom": (0, 1), "right": (1, 2), "left": (2, 0)}
LABEL_TRACES = ("composition tick labels", "composition axis titles", "corner labels")
KINDS = dict(zip(LABEL_TRACES, ("tick", "title", "corner")))

AMS_LD = {
    "Al-Mg": [-9000.0, 1.0, 2000.0, -0.5],
    "Mg-Si": [-20000.0, 3.0, -5000.0, 1.0],
    "Si-Al": [-4000.0, 0.5, 1000.0, 0.0],
}


def _fractions(x, y):
    """Mole fractions of (components[0], components[1], components[2]) at plot point (x, y)."""
    x1 = y / H
    x0 = x - 0.5 * x1
    return np.array([1.0 - x0 - x1, x0, x1])


def _inside(x, y, tol=1e-9):
    return bool(np.all(_fractions(x, y) >= -tol))


def _segments(trace):
    """The (start, end) points of a lines trace whose segments are separated by None."""
    pts = [(x, y) for x, y in zip(trace.x, trace.y)]
    segs, current = [], []
    for p in pts + [(None, None)]:
        if p[0] is None:
            if current:
                segs.append(current)
            current = []
        else:
            current.append(np.array(p, dtype=float))
    return segs


def _named(fig, name):
    matches = [t for t in fig.data if t.name == name]
    assert len(matches) == 1, f"expected one trace named {name!r}, found {len(matches)}"
    return matches[0]


def _drawn(trace):
    """The view a label mesh draws, after checking its x, y are that view's."""
    view = trace.meta[tl.VIEW_KEY]
    x, y = trace.meta[tl.XY_KEY][view]
    assert list(trace.x) == list(x) and list(trace.y) == list(y)
    return view


def _rising(edge):
    """Unit vector along ``edge`` in the direction its value rises."""
    ticks = [t for t in tl.composition_ticks(COMPONENTS) if t["edge"] == edge]
    d = np.subtract(ticks[-1]["point"], ticks[0]["point"])
    return d / np.linalg.norm(d)


def _ink(label):
    """The glyph vertices of ``label`` in plot coordinates."""
    return tl.label_outline(label) + label["center"]


def _box(label):
    """``label``'s ink box in the frame of its baseline: (x0, y0, x1, y1)."""
    a = label["angle"]
    local = _ink(label) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (*local.min(axis=0), *local.max(axis=0))


def _rotate(points, degrees):
    t = np.radians(degrees)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(points) - CENTROID) @ r.T + CENTROID


def _triangle_area(vertices, faces):
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * float(np.abs(cross).sum())


def _camera(eye, center=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    return {name: dict(zip("xyz", v)) for name, v in (("eye", eye), ("center", center), ("up", up))}


class TestCompositionTicks:
    def test_nine_ticks_per_edge_at_each_tenth(self):
        ticks = tl.composition_ticks(COMPONENTS)
        for edge, index in EDGE_ELEMENT.items():
            on_edge = [t for t in ticks if t["edge"] == edge]
            assert [t["value"] for t in on_edge] == pytest.approx(np.arange(1, 10) / 10)
            assert {t["element"] for t in on_edge} == {COMPONENTS[index]}

    def test_each_point_is_on_its_edge_at_its_value(self):
        for tick in tl.composition_ticks(COMPONENTS):
            frac = _fractions(*tick["point"])
            assert frac[EDGE_ELEMENT[tick["edge"]]] == pytest.approx(tick["value"])
            assert frac.min() == pytest.approx(0.0, abs=1e-12)

    def test_direction_follows_constant_composition_into_the_triangle(self):
        for tick in tl.composition_ticks(COMPONENTS):
            p, d = np.array(tick["point"]), np.array(tick["direction"])
            assert np.linalg.norm(d) == pytest.approx(1.0)
            inward = p + 0.01 * d
            assert _inside(*inward)
            index = EDGE_ELEMENT[tick["edge"]]
            assert _fractions(*inward)[index] == pytest.approx(tick["value"])

    def test_normal_points_away_from_the_triangle(self):
        for tick in tl.composition_ticks(COMPONENTS):
            p, n = np.array(tick["point"]), np.array(tick["normal"])
            assert np.linalg.norm(n) == pytest.approx(1.0)
            assert np.dot(n, p - CENTROID) > 0
            assert not _inside(*(p + 0.01 * n))


class TestLabelView:
    def test_the_default_camera_looks_over_the_right_edge(self):
        assert tl.label_view() == "right"
        assert tl.label_view({}) == "right"
        assert tl.label_view({"projection": {"type": "orthographic"}}) == "right"

    @pytest.mark.parametrize("edge", tl.VIEWS)
    def test_a_camera_over_an_edge_shows_its_view(self, edge):
        nx, ny = EDGE_NORMAL[edge]
        assert tl.label_view(_camera((2 * nx, 2 * ny, 1.0))) == edge

    def test_the_view_is_the_edge_whose_midpoint_is_nearest_the_camera(self):
        """With ``center`` at the triangle's centroid, ``up`` along z and the camera above."""
        midpoints = {e: (H / 3) * np.array(n) for e, n in EDGE_NORMAL.items()}
        for degrees in np.arange(0.25, 360, 0.5):
            eye = np.array([2 * np.cos(np.radians(degrees)), 2 * np.sin(np.radians(degrees))])
            for height in (0.3, 1.25, 4.0):
                nearest = min(midpoints, key=lambda e: np.linalg.norm(eye - midpoints[e]))
                assert tl.label_view(_camera((*eye, height))) == nearest, (degrees, height)

    def test_labels_flip_where_two_midpoints_are_equally_near(self):
        # The lower right corner lies at -30 degrees from the centroid.
        for degrees, view in ((-29.99, "right"), (-30.01, "bottom")):
            eye = (2 * np.cos(np.radians(degrees)), 2 * np.sin(np.radians(degrees)), 1.0)
            assert tl.label_view(_camera(eye)) == view

    def test_panning_keeps_the_view(self):
        for eye in ((1.25, 1.25, 1.25), (0.3, -2.0, 1.0), (-1.9, 0.9, 1.0)):
            shift = np.array([0.4, -0.7, 0.2])
            panned = _camera(tuple(np.add(eye, shift)), center=tuple(shift))
            assert tl.label_view(panned) == tl.label_view(_camera(eye))

    @pytest.mark.parametrize("edge", tl.VIEWS)
    def test_a_top_view_shows_the_edge_at_the_bottom_of_the_screen(self, edge):
        nx, ny = EDGE_NORMAL[edge]
        assert tl.label_view(_camera((0.0, 0.0, 2.5), up=(-nx, -ny, 0.0))) == edge

    def test_a_camera_level_with_the_floor_shows_the_bottom_view(self):
        assert tl.label_view(_camera((2.0, 0.0, 0.0))) == "bottom"
        assert tl.label_view(_camera((0.0, 0.0, 0.0))) == "bottom"

    def test_plotly_camera_objects_are_read_like_dicts(self):
        camera = go.layout.scene.Camera(eye=dict(x=-1.9, y=0.9, z=1.0))
        assert tl.label_view(camera) == tl.label_view({"eye": {"x": -1.9, "y": 0.9, "z": 1.0}})
        assert tl.label_view(camera) == "left"


class TestCompositionLabels:
    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_a_number_per_tick_a_title_per_edge_and_a_name_per_corner(self, view):
        labels = tl.composition_labels(COMPONENTS, view)
        for edge, index in EDGE_ELEMENT.items():
            numbers = [
                lab["text"] for lab in labels if lab["edge"] == edge and lab["kind"] == "tick"
            ]
            assert numbers == [str(10 * k) for k in range(1, 10)]
            titles = [
                lab["text"] for lab in labels if lab["edge"] == edge and lab["kind"] == "title"
            ]
            assert titles == [f"at. % {COMPONENTS[index]}"]
        names = [lab for lab in labels if lab["kind"] == "corner"]
        assert [(lab["text"], lab["element"], lab["edge"]) for lab in names] == [
            (el, el, None) for el in COMPONENTS
        ]
        assert {lab["weight"] for lab in names} == {"bold"}
        assert {lab["weight"] for lab in labels if lab["kind"] != "corner"} == {"normal"}

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_every_label_reads_upright_from_the_view_edge(self, view):
        inward = -np.array(EDGE_NORMAL[view])
        for label in tl.composition_labels(COMPONENTS, view):
            a = label["angle"]
            baseline, top = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            assert baseline == pytest.approx(_rising(view))
            assert top == pytest.approx(inward)

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_numbers_are_set_off_their_tick_along_the_edge_normal(self, view):
        ticks = tl.composition_ticks(COMPONENTS)
        numbers = [lab for lab in tl.composition_labels(COMPONENTS, view) if lab["kind"] == "tick"]
        for tick, label in zip(ticks, numbers, strict=True):
            p, n = np.array(tick["point"]), np.array(tick["normal"])
            offset = np.subtract(label["center"], p)
            assert offset - (offset @ n) * n == pytest.approx([0.0, 0.0], abs=1e-12)
            tick_end = tl.TICK_LENGTH * -float(np.dot(tick["direction"], n))
            nearest = float(np.min((_ink(label) - p) @ n))
            assert nearest == pytest.approx(tick_end + tl.TICK_LABEL_GAP)

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_titles_sit_beyond_their_edges_numbers(self, view):
        labels = tl.composition_labels(COMPONENTS, view)
        for edge, n in ((e, np.array(v)) for e, v in EDGE_NORMAL.items()):
            start = np.array(CORNERS[(EDGE_ELEMENT[edge] + 2) % 3])
            numbers = [lab for lab in labels if lab["edge"] == edge and lab["kind"] == "tick"]
            (title,) = [lab for lab in labels if lab["edge"] == edge and lab["kind"] == "title"]
            far = max(float(np.max((_ink(lab) - start) @ n)) for lab in numbers)
            nearest = float(np.min((_ink(title) - start) @ n))
            assert nearest == pytest.approx(far + tl.TICK_TITLE_GAP)
            midpoint = start + 0.5 * np.subtract(CORNERS[(EDGE_ELEMENT[edge]) % 3], start)
            offset = np.subtract(title["center"], midpoint)
            assert offset - (offset @ n) * n == pytest.approx([0.0, 0.0], abs=1e-12)

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_names_sit_beyond_their_own_corner(self, view):
        """The view edge's corners are named directly away from the centroid; the far corner
        beside itself, along the baseline."""
        for label in tl.composition_labels(COMPONENTS, view):
            if label["kind"] != "corner":
                continue
            k = COMPONENTS.index(label["element"])
            corner = np.array(CORNERS[k])
            near = k in EDGE_CORNERS[view]
            outward = (corner - CENTROID) / np.linalg.norm(corner - CENTROID)
            outward = outward if near else _rising(view)
            offset = np.subtract(label["center"], corner)
            assert offset - (offset @ outward) * outward == pytest.approx([0, 0], abs=1e-12)
            assert offset @ outward > 0
            nearest = float(np.min((_ink(label) - corner) @ outward))
            assert nearest >= tl.CORNER_LABEL_GAP - 1e-12
            own = min(range(3), key=lambda j: np.hypot(*np.subtract(label["center"], CORNERS[j])))
            assert COMPONENTS[own] == label["element"]

    @pytest.mark.parametrize("view", tl.VIEWS)
    @pytest.mark.parametrize("components", [["Al", "Mg", "Si"], ["W", "Mo", "I"]])
    def test_names_stay_clear_of_the_line_above_their_corner(self, view, components):
        """Every phase draws a vertical line up from its composition, so each corner carries
        one. Under an orthographic camera looking down from azimuth ``a``, the line covers, on
        the floor, the ray from the corner pointing away from the camera; no name ink above its
        corner on screen may come within ``CORNER_LINE_GAP`` of it, for any camera that shows
        the view."""
        names = [lab for lab in tl.composition_labels(components, view) if lab["kind"] == "corner"]
        centre = np.degrees(np.arctan2(*EDGE_NORMAL[view][::-1]))
        checked = 0
        for degrees in np.arange(centre - 59.75, centre + 60, 0.5):
            towards = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
            assert tl.label_view(_camera((*(2 * towards), 1.0))) == view
            right = np.array([-towards[1], towards[0]])
            for label in names:
                ink = _ink(label) - CORNERS[components.index(label["element"])]
                above = ink @ -towards > 0
                assert np.all(np.abs(ink[above] @ right) >= tl.CORNER_LINE_GAP - 1e-12), (
                    label["text"],
                    degrees,
                )
                checked += int(above.sum())
        assert checked > 0

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_no_label_touches_another_the_triangle_or_a_tick(self, view):
        labels = tl.composition_labels(["Al", "Mg", "Si"], view)
        boxes = [_box(lab) for lab in labels]
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = boxes[a], boxes[b]
                apart = ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0
                assert apart, (labels[a]["text"], labels[b]["text"])
        angle = labels[0]["angle"]
        to_local = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        samples = [
            np.array(t["point"]) - s * tl.TICK_LENGTH * np.array(t["direction"])
            for t in tl.composition_ticks(COMPONENTS)
            for s in np.linspace(0.0, 1.0, 26)
        ] + [
            np.array(CORNERS[k]) + s * np.subtract(CORNERS[(k + 1) % 3], CORNERS[k])
            for k in range(3)
            for s in np.linspace(0.0, 1.0, 201)
        ]
        local = np.array(samples) @ to_local
        for label, (x0, y0, x1, y1) in zip(labels, boxes):
            hit = (
                (local[:, 0] >= x0)
                & (local[:, 0] <= x1)
                & (local[:, 1] >= y0)
                & (local[:, 1] <= y1)
            )
            assert not hit.any(), label["text"]
            assert not any(_inside(x, y) for x, y in _ink(label)), label["text"]

    def test_the_numbers_of_each_view_are_those_of_the_last_turned_a_third(self):
        cycle = {"bottom": "right", "right": "left", "left": "bottom"}
        for view in tl.VIEWS:
            here = [lab for lab in tl.composition_labels(COMPONENTS, view) if lab["kind"] == "tick"]
            there = [
                lab
                for lab in tl.composition_labels(COMPONENTS, cycle[view])
                if lab["kind"] == "tick"
            ]
            for edge in tl.VIEWS:
                turned = _rotate([lab["center"] for lab in here if lab["edge"] == edge], 120)
                after = [lab["center"] for lab in there if lab["edge"] == cycle[edge]]
                assert turned == pytest.approx(np.array(after), abs=1e-12)

    def test_sizes_rank_names_titles_numbers(self):
        sizes = {lab["kind"]: lab["size"] for lab in tl.composition_labels(COMPONENTS)}
        assert sizes == {
            "tick": tl.TICK_LABEL_SIZE,
            "title": tl.TICK_TITLE_SIZE,
            "corner": tl.CORNER_LABEL_SIZE,
        }
        assert tl.CORNER_LABEL_SIZE > tl.TICK_TITLE_SIZE > tl.TICK_LABEL_SIZE

    def test_an_unknown_view_is_refused(self):
        with pytest.raises(ValueError, match="sideways"):
            tl.composition_labels(COMPONENTS, "sideways")


class TestGlyphMesh:
    @pytest.mark.parametrize(
        ("text", "weight"),
        [("10", "normal"), ("80", "normal"), ("at. % Mg", "normal"), ("at. % Bi", "normal")]
        + [("Mg", "bold"), ("Si", "bold"), ("Al", "bold")],
    )
    def test_triangles_fill_exactly_the_glyph_outlines(self, text, weight):
        vertices, faces = tl.glyph_mesh(text, weight)
        font = FontProperties(family=tl.LABEL_FONT, weight=weight)
        rings = [
            np.asarray(r)[:-1] for r in TextPath((0, 0), text, size=1.0, prop=font).to_polygons()
        ]
        signed = sum(
            0.5 * (np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))
            for r in rings
        )
        assert _triangle_area(vertices, faces) == pytest.approx(abs(signed), rel=1e-9)
        shift = np.array(vertices[0]) - np.asarray(rings[0][0])
        centroids = vertices[faces].mean(axis=1) - shift
        crossings = sum(Path(r).contains_points(centroids).astype(int) for r in rings)
        assert (crossings % 2 == 1).all()

    def test_bold_is_heavier_than_normal(self):
        normal, bold = (tl.glyph_mesh("Mg", w) for w in ("normal", "bold"))
        assert _triangle_area(*bold) > 1.2 * _triangle_area(*normal)

    def test_text_is_centred_on_its_ink_width_and_capital_height(self):
        vertices, _ = tl.glyph_mesh("50")
        assert vertices[:, 0].min() == pytest.approx(-vertices[:, 0].max())
        font = FontProperties(family=tl.LABEL_FONT)
        cap = TextPath((0, 0), "M", size=1.0, prop=font).get_extents().ymax
        assert vertices[:, 1].max() == pytest.approx(cap / 2, abs=0.02)

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_labels_are_their_font_outlines_turned_to_the_view(self, view):
        """Checked against matplotlib's outlines directly, so a label drawn mirrored or turned
        half a turn fails."""
        labels = tl.composition_labels(["Al", "Mg", "Si"], view)
        picks = [labels[0], *(lab for lab in labels if lab["kind"] != "tick")]
        for label in picks:
            font = FontProperties(family=tl.LABEL_FONT, weight=label["weight"])
            rings = TextPath((0, 0), label["text"], size=1.0, prop=font).to_polygons()
            points = np.concatenate([np.asarray(r)[:-1] for r in rings])
            cap = TextPath((0, 0), "M", size=1.0, prop=font).get_extents().ymax
            middle = [(points[:, 0].min() + points[:, 0].max()) / 2, cap / 2]
            c, s = np.cos(label["angle"]), np.sin(label["angle"])
            turn = np.array([[c, -s], [s, c]])
            expected = label["size"] * (points - middle) @ turn.T + label["center"]
            ink = _ink(label)
            gaps = np.min(np.linalg.norm(expected[:, None, :] - ink[None, :, :], axis=2), axis=1)
            assert gaps.max() < 1e-9, label["text"]

    def test_meshes_are_cached_and_read_only(self):
        vertices, faces = tl.glyph_mesh("at. % Mg")
        assert tl.glyph_mesh("at. % Mg")[0] is vertices
        with pytest.raises(ValueError):
            vertices[0, 0] = 0.0
        with pytest.raises(ValueError):
            faces[0, 0] = 0


@pytest.fixture(scope="module")
def plotter():
    p = TLIPlotter.from_components(
        ["Al", "Mg", "Si"],
        delta=0.25,
        xs_mix={k: list(v) for k, v in AMS_LD.items()},
        T_incr=100,
        param_format="linear",
    )
    p.process_data()
    return p


@pytest.fixture(scope="module")
def figure(plotter):
    return plotter.get_plot("tx")


def _floor(conds):
    lo, hi = float(conds[0]), float(conds[1])
    return lo + ts.FLOOR_INSET_FRAC * (hi - lo)


class TestTxFigureLabels:
    def test_tick_marks_run_outward_along_constant_composition(self, figure, plotter):
        trace = _named(figure, "composition ticks")
        segs = _segments(trace)
        assert len(segs) == 27
        for start, end in segs:
            frac_start, frac_end = _fractions(*start), _fractions(*end)
            assert frac_start.min() == pytest.approx(0.0, abs=1e-12)
            assert not _inside(*end)
            on_line = np.isclose(frac_start, frac_end) & ~np.isclose(frac_start, 0.0)
            assert on_line.sum() == 1, (start, end)
        z = [v for v in trace.z if v is not None]
        assert z and all(v == pytest.approx(_floor(plotter.conds)) for v in z)

    @pytest.mark.parametrize("view", tl.VIEWS)
    def test_each_label_mesh_carries_the_views_flat_labels(self, figure, plotter, view):
        labels = tl.composition_labels(plotter.components, view)
        for name, kind in KINDS.items():
            trace = _named(figure, name)
            assert trace.type == "mesh3d"
            assert all(v == pytest.approx(_floor(plotter.conds)) for v in trace.z)
            mine = [lab for lab in labels if lab["kind"] == kind]
            expected = np.round(np.concatenate([_ink(lab) for lab in mine]), 5)
            x, y = trace.meta[tl.XY_KEY][view]
            assert np.column_stack([x, y]) == pytest.approx(expected, abs=1e-12)
            faces, offset = [], 0
            for lab in mine:
                vertices, triangles = tl.glyph_mesh(lab["text"], lab["weight"])
                faces.append(triangles + offset)
                offset += len(vertices)
            drawn = np.column_stack([trace.i, trace.j, trace.k])
            assert np.array_equal(drawn, np.concatenate(faces))
            assert np.unique(drawn).size == len(x) == len(trace.z)
            assert not any(_inside(px, py) for px, py in zip(x, y))

    def test_the_label_meshes_draw_the_cameras_view(self, figure):
        tagged = [t for t in figure.data if isinstance(t.meta, dict) and tl.VIEW_KEY in t.meta]
        assert [t.name for t in tagged] == list(LABEL_TRACES)
        assert {_drawn(t) for t in tagged} == {tl.label_view(figure.layout.scene.camera)}
        assert _drawn(tagged[0]) == "right"
        assert all(t.visible is not False for t in tagged)

    def test_orient_labels_holds_up_on_z_where_plotly_does(self, figure):
        """plotly turns a scene as a turntable, with ``up`` on z, unless its dragmode says
        otherwise or its camera's ``up`` is horizontal."""
        fig = go.Figure(figure)
        rolled = _camera((0.0, -2.0, 1.0), up=(1.0, 0.0, 1.0))
        assert tl.label_view(rolled) == "left"
        assert tl.orient_labels(fig, rolled) == "bottom"
        fig.update_layout(scene_dragmode="orbit")
        assert tl.orient_labels(fig, rolled) == "left"
        fig.update_layout(scene_dragmode=None, dragmode="turntable")
        assert tl.orient_labels(fig, rolled) == "bottom"
        fig.update_layout(dragmode="orbit")
        assert tl.orient_labels(fig, rolled) == "left"
        mixed = go.Figure(fig)
        mixed.add_trace(go.Scatter(x=[0.0], y=[0.0]))
        assert tl.orient_labels(mixed, rolled) == "bottom"
        fig.update_layout(dragmode=None, template=dict(layout=dict(scene=dict(dragmode="orbit"))))
        assert tl.orient_labels(fig, rolled) == "left"
        fig.update_layout(template=dict(layout=dict(scene=dict(dragmode=None))))
        top = _camera((0.0, 0.0, 2.5), up=(-H, -0.5, 0.0))
        assert tl.orient_labels(fig, top) == "right"

    def test_orient_labels_shows_the_view_of_a_new_camera(self, figure):
        fig = go.Figure(figure)
        fig.update_layout(scene_camera_eye=dict(x=-1.9, y=0.9, z=1.0))
        assert tl.orient_labels(fig) == "left"
        for name in LABEL_TRACES:
            assert _drawn(_named(fig, name)) == "left"
            assert _drawn(_named(figure, name)) == "right"
        assert tl.orient_labels(fig, _camera((0.3, -2.0, 1.0))) == "bottom"
        assert {_drawn(_named(fig, name)) for name in LABEL_TRACES} == {"bottom"}
        assert fig.layout.scene.camera.eye.x == -1.9

    def test_the_figure_carries_the_spec_of_its_views(self, figure):
        spec = figure.layout.meta[tl.SPEC_KEY]
        assert spec["version"] == tl.SPEC_VERSION
        assert list(spec["normals"]) == list(tl.VIEWS)
        for view, normal in spec["normals"].items():
            assert normal == pytest.approx(EDGE_NORMAL[view])

    def test_titles_and_names_are_the_components(self, plotter):
        labels = tl.composition_labels(plotter.components)
        titles = {lab["text"] for lab in labels if lab["kind"] == "title"}
        names = {lab["text"] for lab in labels if lab["kind"] == "corner"}
        assert titles == {f"at. % {el}" for el in plotter.components}
        assert names == set(plotter.components)

    def test_x_and_y_share_one_scale_and_hold_every_view(self, figure):
        scene = figure.layout.scene
        (x0, x1), (y0, y1) = scene.xaxis.range, scene.yaxis.range
        assert scene.aspectmode == "manual"
        assert scene.aspectratio.y / scene.aspectratio.x == pytest.approx((y1 - y0) / (x1 - x0))
        drawn = [(t.x, t.y) for t in figure.data if t.x is not None]
        views = [
            xy for t in figure.data if t.name in LABEL_TRACES for xy in t.meta[tl.XY_KEY].values()
        ]
        assert len(views) == 9
        for x, y in drawn + views:
            xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            assert (xs[~np.isnan(xs)] > x0).all() and (xs[~np.isnan(xs)] < x1).all()
            assert (ys[~np.isnan(ys)] > y0).all() and (ys[~np.isnan(ys)] < y1).all()

    def test_decoration_is_not_hoverable_nor_in_the_legend(self, figure):
        decoration = [t for t in figure.data if t.name in ("composition ticks", *LABEL_TRACES)]
        assert len(decoration) == 4
        for trace in decoration:
            assert trace.hoverinfo == "skip"
            assert trace.showlegend is False

    def test_a_highlighted_view_keeps_one_mesh_of_each_label_kind(self, plotter):
        phase = next(p for p in plotter.color_map if p != "L")
        fig = plotter.get_plot("tx", highlight_phase=phase)
        _named(fig, "composition ticks")
        _named(fig, "liquidus")
        for name in LABEL_TRACES:
            trace = _named(fig, name)
            assert trace.visible is not False
            assert _drawn(trace) == "right"

    def test_an_interactive_legend_keeps_the_label_spec(self, plotter):
        fig = plotter.get_plot("tx", interactive_legend=True)
        assert set(fig.layout.meta) == {tl.SPEC_KEY, phase_highlight.SPEC_KEY}


class TestScripts:
    def test_post_script_attaches_the_label_views_and_the_legend(self):
        script = ts.post_script()
        assert "{plot_id}" in script
        assert tl.ATTACH_JS in script and phase_highlight.ATTACH_JS in script
        for name in ("gliquidAttachLabelViews", "gliquidAttachPhaseHighlight"):
            assert f"window.{name}(gd)" in script

    def test_dash_function_attaches_the_label_views_and_the_legend(self):
        function = ts.dash_clientside_function()
        assert function.startswith("function (figure, graphId)")
        assert tl.ATTACH_JS in function and phase_highlight.ATTACH_JS in function
        for name in ("gliquidAttachLabelViews", "gliquidAttachPhaseHighlight"):
            assert f"window.{name}(gd)" in function

    def test_label_script_reads_the_figures_keys(self):
        for token in (
            f'KEY = "{tl.SPEC_KEY}"',
            f'TAG = "{tl.VIEW_KEY}"',
            f'XY = "{tl.XY_KEY}"',
            f"VERSION = {tl.SPEC_VERSION}",
        ):
            assert token in tl.ATTACH_JS
        for script in (tl.ATTACH_JS, ts.post_script(), ts.dash_clientside_function()):
            assert re.search(r"__[A-Z_]+__", script) is None

    def test_show_runs_the_post_script_outside_a_notebook(self, plotter, monkeypatch):
        calls = []
        monkeypatch.setattr(ts, "_in_notebook_kernel", lambda: False)
        monkeypatch.setattr(go.Figure, "show", lambda fig, *a, **kw: calls.append(kw))
        plotter.show("tx")
        assert calls == [{"post_script": [ts.post_script()]}]


class TestLiquidusFigure:
    def test_the_legend_is_interactive_unless_turned_off(self, plotter, figure):
        assert isinstance(figure, ts.LiquidusFigure)
        assert phase_highlight.SPEC_KEY in figure.layout.meta
        off = plotter.get_plot("tx", interactive_legend=False)
        assert set(off.layout.meta) == {tl.SPEC_KEY}

    def test_html_runs_the_script_before_the_callers(self, figure):
        html = figure.to_html(include_plotlyjs=False, post_script="window.callerRan = true;")
        plot_id = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html).group(1)
        attach = f"document.getElementById('{plot_id}')"
        assert attach in html and "window.callerRan = true;" in html
        assert html.index(attach) < html.index("window.callerRan")

    def test_plotter_writes_a_standalone_scripted_file(self, plotter, tmp_path):
        path = tmp_path / "tx.html"
        plotter.write_html("tx", path)
        html = path.read_text(encoding="utf-8")
        assert "window.gliquidAttachPhaseHighlight(gd)" in html
        assert "{plot_id}" not in html

    def test_a_notebook_cell_draws_the_scripted_html(self, figure, monkeypatch):
        pytest.importorskip("IPython")
        import IPython.display

        shown = []
        monkeypatch.setattr(ts, "_in_notebook_kernel", lambda: True)
        monkeypatch.setattr(IPython.display, "display", shown.append)
        monkeypatch.setattr(go.Figure, "show", lambda *a, **kw: pytest.fail("plotly show ran"))
        warnings.simplefilter("error")
        figure._ipython_display_()
        figure.show(config={"scrollZoom": False})
        assert len(shown) == 2
        pages = []
        for out in shown:
            srcdoc = re.search(r'<iframe srcdoc="([^"]*)"', out.data)
            pages.append(unescape(srcdoc.group(1)))
        for page in pages:
            plot_id = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', page).group(1)
            assert f"document.getElementById('{plot_id}')" in page
            assert 'src="https://cdn.plot.ly/' in page
        assert '"scrollZoom": false' in pages[1]

    def test_copies_stay_liquidus_figures(self, figure):
        import copy
        import pickle

        for clone in (copy.deepcopy(figure), pickle.loads(pickle.dumps(figure))):
            assert isinstance(clone, ts.LiquidusFigure)
            assert clone.update_layout(title="x") is clone
