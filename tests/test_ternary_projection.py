"""Tests for gliquid.plotting.ternary_projection on synthetic liquidus records."""

import inspect
import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from gliquid.plotting import ternary_projection as tp  # noqa: E402

COMPONENTS = ["A", "B", "C"]
ORDER = ["A", "B", "C"]
WINDOW = (0.15, 0.6, 0.05, 0.41)


def _record(delta=0.025):
    """A full-triangle record with three primary fields, one of them imputed."""
    n = int(round(1.0 / delta))
    nodes = {"x0": [], "x1": [], "T_K": [], "primary": []}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            x0, x1 = round(i * delta, 6), round(j * delta, 6)
            if x0 >= 0.4 and x0 >= x1:
                phase = "Beta"
            elif x1 >= 0.4:
                phase = "Gamma"
            else:
                phase = "Alpha"
            nodes["x0"].append(x0)
            nodes["x1"].append(x1)
            nodes["T_K"].append(1000.0 + 400.0 * x0 + 150.0 * x1)
            nodes["primary"].append(phase)
    return {"nodes": nodes, "delta": delta, "imputed_phase_names": ["Beta"]}


def _overlapping(boxes):
    return [(a, b) for a, b in itertools.combinations(boxes, 2) if tp.boxes_overlap(a, b)]


@pytest.fixture
def record():
    return _record()


@pytest.mark.parametrize("ticks", [True, False])
def test_a_full_panel_renders(record, ticks):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    contour = tp.draw_liquidus_panel(
        ax, record, COMPONENTS, ORDER, tp.levels_for([record]), ticks=ticks
    )
    fig.canvas.draw()
    texts = {t.get_text() for t in ax.texts}
    low = ax.get_ylim()[0]
    plt.close(fig)
    assert contour is not None
    assert {"Alpha", "Beta*", "Gamma", "A", "B", "C"} <= texts
    titles = {"at.% A", "at.% B", "at.% C"}
    if ticks:
        assert titles <= texts and "50" in texts
        assert low == pytest.approx(-0.17)
    else:
        assert not titles & texts
        assert low == pytest.approx(-0.12)


@pytest.mark.parametrize("ticks", [True, False])
def test_a_closeup_panel_renders(record, ticks):
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_axes([0.14, 0.16, 0.72, 0.72])
    levels = tp.window_levels([record], COMPONENTS, ORDER, WINDOW)
    contour, labels = tp.draw_closeup_panel(
        ax, record, COMPONENTS, ORDER, levels, WINDOW, ticks=ticks
    )
    fig.canvas.draw()
    texts = [t.get_text() for t in ax.texts]
    limits = ax.get_xlim() + ax.get_ylim()
    plt.close(fig)
    assert contour is not None
    assert {p for _, _, p in labels} == {"Alpha", "Beta", "Gamma"}
    assert limits == pytest.approx(WINDOW)
    assert any(t.startswith("at.% ") for t in texts) is ticks


def test_field_labels_do_not_overlap():
    delta = 0.05
    n = int(round(1 / delta))
    lattice = {
        (i, j): f"PhaseLongName{min(3, i // 5)}" for i in range(n + 1) for j in range(n + 1 - i)
    }
    placed = tp.place_field_labels(lattice, delta, COMPONENTS, ORDER)
    assert len(placed) == 4
    assert not _overlapping([tp.label_box(x, y, p) for x, y, p in placed])


def test_closeup_labels_do_not_overlap(record):
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_axes([0.14, 0.16, 0.72, 0.72])
    levels = tp.window_levels([record], COMPONENTS, ORDER, WINDOW)
    _contour, labels = tp.draw_closeup_panel(
        ax, record, COMPONENTS, ORDER, levels, WINDOW, ticks=False
    )
    names = {p + ("*" if p == "Beta" else "") for _, _, p in labels}
    fields = [t for t in ax.texts if t.get_text() in names]
    isotherms = [t for t in ax.texts if t.get_text().isdigit() and t not in fields]
    field_boxes = tp.artist_boxes(ax, fields)
    isotherm_boxes = tp.artist_boxes(ax, isotherms)
    plt.close(fig)
    assert len(fields) == 3 and len(isotherms) >= 4
    assert not _overlapping(field_boxes)
    assert not any(tp.boxes_overlap(a, b) for a in field_boxes for b in isotherm_boxes)


SIGNATURES = {
    "draw_liquidus_panel": "(ax, record: 'dict', sorted_components, order, levels: 'dict', "
    "imputed=None, fonts: 'dict | None' = None, colours: 'dict | None' = None, "
    "frame: 'bool' = True, isotherm_labels: 'bool' = True, highlight: 'dict | None' = None, "
    "ticks: 'bool' = True)",
    "draw_closeup_panel": "(ax, record: 'dict', sorted_components, order, levels: 'dict', "
    "window, imputed=None, fonts: 'dict | None' = None, colours: 'dict | None' = None, "
    "label_style: 'str' = 'halo', isotherm_spacing: 'float' = 0.5, ticks: 'bool' = True)",
    "levels_for": "(records, n_fill: 'int' = 25) -> 'dict'",
    "window_levels": "(records, sorted_components, order, window, n_fill: 'int' = 25, "
    "target: 'int' = 12, steps: 'tuple' = (2, 5, 10, 20, 25, 50, 100, 200), "
    "pad: 'float' = 0.03) -> 'dict'",
    "place_field_labels": "(lattice: 'dict', delta: 'float', sorted_components, order, "
    "imputed=frozenset(), min_nodes: 'int' = 12, priority=frozenset(), window=None, "
    "avoid=(), box=None, lines=None) -> 'list'",
    "triangle_frame": "(ax, order, fontsize: 'float' = 17, ticks: 'bool' = True, "
    "tick_fontsize: 'float' = 10)",
    "draw_window_frame": "(ax, order, window, fontsize: 'float' = 9.0, step: 'float' = 0.1, "
    "number_sides=('bottom', 'left', 'right'), title_sides=('bottom', 'left', 'right'), "
    "length_pt: 'float' = 6.0, colour: 'str' = '#3d3c39') -> 'None'",
    "draw_locator": "(ax, order, window, fontsize: 'float' = 9, colour: 'str' = '#b3261e') "
    "-> 'None'",
    "monovariant_polylines": "(nodes: 'dict', delta: 'float', sorted_components, order, "
    "window: 'int' = 7, rounds: 'int' = 4) -> 'list'",
    "primary_lattice": "(nodes: 'dict', delta: 'float') -> 'dict'",
}


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_public_signatures(name):
    assert str(inspect.signature(getattr(tp, name))) == SIGNATURES[name]
