"""The binary solidus lines of the ternary liquidus figure and the script that draws them."""

from gliquid.plotting import solidus_view, ternary_surface
from gliquid.plotting.style import SOLIDUS_LINE_COLOR


def test_the_trace_draws_each_line_and_carries_what_the_script_reads():
    solidus = [([[0.0, 0.0], [0.5, 0.0]], 500.0), ([[1.0, 0.0], [0.75, 0.4]], 300.0)]
    (trace,) = solidus_view.traces(solidus)

    assert list(trace.x) == [0.0, 0.5, None, 1.0, 0.75]
    assert list(trace.y) == [0.0, 0.0, None, 0.0, 0.4]
    assert list(trace.z) == [500.0, 500.0, None, 300.0, 300.0]
    assert trace.line.color == SOLIDUS_LINE_COLOR
    lines = trace.meta[solidus_view.META_KEY]["lines"]
    assert [line["T"] for line in lines] == [500.0, 300.0]
    assert lines[1]["xy"] == [[1.0, 0.0], [0.75, 0.4]]
    assert lines[0]["text"] == trace.text[0] == "<b>Binary solidus</b><br>T: 500.0°C"


def test_both_figure_scripts_attach_the_solidus_view():
    for script in (ternary_surface.post_script(), ternary_surface.dash_clientside_function()):
        assert "window.gliquidAttachSolidusView = function" in script
        assert "window.gliquidAttachSolidusView(gd);" in script
