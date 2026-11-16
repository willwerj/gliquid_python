"""The solidus traces of the ternary liquidus figure and the script that draws them."""

from gliquid.plotting import solidus_view, ternary_surface
from gliquid.plotting.style import SOLIDUS_FILL_OPACITY


def test_the_fill_is_transparent_and_both_traces_carry_what_the_script_reads():
    solidus = [
        (("A", "B", "C"), [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]], 500.0),
        (("A", "C", "D"), [[0.0, 0.0], [0.5, 0.8], [0.2, 0.6]], 300.0),
    ]
    colours = {"A": "#FF0000", "B": "rgb(0, 128, 255)", "C": "Silver", "D": "#000000"}
    fill, outline = solidus_view.traces(solidus, colours, str.lower)

    spec = fill.meta[solidus_view.META_KEY]
    assert spec["role"] == "fill"
    assert spec["colors"][:3] == ["255,0,0", "0,128,255", "192,192,192"]
    assert spec["opacity"] == SOLIDUS_FILL_OPACITY
    assert list(fill.vertexcolor) == [f"rgba({colour},0)" for colour in spec["colors"]]

    spec = outline.meta[solidus_view.META_KEY]
    assert spec["role"] == "outline"
    assert [triangle["T"] for triangle in spec["triangles"]] == [500.0, 300.0]
    assert spec["triangles"][1]["xy"] == [[0.0, 0.0], [0.5, 0.8], [0.2, 0.6]]
    assert spec["triangles"][0]["text"] == fill.text[0] == outline.text[0]
    assert fill.text[0] == "<b>Solidus</b><br>a + b + c<br>T: 500.0°C"


def test_both_figure_scripts_attach_the_solidus_view():
    for script in (ternary_surface.post_script(), ternary_surface.dash_clientside_function()):
        assert "window.gliquidAttachSolidusView = function" in script
        assert "window.gliquidAttachSolidusView(gd);" in script
