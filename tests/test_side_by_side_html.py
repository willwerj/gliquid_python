"""``side_by_side_html`` draws Plotly figures in equal columns from one HTML fragment.

Each column is an iframe of the figure's standalone page, loading plotly.js from the CDN, so the
fragment runs no script in the notebook page itself.
"""

import re
from html import unescape

import plotly.graph_objects as go
import pytest
from plotly.offline import get_plotlyjs_version

from gliquid.plotting.export import side_by_side_html


def _figure(name):
    return go.Figure(go.Scatter(x=[0, 1], y=[1, 0], name=name))


def _pages(html):
    return [unescape(page) for page in re.findall(r'<iframe srcdoc="([^"]*)"', html)]


def _plot_id(page):
    return re.search(r'<div id="([^"]+)" class="plotly-graph-div"', page).group(1)


def test_one_column_per_figure_in_order():
    html = side_by_side_html([_figure("left"), _figure("right")], height="400px")
    pages = _pages(html)
    assert len(pages) == 2
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in html
    assert html.count("height:400px") == 2
    assert '"name":"left"' in pages[0] and '"name":"right"' in pages[1]
    url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    assert all(f'src="{url}"' in page for page in pages)


def test_configs_merge_over_responsive():
    html = side_by_side_html([_figure("a"), _figure("b")], configs=[None, {"scrollZoom": False}])
    first, second = _pages(html)
    assert '{"responsive": true}' in first
    assert '{"responsive": true, "scrollZoom": false}' in second


def test_a_post_script_runs_against_its_own_figure():
    script = "window.attached = document.getElementById('{plot_id}');"
    first, second = _pages(
        side_by_side_html([_figure("a"), _figure("b")], post_scripts=[script, None])
    )
    assert f"window.attached = document.getElementById('{_plot_id(first)}');" in first
    assert "window.attached" not in second


def test_no_markup_of_the_data_reaches_the_notebook_page():
    html = side_by_side_html([_figure("</script><b>x</b>")])
    assert "<script" not in html and "<b>" not in html
    assert len(_pages(html)) == 1


def test_a_list_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match="one entry per figure"):
        side_by_side_html([_figure("a"), _figure("b")], configs=[None])
    with pytest.raises(ValueError, match="one entry per figure"):
        side_by_side_html([_figure("a")], post_scripts=[None, None])
