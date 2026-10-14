"""``show_figure`` displays a Matplotlib figure under the inline backend Jupyter and Colab use.

Under ``matplotlib_inline`` the figure goes to ``IPython.display.display`` once and is closed,
so the backend's end-of-cell flush does not draw it a second time. Every other backend gets
``Figure.show()``.
"""

import warnings

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from gliquid.plotting import export  # noqa: E402

from ._mpds_builders import close_figures  # noqa: F401

INLINE = "module://matplotlib_inline.backend_inline"


@pytest.fixture
def displayed(monkeypatch):
    shown = []
    ipython_display = pytest.importorskip("IPython.display")
    monkeypatch.setattr(ipython_display, "display", lambda obj: shown.append(obj))
    return shown


def test_inline_backend_displays_the_figure_once_and_closes_it(monkeypatch, displayed):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: INLINE)
    fig, _ = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        export.show_figure(fig)
    assert displayed == [fig]
    assert not plt.fignum_exists(fig.number)


def test_other_backends_call_figure_show(monkeypatch, displayed):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    calls = []
    fig, _ = plt.subplots()
    monkeypatch.setattr(fig, "show", lambda: calls.append(fig))
    export.show_figure(fig)
    assert calls == [fig]
    assert displayed == []
    assert plt.fignum_exists(fig.number)
