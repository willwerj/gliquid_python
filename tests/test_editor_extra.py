"""The editor's widget stack (the `notebook` extra) is guarded, and says so by name.

ipywidgets/IPython back ConvexHullEditor only, and are NOT base dependencies. Without the
extra the module still imports and every lazy export still resolves; CONSTRUCTING an
editor is what fails, with a message naming `pip install gliquid[notebook]`.

The end-to-end check runs in a subprocess that blocks both modules at the import
system.
"""

import pytest

from gliquid.hull_editor import _EDITOR_EXTRA_HINT, _MissingEditorDep, _require_editor_deps

from ._fresh import ok, run_fresh

EXTRA = "pip install gliquid[notebook]"

# Runs with ipywidgets and IPython blocked, which is what a bare install looks like from
# inside hull_editor.
_BARE_INSTALL = f"""
import gliquid

# 1. The regression: every lazy export must still resolve without the extra.
for _name in gliquid.__all__:
    getattr(gliquid, _name)

# 2. Constructing must fail with a message naming the extra. The dependency check
#    runs ahead of source validation, so the argument here is irrelevant.
try:
    gliquid.ConvexHullEditor(None)
except ImportError as exc:
    assert {EXTRA!r} in str(exc), f"message names no remedy: {{exc}}"
else:
    raise AssertionError("ConvexHullEditor(None) did not raise without ipywidgets")

print("OK")
"""


def test_bare_install_resolves_exports_and_names_the_extra(tmp_path):
    """No ipywidgets/IPython: exports resolve, construction names `gliquid[notebook]`."""
    ok(run_fresh(_BARE_INSTALL, tmp_path, blocked=("ipywidgets", "IPython")))


def test_hint_names_the_extra():
    assert EXTRA in _EDITOR_EXTRA_HINT


def test_require_editor_deps_passes_when_installed():
    """Where the extra IS installed, the guard is a no-op; skipped where it is not."""
    pytest.importorskip("ipywidgets", reason="the editor extra is not installed here")
    pytest.importorskip("IPython", reason="the editor extra is not installed here")
    _require_editor_deps()


def test_placeholder_attribute_access_names_the_extra():
    """A stray `widgets.X` reports the remedy instead of an opaque AttributeError."""
    placeholder = _MissingEditorDep("ipywidgets")
    try:
        _ = placeholder.HTML
    except ImportError as exc:
        assert EXTRA in str(exc)
        assert "ipywidgets" in str(exc)
    else:
        raise AssertionError("attribute access on the placeholder did not raise")


def test_placeholder_call_names_the_extra():
    """`display(...)` is called, not attribute-accessed, so the placeholder is callable."""
    placeholder = _MissingEditorDep("IPython")
    try:
        placeholder("widget")
    except ImportError as exc:
        assert EXTRA in str(exc)
        assert "IPython" in str(exc)
    else:
        raise AssertionError("calling the placeholder did not raise")
