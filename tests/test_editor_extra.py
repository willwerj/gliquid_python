"""The optional `editor` extra is guarded, and says so by name (OQ46 item 3).

ipywidgets/IPython back ConvexHullEditor only. They are NOT base dependencies, but
hull_editor.py imported them at module top while gliquid/__init__.py exposed
ConvexHullEditor through its lazy __getattr__ -- so on a bare `pip install gliquid`
merely naming the export raised `ModuleNotFoundError: No module named 'ipywidgets'`,
with nothing pointing at the remedy.

The contract now: without the extra the module still imports and every lazy export
still resolves; CONSTRUCTING an editor is what fails, with a message naming
`pip install gliquid[editor]`.

The end-to-end check runs in a subprocess that blocks both modules at the import
system, since this test session has them installed and unloading them mid-run would
strand the other hull_editor tests.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gliquid.hull_editor import _EDITOR_EXTRA_HINT, _MissingEditorDep, _require_editor_deps

EXTRA = "pip install gliquid[editor]"

# Blocks ipywidgets/IPython for the whole subprocess by raising from a meta-path
# finder, which is what a bare install looks like from inside hull_editor.
_BARE_INSTALL = f"""
import sys

BLOCKED = ("ipywidgets", "IPython")


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}")
        return None


for _mod in [m for m in list(sys.modules) if m.split(".")[0] in BLOCKED]:
    del sys.modules[_mod]
sys.meta_path.insert(0, _Block())

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


def test_bare_install_resolves_exports_and_names_the_extra():
    """No ipywidgets/IPython: exports resolve, construction names `gliquid[editor]`."""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _BARE_INSTALL],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_hint_names_the_extra():
    assert EXTRA in _EDITOR_EXTRA_HINT


def test_require_editor_deps_passes_when_installed():
    """Where the extra IS installed, the guard must be a no-op.

    Skipped rather than failed where it is not: the compat matrix installs only
    ``extras = test``, so ipywidgets is genuinely absent there and the guard raising is
    correct behaviour, not a regression. Asserting unconditionally made all eight tox
    environments red for a reason that had nothing to do with the package.
    """
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
