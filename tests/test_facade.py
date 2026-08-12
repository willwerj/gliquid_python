"""API façade contract (PEP-562 lazy exports in gliquid/__init__.py).

The façade IS the public surface: every name in __all__ must resolve through
`import gliquid`, and the bare import must stay light (no matplotlib/plotly).
Any change to __all__ is a breaking change needing explicit sign-off.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import gliquid


class TestFacadeExports:
    def test_every_all_name_resolves(self):
        for name in gliquid.__all__:
            assert getattr(gliquid, name) is not None, name

    def test_exports_match_declared_surface(self):
        assert set(gliquid.__all__) == {*gliquid._EXPORTS, "__version__"}

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            gliquid.not_a_real_export  # noqa: B018

    def test_version_is_nonempty_string(self):
        assert isinstance(gliquid.__version__, str) and gliquid.__version__

    def test_dir_lists_exports(self):
        listing = dir(gliquid)
        assert "BinaryLiquid" in listing and "set_data_dir" in listing


def test_import_gliquid_is_lazy():
    """A bare `import gliquid` must not drag in matplotlib or plotly."""
    code = (
        "import gliquid, sys; "
        "heavy = [m for m in ('matplotlib', 'plotly') if m in sys.modules]; "
        "assert not heavy, f'heavy imports leaked: {heavy}'"
    )
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
