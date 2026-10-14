"""Every shipped notebook installs one pinned GLiquid on Colab, and all pin the same version.

``colab_demo.ipynb`` runs its setup cell as is; the other notebooks carry the same lines
commented out under ``## UNCOMMENT THIS BLOCK FOR GOOGLE COLAB``. Each installs the package
by version, with no extras, and clones the repository only for its ``cache/`` data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from packaging.version import Version

from ._mpds_builders import code_cells, shipped_notebooks

_NOTEBOOKS = shipped_notebooks()

pytestmark = pytest.mark.skipif(
    not _NOTEBOOKS, reason="notebooks/ does not ship in this installation"
)

_INSTALL = re.compile(r"^(#\s*)?!pip install\b.*$", re.MULTILINE)
_PIN = re.compile(r'"gliquid==([^"\[\]]+)"')
_CLONE = re.compile(
    r"^(#\s*)?!git clone --depth 1 https://github\.com/willwerj/gliquid_python\.git"
)


def _code(path: Path) -> list[str]:
    return code_cells(json.loads(path.read_text(encoding="utf-8")))


def _install_lines(path: Path) -> list[str]:
    return [m.group(0) for source in _code(path) for m in _INSTALL.finditer(source)]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.stem)
def test_one_pinned_install_without_extras(path):
    (line,) = _install_lines(path)
    (version,) = _PIN.findall(line)
    Version(version)
    assert "gliquid[" not in line
    assert (line.lstrip().startswith("#")) is (path.stem != "colab_demo")


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.stem)
def test_the_data_comes_from_a_shallow_clone(path):
    assert any(_CLONE.search(line) for source in _code(path) for line in source.splitlines())


def test_every_notebook_pins_the_same_version():
    pins = {path.stem: _PIN.findall(_install_lines(path)[0])[0] for path in _NOTEBOOKS}
    assert len(set(pins.values())) == 1, pins
