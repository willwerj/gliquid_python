"""The shipped notebooks load the model ``ProductionModelRunner()`` loads by default.

Notebook code cells are read as JSON and parsed; nothing is executed. Live code is what
``ast`` sees once shell/magic lines are dropped, so a commented-out line never counts as
live, and no notebook keeps a commented-out ``pred_params = [...]`` line from the
``20260329_022905`` model.
"""

from __future__ import annotations

import ast
import json
import re

import pytest

from ._mpds_builders import code_cells, shipped_notebooks

LEGACY_BUNDLE_ID = "20260329_022905"
RUNNER_NOTEBOOKS = ["GaRu_adjustment.ipynb", "PbTb_adjustment.ipynb", "colab_demo.ipynb"]

LIVE_LEGACY_PATTERNS = (re.escape(LEGACY_BUNDLE_ID), r"\bbundle_dir\b")
LIVE_DATA_DIR_PATTERNS = (r"\bset_data_dir\b", r"\bcfg\.data_dir\b")
LEGACY_LINE = re.compile(
    r"^\s*#\s*pred_params\s*=\s*(\[[^\]]*\])\s*#\s*" + re.escape(LEGACY_BUNDLE_ID) + r" model\s*$"
)


_NOTEBOOK_PATHS = {p.name: p for p in shipped_notebooks()}
_NOTEBOOKS = sorted(_NOTEBOOK_PATHS)

pytestmark = pytest.mark.skipif(
    not _NOTEBOOKS, reason="notebooks/ does not ship in this installation"
)


def _code_cells(name: str) -> list[str]:
    return code_cells(json.loads(_NOTEBOOK_PATHS[name].read_text(encoding="utf-8")))


def live_code(source: str) -> str:
    """``source`` with comments dropped: shell/magic lines removed, then round-tripped
    through ``ast``."""
    lines = ["" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()]
    return ast.unparse(ast.parse("\n".join(lines)))


def live_hits(source: str, patterns) -> list[str]:
    code = live_code(source)
    return [pattern for pattern in patterns if re.search(pattern, code)]


def legacy_lines(source: str) -> list[list]:
    """The parameter lists of every commented-out ``pred_params = [...]`` legacy line."""
    return [
        ast.literal_eval(match.group(1))
        for line in source.splitlines()
        if (match := LEGACY_LINE.match(line))
    ]


def bare_runner_calls(source: str) -> int:
    """Live ``ProductionModelRunner()`` calls with no arguments."""
    count = 0
    for node in ast.walk(ast.parse(live_code(source))):
        if isinstance(node, ast.Call) and not node.args and not node.keywords:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            count += name == "ProductionModelRunner"
    return count


@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_no_live_legacy_bundle_reference(notebook):
    hits = [hit for cell in _code_cells(notebook) for hit in live_hits(cell, LIVE_LEGACY_PATTERNS)]
    assert hits == []


@pytest.mark.parametrize("notebook", RUNNER_NOTEBOOKS)
def test_runner_notebook_has_no_live_data_dir(notebook):
    hits = [
        hit for cell in _code_cells(notebook) for hit in live_hits(cell, LIVE_DATA_DIR_PATTERNS)
    ]
    assert hits == []


@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_no_legacy_parameter_lines(notebook):
    assert [params for cell in _code_cells(notebook) for params in legacy_lines(cell)] == []


@pytest.mark.parametrize("notebook", RUNNER_NOTEBOOKS)
def test_runner_is_constructed_with_no_arguments(notebook):
    assert sum(bare_runner_calls(cell) for cell in _code_cells(notebook)) >= 1


def test_the_scan_reads_live_code_and_skips_comments():
    source = (
        "runner = ProductionModelRunner()\n"
        "bundle_dir = 'x'\n"
        f"# path = '{LEGACY_BUNDLE_ID}'\n"
        "!pip install -q .[ml]\n"
        "cfg.set_cache_dir('y')  # set_data_dir in a comment\n"
        "# pred_params = [1.0, -2.5, 3, 0]  # 20260329_022905 model\n"
    )
    assert live_hits(source, LIVE_LEGACY_PATTERNS) == [r"\bbundle_dir\b"]
    assert live_hits(source, LIVE_DATA_DIR_PATTERNS) == []
    assert legacy_lines(source) == [[1.0, -2.5, 3, 0]]
    assert bare_runner_calls(source) == 1
