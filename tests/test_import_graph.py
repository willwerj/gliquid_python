"""Import direction inside ``gliquid.imputation`` and between it, ``gliquid.ternary`` and
``gliquid.mpds``.

Checked in a fresh interpreter per module, on the checkout's ``src``, from an empty working
directory. ``gliquid.ternary``, ``gliquid.imputation``, every ``gliquid.imputation`` submodule,
``gliquid.mpds`` and ``gliquid.solid_coverage`` each import, print nothing, leave the
directory empty and load none of the modules ``FORBIDDEN`` lists for them:
  * ``gliquid.imputation`` loads none of its submodules;
  * ``gliquid.imputation.core`` loads neither ``gliquid.binary`` nor ``gliquid.ternary``;
  * ``gliquid.imputation.lifecycle`` does not load ``gliquid.ternary``;
  * ``gliquid.mpds`` does not load ``gliquid.solid_coverage``;
  * ``gliquid.solid_coverage`` loads none of ``gliquid.binary``, ``gliquid.imputation`` and
    ``gliquid.ternary``.
The removed module paths (``REMOVED``) raise ``ModuleNotFoundError``.

Checked on the source of every module in the ``gliquid.imputation`` package directory:
  * none imports ``gliquid.ternary`` at module level;
  * ``core`` imports neither ``gliquid.binary`` nor ``gliquid.ternary`` at any level;
  * neither ``binary_imputation`` nor ``ternary_imputation`` imports
    ``gliquid.imputation.lifecycle`` at any level;
  * ``gliquid.mpds`` imports ``gliquid.solid_coverage`` at no level;
  * ``gliquid.solid_coverage`` imports none of ``gliquid.binary``, ``gliquid.imputation*``,
    ``gliquid.ternary`` and ``gliquid.plotting*`` at any level.

The AST checker counts an import as module level unless it sits inside a function or lambda;
``if`` / ``try`` / class bodies count as module level. ``from P import n`` imports ``P`` and
``P.n``; relative imports resolve against the importing module's package.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from ._fresh import run_fresh

PACKAGE = "gliquid.imputation"
CORE = "gliquid.imputation.core"
BINARY_IMPUTATION = "gliquid.imputation.binary_imputation"
TERNARY_IMPUTATION = "gliquid.imputation.ternary_imputation"
LIFECYCLE = "gliquid.imputation.lifecycle"
REMOVED = [
    "gliquid.imputation_core",
    "gliquid.phase_imputation",
    "gliquid.ternary_imputation",
    "gliquid.phase_ablation",
    "gliquid.imputation.binary",
    "gliquid.imputation.ternary",
]
MODEL_MODULES = {"gliquid.binary", "gliquid.ternary"}
TERNARY = "gliquid.ternary"
MPDS = "gliquid.mpds"
SOLID_COVERAGE = "gliquid.solid_coverage"
#: Module-name prefixes each target must not load when it is the first gliquid import.
FORBIDDEN = {
    PACKAGE: (PACKAGE + ".",),
    CORE: tuple(sorted(MODEL_MODULES)),
    LIFECYCLE: (TERNARY,),
    MPDS: (SOLID_COVERAGE,),
    SOLID_COVERAGE: ("gliquid.binary", PACKAGE, TERNARY),
}


def _origin(name: str) -> str | None:
    """The file ``name`` resolves to in the imported tree, or ``None`` when it does not exist."""
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    return None if spec is None else spec.origin


def _package_of(name: str) -> str:
    origin = _origin(name) or ""
    return name if Path(origin).name == "__init__.py" else name.rpartition(".")[0]


def _family() -> list[str]:
    """``gliquid.imputation`` and one module name per ``.py`` file in its directory."""
    origin = _origin(PACKAGE)
    assert origin is not None, f"{PACKAGE} is not importable from the imported tree"
    names = [PACKAGE]
    for path in sorted(Path(origin).parent.glob("*.py")):
        if path.name != "__init__.py":
            names.append(f"{PACKAGE}.{path.stem}")
    return names


def import_records(source: str, package: str) -> list[tuple[str, bool, str | None]]:
    """``(module, at_module_level, enclosing_function)`` for every module ``source`` imports."""
    records: list[tuple[str, bool, str | None]] = []

    def resolve(node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = package.split(".")
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def visit(node, top: bool, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top, function = False, node.name
        elif isinstance(node, ast.Lambda):
            top, function = False, "<lambda>"
        if isinstance(node, ast.Import):
            records.extend((alias.name, top, function) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = resolve(node)
            records.append((module, top, function))
            records.extend((f"{module}.{alias.name}", top, function) for alias in node.names)
        for child in ast.iter_child_nodes(node):
            visit(child, top, function)

    visit(ast.parse(source), True, None)
    return records


def _records_of(name: str) -> list[tuple[str, bool, str | None]]:
    return import_records(Path(_origin(name)).read_text(encoding="utf-8"), _package_of(name))


def _imports(name: str, target: str, module_level_only: bool) -> bool:
    return any(
        module == target and (top or not module_level_only)
        for module, top, _function in _records_of(name)
    )


CHECKER_CASES = [
    ("import gliquid.ternary\n", "gliquid", True),
    ("def f():\n    import gliquid.ternary\n", "gliquid", False),
    ("from gliquid import ternary\n", "gliquid", True),
    ("from gliquid.imputation import ternary\n", "gliquid", False),
    ("from .. import ternary\n", "gliquid.imputation", True),
]


@pytest.mark.parametrize(
    ("source", "package", "hits"),
    CHECKER_CASES,
    ids=["top-level", "in-function", "from-parent", "from-sibling-package", "relative"],
)
def test_the_checker(source, package, hits):
    """The checker's own controls: which synthetic imports count as a module-level ternary import."""
    records = import_records(source, package)
    assert any(m == TERNARY and top for m, top, _f in records) is hits


@pytest.mark.parametrize("module", [TERNARY, *_family(), MPDS, SOLID_COVERAGE])
def test_fresh_import(module, tmp_path):
    """``module`` imports first in a fresh interpreter, prints nothing, writes nothing to the
    working directory and loads none of its ``FORBIDDEN`` prefixes."""
    result = run_fresh(
        f"import json, sys, {module}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('gliquid'))))\n",
        tmp_path,
        env={"GLIQUID_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert list(tmp_path.iterdir()) == []
    lines = result.stdout.splitlines()
    assert len(lines) == 1, result.stdout
    loaded = json.loads(lines[0])
    assert module in loaded
    assert [m for m in loaded if m.startswith(FORBIDDEN.get(module, ()))] == []


def test_the_removed_module_paths_are_gone(tmp_path):
    code = (
        "import importlib, json\n"
        "out = {}\n"
        f"for name in {REMOVED!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        out[name] = 'imported'\n"
        "    except ModuleNotFoundError as exc:\n"
        "        out[name] = exc.name\n"
        "print(json.dumps(out))\n"
    )
    result = run_fresh(code, tmp_path, env={"GLIQUID_OFFLINE": "1"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {name: name for name in REMOVED}


def test_no_imputation_module_imports_gliquid_ternary_at_module_level():
    importers = {name for name in _family() if _imports(name, TERNARY, True)}
    assert importers == set()


def test_imputation_core_imports_no_model_module():
    imported = {module for module, _top, _f in _records_of(CORE)}
    assert not (imported & MODEL_MODULES)


def test_binary_and_ternary_do_not_import_lifecycle():
    for name in (BINARY_IMPUTATION, TERNARY_IMPUTATION):
        assert not _imports(name, LIFECYCLE, False), name


def test_mpds_does_not_import_solid_coverage():
    assert not _imports(MPDS, SOLID_COVERAGE, False)


def test_solid_coverage_imports_no_model_imputation_or_plotting_module():
    imported = {module for module, _top, _f in _records_of(SOLID_COVERAGE)}
    assert MPDS in imported
    forbidden = {
        module
        for module in imported
        if module in MODEL_MODULES or module.startswith((PACKAGE, "gliquid.plotting"))
    }
    assert forbidden == set()
