"""The shipped runner returns real parameter values for the notebook systems.

No test here asserts what the values are. :func:`literal_comparisons` scans ``tests/`` and
``tests_internal/`` for a comparison between a named system's ``predict_system`` output and
a numeric literal, and must find none.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Element

pytest.importorskip("xgboost")

from gliquid.binary import BinaryLiquid  # noqa: E402
from gliquid.production_model_runner import ProductionModelRunner  # noqa: E402

from ._mpds_builders import hermetic_mpds  # noqa: E402

SYSTEMS = ("Ga-Ru", "Pb-Tb", "Cu-Mg")
TEST_ROOTS = ("tests", "tests_internal")


@pytest.fixture(scope="module")
def runner():
    return ProductionModelRunner()


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_shipped_runner_returns_real_values(runner, system):
    values = runner.predict_system(system)
    assert len(values) == len(runner.targets)
    for value in values:
        assert isinstance(value, (float, np.floating))
        assert not isinstance(value, (bool, np.bool_))
        assert np.isfinite(value)
    assert runner.feature_source.startswith("bundle:")


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_values_draw_a_finite_liquidus(runner, system):
    values = runner.predict_system(system)
    with hermetic_mpds():
        bl = BinaryLiquid.from_cache(system, param_format="comb-exp")
        bl.update_params(values + [0])
    points = bl.phases[-1].points
    assert len(points) > 0
    assert np.all(np.isfinite([point[1] for point in points]))


# -- No test compares a named system's predict_system output with a numeric literal --

_ELEMENTS = {element.symbol for element in Element}
_CLOSE_CALLS = ("allclose", "isclose", "array_equal", "array_almost_equal")
_LITERAL_WRAPPERS = ("approx", "array", "asarray")


def _names_a_system(value) -> bool:
    parts = value.split("-") if isinstance(value, str) else []
    return len(parts) > 1 and all(part in _ELEMENTS for part in parts)


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _tail(node) -> str:
    return _dotted(node).split(".")[-1]


def _is_numeric_literal(node) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_numeric_literal(node.operand)
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts) and all(_is_numeric_literal(e) for e in node.elts)
    if isinstance(node, ast.Call) and node.args and _tail(node.func) in _LITERAL_WRAPPERS:
        return _is_numeric_literal(node.args[0])
    return False


class _Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.strings: dict[str, list[str]] = {}
        self.tainted: set[str] = set()

    def strings_of(self, name):
        scope = self
        while scope is not None:
            if name in scope.strings:
                return scope.strings[name]
            scope = scope.parent
        return []

    def is_tainted(self, name):
        scope = self
        while scope is not None:
            if name in scope.tainted:
                return True
            scope = scope.parent
        return False


def _string_values(node, scope) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [v for e in node.elts for v in _string_values(e, scope)]
    if isinstance(node, ast.Name):
        return scope.strings_of(node.id)
    return []


def _parametrized_strings(fn, scope) -> dict[str, list[str]]:
    """{argname: string values} from ``pytest.mark.parametrize`` decorators on ``fn``."""
    out: dict[str, list[str]] = {}
    for dec in fn.decorator_list:
        if not (isinstance(dec, ast.Call) and _dotted(dec.func).endswith("parametrize")):
            continue
        if len(dec.args) < 2 or not isinstance(dec.args[0], ast.Constant):
            continue
        names = [n.strip() for n in str(dec.args[0].value).split(",")]
        values = dec.args[1]
        if len(names) == 1:
            out[names[0]] = _string_values(values, scope)
        elif isinstance(values, (ast.List, ast.Tuple)):
            for row in values.elts:
                if isinstance(row, (ast.List, ast.Tuple)):
                    for name, item in zip(names, row.elts):
                        out.setdefault(name, []).extend(_string_values(item, scope))
    return out


class _Scanner(ast.NodeVisitor):
    def __init__(self):
        self.scope = _Scope()
        self.hits: list[int] = []

    def _derives(self, node) -> bool:
        """``node`` is, or derives by name from, a predict_system call on a named system."""
        if isinstance(node, ast.Call) and _dotted(node.func) == "len":
            return False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and self.scope.is_tainted(sub.id):
                return True
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "predict_system"
                and sub.args
                and any(_names_a_system(v) for v in _string_values(sub.args[0], self.scope))
            ):
                return True
        return False

    def _flag(self, a, b, lineno):
        if (self._derives(a) and _is_numeric_literal(b)) or (
            self._derives(b) and _is_numeric_literal(a)
        ):
            self.hits.append(lineno)

    def visit_FunctionDef(self, node):
        outer = self.scope
        self.scope = _Scope(outer)
        self.scope.strings.update(_parametrized_strings(node, outer))
        for stmt in node.body:
            self.visit(stmt)
        self.scope = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        self.generic_visit(node)
        strings = _string_values(node.value, self.scope)
        derived = self._derives(node.value)
        for target in node.targets:
            for name in (n.id for n in ast.walk(target) if isinstance(n, ast.Name)):
                if strings:
                    self.scope.strings[name] = strings
                if derived:
                    self.scope.tainted.add(name)

    def visit_For(self, node):
        if self._derives(node.iter):
            for name in (n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)):
                self.scope.tainted.add(name)
        strings = _string_values(node.iter, self.scope)
        if strings:
            for name in (n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)):
                self.scope.strings[name] = strings
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            operands = [node.left, *node.comparators]
            for a, b in zip(operands, operands[1:]):
                self._flag(a, b, node.lineno)

    def visit_Call(self, node):
        self.generic_visit(node)
        name = _dotted(node.func)
        tail = name.split(".")[-1]
        if len(node.args) >= 2 and (
            tail in _CLOSE_CALLS or tail.startswith("assert_") or ".testing." in f".{name}."
        ):
            self._flag(node.args[0], node.args[1], node.lineno)


def literal_comparisons(source: str) -> list[int]:
    """Line numbers of every comparison between a named system's ``predict_system`` output
    and a numeric literal in ``source``."""
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    return sorted(set(scanner.hits))


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_test_pins_predict_system_output_to_a_literal():
    root = _package_root()
    files = [p for sub in TEST_ROOTS for p in sorted((root / sub).rglob("*.py"))]
    assert any(p.name == "test_portable_model_bundle.py" for p in files)
    found = {
        p.relative_to(root).as_posix(): hits
        for p in files
        if (hits := literal_comparisons(p.read_text(encoding="utf-8")))
    }
    assert found == {}


def test_the_scan_flags_a_planted_literal_comparison():
    planted = (
        "import numpy as np\n"
        "import pytest\n"
        "\n"
        "SYSTEM = 'Cu-Mg'\n"
        "\n"
        "def test_planted(runner):\n"
        "    values = runner.predict_system('Cu-Mg')\n"
        "    assert values == [-42972.5, -6.8, -9145.3]\n"
        "    assert values[0] == pytest.approx(-42972.5)\n"
        "    np.testing.assert_allclose(runner.predict_system(SYSTEM), [1.0, 2.0, 3.0])\n"
        "    assert len(values) == 3\n"
        "    assert values == runner.predict_system('Cu-Mg')\n"
        "\n"
        "@pytest.mark.parametrize('system', ['Pb-Tb', 'Ga-Ru'])\n"
        "def test_planted_parametrized(runner, system):\n"
        "    assert np.allclose(runner.predict_system(system), (1.0, -2.0, 3.0))\n"
    )
    assert literal_comparisons(planted) == [8, 9, 10, 16]
