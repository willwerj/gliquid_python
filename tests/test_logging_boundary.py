"""The print/logging boundary, as an enforced contract rather than a convention.

The library reports through ``logging`` (one ``logging.getLogger(__name__)`` per module,
all under the ``gliquid`` parent), and configures no handlers of its own -- handlers belong
to the application. ``dev/scripts/Fit_Binary_Systems.py::_attach_gliquid_log_handler`` is
the driver-side half, deliberately attached AFTER its stdout tee/queue swap so library
records land in fit logs exactly where the prints used to.

Three ``print()`` sites survive that conversion on purpose, each carrying an in-code
"documented logging exemption" comment:

  * ``mpds.print_phase_mismatch_chart`` -- the function *is* a console chart renderer; the
    aligned monospace rows are the product, not diagnostics.
  * ``hull_editor._log`` -- writes into an ``ipywidgets`` Output pane. Widget UI.
  * ``cache._emit`` -- the single output site of ``python -m gliquid.cache``. A CLI's
    stdout is its product, and the alternative is worse: a library may not call
    ``basicConfig``/``addHandler`` (see ``TestLibraryConfiguresNoLogging`` below), so a
    ``logger`` call there would print nothing at all under the default configuration.
    Library code in that module still reports through ``logger``; only the CLI section
    below the marked divider uses ``_emit``.

``EXEMPT`` below is deliberately a short, explicit list: adding a fourth exemption requires
editing this file, which is the point. These tests scan the SOURCE TREE, not the import
graph, so a print added inside a branch no test happens to execute is still caught.
"""

import ast
from pathlib import Path

import pytest

import gliquid

# Prefer the repo's src/ layout; fall back to the imported package (installed checkout).
_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "gliquid"
SRC = _REPO_SRC if _REPO_SRC.is_dir() else Path(gliquid.__file__).resolve().parent

# (file name, enclosing function) pairs allowed to call print(). Keep this SHORT.
EXEMPT = {
    ("mpds.py", "print_phase_mismatch_chart"),
    ("hull_editor.py", "_log"),
    ("cache.py", "_emit"),
}

# Names that would mean the library configured logging for its application -- the thing a
# library must never do. Matched on the AST, so prose mentioning them is fine.
HANDLER_CONFIG_NAMES = {
    "basicConfig",
    "addHandler",
    "removeHandler",
    "setLevel",
    "StreamHandler",
    "FileHandler",
    "NullHandler",
}


def _source_files():
    files = sorted(SRC.rglob("*.py"))
    assert files, f"no package sources found under {SRC}"
    return files


class _CallScanner(ast.NodeVisitor):
    """Collect ``print(...)`` and direct ``sys.stdout/stderr.write(...)`` calls with scope."""

    def __init__(self):
        self.scope: list[str] = []
        self.prints: list[tuple[int, str]] = []  # (lineno, enclosing function or '<module>')
        self.stream_writes: list[tuple[int, str]] = []

    def _enclosing(self) -> str:
        return self.scope[-1] if self.scope else "<module>"

    def _visit_scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped
    visit_ClassDef = _visit_scoped

    def visit_Call(self, node):
        func = node.func
        # A bare print(...) -- NOT pprint(...), NOT obj.print(...), and never a string or
        # comment that merely contains the text, since those are not Call nodes at all.
        if isinstance(func, ast.Name) and func.id == "print":
            self.prints.append((node.lineno, self._enclosing()))
        # The obvious way around a print ban.
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "write"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr in ("stdout", "stderr")
        ):
            self.stream_writes.append((node.lineno, self._enclosing()))
        self.generic_visit(node)


def _scan(path: Path) -> _CallScanner:
    scanner = _CallScanner()
    scanner.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return scanner


def _census() -> dict[tuple[str, str], list[str]]:
    """{(file name, function): ['<relpath>:<lineno>', ...]} for every print() in the tree."""
    found: dict[tuple[str, str], list[str]] = {}
    for path in _source_files():
        for lineno, func in _scan(path).prints:
            key = (path.name, func)
            found.setdefault(key, []).append(f"{path.relative_to(SRC).as_posix()}:{lineno}")
    return found


class TestPrintBoundary:
    def test_no_bare_prints_outside_the_two_exemptions(self):
        offenders = sorted(
            loc for key, locs in _census().items() if key not in EXEMPT for loc in locs
        )
        assert not offenders, (
            "print() in library code -- use the module logger "
            "(logging.getLogger(__name__)) instead:\n  "
            + "\n  ".join(offenders)
            + "\n\nIf this really is console-rendering or widget UI rather than a "
            "diagnostic, add it to EXEMPT in this file and say why in the code."
        )

    def test_every_exemption_is_still_exercised(self):
        """A stale exemption is a hole: it silently re-authorizes a future print there."""
        census = _census()
        stale = sorted(key for key in EXEMPT if key not in census)
        assert not stale, (
            f"these exemptions no longer match any print() and should be dropped: {stale}"
        )

    def test_no_direct_stdout_writes(self):
        offenders = sorted(
            f"{path.relative_to(SRC).as_posix()}:{lineno} (in {func})"
            for path in _source_files()
            for lineno, func in _scan(path).stream_writes
        )
        assert not offenders, (
            "sys.stdout/stderr.write in library code bypasses logging just as print does:\n  "
            + "\n  ".join(offenders)
        )


class TestLibraryConfiguresNoLogging:
    """A library attaches no handlers and sets no levels; the application does.

    ``Fit_Binary_Systems`` attaches a StreamHandler to the ``gliquid`` logger itself, and
    a basicConfig here would race that (and any other consumer's) configuration.
    """

    def test_no_handler_or_level_configuration_in_the_library(self):
        offenders = []
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                name = (
                    node.attr
                    if isinstance(node, ast.Attribute)
                    else node.id
                    if isinstance(node, ast.Name)
                    else None
                )
                if name in HANDLER_CONFIG_NAMES:
                    offenders.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno} ({name})")
        assert not offenders, (
            "logging handler/level configuration inside the library:\n  "
            + "\n  ".join(sorted(offenders))
        )


class TestModuleLoggersAreConventional:
    """Every module that logs does so through ``logger = logging.getLogger(__name__)``."""

    @pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
    def test_logger_is_named_logger_and_bound_to_dunder_name(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            if not (isinstance(func, ast.Attribute) and func.attr == "getLogger"):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            assert targets == ["logger"], (
                f"{path.name}:{node.lineno} names its logger {targets}; the package "
                f"convention is `logger = logging.getLogger(__name__)`"
            )
            args = node.value.args
            assert len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "__name__", (
                f"{path.name}:{node.lineno} must use getLogger(__name__) so every module "
                f"logger is a child of the `gliquid` logger a consumer configures"
            )
