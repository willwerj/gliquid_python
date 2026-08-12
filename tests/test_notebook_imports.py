"""Static drift guard for the shipped notebooks in ``notebooks/``.

Every gliquid symbol a notebook uses is resolved against the *current* package: module
paths must import, attributes must exist on the module/class/instance they are read from,
and keyword arguments must be accepted by the callable's signature. Nothing is executed —
no data corpus, no API keys, no kernel — so this runs anywhere the package imports.

What it does NOT do is check that a notebook produces the right numbers. It catches the
one failure mode nothing else did: the package API moves, and a notebook that a new user
runs first keeps a dead call in it until someone happens to execute cell 11.

Inference is deliberately shallow. A name is tracked only when its origin is unambiguous
(a gliquid import, a constructor call, a classmethod on a gliquid class, an annotated
parameter, or a loop over a container listed in ``_ITERABLE_ELEMENTS``); everything else
is UNKNOWN and simply not checked. False negatives are fine here, false alarms are not.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import logging
from pathlib import Path

import pytest

from gliquid.binary import FIT_KWARGS, RETIRED_FIT_KWARGS, BinaryLiquid
from gliquid.phase import Phase


def _notebooks_dir() -> Path | None:
    """``notebooks/`` under the repository root, or None where it does not ship.

    Anchored by NAME rather than a parent index so moving this file between test tiers
    cannot silently point it at the wrong directory.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == "gliquid_python":
            candidate = parent / "notebooks"
            return candidate if candidate.is_dir() else None
    return None


_NB_DIR = _notebooks_dir()
_NOTEBOOKS = sorted(p.name for p in _NB_DIR.glob("*.ipynb")) if _NB_DIR else []

_needs_notebooks = pytest.mark.skipif(
    not _NOTEBOOKS, reason="notebooks/ does not ship in this installation"
)


# Callables whose **kwargs are NOT a blank cheque: the declared keyword set is the contract,
# and passing anything else raises. Without this the signature check below would wave through
# every keyword these functions accept only to ignore.
_DECLARED_KWARGS = {
    BinaryLiquid.fit_parameters: FIT_KWARGS | RETIRED_FIT_KWARGS,
}

# Element type of the gliquid containers the notebooks iterate over. These attributes carry
# no annotation, so without the map a notebook can treat a dataclass element as a dict
# (``p['name']``, ``'comp' in p``) and nothing static would notice.
_ITERABLE_ELEMENTS = {
    (BinaryLiquid, "phases"): Phase,
}

UNKNOWN = object()


def _binding_names(target: ast.expr) -> list[str]:
    """Names an assignment target actually rebinds.

    Only plain names and the names inside tuple/list unpacking. ``bl.nmpath = x`` and
    ``bl.eqs[k] = v`` mutate ``bl``; they do not rebind it, and treating them as a rebind
    would blind every check that follows to what ``bl`` is.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _binding_names(element)]
    if isinstance(target, ast.Starred):
        return _binding_names(target.value)
    return []


class _Instance:
    """Marker: the name holds an instance of ``cls`` (not the class object itself)."""

    def __init__(self, cls):
        self.cls = cls

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<instance of {self.cls.__name__}>"


# --------------------------------------------------------------------------------------
# Notebook -> code
# --------------------------------------------------------------------------------------


def _notebook_code(path: Path) -> str:
    """Concatenated code cells with IPython shell/magic lines removed."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = source.lstrip()
        if stripped.startswith("%%"):
            continue  # cell magic: the body is not Python
        lines = [line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))]
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def _parse(name: str) -> ast.Module:
    return ast.parse(_notebook_code(_NB_DIR / name), filename=name)


# --------------------------------------------------------------------------------------
# Attribute surfaces
# --------------------------------------------------------------------------------------


def _self_assigned(cls) -> set[str]:
    """Attribute names the class assigns to ``self`` anywhere in its own source."""
    try:
        tree = ast.parse(inspect.getsource(cls))
    except (OSError, TypeError, SyntaxError):  # pragma: no cover - C/builtin classes
        return set()
    found = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                ):
                    found.add(sub.attr)
    return found


def _instance_attrs(cls) -> set[str]:
    attrs = set(dir(cls))
    for klass in cls.__mro__:
        if klass is object:
            continue
        attrs |= _self_assigned(klass)
    return attrs


# --------------------------------------------------------------------------------------
# The resolver
# --------------------------------------------------------------------------------------


class _Resolver(ast.NodeVisitor):
    """Walks a notebook in document order, tracking what each name refers to."""

    def __init__(self, notebook: str):
        self.notebook = notebook
        self.env: dict[str, object] = {}
        self.import_failures: list[str] = []
        self.attr_failures: list[str] = []
        self.kwarg_failures: list[str] = []
        self.usage_failures: list[str] = []

    # -- helpers ----------------------------------------------------------------------

    def _import_module(self, dotted: str):
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:
            self.import_failures.append(f"{self.notebook}: cannot import {dotted!r} ({exc})")
            return UNKNOWN

    def _bind(self, name: str, value) -> None:
        self.env[name] = value

    def _lookup(self, node: ast.expr):
        """What ``node`` refers to, or UNKNOWN."""
        if isinstance(node, ast.Name):
            return self.env.get(node.id, UNKNOWN)
        if isinstance(node, ast.Attribute):
            base = self._lookup(node.value)
            if base is UNKNOWN:
                return UNKNOWN
            if isinstance(base, _Instance):
                return getattr(base.cls, node.attr, UNKNOWN)
            return getattr(base, node.attr, UNKNOWN)
        return UNKNOWN

    def _class_from_annotation(self, annotation):
        target = self._lookup(annotation) if annotation is not None else UNKNOWN
        return target if inspect.isclass(target) else None

    # -- bindings ---------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not alias.name.startswith("gliquid"):
                self._bind((alias.asname or alias.name).split(".")[0], UNKNOWN)
                continue
            module = self._import_module(alias.name)
            if alias.asname:
                self._bind(alias.asname, module)
            else:
                root = alias.name.split(".")[0]
                self._bind(root, self._import_module(root))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not (node.module or "").startswith("gliquid"):
            for alias in node.names:
                self._bind(alias.asname or alias.name, UNKNOWN)
            return
        module = self._import_module(node.module)
        for alias in node.names:
            bound = alias.asname or alias.name
            if module is UNKNOWN:
                self._bind(bound, UNKNOWN)
                continue
            if not hasattr(module, alias.name):
                self.import_failures.append(
                    f"{self.notebook}: {node.module}.{alias.name} does not exist"
                )
                self._bind(bound, UNKNOWN)
            else:
                self._bind(bound, getattr(module, alias.name))

    def _value_of(self, node: ast.expr):
        """What an assigned expression evaluates to, as far as we can tell."""
        if isinstance(node, ast.Call):
            func = self._lookup(node.func)
            if inspect.isclass(func) and func.__module__.startswith("gliquid"):
                return _Instance(func)
            # A classmethod on a gliquid class is an alternate constructor.
            if isinstance(node.func, ast.Attribute):
                owner = self._lookup(node.func.value)
                if (
                    inspect.isclass(owner)
                    and owner.__module__.startswith("gliquid")
                    and isinstance(inspect.getattr_static(owner, node.func.attr, None), classmethod)
                ):
                    return _Instance(owner)
            return UNKNOWN
        return self._lookup(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._value_of(node.value)
        for target in node.targets:
            names = _binding_names(target)
            if isinstance(target, ast.Name):
                self._bind(target.id, value)
            else:
                for name in names:
                    self._bind(name, UNKNOWN)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name, UNKNOWN)
        saved = dict(self.env)  # parameters are local; they must not leak past the body
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            cls = self._class_from_annotation(arg.annotation)
            self._bind(arg.arg, _Instance(cls) if cls is not None else UNKNOWN)
        for statement in node.body:
            self.visit(statement)
        self.env = saved

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name, UNKNOWN)
        for statement in node.body:
            self.visit(statement)

    def _element_type(self, iterable: ast.expr):
        """Element type of a mapped gliquid container, else None."""
        if isinstance(iterable, ast.Attribute):
            base = self._lookup(iterable.value)
            if isinstance(base, _Instance):
                return _ITERABLE_ELEMENTS.get((base.cls, iterable.attr))
        return None

    def _bind_loop_target(self, target: ast.expr, iterable: ast.expr) -> None:
        element = self._element_type(iterable)
        if isinstance(target, ast.Name):
            self._bind(target.id, _Instance(element) if element else UNKNOWN)
        else:
            for name in _binding_names(target):
                self._bind(name, UNKNOWN)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_loop_target(node.target, node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def _visit_comprehension(self, node, value_fields) -> None:
        saved = dict(self.env)  # comprehension targets are local to the comprehension
        for generator in node.generators:
            self.visit(generator.iter)
            self._bind_loop_target(generator.target, generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for field in value_fields:
            self.visit(getattr(node, field))
        self.env = saved

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, ("elt",))

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, ("key", "value"))

    # -- checks -----------------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load):
            return  # a write may legitimately create a new attribute
        base = self._lookup(node.value)
        if base is UNKNOWN:
            return
        if isinstance(base, _Instance):
            if node.attr not in _instance_attrs(base.cls):
                self.attr_failures.append(
                    f"{self.notebook}: {base.cls.__name__} instance has no attribute {node.attr!r}"
                )
        elif inspect.ismodule(base) or inspect.isclass(base):
            if not hasattr(base, node.attr):
                label = getattr(base, "__name__", repr(base))
                self.attr_failures.append(
                    f"{self.notebook}: {label} has no attribute {node.attr!r}"
                )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)
        base = self._lookup(node.value)
        if isinstance(base, _Instance) and not hasattr(base.cls, "__getitem__"):
            self.usage_failures.append(
                f"{self.notebook}: {base.cls.__name__} is not subscriptable (indexed like a dict)"
            )

    def visit_Compare(self, node: ast.Compare) -> None:
        self.generic_visit(node)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            base = self._lookup(comparator)
            if (
                isinstance(base, _Instance)
                and not hasattr(base.cls, "__contains__")
                and not hasattr(base.cls, "__iter__")
            ):
                self.usage_failures.append(
                    f"{self.notebook}: {base.cls.__name__} does not support 'in' "
                    f"(membership-tested like a dict)"
                )

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        keywords = [kw.arg for kw in node.keywords if kw.arg]
        if not keywords:
            return
        func = self._lookup(node.func)
        if func is UNKNOWN or not (inspect.isroutine(func) or inspect.isclass(func)):
            return
        module = getattr(func, "__module__", "") or ""
        if not module.startswith("gliquid"):
            return
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - unsupported callable
            return
        named = {
            name
            for name, p in signature.parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        var_keyword = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
        declared = _DECLARED_KWARGS.get(func)
        allowed = named | (declared or set())
        if var_keyword and declared is None:
            return  # genuinely open-ended
        for keyword in keywords:
            if keyword not in allowed:
                self.kwarg_failures.append(
                    f"{self.notebook}: {func.__qualname__}() does not accept keyword {keyword!r}"
                )


def _resolve(name: str) -> _Resolver:
    resolver = _Resolver(name)
    resolver.visit(_parse(name))
    return resolver


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@_needs_notebooks
@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_notebook_code_parses(notebook):
    """Every code cell is valid Python once shell/magic lines are stripped."""
    _parse(notebook)


@_needs_notebooks
@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_notebook_gliquid_imports_resolve(notebook):
    """Every ``import gliquid...`` / ``from gliquid... import x`` still exists."""
    assert _resolve(notebook).import_failures == []


@_needs_notebooks
@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_notebook_gliquid_attributes_exist(notebook):
    """Every attribute read off a gliquid module, class or instance still exists."""
    assert _resolve(notebook).attr_failures == []


@_needs_notebooks
@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_notebook_gliquid_call_kwargs_accepted(notebook):
    """Every keyword passed to a gliquid callable is one that callable accepts."""
    assert _resolve(notebook).kwarg_failures == []


@_needs_notebooks
@pytest.mark.parametrize("notebook", _NOTEBOOKS)
def test_notebook_gliquid_objects_used_as_objects(notebook):
    """gliquid dataclasses are not indexed or membership-tested like dicts."""
    assert _resolve(notebook).usage_failures == []


# The kwarg allow-list above is only a guard if the package enforces the same list at
# runtime; these two pin that contract. A bare instance is enough because the check runs
# before fit_parameters touches any state.


def test_unknown_fit_kwarg_raises_typeerror():
    bl = BinaryLiquid.__new__(BinaryLiquid)
    with pytest.raises(TypeError) as excinfo:
        bl.fit_parameters(nonsense_kwarg=1, params_init=[0, 0])
    message = str(excinfo.value)
    assert "nonsense_kwarg" in message and "params_init" in message


def test_retired_fit_kwarg_logs_and_continues(caplog):
    bl = BinaryLiquid.__new__(BinaryLiquid)
    with caplog.at_level(logging.WARNING, logger="gliquid.binary"):
        # Past the kwarg gate, the bare instance is what stops it -- not a TypeError.
        with pytest.raises(AttributeError):
            bl.fit_parameters(check_phase_mismatch=False)
    assert "'check_phase_mismatch' is retired" in caplog.text
