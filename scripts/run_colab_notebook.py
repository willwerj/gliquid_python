"""Execute one notebook the way Google Colab runs it, in a Colab-like Linux container.

Usage::

    python scripts/run_colab_notebook.py                          # colab_demo at HEAD
    python scripts/run_colab_notebook.py --notebook fitting_demo --uncomment-colab
    python scripts/run_colab_notebook.py --ref v0.3.0 --freeze saved/colab-pip-freeze.txt
    python scripts/run_colab_notebook.py --notebook notebooks/colab_demo.ipynb   # working copy
    python scripts/run_colab_notebook.py --pin-from-ref HEAD     # version for a gliquid== pin

What a run does:

1. Reads Colab's published runtime package set (``pip-freeze.txt``) and OS/Python version
   (``os-info.txt``) from the ``googlecolab/backend-info`` repository, or from ``--freeze`` /
   ``--os-info``. Both are saved into the output directory.
2. Builds a ``python:<major.minor>-slim`` image with ``git`` added (``--image`` overrides the
   base) and runs the notebook there with the working directory ``/content``.
3. ``git clone``\\ s this repository at ``--ref`` (checked out as ``main``; every other branch
   kept) into a scratch directory and builds a wheel from that clone. The clone is mounted
   read-only at ``/src`` and the wheel directory at ``/wheels``.
4. Resolves the dependency closure of the notebook's own ``pip install`` lines with
   ``pip install --dry-run --report``, intersects it with the Colab freeze, and installs that
   intersection at Colab's pinned versions (plus Colab's pip, setuptools and notebook tooling)
   before the notebook starts. :data:`MODULE_ALIASES` adds the Colab distribution that provides
   the same import package under another name (Colab's ``xgboost`` for ``xgboost-cpu``).
   ``--closure-extras`` (default ``ml``) adds gliquid extras to that resolution only, so Colab's
   copies of gliquid's optional dependencies are present even when the notebook installs
   gliquid without the extra.
5. Rewrites the notebook's shell magics, leaving every other line verbatim:
   ``!git clone https://github.com/willwerj/gliquid_python.git`` clones ``file:///src`` with the
   same flags; ``!pip install gliquid[...]==VERSION`` gets ``--find-links /wheels`` when a local
   wheel of VERSION exists (a ``X.Y.Z.devN+g<sha>`` pin builds that commit first) and is left
   for PyPI otherwise; ``git+`` URLs of the GitHub repository point at ``/src``.
   ``--uncomment-colab`` activates the ``## UNCOMMENT THIS BLOCK FOR GOOGLE COLAB`` cell.
6. Executes the notebook with nbclient inside the container, stopping at the first failing
   cell unless ``--allow-errors``. No API key is injected.
7. Writes ``report.json``, ``report.txt`` and the executed notebook to the output directory
   (``--out``, default a new directory under ``~/.cache/gliquid-colab-harness/runs/``).

The report lists per-cell status, elapsed time, stderr and warnings; every Colab-preinstalled
distribution the notebook's install changed ("Colab runtime-restart hazard"); every import
package provided by two installed distributions ("module collision"); and the cache records
and HTTP requests the kernel attempted.

``--notebook NAME`` reads ``notebooks/NAME.ipynb`` from the ``--ref`` checkout;
``--notebook PATH`` reads that file instead, so a working copy can be run against a committed
source tree.

Exit status is 0 when every code cell ran without error and no module collision was found,
1 when a cell failed or a collision was found, and 2 for a usage or infrastructure error
(docker unavailable, notebook missing, clone or build failure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

DEFAULT_NOTEBOOK = "colab_demo"
DEFAULT_TIMEOUT = 900
DEFAULT_CLOSURE_EXTRAS = "ml"

COLAB_FREEZE_URL = "https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.txt"
COLAB_OS_INFO_URL = "https://raw.githubusercontent.com/googlecolab/backend-info/main/os-info.txt"

FREEZE_FILE = "colab-pip-freeze.txt"
OS_INFO_FILE = "colab-os-info.txt"

CONTAINER_SRC = "/src"
CONTAINER_WHEELS = "/wheels"
CONTAINER_OUT = "/out"
CONTAINER_HARNESS = "/harness"
CONTENT_DIR = "/content"
SOURCE_URL = f"file://{CONTAINER_SRC}"

IMAGE_PREFIX = "gliquid-colab-harness"
PIP_CACHE_VOLUME = "gliquid-colab-pip-cache"

COLAB_BANNER = "## UNCOMMENT THIS BLOCK FOR GOOGLE COLAB"

# Distributions that install the same import package under a different distribution name,
# mapped to the Colab distribution names they stand in for.
MODULE_ALIASES: dict[str, tuple[str, ...]] = {
    "xgboost-cpu": ("xgboost",),
}

# Installed at Colab's pinned version before the notebook runs.
NOTEBOOK_TOOLS = (
    "pip",
    "setuptools",
    "wheel",
    "ipykernel",
    "ipython",
    "ipywidgets",
    "matplotlib-inline",
    "traitlets",
    "jupyter-client",
    "jupyter-core",
    "nbformat",
    "nbclient",
)

# The subset of NOTEBOOK_TOOLS the executor cannot run without.
EXECUTOR_TOOLS = ("ipykernel", "jupyter-client", "nbformat", "nbclient")

_GITHUB_REPO_URL = (
    r"(?:git\+)?(?:https?://|git@|ssh://git@)github\.com[/:]willwerj/gliquid_python(?:\.git)?/?"
)
_GITHUB_REPO = re.compile("^" + _GITHUB_REPO_URL, re.IGNORECASE)
_GITHUB_REPO_ANY = re.compile(_GITHUB_REPO_URL, re.IGNORECASE)
_GIT_CLONE = re.compile(r"^(?P<prefix>\s*!\s*)git\s+clone\b(?P<args>.*)$")
_PIP_INSTALL = re.compile(
    r"^(?P<prefix>\s*(?:!\s*|%)(?:python3?\s+-m\s+)?pip3?\s+install)\b(?P<args>.*)$"
)
_GLIQUID_REQ = re.compile(
    r"^gliquid(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>(?:[=<>!~;@].*)?)$", re.IGNORECASE
)
_CHECKOUT_REQ = re.compile(r"^\.(?:/)?(?:\[(?P<extras>[^\]]*)\])?$")
_DEV_VERSION = re.compile(
    r"^(?P<base>\d+(?:\.\d+)*)\.dev(?P<distance>\d+)\+g(?P<sha>[0-9a-f]{7,40})"
    r"(?P<dirty>\.d\d{8})?$"
)
_WHEEL_NAME = re.compile(r"^gliquid-(?P<version>[^-]+)-.*\.whl$")
_PYTHON_LINE = re.compile(r"^Python\s+(\d+)\.(\d+)(?:\.(\d+))?", re.MULTILINE)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PROGRESS_BAR = re.compile(r"\d+%\||\d+it/s\]|\ds/it\]")
_WARNING_LINE = re.compile(
    r"\b\w*Warning\b:|^\s*(?:WARNING|ERROR|DEPRECATION)\b:?|^\s*(?:warning|fatal|error):",
)

# git clone options that consume the following token.
_CLONE_VALUE_OPTS = frozenset(
    {
        "-b",
        "--branch",
        "--depth",
        "-o",
        "--origin",
        "-c",
        "--config",
        "-j",
        "--jobs",
        "--filter",
        "--reference",
        "--reference-if-able",
        "--separate-git-dir",
        "--template",
        "-u",
        "--upload-pack",
        "--shallow-since",
        "--shallow-exclude",
        "--server-option",
        "--bundle-uri",
        "--ref-format",
        "--revision",
    }
)

# pip install options that consume the following token.
_PIP_VALUE_OPTS = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-t",
        "--target",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
        "--root",
        "--prefix",
        "--src",
        "--upgrade-strategy",
        "--global-option",
        "-C",
        "--config-settings",
        "--no-binary",
        "--only-binary",
        "--progress-bar",
        "--root-user-action",
        "--report",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "--trusted-host",
        "--proxy",
        "--retries",
        "--timeout",
        "--exists-action",
        "--log",
        "--cert",
        "--client-cert",
        "--cache-dir",
        "--keyring-provider",
        "--python",
        "--use-feature",
        "--use-deprecated",
    }
)


class InfraError(RuntimeError):
    """A usage or infrastructure failure; maps to exit status 2."""


# -- pure helpers ------------------------------------------------------------------------


class FreezeEntry(NamedTuple):
    """One ``pip freeze`` line: the distribution name as written, its version, the raw line."""

    name: str
    version: str | None
    line: str


def normalize_name(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(text: str) -> dict[str, FreezeEntry]:
    """``{normalised name: FreezeEntry}`` for every requirement line of a ``pip freeze``.

    ``name==version`` lines carry a version; ``name @ url`` lines carry None. Comments,
    blank lines and ``-e`` lines are skipped.
    """
    entries: dict[str, FreezeEntry] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if " @ " in line:
            name = line.split(" @ ", 1)[0].strip()
            version = None
        elif "==" in line:
            name, version = (part.strip() for part in line.split("==", 1))
        else:
            continue
        name = name.split("[", 1)[0]
        entries[normalize_name(name)] = FreezeEntry(name, version, line)
    return entries


def parse_python_version(os_info: str) -> tuple[int, int, int | None] | None:
    """``(major, minor, micro)`` from Colab's ``os-info.txt`` (``Python 3.13.15``), or None."""
    match = _PYTHON_LINE.search(os_info)
    if match is None:
        return None
    micro = int(match.group(3)) if match.group(3) is not None else None
    return int(match.group(1)), int(match.group(2)), micro


def constraints_from_freeze(freeze: dict[str, FreezeEntry]) -> list[str]:
    """``name==version`` constraint lines for every pinned freeze entry, sorted by name."""
    return sorted(
        f"{entry.name}=={entry.version}" for entry in freeze.values() if entry.version is not None
    )


def parse_local_dev_version(version: str) -> tuple[str, bool] | None:
    """``(sha, dirty)`` for a hatch-vcs ``X.Y.Z.devN+g<sha>[.dYYYYMMDD]`` version, else None."""
    match = _DEV_VERSION.match(version.strip())
    if match is None:
        return None
    return match.group("sha"), match.group("dirty") is not None


def wheel_version(filename: str) -> str | None:
    """The version field of a ``gliquid-<version>-*.whl`` filename, or None."""
    match = _WHEEL_NAME.match(Path(filename).name)
    return match.group("version") if match else None


def _split_extras(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(sorted({part.strip() for part in text.split(",") if part.strip()}))


def _comment_start(line: str) -> int:
    """Index of an unquoted shell comment (``#`` after whitespace) in ``line``, or len(line)."""
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return index
    return len(line)


def _positionals(tokens: list[str], value_opts: frozenset[str]) -> list[tuple[int, str]]:
    """``(index, token)`` of every positional argument, skipping options and their values."""
    found = []
    skip = False
    for index, token in enumerate(tokens):
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            if token in value_opts:
                skip = True
            continue
        found.append((index, token))
    return found


def _option_value(tokens: list[str], names: tuple[str, ...]) -> str | None:
    for index, token in enumerate(tokens):
        if token in names and index + 1 < len(tokens):
            return tokens[index + 1]
        for name in names:
            if name.startswith("--") and token.startswith(name + "="):
                return token.split("=", 1)[1]
    return None


class CloneRewrite(NamedTuple):
    """A rewritten ``!git clone`` line and the branch it asks for (None for the default)."""

    line: str
    branch: str | None
    directory: str


def rewrite_clone_line(line: str, source_url: str = SOURCE_URL) -> CloneRewrite | None:
    """Point a ``!git clone`` of the GitHub repository at ``source_url``.

    Every flag is kept. When the line names no destination directory, the GitHub URL's
    directory name (``gliquid_python``) is appended, since cloning ``source_url`` would
    otherwise name the directory after it. Returns None for any other line.
    """
    match = _GIT_CLONE.match(line)
    if match is None:
        return None
    body_end = _comment_start(line)
    try:
        tokens = shlex.split(line[match.start("args") : body_end])
    except ValueError:
        return None
    positionals = _positionals(tokens, _CLONE_VALUE_OPTS)
    if not positionals or not _GITHUB_REPO.match(positionals[0][1]):
        return None
    url = positionals[0][1]
    directory = positionals[1][1] if len(positionals) > 1 else None
    new = line.replace(url, source_url, 1)
    if directory is None:
        directory = "gliquid_python"
        cut = _comment_start(new)
        new = f"{new[:cut].rstrip()} {directory}" + (f"  {new[cut:]}" if cut < len(new) else "")
    branch = _option_value(tokens, ("-b", "--branch"))
    return CloneRewrite(new, branch, directory)


class Requirement(NamedTuple):
    """One requirement token of a ``pip install`` line.

    ``kind`` is ``checkout`` (``.`` / ``.[extras]``), ``gliquid`` (a ``gliquid`` requirement,
    ``version`` set when pinned with ``==``), ``gliquid-git`` (a ``git+`` URL of the GitHub
    repository) or ``other``.
    """

    token: str
    kind: str
    extras: tuple[str, ...]
    version: str | None


def classify_requirement(token: str) -> Requirement:
    """Classify one ``pip install`` requirement token."""
    checkout = _CHECKOUT_REQ.match(token)
    if checkout:
        return Requirement(token, "checkout", _split_extras(checkout.group("extras")), None)
    if _GITHUB_REPO.match(token):
        return Requirement(token, "gliquid-git", (), None)
    gliquid = _GLIQUID_REQ.match(token)
    if gliquid:
        extras = _split_extras(gliquid.group("extras"))
        spec = gliquid.group("spec").strip()
        if spec.startswith("@"):
            url = spec[1:].strip()
            kind = "gliquid-git" if _GITHUB_REPO.match(url) else "other"
            return Requirement(token, kind, extras, None)
        pinned = re.fullmatch(r"==\s*([^\s,;]+)", spec)
        return Requirement(token, "gliquid", extras, pinned.group(1) if pinned else None)
    return Requirement(token, "other", (), None)


def parse_pip_line(line: str) -> list[Requirement] | None:
    """The requirements of a ``!pip install`` / ``%pip install`` line, or None for other lines."""
    match = _PIP_INSTALL.match(line)
    if match is None:
        return None
    try:
        tokens = shlex.split(line[match.start("args") : _comment_start(line)])
    except ValueError:
        return None
    requirements = []
    skip = False
    for index, token in enumerate(tokens):
        if skip:
            skip = False
            continue
        if token in ("-e", "--editable") and index + 1 < len(tokens):
            requirements.append(classify_requirement(tokens[index + 1]))
            skip = True
            continue
        if token.startswith("-"):
            skip = token in _PIP_VALUE_OPTS
            continue
        requirements.append(classify_requirement(token))
    return requirements


def gliquid_pin_action(version: str, built_versions: set[str] | frozenset[str]) -> str:
    """How a ``gliquid==version`` pin is served.

    ``local-wheel`` when a wheel of exactly that version was built; ``build-sha`` when the
    version is a hatch-vcs ``X.Y.Z.devN+g<sha>`` version whose commit still needs a wheel;
    ``pypi`` otherwise (the line is left for the package index).
    """
    if version in built_versions:
        return "local-wheel"
    if parse_local_dev_version(version) is not None:
        return "build-sha"
    return "pypi"


def rewrite_pip_line(
    line: str,
    built_versions: set[str] | frozenset[str],
    find_links: str = CONTAINER_WHEELS,
    source_url: str = SOURCE_URL,
) -> tuple[str, list[str]]:
    """Rewrite one ``pip install`` line for the local source. Returns ``(line, notes)``.

    A pinned gliquid requirement whose version is in ``built_versions`` gets
    ``--find-links <find_links>``; a ``git+`` URL of the GitHub repository is pointed at
    ``source_url``; ``.`` / ``.[extras]`` and everything else is left as written.
    """
    requirements = parse_pip_line(line)
    if requirements is None:
        return line, []
    notes = []
    new = line
    add_find_links = False
    for req in requirements:
        if req.kind == "checkout":
            notes.append(f"{req.token}: installs the /content clone (unchanged)")
        elif req.kind == "gliquid-git":
            repo = _GITHUB_REPO_ANY.search(req.token)
            local = "git+" + source_url
            new = new.replace(
                req.token, req.token[: repo.start()] + local + req.token[repo.end() :]
            )
            notes.append(f"{req.token}: GitHub URL redirected to {source_url}")
        elif req.kind == "gliquid" and req.version is None:
            notes.append(f"{req.token}: unpinned, resolved from PyPI (unchanged)")
        elif req.kind == "gliquid":
            action = gliquid_pin_action(req.version, built_versions)
            if action == "local-wheel":
                add_find_links = True
                notes.append(f"{req.token}: local wheel {req.version} via --find-links")
            else:
                notes.append(f"{req.token}: no local wheel of {req.version}; PyPI (unchanged)")
            if "+" in req.version:
                notes.append(
                    f"{req.version} has a local version label: PyPI cannot host it, so on "
                    "Colab this line needs its own --find-links/--index-url"
                )
    if add_find_links:
        cut = _comment_start(new)
        tail = f"  {new[cut:]}" if cut < len(new) else ""
        new = f"{new[:cut].rstrip()} --find-links {find_links}{tail}"
    return new, notes


def closure_specs(
    requirements: list[Requirement], ref_version: str, add_extras: tuple[str, ...] = ()
) -> list[str]:
    """Requirement strings for the dry-run that resolves a notebook's install closure.

    The local checkout and ``git+`` installs resolve as ``gliquid[extras]==ref_version``;
    every gliquid requirement also gets ``add_extras``; other requirements pass through.
    """
    specs = []
    for req in requirements:
        extras = sorted(set(req.extras) | set(add_extras))
        bracket = f"[{','.join(extras)}]" if extras else ""
        if req.kind in ("checkout", "gliquid-git"):
            specs.append(f"gliquid{bracket}=={ref_version}")
        elif req.kind == "gliquid":
            spec = _GLIQUID_REQ.match(req.token).group("spec").strip()
            specs.append(f"gliquid{bracket}{spec}")
        else:
            specs.append(req.token)
    return list(dict.fromkeys(specs))


def uncomment_colab_block(source: str, banner: str = COLAB_BANNER) -> str | None:
    """The cell source with its Colab block uncommented, or None if it is not that cell.

    The cell qualifies when its first non-blank line starts with ``banner``. Each later line
    loses one leading ``# `` (a bare ``#`` becomes empty); the banner line is kept.
    """
    lines = source.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or not lines[first].strip().startswith(banner):
        return None
    out = lines[: first + 1]
    for line in lines[first + 1 :]:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.rstrip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def select_preinstall(
    closure: list[str],
    freeze: dict[str, FreezeEntry],
    aliases: dict[str, tuple[str, ...]] = MODULE_ALIASES,
    tools: tuple[str, ...] = NOTEBOOK_TOOLS,
) -> tuple[list[str], list[str], list[str]]:
    """What to install before the notebook runs so the container starts as Colab does.

    Returns ``(pins, absent_tools, unpinnable)``: ``name==version`` for every closure
    distribution (and alias) that Colab ships, plus every tool Colab ships; the tools Colab
    does not ship (installed unpinned); and closure distributions Colab ships only as a
    direct URL.
    """
    wanted: dict[str, None] = {}
    for name in closure:
        key = normalize_name(name)
        wanted[key] = None
        for alias in aliases.get(key, ()):
            wanted[normalize_name(alias)] = None
    pins: dict[str, str] = {}
    unpinnable = []
    for key in wanted:
        entry = freeze.get(key)
        if entry is None:
            continue
        if entry.version is None:
            unpinnable.append(entry.name)
        else:
            pins[key] = f"{entry.name}=={entry.version}"
    absent_tools = []
    for tool in tools:
        key = normalize_name(tool)
        entry = freeze.get(key)
        if entry is not None and entry.version is not None:
            pins[key] = f"{entry.name}=={entry.version}"
        elif entry is None:
            absent_tools.append(tool)
    return sorted(pins.values(), key=str.lower), absent_tools, sorted(unpinnable)


def provided_modules(record_paths: list[str]) -> list[str]:
    """Top-level import names a distribution provides, from its ``RECORD`` paths.

    A name counts when the distribution ships ``<name>/__init__.py``, ``<name>.py`` or a
    top-level extension module. Namespace packages (no ``__init__.py``) do not count.
    """
    modules = set()
    for raw in record_paths:
        path = raw.replace("\\", "/")
        if path.startswith(("/", "../")):
            continue
        parts = path.split("/")
        head = parts[0]
        if head.endswith((".dist-info", ".data", ".egg-info")) or head == "__pycache__":
            continue
        if len(parts) == 1:
            if head.endswith(".py") and head != "__init__.py":
                modules.add(head[:-3])
            elif head.endswith((".so", ".pyd")):
                modules.add(head.split(".", 1)[0])
        elif len(parts) == 2 and parts[1] == "__init__.py":
            modules.add(head)
    return sorted(modules)


def find_collisions(dist_modules: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """``(module, [distributions])`` for every top-level module more than one dist provides."""
    providers: dict[str, set[str]] = {}
    for dist, modules in dist_modules.items():
        for module in modules:
            providers.setdefault(module, set()).add(dist)
    return sorted((module, sorted(dists)) for module, dists in providers.items() if len(dists) > 1)


def _version_direction(before: str | None, after: str | None) -> str:
    try:
        from packaging.version import Version

        if Version(after) > Version(before):
            return "upgraded"
        if Version(after) < Version(before):
            return "downgraded"
        return "changed"
    except Exception:  # noqa: BLE001 - unparsable or missing versions
        return "changed"


def freeze_changes(
    before: dict[str, FreezeEntry],
    after: dict[str, FreezeEntry],
    colab: dict[str, FreezeEntry],
) -> list[dict]:
    """Colab distributions the notebook's own install changed.

    Compares two ``pip freeze`` snapshots taken around the notebook run. Reports each
    distribution Colab ships that was upgraded, downgraded, changed or removed, and each
    newly installed distribution Colab ships at a different version.
    """
    changes = []
    for key, old in sorted(before.items()):
        if key not in colab:
            continue
        new = after.get(key)
        if new is None:
            change = "removed"
        elif new.version != old.version:
            change = _version_direction(old.version, new.version)
        else:
            continue
        changes.append(
            {
                "name": old.name,
                "colab": colab[key].version,
                "before": old.version,
                "after": new.version if new else None,
                "change": change,
            }
        )
    for key, new in sorted(after.items()):
        if key in before or key not in colab or colab[key].version == new.version:
            continue
        changes.append(
            {
                "name": new.name,
                "colab": colab[key].version,
                "before": None,
                "after": new.version,
                "change": "differs-from-colab",
            }
        )
    return changes


def strip_ansi(text: str) -> str:
    """``text`` without ANSI escapes, with each ``\\r``-overwritten line reduced to its last state."""
    text = _ANSI.sub("", text)
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def stderr_messages(text: str) -> list[str]:
    """Distinct messages in a cell's stderr: every unindented, non-blank line.

    Indented lines (the source line ``warnings`` echoes under a warning) and tqdm progress
    bars are skipped; logged warnings and errors have no category prefix, so they count too.
    """
    found = []
    for line in strip_ansi(text).splitlines():
        if not line.strip() or line[0].isspace() or _PROGRESS_BAR.search(line):
            continue
        if line.rstrip() not in found:
            found.append(line.rstrip())
    return found


def warning_lines(text: str) -> list[str]:
    """Lines of ``text`` that carry a Python warning category, or a pip/git WARNING/ERROR."""
    return [line.rstrip() for line in strip_ansi(text).splitlines() if _WARNING_LINE.search(line)]


def _last_line(text: str) -> str:
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def summarize_cells(notebook: dict, executed_order: list[int], elapsed: dict) -> list[dict]:
    """Per-code-cell records from an executed notebook (nbformat v4 as a dict).

    ``executed_order`` lists the indices of the cells the kernel ran, ``elapsed`` maps an
    index (int or str) to seconds. Status is ``ok``, ``error``, ``not-run`` or ``empty``.
    """
    ran = set(executed_order)
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        stdout, stderr, error = [], [], None
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                text = output.get("text", "")
                text = "".join(text) if isinstance(text, list) else text
                (stderr if output.get("name") == "stderr" else stdout).append(text)
            elif kind == "error":
                traceback = strip_ansi("\n".join(output.get("traceback", [])))
                error = {
                    "ename": output.get("ename", ""),
                    "evalue": strip_ansi(output.get("evalue", "")),
                    "last_line": _last_line(traceback),
                }
        stderr_text = strip_ansi("".join(stderr))
        stdout_text = strip_ansi("".join(stdout))
        if not source.strip():
            status = "empty"
        elif error is not None:
            status = "error"
        elif index in ran:
            status = "ok"
        else:
            status = "not-run"
        seconds = elapsed.get(index, elapsed.get(str(index)))
        cells.append(
            {
                "index": index,
                "status": status,
                "elapsed": round(seconds, 2) if seconds is not None else None,
                "first_line": next((ln for ln in source.splitlines() if ln.strip()), ""),
                "stderr": stderr_text,
                "stdout_tail": stdout_text[-4000:],
                "warnings": list(
                    dict.fromkeys(stderr_messages(stderr_text) + warning_lines(stdout_text))
                ),
                "error": error,
            }
        )
    return cells


def summarize_audit(records: list[dict], executed_order: list[int]) -> dict:
    """Cache lookups and network requests from the kernel audit log, keyed to cell indices.

    Each record's ``seq`` is the 1-based position of the cell in ``executed_order``.
    """

    def cell_of(record: dict) -> int | None:
        seq = record.get("seq", 0)
        return executed_order[seq - 1] if 0 < seq <= len(executed_order) else None

    hits: set[str] = set()
    lookups: dict[str, dict] = {}
    writes: dict[str, dict] = {}
    http: dict[tuple, dict] = {}
    hosts: dict[str, set] = {}
    for record in records:
        event = record.get("event")
        cell = cell_of(record)
        if event == "cache-exists":
            key = record.get("location") or record.get("key", "")
            if record.get("result"):
                hits.add(key)
            entry = lookups.setdefault(key, {"key": record.get("key"), "cells": set()})
            entry["cells"].add(cell)
        elif event == "cache-write":
            key = record.get("location") or record.get("key", "")
            entry = writes.setdefault(key, {"key": record.get("key"), "cells": set()})
            entry["cells"].add(cell)
        elif event == "http":
            ident = (record.get("method"), record.get("host"), record.get("path"))
            http.setdefault(ident, {"cells": set()})["cells"].add(cell)
        elif event == "dns":
            hosts.setdefault(record.get("host", ""), set()).add(cell)

    def cells(values: set) -> list:
        return sorted(v for v in values if v is not None)

    return {
        "cache_misses": [
            {"location": key, "key": entry["key"], "cells": cells(entry["cells"])}
            for key, entry in sorted(lookups.items())
            if key not in hits
        ],
        "cache_writes": [
            {"location": key, "key": entry["key"], "cells": cells(entry["cells"])}
            for key, entry in sorted(writes.items())
        ],
        "http_requests": [
            {"method": m, "host": h, "path": p, "cells": cells(entry["cells"])}
            for (m, h, p), entry in sorted(http.items(), key=lambda item: str(item[0]))
        ],
        "hosts": [{"host": h, "cells": cells(c)} for h, c in sorted(hosts.items())],
    }


def deduplicate_warnings(cells: list[dict]) -> list[dict]:
    """``[{"text", "cells"}]`` for every distinct warning line across ``cells``."""
    seen: dict[str, list[int]] = {}
    for cell in cells:
        for line in cell["warnings"]:
            text = line.strip()
            indices = seen.setdefault(text, [])
            if cell["index"] not in indices:
                indices.append(cell["index"])
    return [{"text": text, "cells": indices} for text, indices in seen.items()]


# -- notebook preparation (host) ---------------------------------------------------------


def _cell_text(cell) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def notebook_install_lines(notebook: dict) -> list[tuple[int, str]]:
    """``(cell index, line)`` for every ``pip install`` line in the notebook's code cells."""
    found = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for line in _cell_text(cell).split("\n"):
            if parse_pip_line(line) is not None:
                found.append((index, line))
    return found


def uncomment_notebook(notebook: dict) -> int | None:
    """Uncomment the Colab block in place; returns its cell index, or None if absent."""
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        new = uncomment_colab_block(_cell_text(cell))
        if new is not None:
            cell["source"] = new
            return index
    return None


def rewrite_notebook(
    notebook: dict, built_versions: set[str] | frozenset[str]
) -> tuple[list[dict], list[str]]:
    """Rewrite the clone and pip magics of every code cell in place.

    Returns ``(rewrites, branches)``: one record per touched line, and every ``--branch``
    value a rewritten clone asks for.
    """
    rewrites, branches = [], []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = _cell_text(cell).split("\n")
        changed = False
        for pos, line in enumerate(lines):
            clone = rewrite_clone_line(line)
            if clone is not None:
                lines[pos] = clone.line
                changed = True
                if clone.branch:
                    branches.append(clone.branch)
                rewrites.append({"cell": index, "original": line, "rewritten": clone.line})
                continue
            new, notes = rewrite_pip_line(line, built_versions)
            if notes:
                rewrites.append({"cell": index, "original": line, "rewritten": new, "notes": notes})
            if new != line:
                lines[pos] = new
                changed = True
        if changed:
            cell["source"] = "\n".join(lines)
    return rewrites, branches


# -- host infrastructure -----------------------------------------------------------------


def harness_home() -> Path:
    """``~/.cache/gliquid-colab-harness``, holding ``runs/``, ``work/`` and the build venv.

    Docker bind-mounts from here. It is outside AppData: a packaged (MSIX) Windows app's
    AppData writes are virtualised and invisible to the docker daemon.
    """
    base = Path.home() / ".cache" / "gliquid-colab-harness"
    for sub in ("runs", "work"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def repo_root() -> Path | None:
    """The first ancestor of this script named ``gliquid_python``, or None."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "gliquid_python":
            return parent
    return None


def _run(cmd: list[str], what: str, cwd: Path | None = None, input: str | None = None) -> str:
    """Run ``cmd``; return its stdout, or raise InfraError naming ``what`` on failure.

    ``input`` is sent as UTF-8 bytes with ``\\n`` line endings on every platform.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            input=input.encode("utf-8") if input is not None else None,
        )
    except FileNotFoundError as exc:
        raise InfraError(f"{what}: {cmd[0]} not found") from exc
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (stderr or stdout).strip()[-2000:]
        raise InfraError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return stdout


def docker_available() -> bool:
    """True when ``docker info`` reaches a running daemon."""
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8")
    except OSError as exc:
        raise InfraError(f"fetching {url}: {exc}") from exc


def load_colab_reference(args, out: Path) -> tuple[str, str]:
    """``(freeze text, os-info text)``, fetched live or read from files, saved into ``out``."""
    if args.freeze:
        freeze_text = Path(args.freeze).read_text(encoding="utf-8")
        freeze_source = str(Path(args.freeze).resolve())
    else:
        freeze_text = _fetch(COLAB_FREEZE_URL)
        freeze_source = COLAB_FREEZE_URL
    if args.os_info:
        os_text = Path(args.os_info).read_text(encoding="utf-8")
        os_source = str(Path(args.os_info).resolve())
    elif args.freeze and (Path(args.freeze).parent / OS_INFO_FILE).is_file():
        os_path = Path(args.freeze).parent / OS_INFO_FILE
        os_text = os_path.read_text(encoding="utf-8")
        os_source = str(os_path.resolve())
    else:
        os_text = _fetch(COLAB_OS_INFO_URL)
        os_source = COLAB_OS_INFO_URL
    (out / FREEZE_FILE).write_text(freeze_text, encoding="utf-8", newline="\n")
    (out / OS_INFO_FILE).write_text(os_text, encoding="utf-8", newline="\n")
    args.freeze_source, args.os_info_source = freeze_source, os_source
    return freeze_text, os_text


def ensure_image(base: str) -> str:
    """Build (or reuse from the layer cache) ``base`` plus git and ``/content``; return its tag."""
    tag = f"{IMAGE_PREFIX}:{re.sub(r'[^A-Za-z0-9_.-]+', '-', base)}"
    dockerfile = (
        f"FROM {base}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        f"RUN mkdir -p {CONTENT_DIR}\n"
        f"WORKDIR {CONTENT_DIR}\n"
    )
    _run(["docker", "build", "-q", "-t", tag, "-"], f"docker build {tag}", input=dockerfile)
    return tag


def clone_source(root: Path, ref: str, dest: Path) -> str:
    """Clone ``root`` into ``dest`` with ``ref`` checked out as ``main``; return the commit.

    Every branch of ``root`` becomes a local branch of the clone, so a notebook's
    ``git clone --branch X`` of the clone resolves as it would against GitHub.
    """
    sha = _run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        f"resolving --ref {ref}",
    ).strip()
    _run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.longpaths=true",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--config",
            "core.autocrlf=false",
            "--config",
            "core.longpaths=true",
            str(root),
            str(dest),
        ],
        "git clone of the local repository",
    )
    _run(["git", "-C", str(dest), "checkout", "--quiet", "-B", "main", sha], f"checkout {sha}")
    listing = _run(
        [
            "git",
            "-C",
            str(dest),
            "for-each-ref",
            "--format=%(objectname) %(refname:lstrip=3)",
            "refs/remotes/origin",
        ],
        "listing branches",
    )
    updates = []
    for row in listing.splitlines():
        objectname, _, name = row.partition(" ")
        if name and name not in ("HEAD", "main"):
            updates.append(f"update refs/heads/{name} {objectname}\n")
    if updates:
        _run(
            ["git", "-C", str(dest), "update-ref", "--stdin"],
            "creating local branches",
            input="".join(updates),
        )
    return sha


def has_ref(repo: Path, name: str) -> bool:
    """True when ``name`` is a branch or tag of ``repo``."""
    for ref in (f"refs/heads/{name}", f"refs/tags/{name}"):
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref],
            capture_output=True,
        )
        if proc.returncode == 0:
            return True
    return False


def _builder_python(cache_dir: Path) -> str:
    """A Python that can run ``python -m build``: this one, or a reusable throwaway venv."""
    probe = subprocess.run(
        [sys.executable, "-m", "build", "--version"], capture_output=True, text=True
    )
    if probe.returncode == 0:
        return sys.executable
    venv = cache_dir / "buildenv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.is_file():
        probe = subprocess.run([str(python), "-m", "build", "--version"], capture_output=True)
        if probe.returncode == 0:
            return str(python)
    _run([sys.executable, "-m", "venv", "--clear", str(venv)], "creating the build venv")
    _run([str(python), "-m", "pip", "install", "--quiet", "build"], "installing build")
    return str(python)


def build_wheel(src: Path, wheels: Path, cache_dir: Path) -> tuple[Path, str]:
    """Build a wheel of ``src`` into ``wheels``; return ``(path, version)``.

    Raises InfraError when the version is ``0.0.0``, which means hatch-vcs found no git data.
    """
    before = set(wheels.glob("gliquid-*.whl"))
    python = _builder_python(cache_dir)
    _run(
        [python, "-m", "build", "--wheel", "--outdir", str(wheels), str(src)],
        f"building a wheel of {src.name}",
    )
    built = sorted(set(wheels.glob("gliquid-*.whl")) - before)
    if not built:
        raise InfraError(f"building a wheel of {src.name}: no gliquid wheel produced")
    version = wheel_version(built[-1].name) or ""
    if version == "0.0.0" or version.startswith("0.0.0+"):
        raise InfraError(f"wheel version is {version}: hatch-vcs saw no git metadata")
    return built[-1], version


def _rmtree(path: Path) -> None:
    def make_writable(func, target, _exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.exists():
        shutil.rmtree(path, onerror=make_writable)


def _mount(source: Path, target: str, readonly: bool = False) -> list[str]:
    spec = f"type=bind,source={source},target={target}"
    return ["--mount", spec + (",readonly" if readonly else "")]


def run_container(image: str, mounts: list[str], log_path: Path, pip_cache: bool) -> int:
    """Run the in-container stage, streaming its output to the console and ``log_path``."""
    name = f"gliquid-colab-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--name", name, "-w", CONTENT_DIR, "-e", "PYTHONUNBUFFERED=1"]
    cmd += mounts
    if pip_cache:
        cmd += ["--mount", f"type=volume,source={PIP_CACHE_VOLUME},target=/root/.cache/pip"]
    cmd += [
        image,
        "python",
        f"{CONTAINER_HARNESS}/{Path(__file__).name}",
        "--inner",
        f"{CONTAINER_OUT}/plan.json",
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            raise


# -- in-container stage ------------------------------------------------------------------

_AUDIT_STARTUP = r"""
def _gliquid_colab_harness_audit():
    import functools
    import importlib.abc
    import importlib.util
    import json
    import sys
    import threading

    log_path = "/out/audit.jsonl"
    state = {"seq": 0}
    lock = threading.Lock()

    def emit(record):
        record["seq"] = state["seq"]
        line = json.dumps(record, default=str)
        with lock:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def pre_run_cell(info=None):
        state["seq"] += 1

    shell = get_ipython()  # noqa: F821
    if shell is not None:
        shell.events.register("pre_run_cell", pre_run_cell)

    def hook(event, args):
        if event == "http.client.send":
            try:
                data = args[1]
                if isinstance(data, (bytes, bytearray)) and b"HTTP/1." in data[:4096]:
                    head = bytes(data[:4096]).split(b"\r\n")
                    method, _, rest = head[0].decode("latin-1").partition(" ")
                    path = rest.rsplit(" ", 1)[0]
                    host = next(
                        (h[5:].strip().decode("latin-1") for h in head[1:]
                         if h[:5].lower() == b"host:"),
                        "",
                    )
                    emit({"event": "http", "method": method, "host": host, "path": path[:300]})
            except Exception:
                pass
        elif event == "socket.getaddrinfo":
            try:
                emit({"event": "dns", "host": str(args[0])})
            except Exception:
                pass

    sys.addaudithook(hook)

    def wrap(module):
        for obj in list(vars(module).values()):
            if not isinstance(obj, type) or obj.__module__ != module.__name__:
                continue
            if getattr(obj, "_is_protocol", False):
                continue
            if "exists" not in vars(obj) or not callable(getattr(obj, "locate", None)):
                continue

            def make_exists(original, cls_name):
                @functools.wraps(original)
                def exists(self, key):
                    result = original(self, key)
                    try:
                        location = self.locate(key)
                    except Exception:
                        location = None
                    emit({"event": "cache-exists", "backend": cls_name, "key": repr(key),
                          "location": location, "result": bool(result)})
                    return result
                return exists

            def make_write(original, cls_name):
                @functools.wraps(original)
                def write_json(self, key, payload, *a, **kw):
                    try:
                        location = self.locate(key)
                    except Exception:
                        location = None
                    emit({"event": "cache-write", "backend": cls_name, "key": repr(key),
                          "location": location})
                    return original(self, key, payload, *a, **kw)
                return write_json

            obj.exists = make_exists(vars(obj)["exists"], obj.__name__)
            if "write_json" in vars(obj):
                obj.write_json = make_write(vars(obj)["write_json"], obj.__name__)

    class _WrapOnImport(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "gliquid.cache":
                return None
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                wrap(module)

            spec.loader.exec_module = exec_module
            return spec

    if "gliquid.cache" in sys.modules:
        wrap(sys.modules["gliquid.cache"])
    else:
        sys.meta_path.insert(0, _WrapOnImport())


_gliquid_colab_harness_audit()
del _gliquid_colab_harness_audit
"""


def _inner_step(steps: list, name: str, cmd: list[str], check: bool = False) -> bool:
    print(f"[harness] {name}: {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    output = (proc.stdout + proc.stderr).strip()
    steps.append(
        {
            "name": name,
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.monotonic() - started, 1),
            "output_tail": output[-3000:],
        }
    )
    if proc.returncode != 0:
        print(output[-3000:], flush=True)
        if check:
            raise InfraError(f"{name} failed (exit {proc.returncode})")
    return proc.returncode == 0


def _dump_modules(path: Path) -> None:
    import importlib.metadata as metadata

    modules: dict[str, set] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        paths = [str(p) for p in (dist.files or [])]
        modules.setdefault(name, set()).update(provided_modules(paths))
    path.write_text(json.dumps({k: sorted(v) for k, v in sorted(modules.items())}, indent=1))


def _pip_freeze(path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True
    )
    path.write_text(proc.stdout)


def inner_main(plan_path: str) -> int:
    """The in-container stage: preinstall Colab's packages, execute the notebook, snapshot."""
    plan = json.loads(Path(plan_path).read_text())
    out = Path(CONTAINER_OUT)
    result: dict = {"steps": [], "python": sys.version.split()[0]}
    steps = result["steps"]
    freeze = parse_freeze((out / FREEZE_FILE).read_text())

    try:
        pip_entry = freeze.get("pip")
        if pip_entry and pip_entry.version:
            _inner_step(
                steps,
                "pin pip",
                [sys.executable, "-m", "pip", "install", "-q", f"pip=={pip_entry.version}"],
            )
        _inner_step(
            steps,
            "git safe.directory",
            ["git", "config", "--global", "--add", "safe.directory", "*"],
            check=True,
        )

        closure: list[str] = []
        if plan["closure_specs"]:
            report = out / "closure-report.json"
            ok = _inner_step(
                steps,
                "resolve install closure",
                [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "-q"]
                + ["--report", str(report), "--find-links", CONTAINER_WHEELS]
                + plan["closure_specs"],
            )
            if ok:
                data = json.loads(report.read_text())
                closure = [item["metadata"]["name"] for item in data.get("install", [])]
        result["closure"] = closure

        pins, absent, unpinnable = select_preinstall(closure, freeze)
        result.update(preinstall=pins, absent_tools=absent, unpinnable=unpinnable)
        constraints = out / "colab-constraints.txt"
        constraints.write_text("\n".join(constraints_from_freeze(freeze)) + "\n")
        install = [sys.executable, "-m", "pip", "install", "-q"]
        if _inner_step(
            steps,
            "preinstall (Colab-constrained)",
            install + ["-c", str(constraints)] + pins + absent,
        ):
            result["preinstall_mode"] = "constrained"
        else:
            _inner_step(steps, "preinstall (unconstrained)", install + pins + absent, check=True)
            result["preinstall_mode"] = "unconstrained"
    except InfraError as exc:
        result["infra_error"] = str(exc)
        (out / "inner.json").write_text(json.dumps(result, indent=1))
        return 2

    _pip_freeze(out / "freeze-before.txt")
    _dump_modules(out / "modules-before.json")

    startup = Path.home() / ".ipython/profile_default/startup/00-gliquid-colab-harness.py"
    startup.parent.mkdir(parents=True, exist_ok=True)
    startup.write_text(_AUDIT_STARTUP)
    (out / "audit.jsonl").write_text("")

    result.update(_execute(plan))

    _pip_freeze(out / "freeze-after.txt")
    _dump_modules(out / "modules-after.json")
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib.metadata as m; print(m.version('gliquid'))"],
        capture_output=True,
        text=True,
    )
    result["gliquid_version_after"] = probe.stdout.strip() or None
    (out / "inner.json").write_text(json.dumps(result, indent=1))
    return 0


def _execute(plan: dict) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    notebook = nbformat.read(plan["notebook"], as_version=4)
    order: list[int] = []
    started: dict[int, float] = {}
    elapsed: dict[int, float] = {}

    def on_execute(cell=None, cell_index=None, **_):
        order.append(cell_index)
        started[cell_index] = time.monotonic()
        print(f"[cell {cell_index}] running: {_cell_text(cell).strip()[:70]!r}", flush=True)

    def on_executed(cell=None, cell_index=None, execute_reply=None, **_):
        elapsed[cell_index] = time.monotonic() - started.get(cell_index, time.monotonic())
        status = (execute_reply or {}).get("content", {}).get("status", "?")
        print(f"[cell {cell_index}] {status} in {elapsed[cell_index]:.1f}s", flush=True)

    client = NotebookClient(
        notebook,
        timeout=plan["timeout"],
        startup_timeout=180,
        kernel_name="python3",
        allow_errors=plan["allow_errors"],
        resources={"metadata": {"path": CONTENT_DIR}},
        on_cell_execute=on_execute,
        on_cell_executed=on_executed,
    )
    abort = None
    t0 = time.monotonic()
    try:
        client.execute()
    except CellTimeoutError as exc:
        abort = {"kind": "timeout", "cell": order[-1] if order else None, "message": str(exc)}
    except CellExecutionError:
        abort = {"kind": "cell-error", "cell": order[-1] if order else None}
    except DeadKernelError as exc:
        abort = {"kind": "dead-kernel", "cell": order[-1] if order else None, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - kernel start or executor failure is reported
        abort = {"kind": "executor", "message": f"{type(exc).__name__}: {exc}"}
    for index in order:
        if index not in elapsed and index in started:
            elapsed[index] = time.monotonic() - started[index]
    nbformat.write(notebook, plan["executed"])
    return {
        "order": order,
        "elapsed": elapsed,
        "abort": abort,
        "notebook_seconds": round(time.monotonic() - t0, 1),
    }


# -- reporting (host) --------------------------------------------------------------------


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def build_report(out: Path, context: dict) -> dict:
    """Assemble ``report.json`` from the in-container outputs in ``out``."""
    inner = _read_json(out / "inner.json", {})
    executed = _read_json(out / context["executed_name"], {"cells": []})
    order = inner.get("order", [])
    cells = summarize_cells(executed, order, inner.get("elapsed", {}))
    abort = inner.get("abort") or {}
    if abort.get("kind") == "timeout" and abort.get("cell") is not None:
        for cell in cells:
            if cell["index"] == abort["cell"]:
                cell["status"] = "timeout"

    colab = parse_freeze((out / FREEZE_FILE).read_text(encoding="utf-8"))
    before = parse_freeze(_text(out / "freeze-before.txt"))
    after = parse_freeze(_text(out / "freeze-after.txt"))
    hazards = freeze_changes(before, after, colab)
    modules_before = _read_json(out / "modules-before.json", {})
    modules_after = _read_json(out / "modules-after.json", {})
    collisions = []
    for module, dists in find_collisions(modules_after):
        new = [d for d in dists if d not in modules_before]
        collisions.append({"module": module, "distributions": dists, "added_by_notebook": new})
        colab_dists = [d for d in dists if normalize_name(d) in colab]
        if new and colab_dists:
            hazards.append(
                {
                    "name": ", ".join(colab_dists),
                    "colab": ", ".join(
                        colab[normalize_name(d)].version or "?" for d in colab_dists
                    ),
                    "before": None,
                    "after": None,
                    "change": f"module '{module}' overwritten by {', '.join(new)}",
                }
            )

    audit = []
    for line in _text(out / "audit.jsonl").splitlines():
        try:
            audit.append(json.loads(line))
        except ValueError:
            continue

    failed = [c for c in cells if c["status"] in ("error", "timeout")]
    not_run = [c for c in cells if c["status"] == "not-run"]
    if inner.get("infra_error") or (abort and abort.get("kind") in ("executor", "dead-kernel")):
        exit_code = 2 if not order else 1
    elif failed or not_run or collisions:
        exit_code = 1
    else:
        exit_code = 0

    return {
        **context,
        "container_python": inner.get("python"),
        "gliquid_version_after": inner.get("gliquid_version_after"),
        "notebook_seconds": inner.get("notebook_seconds"),
        "abort": inner.get("abort"),
        "infra_error": inner.get("infra_error"),
        "steps": inner.get("steps", []),
        "cells": cells,
        "warnings": deduplicate_warnings(cells),
        "install": {
            "closure": inner.get("closure", []),
            "preinstalled": inner.get("preinstall", []),
            "preinstall_mode": inner.get("preinstall_mode"),
            "tools_absent_from_freeze": inner.get("absent_tools", []),
            "unpinnable": inner.get("unpinnable", []),
            "restart_hazards": hazards,
            "module_collisions": collisions,
        },
        "fetches": summarize_audit(audit, order),
        "exit_code": exit_code,
    }


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def format_report(report: dict) -> str:
    """The human-readable ``report.txt``."""
    lines = [
        f"notebook        : {report['notebook']} ({report['notebook_source']})",
        f"ref             : {report['ref']} = {report['sha']}",
        f"wheel version   : {report['wheel_version']}",
        f"installed after : gliquid {report.get('gliquid_version_after')}",
        f"image           : {report['base_image']} (container Python "
        f"{report.get('container_python')}; Colab Python {report['colab_python']})",
        f"Colab freeze    : {report['freeze_source']}",
        f"wall time       : {report['wall_seconds']:.0f}s total, notebook "
        f"{report.get('notebook_seconds') or 0:.0f}s",
    ]
    if report.get("uncommented_cell") is not None:
        lines.append(f"uncommented     : cell {report['uncommented_cell']} (Colab block)")
    if report.get("infra_error"):
        lines.append(f"INFRA ERROR     : {report['infra_error']}")
    abort = report.get("abort")
    if abort:
        lines.append(f"stopped         : {abort.get('kind')} at cell {abort.get('cell')}")

    lines += ["", "INSTALL REWRITES"]
    for item in report["rewrites"] or [{"cell": "-", "original": "(none)", "rewritten": ""}]:
        lines.append(f"  cell {item['cell']}: {item['original'].strip()}")
        if item["rewritten"] and item["rewritten"] != item["original"]:
            lines.append(f"      -> {item['rewritten'].strip()}")
        for note in item.get("notes", []):
            lines.append(f"      {note}")
    for note in report.get("notes", []):
        lines.append(f"  note: {note}")

    lines += ["", "CELLS"]
    for cell in report["cells"]:
        if cell["status"] == "empty":
            continue
        seconds = f"{cell['elapsed']:.1f}s" if cell["elapsed"] is not None else "-"
        lines.append(
            f"  [{cell['index']:>2}] {cell['status']:8} {seconds:>8}  {cell['first_line'][:70]}"
        )
        if cell["error"]:
            err = cell["error"]
            lines.append(f"       {err['ename']}: {err['evalue'][:300]}")
            if err["last_line"] and err["last_line"] not in f"{err['ename']}: {err['evalue']}":
                lines.append(f"       last line: {err['last_line'][:300]}")
        stderr = [ln for ln in cell["stderr"].splitlines() if ln.strip()]
        for ln in stderr[:15]:
            lines.append(f"       stderr| {ln[:200]}")
        if len(stderr) > 15:
            lines.append(f"       stderr| ... {len(stderr) - 15} more lines in report.json")

    lines += ["", "WARNINGS (deduplicated stderr messages; pip/git warnings from stdout)"]
    for item in report["warnings"] or [{"text": "(none)", "cells": []}]:
        where = ", ".join(str(c) for c in item["cells"])
        lines.append(f"  [cell {where}] {item['text'][:300]}" if where else f"  {item['text']}")

    install = report["install"]
    lines += ["", "INSTALL"]
    lines.append("  closure resolved for: " + " ".join(report.get("closure_specs", [])))
    lines.append(
        f"  preinstalled at Colab versions ({install['preinstall_mode']}): "
        f"{len(install['preinstalled'])} distributions"
    )
    if install["tools_absent_from_freeze"]:
        lines.append(
            "  not in the Colab freeze, installed unpinned: "
            + ", ".join(install["tools_absent_from_freeze"])
        )
    if install["unpinnable"]:
        lines.append(
            "  Colab ships as a direct URL, not preinstalled: " + ", ".join(install["unpinnable"])
        )
    lines.append("  Colab runtime-restart hazards:")
    for item in install["restart_hazards"] or [{"name": "(none)"}]:
        if "change" not in item:
            lines.append("    (none)")
            continue
        lines.append(
            f"    {item['name']}: Colab {item['colab']}, before {item['before']}, "
            f"after {item['after']} ({item['change']})"
        )
    lines.append("  Module collisions:")
    for item in install["module_collisions"] or [{"module": None}]:
        if item["module"] is None:
            lines.append("    (none)")
            continue
        added = (
            f" (added by the notebook: {', '.join(item['added_by_notebook'])})"
            if item["added_by_notebook"]
            else ""
        )
        lines.append(f"    {item['module']}: {', '.join(item['distributions'])}{added}")

    fetches = report["fetches"]
    lines += ["", "FETCHES (kernel only; shell magics are not traced)"]
    lines.append("  cache records looked up and not found:")
    for item in fetches["cache_misses"] or [{"location": "(none)", "cells": []}]:
        lines.append(f"    {item['location']}  cells {item['cells']}")
    if fetches["cache_writes"]:
        lines.append("  cache records written:")
        for item in fetches["cache_writes"]:
            lines.append(f"    {item['location']}  cells {item['cells']}")
    lines.append("  HTTP requests:")
    for item in fetches["http_requests"] or [
        {"method": "(none)", "host": "", "path": "", "cells": []}
    ]:
        lines.append(f"    {item['method']} {item['host']}{item['path']}  cells {item['cells']}")
    if fetches["hosts"]:
        lines.append(
            "  hosts resolved: " + ", ".join(f"{h['host']} {h['cells']}" for h in fetches["hosts"])
        )

    verdict = {0: "PASS", 1: "FAIL", 2: "INFRA ERROR"}[report["exit_code"]]
    lines += ["", f"RESULT: {verdict} (exit {report['exit_code']})"]
    return "\n".join(lines) + "\n"


# -- host driver -------------------------------------------------------------------------


def _resolve_notebook(requested: str, clone: Path) -> tuple[Path, str]:
    candidate = Path(requested)
    if candidate.suffix == ".ipynb" and candidate.is_file():
        return candidate.resolve(), "working file"
    name = requested if requested.endswith(".ipynb") else f"{requested}.ipynb"
    path = clone / "notebooks" / Path(name).name
    if not path.is_file():
        available = sorted(p.name for p in (clone / "notebooks").glob("*.ipynb"))
        raise InfraError(f"no notebook {name!r} at the ref; available: {', '.join(available)}")
    return path, "the --ref checkout"


def pin_from_ref(ref: str) -> int:
    """Build a wheel from ``ref`` and print its version and the matching pip line."""
    root = repo_root()
    if root is None:
        print("error: no gliquid_python directory above this script")
        return 2
    base = harness_home()
    work = Path(tempfile.mkdtemp(prefix="pin-", dir=base / "work"))
    try:
        sha = clone_source(root, ref, work / "src")
        (work / "wheels").mkdir()
        _, version = build_wheel(work / "src", work / "wheels", base)
    except InfraError as exc:
        print(f"error: {exc}")
        return 2
    finally:
        _rmtree(work)
    print(f"{ref} = {sha}")
    print(version)
    print(f'!pip install -q "gliquid=={version}"')
    return 0


def host_main(args) -> int:
    import nbformat

    started = time.monotonic()
    root = repo_root()
    if root is None:
        print("error: no gliquid_python directory above this script")
        return 2
    if not docker_available():
        print("error: docker is not available (`docker info` failed); start Docker and retry")
        return 2

    base = harness_home()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.out:
        out = Path(args.out).resolve()
    else:
        out = Path(
            tempfile.mkdtemp(prefix=f"{stamp}-{Path(args.notebook).stem}-", dir=base / "runs")
        )
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=base / "work"))
    print(f"output directory: {out}")
    try:
        freeze_text, os_text = load_colab_reference(args, out)
        py = parse_python_version(os_text)
        if py is None and not args.image:
            raise InfraError("could not read the Python version from os-info; pass --image")
        base_image = args.image or f"python:{py[0]}.{py[1]}-slim"
        colab_python = ".".join(str(p) for p in py if p is not None) if py else "unknown"

        print(f"==> image {base_image}", flush=True)
        image = ensure_image(base_image)

        print(f"==> cloning --ref {args.ref}", flush=True)
        src = work / "src"
        sha = clone_source(root, args.ref, src)
        wheels = work / "wheels"
        wheels.mkdir()
        print("==> building the wheel", flush=True)
        _, ref_version = build_wheel(src, wheels, base)
        built = {ref_version}
        notes = []
        if ref_version.startswith("0.0."):
            notes.append(f"wheel version {ref_version}: no tag reachable from {sha[:10]}")

        nb_path, nb_source = _resolve_notebook(args.notebook, src)
        notebook = nbformat.read(nb_path, as_version=4)
        uncommented = None
        if args.uncomment_colab:
            uncommented = uncomment_notebook(notebook)
            if uncommented is None:
                raise InfraError(f"{nb_path.name} has no '{COLAB_BANNER}' cell")

        for _, line in notebook_install_lines(notebook):
            for req in parse_pip_line(line) or []:
                if req.kind != "gliquid" or req.version is None:
                    continue
                if gliquid_pin_action(req.version, built) != "build-sha":
                    continue
                pin_sha, dirty = parse_local_dev_version(req.version)
                if dirty:
                    notes.append(f"{req.version}: dirty-tree version cannot be rebuilt; PyPI")
                    continue
                probe = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "rev-parse",
                        "--verify",
                        "--quiet",
                        f"{pin_sha}^{{commit}}",
                    ],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    notes.append(
                        f"{req.version}: commit {pin_sha} not in the local repository; PyPI"
                    )
                    continue
                print(f"==> building the pinned commit {pin_sha}", flush=True)
                pin_src = work / f"src-{pin_sha}"
                clone_source(root, probe.stdout.strip(), pin_src)
                _, pin_version = build_wheel(pin_src, wheels, base)
                built.add(pin_version)
                notes.append(f"{req.version}: built commit {pin_sha} as {pin_version}")

        requirements = [
            req
            for _, line in notebook_install_lines(notebook)
            for req in parse_pip_line(line) or []
        ]
        rewrites, branches = rewrite_notebook(notebook, built)
        for branch in branches:
            if not has_ref(src, branch):
                notes.append(f"--branch {branch}: not a branch or tag of the local repository")
        harness = work / "harness"
        harness.mkdir()
        shutil.copy2(Path(__file__).resolve(), harness / Path(__file__).name)
        stem = nb_path.stem
        input_name, executed_name = f"{stem}.input.ipynb", f"{stem}.executed.ipynb"
        nbformat.write(notebook, out / input_name)
        for stale in ("audit.jsonl", "inner.json", "freeze-before.txt", "freeze-after.txt"):
            (out / stale).unlink(missing_ok=True)
        plan = {
            "notebook": f"{CONTAINER_OUT}/{input_name}",
            "executed": f"{CONTAINER_OUT}/{executed_name}",
            "timeout": args.timeout,
            "allow_errors": args.allow_errors,
            "closure_specs": closure_specs(requirements, ref_version, args.closure_extras),
        }
        (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")

        print(f"==> running {nb_path.name} in {base_image}", flush=True)
        mounts = (
            _mount(src, CONTAINER_SRC, readonly=True)
            + _mount(wheels, CONTAINER_WHEELS, readonly=True)
            + _mount(out, CONTAINER_OUT)
            + _mount(harness, CONTAINER_HARNESS, readonly=True)
        )
        code = run_container(image, mounts, out / "container.log", not args.no_pip_cache)
        if not (out / "inner.json").is_file():
            raise InfraError(f"the container exited {code} without a result; see container.log")

        context = {
            "notebook": nb_path.name,
            "notebook_source": nb_source,
            "ref": args.ref,
            "sha": sha,
            "wheel_version": ref_version,
            "wheels_built": sorted(built),
            "base_image": base_image,
            "colab_python": colab_python,
            "freeze_source": args.freeze_source,
            "os_info_source": args.os_info_source,
            "uncommented_cell": uncommented,
            "rewrites": rewrites,
            "closure_specs": plan["closure_specs"],
            "notes": notes,
            "executed_name": executed_name,
            "wall_seconds": time.monotonic() - started,
        }
        report = build_report(out, context)
    except InfraError as exc:
        print(f"error: {exc}")
        (out / "report.txt").write_text(f"INFRA ERROR: {exc}\n", encoding="utf-8")
        return 2
    finally:
        if args.keep_work:
            print(f"work directory kept: {work}")
        else:
            _rmtree(work)

    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    text = format_report(report)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"report: {out / 'report.txt'}")
    return report["exit_code"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a notebook as Google Colab would, in a Colab-like container.",
        epilog=(
            "Exit status: 0 if every cell ran and no module collision was found, 1 if a cell "
            "failed or a collision was found, 2 for a usage or infrastructure error."
        ),
    )
    parser.add_argument(
        "--notebook",
        default=DEFAULT_NOTEBOOK,
        metavar="NAME|PATH",
        help=f"notebook name in notebooks/ at --ref, or a path to an .ipynb file "
        f"(default: {DEFAULT_NOTEBOOK})",
    )
    parser.add_argument(
        "--ref", default="HEAD", help="git ref of this repository to clone (default: HEAD)"
    )
    parser.add_argument("--freeze", metavar="FILE", help="saved Colab pip-freeze.txt to use")
    parser.add_argument(
        "--os-info",
        metavar="FILE",
        help="saved Colab os-info.txt (default: the colab-os-info.txt beside --freeze, else fetched)",
    )
    parser.add_argument(
        "--image", metavar="IMAGE", help="base image (default: python:<Colab major.minor>-slim)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="S",
        help=f"per-cell execution timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="keep executing after a failing cell (default: stop, like Colab's Run all)",
    )
    parser.add_argument(
        "--uncomment-colab",
        action="store_true",
        help=f"uncomment the '{COLAB_BANNER}' cell before running",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        help="output directory (default: a new directory under ~/.cache/gliquid-colab-harness/runs)",
    )
    parser.add_argument(
        "--keep-work", action="store_true", help="keep the temporary clone and wheel directory"
    )
    parser.add_argument(
        "--no-pip-cache",
        action="store_true",
        help=f"do not mount the '{PIP_CACHE_VOLUME}' docker volume as pip's cache",
    )
    parser.add_argument(
        "--closure-extras",
        default=DEFAULT_CLOSURE_EXTRAS,
        metavar="EXTRAS",
        help="gliquid extras added to every gliquid requirement when resolving which Colab "
        "packages to preinstall, so Colab's copies of optional dependencies are present even "
        "when the notebook installs gliquid without them; '' for none "
        f"(default: {DEFAULT_CLOSURE_EXTRAS})",
    )
    parser.add_argument(
        "--pin-from-ref",
        metavar="REF",
        help="print the version a wheel built from REF carries (the value for a "
        "`gliquid==VERSION` pin) and exit; needs git, not docker",
    )
    parser.add_argument("--inner", metavar="PLAN", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.closure_extras = _split_extras(args.closure_extras)
    if args.inner:
        return inner_main(args.inner)
    if args.pin_from_ref:
        return pin_from_ref(args.pin_from_ref)
    return host_main(args)


if __name__ == "__main__":
    sys.exit(main())
