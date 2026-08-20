"""Execute the shipped notebooks in ``notebooks/`` and report pass / warn / fail.

This is the **only** notebook-execution surface in the repository, and it is driven by
``tox`` (the ``py{310,311,312,313}-max-notebooks`` environments), deliberately not by the
public test suite: executing a notebook needs the external data corpus and a Materials
Project API key, neither of which a clean CI checkout has. The CI-safe half is
``tests/test_notebook_imports.py``, which resolves every gliquid symbol a notebook uses
without running a kernel.

Three outcomes, not two:

``pass``
    Every cell executed.
``warn``
    The notebook cannot run *here* for a structural reason -- it is Colab-only by
    construction, or it drives an ``ipywidgets`` UI that has no front-end headlessly, or it
    needs a key/network/corpus this environment does not have. Reported, not failed.
``fail``
    A cell raised, or the notebook exceeded the timeout. This is the signal.

Exit status is 0 when every notebook passed or warned, 1 when any failed, and 2 for a usage
error (no ``notebooks/`` directory, or ``--notebook`` naming a file that does not exist).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 900

PASS = "pass"
WARN = "warn"
FAIL = "fail"


def notebooks_dir() -> Path | None:
    """``notebooks/`` under the repository root, or None if it is not there.

    Anchored by NAME -- the first ancestor called ``gliquid_python`` -- rather than a
    ``parents[N]`` index, so moving this script between directories cannot silently point it
    at some other tree.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == "gliquid_python":
            candidate = parent / "notebooks"
            return candidate if candidate.is_dir() else None
    return None


# An UNCOMMENTED Colab bootstrap line. colab_demo.ipynb clones the repository into
# /content/ and installs it there; running that anywhere else either fails on a path that
# does not exist or, worse, clones into the working tree. Detected before execution so the
# notebook is never started.
_COLAB_BOOTSTRAP = re.compile(r"^\s*(![ \t]*git[ \t]+clone|%cd[ \t]+/content)", re.MULTILINE)

# Failure texts that mean "this environment is missing something", not "the notebook is
# broken". Kept short and explicit on purpose: every pattern added here is a class of real
# failure that stops being reported as one.
_STRUCTURAL = (
    (
        re.compile(r"NoSuchKernel|No such kernel|kernelspec", re.IGNORECASE),
        "no usable Jupyter kernel in this environment",
    ),
    (
        re.compile(r"ipywidgets|No module named ['\"]?IPython"),
        "drives an ipywidgets UI; needs the `editor` extra and a live front-end",
    ),
    (
        re.compile(
            r"NEW_MP_API_KEY|MPRestError|MPDSError|mpds_client|YOUR_API_KEY_HERE"
            # What pymatgen/mp-api actually raise when the key is absent, stale or the
            # wrong length -- neither message names the environment variable.
            r"|Please use a new API key|materialsproject\.org/api"
        ),
        "needs a Materials Project / MPDS API key",
    ),
    (
        re.compile(
            r"ConnectionError|ConnectTimeout|ReadTimeout|Max retries exceeded"
            r"|Temporary failure in name resolution|Name or service not known"
        ),
        "needs network access",
    ),
    (
        re.compile(r"ConfigError|GLIQUID_(?:CACHE|DATA)_DIR"),
        "needs the external cache corpus (set GLIQUID_CACHE_DIR)",
    ),
)


# Kernel tracebacks arrive with IPython's colour escapes embedded; they turn the summary
# table into noise and would also let a pattern above miss a match split by an escape.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def classify_error(text: str) -> tuple[str, str]:
    """``(status, note)`` for a failed execution, given its traceback text."""
    text = _ANSI.sub("", text)
    for pattern, reason in _STRUCTURAL:
        if pattern.search(text):
            return WARN, reason
    last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return FAIL, last[:110]


def colab_only(path: Path) -> bool:
    """True when a code cell carries an active ``!git clone`` / ``%cd /content`` line."""
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    return any(
        cell.get("cell_type") == "code" and _COLAB_BOOTSTRAP.search(cell.get("source", ""))
        for cell in notebook.cells
    )


def run_one(path: Path, timeout: int) -> tuple[str, str, float]:
    """Execute one notebook. Returns ``(status, note, seconds)``."""
    import nbformat
    from nbclient.exceptions import CellTimeoutError
    from nbconvert.preprocessors import ExecutePreprocessor

    started = time.monotonic()
    try:
        notebook = nbformat.read(path, as_version=4)
        executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        # Run with the notebook's own directory as cwd: several notebooks reach the corpus
        # through a relative path (`Path.cwd().parent / 'cache'`).
        executor.preprocess(notebook, {"metadata": {"path": str(path.parent)}})
    except CellTimeoutError:
        return FAIL, f"exceeded the {timeout}s timeout", time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - any kernel-side error is a result, not a crash
        status, note = classify_error(f"{type(exc).__name__}: {exc}")
        return status, note, time.monotonic() - started
    return PASS, "", time.monotonic() - started


def select(directory: Path, requested: str | None) -> list[Path] | None:
    """The notebooks to run, or None if ``requested`` names one that is not there."""
    available = sorted(directory.glob("*.ipynb"))
    if requested is None:
        return available
    wanted = requested if requested.endswith(".ipynb") else f"{requested}.ipynb"
    match = [p for p in available if p.name == wanted]
    if not match:
        print(f"error: no notebook named {wanted!r} in {directory}")
        print("available: " + ", ".join(p.name for p in available))
        return None
    return match


def report(results: list[tuple[str, str, str, float]]) -> None:
    """Print the per-notebook summary table."""
    width = max((len(name) for name, *_ in results), default=8)
    print()
    print(f"{'notebook'.ljust(width)}  {'status':6}  {'time':>8}  note")
    print(f"{'-' * width}  {'-' * 6}  {'-' * 8}  {'-' * 40}")
    for name, status, note, seconds in results:
        print(f"{name.ljust(width)}  {status:6}  {seconds:7.1f}s  {note}")
    tally = {state: sum(1 for r in results if r[1] == state) for state in (PASS, WARN, FAIL)}
    print()
    print(f"{tally[PASS]} passed, {tally[WARN]} warned, {tally[FAIL]} failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the notebooks in notebooks/ and report pass / warn / fail.",
        epilog=(
            "Exit status: 0 if every notebook passed or warned, 1 if any failed, "
            "2 for a usage error."
        ),
    )
    parser.add_argument(
        "--notebook",
        metavar="NAME",
        help="run only this notebook (with or without the .ipynb suffix)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="S",
        help=f"per-cell execution timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args(argv)

    directory = notebooks_dir()
    if directory is None:
        print("error: no notebooks/ directory found above this script")
        return 2

    selected = select(directory, args.notebook)
    if selected is None:
        return 2
    if not selected:
        print(f"error: {directory} contains no .ipynb files")
        return 2

    results: list[tuple[str, str, str, float]] = []
    for path in selected:
        print(f"==> {path.name}", flush=True)
        if colab_only(path):
            results.append((path.name, WARN, "Colab-only (clones into /content/)", 0.0))
            continue
        status, note, seconds = run_one(path, args.timeout)
        results.append((path.name, status, note, seconds))

    report(results)
    return 1 if any(status == FAIL for _, status, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
