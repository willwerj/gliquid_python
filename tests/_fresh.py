"""Run code in a fresh interpreter on this checkout's ``src``.

``run_fresh(code, cwd)`` runs ``code`` with ``sys.executable -c`` from ``cwd``, with ``src``
first on ``PYTHONPATH``, and returns the completed process with stdout and stderr captured as
text. ``blocked`` names modules the child cannot import: importing one of them or any of its
submodules raises ``ModuleNotFoundError``, and each block is checked to have taken before
``code`` runs. ``env`` sets variables in the child's environment; a ``None`` value removes one.

``ok(result)`` asserts a zero exit status and an ``OK`` line on stdout, and returns stdout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

_BLOCKER = """\
import importlib, sys

_BLOCKED = {blocked!r}


def _is_blocked(name):
    return any(name == b or name.startswith(b + ".") for b in _BLOCKED)


class _Block:
    def find_spec(self, name, path=None, target=None):
        if _is_blocked(name):
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


for _name in [m for m in sys.modules if _is_blocked(m)]:
    del sys.modules[_name]
sys.meta_path.insert(0, _Block())
for _name in _BLOCKED:
    try:
        importlib.import_module(_name)
    except ImportError:
        pass
    else:
        raise AssertionError(f"the {{_name}} block did not take; the test is vacuous")
"""


def run_fresh(
    code: str,
    cwd: Path,
    *,
    blocked: tuple[str, ...] = (),
    env: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter from ``cwd`` and return the completed process."""
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
    )
    for name, value in (env or {}).items():
        if value is None:
            child_env.pop(name, None)
        else:
            child_env[name] = value
    preamble = _BLOCKER.format(blocked=tuple(blocked)) if blocked else ""
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(code)],
        cwd=str(cwd),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def ok(result: subprocess.CompletedProcess) -> str:
    """Assert ``result`` exited 0 and printed ``OK``; return its stdout."""
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout
