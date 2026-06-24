"""Pytest configuration for the gliquid test suite.

Makes secrets (e.g. ``NEW_MP_API_KEY``) discoverable for tests that exercise the
live Materials Project API path, without ever committing them.

Resolution order for each key:
  1. Already present in ``os.environ`` -- CI provides ``NEW_MP_API_KEY`` via a repo
     secret (see .github/workflows/tests.yml), so it always wins there.
  2. A gitignored ``.env`` file at the repository root (``KEY=VALUE`` per line).

Tests that need the API gate on ``os.getenv('NEW_MP_API_KEY')`` and skip when it is
absent, so the offline (cached) suite still runs everywhere.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Populate ``os.environ`` from a repo-root ``.env`` for keys not already set."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
