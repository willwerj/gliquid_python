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

from pathlib import Path

from gliquid.api import load_dotenv

# Explicit repo-root path so test behavior is independent of config.find_project_root's cwd walk.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
