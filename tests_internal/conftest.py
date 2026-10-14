"""Pytest configuration for the maintainer-only tier.

Mirrors ``tests/conftest.py``'s ``.env`` bootstrap, so ``pytest tests_internal`` alone still
puts ``NEW_MP_API_KEY`` into ``os.environ`` for the live fetch in
``test_dft_data_loading.py``.

The shared fixtures are defined in ``tests/_mpds_builders.py``.
"""

from pathlib import Path

from gliquid.api import load_dotenv
from tests._mpds_builders import (  # noqa: F401
    matrix_data_env,
    matrix_data_store,
    ss_from_cache,
)

# Explicit repo-root path so test behavior is independent of config.find_project_root's cwd walk.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
