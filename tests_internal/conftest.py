"""Pytest configuration for the maintainer-only tier.

Mirrors ``tests/conftest.py``'s ``.env`` bootstrap so ``pytest tests_internal`` works on its
own. Without it the only consumer of a secret in this tier -- the live Materials Project
fetch in ``test_dft_data_loading.py`` -- would silently SKIP whenever ``tests/`` was not also
collected in the same run, because ``tests/conftest.py`` is what puts ``NEW_MP_API_KEY`` into
``os.environ``. A network test that quietly stops running is worse than one that fails.
"""

from pathlib import Path

from gliquid.api import load_dotenv

# Explicit repo-root path so test behavior is independent of config.find_project_root's cwd walk.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
