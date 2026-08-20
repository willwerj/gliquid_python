"""Tests for gliquid.config path resolution and the unary registry's load behavior.

gliquid's data comes in two kinds and they resolve differently:

  BUNDLED   phase_transitions.json, omegas_hcp.json, spurious_structures.json ship inside
            the package (src/gliquid/reference/). An installed gliquid always has them, so
            it can never reach the state where the unary registry is empty and every
            element reference evaluates to zero.
  EXTERNAL  the per-system DFT entry caches, MPDS diagrams and the model bundle. Not
            shipped. Reachable only through set_data_dir() or GLIQUID_DATA_DIR.

Resolution order is set_data_dir() -> GLIQUID_DATA_DIR -> a source checkout's cache/ (found
by walking __file__'s parents for 'gliquid_python') -> bundled -> ConfigError. There is no
working-directory branch at any step: config used to walk up from Path.cwd() and fall back
to cwd itself, so a consumer running outside gliquid_python/ silently got data_dir=<cwd>/data.
The subprocess tests below run from a foreign cwd on purpose, and the "installed" ones
import a copy of the package from outside any checkout so the __file__ walk cannot fire --
that is the only way to exercise what a pip-installed user actually gets.

Run with: python -m pytest tests/test_config.py -v
"""

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src"
sys.path.insert(0, str(_SRC))

_PROBE = (
    "import json, gliquid.config as config, gliquid.phase as phase\n"
    "print(json.dumps({\n"
    "    'project_root': None if config.project_root is None else str(config.project_root),\n"
    "    'data_dir': None if config.data_dir is None else str(config.data_dir),\n"
    "    'phase_transitions_file': str(config.phase_transitions_file),\n"
    "    'n_elements': len(phase.UNARY.elements),\n"
    "    'al_t_fusion': phase.UNARY['Al'].t_fusion,\n"
    "}))\n"
)


def _run_probe(
    code: str, cwd: Path, pkg_path: Path, env_extra: dict | None = None
) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(pkg_path)}
    # BOTH spellings, and this is not belt-and-braces. These tests assert on what an
    # installed gliquid resolves with NOTHING configured, so any inherited corpus variable
    # makes them assert the opposite of their point. Popping only the legacy name is exactly
    # how every tox env broke when tox.ini moved to GLIQUID_CACHE_DIR: the probe silently
    # inherited a configured corpus, config.data_dir came back a Path instead of None, and
    # the failure surfaced as "WindowsPath is not JSON serializable" — nothing that names
    # the real cause. Add any future corpus variable here at the same time it is introduced.
    for _corpus_var in ("GLIQUID_CACHE_DIR", "GLIQUID_DATA_DIR"):
        env.pop(_corpus_var, None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _probe_from(cwd: Path, pkg_path: Path = _SRC, env_extra: dict | None = None) -> dict:
    """Import gliquid.config + phase in a subprocess started in ``cwd``."""
    result = _run_probe(_PROBE, cwd, pkg_path, env_extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _bundled_dir() -> Path:
    """Where the IMPORTED gliquid keeps its package data.

    Derived from the imported module rather than from ``_SRC``: under tox the suite runs
    against an installed wheel, so the package data lives in site-packages and a
    source-tree constant would assert against a directory this process never reads.
    """
    import gliquid.config as config

    return Path(config._BUNDLED_REFERENCE_DIR)


@pytest.fixture(scope="module")
def installed_pkg(tmp_path_factory) -> Path:
    """A copy of the package outside any 'gliquid_python' ancestor — i.e. site-packages.

    Copying rather than mocking is deliberate: the thing under test is what
    ``Path(__file__).resolve().parents`` yields, which nothing but a real relocation
    changes. ``__pycache__`` is skipped so the copy cannot import stale bytecode.
    """
    dest = tmp_path_factory.mktemp("site_packages")
    shutil.copytree(
        _SRC / "gliquid", dest / "gliquid", ignore=shutil.ignore_patterns("__pycache__")
    )
    return dest


# ---------------------------------------------------------------------------------------
# Source checkout: cwd-independent, and the corpus resolves without configuration
# ---------------------------------------------------------------------------------------


def test_data_dir_resolves_from_foreign_cwd(tmp_path):
    """Importing from an unrelated cwd must still find the checkout's cache directory."""
    probe = _probe_from(tmp_path)
    assert Path(probe["data_dir"]) == _PKG_ROOT / "cache"
    assert probe["n_elements"] > 50, "unary registry silently loaded empty"
    assert probe["al_t_fusion"] == 933.5


def test_data_dir_resolves_from_package_cwd():
    """The historical good case (cwd inside gliquid_python) keeps working."""
    probe = _probe_from(_PKG_ROOT)
    assert Path(probe["data_dir"]) == _PKG_ROOT / "cache"
    assert probe["n_elements"] > 50


def test_reference_tables_come_from_the_package_not_the_corpus():
    """The three reference tables are package data, not corpus data."""
    import gliquid.config as config

    bundled = _bundled_dir()
    for path in (
        config.phase_transitions_file,
        config.omegas_file,
        config.spurious_structures_file,
    ):
        assert Path(path).parent == bundled
        assert Path(path).exists()


# ---------------------------------------------------------------------------------------
# Installed package: no cwd guessing, loud failure, env-var escape hatch
# ---------------------------------------------------------------------------------------


def test_installed_package_ships_working_reference_data(installed_pkg, tmp_path):
    """pip-installed, no configuration: element references are real, and nothing warns."""
    code = (
        "import warnings, json, gliquid.config as config\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always', UserWarning)\n"
        "    import gliquid.phase as phase\n"
        "caught = [w for w in caught if issubclass(w.category, UserWarning)]\n"
        "print(json.dumps({\n"
        "    'project_root': config.project_root,\n"
        "    'data_dir': config.data_dir,\n"
        "    'n_elements': len(phase.UNARY.elements),\n"
        "    'fe_t_fusion': phase.UNARY['Fe'].t_fusion,\n"
        "    'warnings': [str(w.message) for w in caught],\n"
        "}))\n"
    )
    result = _run_probe(code, tmp_path, installed_pkg)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])

    assert probe["project_root"] is None, "the __file__ walk must not match outside a checkout"
    assert probe["data_dir"] is None, "no corpus configured -> data_dir must stay unset"
    assert probe["n_elements"] > 50, "bundled reference tables did not ship"
    assert probe["fe_t_fusion"] == 1811.0
    assert probe["warnings"] == [], f"import warned about an empty registry: {probe['warnings']}"


def test_installed_package_never_falls_back_to_cwd(installed_pkg, tmp_path):
    """A cwd holding a plausible cache/ must NOT be adopted. This is the deleted behavior.

    Run from a directory named 'gliquid_python' with a populated cache/ underneath — the
    exact shape the old cwd walk searched for and returned. The decoy has to carry the
    CURRENT corpus directory name; spelled 'data' it would be a decoy nothing looks for,
    and the assertions below would pass without exercising anything.
    """
    decoy = tmp_path / "gliquid_python"
    (decoy / "cache").mkdir(parents=True)
    (decoy / "cache" / "phase_transitions.json").write_text('{"elements": {}}')

    probe = _probe_from(decoy, pkg_path=installed_pkg)
    assert probe["data_dir"] is None, "cwd was adopted as the cache directory"
    assert probe["project_root"] is None
    # and the decoy's empty table must not have displaced the real bundled one
    assert probe["n_elements"] > 50
    assert Path(probe["phase_transitions_file"]).parent == installed_pkg / "gliquid" / "reference"


def test_installed_package_corpus_access_raises_config_error(installed_pkg, tmp_path):
    """Reading the external corpus with nothing configured must name the two remedies."""
    code = (
        "import gliquid.config as config\n"
        "from gliquid import api\n"
        "from gliquid.cache import CacheKey\n"
        "try:\n"
        "    api.resolve_cache_path(CacheKey('Cu-Mg', 'dft_entries', 'GGA'))\n"
        "except config.ConfigError as exc:\n"
        "    print('CONFIG_ERROR:' + str(exc))\n"
        "else:\n"
        "    raise AssertionError('corpus access silently resolved')\n"
    )
    result = _run_probe(code, tmp_path, installed_pkg)
    assert result.returncode == 0, result.stderr
    message = result.stdout.split("CONFIG_ERROR:", 1)[1]
    assert "set_data_dir" in message
    assert "GLIQUID_DATA_DIR" in message
    assert "Cu-Mg" in message


def test_gliquid_data_dir_env_var_supplies_the_corpus(installed_pkg, tmp_path):
    """GLIQUID_DATA_DIR is the environment-level equivalent of set_data_dir()."""
    probe = _probe_from(
        tmp_path, pkg_path=installed_pkg, env_extra={"GLIQUID_DATA_DIR": str(_PKG_ROOT / "cache")}
    )
    assert Path(probe["data_dir"]) == _PKG_ROOT / "cache"
    assert probe["n_elements"] > 50


def test_env_var_loses_to_an_explicit_set_data_dir(installed_pkg, tmp_path):
    """set_data_dir() sits above the env var in the resolution order."""
    code = (
        "import json, gliquid.config as config\n"
        "before = str(config.data_dir)\n"
        "config.set_data_dir(r'" + str(tmp_path / "chosen") + "')\n"
        "print(json.dumps({'before': before, 'after': str(config.data_dir)}))\n"
    )
    result = _run_probe(
        code, tmp_path, installed_pkg, env_extra={"GLIQUID_DATA_DIR": str(_PKG_ROOT / "cache")}
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(probe["before"]) == _PKG_ROOT / "cache"
    assert Path(probe["after"]) == tmp_path / "chosen"


# ---------------------------------------------------------------------------------------
# set_data_dir semantics under the two-root split (reagent D60)
# ---------------------------------------------------------------------------------------


def test_explicit_set_data_dir_still_wins(tmp_path):
    """set_data_dir/set_project_root overrides are the API for custom stores."""
    import gliquid.config as config

    old_root, old_dir = config.project_root, config.data_dir
    try:
        config.set_data_dir(tmp_path)
        assert config.data_dir == tmp_path
    finally:
        config.set_project_root(old_root)
        config.set_data_dir(old_dir)


def test_reference_files_fall_back_to_bundled_for_a_partial_corpus(tmp_path):
    """Pointing at a corpus directory with no reference tables still yields working ones.

    This is the decided semantics: `data_dir/<name>` IF that file exists, else bundled.
    A directory holding only per-system caches must not produce a zeroed unary registry.
    """
    import gliquid.config as config

    old_dir = config.data_dir
    try:
        config.set_data_dir(tmp_path)  # empty directory
        bundled = _bundled_dir()
        assert Path(config.phase_transitions_file) == bundled / "phase_transitions.json"
        assert Path(config.omegas_file) == bundled / "omegas_hcp.json"
        assert Path(config.spurious_structures_file) == bundled / "spurious_structures.json"
    finally:
        config.set_data_dir(old_dir)


def test_reference_files_prefer_a_copy_present_in_the_data_dir(tmp_path):
    """A data dir carrying its own reference table overrides the shipped one.

    This is how the unary database is iterated on: drop an edited phase_transitions.json
    next to the caches and it wins, exactly as it did before the split.
    """
    import gliquid.config as config

    (tmp_path / "phase_transitions.json").write_text('{"elements": {}}')
    old_dir = config.data_dir
    try:
        config.set_data_dir(tmp_path)
        assert Path(config.phase_transitions_file) == tmp_path / "phase_transitions.json"
        # the two absent ones still come from the package -- the check is per file
        assert Path(config.omegas_file) == _bundled_dir() / "omegas_hcp.json"
    finally:
        config.set_data_dir(old_dir)


def test_set_omegas_file_still_overrides(tmp_path):
    """The per-file override predates the split and must keep working."""
    import gliquid.config as config

    old_omegas, old_dir = config.omegas_file, config.data_dir
    try:
        custom = tmp_path / "my_omegas.json"
        config.set_omegas_file(custom)
        assert config.omegas_file == custom
        # ...but set_data_dir re-resolves all three, which is its documented job
        config.set_data_dir(tmp_path)
        assert Path(config.omegas_file) == _bundled_dir() / "omegas_hcp.json"
    finally:
        config.set_data_dir(old_dir)
        config.omegas_file = old_omegas


def test_require_data_dir_returns_the_corpus_when_configured(tmp_path):
    import gliquid.config as config

    old_dir = config.data_dir
    try:
        config.set_data_dir(tmp_path)
        assert config.require_data_dir() == tmp_path
        config.set_data_dir(None)
        with pytest.raises(config.ConfigError, match="GLIQUID_DATA_DIR"):
            config.require_data_dir()
    finally:
        config.set_data_dir(old_dir)


def test_find_project_root_returns_none_for_an_unknown_dirname():
    """No match is None now, not the working directory."""
    import gliquid.config as config

    assert config.find_project_root() == _PKG_ROOT
    assert config.find_project_root(dirname="not_a_real_ancestor_name") is None


# ---------------------------------------------------------------------------------------
# Registry load behavior
# ---------------------------------------------------------------------------------------


def test_empty_registry_load_warns(tmp_path):
    """An unreadable phase_transitions.json must load loudly, not silently.

    The file is overridden directly rather than via set_data_dir: an empty data dir now
    falls back to the bundled table (by design), so pointing at one no longer produces an
    empty registry. The warning path itself is unchanged and still worth pinning.
    """
    import gliquid.config as config
    import gliquid.phase as phase

    old_file = config.phase_transitions_file
    try:
        config.phase_transitions_file = tmp_path / "phase_transitions.json"  # absent
        with pytest.warns(UserWarning, match="empty"):
            phase.reload(require=False)
        assert len(phase.UNARY.elements) == 0
    finally:
        config.phase_transitions_file = old_file
        phase.reload(require=True)
        assert len(phase.UNARY.elements) > 50


def test_unknown_element_zero_default_is_preserved():
    """Per-symbol zero default for unknown elements is pinned behavior (test_ternary.py)."""
    import gliquid.phase as phase

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn for a per-symbol miss
        ref = phase.UNARY["Xx"]
    assert ref.h_liq == 0.0 and ref.t_fusion == 0.0


def test_solid_solution_switch_defaults():
    """The package-wide SS switch defaults OFF, ref_mode defaults to from_unary_db."""
    import gliquid.config as config
    import gliquid.solution as solution

    assert config.solid_solutions is False
    assert config.ss_ref_mode == "from_unary_db"
    # keep config.ss_ref_mode in sync with solution.DEFAULT_REF_MODE (no shared import).
    assert config.ss_ref_mode == solution.DEFAULT_REF_MODE


def test_set_solid_solutions_and_ss_ref_mode():
    """Setters flip the globals; ss_ref_mode validates against the allowed set."""
    import gliquid.config as config

    old_ss, old_mode = config.solid_solutions, config.ss_ref_mode
    try:
        config.set_solid_solutions(True)
        assert config.solid_solutions is True
        config.set_solid_solutions(0)  # coerced to bool
        assert config.solid_solutions is False

        config.set_ss_ref_mode("from_omegas_file")
        assert config.ss_ref_mode == "from_omegas_file"
        with pytest.raises(ValueError):
            config.set_ss_ref_mode("bogus")
    finally:
        config.set_solid_solutions(old_ss)
        config.set_ss_ref_mode(old_mode)


def test_coverage_threshold_defaults():
    """Shipped defaults for the solid-energy coverage gate."""
    import gliquid.config as config

    assert config.coverage_skip_frac == 0.45
    assert config.coverage_min_missing == 2
    assert config.coverage_missing_frac == 0.60
    assert config.coverage_ss_narrow_tol == 0.10
    assert config.coverage_dft_cover_tol == 0.10
    # Load-bearing: uncapped, one DFT compound would rescue an arbitrarily wide solid-solution
    # field and complete solid solutions with no models would read as fully supported.
    assert config.coverage_ss_rescue_max_width == 0.25


def test_set_coverage_thresholds_partial_update_and_validation():
    """Only the passed arguments change; fractions must lie in (0, 1]."""
    import gliquid.config as config

    old = (
        config.coverage_skip_frac,
        config.coverage_min_missing,
        config.coverage_missing_frac,
        config.coverage_ss_narrow_tol,
        config.coverage_dft_cover_tol,
        config.coverage_ss_rescue_max_width,
    )
    try:
        config.set_coverage_thresholds(skip_frac=0.3)
        assert config.coverage_skip_frac == 0.3
        assert config.coverage_min_missing == old[1]  # untouched

        config.set_coverage_thresholds(min_missing=3, ss_rescue_max_width=0.4)
        assert config.coverage_min_missing == 3
        assert config.coverage_ss_rescue_max_width == 0.4

        for kwargs in (
            {"skip_frac": 0},
            {"skip_frac": 1.5},
            {"missing_frac": -0.1},
            {"dft_cover_tol": 2},
            {"ss_rescue_max_width": 0},
            {"min_missing": 0},
        ):
            with pytest.raises(ValueError):
                config.set_coverage_thresholds(**kwargs)
    finally:
        config.set_coverage_thresholds(
            skip_frac=old[0],
            min_missing=old[1],
            missing_frac=old[2],
            ss_narrow_tol=old[3],
            dft_cover_tol=old[4],
            ss_rescue_max_width=old[5],
        )
