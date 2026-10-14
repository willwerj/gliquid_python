"""Tests for gliquid.config path resolution and the unary registry's load behavior.

gliquid's data comes in two kinds and they resolve differently:

  BUNDLED   phase_transitions.json, omegas_hcp.json, spurious_structures.json ship inside
            the package (src/gliquid/reference/), so the unary registry is never empty.
  EXTERNAL  the per-system DFT entry caches, MPDS diagrams and the model bundle. Not
            shipped. Configured with set_cache_dir() or GLIQUID_CACHE_DIR.

Resolution order is set_cache_dir() -> GLIQUID_CACHE_DIR -> a source checkout's cache/
(found by walking __file__'s parents for 'gliquid_python') -> <cwd>/gliquid_cache, a flat
directory store created on its first write. The cwd default is an implicit store: it never
supplies the reference tables, and a cwd's cache/ is never adopted. set_cache_dir(None)
unsets the store, after which corpus access raises ConfigError. The subprocess tests run
from a foreign cwd, and the "installed" ones import a copy of the package from outside any
checkout so the __file__ walk cannot fire.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src"
sys.path.insert(0, str(_SRC))

import gliquid.config as config

_NAMED_CHECKOUT = pytest.mark.skipif(
    _PKG_ROOT.name != "gliquid_python",
    reason="checkout resolution needs a checkout directory named gliquid_python",
)

_PROBE = (
    "import json, gliquid.config as config, gliquid.phase as phase\n"
    "print(json.dumps({\n"
    "    'project_root': None if config.project_root is None else str(config.project_root),\n"
    "    'cache_dir': None if config.cache_dir is None else str(config.cache_dir),\n"
    "    'phase_transitions_file': str(config.phase_transitions_file),\n"
    "    'n_elements': len(phase.UNARY.elements),\n"
    "    'al_t_fusion': phase.UNARY['Al'].t_fusion,\n"
    "}))\n"
)


def _run_probe(
    code: str, cwd: Path, pkg_path: Path, env_extra: dict | None = None
) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(pkg_path)}
    # The probes assert what gliquid resolves with NOTHING configured.
    env.pop("GLIQUID_CACHE_DIR", None)
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


def _json_probe(code: str, cwd: Path, pkg_path: Path, env_extra: dict | None = None) -> dict:
    """Run ``code`` in a subprocess started in ``cwd``; parse its last stdout line as JSON."""
    result = _run_probe(code, cwd, pkg_path, env_extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _bundled_dir() -> Path:
    """Where the IMPORTED gliquid keeps its package data: site-packages under tox, so never
    derive it from ``_SRC``."""
    import gliquid.config as config

    return Path(config._BUNDLED_REFERENCE_DIR)


@pytest.fixture(scope="module")
def installed_pkg(tmp_path_factory) -> Path:
    """A copy of the package outside any 'gliquid_python' ancestor — i.e. site-packages.

    A real copy, not a mock: what is under test is what ``Path(__file__).resolve().parents``
    yields. ``__pycache__`` is skipped so the copy cannot import stale bytecode.
    """
    dest = tmp_path_factory.mktemp("site_packages")
    shutil.copytree(
        _SRC / "gliquid", dest / "gliquid", ignore=shutil.ignore_patterns("__pycache__")
    )
    return dest


# -- Source checkout: cwd-independent, and the corpus resolves without configuration --


@_NAMED_CHECKOUT
def test_cache_dir_resolves_from_foreign_cwd(tmp_path):
    """Importing from an unrelated cwd must still find the checkout's cache directory."""
    probe = _probe_from(tmp_path)
    assert Path(probe["cache_dir"]) == _PKG_ROOT / "cache"
    assert probe["n_elements"] > 50, "unary registry silently loaded empty"
    assert probe["al_t_fusion"] == 933.5


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


# -- Installed package: the cwd default store, no cache/ adoption, env-var escape hatch --


def test_installed_package_ships_working_reference_data(installed_pkg, tmp_path):
    """pip-installed, no configuration: the store is <cwd>/gliquid_cache, element references
    are real, and nothing warns."""
    code = (
        "import warnings, json, gliquid.config as config\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always', UserWarning)\n"
        "    import gliquid.phase as phase\n"
        "caught = [w for w in caught if issubclass(w.category, UserWarning)]\n"
        "print(json.dumps({\n"
        "    'project_root': config.project_root,\n"
        "    'cache_dir': None if config.cache_dir is None else str(config.cache_dir),\n"
        "    'n_elements': len(phase.UNARY.elements),\n"
        "    'fe_t_fusion': phase.UNARY['Fe'].t_fusion,\n"
        "    'warnings': [str(w.message) for w in caught],\n"
        "}))\n"
    )
    result = _run_probe(code, tmp_path, installed_pkg)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])

    assert probe["project_root"] is None, "the __file__ walk must not match outside a checkout"
    assert probe["cache_dir"] is not None, "no default store was installed"
    assert Path(probe["cache_dir"]) == (tmp_path / "gliquid_cache").resolve()
    assert probe["n_elements"] > 50, "bundled reference tables did not ship"
    assert probe["fe_t_fusion"] == 1811.0
    assert probe["warnings"] == [], f"import warned about an empty registry: {probe['warnings']}"


def test_installed_package_never_falls_back_to_cwd(installed_pkg, tmp_path):
    """A cwd holding a plausible cache/ must NOT be adopted; the default store is used.

    Run from a directory named 'gliquid_python' with a populated cache/ underneath, the
    shape a working-directory walk would search for. The decoy carries the current corpus
    directory name, so a lookup of that name would find it.
    """
    decoy = tmp_path / "gliquid_python"
    (decoy / "cache").mkdir(parents=True)
    (decoy / "cache" / "phase_transitions.json").write_text('{"elements": {}}')

    probe = _probe_from(decoy, pkg_path=installed_pkg)
    assert probe["cache_dir"] is not None, "no default store was installed"
    assert Path(probe["cache_dir"]) == (decoy / "gliquid_cache").resolve()
    assert Path(probe["cache_dir"]) != decoy / "cache", "cwd's cache/ was adopted"
    assert probe["project_root"] is None
    # and the decoy's empty table must not have displaced the real bundled one
    assert probe["n_elements"] > 50
    assert Path(probe["phase_transitions_file"]).parent == installed_pkg / "gliquid" / "reference"


def test_env_var_loses_to_an_explicit_set_cache_dir(installed_pkg, tmp_path):
    """set_cache_dir() sits above the env var in the resolution order."""
    code = (
        "import json, gliquid.config as config\n"
        "before = str(config.cache_dir)\n"
        "config.set_cache_dir(r'" + str(tmp_path / "chosen") + "')\n"
        "print(json.dumps({'before': before, 'after': str(config.cache_dir)}))\n"
    )
    result = _run_probe(
        code, tmp_path, installed_pkg, env_extra={"GLIQUID_CACHE_DIR": str(_PKG_ROOT / "cache")}
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(probe["before"]) == _PKG_ROOT / "cache"
    assert Path(probe["after"]) == tmp_path / "chosen"


# -- Installed package: the implicit default store <cwd>/gliquid_cache --

_DEFAULT_STORE_READS = (
    "import json\n"
    "from pathlib import Path\n"
    "import gliquid.config as config\n"
    "from gliquid import api\n"
    "from gliquid.cache import CacheKey, resolve_backend\n"
    "from gliquid.ternary import TernaryLiquidInterpolation\n"
    "store = Path.cwd() / 'gliquid_cache'\n"
    "state = {'cache_dir': str(config.cache_dir), 'cache_mode': config.cache_mode,\n"
    "         'dir_structure': config.dir_structure, 'after_import': store.exists()}\n"
    "backend = resolve_backend(None)\n"
    "key = CacheKey('Cu-Mg', 'dft_entries', 'GGA')\n"
    "state['exists'] = backend.exists(key)\n"
    "state['variants'] = backend.variants('Cu-Mg', 'mpds')\n"
    "state['resolved'] = str(api.resolve_cache_path(key))\n"
    "try:\n"
    "    backend.read_json(key)\n"
    "except FileNotFoundError:\n"
    "    state['read_json'] = 'FileNotFoundError'\n"
    "ti = TernaryLiquidInterpolation(\n"
    "    ['Al', 'Mg', 'Si'], data_dir=Path.cwd() / 'ternary_store', delta=0.25,\n"
    "    xs_mix={'Al-Mg': [0.0] * 4, 'Mg-Si': [0.0] * 4, 'Si-Al': [0.0] * 4},\n"
    ")\n"
    "try:\n"
    "    state['floor'] = repr(ti._reference_floor())\n"
    "except config.ConfigError as exc:\n"
    "    state['floor_error'] = str(exc)\n"
    "state['implicit'] = config._cache_dir_is_implicit()\n"
    "state['after_reads'] = store.exists()\n"
    "config.set_cache_dir(None)\n"
    "try:\n"
    "    api.resolve_cache_path(key)\n"
    "except config.ConfigError as exc:\n"
    "    state['unset_error'] = str(exc)\n"
    "print(json.dumps(state))\n"
)

_DEFAULT_STORE_WRITES = (
    "import json, logging, sys\n"
    "from pathlib import Path\n"
    "logging.basicConfig(level=logging.INFO, stream=sys.stderr,\n"
    "                    format='LOG %(name)s %(levelname)s %(message)s')\n"
    "import gliquid.config as config\n"
    "from gliquid.cache import CacheKey, resolve_backend\n"
    "backend = resolve_backend(None)\n"
    "backend.write_json(CacheKey('Cu-Mg', 'dft_entries', 'GGA'), [{'entry_id': 'mp-1'}])\n"
    "backend.write_json(CacheKey('Cu-Mg', 'mpds', ''), {'reference': None})\n"
    "print(json.dumps({'store': str((Path.cwd() / 'gliquid_cache').resolve())}))\n"
)


def test_installed_default_store_is_cwd_gliquid_cache_and_reads_create_nothing(
    installed_pkg, tmp_path
):
    """Nothing configured: a flat directory store at <cwd>/gliquid_cache, absent until a
    write; import, read misses and a ternary on its own store leave it absent.

    That ternary's reference floor raises ConfigError naming both stores. After
    set_cache_dir(None), corpus access raises ConfigError naming the remedies and the system.
    """
    ternary_store = tmp_path / "ternary_store"
    ternary_store.mkdir()
    probe = _json_probe(
        _DEFAULT_STORE_READS, tmp_path, installed_pkg, env_extra={"GLIQUID_OFFLINE": "1"}
    )
    store = (tmp_path / config.DEFAULT_CACHE_DIR_NAME).resolve()

    assert Path(probe["cache_dir"]) == store
    assert probe["cache_mode"] == "directory"
    assert probe["dir_structure"] == "flat"
    assert probe["exists"] is False
    assert probe["variants"] == []
    assert Path(probe["resolved"]) == store / "Cu-Mg_ENTRIES_MP_GGA.json"
    assert probe["read_json"] == "FileNotFoundError"
    assert probe["after_import"] is False
    assert probe["after_reads"] is False
    assert not store.exists()

    assert probe["implicit"] is True
    assert "floor_error" in probe, probe
    assert str(ternary_store.resolve()) in probe["floor_error"]
    assert str(store) in probe["floor_error"]

    assert "unset_error" in probe, "corpus access silently resolved"
    for remedy in ("set_cache_dir", "GLIQUID_CACHE_DIR", "Cu-Mg"):
        assert remedy in probe["unset_error"]


def test_installed_first_write_creates_the_flat_default_store_and_logs_once(
    installed_pkg, tmp_path
):
    """The first write creates <cwd>/gliquid_cache and lands flat; one INFO line names the
    path and both ways to change it."""
    result = _run_probe(_DEFAULT_STORE_WRITES, tmp_path, installed_pkg)
    assert result.returncode == 0, result.stderr
    reported = json.loads(result.stdout.strip().splitlines()[-1])["store"]
    store = (tmp_path / "gliquid_cache").resolve()
    assert Path(reported) == store

    assert sorted(p.name for p in store.iterdir()) == ["Cu-Mg.json", "Cu-Mg_ENTRIES_MP_GGA.json"]
    assert json.loads((store / "Cu-Mg_ENTRIES_MP_GGA.json").read_text()) == [{"entry_id": "mp-1"}]
    assert json.loads((store / "Cu-Mg.json").read_text()) == {"reference": None}

    announced = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("LOG ") and " INFO " in line and reported in line
    ]
    assert len(announced) == 1, result.stderr
    assert "set_cache_dir" in announced[0]
    assert "GLIQUID_CACHE_DIR" in announced[0]


def test_cache_dir_env_var_beats_the_cwd_default(installed_pkg, tmp_path):
    """GLIQUID_CACHE_DIR sits above the cwd default in the resolution order."""
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    probe = _probe_from(cwd, pkg_path=installed_pkg, env_extra={"GLIQUID_CACHE_DIR": str(chosen)})
    assert Path(probe["cache_dir"]) == chosen
    assert not (cwd / "gliquid_cache").exists()


def _decoy_tables_cwd(tmp_path: Path) -> Path:
    """A cwd whose gliquid_cache/ and cache/ each hold an empty phase_transitions.json."""
    cwd = tmp_path / "work"
    for name in ("gliquid_cache", "cache"):
        (cwd / name).mkdir(parents=True)
        (cwd / name / "phase_transitions.json").write_text('{"elements": {}}')
    return cwd


def test_installed_default_store_never_supplies_the_reference_tables(installed_pkg, tmp_path):
    """The implicit store is <cwd>/gliquid_cache, yet the bundled tables stay in use and the
    cwd's cache/ is not adopted."""
    cwd = _decoy_tables_cwd(tmp_path)
    probe = _probe_from(cwd, pkg_path=installed_pkg)

    assert probe["cache_dir"] is not None, "no default store was installed"
    assert Path(probe["cache_dir"]) == (cwd / "gliquid_cache").resolve()
    assert Path(probe["cache_dir"]) != cwd / "cache"
    assert probe["n_elements"] > 50
    assert Path(probe["phase_transitions_file"]).parent == installed_pkg / "gliquid" / "reference"


def test_explicit_store_at_the_default_path_supplies_its_reference_table(installed_pkg, tmp_path):
    """An explicit set_cache_dir to the same directory lets its own table override."""
    cwd = _decoy_tables_cwd(tmp_path)
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import gliquid.config as config\n"
        "config.set_cache_dir(Path.cwd() / 'gliquid_cache')\n"
        "print(json.dumps({'phase_transitions_file': str(config.phase_transitions_file)}))\n"
    )
    probe = _json_probe(code, cwd, installed_pkg)
    assert Path(probe["phase_transitions_file"]).resolve() == (
        (cwd / "gliquid_cache" / "phase_transitions.json").resolve()
    )


def test_installed_default_store_blocked_by_a_file_raises_config_error(installed_pkg, tmp_path):
    """A regular file named gliquid_cache in the cwd: the store stays a directory store, and
    a write into it raises ConfigError naming the path and both remedies."""
    (tmp_path / "gliquid_cache").write_text("not a directory")
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import gliquid.config as config\n"
        "from gliquid.cache import CacheKey, DirectoryBackend, resolve_backend\n"
        "state = {'cache_mode': config.cache_mode, 'cache_dir': str(config.cache_dir),\n"
        "         'expected': str((Path.cwd() / 'gliquid_cache').resolve()), 'errors': []}\n"
        "key = CacheKey('Cu-Mg', 'dft_entries', 'GGA')\n"
        "for backend in (resolve_backend(None), DirectoryBackend(config.cache_dir)):\n"
        "    try:\n"
        "        backend.write_json(key, [])\n"
        "    except config.ConfigError as exc:\n"
        "        state['errors'].append({'message': str(exc),\n"
        "                                'cause_is_oserror': isinstance(exc.__cause__, OSError)})\n"
        "print(json.dumps(state))\n"
    )
    probe = _json_probe(code, tmp_path, installed_pkg)

    assert probe["cache_mode"] == "directory"
    assert Path(probe["cache_dir"]) == (tmp_path / "gliquid_cache").resolve()
    assert len(probe["errors"]) == 2, "a write into the blocked default store did not raise"
    for error in probe["errors"]:
        assert probe["expected"] in error["message"]
        assert "set_cache_dir" in error["message"]
        assert "GLIQUID_CACHE_DIR" in error["message"]
        assert error["cause_is_oserror"] is True
    assert (tmp_path / "gliquid_cache").read_text() == "not a directory"


def test_implicit_store_ends_on_set_cache_dir_or_direct_assignment(installed_pkg, tmp_path):
    """Implicit right after import; not after a direct assignment of cache_dir; not after
    set_cache_dir, even to the same path."""
    code = (
        "import json\n"
        "import gliquid.config as config\n"
        "default = config.cache_dir\n"
        "state = {'after_import': config._cache_dir_is_implicit()}\n"
        "config.cache_dir = default.parent / 'elsewhere'\n"
        "state['after_assignment'] = config._cache_dir_is_implicit()\n"
        "config.cache_dir = default\n"
        "state['after_restoring'] = config._cache_dir_is_implicit()\n"
        "config.set_cache_dir(default)\n"
        "state['after_set_cache_dir'] = config._cache_dir_is_implicit()\n"
        "state['same_path'] = config.cache_dir == default\n"
        "print(json.dumps(state))\n"
    )
    probe = _json_probe(code, tmp_path, installed_pkg)
    assert probe == {
        "after_import": True,
        "after_assignment": False,
        "after_restoring": True,
        "after_set_cache_dir": False,
        "same_path": True,
    }


def test_direct_assignment_of_cache_dir_is_not_implicit(tmp_path, monkeypatch):
    """A cache_dir assigned directly, bypassing set_cache_dir, is never the implicit store."""
    import gliquid.config as config

    monkeypatch.setattr(config, "cache_dir", tmp_path / "other")
    assert config._cache_dir_is_implicit() is False


def test_cache_dir_env_var_expands_the_home_directory(installed_pkg, tmp_path):
    """GLIQUID_CACHE_DIR=~/<name> resolves under the home directory."""
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import gliquid.config as config\n"
        "print(json.dumps({'cache_dir': str(config.cache_dir),\n"
        "                  'expected': str(Path.home() / 'gliquid_probe_x')}))\n"
    )
    probe = _json_probe(
        code, tmp_path, installed_pkg, env_extra={"GLIQUID_CACHE_DIR": "~/gliquid_probe_x"}
    )
    assert probe["cache_dir"] == probe["expected"]


def test_set_cache_dir_expands_the_home_directory(monkeypatch):
    """set_cache_dir('~/<name>') resolves under the home directory."""
    import gliquid.config as config

    for name in (
        "cache_dir",
        "cache_mode",
        "phase_transitions_file",
        "omegas_file",
        "spurious_structures_file",
    ):
        monkeypatch.setattr(config, name, getattr(config, name))
    config.set_cache_dir("~/gliquid_probe_x")
    assert config.cache_dir == Path.home() / "gliquid_probe_x"


# -- set_cache_dir semantics under the two-root split --


def test_explicit_set_cache_dir_wins(tmp_path):
    """set_cache_dir/set_project_root overrides are the API for custom stores."""
    import gliquid.config as config

    old_root, old_dir = config.project_root, config.cache_dir
    try:
        config.set_cache_dir(tmp_path)
        assert config.cache_dir == tmp_path
    finally:
        config.set_project_root(old_root)
        config.set_cache_dir(old_dir)


def test_reference_files_fall_back_to_bundled_for_a_partial_corpus(tmp_path):
    """Pointing at a corpus directory with no reference tables still yields working ones:
    `cache_dir/<name>` IF that file exists, else bundled.
    """
    import gliquid.config as config

    old_dir = config.cache_dir
    try:
        config.set_cache_dir(tmp_path)  # empty directory
        bundled = _bundled_dir()
        assert Path(config.phase_transitions_file) == bundled / "phase_transitions.json"
        assert Path(config.omegas_file) == bundled / "omegas_hcp.json"
        assert Path(config.spurious_structures_file) == bundled / "spurious_structures.json"
    finally:
        config.set_cache_dir(old_dir)


def test_reference_files_prefer_a_copy_present_in_the_cache_dir(tmp_path):
    """A cache dir carrying its own reference table overrides the shipped one.

    This is how the unary database is iterated on: drop an edited phase_transitions.json
    next to the caches and it wins.
    """
    import gliquid.config as config

    (tmp_path / "phase_transitions.json").write_text('{"elements": {}}')
    old_dir = config.cache_dir
    try:
        config.set_cache_dir(tmp_path)
        assert Path(config.phase_transitions_file) == tmp_path / "phase_transitions.json"
        # the two absent ones still come from the package -- the check is per file
        assert Path(config.omegas_file) == _bundled_dir() / "omegas_hcp.json"
    finally:
        config.set_cache_dir(old_dir)


def test_set_omegas_file_still_overrides(tmp_path):
    """``set_omegas_file`` overrides the omegas file until ``set_cache_dir`` re-resolves it."""
    import gliquid.config as config

    old_omegas, old_dir = config.omegas_file, config.cache_dir
    try:
        custom = tmp_path / "my_omegas.json"
        config.set_omegas_file(custom)
        assert config.omegas_file == custom
        # ...but set_cache_dir re-resolves all three, which is its documented job
        config.set_cache_dir(tmp_path)
        assert Path(config.omegas_file) == _bundled_dir() / "omegas_hcp.json"
    finally:
        config.set_cache_dir(old_dir)
        config.omegas_file = old_omegas


def test_require_cache_dir_returns_the_corpus_when_configured(tmp_path):
    import gliquid.config as config

    old_dir = config.cache_dir
    try:
        config.set_cache_dir(tmp_path)
        assert config.require_cache_dir() == tmp_path
        config.set_cache_dir(None)
        with pytest.raises(config.ConfigError, match="GLIQUID_CACHE_DIR"):
            config.require_cache_dir()
    finally:
        config.set_cache_dir(old_dir)


@_NAMED_CHECKOUT
def test_find_project_root_returns_none_for_an_unknown_dirname():
    """No match is None, not the working directory."""
    import gliquid.config as config

    assert config.find_project_root() == _PKG_ROOT
    assert config.find_project_root(dirname="not_a_real_ancestor_name") is None


# -- Registry load behavior --


def test_empty_registry_load_warns(tmp_path):
    """An unreadable phase_transitions.json loads loudly, not silently.

    The file is overridden directly: an empty cache dir falls back to the bundled table and
    never produces an empty registry.
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
    """Shipped defaults for the solid-energy coverage gate.

    The verdict is ``min_supported_total``: a floor on how much supported liquidus there is
    in TOTAL, as a fraction of the whole composition axis. Support that arrives in several
    pieces still constrains the liquid parameters, so the widest single run does not gate.

    ``subliquidus_margin`` ships at 0 K -- the strict form of the rule that an unsupported
    phase masks nothing unless its digitized field reaches the liquidus. It is kelvin, not a
    fraction, and ``math.inf`` is the documented off switch.
    """
    import gliquid.config as config

    assert config.coverage_min_supported_total == 0.30

    assert config.coverage_subliquidus_margin == 0.0
    assert config.coverage_ss_narrow_tol == 0.10
    assert config.coverage_dft_cover_tol == 0.10
    assert config.coverage_dft_edge_tol == 0.01
    assert config.coverage_dft_structure_tol == 0.05
    # Load-bearing: uncapped, one DFT compound would rescue an arbitrarily wide solid-solution
    # field and complete solid solutions with no models would read as fully supported.
    assert config.coverage_ss_rescue_max_width == 0.25
    # An unsupported component solution field masks at most this far past its drawn bounds,
    # unless the drawn fields together cover ss_width_guard of the axis.
    assert config.coverage_ss_extension_cap == 0.05
    assert config.coverage_ss_width_guard == 0.67


def test_set_coverage_thresholds_partial_update_and_validation():
    """Only the passed arguments change; fractions must lie in (0, 1].

    ``subliquidus_margin`` is the one knob here that is NOT a fraction -- it is kelvin below
    the digitized liquidus minimum, unbounded above, with ``math.inf`` the documented
    "never exclude a phase" setting -- so it carries its own >= 0 check.
    """
    import math

    import gliquid.config as config

    old = (
        config.coverage_ss_narrow_tol,
        config.coverage_dft_cover_tol,
        config.coverage_ss_rescue_max_width,
        config.coverage_min_supported_total,
        config.coverage_subliquidus_margin,
        config.coverage_dft_edge_tol,
        config.coverage_dft_structure_tol,
    )
    try:
        config.set_coverage_thresholds(dft_edge_tol=0, dft_structure_tol=0)
        assert config.coverage_dft_edge_tol == 0.0
        assert config.coverage_dft_structure_tol == 0.0
        config.set_coverage_thresholds(dft_structure_tol=0.07)
        assert config.coverage_dft_structure_tol == 0.07
        assert config.coverage_dft_edge_tol == 0.0  # untouched

        config.set_coverage_thresholds(ss_narrow_tol=0.2)
        assert config.coverage_ss_narrow_tol == 0.2
        assert config.coverage_dft_cover_tol == old[1]  # untouched

        config.set_coverage_thresholds(ss_rescue_max_width=0.4)
        assert config.coverage_ss_rescue_max_width == 0.4

        config.set_coverage_thresholds(min_supported_total=0.4)
        assert config.coverage_min_supported_total == 0.4
        # 0 is the "assess but do not gate" setting, and is admissible only for this knob.
        config.set_coverage_thresholds(min_supported_total=0)
        assert config.coverage_min_supported_total == 0.0

        config.set_coverage_thresholds(subliquidus_margin=25)
        assert config.coverage_subliquidus_margin == 25.0
        config.set_coverage_thresholds(subliquidus_margin=0)
        assert config.coverage_subliquidus_margin == 0.0
        config.set_coverage_thresholds(subliquidus_margin=math.inf)
        assert config.coverage_subliquidus_margin == math.inf

        for kwargs in (
            {"ss_narrow_tol": 0},
            {"ss_narrow_tol": 1.5},
            {"dft_cover_tol": 2},
            {"ss_rescue_max_width": 0},
            {"ss_width_guard": 0},
            {"min_supported_total": -0.1},
            {"min_supported_total": 1.5},
            {"subliquidus_margin": -1},
            {"dft_edge_tol": -0.01},
            {"dft_edge_tol": 1.5},
            {"dft_structure_tol": -0.01},
            {"dft_structure_tol": 1.5},
        ):
            with pytest.raises(ValueError):
                config.set_coverage_thresholds(**kwargs)
    finally:
        config.set_coverage_thresholds(
            ss_narrow_tol=old[0],
            dft_cover_tol=old[1],
            ss_rescue_max_width=old[2],
            min_supported_total=old[3],
            subliquidus_margin=old[4],
            dft_edge_tol=old[5],
            dft_structure_tol=old[6],
        )


@pytest.mark.parametrize("name", ["skip_frac", "min_missing", "missing_frac"])
def test_removed_coverage_knobs_raise_naming_the_verdict(name):
    import gliquid.config as config

    with pytest.raises(TypeError, match=f"'{name}' \\(use min_supported_total\\)"):
        config.set_coverage_thresholds(**{name: 0.5})
    with pytest.raises(ValueError, match="use min_supported_total"):
        config.coverage_settings({name: 0.5})


class TestCoverageSettings:
    def test_overrides_layer_on_config(self):
        import gliquid.config as config

        got = config.coverage_settings({"ss_extension_cap": 1, "dft_cover_tol": None})
        assert set(got) == set(config._COVERAGE_KNOBS)
        assert got["ss_extension_cap"] == 1.0 and isinstance(got["ss_extension_cap"], float)
        assert got["dft_cover_tol"] == config.coverage_dft_cover_tol

    def test_a_config_only_knob_raises(self):
        import gliquid.config as config

        with pytest.raises(ValueError, match="low_temp_axis_fraction cannot be overridden"):
            config.coverage_settings({"low_temp_axis_fraction": 0.1})

    @pytest.mark.parametrize("overrides", [{"dft_edge_tolerance": 0.1}, {"min_supported_total": 2}])
    def test_an_unknown_knob_or_bad_value_raises(self, overrides):
        import gliquid.config as config

        with pytest.raises(ValueError):
            config.coverage_settings(overrides)


class TestCoverageDefault:
    """``config.coverage_default(name, value)``: ``value`` unchanged unless it is None, then
    the current ``config.coverage_<name>``; an unknown name raises."""

    @pytest.mark.parametrize("value", [0, 0.0, False, 0.125], ids=repr)
    def test_a_given_value_is_returned_as_the_same_object(self, value):
        import gliquid.config as config

        assert config.coverage_default("dft_edge_tol", value) is value

    def test_none_reads_the_config_value_at_call_time(self, monkeypatch):
        import gliquid.config as config

        for knob in sorted(config._COVERAGE_KNOBS):
            sentinel = object()
            monkeypatch.setattr(config, f"coverage_{knob}", sentinel)
            assert config.coverage_default(knob, None) is sentinel

    def test_none_follows_set_coverage_thresholds(self, monkeypatch):
        import gliquid.config as config

        monkeypatch.setattr(config, "coverage_dft_edge_tol", config.coverage_dft_edge_tol)
        config.set_coverage_thresholds(dft_edge_tol=0.02)
        assert config.coverage_default("dft_edge_tol", None) == 0.02

    @pytest.mark.parametrize("value", [None, 0.1], ids=repr)
    @pytest.mark.parametrize("name", ["dft_edge_tolerance", "coverage_dft_edge_tol"])
    def test_an_unknown_name_raises_whatever_the_value(self, name, value):
        import gliquid.config as config

        with pytest.raises(ValueError, match=f"unknown coverage knob '{name}'"):
            config.coverage_default(name, value)

    @pytest.mark.parametrize("value", [None, 1], ids=repr)
    def test_default_is_not_a_knob(self, value):
        import gliquid.config as config

        with pytest.raises(ValueError, match="unknown coverage knob 'default'"):
            config.coverage_default("default", value)

    def test_the_knobs_are_the_keyword_parameters_of_set_coverage_thresholds(self):
        import inspect

        import gliquid.config as config

        params = inspect.signature(config.set_coverage_thresholds).parameters
        keywords = {n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
        assert keywords == set(params)
        assert set(config._COVERAGE_KNOBS) == keywords
        for knob in keywords:
            assert f"coverage_{knob}" in vars(config)


@pytest.fixture
def restore_cache_dir():
    saved_dir, saved_mode = config.cache_dir, config.cache_mode
    try:
        yield
    finally:
        config.set_cache_dir(saved_dir)
        config.set_cache_mode(saved_mode)


# -- cache_mode, and its orthogonality to dir_structure --


class TestCacheMode:
    @pytest.mark.directory_only  # asserts the UNSWAPPED process default
    def test_defaults_to_directory(self):
        assert config.cache_mode == "directory"

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="cache_mode"):
            config.set_cache_mode("parquet")

    @pytest.mark.parametrize("name", ["corpus.sqlite", "corpus.sqlite3", "corpus.db"])
    def test_set_cache_dir_infers_sqlite_from_a_single_file_path(
        self, tmp_path, restore_cache_dir, name
    ):
        """Being handed one file and having it just work is the point of the format."""
        config.set_cache_dir(tmp_path / name)
        assert config.cache_mode == "sqlite"

    def test_set_cache_dir_infers_sqlite_from_an_existing_file(self, tmp_path, restore_cache_dir):
        store = tmp_path / "store_without_a_telling_suffix"
        store.write_bytes(b"")
        config.set_cache_dir(store)
        assert config.cache_mode == "sqlite"

    def test_a_directory_gives_directory_mode(self, tmp_path, restore_cache_dir):
        """POSITIVE CONTROL: the inference above is not simply always answering sqlite."""
        config.set_cache_dir(tmp_path)
        assert config.cache_mode == "directory"

    def test_set_dir_structure_under_sqlite_logs_and_does_not_raise(
        self, tmp_path, restore_cache_dir, caplog
    ):
        """``set_dir_structure()`` on a single-file store logs and is ignored, so callers
        that set it unconditionally keep working."""
        import logging

        config.set_cache_dir(tmp_path / "corpus.sqlite")
        before = config.dir_structure
        with caplog.at_level(logging.INFO, logger=config.logger.name):
            config.set_dir_structure("nested")  # must NOT raise
        assert config.dir_structure == before, "the ignored setting must not take effect"
        assert "sqlite" in caplog.text

    def test_set_dir_structure_still_rejects_a_typo_under_sqlite(self, tmp_path, restore_cache_dir):
        """Ignoring an inapplicable setting is not the same as accepting a misspelled one."""
        config.set_cache_dir(tmp_path / "corpus.sqlite")
        with pytest.raises(ValueError, match="dir_structure"):
            config.set_dir_structure("nsted")

    def test_dir_structure_still_works_in_directory_mode(self, tmp_path, restore_cache_dir):
        config.set_cache_dir(tmp_path)
        saved = config.dir_structure
        try:
            config.set_dir_structure("nested")
            assert config.dir_structure == "nested"
        finally:
            config.set_dir_structure(saved)
