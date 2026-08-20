"""The ``data_*`` -> ``cache_*`` rename, and the deprecated aliases that keep 0.1.0 working.

0.1.0 is public and the old names are load-bearing outside this repo: hundreds of driver
scripts under ``dev/scripts``, the notebooks, and both test suites spell the corpus
``data_dir``. The rename is therefore additive — every old name keeps working and says so
once.

The alias that is easy to get WRONG is the module attribute. A bare PEP 562 module-level
``__getattr__`` fires only for MISSING attributes, so the first ``config.data_dir = X``
(real callers do this: ``tests_internal/test_dft_data_loading.py``, and every
``monkeypatch.setattr(config, "data_dir", ...)``) plants a genuine global that SHADOWS the
alias. Reads of the two names then disagree forever and nothing raises. The write path is
tested here explicitly, not just the read path.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import gliquid.config as config

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src"


@pytest.fixture
def fresh_warnings():
    """Forget which deprecations have already fired, so ``pytest.warns`` can see them.

    The warnings are once-per-process by design (``config.data_dir`` is read inside loops
    over thousands of systems); a test that wants to observe one has to reset the ledger.
    """
    saved = set(config._DEPRECATION_WARNED)
    config._DEPRECATION_WARNED.clear()
    try:
        yield
    finally:
        config._DEPRECATION_WARNED.clear()
        config._DEPRECATION_WARNED.update(saved)


@pytest.fixture
def restore_cache_dir():
    saved_dir, saved_mode = config.cache_dir, config.cache_mode
    try:
        yield
    finally:
        config.set_cache_dir(saved_dir)
        config.set_cache_mode(saved_mode)


# ---------------------------------------------------------------------------------------
# The two names are ONE variable, in both directions
# ---------------------------------------------------------------------------------------


class TestDataDirIsAnAliasNotACopy:
    def test_writing_data_dir_is_observable_as_cache_dir(self, tmp_path, restore_cache_dir):
        """The ``__setattr__`` path — the one a module ``__getattr__`` alone cannot serve."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config.data_dir = tmp_path
            assert config.cache_dir == tmp_path
            # ...and the alias did NOT become a shadowing global of its own
            assert "data_dir" not in config.__dict__

    def test_writing_cache_dir_is_observable_as_data_dir(self, tmp_path, restore_cache_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config.cache_dir = tmp_path
            assert config.data_dir == tmp_path

    def test_repeated_writes_do_not_diverge(self, tmp_path, restore_cache_dir):
        """The exact failure mode of a bare module ``__getattr__``: silent divergence."""
        first, second = tmp_path / "a", tmp_path / "b"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config.data_dir = first
            config.cache_dir = second
            assert config.data_dir == second
            config.data_dir = first
            assert config.cache_dir == first

    def test_monkeypatch_setattr_round_trips(self, tmp_path, monkeypatch, restore_cache_dir):
        """``monkeypatch.setattr(config, 'data_dir', ...)`` is used in the suites verbatim."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            before = config.cache_dir
            monkeypatch.setattr(config, "data_dir", tmp_path)
            assert config.cache_dir == tmp_path
            monkeypatch.undo()
            assert config.cache_dir == before

    def test_set_data_dir_updates_cache_dir(self, tmp_path, restore_cache_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config.set_data_dir(tmp_path)
        assert config.cache_dir == tmp_path

    def test_unknown_attribute_still_raises(self):
        with pytest.raises(AttributeError):
            config.not_a_real_setting  # noqa: B018


# ---------------------------------------------------------------------------------------
# Each deprecated spelling warns, and each still works
# ---------------------------------------------------------------------------------------


class TestDeprecationWarnings:
    def test_reading_data_dir_warns(self, fresh_warnings):
        with pytest.warns(DeprecationWarning, match="cache_dir"):
            value = config.data_dir
        assert value == config.cache_dir, "the warning must not cost the right answer"

    def test_set_data_dir_warns(self, fresh_warnings, tmp_path, restore_cache_dir):
        with pytest.warns(DeprecationWarning, match="set_cache_dir"):
            config.set_data_dir(tmp_path)
        assert config.cache_dir == tmp_path

    def test_require_data_dir_warns_and_still_returns_the_corpus(
        self, fresh_warnings, tmp_path, restore_cache_dir
    ):
        config.set_cache_dir(tmp_path)
        with pytest.warns(DeprecationWarning, match="require_cache_dir"):
            assert config.require_data_dir() == tmp_path

    def test_data_dir_env_var_constant_warns_and_keeps_its_own_value(self, fresh_warnings):
        """It names a DIFFERENT variable from ``CACHE_DIR_ENV_VAR``, so it cannot forward."""
        with pytest.warns(DeprecationWarning, match="CACHE_DIR_ENV_VAR"):
            assert config.DATA_DIR_ENV_VAR == "GLIQUID_DATA_DIR"
        assert config.CACHE_DIR_ENV_VAR == "GLIQUID_CACHE_DIR"

    def test_each_name_warns_only_once(self, fresh_warnings):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            for _ in range(5):
                _ = config.data_dir
        assert len([w for w in caught if issubclass(w.category, DeprecationWarning)]) == 1

    def test_the_new_names_do_not_warn(self, tmp_path, restore_cache_dir):
        """POSITIVE CONTROL for the tests above: the warning is attached to the OLD name."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config.set_cache_dir(tmp_path)
            assert config.cache_dir == tmp_path
            assert config.require_cache_dir() == tmp_path


def _probe(code: str, env_extra: dict) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    env.pop("GLIQUID_DATA_DIR", None)
    env.pop("GLIQUID_CACHE_DIR", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-W", "always::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


class TestEnvironmentVariables:
    """Resolution order: ``GLIQUID_CACHE_DIR`` -> ``GLIQUID_DATA_DIR`` (deprecated) -> walk."""

    CODE = "import gliquid.config as c; print('RESOLVED:' + str(c.cache_dir))"

    def test_new_env_var_supplies_the_corpus_without_warning(self, tmp_path):
        result = self._run(tmp_path, {"GLIQUID_CACHE_DIR": str(tmp_path)})
        assert "GLIQUID_DATA_DIR" not in result.stderr

    def test_legacy_env_var_still_supplies_the_corpus_and_warns(self, tmp_path):
        result = self._run(tmp_path, {"GLIQUID_DATA_DIR": str(tmp_path)})
        assert "DeprecationWarning" in result.stderr, result.stderr
        assert "GLIQUID_CACHE_DIR" in result.stderr

    def test_new_env_var_wins_over_the_legacy_one(self, tmp_path):
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        result = _probe(
            self.CODE,
            {"GLIQUID_CACHE_DIR": str(chosen), "GLIQUID_DATA_DIR": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        assert self._resolved(result) == chosen

    def _run(self, tmp_path, env_extra):
        result = _probe(self.CODE, env_extra)
        assert result.returncode == 0, result.stderr
        assert self._resolved(result) == tmp_path
        return result

    @staticmethod
    def _resolved(result) -> Path:
        line = next(x for x in result.stdout.splitlines() if x.startswith("RESOLVED:"))
        return Path(line.split("RESOLVED:", 1)[1])


# ---------------------------------------------------------------------------------------
# cache_mode, and its orthogonality to dir_structure
# ---------------------------------------------------------------------------------------


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
        """20+ dev scripts call set_dir_structure() unconditionally at import.

        Raising here would make a single-file store unusable from every one of them, over a
        setting that simply does not apply to it.
        """
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


class TestTopLevelFacade:
    def test_set_cache_dir_is_importable_from_the_package_namespace(self):
        import gliquid

        assert callable(gliquid.set_cache_dir)

    def test_set_data_dir_is_still_importable(self):
        import gliquid

        assert callable(gliquid.set_data_dir)
