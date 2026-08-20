"""The ML stack is an OPTIONAL extra — pinned here, because nothing else can catch it.

XGBoost, SHAP, joblib and scikit-learn are reachable from exactly one module,
``gliquid.production_model_runner``. As base dependencies they cost ~935 MB of
site-packages on an image whose app never constructs a runner. The split is easy to make
and easy to undo by accident: adding ``import xgboost`` back to the top of that module, or
one line back to ``[project.dependencies]``, restores the whole payload and breaks nothing
in an environment that already has the packages — which is every developer environment.
So the checks here run the way that failure would have to be seen:

* **The declared metadata**, read from the INSTALLED distribution rather than from
  ``pyproject.toml``, so it also catches a stale build.
* **In a subprocess with the package blocked at import**, because an in-process test cannot
  observe an eager import in an environment that has the module — ``sys.modules`` is
  already populated by the time the test runs. This is the same technique
  ``test_portable_model_bundle.py`` uses for ``openpyxl``, and for the same reason.

The claim in each case is BOTH halves: that the module still imports, and that the failure
when the extra is genuinely needed names the extra rather than raising a bare
``ModuleNotFoundError`` at whatever depth it happens to surface.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _dist_name(requirement: str) -> str:
    """The bare distribution name of a core-metadata ``Requires-Dist`` string.

    ``xgboost-cpu<4,>=2.1.1; sys_platform != "darwin" and extra == "ml"`` -> ``xgboost-cpu``.
    """
    head = requirement.split(";")[0]
    for separator in (">=", "==", "<", ">", "!=", "~=", "[", " ", "("):
        head = head.split(separator)[0]
    return head.strip().lower()

#: Distributions that must not be reachable from a bare ``pip install gliquid``.
ML_DISTRIBUTIONS = ("xgboost", "xgboost-cpu", "shap", "scikit-learn", "joblib", "numba")

#: Extras that must exist, and the runner capability each one buys.
EXPECTED_EXTRAS = {"ml", "ml-gpu", "shap", "models", "mpds", "editor", "test", "notebook"}


def _run_blocked(blocked: tuple[str, ...], body: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess where importing any of ``blocked`` raises ImportError.

    The block is asserted to have TAKEN before the body runs; without that the test passes
    just as happily in an environment where the module was never importable for some other
    reason, and would keep passing after the guard it checks was deleted.

    cwd is a tmp_path outside the repo: ``gliquid.config`` walks out from the package's own
    ``__file__`` to adopt a ``cache/``, and several of these bodies want no cache.
    """
    preamble = textwrap.dedent(
        f"""
        import builtins, sys
        _BLOCKED = {tuple(blocked)!r}
        _real_import = builtins.__import__
        def _blocker(name, *args, **kwargs):
            if name.split(".")[0] in _BLOCKED:
                raise ImportError(name + " is blocked for this test")
            return _real_import(name, *args, **kwargs)
        builtins.__import__ = _blocker
        # Drop anything already imported, or a cached module would satisfy an import that
        # the blocker never sees and the block would be half-applied.
        for _name in [m for m in sys.modules if m.split(".")[0] in _BLOCKED]:
            del sys.modules[_name]
        for _name in _BLOCKED:
            try:
                __import__(_name)
            except ImportError:
                pass
            else:
                raise AssertionError("the " + _name + " block did not take; test is vacuous")
        """
    )
    env = os.environ.copy()
    for name in ("GLIQUID_CACHE_DIR", "GLIQUID_DATA_DIR"):
        env.pop(name, None)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )


def _ok(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
    return result.stdout


# =======================================================================================
# What the distribution DECLARES
# =======================================================================================


class TestDeclaredDependencies:
    """Read from the installed distribution, so a stale build fails rather than passes."""

    @staticmethod
    def _requirements() -> list[str]:
        reqs = metadata.requires("gliquid")
        assert reqs, (
            "gliquid declares no requirements at all, which cannot be right. If this is an "
            "editable install whose metadata predates the pyproject change, reinstall it: "
            "`pip install -e .`"
        )
        return reqs

    @staticmethod
    def _is_unconditional(req: str) -> bool:
        """True when ``req`` applies to a plain ``pip install gliquid``.

        A requirement carrying ``extra == "..."`` is opt-in; anything else is not. Markers
        that merely narrow the PLATFORM (``sys_platform != 'darwin'``) still count as
        unconditional, which is the point — a Linux-only base dependency is exactly the
        shape of the payload this test exists to keep out.
        """
        return 'extra == "' not in req and "extra == '" not in req

    def test_no_ml_distribution_is_a_base_dependency(self):
        offenders = [
            req
            for req in self._requirements()
            if self._is_unconditional(req) and _dist_name(req) in ML_DISTRIBUTIONS
        ]
        assert not offenders, (
            f"these are declared as BASE dependencies: {offenders}. Every one of them is "
            f"reachable only from gliquid.production_model_runner, and together they cost "
            f"~935 MB of site-packages to a consumer that never constructs a runner. They "
            f"belong in the `ml` / `shap` / `models` extras."
        )

    def test_the_base_dependency_set_is_the_expected_one(self):
        """A positive control: the check above would also pass on an EMPTY requirement list."""
        base = {_dist_name(req) for req in self._requirements() if self._is_unconditional(req)}
        assert base == {
            "plotly",
            "emmet-core",
            "mp-api",
            "pymatgen",
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "sympy",
            "tqdm",
        }, sorted(base)

    def test_the_extras_are_declared(self):
        declared = set(metadata.metadata("gliquid").get_all("Provides-Extra") or [])
        missing = EXPECTED_EXTRAS - declared
        assert not missing, f"missing extras {sorted(missing)}; declared: {sorted(declared)}"

    def test_ml_and_models_agree_on_one_xgboost_distribution(self):
        """``gliquid[ml,models]`` must not resolve to xgboost AND xgboost-cpu at once.

        Both install a package directory named ``xgboost``; an environment holding both
        keeps whichever pip wrote last, and uninstalling either breaks the other.
        """
        by_extra: dict[str, set[str]] = {}
        for req in self._requirements():
            if "extra ==" not in req:
                continue
            extra = req.split("extra ==")[1].strip().strip("\"'")
            name = _dist_name(req)
            if name.startswith("xgboost"):
                by_extra.setdefault(extra, set()).add(name)
        assert by_extra.get("ml") == by_extra.get("models"), (
            f"`ml` asks for {sorted(by_extra.get('ml') or [])} and `models` for "
            f"{sorted(by_extra.get('models') or [])}; installing both extras together would "
            f"put two conflicting xgboost distributions in one environment."
        )


# =======================================================================================
# What happens with the extras genuinely ABSENT
# =======================================================================================


class TestImportsWithoutTheMlStack:
    #: ``joblib`` is deliberately NOT blocked here, and the reason is measured rather than
    #: assumed. It is not optional for gliquid at all: ``pymatgen/analysis/compatibility/
    #: __init__.py`` runs ``from joblib import Parallel, delayed`` at import time, and
    #: ``gliquid.api`` imports ``pymatgen.entries.computed_entries``, which re-exports from
    #: it — so every gliquid install has joblib whatever gliquid declares (``pymatgen-core``
    #: requires ``joblib>=1.3.2``). The compat matrix is what settled it: blocking joblib
    #: here passed on py310 and on every -min env and failed on py311/312/313-max, because
    #: only the newest pymatgen imports it eagerly. gliquid still DECLARES joblib under
    #: `models`, since _load_legacy_artifacts imports it directly and a transitive
    #: dependency is not a contract — but a test asserting `import gliquid` survives without
    #: it would be asserting something false about the dependency graph.
    BLOCKED_FOR_FACADE = ("xgboost", "shap", "sklearn", "numba")

    def test_importing_gliquid_needs_none_of_it(self, tmp_path):
        """The façade is lazy; naming the export must not drag the stack in either."""
        blocked = self.BLOCKED_FOR_FACADE
        out = _ok(
            _run_blocked(
                blocked,
                f"""
                import gliquid
                assert "ProductionModelRunner" in dir(gliquid)
                assert gliquid.BinaryLiquid is not None
                assert gliquid.lower_convex_hull is not None
                import sys
                leaked = sorted(
                    m for m in sys.modules if m.split(".")[0] in {set(blocked)!r}
                )
                assert not leaked, leaked
                print("OK")
                """,
                tmp_path,
            )
        )
        assert "OK" in out

    def test_the_runner_module_imports_without_the_ml_stack(self, tmp_path):
        """Importing the module is what must not raise. Constructing is what raises."""
        _ok(
            _run_blocked(
                ("xgboost", "shap", "sklearn", "joblib"),
                """
                from gliquid.production_model_runner import (
                    DEFAULT_BUNDLE_ID,
                    ProductionModelRunner,
                    default_bundle_dir,
                    derive_frame,
                    golden_digest,
                )
                # Module-level work that touches no optional dependency still works.
                assert default_bundle_dir().name == DEFAULT_BUNDLE_ID
                assert derive_frame(2, {"symmetric": ["a", "b"], "antisymmetric": ["c"]}) == "symmetric"
                assert golden_digest([1.0, 2.0])
                print("OK")
                """,
                tmp_path,
            )
        )


class TestErrorsNameTheirExtra:
    def test_constructing_without_xgboost_names_the_ml_extra(self, tmp_path):
        _ok(
            _run_blocked(
                ("xgboost",),
                """
                from gliquid.production_model_runner import ProductionModelRunner
                try:
                    ProductionModelRunner()
                except ImportError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("constructing succeeded with xgboost blocked")
                assert "gliquid[ml]" in message, message
                assert "XGBoost" in message, message
                print("OK")
                """,
                tmp_path,
            )
        )

    def test_a_legacy_bundle_without_joblib_names_the_models_extra(self, tmp_path):
        """joblib is blocked but xgboost is not, so this reaches the legacy branch."""
        bundle = tmp_path / "legacy"
        (bundle / "model").mkdir(parents=True)
        (bundle / "model" / "L0_a_model.joblib").write_bytes(b"not really a pickle")
        _ok(
            _run_blocked(
                ("joblib",),
                f"""
                from gliquid.production_model_runner import ProductionModelRunner
                try:
                    ProductionModelRunner(r{str(bundle)!r})
                except ImportError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("constructing succeeded with joblib blocked")
                assert "gliquid[models]" in message, message
                print("OK")
                """,
                tmp_path,
            )
        )

    def test_shap_explanations_without_shap_name_the_shap_extra(self, tmp_path):
        """And the PREDICTION path still works in the same process — that is the split."""
        _ok(
            _run_blocked(
                ("shap",),
                """
                import numpy as np, pandas as pd
                import gliquid.config as config
                config.set_cache_dir(None)

                from gliquid.production_model_runner import ProductionModelRunner
                runner = ProductionModelRunner()

                rng = np.random.default_rng(3)
                symm = pd.DataFrame(
                    rng.random((1, len(runner.feature_names["symmetric"]))),
                    columns=runner.feature_names["symmetric"],
                )
                anti = pd.DataFrame(
                    rng.random((1, len(runner.feature_names["antisymmetric"]))),
                    columns=runner.feature_names["antisymmetric"],
                )
                values = runner.predict_from_dataframes(symm, anti)
                assert len(values) == 3 and all(np.isfinite(values)), values

                try:
                    runner.get_shap_explanation("L0_a", symm[runner.feature_names["symmetric"]])
                except ImportError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("get_shap_explanation succeeded with shap blocked")
                assert "gliquid[shap]" in message, message
                print("OK")
                """,
                tmp_path,
            )
        )


class TestShapPatchesStillApply:
    """Deferring ``apply_patches`` must not mean it stops happening.

    It used to run at module import, while ``shap`` was a base dependency. It now runs
    inside the ``_shap()`` accessor — which every SHAP call site in the runner goes through
    — so the ordering ``shap_compat`` documents ("before any TreeExplainer is created") is
    preserved. Without this, the deferral would silently drop the ``base_score`` fix.
    """

    def test_the_patches_are_applied_before_the_first_explainer(self):
        shap = pytest.importorskip("shap")
        import shap.explainers._tree as tree_mod

        from gliquid import shap_compat
        from gliquid.production_model_runner import ProductionModelRunner, _shap

        # The patch is an ASSIGNMENT into the module namespace and unpatching DELETES it,
        # so "off" is the attribute being absent -- LOAD_GLOBAL then falls back to builtins.
        shap_compat.unpatch_xgb_base_score()
        assert getattr(tree_mod, "float", float) is float, "precondition: the patch is off"

        assert _shap() is shap
        assert getattr(tree_mod, "float", float) is not float, (
            "_shap() did not apply the base_score patch; deferring the import must not mean "
            "dropping the patches that used to run beside it at module import"
        )

        runner = ProductionModelRunner()
        runner._ensure_explainer(runner.targets[0])
        assert getattr(tree_mod, "float", float) is not float

    def test_importing_the_runner_module_no_longer_imports_shap(self, tmp_path):
        """The deferral itself, stated as a fact about sys.modules rather than about code."""
        _ok(
            _run_blocked(
                (),
                """
                import sys
                assert "shap" not in sys.modules
                import gliquid.production_model_runner  # noqa: F401
                assert "shap" not in sys.modules, "importing the runner module imported shap"
                assert "xgboost" not in sys.modules, "importing the runner module imported xgboost"
                print("OK")
                """,
                tmp_path,
            )
        )
