"""The ML stack is an OPTIONAL extra.

XGBoost, SHAP, joblib and scikit-learn are reachable from exactly one module,
``gliquid.production_model_runner``. An eager import there, or a line back in
``[project.dependencies]``, breaks nothing in an environment that already has the packages,
so the checks here read:

* **The declared metadata** from the INSTALLED distribution rather than from
  ``pyproject.toml``, so a stale build fails too.
* **A subprocess with the package blocked at import**, since in-process ``sys.modules`` is
  already populated.

Each case asserts BOTH halves: the module still imports, and the failure when the extra is
genuinely needed names the extra rather than raising a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from ._fresh import ok, run_fresh


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
EXPECTED_EXTRAS = {"ml", "mpds", "test", "notebook"}


def _run_blocked(
    blocked: tuple[str, ...], body: str, tmp_path: Path
) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter from ``tmp_path`` with ``blocked`` unimportable
    and ``GLIQUID_CACHE_DIR`` unset."""
    return run_fresh(body, tmp_path, blocked=blocked, env={"GLIQUID_CACHE_DIR": None})


# -- What the distribution DECLARES --


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
        unconditional.
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
            f"belong in the `ml` extra or an explicit install."
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


# -- What happens with the extras genuinely ABSENT --


class TestImportsWithoutTheMlStack:
    #: ``joblib`` is NOT blocked: the newest pymatgen imports it eagerly through
    #: ``pymatgen.analysis.compatibility``, which ``gliquid.api`` reaches, so ``import gliquid``
    #: does not survive without it there. gliquid still declares joblib under ``models``,
    #: since ``_load_legacy_artifacts`` imports it directly.
    BLOCKED_FOR_FACADE = ("xgboost", "shap", "sklearn", "numba")

    def test_importing_gliquid_needs_none_of_it(self, tmp_path):
        """The façade is lazy; naming the export must not drag the stack in either."""
        blocked = self.BLOCKED_FOR_FACADE
        out = ok(
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
        ok(
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
        ok(
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
        ok(
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
                assert "pip install joblib openpyxl scikit-learn" in message, message
                print("OK")
                """,
                tmp_path,
            )
        )

    def test_shap_explanations_without_shap_name_the_shap_install(self, tmp_path):
        """And the PREDICTION path still works in the same process — that is the split."""
        ok(
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
                assert "pip install shap" in message, message
                print("OK")
                """,
                tmp_path,
            )
        )


class TestShapPatchesStillApply:
    """``apply_patches`` runs inside the ``_shap()`` accessor, which every SHAP call site in
    the runner goes through, so it precedes any TreeExplainer as ``shap_compat`` documents.
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
        """Importing the runner module loads none of shap, xgboost and matplotlib."""
        ok(
            _run_blocked(
                (),
                """
                import sys
                assert "shap" not in sys.modules
                import gliquid.production_model_runner  # noqa: F401
                assert "shap" not in sys.modules, "importing the runner module imported shap"
                assert "xgboost" not in sys.modules, "importing the runner module imported xgboost"
                assert "matplotlib" not in sys.modules, "matplotlib imported by the runner"
                print("OK")
                """,
                tmp_path,
            )
        )
