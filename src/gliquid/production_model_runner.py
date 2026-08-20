"""Production runner for the pre-trained L0/L1 models — weights in the wheel, corpus in the cache.

**The split this module exists to make.** A trained bundle is two unrelated things wearing
one name:

* the **model** — three (or four) gradient-boosted regressors plus the affine scalers around
  them. Small, fixed, and useless apart from the code that calls it. It ships INSIDE the
  wheel at ``gliquid/models/<bundle_id>/``.
* the **feature corpus** — one row of model inputs per system, 2,415 symmetric x 51 columns
  and 4,830 antisymmetric x 37. It grows with the corpus, not with the model, and it lives
  where the corpus lives: the cache's ``ml_features`` / ``ml_feature_columns`` tables.

Before that split the runner read both xlsx sheets on EVERY instantiation, through an
unconditional ``pd.read_excel`` — even though only ``get_rows_for_system`` ever touched
them, and ``predict_from_dataframes`` never did. That also made a bare ``pip install
gliquid`` die inside ``__init__``, because ``openpyxl`` is declared only in the ``models``
extra. Constructing a runner now loads weights only; the corpus is loaded lazily, and
``predict_from_dataframes`` works with **no cache, no xlsx and no openpyxl** at all.

**No pickle in the shipped bundle.** A joblib bundle is a pickle graph of
``sklearn.pipeline.Pipeline`` objects, so it unpickles correctly only against the library
versions that wrote it — the reason the ``models`` extra pins ``scikit-learn==1.7.1`` and
``xgboost==3.1.3``, and the reason loading one under a newer scikit-learn emits
``InconsistentVersionWarning``. The portable bundle is XGBoost's own forward-compatible
UBJSON plus JSON for the scaler coefficients, so it carries no version coupling at all.
Bundles are converted by ``dev/scripts/export_portable_model_bundle.py``, which pins the
conversion with a golden vector rather than with a claim.

**That fragility is measured, not asserted.** Running the tox matrix over this module caught
scikit-learn changing its own arithmetic: ``StandardScaler.inverse_transform`` rounds the
scale/mean coefficient to float32 BEFORE multiplying from 1.9.0 onward (the Array-API
migration) and AFTER it in 1.7.x. On 4,096 float32 samples that moves **2,530 of 4,096**
values by up to **9.54e-7**. So a joblib bundle's predictions are not stable across a
scikit-learn upgrade — while the shipped bundle's golden vector, recorded under
scikit-learn 1.7.1, reproduced **bit-for-bit in all eight tox environments** through the
portable path, including under 1.9.0. The op order is recorded per target in
``preprocess.json`` and replayed by :class:`_StandardInverse`, so a bundle keeps predicting
what it predicted at export whichever scikit-learn is installed later — or none at all.

**Legacy bundles still load.** A directory holding ``model/L0_a_model.joblib`` takes the old
path unchanged — joblib pipelines, and the feature corpus read from the bundle's own xlsx —
so the notebooks and ``dev/scripts/Interactive_Matrix_Plotter.py`` keep running against
``dev/model_bundles/`` and ``cache/20260329_022905/`` with no edits.

**The ML stack is an OPTIONAL EXTRA, and this module is where that line is drawn.** Nothing
else in gliquid imports XGBoost, SHAP, joblib or scikit-learn: the hull walk, the phase
diagrams and the ternary interpolation reach none of them. Carrying them as base
dependencies cost every consumer ~935 MB of site-packages for code it never called —
measured on the ``whsun-viz`` image, whose ternary app uses only the hull path and never
constructs a runner: ``nvidia`` (NCCL, pulled by ``xgboost`` on Linux) 454 MB, ``xgboost``
228 MB, ``llvmlite`` 173 MB, ``scikit-learn`` 50 MB, ``numba`` 35 MB. So they moved behind
extras, and the imports below are deferred to the exact call that needs one:

===================  ================  =====================================================
what you are doing   extra             what it actually needs
===================  ================  =====================================================
``import gliquid``   *(none)*          nothing here — the façade is lazy, see ``__init__``
portable bundle      ``ml``            ``xgboost`` only: ``Booster`` + JSON scaler coefficients
SHAP explanations    ``shap``          ``shap`` (which drags ``numba``/``llvmlite``)
legacy joblib bundle ``models``        ``joblib`` + the pinned ``scikit-learn`` to unpickle into
===================  ================  =====================================================

The split is not cosmetic: the *portable* path genuinely needs less than the legacy one.
Its scalers are :class:`_AffineScaler` / :class:`_StandardInverse` — plain NumPy replaying
recorded coefficients — so it never imports scikit-learn at all, and its weights are UBJSON
rather than a pickle graph, so it never imports joblib. ``shap`` is needed by no prediction
path whatsoever, only by the explanation and figure methods.

Each accessor raises an ``ImportError`` naming its extra, following the same pattern
``hull_editor`` uses for ``editor``. ``gliquid/__init__.py`` resolves ``ProductionModelRunner``
lazily, so a bare install can still *name* the export; CONSTRUCTING one is what raises.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gliquid.cache as gliquid_cache
import gliquid.config as config

if TYPE_CHECKING:
    # Annotations only. `from __future__ import annotations` keeps every annotation a
    # string, so nothing below evaluates these names at runtime.
    import shap
    import xgboost

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# Optional ML dependencies -- imported at the call that needs them, never at module import
# ---------------------------------------------------------------------------------------

_ML_EXTRA_HINT = (
    "ProductionModelRunner needs XGBoost, which gliquid ships as an optional extra "
    "(gliquid has no other use for it). Install it with `pip install gliquid[ml]` or "
    "`pip install xgboost`."
)

_SHAP_EXTRA_HINT = (
    "SHAP explanations need the `shap` package, which gliquid ships as an optional extra "
    "because it pulls numba and llvmlite (~200 MB) that no prediction path uses. Install it "
    "with `pip install gliquid[shap]` or `pip install shap`. Predicting needs none of this."
)

_MODELS_EXTRA_HINT = (
    "Loading a LEGACY joblib bundle needs joblib and the pinned scikit-learn it was "
    "pickled against, which gliquid ships as the optional `models` extra. Install it with "
    "`pip install gliquid[models]`. The bundle shipped inside the wheel is pickle-free and "
    "needs none of it: construct ProductionModelRunner() with no arguments."
)


def _xgboost():
    """The ``xgboost`` module, or an ``ImportError`` naming the ``ml`` extra.

    Required by BOTH bundle formats — portable bundles load a ``Booster`` directly, and a
    legacy bundle's pickled pipelines unpickle ``xgboost.sklearn.XGBRegressor`` — so it is
    resolved once in :meth:`ProductionModelRunner._load_model_artifacts` rather than at each
    use. Raising there means a missing extra surfaces as a message naming it, not as a
    ``ModuleNotFoundError`` from inside ``joblib.load``.
    """
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess, see tests
        raise ImportError(_ML_EXTRA_HINT) from exc
    return xgboost


def _shap():
    """The ``shap`` module with :mod:`gliquid.shap_compat` applied, or an ``ImportError``.

    The patches were applied at module import while ``shap`` was a base dependency. They are
    applied here instead, which preserves the ordering ``shap_compat`` documents — "before
    any ``shap.TreeExplainer`` is created or any ``shap.plots.waterfall`` is called" — since
    every such call in this module goes through this accessor first. ``apply_patches`` is
    idempotent, so the repeat calls cost a set lookup.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess, see tests
        raise ImportError(_SHAP_EXTRA_HINT) from exc
    from gliquid.shap_compat import apply_patches

    apply_patches()
    return shap


def _joblib():
    """The ``joblib`` module, or an ``ImportError`` naming the ``models`` extra.

    Reached only by :meth:`ProductionModelRunner._load_legacy_artifacts`. The portable
    bundle in the wheel is UBJSON plus JSON and never unpickles anything.
    """
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess, see tests
        raise ImportError(_MODELS_EXTRA_HINT) from exc
    return joblib

#: The ONE bundle that ships in the wheel. Policy is replace, never accumulate: binary
#: artifacts in git are forever, and two bundles in ``gliquid/models/`` would double the
#: wheel to hold a copy nothing selects. Joblib originals stay outside the package.
#:
#: 0.2.0 repoints this from the v22.02-era 20260329_022905 to 20260817_112204 — trained on
#: the NOSS1 (no-solid-solution) corpus with the ``mean_skill_huber`` objective. Chosen on
#: measured liquidus error, not CV R²: paired over the 744 systems shared with v22.02 and
#: rescored under current code, huber came in at ΔMAPE −0.16 against +1.11 for the v22.02
#: recipe's ``mean_r2`` and +1.16 for ``tail_p90_error``. Same three targets (L1_b is pinned
#: to zero by the comb-exp fit format, so it is not a model).
#:
#: NOTE the previous bundle is still present in ``gliquid/models/`` pending whsun-viz
#: deployment and notebook verification, which violates the replace-never-accumulate policy
#: above for exactly as long as that takes. It must be removed before 0.2.0 is published, or
#: the wheel ships two bundles and 3 MB nothing selects.
DEFAULT_BUNDLE_ID = "20260817_112204"

#: Bumped when the portable bundle layout changes incompatibly. A bundle whose
#: ``manifest.json`` declares a HIGHER schema is refused rather than read with the wrong
#: meanings — the same rule ``cache.SqliteBackend`` applies to a store.
BUNDLE_SCHEMA_VERSION = 1

#: Files a portable bundle must contain, beyond one ``<target>.ubj`` per target.
PORTABLE_REQUIRED = ("manifest.json", "preprocess.json", "feature_names.json")

#: The two feature spaces, named as ``feature_names.json`` and ``_prepare_row`` name them.
FRAMES = ("symmetric", "antisymmetric")

_XLSX_NAMES = {
    "symmetric": "prediction_dataset_symmetric.xlsx",
    "antisymmetric": "prediction_dataset_antisymmetric.xlsx",
}


def default_bundle_dir() -> Path:
    """The in-wheel bundle directory — ``gliquid/models/<DEFAULT_BUNDLE_ID>/``.

    Resolved through ``importlib.resources`` rather than ``__file__`` so it is correct for a
    zipped or relocated install, and returned as a plain ``Path`` because everything
    downstream (``xgboost.Booster.load_model``) wants a filesystem path anyway.
    """
    return Path(str(importlib.resources.files("gliquid") / "models" / DEFAULT_BUNDLE_ID))


def derive_frame(n_features: int, feature_names: dict[str, list[str]]) -> str:
    """Which feature frame a model with ``n_features`` inputs was trained in.

    Derived from the WIDTH rather than from the target's name, because the name rule the old
    runner used (``'antisymmetric' if target == 'L1_a' else 'symmetric'``) is hardcoded to
    one bundle's three targets and silently mislabels a v23 bundle's fourth (``L1_b``).
    Width is a property of the artifact; a name is a convention.

    Raises:
        ValueError: if no frame or more than one frame has that width. Both are ambiguous
            and guessing would send a row of the wrong features into a model that accepts
            them without complaint.
    """
    matches = [frame for frame, names in feature_names.items() if len(names) == n_features]
    if len(matches) == 1:
        return matches[0]
    widths = {frame: len(names) for frame, names in feature_names.items()}
    raise ValueError(
        f"Cannot tell which feature frame a {n_features}-input model belongs to: the "
        f"bundle's frames are {widths}, which gives {len(matches)} match(es). A frame must "
        f"be identifiable by width; two frames of equal width need an explicit mapping."
    )


class _AffineScaler:
    """``(X - center) / scale`` — the portable form of a fitted ``RobustScaler``.

    ``RobustScaler.transform`` is exactly this: ``check_array`` to a float copy, then
    ``X -= center_`` and ``X /= scale_`` in place. Verified ``np.array_equal`` against the
    fitted estimator over the real corpus, which is why replacing it costs nothing
    numerically. The dtype rule mirrors sklearn's ``FLOAT_DTYPES``: float32 and float64
    inputs keep their width, anything else becomes float64.
    """

    def __init__(self, center, scale):
        self.center_ = np.asarray(center, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, X):
        arr = np.asarray(X)
        dtype = arr.dtype if arr.dtype in (np.float32, np.float64) else np.float64
        out = arr.astype(dtype, copy=True)
        out -= self.center_
        out /= self.scale_
        return out


#: The two op orders ``StandardScaler.inverse_transform`` has actually shipped with. Both
#: are "multiply then add, in place, at the input's width"; they differ ONLY in whether the
#: coefficient is rounded to float32 before the arithmetic or after it. See
#: :class:`_StandardInverse`. A bundle records which one reproduces the sklearn that
#: exported it, so a bundle exported on either era stays exactly reproducible on the other.
STANDARD_INVERSE_CONVENTIONS = ("float64-coeff", "float32-coeff")

#: What a bundle that predates the recorded field used — sklearn <= 1.7.x.
DEFAULT_STANDARD_INVERSE_CONVENTION = "float64-coeff"


class _StandardInverse:
    """``z -> y`` for a fitted ``StandardScaler`` — replaying sklearn's op order exactly.

    **This is float32-path-sensitive and the order is load-bearing.** XGBoost predicts in
    float32; sklearn's ``inverse_transform`` runs ``check_array``, which PRESERVES float32
    rather than widening it, then multiplies and adds IN PLACE — so both arithmetic steps
    round to float32, and the second rounds a value that was already rounded. Computing
    ``z.astype(float64) * scale + mean`` instead is the same algebra in a different order
    and drifts up to ~1e-6 relative on the shipped bundle. That is small enough to look
    like noise and large enough to destroy the point of this module: the acceptance
    criterion for a joblib -> portable conversion is ``np.array_equal``, not
    ``np.allclose``, and a conversion that is merely close is one nobody can re-derive.

    **sklearn has shipped TWO such orders, and the difference is measured, not theoretical.**

    ``float64-coeff`` (sklearn <= 1.7.x)::

        X *= self.scale_          # float32 *= float64: numpy computes at float64,
        X += self.mean_           # rounding to float32 once, on the RESULT

    ``float32-coeff`` (sklearn >= 1.9, after the Array-API migration)::

        X *= xp.astype(self.scale_, X.dtype)   # coefficient rounded to float32 FIRST,
        X += xp.astype(self.mean_, X.dtype)    # so the product rounds twice

    Measured across the tox matrix on 4,096 float32 samples: sklearn 1.7.1 and 1.7.2 agree
    with ``float64-coeff`` exactly; sklearn 1.9.0 differs from it on **2,530 of 4,096**
    values, by up to **9.54e-7** — about two float32 ULP. That is sklearn's own output
    moving between releases, which is precisely the fragility a pickled bundle inherits and
    this format exists to escape: the convention is RECORDED in the bundle, so the numbers a
    bundle was exported with stay reproducible whichever sklearn is installed later — or
    whether sklearn is installed at all.
    """

    def __init__(self, mean, scale, convention: str = DEFAULT_STANDARD_INVERSE_CONVENTION):
        if convention not in STANDARD_INVERSE_CONVENTIONS:
            raise ValueError(
                f"Unknown target-inverse convention {convention!r}; this build understands "
                f"{list(STANDARD_INVERSE_CONVENTIONS)}. Re-export the bundle with a matching "
                f"gliquid rather than guessing an op order."
            )
        self.convention = convention
        self.mean_ = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.scale_ = np.asarray(scale, dtype=np.float64).reshape(-1)

    def inverse_transform(self, X):
        arr = np.asarray(X)
        dtype = arr.dtype if arr.dtype in (np.float32, np.float64) else np.float64
        out = arr.astype(dtype, copy=True)
        # In place, in this order, at this width -- see the class docstring.
        if self.convention == "float32-coeff":
            out *= self.scale_.astype(dtype)
            out += self.mean_.astype(dtype)
        else:
            out *= self.scale_
            out += self.mean_
        return out


def detect_standard_inverse_convention(transformer, *, n_probe: int = 4096, seed: int = 0) -> str:
    """Which op order the INSTALLED sklearn's ``inverse_transform`` implements.

    Probed against the fitted ``transformer`` itself rather than inferred from
    ``sklearn.__version__``: a version string says which release is installed, not what its
    arithmetic does, and the two orders differ only at float32 where the answer is decidable
    by measurement in microseconds.

    The probe is float32 because that is the dtype XGBoost predicts in and the only width at
    which the conventions differ at all — probing at float64 would return "both match" and
    record a convention that has never been tested.

    Raises:
        ValueError: when NEITHER convention reproduces the installed sklearn. That means a
            third op order has shipped, and the honest response is to stop: writing a bundle
            whose recorded convention does not reproduce its own source would bake in a
            golden vector the portable path can never match.
    """
    mean = np.asarray(transformer.mean_, dtype=np.float64).reshape(-1)
    scale = np.asarray(transformer.scale_, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    # Spread over several magnitudes: the two orders agree on plenty of individual values,
    # so a narrow probe could match both and pick the wrong one.
    probe = (rng.random((int(n_probe), mean.size)) * 2.0 - 1.0).astype(np.float32)
    probe *= np.float32(10.0) ** rng.integers(-4, 5, size=probe.shape).astype(np.float32)
    expected = np.asarray(transformer.inverse_transform(probe.copy()))
    for convention in STANDARD_INVERSE_CONVENTIONS:
        replay = _StandardInverse(mean, scale, convention).inverse_transform(probe.copy())
        if np.array_equal(replay, expected):
            return convention
    import sklearn

    raise ValueError(
        f"scikit-learn {sklearn.__version__}'s StandardScaler.inverse_transform matches "
        f"neither op order this build of gliquid knows how to replay "
        f"({list(STANDARD_INVERSE_CONVENTIONS)}). Exporting a bundle here would record a "
        f"golden vector the pickle-free path cannot reproduce, so it is refused. Add the new "
        f"convention to gliquid.production_model_runner, or export on an sklearn that "
        f"matches one of the known orders."
    )


class _PortableModel:
    """One target's scaler + booster, shaped like the ``Pipeline`` it replaces.

    ``steps`` is not decoration. ``_get_tree_model`` and ``_transform_row_for_explainer``
    below already walk ``model.steps`` to find the estimator SHAP explains and to apply
    everything in front of it; presenting the same two-step shape means the whole SHAP path
    works on a portable bundle with no branching at all.
    """

    def __init__(self, scaler: _AffineScaler, booster: xgboost.Booster):
        self.scaler = scaler
        self.booster = booster
        self.steps = [("scaler", scaler), ("model", booster)]

    def predict(self, X):
        scaled = self.scaler.transform(X)
        # A sys.modules hit after the runner's constructor already resolved it. Called once
        # per predict() -- and the golden draw pushes all 4,096 rows through a single call.
        return self.booster.predict(_xgboost().DMatrix(scaled))


class ProductionModelRunner:
    """Standalone model runner for production-style single-sample inference.

    Args:
        bundle_dir: A portable bundle (``manifest.json`` + ``<target>.ubj``) or a legacy
            joblib bundle (``model/<target>_model.joblib``). ``None`` — the default — uses
            the bundle shipped inside the wheel, :func:`default_bundle_dir`.

    Raises:
        ImportError: when the optional ML stack is absent. CONSTRUCTING a runner is what
            raises — importing this module, and naming ``gliquid.ProductionModelRunner``
            through the lazy façade, both still work on a bare install. ``ml`` covers the
            shipped portable bundle; a legacy bundle additionally needs ``models``. See the
            module docstring for the full table.
    """

    def __init__(self, bundle_dir: str | Path | None = None):
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else default_bundle_dir()
        self.bundle_format = self._detect_format(self.bundle_dir)
        # Legacy bundles keep the artifacts one level down; portable bundles are flat.
        legacy = self.bundle_format == "legacy"
        self.model_dir = self.bundle_dir / "model" if legacy else self.bundle_dir

        self.bundle_id: str = self.bundle_dir.name
        self.targets: list[str] = []
        self.target_frames: dict[str, str] = {}
        self.models: dict[str, object] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.target_transformers: dict[str, object] = {}
        self.explainers: dict[str, object] = {}
        self.manifest: dict = {}

        self._frames: dict[str, pd.DataFrame] = {}
        self._feature_source: str | None = None

        self._load_model_artifacts()

    # -- bundle discovery ---------------------------------------------------------------

    @staticmethod
    def _detect_format(bundle_dir: Path) -> str:
        """``'portable'`` or ``'legacy'``, decided by what is actually on disk.

        Portable wins when both are present, which only happens mid-conversion. The check is
        for the manifest rather than for a ``.ubj`` glob so that a half-written bundle reads
        as absent instead of as a portable bundle with missing targets.
        """
        if (bundle_dir / "manifest.json").is_file():
            return "portable"
        if any((bundle_dir / "model").glob("*_model.joblib")):
            return "legacy"
        raise FileNotFoundError(
            f"{bundle_dir} is not a model bundle: it has neither manifest.json (a portable "
            f"bundle, as shipped in gliquid/models/) nor model/*_model.joblib (a legacy "
            f"joblib bundle). Pass ProductionModelRunner() with no arguments to use the "
            f"bundle shipped in the wheel."
        )

    # -- weights ------------------------------------------------------------------------

    def _load_model_artifacts(self) -> None:
        """Load the WEIGHTS only. The feature corpus is loaded lazily, on first use."""
        # Resolved here, before either branch, because BOTH need it and only one names it
        # directly: a legacy bundle reaches xgboost through joblib's unpickling, so without
        # this the `models` path would report a bare ModuleNotFoundError raised inside
        # joblib.load rather than a message naming the extra to install.
        _xgboost()
        if self.bundle_format == "portable":
            self._load_portable_artifacts()
        else:
            self._load_legacy_artifacts()

    def _load_portable_artifacts(self) -> None:
        manifest_path = self.bundle_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = int(self.manifest.get("schema", 0))
        if schema > BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"{manifest_path} declares bundle schema {schema} and this build of gliquid "
                f"understands up to {BUNDLE_SCHEMA_VERSION}. Reading it here would "
                f"misinterpret its contents, so it is refused. Upgrade gliquid, or re-export "
                f"the bundle with this version."
            )
        self.bundle_id = str(self.manifest.get("bundle_id", self.bundle_dir.name))
        self.targets = [str(t) for t in self.manifest["targets"]]
        self.target_frames = {str(k): str(v) for k, v in self.manifest["target_frames"].items()}

        self.feature_names = {
            frame: [str(c) for c in names]
            for frame, names in json.loads(
                (self.bundle_dir / "feature_names.json").read_text(encoding="utf-8")
            ).items()
        }
        preprocess = json.loads((self.bundle_dir / "preprocess.json").read_text(encoding="utf-8"))

        required = [self.bundle_dir / name for name in PORTABLE_REQUIRED]
        required += [self.bundle_dir / f"{target}.ubj" for target in self.targets]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required bundle files:\n" + "\n".join(missing))

        for target in self.targets:
            booster = _xgboost().Booster()
            booster.load_model(str(self.bundle_dir / f"{target}.ubj"))
            feature_block = preprocess["features"][target]
            scaler = _AffineScaler(feature_block["center"], feature_block["scale"])
            self.models[target] = _PortableModel(scaler, booster)
            target_block = preprocess["targets"].get(target)
            if target_block is not None:
                self.target_transformers[target] = _StandardInverse(
                    target_block["mean"],
                    target_block["scale"],
                    # Absent in a bundle written before the field existed, which can only be
                    # a bundle exported under sklearn <= 1.7.x -- exactly the default.
                    target_block.get("convention", DEFAULT_STANDARD_INVERSE_CONVENTION),
                )

    def _load_legacy_artifacts(self) -> None:
        """The pre-portable path: joblib pipelines, discovered rather than hardcoded.

        The target list comes from the directory contents, not from a literal
        ``["L0_a", "L0_b", "L1_a"]``. That literal is exactly what stops a v23 four-target
        bundle from loading, and the fix is the same one the converter's ``--targets`` makes.

        This is the ONLY path that unpickles, hence the only one that needs ``joblib`` and
        the pinned ``scikit-learn`` of the ``models`` extra.
        """
        joblib = _joblib()
        suffix = "_model.joblib"
        self.targets = sorted(p.name[: -len(suffix)] for p in self.model_dir.glob("*" + suffix))
        if not self.targets:  # pragma: no cover - _detect_format already required one
            raise FileNotFoundError(f"No model/*_model.joblib files under {self.model_dir}")

        required = [self.model_dir / f"{t}_model.joblib" for t in self.targets]
        required += [
            self.model_dir / "target_transformers.joblib",
            self.model_dir / "feature_names_symm.joblib",
            self.model_dir / "feature_names_anti.joblib",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required bundle files:\n" + "\n".join(missing))

        self.models = {t: joblib.load(self.model_dir / f"{t}_model.joblib") for t in self.targets}
        self.feature_names = {
            "symmetric": list(joblib.load(self.model_dir / "feature_names_symm.joblib")),
            "antisymmetric": list(joblib.load(self.model_dir / "feature_names_anti.joblib")),
        }
        self.target_transformers = joblib.load(self.model_dir / "target_transformers.joblib")
        self.target_frames = {
            target: derive_frame(self._legacy_input_width(model), self.feature_names)
            for target, model in self.models.items()
        }

    @staticmethod
    def _legacy_input_width(model) -> int:
        """How many features a fitted legacy pipeline accepts."""
        width = getattr(model, "n_features_in_", None)
        if width is None and hasattr(model, "steps"):  # pragma: no cover - defensive
            width = getattr(model.steps[0][1], "n_features_in_", None)
        if width is None:  # pragma: no cover - defensive
            raise ValueError(f"Cannot determine the input width of {model!r}.")
        return int(width)

    # -- the feature corpus (lazy) ------------------------------------------------------

    def _frame(self, frame: str) -> pd.DataFrame:
        """The feature table for ``frame``, loaded on first use and then memoized."""
        if frame not in self._frames:
            self._frames[frame] = self._load_frame(frame)
        return self._frames[frame]

    def _load_frame(self, frame: str) -> pd.DataFrame:
        """Load one feature frame from the bundle's xlsx, else from the configured cache.

        The bundle's own xlsx wins when present: they are the corpus that bundle was
        exported with, so they cannot disagree with its models. A portable bundle has none,
        and falls through to the cache — which is the whole point of the split.
        """
        xlsx = self._xlsx_path(frame)
        if xlsx is not None:
            # Imported here and nowhere else: pandas needs openpyxl to read xlsx, openpyxl
            # is declared only in the `models` extra, and this is the ONLY code path that
            # requires it. An unconditional read here is the bug this module used to have.
            self._feature_source = f"xlsx:{xlsx}"
            return pd.read_excel(xlsx)
        return self._load_frame_from_cache(frame)

    def _xlsx_path(self, frame: str) -> Path | None:
        """The bundle's own feature sheet for ``frame``, or ``None``.

        Two layouts, both real: ``cache/20260329_022905/`` keeps the sheets beside
        ``model/``, while ``dev/model_bundles/bundle_*/`` keeps them under ``data/``.
        """
        name = _XLSX_NAMES.get(frame)
        if name is None:
            return None
        for candidate in (self.bundle_dir / name, self.bundle_dir / "data" / name):
            if candidate.is_file():
                return candidate
        return None

    def _load_frame_from_cache(self, frame: str) -> pd.DataFrame:
        backend = self._ml_backend(frame)
        columns = backend.ml_feature_columns(frame)
        expected = self.feature_names.get(frame)
        if expected is not None and columns != expected:
            raise config.ConfigError(
                f"The cache at {backend.path} carries ML features for frame {frame!r} whose "
                f"columns do not match model bundle {self.bundle_id!r} "
                f"(cache: {len(columns)} columns exported for bundle "
                f"{backend.ml_features_bundle_id(frame)!r}; bundle: {len(expected)}). "
                f"Predicting from mismatched columns would feed the model a row of the "
                f"wrong features, which it would accept without complaint. Re-export the "
                f"feature tables for this bundle with "
                f"`python dev/scripts/export_portable_model_bundle.py --features-out ...`."
            )
        rows = backend.ml_feature_rows(frame)
        if not rows:
            raise config.ConfigError(
                f"The cache at {backend.path} declares ML feature frame {frame!r} but holds "
                f"no rows for it."
            )
        self._feature_source = f"cache:{backend.path}"
        frame_df = pd.DataFrame(
            [[system, *values] for system, values in rows], columns=["system", *columns]
        )
        return frame_df

    def _ml_backend(self, frame: str):
        """The configured cache, if it can serve ML features. ``ConfigError`` otherwise.

        A ``ConfigError`` and not a ``FileNotFoundError``: nothing is missing from the
        BUNDLE — the bundle is complete and its weights already loaded. What is missing is
        the pointer to a corpus, which is a configuration fault, and the message says how to
        fix it rather than naming a path that was never going to exist.
        """
        how_to = (
            f"Point gliquid at a SQLite cache store carrying them with "
            f"gliquid.config.set_cache_dir('<store>.sqlite') (or the "
            f"{config.CACHE_DIR_ENV_VAR} environment variable), and build one with "
            f"`python dev/scripts/export_portable_model_bundle.py --joblib-bundle <bundle> "
            f"--out <bundle> --features-out <store>.sqlite`. Predicting from feature rows "
            f"you already hold needs none of this: use predict_from_dataframes()."
        )
        if config.cache_dir is None:
            raise config.ConfigError(
                f"Model bundle {self.bundle_id!r} ships weights only; the per-system feature "
                f"corpus it predicts FROM is not in the wheel, and no gliquid cache is "
                f"configured. {how_to}"
            )
        backend = gliquid_cache.resolve_backend()
        if not isinstance(backend, gliquid_cache.SqliteBackend) or not backend.has_ml_features:
            raise config.ConfigError(
                f"Model bundle {self.bundle_id!r} needs the ML feature frame {frame!r}, but "
                f"the configured cache ({config.cache_dir}) does not carry the ml_features "
                f"tables — only a SQLite store can. {how_to}"
            )
        return backend

    @property
    def df_symm(self) -> pd.DataFrame:
        """The symmetric feature table. Loaded on first access, not on construction."""
        return self._frame("symmetric")

    @property
    def df_anti(self) -> pd.DataFrame:
        """The antisymmetric feature table. Loaded on first access, not on construction."""
        return self._frame("antisymmetric")

    @property
    def feature_source(self) -> str | None:
        """Where the feature rows came from, once any have been loaded. Diagnostics only."""
        return self._feature_source

    # -- prediction ---------------------------------------------------------------------

    @staticmethod
    def _mirror_system(system_name: str) -> str | None:
        for sep in ("-", "_"):
            if sep in system_name:
                parts = system_name.split(sep)
                if len(parts) == 2:
                    return f"{parts[1]}{sep}{parts[0]}"
        return None

    @staticmethod
    def _extract_scalar(x: object) -> float:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.ravel()[0])

    def _prepare_row(self, row_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Align row dataframe to required feature order."""
        if len(row_df) != 1:
            raise ValueError("Input must be a single-row DataFrame.")

        if mode not in self.feature_names:
            raise ValueError(f"mode must be one of {sorted(self.feature_names)}.")

        req_features = self.feature_names[mode]
        missing = [c for c in req_features if c not in row_df.columns]
        if missing:
            raise KeyError(
                f"Missing {mode} features: {missing[:10]}" + (" ..." if len(missing) > 10 else "")
            )

        return row_df[req_features].copy()

    def _predict_single_target(self, target: str, X_row: pd.DataFrame) -> float:
        pred = self.models[target].predict(X_row.values)
        pred = np.asarray(pred).reshape(-1)

        if target in self.target_transformers:
            pred = self.target_transformers[target].inverse_transform(pred.reshape(-1, 1)).ravel()

        return float(pred[0])

    def get_rows_for_system(
        self, system_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Fetch one symmetric row, one antisymmetric row, and optional antisymmetric mirror row.

        Raises:
            gliquid.ConfigError: when the feature corpus cannot be reached — the bundle
                carries weights only. See :meth:`_ml_backend`.
            ValueError: when the corpus is reachable but has no row for *system_name*.
        """
        df_symm = self._frame("symmetric")
        df_anti = self._frame("antisymmetric")
        symm_row = df_symm[df_symm["system"] == system_name].head(1).copy()
        anti_row = df_anti[df_anti["system"] == system_name].head(1).copy()

        if symm_row.empty:
            raise ValueError(f"System '{system_name}' not found in symmetric dataset.")
        if anti_row.empty:
            raise ValueError(f"System '{system_name}' not found in antisymmetric dataset.")

        mirror_name = self._mirror_system(system_name)
        anti_mirror_row = None
        if mirror_name is not None:
            mirror = df_anti[df_anti["system"] == mirror_name].head(1).copy()
            if not mirror.empty:
                anti_mirror_row = mirror

        return symm_row, anti_row, anti_mirror_row

    def predict_from_dataframes(
        self,
        symm_row_df: pd.DataFrame,
        anti_row_df: pd.DataFrame,
        anti_mirror_row_df: pd.DataFrame | None = None,
    ) -> list[float]:
        """Predict every target of the bundle, in the bundle's target order.

        For the shipped three-target bundle that is ``[L0_a, L0_b, L1_a]``, unchanged.

        If ``anti_mirror_row_df`` is provided, the antisymmetry constraint is enforced on
        every ANTISYMMETRIC target::

            y = 0.5 * (pred(A-B) - pred(B-A))

        which used to be hardcoded to ``L1_a`` alone and so silently skipped a v23 bundle's
        second antisymmetric target.

        Needs no cache, no xlsx and no ``openpyxl``: the caller already holds the rows.
        """
        prepared: dict[str, pd.DataFrame] = {}
        mirrored: dict[str, pd.DataFrame] = {}
        for frame in sorted({self.target_frames[t] for t in self.targets}):
            source = anti_row_df if frame == "antisymmetric" else symm_row_df
            prepared[frame] = self._prepare_row(source, mode=frame)
            if frame == "antisymmetric" and anti_mirror_row_df is not None:
                mirrored[frame] = self._prepare_row(anti_mirror_row_df, mode=frame)

        predictions: list[float] = []
        for target in self.targets:
            frame = self.target_frames[target]
            raw = self._predict_single_target(target, prepared[frame])
            if frame in mirrored:
                raw = 0.5 * (raw - self._predict_single_target(target, mirrored[frame]))
            predictions.append(raw)
        return predictions

    def predict_system(self, system_name: str) -> list[float]:
        """Predict every target for a named system, reading its features from the corpus."""
        symm_row, anti_row, anti_mirror_row = self.get_rows_for_system(system_name)
        return self.predict_from_dataframes(symm_row, anti_row, anti_mirror_row)

    @staticmethod
    def apply_feature_updates(row_df: pd.DataFrame, updates: dict[str, float]) -> pd.DataFrame:
        """Return a modified copy of row_df with selected feature updates."""
        if len(row_df) != 1:
            raise ValueError("row_df must be single-row.")
        out = row_df.copy()
        for key, value in updates.items():
            if key not in out.columns:
                raise KeyError(f"Feature '{key}' not found in input row.")
            out.loc[out.index[0], key] = value
        return out

    def _get_tree_model(self, model: object) -> object:
        """Extract final tree model for SHAP TreeExplainer."""
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
        return model

    def _transform_row_for_explainer(self, model: object, features_df: pd.DataFrame) -> np.ndarray:
        """Apply all preprocessing steps before final estimator."""
        X = features_df.values
        if hasattr(model, "steps") and len(model.steps) > 1:
            for _, transformer in model.steps[:-1]:
                if transformer is not None:
                    X = transformer.transform(X)
        return X

    def _ensure_explainer(self, target: str) -> None:
        if target in self.explainers:
            return
        tree_model = self._get_tree_model(self.models[target])
        self.explainers[target] = _shap().TreeExplainer(tree_model)

    def _target_inverse_affine(self, target: str) -> tuple[float, float] | None:
        """Return ``(intercept, slope)`` of the target's inverse transform.

        A target transformer maps physical units ``y`` → model space ``z``.
        SHAP values are computed in ``z`` space; to express them in physical
        units we need the inverse map ``z → y``.  When that inverse is *affine*
        (``y = intercept + slope · z`` with a constant slope) SHAP additivity is
        preserved: every value scales by ``slope`` and the base value maps to
        ``intercept + slope · base``.

        Rather than reach into scaler-specific attributes (``mean_``,
        ``center_``, ``min_``, …) we recover the two coefficients by probing
        ``inverse_transform`` at ``z = 0, 1, 2``.  This transparently supports
        any affine target transform (``StandardScaler``, ``RobustScaler``,
        ``MinMaxScaler``, ``MaxAbsScaler``, …) regardless of its internal
        parameter names or ``with_centering`` / ``with_scaling`` flags — and,
        since the portable bundle's :class:`_StandardInverse` answers the same
        probe, it needs no branch for the pickle-free path either.

        Returns
        -------
        (intercept, slope) : tuple of float
            If *target* has an affine transformer.
        None
            If *target* has no transformer (SHAP already in physical units).

        Raises
        ------
        NotImplementedError
            If the transformer is **non-affine** (e.g. ``QuantileTransformer``,
            ``PowerTransformer``).  SHAP values cannot be linearly rescaled
            through a nonlinear inverse without breaking additivity, so we fail
            loudly instead of silently mislabelling z-scores as physical units.
        """
        if target not in self.target_transformers:
            return None

        transformer = self.target_transformers[target]
        probe = np.array([[0.0], [1.0], [2.0]])
        y = np.asarray(transformer.inverse_transform(probe), dtype=float).ravel()

        intercept = float(y[0])
        slope = float(y[1] - y[0])

        # Affine <=> equally spaced outputs: y(2) must equal intercept + 2·slope.
        if not np.isclose(y[2], intercept + 2.0 * slope, rtol=1e-6, atol=1e-8):
            raise NotImplementedError(
                f"Target transformer for {target!r} "
                f"({type(transformer).__name__}) is non-affine; SHAP values "
                f"cannot be linearly inverse-scaled to physical units. Only "
                f"affine target transforms (StandardScaler, RobustScaler, "
                f"MinMaxScaler, …) are supported."
            )
        return intercept, slope

    def get_shap_explanation(self, target: str, features_df: pd.DataFrame) -> shap.Explanation:
        """Compute a SHAP Explanation for *target* in original parameter space.

        If an affine transform was applied to the target during training the
        SHAP values and base value live in scaled (e.g. z-score) space.  This
        method inverse-scales them via :meth:`_target_inverse_affine` so the
        explanation is directly interpretable in physical units:

        * ``values``      → multiplied by ``slope``
        * ``base_values``  → ``intercept + slope · base``

        Non-affine target transforms (e.g. ``QuantileTransformer``) raise
        ``NotImplementedError`` rather than return misleading values.

        Args:
            target (str): Model key, e.g. ``"L0_a"``, ``"L0_b"``, ``"L1_a"``.
            features_df (pd.DataFrame): Single-row DataFrame already aligned to the
                correct feature order (output of :meth:`_prepare_row`).

        Returns:
            shap.Explanation: One-dimensional Explanation with values in original
            target space.

        Raises:
            ImportError: when ``shap`` is absent — it is the optional ``shap`` extra. No
                PREDICTION path needs it; see the module docstring.
        """
        shap = _shap()
        self._ensure_explainer(target)
        model = self.models[target]
        explainer = self.explainers[target]

        # Which frame this target was trained in -- recorded per target rather than derived
        # from its name, so a bundle with a second antisymmetric target explains correctly.
        mode = self.target_frames[target]
        feature_names = self.feature_names[mode]

        # Preprocess features through pipeline (minus final estimator)
        X = self._transform_row_for_explainer(model, features_df)

        # Raw SHAP values (z-score space if scaler exists)
        sv = explainer.shap_values(X)
        base = float(explainer.expected_value)

        values = np.array(sv[0], dtype=float)

        # Inverse-scale into physical units if an affine target transform exists
        scaling = self._target_inverse_affine(target)
        if scaling is not None:
            intercept, slope = scaling
            values = values * slope
            base = intercept + slope * base

        return shap.Explanation(
            values=values,
            base_values=base,
            data=features_df.iloc[0].values,
            feature_names=feature_names,
        )

    def create_compact_prediction_figure(self, system_name, output_file, max_display_features=6):
        """
        Create a compact 300x480px figure with vertical stack of waterfall plots.

        Generates a publication-ready compact figure showing SHAP waterfall plots for
        all three parameters (L0_a, L0_b, L1_a) in a vertical stack. Each subplot is
        100px tall by 480px wide, totaling 300x480px. Features are labeled in order
        of importance up to the maximum that can be reasonably displayed.

        Args:
            system_name (str): Name of the system to analyze.
            output_file (str | None): Path of the SVG file to write. ``None`` shows the
                figure interactively and returns ``None`` instead of saving.
            max_display_features (int, optional): Maximum number of features to label. If
                None, automatically determines based on space available. Features are
                labeled in order of absolute SHAP value (most important first).

        Returns:
            str: Path to saved figure file, or ``None`` when shown interactively.

        Raises:
            ImportError: when ``shap`` is absent — the optional ``shap`` extra.
        """
        shap = _shap()

        # Fetch feature rows using the existing workflow
        symm_row, anti_row, _ = self.get_rows_for_system(system_name)
        l0_features_df = self._prepare_row(symm_row, mode="symmetric")
        l1_features_df = self._prepare_row(anti_row, mode="antisymmetric")

        # Ensure all explainers are ready
        for param in ["L0_a", "L0_b", "L1_a"]:
            self._ensure_explainer(param)

        # Create figure with 3 subplots vertically stacked
        # Figure size in inches - will be saved as SVG (scalable)
        # Using nominal size that gives good proportions
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(4.8, 3.0),
            gridspec_kw={"hspace": 0.23, "left": 0.305, "right": 0.955, "top": 0.95, "bottom": 0},
        )

        params = ["L0_a", "L0_b", "L1_a"]

        for param, ax in zip(params, axes):
            logger.info("Creating waterfall plot for %s...", param)

            # Determine features based on parameter
            if param in ["L0_a", "L0_b"]:
                features_df = l0_features_df
            else:  # L1_a
                features_df = l1_features_df

            # Get SHAP explanation in original parameter space
            shap_explanation = self.get_shap_explanation(param, features_df)

            # Create waterfall plot on this axis
            plt.sca(ax)  # Set current axis
            shap.plots.waterfall(shap_explanation, max_display=max_display_features, show=False)
            # Get y-axis tick labels and modify them
            y_labels = ax.get_yticklabels()[:max_display_features]
            ax.set_yticks(ax.get_yticks()[:max_display_features])
            new_labels = []

            for label in y_labels:
                label_text = label.get_text()
                if " = " in label_text:
                    parts = label_text.split(" = ")
                    feature_name = parts[1]
                    value_str = parts[0]
                    value = float(value_str.replace("−", "-"))
                    formatted_value = f"{value:.2g}"
                    feature_name = feature_name.replace("metastable", "ms")
                    feature_name = feature_name.replace("valence", "val")
                    feature_name = feature_name.replace("enthalpy", "enth")
                    feature_name = feature_name.replace("change", "Δ")
                    feature_name = feature_name.replace("comp_skew", "skew")
                    new_labels.append(f"{feature_name} = {formatted_value}")
                else:
                    new_labels.append(label_text)

            ax.set_yticklabels(new_labels)
            ax.tick_params(axis="y", labelsize=11, labelcolor="black")
            ax.set_xlabel("")  # Remove x-axis label

            # SHAP value formatting
            for text in ax.texts:
                text.set_fontsize(10)
                text_content = text.get_text()
                if param == "L0_b":
                    value = float(text_content.replace("−", "-"))
                    text.set_text(f"{value:.2f}")
                elif param in ["L0_a", "L1_a"]:
                    value = float(text_content.replace("−", "-"))
                    text.set_text(f"{int(round(value))}")

            xlim = ax.get_xlim()
            x_min, x_max = xlim
            x_range = x_max - x_min
            ideal_step = x_range / 4  # 4 intervals for 5 ticks

            # Determine appropriate step with at most 2 significant figures
            if ideal_step == 0:
                step = 1
            else:
                # Find the order of magnitude
                magnitude = 10 ** int(np.floor(np.log10(abs(ideal_step))))

                # Normalize to 1-10 range
                normalized = ideal_step / magnitude

                # Pick step with 2 significant figures
                if normalized <= 1.0:
                    step_normalized = 1.0
                elif normalized <= 1.5:
                    step_normalized = 1.5
                elif normalized <= 2.0:
                    step_normalized = 2.0
                elif normalized <= 2.5:
                    step_normalized = 2.5
                elif normalized <= 3.0:
                    step_normalized = 3.0
                elif normalized <= 4.0:
                    step_normalized = 4.0
                elif normalized <= 5.0:
                    step_normalized = 5.0
                elif normalized <= 7.5:
                    step_normalized = 7.5
                else:
                    step_normalized = 10.0

                step = step_normalized * magnitude

            # Find starting tick (should be multiple of step and <= x_min)
            start_tick = int(np.floor(x_min / step)) * step

            # If range crosses zero, ensure 0 is included
            if x_min <= 0 <= x_max:
                # Adjust start_tick to ensure 0 is one of the 5 ticks
                ticks_before_zero = 2  # Use 2 ticks before zero for balance
                start_tick = -ticks_before_zero * step

            # Generate 5 evenly spaced ticks
            tick_values = [start_tick + i * step for i in range(5)]

            # Filter to only include ticks within the visible range
            tick_values = [t for t in tick_values if x_min <= t <= x_max]

            # Ensure we have at least 3 ticks for readability
            if len(tick_values) < 3:
                # Fall back to simple approach
                if x_min <= 0 <= x_max:
                    tick_values = [int(x_min), 0, int(x_max)]
                else:
                    tick_values = [int(x_min), int((x_min + x_max) / 2), int(x_max)]

            x_min = x_max - x_range * 1.08
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(tick_values)
            ax.set_xticklabels([str(int(t)) for t in tick_values], fontsize=9)

            # Remove grid lines
            ax.grid(False)

            # Add parameter label in top left
            ax.text(
                0.01,
                1,
                param,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )
            # Add figure title in top left corner
            fig.suptitle(
                "SHAP Analysis", fontsize=14, fontweight="bold", x=0.025, y=1, ha="left", va="top"
            )

        if output_file is None:
            plt.show()
            return None
        else:
            plt.savefig(output_file, format="svg", bbox_inches="tight", dpi=400)
            plt.close()

            return output_file

    def create_beeswarm_figures(
        self,
        output_dir: str | None = None,
        max_display: int = 6,
        sample_fraction: float = 1.0,
        publication: bool = False,
    ) -> list[str | None]:
        """Create beeswarm SHAP plots for every target in the bundle.

        Args:
            output_dir (str | None): Directory to save SVG files.  If ``None``, plots are
                shown interactively with ``plt.show()`` instead.
            max_display (int): Maximum number of features to display (default 6).
            sample_fraction (float): Fraction of the dataset to use (0–1).  Values < 1
                speed up computation for quick visual checks.
            publication (bool): If ``True``, strip **all** text (axis labels, tick labels,
                feature names, colorbar labels) so only points and lines remain, then save
                as a high-resolution SVG.

        Returns:
            list[str | None]: Paths to saved files, or ``None`` entries when shown
            interactively.

        Raises:
            gliquid.ConfigError: reads the whole feature corpus, so it needs one — unlike
                :meth:`predict_from_dataframes`.
            ImportError: when ``shap`` is absent — the optional ``shap`` extra.
        """
        shap = _shap()
        results: list[str | None] = []

        for target in self.targets:
            logger.info(
                "Computing beeswarm for %s (sample_fraction=%.0f%%)...",
                target,
                sample_fraction * 100,
            )

            # --- select the right dataset & feature set -----------------------
            mode = self.target_frames[target]
            df = self._frame(mode)
            req_features = self.feature_names[mode]

            # --- optional subsampling -----------------------------------------
            if sample_fraction < 1.0:
                df = df.sample(frac=sample_fraction, random_state=42)

            features_df = df[req_features].copy()

            # --- batch SHAP in original parameter space -----------------------
            self._ensure_explainer(target)
            model = self.models[target]
            explainer = self.explainers[target]

            X = self._transform_row_for_explainer(model, features_df)
            sv = explainer.shap_values(X)
            base = float(explainer.expected_value)
            values = np.array(sv, dtype=float)

            scaling = self._target_inverse_affine(target)
            if scaling is not None:
                intercept, slope = scaling
                values = values * slope
                base = intercept + slope * base

            explanation = shap.Explanation(
                values=values,
                base_values=np.full(len(features_df), base),
                data=features_df.values,
                feature_names=req_features,
            )

            # --- plot ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.sca(ax)
            shap.plots.beeswarm(explanation, max_display=max_display, show=False)

            if publication:
                # Strip ALL text; keep only points, lines, and the zero vline.
                ax = plt.gca()
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("")
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Clear colorbar / any other axes
                for other_ax in fig.get_axes():
                    if other_ax is not ax:
                        other_ax.set_ylabel("")
                        other_ax.set_xlabel("")
                        other_ax.set_title("")
                        other_ax.set_xticklabels([])
                        other_ax.set_yticklabels([])

                # Remove any standalone figure-level text
                for txt in fig.texts:
                    txt.set_visible(False)

            # --- save or show -------------------------------------------------
            if output_dir is not None:
                out_path = Path(output_dir) / f"{target}_beeswarm.svg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(out_path), format="svg", bbox_inches="tight", dpi=400)
                plt.close()
                logger.info("  [OK] Saved: %s", out_path)
                results.append(str(out_path))
            else:
                plt.show()
                results.append(None)

        return results


# ---------------------------------------------------------------------------------------
# Golden vector -- the conversion's proof, and the only thing that makes "portable" checkable
# ---------------------------------------------------------------------------------------
#
# A converted bundle that merely LOADS proves nothing: every failure mode here (a transposed
# scaler, a dropped booster iteration, the float64 inverse-transform route) produces numbers
# that look entirely reasonable. So the exporter draws N deterministic rows, pushes them
# through the ORIGINAL joblib pipeline, and records the sha256 of the float64 output in the
# manifest. Re-running the same draw through the PORTABLE path must reproduce it bit for bit,
# on any Python and any library version -- which is also what makes the tox matrix's
# agreement on one hash a real cross-version claim rather than eight independent runs.
#
# The draw lives here rather than in the exporter so that the package can re-check its own
# shipped bundle with no dev checkout present.


def golden_rows(frame_index: int, n_rows: int, scaler_center, scaler_scale, seed: int = 0):
    """``n_rows`` deterministic feature rows spanning ±3 robust-scale units about ``center``.

    Seeded per frame (``default_rng([seed, frame_index])``) so adding a frame cannot shift
    another frame's draw, and built from ``Generator.random`` — a uniform stream NumPy's
    versioning policy holds fixed — rather than from ``standard_normal``. Scaling by the
    model's own ``center_``/``scale_`` puts the rows where the trees actually have splits;
    unit-normal rows would land almost every sample in the same handful of leaves and make
    the hash agree for the wrong reason.
    """
    center = np.asarray(scaler_center, dtype=np.float64)
    scale = np.asarray(scaler_scale, dtype=np.float64)
    rng = np.random.default_rng([int(seed), int(frame_index)])
    unit = rng.random((int(n_rows), center.size))
    return center + scale * (6.0 * unit - 3.0)


def golden_digest(values) -> str:
    """sha256 of a float64 array's raw bytes — the comparison the acceptance test makes."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_golden(
    runner: ProductionModelRunner, frames_order, n_rows: int, seed: int, targets=None
) -> dict:
    """Run the golden draw through ``runner`` and return the manifest's ``golden`` block.

    Deliberately takes a *runner*, so the exporter can call it on a legacy joblib bundle and
    the package test can call it on the portable one and the two are the same code.

    ``targets`` overrides the runner's own list, which matters on the export side: a legacy
    bundle's targets are DISCOVERED from its directory and may be a superset of the
    ``--targets`` being converted, and a golden vector over a different set of targets than
    the bundle carries could never be reproduced.
    """
    targets = list(runner.targets if targets is None else targets)
    rows = {}
    for index, frame in enumerate(frames_order):
        # Sorted, so the draw does not depend on the order --targets was typed in. Any
        # target of the frame will do: the rows only need to land where that frame's trees
        # have splits, and every model of a frame shares its training distribution.
        anchor = sorted(t for t in targets if runner.target_frames[t] == frame)[0]
        scaler = runner.models[anchor].steps[0][1]
        rows[frame] = golden_rows(index, n_rows, scaler.center_, scaler.scale_, seed=seed)

    per_target = {}
    stacked = []
    for target in targets:
        frame = runner.target_frames[target]
        raw = np.asarray(runner.models[target].predict(rows[frame])).reshape(-1)
        if target in runner.target_transformers:
            raw = runner.target_transformers[target].inverse_transform(raw.reshape(-1, 1)).ravel()
        out = np.asarray(raw, dtype=np.float64)
        per_target[target] = {
            "sha256": golden_digest(out),
            "first16": [float(v) for v in out[:16]],
        }
        stacked.append(out)

    combined = np.concatenate(stacked) if stacked else np.zeros(0, dtype=np.float64)
    return {
        "n_rows": int(n_rows),
        "seed": int(seed),
        "frame_order": list(frames_order),
        "targets": per_target,
        "combined_sha256": golden_digest(combined),
    }
