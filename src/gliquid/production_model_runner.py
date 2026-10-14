"""Production runner for the pre-trained L0/L1 models: weights and feature table in the wheel.

A model bundle holds the **model** (gradient-boosted regressors and the affine scalers around
them; the shipped bundle is ``gliquid/models/<bundle_id>/``) and the **feature table** (one row
of model inputs per system per frame, the ``features_<frame>.json`` files named by the
``features`` key of its ``manifest.json``).

:meth:`ProductionModelRunner.predict_system` reads a system's rows from the first source that
exists: the bundle's own xlsx sheets (legacy bundles), the manifest's feature table, or a
configured SQLite cache store carrying the ``ml_features`` / ``ml_feature_columns`` tables.
Constructing a runner loads the weights; the feature table is loaded on first use.
:meth:`ProductionModelRunner.predict_from_dataframes` takes the rows from the caller.

The portable bundle is XGBoost UBJSON plus JSON scaler coefficients, checked against a golden
vector recorded in its manifest. The ``StandardScaler.inverse_transform`` op order is recorded
per target in ``preprocess.json`` and replayed by :class:`_StandardInverse`. A directory
holding ``model/L0_a_model.joblib`` loads as a legacy joblib bundle with its xlsx corpus.

The ML stack is optional; each dependency is imported at the call that needs it, and a missing
one raises ``ImportError`` naming what to install:

===================  ================  =====================================================
what you are doing   install           what it needs
===================  ================  =====================================================
``import gliquid``   *(none)*          nothing
portable bundle      ``gliquid[ml]``   ``xgboost``
SHAP explanations    ``shap``          ``shap``
legacy joblib bundle ``joblib`` ...    ``joblib`` + the pinned ``scikit-learn``
===================  ================  =====================================================
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import gliquid.cache as gliquid_cache
import gliquid.config as config

if TYPE_CHECKING:
    # Annotations only.
    import shap
    import xgboost

logger = logging.getLogger(__name__)


# -- Optional ML dependencies, imported at the call that needs them --

_ML_EXTRA_HINT = (
    "ProductionModelRunner needs XGBoost, which gliquid ships as an optional extra "
    "(gliquid has no other use for it). Install it with `pip install gliquid[ml]` or "
    "`pip install xgboost`."
)

_SHAP_EXTRA_HINT = (
    "SHAP explanations need the `shap` package. Install it with `pip install shap`. "
    "Predicting needs none of this."
)

_MODELS_EXTRA_HINT = (
    "Loading a LEGACY joblib bundle needs joblib, openpyxl and the scikit-learn it was "
    "pickled against. Install them with "
    "`pip install joblib openpyxl scikit-learn==1.7.1`. The bundle shipped inside the wheel "
    "is pickle-free and needs none of it: construct ProductionModelRunner() with no arguments."
)


def _xgboost():
    """The ``xgboost`` module, or an ``ImportError`` naming the ``ml`` extra.

    Required by BOTH bundle formats — portable bundles load a ``Booster`` directly, and a
    legacy bundle's pickled pipelines unpickle ``xgboost.sklearn.XGBRegressor`` — so it is
    resolved once in :meth:`ProductionModelRunner._load_model_artifacts` rather than at each
    use. Raising there means a missing extra surfaces as a message naming it, not as a
    ``ModuleNotFoundError`` from inside ``joblib.load``.
    """
    return config._require_extra("xgboost", _ML_EXTRA_HINT)


def _shap():
    """The ``shap`` module with :mod:`gliquid.shap_compat` applied, or an ``ImportError``.

    The patches are applied here, which preserves the ordering ``shap_compat`` documents —
    "before any ``shap.TreeExplainer`` is created or any ``shap.plots.waterfall`` is called"
    — since every such call in this module goes through this accessor first.
    ``apply_patches`` is idempotent, so the repeat calls cost a set lookup.
    """
    shap = config._require_extra("shap", _SHAP_EXTRA_HINT)
    from gliquid.shap_compat import apply_patches

    apply_patches()
    return shap


def _joblib():
    """The ``joblib`` module, or an ``ImportError`` naming the legacy-bundle install.

    Reached only by :meth:`ProductionModelRunner._load_legacy_artifacts`. The portable
    bundle in the wheel is UBJSON plus JSON and never unpickles anything.
    """
    return config._require_extra("joblib", _MODELS_EXTRA_HINT)


#: The bundle that ships in the wheel. L1_b is pinned to zero by the comb-exp fit format, so it
#: has no model.
DEFAULT_BUNDLE_ID = "20261013_191431"

#: Bundle layout version; a bundle whose ``manifest.json`` declares a higher one is refused.
BUNDLE_SCHEMA_VERSION = 1

#: Files a portable bundle contains, beyond one ``<target>.ubj`` per target.
PORTABLE_REQUIRED = ("manifest.json", "preprocess.json", "feature_names.json")

#: Released alias of ``gliquid.cache.ML_FRAMES``, the two feature spaces.
FRAMES = gliquid_cache.ML_FRAMES

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

    Derived from the WIDTH rather than from the target's name: width is a property of the
    artifact; a name is a convention.

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


def _brief(value: object) -> str:
    """A short description of a feature-table field value, for error messages."""
    if isinstance(value, list):
        return f"a list of {len(value)} starting {value[:3]!r}"
    return repr(value)


class _AffineScaler:
    """``(X - center) / scale`` — the portable form of a fitted ``RobustScaler``.

    ``RobustScaler.transform`` is exactly this: ``check_array`` to a float copy, then
    ``X -= center_`` and ``X /= scale_`` in place. The dtype rule mirrors sklearn's
    ``FLOAT_DTYPES``: float32 and float64
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


#: The two op orders of ``StandardScaler.inverse_transform``: multiply then add in place, with
#: the coefficient rounded to float32 before or after. A bundle records which one it uses.
STANDARD_INVERSE_CONVENTIONS = ("float64-coeff", "float32-coeff")

#: What a bundle that predates the recorded field used — sklearn <= 1.7.x.
DEFAULT_STANDARD_INVERSE_CONVENTION = "float64-coeff"


class _StandardInverse:
    """``z -> y`` for a fitted ``StandardScaler``, replaying sklearn's float32 op order exactly.

    The input stays float32 and is multiplied, then added to, in place. The recorded convention
    picks one of sklearn's two orders:

    ``float64-coeff`` (sklearn <= 1.7.x)::

        X *= self.scale_          # float32 *= float64, rounded to float32 once
        X += self.mean_

    ``float32-coeff`` (sklearn >= 1.9)::

        X *= xp.astype(self.scale_, X.dtype)   # coefficient rounded to float32 first
        X += xp.astype(self.mean_, X.dtype)
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
        ValueError: when NEITHER convention reproduces the installed sklearn, i.e. a third op
            order has shipped.
    """
    mean = np.asarray(transformer.mean_, dtype=np.float64).reshape(-1)
    scale = np.asarray(transformer.scale_, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    # Probe values span several magnitudes.
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
        # A sys.modules hit after the runner's constructor already resolved it.
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
        # Resolved before either branch, so a missing extra is named.
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
                    # Absent in older bundles, which used the default.
                    target_block.get("convention", DEFAULT_STANDARD_INVERSE_CONVENTION),
                )

    def _load_legacy_artifacts(self) -> None:
        """The pre-portable path: joblib pipelines, discovered rather than hardcoded.

        The target list comes from the directory contents.

        This is the ONLY path that unpickles, hence the only one that needs ``joblib`` and
        the ``scikit-learn`` it was pickled against.
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
        """Load one feature frame from the first source that exists.

        1. the bundle's own xlsx sheet (legacy bundles);
        2. the file the portable manifest's ``features`` key names for ``frame``;
        3. the configured cache store, which must be a SQLite store carrying ``ml_features``.

        Raises:
            FileNotFoundError: the manifest names a feature table the bundle does not hold.
            ValueError: that table disagrees with the bundle (see
                :meth:`_load_frame_from_bundle`).
            gliquid.ConfigError: neither bundle source exists and the configured cache cannot
                serve ML features (see :meth:`_ml_backend`).
        """
        xlsx = self._xlsx_path(frame)
        if xlsx is not None:
            # openpyxl is part of the legacy-bundle install.
            self._feature_source = f"xlsx:{xlsx}"
            return pd.read_excel(xlsx)
        table = self._feature_table_path(frame)
        if table is not None:
            return self._load_frame_from_bundle(frame, table)
        return self._load_frame_from_cache(frame)

    def _feature_table_path(self, frame: str) -> Path | None:
        """The bundle file the manifest's ``features`` key names for ``frame``, or ``None``."""
        name = (self.manifest.get("features") or {}).get(frame)
        return None if name is None else self.bundle_dir / str(name)

    def _load_frame_from_bundle(self, frame: str, path: Path) -> pd.DataFrame:
        """Read ``frame`` from the bundle's feature table at ``path``. Reads nothing from config.

        The table is ``{"bundle_id", "frame", "columns", "rows": {system: [floats]}}``; the
        frame is ``system`` plus ``columns``, float64, in the file's row order.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: the table's ``bundle_id``, ``frame`` or ``columns`` differ from the
                bundle's, or a row's width differs from ``columns``.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"Model bundle {self.bundle_id!r} names {path.name!r} as its {frame!r} feature "
                f"table, but {path} does not exist. The bundle is incomplete: reinstall gliquid "
                f"or re-export the bundle."
            )
        table = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "bundle_id": self.bundle_id,
            "frame": frame,
            "columns": self.feature_names.get(frame),
        }
        for field, want in expected.items():
            got = table.get(field)
            if got != want:
                raise ValueError(
                    f"The feature table {path} does not match model bundle "
                    f"{self.bundle_id!r}: its {field!r} is {_brief(got)}, expected {_brief(want)}."
                )
        columns = table["columns"]
        rows = table["rows"]
        ragged = [system for system, values in rows.items() if len(values) != len(columns)]
        if ragged:
            raise ValueError(
                f"The feature table {path} has {len(ragged)} row(s) whose width differs from "
                f"its {len(columns)} 'columns', starting with {ragged[:3]!r}."
            )
        values = np.asarray(list(rows.values()), dtype=np.float64).reshape(len(rows), len(columns))
        frame_df = pd.DataFrame(values, columns=columns)
        frame_df.insert(0, "system", list(rows))
        self._feature_source = f"bundle:{path}"
        return frame_df

    def _xlsx_path(self, frame: str) -> Path | None:
        """The bundle's own feature sheet for ``frame``, or ``None``.

        Two layouts: the sheets beside ``model/``, or under ``data/``.
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
                f"wrong features, which it would accept without complaint. Point gliquid at "
                f"a cache store whose feature tables were exported for this bundle."
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
        the pointer to a corpus, which is a configuration fault.
        """
        how_to = (
            f"Point gliquid at a SQLite cache store carrying them with "
            f"gliquid.config.set_cache_dir('<store>.sqlite') (or the "
            f"{config.CACHE_DIR_ENV_VAR} environment variable); the tables are exported "
            f"alongside the bundle and are not part of the wheel. Predicting from feature "
            f"rows you already hold needs none of this: use predict_from_dataframes()."
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

        The symmetric features do not depend on element order, so a system whose symmetric
        row is stored under the reversed name (``"Mg-Cu"`` for ``"Cu-Mg"``) uses that row.
        With both antisymmetric orders stored, the antisymmetric targets of the reversed
        name are then exactly the negation of the stored name's.

        Raises:
            gliquid.ConfigError: when the bundle carries no feature table and the configured
                cache cannot serve ML features. See :meth:`_ml_backend`.
            FileNotFoundError: when the manifest names a feature table the bundle lacks.
            ValueError: when a feature table disagrees with the bundle, or the corpus has no
                row for *system_name*.
        """
        df_symm = self._frame("symmetric")
        df_anti = self._frame("antisymmetric")
        mirror_name = self._mirror_system(system_name)
        symm_row = df_symm[df_symm["system"] == system_name].head(1).copy()
        if symm_row.empty and mirror_name is not None:
            symm_row = df_symm[df_symm["system"] == mirror_name].head(1).copy()
        anti_row = df_anti[df_anti["system"] == system_name].head(1).copy()

        if symm_row.empty:
            raise ValueError(f"System '{system_name}' not found in symmetric dataset.")
        if anti_row.empty:
            raise ValueError(f"System '{system_name}' not found in antisymmetric dataset.")

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

        For the shipped three-target bundle that is ``[L0_a, L0_b, L1_a]``.

        If ``anti_mirror_row_df`` is provided, the antisymmetry constraint is enforced on
        every ANTISYMMETRIC target::

            y = 0.5 * (pred(A-B) - pred(B-A))

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
        """Return ``(intercept, slope)`` of the target's inverse transform ``z -> y``.

        The coefficients are recovered by probing ``inverse_transform`` at ``z = 0, 1, 2``, so any
        affine target transform works (``StandardScaler``, ``RobustScaler``, ``MinMaxScaler``,
        :class:`_StandardInverse`, ...).

        Returns
        -------
        (intercept, slope) : tuple of float
            If *target* has an affine transformer.
        None
            If *target* has no transformer (SHAP already in physical units).

        Raises
        ------
        NotImplementedError
            If the transformer is non-affine (e.g. ``QuantileTransformer``,
            ``PowerTransformer``).
        """
        if target not in self.target_transformers:
            return None

        transformer = self.target_transformers[target]
        probe = np.array([[0.0], [1.0], [2.0]])
        y = np.asarray(transformer.inverse_transform(probe), dtype=float).ravel()

        intercept = float(y[0])
        slope = float(y[1] - y[0])

        # Affine <=> equally spaced outputs: y(2) == intercept + 2·slope.
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

        Values in a scaled target space are mapped back through :meth:`_target_inverse_affine`:
        ``values`` are multiplied by ``slope`` and ``base_values`` become ``intercept + slope · base``.

        Args:
            target (str): Model key, e.g. ``"L0_a"``, ``"L0_b"``, ``"L1_a"``.
            features_df (pd.DataFrame): Single-row DataFrame already aligned to the
                correct feature order (output of :meth:`_prepare_row`).

        Returns:
            shap.Explanation: One-dimensional Explanation with values in original
            target space.

        Raises:
            ImportError: when ``shap`` is absent.
            NotImplementedError: for a non-affine target transform.
        """
        shap = _shap()
        self._ensure_explainer(target)
        model = self.models[target]
        explainer = self.explainers[target]

        # The frame this target was trained in, as recorded in the bundle.
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
        """Create a compact 300x480 px figure of stacked SHAP waterfall plots for L0_a, L0_b and L1_a.

        Args:
            system_name (str): Name of the system to analyze.
            output_file (str | None): Path of the SVG file to write. ``None`` shows the
                figure interactively and returns ``None`` instead of saving.
            max_display_features (int, optional): Maximum number of features to label, in order
                of absolute SHAP value. None fits as many as the space allows.

        Returns:
            str: Path to saved figure file, or ``None`` when shown interactively.

        Raises:
            ImportError: when ``shap`` is absent.
        """
        import matplotlib.pyplot as plt

        shap = _shap()

        # Fetch feature rows using the existing workflow
        symm_row, anti_row, _ = self.get_rows_for_system(system_name)
        l0_features_df = self._prepare_row(symm_row, mode="symmetric")
        l1_features_df = self._prepare_row(anti_row, mode="antisymmetric")

        # Ensure all explainers are ready
        for param in ["L0_a", "L0_b", "L1_a"]:
            self._ensure_explainer(param)

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


# -- Golden vector: N deterministic rows pushed through the original joblib pipeline, with the
# sha256 of the float64 output recorded in the manifest; the portable path reproduces it --


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

    Works on a legacy joblib runner and on a portable one alike. ``targets`` overrides the
    runner's own list.
    """
    targets = list(runner.targets if targets is None else targets)
    rows = {}
    for index, frame in enumerate(frames_order):
        # The frame's first target in sorted order sets the draw.
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
