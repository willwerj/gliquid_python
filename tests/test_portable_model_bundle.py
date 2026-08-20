"""The portable model bundle: pickle-free weights in the wheel, feature corpus in the cache.

Four claims are pinned here, and each is checked against something other than the code that
produced it:

* **The shipped bundle is intact and reproduces its golden vector.** ``manifest.json`` holds
  the sha256 of every other file, plus the sha256 of 4,096 deterministic predictions made by
  the ORIGINAL joblib pipelines at export time. Re-running that draw through the portable
  path must return the SAME BYTES — ``array_equal``, never ``allclose``. Every failure mode
  of this conversion (a transposed scaler, a booster read at the wrong iteration, the naive
  float64 route through ``StandardScaler.inverse_transform``) yields numbers that look
  entirely reasonable, so a tolerance here would accept the bug it exists to catch. Measured:
  the float64 route drifts up to ~1e-6 relative on this bundle and ``np.allclose`` passes.

  This test is also the CROSS-VERSION acceptance. It runs in all eight tox environments and
  they must all agree on one hash — which is a claim about version independence only because
  the comparison is exact.

* **The portable path carries no library-version coupling.** Loading the joblib bundle under
  a newer scikit-learn emits ``InconsistentVersionWarning``; loading this one must emit none,
  on any version. That contrast is the whole point of the format.

* **Weights and corpus are genuinely separable.** ``ProductionModelRunner()`` with no
  arguments, no cache and no ``openpyxl`` constructs and predicts. That is checked in a
  SUBPROCESS with ``openpyxl`` blocked at import, because the failure it guards against —
  an unconditional ``pd.read_excel`` in the constructor — is invisible to a test running in
  an environment that happens to have the package.

* **Legacy joblib bundles still load.** Built here from scratch rather than read from disk,
  so the test says nothing about which scikit-learn wrote the shipped one, and with FOUR
  targets across two antisymmetric frames — the shape a v23 bundle has and the shape the old
  hardcoded ``["L0_a", "L0_b", "L1_a"]`` could not express.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gliquid.config as config
from gliquid.cache import SqliteBackend, close_sqlite_backends
from gliquid.production_model_runner import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_BUNDLE_ID,
    DEFAULT_STANDARD_INVERSE_CONVENTION,
    STANDARD_INVERSE_CONVENTIONS,
    ProductionModelRunner,
    _StandardInverse,
    compute_golden,
    default_bundle_dir,
    derive_frame,
    detect_standard_inverse_convention,
)

SHIPPED = default_bundle_dir()


def _new_store(path: Path) -> None:
    """Create an empty store ON DISK.

    ``SqliteBackend`` connects lazily, so ``SqliteBackend(p, create=True).close()`` creates
    nothing at all — a trap worth naming once here rather than debugging per test.
    """
    backend = SqliteBackend(path, create=True)
    try:
        backend.set_meta("created_by", "test_portable_model_bundle")
    finally:
        backend.close()


# =======================================================================================
# The shipped bundle
# =======================================================================================


class TestShippedBundle:
    def test_bundle_is_present_in_the_installed_package(self):
        """A wheel built before ``git add`` would carry no bundle and raise nothing else."""
        assert SHIPPED.is_dir(), (
            f"no model bundle at {SHIPPED}. Hatchling ships TRACKED files: an exported "
            f"bundle that was never `git add`ed is absent from the wheel with no error."
        )
        assert (SHIPPED / "manifest.json").is_file()

    def test_no_pickle_in_the_shipped_bundle(self):
        """The format's premise, as an assertion rather than a claim in a docstring."""
        pickled = sorted(p.name for p in SHIPPED.iterdir() if p.suffix in (".joblib", ".pkl"))
        assert not pickled, f"pickled artifacts in the portable bundle: {pickled}"

    def test_every_file_matches_its_recorded_sha256(self):
        manifest = json.loads((SHIPPED / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["files"], "manifest records no files; the check would be vacuous"
        import hashlib

        for name, entry in sorted(manifest["files"].items()):
            path = SHIPPED / name
            assert path.is_file(), f"manifest names {name}, which is not in the bundle"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entry["sha256"], f"{name} does not match its recorded sha256"

    def test_manifest_schema_is_understood(self):
        manifest = json.loads((SHIPPED / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] <= BUNDLE_SCHEMA_VERSION
        assert manifest["bundle_id"] == DEFAULT_BUNDLE_ID

    def test_golden_vector_reproduces_byte_for_byte(self):
        """The cross-version acceptance: one hash, identical in every tox environment."""
        manifest = json.loads((SHIPPED / "manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["golden"]
        runner = ProductionModelRunner()
        replayed = compute_golden(
            runner, recorded["frame_order"], recorded["n_rows"], recorded["seed"]
        )
        assert replayed["combined_sha256"] == recorded["combined_sha256"], (
            "the portable path no longer reproduces the joblib pipelines' predictions. "
            "This is exact on purpose: see the module docstring."
        )
        for target, block in sorted(recorded["targets"].items()):
            assert replayed["targets"][target]["sha256"] == block["sha256"], target
            assert replayed["targets"][target]["first16"] == block["first16"], target

    def test_portable_load_emits_no_inconsistent_version_warning(self):
        """The joblib path emits two of these under a newer sklearn. This one emits zero."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner = ProductionModelRunner()
            manifest = runner.manifest["golden"]
            compute_golden(runner, manifest["frame_order"], 16, manifest["seed"])
        offenders = [w for w in caught if "InconsistentVersion" in w.category.__name__]
        assert not offenders, [str(w.message) for w in offenders]

    def test_constructing_the_default_runner_loads_no_feature_corpus(self):
        """Weights only. Touching the corpus is what needs a cache, and it is lazy."""
        runner = ProductionModelRunner()
        assert runner.bundle_format == "portable"
        assert runner.targets == ["L0_a", "L0_b", "L1_a"]
        assert runner.target_frames == {
            "L0_a": "symmetric",
            "L0_b": "symmetric",
            "L1_a": "antisymmetric",
        }
        assert runner.feature_source is None
        # A deliberate pin on the SHIPPED artifact -- a canary that the wheel carries the
        # bundle we think it does, so it is expected to move when DEFAULT_BUNDLE_ID does.
        # 51/37 was the v22.02-era 20260329_022905 bundle; 49/36 is 20260817_112204.
        # The three absent features (max_s_unfilled, min_f_unfilled, diff_s_unfilled) were
        # dropped from the dataset generator INTENTIONALLY between v22.02 and the v23
        # series -- v23.92, built months earlier, already carries the same five 'unfilled'
        # columns. Not a side effect of scoping v24.01 to 60 elements; min_f_unfilled was
        # a constant column (nunique=1) in v22.02 regardless.
        assert {f: len(n) for f, n in runner.feature_names.items()} == {
            "symmetric": 49,
            "antisymmetric": 36,
        }


# =======================================================================================
# The split: predicting without a corpus, and failing loudly when one is needed
# =======================================================================================


_NO_CACHE_NO_OPENPYXL = textwrap.dedent(
    """
    import builtins, sys
    _real_import = builtins.__import__
    def _blocked(name, *args, **kwargs):
        if name == "openpyxl" or name.startswith("openpyxl."):
            raise ImportError("openpyxl is blocked for this test")
        return _real_import(name, *args, **kwargs)
    builtins.__import__ = _blocked
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("the openpyxl block did not take; the test would be vacuous")

    import numpy as np, pandas as pd
    import gliquid.config as config

    # Unset explicitly rather than relying on the environment: config finds a corpus by
    # walking out from the PACKAGE's own __file__, so a source checkout adopts its cache/
    # no matter what cwd or GLIQUID_CACHE_DIR say. The claim under test is that no corpus
    # is needed, and this is how you get "no corpus" here.
    config.set_cache_dir(None)
    assert config.cache_dir is None, f"cache still configured: {config.cache_dir}"

    from gliquid.production_model_runner import ProductionModelRunner
    runner = ProductionModelRunner()

    rng = np.random.default_rng(7)
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

    mirror = pd.DataFrame(
        rng.random((1, len(runner.feature_names["antisymmetric"]))),
        columns=runner.feature_names["antisymmetric"],
    )
    mirrored = runner.predict_from_dataframes(symm, anti, mirror)
    assert mirrored[:2] == values[:2], "the mirror branch must not touch symmetric targets"
    assert "openpyxl" not in sys.modules
    print("OK")
    """
)


def test_predict_from_dataframes_needs_no_cache_no_xlsx_and_no_openpyxl(tmp_path):
    """Acceptance 4: the split line. Run in a subprocess so the block is real.

    ``openpyxl`` is declared only in the ``models`` extra, and the constructor used to call
    ``pd.read_excel`` unconditionally — so a bare ``pip install gliquid`` imported fine and
    then died on instantiation. An in-process test cannot see that in an environment that
    has the package installed, which this one does.
    """
    env = os.environ.copy()
    for name in ("GLIQUID_CACHE_DIR", "GLIQUID_DATA_DIR"):
        env.pop(name, None)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    # cwd outside the repo so config's project-root walk finds no cache/ to adopt.
    result = subprocess.run(
        [sys.executable, "-c", _NO_CACHE_NO_OPENPYXL],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


class TestCorpusRequired:
    def test_predict_system_without_a_cache_raises_config_error(self, monkeypatch):
        """Acceptance 5. A ConfigError, not a FileNotFoundError: nothing is MISSING from
        the bundle — the pointer to a corpus is."""
        monkeypatch.setattr(config, "cache_dir", None)
        runner = ProductionModelRunner()
        with pytest.raises(config.ConfigError) as excinfo:
            runner.predict_system("Cu-Mg")
        message = str(excinfo.value)
        assert "set_cache_dir" in message
        assert "predict_from_dataframes" in message

    def test_directory_cache_cannot_serve_features_and_says_so(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "cache_mode", "directory")
        runner = ProductionModelRunner()
        with pytest.raises(config.ConfigError, match="ml_features"):
            runner.predict_system("Cu-Mg")


# =======================================================================================
# A synthetic bundle: four targets, two frames, both directions
# =======================================================================================


SYMM_FEATURES = [f"s{i}" for i in range(6)]
ANTI_FEATURES = [f"a{i}" for i in range(4)]
SYNTH_TARGETS = ["L0_a", "L0_b", "L1_a", "L1_b"]
SYNTH_FRAMES = {"L0_a": "symmetric", "L0_b": "symmetric", "L1_a": "antisymmetric", "L1_b": "antisymmetric"}
SYNTH_SYSTEMS = ["Ag-Au", "Au-Ag", "Cu-Mg", "Mg-Cu", "Ni-Ti", "Ti-Ni"]


def _fit_pipeline(rng, n_features, n_rows=160):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBRegressor

    X = rng.normal(size=(n_rows, n_features)) * rng.uniform(0.5, 3.0, n_features)
    y = X @ rng.normal(size=n_features) + rng.normal(size=n_rows) * 0.1
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", XGBRegressor(n_estimators=12, max_depth=3, random_state=0)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline, y


@pytest.fixture(scope="module")
def synthetic_legacy_bundle(tmp_path_factory):
    """A four-target joblib bundle, built here rather than read from disk.

    Two targets per frame, so the antisymmetric mirror branch has to apply to BOTH
    antisymmetric targets — the generalization the old hardcoded ``L1_a`` check could not
    express, and the one a v23 bundle needs.
    """
    import joblib
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(11)
    root = tmp_path_factory.mktemp("legacy_bundle")
    model_dir = root / "model"
    model_dir.mkdir()

    transformers = {}
    for target in SYNTH_TARGETS:
        width = len(SYMM_FEATURES) if SYNTH_FRAMES[target] == "symmetric" else len(ANTI_FEATURES)
        pipeline, y = _fit_pipeline(rng, width)
        joblib.dump(pipeline, model_dir / f"{target}_model.joblib")
        transformers[target] = StandardScaler().fit(y.reshape(-1, 1))
    joblib.dump(transformers, model_dir / "target_transformers.joblib")
    joblib.dump(SYMM_FEATURES, model_dir / "feature_names_symm.joblib")
    joblib.dump(ANTI_FEATURES, model_dir / "feature_names_anti.joblib")
    return root


@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory, synthetic_legacy_bundle):
    """A cache store holding a feature row per system, per frame — including both mirrors."""
    rng = np.random.default_rng(23)
    store = tmp_path_factory.mktemp("features") / "ml.sqlite"
    backend = SqliteBackend(store, create=True)
    try:
        backend.ensure_ml_features()
        with backend.bulk_write():
            for frame, columns in (
                ("symmetric", SYMM_FEATURES),
                ("antisymmetric", ANTI_FEATURES),
            ):
                rows = [(name, rng.normal(size=len(columns))) for name in SYNTH_SYSTEMS]
                backend.write_ml_features(frame, "synthetic", columns, rows)
    finally:
        backend.close()
    close_sqlite_backends()
    return store


def _write_portable(source: ProductionModelRunner, out_dir: Path, bundle_id: str) -> Path:
    """Hand-write a portable bundle from a loaded legacy runner.

    Deliberately NOT the converter: ``dev/scripts/export_portable_model_bundle.py`` is not
    importable from the package suite, and a reader of these tests should be able to see the
    whole format in one screen.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in source.targets:
        source.models[target].steps[-1][1].get_booster().save_model(str(out_dir / f"{target}.ubj"))
    (out_dir / "feature_names.json").write_text(json.dumps(source.feature_names), encoding="utf-8")
    (out_dir / "preprocess.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA_VERSION,
                "features": {
                    target: {
                        "kind": "affine",
                        "center": list(map(float, source.models[target].steps[0][1].center_)),
                        "scale": list(map(float, source.models[target].steps[0][1].scale_)),
                    }
                    for target in source.targets
                },
                "targets": {
                    target: {
                        "kind": "standard",
                        # Measured against the INSTALLED sklearn, exactly as the converter
                        # does. sklearn changed this op order between 1.7.2 and 1.9.0, so a
                        # hardcoded convention here would make this test assert that gliquid
                        # inherits sklearn's drift -- the opposite of the point.
                        "convention": detect_standard_inverse_convention(transformer),
                        "mean": list(map(float, transformer.mean_)),
                        "scale": list(map(float, transformer.scale_)),
                    }
                    for target, transformer in source.target_transformers.items()
                },
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA_VERSION,
                "bundle_id": bundle_id,
                "targets": list(source.targets),
                "target_frames": dict(source.target_frames),
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    return out_dir


@pytest.fixture(scope="module")
def synthetic_pair(tmp_path_factory, synthetic_legacy_bundle):
    legacy = ProductionModelRunner(synthetic_legacy_bundle)
    portable_dir = tmp_path_factory.mktemp("portable_bundle") / "synthetic"
    _write_portable(legacy, portable_dir, "synthetic")
    return legacy, ProductionModelRunner(portable_dir)


class TestFourTargetBundle:
    def test_legacy_targets_are_discovered_not_hardcoded(self, synthetic_pair):
        legacy, portable = synthetic_pair
        assert legacy.bundle_format == "legacy"
        assert legacy.targets == SYNTH_TARGETS
        assert legacy.target_frames == SYNTH_FRAMES
        assert portable.targets == SYNTH_TARGETS

    def test_portable_matches_legacy_exactly(self, synthetic_pair):
        """``array_equal``, over both frames and with the mirror branch engaged."""
        legacy, portable = synthetic_pair
        rng = np.random.default_rng(5)
        for _ in range(50):
            symm = pd.DataFrame(rng.normal(size=(1, len(SYMM_FEATURES))), columns=SYMM_FEATURES)
            anti = pd.DataFrame(rng.normal(size=(1, len(ANTI_FEATURES))), columns=ANTI_FEATURES)
            mirror = pd.DataFrame(rng.normal(size=(1, len(ANTI_FEATURES))), columns=ANTI_FEATURES)
            assert np.array_equal(
                legacy.predict_from_dataframes(symm, anti),
                portable.predict_from_dataframes(symm, anti),
            )
            assert np.array_equal(
                legacy.predict_from_dataframes(symm, anti, mirror),
                portable.predict_from_dataframes(symm, anti, mirror),
            )

    def test_the_mirror_branch_applies_to_every_antisymmetric_target(self, synthetic_pair):
        """The old code mirrored ``L1_a`` alone, which silently skipped a bundle's second
        antisymmetric target."""
        _, portable = synthetic_pair
        rng = np.random.default_rng(6)
        symm = pd.DataFrame(rng.normal(size=(1, len(SYMM_FEATURES))), columns=SYMM_FEATURES)
        anti = pd.DataFrame(rng.normal(size=(1, len(ANTI_FEATURES))), columns=ANTI_FEATURES)
        mirror = pd.DataFrame(rng.normal(size=(1, len(ANTI_FEATURES))), columns=ANTI_FEATURES)

        plain = portable.predict_from_dataframes(symm, anti)
        mirrored = portable.predict_from_dataframes(symm, anti, mirror)
        assert plain[:2] == mirrored[:2], "symmetric targets must be untouched"
        for index, target in enumerate(SYNTH_TARGETS):
            if SYNTH_FRAMES[target] != "antisymmetric":
                continue
            expected = 0.5 * (
                portable._predict_single_target(target, portable._prepare_row(anti, "antisymmetric"))
                - portable._predict_single_target(
                    target, portable._prepare_row(mirror, "antisymmetric")
                )
            )
            assert mirrored[index] == expected, target

    def test_predict_system_reads_the_cache_and_mirrors(
        self, synthetic_pair, synthetic_features, monkeypatch
    ):
        _, portable = synthetic_pair
        monkeypatch.setattr(config, "cache_dir", synthetic_features)
        monkeypatch.setattr(config, "cache_mode", "sqlite")
        try:
            values = portable.predict_system("Ag-Au")
            assert len(values) == 4
            assert portable.feature_source.startswith("cache:")
            # 'Ag-Au' and 'Au-Ag' are DISTINCT antisymmetric rows, so the mirror is real.
            mirrored = portable.predict_system("Au-Ag")
            for index, target in enumerate(SYNTH_TARGETS):
                if SYNTH_FRAMES[target] == "antisymmetric":
                    assert values[index] == pytest.approx(-mirrored[index], abs=0, rel=1e-12)
        finally:
            portable._frames.clear()
            close_sqlite_backends()

    def test_golden_vector_is_reproducible_for_an_arbitrary_bundle(self, synthetic_pair):
        legacy, portable = synthetic_pair
        recorded = compute_golden(legacy, ["antisymmetric", "symmetric"], 128, 3)
        replayed = compute_golden(portable, ["antisymmetric", "symmetric"], 128, 3)
        assert replayed == recorded


class TestLegacyXlsxCorpus:
    """Acceptance 7: a legacy bundle that ships its own xlsx keeps serving from them."""

    @pytest.mark.parametrize("layout", ["beside-model", "under-data"])
    def test_legacy_bundle_reads_its_own_xlsx(self, tmp_path, synthetic_legacy_bundle, layout):
        pytest.importorskip("openpyxl")
        import shutil

        root = tmp_path / layout
        shutil.copytree(synthetic_legacy_bundle, root)
        target_dir = root if layout == "beside-model" else root / "data"
        target_dir.mkdir(exist_ok=True)

        rng = np.random.default_rng(31)
        for columns, name in (
            (SYMM_FEATURES, "prediction_dataset_symmetric.xlsx"),
            (ANTI_FEATURES, "prediction_dataset_antisymmetric.xlsx"),
        ):
            frame_df = pd.DataFrame(
                rng.normal(size=(len(SYNTH_SYSTEMS), len(columns))), columns=columns
            )
            frame_df.insert(0, "system", SYNTH_SYSTEMS)
            frame_df.to_excel(target_dir / name, index=False)

        runner = ProductionModelRunner(root)
        values = runner.predict_system("Cu-Mg")
        assert len(values) == 4
        assert runner.feature_source.startswith("xlsx:")


class TestBundleGuards:
    def test_a_directory_that_is_neither_format_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a model bundle"):
            ProductionModelRunner(tmp_path)

    def test_a_newer_bundle_schema_is_refused(self, tmp_path, synthetic_pair):
        import shutil

        _, portable = synthetic_pair
        clone = tmp_path / "future"
        shutil.copytree(portable.bundle_dir, clone)
        manifest = json.loads((clone / "manifest.json").read_text(encoding="utf-8"))
        manifest["schema"] = BUNDLE_SCHEMA_VERSION + 1
        (clone / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="bundle schema"):
            ProductionModelRunner(clone)

    def test_a_missing_booster_is_named(self, tmp_path, synthetic_pair):
        import shutil

        _, portable = synthetic_pair
        clone = tmp_path / "incomplete"
        shutil.copytree(portable.bundle_dir, clone)
        (clone / "L1_b.ubj").unlink()
        with pytest.raises(FileNotFoundError, match="L1_b.ubj"):
            ProductionModelRunner(clone)

    def test_frame_derivation_refuses_an_ambiguous_width(self):
        assert derive_frame(6, {"symmetric": SYMM_FEATURES, "antisymmetric": ANTI_FEATURES})
        with pytest.raises(ValueError, match="identifiable by width"):
            derive_frame(99, {"symmetric": SYMM_FEATURES, "antisymmetric": ANTI_FEATURES})
        with pytest.raises(ValueError, match="identifiable by width"):
            derive_frame(6, {"symmetric": SYMM_FEATURES, "antisymmetric": SYMM_FEATURES})


class TestStandardInverseConvention:
    """The float32 op order inside ``StandardScaler.inverse_transform``, which sklearn changed.

    Measured across the tox matrix: sklearn 1.7.1/1.7.2 round the coefficient AFTER
    multiplying; 1.9.0 rounds it BEFORE (the Array-API migration), moving 2,530 of 4,096
    float32 samples by up to 9.54e-7. A bundle therefore records which order reproduces the
    sklearn that exported it, instead of gliquid hardcoding one and silently inheriting the
    drift on the next upgrade.
    """

    @staticmethod
    def _fitted_scaler():
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(2)
        return StandardScaler().fit(rng.normal(size=(500, 1)) * 137.0 - 4.0)

    def test_the_installed_sklearn_matches_a_known_convention(self):
        """If sklearn ships a THIRD order, this fails here rather than in a silent export."""
        assert detect_standard_inverse_convention(self._fitted_scaler()) in (
            STANDARD_INVERSE_CONVENTIONS
        )

    def test_detection_agrees_with_sklearn_bit_for_bit(self):
        scaler = self._fitted_scaler()
        convention = detect_standard_inverse_convention(scaler)
        rng = np.random.default_rng(9)
        z = (rng.normal(size=(4096, 1)) * 3.0).astype(np.float32)
        expected = scaler.inverse_transform(z.copy())
        replay = _StandardInverse(scaler.mean_, scaler.scale_, convention).inverse_transform(
            z.copy()
        )
        assert np.array_equal(replay, expected)
        assert replay.dtype == np.float32, "float32 in must stay float32 out"

    def test_the_two_conventions_are_genuinely_different(self):
        """Otherwise detection would be picking arbitrarily and the recorded field vacuous."""
        scaler = self._fitted_scaler()
        rng = np.random.default_rng(9)
        z = (rng.normal(size=(4096, 1)) * 3.0).astype(np.float32)
        outputs = [
            _StandardInverse(scaler.mean_, scaler.scale_, name).inverse_transform(z.copy())
            for name in STANDARD_INVERSE_CONVENTIONS
        ]
        assert not np.array_equal(outputs[0], outputs[1])
        assert np.allclose(outputs[0], outputs[1], rtol=1e-5), "and yet only ULPs apart"

    def test_float64_input_is_unaffected_by_the_convention(self):
        """The orders differ only at float32; a float64 probe cannot tell them apart, which
        is why detection probes at float32."""
        scaler = self._fitted_scaler()
        rng = np.random.default_rng(9)
        z = rng.normal(size=(256, 1))
        outputs = [
            _StandardInverse(scaler.mean_, scaler.scale_, name).inverse_transform(z.copy())
            for name in STANDARD_INVERSE_CONVENTIONS
        ]
        assert np.array_equal(outputs[0], outputs[1])

    def test_an_unknown_convention_is_refused(self):
        with pytest.raises(ValueError, match="target-inverse convention"):
            _StandardInverse([0.0], [1.0], "float128-vibes")

    def test_the_shipped_bundle_records_its_convention(self):
        preprocess = json.loads((SHIPPED / "preprocess.json").read_text(encoding="utf-8"))
        for target, block in preprocess["targets"].items():
            assert block["kind"] == "standard", target
            assert block["convention"] in STANDARD_INVERSE_CONVENTIONS, target

    def test_a_bundle_without_the_field_takes_the_pre_1_9_order(self):
        """Back-compat: the field postdates the format, and only a <=1.7.x export lacks it."""
        assert DEFAULT_STANDARD_INVERSE_CONVENTION == "float64-coeff"
        rng = np.random.default_rng(1)
        z = (rng.normal(size=(64, 1)) * 5.0).astype(np.float32)
        assert np.array_equal(
            _StandardInverse([3.0], [7.0]).inverse_transform(z.copy()),
            _StandardInverse([3.0], [7.0], "float64-coeff").inverse_transform(z.copy()),
        )


class TestFeatureColumnMismatch:
    def test_columns_that_do_not_match_the_bundle_are_refused(
        self, tmp_path, synthetic_pair, monkeypatch
    ):
        """A cache holding another bundle's feature columns would feed the model a row of
        the wrong features, which it accepts without complaint."""
        _, portable = synthetic_pair
        store = tmp_path / "wrong.sqlite"
        backend = SqliteBackend(store, create=True)
        try:
            backend.ensure_ml_features()
            backend.write_ml_features(
                "symmetric",
                "some-other-bundle",
                [f"x{i}" for i in range(len(SYMM_FEATURES))],
                [(name, np.zeros(len(SYMM_FEATURES))) for name in SYNTH_SYSTEMS],
            )
        finally:
            backend.close()
        close_sqlite_backends()

        monkeypatch.setattr(config, "cache_dir", store)
        monkeypatch.setattr(config, "cache_mode", "sqlite")
        try:
            with pytest.raises(config.ConfigError, match="do not match model bundle"):
                portable.predict_system("Ag-Au")
        finally:
            portable._frames.clear()
            close_sqlite_backends()


# =======================================================================================
# The cache tables themselves
# =======================================================================================


class TestMlFeatureTables:
    def test_absent_until_explicitly_created(self, tmp_path):
        backend = SqliteBackend(tmp_path / "s.sqlite", create=True)
        try:
            assert backend.has_ml_features is False
            assert backend.ml_feature_columns("symmetric") is None
            assert backend.ml_feature_rows("symmetric") == []
            assert backend.ml_feature_stats() == {}
            backend.ensure_ml_features()
            assert backend.has_ml_features is True
        finally:
            backend.close()

    def test_roundtrip_is_bit_exact(self, tmp_path):
        """Raw float64, not json: the acceptance downstream is ``array_equal``."""
        rng = np.random.default_rng(3)
        values = {
            name: rng.normal(size=4) * 10.0 ** float(rng.integers(-8, 8))
            for name in SYNTH_SYSTEMS
        }
        values["Nan-Sys"] = np.array([np.nan, np.inf, -np.inf, -0.0])
        backend = SqliteBackend(tmp_path / "s.sqlite", create=True)
        try:
            backend.ensure_ml_features()
            written = backend.write_ml_features(
                "antisymmetric", "b1", ANTI_FEATURES, list(values.items())
            )
            assert written == len(values)
            assert backend.ml_feature_columns("antisymmetric") == ANTI_FEATURES
            assert backend.ml_features_bundle_id("antisymmetric") == "b1"
            read = dict(backend.ml_feature_rows("antisymmetric"))
            assert set(read) == set(values)
            for name, expected in values.items():
                assert np.array_equal(read[name], expected, equal_nan=True), name
        finally:
            backend.close()

    def test_ordered_system_names_are_distinct_rows(self, tmp_path):
        """``Ag-Au`` and ``Au-Ag`` are the mirror pair the antisymmetry constraint reads."""
        backend = SqliteBackend(tmp_path / "s.sqlite", create=True)
        try:
            backend.ensure_ml_features()
            backend.write_ml_features(
                "antisymmetric",
                "b1",
                ANTI_FEATURES,
                [("Ag-Au", np.ones(4)), ("Au-Ag", -np.ones(4))],
            )
            read = dict(backend.ml_feature_rows("antisymmetric"))
            assert np.array_equal(read["Ag-Au"], np.ones(4))
            assert np.array_equal(read["Au-Ag"], -np.ones(4))
        finally:
            backend.close()

    def test_a_short_row_is_refused_not_truncated(self, tmp_path):
        backend = SqliteBackend(tmp_path / "s.sqlite", create=True)
        try:
            backend.ensure_ml_features()
            with pytest.raises(ValueError, match="feature values"):
                backend.write_ml_features("symmetric", "b1", SYMM_FEATURES, [("X-Y", [1.0, 2.0])])
        finally:
            backend.close()

    def test_an_unknown_frame_is_refused(self, tmp_path):
        backend = SqliteBackend(tmp_path / "s.sqlite", create=True)
        try:
            backend.ensure_ml_features()
            with pytest.raises(ValueError, match="Unknown ML feature frame"):
                backend.write_ml_features("sideways", "b1", SYMM_FEATURES, [])
        finally:
            backend.close()

    def test_a_read_only_store_cannot_grow_the_tables(self, tmp_path):
        from gliquid.cache import CacheModeError

        path = tmp_path / "s.sqlite"
        _new_store(path)
        backend = SqliteBackend(path)
        try:
            assert backend.has_ml_features is False  # the store IS readable: not vacuous
            with pytest.raises(CacheModeError):
                backend.ensure_ml_features()
        finally:
            backend.close()

    def test_payload_is_the_documented_codec(self, tmp_path):
        """``float64-le``, checked against ``struct`` rather than against the writer."""
        import sqlite3

        from gliquid.cache import ML_FEATURE_CODEC

        assert ML_FEATURE_CODEC == "float64-le"
        path = tmp_path / "s.sqlite"
        backend = SqliteBackend(path, create=True)
        try:
            backend.ensure_ml_features()
            backend.write_ml_features(
                "antisymmetric", "b1", ANTI_FEATURES, [("A-B", [1.5, -2.25, 0.0, 1e300])]
            )
        finally:
            backend.close()
        conn = sqlite3.connect(str(path))
        blob = conn.execute("SELECT features FROM ml_features").fetchone()[0]
        conn.close()
        assert len(blob) == 4 * 8
        assert struct.unpack("<4d", blob) == (1.5, -2.25, 0.0, 1e300)

    def test_info_reports_the_ml_section_only_when_present(self, tmp_path, capsys):
        from gliquid.cache import info

        path = tmp_path / "s.sqlite"
        _new_store(path)
        assert info(path) == 0
        assert "ml_features" not in capsys.readouterr().out

        backend = SqliteBackend(path, writable=True)
        try:
            backend.ensure_ml_features()
            backend.write_ml_features(
                "symmetric", "b1", SYMM_FEATURES, [(n, np.zeros(6)) for n in SYNTH_SYSTEMS]
            )
        finally:
            backend.close()
        assert info(path) == 0
        out = capsys.readouterr().out
        assert "ml_features" in out
        assert f"{len(SYNTH_SYSTEMS)} rows x 6 columns" in out
