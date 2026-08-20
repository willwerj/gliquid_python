# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The distribution version is
derived from the git tag by `hatch-vcs` — tagging `vX.Y.Z` *is* the release — so nothing in the
source tree carries a version literal that could disagree with this file.

## [Unreleased]

### Changed

- **Two directories no longer share the name `data`.** The bundled reference tables moved
  from `src/gliquid/data/` to `src/gliquid/reference/` (in the wheel: `gliquid/data/` →
  `gliquid/reference/`), and a source checkout's external corpus moved from `data/` to
  `cache/`. Nothing about how either is *reached* changed: the reference tables still ship
  and still load with no configuration, and the corpus is still found through
  `set_cache_dir()` / `GLIQUID_CACHE_DIR` (or the deprecated `set_data_dir()` /
  `GLIQUID_DATA_DIR`), with a checkout's own `cache/` as the last fallback. A clone that
  predates this rename should rename its `data/` directory to `cache/`.
- Added an empty `src/gliquid/models/` for the portable model bundle.

## [0.1.0] - 2026-08-12

First public release. `gliquid` was developed privately before this point, so there are no
earlier published versions to diff against; this entry describes what the package **is** at
0.1.0 rather than what changed.

### Added

- **DFT-referenced binary liquid fitting** (`gliquid.binary`). `BinaryLiquid` assembles a
  two-component system from cached Materials Project DFT entries plus a digitized experimental
  phase diagram, and `fit_parameters()` optimizes non-ideal liquid mixing parameters against the
  measured phase boundaries, reporting per-fit MAE. `BinaryLiquid.from_cache()` is the usual
  entry point.
- **Solution models** (`gliquid.solution`). `SolutionModel` and the `RKPolyExp`
  Redlich–Kister/exponential family, with four registered parameter formats — `linear`,
  `combined`, `comb-exp`, `regular` — selected by name through `param_format=`, plus the
  temperature scale `tau`.
- **Phase-equilibrium machinery** (`gliquid.hsx`). `HSX` and `lower_convex_hull` build the
  enthalpy–entropy–composition lower convex hull that turns free-energy curves into phase
  fields, liquidus/solidus boundaries and invariant points.
- **Ternary interpolation** (`gliquid.ternary`). `TernaryLiquidInterpolation` interpolates a
  ternary liquid free-energy surface from three fitted binary sub-systems, with `TLIPlotter`
  for isothermal sections and liquidus-surface views.
- **Unary reference database** (`gliquid.phase`). `UNARY`, `ComponentRef` and `Phase` expose
  per-element melting points, transition temperatures and lattice stabilities used as the
  reference states for every mixing calculation.
- **Experimental phase-diagram ingest** (`gliquid.mpds`). `load_mpds_data()` reads digitized
  MPDS diagrams, and `identify_invariant_points()` extracts eutectics, peritectics and
  congruent points from them.
- **Plotting layer** (`gliquid.plotting`, plus the `BLPlotter`/`TLIPlotter` façades). Plotly
  binary T–x diagrams, convex-hull and free-energy views, ternary surfaces, a shared style/color
  module, and figure export helpers.
- **Interactive hull editor** (`gliquid.hull_editor`). `ConvexHullEditor` is an `ipywidgets`
  tool for hand-adjusting hull endpoints; the module imports on a bare install and raises a
  message naming the `editor` extra only when one is actually constructed.
- **Pre-trained model runner** (`gliquid.production_model_runner`). Loads an exported L0/L1
  model bundle, predicts from single-row frames while preserving the L1 antisymmetric
  constraint, and renders compact SHAP force plots.
- **Bundled reference tables.** `phase_transitions.json`, `omegas_hcp.json` and
  `spurious_structures.json` ship inside the wheel at `gliquid/data/`, so element free-energy
  references work immediately after `pip install gliquid` with no external files.
- **External data-corpus contract** (`gliquid.config`). The per-system DFT entry caches,
  digitized diagrams and model bundle are megabytes of per-system data that no distribution
  carries. They are located through `set_data_dir()`, then `GLIQUID_DATA_DIR`, then a source
  checkout's own `data/`; with none of the three available a call that needs the corpus raises
  `ConfigError` naming both remedies rather than guessing a directory. Each shipped reference
  table is overridden by a same-named file in the data directory when one is present.
- **Optional extras.** `mpds` (live MPDS downloads), `editor` (`ConvexHullEditor` widgets),
  `models` (the exact `scikit-learn`/`xgboost` versions the serialized model artifacts were
  pickled against), `notebook` (local Jupyter tooling), `test` (`pytest`, `ruff`).
- **Library-grade logging.** All output goes through the `gliquid` logger tree, one child
  logger per module; the package installs no handlers and sets no levels, leaving that to the
  application. `tests/test_logging_boundary.py` enforces the boundary against the source tree,
  with two documented exemptions (`mpds.print_phase_mismatch_chart`, `hull_editor._log`).
- **Demonstration notebooks.** `notebooks/fitting_demo.ipynb`,
  `notebooks/interpolation_demo.ipynb` and a Colab-ready `notebooks/colab_demo.ipynb`.
- **Two-tier test suite.** A public `tests/` suite plus a maintainer `tests_internal/` tier of
  value pins, figure goldens and cache/network-gated families, with `pins`, `needs_cache`,
  `needs_network` and `slow` markers under `--strict-markers`. See `CONTRIBUTING.md`.
- **Packaging.** Hatchling with `hatch-vcs`, so the version is single-sourced from the git tag;
  Python 3.10–3.13 supported and exercised by a `tox` matrix.

### Known limitations

- **`dft_type` supports `GGA` only.** `R2SCAN` and `MIXED` remain recognized names, but neither
  can currently be fetched, for reasons upstream of this package; both now raise a `ValueError`
  naming the cause before any network call rather than failing obscurely mid-fetch.
  `R2SCAN` — the Materials Project stores the thermo type as the literal `r2SCAN` while
  `emmet.core.thermo.ThermoType.R2SCAN` is `R2SCAN`; `mp_api` validates the argument against the
  enum and forwards that casing, so the query matches nothing and the fetch comes back empty.
  `MIXED` — pymatgen's `MaterialsProjectDFTMixingScheme` hashes entry ids, but new-API entries
  carry `entry_id` as a dict, raising `TypeError: unhashable type: 'dict'`; past that, its
  r2SCAN half would be empty for the reason above, making it silently GGA-only. The bundled data
  corpus and every published result are built on `GGA`.
- **An empty Materials Project fetch now raises rather than being cached.** A zero-entry result
  used to be written out as a 2-byte `[]`, which every later read then accepted as a warm cache.

[Unreleased]: https://github.com/willwerj/gliquid_python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/willwerj/gliquid_python/releases/tag/v0.1.0
