# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The distribution version is
derived from the git tag by `hatch-vcs` — tagging `vX.Y.Z` *is* the release — so nothing in the
source tree carries a version literal that could disagree with this file.

## [Unreleased]

## [0.2.0] - 2026-08-20

### Added

- **A pluggable cache layer.** A cached record is now named by a `CacheKey` and read through a
  `CacheBackend`, rather than by resolving a directory. `DirectoryBackend` is the per-system
  JSON tree gliquid has always used, with identical filenames; `SqliteBackend` is a
  single-file, read-mostly store you can ship. `set_cache_dir()` accepts either — a path
  ending in `.sqlite` selects the single-file mode. `python -m gliquid.cache --help` migrates,
  verifies and inspects a store.
- **A pickle-free model bundle.** The shipped bundle is XGBoost's own forward-compatible
  UBJSON plus JSON scaler coefficients, so it carries no scikit-learn or joblib version
  coupling. Legacy joblib bundles still load unchanged.
- **Polymorph transitions in the ternary figure.** Each elemental corner now draws its full
  polymorph ladder, one coloured segment per phase, joined at the transition temperatures.

### Changed

- **The v24.01 model is the default bundle**, trained on a corpus fitted without solid
  solutions. Three targets (`L0_a`, `L0_b`, `L1_a`); `L1_b` is pinned to zero by the comb-exp
  parameter format.
- **One naming convention for elemental polymorphs.** Every solid phase in
  `phase_transitions.json` now reads `<name>-<El> (<structure>)`, or `<El> (<structure>)`
  where the phase has no name — so `hcp-Cd` is `Cd (hcp)`, `Diamond cubic Si` is
  `Si (diamond cubic)`, and `Graphite` is `C (graphite)`. 56 of 120 names changed. Code
  matching on these strings must be updated.
- **The binary and ternary figures share one phase-label formatter**, so a phase reads the
  same in both: greek symbol, structure abbreviated in parentheses, formula digits
  subscripted (`α-Fe (bcc)`, `Si (dc)`, `Ce(FeSi)₂`). Phase *colours* are assigned in
  phase-name order, so the renaming above changes the palette assignment for some systems.


- **The machine-learning stack is now an optional extra, not a base dependency.**
  `scikit-learn`, `shap`, `joblib` and `xgboost` left `[project.dependencies]` for the new
  `ml` / `shap` extras and the existing `models` one. They are reachable from exactly one
  module, `gliquid.production_model_runner`; every other capability — fitting, the convex
  hull, the phase diagrams, the ternary interpolation — ran without them and paid for them
  anyway. Measured over the full dependency closure on a linux-x86_64 / py312 resolve, a
  plain `pip install gliquid` goes from **1088.5 MB to 469.4 MB** of installed packages, and
  from 535.4 MB to 151.4 MB of downloads — 70 distributions to 60. Most of that is not the ML
  libraries themselves but what they drag: `xgboost` declares NVIDIA's NCCL on Linux
  (301.7 MB unpacked) and `shap` pulls `numba` → `llvmlite` (195.2 MB).

  Installing the capability back still costs less than the old default did: `gliquid[ml]`
  resolves to 492.8 MB, `gliquid[models]` to 520.9 MB, and `gliquid[shap]` — everything —
  to 722.8 MB.

  Nothing about how the runner is *reached* changed: `import gliquid` still works, and
  `gliquid.ProductionModelRunner` still resolves through the lazy façade. Constructing one
  without the extra now raises an `ImportError` naming it, the way `ConvexHullEditor` already
  did for `editor`. Which extra depends on what you are doing — `ml` for the pickle-free
  bundle that ships in the wheel, `shap` for explanations and their figures, `models` for a
  legacy joblib bundle. Callers who were relying on `pip install gliquid` to supply these
  need `pip install gliquid[ml]` (or `[shap]` / `[models]`).

- `ml` and `models` install **`xgboost-cpu`** rather than `xgboost` on every platform that
  publishes it (all but macOS). Same library, same import name, same 3.1.3 for the pinned
  legacy path; `xgboost` differs only in also declaring NVIDIA NCCL on Linux, which gliquid
  cannot use — there is no GPU code path in the package. `ml-gpu` installs the CUDA-capable
  build for callers who want it, and must be used *instead of* `ml`: both distributions
  install a package directory named `xgboost`.

- **Two directories no longer share the name `data`.** The bundled reference tables moved
  from `src/gliquid/data/` to `src/gliquid/reference/` (in the wheel: `gliquid/data/` →
  `gliquid/reference/`), and a source checkout's external corpus moved from `data/` to
  `cache/`. Nothing about how either is *reached* changed: the reference tables still ship
  and still load with no configuration, and the corpus is still found through
  `set_cache_dir()` / `GLIQUID_CACHE_DIR` (or the deprecated `set_data_dir()` /
  `GLIQUID_DATA_DIR`), with a checkout's own `cache/` as the last fallback. A clone that
  predates this rename should rename its `data/` directory to `cache/`.
- The portable model bundle ships in the wheel at `gliquid/models/<bundle_id>/`. Exactly one
  bundle ships at a time; the feature tables it predicts from live in the cache store, not the
  wheel.

### Fixed

- **Elemental corners of a ternary melted below their own melting point.** The ternary hull
  took its solid side from the DFT stable entries alone, so a component whose polymorph ladder
  carries a sub-melting transition had its corner set by `h_liq / s_liq` instead of
  `t_fusion`. 27 of 84 elements with a liquid reference were affected.
- **Solid-solution reconciliation picked the wrong polymorph.** Steps now resolve by the
  polymorph stable just below the melt rather than by first spacegroup match, so an element
  carrying several polymorphs of one spacegroup no longer loses the hot one.
- **One temperature convention for the ternary**: `T_grid` is Kelvin, clamped at absolute
  zero, and `conds` is the same window in Celsius derived from it, so the two cannot drift.
- **The 3D ternary scene used 61% of its figure.** The legend was anchored so that plotly
  reserved its width outside the plot area. The scene now fills the figure, and zooming is
  bounded by the whole plot area rather than by a smaller inner rectangle.
- **`Ce(FeSi)2` and other parenthesised formulas** rendered as `Ce (FeSi) 2` in figure labels.
- **The shipped model bundle's checksums did not verify off Windows.** `manifest.json`
  recorded each member's sha256 over CRLF bytes, while git hands the JSON and txt members
  back with LF on checkout, so `ProductionModelRunner`'s integrity check failed on Linux and
  macOS. The bundle is now excluded from end-of-line normalization and its checksums are
  recorded over the bytes that ship.

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

[Unreleased]: https://github.com/willwerj/gliquid_python/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/willwerj/gliquid_python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/willwerj/gliquid_python/releases/tag/v0.1.0
