# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The distribution version is
derived from the git tag by `hatch-vcs` — tagging `vX.Y.Z` *is* the release — so nothing in the
source tree carries a version literal that could disagree with this file.

## [Unreleased]

### Changed

- **The liquid is linearized at its own liquidus, in binaries and ternaries alike.** A liquid whose
  excess terms depend on T ('combined', 'comb-exp') takes its H and S at a composition-dependent
  temperature T*(x), not at the mean component melting point. The first hull takes the
  x-weighted melting point. Each further hull re-evaluates every row at the previous hull's
  liquidus, the step halved where it grew, until no row's |G_lin(T_liq; T*) - G(T_liq)| exceeds
  `DEFAULT_TSTAR_TOL` (0.2143 J/mol, about 2 K of liquidus) or `DEFAULT_TSTAR_MAX_HULLS` (60)
  hulls are built. A binary also needs every row's estimated liquidus shift, that difference
  over the entropy step across its liquidus, at most `DEFAULT_TSTAR_LIQ_TOL` (1 K): near the
  top of a liquid miscibility gap the entropy step vanishes, and a row that meets the energy
  tolerance can still sit tens of K off its converged liquidus as a one-row spike
  (`BinaryLiquid.liquidus_shift_estimate`). A binary checks the rows inside its digitized liquidus; a ternary checks
  every liquid node and stops after `DEFAULT_TERNARY_TSTAR_MAX_HULLS` (3) hulls, the B pass and
  two D passes. On the 96 ternaries of the melting-point benchmark that cap leaves every node's
  liquidus within 3.7 K of the converged one and no melting point or eutectic moved, while
  holding the slowest systems to about 1.7 s per interpolation instead of 2.9 s.
  - The first pass restricts exactly to each binary on a ternary edge, so a ternary edge no longer
    depends on the third element.
  - `liquid_tstar` (on `BinaryLiquid` and `TernaryLiquidInterpolation`) holds the last
    evaluation's T*.
  - Parameters fitted before this change, and the v29.05 model's predictions, render
    differently and should be refit and retrained: the notebook systems' liquidus moves by up to
    10.5 K (Pb-Tb).
  - Refit over the 853 SOLIDCOV_Q binaries, the fits' exact comb-exp liquidus matches MPDS with a
    mean absolute error of 77.4 K, against 79.9 K for fits under the mean melting point, and
    fitting takes about twice as long.
- **The unexpected-gap penalty counts only real gaps.** A two-liquid tie of the fitted hull
  counts only where the liquid's exact Gibbs energy, with the solids, also ties two liquids at
  that temperature.
- **Nelder-Mead returns its best vertex at `max_iter`** instead of raising.
- **Every initial simplex spans at least `DEFAULT_B_SPREAD_MIN` in L*_b.** A small L*_b vertex
  is moved by the spread rather than scaled by 0.8.
- **`liquidus_is_continuous` allows one missing grid row**, as its tolerance intends, despite
  floating-point error in the grid spacing.
- **Ternary imputation reads the liquid's Gibbs energy at the target temperature.**
  `TernaryLiquidInterpolation.liquid_gibbs` takes H and S at the temperature asked for, as the
  binary imputation does. A ternary imputed entry with a T-dependent edge recorded before this
  change reports itself stale (`liquid_gibbs` provenance).
- **`LiquidusSheets` and `Binodal` take a `TstarField`** (the liquid's T*(x)) in place of one
  temperature. `TernaryLiquidInterpolation.liquid_tstar_field()` builds it.
- **One test suite.** `tests_internal/` is merged into `tests/`, so a bare `pytest` also runs the
  value pins, figure goldens and cache/network-gated families. Deselect them by marker
  (`-m "not pins"`, `-m "not needs_cache"`).
- **The unary reference data are anchored on Materials Project GGA/GGA+U energies.**
  `phase_transitions.json` was built from MP's mixed GGA/GGA+U/r2SCAN energies while every
  convex hull uses GGA entries; every DFT energy in it is now the GGA/GGA+U thermo energy. Sources
  are used in a fixed order: MP-GGA for the ground state and every phase it has, the omegas GGA
  lattice stabilities (Chen et al., Nat. Commun. 14, 2856 (2023)) for BCC/FCC/HCP references MP
  lacks or places below the ground state, and SGTE or an assessed literature value only where the
  observed 1 atm phase order needs it or no DFT BCC/FCC/HCP value lands above the ground state.
  - Ten ground states move to the stable 1 atm structure: K and Cs to bcc, Kr and Xe to fcc,
    As to rhombohedral, Se to trigonal, Pu to alpha-Pu (monoclinic), N to alpha-N2 (Pa-3), H to
    solid para-H2 (hcp, mp-24504), and Na to its 9R martensite (R-3m), with bcc from 36 K.
    `expected_gs_spacegroup` in `spurious_structures.json` now covers 85 elements (62 observed
    ground states, the rest from SGTE's stable element reference or the existing anchor), so the
    hull anchor guard covers them too.
  - Every real phase whose ladder enthalpy is SGTE or assessed (28 MP entries, such as gamma-Fe,
    fcc-Co, beta-Zr and the Pu allotropes) is in `spurious_structures.json` with
    `category: "sgte_corrected"`, so no corrected DFT entry reaches a hull; the ladder phase keeps
    the structure and its id becomes `sgte-<El>-<spacegroup>`. 26 further entries with no 1 atm
    stability field are blacklisted as `spurious`, each with a literature source, among them the
    former H anchor (mp-730101, the H sublattice of ammonium dideuterium phosphate); only fibrous
    red P (mp-1198724) is still marked `status: "derived; literature review pending"`. The 7
    computational BCC/FCC/HCP prototypes among them carry `ss_structure`.
  - Liquid references move for fifteen elements: Pu by -47568 J/mol (every Pu step is now SGTE),
    Ti by -2765 J/mol (beta-Ti from 13146.9 to 10381.94 J/mol), Li by -436 J/mol and
    -4.955 J/mol/K (the 9R -> bcc step at 74 K instead of 80 K; Maier et al., Phys. Rev. B 55,
    113 (1997)), Na by +429 J/mol and +11.93 J/mol/K (the 9R -> bcc step at 36 K), Sr by
    +576 J/mol and +0.628 J/mol/K, Ba by +730 J/mol and +0.73 J/mol/K, and Ca by -31 J/mol and
    -0.043 J/mol/K. Ca, Sr and Ba take the assessment of Alcock, Chase & Itkin (J. Phys. Chem.
    Ref. Data 22, 1 (1993)). Ca: fcc -> bcc at 716 K with 930 J/mol instead of an unverified
    961 J/mol. Sr: fcc -> bcc at 820 K with 850 J/mol, and fusion at 1041 K with 8000 J/mol,
    instead of SGTE's 844 J/mol at 830 K and CRC's 7430 J/mol at 1050 K. Ba: fusion with
    7850 J/mol instead of CRC's 7120 J/mol, at the same 1000 K. La, Ce, Sm and Gd take Konings &
    Beneš (J. Phys. Chem. Ref. Data 39, 043102 (2010)), Mn, Fe and the Be transition take
    NIST-JANAF (Chase 1998), and Be's fusion takes Alcock et al.: Sm by +1063 J/mol and
    +1.135 J/mol/K (alpha -> beta at 970 K and beta -> gamma at 1185 K instead of 1190 K and
    1345 K, with 100, 3170 and 9100 J/mol steps), Be by +1028 J/mol and +0.659 J/mol/K (fusion
    8000 J/mol instead of 6972 J/mol), Ce by -620 J/mol and -0.579 J/mol/K (fusion 5580 J/mol
    instead of 6200 J/mol), Gd by +183 J/mol and +0.064 J/mol/K (alpha -> beta at 1534 K with
    3940 J/mol, fusion at 1591 K with 9800 J/mol), Tl by +171 J/mol and +0.289 J/mol/K (fusion
    4150 J/mol and SGTE's 360 J/mol beta step), Sn by -124 J/mol and -0.433 J/mol/K (SGTE's
    1966 J/mol beta step instead of 2090 J/mol), Mn by -81 J/mol and +0.034 J/mol/K (2226, 2122
    and 1879 J/mol steps at 980, 1361 and 1412 K) and La by -36 J/mol (beta at 566 K, gamma at
    1140 K, fusion 6140 J/mol at 1192 K).
  - Solid-solution references (`lattice_stabilities`): 95 change by more than 0.5 J/mol, 21 are
    new and 7 are dropped. Omegas values are measured from the element's MP-GGA ground-state
    energy instead of a shared structure, so FCC-Ga becomes +1923 J/mol instead of 0. Mg and Cd
    FCC come from
    omegas, and In FCC and Hg FCC/HCP from SGTE, because the DFT values before them land below
    the ground state. Hg BCC, which no DFT or SGTE source places above the ground state, takes
    the enthalpy of the fitted end member of the Hg-Tl assessment (Gierlotka, Sopousek &
    Fitzner, Calphad 30, 425 (2006)): +5540 J/mol, id `assessed-Hg-229`, instead of +336 J/mol.
    A structure with no source above the ground state is not emitted and is listed under the
    element's new `excluded_lattice_stabilities` (Eu FCC, Xe BCC).
  Fitted parameters made before this change do not carry over and must be refit.

- **A ladder phase is a solid solution's end member only if it has that structure's
  prototype.** Spacegroup 194 holds hcp (hP2), dhcp and graphite (hP4); `Phase` now carries
  `pearson_symbol`, and `ComponentRef.ss_polymorphs` picks the end member by spacegroup and
  prototype. A dhcp or graphite ground state stands in for HCP only when no hP2 lattice stability
  exists, and gliquid logs a warning when it does. Pr's HCP solid solution now uses the hP2
  lattice stability (+2792.66 J/mol) and alpha-Pr (dhcp) stays a line compound beside it; C, La,
  Nd and Sm keep their SG 194 phase as a stand-in.

- **Candidate Nelder-Mead fits converge on one scale and start Lupis-Elliott-consistent.** The
  convergence test weighs an `L*_b` coordinate by the mean liquidus temperature, so `L*_a` and
  `L*_b` are measured by their contribution to `L_k` (J/mol); the tolerance is
  `CANDIDATE_TOL_J_MOL` (100 J/mol). `BinaryLiquid.nelder_mead` takes the per-coordinate `scale`.
  The offset vertex of a candidate's initial simplex moves `L*_b` toward excess entropy of the
  sign of `L*_a` at 0 K, in the linear and `exp(-T/tau)` forms alike.

### Fixed

- **The Bi-Sr hull follows the assessed diagram's compounds.** The Materials Project hull has
  vertices at Sr2Bi3 (mp-31150, x_Sr = 0.400) and Sr4Bi3 (mp-30893, x_Sr = 0.571), which the
  assessed Bi-Sr diagram does not draw, and takes the P6_3/mcm polymorph of Sr5Bi3 (mp-29620).
  All three are now in `spurious_structures.json`; the hull becomes SrBi3, Sr11Bi10 and the
  Pnma Sr5Bi3 (mp-1197242).

## [0.3.1] - 2026-10-17

### Added

- **Seed-averaged model bundles.** `ProductionModelRunner` loads a bundle of several models
  trained at different random seeds and predicts their mean, target by target in physical units.
  A portable bundle declares them as `members` (bundle schema 2, one subdirectory per model with
  its own `preprocess.json` and `<target>.ubj`); a joblib bundle keeps them in
  `model/seed_<s>/`, ordered by `bundle_manifest.json`'s `seeds`. SHAP explanations of such a
  bundle are the mean of its members' explanations, and its golden vector is computed through the
  same mean. Single-model bundles (schema 1) load and predict exactly as before.

### Changed

- **Titanium is referenced to alpha-Ti (hcp), not omega-Ti.** omega-Ti (mp-72, P6/mmm) is a
  high-pressure phase that GGA ranks 15 meV/atom below hcp; it is now in
  `spurious_structures.json`, and Ti's expected ground state is SG 194. The ladder loses its
  omega-to-alpha step at 180 K: beta-Ti moves from H = 14610.62 J/mol, S = 19.514 J/(mol K) to
  H = 13146.9 J/mol, S = 11.3826 J/(mol K), and liquid Ti from H = 28760.62 J/mol,
  S = 26.7966 J/(mol K) to H = 27296.9 J/mol, S = 18.6652 J/(mol K). The melting point stays at
  1943 K. Every hull containing Ti changes, since Ti compound formation energies are re-referenced
  to alpha-Ti. Fitted parameters for Ti binaries made before this change do not carry over and must
  be refit. Refit with the 0.3.0 campaign settings, the six Ti-containing compounds in the
  101-compound ternary melting-point benchmark go from 292.5 K to 139.2 K mean absolute error: the
  RE-Ti-Ge compounds move from a mean error of -332 K to -130 K, while TiAlAu moves from -93 K to
  -183 K. The full benchmark goes from 151.2 K to 142.1 K. The 32 fitted Ti binaries lose some
  accuracy, from 81.9 K to 90.6 K mean liquidus error, mostly in refractory compound-forming
  systems (Ru-Ti, C-Ti, B-Ti, Re-Ti, Fe-Ti).

- **The compact SHAP figure has a fixed layout.**
  `ProductionModelRunner.create_compact_prediction_figure` saves at a fixed size
  (`COMPACT_FIGURE_SIZE`, 8.0 x 4.915 in), sizes its label column for the widest feature name of
  the naming scheme so every bundle draws its bars at the same place, and labels each bar with the
  feature name as written.

- **The packaged model is v29.05** (bundle `20261017_003407`, replacing `20261013_191431`).
  It is the mean of eight XGBoost models trained at different seeds (bundle schema 2), on the fits
  refit for the alpha-Ti and Au-Tl changes above, with 49 symmetric and 30 antisymmetric features.
  On its 100 held-out test systems it predicts the liquidus with a mean absolute percentage error
  of 11.99 % (median 10.49 %) and a mean absolute error of 178.7 K. Every predicted parameter
  changes from 0.3.0.

### Fixed

- **The Au-Tl hull no longer carries AuTl2.** The assessed Au-Tl diagram is a simple eutectic
  and draws no field at x_Tl = 0.667, but the Materials Project hull has an AuTl2 vertex there
  (mp-30571). It and its same-composition polymorph mp-755435, which would otherwise take its
  place on the hull, are now in `spurious_structures.json`.

- **The spurious-structure blacklist matches new-API Materials Project ids.** An entry whose id
  is in the new API's letter spelling (`mp-aaaaadxv`) is matched against a rule written with the
  classic number (`mp-2647`), whether the id is a bare string or a dict. Such entries used to
  pass the blacklist unfiltered.

- **The Colab notebooks install on current Colab.** Their install cell adds `pymatgen<2026`, which
  keeps Colab's preinstalled `requests` 2.32.4, and enables Colab's custom widget manager, which
  the hull editor needs.

## [0.3.0] - 2026-10-13

### Added

- **Phase-energy imputation, in the new `gliquid.imputation` package.** When the DFT hull lacks a
  solid phase the assessed diagram draws, imputation gives it a formation enthalpy from the fitted
  liquid and places it on the hull as a tagged entry (`entry_id` prefix `IMPUTED_ID_PREFIX`), drawn
  dashed. `impute_phases`, `drop_imputed_phases`, `imputed_entry_staleness` and
  `stale_imputed_entries` take a `BinaryLiquid` or a `TernaryLiquidInterpolation`. The submodules
  load on first use.

- **Binary imputation.** `impute_phases(bl, fitting_data=...)` imputes each low-temperature solid
  bordering the liquid that the hull lacks (`list_binary_imputation_targets(bl)` lists them), at the
  enthalpy that reproduces the invariant it controls. `protect_dft_hull=True` (the default) keeps
  drawn DFT phases on the hull. The model changes in memory only: `write_cache=True` stores the
  entries, and a fit reads stored ones only with `include_imputed=True`.

- **Ternary imputation.** `TernaryLiquidInterpolation.impute_phase(composition, temperature, ...)`,
  `TLIPlotter.impute_phase` and `impute_phases(tli, targets)` place a solid that leaves the hull at
  that temperature; `liquid_composition=` reads the enthalpy off the liquid's tangent instead, and a
  phase that would displace a stable one warns and is placed only with `force=True`.
  `adjust_phase_temperature(phase, temperature)` moves an existing phase. Each call returns a
  `TernaryImputationResult`.

- **New ternary model methods:** `liquid_gibbs`, `liquid_model`, `hull_table` and `solid_hull`
  (both take `exclude=`), `all_imputed_entries`, `imputation_params`, `reset_derived_state` and
  `imputed_phase_names`. `adjust_ternary_phase_temperature(tli, phase, temperature)` is the
  function form of `adjust_phase_temperature`.

- **Ternary imputed phases are stored only on request.** They are written with `write_cache=True`,
  or by default after `config.set_persist_ternary_imputed(True)` /
  `GLIQUID_PERSIST_TERNARY_IMPUTED=1`. `cache.OverlayBackend(base)` keeps writes in memory for one
  session. `TernaryLiquidInterpolation(..., include_imputed=True)` places the stored imputed entries
  of the system and its three edges.

- **Imputed entries in either store.** `cache.KIND_IMPUTED_ENTRIES` names a system's imputed
  entries in a directory or SQLite store; `api.get_dft_convexhull(..., include_imputed=True)` reads
  them from either, and `api.read_imputed_entries`, `api.cache_imputed_entries` and
  `api.drop_imputed_entries` read, write and clear them.

- **Imputed entries on any hull.** `reanchor_imputed_entries(hull, entry_dicts)` rebuilds tagged
  entries on another hull from their stored formation enthalpy, and `imputed_hull(components,
  entry_dicts, ...)` returns the cached DFT hull with them. `TernaryLiquidInterpolation(
  imputed_entries=...)` and `from_binaries(..., include_imputed=True)` place them on a ternary hull.

- **`gliquid.imputation.core`** holds what binary and ternary imputation share, including
  `provenance_header` and `hull_with_reanchored_entries`.

- **Phase ablation, in `gliquid.imputation.ablation`.** `ablate_phases` removes interior DFT phases
  from a system's hull for a re-fit, `restore_phases` puts them back, and `list_ablatable_phases`
  lists the candidates.

- **`gliquid.solid_coverage`.** `assess_solid_coverage` measures how much of a system's digitized
  liquidus has a solid free-energy reference and returns a `SolidCoverageReport` of `PhaseCoverage`
  records (moved from `gliquid.mpds`; see Removed). `assign_dft_phases` matches DFT hull phases to
  MPDS phases one-to-one.

- **Public helpers in `gliquid.mpds`:** `compound_key`, `formula_comp`, `FORMULA_TOL`,
  `is_ordered_field_label`, `require_full_record`, `liquidus_crossings`, `find_local_maxima`,
  `find_local_minima`, `SEGMENT_TOL`, `mig_feet`, `mig_feet_on_liquidus`, `validated_mig_feet`,
  `bounding_invariant_comps`, `subtract_ranges` and `flank_bound`.
- **Public helpers elsewhere:** `config.coverage_default` and `config.DEFAULT_CACHE_DIR_NAME`;
  `cache.is_imputed_entry_dict` and `cache.imputed_entry_label`; `api.hull_entry_name` and
  `api.DEFAULT_DFT_TYPE`; `phase.polymorph_ladder`; `plotting.style.abbreviate_phase_name`;
  `BinaryLiquid.rebuild_hull_interp()` and `BinaryLiquid.reset_fit_state()`.

- **Fit results name their constraint temperatures and dropped candidates.** Fit result dicts carry
  `constr_temps`, and `BinaryLiquid.dropped_candidates` names the candidates that never ran.
- **Hopeless candidates are not optimised.** `fit_parameters` skips a candidate whose initial
  objective exceeds `init_objective_cutoff` times the best candidate's (default
  `DEFAULT_INIT_OBJECTIVE_CUTOFF`, 100 000) and lists it on `BinaryLiquid.pruned_candidates`.
  `init_objective_cutoff=None` runs every candidate.
- **The fit objective penalises a liquid miscibility gap the assessed diagram does not draw.** A
  generated two-liquid tie outside `BinaryLiquid.expected_gap_ranges`, widened by
  `config.gap_match_tol` (`set_gap_match_tol`), scales the objective by
  `(1 + width / GAP_WIDTH_SCALE) ** GAP_EXPONENT`. `suppress_unexpected_migs=False` turns it off.
- **`mig` invariant records carry `apex`, `partial` and `source`.** A dome with no usable apex is
  `partial` with `apex` None; `mpds.mig_apex` reads old and new records. Every printed two-liquid
  label is processed (`mpds.is_two_liquid_label`).
- `HSX.two_liquid_spans` and `hsx.two_liquid_spans_from_facets`.
- **`min_comp_span` sets the composition span a liquidus comparison must cover**, on
  `BinaryLiquid.calculate_deviation_metrics` and `fit_parameters`. A narrower comparison reports
  `inf`; `None` keeps 0.7. `min_comp_span=0` with `allow_sparse_data=True` scores a few points.
- **User-defined phases.** A phase added or edited in `ConvexHullEditor`, or an entry with no
  `entry_id`, is user-defined: tagged by `USER_ID_PREFIX` (`"user:"`) or
  `data[NON_DFT_KEY] == "user"`, built by `api.user_entry(...)`, and reported by
  `cache.non_dft_entry_kind`, `cache.non_dft_entry_label`, `Phase.non_dft` and `phase.non_dft_kind`.
  Figures draw it dashed under `plotting.style.USER_DEFINED_LEGEND_LABEL` (see
  `non_dft_legend_label`); `plot_tx(non_dft_label=)` sets that legend text.
- **Ternaries carry the edges' user-defined phases and removals.**
  `TernaryLiquidInterpolation(user_entries=..., excluded_entries=...)` places user-defined entries
  in place of the DFT entries each lists under `data['replaces']`, with any entropy in
  `data[cache.ENTROPY_KEY]`, and leaves excluded ids off; `from_binaries` collects both from its
  edges (`ConvexHullEditor.apply()` sets `removed_entry_ids`). `user_phase_names()` names them.
- **`TernaryLiquidInterpolation.edge_hull(edge)`** is the hull the ternary places on one binary
  edge, and the edge figures draw from it. `BinaryLiquid.from_cache(hull=...)` builds from a given
  hull. `cache.canonical_entry_id` spells classic and new-API Materials Project ids alike.

- **One solid phase's liquidus field, highlighted on the ternary figure.**
  `TLIPlotter.get_plot('tx', highlight_phase=<phase>)` draws that phase's primary field in colour
  over a desaturated surface. The fields come from per-phase liquidus sheets
  (`liquidus_sheets.LiquidusSheets`, `plotting.liquidus_fields.partition_fields`); a liquid
  miscibility gap is its own `L1 + L2` field. Where the sheets do not describe the hull's liquidus,
  the surface is drawn without fields and a warning names the system.
- **Clicking a legend entry of the ternary figure highlights that phase's field.**
  `get_plot('tx')` embeds every legend phase's highlighted view (`interactive_legend=False` leaves
  them out, for a figure about 2.5x smaller); `ternary_surface.dash_clientside_function()` attaches
  the handler in Dash.
- **The ternary liquidus figure runs its browser script wherever it is drawn.** `get_plot('tx')`
  returns a `ternary_surface.LiquidusFigure`, whose `show()`, `write_html`/`to_html`, and display
  as a notebook cell's result carry `ternary_surface.post_script()` (in a notebook kernel, `show()`
  draws it with `side_by_side_html`, so it is interactive in JupyterLab, VS Code and Colab). New
  `TLIPlotter.write_html(plot_type, file, **kwargs)`.

- **Composition ticks and flat labels on the ternary liquidus figure.** `get_plot('tx')` ticks each
  edge every 10 at.%, titles each edge and names each corner, with labels lying flat on the floor.
  In a browser they turn with the scene to stay readable; `ternary_labels.orient_labels(fig)` fits
  them to a camera set in Python. The scene's x and y axes share one scale.
- **Ternary mixing models accept each binary pair in either order.** `xs_mix` and `fit_or_pred` on
  `TernaryLiquidInterpolation` (and `TLIPlotter.from_components`) may name a pair against the edge
  convention, e.g. `'Al-Si'` for the `'Si-Al'` edge; it is re-keyed with the odd Redlich-Kister
  orders flipped (`ternary.orient_to_edges`). An `'A-B-C'` triplet entry may name its elements
  in any order. Naming one pair or triplet in two orders raises `ValueError`.
- **`get_convex_hull` and `get_plot('ch', T_celsius=...)` work in `hull_mode='hsx'`**, from the
  interpolated surface at the nearest grid temperature.
- **`scripts/run_colab_notebook.py` runs a notebook as Google Colab does**, in a Linux container
  with Colab's package versions, against a wheel built from a local commit, and reports each cell's
  warnings and errors and every Colab package the notebook's `pip install` replaces.
- **`gliquid.plotting.export.side_by_side_html(figures, configs=, post_scripts=, height=)`** lays
  Plotly figures out in equal columns as one HTML fragment for `IPython.display.HTML`, each column
  an iframe of the figure's standalone page (a `LiquidusFigure` brings its own script), drawing in
  Colab, JupyterLab, VS Code, classic Jupyter and exported notebooks.
- **Flat ternary liquidus projections, in `gliquid.plotting.ternary_projection`.**
  `draw_liquidus_panel(ax, record, sorted_components, order, levels)` draws the composition
  triangle and `draw_closeup_panel(..., window)` a window of it: filled and labelled isotherms,
  monovariant lines, one label per primary field, and outer composition ticks (`ticks=False` leaves
  them off). `levels_for` and `window_levels` build the colour scales.
- **A liquidus-MAPE chip on fitted ternary edge figures.** `TLIPlotter(..., edge_mape=True)` (and
  `build_binary_edge_figures(ti, edge_mape=True)`) puts `MAPE x.x%` above the top-left corner of
  each fitted edge's plot frame, scored against the edge's digitized MPDS liquidus on the hull the
  edge is drawn with. The chip is colored by `gliquid.plotting.style.mape_color` (reversed
  `RdYlGn` over 0-40 %, piecewise linear through `MAPE_COLOR_KNOTS` so it changes fastest between
  5 and 20 %) with black or white text from `contrast_text`; predicted edges get none.
  `gliquid.plotting.binary_tx.add_corner_note(fig, text, fill)` places such a chip on any figure,
  as an annotation named `CORNER_NOTE_NAME`.
- **Predicted ternary edges can draw their digitized MPDS liquidus.** `TLIPlotter(...,
  pred_liquidus=True)` (and `build_binary_edge_figures(ti, pred_liquidus=True)`) draws the assessed
  liquidus on every predicted edge that has one, beside the predicted liquidus; with `edge_mape`
  such an edge also gets the MAPE chip.
- **Progress reporting for ternary plots.** `TLIPlotter.process_data(progress=fn)` calls
  `fn("model", 1, 1)`, then `fn("edges", k, n)` after each binary edge figure, then
  `fn("hull", 1, 1)`; `build_binary_edge_figures(ti, progress=fn)` reports the edges alone.

### Changed

- **The default model is v29.03** (`gliquid/models/20261013_191431/`), replacing v24.01. It keeps
  the v24.01 architecture (per-target pseudo-Huber XGBoost; `L1_a = 0.5 * (f(A-B) - f(B-A))`) and
  is retrained on the 0.3.0 fits (DFT hull only, no imputation), with 50 order-invariant and 31
  order-dependent features that add DFT hull and hull-area descriptors to the element features.

- **`ProductionModelRunner.predict_system` accepts either element order.** `predict_system("Mg-Cu")`
  returns the `Cu-Mg` prediction with `L1_a` negated; `L0_a` and `L0_b` are identical.

- **Imputed and user-defined phases are drawn with a finer dash in the `tx` plots,** and the
  unsupported stretches of the assessed liquidus are dash-dotted instead of dotted. The imputed
  phase lines and their legend key use `plotting.style.IMPUTED_LINE_DASH` (`"4px,2px"`, was
  plotly's `"dash"`); the ternary edge figures, which share the `tx` renderer, change with them.
  The 3D ternary surface keeps `"dash"`.

- **`fit+liq` and `pred+liq` mark the unsupported stretches of the assessed liquidus only for
  `unsupported_range_mode='ignore'` and `'cap'` fits.** A `'score'` fit (the default) scores those
  ranges like any other composition, so its assessed liquidus is drawn in one style.
  `fit_parameters` stores the mode it ran under as `BinaryLiquid.fit_unsupported_range_mode`
  (`None` before any fit, which marks `unsupported_comp_ranges` as before); `reset_fit_state`
  clears it and `with_component_order` carries it.

- **Ternary hover text gives the composition in at. % of all three elements,** as
  `<element>: <x> at. %` lines in the order of the plotter's `sys_name`, on the `tx` liquidus
  surface and its phase-field meshes and on the `ch` hull slice. They replaced the mole fractions
  of the two non-base components (`x_<element>`); the trace `customdata` columns are now the three
  at. % values, to one decimal place, followed by the trace's other hover fields.

- **Lean MPDS records keep only what a fit+liquidus plot reads:** the pre-fill stitched liquidus,
  its covered regions, `chemical_elements`, `temp`, `comp_range` and `reference`. `labels` and the
  top-level `entry`, `jcode` and `year` are dropped; `cache.verify` compares those four header
  fields on a lean store. Full records and full stores are unchanged.

- **A compound melting as a distinct high-temperature polymorph is fitted, not masked.** Its
  thermal forms are one phase: the low-temperature form's hull enthalpy stands in for the melting
  form, the liquidus beside it is fitted, and the invariants it controls become constraint
  candidates. 0.2.0 fits of such systems cannot be reproduced.
- **`dft_type='GGA'` fetches carry composition-guessed MP2020 corrections.**
  `api.apply_mp2020_corrections(entry_dicts)` recomputes `correction` and `energy_adjustments` with
  oxidation states guessed from each composition, as the shipped binary corpus does, and the GGA
  fetch applies it before caching. `MIXED` and `R2SCAN` fetches are unchanged; caches written
  before this change keep the served corrections.
- **A ternary hull warns when it disagrees with its binary edges.** The first load of the hull of a
  ternary built `from_binaries` logs a warning when an entry shared with an edge hull carries a
  different energy (`cache.entry_energies_by_id`, `cache.entry_energy_drift`).

- **The ternary liquidus surface is drawn at the liquidus of the sheets.** `get_plot('tx')` draws
  the surface through the hull's liquidus nodes and the field-boundary points, each at the hottest
  sheet's temperature, so its valleys lie on the field boundaries; it lies above the hull's node
  temperatures by the hull's grid error. Without sheets the surface is the hull's.
- **Ternary isotherms are continuous.** Contour segments are joined through the mesh edges they
  cross, so contours no longer break into pieces on fine or uneven meshes.
- **A ternary liquidus node's "Coexistent Phases" are those of every hull facet meeting it at its
  liquidus**, so the hover names every phase whose field meets that point.
- **The ternary liquidus legend lists solid phases only.** The `L` entry is gone; `color_map`
  still carries it. `get_plot('tx')` raises `TypeError` on a keyword it does not take, and
  `highlight_phase` / `interactive_legend` raise `ValueError` with `get_plot('ch')`.

- **With nothing configured, per-system data is stored in `./gliquid_cache`**
  (`config.DEFAULT_CACHE_DIR_NAME`, fixed when G<sub>Liquid</sub> is imported, created by the first write)
  instead of raising `ConfigError`. The store is flat: `<A-B>_ENTRIES_MP_GGA.json` and `<A-B>.json`.
  `set_cache_dir()`, `GLIQUID_CACHE_DIR` and a source checkout's `cache/` still come first; the
  default store never supplies the reference tables.
- **A directory store is created on its first write, and `~` expands** in `set_cache_dir("~/data")`
  and `GLIQUID_CACHE_DIR=~/data`.
- **The ML feature table ships in the wheel.** The shipped model bundle carries its feature table
  (`features_symmetric.json`, `features_antisymmetric.json`), which `ProductionModelRunner` reads
  first, so with `pip install gliquid[ml]` `ProductionModelRunner().predict_system("Cu-Mg")` works
  with no store configured.
- **The notebooks run on the model that ships in the wheel.** The runner notebooks construct
  `ProductionModelRunner()` with no arguments, needing no joblib bundle.
- **A failed MPDS request is no longer cached as "no diagram".** Only the service's "no results"
  answer (code 204) writes the `{"reference": None}` placeholder; any other client error is logged
  and raised, and nothing is written. A placeholder an earlier version wrote after a failed request
  still reads as no diagram: delete that system's MPDS file (e.g. `Cu-Mg.json`) to fetch it again.
- **A read-only store fails before fetching.** On a system missing from a store G<sub>Liquid</sub> cannot
  write to, `api.get_dft_convexhull`, `api.get_dft_structure_entries`, `api.cache_imputed_entries`
  and `mpds.load_mpds_data` raise `CacheModeError` without contacting the Materials Project or
  MPDS. In offline mode they still raise `OfflineError`.
- **A ternary refuses to read its binary edges from the default store when it has its own.** A
  `TernaryLiquidInterpolation(data_dir=...)` on the implicit `./gliquid_cache` raises
  `ConfigError` naming both locations; call `config.set_cache_dir()` with the ternary's store first.
- **`hull_editor.J_PER_MOL_PER_EV` is now `96485.0`**, a float: it is the same object as
  `phase.EV_ATOM_TO_J_MOL`.
- **The solid-coverage gate decides on how much constraining liquidus is left.** A system is
  skipped when its supported liquidus plus any confirmed two-liquid dome covers less than
  `min_supported_total` of the composition axis (default 0.30; `config.set_coverage_thresholds`,
  or per fit `coverage_thresholds={'min_supported_total': ...}`). `skip_frac`, `min_missing` and
  `missing_frac` are gone (see Removed); the report's `reason` and `thresholds` record each rule.
  The floor rises with the axis that carries no digitized liquidus: it is `min_supported_total +
  ignored_floor_slope * ignored_total`, with the new coverage knob `ignored_floor_slope` defaulting
  to 0.5 (0 restores a constant floor).
- **`binary.MASK_FRACTION_CAP` is 0.55** (was 0.60): `fit_parameters` skips a system with
  `mask_fraction_exceeded` when the invariant-derived unsupported ranges mask more than 55% of the
  digitized liquidus span.
- **An unsupported component solid-solution field masks the liquidus it controls**: its drawn
  range plus the primary field its flanking invariants give it, extended at most
  `ss_extension_cap` of the axis per side (default 0.05) unless the masked component fields cover
  at least `ss_width_guard` (default 0.67). Both are coverage thresholds, and `math.inf` switches
  either off; the report's `thresholds` record `ss_drawn_width` and `ss_guard_fired`.
- **DFT phases are matched to the assessed phases one-to-one** by
  `solid_coverage.assign_dft_phases`, read by the coverage gate, `matched_dft_comps` and the
  imputation target check: at the phase's stoichiometry or drawn range within `dft_edge_tol`
  (default 0.01), or within `dft_structure_tol` (default 0.05) when the space groups match.
  `dft_cover_tol` now applies only to `fit_parameters`' invariant-derived masking.
- **A printed two-liquid region is not masked as a missing solid's field.** Inside a region the
  diagram prints as `L1 + L2`, located by its `mig` record where there is one and by the label
  otherwise, an unsupported solid's field is not masked.
- **The dome constraints are gated on what the record knows.** The top-of-dome equation is emitted
  only when the apex is known, and the bottom-of-dome equation only when both feet share one
  isotherm and rest on the digitized liquidus within `coverage_mig_foot_tol`.
- **`ConvexHullEditor` keeps every entry it does not change**, with its object, `entry_id`, `data`
  and structure. An added or edited entry becomes a user-defined `ComputedEntry` with the id
  `user:<name>` (suffixed `#2`, `#3`, ... on a clash), and an edited DFT entry lists the id it
  replaces under `data['replaces']`. `editor.entries` rows gain `non_dft`. Removing an entry
  changes only the editor's hull.
- **`TernaryLiquidInterpolation.from_binaries` sets `fit_or_pred`**, so `TLIPlotter` draws the
  binary-edge figures of a ternary built from binaries: `'fit'` for an edge with a digitized
  liquidus, `'pred'` otherwise; an explicit `fit_or_pred=` still wins. Edge figures take their
  phases from the ternary's own hull on that edge.

- **`hull_mode='hsx'` is the default for ternary interpolation** in `TernaryLiquidInterpolation`
  and `TLIPlotter.from_components`. Pass `hull_mode='gtx'` for the per-temperature Gibbs slices
  on the `T_incr` grid.
- **The notebooks install G<sub>Liquid</sub> by version on Colab.** Each Colab setup block pins one
  `gliquid==` version with no extras and clones the repository only for its `cache/` data; the
  notebooks configure the store with `set_cache_dir` and display figures through `BLPlotter.show`.
- **`colab_demo.ipynb` runs the whole pipeline on Ga–Ru–Si**: the assessed Ru–Si diagram, a hull
  edit, Ru–Si and Ga–Si fits, an ML prediction of Ga–Ru, the interpolated ternary liquidus with the
  Ru₂Si₃ field highlighted, and the DSC-based revision from `GaRu_adjustment.ipynb`. Its data ships
  in `cache/`, so it runs without API keys.
- **Importing `gliquid.api` no longer imports `pymatgen.entries.compatibility`.** It loads on first
  use, so importing `gliquid.api`, `gliquid.binary` or `gliquid.ternary` is faster.
- **`TernaryLiquidInterpolation(solid_solutions=)` defaults to `config.solid_solutions`**, as
  `BinaryLiquid.from_cache` does (was `False`), and its solid-solution models load with
  `config.ss_ref_mode` unless `ss_kwargs` names a `ref_mode`.
- **`param_format` defaults to `'comb-exp'`** (was `'linear'`) in `BinaryLiquid`,
  `BinaryLiquid.from_cache`, `RKPolyExp` and `TernaryLiquidInterpolation`. All three formats
  take the same four parameters, so a parameter list given without a format is now read as
  comb-exp, and a fit without one pins `L1_b`; pass `param_format='linear'` for the old reading.

### Removed

- **The `editor`, `shap`, `models` and `ml-gpu` extras.** The extras are now `mpds`, `ml` and
  `notebook`. `gliquid[notebook]` installs the widget stack `ConvexHullEditor` needs, alongside
  Jupyter. SHAP explanations need `pip install shap`, and a legacy joblib bundle needs
  `pip install joblib openpyxl scikit-learn==1.7.1`; each error message names its install.
  For a CUDA build of XGBoost, install `xgboost` yourself and leave out `[ml]`.

- **The `data_*` spellings of the cache configuration**, with no alias: `config.set_data_dir()`,
  `config.require_data_dir()`, `config.data_dir`, `config.DATA_DIR_ENV_VAR`, `gliquid.set_data_dir`
  and `GLIQUID_DATA_DIR`. Use `set_cache_dir()`, `require_cache_dir()`, `config.cache_dir`,
  `CACHE_DIR_ENV_VAR` and `GLIQUID_CACHE_DIR`. `api._resolve_sys_dir()` is gone;
  `api.resolve_cache_path(CacheKey(...))` names a record's path. The per-call `data_dir=` keyword
  of `TernaryLiquidInterpolation`, `TLIPlotter.from_components` and `get_dft_convexhull` is unchanged.
- **The 0.2.0 parameter `input` is renamed with no alias:** `system` on
  `api.get_dft_convexhull`, `api.get_dft_structure_entries`, `api.cache_imputed_entries`,
  `BinaryLiquid.from_cache`, `mpds.load_mpds_data` and `phase.validate_and_format_system`, and
  `params` on `BinaryLiquid.update_params`. Passing `input=` raises `TypeError` naming the new
  keyword.
- **`assess_solid_coverage`, `SolidCoverageReport` and `PhaseCoverage` live only in
  `gliquid.solid_coverage`.** `gliquid.mpds` no longer resolves them.
- `ProductionModelRunner.create_beeswarm_figures` and
  `ProductionModelRunner.apply_feature_updates`, and the `gliquid.cache` constants
  `ML_FEATURE_CODEC` and `POOL_META_KEYS`.
- BLPlotter plot type `'vch'` (it drew the `'ch+g'` figure); use `'ch'` or `'ch+g'`.
- `gliquid.ternary.build_polymorph_transitions`. Import it from `gliquid.plotting.binary_tx`.
- **0.2.0 spellings renamed since**, each raising `TypeError` naming the new one:
  `fit_parameters(ignored_ranges=)` and `calculate_deviation_metrics(ignored_ranges=)` are
  `unsupported_range_mode='ignore'` or `'score'`; `BinaryLiquid(ignored_comp_ranges=)` and its
  attribute are `unsupported_comp_ranges`; `from_cache(comp_range_fit_lim=)` is
  `assessed_crange_floor`. Assigning `bl.comp_range_fit_lim` has no effect.
- **`BinaryLiquid.to_HSX` is renamed `to_hsx`,** with the same parameters and no alias: calling
  `to_HSX` raises `AttributeError`.
- **`fit_parameters(check_phase_mismatch=)`**, which has gated nothing since 0.1.0, and
  **`fit_parameters(num_points=)`** raise `TypeError`; `RETIRED_FIT_KWARGS` is gone.
- **`BinaryLiquid(...)` takes its options as keyword-only parameters.** An unknown keyword
  raises `TypeError` instead of being ignored, and `init_triangle=` is no longer accepted.
- **Per-knob coverage keywords.** Every coverage knob is overridden for one call through one
  dict, `fit_parameters(coverage_thresholds={...})` or `assess_solid_coverage(thresholds={...})`,
  validated by `config.coverage_settings`; `low_temp_axis_fraction` is config-only. The 0.2.0
  keywords `fit_parameters(coverage_ss_narrow_tol=, coverage_dft_cover_tol=,
  coverage_ss_rescue_max_width=)` and `assess_solid_coverage(ss_narrow_tol=, dft_cover_tol=,
  ss_rescue_max_width=)` raise `TypeError` naming the dict entry that replaces them.
- **`set_coverage_thresholds(skip_frac=, min_missing=, missing_frac=)`.** Passing one raises
  `TypeError` naming `min_supported_total`, and the keys leave `report.thresholds`.
- The `'distinct_melting_polymorph'` and `'polymorph_transition_temp'` keys of the phase
  dicts `mpds.collapse_polymorphs` and `mpds.identify_invariant_points` return.
- The `'scale'` key of `lupis_elliott_cfg`, an alias of `'strength'`. Passing it raises
  `TypeError` naming `'strength'`.
- `gliquid.mpds.SS_SPACEGROUPS` and `gliquid.plotting.binary_tx.subscript_formula`. Import them
  from `gliquid.phase` and `gliquid.plotting.style`.
- `gliquid.api.atomic_write_json` and its alias `gliquid.api._atomic_write_json`. Import
  `atomic_write_json` from `gliquid.cache`.
- **`TLIPlotter(temp_slider=, T_incr=, hull_mode=)`.** The plotter uses the hull settings of the
  interpolation it wraps: set them on `TernaryLiquidInterpolation`, or pass them to
  `TLIPlotter.from_components`, which forwards them. Passing one to `TLIPlotter` raises
  `TypeError` naming `TernaryLiquidInterpolation`.
- **`TernaryLiquidInterpolation(ternary_l0=)`.** Give the ternary term as an `'A-B-C'` entry of
  `xs_mix`; on a ternary built with `from_binaries`, assign
  `ti.xs_mix['A-B-C'] = RKPolyExp('regular', [L0])` before interpolating. Passing `ternary_l0=`
  raises `TypeError` naming the `xs_mix` entry.
- **`TernaryLiquidInterpolation.from_binaries(components=)`.** `order=[...]` builds in the same
  order. Passing `components=` raises `TypeError` naming `order=`.

### Fixed

- **An entropy-stabilized compound is drawn from its formation temperature up.** On a `fit+liq`
  T-x diagram, a compound above the 0 K enthalpy hull (stable only where its entropy lowers its
  free energy, as after a positive solid entropy adjustment) starts at the solid-state reaction
  that forms it; it used to be drawn down to the plot floor.
- **A per-fit `dft_cover_tol` reaches the invariant-derived masking.**
  `fit_parameters(coverage_thresholds={'dft_cover_tol': ...})` sets the tolerance within which
  a DFT phase covers a missing terminal phase; the per-fit value used to be recorded on the
  report and ignored.
- **`fit_parameters` with one attempt logs a failing attempt and returns `[]`,** as it does
  with several; an exception other than `RuntimeError` from the only attempt used to
  propagate.
- **`BinaryLiquid` docstrings state the contract the code keeps.** `f` returns the liquidus
  RMSE times its penalty factors, not the MAE; `check_full_ss` stops invariant identification
  early; `get_plot` lists every plot type; the class docstring lists every attribute.
- **`lower_convex_hull` drops vertical simplices on every edge of a ternary.** Zero-area
  simplices on the `x0 + x1 = 1` edge stayed on the gtx hull and tied solids to liquids across
  that edge. Liquidus temperatures are unchanged; the `get_plot('ch')` slice loses its vertical
  curtain on that edge.

- **Coverage and fitting run on a hull edited in `ConvexHullEditor`.** `fit_parameters`,
  `dft_assignment` and `assess_solid_coverage` raised `AttributeError` on an edited or
  hand-built hull. An entry with no id, and any imputed or user-defined entry, has no space
  group; DFT entries the editor leaves alone keep theirs.
- **`ConvexHullEditor.apply()` keeps the model consistent with the edited hull.** It keeps
  the model's solid-solution phases (it dropped them, even with no edit), orders endpoint
  polymorphs as `from_cache` does, rebuilds `eqs['h_hull_interp']` from the edited 0 K hull, and
  recomputes the phase points and an already computed `hsx`, so later fits and `BLPlotter`
  figures see the edited hull.
- **`ConvexHullEditor`'s temperature no longer drops compounds.** `apply()` keeps every phase
  stable at 0 K plus those stable at the editor temperature, including a differently named
  phase at a 0 K compound's composition, and elemental reference entries keep S = 0 (they were
  seeded with a high-temperature polymorph's entropy). The editor's table marks a phase kept
  because it is stable at 0 K, and `entries` reports `stable_at_0k`.
- **`ConvexHullEditor` loads each phase's entropy onto one entry**: the closest in formation
  enthalpy among the same-named entries at its composition, else among all entries there, not
  every entry at that composition. Re-running the editor no longer hands a phase's entropy to
  its metastable twins.
- **`TernaryLiquidInterpolation.from_binaries` gives each user-defined entry its own phase's
  entropy.** Two user entries at one edge composition each took the entropy of whichever
  phase there was listed last.
- **`ConvexHullEditor.add_entry` rejects a name already in use**: `'L'`, a solid-solution or
  endpoint polymorph phase, or an entry at another composition. Its change log no longer
  raises `UnicodeEncodeError` on a console that cannot encode the log's symbols.
- **`BLPlotter`'s `'hsx'` and `'scatter'` figures follow the plotted model**: a reused plotter
  colours them by the model's current phases, and a model whose `hsx` was dropped (by a copy
  or a pickle) has it recomputed.
- **Offline mode proceeds without an MPDS diagram when no MPDS key is set.** On an MPDS cache
  miss with no `MPDS_API_KEY`, `mpds.load_mpds_data` logs a warning and returns
  `{"reference": None}` in either mode instead of raising `OfflineError` offline. With a key set,
  offline mode still raises `OfflineError` before the fetch.
- **An invariant-constraint candidate whose equations do not determine the fit's solve
  parameters is dropped** before scoring and named on `dropped_candidates`. It was admitted, ran
  free of the equation it was named for, and could be reported as the winning constraint.
- **The Nelder-Mead simplex update follows the textbook rules.** A reflection better than the
  second-worst vertex is accepted, a shrink moves every vertex but the best halfway to it, and
  infinite vertices of the initial simplex are moved onto the feasible side. Vertex objectives are
  cached between iterations. Fits change; a campaign is not comparable across this change.
- **Initial `L*_b` guesses of the `exp(-T/tau)` formats start on the S_xs(0 K) = 0 line**,
  at `L*_a / tau`, rather than on the linear formats' tau line, and a zero `L*_b` reference no
  longer collapses the H-S partition simplex.
- **The eutectic `0th` equation used the liquid free energy where its slope belongs.** The
  tangent condition is `g_L + g_L' (x_s − x_e) = g_s`. A fit whose winner was an `eut 0th`
  candidate is affected.
- **`comb-exp` fits no longer discard their hottest invariant equation.** Every equation is now its
  own candidate for the one-equation format, so a system whose only equation is a dome's feet
  equation has an invariant-constrained candidate.
- **Reversed-frame diagrams can match their dome polygon.** Label positions and shape
  coordinates were read in the record's own component frame while the liquidus was in the
  evaluation frame.

- **`BLPlotter.show` displays Matplotlib figures in Jupyter and Colab.** Under the inline
  backend the `'pc'` and `'nmp'` figures are displayed where `show` is called, once, instead of
  warning `FigureCanvasAgg is non-interactive`.
- **The `'pc'` phase-comparison figure no longer warns about tick labels.** Its composition
  axis sets the tick positions together with the at.% labels.
- **The adjustment notebooks fit again.** `GaRu_adjustment.ipynb` and `PbTb_adjustment.ipynb`
  passed `check_phase_mismatch=False`, which had no effect, so the coverage gate skipped both
  fits; they pass `check_solid_coverage=False`.
- **Decimal stoichiometries are subscripted whole.** `subscript_formula` and
  `abbreviate_phase_name` render `Fe1.7Ge` as `Fe<sub>1.7</sub>Ge` (it was
  `Fe<sub>1</sub>.7Ge`), in the ternary legend and hover text and in binary field labels
  written as atomic fractions (`Y<sub>0.177</sub>Zn<sub>0.823</sub>`).

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

[Unreleased]: https://github.com/willwerj/gliquid_python/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/willwerj/gliquid_python/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/willwerj/gliquid_python/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/willwerj/gliquid_python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/willwerj/gliquid_python/releases/tag/v0.1.0
