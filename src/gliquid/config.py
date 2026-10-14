import functools
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

_DIR_STRUCT_OPTS = ["flat", "nested"]
_CACHE_MODE_OPTS = ["directory", "sqlite"]
_SS_REF_MODE_OPTS = ["from_omegas_file", "from_dft_entries", "from_unary_db"]

#: Environment variable naming the external cache corpus. See ``set_cache_dir``.
CACHE_DIR_ENV_VAR = "GLIQUID_CACHE_DIR"

#: Directory name of the default store, ``<cwd>/gliquid_cache``, created on the first write.
DEFAULT_CACHE_DIR_NAME = "gliquid_cache"

#: Environment variable that turns OFFLINE mode on at import. See :func:`set_offline`.
OFFLINE_ENV_VAR = "GLIQUID_OFFLINE"

# Recognized spellings for OFFLINE_ENV_VAR; any other value turns offline mode on, with a warning.
_OFFLINE_TRUE = frozenset({"1", "true", "yes", "on"})
_OFFLINE_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Environment variable that makes ternary phase imputation write to the cache store by default
#: (:func:`set_persist_ternary_imputed`). Same spellings as :data:`OFFLINE_ENV_VAR`; anything else
#: reads as off.
PERSIST_TERNARY_IMPUTED_ENV_VAR = "GLIQUID_PERSIST_TERNARY_IMPUTED"

# Suffixes that name a single-file cache store rather than a directory tree.
_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}

# Bundled reference tables ship in the wheel. The external store (``cache_dir``) holds DFT entry
# caches, MPDS diagrams and the model bundle, set by ``set_cache_dir()`` or ``GLIQUID_CACHE_DIR``.
_BUNDLED_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PHASE_TRANSITIONS_NAME = "phase_transitions.json"
_OMEGAS_NAME = "omegas_hcp.json"
_SPURIOUS_STRUCTURES_NAME = "spurious_structures.json"


class ConfigError(RuntimeError):
    """The external cache corpus is needed and no usable cache location is configured.

    Raised after ``set_cache_dir(None)``, and when the default ``<cwd>/gliquid_cache`` store
    cannot be created. Fix with ``gliquid.config.set_cache_dir(...)`` or by setting
    ``GLIQUID_CACHE_DIR``.
    """


class CacheModeError(ConfigError):
    """The requested operation needs a capability the configured store does not have.

    Raised for a write against a read-only single-file store, and for a lean MPDS record (see
    :func:`gliquid.mpds.record_mode`) reaching a consumer that needs the digitized ``shapes``.
    A :class:`ConfigError`; ``gliquid.cache.CacheModeError`` is this same class, re-exported.
    """


class OfflineError(ConfigError):
    """A remote fetch was attempted while gliquid is in OFFLINE mode.

    With :func:`set_offline` on, every remote path raises this at the call site, naming the
    system it was asked to fetch, instead of reaching the network. A ``ConfigError``: offline
    mode is a deliberate configuration of the package, not a fault in the data.
    """


def _removed_keywords(replacements: dict[str, str]):
    """Decorator factory: a call passing a keyword in ``replacements`` raises ``TypeError``
    naming what replaces it. ``inspect.signature`` shows the wrapped function's signature."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            removed = sorted(set(kwargs) & set(replacements))
            if removed:
                raise TypeError(
                    f"{func.__qualname__}() no longer takes "
                    + "; ".join(f"'{name}' (use {replacements[name]})" for name in removed)
                )
            return func(*args, **kwargs)

        return wrapper

    return decorate


def _require_extra(module_name: str, hint: str):
    """Import and return the top-level module ``module_name``.

    Raises:
        ImportError: with message ``hint``, chained from the failed import.
    """
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(hint) from exc


project_root = None
cache_dir = None
# The implicit <cwd>/gliquid_cache path installed at import; None once set_cache_dir runs.
_implicit_cache_dir = None
cache_mode = None
dir_structure = None
offline = None
persist_ternary_imputed = None
phase_transitions_file = None
omegas_file = None
spurious_structures_file = None
solid_solutions = None
ss_ref_mode = None
coverage_min_supported_total = None
coverage_ignored_floor_slope = None
coverage_ss_narrow_tol = None
coverage_dft_cover_tol = None
coverage_dft_edge_tol = None
coverage_dft_structure_tol = None
coverage_ss_rescue_max_width = None
coverage_subliquidus_margin = None
coverage_field_ceiling_tol = None
coverage_bound_by_supported = None
coverage_mask_miscibility_gaps = None
coverage_mig_foot_tol = None
coverage_ss_extension_cap = None
coverage_ss_width_guard = None
coverage_low_temp_axis_fraction = None
coverage_polymorph_pair_tol = None
gap_match_tol = None
liquidus_max_gap = None
liquidus_min_coverage = None
liquidus_gap_tol = None
assessed_crange_floor = None


def set_project_root(path: Path):
    global project_root
    project_root = path


def _reference_file(name: str) -> Path:
    """Resolve one reference table: ``cache_dir/name`` if that file exists, else bundled.

    Only an explicitly configured store is consulted, never the implicit default store (see
    :func:`_cache_dir_is_implicit`).
    """
    if cache_dir is not None and not _cache_dir_is_implicit():
        candidate = Path(cache_dir) / name
        if candidate.exists():
            return candidate
    return _BUNDLED_REFERENCE_DIR / name


def _cache_dir_is_implicit() -> bool:
    """Whether ``cache_dir`` is the implicit ``<cwd>/gliquid_cache`` default.

    True only while ``cache_dir`` equals the path installed at import because nothing was
    configured. :func:`set_cache_dir` (to any path, the same one included) ends it, and so
    does a direct assignment of ``cache_dir`` to a different path.
    """
    return (
        _implicit_cache_dir is not None
        and cache_dir is not None
        and Path(cache_dir) == _implicit_cache_dir
    )


def _is_file_store(path: Path) -> bool:
    """Whether ``path`` names a single-file cache store rather than a directory tree."""
    return path.suffix.lower() in _SQLITE_SUFFIXES or path.is_file()


def set_cache_mode(mode: str):
    """Select the cache backend: ``'directory'`` (a tree of JSON files) or ``'sqlite'``.

    Orthogonal to :func:`set_dir_structure`, which describes one particular on-disk
    arrangement WITHIN the directory backend and means nothing to a single-file store.
    """
    global cache_mode
    if mode not in _CACHE_MODE_OPTS:
        raise ValueError(f"cache_mode must be one of {_CACHE_MODE_OPTS}")
    cache_mode = mode


def set_cache_dir(path: Path | str | None):
    """Point gliquid at an external cache corpus (per-system DFT caches, MPDS diagrams).

    Accepts a directory OR a single file. A path that is an existing file, or whose suffix
    is one of ``.sqlite`` / ``.sqlite3`` / ``.db``, sets ``cache_mode='sqlite'``; anything
    else sets ``cache_mode='directory'``.

    Also re-resolves the three reference tables, each taken from ``path`` when a file of
    that name exists there and from the copy shipped inside the package otherwise. Passing
    ``None`` unsets the corpus (leaving ``cache_mode`` alone): reference tables then come
    from the package, and any read of the corpus raises ``ConfigError`` rather than
    guessing.

    A leading ``~`` in ``path`` expands to the home directory; a relative path is stored as
    given. Every call makes the store explicit, including one naming the
    ``<cwd>/gliquid_cache`` default's own path.
    """
    global _implicit_cache_dir

    _implicit_cache_dir = None
    _install_cache_dir(Path(path).expanduser() if path is not None else None)


def _install_cache_dir(path: Path | None, mode: str | None = None) -> None:
    """Set ``cache_dir`` to ``path`` and re-resolve the three reference tables.

    ``mode`` sets ``cache_mode`` when ``path`` is not ``None``; ``None`` infers it from the
    shape of ``path`` (see :func:`set_cache_dir`). Does not touch the implicit marker.
    """
    global cache_dir
    global phase_transitions_file
    global omegas_file
    global spurious_structures_file

    cache_dir = path
    if cache_dir is not None:
        if mode is None:
            mode = "sqlite" if _is_file_store(cache_dir) else "directory"
        set_cache_mode(mode)
    phase_transitions_file = _reference_file(_PHASE_TRANSITIONS_NAME)
    omegas_file = _reference_file(_OMEGAS_NAME)
    spurious_structures_file = _reference_file(_SPURIOUS_STRUCTURES_NAME)


def require_cache_dir(purpose: str = "This operation") -> Path:
    """The external corpus location, or ``ConfigError`` naming how to configure one.

    Directory stores resolve their root through here. ``cache_dir`` is ``None`` only after
    ``set_cache_dir(None)`` (or a direct assignment of ``None``); import always installs a
    location. The error names ``purpose``, ``set_cache_dir`` and ``GLIQUID_CACHE_DIR``.
    """
    if cache_dir is None:
        raise ConfigError(
            f"{purpose} needs the gliquid data corpus (per-system DFT entry caches and "
            f"digitized MPDS diagrams), and no cache store is set. Point "
            f"gliquid at a copy with gliquid.config.set_cache_dir('/path/to/cache'), or "
            f"set the {CACHE_DIR_ENV_VAR} environment variable before importing gliquid. "
            f"The reference tables ({_PHASE_TRANSITIONS_NAME}, {_OMEGAS_NAME}, "
            f"{_SPURIOUS_STRUCTURES_NAME}) ship with the package and are already loaded."
        )
    return Path(cache_dir)


def set_offline(enabled: bool):
    """Turn OFFLINE mode on or off. On, every remote fetch raises :class:`OfflineError`.

    Covers the Materials Project DFT-entry fetch (``api._get_dft_entries_from_components`` and
    the client constructors that reach it) and the MPDS live diagram fetch (``load_mpds_data``'s
    cache-miss branch and ``api.get_mpds_client``). An MPDS cache miss with no ``MPDS_API_KEY``
    fetches nothing in either mode: ``load_mpds_data`` logs a warning and returns
    ``{"reference": None}``.

    Also set at import by ``GLIQUID_OFFLINE=1`` (see :data:`OFFLINE_ENV_VAR`).
    """
    global offline
    offline = bool(enabled)


def require_online(purpose: str) -> None:
    """Raise :class:`OfflineError` naming ``purpose`` when offline mode is on.

    Called at the top of every remote path, before any credential lookup or client construction.
    """
    if not offline:
        return
    raise OfflineError(
        f"{purpose} needs a network fetch, and gliquid is in OFFLINE mode. Nothing was "
        f"requested. This means the configured cache store does not cover what was asked "
        f"for: either point gliquid at a store that does "
        f"(gliquid.config.set_cache_dir(...)), or -- if reaching out is genuinely wanted "
        f"here -- turn offline mode off with gliquid.config.set_offline(False) or by "
        f"unsetting {OFFLINE_ENV_VAR}."
    )


def _initial_offline() -> bool:
    """Offline mode at import, from :data:`OFFLINE_ENV_VAR`. Unset means online."""
    raw = os.environ.get(OFFLINE_ENV_VAR)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _OFFLINE_TRUE:
        return True
    if value in _OFFLINE_FALSE:
        return False
    logger.warning(
        "%s is set to %r, which is not one of %s or %s. Treating it as ON: setting the "
        "variable at all is an attempt to switch offline mode on, and reading an "
        "unrecognized value as 'stay online' would silently permit the network fetches "
        "this mode exists to forbid.",
        OFFLINE_ENV_VAR,
        raw,
        sorted(_OFFLINE_TRUE),
        sorted(_OFFLINE_FALSE - {""}),
    )
    return True


def set_persist_ternary_imputed(enabled: bool):
    """Whether ternary phase imputation writes its phases to the cache store by default.

    Off, :func:`gliquid.imputation.ternary_imputation.impute_ternary_phases` keeps imputed phases
    on the ternary object only. A call's own ``write_cache=`` overrides this setting either way.
    Also settable at import with ``GLIQUID_PERSIST_TERNARY_IMPUTED=1`` (see
    :data:`PERSIST_TERNARY_IMPUTED_ENV_VAR`).
    """
    global persist_ternary_imputed
    persist_ternary_imputed = bool(enabled)


def _initial_persist_ternary_imputed() -> bool:
    """Ternary imputed-phase persistence at import, from
    :data:`PERSIST_TERNARY_IMPUTED_ENV_VAR`. Unset or unrecognized means off."""
    raw = os.environ.get(PERSIST_TERNARY_IMPUTED_ENV_VAR)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _OFFLINE_TRUE:
        return True
    if value in _OFFLINE_FALSE:
        return False
    logger.warning(
        "%s is set to %r, which is not one of %s or %s. Treating it as OFF: ternary imputed "
        "phases are not written to the cache store unless a call asks for it.",
        PERSIST_TERNARY_IMPUTED_ENV_VAR,
        raw,
        sorted(_OFFLINE_TRUE),
        sorted(_OFFLINE_FALSE - {""}),
    )
    return False


def set_omegas_file(path: Path):
    global omegas_file
    omegas_file = Path(path)


def set_solid_solutions(enabled: bool):
    """Package-wide default for the ``solid_solutions`` param of ``BinaryLiquid.from_cache`` and
    ``TernaryLiquidInterpolation``.

    OFF (``False``) is the default: solid-solution models load only when this is truthy OR a
    caller passes an explicit ``solid_solutions=`` override. This is the single, uniform SS switch.
    """
    global solid_solutions
    solid_solutions = bool(enabled)


def set_ss_ref_mode(mode: str):
    """Default ``ref_mode`` used to resolve SS references when ``solid_solutions`` is on
    and the caller's ``ss_kwargs`` doesn't specify one. Mirrors
    ``solution.DEFAULT_REF_MODE`` (kept in sync by hand — ``solution`` imports ``config``,
    so ``config`` cannot import ``solution`` back).
    """
    global ss_ref_mode
    if mode not in _SS_REF_MODE_OPTS:
        raise ValueError(f"ss_ref_mode must be one of {_SS_REF_MODE_OPTS}")
    ss_ref_mode = mode


# Every coverage knob: name -> (cast, valid, requirement in the ValueError).
_COVERAGE_KNOBS = {
    "min_supported_total": (float, lambda v: 0 <= v <= 1, "in [0, 1]"),
    "ignored_floor_slope": (float, lambda v: v >= 0, ">= 0"),
    "ss_narrow_tol": (float, lambda v: 0 < v <= 1, "in (0, 1]"),
    "dft_cover_tol": (float, lambda v: 0 < v <= 1, "in (0, 1]"),
    "dft_edge_tol": (float, lambda v: 0 <= v <= 1, "in [0, 1]"),
    "dft_structure_tol": (float, lambda v: 0 <= v <= 1, "in [0, 1]"),
    "ss_rescue_max_width": (float, lambda v: 0 < v <= 1, "in (0, 1]"),
    "subliquidus_margin": (float, lambda v: v >= 0, ">= 0 K"),
    "field_ceiling_tol": (float, lambda v: v >= 0, ">= 0 K"),
    "bound_by_supported": (bool, lambda v: True, ""),
    "mask_miscibility_gaps": (bool, lambda v: True, ""),
    "mig_foot_tol": (float, lambda v: v >= 0, ">= 0 K"),
    "ss_extension_cap": (float, lambda v: v >= 0, "non-negative"),
    "ss_width_guard": (float, lambda v: v > 0, "positive"),
    "low_temp_axis_fraction": (float, lambda v: 0 < v <= 1, "in (0, 1]"),
    "polymorph_pair_tol": (float, lambda v: 0 <= v <= 1, "in [0, 1]"),
}

# Knobs a per-call override cannot reach: they act before the coverage assessment runs.
_CONFIG_ONLY_KNOBS = frozenset({"low_temp_axis_fraction"})

_RETIRED_COVERAGE_KNOBS = {
    name: "min_supported_total" for name in ("skip_frac", "min_missing", "missing_frac")
}


def _validated_coverage(values: dict) -> dict:
    """``values`` cast to each knob's type; raises ValueError on an unknown knob or a bad value."""
    unknown = sorted(set(values) - set(_COVERAGE_KNOBS))
    if unknown:
        retired = [repr(name) for name in unknown if name in _RETIRED_COVERAGE_KNOBS]
        raise ValueError(
            f"unknown coverage knob(s) {', '.join(map(repr, unknown))}"
            + (
                f"; {', '.join(retired)} removed in 0.3.0, use min_supported_total"
                if retired
                else ""
            )
            + f". The knobs are: {', '.join(_COVERAGE_KNOBS)}"
        )
    resolved = {}
    for name, value in values.items():
        cast, valid, requirement = _COVERAGE_KNOBS[name]
        resolved[name] = cast(value)
        if not valid(resolved[name]):
            raise ValueError(f"{name} must be {requirement}, got {value}")
    return resolved


@_removed_keywords(_RETIRED_COVERAGE_KNOBS)
def set_coverage_thresholds(
    *,
    min_supported_total: float | None = None,
    ignored_floor_slope: float | None = None,
    ss_narrow_tol: float | None = None,
    dft_cover_tol: float | None = None,
    dft_edge_tol: float | None = None,
    dft_structure_tol: float | None = None,
    ss_rescue_max_width: float | None = None,
    subliquidus_margin: float | None = None,
    field_ceiling_tol: float | None = None,
    bound_by_supported: bool | None = None,
    mask_miscibility_gaps: bool | None = None,
    mig_foot_tol: float | None = None,
    ss_extension_cap: float | None = None,
    ss_width_guard: float | None = None,
    low_temp_axis_fraction: float | None = None,
    polymorph_pair_tol: float | None = None,
):
    """Package-wide defaults for the solid-energy coverage gate (``solid_coverage.assess_solid_coverage``).

    This is the single home of every coverage threshold. Only the arguments passed are changed.
    Every knob but ``low_temp_axis_fraction`` can also be overridden for one call through
    ``fit_parameters(coverage_thresholds=...)`` or ``assess_solid_coverage(thresholds=...)``
    (see :func:`coverage_settings`). Composition widths are fractions of the composition axis.

    Args:
        min_supported_total: The verdict: skip a system when the total width of liquidus with a
            solid free-energy reference behind it is below this plus ``ignored_floor_slope``
            times the axis width with no digitized liquidus. In [0, 1]; 0 with a zero slope
            admits everything.
        ignored_floor_slope: How much the ``min_supported_total`` floor rises per unit of
            composition axis carrying no digitized liquidus. >= 0; 0 makes the floor constant.
        ss_narrow_tol: Solid-solution fields no wider than this count as represented by their
            endpoint line compounds and are not scored. In (0, 1].
        dft_cover_tol: Composition tolerance for "a DFT phase covers this composition" in
            ``BinaryLiquid.fit_parameters``' invariant-derived masking. In (0, 1].
        dft_edge_tol: How far past a drawn field's edges, or from a phase drawn at one
            composition, a DFT hull phase may lie and still match it whatever its structure.
            In [0, 1].
        dft_structure_tol: How far off an MPDS phase's stoichiometry or drawn range a DFT hull
            phase sharing its space group may lie and still match it (space groups 1 and 2 are
            never shared). In [0, 1]; 0 admits only exact and on-the-drawing matches.
        ss_rescue_max_width: Widest solid-solution field a nearby DFT compound may rescue.
            In (0, 1].
        subliquidus_margin: Kelvin. An unsupported phase whose digitized field tops out more than
            this below the lowest liquidus temperature across its own field masks nothing. 0 is
            the strict criterion; ``math.inf`` masks every unsupported phase. >= 0.
        field_ceiling_tol: Kelvin of slack when narrowing a masked field to where the liquidus is
            no hotter than the phase can exist. >= 0.
        bound_by_supported: Let a supported phase that meets the liquid bound a neighbour's
            primary crystallization field. Support verdicts (``narrow_field``,
            ``component_phase``, ``unknown_structure_ss_loaded``) and sub-liquidus phases do not
            bound.
        mask_miscibility_gaps: False removes a confirmed two-liquid dome from the mask.
        mig_foot_tol: Kelvin a dome's foot may sit off the digitized liquidus before its ``mig``
            record is refused. >= 0.
        ss_extension_cap: Most an unsupported component solution field's mask may extend past its
            drawn bounds on each side; ``math.inf`` keeps the full invariant-bounded extension.
            >= 0.
        ss_width_guard: When the drawn component fields masked together cover at least this much
            of the axis, ``ss_extension_cap`` is not applied; ``math.inf`` never guards. > 0.
        low_temp_axis_fraction: Fraction of a diagram's temperature-axis span that, added to the
            axis bottom, is the low-temperature cutoff (``mpds.low_temp_threshold``). In (0, 1].
        polymorph_pair_tol: Composition tolerance within which two thermal forms of one compound
            are the same compound (``mpds.collapse_polymorphs``, ``mpds.mark_unpaired_ht_forms``).
            In [0, 1].

    Raises:
        ValueError: for a value outside its range; nothing is changed then.
        TypeError: for a knob removed in 0.3.0 (``skip_frac``, ``min_missing``,
            ``missing_frac``).
    """
    passed = {name: value for name, value in locals().items() if value is not None}
    for name, value in _validated_coverage(passed).items():
        globals()[f"coverage_{name}"] = value


def coverage_settings(overrides: dict | None = None) -> dict:
    """Every coverage knob's value for one call: ``config`` with ``overrides`` on top.

    Args:
        overrides: knob name to value, any knob of :func:`set_coverage_thresholds` but
            ``low_temp_axis_fraction``; None values defer to ``config``.

    Returns:
        dict: knob name to value, cast to the knob's type, for every knob.

    Raises:
        ValueError: for an unknown or config-only knob, or a value outside its range.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    config_only = sorted(set(overrides) & _CONFIG_ONLY_KNOBS)
    if config_only:
        raise ValueError(
            f"{', '.join(config_only)} cannot be overridden per call; "
            f"set it with gliquid.config.set_coverage_thresholds"
        )
    current = {name: globals()[f"coverage_{name}"] for name in _COVERAGE_KNOBS}
    return {**current, **_validated_coverage(overrides)}


def coverage_default(name: str, value):
    """Return ``value``, or the package default of the coverage knob ``name`` when it is None.

    Args:
        name: A keyword parameter of :func:`set_coverage_thresholds`, e.g. ``"dft_edge_tol"``.
        value: The caller's value for the knob, or None for the package default.

    Returns:
        ``value`` itself, unconverted, when it is not None (``0``, ``0.0`` and ``False`` are
        values); otherwise the current ``config.coverage_<name>``, read at call time.

    Raises:
        ValueError: If ``name`` is not a coverage knob, whatever ``value`` is.
    """
    if name not in _COVERAGE_KNOBS:
        raise ValueError(
            f"unknown coverage knob {name!r}; the knobs are the keyword parameters of "
            f"set_coverage_thresholds: {', '.join(sorted(_COVERAGE_KNOBS))}"
        )
    if value is not None:
        return value
    return globals()[f"coverage_{name}"]


def set_gap_match_tol(value: float):
    """Package-wide composition slack for matching a generated two-liquid span to an
    assessed miscibility gap in the fitting objective.

    Args:
        value: composition width added to each side of every expected gap range before a
            generated span is tested against it; must be non-negative.

    Raises:
        ValueError: for a negative value.
    """
    global gap_match_tol
    value = float(value)
    if value < 0:
        raise ValueError(f"gap_match_tol must be non-negative, got {value}")
    gap_match_tol = value


def set_liquidus_coverage_thresholds(
    *,
    max_gap: float | None = None,
    min_coverage: float | None = None,
    gap_tol: float | None = None,
    assessed_crange: float | None = None,
):
    """Package-wide defaults for the liquidus interior-coverage gate in ``BinaryLiquid.from_cache``.

    ``from_cache`` measures the pre-fill digitized liquidus (``mpds.liquidus_coverage``) and flags
    ``init_error`` when it is interior-sparse. Only the arguments passed are changed. Every value
    is a fraction of the composition axis in (0, 1].

    Args:
        max_gap: Reject when the widest interval between consecutive digitized liquidus points
            exceeds this.
        min_coverage: Reject when less than this fraction of the stitched span is made of
            inter-point gaps no wider than ``gap_tol``.
        assessed_crange: Reject when less than this much of the axis carries digitized liquidus
            (the complement of the ignored range: outside the endpoints plus unbridged holes).
        gap_tol: Gap width up to which a stretch counts as locally sampled.

    Raises:
        ValueError: for a value outside (0, 1].
    """
    global liquidus_max_gap, liquidus_min_coverage, liquidus_gap_tol, assessed_crange_floor
    for name, value in {
        "max_gap": max_gap,
        "min_coverage": min_coverage,
        "gap_tol": gap_tol,
        "assessed_crange": assessed_crange,
    }.items():
        if value is not None and not 0 < float(value) <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {value}")
    if max_gap is not None:
        liquidus_max_gap = float(max_gap)
    if min_coverage is not None:
        liquidus_min_coverage = float(min_coverage)
    if gap_tol is not None:
        liquidus_gap_tol = float(gap_tol)
    if assessed_crange is not None:
        assessed_crange_floor = float(assessed_crange)


def set_dir_structure(structure: str):
    """How the DIRECTORY backend arranges files: ``'flat'`` or ``'nested'`` per system.

    A ``DirectoryBackend``-only knob, orthogonal to :func:`set_cache_mode`. Under
    ``cache_mode='sqlite'`` there are no directories to arrange, and this call LOGS and
    returns rather than raising.
    """
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    if cache_mode == "sqlite":
        logger.info(
            "Ignoring set_dir_structure('%s'): cache_mode is 'sqlite', which stores every "
            "record in one file and has no directory layout to choose.",
            structure,
        )
        return
    dir_structure = structure


def find_project_root(dirname="gliquid_python") -> Path | None:
    """The source checkout this file lives in, or ``None`` when there is not one.

    Package-anchored ONLY: with the src layout this file is at
    ``<project>/src/gliquid/config.py``, so a development checkout is always an ancestor of
    ``__file__``. Installed into site-packages nothing matches and the answer is ``None``.

    There is no working-directory walk, so a cwd's ``cache/`` is never adopted as the
    corpus. An installed package falls back to ``<cwd>/gliquid_cache`` instead (see
    :func:`_initial_cache_dir`), an implicit store that never supplies the reference
    tables.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == dirname:
            return parent
    return None


def _initial_cache_dir() -> tuple[Path, bool]:
    """Store at import and whether it is the implicit default.

    Order: ``GLIQUID_CACHE_DIR`` -> a checkout's ``cache/`` ->
    ``(<cwd>/gliquid_cache).resolve()``. Only the last is implicit (``True``). An environment path has a leading ``~`` expanded. Nothing is created.
    ``set_cache_dir()`` sits above all of them and is applied by the caller afterwards.
    """
    env_value = os.environ.get(CACHE_DIR_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_dir() and not _is_file_store(candidate):
            # Logged, not raised: an unreadable corpus fails where it is read.
            logger.warning(
                "%s is set to '%s', which is not a directory. gliquid will use "
                "it anyway; reads of the data corpus will fail there.",
                CACHE_DIR_ENV_VAR,
                candidate,
            )
        return candidate, False
    root = find_project_root()
    if root is not None and (root / "cache").is_dir():
        return root / "cache", False
    return (Path.cwd() / DEFAULT_CACHE_DIR_NAME).resolve(), True


def _apply_initial_cache_dir() -> None:
    """Install the import-time store from :func:`_initial_cache_dir`.

    The implicit default is marked before its reference tables are resolved, and is always
    a directory store, even when a regular file of that name exists.
    """
    global _implicit_cache_dir

    path, implicit = _initial_cache_dir()
    if not implicit:
        set_cache_dir(path)
        return
    _implicit_cache_dir = path
    _install_cache_dir(path, mode="directory")


class _ConfigModule(ModuleType):
    """Module type that rejects writes to the removed ``data_dir`` attribute."""

    def __setattr__(self, name, value):
        if name == "data_dir":
            raise AttributeError("gliquid.config.data_dir was removed; use cache_dir")
        super().__setattr__(name, value)


set_project_root(find_project_root())
# cache_mode and dir_structure are applied before _apply_initial_cache_dir().
set_cache_mode(_CACHE_MODE_OPTS[0])
set_dir_structure(_DIR_STRUCT_OPTS[0])
_apply_initial_cache_dir()
set_offline(_initial_offline())
set_persist_ternary_imputed(_initial_persist_ternary_imputed())

sys.modules[__name__].__class__ = _ConfigModule
set_solid_solutions(False)
set_ss_ref_mode("from_unary_db")
# Coverage verdict: a floor on the total supported width, as a fraction of the composition axis.
set_coverage_thresholds(
    min_supported_total=0.30,
    ignored_floor_slope=0.5,
    ss_narrow_tol=0.10,
    dft_cover_tol=0.10,
    dft_edge_tol=0.01,
    dft_structure_tol=0.05,
    ss_rescue_max_width=0.25,
    subliquidus_margin=0.0,
    field_ceiling_tol=10.0,
    bound_by_supported=True,
    mask_miscibility_gaps=False,
    mig_foot_tol=20.0,
    ss_extension_cap=0.05,
    ss_width_guard=0.67,
    low_temp_axis_fraction=0.05,
    polymorph_pair_tol=0.04,
)
set_liquidus_coverage_thresholds(max_gap=0.45, min_coverage=0.50, gap_tol=0.10, assessed_crange=0.7)
set_gap_match_tol(0.10)
