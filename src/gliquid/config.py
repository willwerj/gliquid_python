import logging
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

_DIR_STRUCT_OPTS = ["flat", "nested"]
_CACHE_MODE_OPTS = ["directory", "sqlite"]
_SS_REF_MODE_OPTS = ["from_omegas_file", "from_dft_entries", "from_unary_db"]

#: Environment variable naming the external cache corpus. See ``set_cache_dir``.
CACHE_DIR_ENV_VAR = "GLIQUID_CACHE_DIR"

#: Environment variable that turns OFFLINE mode on at import. See :func:`set_offline`.
OFFLINE_ENV_VAR = "GLIQUID_OFFLINE"

# Recognized spellings for OFFLINE_ENV_VAR. Anything else FAILS CLOSED (offline on) with a
# warning: setting the variable at all is an attempt to switch offline mode ON.
_OFFLINE_TRUE = frozenset({"1", "true", "yes", "on"})
_OFFLINE_FALSE = frozenset({"", "0", "false", "no", "off"})

#: The pre-0.2 spelling of :data:`CACHE_DIR_ENV_VAR`. Still honored, deprecated. Served to
#: consumers as ``config.DATA_DIR_ENV_VAR`` (with a ``DeprecationWarning``) rather than as a
#: plain global, so the package's own uses of it stay silent.
_LEGACY_DIR_ENV_VAR = "GLIQUID_DATA_DIR"

# Suffixes that name a SINGLE-FILE cache store rather than a directory tree. Handing
# someone one file and having it just work is the point of the sqlite format, so
# ``set_cache_dir`` infers the mode from the shape of the argument.
_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}

# gliquid's data comes in two kinds, in two places.
#
#   BUNDLED (inside the wheel) -- the three reference tables below. Small, and the library
#   cannot compute anything without them.
#
#   EXTERNAL (``cache_dir``) -- the per-system DFT entry caches, the MPDS diagrams and the
#   model bundle. Megabytes of corpus, not shipped, reachable only through
#   ``set_cache_dir()`` or ``GLIQUID_CACHE_DIR``. In a source checkout it is ``cache/``.
_BUNDLED_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PHASE_TRANSITIONS_NAME = "phase_transitions.json"
_OMEGAS_NAME = "omegas_hcp.json"
_SPURIOUS_STRUCTURES_NAME = "spurious_structures.json"


class ConfigError(RuntimeError):
    """The external cache corpus is needed and no cache location is configured.

    Raised in place of guessing a location. Fix with ``gliquid.config.set_cache_dir(...)``
    or by setting ``GLIQUID_CACHE_DIR``.
    """


class CacheModeError(ConfigError):
    """The requested operation needs a capability the configured store does not have.

    Raised in place of inventing an answer. Three situations reach it, and they share a
    shape -- *the store you configured cannot answer this question, and guessing would be
    worse than stopping*:

    * ``api._resolve_sys_dir`` asks "which DIRECTORY holds this system", which is
      unanswerable once the store is a single file;
    * a write against a read-only single-file store (the default and the design);
    * a **lean** MPDS record reaching a consumer that needs the digitized ``shapes`` --
      see :func:`gliquid.mpds.record_mode`. That one is the dangerous case, because the
      shapeless record does not crash anything: ``identify_mpds_phases`` would return
      ``[]``, the solid-coverage gate would read "no unsupported compounds", and the gate
      would PASS. The raise exists so a reduced store cannot silently answer a question it
      threw the evidence away for.

    Lives here rather than in ``gliquid.cache`` because it is a *configuration* fault --
    the store, or the mode it was built in, does not match what the caller needs -- and so
    that ``except gliquid.config.ConfigError`` catches every "gliquid is pointed at the
    wrong data" failure in one place. ``gliquid.cache.CacheModeError`` is this same class,
    re-exported, so existing imports and ``except`` clauses are unaffected.
    """


class OfflineError(ConfigError):
    """A remote fetch was attempted while gliquid is in OFFLINE mode.

    Offline mode exists because "has no API key" is not a property anything enforces. A
    deployment that simply *never sets* ``NEW_MP_API_KEY`` still runs every network code
    path: ``api.get_api_key`` falls through to a gitignored ``.env``, an operator's shell
    may export the variable, and a cache miss on one uncovered system then reaches out --
    slowly, and to an endpoint the deployment has no credentials for. The failure surfaces
    as a timeout or a 401 raised deep inside ``mp_api``, naming nothing about the cause.

    With :func:`set_offline` on, every remote path raises THIS instead, at the call site,
    naming the system it was asked to fetch. A missing record then reads as "this store
    does not cover that system", which is a fact about the store and is actionable, rather
    than as an opaque client error. A ``ConfigError`` because it is a deliberate
    configuration of the package, not a fault in the data.
    """


# ---------------------------------------------------------------------------------------
# Deprecated ``data_*`` spellings of the ``cache_*`` names.
#
# 0.1.0 is public and the old names are everywhere -- hundreds of driver scripts, the
# notebooks and the test suites -- so they stay WORKING aliases rather than becoming
# errors. Each one announces itself once per process; see ``_ConfigModule`` at the bottom
# of this file for why a bare module ``__getattr__`` is not enough to implement them.
# ---------------------------------------------------------------------------------------

# old name -> (the global that actually holds the value, the name to recommend instead).
# The two differ for DATA_DIR_ENV_VAR: its VALUE is still the legacy 'GLIQUID_DATA_DIR'
# string (it names a different environment variable, so it cannot simply forward to
# CACHE_DIR_ENV_VAR), but the name to migrate to is CACHE_DIR_ENV_VAR.
_DEPRECATED_ATTRS = {
    "data_dir": ("cache_dir", "cache_dir"),
    "DATA_DIR_ENV_VAR": ("_LEGACY_DIR_ENV_VAR", "CACHE_DIR_ENV_VAR"),
}

_DEPRECATION_WARNED: set[str] = set()


def _warn_deprecated(key: str, message: str) -> None:
    """Emit ``message`` as a ``DeprecationWarning`` the first time ``key`` is used.

    Once per process, not once per call: ``config.data_dir`` is read inside loops over
    thousands of systems, and a warning per read would bury everything else.
    """
    if key in _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _warn_deprecated_attr(old: str, new: str) -> None:
    _warn_deprecated(
        old,
        f"gliquid.config.{old} is deprecated and will be removed in a future release; "
        f"use gliquid.config.{new} instead. The old name still works and stays in sync "
        f"with the new one.",
    )


project_root = None
cache_dir = None
cache_mode = None
dir_structure = None
offline = None
phase_transitions_file = None
omegas_file = None
spurious_structures_file = None
solid_solutions = None
ss_ref_mode = None
coverage_skip_frac = None
coverage_min_missing = None
coverage_missing_frac = None
coverage_ss_narrow_tol = None
coverage_dft_cover_tol = None
coverage_ss_rescue_max_width = None
liquidus_max_gap = None
liquidus_min_coverage = None
liquidus_gap_tol = None


def set_project_root(path: Path):
    global project_root
    project_root = path


def _reference_file(name: str) -> Path:
    """Resolve one reference table: ``cache_dir/name`` if that file exists, else bundled.

    The existence check is what lets ``cache_dir`` point at a PARTIAL corpus -- a directory
    holding only per-system caches still yields a working unary registry -- while a
    directory that does carry its own copy keeps overriding the shipped one, which is how
    the reference tables are iterated on during development.
    """
    if cache_dir is not None:
        candidate = Path(cache_dir) / name
        if candidate.exists():
            return candidate
    return _BUNDLED_REFERENCE_DIR / name


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
    else sets ``cache_mode='directory'``. Being handed one file and having it just work is
    the point of the single-file format, so the shape of the argument decides rather than a
    second call the caller has to remember.

    Also re-resolves the three reference tables, each taken from ``path`` when a file of
    that name exists there and from the copy shipped inside the package otherwise. Passing
    ``None`` unsets the corpus (leaving ``cache_mode`` alone): reference tables then come
    from the package, and any read of the corpus raises ``ConfigError`` rather than
    guessing.
    """
    global cache_dir
    global phase_transitions_file
    global omegas_file
    global spurious_structures_file

    cache_dir = Path(path) if path is not None else None
    if cache_dir is not None:
        set_cache_mode("sqlite" if _is_file_store(cache_dir) else "directory")
    phase_transitions_file = _reference_file(_PHASE_TRANSITIONS_NAME)
    omegas_file = _reference_file(_OMEGAS_NAME)
    spurious_structures_file = _reference_file(_SPURIOUS_STRUCTURES_NAME)


def require_cache_dir(purpose: str = "This operation") -> Path:
    """The external corpus location, or ``ConfigError`` naming how to configure one.

    Every read of the external corpus resolves through here rather than reading
    ``cache_dir`` directly, so an unconfigured corpus is a loud error instead of a path
    built from a guess. There is deliberately no working-directory fallback: guessing is
    what turned a configuration error into wrong numbers.
    """
    if cache_dir is None:
        raise ConfigError(
            f"{purpose} needs the gliquid data corpus (per-system DFT entry caches and "
            f"digitized MPDS diagrams), which is not shipped with the package. Point "
            f"gliquid at a copy with gliquid.config.set_cache_dir('/path/to/cache'), or "
            f"set the {CACHE_DIR_ENV_VAR} environment variable before importing gliquid. "
            f"(The former spellings set_data_dir(...) and {_LEGACY_DIR_ENV_VAR} still work "
            f"and are deprecated.) "
            f"The reference tables ({_PHASE_TRANSITIONS_NAME}, {_OMEGAS_NAME}, "
            f"{_SPURIOUS_STRUCTURES_NAME}) ship with the package and are already loaded."
        )
    return Path(cache_dir)


def set_offline(enabled: bool):
    """Turn OFFLINE mode on or off. On, every remote fetch raises :class:`OfflineError`.

    Two paths are covered, and they are the only two in the package that talk to a remote
    host: the Materials Project DFT-entry fetch (``api._get_dft_entries_from_components``,
    plus the client constructors that reach it) and the MPDS live diagram fetch
    (``mpds.load_mpds_data``'s cache-miss branch, plus ``api.get_mpds_client``).

    RAISES rather than degrades, deliberately, and the MPDS path is why. On a cache miss
    with no ``MPDS_API_KEY`` that function already logs a warning and returns
    ``{"reference": None}`` -- a record shaped exactly like "this system has no digitized
    diagram". A silent skip under offline mode would be indistinguishable from that real
    answer, and a fit would proceed against a diagram that was never consulted.

    Also settable at import with ``GLIQUID_OFFLINE=1`` (see :data:`OFFLINE_ENV_VAR`),
    which is how a container turns it on without editing code.
    """
    global offline
    offline = bool(enabled)


def require_online(purpose: str) -> None:
    """Raise :class:`OfflineError` naming ``purpose`` when offline mode is on.

    Called at the top of every remote path, BEFORE any credential lookup or client
    construction, so offline mode is decided by configuration rather than by whether a key
    happens to be installed.
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


def set_data_dir(path: Path | str | None):
    """Deprecated alias of :func:`set_cache_dir`."""
    _warn_deprecated(
        "set_data_dir",
        "gliquid.config.set_data_dir() is deprecated and will be removed in a future "
        "release; use gliquid.config.set_cache_dir() instead. The old name still works.",
    )
    set_cache_dir(path)


def require_data_dir(purpose: str = "This operation") -> Path:
    """Deprecated alias of :func:`require_cache_dir`."""
    _warn_deprecated(
        "require_data_dir",
        "gliquid.config.require_data_dir() is deprecated and will be removed in a future "
        "release; use gliquid.config.require_cache_dir() instead. The old name still works.",
    )
    return require_cache_dir(purpose)


def set_omegas_file(path: Path):
    global omegas_file
    omegas_file = Path(path)


def set_solid_solutions(enabled: bool):
    """Package-wide default for ``BinaryLiquid.from_cache``'s ``solid_solutions`` param.

    OFF (``False``, the default) is byte-identical to pre-SS behavior: ``from_cache``
    loads ``ss_models`` only when this is truthy OR a caller passes an explicit
    ``solid_solutions=`` override. This is the single, uniform SS switch.
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


def set_coverage_thresholds(
    *,
    skip_frac: float | None = None,
    min_missing: int | None = None,
    missing_frac: float | None = None,
    ss_narrow_tol: float | None = None,
    dft_cover_tol: float | None = None,
    ss_rescue_max_width: float | None = None,
):
    """Package-wide defaults for the solid-energy coverage gate (``mpds.assess_solid_coverage``).

    The gate skips a system when too much of its liquidus has no solid free-energy reference
    behind it. ``BinaryLiquid.fit_parameters`` reads these unless the caller passes an explicit
    per-call override, so a campaign can tune the gate without touching every call site.
    Only the arguments you pass are changed; the rest keep their current values.

    Args:
        skip_frac: Skip when the unsupported fraction of the liquidus span exceeds this.
        min_missing: Minimum count of DFT-less compounds before ``missing_frac`` can fire.
            Guards against single-compound systems tripping the gate on a tiny composition range.
        missing_frac: Skip when this fraction of the interior MPDS compounds have no DFT
            counterpart (and at least ``min_missing`` of them are missing).
        ss_narrow_tol: Solid-solution fields no wider than this are treated as adequately
            represented by their endpoint line compounds and are not scored.
        dft_cover_tol: Composition tolerance for "a DFT phase covers this composition".
        ss_rescue_max_width: Maximum width of a solid-solution field that a nearby DFT compound
            is allowed to rescue. **Load-bearing**: uncapped, one interior DFT compound would
            rescue an arbitrarily wide field, and complete solid solutions with no
            solid-solution models (Ag-Au, Ta-W, Se-Te, ...) would score as fully supported --
            exactly the failure this gate exists to catch.
    """
    global coverage_skip_frac, coverage_min_missing, coverage_missing_frac
    global coverage_ss_narrow_tol, coverage_dft_cover_tol, coverage_ss_rescue_max_width

    fractions = {
        "skip_frac": skip_frac,
        "missing_frac": missing_frac,
        "ss_narrow_tol": ss_narrow_tol,
        "dft_cover_tol": dft_cover_tol,
        "ss_rescue_max_width": ss_rescue_max_width,
    }
    for name, value in fractions.items():
        if value is not None and not 0 < float(value) <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {value}")
    if min_missing is not None and int(min_missing) < 1:
        raise ValueError(f"min_missing must be >= 1, got {min_missing}")

    if skip_frac is not None:
        coverage_skip_frac = float(skip_frac)
    if min_missing is not None:
        coverage_min_missing = int(min_missing)
    if missing_frac is not None:
        coverage_missing_frac = float(missing_frac)
    if ss_narrow_tol is not None:
        coverage_ss_narrow_tol = float(ss_narrow_tol)
    if dft_cover_tol is not None:
        coverage_dft_cover_tol = float(dft_cover_tol)
    if ss_rescue_max_width is not None:
        coverage_ss_rescue_max_width = float(ss_rescue_max_width)


def set_liquidus_coverage_thresholds(
    *, max_gap: float | None = None, min_coverage: float | None = None, gap_tol: float | None = None
):
    """Package-wide defaults for the liquidus interior-coverage gate in
    ``BinaryLiquid.from_cache``.

    The endpoint-span gate (``comp_range_fit_lim``) cannot see interior holes:
    ``mpds.extract_digitized_liquidus`` linearly fills every composition gap wider than
    0.06 before anyone downstream can measure it, so a liquidus digitized only near the
    two pure ends (Bi-Si class: wedges at 0-7 and 92-100 at.%) is admitted with ~85% of
    its interior fabricated. ``from_cache`` therefore also measures the PRE-fill curve
    (``mpds.liquidus_coverage``) and flags ``init_error`` when it is interior-sparse.
    Only the arguments you pass are changed; the rest keep their current values.

    Args:
        max_gap: Reject when the widest composition interval between consecutive
            digitized liquidus points exceeds this fraction of the composition axis.
        min_coverage: Reject when less than this fraction of the stitched span is made of
            inter-point gaps no wider than ``gap_tol`` (i.e. when more than
            ``1 - min_coverage`` of the span is fabricated across large holes).
        gap_tol: Gap width up to which a stretch counts as locally sampled. Deliberately
            looser than the extractor's 0.06 fill threshold: ~10% of the admitted corpus
            is uniformly digitized at 0.06-0.10 spacing (simple lens diagrams with ~19
            points), which the fill interpolates faithfully — only wider holes fabricate.
    """
    global liquidus_max_gap, liquidus_min_coverage, liquidus_gap_tol
    for name, value in {
        "max_gap": max_gap,
        "min_coverage": min_coverage,
        "gap_tol": gap_tol,
    }.items():
        if value is not None and not 0 < float(value) <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {value}")
    if max_gap is not None:
        liquidus_max_gap = float(max_gap)
    if min_coverage is not None:
        liquidus_min_coverage = float(min_coverage)
    if gap_tol is not None:
        liquidus_gap_tol = float(gap_tol)


def set_dir_structure(structure: str):
    """How the DIRECTORY backend arranges files: ``'flat'`` or ``'nested'`` per system.

    A ``DirectoryBackend``-only knob, orthogonal to :func:`set_cache_mode`. Under
    ``cache_mode='sqlite'`` there are no directories to arrange, and this call LOGS and
    returns rather than raising: driver scripts call it unconditionally at import, and an
    exception would make a single-file store unusable for a setting that does not apply.
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

    There is deliberately no working-directory walk and no bare ``Path.cwd()`` fallback:
    either would silently resolve ``cache_dir`` to ``<cwd>/cache`` for an installed
    package, the unary registry would then load empty and every element reference would
    evaluate to zero — a configuration error rendered as wrong numbers.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == dirname:
            return parent
    return None


def _initial_cache_dir() -> Path | None:
    """Corpus at import: ``GLIQUID_CACHE_DIR`` -> ``GLIQUID_DATA_DIR`` (deprecated) ->
    a checkout's ``cache/`` -> none.

    ``set_cache_dir()`` sits above all of them and is applied by the caller afterwards.
    """
    for var in (CACHE_DIR_ENV_VAR, _LEGACY_DIR_ENV_VAR):
        env_value = os.environ.get(var)
        if not env_value:
            continue
        if var == _LEGACY_DIR_ENV_VAR:
            _warn_deprecated(
                _LEGACY_DIR_ENV_VAR,
                f"The {_LEGACY_DIR_ENV_VAR} environment variable is deprecated and will be "
                f"removed in a future release; use {CACHE_DIR_ENV_VAR} instead. "
                f"{_LEGACY_DIR_ENV_VAR} is still honored.",
            )
        candidate = Path(env_value)
        if not candidate.is_dir() and not _is_file_store(candidate):
            # Loud (WARNING reaches stderr through logging.lastResort even with no handler
            # configured) but not fatal: an unreadable corpus should fail where it is read.
            logger.warning(
                "%s is set to '%s', which is not a directory. gliquid will use "
                "it anyway; reads of the data corpus will fail there.",
                var,
                candidate,
            )
        return candidate
    root = find_project_root()
    if root is not None and (root / "cache").is_dir():
        return root / "cache"
    return None


class _ConfigModule(ModuleType):
    """Module type that keeps the deprecated ``data_*`` names ALIASES, not copies.

    A bare PEP 562 module-level ``__getattr__`` is not sufficient here and fails
    silently. ``__getattr__`` fires only for attributes that are MISSING from the module
    dict, so the first ``config.data_dir = X`` -- which real callers do, e.g.
    ``tests_internal/test_dft_data_loading.py`` and ``monkeypatch.setattr(config,
    "data_dir", ...)`` -- creates a genuine global that SHADOWS the alias. From then on
    reads of ``data_dir`` return the shadow and reads of ``cache_dir`` return the old
    value, the two names have silently diverged, and nothing raises.

    Overriding ``__setattr__`` as well is what makes them one variable in both directions.
    """

    def __getattr__(self, name):
        alias = _DEPRECATED_ATTRS.get(name)
        if alias is not None and alias[0] in self.__dict__:
            _warn_deprecated_attr(name, alias[1])
            return self.__dict__[alias[0]]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        alias = _DEPRECATED_ATTRS.get(name)
        if alias is not None:
            _warn_deprecated_attr(name, alias[1])
            name = alias[0]
        super().__setattr__(name, value)


set_project_root(find_project_root())
# cache_mode and dir_structure both before set_cache_dir(): it INFERS the mode from the
# shape of its argument, and set_dir_structure() is a no-op once the mode is 'sqlite'.
set_cache_mode(_CACHE_MODE_OPTS[0])
set_dir_structure(_DIR_STRUCT_OPTS[0])
set_cache_dir(_initial_cache_dir())
set_offline(_initial_offline())

sys.modules[__name__].__class__ = _ConfigModule
set_solid_solutions(False)
set_ss_ref_mode("from_unary_db")
# missing_frac 0.50 -> 0.60, calibrated on the 1457-system coverage sweep (2026-08-08):
# the comparison is `>=`, so 0.50 skipped every system with EXACTLY half its compounds missing.
# 0.60 admits 12 (ss_on skipped 254 -> 242) -- ten of them at exactly 0.500, plus Co-Pr
# (5/9 = 0.556) and Pu-Sn (4/7 = 0.571). Rb-Sn and Mn-Y (both 2/3 = 0.667) still fire on
# criterion 2; Na-Si (0.544 unsupported) still fires on criterion 1, which is why skip_frac
# must stay strictly below 0.544 and is left at 0.45.
set_coverage_thresholds(
    skip_frac=0.45,
    min_missing=2,
    missing_frac=0.60,
    ss_narrow_tol=0.10,
    dft_cover_tol=0.10,
    ss_rescue_max_width=0.25,
)
# Calibrated on the 1433 span-admitted cached systems (dev scan liquidus_coverage_scan):
# admitted max_gap p50/p95/p99 = 0.033/0.141/0.330; these defaults newly reject only the 9
# catastrophic systems (0.6%) whose interiors are ~half fabricated or worse (La-Y, Pu-Ti,
# Ce-Er, La-Lu, In-Tl, Cr-Os, Bi-Sn, Mg-Y, Mo-Re) — and the Bi-Si class the gate exists for.
set_liquidus_coverage_thresholds(max_gap=0.45, min_coverage=0.50, gap_tol=0.10)
