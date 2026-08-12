import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR_STRUCT_OPTS = ["flat", "nested"]
_SS_REF_MODE_OPTS = ["from_omegas_file", "from_dft_entries", "from_unary_db"]

#: Environment variable naming the external data corpus. See ``set_data_dir``.
DATA_DIR_ENV_VAR = "GLIQUID_DATA_DIR"

# gliquid's data comes in two kinds, and they live in two different places.
#
#   BUNDLED (here, inside the wheel) -- the three reference tables below. Small, and the
#   library cannot compute anything without them, so an installed gliquid is never in the
#   zero-energy state where every element reference evaluates to 0.
#
#   EXTERNAL (``data_dir``) -- the per-system DFT entry caches, the MPDS diagrams and the
#   model bundle. Megabytes of corpus, not shipped, reachable only through an explicit
#   ``set_data_dir()`` or ``GLIQUID_DATA_DIR``.
_BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"
_PHASE_TRANSITIONS_NAME = "phase_transitions.json"
_OMEGAS_NAME = "omegas_hcp.json"
_SPURIOUS_STRUCTURES_NAME = "spurious_structures.json"


class ConfigError(RuntimeError):
    """The external data corpus is needed and no data directory is configured.

    Raised in place of guessing a location. Fix with ``gliquid.config.set_data_dir(...)``
    or by setting ``GLIQUID_DATA_DIR``.
    """


project_root = None
data_dir = None
dir_structure = None
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
    """Resolve one reference table: ``data_dir/name`` if that file exists, else bundled.

    The existence check is what lets ``data_dir`` point at a PARTIAL corpus -- a directory
    holding only per-system caches still yields a working unary registry -- while a
    directory that does carry its own copy keeps overriding the shipped one, which is how
    the reference tables are iterated on during development.
    """
    if data_dir is not None:
        candidate = Path(data_dir) / name
        if candidate.exists():
            return candidate
    return _BUNDLED_DATA_DIR / name


def set_data_dir(path: Path | str | None):
    """Point gliquid at an external data corpus (per-system DFT caches, MPDS diagrams).

    Also re-resolves the three reference tables, each taken from ``path`` when a file of
    that name exists there and from the copy shipped inside the package otherwise. Passing
    ``None`` unsets the corpus: reference tables then come from the package, and any read
    of the corpus raises ``ConfigError`` rather than guessing.
    """
    global data_dir
    global phase_transitions_file
    global omegas_file
    global spurious_structures_file

    data_dir = Path(path) if path is not None else None
    phase_transitions_file = _reference_file(_PHASE_TRANSITIONS_NAME)
    omegas_file = _reference_file(_OMEGAS_NAME)
    spurious_structures_file = _reference_file(_SPURIOUS_STRUCTURES_NAME)


def require_data_dir(purpose: str = "This operation") -> Path:
    """The external corpus directory, or ``ConfigError`` naming how to configure one.

    Every read of the external corpus resolves through here rather than reading
    ``data_dir`` directly, so an unconfigured corpus is a loud error instead of a path
    built from a guess. There is deliberately no working-directory fallback: guessing is
    what turned a configuration error into wrong numbers.
    """
    if data_dir is None:
        raise ConfigError(
            f"{purpose} needs the gliquid data corpus (per-system DFT entry caches and "
            f"digitized MPDS diagrams), which is not shipped with the package. Point "
            f"gliquid at a copy with gliquid.config.set_data_dir('/path/to/data'), or set "
            f"the {DATA_DIR_ENV_VAR} environment variable before importing gliquid. "
            f"The reference tables ({_PHASE_TRANSITIONS_NAME}, {_OMEGAS_NAME}, "
            f"{_SPURIOUS_STRUCTURES_NAME}) ship with the package and are already loaded."
        )
    return Path(data_dir)


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
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    dir_structure = structure


def find_project_root(dirname="gliquid_python") -> Path | None:
    """The source checkout this file lives in, or ``None`` when there is not one.

    Package-anchored ONLY: with the src layout this file is at
    ``<project>/src/gliquid/config.py``, so a development checkout is always an ancestor of
    ``__file__``. Installed into site-packages nothing matches and the answer is ``None``.

    There is deliberately no working-directory walk and no bare ``Path.cwd()`` fallback:
    either would silently resolve ``data_dir`` to ``<cwd>/data`` for an installed
    package, the unary registry would then load empty and every element reference would
    evaluate to zero — a configuration error rendered as wrong numbers.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == dirname:
            return parent
    return None


def _initial_data_dir() -> Path | None:
    """Corpus location at import: ``GLIQUID_DATA_DIR`` -> a checkout's ``data/`` -> none.

    ``set_data_dir()`` sits above both and is applied by the caller afterwards.
    """
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_dir():
            # Loud (WARNING reaches stderr through logging.lastResort even with no handler
            # configured) but not fatal: an unreadable corpus should fail where it is read.
            logger.warning(
                "%s is set to '%s', which is not a directory. gliquid will use "
                "it anyway; reads of the data corpus will fail there.",
                DATA_DIR_ENV_VAR,
                candidate,
            )
        return candidate
    root = find_project_root()
    if root is not None and (root / "data").is_dir():
        return root / "data"
    return None


set_project_root(find_project_root())
set_data_dir(_initial_data_dir())
set_dir_structure(_DIR_STRUCT_OPTS[0])
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
