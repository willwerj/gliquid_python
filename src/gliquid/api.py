"""
Author: Joshua Willwerth
Description: Single home for everything in gliquid that talks to a remote API: credential
resolution (environment variables first, then an optional gitignored ``.env`` file — existing
environment variables are never overwritten), lazy construction of the Materials Project and
MPDS clients (so importing any gliquid module works offline, without API keys installed), and
the DFT-entry cache/load layer — fetching Materials Project entries for an n-component system,
caching them as ``{sys}_ENTRIES_MP_{type}.json``, and building the (Compound)PhaseDiagram
convex hull from the cache.

MPDS digitized-phase-diagram parsing is data-layout policy for a different source and lives in
``gliquid.mpds``; the raw MPDS bibliography/journal-code HTTP helpers are a driver-side
concern and are not part of this package.

IMPORT HYGIENE: importing this module must not import ``mp_api``, ``mpds_client``, or
``requests`` (pinned by tests/test_api.py) — client libraries and ``emmet.core`` (which leaks
``requests``) are imported inside functions only.
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

import gliquid.config as config
from gliquid.phase import validate_and_format_system

logger = logging.getLogger(__name__)

MP_KEY_VAR = "NEW_MP_API_KEY"
MPDS_KEY_VAR = "MPDS_API_KEY"


# --------------------------------------------------------------------------------------
# Credential resolution
# --------------------------------------------------------------------------------------


def load_dotenv(env_file: Path | str | None = None) -> None:
    """Populate ``os.environ`` from a ``KEY=VALUE`` file for keys not already set.

    Never overwrites an existing environment variable (``os.environ.setdefault`` only).
    Defaults to a gitignored ``.env`` at the repository root (``config.project_root``);
    silently a no-op when the file does not exist, or when there is no repository root at
    all (an installed package -- pass ``env_file`` or set the variable directly). Lines
    starting with ``#`` and lines without ``=`` are ignored; surrounding single/double
    quotes on values are stripped.
    """
    if env_file is None:
        if config.project_root is None:
            return
        env_file = Path(config.project_root) / ".env"
    env_file = Path(env_file)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_dotenv_loaded = False


def get_api_key(name: str) -> str | None:
    """Resolve an API key: ``os.environ`` first, then a one-shot ``load_dotenv()`` fallback."""
    value = os.getenv(name)
    if value:
        return value
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True
    return os.getenv(name)


# --------------------------------------------------------------------------------------
# Materials Project clients (mp_api imported lazily, inside functions only)
# --------------------------------------------------------------------------------------


def _construct_mprester(api_key: str):
    """Construct an MPRester, preferring raw dicts over monty/document decoding.

    The ``TypeError`` fallback covers an mp_api too old to accept these keyword
    arguments. It is unreachable on every mp_api version this package has been tested
    against, which is exactly why it must announce itself: the fallback client decodes
    responses into DOCUMENT MODELS instead of dicts, so a different data shape reaches
    code that expects dicts, and the resulting failure would surface far from here.
    """
    from mp_api.client import MPRester

    try:
        return MPRester(api_key, monty_decode=False, use_document_model=False)
    except TypeError:
        logger.warning(
            "MPRester rejected monty_decode=False/use_document_model=False (mp_api too "
            "old); falling back to a client that returns DOCUMENT MODELS rather than "
            "dicts. Entry handling in gliquid.api expects dicts, so downstream failures "
            "here are a symptom of this fallback -- upgrade mp_api (>=0.45.1)."
        )
        return MPRester(api_key)


_mpr = None


def get_mpr():
    """Lazily instantiate the shared MPRester client on first use.

    Avoids requiring NEW_MP_API_KEY (and a live MP connection) merely to import gliquid
    modules, so offline consumers (tests, reference-state lookups) work without a key.
    """
    global _mpr
    if _mpr is None:
        api_key = get_api_key(MP_KEY_VAR)
        if not api_key:
            raise ValueError(f"{MP_KEY_VAR} not found in environment variables!")
        _mpr = _construct_mprester(api_key)
    return _mpr


def mp_rester(api_key: str | None = None):
    """Return a fresh MPRester instance for ``with``-block use.

    The shared ``get_mpr()`` singleton must not be used as a context manager — exiting the
    block closes its session and would poison every later call.
    """
    api_key = api_key or get_api_key(MP_KEY_VAR)
    if not api_key:
        raise ValueError(f"{MP_KEY_VAR} not found in environment variables!")
    return _construct_mprester(api_key)


# --------------------------------------------------------------------------------------
# MPDS client (mpds_client is an optional dependency, imported lazily)
# --------------------------------------------------------------------------------------


def mpds_api_error() -> type[Exception]:
    """The mpds_client APIError class, or ``Exception`` when mpds-client is not installed."""
    try:
        from mpds_client import APIError

        return APIError
    except ImportError:
        return Exception


def get_mpds_client(dtype: str = "PEER_REVIEWED"):
    """Construct an authenticated MPDSDataRetrieval client (key via ``get_api_key``)."""
    try:
        from mpds_client import MPDSDataRetrieval, MPDSDataTypes
    except ImportError as exc:
        raise ImportError(
            "mpds-client is required for live MPDS retrieval. "
            "Install it with `pip install gliquid[mpds]` or `pip install mpds-client`."
        ) from exc
    api_key = get_api_key(MPDS_KEY_VAR)
    if not api_key:
        raise ValueError(f"{MPDS_KEY_VAR} not found in environment variables!")
    client = MPDSDataRetrieval(api_key=api_key)
    client.dtype = getattr(MPDSDataTypes, dtype)
    return client


# --------------------------------------------------------------------------------------
# DFT entry cache/load — n-component Materials Project entries and convex hulls.
# Cache files are ``{sys_name}_ENTRIES_MP_{dft_type}.json`` under the resolved system
# directory (``config.dir_structure`` flat/nested, or flat inside an explicit
# ``data_dir`` override such as TernaryLiquidInterpolation's).
# --------------------------------------------------------------------------------------

_LEGACY_MONTY_MODULES = {
    "pymatgen.core.entries": "pymatgen.entries.computed_entries",
    "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility",
}

# The dft_type names this package recognizes -- not the same thing as the ones that WORK.
SUPPORTED_DFT_TYPES = ("GGA", "R2SCAN", "MIXED")

# Two of the three recognized names cannot be fetched at all, each for an UPSTREAM reason
# gliquid cannot fix from here. They are rejected at the entry points below, before any
# network call, rather than left to fail deep inside a fetch with an error that points
# nowhere near the cause. Each string is the one-line diagnosis a user needs to check
# whether the upstream fix has landed; when one has, delete its entry and the
# tests_internal/test_dft_data_loading.py canary for it turns XPASS.
_BLOCKED_DFT_TYPES = {
    "R2SCAN": (
        "the Materials Project stores this thermo type as the literal 'r2SCAN' while "
        "emmet.core.thermo.ThermoType.R2SCAN is 'R2SCAN'; mp_api validates the argument "
        "against the enum (rejecting a literal 'r2SCAN') and then forwards that casing, so "
        "the query matches no document and the fetch returns zero entries"
    ),
    "MIXED": (
        "pymatgen's MaterialsProjectDFTMixingScheme hashes entry ids "
        "({e.entry_id for e in ...}, mixing_scheme.py), but new-API entries carry entry_id "
        "as a dict, raising TypeError: unhashable type: 'dict' -- and even past that its "
        "r2SCAN half would be empty for the R2SCAN reason above, making it silently GGA-only"
    ),
}


def _validate_dft_type(dft_type: str) -> None:
    """Reject an unknown ``dft_type``, and the accepted-but-broken ones, before any fetch.

    Called at the public entry points only. ``_get_dft_entries_from_components`` is
    deliberately left reachable for the blocked types so the maintainer canaries in
    ``tests_internal/test_dft_data_loading.py`` can keep probing the real upstream failure
    and report an upstream fix as XPASS.
    """
    if dft_type not in SUPPORTED_DFT_TYPES:
        raise ValueError(
            f"dft_type '{dft_type}' is not currently supported! "
            f"Please specify as one of the following: {', '.join(SUPPORTED_DFT_TYPES)}"
        )
    cause = _BLOCKED_DFT_TYPES.get(dft_type)
    if cause is not None:
        raise ValueError(
            f"dft_type='{dft_type}' is accepted as a name but cannot currently be fetched: "
            f"{cause}. Use dft_type='GGA' (the default) -- it is what this package's data "
            f"corpus and every published result are built on. If the upstream fix has "
            f"landed, drop the '{dft_type}' entry from gliquid.api._BLOCKED_DFT_TYPES and "
            f"re-run the canaries in tests_internal/test_dft_data_loading.py."
        )


def _alias_legacy_monty_modules() -> None:
    """Make the retired pymatgen paths importable, so UPSTREAM monty decodes resolve.

    ``_normalize_entry_dict`` below rewrites ``@module`` on dicts gliquid loads itself. It
    cannot help when the decode happens inside mp_api: ``MPRester.get_entries`` calls
    ``ComputedStructureEntry.from_dict`` on the server's response, and the Materials Project
    still tags nested ``energy_adjustments`` records with ``pymatgen.core.entries``. monty
    then ``__import__``s that dead path and raises ModuleNotFoundError, so every cache-MISS
    fetch fails while a warm cache succeeds -- which is why this hid from local runs.

    Registering the alias fixes the fetch for code gliquid never touches. Each path is only
    aliased if it is genuinely gone, so a future pymatgen that restores one wins.
    """
    import sys
    from importlib import import_module

    for legacy, current in _LEGACY_MONTY_MODULES.items():
        if legacy in sys.modules:
            continue
        try:
            import_module(legacy)
        except ImportError:
            pass
        else:
            continue  # still shipped upstream -- leave it alone
        try:
            sys.modules[legacy] = import_module(current)
        except ImportError:  # pragma: no cover - current path missing is a real breakage
            logger.debug("Could not alias %s -> %s; leaving unmapped.", legacy, current)


_alias_legacy_monty_modules()


def _normalize_entry_dict(entry):
    """Rewrite legacy monty ``@module`` paths so old caches deserialize after upgrades."""
    if isinstance(entry, dict):
        return {
            key: _LEGACY_MONTY_MODULES.get(value, value)
            if key == "@module"
            else _normalize_entry_dict(value)
            for key, value in entry.items()
        }
    if isinstance(entry, list):
        return [_normalize_entry_dict(value) for value in entry]
    return entry


def _computed_entry_from_dict(entry: dict):
    entry = _normalize_entry_dict(entry)
    if entry.get("@class") == "ComputedStructureEntry" or "structure" in entry:
        return ComputedStructureEntry.from_dict(entry)
    return ComputedEntry.from_dict(entry)


# Tier A spurious-structure blacklist (``config.spurious_structures_file``): elemental MP
# structures with no stability field at any T at 1 atm. Filtered at fetch AND at cache
# read (mirroring the Mg149 guard) so every hull references formation energies to the
# same surviving elemental ground states as the unary DB — old caches included, without
# rewriting them. Matching is two-path because entry ids changed generations: classic
# string ids ('mp-8566-GGA') match the blacklist's material_ids; new-API alpha ids
# ({'identifier': 'mp-aaaaaaeu', ...}) are unmappable, so elemental entries fall back to
# the (element, spacegroup) of their structure (symprec=0.1, the MP convention).
_MP_ID_RE = re.compile(r"(mp-[a-z0-9]+)")
_spurious_cache: tuple | None = None  # (path, mtime, ids, pairs, expected_gs)


def _spurious_structure_index() -> tuple[frozenset, frozenset, dict]:
    """(blacklisted material_ids, blacklisted (element, spacegroup) pairs,
    {element: expected ground-state spacegroup})."""
    global _spurious_cache
    path = config.spurious_structures_file
    if path is None or not os.path.exists(path):
        return frozenset(), frozenset(), {}
    mtime = os.path.getmtime(path)
    if _spurious_cache and _spurious_cache[:2] == (str(path), mtime):
        return _spurious_cache[2], _spurious_cache[3], _spurious_cache[4]
    with open(path) as f:
        raw = json.load(f)
    ids, pairs = set(), set()
    for element, entries in raw.get("elements", {}).items():
        for rec in entries:
            if rec.get("material_id"):
                ids.add(rec["material_id"])
            if rec.get("spacegroup_number"):
                pairs.add((element, int(rec["spacegroup_number"])))
    expected = {el: int(sg) for el, sg in raw.get("expected_gs_spacegroup", {}).items()}
    _spurious_cache = (str(path), mtime, frozenset(ids), frozenset(pairs), expected)
    return _spurious_cache[2], _spurious_cache[3], _spurious_cache[4]


def _entry_spacegroup(entry_dict: dict) -> int | None:
    """Spacegroup of an entry dict's structure (MP symprec convention); None if absent."""
    if "structure" not in entry_dict:
        return None
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return SpacegroupAnalyzer(
            Structure.from_dict(_normalize_entry_dict(entry_dict["structure"])), symprec=0.1
        ).get_space_group_number()
    except Exception:
        return None


def _is_spurious_entry_dict(entry_dict: dict) -> bool:
    """True if a cached/fetched entry dict is a blacklisted elemental structure."""
    ids, pairs, _ = _spurious_structure_index()
    if not ids and not pairs:
        return False
    comp = entry_dict.get("composition", {})
    if len(comp) != 1:
        return False  # the blacklist is an elemental ground-state policy
    entry_id = entry_dict.get("entry_id")
    id_str = entry_id.get("identifier") if isinstance(entry_id, dict) else entry_id
    if isinstance(id_str, str):
        m = _MP_ID_RE.match(id_str)
        if m and m.group(1) in ids:
            return True
    element = next(iter(comp))
    if pairs and any(el == element for el, _ in pairs):
        sg = _entry_spacegroup(entry_dict)
        return sg is not None and (element, sg) in pairs
    return False


def _filter_spurious_entries(computed_entry_dicts: list[dict]) -> list[dict]:
    """Tier A for the entry layer: blacklist filter + anchor-consistency guard.

    The blacklist removes known spurious elemental structures. The anchor guard
    then drops any ELEMENTAL entry of an audited element that sits BELOW the
    lowest entry of the element's expected ground-state spacegroup while not
    being of that spacegroup itself. Entry caches include THEORETICAL
    structures the unary builder's theoretical=False filter never sees (9R-Ag
    below FCC, theoretical FCC-Eu below BCC, ...), an unbounded artifact family
    per-id enumeration cannot close; this pins every hull's elemental reference
    to the same experimentally-anchored structure as the unary DB.
    """
    ids, pairs, expected = _spurious_structure_index()
    if not ids and not pairs and not expected:
        return computed_entry_dicts
    survivors = [e for e in computed_entry_dicts if not _is_spurious_entry_dict(e)]
    if not expected:
        return survivors

    # Reference energy per audited element: lowest energy-per-atom among its
    # entries OF the expected spacegroup (None if the cache has no such entry).
    ref_epa: dict[str, float] = {}
    entry_sgs: dict[int, int | None] = {}
    for i, e in enumerate(survivors):
        comp = e.get("composition", {})
        if len(comp) != 1:
            continue
        element = next(iter(comp))
        if element not in expected or e.get("energy") is None:
            continue
        sg = _entry_spacegroup(e)
        entry_sgs[i] = sg
        if sg == expected[element]:
            epa = e["energy"] / sum(comp.values())
            if element not in ref_epa or epa < ref_epa[element]:
                ref_epa[element] = epa

    kept = []
    for i, e in enumerate(survivors):
        comp = e.get("composition", {})
        if len(comp) == 1:
            element = next(iter(comp))
            if (
                element in ref_epa
                and i in entry_sgs
                and entry_sgs[i] != expected[element]
                and e.get("energy") is not None
                and e["energy"] / sum(comp.values()) < ref_epa[element] - 1e-8
            ):
                logger.info(
                    f"Anchor guard: dropping sub-ground-state {element} entry "
                    f"(SG {entry_sgs[i]}) below the expected SG "
                    f"{expected[element]} reference."
                )
                continue
        kept.append(e)
    return kept


def _get_dft_entries_from_components(
    components: list[str], dft_type: str, keep_data=False
) -> list[dict]:
    """Fetches DFT entries for the specified components and DFT functional type."""
    # emmet.core transitively imports requests; keep it out of module import time
    # (tests/test_api.py pins the import hygiene).
    from emmet.core.thermo import ThermoType

    entries = []

    # The chemsys query is expressed in ELEMENT symbols, not in components.
    # ``MPRester.get_entries_in_chemsys`` enumerates every chemsys substring of
    # ``set(elements)``, so a compound component is queried as a chemsys that does not
    # exist ('CuMg') and silently contributes nothing -- a PARTIAL fetch that then gets
    # written to the cache, and whose CompoundPhaseDiagram fails downstream with
    # "Missing terminal entries" rather than at the fetch. A hull over compound terminals
    # needs the whole spanned elemental chemsys anyway (CompoundPhaseDiagram drops what
    # its terminal basis cannot express). For elemental components this is the identity:
    # get_entries_in_chemsys de-duplicates and re-derives the order itself.
    elements = sorted({str(el) for c in components for el in Composition(c).elements})

    def fetch_entries(api_key, thermo_type=None):
        """Helper function to fetch entries from API."""
        with mp_rester(api_key) as MPR:
            criteria = {"thermo_types": [thermo_type]} if thermo_type else {}
            return MPR.get_entries_in_chemsys(elements, additional_criteria=criteria)

    new_mp_api_key = get_api_key(MP_KEY_VAR)
    scan_entries, ggau_entries = [], []

    if dft_type in ["R2SCAN", "MIXED"]:
        scan_entries = fetch_entries(new_mp_api_key, ThermoType.R2SCAN)
    if dft_type in ["GGA", "MIXED"]:
        ggau_entries = fetch_entries(new_mp_api_key, ThermoType.GGA_GGA_U)

    if dft_type == "MIXED":
        # Imported HERE, not at function entry: only this branch uses it, so a pymatgen
        # release that relocates the class must not break GGA and R2SCAN cold fetches for
        # a feature they never touch (the same shape as the emmet.core.thermo coupling).
        from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

        entries = MaterialsProjectDFTMixingScheme().process_entries(
            scan_entries + ggau_entries, verbose=False
        )
    elif dft_type == "GGA":
        entries = ggau_entries
    elif dft_type == "R2SCAN":
        entries = scan_entries

    # With monty_decode=False the client may return raw dicts instead of ComputedEntry
    # objects; accept either so the cache format is unchanged.
    computed_entry_dicts = [e.as_dict() if hasattr(e, "as_dict") else e for e in entries]

    # Filter out Mg149 phase and remove run data to reduce cache size
    computed_entry_dicts = [e for e in computed_entry_dicts if e["composition"].get("Mg", 0) != 149]
    # Tier A: blacklist + anchor-consistency guard at fetch time.
    computed_entry_dicts = _filter_spurious_entries(computed_entry_dicts)
    if not keep_data:
        for e in computed_entry_dicts:
            e.pop("data", None)

    # A fetch that yields nothing must RAISE here, above every cache-write site, rather
    # than return [] for a caller to persist. A 2-byte '[]' cache file is indistinguishable
    # from a real one on the next read, so one bad fetch is served warm forever -- the same
    # defect shape as the partial-chemsys fetch, and this half is gliquid's own. Returning
    # empty without raising is not an option either: the caller then fails somewhere far
    # less legible (PhaseDiagram's "Unable to build phase diagram without entries").
    if not computed_entry_dicts:
        raise ValueError(
            f"The Materials Project returned no usable entries for "
            f"'{'-'.join(components)}' (chemsys {'-'.join(elements)}) with "
            f"dft_type='{dft_type}'; refusing to cache an empty result. Check that every "
            f"component is a real element or compound and that MP holds entries for this "
            f"chemsys under this functional. An empty result for a real chemsys means the "
            f"query matched nothing upstream -- for dft_type='R2SCAN' that is the known "
            f"thermo-type casing bug, not a property of the system."
        )

    return computed_entry_dicts


def _atomic_write_json(path: str, payload) -> None:
    """Serialize ``payload`` to ``path`` atomically, via a temp file + ``os.replace``.

    A plain ``json.dump`` to the cache path writes incrementally, so an interrupted or
    failing write leaves a TRUNCATED file that later reads accept as a valid cache. Worse,
    concurrent cold fetches of the same system interleave into one file --
    ``dev/scripts/Fit_Binary_Systems.py`` fans campaigns out over a ``ProcessPoolExecutor``,
    and a campaign over uncached systems is exactly a burst of concurrent cold fetches.

    The temp file is created in the DESTINATION directory so it shares a filesystem with
    the target; ``os.replace`` is only atomic within one filesystem, and is atomic on both
    POSIX and Windows. A reader therefore sees either the old file or the complete new one,
    never a partial one. Two writers racing still both fetch -- that is accepted -- but
    whichever lands last leaves a whole, valid file.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
    except BaseException:
        # Never leave the scratch file behind on a failed write.
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover - already gone, or undeletable
            pass
        raise


def _canonical_sys_name(components) -> str:
    """Alphabetical hyphenated system name — the on-disk cache-key convention.

    Construction order stays authoritative in memory; canonicalizing only the disk key
    lets 'Zr-Hf' and 'Hf-Zr' share one cache.
    """
    return "-".join(sorted(components))


def _resolve_sys_dir(sys_name: str, data_dir=None) -> str:
    """Return the cache directory for ``sys_name``.

    An explicit ``data_dir`` wins and implies a FLAT layout inside it (the historical
    per-instance ternary convention); otherwise ``config.dir_structure`` decides.
    """
    if data_dir is not None:
        return str(data_dir)
    root = config.require_data_dir(f"Reading the cache for '{sys_name}'")
    if config.dir_structure == "nested":
        sys_dir = os.path.join(root, sys_name)
        os.makedirs(sys_dir, exist_ok=True)
        return sys_dir
    if config.dir_structure == "flat":
        return root
    raise ValueError(f"Invalid dir_structure '{config.dir_structure}'. Must be 'nested' or 'flat'.")


def _is_imputed_entry_dict(entry_dict: dict) -> bool:
    """True if a cached ComputedEntry dict was synthesized by phase-energy imputation.

    Imputed entries are tagged on write with an ``entry_id`` of the form ``"imputed:..."``
    and ``data={'imputed': True}`` (see ``phase_imputation``); either marker is sufficient.
    """
    if str(entry_dict.get("entry_id", "")).startswith("imputed:"):
        return True
    data = entry_dict.get("data") or {}
    return bool(isinstance(data, dict) and data.get("imputed"))


def get_dft_structure_entries(
    input, dft_type="GGA", verbose=False, data_dir=None
) -> list[ComputedStructureEntry]:
    """Cached DFT entries with structures retained, for structure-based analysis.

    Same cache file as ``get_dft_convexhull`` (``<sys>_ENTRIES_MP_<type>.json``), but
    guarantees each returned entry carries its structure (needed e.g. for spacegroup
    identification in ``gliquid.solution``). A cache written without structures is
    refreshed once via the API; imputed entries (no structures by construction) are
    preserved in the cache across a refresh and excluded from the returned list.

    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        dft_type (str): Functional type. Only 'GGA' works; 'R2SCAN' and 'MIXED' are
            recognized names blocked upstream and raise (see ``_BLOCKED_DFT_TYPES``).
        verbose (bool): Whether to print cache activity.
        data_dir: Explicit cache directory override (flat layout inside it).

    Returns:
        list[ComputedStructureEntry]: Real (non-imputed) entries with structures.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    _validate_dft_type(dft_type)

    sys_dir = _resolve_sys_dir(cache_name, data_dir=data_dir)
    dft_entries_file = os.path.join(sys_dir, f"{cache_name}_ENTRIES_MP_{dft_type}.json")

    computed_entry_dicts = None
    if os.path.exists(dft_entries_file):
        with open(dft_entries_file) as f:
            computed_entry_dicts = json.load(f)
        if verbose:
            logger.info("Loading cached DFT entry data.")

    real_entries = [e for e in (computed_entry_dicts or []) if not _is_imputed_entry_dict(e)]
    if computed_entry_dicts is None or any("structure" not in e for e in real_entries):
        imputed = [e for e in (computed_entry_dicts or []) if _is_imputed_entry_dict(e)]
        real_entries = _get_dft_entries_from_components(components, dft_type, keep_data=True)
        computed_entry_dicts = real_entries + imputed
        if verbose:
            logger.info(f"Caching DFT entry data as {dft_entries_file}...")
        _atomic_write_json(dft_entries_file, computed_entry_dicts)

    # Read-time Tier A guard (see get_dft_convexhull): filter in memory, never rewrite.
    return [_computed_entry_from_dict(e) for e in _filter_spurious_entries(real_entries)]


def cache_imputed_entries(
    input, imputed_entry_dicts: list[dict], dft_type="GGA", data_dir=None
) -> str:
    """Append imputed ComputedEntry dicts to the shared DFT entries cache.

    Imputed entries live in the same ``<sys>_ENTRIES_MP_<type>.json`` as the real DFT
    entries but are tagged, so ``get_dft_convexhull(..., include_imputed=False)`` (the
    default) ignores them and the canonical DFT-only hull is unchanged. Re-running is
    idempotent: an existing imputed entry with the same ``entry_id`` is replaced.

    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        imputed_entry_dicts (list[dict]): Tagged ``ComputedEntry.as_dict()`` payloads.
        dft_type (str): Functional type, selecting the cache file.
        data_dir: Explicit cache directory override (flat layout inside it).

    Returns:
        str: Path to the cache file written.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    sys_dir = _resolve_sys_dir(cache_name, data_dir=data_dir)
    dft_entries_file = os.path.join(sys_dir, f"{cache_name}_ENTRIES_MP_{dft_type}.json")

    if os.path.exists(dft_entries_file):
        with open(dft_entries_file) as f:
            existing = json.load(f)
    else:
        existing = _get_dft_entries_from_components(components, dft_type)

    new_ids = {e.get("entry_id") for e in imputed_entry_dicts}
    existing = [e for e in existing if e.get("entry_id") not in new_ids]
    existing.extend(imputed_entry_dicts)
    _atomic_write_json(dft_entries_file, existing)
    return dft_entries_file


def get_dft_convexhull(
    input,
    dft_type="GGA",
    inc_structure_data=False,
    verbose=False,
    include_imputed=False,
    data_dir=None,
) -> tuple[PhaseDiagram, dict]:
    """
    Returns the DFT convex hull of a given n-component system with specified functionals.

    Args:
        input (str or list): System specification (e.g., 'A-B', ['A', 'B'] or
            ['A', 'B', 'C'] — any number of components >= 2).
        dft_type (str): Functional type. Only 'GGA' works; 'R2SCAN' and 'MIXED' are
            recognized names blocked upstream and raise (see ``_BLOCKED_DFT_TYPES``).
        inc_structure_data (bool): Whether to include structural data.
        verbose (bool): Whether to print detailed output.
        include_imputed (bool): If False (default), imputed entries cached by
            ``cache_imputed_entries`` are filtered out so the hull is DFT-only. Set True to
            include them (phase-energy imputation workflow).
        data_dir: Explicit cache directory override (flat layout inside it) — e.g. the
            per-instance ternary ``data_dir``.

    Returns:
        A tuple of the phase diagram and a dictionary of stable entry atomic volumes.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)

    _validate_dft_type(dft_type)
    if verbose:
        logger.info(f"Using DFT entries solved with {dft_type} functionals.")

    sys_dir = _resolve_sys_dir(cache_name, data_dir=data_dir)

    dft_entries_file = os.path.join(sys_dir, f"{cache_name}_ENTRIES_MP_{dft_type}.json")

    # Yb-containing structures are only available with R2SCAN functional
    # See https://docs.materialsproject.org/changes/database-versions#v2023.11.1
    # and https://docs.materialsproject.org/changes/database-versions#v2025.02.12
    if "Yb" in components and not os.path.exists(dft_entries_file):
        logger.warning(
            "Yb-containing structures are only available with R2SCAN or MIXED functionals "
            "on the MP database, and BOTH are blocked upstream (see _BLOCKED_DFT_TYPES), so "
            "there is no working functional for a cold Yb fetch: expect an empty result."
        )

    if os.path.exists(dft_entries_file):
        with open(dft_entries_file) as f:
            computed_entry_dicts = [_normalize_entry_dict(e) for e in json.load(f)]
        if verbose:
            logger.info("Loading cached DFT entry data.")
    else:
        computed_entry_dicts = _get_dft_entries_from_components(components, dft_type)
        if verbose:
            logger.info(f"Caching DFT entry data as {dft_entries_file}...")
        _atomic_write_json(dft_entries_file, computed_entry_dicts)

    if not include_imputed:
        computed_entry_dicts = [e for e in computed_entry_dicts if not _is_imputed_entry_dict(e)]

    # Read-time Mg149 guard (fetch already filters; old ternary caches may predate it).
    computed_entry_dicts = [
        e for e in computed_entry_dicts if e.get("composition", {}).get("Mg", 0) != 149
    ]

    # Read-time Tier A guard (blacklist + anchor consistency): caches of any age
    # may hold blacklisted or sub-anchor elemental structures; filter in memory,
    # never rewrite the cache file.
    computed_entry_dicts = _filter_spurious_entries(computed_entry_dicts)

    if any(len(Composition(c).elements) > 1 for c in components):
        dft_ch = CompoundPhaseDiagram(
            terminal_compositions=[Composition(c) for c in components],
            entries=[_computed_entry_from_dict(e) for e in computed_entry_dicts],
        )
    else:
        dft_ch = PhaseDiagram(
            elements=[Element(c) for c in components],
            entries=[_computed_entry_from_dict(e) for e in computed_entry_dicts],
        )
    if verbose:
        logger.info(
            f"{len(dft_ch.stable_entries) - 2} stable line compound(s) on the DFT convex hull."
        )

    stable_entry_atomic_volumes = {}

    if inc_structure_data:
        for entry in dft_ch.stable_entries:
            entries_matching_composition = [
                e
                for e in computed_entry_dicts
                if Composition.from_dict(e["composition"]) == entry.composition
            ]
            hull_stable_entry = min(entries_matching_composition, key=lambda x: x["energy"])
            hull_stable_structure = Structure.from_dict(hull_stable_entry["structure"])
            ucell_volume = hull_stable_structure.volume  # Volume in cubic angstroms
            ucell_n_atoms = hull_stable_structure.num_sites  # Number of atoms per structure
            atomic_volume = (
                ucell_volume / ucell_n_atoms
            )  # Atomic volume in cubic angstroms per atom
            stable_entry_atomic_volumes[entry.composition.reduced_formula] = atomic_volume

    return dft_ch, stable_entry_atomic_volumes


# --------------------------------------------------------------------------------------
# Hull seams — the ONLY sanctioned ways to read component identities, axis fractions,
# and display names off a hull, valid for both PhaseDiagram and CompoundPhaseDiagram.
# Consumers written against these helpers need no changes when pseudo-binary
# (compound end-member) systems land.
# --------------------------------------------------------------------------------------


def pd_components(pd) -> list[str]:
    """The hull's component identities in ITS axis order.

    Elemental ``PhaseDiagram``: element symbols in the hull's element order (the caller
    order under this package's construction convention). ``CompoundPhaseDiagram``: the
    terminal compositions' reduced formulas in terminal order — pymatgen-normalized
    strings (e.g. 'CuMg' -> 'MgCu'), so compare via ``Composition`` equality rather than
    string equality.
    """
    if isinstance(pd, CompoundPhaseDiagram):
        return [c.reduced_composition.reduced_formula for c in pd.terminal_compositions]
    return [str(el) for el in pd.elements]


def entry_original(entry):
    """The untransformed entry behind a (possibly Transformed) hull entry."""
    return getattr(entry, "original_entry", None) or entry


def entry_display_name(entry) -> str:
    """The entry's phase display name: the ORIGINAL composition's reduced formula.

    Safe for ``TransformedPDEntry`` (whose own composition is in dummy-species
    coordinates); identical to ``entry.composition.reduced_formula`` for plain entries.
    """
    return entry_original(entry).composition.reduced_formula


def entry_frac_along(pd, entry, components=None) -> tuple[float, ...]:
    """The entry's fractions along the hull's evaluation axes (``components[1:]``).

    Elemental ``PhaseDiagram``: atomic fractions of ``components[1:]`` (defaulting to
    the hull's own element order) — bit-identical to
    ``entry.composition.get_atomic_fraction``. ``CompoundPhaseDiagram``: normalized
    dummy-species amounts — the end-member pseudo-fractions on pymatgen's
    atom-normalized terminal basis (``normalize_terminal_compositions=True``), e.g.
    CuMg2 in a CuMg/Mg frame sits at x_Mg = 1/3, not 1/2.

    Note: ``CompoundPhaseDiagram`` requires an entry AT each terminal composition or it
    raises ``Missing terminal entries`` at construction — a future pseudo-binary loader
    must guarantee the end-member entries exist.
    """
    if isinstance(pd, CompoundPhaseDiagram):
        amts = [entry.composition[sp] if sp in entry.composition else 0.0 for sp in pd.elements]
        total = sum(amts)
        if not total:
            return tuple(0.0 for _ in pd.elements[1:])
        return tuple(a / total for a in amts[1:])
    comps = list(components) if components is not None else [str(el) for el in pd.elements]
    comp = entry.composition
    return tuple(comp.get_atomic_fraction(c) for c in comps[1:])
