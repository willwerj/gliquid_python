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
import warnings
from pathlib import Path

from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

import gliquid.config as config
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    CacheKey,
    CacheModeError,
    atomic_write_json,
    resolve_backend,
)
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
        config.require_online("Constructing a Materials Project client")
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
    config.require_online("Constructing a Materials Project client")
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
    config.require_online("Constructing an MPDS client")
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
# Records are addressed as ``CacheKey(sys_name, 'dft_entries', dft_type)`` and resolved by a
# ``gliquid.cache`` backend: the configured store, or the one named by an explicit
# ``data_dir`` override such as TernaryLiquidInterpolation's. Under the DirectoryBackend
# that is still ``{sys_name}_ENTRIES_MP_{dft_type}.json`` under the system directory
# (``config.dir_structure`` flat/nested, or flat inside an explicit override).
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


# Tier A spurious-structure blacklist (``config.spurious_structures_file``). Two sibling
# blocks, two policies:
#   'elements' — elemental MP structures with no stability field at any T at 1 atm.
#   'compounds' — non-elemental artifacts (MP composition artifacts and the like), which
#     the elemental block structurally cannot express because that path refuses any entry
#     of arity != 1.
# Both are filtered at fetch AND at cache read, so every hull references formation
# energies to the same surviving elemental ground states as the unary DB — old caches
# included, without rewriting them. Elemental matching is two-path because entry ids
# changed generations: classic string ids ('mp-8566-GGA') match the blacklist's
# material_ids; new-API alpha ids ({'identifier': 'mp-aaaaaaeu', ...}) are unmappable, so
# elemental entries fall back to the (element, spacegroup) of their structure
# (symprec=0.1, the MP convention).
_MP_ID_RE = re.compile(r"(mp-[a-z0-9]+)")
_spurious_cache: tuple | None = None  # (path, mtime, ids, pairs, expected_gs, compounds)

# Compound-rule predicate forms, in EVALUATION PRECEDENCE. A record's ``match`` object may
# name several; the highest-precedence form present decides that record. 'composition' is
# the preferred shape for new records — it pins every element and so cannot over-match.
# 'element_count' constrains only the elements it names and leaves the rest free, which is
# what makes it the exact translation of a legacy ``comp.get(el, 0) != n`` literal.
_COMPOUND_MATCH_FORMS = ("material_id", "composition", "element_count")


def _normalize_compound_rules(records) -> tuple:
    """``compounds`` block -> ordered ``(form, payload)`` pairs.

    One rule per record: the first form of ``_COMPOUND_MATCH_FORMS`` present in its
    ``match`` object wins. A record naming no recognized form — or naming one with an
    EMPTY payload, which would make ``all(...)`` vacuously true and blacklist the whole
    corpus — is dropped with a warning rather than silently over-matching.
    """
    rules: list[tuple[str, object]] = []
    for rec in records or []:
        match = rec.get("match", {}) if isinstance(rec, dict) else {}
        for form in _COMPOUND_MATCH_FORMS:
            payload = match.get(form) if isinstance(match, dict) else None
            if not payload:
                continue
            if form == "material_id":
                rules.append((form, str(payload)))
            elif form == "composition":
                rules.append((form, tuple(sorted((str(el), int(n)) for el, n in payload.items()))))
            else:  # element_count
                rules.append((form, tuple(sorted((str(el), n) for el, n in payload.items()))))
            break
        else:
            logger.warning(
                "Spurious-structure 'compounds' record names no usable match form %s "
                "(missing or empty); ignoring it: %r",
                _COMPOUND_MATCH_FORMS,
                rec,
            )
    return tuple(rules)


def _spurious_structure_index() -> tuple[frozenset, frozenset, dict, tuple]:
    """(blacklisted material_ids, blacklisted (element, spacegroup) pairs,
    {element: expected ground-state spacegroup}, compound (form, payload) rules)."""
    global _spurious_cache
    path = config.spurious_structures_file
    if path is None or not os.path.exists(path):
        return frozenset(), frozenset(), {}, ()
    mtime = os.path.getmtime(path)
    if _spurious_cache and _spurious_cache[:2] == (str(path), mtime):
        return _spurious_cache[2], _spurious_cache[3], _spurious_cache[4], _spurious_cache[5]
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
    compounds = _normalize_compound_rules(raw.get("compounds", []))
    _spurious_cache = (str(path), mtime, frozenset(ids), frozenset(pairs), expected, compounds)
    return _spurious_cache[2], _spurious_cache[3], _spurious_cache[4], _spurious_cache[5]


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


def _entry_mp_id(entry_dict: dict) -> str | None:
    """Classic 'mp-xxxx' prefix of an entry dict's id (dict- or string-form); None if absent."""
    entry_id = entry_dict.get("entry_id")
    id_str = entry_id.get("identifier") if isinstance(entry_id, dict) else entry_id
    if isinstance(id_str, str):
        m = _MP_ID_RE.match(id_str)
        if m:
            return m.group(1)
    return None


def _matches_compound_rule(entry_dict: dict, comp: dict, rules: tuple) -> bool:
    """True if any ``compounds`` rule matches. Applies at ANY arity, elemental included."""
    for form, payload in rules:
        if form == "material_id":
            if _entry_mp_id(entry_dict) == payload:
                return True
        elif form == "composition":
            try:
                got = tuple(sorted((str(el), int(n)) for el, n in comp.items()))
            except (TypeError, ValueError):
                continue
            if got == payload:
                return True
        else:  # element_count — plain numeric equality, NOT an int cast: this is the
            # bit-exact translation of the legacy ``comp.get(el, 0) != n`` literals.
            # Cached counts are floats, and float == int already compares equal; an
            # int cast would additionally swallow a genuinely fractional near-miss.
            if all(comp.get(el, 0) == n for el, n in payload):
                return True
    return False


def _is_spurious_entry_dict(entry_dict: dict) -> bool:
    """True if a cached/fetched entry dict is blacklisted by either block.

    Dispatches on arity policy, not on file layout: ``compounds`` rules are tested
    against every entry (a composition artifact is not an elemental-ground-state
    question), while the ``elements`` block stays gated to arity-1 entries.
    """
    ids, pairs, _, compounds = _spurious_structure_index()
    if not ids and not pairs and not compounds:
        return False
    comp = entry_dict.get("composition", {}) or {}
    if compounds and _matches_compound_rule(entry_dict, comp, compounds):
        return True
    if not ids and not pairs:
        return False
    if len(comp) != 1:
        return False  # the 'elements' block is an elemental ground-state policy
    if _entry_mp_id(entry_dict) in ids:
        return True
    element = next(iter(comp))
    if pairs and any(el == element for el, _ in pairs):
        sg = _entry_spacegroup(entry_dict)
        return sg is not None and (element, sg) in pairs
    return False


def _filter_spurious_entries(computed_entry_dicts: list[dict]) -> list[dict]:
    """Tier A for the entry layer: blacklist filter + anchor-consistency guard.

    The blacklist removes known spurious elemental structures and, via the
    ``compounds`` block, non-elemental artifacts of any arity. The anchor guard
    then drops any ELEMENTAL entry of an audited element that sits BELOW the
    lowest entry of the element's expected ground-state spacegroup while not
    being of that spacegroup itself. Entry caches include THEORETICAL
    structures the unary builder's theoretical=False filter never sees (9R-Ag
    below FCC, theoretical FCC-Eu below BCC, ...), an unbounded artifact family
    per-id enumeration cannot close; this pins every hull's elemental reference
    to the same experimentally-anchored structure as the unary DB.

    The anchor guard stays deliberately ELEMENTAL-ONLY: its ``len(comp) == 1``
    tests below are load-bearing, not an oversight the compound block relaxes.
    """
    ids, pairs, expected, compounds = _spurious_structure_index()
    if not ids and not pairs and not expected and not compounds:
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
    # THE remote DFT path. Refused first, above the credential lookup and above the import
    # of anything that can open a socket, so offline mode is a property of the
    # configuration and not of whether a key happens to be installed (see
    # config.OfflineError). Every cache miss for a system the store does not cover arrives
    # here, so this one guard covers get_dft_convexhull, get_dft_structure_entries and
    # cache_imputed_entries alike.
    config.require_online(
        f"Fetching Materials Project DFT entries for '{'-'.join(components)}' "
        f"(dft_type={dft_type!r})"
    )
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

    # Tier A: blacklist + anchor-consistency guard at fetch time. The blacklist's
    # 'compounds' block carries the MP composition artifact that used to be an
    # element-specific literal here; run data is dropped just below.
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


# The atomic-write implementation moved to ``gliquid.cache`` (it is the DirectoryBackend's
# job now); the old name stays reachable here because callers and tests use it.
_atomic_write_json = atomic_write_json


def _canonical_sys_name(components) -> str:
    """Alphabetical hyphenated system name — the on-disk cache-key convention.

    Construction order stays authoritative in memory; canonicalizing only the disk key
    lets 'Zr-Hf' and 'Hf-Zr' share one cache.
    """
    return "-".join(sorted(components))


def resolve_cache_path(key: CacheKey, cache=None) -> Path | None:
    """The filesystem path of one cached record, or ``None`` if the store has no paths.

    The supported replacement for ``_resolve_sys_dir``: it names a RECORD rather than a
    directory, which is the only question every backend can answer. ``None`` is a real
    answer, not an error -- a single-file store holds the record but not at a path -- so
    callers that need a path must handle it rather than assume one exists.

    Args:
        key: The record to locate.
        cache: ``None`` for the configured store, or a path / ``CacheBackend`` override.
    """
    backend = resolve_backend(cache)
    path_for = getattr(backend, "path_for", None)
    if path_for is None:
        return None
    return Path(path_for(key))


def _resolve_sys_dir(sys_name: str, data_dir=None) -> str:
    """Deprecated. The cache directory for ``sys_name``; use :func:`resolve_cache_path`.

    Kept with IDENTICAL semantics because callers outside this package depend on them,
    including the side effect below. An explicit ``data_dir`` wins and implies a FLAT
    layout inside it (the historical per-instance ternary convention); otherwise
    ``config.dir_structure`` decides.

    Side effect, deliberately preserved: in nested mode this CREATES the system directory,
    on the read path as much as the write path. ``dev/scripts/Fit_Binary_Systems.py``
    relies on it for cold fetches into the workspace ``matrix_data`` store.

    Raises:
        CacheModeError: under ``cache_mode='sqlite'``, where the question "which directory"
            has no answer and any string returned here would be a fabrication.
    """
    warnings.warn(
        "gliquid.api._resolve_sys_dir() is deprecated and will be removed in a future "
        "release; use gliquid.api.resolve_cache_path(CacheKey(...)) instead, which names a "
        "record rather than a directory and works for single-file stores too.",
        DeprecationWarning,
        stacklevel=2,
    )
    backend = resolve_backend(data_dir)
    sys_location = getattr(backend, "sys_location", None)
    if sys_location is None:
        raise CacheModeError(
            f"The configured gliquid cache store keeps every record in one file, so there "
            f"is no cache DIRECTORY for '{sys_name}' to return. Use "
            f"gliquid.api.resolve_cache_path(CacheKey(...)) to name a record instead."
        )
    return str(sys_location(sys_name))


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
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``. The parameter keeps its historical name.

    Returns:
        list[ComputedStructureEntry]: Real (non-imputed) entries with structures.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    _validate_dft_type(dft_type)

    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)

    computed_entry_dicts = None
    if backend.exists(key):
        computed_entry_dicts = backend.read_json(key)
        if verbose:
            logger.info("Loading cached DFT entry data.")

    real_entries = [e for e in (computed_entry_dicts or []) if not _is_imputed_entry_dict(e)]
    if computed_entry_dicts is None or any("structure" not in e for e in real_entries):
        imputed = [e for e in (computed_entry_dicts or []) if _is_imputed_entry_dict(e)]
        real_entries = _get_dft_entries_from_components(components, dft_type, keep_data=True)
        computed_entry_dicts = real_entries + imputed
        if verbose:
            logger.info(f"Caching DFT entry data as {backend.locate(key)}...")
        backend.write_json(key, computed_entry_dicts)

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
        dft_type (str): Functional type, selecting the cache record.
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``. The parameter keeps its historical name.

    Returns:
        str: A handle to the record written — the cache file path for a directory store.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)

    if backend.exists(key):
        existing = backend.read_json(key)
    else:
        existing = _get_dft_entries_from_components(components, dft_type)

    new_ids = {e.get("entry_id") for e in imputed_entry_dicts}
    existing = [e for e in existing if e.get("entry_id") not in new_ids]
    existing.extend(imputed_entry_dicts)
    backend.write_json(key, existing)
    return backend.locate(key)


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
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``, e.g. the per-instance ternary store. The parameter keeps its
            historical name.

    Returns:
        A tuple of the phase diagram and a dictionary of stable entry atomic volumes.
    """
    components, _, _ = validate_and_format_system(input, allow_compounds=True)
    cache_name = _canonical_sys_name(components)

    _validate_dft_type(dft_type)
    if verbose:
        logger.info(f"Using DFT entries solved with {dft_type} functionals.")

    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)
    cached = backend.exists(key)

    # Yb-containing structures are only available with R2SCAN functional
    # See https://docs.materialsproject.org/changes/database-versions#v2023.11.1
    # and https://docs.materialsproject.org/changes/database-versions#v2025.02.12
    if "Yb" in components and not cached:
        logger.warning(
            "Yb-containing structures are only available with R2SCAN or MIXED functionals "
            "on the MP database, and BOTH are blocked upstream (see _BLOCKED_DFT_TYPES), so "
            "there is no working functional for a cold Yb fetch: expect an empty result."
        )

    if cached:
        computed_entry_dicts = [_normalize_entry_dict(e) for e in backend.read_json(key)]
        if verbose:
            logger.info("Loading cached DFT entry data.")
    else:
        computed_entry_dicts = _get_dft_entries_from_components(components, dft_type)
        if verbose:
            logger.info(f"Caching DFT entry data as {backend.locate(key)}...")
        backend.write_json(key, computed_entry_dicts)

    if not include_imputed:
        computed_entry_dicts = [e for e in computed_entry_dicts if not _is_imputed_entry_dict(e)]

    # Read-time Tier A guard (blacklist + anchor consistency): caches of any age
    # may hold blacklisted elemental structures, sub-anchor elemental structures,
    # or the blacklisted MP composition artifacts old ternary caches predate;
    # filter in memory, never rewrite the cache file.
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
