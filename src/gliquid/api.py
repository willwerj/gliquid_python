"""Author: Joshua Willwerth
Description: Single home for everything in gliquid that talks to a remote API: credential
resolution (environment variables first, then an optional gitignored ``.env`` file; existing
environment variables are kept), lazy construction of the Materials Project and MPDS clients,
and the DFT-entry cache/load layer: fetching Materials Project entries for an n-component
system, caching them as ``{sys}_ENTRIES_MP_{type}.json``, and building the
(Compound)PhaseDiagram convex hull from the cache.

MPDS digitized-phase-diagram parsing lives in ``gliquid.mpds``. Importing this module imports
none of ``mp_api``, ``mpds_client``, ``requests`` or ``emmet.core``.
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import re
import sys
import types
import warnings
from pathlib import Path

from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

import gliquid.config as config
from gliquid.cache import (
    KIND_DFT_ENTRIES,
    KIND_IMPUTED_ENTRIES,
    NON_DFT_KEY,
    USER_ID_PREFIX,
    CacheKey,
    CacheModeError,
    imputed_entry_label,
    is_imputed_entry_dict,
    non_dft_entry_kind,
    non_dft_entry_label,
    resolve_backend,
)
from gliquid.phase import validate_and_format_system

logger = logging.getLogger(__name__)

MP_KEY_VAR = "NEW_MP_API_KEY"
MPDS_KEY_VAR = "MPDS_API_KEY"


# -- Credential resolution --


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


# -- Materials Project clients (mp_api imported inside functions) --


def _construct_mprester(api_key: str):
    """Construct an MPRester that returns raw dicts rather than document models."""
    from mp_api.client import MPRester

    return MPRester(api_key, monty_decode=False, use_document_model=False)


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


# -- MPDS client (optional mpds_client, imported inside functions) --


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
    mpds_client = config._require_extra(
        "mpds_client",
        "mpds-client is required for live MPDS retrieval. "
        "Install it with `pip install gliquid[mpds]` or `pip install mpds-client`.",
    )
    api_key = get_api_key(MPDS_KEY_VAR)
    if not api_key:
        raise ValueError(f"{MPDS_KEY_VAR} not found in environment variables!")
    client = mpds_client.MPDSDataRetrieval(api_key=api_key)
    client.dtype = getattr(mpds_client.MPDSDataTypes, dtype)
    return client


# -- DFT entry cache: n-component Materials Project entries and convex hulls, stored under
# ``CacheKey(sys_name, 'dft_entries', dft_type)`` in the configured or ``data_dir`` backend --

_LEGACY_MONTY_MODULES = {
    "pymatgen.core.entries": "pymatgen.entries.computed_entries",
    "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility",
}

# The dft_type names this package recognizes -- not the same thing as the ones that WORK.
SUPPORTED_DFT_TYPES = ("GGA", "R2SCAN", "MIXED")

# The functional used when none is given.
DEFAULT_DFT_TYPE = "GGA"

# Functionals that cannot be fetched, each with its one-line diagnosis; rejected at the entry
# points below before any network call.
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

    Called at the public entry points only; ``_get_dft_entries_from_components`` still accepts
    the blocked types.
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


class _LegacyModuleAlias(types.ModuleType):
    """``sys.modules`` stand-in for a retired pymatgen path in ``_LEGACY_MONTY_MODULES``.

    The first lookup of an attribute that is not a dunder imports the current module, puts it
    in ``sys.modules`` under the retired name and returns the attribute from it. Dunder
    lookups raise ``AttributeError`` without importing anything.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        current = importlib.import_module(_LEGACY_MONTY_MODULES[self.__name__])
        sys.modules[self.__name__] = current
        return getattr(current, name)


def _alias_legacy_monty_modules() -> None:
    """Make the retired pymatgen paths importable, so UPSTREAM monty decodes resolve.

    ``_normalize_entry_dict`` below rewrites ``@module`` on dicts gliquid loads itself. It
    cannot help when the decode happens inside mp_api: ``MPRester.get_entries`` calls
    ``ComputedStructureEntry.from_dict`` on the server's response, and the Materials Project
    still tags nested ``energy_adjustments`` records with ``pymatgen.core.entries``. monty
    then ``__import__``s that dead path and raises ModuleNotFoundError, so every cache-MISS
    fetch fails while a warm cache succeeds.

    Registering the alias fixes the fetch for code gliquid never touches. Each path is only
    aliased if it is genuinely gone, so a future pymatgen that restores one wins. The alias
    is a :class:`_LegacyModuleAlias`, which imports the current module on first use.
    """
    for legacy in _LEGACY_MONTY_MODULES:
        if legacy in sys.modules:
            continue
        try:
            importlib.import_module(legacy)
        except ImportError:
            sys.modules[legacy] = _LegacyModuleAlias(legacy)


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


@functools.lru_cache(maxsize=8192)
def _guessed_oxidation_states(composition_key: tuple) -> tuple:
    """Best ``Composition.oxi_state_guesses`` result, as ``(element, state)`` pairs."""
    try:
        guesses = Composition(dict(composition_key)).oxi_state_guesses(max_sites=-20)
    except ValueError:
        guesses = ()
    return tuple((guesses[0] if guesses else {}).items())


def apply_mp2020_corrections(entry_dicts: list[dict]) -> list[dict]:
    """``entry_dicts`` with the MP2020 energy corrections recomputed on this machine.

    Oxidation states are guessed from each entry's composition; any ``data['oxidation_states']``
    the entry carries is ignored. ``correction`` and ``energy_adjustments`` are replaced on a
    copy of each dict and every other key is left as it was. Imputed and user-defined entries,
    and entries MP2020 does not accept (a run type other than GGA or GGA+U, or an unexpected
    Hubbard U), are returned unchanged.

    Args:
        entry_dicts: ``ComputedEntry`` or ``ComputedStructureEntry`` ``as_dict()`` payloads.

    Returns:
        list[dict]: one dict per input, in input order.
    """
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    compat = MaterialsProject2020Compatibility(check_potcar=False)
    return [_with_mp2020_adjustments(entry, compat) for entry in entry_dicts]


def _with_mp2020_adjustments(raw: dict, compat) -> dict:
    if non_dft_entry_kind(raw) is not None:
        return raw
    payload = dict(raw)
    if not {"O", "S"} & set(payload.get("composition") or {}):
        payload.pop("structure", None)
        payload["@class"] = "ComputedEntry"
    entry = _computed_entry_from_dict(payload)
    key = tuple(
        sorted((str(el), float(n)) for el, n in entry.composition.reduced_composition.items())
    )
    entry.data = {**entry.data, "oxidation_states": dict(_guessed_oxidation_states(key))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if compat.process_entry(entry, inplace=True, clean=True) is None:
            return raw
    out = dict(raw)
    out["correction"] = float(entry.correction)
    out["energy_adjustments"] = [adjustment.as_dict() for adjustment in entry.energy_adjustments]
    return out


# Tier A spurious-structure blacklist (``config.spurious_structures_file``), applied at fetch and
# at cache read: 'elements' lists elemental structures with no stability field at 1 atm,
# 'compounds' lists non-elemental artifacts. Classic string ids match by material_id; new-API
# ids match by (element, spacegroup) of their structure (symprec=0.1).
_MP_ID_RE = re.compile(r"(mp-[a-z0-9]+)")
_spurious_cache: tuple | None = None  # (path, mtime, ids, pairs, expected_gs, compounds)

# Compound-rule predicate forms in evaluation precedence; the first form present decides a record.
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
        else:  # element_count: plain numeric equality
            # (an int cast would swallow a genuinely fractional near-miss)
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

    The blacklist removes known spurious elemental structures and, via the ``compounds`` block,
    non-elemental artifacts of any arity. The anchor guard then drops any elemental entry of an
    audited element that sits below the lowest entry of the element's expected ground-state
    spacegroup while not being of that spacegroup itself, so every hull's elemental reference is
    the experimentally anchored structure of the unary DB. The anchor guard applies to elemental
    entries only.
    """
    ids, pairs, expected, compounds = _spurious_structure_index()
    if not ids and not pairs and not expected and not compounds:
        return computed_entry_dicts
    survivors = [e for e in computed_entry_dicts if not _is_spurious_entry_dict(e)]
    if not expected:
        return survivors

    # Reference energy per audited element: the lowest energy per atom among its entries of the
    # expected spacegroup, or None.
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
    # Refused in offline mode before the credential lookup and before any network import.
    config.require_online(
        f"Fetching Materials Project DFT entries for '{'-'.join(components)}' "
        f"(dft_type={dft_type!r})"
    )
    from emmet.core.thermo import ThermoType

    entries = []

    # Queried by element symbols, so a compound component contributes its elements.
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
        from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

        entries = MaterialsProjectDFTMixingScheme().process_entries(
            scan_entries + ggau_entries, verbose=False
        )
    elif dft_type == "GGA":
        entries = ggau_entries
    elif dft_type == "R2SCAN":
        entries = scan_entries

    # The client may return raw dicts or ComputedEntry objects.
    computed_entry_dicts = [e.as_dict() if hasattr(e, "as_dict") else e for e in entries]
    if dft_type == "GGA":
        computed_entry_dicts = apply_mp2020_corrections(computed_entry_dicts)

    # Tier A: blacklist + anchor-consistency guard at fetch time; run data is dropped just below.
    computed_entry_dicts = _filter_spurious_entries(computed_entry_dicts)
    if not keep_data:
        for e in computed_entry_dicts:
            e.pop("data", None)

    # An empty fetch raises before any cache write.
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


def _canonical_sys_name(components) -> str:
    """Alphabetical hyphenated system name — the on-disk cache-key convention.

    Construction order stays authoritative in memory; canonicalizing only the disk key
    lets 'Zr-Hf' and 'Hf-Zr' share one cache.
    """
    return "-".join(sorted(components))


def resolve_cache_path(key: CacheKey, cache=None) -> Path | None:
    """The filesystem path of one cached record, or ``None`` if the store has no paths.

    It names a RECORD rather than a directory, which is the only question every backend
    can answer. ``None`` is a real answer, not an error -- a single-file store holds the record but not at a path -- so
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


def _merge_stored_imputed_entries(backend, cache_name, dft_type, entry_dicts) -> list[dict]:
    """``entry_dicts`` with the store's ``KIND_IMPUTED_ENTRIES`` record folded in.

    One entry per ``entry_id``: a stored record REPLACES a tagged entry of the same id in
    place, and anything the DFT payload does not already carry is appended. A directory
    store, which holds its imputed entries inside that payload, gets its list back
    unchanged.
    """
    stored = read_imputed_entries(cache_name, dft_type, data_dir=backend)
    if not stored:
        return entry_dicts

    by_id = {e.get("entry_id"): e for e in stored}
    unused = set(by_id)
    merged = []
    for entry in entry_dicts:
        entry_id = entry.get("entry_id")
        if is_imputed_entry_dict(entry) and entry_id in unused:
            unused.discard(entry_id)
            merged.append(by_id[entry_id])
        else:
            merged.append(entry)
    merged.extend(e for e in stored if e.get("entry_id") in unused)
    return merged


def _require_writable_store(backend, key: CacheKey) -> None:
    """Refuse a fetch whose result ``backend`` could not store under ``key``.

    Call immediately before a remote fetch that writes its result. In offline mode it
    returns without checking, so the fetch's own ``config.require_online`` raises. A backend
    that declares no ``capabilities`` is treated as writable.

    Raises:
        CacheModeError: ``backend`` declares ``capabilities`` without ``'write'``.
    """
    if config.offline:
        return
    capabilities = getattr(backend, "capabilities", None)
    if capabilities is None or "write" in capabilities:
        return
    store = getattr(backend, "path", backend)
    raise CacheModeError(
        f"Fetching {backend.locate(key)} is not possible: the cache store at {store} is "
        f"READ-ONLY, so the fetched record could not be written to it. Nothing was "
        f"requested. To run something that writes cold-fetched records, point gliquid at a "
        f"directory store instead: gliquid.config.set_cache_dir('/path/to/cache/dir'). To "
        f"change the file store's contents, rebuild it: `python -m gliquid.cache migrate "
        f"--from <cache dir> --to {store}`."
    )


@config._removed_keywords({"input": "system"})
def get_dft_structure_entries(
    system, dft_type="GGA", verbose=False, data_dir=None
) -> list[ComputedStructureEntry]:
    """Cached DFT entries with structures retained, for structure-based analysis.

    Same cache file as ``get_dft_convexhull`` (``<sys>_ENTRIES_MP_<type>.json``), but
    guarantees each returned entry carries its structure (needed e.g. for spacegroup
    identification in ``gliquid.solution``). A cache written without structures is
    refreshed once via the API; imputed entries (no structures by construction) are
    preserved in the cache across a refresh and excluded from the returned list.

    Args:
        system (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        dft_type (str): Functional type. Only 'GGA' works; 'R2SCAN' and 'MIXED' are
            recognized names blocked upstream and raise (see ``_BLOCKED_DFT_TYPES``).
        verbose (bool): Whether to print cache activity.
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``.

    Returns:
        list[ComputedStructureEntry]: Real (non-imputed) entries with structures.

    Raises:
        CacheModeError: the store is read-only and its record is missing or lacks
            structures; raised before any fetch.
    """
    components, _, _ = validate_and_format_system(system, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    _validate_dft_type(dft_type)

    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)

    computed_entry_dicts = None
    if backend.exists(key):
        computed_entry_dicts = backend.read_json(key)
        if verbose:
            logger.info("Loading cached DFT entry data.")

    real_entries = [e for e in (computed_entry_dicts or []) if not is_imputed_entry_dict(e)]
    if computed_entry_dicts is None or any("structure" not in e for e in real_entries):
        imputed = [e for e in (computed_entry_dicts or []) if is_imputed_entry_dict(e)]
        _require_writable_store(backend, key)
        real_entries = _get_dft_entries_from_components(components, dft_type, keep_data=True)
        computed_entry_dicts = real_entries + imputed
        if verbose:
            logger.info(f"Caching DFT entry data as {backend.locate(key)}...")
        backend.write_json(key, computed_entry_dicts)

    # Read-time Tier A guard (see get_dft_convexhull): filters in memory, cache unchanged.
    return [_computed_entry_from_dict(e) for e in _filter_spurious_entries(real_entries)]


@config._removed_keywords({"input": "system"})
def cache_imputed_entries(
    system, imputed_entry_dicts: list[dict], dft_type="GGA", data_dir=None
) -> str:
    """Add imputed ComputedEntry dicts to the store's imputed-entry record for a system.

    Written through ``KIND_IMPUTED_ENTRIES``, so where they land is the store's business:
    a directory store keeps them tagged inside ``<sys>_ENTRIES_MP_<type>.json`` beside the
    real DFT entries, a SQLite store keeps them in its own table. Either way
    ``get_dft_convexhull(..., include_imputed=False)`` (the default) ignores them and the
    canonical DFT-only hull is unchanged. Re-running is idempotent: an existing imputed
    entry with the same ``entry_id`` is replaced, and imputed entries under other ids are
    kept. The system's DFT record is fetched and written first when the store has none.

    Args:
        system (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        imputed_entry_dicts (list[dict]): Tagged ``ComputedEntry.as_dict()`` payloads.
        dft_type (str): Functional type, selecting the cache record.
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``.

    Returns:
        str: A handle to the record written — the cache file path for a directory store.

    Raises:
        CacheModeError: the store is read-only and holds no DFT record for the system;
            raised before any fetch.
    """
    components, _, _ = validate_and_format_system(system, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    backend = resolve_backend(data_dir)
    dft_key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)
    key = CacheKey(cache_name, KIND_IMPUTED_ENTRIES, dft_type)

    if not backend.exists(dft_key):
        _require_writable_store(backend, dft_key)
        backend.write_json(dft_key, _get_dft_entries_from_components(components, dft_type))

    existing = backend.read_json(key) if backend.exists(key) else []
    new_ids = {e.get("entry_id") for e in imputed_entry_dicts}
    kept = [e for e in existing if e.get("entry_id") not in new_ids]
    kept.extend(imputed_entry_dicts)
    backend.write_json(key, kept)
    return backend.locate(key)


def read_imputed_entries(system, dft_type="GGA", data_dir=None) -> list[dict]:
    """The store's imputed-entry record for a system. Read twin of
    :func:`cache_imputed_entries`.

    Args:
        system (str or list): System specification (e.g., 'A-B', ['A', 'B', 'C']).
        dft_type (str): Functional type, selecting the cache record.
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``.

    Returns:
        list[dict]: Tagged ``ComputedEntry.as_dict()`` payloads; empty when the store holds
        none for the system.
    """
    components, _, _ = validate_and_format_system(system, allow_compounds=True)
    backend = resolve_backend(data_dir)
    key = CacheKey(_canonical_sys_name(components), KIND_IMPUTED_ENTRIES, dft_type)
    if not backend.exists(key):
        return []
    try:
        stored = backend.read_json(key)
    except FileNotFoundError:
        return []
    return [_normalize_entry_dict(e) for e in stored]


def drop_imputed_entries(system, dft_type="GGA", entry_ids=None, data_dir=None) -> int:
    """Remove imputed entries from the store's imputed-entry record. Inverse of
    :func:`cache_imputed_entries`.

    Use it to retire the entries ``gliquid.imputation.lifecycle.imputed_entry_staleness``
    reports stale.

    Real DFT entries are never touched: only the ``KIND_IMPUTED_ENTRIES`` record is read and
    rewritten. Removing nothing is not an error; a system with no imputed entries returns 0.

    Args:
        system (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        dft_type (str): Functional type, selecting the cache record.
        entry_ids (iterable[str] | None): Restrict to these ``entry_id`` values; ``None``
            (the default) drops every imputed entry for the system.
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``.

    Returns:
        int: how many entries were removed.
    """
    components, _, _ = validate_and_format_system(system, allow_compounds=True)
    cache_name = _canonical_sys_name(components)
    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_IMPUTED_ENTRIES, dft_type)
    if not backend.exists(key):
        return 0

    existing = backend.read_json(key)
    if entry_ids is None:
        kept = []
    else:
        wanted = set(entry_ids)
        kept = [e for e in existing if e.get("entry_id") not in wanted]
    removed = len(existing) - len(kept)
    if removed:
        backend.write_json(key, kept)
    return removed


@config._removed_keywords({"input": "system"})
def get_dft_convexhull(
    system,
    dft_type="GGA",
    inc_structure_data=False,
    verbose=False,
    include_imputed=False,
    data_dir=None,
) -> tuple[PhaseDiagram, dict]:
    """
    Returns the DFT convex hull of a given n-component system with specified functionals.

    Args:
        system (str or list): System specification (e.g., 'A-B', ['A', 'B'] or
            ['A', 'B', 'C'] — any number of components >= 2).
        dft_type (str): Functional type. Only 'GGA' works; 'R2SCAN' and 'MIXED' are
            recognized names blocked upstream and raise (see ``_BLOCKED_DFT_TYPES``).
        inc_structure_data (bool): Whether to include structural data.
        verbose (bool): Whether to print detailed output.
        include_imputed (bool): If False (default), imputed entries cached by
            ``cache_imputed_entries`` are filtered out so the hull is DFT-only and the
            store's ``KIND_IMPUTED_ENTRIES`` record is never read. Set True to merge that
            record in, one entry per ``entry_id`` (phase-energy imputation workflow).
        data_dir: Explicit cache STORE override — a directory (flat layout inside it) or a
            ``CacheBackend``, e.g. the per-instance ternary store.

    Returns:
        A tuple of the phase diagram and a dictionary of stable entry atomic volumes.

    Raises:
        CacheModeError: the store is read-only and holds no record for the system; raised
            before any fetch.
    """
    components, _, _ = validate_and_format_system(system, allow_compounds=True)
    cache_name = _canonical_sys_name(components)

    _validate_dft_type(dft_type)
    if verbose:
        logger.info(f"Using DFT entries solved with {dft_type} functionals.")

    backend = resolve_backend(data_dir)
    key = CacheKey(cache_name, KIND_DFT_ENTRIES, dft_type)
    cached = backend.exists(key)

    # Yb structures exist only under R2SCAN on the Materials Project.
    if "Yb" in components and not cached:
        logger.warning(
            "Yb-containing structures are only available with R2SCAN or MIXED functionals "
            "on the MP database, and BOTH are blocked upstream (see _BLOCKED_DFT_TYPES), so "
            "there is no working functional for a cold Yb fetch: expect an empty result."
        )

    if cached:
        computed_entry_dicts = backend.read_json(key)
        if verbose:
            logger.info("Loading cached DFT entry data.")
    else:
        _require_writable_store(backend, key)
        computed_entry_dicts = _get_dft_entries_from_components(components, dft_type)
        if verbose:
            logger.info(f"Caching DFT entry data as {backend.locate(key)}...")
        backend.write_json(key, computed_entry_dicts)

    if include_imputed:
        computed_entry_dicts = _merge_stored_imputed_entries(
            backend, cache_name, dft_type, computed_entry_dicts
        )
    else:
        computed_entry_dicts = [e for e in computed_entry_dicts if not is_imputed_entry_dict(e)]

    # Tier A guard on read: filters cached entries in memory without rewriting the cache.
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


# -- Hull seams: component identities, axis fractions and display names of a PhaseDiagram or
# CompoundPhaseDiagram --


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


def hull_entry_name(entry) -> str:
    """The display name for a solid hull entry.

    An imputed entry is named by :func:`gliquid.cache.imputed_entry_label` and a user-defined
    one by :func:`gliquid.cache.non_dft_entry_label`; every other entry keeps
    ``api.entry_display_name``. An imputed entry sits at a FRACTIONAL composition, whose
    reduced formula is not a phase name.
    """
    data = getattr(entry, "data", None) or {}
    if data.get("imputed"):
        return imputed_entry_label({"entry_id": entry.entry_id, "data": data})
    if non_dft_entry_kind(entry) == "user":
        return non_dft_entry_label(entry)
    return entry_display_name(entry)


def user_entry(
    composition: Composition, energy: float, label: str, entry_id: str | None = None, **data
) -> ComputedEntry:
    """A user-defined hull entry: a ``ComputedEntry`` tagged as standing in for a DFT compound.

    Args:
        composition: the phase composition.
        energy: total energy, eV, on the same basis as the hull it joins.
        label: the phase name; set as the entry's ``name`` and stored as ``data['label']``.
        entry_id: the id to carry; ``None`` gives ``'user:<label>'``.
        **data: further ``data`` keys, e.g. ``replaces`` for the DFT entry ids it stands in for.

    Returns:
        ComputedEntry: an entry :func:`gliquid.cache.non_dft_entry_kind` reports as ``"user"``.
    """
    entry = ComputedEntry(
        composition,
        energy,
        entry_id=entry_id or f"{USER_ID_PREFIX}{label}",
        data={NON_DFT_KEY: "user", "label": label, **data},
    )
    entry.name = label
    return entry


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
