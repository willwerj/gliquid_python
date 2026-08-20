"""The cache seam: which STORE a cached record lives in, and how to read or write it.

A record is named by a :class:`CacheKey`; a store implements :class:`CacheBackend`.
:class:`DirectoryBackend` is a directory tree of per-system JSON files;
:class:`SqliteBackend` is a single-file store.

``read_json``/``write_json`` exchange **Python objects, not bytes**. A bytes-level API would
force a backend that holds a semantic REDUCTION of a record (the lean MPDS mode) to
fabricate a JSON blob it does not have, and the fabrication would be indistinguishable from
the real thing downstream.

``config.cache_mode`` (``"directory"`` / ``"sqlite"``) selects the backend and is
**orthogonal** to ``config.dir_structure`` (``"flat"`` / ``"nested"``), which is a
:class:`DirectoryBackend`-only knob describing one particular on-disk arrangement.

:class:`SqliteBackend` is the second store: the whole corpus in ONE file, which is what
makes a corpus shippable (a container image, a web app, a colleague's laptop). It is
**read-mostly by design** -- see its docstring for why writes need an explicit opt-in.

A SQLite store may additionally carry a **pooled entry store** (``entry_pool``): DFT
entries deduplicated by ``entry_id`` across systems, read back as
``chemsys IN (<the subsets>)``. It is a fallback source for ``dft_entries`` reads, exists
for the TERNARY scope only, and is licensed by a drift measurement rather than by the
obvious size win -- see the comment above ``_ENTRY_POOL_SQL``.

It may also carry the **ML feature corpus** (``ml_features`` / ``ml_feature_columns``): one
row of model input features per system, per feature frame. The model itself ships inside the
wheel at ``gliquid/models/<bundle_id>/``; these tables are the other half of that split, and
are read only by ``gliquid.production_model_runner``.

Run ``python -m gliquid.cache --help`` for the migrate / verify / info CLI.

IMPORT HYGIENE: this module imports stdlib and ``gliquid.config`` only. ``gliquid.api``
imports it, so it must never import ``gliquid.api`` back.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import sys
import tempfile
import threading
import urllib.request
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import gliquid.config as config

logger = logging.getLogger(__name__)

#: A per-system Materials Project DFT-entry cache. ``variant`` is the ``dft_type``.
KIND_DFT_ENTRIES = "dft_entries"

#: A digitized MPDS phase diagram. ``variant`` is ``str(pd_ind)``, or ``""`` for the
#: indexless ``<sys>.json`` naming used by single-diagram stores.
KIND_MPDS = "mpds"

_KINDS = (KIND_DFT_ENTRIES, KIND_MPDS)


#: Re-exported from :mod:`gliquid.config`. The same class object, so existing
#: ``except CacheModeError`` clauses keep working.
CacheModeError = config.CacheModeError


@dataclass(frozen=True)
class CacheKey:
    """The name of one cached record — deliberately NOT a string.

    A string key would have to be parsed apart again by every backend:
    :class:`DirectoryBackend` needs the parts to rebuild today's filenames, and the SQLite
    store needs them as a composite primary key. Keeping them separate means the on-disk
    naming convention is a property of the backend rather than of the key.

    Attributes:
        sys_name: The canonical system name — the output of ``api._canonical_sys_name``
            (alphabetical and hyphenated, e.g. ``'Cu-Mg'``).
        kind: :data:`KIND_DFT_ENTRIES` or :data:`KIND_MPDS`.
        variant: The ``dft_type`` for DFT entries; ``str(pd_ind)`` for MPDS, where ``""``
            means the indexless ``<sys>.json`` naming.
    """

    sys_name: str
    kind: str
    variant: str = ""


@runtime_checkable
class CacheBackend(Protocol):
    """A store of cached records, addressed by :class:`CacheKey`."""

    @property
    def capabilities(self) -> frozenset[str]:
        """What this store can do — e.g. ``{'read', 'write', 'paths', 'variants'}``.

        ``'paths'`` means every record has a filesystem path (``path_for``); a single-file
        store does not, and callers that need one must degrade rather than assume.
        """

    def exists(self, key: CacheKey) -> bool:
        """Whether a record is present, without reading it."""

    def read_json(self, key: CacheKey) -> Any:
        """The record as a Python object. Raises ``FileNotFoundError`` when absent."""

    def write_json(self, key: CacheKey, payload: Any) -> None:
        """Store ``payload`` under ``key``, atomically where the store can."""

    def variants(self, sys_name: str, kind: str) -> list[str]:
        """Every ``variant`` this store holds for ``(sys_name, kind)``."""

    def locate(self, key: CacheKey) -> str:
        """A path, or a store-relative handle — for log lines and error messages only."""


def atomic_write_json(path: str | os.PathLike, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` atomically, via a temp file + ``os.replace``.

    A plain ``json.dump`` to the cache path writes incrementally, so an interrupted or
    failing write leaves a TRUNCATED file that later reads accept as a valid cache. Worse,
    concurrent cold fetches of the same system interleave into one file, which a campaign
    fanned out over a process pool produces in bursts.

    The temp file is created in the DESTINATION directory so it shares a filesystem with
    the target; ``os.replace`` is only atomic within one filesystem, and is atomic on both
    POSIX and Windows. A reader therefore sees either the old file or the complete new one,
    never a partial one. Two writers racing still both fetch -- that is accepted -- but
    whichever lands last leaves a whole, valid file.
    """
    path = str(path)
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


def record_filename(key: CacheKey) -> str:
    """Today's on-disk filename for a record — the naming contract, in one place.

    Three conventions, all of which exist in the corpus right now:
    ``<sys>_ENTRIES_MP_<dft_type>.json``, ``<sys>_MPDS_PD_<n>.json``, and the indexless
    ``<sys>.json`` that single-diagram stores use.
    """
    if key.kind == KIND_DFT_ENTRIES:
        return f"{key.sys_name}_ENTRIES_MP_{key.variant}.json"
    if key.kind == KIND_MPDS:
        if key.variant == "":
            return f"{key.sys_name}.json"
        return f"{key.sys_name}_MPDS_PD_{key.variant}.json"
    raise ValueError(f"Unknown cache kind {key.kind!r}. Must be one of {list(_KINDS)}.")


#: The inverse of :func:`record_filename`, as patterns. Ordered: the two *decorated*
#: namings are tried before the bare ``<sys>.json``, because ``Cu-Mg_ENTRIES_MP_GGA.json``
#: is also "some name ending in .json" and must not be read as an MPDS record.
_ENTRIES_NAME_RE = re.compile(r"\A(?P<sys>.+)_ENTRIES_MP_(?P<variant>.+)\.json\Z")
_MPDS_NAME_RE = re.compile(r"\A(?P<sys>.+)_MPDS_PD_(?P<variant>.+)\.json\Z")
_PLAIN_NAME_RE = re.compile(r"\A(?P<sys>.+)\.json\Z")


def parse_record_filename(name: str, sys_name: str | None = None) -> CacheKey | None:
    """``record_filename`` inverted: a filename to a :class:`CacheKey`, or ``None``.

    ``None`` means "not a cache record at all" -- corpora hold plenty of neighbours that
    are not (``fit_results_cache_comb-exp.json``, ``all_feature_dft_data.json``, every
    ``.svg`` and ``.webp`` beside them), and reading one as a record would invent a system.

    Args:
        name: The bare filename.
        sys_name: When given, the system the file must belong to -- the nested layout knows
            it from the enclosing directory, and pinning it stops ``Cu-Mg_extra_notes.json``
            from being read as a record of the system ``Cu-Mg_extra_notes``.
    """
    for pattern, kind in ((_ENTRIES_NAME_RE, KIND_DFT_ENTRIES), (_MPDS_NAME_RE, KIND_MPDS)):
        match = pattern.match(name)
        if match and (sys_name is None or match.group("sys") == sys_name):
            return CacheKey(match.group("sys"), kind, match.group("variant"))
    match = _PLAIN_NAME_RE.match(name)
    if match and (sys_name is None or match.group("sys") == sys_name):
        return CacheKey(match.group("sys"), KIND_MPDS, "")
    return None


# Every IUPAC element symbol, used to decide whether a name looks like a system.
# Hard-coded to keep this module stdlib-only: ``gliquid.api`` imports it.
_ELEMENT_SYMBOLS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga
    Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm
    Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th
    Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
)

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def _is_formula(part: str) -> bool:
    """Whether ``part`` is a run of element symbols with optional counts (``Cu``, ``CuMg2``)."""
    if not part:
        return False
    pos = 0
    while pos < len(part):
        match = _FORMULA_TOKEN_RE.match(part, pos)
        if match is None or match.group(1) not in _ELEMENT_SYMBOLS or match.end() == pos:
            return False
        pos = match.end()
    return True


def looks_like_system_name(name: str) -> bool:
    """Whether ``name`` has the shape of a gliquid system key: formulas joined by ``-``.

    ``api._canonical_sys_name`` builds these as ``'-'.join(sorted(components))`` where each
    component is an element or a composition, so ``'Cu-Mg'``, ``'Hf-Ti-Zr'`` and the
    compound-system ``'CuMg-Mg'`` all qualify while ``'fit_results_cache_comb-exp'`` and
    ``'all_feature_dft_data'`` do not.
    """
    parts = name.split("-")
    return len(parts) >= 2 and all(_is_formula(part) for part in parts)


class DirectoryBackend:
    """A directory tree of per-system JSON files — the store gliquid has always had.

    Args:
        root: The corpus directory. ``None`` means "whatever ``config.cache_dir`` is at the
            time of the call", so a ``set_cache_dir()`` part-way through a session is
            observed — which is how the test suites and the fitting campaigns drive it.
        dir_structure: ``"flat"`` (every file directly under ``root``) or ``"nested"``
            (under ``root/<sys_name>/``). ``None`` means "whatever ``config.dir_structure``
            is at the time of the call". An EXPLICIT ``root`` defaults to ``"flat"``,
            which is the historical meaning of passing ``data_dir=`` to
            ``api.get_dft_convexhull`` and friends.
    """

    def __init__(
        self,
        root: str | os.PathLike | None = None,
        dir_structure: str | None = None,
    ):
        self._root = Path(root) if root is not None else None
        if dir_structure is None and root is not None:
            dir_structure = "flat"
        self._dir_structure = dir_structure

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        root = self._root if self._root is not None else "<config.cache_dir>"
        struct = self._dir_structure or "<config.dir_structure>"
        return f"DirectoryBackend(root={root!r}, dir_structure={struct!r})"

    # -- configuration -------------------------------------------------------------

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"read", "write", "paths", "variants"})

    def _root_path(self, purpose: str) -> Path:
        if self._root is not None:
            return self._root
        return config.require_cache_dir(purpose)

    def _structure(self) -> str:
        return self._dir_structure if self._dir_structure is not None else config.dir_structure

    def sys_location(self, sys_name: str) -> Path:
        """The directory holding ``sys_name``'s records.

        Creates it in nested mode, on the READ path as well as the write path. Campaign
        drivers depend on that for cold fetches, so it is preserved deliberately.
        """
        root = self._root_path(f"Reading the cache for '{sys_name}'")
        structure = self._structure()
        if structure == "nested":
            sys_dir = root / sys_name
            os.makedirs(sys_dir, exist_ok=True)
            return sys_dir
        if structure == "flat":
            return root
        raise ValueError(f"Invalid dir_structure '{structure}'. Must be 'nested' or 'flat'.")

    # -- the CacheBackend protocol -------------------------------------------------

    def path_for(self, key: CacheKey) -> Path:
        """The filesystem path of one record. ``DirectoryBackend`` extra (``'paths'``)."""
        return self.sys_location(key.sys_name) / record_filename(key)

    def locate(self, key: CacheKey) -> str:
        return str(self.path_for(key))

    def exists(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def read_json(self, key: CacheKey) -> Any:
        with open(self.path_for(key)) as handle:
            return json.load(handle)

    def write_json(self, key: CacheKey, payload: Any) -> None:
        atomic_write_json(self.path_for(key), payload)

    def variants(self, sys_name: str, kind: str) -> list[str]:
        if kind not in _KINDS:
            raise ValueError(f"Unknown cache kind {kind!r}. Must be one of {list(_KINDS)}.")
        sys_dir = self.sys_location(sys_name)
        try:
            names = os.listdir(sys_dir)
        except (FileNotFoundError, NotADirectoryError):
            return []

        found = []
        for name in names:
            parsed = parse_record_filename(name, sys_name)
            if parsed is not None and parsed.kind == kind:
                found.append(parsed.variant)
        return sorted(found)


@dataclass
class DirectoryScan:
    """Everything a directory store holds, split into records and non-records.

    ``ignored`` is carried rather than dropped so a migration can SAY how many neighbours
    it walked past. A store that silently reports "0 records migrated" and a store that
    silently reports "3 records migrated" out of 11,671 are the same failure, and both are
    invisible without this count.
    """

    root: Path
    dir_structure: str
    records: list[tuple[CacheKey, Path]] = field(default_factory=list)
    ignored: list[Path] = field(default_factory=list)


def scan_directory_store(root: str | os.PathLike, dir_structure: str = "flat") -> DirectoryScan:
    """Every cache record under ``root``, as ``(CacheKey, path)`` pairs.

    Enumerates exactly what :meth:`DirectoryBackend.variants` would find, so a migration
    and a live read agree on what the corpus contains.

    Args:
        root: The corpus directory.
        dir_structure: ``"flat"`` (records directly under ``root``) or ``"nested"``
            (under ``root/<sys_name>/``).
    """
    root = Path(root)
    if dir_structure not in ("flat", "nested"):
        raise ValueError(f"Invalid dir_structure '{dir_structure}'. Must be 'nested' or 'flat'.")
    scan = DirectoryScan(root=root, dir_structure=dir_structure)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if dir_structure == "nested":
        for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not looks_like_system_name(sys_dir.name):
                scan.ignored.append(sys_dir)
                continue
            for entry in sorted(sys_dir.iterdir()):
                if not entry.is_file():
                    continue
                key = parse_record_filename(entry.name, sys_dir.name)
                if key is None:
                    scan.ignored.append(entry)
                else:
                    scan.records.append((key, entry))
    else:
        for entry in sorted(root.iterdir()):
            if not entry.is_file():
                continue
            key = parse_record_filename(entry.name)
            if key is None or not looks_like_system_name(key.sys_name):
                scan.ignored.append(entry)
            else:
                scan.records.append((key, entry))
    return scan


# ---------------------------------------------------------------------------------------
# The single-file store
# ---------------------------------------------------------------------------------------

#: Bumped when the schema changes incompatibly. A store whose ``PRAGMA user_version``
#: EXCEEDS this is refused rather than read with the wrong column meanings.
SQLITE_SCHEMA_VERSION = 1

#: How a DFT payload is encoded. Recorded in ``meta`` so a reader can tell.
DFT_CODEC = "json+zlib6"

_ZLIB_LEVEL = 6

#: MPDS fields lifted out of the payload into addressable columns, for ``info`` and
#: LEAN reads. ``entry``/``jcode``/``year`` are looked for at the top level and under
#: ``reference``.
_MPDS_HEADER_FIELDS = (
    "chemical_elements",
    "temp",
    "comp_range",
    "reference",
    "labels",
    "entry",
    "jcode",
    "year",
)

#: The whole MPDS json, compressed. Everything gliquid can do, at ~93-97% ``shapes``.
MPDS_MODE_FULL = "full"

#: Header plus the pre-fill stitched liquidus and its covered regions; no ``shapes``.
#: Serves ``extract_digitized_liquidus``, ``liquidus_coverage`` and a liquidus plot;
#: every other MPDS consumer raises.
MPDS_MODE_LEAN = "lean"

MPDS_MODES = (MPDS_MODE_FULL, MPDS_MODE_LEAN)

#: Mirrors ``gliquid.mpds._GLIQUID_KEY`` / ``LEAN_SCHEMA``. Duplicated rather than imported
#: because ``mpds`` imports THIS module; the round-trip test pins the two together.
LEAN_BLOCK_KEY = "_gliquid"
LEAN_SCHEMA = 1

# WITHOUT ROWID + no secondary indices: all access is exact-PK or a PK-prefix scan.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dft_entries (
  sys_name    TEXT NOT NULL,
  dft_type    TEXT NOT NULL,
  payload     BLOB NOT NULL,     -- zlib level 6 over UTF-8 json.dumps(list[entry_dict])
  n_entries   INTEGER NOT NULL,
  sha256      TEXT NOT NULL,     -- of the UNCOMPRESSED bytes: the losslessness proof
  written_utc TEXT NOT NULL,
  PRIMARY KEY (sys_name, dft_type)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS mpds_diagrams (
  sys_name    TEXT NOT NULL,
  variant     TEXT NOT NULL,     -- '0','1',... ; '' == the indexless <sys>.json naming.
                                 -- NOT NULL on purpose: NULL breaks primary-key semantics.
  mode        TEXT NOT NULL,     -- 'full' | 'lean'; see MPDS_MODE_* above
  header      TEXT NOT NULL,     -- json: chemical_elements, temp, comp_range, reference,
                                 --       labels, entry, jcode, year
  payload     BLOB,              -- full mode: zlib of the whole mpds_json. NULL in lean.
  stitched    BLOB,              -- lean mode only: zlib of the PRE-fill [[x, T], ...]
  regions     TEXT,              -- lean mode only: json of the covered [[lo, hi], ...]
  written_utc TEXT NOT NULL,
  PRIMARY KEY (sys_name, variant)
) WITHOUT ROWID;
"""

# The pooled entry store, for ternaries. Reading a system is
# ``chemsys IN (<its non-empty subsets>)`` -- 7 for a ternary, 3 for a binary -- over
# ``idx_pool_chemsys``. Payloads are zlib-compressed per entry so a single entry is
# addressable without inflating the rest.
#
# Binaries are NOT pooled: repeated entry_ids there can carry differing payloads.
# ``entry_pool_systems`` records coverage -- an entry query cannot distinguish a system
# with no ternary compounds from one never fetched.
# ``entry_pool`` is deliberately not WITHOUT ROWID: its payloads spill to overflow pages.
_ENTRY_POOL_SQL = """
CREATE TABLE IF NOT EXISTS entry_pool (
  entry_id   TEXT PRIMARY KEY,
  chemsys    TEXT NOT NULL,     -- sorted, '-'-joined element symbols of the entry
  n_elements INTEGER NOT NULL,
  payload    BLOB NOT NULL      -- zlib level 6 over UTF-8 json.dumps(entry_dict, sort_keys)
);

CREATE INDEX IF NOT EXISTS idx_pool_chemsys ON entry_pool(chemsys);

CREATE TABLE IF NOT EXISTS entry_pool_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entry_pool_systems (
  sys_name            TEXT NOT NULL,
  dft_type            TEXT NOT NULL,
  chemsys             TEXT NOT NULL,
  n_entries           INTEGER NOT NULL,
  sha256              TEXT NOT NULL,  -- of the CANONICAL entry list (see entry_list_sha256)
  mp_database_version TEXT NOT NULL,  -- '' when the source could not report one
  fetched_utc         TEXT NOT NULL,
  source              TEXT NOT NULL,  -- 'mp-api' | the file a --from-existing row came from
  PRIMARY KEY (sys_name, dft_type)
) WITHOUT ROWID;
"""

# The ML feature corpus: the per-system model input tables, one row per system per
# frame. The model itself ships in the wheel at ``gliquid/models/<bundle_id>/``.
# ``frame`` is ``'symmetric'`` or ``'antisymmetric'``, named as in ``feature_names.json``
# and ``_prepare_row(mode=...)``.
#
# ``system`` is ORDERED: Ag-Au and Au-Ag are distinct rows. Do not canonicalise.
# ``features`` is raw little-endian float64 in ``ml_feature_columns.columns`` order;
# a row is meaningless without that table, written in the same transaction.
_ML_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS ml_feature_columns (
  frame     TEXT NOT NULL PRIMARY KEY,   -- 'symmetric' | 'antisymmetric'
  bundle_id TEXT NOT NULL,               -- the model bundle these columns were exported for
  columns   TEXT NOT NULL                -- json list of column names, IN ORDER
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ml_features (
  frame    TEXT NOT NULL,
  system   TEXT NOT NULL,   -- ORDERED: 'Ag-Au' and 'Au-Ag' are distinct rows
  features BLOB NOT NULL,   -- struct '<Nd': raw little-endian float64, N = len(columns)
  PRIMARY KEY (frame, system)
) WITHOUT ROWID;
"""

#: How an ``ml_features`` payload is encoded. Recorded in ``meta`` so a reader can tell.
ML_FEATURE_CODEC = "float64-le"

#: The two feature spaces the production model was trained in. Mirrors the keys of the
#: bundle's ``feature_names.json`` and the ``mode`` argument of ``_prepare_row``.
ML_FRAMES = ("symmetric", "antisymmetric")


#: ``entry_pool_meta`` keys with a defined meaning. A pool refreshed against a different
#: MP database version must record that here.
POOL_META_KEYS = (
    "element_scope",  # json list of element symbols the pool was built to cover
    "mp_database_version",
    "fetched_utc",
    "n_systems_covered",
    "builder_version",
)


def chemsys_key(elements) -> str:
    """``['Cu', 'Ag']`` -> ``'Ag-Cu'`` — the pool's sorted, deduplicated chemsys spelling."""
    return "-".join(sorted({str(el) for el in elements}))


def chemsys_subsets(elements) -> list[str]:
    """Every non-empty subset of ``elements`` as a chemsys key — 7 for a ternary, 3 for a binary.

    This is the read query for a system, and it mirrors ``MPRester.get_entries_in_chemsys``
    exactly: that call returns the entries of the chemsys AND of every sub-chemsys, which
    is why a per-system cache of ``A-B-C`` holds all the ``A``, ``B``, ``C``, ``A-B``,
    ``A-C``, ``B-C`` entries too. Reconstructing the same list from the pool therefore means
    selecting the same 7 groups, and any narrower query would return a smaller entry set
    than the fetch did.
    """
    unique = sorted({str(el) for el in elements})
    subsets = []
    for mask in range(1, 1 << len(unique)):
        subsets.append("-".join(el for i, el in enumerate(unique) if mask >> i & 1))
    return sorted(subsets)


def system_elements(sys_name: str) -> list[str]:
    """The element symbols spanned by a system name, sorted. ``[]`` when it is not one.

    Compound components are expanded (``'CuMg-Mg'`` -> ``['Cu', 'Mg']``), which is exactly
    what ``api._get_dft_entries_from_components`` does before querying: the chemsys query is
    expressed in ELEMENTS, never in components.
    """
    if not looks_like_system_name(sys_name):
        return []
    found = set()
    for part in sys_name.split("-"):
        for match in _FORMULA_TOKEN_RE.finditer(part):
            found.add(match.group(1))
    return sorted(found)


def entry_chemsys(entry_dict: dict) -> str:
    """The chemsys key of one entry dict, from its ``composition`` block.

    Zero-amount elements are dropped: a composition that carries an explicit ``0.0`` for an
    element would otherwise land in a chemsys the entry does not belong to, and would then
    be invisible to the narrower query that should have found it.
    """
    composition = entry_dict.get("composition") or {}
    return chemsys_key(el for el, amount in composition.items() if amount)


def entry_pool_id(entry_dict: dict) -> str | None:
    """The pool's primary key for one entry, or ``None`` when it has no usable identity.

    ``entry_id`` is a plain string in classic caches (``'mp-1234-GGA'``) and a DICT in
    new-API ones (``{'identifier': 'mp-aaaaaaeu', ...}``); both are keyed here, the dict by
    its canonical json so two spellings of the same id collide as they should. ``None``
    means the entry cannot be pooled at all -- a caller must report it rather than invent a
    key, because a synthesized key would make two different entries look like one.
    """
    raw = entry_dict.get("entry_id")
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, sort_keys=True)
    return str(raw)


def canonical_entry_list(entries) -> list[dict]:
    """``entries`` in the pool's canonical order: by pool id, then by serialized payload.

    A pooled read has no natural order (rows come back in whatever order the index yields),
    so both the writer's recorded sha256 and the reader's reconstruction sort through here.
    The payload tiebreak keeps the order total even for the pathological case of two
    entries sharing an id.
    """
    return sorted(entries, key=lambda e: (entry_pool_id(e) or "", json.dumps(e, sort_keys=True)))


def entry_list_sha256(entries) -> str:
    """sha256 of a canonicalized entry list — the reconstruction proof for one system.

    Recorded per system in ``entry_pool_systems`` at build time and recomputed from the
    POOLED read, so "the pool reproduces this system" is checked against a value written
    before the pool was queried, not against the query's own output.
    """
    payload = json.dumps(canonical_entry_list(entries), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SqliteStoreError(RuntimeError):
    """The file handed to :class:`SqliteBackend` is not a usable gliquid cache store."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _gliquid_version() -> str:
    """The installed gliquid version, or ``'unknown'`` — recorded in ``meta`` only.

    Imported lazily: ``gliquid/__init__`` imports ``api`` which imports this module, so a
    module-level import of the package would be circular.
    """
    try:
        from importlib.metadata import version

        return version("gliquid")
    except Exception:  # pragma: no cover - version metadata absent in odd installs
        return "unknown"


def _read_only_uri(path: Path, immutable: bool) -> str:
    """``file:`` URI for a read-only open, correct for Windows drive letters and spaces."""
    uri = "file:" + urllib.request.pathname2url(str(path)) + "?mode=ro"
    return uri + "&immutable=1" if immutable else uri


class SqliteBackend:
    """The whole corpus in one SQLite file — the DISTRIBUTION format.

    **Read-mostly by design, not by accident.** ``write_json`` raises unless the store was
    opened ``writable=True``. SQLite admits exactly one writer, and a campaign fanned out
    over a process pool is a burst of concurrent cold-fetch writes -- which is why
    :func:`atomic_write_json` exists on the directory side. Pointed at one file those
    writers would serialize behind ``database is locked``. So: the directory tree
    stays the WORKING format, and this is the format you hand to someone else. Populate it
    with ``python -m gliquid.cache migrate``.

    Args:
        path: The store file.
        writable: Open for writing. Also required to create tables.
        create: Create the file and its schema if it is missing. Implies ``writable``.
        immutable: Add ``immutable=1`` to a read-only open, which lets SQLite skip locking
            entirely — the right setting for a shipped, never-modified store. ``None``
            (default) decides per open: immutable only when the store has no ``-wal`` /
            ``-journal`` sidecar, i.e. was closed cleanly. That matters because
            ``immutable=1`` makes SQLite IGNORE the write-ahead log, and doing so on a
            store with a live WAL would serve stale rows with no error at all.
        timeout: Seconds to wait on a locked database.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        writable: bool = False,
        create: bool = False,
        immutable: bool | None = None,
        timeout: float = 60.0,
    ):
        self._path = Path(path)
        self._writable = bool(writable or create)
        self._create = bool(create)
        self._immutable = immutable
        self._timeout = float(timeout)
        # One connection per (process, thread). sqlite3 connections are neither picklable
        # nor safe to share across a fork, and gliquid's drivers do both fork and thread.
        self._connections: dict[tuple[int, int], sqlite3.Connection] = {}
        self._lock = threading.Lock()
# Whether this store carries the pooled-entry tables. Memoized; invalidated by
# `ensure_entry_pool`.
        self._pool_present: bool | None = None
        # Same memoization, same invalidation rule, for the ML feature tables.
        self._ml_features_present: bool | None = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        mode = "rw" if self._writable else "ro"
        return f"SqliteBackend(path={str(self._path)!r}, mode={mode!r})"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def writable(self) -> bool:
        return self._writable

    # -- connection management -----------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        if self._writable:
            if not self._create and not self._path.exists():
                raise SqliteStoreError(
                    f"No SQLite cache store at {self._path}. Build one with "
                    f"`python -m gliquid.cache migrate --from <cache dir> --to {self._path}`."
                )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), timeout=self._timeout, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            if self._create:
                conn.executescript(_SCHEMA_SQL)
                if self._user_version(conn) == 0:
                    conn.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")
        else:
            if not self._path.is_file():
                raise SqliteStoreError(
                    f"No SQLite cache store at {self._path}. Point gliquid at one with "
                    f"gliquid.config.set_cache_dir('<store>.sqlite'), or build one with "
                    f"`python -m gliquid.cache migrate --from <cache dir> --to <store>.sqlite`."
                )
            immutable = self._immutable
            if immutable is None:
                sidecars = (
                    self._path.with_name(self._path.name + "-wal"),
                    self._path.with_name(self._path.name + "-journal"),
                )
                immutable = not any(s.exists() for s in sidecars)
            conn = sqlite3.connect(
                _read_only_uri(self._path, immutable), uri=True, timeout=self._timeout
            )
        self._check_store(conn)
        return conn

    @staticmethod
    def _user_version(conn: sqlite3.Connection) -> int:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def _check_store(self, conn: sqlite3.Connection) -> None:
        """Refuse a file that is not a gliquid store of a version this build understands."""
        try:
            version = self._user_version(conn)
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise SqliteStoreError(f"{self._path} is not a SQLite database ({exc}).") from exc
        if version == 0:
            conn.close()
            raise SqliteStoreError(
                f"{self._path} is a SQLite database but not a gliquid cache store "
                f"(PRAGMA user_version is 0, expected {SQLITE_SCHEMA_VERSION}). Build one "
                f"with `python -m gliquid.cache migrate`."
            )
        if version > SQLITE_SCHEMA_VERSION:
            conn.close()
            raise SqliteStoreError(
                f"{self._path} was written by a NEWER gliquid: its cache schema is version "
                f"{version} and this build understands up to {SQLITE_SCHEMA_VERSION}. "
                f"Reading it here would interpret its columns wrongly, so it is refused. "
                f"Upgrade gliquid, or re-migrate the source corpus with this version."
            )

    def _conn(self) -> sqlite3.Connection:
        token = (os.getpid(), threading.get_ident())
        conn = self._connections.get(token)
        if conn is None:
            with self._lock:
                conn = self._connections.get(token)
                if conn is None:
                    conn = self._open()
                    self._connections[token] = conn
        return conn

    def close(self) -> None:
        """Close every connection this backend opened. Idempotent."""
        with self._lock:
            for conn in self._connections.values():
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._connections.clear()

    @contextlib.contextmanager
    def bulk_write(self) -> Iterator[None]:
        """One explicit transaction around many :meth:`write_json` calls.

        Autocommit (one transaction per row) is correct but pays a WAL frame flush per
        record; a migration writes tens of thousands.
        """
        self._require_writable("Bulk writing")
        conn = self._conn()
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def vacuum(self) -> None:
        """Rebuild the file compactly. For a store about to be SHIPPED, not for a working one.

        Worth ~1% on a ``migrate``d store (spec 04 measured 20 KB of 45 MiB) but ~10% on an
        incrementally built pool, where rows arrive interleaved across chemsystems and leave
        partially filled pages behind. Must run outside a transaction, so it is a method
        rather than something ``bulk_write`` could do.
        """
        self._require_writable("Vacuuming the store")
        self._conn().execute("VACUUM")

    # -- meta ----------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self._require_writable("Writing store metadata")
        self._conn().execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def meta(self) -> dict[str, str]:
        return dict(self._conn().execute("SELECT key, value FROM meta").fetchall())

    # -- the CacheBackend protocol -------------------------------------------------

    @property
    def capabilities(self) -> frozenset[str]:
        """No ``'paths'``: a record here is a row, and no filesystem path names it."""
        caps = {"read", "variants"}
        if self._writable:
            caps.add("write")
        return frozenset(caps)

    def locate(self, key: CacheKey) -> str:
        """``<store>#<the filename this record would have had>`` — for messages only."""
        return f"{self._path}#{record_filename(key)}"

    def exists(self, key: CacheKey) -> bool:
        if key.kind == KIND_DFT_ENTRIES:
            sql = "SELECT 1 FROM dft_entries WHERE sys_name = ? AND dft_type = ?"
        elif key.kind == KIND_MPDS:
            sql = "SELECT 1 FROM mpds_diagrams WHERE sys_name = ? AND variant = ?"
        else:
            raise ValueError(f"Unknown cache kind {key.kind!r}. Must be one of {list(_KINDS)}.")
        if self._conn().execute(sql, (key.sys_name, key.variant)).fetchone() is not None:
            return True
# A record the pool covers must report present here: `get_dft_convexhull` decides
# whether to fetch on this answer alone.
        return key.kind == KIND_DFT_ENTRIES and self.pool_covers(key.sys_name, key.variant)

    def read_json(self, key: CacheKey) -> Any:
        conn = self._conn()
        if key.kind == KIND_DFT_ENTRIES:
            row = conn.execute(
                "SELECT payload FROM dft_entries WHERE sys_name = ? AND dft_type = ?",
                (key.sys_name, key.variant),
            ).fetchone()
            if row is not None:
                return json.loads(zlib.decompress(row[0]))
            pooled = self.pooled_record(key)
            if pooled is not None:
                return pooled
            raise FileNotFoundError(f"No cached record {self.locate(key)}")
        if key.kind == KIND_MPDS:
            row = conn.execute(
                "SELECT mode, payload, stitched, regions, header FROM mpds_diagrams "
                "WHERE sys_name = ? AND variant = ?",
                (key.sys_name, key.variant),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"No cached record {self.locate(key)}")
            mode, payload, stitched, regions, header = row
            if mode == MPDS_MODE_FULL and payload is not None:
                return json.loads(zlib.decompress(payload))
            if mode == MPDS_MODE_LEAN:
                # Reassembled into the SAME dict shape a full record has, plus the reserved
                # block -- see gliquid.mpds for why this is a key and not a class.
                record = json.loads(header)
                record[LEAN_BLOCK_KEY] = {
                    "mode": MPDS_MODE_LEAN,
                    "schema": LEAN_SCHEMA,
                    "stitched": None if stitched is None else json.loads(zlib.decompress(stitched)),
                    "covered": None if regions is None else json.loads(regions),
                }
                return record
            raise CacheModeError(
                f"{self.locate(key)} is stored in MPDS mode {mode!r}, which this build "
                f"of gliquid cannot reconstruct a record from. Use a store migrated with "
                f"--mpds-mode full or --mpds-mode lean."
            )
        raise ValueError(f"Unknown cache kind {key.kind!r}. Must be one of {list(_KINDS)}.")

    def _require_writable(self, what: str) -> None:
        if self._writable:
            return
        raise CacheModeError(
            f"{what} is not possible: the SQLite cache store at {self._path} is open "
            f"READ-ONLY, which is the default and the design. SQLite admits one writer and "
            f"gliquid campaigns fan out over a ProcessPoolExecutor, so the single-file "
            f"store is the DISTRIBUTION format, not the working one. To change its "
            f"contents, rebuild it: `python -m gliquid.cache migrate --from <cache dir> "
            f"--to {self._path}`. To run something that writes cold-fetched records, point "
            f"gliquid at a directory store instead: "
            f"gliquid.config.set_cache_dir('/path/to/cache/dir')."
        )

    def write_json(self, key: CacheKey, payload: Any) -> None:
        """Store ``payload`` under ``key``.

        An MPDS payload carrying the reserved ``_gliquid`` lean block is stored **as a lean
        row** — header plus stitched curve plus regions, no ``payload`` blob — so that
        ``write_json`` → ``read_json`` round-trips whatever it was handed. The alternative,
        compressing the lean dict whole into the ``payload`` column, would leave a row
        labelled ``mode='full'`` that has no ``shapes``: the one state the whole contract
        exists to make impossible.
        """
        self._require_writable(f"Writing {record_filename(key)}")
        if key.kind == KIND_MPDS and _is_lean_record(payload):
            self._write_lean_mpds(key, payload)
            return
        blob = json.dumps(payload).encode("utf-8")
        compressed = zlib.compress(blob, _ZLIB_LEVEL)
        now = _utc_now()
        conn = self._conn()
        if key.kind == KIND_DFT_ENTRIES:
            conn.execute(
                "INSERT INTO dft_entries "
                "(sys_name, dft_type, payload, n_entries, sha256, written_utc) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sys_name, dft_type) DO UPDATE SET "
                "payload = excluded.payload, n_entries = excluded.n_entries, "
                "sha256 = excluded.sha256, written_utc = excluded.written_utc",
                (
                    key.sys_name,
                    key.variant,
                    compressed,
                    len(payload) if isinstance(payload, (list, dict)) else 0,
                    hashlib.sha256(blob).hexdigest(),
                    now,
                ),
            )
        elif key.kind == KIND_MPDS:
            conn.execute(
                "INSERT INTO mpds_diagrams "
                "(sys_name, variant, mode, header, payload, stitched, regions, written_utc) "
                "VALUES (?, ?, 'full', ?, ?, NULL, NULL, ?) "
                "ON CONFLICT(sys_name, variant) DO UPDATE SET "
                "mode = excluded.mode, header = excluded.header, payload = excluded.payload, "
                "stitched = excluded.stitched, regions = excluded.regions, "
                "written_utc = excluded.written_utc",
                (
                    key.sys_name,
                    key.variant,
                    json.dumps(mpds_header(payload)),
                    compressed,
                    now,
                ),
            )
        else:
            raise ValueError(f"Unknown cache kind {key.kind!r}. Must be one of {list(_KINDS)}.")

    def _write_lean_mpds(self, key: CacheKey, payload: dict) -> None:
        """Store a lean MPDS record across the ``header`` / ``stitched`` / ``regions`` columns."""
        block = payload[LEAN_BLOCK_KEY]
        header = {k: v for k, v in payload.items() if k != LEAN_BLOCK_KEY}
        stitched = block.get("stitched")
        covered = block.get("covered")
        self._conn().execute(
            "INSERT INTO mpds_diagrams "
            "(sys_name, variant, mode, header, payload, stitched, regions, written_utc) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?, ?) "
            "ON CONFLICT(sys_name, variant) DO UPDATE SET "
            "mode = excluded.mode, header = excluded.header, payload = excluded.payload, "
            "stitched = excluded.stitched, regions = excluded.regions, "
            "written_utc = excluded.written_utc",
            (
                key.sys_name,
                key.variant,
                MPDS_MODE_LEAN,
                json.dumps(header),
                None
                if stitched is None
                else zlib.compress(json.dumps(stitched).encode("utf-8"), _ZLIB_LEVEL),
                None if covered is None else json.dumps(covered),
                _utc_now(),
            ),
        )

    def variants(self, sys_name: str, kind: str) -> list[str]:
        if kind == KIND_DFT_ENTRIES:
            sql = "SELECT dft_type FROM dft_entries WHERE sys_name = ?"
        elif kind == KIND_MPDS:
            sql = "SELECT variant FROM mpds_diagrams WHERE sys_name = ?"
        else:
            raise ValueError(f"Unknown cache kind {kind!r}. Must be one of {list(_KINDS)}.")
        found = {row[0] for row in self._conn().execute(sql, (sys_name,))}
        if kind == KIND_DFT_ENTRIES and self.has_entry_pool:
            # Kept in step with `exists`: a backend that answers True there and [] here
            # would be internally inconsistent, and the inconsistency would only surface in
            # whichever consumer happened to ask the other question.
            found.update(
                row[0]
                for row in self._conn().execute(
                    "SELECT dft_type FROM entry_pool_systems WHERE sys_name = ?", (sys_name,)
                )
            )
        return sorted(found)

    # -- store-wide queries (CLI / tooling) -----------------------------------------

    def keys(self) -> list[CacheKey]:
        """Every record in the store, sorted — for ``verify`` and ``info``."""
        conn = self._conn()
        found = [
            CacheKey(sys_name, KIND_DFT_ENTRIES, variant)
            for sys_name, variant in conn.execute("SELECT sys_name, dft_type FROM dft_entries")
        ]
        found += [
            CacheKey(sys_name, KIND_MPDS, variant)
            for sys_name, variant in conn.execute("SELECT sys_name, variant FROM mpds_diagrams")
        ]
        return sorted(found, key=lambda k: (k.sys_name, k.kind, k.variant))

    def stored_sha256(self, key: CacheKey) -> str | None:
        """The sha256 of a DFT record's UNCOMPRESSED bytes, as written. ``None`` otherwise."""
        if key.kind != KIND_DFT_ENTRIES:
            return None
        row = (
            self._conn()
            .execute(
                "SELECT sha256 FROM dft_entries WHERE sys_name = ? AND dft_type = ?",
                (key.sys_name, key.variant),
            )
            .fetchone()
        )
        return None if row is None else row[0]

    # -- the pooled entry store ------------------------------------------------------
    #
    # A fallback for `dft_entries`: a per-system blob, when one exists, always wins.

    def ensure_entry_pool(self) -> None:
        """Create the pooled-entry tables if this store does not have them yet.

        Explicit rather than part of the default schema so that ``has_entry_pool`` means
        "this store was built to carry a pool", not "this store was opened by a build of
        gliquid that knows what a pool is". A ``migrate``d store has no pool tables at all.
        """
        self._require_writable("Creating the pooled-entry tables")
        self._conn().executescript(_ENTRY_POOL_SQL)
        self._pool_present = None

    @property
    def has_entry_pool(self) -> bool:
        """Whether this store carries the pooled-entry tables."""
        if self._pool_present is None:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
                "('entry_pool', 'entry_pool_systems', 'entry_pool_meta')"
            ).fetchone()
            self._pool_present = row[0] == 3
        return self._pool_present

    def set_pool_meta(self, key: str, value: str) -> None:
        self._require_writable("Writing pool metadata")
        self._conn().execute(
            "INSERT INTO entry_pool_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def pool_meta(self) -> dict[str, str]:
        """``entry_pool_meta`` as a dict — ``{}`` when this store carries no pool."""
        if not self.has_entry_pool:
            return {}
        return dict(self._conn().execute("SELECT key, value FROM entry_pool_meta").fetchall())

    def write_pool_entries(self, entries) -> tuple[int, list[str], list[dict]]:
        """Add ``entries`` to the pool. Returns ``(written, drifted_ids, unkeyable)``.

        **Drift is reported, never resolved.** When an ``entry_id`` is already present with
        a DIFFERENT payload the existing row is kept and the id is returned, because there
        is no principled way to choose between two payloads the Materials Project served
        for the same id at different times -- that is precisely why the binary corpus is not
        pooled. A builder that ignores this return value has re-created the failure the
        drift measurement ruled out.

        ``unkeyable`` carries entries with no usable ``entry_id``; they are NOT stored,
        because a synthesized key would merge distinct entries into one row.
        """
        self._require_writable("Writing pooled entries")
        conn = self._conn()
        keyed: list[tuple[str, str, dict]] = []
        unkeyable: list[dict] = []
        for entry in entries:
            pool_id = entry_pool_id(entry)
            if pool_id is None:
                unkeyable.append(entry)
                continue
            keyed.append((pool_id, json.dumps(entry, sort_keys=True), entry))

        existing: dict[str, str] = {}
        ids = [pool_id for pool_id, _, _ in keyed]
        for start in range(0, len(ids), 400):
            batch = ids[start : start + 400]
            placeholders = ",".join("?" * len(batch))
            for pool_id, blob in conn.execute(
                f"SELECT entry_id, payload FROM entry_pool WHERE entry_id IN ({placeholders})",
                batch,
            ):
                existing[pool_id] = zlib.decompress(blob).decode("utf-8")

        written = 0
        drifted: list[str] = []
        for pool_id, canonical, entry in keyed:
            previous = existing.get(pool_id)
            if previous is not None:
                if previous != canonical:
                    drifted.append(pool_id)
                continue
            conn.execute(
                "INSERT INTO entry_pool (entry_id, chemsys, n_elements, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    pool_id,
                    entry_chemsys(entry),
                    len([el for el, amount in (entry.get("composition") or {}).items() if amount]),
                    zlib.compress(canonical.encode("utf-8"), _ZLIB_LEVEL),
                ),
            )
            existing[pool_id] = canonical
            written += 1
        return written, drifted, unkeyable

    def record_pool_system(
        self,
        sys_name: str,
        dft_type: str,
        entries,
        *,
        mp_database_version: str = "",
        fetched_utc: str | None = None,
        source: str = "",
    ) -> None:
        """Record that ``sys_name`` is COVERED by the pool, with its reconstruction sha256.

        Coverage is a fact the pool cannot infer from its own rows -- see the comment on
        ``entry_pool_systems``. Call this only once every entry of the system is stored.
        """
        self._require_writable("Recording pool coverage")
        self._conn().execute(
            "INSERT INTO entry_pool_systems "
            "(sys_name, dft_type, chemsys, n_entries, sha256, mp_database_version, "
            " fetched_utc, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(sys_name, dft_type) DO UPDATE SET "
            "chemsys = excluded.chemsys, n_entries = excluded.n_entries, "
            "sha256 = excluded.sha256, mp_database_version = excluded.mp_database_version, "
            "fetched_utc = excluded.fetched_utc, source = excluded.source",
            (
                sys_name,
                dft_type,
                chemsys_key(system_elements(sys_name)),
                len(entries),
                entry_list_sha256(entries),
                mp_database_version,
                fetched_utc or _utc_now(),
                source,
            ),
        )

    def pool_covers(self, sys_name: str, dft_type: str) -> bool:
        """Whether the pool was BUILT for this system — not merely whether rows exist."""
        if not self.has_entry_pool:
            return False
        row = self._conn().execute(
            "SELECT 1 FROM entry_pool_systems WHERE sys_name = ? AND dft_type = ?",
            (sys_name, dft_type),
        ).fetchone()
        return row is not None

    def pool_systems(self) -> list[tuple]:
        """``(sys_name, dft_type, n_entries, sha256, fetched_utc)`` per covered system."""
        if not self.has_entry_pool:
            return []
        return list(
            self._conn().execute(
                "SELECT sys_name, dft_type, n_entries, sha256, fetched_utc "
                "FROM entry_pool_systems ORDER BY sys_name, dft_type"
            )
        )

    def read_pool_entries(self, elements) -> list[dict]:
        """Every pooled entry in the chemsys spanned by ``elements``, canonically ordered.

        ``chemsys IN (<the non-empty subsets>)`` -- one indexed lookup per subset group and
        no join, which is why the pool needs no per-system rows at all. Ordering comes from
        :func:`canonical_entry_list` rather than from the index, so two reads of the same
        system are identical lists and not merely equal sets.
        """
        if not self.has_entry_pool:
            return []
        subsets = chemsys_subsets(elements)
        if not subsets:
            return []
        placeholders = ",".join("?" * len(subsets))
        rows = self._conn().execute(
            f"SELECT payload FROM entry_pool WHERE chemsys IN ({placeholders})", subsets
        )
        return canonical_entry_list(json.loads(zlib.decompress(row[0])) for row in rows)

    def pooled_record(self, key: CacheKey) -> list[dict] | None:
        """The pooled reconstruction of a DFT record, or ``None`` when the pool has none."""
        if key.kind != KIND_DFT_ENTRIES or not self.pool_covers(key.sys_name, key.variant):
            return None
        elements = system_elements(key.sys_name)
        if not elements:  # pragma: no cover - coverage rows are only ever written for systems
            return None
        return self.read_pool_entries(elements)

    def pool_stats(self) -> dict:
        """Row/byte counts for ``info``. ``{}`` when this store carries no pool."""
        if not self.has_entry_pool:
            return {}
        conn = self._conn()
        entries, chemsystems, payload = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT chemsys), COALESCE(SUM(LENGTH(payload)), 0) "
            "FROM entry_pool"
        ).fetchone()
        by_arity = dict(
            conn.execute("SELECT n_elements, COUNT(*) FROM entry_pool GROUP BY n_elements")
        )
        covered, pooled_rows = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(n_entries), 0) FROM entry_pool_systems"
        ).fetchone()
        return {
            "entries": entries,
            "chemsystems": chemsystems,
            "payload_bytes": payload,
            "by_n_elements": by_arity,
            "systems_covered": covered,
            "entry_rows_reconstructed": pooled_rows,
        }

    # -- the ML feature corpus --------------------------------------------------------
    #
    # Model input features, read by ``gliquid.production_model_runner.get_rows_for_system``.

    def ensure_ml_features(self) -> None:
        """Create the ML feature tables if this store does not have them yet.

        Explicit, exactly as :meth:`ensure_entry_pool` is: ``has_ml_features`` must mean
        "this store was built to carry the feature corpus", not "this store was opened by a
        build of gliquid that knows what one is".
        """
        self._require_writable("Creating the ML feature tables")
        self._conn().executescript(_ML_FEATURES_SQL)
        self._ml_features_present = None

    @property
    def has_ml_features(self) -> bool:
        """Whether this store carries the ML feature tables."""
        if self._ml_features_present is None:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN "
                "('ml_features', 'ml_feature_columns')"
            ).fetchone()
            self._ml_features_present = row[0] == 2
        return self._ml_features_present

    def write_ml_features(self, frame: str, bundle_id: str, columns, rows) -> int:
        """Store one frame of the feature corpus. Returns the number of rows written.

        ``rows`` is an iterable of ``(system, sequence-of-floats)``; every sequence must be
        ``len(columns)`` long, and a row that is not is refused rather than truncated -- a
        short row would decode as a valid but WRONG feature vector, which is the one failure
        mode a raw-float encoding has and json does not.
        """
        self._require_writable("Writing ML features")
        if frame not in ML_FRAMES:
            raise ValueError(f"Unknown ML feature frame {frame!r}. Must be one of {ML_FRAMES}.")
        columns = [str(c) for c in columns]
        width = len(columns)
        if not width:
            raise ValueError(f"Frame {frame!r} was given no feature columns.")
        packer = struct.Struct(f"<{width}d")
        conn = self._conn()
        conn.execute(
            "INSERT INTO ml_feature_columns (frame, bundle_id, columns) VALUES (?, ?, ?) "
            "ON CONFLICT(frame) DO UPDATE SET bundle_id = excluded.bundle_id, "
            "columns = excluded.columns",
            (frame, str(bundle_id), json.dumps(columns)),
        )
        written = 0
        for system, values in rows:
            values = list(values)
            if len(values) != width:
                raise ValueError(
                    f"System {system!r} in frame {frame!r} has {len(values)} feature values "
                    f"but the frame declares {width} columns."
                )
            conn.execute(
                "INSERT INTO ml_features (frame, system, features) VALUES (?, ?, ?) "
                "ON CONFLICT(frame, system) DO UPDATE SET features = excluded.features",
                (frame, str(system), packer.pack(*(float(v) for v in values))),
            )
            written += 1
        return written

    def ml_feature_columns(self, frame: str) -> list[str] | None:
        """The ordered column names of ``frame``, or ``None`` when the store has none."""
        if not self.has_ml_features:
            return None
        row = self._conn().execute(
            "SELECT columns FROM ml_feature_columns WHERE frame = ?", (frame,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def ml_features_bundle_id(self, frame: str) -> str | None:
        """The model bundle ``frame``'s columns were exported for. Diagnostics only."""
        if not self.has_ml_features:
            return None
        row = self._conn().execute(
            "SELECT bundle_id FROM ml_feature_columns WHERE frame = ?", (frame,)
        ).fetchone()
        return None if row is None else row[0]

    def ml_feature_rows(self, frame: str) -> list[tuple[str, list[float]]]:
        """Every ``(system, values)`` pair of ``frame``, ordered by system name.

        The whole frame rather than a point read on purpose: the consumer
        (``ProductionModelRunner``) materializes both frames once into DataFrames and then
        answers thousands of ``predict_system`` calls from memory, and its beeswarm plots
        need the full frame anyway. A per-system query would pay a round trip per call for
        a table that is 2.6 MB whole.
        """
        if not self.has_ml_features:
            return []
        columns = self.ml_feature_columns(frame)
        if not columns:
            return []
        unpacker = struct.Struct(f"<{len(columns)}d")
        return [
            (system, list(unpacker.unpack(blob)))
            for system, blob in self._conn().execute(
                "SELECT system, features FROM ml_features WHERE frame = ? ORDER BY system",
                (frame,),
            )
        ]

    def ml_feature_stats(self) -> dict:
        """Row/byte counts per frame for ``info``. ``{}`` when this store carries none."""
        if not self.has_ml_features:
            return {}
        conn = self._conn()
        widths = {
            frame: len(json.loads(columns))
            for frame, columns in conn.execute("SELECT frame, columns FROM ml_feature_columns")
        }
        if not widths:
            return {}
        frames = {}
        for frame, rows, payload in conn.execute(
            "SELECT frame, COUNT(*), COALESCE(SUM(LENGTH(features)), 0) FROM ml_features "
            "GROUP BY frame"
        ):
            frames[frame] = {
                "rows": rows,
                "columns": widths.get(frame, 0),
                "payload_bytes": payload,
                "bundle_id": self.ml_features_bundle_id(frame) or "",
            }
        for frame, width in widths.items():  # a declared frame with zero rows still shows
            frames.setdefault(
                frame,
                {
                    "rows": 0,
                    "columns": width,
                    "payload_bytes": 0,
                    "bundle_id": self.ml_features_bundle_id(frame) or "",
                },
            )
        return frames


def _drop_null_frame(header: dict) -> dict:
    """A lean header made comparable to ``mpds_header`` of the record it came from.

    ``gliquid.mpds.lean_record`` writes ``chemical_elements: null`` when the source json
    carries no frame block, because ``mpds_frame_matches`` treats an ABSENT frame as
    *matching* the caller and a silently mis-mirrored construction is the bug that would
    cause. ``mpds_header`` simply omits the key in that case, so the explicit null is the
    one intended difference between the two and is dropped before comparison.
    """
    if header.get("chemical_elements") is None:
        return {k: v for k, v in header.items() if k != "chemical_elements"}
    return header


def _is_lean_record(payload: Any) -> bool:
    """Whether ``payload`` is a lean MPDS record — mirrors ``gliquid.mpds.record_mode``."""
    if not isinstance(payload, dict):
        return False
    block = payload.get(LEAN_BLOCK_KEY)
    return isinstance(block, dict) and block.get("mode") == MPDS_MODE_LEAN


def mpds_header(mpds_json: Any) -> dict:
    """The addressable header columns of an MPDS record.

    Tolerates the ``{"reference": None}`` placeholder that ``load_mpds_data`` caches when a
    system has no digitized diagram — that IS a record, and dropping it would make the
    system look uncached and trigger a network fetch on every read.
    """
    if not isinstance(mpds_json, dict):
        return {}
    reference = mpds_json.get("reference")
    header: dict[str, Any] = {}
    for name in _MPDS_HEADER_FIELDS:
        if name in mpds_json:
            header[name] = mpds_json[name]
        elif isinstance(reference, dict) and name in reference:
            header[name] = reference[name]
    return header


#: The store ``resolve_backend(None)`` hands out. Reads ``config`` at call time rather
#: than snapshotting it, so a mid-session ``set_cache_dir`` is observed.
_GLOBAL_DIRECTORY_BACKEND = DirectoryBackend()

_BACKEND_METHODS = ("exists", "read_json", "write_json", "variants", "locate")


def store_label(backend: CacheBackend, sys_name: str) -> str:
    """A human name for where ``sys_name``'s records live — for log lines only.

    Directory stores answer with the system directory, which is what the pre-seam log
    lines printed; other stores fall back to a record handle.
    """
    sys_location = getattr(backend, "sys_location", None)
    if sys_location is not None:
        return str(sys_location(sys_name))
    return str(backend.locate(CacheKey(sys_name, KIND_MPDS, "")))


#: Read-only ``SqliteBackend``s by store path, opened once per process. Keyed by path so
#: a mid-session ``set_cache_dir`` to a different store is observed.
_SQLITE_BACKENDS: dict[str, SqliteBackend] = {}


def sqlite_backend_for(path: str | os.PathLike) -> SqliteBackend:
    """The process-wide read-only :class:`SqliteBackend` for ``path``, opening it once."""
    resolved = str(Path(path))
    backend = _SQLITE_BACKENDS.get(resolved)
    if backend is None:
        backend = SqliteBackend(resolved)
        _SQLITE_BACKENDS[resolved] = backend
    return backend


def close_sqlite_backends() -> None:
    """Close and forget every cached read-only store. For tests and teardown."""
    for backend in list(_SQLITE_BACKENDS.values()):
        backend.close()
    _SQLITE_BACKENDS.clear()


def resolve_backend(override: CacheBackend | str | os.PathLike | None = None) -> CacheBackend:
    """The store to use for one call.

    Args:
        override: ``None`` uses the configured global store. A ``str``/``Path`` builds a
            :class:`DirectoryBackend` with a **flat** layout inside it — the historical
            meaning of an explicit ``data_dir=`` argument, which is why it does not consult
            ``config.dir_structure`` — unless the path names a single FILE store
            (``.sqlite`` / ``.sqlite3`` / ``.db``), which opens read-only.
            An already-constructed backend is used as-is.
    """
    if override is None:
        if config.cache_mode == "sqlite":
            store = config.require_cache_dir("Reading the cache")
            if store.is_dir():
                raise CacheModeError(
                    f"config.cache_mode is 'sqlite' but config.cache_dir ({store}) is a "
                    f"DIRECTORY. Set them together with "
                    f"gliquid.config.set_cache_dir(...), which infers the mode from the "
                    f"shape of the path, rather than assigning either one on its own."
                )
            return sqlite_backend_for(store)
        if config.cache_mode != "directory":
            raise CacheModeError(
                f"Unknown config.cache_mode {config.cache_mode!r}; expected 'directory' "
                f"or 'sqlite'."
            )
        return _GLOBAL_DIRECTORY_BACKEND
    if isinstance(override, (str, os.PathLike)):
        # One rule for "is this a single-file store", shared with config.set_cache_dir, so
        # an explicit data_dir= and a configured cache_dir cannot disagree about a path.
        if config._is_file_store(Path(override)):
            return sqlite_backend_for(override)
        return DirectoryBackend(override)
    if all(hasattr(override, name) for name in _BACKEND_METHODS):
        return override
    raise TypeError(
        f"Cannot use {override!r} as a gliquid cache store: expected a path, a CacheBackend, "
        f"or None."
    )


# ---------------------------------------------------------------------------------------
# CLI -- ``python -m gliquid.cache {migrate,verify,info}``
#
# Everything below writes to stdout; library code above this line reports through
# ``logger`` (exemption registered in tests/test_logging_boundary.py::EXEMPT).
# ---------------------------------------------------------------------------------------


def _emit(message: str = "") -> None:
    """Write one line of CLI output. The module's only print site."""
    print(message)  # documented logging exemption: CLI stdout is the product


def _human_bytes(count: float) -> str:
    if count < 1024:
        return f"{int(count):,} B"
    for unit in ("KB", "MB", "GB"):
        count /= 1024
        if count < 1024 or unit == "GB":
            return f"{count:,.1f} {unit}"
    raise AssertionError("unreachable")  # pragma: no cover


def _json_equivalent(left: Any, right: Any) -> bool:
    """Deep equality that treats two NaNs as equal.

    ``json`` round-trips a non-standard ``NaN`` literal happily in both directions, but
    ``float('nan') != float('nan')``, so a plain ``==`` would report a phantom mismatch on
    a record that is in fact byte-identical.
    """
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (left != left and right != right)
    if isinstance(left, dict) and isinstance(right, dict):
        return len(left) == len(right) and all(
            key in right and _json_equivalent(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) != isinstance(right, bool):
        return False  # json distinguishes true from 1; Python's == does not
    return left == right


def _report_scan(scan: DirectoryScan) -> None:
    _emit(f"source               : {scan.root}  (--dir-structure {scan.dir_structure})")
    _emit(f"cache records        : {len(scan.records):,}")
    _emit(f"ignored (not records): {len(scan.ignored):,}")
    for path in scan.ignored[:10]:
        _emit(f"    - {path.name}")
    if len(scan.ignored) > 10:
        _emit(f"    ... and {len(scan.ignored) - 10:,} more")


def _explain_empty_scan(scan: DirectoryScan) -> str | None:
    """A message explaining a zero-record scan, or ``None`` when there ARE records.

    A migration that quietly produces an empty store is the failure this whole tool exists
    to avoid, and the overwhelmingly likely cause is the wrong ``--dir-structure``.
    """
    if scan.records:
        return None
    other = "nested" if scan.dir_structure == "flat" else "flat"
    try:
        alternative = len(scan_directory_store(scan.root, other).records)
    except OSError:  # pragma: no cover - the root was already walked once
        alternative = 0
    hint = (
        f" --dir-structure {other} would find {alternative:,}; the corpus is arranged that way."
        if alternative
        else ""
    )
    return (
        f"No cache records found under {scan.root} with --dir-structure {scan.dir_structure}.{hint}"
    )


def migrate(
    source: str | os.PathLike,
    dest: str | os.PathLike,
    *,
    dir_structure: str = "flat",
    mpds_mode: str = "full",
    skip_unparseable: bool = False,
    overwrite: bool = False,
) -> int:
    """Build a single-file store from a directory corpus. Returns a process exit code.

    Unparseable source files are **reported and refused**, not skipped: a corpus holding a
    truncated cache file is a corpus with a silently wrong answer in it, and a migration
    that drops it only moves the silence somewhere new. ``skip_unparseable`` is the
    explicit opt-out, and it still reports every file it passed over.

    ``mpds_mode='lean'`` reduces every MPDS record to its header plus the PRE-fill stitched
    liquidus (``gliquid.mpds.lean_record``), which is ~93-97% smaller and is all a fit or a
    liquidus plot reads. Every OTHER MPDS consumer raises on such a record rather than
    answering from a missing block — see ``gliquid.mpds.record_mode``.
    """
    source, dest = Path(source), Path(dest)
    if mpds_mode not in MPDS_MODES:
        _emit(
            f"ERROR: --mpds-mode {mpds_mode!r} is not a known mode. "
            f"Choose one of {', '.join(MPDS_MODES)}."
        )
        return 2
    if dest.exists() and not overwrite:
        _emit(f"ERROR: {dest} already exists. Pass --overwrite to replace it.")
        return 2

    scan = scan_directory_store(source, dir_structure)
    _report_scan(scan)
    empty = _explain_empty_scan(scan)
    if empty is not None:
        _emit(f"ERROR: {empty}")
        return 1

    # Built under a .partial name and moved into place only on success, so a refused
    # migration leaves no store at all rather than a plausible-looking incomplete one.
    partial = dest.with_name(dest.name + ".partial")
    for stale in (partial, *(partial.with_name(partial.name + s) for s in ("-wal", "-shm"))):
        with contextlib.suppress(OSError):
            stale.unlink()

    failures: list[tuple[Path, str]] = []
    written = {KIND_DFT_ENTRIES: 0, KIND_MPDS: 0}
    source_bytes = 0
    # Lazy: gliquid.mpds imports this module, so a module-level import would be circular.
    reduce = None
    if mpds_mode == MPDS_MODE_LEAN:
        from gliquid.mpds import lean_record as reduce
    backend = SqliteBackend(partial, create=True)
    try:
        backend.set_meta("schema_version", str(SQLITE_SCHEMA_VERSION))
        backend.set_meta("mpds_mode", mpds_mode)
        backend.set_meta("dft_codec", DFT_CODEC)
        backend.set_meta("gliquid_version_written", _gliquid_version())
        backend.set_meta("created_utc", _utc_now())
        backend.set_meta("source_store", str(source.resolve()))
        backend.set_meta("source_dir_structure", dir_structure)

        chunk = 250
        total = len(scan.records)
        for start in range(0, total, chunk):
            with backend.bulk_write():
                for key, path in scan.records[start : start + chunk]:
                    try:
                        raw = path.read_bytes()
                        payload = json.loads(raw)
                    except (OSError, ValueError, UnicodeDecodeError) as exc:
                        failures.append((path, f"{type(exc).__name__}: {exc}"))
                        continue
                    source_bytes += len(raw)
                    if key.kind == KIND_MPDS and mpds_mode == MPDS_MODE_LEAN:
                        try:
                            payload = reduce(payload)
                        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                            failures.append((path, f"lean reduction failed: {type(exc).__name__}: {exc}"))
                            continue
                    backend.write_json(key, payload)
                    written[key.kind] += 1
            done = min(start + chunk, total)
            if done % 2500 == 0 or done == total:
                _emit(f"    migrated {done:,}/{total:,} records")
        backend.set_meta("source_bytes", str(source_bytes))
        backend.set_meta("record_count", str(sum(written.values())))
    finally:
        backend.close()

    if failures:
        _emit("")
        _emit(f"{len(failures):,} source file(s) could not be parsed as JSON:")
        for path, why in failures:
            _emit(f"    {path}")
            _emit(f"        {why}")

    if failures and not skip_unparseable:
        with contextlib.suppress(OSError):
            partial.unlink()
        _emit("")
        _emit(
            f"REFUSED: no store was written to {dest}. Fix or remove the file(s) above, or "
            f"re-run with --skip-unparseable to build a store that deliberately omits them."
        )
        return 1

    os.replace(partial, dest)
    size = dest.stat().st_size
    _emit("")
    _emit(f"wrote                : {dest}")
    _emit(f"  dft_entries        : {written[KIND_DFT_ENTRIES]:,} records")
    _emit(f"  mpds_diagrams      : {written[KIND_MPDS]:,} records  (mode={mpds_mode})")
    if failures:
        _emit(f"  SKIPPED            : {len(failures):,} unparseable source file(s), listed above")
    _emit(f"  source bytes       : {_human_bytes(source_bytes)}")
    _emit(f"  store bytes        : {_human_bytes(size)}")
    if size:
        _emit(f"  compression        : {source_bytes / size:.2f}x")
    return 0


def verify(
    directory: str | os.PathLike,
    sqlite_path: str | os.PathLike,
    *,
    dir_structure: str | None = None,
) -> int:
    """Compare a directory corpus against a single-file store, record by record.

    The comparison is on the PARSED PYTHON OBJECTS, which is the level the seam exchanges
    (see the module docstring). For DFT records the stored sha256 of the uncompressed bytes
    is re-derived as well, so losslessness is proved on the bytes and not only through an
    equality operator.

    Against a **lean** store, object equality is the wrong question — the reduction threw
    ``shapes`` away on purpose. MPDS records are checked on what a lean record CLAIMS to
    preserve instead: identical ``extract_digitized_liquidus`` output and identical
    ``liquidus_coverage`` metrics, plus identical header fields. That equivalence is the
    whole justification for the reduction, so it is checked here rather than asserted, and
    a lean store whose MPDS half was compared ZERO times fails.
    """
    directory, sqlite_path = Path(directory), Path(sqlite_path)
    backend = SqliteBackend(sqlite_path)
    try:
        meta = backend.meta()
        mpds_mode = meta.get("mpds_mode", MPDS_MODE_FULL)
        equivalence = None
        if mpds_mode == MPDS_MODE_LEAN:
            from gliquid.mpds import extract_digitized_liquidus, liquidus_coverage

            def equivalence(full, lean):  # noqa: F811 - defined only in lean mode
                """(same_liquidus, same_coverage) between a full record and its reduction."""
                return (
                    _json_equivalent(
                        list(extract_digitized_liquidus(full)),
                        list(extract_digitized_liquidus(lean)),
                    ),
                    _json_equivalent(liquidus_coverage(full), liquidus_coverage(lean)),
                )

        if dir_structure is None:
            dir_structure = meta.get("source_dir_structure", "flat")
        scan = scan_directory_store(directory, dir_structure)
        _report_scan(scan)
        _emit(f"sqlite               : {sqlite_path}")
        empty = _explain_empty_scan(scan)
        if empty is not None:
            _emit(f"ERROR: {empty}")
            return 1

        compared = 0
        lean_compared = 0
        mismatches: list[str] = []
        missing: list[str] = []
        sha_mismatches: list[str] = []
        unparseable: list[str] = []
        liquidus_divergences: list[str] = []
        coverage_divergences: list[str] = []
        for key, path in scan.records:
            try:
                expected = json.loads(path.read_bytes())
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                unparseable.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if not backend.exists(key):
                missing.append(backend.locate(key))
                continue
            compared += 1
            stored_record = backend.read_json(key)
            if equivalence is not None and key.kind == KIND_MPDS:
                lean_compared += 1
                same_liquidus, same_coverage = equivalence(expected, stored_record)
                if not same_liquidus:
                    liquidus_divergences.append(backend.locate(key))
                if not same_coverage:
                    coverage_divergences.append(backend.locate(key))
                header = {k: v for k, v in stored_record.items() if k != LEAN_BLOCK_KEY}
                if not _json_equivalent(mpds_header(expected), _drop_null_frame(header)):
                    mismatches.append(backend.locate(key))
            elif not _json_equivalent(expected, stored_record):
                mismatches.append(backend.locate(key))
            stored = backend.stored_sha256(key)
            if stored is not None:
                rederived = hashlib.sha256(json.dumps(expected).encode("utf-8")).hexdigest()
                if stored != rederived:
                    sha_mismatches.append(backend.locate(key))

        directory_keys = {key for key, _ in scan.records}
        extra = [backend.locate(k) for k in backend.keys() if k not in directory_keys]

        _emit("")
        _emit(f"keys compared        : {compared:,}")
        _emit(f"object mismatches    : {len(mismatches):,}")
        _emit(f"dft sha256 mismatches: {len(sha_mismatches):,}")
        _emit(f"missing from sqlite  : {len(missing):,}")
        _emit(f"extra in sqlite      : {len(extra):,}")
        _emit(f"unparseable in source: {len(unparseable):,}")
        if equivalence is not None:
            _emit(f"mpds mode            : {mpds_mode}")
            _emit(f"lean mpds compared   : {lean_compared:,}")
            _emit(f"liquidus divergences : {len(liquidus_divergences):,}")
            _emit(f"coverage divergences : {len(coverage_divergences):,}")
        for label, items in (
            ("MISMATCH", mismatches),
            ("SHA MISMATCH", sha_mismatches),
            ("MISSING", missing),
            ("EXTRA", extra),
            ("UNPARSEABLE", unparseable),
            ("LIQUIDUS DIVERGENCE", liquidus_divergences),
            ("COVERAGE DIVERGENCE", coverage_divergences),
        ):
            for item in items[:20]:
                _emit(f"    {label}: {item}")
            if len(items) > 20:
                _emit(f"    ... and {len(items) - 20:,} more {label}")

        bad = (
            mismatches
            or sha_mismatches
            or missing
            or extra
            or unparseable
            or liquidus_divergences
            or coverage_divergences
        )
        if compared == 0:
            _emit("FAIL: compared 0 keys -- a verify that checks nothing is not a pass.")
            return 1
        if equivalence is not None and lean_compared == 0:
            _emit(
                "FAIL: this store is mpds_mode=lean and 0 MPDS records were compared -- the "
                "lean/full equivalence this store rests on was not checked at all."
            )
            return 1
        if bad:
            _emit("FAIL")
        elif equivalence is not None:
            _emit(
                f"OK: {compared:,} records verified; {lean_compared:,} lean MPDS records "
                f"reproduce their full records' liquidus and coverage exactly."
            )
        else:
            _emit(f"OK: {compared:,} records round-tripped losslessly.")
        return 1 if bad else 0
    finally:
        backend.close()


def info(sqlite_path: str | os.PathLike) -> int:
    """Print what a single-file store contains."""
    sqlite_path = Path(sqlite_path)
    backend = SqliteBackend(sqlite_path)
    try:
        conn = backend._conn()
        _emit(f"store                : {sqlite_path}")
        _emit(f"file size            : {_human_bytes(sqlite_path.stat().st_size)}")
        _emit(f"user_version         : {SqliteBackend._user_version(conn)}")
        _emit("")
        _emit("meta")
        for key, value in sorted(backend.meta().items()):
            _emit(f"  {key:<24}: {value}")

        rows, systems, entries, payload = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sys_name), COALESCE(SUM(n_entries), 0), "
            "COALESCE(SUM(LENGTH(payload)), 0) FROM dft_entries"
        ).fetchone()
        _emit("")
        _emit("dft_entries")
        _emit(f"  records                 : {rows:,}")
        _emit(f"  distinct systems        : {systems:,}")
        _emit(f"  entry rows (n_entries)  : {entries:,}")
        _emit(f"  compressed payload      : {_human_bytes(payload)}")

        rows, systems, payload, header, stitched, regions = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sys_name), COALESCE(SUM(LENGTH(payload)), 0), "
            "COALESCE(SUM(LENGTH(header)), 0), COALESCE(SUM(LENGTH(stitched)), 0), "
            "COALESCE(SUM(LENGTH(regions)), 0) FROM mpds_diagrams"
        ).fetchone()
        _emit("")
        _emit("mpds_diagrams")
        _emit(f"  records                 : {rows:,}")
        _emit(f"  distinct systems        : {systems:,}")
        _emit(f"  compressed payload      : {_human_bytes(payload)}")
        _emit(f"  header text             : {_human_bytes(header)}")
        _emit(f"  stitched liquidus       : {_human_bytes(stitched)}")
        _emit(f"  covered regions         : {_human_bytes(regions)}")
        # The number spec 05 is measured on: everything the MPDS half of the store costs.
        _emit(f"  MPDS TOTAL              : {_human_bytes(payload + header + stitched + regions)}")
        for mode, count in conn.execute("SELECT mode, COUNT(*) FROM mpds_diagrams GROUP BY mode"):
            _emit(f"  mode {mode:<18} : {count:,}")
        indexless = conn.execute("SELECT COUNT(*) FROM mpds_diagrams WHERE variant = ''").fetchone()
        _emit(f"  indexless (<sys>.json)  : {indexless[0]:,}")

        # Printed only when the store actually carries a pool, so a `migrate`d store's
        # output is unchanged and the presence of this section is itself the signal.
        stats = backend.pool_stats()
        if stats:
            _emit("")
            _emit("entry_pool")
            _emit(f"  distinct entries        : {stats['entries']:,}")
            _emit(f"  distinct chemsystems    : {stats['chemsystems']:,}")
            _emit(f"  compressed payload      : {_human_bytes(stats['payload_bytes'])}")
            for arity, count in sorted(stats["by_n_elements"].items()):
                _emit(f"  {arity}-element entries      : {count:,}")
            _emit(f"  systems covered         : {stats['systems_covered']:,}")
            reconstructed = stats["entry_rows_reconstructed"]
            _emit(f"  entry rows reconstructed: {reconstructed:,}")
            if stats["entries"]:
                _emit(f"  dedup factor            : {reconstructed / stats['entries']:.2f}x")
            _emit("")
            _emit("entry_pool_meta")
            for key, value in sorted(backend.pool_meta().items()):
                shown = value if len(value) <= 96 else value[:93] + "..."
                _emit(f"  {key:<24}: {shown}")

        # Same rule as the pool: printed only when the store carries the tables, so its
        # presence in the output IS the signal that this store can serve `predict_system`.
        ml_stats = backend.ml_feature_stats()
        if ml_stats:
            _emit("")
            _emit("ml_features")
            total = 0
            for frame in sorted(ml_stats):
                stats = ml_stats[frame]
                total += stats["payload_bytes"]
                _emit(
                    f"  {frame:<24}: {stats['rows']:,} rows x {stats['columns']} columns  "
                    f"({_human_bytes(stats['payload_bytes'])}, bundle {stats['bundle_id']})"
                )
            _emit(f"  {'ML TOTAL':<24}: {_human_bytes(total)}")
        return 0
    finally:
        backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gliquid.cache",
        description=(
            "Build, check and inspect a single-file gliquid cache store. The directory "
            "tree stays the working format; the SQLite store is the distribution format "
            "and is opened read-only."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mig = sub.add_parser("migrate", help="build a SQLite store from a directory corpus")
    mig.add_argument("--from", dest="source", required=True, help="the directory corpus")
    mig.add_argument("--to", dest="dest", required=True, help="the .sqlite file to write")
    mig.add_argument(
        "--dir-structure",
        choices=("flat", "nested"),
        default="flat",
        help="how the source arranges records (default: flat)",
    )
    # Deliberately NOT argparse `choices`: every other error in this CLI is an `_emit`ed
    # "ERROR: ..." line on stdout with an exit code, and an argparse rejection would go to
    # stderr instead, which is the one place a caller piping this tool is not reading.
    mig.add_argument(
        "--mpds-mode",
        default=MPDS_MODE_FULL,
        help=(
            "MPDS record mode: 'full' keeps the whole json; 'lean' keeps the header and "
            "the pre-fill stitched liquidus only (~93-97%% smaller). A lean store fits and "
            "plots liquidi; every other MPDS consumer raises on it."
        ),
    )
    mig.add_argument(
        "--skip-unparseable",
        action="store_true",
        help="build the store anyway, omitting (but still reporting) unparseable sources",
    )
    mig.add_argument("--overwrite", action="store_true", help="replace an existing store")

    ver = sub.add_parser("verify", help="compare a directory corpus against a SQLite store")
    ver.add_argument("--directory", required=True)
    ver.add_argument("--sqlite", dest="sqlite_path", required=True)
    ver.add_argument(
        "--dir-structure",
        choices=("flat", "nested"),
        default=None,
        help="default: whatever the store records as its source layout",
    )

    nfo = sub.add_parser("info", help="describe a SQLite store")
    nfo.add_argument("sqlite_path", metavar="db")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "migrate":
            return migrate(
                args.source,
                args.dest,
                dir_structure=args.dir_structure,
                mpds_mode=args.mpds_mode,
                skip_unparseable=args.skip_unparseable,
                overwrite=args.overwrite,
            )
        if args.command == "verify":
            return verify(args.directory, args.sqlite_path, dir_structure=args.dir_structure)
        return info(args.sqlite_path)
    except (SqliteStoreError, CacheModeError, NotADirectoryError, ValueError) as exc:
        _emit(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
