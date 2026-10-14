"""Repo-root pytest configuration: one OpenMP thread, tracked-only collection, the DFT hull memo
and the optional single-file store.

One OpenMP thread. ``OMP_NUM_THREADS`` is set to 1 for the session (XGBoost and OpenBLAS in
every xdist worker and every fresh interpreter a test starts), before anything imports numpy
or xgboost. A value already in the environment is kept.

Tracked-only collection. Under ``tests/`` and ``tests_internal/`` only the ``test_*.py`` files
that git tracks are collected; an untracked or git-ignored one is skipped and listed in the
terminal summary. A file named on the command line is always collected. Outside a git work
tree, or without ``git`` on PATH, every file is collected.

Hull memo. For the session, ``gliquid.api.get_dft_convexhull`` hands back one shared hull per
key: the components in the order given, ``dft_type``, the content of the record the store
serves, and the spurious-structure blacklist file. Directory stores, SQLite stores and
overlays on either are memoised. A call with ``verbose``, ``include_imputed`` or
``inc_structure_data``, a call whose store cannot be resolved or holds no record, and any
other store go to the real function uncached. Each shared hull is fingerprinted (entry ids,
energies, corrections, ``data`` keys, the stable set) and re-checked when it is handed out and
at the teardown of every test that touched it, so a test that mutates one fails at its own
teardown. ``GLIQUID_TEST_NO_HULL_MEMO=1`` turns the memo off. The terminal summary reports
hits, misses and bypasses.

Single-file store. Set ``GLIQUID_TEST_SQLITE_STORE=<path>.sqlite`` and the whole suite runs
against a SQLite cache store instead of the ``cache/`` directory tree::

    python -m gliquid.cache migrate --from cache --to <path>.sqlite
    GLIQUID_TEST_SQLITE_STORE=<path>.sqlite python -m pytest tests tests_internal

Unset, the suite runs against the directory store.

``@pytest.mark.directory_only`` marks the tests that are ABOUT the directory backend
(filenames on disk, ``dir_structure``, the atomic-write path, the cold-fetch write path).
They are skipped under the swap. The marker means "this test needs a directory tree", never
"this test fails under sqlite" -- if a test fails under the swap for any other reason, that
is a real finding.
"""

import functools
import hashlib
import inspect
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

_SQLITE_STORE_ENV = "GLIQUID_TEST_SQLITE_STORE"
_NO_HULL_MEMO_ENV = "GLIQUID_TEST_NO_HULL_MEMO"
_HULL_MEMO_ENTRIES = 4000  # entries held across all memoised hulls; least recently used go first
_GATED_DIRS = ("tests", "tests_internal")
_UNTRACKED_KEY = "gliquid_untracked_test_files"
_HULL_MEMO_KEY = "gliquid_hull_memo_stats"
_tracked_key = pytest.StashKey[frozenset | None]()
_skipped_key = pytest.StashKey[set]()
_memo_key = pytest.StashKey["_HullMemo"]()
_memo_stats_key = pytest.StashKey[Counter]()


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _tracked_files(config) -> frozenset[str] | None:
    """Return the git-tracked files under the gated dirs, or None when git cannot say."""
    if _tracked_key not in config.stash:
        root = config.rootpath
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "--", *_GATED_DIRS],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = b""
        names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
        config.stash[_tracked_key] = frozenset(_norm(root / n) for n in names) or None
    return config.stash[_tracked_key]


def pytest_ignore_collect(collection_path, config):
    if not (collection_path.name.startswith("test_") and collection_path.suffix == ".py"):
        return None
    path = _norm(collection_path)
    gated = [_norm(config.rootpath / d) + os.sep for d in _GATED_DIRS]
    if not any(path.startswith(prefix) for prefix in gated):
        return None
    tracked = _tracked_files(config)
    if tracked is None or path in tracked:
        return None
    rel = os.path.relpath(collection_path, config.rootpath).replace(os.sep, "/")
    config.stash.setdefault(_skipped_key, set()).add(rel)
    return True


def pytest_sessionfinish(session):
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput[_UNTRACKED_KEY] = sorted(session.config.stash.get(_skipped_key, ()))
        workeroutput[_HULL_MEMO_KEY] = dict(session.config.stash.get(_memo_stats_key, {}))


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    output = getattr(node, "workeroutput", {})
    node.config.stash.setdefault(_skipped_key, set()).update(output.get(_UNTRACKED_KEY, ()))
    stats = output.get(_HULL_MEMO_KEY)
    if stats:
        node.config.stash.setdefault(_memo_stats_key, Counter()).update(stats)


def pytest_terminal_summary(terminalreporter, config):
    if hasattr(config, "workerinput"):
        return
    skipped = sorted(config.stash.get(_skipped_key, ()))
    if skipped:
        terminalreporter.write_line(
            "untracked test files not collected (git add one, or name it, to run it): "
            + ", ".join(skipped)
        )
    if _hull_memo_off():
        terminalreporter.write_line(f"hull memo: off ({_NO_HULL_MEMO_ENV})")
        return
    stats = config.stash.get(_memo_stats_key, None)
    if stats:
        systems = sorted(
            ((name[5:], n) for name, n in stats.items() if name.startswith("hits ")),
            key=lambda item: -item[1],
        )
        terminalreporter.write_line(
            f"hull memo: {stats['hits']} hits, {stats['misses']} misses, "
            f"{stats['bypasses']} bypasses, {stats['trips']} mutations caught; hits by system: "
            + ", ".join(f"{name} {n}" for name, n in systems[:6])
        )


def _hull_memo_off() -> bool:
    return os.environ.get(_NO_HULL_MEMO_ENV, "") not in ("", "0")


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "collection").rsplit(" ", 1)[0]


def _hull_fingerprint(hull) -> tuple:
    """Per entry: id, energy, correction and sorted ``data`` keys; then the stable set."""
    rows = []
    for entry in hull.all_entries:
        source = getattr(entry, "original_entry", None) or entry
        entry_id = source.entry_id
        rows.append(
            (
                entry_id if isinstance(entry_id, str) else repr(entry_id),
                entry.energy,
                getattr(source, "correction", None),
                tuple(sorted(map(str, getattr(source, "data", None) or ()))),
            )
        )
    return tuple(rows), frozenset(map(id, hull.stable_entries))


def _record_digest(backend, record) -> str | None:
    """sha1 of the record ``backend`` serves, or None when it holds none or is not memoised."""
    from gliquid.cache import DirectoryBackend, OverlayBackend, SqliteBackend

    if type(backend) is DirectoryBackend:
        path = backend.path_for(record)
        return hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None
    base = backend.base if type(backend) is OverlayBackend else backend
    if type(base) not in (DirectoryBackend, SqliteBackend) or not backend.exists(record):
        return None
    payload = json.dumps(backend.read_json(record), sort_keys=True, separators=(",", ":"))
    return "json:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _blacklist_stamp(config) -> tuple | None:
    """Path, mtime, size and sha1 of the spurious-structure blacklist, or None without one."""
    path = config.spurious_structures_file
    if path is None or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    return str(path), stat.st_mtime_ns, stat.st_size, digest


class _HullMemo:
    """Content-keyed memo in front of ``api.get_dft_convexhull``; see the module docstring."""

    def __init__(self, api, stats: Counter):
        self.api = api
        self.real = api.get_dft_convexhull
        self.signature = inspect.signature(self.real)
        self.hulls: dict[tuple, list] = {}  # key -> [hull, fingerprint, last check], LRU order
        self.touched: set[tuple] = set()
        self.stats = stats

    def key(self, args, kwargs) -> tuple | None:
        """The memo key of one call, or None when the call goes to the real function."""
        api = self.api
        try:
            call = self.signature.bind(*args, **kwargs)
        except TypeError:
            return None
        call.apply_defaults()
        call = call.arguments
        if any(call.get(name) for name in ("verbose", "include_imputed", "inc_structure_data")):
            return None
        try:
            components, _, _ = api.validate_and_format_system(call["system"], allow_compounds=True)
            api._validate_dft_type(call["dft_type"])
            backend = api.resolve_backend(call["data_dir"])
            record = api.CacheKey(
                api._canonical_sys_name(components), api.KIND_DFT_ENTRIES, call["dft_type"]
            )
            digest = _record_digest(backend, record)
        except Exception:
            return None
        if digest is None:
            return None
        return tuple(components), call["dft_type"], digest, _blacklist_stamp(api.config)

    def __call__(self, *args, **kwargs):
        key = self.key(args, kwargs)
        if key is None:
            self.stats["bypasses"] += 1
            return self.real(*args, **kwargs)
        system = "-".join(key[0])
        self.touched.add(key)
        held = self.hulls.pop(key, None)
        if held is not None:
            if _hull_fingerprint(held[0]) != held[1]:
                self.stats["trips"] += 1
                pytest.fail(
                    f"hull memo: the shared {system} DFT hull changed after {held[2]}: a test "
                    f"holding it through a module, class or session fixture mutated it in place.",
                    pytrace=False,
                )
            self.hulls[key] = held
            self.stats["hits"] += 1
            self.stats[f"hits {system}"] += 1
            return held[0], {}
        hull, volumes = self.real(*args, **kwargs)
        self.stats["misses"] += 1
        self.hulls[key] = [hull, _hull_fingerprint(hull), f"{_current_test()} loaded it"]
        while len(self.hulls) > 1 and (
            sum(len(h[0].all_entries) for h in self.hulls.values()) > _HULL_MEMO_ENTRIES
        ):
            del self.hulls[next(iter(self.hulls))]
        return hull, volumes

    def check(self, nodeid: str) -> None:
        """Re-fingerprint the hulls touched since the last check; fail naming ``nodeid``."""
        changed = []
        for key in list(self.touched):
            held = self.hulls.get(key)
            if held is None:
                continue
            if _hull_fingerprint(held[0]) == held[1]:
                held[2] = f"the teardown of {nodeid} checked it"
            else:
                del self.hulls[key]
                changed.append("-".join(key[0]))
        self.touched.clear()
        if changed:
            self.stats["trips"] += len(changed)
            pytest.fail(
                f"hull memo: {nodeid} mutated the shared {', '.join(changed)} DFT hull that "
                f"gliquid.api.get_dft_convexhull handed it; copy the hull before changing it.",
                pytrace=False,
            )


@pytest.fixture(scope="session", autouse=True)
def _hull_memo(request):
    """Route ``gliquid.api.get_dft_convexhull`` through the hull memo for the session."""
    if _hull_memo_off():
        yield
        return
    import gliquid.api as api

    stats = request.config.stash.setdefault(_memo_stats_key, Counter())
    memo = request.config.stash[_memo_key] = _HullMemo(api, stats)

    @functools.wraps(memo.real)
    def get_dft_convexhull(*args, **kwargs):
        return memo(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "get_dft_convexhull", get_dft_convexhull)
        yield


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    memo = item.config.stash.get(_memo_key, None)
    if memo is not None:
        memo.check(item.nodeid)


def _configured_store() -> Path | None:
    value = os.environ.get(_SQLITE_STORE_ENV)
    return Path(value) if value else None


def pytest_configure(config):
    store = _configured_store()
    if store is None:
        return
    if not store.is_file():
        raise pytest.UsageError(
            f"{_SQLITE_STORE_ENV} is set to '{store}', which is not a file. Build one with "
            f"`python -m gliquid.cache migrate --from cache --to {store}`."
        )
    import gliquid.config as gliquid_config

    gliquid_config.set_cache_dir(store)
    if gliquid_config.cache_mode != "sqlite":  # pragma: no cover - defensive
        raise pytest.UsageError(
            f"set_cache_dir('{store}') left cache_mode {gliquid_config.cache_mode!r}; the "
            f"suite would have run against a directory store while claiming otherwise."
        )


def pytest_report_header(config):
    """Say which store the run used. A silent swap is indistinguishable from no swap."""
    store = _configured_store()
    if store is None:
        return None
    return f"gliquid cache store: SQLITE (single file) {store}"


def pytest_collection_modifyitems(config, items):
    if _configured_store() is None:
        return
    skip = pytest.mark.skip(
        reason=f"directory-only test, and {_SQLITE_STORE_ENV} selected a single-file store"
    )
    for item in items:
        if "directory_only" in item.keywords:
            item.add_marker(skip)
