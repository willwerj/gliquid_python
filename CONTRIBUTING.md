# Contributing to G<sub>Liquid</sub>

Pull requests are welcome. For anything larger than a bug fix, please open an issue first so
the change can be discussed before it is written.

## Setup

Work from a clone, in an isolated environment. The convention among the maintainers is a conda
environment named `gliquidenv`:

```bash
git clone https://github.com/willwerj/gliquid_python.git
cd gliquid_python
conda create --name gliquidenv python=3.10   # 3.11, 3.12 and 3.13 are also supported
conda activate gliquidenv
pip install -e ".[test]"
```

`.[test]` adds `pytest`, `pytest-xdist`, `ruff` and the ML test stack (`gliquid[ml]`, `scikit-learn`, `joblib`, `openpyxl`) on top of the runtime dependencies. Add other extras as the
work needs them — `.[mpds]` for live MPDS downloads, `.[ml]` for `ProductionModelRunner`, and
`.[notebook]` for local Jupyter and `ConvexHullEditor`.

Working from a clone also means the data corpus resolves automatically when the checkout
directory is named `gliquid_python`. The store is the first of: `set_cache_dir()`,
`GLIQUID_CACHE_DIR`, the checkout's `cache/`, then `./gliquid_cache`.

## Running the tests

Run `pytest` from the repository root. It collects `tests/`, which is excluded from the sdist.
The value pins, figure goldens and gated families carry markers (below), so `-m "not pins"`
or `-m "not needs_cache"` deselects them.

The gate is the whole suite in parallel under pytest-xdist. Run it before you push; CI runs the
same command with `-m "not needs_network"`:

```bash
pytest -n auto --dist worksteal
```

- Use `-n 8` instead of `-n auto` while a fit campaign or another heavy job runs: each worker
  commits about 1.7 GB of memory.
- Run serially (no `-n`) to attribute a failure to one test.
- Never put `-n` in `addopts`.

For the inner loop, `pytest-testmon` reruns only the tests that execute code you changed. It is
optional and not part of `.[test]`:

```bash
pip install pytest-testmon
pytest --testmon   # first run: the whole suite, recording what each test executes
pytest --testmon   # later runs: only the tests your edits affect
```

It keeps its record in `.testmondata` at the repository root (git-ignored), needs the cache
provider (do not pass `-p no:cacheprovider`), and also works with `-n auto`. An edit that changes
no code, such as a comment, selects nothing. Tests that start a fresh interpreter, or read the
source as text, are not selected by a change under `src/`, so run the gate before you push.

Without testmon, run one module (`pytest tests/test_phase.py`), or rerun the last failures with
`--lf` (it also needs the cache provider).

Only git-tracked test files are collected from `tests/`. An untracked or
ignored `test_*.py` is skipped and listed in the terminal summary; `git add` it, or name it on
the command line, to run it.

### Markers

Four markers are declared, and `addopts = "--strict-markers"` means an undeclared or misspelled
one fails the run instead of silently applying no marker at all.

| marker | meaning |
| --- | --- |
| `pins` | A frozen-value characterization pin. It asserts that today's numbers equal the numbers captured when the pin was frozen. |
| `needs_cache` | Needs the maintainer's per-system corpus store. **Skips**, rather than fails, without it. |
| `needs_network` | Needs `NEW_MP_API_KEY` and a live Materials Project connection. |
| `directory_only` | Needs a directory cache store. Skipped when `GLIQUID_TEST_SQLITE_STORE` swaps in a single-file store. |

### Writing a test

- A test passes alone and in any order: the gate's workers reorder tests. Build what you read,
  and never rely on another test having run.
- Build expensive objects once.
  - The root `conftest.py` memoises DFT hulls for the session, so do not cache hulls yourself.
  - Reuse the session fits and builders in `tests/_mpds_builders.py`, and scope your own
    fixtures to the module or session.
  - Deep-copy a shared object before you change it: the hull memo fails a test that mutates a
    shared hull.
- Fit each system once per module, with the smallest `n_opts` and `max_iter` that still return a
  fit.
- Start a fresh interpreter only for an import or install contract that an in-process test
  cannot check. Do it through `tests/_fresh.py`, with one probe per target.
- Keep a test under about 5 s; CI prints the 25 slowest. There is no slow marker: the parallel
  gate is the fast path.
- Assert something no other test asserts, and make each pin case change the pinned output.
- Do not assume the checkout directory is named `gliquid_python`, and do not build paths from
  `config.project_root`. Anchor on the test file's own location.
- A removed released name keeps one test that it raises `TypeError` (`AttributeError` for a
  removed attribute). A name that never shipped
  gets no absence test.
- A comment says what the test checks, never why it changed. A docstring states what the test
  asserts and, for a pin, which script re-freezes it.

### Re-freezing a pin

Every pin file's module docstring names the script that froze its fixture (the freezer scripts
live with the maintainers' driver scripts, outside this repository). Re-freezing is a deliberate
act: run that script, inspect the diff in the fixture JSON, and say in the commit message *why*
the values moved.

**Never edit a pin to make it pass.** A pin that changed is telling you the behavior changed. If
that change was intended, re-freeze it through the freezer and justify it. If it was not, you
have found the bug the pin exists to catch.

## Formatting and linting

`ruff` is the formatter, configured in `pyproject.toml` at `line-length = 100` — not the 88
default, because this codebase already writes to roughly 95–100 characters and 88 would split
calls that read fine today.

```bash
ruff format src/gliquid tests   # before you push
ruff format --check src/gliquid tests
ruff check src/gliquid tests     # CI fails on any finding, as on unformatted code
```

A pre-commit hook in `.githooks/` runs `ruff format` over the staged files in those same two
paths and re-stages them, so a commit records formatted code. It is **not active until you
switch git's hook path to it** — that setting lives in `.git/config` and is not carried by a
clone:

```bash
git config core.hooksPath .githooks   # once per clone
git commit --no-verify                # skip the hook for one commit
```

The hook formats only the paths above, and only files that are wholly staged: a file with
unstaged changes alongside staged ones is left alone, and the commit aborts only if ruff would
reformat it, because `git add` on a partially staged file would sweep the unstaged hunks in as
well. If neither `ruff` nor `python -m ruff` is available it warns and lets the commit through rather than blocking you; set `GLIQUID_RUFF` to an
explicit binary if it lives somewhere unusual.

Scattered `# noqa: WPS...` comments survive in the source. They are inherited from a retired
wemake-python-styleguide configuration and **are enforced by nothing today** — no tool in the
current setup reads them. Do not add new ones, and do not treat an existing one as evidence that
some rule is still active.

## The print/logging boundary

The library reports through `logging` — one `logger = logging.getLogger(__name__)` per module,
all children of the `gliquid` logger — and installs no handlers and sets no levels of its own.
Handler and level configuration belongs to the application consuming the library, not to the
library.

Consequently, **a bare `print()` added anywhere in `src/gliquid` fails
`tests/test_logging_boundary.py`.** So does a direct `sys.stdout.write` / `sys.stderr.write`, and
so does any call to `basicConfig`, `addHandler`, `setLevel`, `StreamHandler`, `FileHandler` or
`NullHandler` inside the package. The scan is over the source tree, not the import graph, so a
`print()` on a branch no test happens to execute is still caught.

There are three documented exemptions, listed in `EXEMPT` in that test file:

- `mpds.print_phase_mismatch_chart` — the function *is* a console chart renderer; the aligned
  monospace rows are the product, not diagnostics.
- `hull_editor._log` — writes into an `ipywidgets` `Output` pane. Widget UI, not logging.
- `cache._emit` — prints the `python -m gliquid.cache` command-line tool's output.

Adding another exemption requires editing that test, which is the point. If your output is
diagnostic, use the module logger.

## Release checklist

The distribution version is derived from the git tag by `hatch-vcs` — tagging is the release,
and no version literal lives in `src/`. `tests/test_version.py` enforces that.

1. **Bump `CITATION.cff` by hand.** Its `version:` and `date-released:` are literals and nothing
   derives them.
2. **Repoint the notebooks' Colab pin.** Every notebook's Colab cell installs
   `gliquid==<version>`; set it to `X.Y.Z` in all of them. `tests/test_notebook_colab_pins.py`
   checks they agree.
3. **Add the `CHANGELOG.md` entry.** Move what is under `## [Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the link definitions at the bottom.
4. **Commit** those changes.
5. **Tag** the commit `vX.Y.Z` (SemVer, `v` prefix) and **push the tag**.

The release workflow does the rest — building the sdist and wheel from the tag and publishing
them. Do not hand-edit a version anywhere else; if the built artifact reports `0.0.x`, the
`hatch-vcs` fallback fired and the build did not see the tag. Investigate rather than
overriding it.
