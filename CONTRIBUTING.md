# Contributing to GLiquid

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
pip install -e .[test]
```

`.[test]` adds `pytest` and `ruff` on top of the runtime dependencies. Add other extras as the
work needs them — `.[editor]` for `ConvexHullEditor`, `.[mpds]` for live MPDS downloads,
`.[notebook]` for local Jupyter, `.[models]` for the exact `scikit-learn`/`xgboost` versions the
serialized model artifacts were pickled against.

Working from a clone also means the data corpus resolves automatically: `gliquid.config` falls
back to the checkout's own `data/` directory when neither `set_data_dir()` nor
`GLIQUID_DATA_DIR` has been used. See the README for the full resolution order.

## Running the tests

Run `pytest` from the repository root. The suite is tiered, and which tier you get depends on
what you pass:

```bash
pytest                        # the public tests/ suite -- the default, and what a contributor runs
pytest tests_internal         # the maintainer tier: value pins, figure goldens, gated families
pytest tests tests_internal   # everything; this is what CI runs
```

A bare `pytest` collects only `tests/` because `testpaths = ["tests"]` says so in
`pyproject.toml`; `tests_internal/` is tracked in git but never collected implicitly, and it is
excluded from the sdist.

For the fast edit/run loop, deselect the slow fits:

```bash
pytest -m "not slow"
```

That is roughly a 2.7x speedup on the public suite — measured at 84.7 s against 225.7 s for the
full run on a maintainer machine. It is a faster loop, not an instant one; run the full suite
before you push.

### Markers

Four markers are declared, and `addopts = "--strict-markers"` means an undeclared or misspelled
one fails the run instead of silently applying no marker at all.

| marker | meaning |
| --- | --- |
| `pins` | A frozen-value characterization pin. It asserts that today's numbers equal the numbers captured when the pin was frozen. |
| `needs_cache` | Needs the workspace `matrix_data/` store of per-system DFT and digitized-diagram caches. **Skips**, rather than fails, on a checkout without it. |
| `needs_network` | Needs `NEW_MP_API_KEY` and a live Materials Project connection. |
| `slow` | A multi-second Nelder–Mead fit or hull build. Deselect with `-m "not slow"`. |

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
ruff format src/gliquid tests tests_internal   # before you push
ruff format --check src/gliquid tests tests_internal
ruff check src/gliquid                          # advisory for now, not yet a gate
```

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

There are exactly two documented exemptions, listed in `EXEMPT` in that test file:

- `mpds.print_phase_mismatch_chart` — the function *is* a console chart renderer; the aligned
  monospace rows are the product, not diagnostics.
- `hull_editor._log` — writes into an `ipywidgets` `Output` pane. Widget UI, not logging.

Adding a third exemption requires editing that test, which is the point. If your output is
diagnostic, use the module logger.

## Release checklist

The distribution version is derived from the git tag by `hatch-vcs` — tagging is the release,
and no version literal lives in `src/`. `tests/test_version.py` enforces that.

1. **Bump `CITATION.cff` by hand.** Its `version:` and `date-released:` are literals and nothing
   derives them; this is the one file that must be edited at every release.
2. **Add the `CHANGELOG.md` entry.** Move what is under `## [Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the link definitions at the bottom.
3. **Commit** those two changes.
4. **Tag** the commit `vX.Y.Z` (SemVer, `v` prefix) and **push the tag**.

The release workflow does the rest — building the sdist and wheel from the tag and publishing
them. Do not hand-edit a version anywhere else; if the built artifact reports `0.0.x`, the
`hatch-vcs` fallback fired and the build did not see the tag. Investigate rather than
overriding it.
