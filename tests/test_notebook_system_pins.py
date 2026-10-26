"""Fixed-parameter pins for the systems the shipped notebooks predict.

Frozen by freeze_notebook_system_pins.py into fixtures/notebook_system_pins.json, which calls
:func:`compute_pins` below. Each system is built as the notebooks build it:
``BinaryLiquid.from_cache(system, param_format="comb-exp")``, then
``update_params(FIXED[system] + [0])``, with ``FIXED`` the ``20260329_022905`` model's
parameters held as literals here.

Pinned per system:
  * ``liquidus``: the predicted liquidus the notebooks overlay, ``bl.phases[-1].points`` as
    ``[x, T]`` pairs.
  * ``pred``: the ``'pred'`` figure, captured as trace rows, annotations and y-range by
    ``capture_figure`` from _mpds_builders.py, as test_decomposition_pins.py's figure
    summaries are.

Everything runs inside :func:`hermetic_mpds`: online mode with no API key and tripwired
network clients, so a system with no cached MPDS record takes ``load_mpds_data``'s no-key
path. The store's file listing must not change.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid, BLPlotter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402
from _mpds_builders import capture_figure, hermetic_mpds  # noqa: E402

pytestmark = pytest.mark.pins

PINS_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "notebook_system_pins.json"

FIXED = {
    "Pb-Tb": [-144129.625, -19.75286865234375, 57807.396484375],
    "Ga-Ru": [-90334.796875, -10.073455810546875, -28111.484375],
    "Cu-Mg": [-27056.70703125, -5.751094341278076, -17084.64501953125],
}
SYSTEMS = tuple(FIXED)


def compute_system(system: str) -> dict:
    """The pinned blocks for one system."""
    with hermetic_mpds():
        bl = BinaryLiquid.from_cache(system, param_format="comb-exp")
        bl.update_params(FIXED[system] + [0])
        liquidus = [[p[0], p[1]] for p in bl.phases[-1].points]
        pred = capture_figure(BLPlotter(bl).get_plot("pred"))
    return pu.to_jsonable({"liquidus": liquidus, "pred": pred})


def compute_pins() -> dict:
    """``{system: {"liquidus": ..., "pred": ...}}`` for every system in ``FIXED``."""
    return {system: compute_system(system) for system in SYSTEMS}


def store_listing() -> dict:
    """The active store's location, whether it exists, and its files' names and sizes."""
    root = Path(config.cache_dir) if config.cache_dir is not None else None
    files = {}
    if root is not None and root.is_dir():
        files = {p.name: p.stat().st_size for p in sorted(root.iterdir()) if p.is_file()}
    return {
        "root": str(root),
        "exists": bool(root is not None and root.exists()),
        "files": files,
    }


@pytest.fixture(scope="module")
def pins():
    return json.loads(PINS_PATH.read_text())


@pytest.fixture(scope="module")
def live():
    before = store_listing()
    computed = compute_pins()
    return computed, before, store_listing()


@pytest.mark.parametrize("system", SYSTEMS)
def test_predicted_liquidus_pinned(pins, live, system):
    pu.assert_deep_approx(pins[system]["liquidus"], live[0][system]["liquidus"])


@pytest.mark.parametrize("system", SYSTEMS)
def test_pred_figure_pinned(pins, live, system):
    pu.assert_deep_approx(pins[system]["pred"], live[0][system]["pred"])


def test_the_store_listing_is_unchanged(live):
    _, before, after = live
    assert after == before


def test_the_pins_reference_no_model_runner():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {
        alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    names |= {
        node.module.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"ProductionModelRunner", "predict_system", "DEFAULT_BUNDLE_ID"}
    assert names & forbidden == set()
    assert {"BinaryLiquid", "update_params"} <= names
