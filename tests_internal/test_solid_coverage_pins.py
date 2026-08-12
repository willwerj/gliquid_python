"""Frozen solid-coverage verdicts for the hand-labelled acceptance set.

Split out of ``tests/test_solid_coverage_gate.py``, which keeps the behaviour of the gate
itself (skip/proceed, the threshold overrides, the chart relocation). What lives here is the
pin: 52 parametrized cases read from ``tests/fixtures/solid_coverage_pins.json`` and replayed
against the workspace ``matrix_data`` store they were frozen against. Both are maintainer
machinery -- the fixture is a value freeze, and the whole set skips on a checkout without the
cache.

The fixture stays in ``tests/fixtures/``; this file reaches back out to it.
"""

import json
from pathlib import Path

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid

pytestmark = [pytest.mark.pins, pytest.mark.needs_cache]

PINS_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "solid_coverage_pins.json"


def _matrix_data_dir():
    """The workspace DFT/MPDS cache the pins were frozen against, if this checkout has it.

    Anchored by name, not by parent depth, per the workspace convention.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "matrix_data"
        if candidate.is_dir() and (candidate / "omegas_hcp.json").exists():
            return candidate
    return None


def _pin_cases():
    """(ref_mode, system, expected) for every pinned case, or nothing if pins are absent.

    ``expected`` carries a resolved ``pd_ind``: the fixture's top-level default, overridden
    per system when an entry names its own. Load-bearing -- see the note in
    ``test_ground_truth_coverage_pins`` on why the index may not be left implicit.
    """
    if not PINS_PATH.exists():
        return []
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    default_pd_ind = pins["pd_ind"]
    return [
        (ref_mode, system, {"pd_ind": default_pd_ind, **expected})
        for ref_mode, arm in pins["arms"].items()
        for system, expected in arm.items()
    ]


@pytest.fixture
def matrix_data_env(monkeypatch):
    """Point config at the workspace cache the pins were frozen against.

    Load-bearing: the shipped ``data/omegas_hcp.json`` is a 3-pair test fixture, while the
    pins were frozen against the 780-pair production file. Without this the same systems
    resolve no solid-solution models and every pinned ss_models list is empty.
    """
    matrix_data = _matrix_data_dir()
    if matrix_data is None:
        pytest.skip("workspace matrix_data cache not present in this checkout")
    original_dir, original_struct = config.data_dir, config.dir_structure
    config.set_data_dir(matrix_data)
    # matrix_data is per-system subdirectories, not the flat shipped-fixture layout.
    config.set_dir_structure("nested")
    monkeypatch.setattr(config, "solid_solutions", True)
    try:
        yield matrix_data
    finally:
        config.set_data_dir(original_dir)
        config.set_dir_structure(original_struct)


@pytest.mark.parametrize(
    "ref_mode,system,expected", _pin_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_ground_truth_coverage_pins(ref_mode, system, expected, matrix_data_env):
    """Frozen verdicts for the hand-labelled acceptance set, per SS reference mode.

    Needs the full workspace cache, so the whole set skips on a checkout without it.

    ``pd_ind`` is passed explicitly and must stay that way. ``load_mpds_data(pd_ind=None)``
    resolves the indexless ``matrix_data/<sys>/<sys>.json`` and only falls back to
    ``<sys>_MPDS_PD_0.json`` when that is absent, so leaving it implicit pins the verdicts to
    whichever of the two files happens to be on disk. That is not hypothetical: the indexless
    jsons were swept out of ``matrix_data`` as debris on 2026-08-10, which silently moved
    Cu-Mg from entry C100123 (no compound shape) to C900864 (``Mg2Cu`` as a genuine line
    compound) and turned a green pin red without a line of package code changing.

    The two arms encode the load-bearing property that the verdict follows the *loaded*
    energies, not the system alone. Lu-Nd, Th-Tm, Ho-Pr, Er-Th, Mn-Y, Rb-Sn, Ho-Zr and Se-Te
    are skipped under BOTH ref modes: their elements are absent from the omegas file entirely,
    so no configuration and no fallback rescues them.

    The arms now agree on every verdict. They did not before: Ag-Au, Ag-Pd, Ta-W, Bi-Sb and
    As-Sb were skipped only under 'from_unary_db', because the unary db has no polymorph entry
    for structures those elements never exhibit (Ag has no FCC entry, Ta no BCC entry at all)
    even though the omegas file carries the energies. The omegas fallback in
    solution._apply_omegas_fallback closes that gap, so the two reference sources now differ in
    the VALUES they resolve, not in which systems they can model. The arms are kept separate
    because the values still differ -- and because a future divergence in verdicts is exactly
    the kind of regression this fixture exists to catch.
    """
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    for key, value in pins["thresholds"].items():
        assert getattr(config, f"coverage_{key}") == value, (
            f"pins were frozen with coverage_{key}={value}; re-freeze them if the default moved"
        )

    try:
        bl = BinaryLiquid.from_cache(
            system,
            param_format="comb-exp",
            pd_ind=expected["pd_ind"],
            ss_kwargs={"ref_mode": ref_mode},
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{system} not available in the local cache ({exc})")
    if bl.init_error or not bl.digitized_liq:  # pragma: no cover
        pytest.skip(f"{system} is excluded upstream of the coverage gate")

    bl.find_invariant_points(verbose=False)
    cov = bl.assess_solid_coverage()
    assert list(cov.ss_models) == expected["ss_models"]
    assert cov.unsupported_fraction == pytest.approx(expected["unsupported_fraction"], abs=1e-3)
    assert cov.n_compounds == expected["n_compounds"]
    assert cov.n_missing_compounds == expected["n_missing_compounds"]
    assert cov.is_insufficient()[0] is expected["skipped"]
