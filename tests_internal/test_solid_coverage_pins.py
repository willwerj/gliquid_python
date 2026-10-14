"""Frozen solid-coverage verdicts for the hand-labelled acceptance set.

``tests/test_solid_coverage_gate.py`` keeps the behaviour of the gate itself. The pin here is
the parametrized cases read from ``tests/fixtures/solid_coverage_pins.json`` and replayed
against the maintainer's per-system corpus store they were frozen against; the whole set
skips on a checkout without it. Re-frozen by refreeze_spec10_pins.py
(refreeze_solid_coverage_pins.py diffs them under an explicit ``pd_ind``).
"""

import json
from pathlib import Path

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid
from tests._mpds_builders import coverage_pin_cases

pytestmark = [pytest.mark.pins, pytest.mark.needs_cache]

PINS_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "solid_coverage_pins.json"


@pytest.mark.parametrize(
    "ref_mode,system,expected",
    coverage_pin_cases(PINS_PATH),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_ground_truth_coverage_pins(ref_mode, system, expected, matrix_data_env):
    """Frozen verdicts for the hand-labelled acceptance set, per SS reference mode.

    ``pd_ind`` is passed explicitly and must stay that way: ``load_mpds_data(pd_ind=None)``
    resolves the indexless ``<sys>/<sys>.json`` and only falls back to ``<sys>_MPDS_PD_0.json``
    when that is absent, so an implicit index pins the verdicts to whichever file is on disk.

    The two arms assert that the verdict follows the *loaded* energies, not the system alone.
    Lu-Nd, Th-Tm, Ho-Pr, Er-Th, Mn-Y, Rb-Sn, Ho-Zr and Se-Te are skipped under BOTH ref modes:
    their elements are absent from the omegas file. The arms agree on every verdict and differ
    in the values they resolve.
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
    # The statistic the verdict is taken on, pinned beside the verdict itself: a criterion
    # change that moves a system across the floor has to move this first.
    assert cov.largest_supported_segment == pytest.approx(
        expected["largest_supported_segment"], abs=1e-3
    )
    assert cov.n_supported_segments == expected["n_supported_segments"]
    assert cov.n_compounds == expected["n_compounds"]
    assert cov.n_missing_compounds == expected["n_missing_compounds"]
    assert cov.is_insufficient()[0] is expected["skipped"]
