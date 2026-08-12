"""Characterization pins for ternary liquid surfaces of diverse mixing character.

Extends the Hf-Ti-Zr / Al-Mg-Si net (test_ternary_pipeline.py) with triads whose
binary L0 terms differ in sign and magnitude:

  * Al-Ca-Ni  — strongly compound-forming (large negative L0 on every edge)
  * Ag-Cu-Pb  — demixing / monotectic-like (positive L0 on every edge)
  * Cu-Fe-Si  — mixed-sign, L1-heavy asymmetric edges (linear + Muggianu)

Frozen pre-refactor by dev/scripts/_generated/freeze_diverse_ternary_pins.py into
fixtures/ternary_diverse_pins.json. The fixture is self-describing: each entry
embeds its construction config (elements, L_dict, delta, interp_type,
param_format) beside the pinned x0/x1/H/S/phase arrays, so parameters are defined
in exactly one place. Gate: rtol <= 1e-9. Runs offline (unary registry only).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import pin_utils as pu  # noqa: E402

pytestmark = pytest.mark.pins

PINS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ternary_diverse_pins.json"
    ).read_text()
)


@pytest.mark.parametrize("key", sorted(PINS))
def test_liquid_surface(key):
    blk = PINS[key]
    cfg = blk["config"]
    tmod = pu.get_ternary_mod()
    ti = tmod.TernaryLiquidInterpolation(
        list(cfg["elements"]),
        xs_mix={k: list(v) for k, v in cfg["L_dict"].items()},
        delta=cfg["delta"],
        interp_scheme=cfg["interp_type"],
        param_format=cfg["param_format"],
    )
    ti.interpolate_liquid_surface()
    df = ti.hsx_df
    pu.assert_deep_approx(blk["x0"], df["x0"].to_numpy(dtype=float))
    pu.assert_deep_approx(blk["x1"], df["x1"].to_numpy(dtype=float))
    pu.assert_deep_approx(blk["H"], df["H"].to_numpy(dtype=float))
    pu.assert_deep_approx(blk["S"], df["S"].to_numpy(dtype=float))
    assert df["Phase Name"].tolist() == blk["phase"]
