"""Shared helpers for the characterization-pin tests.

Fixture ref-mode relabelling, the comb-exp liquid builder, deep comparison at ``RTOL`` and
canonical forms of hull simplices and solid-solution models. Test modules import it flat,
with ``tests/`` on ``sys.path``.
"""

from __future__ import annotations

import math

import numpy as np

RTOL = 1e-9
ABS_FLOOR = 1e-12

# Ref-mode labels. The pin fixtures stay byte-frozen with their own labels
# ('omegas-legacy'/'binary-cache'/'element-db'); the code emits the API names, so tests
# translate at the fixture boundary. Never regenerate a fixture to match.
FIXTURE_REF_MODE = {  # new API name -> frozen fixture key/label
    "from_omegas_file": "omegas-legacy",
    "from_dft_entries": "binary-cache",
    "from_unary_db": "element-db",
}
_REF_MODE_RELABEL = {old: new for new, old in FIXTURE_REF_MODE.items()}


def relabel_ref_modes(obj):
    """Recursively rewrite frozen ref-mode/source label strings to the new API names,
    so a byte-frozen pin compares equal to renamed live output."""
    if isinstance(obj, dict):
        return {k: relabel_ref_modes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relabel_ref_modes(v) for v in obj]
    if isinstance(obj, str):
        return _REF_MODE_RELABEL.get(obj, obj)
    return obj


def build_combexp_eqs(ga_expr, gb_expr, tau):
    """The comb-exp binary liquid expressions at ``tau``, from ``SolutionModel.binary_eqs``."""
    from gliquid.solution import RKPolyExp, SolutionModel

    return SolutionModel(
        ("A", "B"), (ga_expr, gb_expr), {(0, 1): RKPolyExp("comb-exp", tau=tau)}
    ).binary_eqs()


# -- Comparison helpers --


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def assert_deep_approx(pinned, live, rtol=RTOL, path="$"):
    """Recursive equality: floats at rtol (abs floor 1e-12), everything else exact."""
    live = to_jsonable(live)
    _cmp(pinned, live, rtol, path)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cmp(a, b, rtol, path):
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), f"{path}: type {type(a)} vs {type(b)}"
        assert set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            _cmp(a[k], b[k], rtol, f"{path}.{k}")
    elif isinstance(a, list) or isinstance(b, list):
        assert isinstance(a, list) and isinstance(b, list), f"{path}: type {type(a)} vs {type(b)}"
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, rtol, f"{path}[{i}]")
    elif _is_number(a) and _is_number(b):
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), f"{path}: {a!r} != {b!r}"
        else:
            assert math.isclose(a, b, rel_tol=rtol, abs_tol=ABS_FLOOR), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def canonical_simplices(simplices):
    """Order-insensitive exact form of a simplex index array: vertices sorted within each
    simplex, simplices sorted lexicographically. Robust to qhull enumeration order."""
    arr = [sorted(int(v) for v in row) for row in to_jsonable(simplices)]
    return sorted(arr)


def canonical_ss_model(model, components):
    """Element/pair-keyed value view of one per-phase SS model dict, old or new schema."""
    comps = sorted(components)
    if "omega_jmol" in model:  # binary-locked schema
        pair = "-".join(comps)
        return {
            "omega": {pair: float(model["omega_jmol"])},
            "delta_h": {
                comps[0]: float(model["deltaH_a_jmol"]),
                comps[1]: float(model["deltaH_b_jmol"]),
            },
            "delta_s": {
                comps[0]: float(model["deltaS_a_jmol_k"]),
                comps[1]: float(model["deltaS_b_jmol_k"]),
            },
        }
    return {
        "omega": {k: float(v) for k, v in model["omega"].items()},
        "delta_h": {k: float(v) for k, v in model["delta_h"].items()},
        "delta_s": {k: float(v) for k, v in model["delta_s"].items()},
    }
