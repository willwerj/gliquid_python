"""Shared helpers for the nd-reductionist-refactor characterization-pin tests.

Used by test_hull_numerics.py and test_ternary_pipeline.py. The resolver helpers make the
pin tests SELF-ADAPTING across the refactor's atomic steps (module renames, builder moves,
SS-schema change) without ever touching the pinned VALUES — the numbers frozen by
dev/scripts/_scratch/freeze_refactor_pins.py are the invariant.
"""

from __future__ import annotations

import math

import numpy as np

RTOL = 1e-9
ABS_FLOOR = 1e-12

# Ref-mode label rename (S14). The pin fixtures stay byte-frozen with the OLD labels
# ('omegas-legacy'/'binary-cache'/'element-db'); the live code emits the NEW API names,
# so tests translate at the fixture boundary rather than regenerating any fixture.
FIXTURE_REF_MODE = {  # new API name -> frozen fixture key/label
    "from_omegas_file": "omegas-legacy",
    "from_dft_entries": "binary-cache",
    "from_unary_db": "element-db",
}
_REF_MODE_RELABEL = {old: new for new, old in FIXTURE_REF_MODE.items()}


# Tiered-policy renames (2026-08): the from_unary_db runtime omegas fallback became the
# builder-baked lattice_stabilities block. Same energies and anchoring math, new provenance
# labels — translated here so the byte-frozen pins keep comparing equal. These labels only
# ever appear under the element-db pin; the other modes still use the runtime fallback and
# keep their original labels.
_UNARY_FALLBACK_SOURCE_PREFIX = "from_unary_db+omegas_fallback"
_LATTICE_STABILITY_SOURCE = (
    "from_unary_db:lattice_stability (omegas_hcp.json; Chen et al., Nat. Commun. 14, 2856 (2023))"
)


def relabel_ref_modes(obj):
    """Recursively rewrite frozen ref-mode/source label strings to the new API names,
    so a byte-frozen pin compares equal to renamed live output."""
    if isinstance(obj, dict):
        return {k: relabel_ref_modes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relabel_ref_modes(v) for v in obj]
    if isinstance(obj, str):
        if obj.startswith(_UNARY_FALLBACK_SOURCE_PREFIX):
            return _LATTICE_STABILITY_SOURCE
        if obj == "omegas_fallback":
            return "omegas_hcp"
        return _REF_MODE_RELABEL.get(obj, obj)
    return obj


# ----------------------------------------------------------------------------------
# Module/function resolution across refactor steps
# ----------------------------------------------------------------------------------


def get_lower_hull_fn():
    """The live lower-hull function: gliquid.hsx.lower_convex_hull once absorbed,
    gliquid.extensive_hull_main.gliq_lowerhull3 before."""
    try:
        from gliquid.hsx import lower_convex_hull

        return lower_convex_hull
    except ImportError:
        from gliquid.extensive_hull_main import gliq_lowerhull3

        return gliq_lowerhull3


def get_solution_mod():
    """gliquid.solution once renamed, gliquid.solution_data before."""
    try:
        import gliquid.solution as m

        return m
    except ImportError:
        import gliquid.solution_data as m

        return m


def get_ternary_mod():
    """gliquid.ternary once renamed, gliquid.hsx_ternary before."""
    try:
        import gliquid.ternary as m

        return m
    except ImportError:
        import gliquid.hsx_ternary as m

        return m


def get_builder_and_symbols():
    """(binary eqs builder, t_sym, binary module) wherever the builder lives.

    Post-RK-refactor there is no standalone binary wrapper; an equivalent callable is
    assembled over SolutionModel.binary_eqs (certified equivalent at rtol<=1e-9)."""
    import gliquid.binary as binary

    try:
        from gliquid.solution import build_thermodynamic_expressions, t_sym

        return build_thermodynamic_expressions, t_sym, binary
    except ImportError:
        try:
            return binary.build_thermodynamic_expressions, binary.t_sym, binary
        except AttributeError:
            from gliquid.solution import DEFAULT_TAU, RKPolyExp, SolutionModel, t_sym

            def build_thermodynamic_expressions(
                param_format="linear", ga_expr=0 * t_sym, gb_expr=0 * t_sym, tau=DEFAULT_TAU
            ):
                return SolutionModel(
                    ("A", "B"), (ga_expr, gb_expr), {(0, 1): RKPolyExp(param_format, tau=tau)}
                ).binary_eqs()

            return build_thermodynamic_expressions, t_sym, binary


def get_x_vals():
    try:
        from gliquid.solution import x_vals

        return np.asarray(x_vals, dtype=float)
    except ImportError:
        from gliquid.binary import _x_vals

        return np.asarray(_x_vals, dtype=float)


def build_combexp_eqs(ga_expr, gb_expr, tau):
    """comb-exp eqs at a given tau via the first-class tau kwarg.

    The pinned tau=3000 values were originally frozen through the retired module-global
    monkeypatch; the kwarg reproducing them at rtol<=1e-9 IS the tau certificate.
    """
    builder, _, _binary = get_builder_and_symbols()
    return builder("comb-exp", ga_expr=ga_expr, gb_expr=gb_expr, tau=tau)


# ----------------------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------------------


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
    if "omega_jmol" in model:  # pre-refactor binary-locked schema
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
