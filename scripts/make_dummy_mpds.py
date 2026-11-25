"""Write the synthetic MPDS-format phase diagrams the test suite reads.

Every diagram in ``tests/fixtures/mpds_dummy/`` is made here. A diagram's liquidus is the
liquidus of a gliquid ``BinaryLiquid`` on the cached DFT hull at the fixed parameters in
``SPECS`` (Hf-Zr's is the closed-form curve of ``hf_zr_liquidus``), and its solid fields,
two-phase fields and labels are drawn around that liquidus from the hand-written field lists
below. Entry ids read ``dummy-<system>`` and every
``reference`` names ``SYNTHETIC_REFERENCE``. ``PLACEHOLDERS`` are written as the
``{"reference": null}`` record of a system with no diagram.

No MPDS record is read: ``mpds.load_mpds_data`` is replaced by a stub that returns only the
temperature frame.

A ``fields`` entry of ``SPECS`` is one of (compositions as fractions of the second element,
temperatures in degC):

* ``("end", label, phase, x_solvus_top, x_solvus_bottom)``: a terminal field on the axis the
  label names, its solidus running from the melting point to ``x_solvus_top`` at the
  neighbouring liquidus minimum;
* ``("line", label, phase, x, t_top)``: a line compound from the axis bottom to ``t_top``, or
  to the liquidus when ``t_top`` is None;
* ``("poly", label, phase, [[x_pct, t_c], ...])``: a field drawn as the given polygon.

Run from the repository root::

    python scripts/make_dummy_mpds.py            # rewrite every diagram
    python scripts/make_dummy_mpds.py --check    # exit 1 when a written diagram is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _mpds_builders import (  # noqa: E402
    bare_field_shape,
    bare_line_shape,
    field_shape,
    line_shape,
    liquid_shape,
    svgpath,
)

import gliquid.config as config  # noqa: E402
import gliquid.mpds as mpds  # noqa: E402
from gliquid.binary import BinaryLiquid  # noqa: E402
from gliquid.phase import UNARY  # noqa: E402

OUT_DIR = ROOT / "tests" / "fixtures" / "mpds_dummy"
SYNTHETIC_REFERENCE = "synthetic:gliquid-test-diagram"
SCHEMA = "https://developer.mpds.io/mpds.schema.json#/definitions/phase_diagram"

SPECS = {
    "Cu-Mg": {
        "flavour": "website",
        "temp": [400.0, 1100.0],
        "params": [-41302.26, -5.1627, -8148.05, 0.0],
        "fields": [
            ("end", "(Cu)", "Cu/225/cF4", 0.035, 0.012),
            ("line", "MgCu2", "MgCu2/227/cF24", 1 / 3, None),
            ("line", "Mg2Cu", "Mg2Cu/70/oF48", 2 / 3, None),
            ("end", "(Mg)", "Mg/194/hP2", 0.99, 0.996),
        ],
    },
    "Ag-V": {
        "flavour": "website",
        "temp": [0.0, 3000.0],
        "params": [54923.70, 6.8246, 608.58, 0.0],
        "fields": [
            ("end", "(Ag)", "Ag/225/cF4", 0.004, 0.001),
            ("end", "(V)", "V/229/cI2", 0.985, 0.995),
        ],
        "two_liquid_label": "L<sub>1</sub> + L<sub>2</sub>",
    },
    "Al-Cu": {
        "flavour": "website",
        "temp": [300.0, 1100.0],
        "params": [-68087.2, -3.6819, 2945.42, 0.0],
        "fields": [
            ("end", "(Al)", "Al/225/cF4", 0.025, 0.004),
            ("line", "CuAl2", "CuAl2/140/tI12", 1 / 3, None),
            ("line", "AlCu rt", "AlCu/12/mC20", 0.50, 560.0),
            (
                "poly",
                "AlCu ht",
                "AlCu/74/oI16",
                [[51.0, 560.0], [51.0, 640.0], [53.0, 640.0], [53.0, 560.0]],
            ),
            ("line", "Al2Cu3", "Al2Cu3/164/hP5", 0.60, 700.0),
            ("line", "Al4Cu9", "Cu9Al4/215/cP52", 9 / 13, 780.0),
            ("line", "AlCu3", "Cu3Al/229/cI16", 0.75, None),
            (
                "poly",
                "(Cu)",
                "Cu/225/cF4",
                [[100.0, 300.0], [100.0, 1084.62], [89.5, 872.0], [84.0, 300.0]],
            ),
        ],
    },
    "C-Nb": {
        "flavour": "website",
        "temp": [700.0, 4700.0],
        "params": [-90000.0, -15.0, -20000.0, 0.0],
        "fields": [
            ("end", "(C)", "C/194/hP4", 0.004, 0.001),
            (
                "poly",
                "NbC",
                "NbC/225/cF8",
                [[50.0, 700.0], [50.0, 3500.0], [55.0, 4121.0], [58.0, 3200.0], [56.0, 700.0]],
            ),
            ("line", "Nb2C", "Nb2C/60/oP12", 2 / 3, 2900.0),
            ("end", "(Nb)", "Nb/229/cI2", 0.95, 0.985),
        ],
    },
    "Ru-Si": {
        "flavour": "api",
        "temp": [400.0, 2400.0],
        "params": [-150000.0, 0.0, 0.0, 0.0],
        "fields": [
            ("end", "(Ru)", None, 0.02, 0.005),
            ("line", "RuSi", None, 0.5, None),
            ("line", "Ru2Si3", None, 0.6, 1700.0),
            ("line", "RuSi2", None, 2 / 3, 1600.0),
            ("end", "(Si)", None, 0.995, 0.999),
        ],
    },
    "Ga-Si": {
        "flavour": "api",
        "temp": [0.0, 1600.0],
        "params": [10000.0, 0.0, 0.0, 0.0],
        "fields": [],
    },
    "Cr-Eu": {
        "flavour": "website",
        "temp": [600.0, 2000.0],
        "params": [40000.0, 0.0, 0.0, 0.0],
        "fields": [],
    },
    "Hf-Zr": {"flavour": "api", "temp": [800.0, 2400.0], "fields": []},
}

# Hf-Zr is drawn in closed form: a monotone liquidus over a full-width solid solution, sampled
# off the round compositions.
HF_ZR_XS = [round(0.035 * k, 4) for k in range(29)] + [1.0]
HF_T_ALPHA_BETA, ZR_T_ALPHA_BETA = 1743.0, 863.0


def hf_zr_liquidus(x):
    return 2232.85 - 379.0 * x - 120.0 * x * (1 - x)


def hf_zr_solidus(x):
    return hf_zr_liquidus(x) - 90.0 * x * (1 - x)


def hf_zr_shapes(spec):
    """L, the '(Hf, Zr) ht' and '(Hf, Zr) rt' fields and the two two-phase fields."""
    t_lo, t_hi = spec["temp"]
    liquidus = [[x, hf_zr_liquidus(x)] for x in HF_ZR_XS]
    solidus = [[x, hf_zr_solidus(x)] for x in HF_ZR_XS]

    def transus(x, drop=0.0):
        return HF_T_ALPHA_BETA + (ZR_T_ALPHA_BETA - HF_T_ALPHA_BETA) * x - drop * x * (1 - x)

    beta_floor = [[x, transus(x)] for x in (1.0, 0.75, 0.5, 0.25, 0.0)]
    alpha_top = [[x, transus(x, 160.0)] for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    shapes = [
        liquid_shape(pct(liquidus + [[1.0, t_hi], [0.0, t_hi]])),
        bare_field_shape("(Hf, Zr) ht", pct(solidus + beta_floor)),
        bare_field_shape("(Hf, Zr) rt", pct([[0.0, t_lo]] + alpha_top + [[1.0, t_lo]])),
        _two_phase(liquidus + solidus[::-1]),
        _two_phase(beta_floor + alpha_top),
    ]
    labels = [
        ["L", [50.0, 2300.0], None],
        ["(Hf, Zr) ht", [50.0, 1700.0], None],
        ["(Hf, Zr) rt", [50.0, 1000.0], None],
    ]
    return liquidus, shapes, labels


def _frame_stub(temp):
    def load_mpds_data(system, pd_ind=None):
        return {"temp": list(temp), "comp_range": [0, 100], "reference": None}, (None, False)

    return load_mpds_data


def model_liquidus(system, temp, params):
    """``[[x, T_C], ...]`` of a ``BinaryLiquid``'s liquidus at ``params`` on the cached hull."""
    real = mpds.load_mpds_data
    mpds.load_mpds_data = _frame_stub(temp)
    try:
        bl = BinaryLiquid.from_cache(system, param_format="comb-exp", solid_solutions=False)
        bl.update_params(list(params))
        points = sorted((float(x), float(t) - 273.15) for x, t in bl.phases[-1].points)
    finally:
        mpds.load_mpds_data = real
    return [[x, t] for x, t in points]


def interp(liquidus, x):
    xs, ts = zip(*liquidus)
    return float(np.interp(x, xs, ts))


def extrema(liquidus):
    """Interior ``(kind, index)`` pairs of the sampled liquidus, in composition order."""
    found = []
    for i in range(1, len(liquidus) - 1):
        prev, here, nxt = liquidus[i - 1][1], liquidus[i][1], liquidus[i + 1][1]
        if here > prev and here >= nxt:
            found.append(("max", i))
        elif here < prev and here <= nxt:
            found.append(("min", i))
    return found


def pct(points):
    return [[round(100 * x, 4), round(t, 2)] for x, t in points]


def build_shapes(spec, elements, liquidus):
    """Liquid, solid and two-phase shapes plus labels around ``liquidus``."""
    t_lo, t_hi = spec["temp"]
    mins = [i for kind, i in extrema(liquidus) if kind == "min"]
    shapes = [liquid_shape(pct(liquidus + [[1.0, t_hi], [0.0, t_hi]]))]
    labels = [["L", [50.0, round(t_hi - 0.08 * (t_hi - t_lo), 2)], None]]
    first_min = liquidus[mins[0]][1] if mins else t_lo + 0.5 * (t_hi - t_lo)
    last_min = liquidus[mins[-1]][1] if mins else first_min
    solids = []
    bare = spec["flavour"] == "api"
    for entry in spec["fields"]:
        kind, label, phase = entry[:3]
        field, line = (bare_field_shape, bare_line_shape) if bare else (field_shape, line_shape)
        if kind == "end":
            x_top, x_bottom = entry[3], entry[4]
            left = x_top < 0.5
            end_x = 0.0 if left else 1.0
            t_melt = UNARY[elements[0 if left else 1]].t_fusion - 273.15
            t_inv = first_min if left else last_min
            ring = [[end_x, t_lo], [end_x, t_melt], [x_top, t_inv], [x_bottom, t_lo]]
            solids.append((min(x_top, x_bottom, end_x), max(x_top, x_bottom, end_x), t_inv))
            shapes.append(field(label, pct(ring), phase))
            labels.append([label, [round(100 * (end_x + x_top) / 2, 4), t_lo + 20.0], None])
        elif kind == "line":
            x, t_top = entry[3], entry[4]
            t_top = interp(liquidus, x) if t_top is None else t_top
            shapes.append(line(label, round(100 * x, 4), t_lo, round(t_top, 2), phase))
            solids.append((x, x, t_top))
            labels.append([label, [round(100 * x, 4), round(t_lo + 0.3 * (t_top - t_lo), 2)], None])
        else:
            ring = entry[3]
            shapes.append(field(label, [[round(x, 4), round(t, 2)] for x, t in ring], phase))
            xs = [x / 100 for x, _ in ring]
            solids.append((min(xs), max(xs), max(t for _, t in ring)))
            labels.append([label, [round(sum(x for x, _ in ring) / len(ring), 4), t_lo + 20], None])
    shapes += two_phase_fields(liquidus, mins, sorted(solids), t_lo)
    return shapes, labels


def two_phase_fields(liquidus, mins, solids, t_lo):
    """Unlabelled two-phase fields: L + solid under each liquidus branch, solid + solid below
    each liquidus minimum."""
    fields = []
    bounds = [0] + mins + [len(liquidus) - 1]
    for a, b in zip(bounds, bounds[1:]):
        branch = liquidus[a : b + 1]
        floor = min(branch[0][1], branch[-1][1])
        top = max(range(len(branch)), key=lambda k: branch[k][1])
        for part in (branch[: top + 1], branch[top:]):
            if len(part) < 2:
                continue
            x_peak = branch[top][0]
            ring = (
                part + [[x_peak, floor]] if part[-1][0] != x_peak else part + [[part[0][0], floor]]
            )
            fields.append(_two_phase(ring))
    for i in mins:
        x_e, t_e = liquidus[i]
        left = max((s for s in solids if s[1] <= x_e), key=lambda s: s[1], default=None)
        right = min((s for s in solids if s[0] >= x_e), key=lambda s: s[0], default=None)
        if left is None or right is None:
            continue
        x_a, x_b = left[1], right[0]
        fields.append(_two_phase([[x_a, t_lo], [x_a, t_e], [x_b, t_e], [x_b, t_lo]]))
    return fields


def _two_phase(points):
    return {"nphases": 2, "is_solid": True, "kind": "phase", "svgpath": svgpath(pct(points))}


def header(system, spec):
    elements = system.split("-")
    entry = f"dummy-{system}"
    if spec["flavour"] == "api":
        return {
            "chemical_elements": elements,
            "entry": entry,
            "comp_range": [0.0, 100.0],
            "temp": list(spec["temp"]),
            "reference": {"entry": SYNTHETIC_REFERENCE},
            "unlocked": False,
        }
    return {
        "$schema": SCHEMA,
        "version": "1.2.0",
        "object_type": "C",
        "use_visavis_type": "pd",
        "diatype": "binary",
        "arity": 2,
        "naxes": 2,
        "entry": entry,
        "title_a": elements[0],
        "title_b": elements[1],
        "comp_start": {elements[0]: 1, elements[1]: 0},
        "comp_end": {elements[0]: 0, elements[1]: 1},
        "comp_range": [0.0, 100.0],
        "chemical_elements": elements,
        "temp": list(spec["temp"]),
        "reference": {"entry": SYNTHETIC_REFERENCE},
    }


def build(system):
    spec = SPECS[system]
    elements = system.split("-")
    if system == "Hf-Zr":
        liquidus, shapes, labels = hf_zr_shapes(spec)
    else:
        liquidus = model_liquidus(system, spec["temp"], spec["params"])
        shapes, labels = build_shapes(spec, elements, liquidus)
    record = header(system, spec)
    if spec.get("two_liquid_label"):
        peak = max(liquidus, key=lambda p: p[1])
        labels.append(
            [spec["two_liquid_label"], [round(100 * peak[0], 4), round(peak[1] - 150, 2)], None]
        )
    if spec["flavour"] == "website":
        for shape in shapes:
            if "phase" in shape:
                shape["chemical_elements"] = list(elements)
    record["labels"] = labels
    record["shapes"] = shapes
    return record


#: Systems whose diagram is the ``{"reference": null}`` placeholder of a system MPDS has none for.
PLACEHOLDERS = ("Ga-Ru",)


def render(record) -> str:
    return json.dumps(record, indent=1) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("systems", nargs="*", help="systems to write (default: all)")
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    config.set_offline(True)
    stale = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for system in args.systems or [*SPECS, *PLACEHOLDERS]:
        text = render({"reference": None} if system in PLACEHOLDERS else build(system))
        path = OUT_DIR / f"{system}.json"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(system)
            continue
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print("stale: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
