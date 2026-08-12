"""Native 'drawing' shapes (tie/construction lines) carry no ``nphases`` key.

``identify_invariant_points``'s miscibility-gap loop indexed ``shape['nphases']``
directly and crashed with KeyError on such shapes (Mn-Pb, Be-Li, Lu-V). The
path was latent until multi-L extraction started returning a liquidus for
disjoint-L diagrams, which made the misc-gap branch reachable for exactly the
systems that draw tie lines without phase counts.
"""

from types import SimpleNamespace

import gliquid.mpds as mpds


def _misc_gap_json():
    return {
        "chemical_elements": ["A", "B"],
        "comp_range": [0.0, 100.0],
        "temp": [400.0, 1300.0],
        "reference": {"entry": "https://mpds.io/entry/C000001"},
        "labels": [
            ["L", [50.0, 1250.0], None],
            ["L<sub>1</sub> + L<sub>2</sub>", [50.0, 800.0], None],
        ],
        "shapes": [
            {
                "kind": "phase",
                "label": "L",
                "nphases": 1,
                "is_solid": False,
                "svgpath": "M 0,700 L 50,900 L 100,700 L 100,1300 L 0,1300 Z",
            },
            # the regression: native construction line with NO nphases key
            {"kind": "drawing", "svgpath": "M 0,600 L 100,600"},
        ],
    }


def test_misc_gap_loop_survives_drawing_shape_without_nphases():
    liq = [
        [0.0, 1173.0],
        [0.25, 1373.0],
        [0.5, 1473.0],
        [0.75, 1373.0],
        [1.0, 1173.0],
    ]  # interior maximum -> misc-gap branch runs
    comp_data = {"A": SimpleNamespace(t_fusion=1173.0), "B": SimpleNamespace(t_fusion=1173.0)}
    out = mpds.identify_invariant_points(
        _misc_gap_json(), ["A", "B"], liq, comp_data, [673.0, 1573.0]
    )
    assert isinstance(out, tuple) and len(out) == 3
