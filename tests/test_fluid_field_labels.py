"""A liquid or gas field tagged ``is_solid`` must not become a compound named 'L'.

MPDS normally draws the liquid field as an area (``kind: 'phase'``, ``is_solid: false``).
Some diagrams draw it as a STROKE instead and flag it solid, in two forms:

* a terminal liquid sliver too narrow to fill, emitted as a 2- or 3-point line close to a
  pure-element axis, often ending on that component's melting point -- a real liquid
  boundary wearing the wrong flag;
* a frame edge: a 2-point vertical just past x = 100 at.% spanning the whole temperature
  frame, which captured a nearby printed 'L' text.

``identify_mpds_phases`` names a stroke from ``split_phase_label(...)[0]``; none of these
may come back as a line compound called 'L'. ``liquid_shape_paths`` refuses them on the
liquidus side with a ``kind in (None, 'phase')`` filter, which cannot serve the phase-list
gate (every genuine line compound is ``kind: 'compound'``): the label is the discriminator.

Fixtures are synthetic strokes of both forms, on the element pairs and temperature frames
such diagrams use.
"""

import pytest

from gliquid import mpds

from ._mpds_builders import line_shape as _line
from ._mpds_builders import mpds_record


def _stroke(label, svgpath, phase=None):
    """A liquid field MPDS drew as a line and flagged solid -- the shape under test."""
    return {
        "nphases": 1,
        "is_solid": True,
        "kind": "compound",
        "label": label,
        "phase": phase,
        "svgpath": svgpath,
    }


def _json(shapes, temp, elements):
    return mpds_record(shapes, elements, temp)


# (system, svgpath, temp frame, elements): two- and three-point slivers near either axis, one
# ending on the low-melting component's melting point, and a full-height frame edge past 100.
HG_PD_STROKE = "M 0.25,250 L 0.1,100 L 0,-38.8"
STROKES = [
    ("Er-Nb", "M 98.6,2420 L 98.5,2499", (500.0, 2500.0), ("Er", "Nb")),
    ("Fe-Sn", "M 99.6,800 L 99.7,600 L 99.8,450", (400.0, 1600.0), ("Fe", "Sn")),
    ("Hg-Pd", HG_PD_STROKE, (-100.0, 600.0), ("Hg", "Pd")),
    ("Li-Mo", "M 100.02,5000 L 100.02,0", (0.0, 5000.0), ("Li", "Mo")),
    ("Tl-V", "M 0.15,1900 L 0,304", (0.0, 3000.0), ("Tl", "V")),
]


class TestLabelRecognition:
    @pytest.mark.parametrize("label", ["L", "G", "L rt", "G ht1"])
    def test_fluid_field_labels_are_recognized(self, label):
        assert mpds.is_fluid_phase_label(label) is True

    @pytest.mark.parametrize(
        "label",
        [
            # gallium and gadolinium are the near misses the base-name match must survive
            "Ga",
            "GaSb",
            "Ga2Se3",
            "(Ga)",
            "Gd",
            "(Gd)",
            "Cu5.4Gd0.8 ht",
            "GaHo121",
            # ordinary phases
            "HgZr3",
            "CeC2 rt",
            "(Zr,Ti) rt",
            "W2Ge3 rt",
        ],
    )
    def test_real_phase_names_are_not(self, label):
        assert mpds.is_fluid_phase_label(label) is False

    def test_empty_label_degrades_quietly(self):
        assert mpds.is_fluid_phase_label("") is False
        assert mpds.is_fluid_phase_label(None) is False

    @pytest.mark.parametrize("label", ["CuSe2 rt + L", "(Zr) rt + G"])
    def test_a_solid_plus_fluid_field_is_the_other_guard_s_job(self, label):
        """Base name is the solid, so only is_multiphase_field_label claims these."""
        assert mpds.is_fluid_phase_label(label) is False
        assert mpds.is_multiphase_field_label(label) is True

    def test_l_plus_g_is_claimed_by_both_guards(self):
        """'L + G' splits to base 'L', so the two guards overlap on it;
        is_multiphase_field_label runs first and drops it with the more specific reason."""
        assert mpds.is_multiphase_field_label("L + G") is True
        assert mpds.is_fluid_phase_label("L + G") is True

    def test_a_fluid_field_annotation_is_dropped_however_it_is_spelled(self):
        data = _json(
            [_line("L + G", 42.76, 800.0, 3400.0), _line("MoSi2", 66.6, 400.0, 2020.0)],
            (0.0, 5000.0),
            ("Li", "Mo"),
        )
        phases = mpds.identify_mpds_phases(data, elements=("Li", "Mo"))
        assert [p["name"] for p in phases] == ["MoSi2"]


class TestTheFiveRealShapes:
    @pytest.mark.parametrize("system,svgpath,temp,elements", STROKES, ids=[r[0] for r in STROKES])
    def test_no_compound_named_l_is_emitted(self, system, svgpath, temp, elements):
        data = _json([_stroke("L", svgpath)], temp, elements)
        phases = mpds.identify_mpds_phases(data, elements=elements)
        assert [p["name"] for p in phases] == []

    def test_the_phantom_does_not_crowd_out_real_compounds(self):
        """Hg-Pd: the stroke goes, its four genuine neighbours stay."""
        data = _json(
            [
                _stroke("L", HG_PD_STROKE),
                _line("Hg4Pd", 20.0, -99.8, 100.0),
                _line("Hg5Pd2", 28.6, -99.8, 250.0),
                _line("HgPd", 50.0, -99.8, 480.0),
                _line("(Pd)", 86.0, -99.8, 400.0),
            ],
            (-100.0, 600.0),
            ("Hg", "Pd"),
        )
        phases = mpds.identify_mpds_phases(data, elements=("Hg", "Pd"))
        assert [p["name"] for p in phases] == ["Hg4Pd", "Hg5Pd2", "HgPd", "(Pd)"]
        assert phases[-1]["type"] == "comp"


class TestWhatMustNotChange:
    def test_a_liquid_field_drawn_as_an_area_never_reached_this_gate(self):
        """The normal spelling is is_solid=false, so the guard is not what excludes it."""
        area = {
            "nphases": 1,
            "is_solid": False,
            "kind": "phase",
            "label": "L",
            "svgpath": "M 0,900 L 50,1200 L 100,1500",
        }
        data = _json([area], (0.0, 2000.0), ("Ge", "W"))
        assert mpds.identify_mpds_phases(data, elements=("Ge", "W")) == []

    def test_a_liquid_area_is_still_the_liquidus_source(self):
        """Nothing here may narrow liquid_shape_paths, which reads label=='L' by design."""
        area = {
            "nphases": 1,
            "is_solid": False,
            "kind": "phase",
            "label": "L",
            "svgpath": "M 0,900 L 50,1200 L 100,1500",
        }
        data = _json([area], (0.0, 2000.0), ("Ge", "W"))
        assert mpds.liquid_shape_paths(data) == [area["svgpath"]]

    def test_a_gallium_line_compound_survives_at_its_own_stoichiometry(self):
        data = _json(
            [_line("GaSb", 50.0, 100.0, 712.0), _line("Ga2Se3", 60.0, 100.0, 1005.0)],
            (0.0, 1200.0),
            ("Ga", "Sb"),
        )
        phases = mpds.identify_mpds_phases(data, elements=("Ga", "Sb"))
        assert [p["name"] for p in phases] == ["GaSb", "Ga2Se3"]
