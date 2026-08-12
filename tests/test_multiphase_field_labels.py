"""Two-phase field annotations must not become phantom compounds.

MPDS labels a field between two phases by joining their names with '+', dropping a
component phase's parentheses ('HgZr3 + G', 'Mn2B Mn + rt' for the figure's
'Mn2B + (Mn) rt'), and routinely tags the shape ``nphases: 1``. Since
``identify_mpds_phases`` names a boundary line from ``shape['label'].split()[0]``, the
first constituent was being promoted to a compound sitting at the FIELD's composition.

The two cases below are real and were confirmed against the source figures in
``dev/data/mpds_source_images``: Hg-Zr draws Hg3Zr at 25 at.% Zr and HgZr3 at 75 at.%,
and the 25 at.% line came back named 'HgZr3'; B-Mn draws Mn2B at 66.7 at.% Mn and grew a
second 'Mn2B' at 80 at.%, where the figure prints only a two-phase field annotation.
Across the cache the pattern covers 23 shapes in 18 systems -- these two are merely the
ones whose phantom collided with a correctly named shape and so were visible at all.

Fixtures are synthetic but carry the real labels, compositions and temperature frames.
"""

import pytest

from gliquid import mpds


def _line(label, x, y_lo, y_hi, phase=None):
    """A line compound / boundary: MPDS draws these as a bare vertical line. y is degC."""
    return {
        "nphases": 1,
        "is_solid": True,
        "kind": "compound",
        "label": label,
        "phase": phase,
        "svgpath": f"M {x},{y_lo} L {x},{y_hi}",
    }


def _field(label, points, phase=None):
    """A shape MPDS typed as an area (kind='phase'); it would carry cbounds."""
    return {
        "nphases": 1,
        "is_solid": True,
        "kind": "phase",
        "label": label,
        "phase": phase,
        "svgpath": "M " + " L ".join(f"{x},{y}" for x, y in points),
    }


def _json(shapes, temp=(-200.0, 2000.0), labels=(), elements=("Hg", "Zr")):
    return {
        "reference": {"entry": "synthetic"},
        "chemical_elements": list(elements),
        "temp": list(temp),
        "comp_range": [0.0, 100.0],
        "shapes": list(shapes),
        "labels": [list(entry) for entry in labels],
    }


class TestLabelRecognition:
    @pytest.mark.parametrize(
        "label",
        [
            "HgZr3 + G",  # Hg-Zr: compound + gas
            "Mn2B Mn + rt",  # B-Mn:  the figure's 'Mn2B + (Mn) rt'
            "CuSe2 rt + L",  # Cu-Se: solid + liquid
            "Au3Zn Zn + rt",  # Au-Zn: one of a flanking pair
            "Pt3Ga Pt + rt",  # Ga-Pt
            "Pd1.5Sn Sn + rt",  # Pd-Sn
            "(Zr) rt + G",  # a component phase + gas
        ],
    )
    def test_field_annotations_are_recognized(self, label):
        assert mpds.is_multiphase_field_label(label) is True

    @pytest.mark.parametrize(
        "label",
        [
            "HgZr3",
            "Mn2B tet",
            "CeC2 rt",
            "(Ce) ht1",
            "(Zr,Ti) rt",
            "Cu5.4Gd0.8 ht",
            "HfFe2 hex1",
            "Th2Co7 rt",
        ],
    )
    def test_ordinary_phase_names_are_not(self, label):
        assert mpds.is_multiphase_field_label(label) is False

    def test_empty_label_degrades_quietly(self):
        assert mpds.is_multiphase_field_label("") is False
        assert mpds.is_multiphase_field_label(None) is False


class TestHgZr:
    """Hg-Zr: Hg3Zr at 25 at.% Zr wore the 'HgZr3 + G' field text (mpds.io/entry/C100194)."""

    @staticmethod
    def _data():
        return _json(
            [
                _line("HgZr3", 74.9905, -200, 559.909, "HgZr3/223/cP8"),
                _line("HgZr", 50.0, -200, 423.4, "HgZr/123/tP2"),
                _line("HgZr3 + G", 24.9905, -200, 404.916, "HgZr3/223/cP8"),
            ]
        )

    def test_the_field_annotated_shape_is_dropped(self):
        comps = [round(p["comp"], 4) for p in mpds.identify_mpds_phases(self._data())]
        assert 0.2499 not in comps

    def test_the_real_compound_survives_at_its_own_stoichiometry(self):
        phases = mpds.identify_mpds_phases(self._data())
        hgzr3 = [p for p in phases if p["name"] == "HgZr3"]
        assert len(hgzr3) == 1
        assert hgzr3[0]["comp"] == pytest.approx(0.75, abs=1e-3)

    def test_no_name_is_claimed_at_two_compositions(self):
        phases = mpds.identify_mpds_phases(self._data())
        names = [p["name"] for p in phases]
        assert len(names) == len(set(names))


class TestBMn:
    """B-Mn: an 'Mn2B' appeared at 80 at.% Mn beside the real one at 66.7 (entry C900298)."""

    @staticmethod
    def _data():
        return _json(
            [
                _line("Mn2B Mn + rt", 80.0038, 599.595, 1120.22),
                _line("Mn2B tet", 66.6413, 599.189, 1580.03, "Mn2B/140/tI12"),
                _line("MnB", 50.0, 599.595, 1889.85),
            ],
            temp=(600.0, 2200.0),
            elements=("B", "Mn"),
        )

    def test_only_the_real_compound_is_identified(self):
        phases = mpds.identify_mpds_phases(self._data(), elements=("B", "Mn"))
        mn2b = [p for p in phases if p["name"] == "Mn2B"]
        assert len(mn2b) == 1
        assert mn2b[0]["comp"] == pytest.approx(2 / 3, abs=1e-3)

    def test_the_polymorph_key_still_rides_when_requested(self):
        """The guard must not disturb the opt-in key set on the shapes it keeps."""
        phases = mpds.identify_mpds_phases(self._data(), elements=("B", "Mn"), with_polymorph=True)
        mn2b = next(p for p in phases if p["name"] == "Mn2B")
        assert mn2b["polymorph"] == "tet"


class TestAreaShapesToo:
    """Au-Zn, Ga-Pt, Pd-Sn and Cu-Pd carry the pattern on kind='phase' area shapes."""

    def test_a_field_annotated_area_shape_is_dropped(self):
        data = _json(
            [
                _field("Au3Zn Zn + rt", [(26.0, 100), (30.0, 100), (30.0, 400), (26.0, 400)]),
                _field("Au3Zn Au + rt", [(20.0, 100), (24.0, 100), (24.0, 400), (20.0, 400)]),
                _line("Au3Zn", 25.0, 100, 664.0),
            ],
            temp=(0.0, 1100.0),
            elements=("Au", "Zn"),
        )
        phases = mpds.identify_mpds_phases(data, elements=("Au", "Zn"))
        assert [p["name"] for p in phases] == ["Au3Zn"]
        assert phases[0]["type"] == "lc"


class TestWhatMustNotChange:
    def test_a_diagram_without_field_labels_is_untouched(self):
        """The guard is a filter, not a rewrite: unaffected shapes keep their exact dicts."""
        data = _json(
            [
                _line("CeC2 rt", 33.3016, 218.246, 2249.67, "CeC2/139/tI6"),
                _line("Ce2C3", 39.9924, 199.392, 1899.34, "Ce2C3/220/cI40"),
            ],
            temp=(200.0, 2600.0),
            elements=("C", "Ce"),
        )
        phases = mpds.identify_mpds_phases(data)
        assert [p["name"] for p in phases] == ["CeC2", "Ce2C3"]
        assert set(phases[0]) == {"type", "name", "comp", "tbounds"}

    def test_component_phases_are_still_typed_comp(self):
        data = _json([_line("(Zr) rt", 98.0, -200, 863.0)], elements=("Hg", "Zr"))
        phases = mpds.identify_mpds_phases(data, elements=("Hg", "Zr"))
        assert [p["type"] for p in phases] == ["comp"]


class TestCollapseLabelBlock:
    """The labels block goes through the same rule, so the two blocks stay joinable."""

    def test_a_field_annotation_does_not_seed_a_thermal_form(self):
        # Threshold for temp=(200, 2600) is 713.15 K == 440 degC; the dome bottom sits
        # above it, so an 'rt' form named in the labels block would pull tbounds down.
        data = _json(
            [_line("XY2 ht", 33.33, 600.0, 1800.0)],
            temp=(200.0, 2600.0),
            elements=("X", "Y"),
            labels=[["XY2 X + rt", [33.0, 100.0]]],
        )
        collapsed = mpds.collapse_polymorphs(
            mpds.identify_mpds_phases(data, with_polymorph=True), data
        )
        xy2 = next(p for p in collapsed if p["name"] == "XY2")
        assert xy2["tbounds"][0][1] == pytest.approx(873.15)
        assert "distinct_melting_polymorph" not in xy2

    def test_a_genuine_rt_label_still_seeds_one(self):
        """Guard rails only the '+' case -- the C-La recovery must keep working."""
        data = _json(
            [_line("XY2 ht", 33.33, 600.0, 1800.0)],
            temp=(200.0, 2600.0),
            elements=("X", "Y"),
            labels=[["XY2 rt", [33.0, 100.0]]],
        )
        collapsed = mpds.collapse_polymorphs(
            mpds.identify_mpds_phases(data, with_polymorph=True), data
        )
        xy2 = next(p for p in collapsed if p["name"] == "XY2")
        assert xy2["tbounds"][0][1] == pytest.approx(373.15)
        assert xy2["distinct_melting_polymorph"] is True
