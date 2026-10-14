"""How a printed two-liquid label becomes a ``mig`` invariant record.

``identify_invariant_points`` seeds each record from a label, takes the apex from the
nearest interior liquidus maximum, the feet from a two-phase polygon topping out there or
else from the walk off the adjacent monotectic minimum, and marks the record ``partial``
(apex unknown) when a liquidus hole lies between the feet or the polygon reaches the frame
top. Synthetic diagrams throughout; the frame is A-B unless a case reverses it.
"""

from types import SimpleNamespace

import pytest

import gliquid.mpds as mpds

COMP_DATA = {"A": SimpleNamespace(t_fusion=1173.0), "B": SimpleNamespace(t_fusion=1173.0)}
TEMP_RANGE = [673.15, 1573.15]


def _json(labels, shapes, elements=("A", "B")):
    return {
        "chemical_elements": list(elements),
        "comp_range": [0.0, 100.0],
        "temp": [400.0, 1300.0],
        "reference": {"entry": "https://mpds.io/entry/C000001"},
        "labels": [[text, [x, t], None] for text, x, t in labels],
        "shapes": [
            {
                "kind": "phase",
                "label": "L",
                "nphases": 1,
                "is_solid": False,
                "svgpath": "M 0,900 L 100,900 L 100,1300 L 0,1300 Z",
            },
            *(
                {"kind": "phase", "label": None, "nphases": 2, "is_solid": True, "svgpath": path}
                for path in shapes
            ),
        ],
    }


def _svg(points):
    return "M " + " L ".join(f"{x * 100:g},{t - 273.15:g}" for x, t in points)


# A complete dome: feet at 0.2 and 0.8 on a 1000 K monotectic, apex (0.5, 1200 K), and the
# digitized liquidus runs over the dome.
DOME_LIQ = [
    [0.0, 1173.0],
    [0.1, 1080.0],
    [0.2, 1000.0],
    [0.3, 1120.0],
    [0.4, 1180.0],
    [0.5, 1200.0],
    [0.6, 1180.0],
    [0.7, 1120.0],
    [0.8, 1000.0],
    [0.9, 1080.0],
    [1.0, 1173.0],
]
DOME_POLY = _svg([[0.2, 1000.0], [0.5, 1200.0], [0.8, 1000.0]])
LABEL = ("L<sub>1</sub> + L<sub>2</sub>", 50.0, 800.0)


def _records(json_, liq, **kwargs):
    invariants, _, _ = mpds.identify_invariant_points(
        json_, ["A", "B"], [list(p) for p in liq], COMP_DATA, TEMP_RANGE, **kwargs
    )
    return [iv for iv in invariants if iv["type"] == "mig"]


class TestCompleteDome:
    def test_polygon_supplies_feet_and_apex(self):
        (rec,) = _records(_json([LABEL], [DOME_POLY]), DOME_LIQ)
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.8])
        assert rec["cbounds"][0][1] == pytest.approx(1000.0)
        assert rec["apex"] == pytest.approx([0.5, 1200.0])
        assert rec["partial"] is False
        assert rec["source"] == "label"
        assert mpds.mig_apex(rec) == pytest.approx([0.5, 1200.0])

    def test_record_keeps_the_legacy_keys(self):
        (rec,) = _records(_json([LABEL], [DOME_POLY]), DOME_LIQ)
        assert {"type", "comp", "cbounds", "tbounds", "phases", "phase_comps"} <= set(rec)
        assert rec["comp"] == pytest.approx(0.5)
        assert rec["tbounds"][1] == pytest.approx([0.5, 1200.0])

    def test_label_spellings(self):
        for text in ("L1 + L2", "L1+L2", "L' + L''", "L<sub>1</sub> + L<sub>2</sub>"):
            assert mpds.is_two_liquid_label(text), text
        for text in ("L", "L + (A)", "(A, B)", "L1 + AB", "", None):
            assert not mpds.is_two_liquid_label(text), text

    def test_label_without_an_interior_maximum_yields_nothing(self):
        monotone = [[0.0, 1173.0], [0.5, 1000.0], [1.0, 900.0]]
        assert _records(_json([LABEL], [DOME_POLY]), monotone) == []


class TestPartialDome:
    HOLE_LIQ = [
        [0.0, 1173.0],
        [0.1, 1080.0],
        [0.2, 1000.0],
        [0.25, 1150.0],
        [0.75, 1150.0],
        [0.8, 1000.0],
        [0.9, 1080.0],
        [1.0, 1173.0],
    ]
    FLANK_POLY = _svg([[0.2, 1000.0], [0.75, 1150.0], [0.8, 1000.0]])

    def test_hole_between_the_feet_hides_the_apex(self):
        (rec,) = _records(
            _json([LABEL], [self.FLANK_POLY]), self.HOLE_LIQ, liquidus_holes=[[0.25, 0.75]]
        )
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.8])
        assert rec["partial"] is True
        assert rec["apex"] is None
        assert mpds.mig_apex(rec) is None

    def test_unknown_holes_leave_the_dome_complete(self):
        (rec,) = _records(_json([LABEL], [self.FLANK_POLY]), self.HOLE_LIQ)
        assert rec["partial"] is False
        assert rec["apex"] == pytest.approx([0.75, 1150.0])

    def test_polygon_at_the_frame_top_hides_the_apex(self):
        liq = [list(p) for p in DOME_LIQ]
        liq[5] = [0.5, TEMP_RANGE[1]]
        liq[4] = [0.4, 1400.0]
        liq[6] = [0.6, 1400.0]
        poly = _svg([[0.2, 1000.0], [0.5, TEMP_RANGE[1]], [0.8, 1000.0]])
        (rec,) = _records(_json([LABEL], [poly]), liq)
        assert rec["partial"] is True
        assert rec["apex"] is None
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.8])

    def test_complete_dome_is_not_partial(self):
        (rec,) = _records(_json([LABEL], [DOME_POLY]), DOME_LIQ, liquidus_holes=[])
        assert rec["partial"] is False


class TestFeet:
    def test_polygon_extremes_are_the_feet(self):
        poly = _svg([[0.15, 1100.0], [0.2, 1000.0], [0.5, 1200.0], [0.8, 1000.0], [0.85, 1100.0]])
        (rec,) = _records(_json([LABEL], [poly]), DOME_LIQ)
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.15, 0.85])
        assert not mpds.mig_feet_on_liquidus(rec, [list(p) for p in DOME_LIQ], 20.0)

    def test_feet_on_the_isotherm_are_genuine(self):
        (rec,) = _records(_json([LABEL], [DOME_POLY]), DOME_LIQ)
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.8])
        assert mpds.mig_feet_on_liquidus(rec, [list(p) for p in DOME_LIQ], 20.0)


class TestLabels:
    TWO_LIQ = [
        [0.0, 1173.0],
        [0.05, 1000.0],
        [0.25, 1150.0],
        [0.45, 1000.0],
        [0.5, 1050.0],
        [0.55, 1000.0],
        [0.75, 1150.0],
        [0.95, 1000.0],
        [1.0, 1173.0],
    ]
    POLY_1 = _svg([[0.05, 1000.0], [0.25, 1150.0], [0.45, 1000.0]])
    POLY_2 = _svg([[0.55, 1000.0], [0.75, 1150.0], [0.95, 1000.0]])

    def test_every_label_is_processed(self):
        labels = [("L1 + L2", 25.0, 800.0), ("L1 + L2", 75.0, 800.0)]
        recs = _records(_json(labels, [self.POLY_1, self.POLY_2]), self.TWO_LIQ)
        assert sorted(rec["comp"] for rec in recs) == pytest.approx([0.25, 0.75])

    def test_a_single_label_yields_a_single_record(self):
        recs = _records(_json([("L1 + L2", 25.0, 800.0)], [self.POLY_1, self.POLY_2]), self.TWO_LIQ)
        assert [rec["comp"] for rec in recs] == pytest.approx([0.25])

    def test_a_label_inside_a_recorded_dome_is_skipped(self):
        labels = [("L1 + L2", 50.0, 800.0), ("L1 + L2", 55.0, 850.0)]
        recs = _records(_json(labels, [DOME_POLY]), DOME_LIQ)
        assert len(recs) == 1


class TestFrame:
    LIQ = [
        [0.0, 1173.0],
        [0.1, 1080.0],
        [0.2, 1000.0],
        [0.3, 1150.0],
        [0.4, 1200.0],
        [0.5, 1150.0],
        [0.6, 1080.0],
        [0.7, 1000.0],
        [0.85, 1080.0],
        [1.0, 1173.0],
    ]

    def test_reversed_json_frame_is_mirrored(self):
        poly = _svg([[1 - 0.2, 1000.0], [1 - 0.4, 1200.0], [1 - 0.7, 1000.0]])
        reversed_json = _json([("L1 + L2", 60.0, 800.0)], [poly], elements=("B", "A"))
        (rec,) = _records(reversed_json, self.LIQ)
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.7])
        assert rec["apex"] == pytest.approx([0.4, 1200.0])

    def test_matching_frame_is_untouched(self):
        poly = _svg([[0.2, 1000.0], [0.4, 1200.0], [0.7, 1000.0]])
        (rec,) = _records(_json([("L1 + L2", 40.0, 800.0)], [poly]), self.LIQ)
        assert [foot[0] for foot in rec["cbounds"]] == pytest.approx([0.2, 0.7])


class TestASyntecticKeepsItsMaximum:
    """A solid crystallizing at a dome's crest melts by ``L1 + L2 -> S``, not congruently.

    The congruent-melting loop in ``identify_invariant_points`` consumes interior maxima
    before ``_miscibility_gap_records`` reads them, so a solid matched to a dome's apex takes
    the maximum the dome needs. ``_two_liquid_apex_maxima`` reserves it first, seeded by the
    printed label exactly as the record itself is.
    """

    # A stroke at the dome's apex composition whose ceiling is inside t_tol of the apex
    # temperature, which is what makes the congruent-melting rule match it.
    APEX_SOLID = _svg([[0.5, 700.0], [0.5, 1190.0]])

    # Two humps: a dome apex at 0.3 with feet at 0.15/0.45, and a plain maximum at 0.7.
    TWO_HUMP_LIQ = [
        [0.0, 1173.0],
        [0.1, 1050.0],
        [0.15, 1000.0],
        [0.2, 1100.0],
        [0.3, 1200.0],
        [0.4, 1100.0],
        [0.45, 1000.0],
        [0.5, 1050.0],
        [0.6, 1120.0],
        [0.7, 1180.0],
        [0.8, 1100.0],
        [0.9, 1050.0],
        [1.0, 1173.0],
    ]
    TWO_HUMP_POLY = _svg([[0.15, 1000.0], [0.3, 1200.0], [0.45, 1000.0]])
    TWO_HUMP_LABEL = ("L<sub>1</sub> + L<sub>2</sub>", 30.0, 800.0)
    DOME_SOLID = _svg([[0.3, 700.0], [0.3, 1190.0]])
    FAR_SOLID = _svg([[0.7, 700.0], [0.7, 1172.0]])

    def _with_solids(self, labels, two_phase, solids):
        """``_json`` plus one-phase solid strokes, which the congruent-melting loop needs."""
        json_ = _json(labels, two_phase)
        json_["shapes"].extend(
            {"kind": "phase", "label": name, "nphases": 1, "is_solid": True, "svgpath": path}
            for name, path in solids
        )
        return json_

    def _invariants(self, json_, liq):
        invariants, _, _ = mpds.identify_invariant_points(
            json_, ["A", "B"], [list(p) for p in liq], COMP_DATA, TEMP_RANGE
        )
        return invariants

    def _apex_case(self, labels, two_phase):
        return self._with_solids(labels, two_phase, [("AB", self.APEX_SOLID)])

    def test_the_dome_is_recorded_and_the_solid_is_not_congruent(self):
        invariants = self._invariants(self._apex_case([LABEL], [DOME_POLY]), DOME_LIQ)
        migs = [iv for iv in invariants if iv["type"] == "mig"]
        assert len(migs) == 1
        assert migs[0]["apex"] == pytest.approx([0.5, 1200.0])
        assert [iv for iv in invariants if iv["type"] == "cmp"] == []

    def test_without_the_label_the_solid_melts_congruently(self):
        """POSITIVE CONTROL: the reservation is label-seeded, so no label reserves nothing."""
        invariants = self._invariants(self._apex_case([], [DOME_POLY]), DOME_LIQ)
        assert [iv for iv in invariants if iv["type"] == "mig"] == []
        (cmp_,) = [iv for iv in invariants if iv["type"] == "cmp"]
        assert cmp_["comp"] == pytest.approx(0.5)

    def test_without_a_lens_the_solid_melts_congruently(self):
        """POSITIVE CONTROL: a label alone does not reserve; a shape must top out there."""
        invariants = self._invariants(self._apex_case([LABEL], []), DOME_LIQ)
        assert [iv for iv in invariants if iv["type"] == "mig"] == []
        assert len([iv for iv in invariants if iv["type"] == "cmp"]) == 1

    def test_only_the_dome_apex_is_reserved(self):
        """A second maximum, clear of the label, keeps the ordinary congruent-melting rule."""
        json_ = self._with_solids(
            [self.TWO_HUMP_LABEL],
            [self.TWO_HUMP_POLY],
            [("AB", self.DOME_SOLID), ("AB3", self.FAR_SOLID)],
        )
        invariants = self._invariants(json_, self.TWO_HUMP_LIQ)
        (mig,) = [iv for iv in invariants if iv["type"] == "mig"]
        (cmp_,) = [iv for iv in invariants if iv["type"] == "cmp"]
        assert mig["apex"] == pytest.approx([0.3, 1200.0])
        assert cmp_["comp"] == pytest.approx(0.7)

    def test_the_feet_come_from_the_lens_not_the_walk(self):
        json_ = self._with_solids(
            [self.TWO_HUMP_LABEL],
            [self.TWO_HUMP_POLY],
            [("AB", self.DOME_SOLID), ("AB3", self.FAR_SOLID)],
        )
        (mig,) = [iv for iv in self._invariants(json_, self.TWO_HUMP_LIQ) if iv["type"] == "mig"]
        assert [foot[0] for foot in mig["cbounds"]] == pytest.approx([0.15, 0.45])
        assert [foot[1] for foot in mig["cbounds"]] == pytest.approx([1000.0, 1000.0])

    def test_the_reserved_set_holds_just_the_apex(self):
        json_ = self._apex_case([LABEL], [DOME_POLY])
        maxima = mpds.find_local_maxima([list(p) for p in DOME_LIQ])
        assert mpds._two_liquid_apex_maxima(json_, ["A", "B"], maxima, t_tol=15) == {(0.5, 1200.0)}

    def test_nothing_is_reserved_without_a_label(self):
        json_ = self._apex_case([], [DOME_POLY])
        maxima = mpds.find_local_maxima([list(p) for p in DOME_LIQ])
        assert mpds._two_liquid_apex_maxima(json_, ["A", "B"], maxima, t_tol=15) == set()

    def test_nothing_is_reserved_without_maxima(self):
        json_ = self._apex_case([LABEL], [DOME_POLY])
        assert mpds._two_liquid_apex_maxima(json_, ["A", "B"], [], t_tol=15) == set()

    def test_the_pool_reads_labels_and_two_phase_shapes_only(self):
        json_ = self._apex_case([LABEL], [DOME_POLY])
        labels, shapes = mpds._two_liquid_lens_pool(json_, ["A", "B"])
        assert labels == [pytest.approx([0.5, 1073.15])]
        assert len(shapes) == 1
