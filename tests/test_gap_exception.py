"""A printed ``L1 + L2`` label keeps the invariant walk from masking across the gap.

``fit_parameters`` walks the invariants and masks the stretch beside any whose conjugate
solid it cannot match to a DFT phase. Where the invariant bounding that stretch is a
monotectic the conjugate on that side is a second LIQUID, so the walk's premise fails and
the range it produced says nothing about solid coverage. The diagram states the gap in
print, so the label is evidence in its own right -- independent of whether the digitized
geometry yields a dome the foot test accepts.

The coverage report applies the same exception to its own drawn-phase mask, through
``printed_gap_comps``; the two paths reach it independently.
"""

import pytest

from gliquid import mpds


class TestTheLabelReader:
    """``two_liquid_label_comps`` on the spellings the cached records actually carry."""

    def test_it_reads_a_markup_label_in_percent(self):
        record = {"labels": [["L<sub>1</sub> + L<sub>2</sub>", [23.2367, 712.7], None]]}
        assert mpds.two_liquid_label_comps(record) == pytest.approx([0.232367])

    def test_it_reads_a_plain_label_as_a_fraction(self):
        record = {"labels": [["L1 + L2", [0.4, 900.0], None]]}
        assert mpds.two_liquid_label_comps(record) == pytest.approx([0.4])

    def test_it_ignores_every_other_label(self):
        record = {
            "labels": [
                ["L", [0.5, 900.0], None],
                ["L + Mg<sub>2</sub>Si", [0.3, 800.0], None],
                ["(Na)", [0.02, 400.0], None],
            ]
        }
        assert mpds.two_liquid_label_comps(record) == []

    def test_it_mirrors_onto_a_reversed_component_frame(self):
        record = {
            "chemical_elements": ["Na", "Si"],
            "labels": [["L<sub>1</sub> + L<sub>2</sub>", [23.0, 700.0], None]],
        }
        assert mpds.two_liquid_label_comps(record, ["Na", "Si"]) == pytest.approx([0.23])
        assert mpds.two_liquid_label_comps(record, ["Si", "Na"]) == pytest.approx([0.77])

    def test_a_record_with_no_labels_block_reads_empty(self):
        assert mpds.two_liquid_label_comps({}) == []
        assert mpds.two_liquid_label_comps({"labels": None}) == []
