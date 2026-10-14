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
from gliquid.binary import BinaryLiquid


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


def _walked(system, drop_labels=False):
    """``(mask, label comps)`` after a stubbed ``fit_parameters`` on a cached system."""
    bl = BinaryLiquid.from_cache(system, param_format="comb-exp", pd_ind=0)
    if bl.init_error or not bl.digitized_liq:  # pragma: no cover
        pytest.skip(f"{system} is excluded upstream of the coverage gate")
    if drop_labels:
        bl.mpds_json["labels"] = [
            entry
            for entry in bl.mpds_json.get("labels") or []
            if not mpds.is_two_liquid_label(entry[0])
        ]
    comps = mpds.two_liquid_label_comps(bl.mpds_json, bl.components)
    original = BinaryLiquid.f
    BinaryLiquid.f = lambda self, guess, **kwargs: float("inf")
    try:
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=True)
    except Exception:  # noqa: BLE001 - the mask is populated before the optimiser runs
        pass
    finally:
        BinaryLiquid.f = original
    return [[float(lo), float(hi)] for lo, hi in bl.unsupported_comp_ranges], comps


class TestTheWalkStopsAtAPrintedGap:
    """Na-Si and Eu-Ni: the Na-rich and Eu-rich arms the monotectic bounds."""

    CASES = [("Na-Si", 0.232367), ("Eu-Ni", 0.205695)]

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system,label_comp", CASES)
    def test_no_masked_range_spans_the_printed_label(self, system, label_comp, matrix_data_env):
        mask, comps = _walked(system)
        assert comps == pytest.approx([label_comp], abs=1e-4)
        assert not [r for r in mask if r[0] <= label_comp <= r[1]]

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system,label_comp", CASES)
    def test_the_same_walk_masks_it_once_the_label_is_gone(
        self, system, label_comp, matrix_data_env
    ):
        """POSITIVE CONTROL: only the printed label differs between the two runs."""
        without, comps = _walked(system, drop_labels=True)
        assert comps == []
        assert [r for r in without if r[0] <= label_comp <= r[1]]

    @pytest.mark.needs_cache
    def test_the_coverage_mask_does_not_reach_this_label(self, matrix_data_env):
        """Na-Si's drawn-phase mask falls clear of its label, so the coverage rule is silent.

        A statement about where Na-Si's mask lands, not about the report being exempt: the
        coverage path honours a printed gap too, and refuses a field that spans one. The mask
        coincides with the dome, whose feet bound the field on both sides.
        """
        bl = BinaryLiquid.from_cache("Na-Si", param_format="comb-exp", pd_ind=0)
        bl.find_invariant_points(verbose=False)
        report = bl.assess_solid_coverage()
        assert report.unsupported_ranges == [
            pytest.approx([0.4741, 0.5288], abs=1e-3),
        ]
        assert report.gap_ranges == [pytest.approx([0.4741, 0.5288], abs=1e-3)]
