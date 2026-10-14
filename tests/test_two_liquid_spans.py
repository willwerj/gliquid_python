"""Two-liquid composition spans of a generated diagram.

``HSX.two_liquid_spans`` reads the facets of the last ``compute_tx`` and returns the merged
composition intervals tied by a two-liquid facet, without a second hull. The facet rule is
the pure function ``two_liquid_spans_from_facets``. Runs offline on the shipped Ag-V, Cu-Mg
and Hf-Zr caches.
"""

import numpy as np
import pytest

from gliquid.binary import BinaryLiquid
from gliquid.hsx import two_liquid_spans_from_facets

AG_V_DOME = [54923.70, 6.8246, 608.58, 0.0]


def _hsx(system, params, **kwargs):
    bl = BinaryLiquid.from_cache(system, param_format="comb-exp", **kwargs)
    assert not bl.init_error
    bl.update_params(params)
    bl.update_phase_points()
    return bl.hsx


class TestFacetRule:
    def test_liquid_liquid_solid_facet_is_a_span(self):
        xs = np.array([[0.1, 0.8, 1.0]])
        labels = np.array([["L", "L", "S"]])
        assert two_liquid_spans_from_facets(xs, labels) == [[0.1, 0.8]]

    def test_adjacent_liquid_vertices_are_not_a_tie(self):
        xs = np.array([[0.10, 0.11, 1.0]])
        labels = np.array([["L", "L", "S"]])
        assert two_liquid_spans_from_facets(xs, labels) == []

    def test_three_liquid_vertices_are_ignored(self):
        xs = np.array([[0.1, 0.5, 0.9]])
        labels = np.array([["L", "L", "L"]])
        assert two_liquid_spans_from_facets(xs, labels) == []

    def test_one_liquid_vertex_is_ignored(self):
        xs = np.array([[0.0, 0.3, 0.5]])
        labels = np.array([["S", "L", "T"]])
        assert two_liquid_spans_from_facets(xs, labels) == []

    def test_overlapping_ties_merge_and_disjoint_ties_stay_apart(self):
        xs = np.array([[0.1, 0.5, 1.0], [0.3, 0.6, 1.0], [0.8, 0.95, 1.0]])
        labels = np.array([["L", "L", "S"], ["L", "L", "S"], ["L", "L", "S"]])
        assert two_liquid_spans_from_facets(xs, labels) == [[0.1, 0.6], [0.8, 0.95]]

    def test_vertex_order_within_a_facet_does_not_matter(self):
        xs = np.array([[1.0, 0.8, 0.1]])
        labels = np.array([["S", "L", "L"]])
        assert two_liquid_spans_from_facets(xs, labels) == [[0.1, 0.8]]

    def test_empty_input(self):
        assert two_liquid_spans_from_facets(np.zeros((0, 3)), np.zeros((0, 3), dtype=str)) == []


class TestOnDiagrams:
    def test_ag_v_dome_is_one_span(self):
        (span,) = _hsx("Ag-V", AG_V_DOME).two_liquid_spans()
        assert span[0] == pytest.approx(0.06, abs=0.02)
        assert span[1] == pytest.approx(0.93, abs=0.02)

    def test_ideal_liquid_has_no_span(self):
        assert _hsx("Cu-Mg", [0.0, 0.0, 0.0, 0.0]).two_liquid_spans() == []

    def test_attractive_liquid_has_no_span(self):
        assert _hsx("Hf-Zr", [-30000.0, 0.0, 0.0, 0.0], pd_ind=0).two_liquid_spans() == []

    def test_span_matches_the_coldest_two_liquid_invariant(self):
        hsx = _hsx("Ag-V", AG_V_DOME)
        (span,) = hsx.two_liquid_spans()
        ties = [
            entry
            for entry in hsx.liquidus_invariants()[0]["Misc Gaps"]
            if list(entry[3]).count("L") == 2
        ]
        coldest = min(ties, key=lambda entry: entry[0])
        feet = sorted(c for c, p in zip(coldest[2], coldest[3]) if p == "L")
        assert span == pytest.approx(feet)

    def test_unchanged_after_liquidus_invariants(self):
        hsx = _hsx("Ag-V", AG_V_DOME)
        before = hsx.two_liquid_spans()
        hsx.liquidus_invariants()
        assert hsx.two_liquid_spans() == before

    def test_reflects_the_current_parameters(self):
        bl = BinaryLiquid.from_cache("Ag-V", param_format="comb-exp")
        bl.update_params(AG_V_DOME)
        bl.update_phase_points()
        assert bl.hsx.two_liquid_spans()
        bl.update_params([0.0, 0.0, 0.0, 0.0])
        bl.update_phase_points()
        assert bl.hsx.two_liquid_spans() == []


class TestTies:
    def test_liquid_liquid_liquid_facet_with_a_real_gap_is_a_tie(self):
        from gliquid.hsx import two_liquid_ties_from_facets

        xs = np.array([[0.1, 0.11, 0.8]])
        labels = np.array([["L", "L", "L"]])
        assert two_liquid_ties_from_facets(xs, labels, [1500.0]) == [[0.1, 0.8, 1500.0]]

    def test_three_adjacent_liquid_vertices_are_not_a_tie(self):
        from gliquid.hsx import two_liquid_ties_from_facets

        xs = np.array([[0.10, 0.11, 0.12]])
        labels = np.array([["L", "L", "L"]])
        assert two_liquid_ties_from_facets(xs, labels, [1500.0]) == []

    def test_ties_come_in_temperature_order(self):
        from gliquid.hsx import two_liquid_ties_from_facets

        xs = np.array([[0.2, 0.7, 1.0], [0.1, 0.8, 1.0], [0.3, 0.6, 0.61]])
        labels = np.array([["L", "L", "S"], ["L", "L", "S"], ["L", "L", "L"]])
        ties = two_liquid_ties_from_facets(xs, labels, [1400.0, 1300.0, 1500.0])
        assert ties == [[0.1, 0.8, 1300.0], [0.2, 0.7, 1400.0], [0.3, 0.61, 1500.0]]

    def test_ag_v_dome_ties_narrow_toward_the_top(self):
        hsx = _hsx("Ag-V", AG_V_DOME)
        ties = hsx.two_liquid_ties()
        assert ties
        (span,) = hsx.two_liquid_spans()
        assert ties[0][:2] == pytest.approx(span)
        temps = [tie[2] for tie in ties]
        assert temps == sorted(temps)
        assert 1500.0 < temps[0] < temps[-1] < 4000.0
        assert ties[-1][1] - ties[-1][0] < ties[0][1] - ties[0][0]

    def test_ties_survive_liquidus_invariants(self):
        hsx = _hsx("Ag-V", AG_V_DOME)
        before = hsx.two_liquid_ties()
        hsx.liquidus_invariants()
        assert hsx.two_liquid_ties() == before

    def test_no_dome_means_no_ties(self):
        assert _hsx("Cu-Mg", [0.0, 0.0, 0.0, 0.0]).two_liquid_ties() == []
