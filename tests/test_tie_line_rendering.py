"""Figure-level checks for the tie-line stack ``plot_tx`` draws: coincident horizontals merged
into one trace, and the ``record_tie_lines`` hook.
"""

import pytest

import gliquid.plotting.binary_tx as btx
from gliquid.binary import plot_tx, record_tie_lines
from gliquid.hsx import HSX


def _synthetic_hsx():
    """A minimal two-element hull."""
    data = [
        [0.0, 0.0, 0.0, "A"],
        [1.0, 0.0, 0.0, "B"],
        [0.0, 1.0, 2.0, "L"],
        [1.0, 1.0, 2.1, "L"],
        [0.5, 0.4, 0.9, "AB2"],
    ]
    phases = ["A", "B", "L", "AB2"]
    return HSX(data_dict={"phases": phases, "comps": ["A", "B"], "data": data}, conds=[0.0, 1500.0])


# -- _TieDrawer.add: coincident horizontals are merged, not drawn twice --
def _tie_traces(fig):
    """The silver horizontals plot_tx draws for tie lines."""
    return [tr for tr in fig.data if getattr(getattr(tr, "line", None), "color", None) == "Silver"]


def _render_with_invariants(monkeypatch, entries, key="Eutectics", sink=None):
    """Render a synthetic hull against a hand-built ``key`` invariant list.

    ``liquidus_invariants`` is called once for real so ``df_tx`` is populated (and its
    temperatures converted to Celsius exactly once), then replaced -- plot_tx reads the
    invariants only through that call. ``sink`` is handed to ``record_tie_lines``. Returns
    ``(fig, ties)``.
    """
    hsx = _synthetic_hsx()
    real_inv, combined, _counts = hsx.liquidus_invariants()
    inv_points = {k: [] for k in real_inv}
    inv_points[key] = [list(e) for e in entries]
    monkeypatch.setattr(hsx, "liquidus_invariants", lambda: (inv_points, combined, {}))
    with record_tie_lines(sink) as ties:
        fig = plot_tx(hsx)
    return fig, ties


def _eut(temp, x_lo, x_hi):
    """One invariant entry: ``[temp, comp_mid, comps (fractions), phases]``."""
    mid = 0.5 * (x_lo + x_hi)
    return [temp, mid, [x_lo, mid, x_hi], ["A", "L", "AB2"]]


def _merge_tolerance():
    """The temperature tolerance plot_tx will use on the synthetic hull, in K.

    The tolerance scales with the PLOTTED temperature span, which plot_tx derives from the
    liquidus and writes back into ``hsx.conds`` in place. This hull carries no polymorph
    labels, so nothing lowers conds after the tie stage and the final range is the one the
    merge used. Deriving it here keeps the tests pinned to the CONTRACT (inside the
    tolerance merges, outside does not) rather than to the constant's current value.
    """
    hsx = _synthetic_hsx()
    plot_tx(hsx)
    return btx._TIE_MERGE_T_FRAC * (hsx.conds[1] - hsx.conds[0])


TIE_TOL = _merge_tolerance()
T0 = 0.0  # a temperature inside the synthetic hull's plotted range
NEAR = T0 + 0.4 * TIE_TOL  # comfortably inside the merge tolerance
FAR = T0 + 4.0 * TIE_TOL  # comfortably outside it


class TestTieMerge:
    def test_the_synthetic_tolerance_is_a_usable_positive_window(self):
        assert TIE_TOL > 0.0

    def test_near_coincident_ties_collapse_to_one_trace_with_the_union_span(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.10, 0.60), _eut(NEAR, 0.12, 0.70)]
        )
        assert len(ties) == 1
        assert (ties[0]["x0"], ties[0]["x1"]) == pytest.approx((10.0, 70.0))
        (tie,) = _tie_traces(fig)
        assert tuple(tie.x) == pytest.approx((10.0, 70.0))
        # the surviving trace keeps the temperature it was drawn at
        assert tuple(tie.y) == pytest.approx((T0, T0))

    def test_both_sources_are_recorded_on_the_surviving_tie(self, monkeypatch):
        _fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.10, 0.60), _eut(NEAR, 0.12, 0.70)]
        )
        assert ties[0]["sources"] == ["invariant:Eutectics", "invariant:Eutectics"]

    def test_ties_outside_the_tolerance_stay_separate(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.10, 0.60), _eut(FAR, 0.12, 0.70)]
        )
        assert len(ties) == 2
        assert len(_tie_traces(fig)) == 2

    def test_disjoint_spans_at_one_temperature_stay_separate(self, monkeypatch):
        """The two halves of a three-phase horizontal around a compound both draw."""
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.02, 0.20), _eut(T0, 0.50, 0.70)]
        )
        assert len(ties) == 2
        assert len(_tie_traces(fig)) == 2

    def test_touching_spans_merge(self, monkeypatch):
        _fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.02, 0.20), _eut(T0, 0.20, 0.50)]
        )
        assert len(ties) == 1
        assert (ties[0]["x0"], ties[0]["x1"]) == pytest.approx((2.0, 50.0))

    def test_a_contained_span_never_shrinks_the_drawn_tie(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.10, 0.80), _eut(T0, 0.30, 0.50)]
        )
        assert len(ties) == 1
        (tie,) = _tie_traces(fig)
        assert tuple(tie.x) == pytest.approx((10.0, 80.0))

    def test_an_exact_duplicate_is_still_drawn_once(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(T0, 0.10, 0.60), _eut(T0, 0.10, 0.60)]
        )
        assert len(ties) == 1
        assert len(_tie_traces(fig)) == 1

    def test_the_drawn_count_is_monotone_in_distinct_ties(self, monkeypatch):
        """Three mutually disjoint horizontals draw three traces; adding a fourth that
        overlaps one of them still draws three."""
        base = [_eut(-2.0 * FAR, 0.02, 0.10), _eut(T0, 0.30, 0.40), _eut(2.0 * FAR, 0.60, 0.70)]
        _fig, ties = _render_with_invariants(monkeypatch, base)
        assert len(ties) == 3
        _fig2, ties2 = _render_with_invariants(monkeypatch, base + [_eut(NEAR, 0.35, 0.55)])
        assert len(ties2) == 3

    def test_the_merge_tolerance_stays_in_the_measured_safe_band(self):
        """On the acceptance corpus, 0.0-0.01 all give the same 30 ties, while 0.02 falsely
        merges Mn-Si's two distinct eutectics (1045.8 and 1070.0 C)."""
        assert 0.0 < btx._TIE_MERGE_T_FRAC <= 0.01


class TestTieRecorder:
    def test_no_sink_is_installed_by_default(self):
        assert btx._TIE_SINK is None

    def test_the_hook_is_restored_after_the_block(self):
        with record_tie_lines():
            assert btx._TIE_SINK is not None
        assert btx._TIE_SINK is None

    def test_the_hook_is_restored_after_an_exception(self):
        with pytest.raises(RuntimeError):
            with record_tie_lines():
                raise RuntimeError("boom")
        assert btx._TIE_SINK is None

    def test_a_caller_supplied_sink_is_filled_and_yielded(self, monkeypatch):
        sink = []
        _fig, yielded = _render_with_invariants(monkeypatch, [_eut(T0, 0.10, 0.60)], sink=sink)
        assert yielded is sink
        assert [r["sources"] for r in sink] == [["invariant:Eutectics"]]

    def test_records_mirror_the_drawn_traces(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(-2.0 * FAR, 0.02, 0.10), _eut(T0, 0.30, 0.40)]
        )
        drawn = [v for tr in _tie_traces(fig) for v in (*tr.x, tr.y[0])]
        recorded = [v for r in ties for v in (r["x0"], r["x1"], r["temp"])]
        assert recorded == pytest.approx(drawn)
