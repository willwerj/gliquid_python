"""Figure-level pins for the solid-solution envelope and the tie-line stack.

Split out of ``tests/test_ss_envelope.py``: everything here drives ``plot_tx`` /
``BLPlotter`` and asserts on the resulting FIGURE -- hatch-filled closed shapes, merged tie
horizontals, the recorder hook, the eutectoid horizontal. The topology / predicate contract
and the pinned ``fixtures/ss_envelope_pins.json`` it reads stay in ``tests/``.

``_facet``, ``_MIN_FLOOR`` and ``_MIN_TOP`` exist on both sides of that split and are
duplicated here rather than imported: neither half may depend on the other.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

import gliquid.plotting.binary_tx as btx
from gliquid.binary import BinaryLiquid, BLPlotter, plot_tx, record_tie_lines
from gliquid.hsx import HSX


# ---------------------------------------------------------------------------
# plot_tx rendering: hatch-filled closed shapes, uniform outline
# ---------------------------------------------------------------------------
def _synthetic_hsx(with_ss: bool):
    """A minimal two-element hull; ``SS`` spans the whole composition axis."""
    data = [
        [0.0, 0.0, 0.0, "A"],
        [1.0, 0.0, 0.0, "B"],
        [0.0, 1.0, 2.0, "L"],
        [1.0, 1.0, 2.1, "L"],
        [0.5, 0.4, 0.9, "AB2"],
    ]
    phases = ["A", "B", "L", "AB2"]
    if with_ss:
        phases.insert(3, "SS")
        data += [[x, 0.5, 0.8, "SS"] for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return HSX(data_dict={"phases": phases, "comps": ["A", "B"], "data": data}, conds=[0.0, 1500.0])


def _ss_traces(fig):
    return [tr for tr in fig.data if tr.fill == "toself"]


@pytest.fixture(scope="module")
def ss_fig():
    """Hf-Zr: the offline full-composition solid-solution system."""
    bl = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    return BLPlotter(bl).get_plot("fit+liq"), set(bl.ss_models)


class TestFigureRendering:
    def test_ss_fields_render_as_filled_closed_shapes(self, ss_fig):
        fig, ss_phases = ss_fig
        traces = _ss_traces(fig)
        assert traces, "no filled solid-solution trace was drawn"
        for tr in traces:
            assert tr.x[0] == pytest.approx(tr.x[-1])
            assert tr.y[0] == pytest.approx(tr.y[-1])

    def test_fill_is_clear_with_a_slash_hatch_in_the_phase_color(self, ss_fig):
        fig, _ = ss_fig
        for tr in _ss_traces(fig):
            assert tr.fillpattern.shape == "/"
            assert tr.fillpattern.fgcolor == tr.line.color
            # a clear fill: the pattern background must be fully transparent
            assert "rgba" in tr.fillcolor and tr.fillcolor.replace(" ", "").endswith(",0)")
            assert tr.fillpattern.bgcolor is None or "rgba" in tr.fillpattern.bgcolor

    def test_outline_is_uniform_no_dotted_lower_branch(self, ss_fig):
        fig, _ = ss_fig
        widths = {tr.line.width for tr in _ss_traces(fig)}
        dashes = {tr.line.dash for tr in _ss_traces(fig)}
        assert len(widths) == 1
        assert dashes <= {None, "solid"}

    def test_one_legend_entry_per_rendered_solid_solution_phase(self, ss_fig):
        """A model with no hull presence draws nothing; every drawn field gets one entry."""
        fig, ss_phases = ss_fig
        traces = _ss_traces(fig)
        named = [tr for tr in traces if tr.showlegend]
        # a field may split into several branch polygons but is named exactly once
        assert len(named) == len({tr.name for tr in traces})
        assert 0 < len(named) <= len(ss_phases)
        assert len({tr.name for tr in named}) == len(named)

    def test_non_ss_plots_draw_no_filled_traces(self):
        fig = plot_tx(_synthetic_hsx(with_ss=False))
        assert _ss_traces(fig) == []

    def test_synthetic_ss_phase_is_filled(self):
        hsx = _synthetic_hsx(with_ss=True)
        fig = plot_tx(hsx, ss_phases={"SS"})
        assert isinstance(fig, go.Figure)
        for tr in _ss_traces(fig):
            assert tr.fillpattern.shape == "/"


# ---------------------------------------------------------------------------
# _add_tie: coincident horizontals are merged, not drawn twice
# ---------------------------------------------------------------------------
def _tie_traces(fig):
    """The silver horizontals plot_tx draws for tie lines."""
    return [tr for tr in fig.data if getattr(getattr(tr, "line", None), "color", None) == "Silver"]


def _render_with_invariants(monkeypatch, entries):
    """Render the synthetic non-SS hull against a hand-built invariant list.

    ``liquidus_invariants`` is called once for real so ``df_tx`` is populated (and its
    temperatures converted to Celsius exactly once), then replaced -- plot_tx reads the
    invariants only through that call.
    """
    hsx = _synthetic_hsx(with_ss=False)
    real_inv, combined, _counts = hsx.liquidus_invariants()
    inv_points = {key: [] for key in real_inv}
    inv_points["Eutectics"] = [list(e) for e in entries]
    monkeypatch.setattr(hsx, "liquidus_invariants", lambda: (inv_points, combined, {}))
    with record_tie_lines() as ties:
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

    This is the one import-time render in the suite, and it is why the tie-merge tests live
    on this side of the split: leaving it in the geometry file made every
    ``pytest --collect-only`` draw a figure.
    """
    hsx = _synthetic_hsx(with_ss=False)
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
        """Measured on the acceptance corpus: 0.0-0.01 all give the same 30 ties, while
        0.02 falsely merges Mn-Si's two distinct eutectics (1045.8 and 1070.0 C)."""
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
        hsx = _synthetic_hsx(with_ss=False)
        real_inv, combined, _c = hsx.liquidus_invariants()
        inv_points = {key: [] for key in real_inv}
        inv_points["Eutectics"] = [_eut(T0, 0.10, 0.60)]
        monkeypatch.setattr(hsx, "liquidus_invariants", lambda: (inv_points, combined, {}))
        with record_tie_lines(sink) as yielded:
            plot_tx(hsx)
        assert yielded is sink
        assert [r["sources"] for r in sink] == [["invariant:Eutectics"]]

    def test_records_mirror_the_drawn_traces(self, monkeypatch):
        fig, ties = _render_with_invariants(
            monkeypatch, [_eut(-2.0 * FAR, 0.02, 0.10), _eut(T0, 0.30, 0.40)]
        )
        drawn = [(tuple(tr.x), tr.y[0]) for tr in _tie_traces(fig)]
        assert [((r["x0"], r["x1"]), r["temp"]) for r in ties] == pytest.approx(drawn)


# ---------------------------------------------------------------------------
# plot_tx wiring: the family maximum is resolved once, before the invariant loop
# ---------------------------------------------------------------------------
def _render_ss_with_misc_gaps(monkeypatch, entries):
    """Render the synthetic SS hull against a hand-built 'Misc Gaps' list."""
    hsx = _synthetic_hsx(with_ss=True)
    real_inv, combined, _counts = hsx.liquidus_invariants()
    inv_points = {key: [] for key in real_inv}
    inv_points["Misc Gaps"] = [list(e) for e in entries]
    monkeypatch.setattr(hsx, "liquidus_invariants", lambda: (inv_points, combined, {}))
    with record_tie_lines() as ties:
        fig = plot_tx(hsx, ss_phases={"SS"})
    return fig, ties


def _lss(temp, x_liq, x_s1, x_s2):
    """One L + SS1 + SS2 entry: ``[temp, comp_mid, comps (fractions), phases]``."""
    comps = sorted([x_liq, x_s1, x_s2])
    phases = ["L" if c == x_liq else "SS" for c in comps]
    return [temp, comps[1], comps, phases]


class TestPlotTxAdmitsOneLPlusTwoSolid:
    """End-to-end: the pre-computed per-phase maximum reaches every per-invariant call."""

    def test_only_the_extremal_genuine_member_is_drawn(self, monkeypatch):
        _fig, ties = _render_ss_with_misc_gaps(
            monkeypatch,
            [
                _lss(T0 + 6.0 * FAR, 0.98, 0.994, 0.996),  # collapsed facet, hottest
                _lss(T0 + 2.0 * FAR, 0.14, 0.288, 0.712),  # the reaction
                _lss(T0, 0.12, 0.30, 0.69),  # genuine width, colder
            ],
        )
        assert len(ties) == 1
        assert ties[0]["temp"] == pytest.approx(T0 + 2.0 * FAR)
        assert (ties[0]["x0"], ties[0]["x1"]) == pytest.approx((14.0, 71.2))

    def test_a_family_of_collapsed_facets_alone_draws_nothing(self, monkeypatch):
        _fig, ties = _render_ss_with_misc_gaps(
            monkeypatch,
            [
                _lss(T0 + 2.0 * FAR, 0.98, 0.994, 0.996),
                _lss(T0, 0.14, 0.286, 0.288),
            ],
        )
        assert ties == []


# ---------------------------------------------------------------------------
# The eutectoid horizontal at a branch's interior temperature minimum
# ---------------------------------------------------------------------------
# The synthetic field: an 80 at.%-wide band sampled every 1 at.%, flat on top at 900 C,
# its lower boundary a V bottoming out at 600 C in the middle. Duplicated from the geometry
# half of the split, which builds the same band to test the ADMISSION predicate.
_MIN_FLOOR = 600.0
_MIN_TOP = 900.0


def _facet(t, *vertices):
    """One hull facet as the three ``df_tx`` rows that share its temperature."""
    return [[x / 100.0, float(t), label] for x, label in vertices]


class _FakeHsx:
    """The four attributes ``plot_tx`` reads, over a hand-built facet table.

    A real hull cannot be steered into a designed eutectoid, and the presentation hulls
    that have one are all workspace-cache systems. This carries a ``df_tx`` written
    directly in ``compute_tx``'s layout -- three rows per facet, one temperature each.
    """

    def __init__(self, rows, inv_points, conds=(0.0, 1500.0)):
        self.df_tx = pd.DataFrame(rows, columns=["x", "t", "label"])
        self.phases = ["A", "B", "L", "CA", "SS", "CB"]
        self.comps = ["A", "B"]
        self.conds = list(conds)
        self._inv = inv_points

    def liquidus_invariants(self):
        return self._inv, [], {}


def _eutectoid_hull(inv_points=None, x_min=50.0, t_min=_MIN_FLOOR, changing=True):
    """A hull whose ``SS`` band bottoms out at ``x_min``, flanked by two line compounds."""
    rows = []
    for x in range(0, 101, 10):  # liquidus, well clear above
        rows += _facet(1200.0, (0, "L"), (x, "L"), (100, "L"))
    for t in (0.0, 500.0, 1000.0):  # the flanking line compounds
        rows += _facet(t, (5, "CA"), (5, "CA"), (5, "CA"))
        rows += _facet(t, (95, "CB"), (95, "CB"), (95, "CB"))
    for x in range(10, 91):  # the band: V-shaped floor, flat top
        rows += _facet(t_min + 5.0 * abs(x - x_min), (x, "SS"), (x, "SS"), (x, "SS"))
        rows += _facet(_MIN_TOP, (x, "SS"), (x, "SS"), (x, "SS"))
    # what coexists at x_min just below and just above the minimum
    rows += _facet(t_min - 100.0, (5, "CA"), (x_min, "CA"), (95, "CB"))
    if changing:
        rows += _facet(t_min + 50.0, (5, "CA"), (x_min, "SS"), (60, "SS"))
    else:
        rows += _facet(t_min + 50.0, (5, "CA"), (x_min, "CA"), (95, "CB"))
    keys = ["Eutectics", "Peritectics", "Congruent Melting", "Misc Gaps", "Solid Ties"]
    inv = {key: [] for key in keys}
    inv.update(inv_points or {})
    return _FakeHsx(rows, inv)


def _render_eutectoid(**kwargs):
    hsx = _eutectoid_hull(**kwargs)
    with record_tie_lines() as ties:
        fig = plot_tx(hsx, ss_phases={"SS"})
    return fig, ties


def _solvus_fan(n):
    """``n`` 'Misc Gaps' slices along the SS band's lower boundary, as the emitter hands
    them over: one per grid step, each a collapsed two-phase facet."""
    return [
        [_MIN_FLOOR + 5.0 * i, 0.5, [0.05, (50 - i) / 100, (50 + i) / 100], ["CA", "SS", "SS"]]
        for i in range(1, n + 1)
    ]


class TestMinimumTieRendering:
    def test_the_eutectoid_horizontal_is_drawn_once(self):
        _fig, ties = _render_eutectoid()
        minima = [t for t in ties if "ss_minimum" in t["sources"]]
        assert len(minima) == 1
        assert minima[0]["temp"] == pytest.approx(_MIN_FLOOR)

    def test_it_spans_the_bounding_fields_not_the_band(self):
        """Grown outward to the two line compounds, not to the SS field's own width."""
        _fig, ties = _render_eutectoid()
        (tie,) = [t for t in ties if "ss_minimum" in t["sources"]]
        assert (tie["x0"], tie["x1"]) == pytest.approx((5.0, 95.0))

    def test_an_unchanged_assemblage_draws_nothing(self):
        _fig, ties = _render_eutectoid(changing=False)
        assert [t for t in ties if "ss_minimum" in t["sources"]] == []

    @pytest.mark.parametrize("n", [1, 4, 16])
    def test_one_tie_per_field_however_many_slices_the_emitter_offers(self, n):
        """The solvus fan is rejected wholesale by _ss_tie_allowed; the minimum contributes
        exactly one horizontal no matter how long the fan is."""
        _fig, ties = _render_eutectoid(inv_points={"Misc Gaps": _solvus_fan(n)})
        assert len(ties) == 1
        assert ties[0]["sources"] == ["ss_minimum"]

    def test_a_coincident_eutectic_is_extended_not_doubled(self):
        """The invariant emitter and the minimum naming the same horizontal must leave one
        trace carrying both sources (Hf-W's 1228.3 C peritectoid)."""
        eutectic = [[_MIN_FLOOR, 0.5, [0.05, 0.5, 0.95], ["CA", "SS", "CB"]]]
        fig, ties = _render_eutectoid(inv_points={"Eutectics": eutectic})
        assert len(ties) == 1
        assert ties[0]["sources"] == ["invariant:Eutectics", "ss_minimum"]
        assert (ties[0]["x0"], ties[0]["x1"]) == pytest.approx((5.0, 95.0))
        assert len(_tie_traces(fig)) == 1
