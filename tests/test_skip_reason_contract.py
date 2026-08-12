"""``BinaryLiquid.skip_reason``: the machine-readable refinement of ``init_error``.

Five independent gates set that one bool, and until now the cause was destroyed at the
point of decision — every skipped system reached the campaign workbook as the same
string. ``skip_reason`` names which gate fired, so downstream consumers can tell "no
usable liquidus" from "not enough liquidus" from "not enough solid free energy".

The contract this pins:
  * ``init_error`` stays a plain bool and is set exactly where and when it was before —
    ``skip_reason`` is additive and nothing in the package branches on it;
  * every flagged system carries **exactly one** reason, the earliest gate to fire
    (``_flag_skip`` is first-wins);
  * ``None`` whenever ``init_error`` is False;
  * the reason survives a frame change (``with_component_order``), like every other
    piece of derived state.

Causes 1-3 fire in ``from_cache`` and are driven here with synthetic MPDS jsons (the
``test_liquidus_coverage_gate`` pattern); causes 4-5 fire inside ``fit_parameters`` and
are driven against the offline-cached Hf-Zr system.
"""

import pytest

import gliquid.mpds as mpds
from gliquid.binary import (
    SKIP_MASK_FRACTION,
    SKIP_NARROW_SPAN,
    SKIP_NO_LIQUIDUS,
    SKIP_REASONS,
    SKIP_SOLID_COVERAGE,
    SKIP_SPARSE_LIQUIDUS,
    BinaryLiquid,
)


def _svgpath(points):
    """[[x_pct, T_celsius], ...] -> MPDS-style svgpath string."""
    return "M " + " L ".join(f"{x},{t}" for x, t in points)


def _make_json(shapes, elements=("Hf", "Zr"), temp=(0, 2600)):
    return {
        "reference": {"entry": "synthetic-test-json"},
        "chemical_elements": list(elements),
        "comp_range": [0, 100],
        "temp": list(temp),
        "labels": [],
        "shapes": shapes,
    }


def _liquidus_json(points):
    return _make_json([{"label": "L", "nphases": 1, "svgpath": _svgpath(points)}])


# No 'L' shape at all: the digitizer finds no liquidus to extract.
NO_LIQUIDUS_JSON = _make_json([])
# Well-sampled, but only half the composition axis -> endpoint span 0.5 < comp_range_fit_lim.
NARROW_POINTS = [[x, 2233 - 8 * x] for x in range(0, 51, 2)]
# Bi-Si class: wedges near the pure ends, an ~85 at.% hole between them.
GAPPY_POINTS = [
    [0, 2233],
    [2, 2100],
    [4, 1950],
    [5.5, 1800],
    [7, 1700],
    [92, 1500],
    [94, 1600],
    [96, 1700],
    [98, 1780],
    [100, 1855],
]
# Ordinary V-shaped liquidus at 2 at.% spacing across the full axis.
DENSE_POINTS = [[x, 2233 - (2233 - 1400) * x / 50] for x in range(0, 50, 2)] + [
    [x, 1400 + (1855 - 1400) * (x - 50) / 50] for x in range(50, 101, 2)
]


def _from_cache(monkeypatch, mpds_json, **kwargs):
    def fake_load(input, pd_ind=None):
        return mpds_json, mpds.extract_digitized_liquidus(mpds_json)

    monkeypatch.setattr(mpds, "load_mpds_data", fake_load)
    return BinaryLiquid.from_cache("Hf-Zr", **kwargs)


class TestFromCacheReasons:
    """Causes 1-3, the liquidus-side gates."""

    def test_no_liquidus_at_all(self, monkeypatch):
        bl = _from_cache(monkeypatch, NO_LIQUIDUS_JSON)
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_NO_LIQUIDUS

    def test_endpoint_span_below_fit_limit(self, monkeypatch):
        bl = _from_cache(monkeypatch, _liquidus_json(NARROW_POINTS))
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_NARROW_SPAN

    def test_interior_sparse_liquidus(self, monkeypatch):
        bl = _from_cache(monkeypatch, _liquidus_json(GAPPY_POINTS))
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_SPARSE_LIQUIDUS

    def test_admitted_system_carries_no_reason(self, monkeypatch):
        bl = _from_cache(monkeypatch, _liquidus_json(DENSE_POINTS))
        assert bl.init_error is False
        assert bl.skip_reason is None

    def test_span_gate_short_circuits_the_sparse_gate(self, monkeypatch):
        """A narrow span is reported as narrow, not re-diagnosed as sparse: the two gates
        are ordered and the first one to fire owns the reason."""
        bl = _from_cache(monkeypatch, _liquidus_json(NARROW_POINTS[:6]))
        assert bl.skip_reason == SKIP_NARROW_SPAN

    def test_relaxing_the_gate_clears_the_reason(self, monkeypatch):
        """The reason tracks the bool exactly — it is not an independent verdict."""
        bl = _from_cache(
            monkeypatch, _liquidus_json(GAPPY_POINTS), liq_max_gap=0.90, liq_min_coverage=0.10
        )
        assert bl.init_error is False
        assert bl.skip_reason is None


class TestFitParametersReasons:
    """Causes 4-5, the gates inside ``fit_parameters``. Hf-Zr pd0 is the offline
    full-composition-SS system: without ss_models its liquidus is 100% unsupported."""

    def test_insufficient_solid_coverage(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        assert bl.skip_reason is None  # it cleared the liquidus gates
        assert bl.fit_parameters() == []
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_SOLID_COVERAGE

    def test_mask_fraction_cap_exceeded(self):
        """Synthetic solid-solution invariants at 0.45 and 0.75 make the auto-detected
        ignored ranges span 70% of the liquidus, over the 60% cap. The solid-coverage
        gate is bypassed so the mask gate is the one under test."""
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.invariants = [
            {
                "type": "cmp",
                "comp": 0.45,
                "temp": 1800,
                "phases": ["(Hf, Zr)", "L"],
                "phase_comps": [0.45, 0.45],
            },
            {
                "type": "per",
                "comp": 0.75,
                "temp": 1900,
                "phases": ["(Hf, Zr)", "L"],
                "phase_comps": [0.85, 0.75],
            },
        ]
        bl.low_t_exp_phases = []
        assert (
            bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False, ignored_ranges=True)
            == []
        )
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_MASK_FRACTION

    def test_fitted_system_carries_no_reason(self):
        """A system that clears every gate keeps skip_reason None even after fitting."""
        bl = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
        )
        bl.fit_parameters(n_opts=1, max_iter=4)
        assert bl.init_error is False
        assert bl.skip_reason is None

    def test_disabled_gate_leaves_no_reason(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.init_error is False
        assert bl.skip_reason is None


class TestReasonIsSingleValued:
    def test_first_cause_wins(self, monkeypatch):
        """A system already flagged at from_cache keeps that reason even if a later gate
        would also fire — so 'exactly one reason per flagged system' holds end to end."""
        bl = _from_cache(monkeypatch, _liquidus_json(GAPPY_POINTS))
        assert bl.skip_reason == SKIP_SPARSE_LIQUIDUS
        bl._flag_skip(SKIP_SOLID_COVERAGE)
        assert bl.skip_reason == SKIP_SPARSE_LIQUIDUS
        assert bl.init_error is True

    def test_flag_skip_sets_the_bool(self, monkeypatch):
        bl = _from_cache(monkeypatch, _liquidus_json(DENSE_POINTS))
        assert bl.init_error is False and bl.skip_reason is None
        bl._flag_skip(SKIP_MASK_FRACTION)
        assert bl.init_error is True
        assert bl.skip_reason == SKIP_MASK_FRACTION

    def test_every_reason_is_declared(self):
        assert set(SKIP_REASONS) == {
            SKIP_NO_LIQUIDUS,
            SKIP_NARROW_SPAN,
            SKIP_SPARSE_LIQUIDUS,
            SKIP_SOLID_COVERAGE,
            SKIP_MASK_FRACTION,
        }
        assert len(set(SKIP_REASONS)) == len(SKIP_REASONS)

    def test_reason_survives_a_frame_change(self, monkeypatch):
        bl = _from_cache(monkeypatch, _liquidus_json(GAPPY_POINTS))
        mirrored = bl.with_component_order(["Zr", "Hf"])
        assert mirrored.init_error is True
        assert mirrored.skip_reason == SKIP_SPARSE_LIQUIDUS


class TestNoBehaviourChange:
    """The bool is what every existing consumer reads; the reason must not move it."""

    @pytest.mark.parametrize(
        "points, expected",
        [
            (DENSE_POINTS, False),
            (GAPPY_POINTS, True),
            (NARROW_POINTS, True),
        ],
    )
    def test_init_error_verdicts_unchanged(self, monkeypatch, points, expected):
        bl = _from_cache(monkeypatch, _liquidus_json(points))
        assert bl.init_error is expected
        assert (bl.skip_reason is not None) is expected
