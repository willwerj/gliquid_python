"""HSX._collapse_gap_runs: thinning continuous two-phase boundaries in 'Misc Gaps'.

A miscibility gap or solvus is a continuous field, but the hull emits one collapsed-
triangle facet per composition grid step along it -- Hf-W 64, Cr-W 75, Hf-Zr-with-SS 220
-- all walking the same boundary. Reported as distinct invariants they inflate the
``migs`` fit column and the matrix-plotter counts with grid artifacts.

The collapse must thin those runs WITHOUT eating genuine three-phase horizontals that
happen to share a phase multiset with a long run (Cr-W's monotectic family at ~1932 C
sits in the same ('BCC','BCC','L') bucket as the W-rich solidus, which climbs to
3338 C). That is what the span-jump run break protects, and it is the case most likely
to regress if the tolerance is ever retuned.
"""

import pytest

from gliquid.hsx import HSX

collapse = HSX._collapse_gap_runs


def entry(temp, comps, phases, mid=None):
    """One raw 'Misc Gaps' record: ``[temp, comp_mid, comps, phases]``."""
    return [temp, mid if mid is not None else comps[len(comps) // 2], list(comps), list(phases)]


def solvus_run(
    n, t0=1200.0, dt=45.0, x0=0.10, dx=0.01, partner=0.67, phases=("BCC", "BCC", "HfW2")
):
    """``n`` consecutive grid slices of one solvus, the Hf-W BCC/HfW2 shape."""
    return [entry(t0 + dt * i, [x0 + dx * i, x0 + dx * (i + 1), partner], phases) for i in range(n)]


class TestRunCollapse:
    def test_long_run_reduces_to_its_two_ends(self):
        run = solvus_run(12)
        out = collapse(run)
        assert len(out) == 2
        assert out[0] is run[0] and out[1] is run[-1]

    @pytest.mark.parametrize("n", [1, 2])
    def test_short_runs_pass_through_untouched(self, n):
        run = solvus_run(n)
        assert collapse(run) == run

    def test_empty_input(self):
        assert collapse([]) == []

    def test_input_order_does_not_matter(self):
        run = solvus_run(9)
        shuffled = [run[i] for i in (4, 0, 8, 2, 6, 1, 7, 3, 5)]
        out = collapse(shuffled)
        assert sorted(e[0] for e in out) == [run[0][0], run[-1][0]]

    def test_collapse_is_idempotent(self):
        once = collapse(solvus_run(15))
        assert collapse(once) == once


class TestBoundarySeparation:
    def test_distinct_phase_multisets_do_not_merge(self):
        a = solvus_run(6, phases=("BCC", "BCC", "HfW2"))
        b = solvus_run(6, t0=2000.0, phases=("BCC", "BCC", "L"))
        out = collapse(a + b)
        assert len(out) == 4
        assert {tuple(sorted(e[3])) for e in out} == {("BCC", "BCC", "HfW2"), ("BCC", "BCC", "L")}

    def test_span_jump_starts_a_new_boundary(self):
        """The Cr-W monotectic guard: a wide three-phase horizontal shares the
        ('BCC','BCC','L') bucket with the solidus run but spans a different width, so it
        must not be swallowed as an interior member of that run."""
        solidus = solvus_run(10, t0=2000.0, x0=0.70, partner=0.95, phases=("BCC", "BCC", "L"))
        monotectic = entry(1932.4, [0.14, 0.29, 0.71], ["L", "BCC", "BCC"])
        out = collapse([monotectic] + solidus)
        temps = sorted(e[0] for e in out)
        assert 1932.4 in temps, "the monotectic horizontal was eaten by the solidus run"
        assert len(out) == 3  # the monotectic, plus the solidus run's two ends

    def test_a_run_broken_in_the_middle_yields_two_pairs(self):
        left = solvus_run(5, x0=0.05)
        right = solvus_run(5, t0=1500.0, x0=0.60)  # far away in composition
        out = collapse(left + right)
        assert len(out) == 4
        assert {e[0] for e in out} == {left[0][0], left[-1][0], right[0][0], right[-1][0]}


class TestToleranceSemantics:
    def test_steps_within_tolerance_chain(self):
        run = [
            entry(1000.0, [0.10, 0.11, 0.67], ["BCC", "BCC", "HfW2"]),
            entry(1050.0, [0.11, 0.12, 0.65], ["BCC", "BCC", "HfW2"]),  # partner -0.02
            entry(1100.0, [0.12, 0.13, 0.63], ["BCC", "BCC", "HfW2"]),
        ]
        assert len(collapse(run, x_tol=0.025)) == 2

    def test_steps_beyond_tolerance_do_not_chain(self):
        run = [
            entry(1000.0, [0.10, 0.11, 0.67], ["BCC", "BCC", "HfW2"]),
            entry(1050.0, [0.30, 0.31, 0.67], ["BCC", "BCC", "HfW2"]),
        ]
        assert len(collapse(run, x_tol=0.025)) == 2  # two singleton runs, both kept
        assert len(collapse(run, x_tol=0.5)) == 2  # one run of two -> still both
