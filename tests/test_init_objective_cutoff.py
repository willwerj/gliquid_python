"""Candidates whose initial objective is far above the best candidate's are not optimised.

``fit_parameters`` sorts its candidate constraint sets by initial-simplex objective and, under
``init_objective_cutoff``, runs only those within that multiple of the best; the rest land on
``pruned_candidates``. The optimisation worker is replaced by a recorder here, so the
candidates are assembled and scored exactly as in a fit but nothing is optimised. Runs
offline on the shipped Cu-Mg cache.
"""

import pytest

from gliquid import binary
from gliquid.binary import DEFAULT_INIT_OBJECTIVE_CUTOFF, BinaryLiquid

FIT = {"n_opts": 50, "verbose": False, "max_iter": 8, "check_lupis_elliott": False}


def _candidates_run(monkeypatch, **kwargs):
    """Initial objectives of the candidates the fit would have optimised, in run order."""
    seen = []

    def recorder(task_index, selected_ics, *args, **kw):
        seen.append((int(task_index), float(selected_ics["f"])))
        return None

    monkeypatch.setattr(binary, "_run_single_optimization_worker", recorder)
    bl = BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp", pd_ind=0)
    assert not bl.init_error
    bl.fit_parameters(**FIT, **kwargs)
    return bl, [f for _, f in sorted(seen)]


@pytest.fixture(scope="module")
def every_candidate(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    bl, fs = _candidates_run(monkeypatch, init_objective_cutoff=None)
    assert len(fs) >= 2, "the fixture system offers too few candidates"
    return fs


def test_no_cutoff_runs_every_candidate_in_ascending_order(monkeypatch, every_candidate):
    assert every_candidate == sorted(every_candidate)


def test_cutoff_prunes_the_tail_and_lists_it(monkeypatch, every_candidate):
    best, worst = every_candidate[0], every_candidate[-1]
    cutoff = worst / max(best, 1.0) / 2
    bl, fs = _candidates_run(monkeypatch, init_objective_cutoff=cutoff)
    ceiling = max(best, 1.0) * cutoff
    assert fs == [f for f in every_candidate if f <= ceiling]
    assert [f for _, f in bl.pruned_candidates] == [f for f in every_candidate if f > ceiling]
    assert all(isinstance(name, str) and name for name, _ in bl.pruned_candidates)


def test_default_cutoff_is_the_module_constant(monkeypatch, every_candidate):
    bl, fs = _candidates_run(monkeypatch)
    ceiling = max(every_candidate[0], 1.0) * DEFAULT_INIT_OBJECTIVE_CUTOFF
    assert fs == [f for f in every_candidate if f <= ceiling]
