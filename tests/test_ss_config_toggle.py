"""End-to-end tests for the uniform, config-level solid-solution (SS) switch.

`BinaryLiquid.from_cache(solid_solutions=None)` and `TernaryLiquidInterpolation(solid_solutions=None)`
(the defaults) defer to the package-wide `config.solid_solutions` flag; an explicit True/False
overrides it. Both load their models with `config.ss_ref_mode` unless `ss_kwargs` names one. These
tests pin the four contract corners through from_cache, entirely offline:

  * config default OFF  == explicit solid_solutions=False
  * config ON + covered == explicit solid_solutions=True + ss_kwargs
  * config ON + UNcovered == OFF (no SS phases, no altered liquid references)
  * an explicit solid_solutions= kwarg always wins over the config default

Covered fixture: Hf-Zr (pd_ind=0). Uncovered fixture: Cr-Eu (Eu absent from the shipped
3-element omegas file). Both have full offline cache in the package data/ dir.
"""

import pytest

import gliquid.config as config
import gliquid.solution as solution
from gliquid.binary import BinaryLiquid
from gliquid.ternary import TernaryLiquidInterpolation


@pytest.fixture(autouse=True)
def _restore_ss_config():
    """Snapshot/restore the global SS switch so a test that flips it never leaks
    SS-on into unrelated tests sharing the same interpreter."""
    old_ss, old_mode = config.solid_solutions, config.ss_ref_mode
    yield
    config.set_solid_solutions(old_ss)
    config.set_ss_ref_mode(old_mode)


def _refs(bl):
    return {el: (bl.component_data[el].h_liq, bl.component_data[el].s_liq) for el in bl.components}


def test_config_default_off_matches_explicit_off():
    assert config.solid_solutions is False
    implicit = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)  # default None -> config OFF
    explicit = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=False)
    assert implicit.ss_models == {} == explicit.ss_models
    assert _refs(implicit) == _refs(explicit)
    assert implicit.to_hsx() == explicit.to_hsx()
    assert [p.name for p in implicit.phases] == [p.name for p in explicit.phases]


def test_config_on_covered_matches_explicit_ss_kwargs():
    config.set_solid_solutions(True)  # ss_ref_mode stays 'from_unary_db'
    resolved = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)  # config-resolved ON
    explicit = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    assert set(resolved.ss_models) == set(explicit.ss_models) == {"BCC", "HCP", "FCC"}
    for phase in resolved.ss_models:
        for field in ("omega", "delta_h", "delta_s"):
            r, e = resolved.ss_models[phase][field], explicit.ss_models[phase][field]
            assert r == pytest.approx(e)
    for el in resolved.components:
        assert resolved.component_data[el].h_liq == pytest.approx(explicit.component_data[el].h_liq)
        assert resolved.component_data[el].s_liq == pytest.approx(explicit.component_data[el].s_liq)


def test_config_on_uncovered_matches_off():
    config.set_solid_solutions(True)
    on = BinaryLiquid.from_cache("Cr-Eu")  # config ON, but Cr-Eu is uncovered
    off = BinaryLiquid.from_cache("Cr-Eu", solid_solutions=False)
    assert on.ss_models == {} == off.ss_models
    # The coverage gate must leave the liquid references byte-identical to the SS-off path;
    # identical ss_models ({}) + identical component_data => identical generated liquidus/hull.
    assert _refs(on) == _refs(off)


def test_explicit_override_wins_over_config():
    config.set_solid_solutions(True)
    forced_off = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=False)
    assert forced_off.ss_models == {}

    config.set_solid_solutions(False)
    forced_on = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    assert set(forced_on.ss_models) == {"BCC", "HCP", "FCC"}


def test_a_ternary_defers_to_the_config_switch():
    config.set_solid_solutions(True)
    assert TernaryLiquidInterpolation(["Hf", "Ti", "Zr"]).solid_solutions is True
    assert (
        TernaryLiquidInterpolation(["Hf", "Ti", "Zr"], solid_solutions=False).solid_solutions
        is False
    )


def test_a_ternary_loads_with_the_config_ss_ref_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(
        solution, "load_solid_solution_models", lambda *a, **kw: seen.append(kw["ref_mode"]) or {}
    )
    config.set_ss_ref_mode("from_omegas_file")
    for ss_kwargs in (None, {"ref_mode": "from_unary_db"}):
        ti = TernaryLiquidInterpolation(
            ["Hf", "Ti", "Zr"], solid_solutions=True, ss_kwargs=ss_kwargs
        )
        ti.init_ref_data()
        ti.append_solid_solution_surfaces()
        assert ti.ss_kwargs == ss_kwargs
    assert seen == ["from_omegas_file", "from_unary_db"]
