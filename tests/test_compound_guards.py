"""Compound-component guards (red->green, S5).

Compound end-members (pseudo-binary systems) are a FUTURE feature: every user entry
point must raise NotImplementedError with a clear message today, while the plumbing
that already speaks CompoundPhaseDiagram stays reachable through the
``allow_compounds`` escape hatch (used internally by the api cache layer and by these
tests). The two solid-solution reference sources that will NEVER support compounds
(omegas file, unary db) raise ValueError instead. The silent-empty registry contract
for unknown/dummy symbols ('Xx') is unchanged.
"""

import json

import pytest
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.solution as sd
from gliquid.binary import BinaryLiquid
from gliquid.hull_editor import ConvexHullEditor
from gliquid.phase import UNARY, ComponentRef, Phase, validate_and_format_system
from gliquid.ternary import TernaryLiquidInterpolation


def _cumg_entry_dicts():
    """Synthetic Cu-Mg chemsys entries incl. both terminal compositions of CuMg-Mg."""
    return [
        ComputedEntry(Composition("Cu"), 0.0, entry_id="s-cu").as_dict(),
        ComputedEntry(Composition("Mg"), 0.0, entry_id="s-mg").as_dict(),
        ComputedEntry(Composition("CuMg"), -0.8, entry_id="s-cumg").as_dict(),
        ComputedEntry(Composition("CuMg2"), -0.9, entry_id="s-cumg2").as_dict(),
    ]


class TestValidatorGuard:
    def test_compound_component_raises_nie(self):
        with pytest.raises(NotImplementedError, match="future release"):
            validate_and_format_system("CuMg-Mg")
        with pytest.raises(NotImplementedError, match="CuMg2"):
            validate_and_format_system(["Cu", "Mg", "CuMg2"])

    def test_escape_hatch_passes_compounds(self):
        components, sys_name, order_changed = validate_and_format_system(
            "CuMg-Mg", allow_compounds=True
        )
        assert components == ["CuMg", "Mg"]
        assert sys_name == "CuMg-Mg"
        assert order_changed is False

    def test_elemental_inputs_unchanged(self):
        assert validate_and_format_system("Mg-Cu") == (["Mg", "Cu"], "Mg-Cu", True)
        with pytest.raises(ValueError):
            validate_and_format_system("Xx-Cu")  # dummy species still rejected


class TestEntryPointGuards:
    def test_from_cache_raises_nie(self):
        with pytest.raises(NotImplementedError, match="future release"):
            BinaryLiquid.from_cache("CuMg-Mg")

    def test_ternary_interpolation_raises_nie(self):
        with pytest.raises(NotImplementedError, match="future release"):
            TernaryLiquidInterpolation(["Cu", "Mg", "CuMg2"])

    def test_hull_editor_raises_nie_on_compound_pd(self):
        pytest.importorskip("ipywidgets")
        entries = [api._computed_entry_from_dict(e) for e in _cumg_entry_dicts()]
        cpd = CompoundPhaseDiagram(entries, [Composition("CuMg"), Composition("Mg")])
        with pytest.raises(NotImplementedError, match="future release"):
            ConvexHullEditor(cpd)

    def test_component_ref_and_registry_guards(self):
        with pytest.raises(NotImplementedError, match="future release"):
            ComponentRef("CuMg")
        with pytest.raises(NotImplementedError, match="future release"):
            UNARY["CuMg"]
        # unknown/dummy symbols keep the pinned silent-empty contract
        ref = UNARY["Xx"]
        assert ref.h_liq == 0.0 and ref.t_fusion == 0.0

    def test_fraction_in_compound_axis_raises_nie(self):
        p = Phase(phase_type="solid", name="CuMg", composition=Composition("CuMg"))
        assert p.fraction_in(["Cu", "Mg"]) == (0.5,)  # elemental axes unchanged
        with pytest.raises(NotImplementedError, match="future release"):
            p.fraction_in(["CuMg", "Mg"])


class TestResolverGuards:
    def test_dft_entries_resolver_raises_nie(self):
        # this source CAN support compounds one day -> NotImplementedError
        with pytest.raises(NotImplementedError, match="future release"):
            sd._resolve_refs_cache(["CuMg", "Mg"], entries=[])

    def test_omegas_and_unary_db_resolvers_raise_valueerror(self):
        # these sources are elemental by nature -> permanent ValueError
        with pytest.raises(ValueError, match="element"):
            sd._resolve_refs_legacy({"elements": {}}, ["CuMg", "Mg"], {})
        with pytest.raises(ValueError, match="element"):
            sd._resolve_refs_db(["CuMg", "Mg"], {})


class TestCompoundPlumbingStaysReachable:
    def test_get_dft_convexhull_builds_compound_pd_from_cache(self, tmp_path, monkeypatch):
        (tmp_path / "CuMg-Mg_ENTRIES_MP_GGA.json").write_text(
            json.dumps(_cumg_entry_dicts()), encoding="utf-8"
        )

        def _no_fetch(*args, **kwargs):
            raise AssertionError("cache miss")

        monkeypatch.setattr(api, "_get_dft_entries_from_components", _no_fetch)
        ch, _ = api.get_dft_convexhull(["CuMg", "Mg"], "GGA", data_dir=tmp_path)
        assert isinstance(ch, CompoundPhaseDiagram)
        # terminal-order dummy species; original entry names survive (pymatgen's
        # reduced-formula convention orders by electronegativity: CuMg -> 'MgCu')
        assert len(ch.elements) == 2
        names = {e.name for e in ch.stable_entries}
        assert {"MgCu", "Mg"} <= names
