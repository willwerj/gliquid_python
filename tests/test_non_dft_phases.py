"""Phases that stand in for DFT compounds: imputed phases and user-defined phases.

A user-defined phase is one added or edited in the hull editor, or a compound entry with no
``entry_id`` at all, as a hand-built hull carries. Both kinds are drawn apart from DFT phases
and count as solid cover with no space group. Only imputed phases take part in the imputation
lifecycle: staleness, drop and store caching.

Everything runs offline against the shipped Cu-Mg, Hf-Zr and Al-Mg-Si caches. Every marked case
carries an unmarked control.
"""

from __future__ import annotations

import copy
import functools
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # noqa: E402
from pymatgen.core import Composition, Element  # noqa: E402
from pymatgen.entries.computed_entries import ComputedEntry  # noqa: E402

import gliquid.api as api  # noqa: E402
import gliquid.cache as cache  # noqa: E402
import gliquid.imputation.lifecycle as lifecycle  # noqa: E402
import gliquid.phase as phase_mod  # noqa: E402
import gliquid.ternary as ternary_mod  # noqa: E402
from gliquid.binary import BinaryLiquid, BLPlotter, build_phases_from_chull  # noqa: E402
from gliquid.phase import EV_ATOM_TO_J_MOL, UNARY, Phase  # noqa: E402
from gliquid.plotting import binary_figs, style  # noqa: E402
from gliquid.ternary import (  # noqa: E402
    TernaryLiquidInterpolation,
    TLIPlotter,
    build_binary_edge_figures,
)

from ._mpds_builders import imputed_entry  # noqa: E402

CU_MG = ["Cu", "Mg"]
CU_MG_PARAMS = [-41302.26, -5.1627, -8148.05, 0.0]
MGCU2_ID = "mp-1038-GGA"
MG2CU_ID = "mp-2481-GGA"
MG2CU_SPACEGROUP = 70
USER_LABEL = "CuMg (user)"
USER_X = 0.5
USER_EF_EV = -0.135  # below the Cu-Mg hull at x = 0.5, displacing neither neighbour


@pytest.fixture(scope="module")
def cu_mg() -> BinaryLiquid:
    return BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp", params=CU_MG_PARAMS)


@pytest.fixture
def editor_cls():
    pytest.importorskip("ipywidgets")
    from gliquid.hull_editor import ConvexHullEditor

    return ConvexHullEditor


def _user_entry(formula="CuMg", energy=-1.0, label=USER_LABEL, **data):
    entry = ComputedEntry(
        Composition(formula),
        energy,
        entry_id=f"user:{label}",
        data={"non_dft": "user", "label": label, **data},
    )
    entry.name = label
    return entry


_imputed_entry = functools.partial(imputed_entry, x_b=0.5, energy=-0.5, label="CuMg")


def _hand_built_hull():
    """Elements and two compounds as bare ``PDEntry`` objects, as a notebook builds one."""
    entries = [
        PDEntry(Composition("Cu"), 0.0, name="Cu"),
        PDEntry(Composition("Mg"), 0.0, name="Mg"),
        PDEntry(Composition("MgCu2"), -0.143 * 3, name="MgCu2"),
        PDEntry(Composition("Mg2Cu"), -0.106 * 3, name="Mg2Cu"),
    ]
    return PhaseDiagram(entries, elements=[Element(c) for c in CU_MG])


def _with_user_phase(editor_cls, bl):
    """A copy of ``bl`` with one user-defined phase added at ``USER_X`` through the editor."""
    work = copy.deepcopy(bl)
    editor = editor_cls(work)
    editor.add_entry("CuMg", USER_EF_EV, name=USER_LABEL)
    editor.apply()
    return work


def _with_imputed_phase(bl):
    """A copy of ``bl`` whose hull holds a tagged imputed vertex at x = 0.5."""
    work = copy.deepcopy(bl)
    ref = sum(0.5 * work.dft_ch.el_refs[Element(c)].energy_per_atom for c in CU_MG)
    imputed = _imputed_entry(energy=ref + USER_EF_EV)
    work.dft_ch = PhaseDiagram(
        list(work.dft_ch.all_entries) + [imputed], elements=list(work.dft_ch.elements)
    )
    work.phases = build_phases_from_chull(work.dft_ch, work.components, work.component_data)
    return work


class TestTheEntryKind:
    def test_an_imputed_entry(self):
        assert cache.non_dft_entry_kind(_imputed_entry().as_dict()) == "imputed"
        assert cache.non_dft_entry_kind(_imputed_entry()) == "imputed"

    def test_a_user_tagged_entry(self):
        entry = _user_entry()
        assert cache.non_dft_entry_kind(entry) == "user"
        assert cache.non_dft_entry_kind(entry.as_dict()) == "user"

    def test_either_user_marker_is_enough(self):
        by_id = ComputedEntry(Composition("CuMg"), -1.0, entry_id="user:X")
        by_data = ComputedEntry(Composition("CuMg"), -1.0, entry_id="x", data={"non_dft": "user"})
        assert cache.non_dft_entry_kind(by_id) == "user"
        assert cache.non_dft_entry_kind(by_data) == "user"

    def test_a_compound_with_no_entry_id_is_user(self):
        bare = PDEntry(Composition("MgCu2"), -1.0)
        idless = ComputedEntry(Composition("MgCu2"), -1.0)
        assert cache.non_dft_entry_kind(bare) == "user"
        assert cache.non_dft_entry_kind(bare.as_dict()) == "user"
        assert cache.non_dft_entry_kind(idless) == "user"

    def test_an_element_is_never_non_dft(self):
        assert cache.non_dft_entry_kind(PDEntry(Composition("Cu"), 0.0)) is None
        assert cache.non_dft_entry_kind(_user_entry("Cu", 0.0, label="Cu")) is None

    def test_a_dft_entry_is_not(self):
        """Positive control, both id generations: a plain string and a serialized MPID."""
        classic = ComputedEntry(Composition("MgCu2"), -1.0, entry_id=MGCU2_ID)
        assert cache.non_dft_entry_kind(classic) is None
        as_dict = classic.as_dict()
        as_dict["entry_id"] = {"identifier": "mp-aaaaamud", "suffix": "GGA", "separator": "-"}
        assert cache.non_dft_entry_kind(as_dict) is None

    def test_the_label(self):
        assert cache.non_dft_entry_label(_user_entry(label="alpha (hp)")) == "alpha (hp)"
        assert cache.non_dft_entry_label(_imputed_entry(label="Mg2Cu")) == "Mg2Cu"
        assert cache.non_dft_entry_label(PDEntry(Composition("MgCu2"), -1.0, name="L1")) == "L1"
        by_id = ComputedEntry(Composition("CuMg"), -1.0, entry_id="user:Foo")
        assert cache.non_dft_entry_label(by_id) == "Foo"


NEW_API_1101851 = {"identifier": "mp-aaackryx", "suffix": "GGA", "separator": "-"}


class TestIdsReadTheSameAcrossStores:
    @pytest.mark.parametrize(
        "entry_id, expected",
        [
            ("mp-1101851-GGA", "mp-1101851-GGA"),
            (NEW_API_1101851, "mp-1101851-GGA"),
            (str(NEW_API_1101851), "mp-1101851-GGA"),
            ("mp-aaackryx", "mp-1101851"),
            ("mp-aaaaacap", "mp-1367"),
            ("mp-1367", "mp-1367"),
        ],
    )
    def test_a_materials_project_id_reads_in_its_classic_spelling(self, entry_id, expected):
        assert cache.canonical_entry_id(entry_id) == expected

    @pytest.mark.parametrize(
        "entry_id", ["user:MgSi (user)", "imputed:Al2Mg3", "mp-al", "s-cu", "Mg2Si", "{not a dict"]
    )
    def test_any_other_id_reads_as_itself(self, entry_id):
        assert cache.canonical_entry_id(entry_id) == entry_id

    def test_no_id_stays_none(self):
        assert cache.canonical_entry_id(None) is None

    def test_distinct_ids_stay_distinct(self):
        assert cache.canonical_entry_id("mp-aaackryx") != cache.canonical_entry_id("mp-aaackryy")
        assert cache.canonical_entry_id("mp-1367-GGA") != cache.canonical_entry_id("mp-1367-r2SCAN")
        assert cache.canonical_entry_id("mp-1367-GGA") != cache.canonical_entry_id("mp-1367")


class TestThePhaseListCarriesTheKind:
    def test_a_hand_built_hull_gives_user_phases(self, cu_mg):
        phases = build_phases_from_chull(_hand_built_hull(), CU_MG, cu_mg.component_data)
        compounds = [p for p in phases if p.composition is not None and len(p.composition) == 2]
        assert [(p.name, p.non_dft, p.imputed) for p in compounds] == [
            ("MgCu2", "user", False),
            ("Mg2Cu", "user", False),
        ]

    def test_a_dft_hull_gives_none(self, cu_mg):
        """Control: every Cu-Mg phase from the shipped hull is DFT."""
        assert {p.non_dft for p in cu_mg.phases} == {None}

    def test_an_imputed_entry_gives_imputed(self, cu_mg):
        work = _with_imputed_phase(cu_mg)
        (imputed,) = [p for p in work.phases if p.imputed]
        assert (imputed.name, imputed.non_dft) == ("CuMg", "imputed")

    def test_the_editor_marks_the_phase_it_adds(self, editor_cls, cu_mg):
        work = _with_user_phase(editor_cls, cu_mg)
        (user,) = [p for p in work.phases if p.non_dft == "user"]
        assert (user.name, user.imputed) == (USER_LABEL, False)
        assert {p.name for p in work.phases if p.non_dft is None} >= {"MgCu2", "Mg2Cu"}

    def test_the_editor_keeps_an_imputed_phase_imputed(self, editor_cls, cu_mg):
        work = _with_imputed_phase(cu_mg)
        editor_cls(work).apply()
        (imputed,) = [p for p in work.phases if p.imputed]
        assert (imputed.name, imputed.non_dft) == ("CuMg", "imputed")
        assert "imputed:CuMg" in {str(e.entry_id) for e in work.dft_ch.all_entries}

    def test_an_older_imputed_phase_reads_as_imputed(self):
        """A ``Phase`` with ``imputed`` set and no ``non_dft`` value reads as imputed."""
        old = phase_mod.Phase(phase_type="solid", name="X", imputed=True)
        assert phase_mod.non_dft_kind(old) == "imputed"
        assert phase_mod.non_dft_kind(phase_mod.Phase(phase_type="solid", name="Y")) is None


class TestTheEditorKeepsWhatItDoesNotTouch:
    def test_apply_with_no_edits_keeps_every_entry(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        before = [(type(e).__name__, str(e.entry_id)) for e in work.dft_ch.all_entries]
        editor_cls(work).apply()
        after = [(type(e).__name__, str(e.entry_id)) for e in work.dft_ch.all_entries]
        assert after == before

    def test_the_space_groups_survive(self, editor_cls, cu_mg):
        expected = copy.deepcopy(cu_mg).dft_spacegroups()
        assert expected[2 / 3] == MG2CU_SPACEGROUP
        work = copy.deepcopy(cu_mg)
        editor_cls(work).apply()
        assert work.dft_spacegroups() == expected

    def test_an_addition_leaves_the_dft_entries_whole(self, editor_cls, cu_mg):
        work = _with_user_phase(editor_cls, cu_mg)
        ids = {str(e.entry_id) for e in work.dft_ch.all_entries}
        assert {MGCU2_ID, MG2CU_ID} <= ids


class TestTheEditorTagsWhatItChanges:
    def test_an_added_entry_is_user_defined(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        editor.add_entry("CuMg", USER_EF_EV, name=USER_LABEL)
        added = editor._entries[-1]
        assert cache.non_dft_entry_kind(added) == "user"
        assert str(added.entry_id) == f"user:{USER_LABEL}"
        assert added.data["label"] == USER_LABEL
        assert added.name == USER_LABEL

    def test_an_energy_edit_is_user_defined_and_replaces_the_dft_entry(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        index = editor.find_entry(name="MgCu2")
        assert str(editor._entries[index].entry_id) == MGCU2_ID
        editor.modify_entry(index, new_ef_per_atom=-0.2)
        edited = editor._entries[index]
        assert cache.non_dft_entry_kind(edited) == "user"
        assert edited.data["replaces"] == [MGCU2_ID]
        assert editor.find_entry(name="MgCu2") == index

    def test_an_edited_imputed_entry_keeps_what_it_replaces(self, editor_cls, cu_mg):
        imputed = _imputed_entry()
        imputed.data["replaces"] = [MG2CU_ID]
        hull = PhaseDiagram(
            list(cu_mg.dft_ch.all_entries) + [imputed], elements=list(cu_mg.dft_ch.elements)
        )
        editor = editor_cls(hull)
        (index,) = [i for i, e in enumerate(editor._entries) if e.entry_id == "imputed:CuMg"]
        editor.modify_entry(index, new_ef_per_atom=-0.1)
        edited = editor._entries[index]
        assert cache.non_dft_entry_kind(edited) == "user"
        assert edited.data["replaces"] == [MG2CU_ID]
        assert edited.data["label"] == "CuMg"

    def test_an_entropy_edit_is_user_defined(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        index = editor.find_entry(name="MgCu2")
        editor.modify_entry(index, new_entropy=3.1, units="J/mol")
        assert cache.non_dft_entry_kind(editor._entries[index]) == "user"

    def test_an_empty_edit_changes_nothing(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        index = editor.find_entry(name="MgCu2")
        editor.modify_entry(index)
        assert str(editor._entries[index].entry_id) == MGCU2_ID

    def test_an_element_edit_stays_a_reference(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        index = editor.find_entry(name="Cu")
        editor.modify_entry(index, new_ef_per_atom=0.01)
        assert cache.non_dft_entry_kind(editor._entries[index]) is None

    def test_undo_restores_the_dft_entry(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        index = editor.find_entry(name="MgCu2")
        editor.modify_entry(index, new_ef_per_atom=-0.2)
        editor.undo()
        assert str(editor._entries[index].entry_id) == MGCU2_ID

    def test_two_additions_under_one_name_get_distinct_ids(self, editor_cls, cu_mg):
        editor = editor_cls(copy.deepcopy(cu_mg))
        editor.add_entry("CuMg", -0.2, name="X")
        editor.add_entry("CuMg", -0.1, name="X")
        ids = [str(e.entry_id) for e in editor._entries[-2:]]
        assert len(set(ids)) == 2


def _index_of(editor, entry_id):
    """The working-entry index of the entry with ``entry_id``."""
    return next(i for i, e in enumerate(editor._entries) if str(e.entry_id) == entry_id)


class TestTheEditorRecordsWhatItRemoves:
    def test_a_removed_dft_entry_is_recorded(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.remove_entry(_index_of(editor, MGCU2_ID))
        editor.apply()
        assert work.removed_entry_ids == [MGCU2_ID]
        assert "MgCu2" not in {p.name for p in work.phases}

    def test_an_apply_without_removals_records_none(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor_cls(work).apply()
        assert work.removed_entry_ids == []

    def test_an_edited_entry_is_replaced_not_removed(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.modify_entry(_index_of(editor, MGCU2_ID), new_ef_per_atom=-0.2)
        editor.apply()
        assert work.removed_entry_ids == []

    def test_a_removed_user_entry_is_not_recorded(self, editor_cls, cu_mg):
        work = _with_user_phase(editor_cls, cu_mg)
        editor = editor_cls(work)
        editor.remove_entry(editor.find_entry(name=USER_LABEL))
        editor.apply()
        assert work.removed_entry_ids == []

    def test_a_second_editor_adds_to_the_first_ones_removals(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        for entry_id in (MGCU2_ID, MG2CU_ID):
            editor = editor_cls(work)
            editor.remove_entry(_index_of(editor, entry_id))
            editor.apply()
        assert work.removed_entry_ids == sorted([MGCU2_ID, MG2CU_ID])

    def test_with_component_order_carries_them(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.remove_entry(_index_of(editor, MGCU2_ID))
        editor.apply()
        assert work.with_component_order(["Mg", "Cu"]).removed_entry_ids == [MGCU2_ID]


class TestTheEditedSystemFits:
    """The notebook paths: a hull built in the editor, then coverage and a fit."""

    def test_a_hand_built_hull_has_no_space_groups(self, editor_cls, cu_mg):
        editor = editor_cls(
            PhaseDiagram(
                [PDEntry(Composition(c), 0.0, name=c) for c in CU_MG],
                elements=[Element(c) for c in CU_MG],
            )
        )
        editor.add_entry("MgCu2", -0.143, name="MgCu2")
        editor.add_entry("Mg2Cu", -0.106, name="Mg2Cu")
        work = copy.deepcopy(cu_mg)
        work.dft_ch = editor.apply()
        work.phases = build_phases_from_chull(work.dft_ch, CU_MG, work.component_data)
        assert work.dft_spacegroups() == {1 / 3: None, 2 / 3: None}

    def test_an_edited_system_keeps_its_untouched_space_groups(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.modify_entry(editor.find_entry(name="MgCu2"), new_entropy=3.1, units="J/mol")
        editor.add_entry("Cu0.97Mg0.03", -5700, name="Cu (x=0.03)", units="J/mol")
        editor.apply()
        groups = work.dft_spacegroups()
        assert groups[2 / 3] == MG2CU_SPACEGROUP
        assert groups[1 / 3] is None
        (added,) = [x for x in groups if abs(x - 0.03) < 1e-9]
        assert groups[added] is None

    def test_fit_parameters_runs(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.add_entry("CuMg", USER_EF_EV, name=USER_LABEL)
        editor.apply()
        work.fit_parameters(n_opts=1, max_iter=16)

    def test_dft_assignment_runs(self, editor_cls, cu_mg):
        work = _with_user_phase(editor_cls, cu_mg)
        work.find_invariant_points()
        phases, assigned = work.dft_assignment()
        assert isinstance(assigned, dict)


class TestFiguresDrawUserPhasesApart:
    def test_the_palette_keeps_each_dft_phase_on_its_color(self, editor_cls, cu_mg):
        plain = copy.deepcopy(cu_mg)
        plain.update_phase_points()
        edited = _with_user_phase(editor_cls, cu_mg)
        edited.update_phase_points()
        assert USER_LABEL in edited.hsx.phases
        before = BLPlotter(plain)._phase_color_map()
        after = BLPlotter(edited)._phase_color_map()
        assert USER_LABEL in after
        for name in ("MgCu2", "Mg2Cu"):
            assert after[name] == before[name]

    def test_the_hull_figure_marks_a_user_vertex(self, editor_cls, cu_mg):
        edited = _with_user_phase(editor_cls, cu_mg)
        fig = binary_figs.render_dft_hull(edited, "ch")
        (trace,) = [t for t in fig.data if t.name == style.USER_DEFINED_LEGEND_LABEL]
        assert list(trace.x) == pytest.approx([USER_X])
        assert list(trace.hovertext) == [USER_LABEL]

    def test_the_hull_figure_marks_nothing_on_a_dft_hull(self, cu_mg):
        fig = binary_figs.render_dft_hull(copy.deepcopy(cu_mg), "ch")
        names = {t.name for t in fig.data}
        assert style.IMPUTED_LEGEND_LABEL not in names
        assert "User-defined" not in names

    def test_the_legend_names_both_kinds_together(self):
        assert style.non_dft_legend_label({"imputed"}) == style.IMPUTED_LEGEND_LABEL
        assert style.non_dft_legend_label({"user"}) == style.USER_DEFINED_LEGEND_LABEL
        assert style.non_dft_legend_label({"imputed", "user"}) == "Imputed / user-defined"
        assert style.USER_DEFINED_LEGEND_LABEL == "User-defined"

    def test_the_tx_figure_dashes_a_user_phase_and_names_it(self, editor_cls, cu_mg):
        edited = _with_user_phase(editor_cls, cu_mg)
        fig = BLPlotter(edited).get_plot("fit")
        legend = [t for t in fig.data if t.name == style.USER_DEFINED_LEGEND_LABEL]
        assert len(legend) == 1
        (field,) = [t for t in fig.data if t.name == USER_LABEL and t.mode == "lines"]
        assert field.line.dash == "4px,2px"

    def test_the_tx_figure_of_a_dft_system_has_no_such_entry(self, cu_mg):
        fig = BLPlotter(copy.deepcopy(cu_mg)).get_plot("fit")
        assert not [t for t in fig.data if t.name == style.USER_DEFINED_LEGEND_LABEL]

    def test_the_phase_comparison_outlines_a_user_band(self, monkeypatch):
        user = _user_entry("CuMg2", -0.9, label="CuMg2 (user)")
        dft = ComputedEntry(Composition("Cu2Mg"), -0.6, entry_id=MGCU2_ID)
        elements = [
            ComputedEntry(Composition("Cu"), 0.0, entry_id="mp-cu"),
            ComputedEntry(Composition("Mg"), 0.0, entry_id="mp-mg"),
        ]
        hull = PhaseDiagram(elements + [user, dft], elements=[Element(c) for c in CU_MG])
        mp_phases = {user.name: ((0.60, 0.72), -0.30), dft.name: ((0.28, 0.40), -0.20)}
        monkeypatch.setattr(
            binary_figs.mpds,
            "get_low_temp_phase_data",
            lambda mpds_json, dft_ch: (({}, {}, 0, {}), (mp_phases, {}, -0.30)),
        )

        class _Model:
            dft_ch = hull
            components = CU_MG
            component_data = None
            mpds_json = {}
            sys_name = "Cu-Mg"
            dft_type = "GGA"
            phases = []
            eqs = {}

        fig = binary_figs.render_phase_comparison(_Model())
        try:
            (text,) = fig.axes[1].get_legend().get_texts()
            assert text.get_text() == style.USER_DEFINED_LEGEND_LABEL
        finally:
            plt.close(fig)


class TestTheLifecycleIsImputedOnly:
    def test_drop_leaves_a_user_phase(self, editor_cls, cu_mg):
        edited = _with_user_phase(editor_cls, cu_mg)
        dropped = lifecycle.drop_imputed_phases(edited)
        assert USER_LABEL in {p.name for p in dropped.phases}
        assert f"user:{USER_LABEL}" in {str(e.entry_id) for e in dropped.dft_ch.all_entries}

    def test_drop_removes_the_imputed_phase_beside_it(self, editor_cls, cu_mg):
        work = _with_imputed_phase(cu_mg)
        editor = editor_cls(work)
        editor.add_entry("CuMg3", -0.1, name="CuMg3 (user)")
        editor.apply()
        assert "CuMg" in {p.name for p in work.phases}
        dropped = lifecycle.drop_imputed_phases(work)
        ids = {str(e.entry_id) for e in dropped.dft_ch.all_entries}
        assert "imputed:CuMg" not in ids
        assert "user:CuMg3 (user)" in ids


class TestOmegasHcpStaysOutOfThePhaseList:
    """``Phase.imputed`` is also set on ``omegas_hcp`` lattice-stability references, which feed
    the solid-solution models and never enter ``bl.phases``."""

    def test_the_element_carries_one(self):
        assert any(p.imputed for p in UNARY["Zr"].lattice_stabilities)

    def test_no_phase_is_imputed_and_drop_changes_nothing(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, solid_solutions=True)
        assert not any(p.imputed for p in bl.phases)
        dropped = lifecycle.drop_imputed_phases(bl)
        assert [p.name for p in dropped.phases] == [p.name for p in bl.phases]
        assert list(dropped.ss_models) == list(bl.ss_models)


# -- Ternaries built from binaries --

TERNARY = ["Al", "Mg", "Si"]
_IDEAL = [0.0, 0.0, 0.0, 0.0]
DEEP_LABEL = "MgSi (user)"
DEEP_H_J = -140000.0  # J/mol-atom: far below anything on the Al-Mg-Si hull
TERNARY_KW = {"delta": 0.1, "hull_mode": "hsx", "T_incr": 5}


class _EdgeBinary:
    """The surface ``from_binaries`` reads: components, mixing model, format, hull."""

    def __init__(self, components, hull, phases=(), removed=()):
        from gliquid.solution import RKPolyExp

        self.components = list(components)
        self.xs_mix = RKPolyExp("linear", list(_IDEAL))
        self.param_format = "linear"
        self.tau = self.xs_mix.tau
        self.ss_models = {}
        self.dft_ch = hull
        self.phases = list(phases)
        self.removed_entry_ids = list(removed)


def _edge_hull(components, *extra):
    """An edge hull whose elements sit at 0 eV/atom, so an entry's energy is its formation."""
    elements = [ComputedEntry(Composition(c), 0.0, entry_id=f"mp-{c.lower()}") for c in components]
    return PhaseDiagram(elements + list(extra), elements=[Element(c) for c in components])


def _edges(*mg_si_extra, mg_si_phases=(), mg_si_removed=()):
    return [
        _EdgeBinary(("Al", "Mg"), _edge_hull(("Al", "Mg"))),
        _EdgeBinary(
            ("Mg", "Si"), _edge_hull(("Mg", "Si"), *mg_si_extra), mg_si_phases, mg_si_removed
        ),
        _EdgeBinary(("Si", "Al"), _edge_hull(("Si", "Al"))),
    ]


def _user_phase(label=DEEP_LABEL, entropy=7.5, h_j=DEEP_H_J):
    """The solid ``Phase`` a binary holds for a user-defined MgSi entry."""
    return Phase(
        phase_type="solid",
        name=label,
        composition=Composition("MgSi"),
        enthalpy=h_j,
        entropy=entropy,
        non_dft="user",
    )


def _deep_user(label=DEEP_LABEL, h_j=DEEP_H_J, **data):
    comp = Composition("MgSi")
    return api.user_entry(comp, h_j / EV_ATOM_TO_J_MOL * comp.num_atoms, label, **data)


@pytest.fixture(scope="module")
def mg2si_id():
    plain = TernaryLiquidInterpolation(list(TERNARY), **TERNARY_KW)
    (entry_id,) = [
        str(e.entry_id)
        for e in plain.solid_hull([]).stable_entries
        if e.composition.reduced_formula == "Mg2Si"
    ]
    return entry_id


def _new_api_id(classic_id: str) -> str:
    """``classic_id`` (``'mp-1367-GGA'``) as the ``str()`` of its new-API dict."""
    prefix, number, suffix = classic_id.split("-")
    n, letters = int(number), ""
    while n:
        n, digit = divmod(n, 26)
        letters = "abcdefghijklmnopqrstuvwxyz"[digit] + letters
    return str(
        {"identifier": f"{prefix}-{letters.rjust(8, 'a')}", "suffix": suffix, "separator": "-"}
    )


def _names(ti) -> set:
    return set(ti.get_ternary_form_en(ti.components)["Phase Name"])


class TestTernariesCarryUserPhases:
    def test_from_binaries_collects_a_user_entry_unasked(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        (entry,) = ti.user_entries
        assert entry["entry_id"] == f"user:{DEEP_LABEL}"
        assert entry["data"]["h_form_j_per_mol"] == pytest.approx(DEEP_H_J, abs=1e-6)
        assert ti.imputed_entries == []

    def test_a_dft_edge_carries_none(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        assert ti.user_entries == []

    def test_a_hand_built_edge_entry_is_carried_under_its_name(self):
        bare = PDEntry(Composition("MgSi"), DEEP_H_J / EV_ATOM_TO_J_MOL * 2, name="MgSi-hand")
        ti = TernaryLiquidInterpolation.from_binaries(_edges(bare), **TERNARY_KW)
        (entry,) = ti.user_entries
        assert entry["entry_id"] == "user:MgSi-hand"
        assert entry["data"]["label"] == "MgSi-hand"

    def test_it_competes_on_the_ternary_hull_under_its_label(self):
        plain = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        assert "Mg2Si" in _names(plain)
        names = _names(ti)
        assert DEEP_LABEL in names
        assert "Mg2Si" not in names
        assert ti.ternary_meta["user_phase_names"] == [DEEP_LABEL]
        assert ti.ternary_meta["n_imputed_entries"] == 0
        assert plain.ternary_meta["user_phase_names"] == []

    def test_its_formation_enthalpy_is_re_anchored(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        df = ti.get_ternary_form_en(ti.components)
        row = df[df["Phase Name"] == DEEP_LABEL].iloc[0]
        assert float(row["H"]) == pytest.approx(DEEP_H_J, abs=1.0)

    def test_an_edited_entry_takes_the_place_of_its_dft_entry(self, mg2si_id):
        shallow = _deep_user(label="Mg2Si (edited)", h_j=50000.0, replaces=[mg2si_id])
        ti = TernaryLiquidInterpolation.from_binaries(_edges(shallow), **TERNARY_KW)
        names = _names(ti)
        assert "Mg2Si" not in names
        assert "Mg2Si (edited)" not in names

    def test_an_edit_made_under_a_new_api_id_takes_the_place_of_its_dft_entry(self, mg2si_id):
        assert cache.canonical_entry_id(_new_api_id(mg2si_id)) == mg2si_id
        shallow = _deep_user(label="Mg2Si (edited)", h_j=50000.0, replaces=[_new_api_id(mg2si_id)])
        ti = TernaryLiquidInterpolation.from_binaries(_edges(shallow), **TERNARY_KW)
        names = _names(ti)
        assert "Mg2Si" not in names
        assert "Mg2Si (edited)" not in names

    def test_an_edit_under_another_id_leaves_the_dft_entry(self, mg2si_id):
        other = _new_api_id(mg2si_id.replace("-", "-9", 1))
        shallow = _deep_user(label="Mg2Si (edited)", h_j=50000.0, replaces=[other])
        ti = TernaryLiquidInterpolation.from_binaries(_edges(shallow), **TERNARY_KW)
        assert "Mg2Si" in _names(ti)

    def test_with_component_order_carries_them(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        assert ti.with_component_order("Si-Al-Mg").user_entries == ti.user_entries

    def test_drop_imputed_phases_leaves_them(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        before = list(ti.user_entries)
        assert lifecycle.drop_imputed_phases(ti, write_cache=False) == []
        assert ti.user_entries == before

    def test_the_surface_draws_it_dashed_and_names_it(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        fig = TLIPlotter(ti, order="given").get_plot("tx")
        dashed = [t for t in fig.data if t.type == "scatter3d" and t.line.dash == "dash"]
        assert len(dashed) == 1 and f"{DEEP_LABEL} (user-defined)" in dashed[0].hovertemplate
        assert f"{DEEP_LABEL} (user-defined)" in {t.name for t in fig.data}

    def test_the_surface_of_a_dft_ternary_marks_nothing(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        fig = TLIPlotter(ti, order="given").get_plot("tx")
        assert not [t for t in fig.data if t.type == "scatter3d" and t.line.dash == "dash"]
        assert not any("(user-defined)" in (t.name or "") for t in fig.data)


class TestTernariesCarryUserEntropy:
    HOT_LABEL = "MgSi (hot)"
    HOT_H_J = -1000.0  # J/mol-atom: above the Al-Mg-Si 0 K hull at x(Si) = 0.5

    @staticmethod
    def _from(*extra, phases=()):
        edges = _edges(*extra, mg_si_phases=phases)
        return TernaryLiquidInterpolation.from_binaries(edges, **TERNARY_KW)

    def test_the_edge_phase_entropy_rides_on_the_user_entry(self):
        ti = self._from(_deep_user(), phases=[_user_phase(entropy=7.5)])
        (entry,) = ti.user_entries
        assert entry["data"][cache.ENTROPY_KEY] == 7.5

    def test_an_entry_whose_phase_holds_no_entropy_carries_none(self):
        zero = self._from(_deep_user(), phases=[_user_phase(entropy=0.0)])
        absent = self._from(_deep_user())
        assert cache.ENTROPY_KEY not in zero.user_entries[0]["data"]
        assert cache.ENTROPY_KEY not in absent.user_entries[0]["data"]

    def test_a_negative_entropy_is_carried(self):
        ti = self._from(_deep_user(), phases=[_user_phase(entropy=-1.4)])
        assert ti.user_entries[0]["data"][cache.ENTROPY_KEY] == -1.4

    def test_the_hull_row_carries_the_entropy(self):
        ti = self._from(_deep_user(), phases=[_user_phase(entropy=7.5)])
        rows = ti.get_ternary_form_en(ti.components).set_index("Phase Name")
        assert rows.loc[DEEP_LABEL, "S"] == 7.5
        assert rows.loc[DEEP_LABEL, "H"] == pytest.approx(DEEP_H_J, abs=1.0)

    def test_a_dft_row_carries_none(self):
        plain = self._from()
        rows = plain.get_ternary_form_en(plain.components).set_index("Phase Name")
        assert rows.loc["Mg2Si", "S"] == 0.0

    def test_an_entropy_entry_above_the_0k_hull_is_placed(self):
        hot = _deep_user(label=self.HOT_LABEL, h_j=self.HOT_H_J)
        phase = _user_phase(self.HOT_LABEL, entropy=7.5, h_j=self.HOT_H_J)
        with_s = self._from(hot, phases=[phase])
        rows = with_s.get_ternary_form_en(with_s.components).set_index("Phase Name")
        assert rows.loc[self.HOT_LABEL, "S"] == 7.5
        assert rows.loc[self.HOT_LABEL, "H"] == pytest.approx(self.HOT_H_J, abs=1.0)
        assert with_s.ternary_meta["user_phase_names"] == [self.HOT_LABEL]

    def test_the_same_entry_without_entropy_stays_off_the_hull(self):
        cold = self._from(_deep_user(label=self.HOT_LABEL, h_j=self.HOT_H_J))
        assert self.HOT_LABEL not in _names(cold)
        assert cold.ternary_meta["user_phase_names"] == []

    def test_the_deepest_formation_energy_ignores_the_entropy_entry(self):
        hot = _deep_user(label=self.HOT_LABEL, h_j=self.HOT_H_J)
        phase = _user_phase(self.HOT_LABEL, entropy=7.5, h_j=self.HOT_H_J)
        with_s = self._from(hot, phases=[phase])
        without = self._from()
        with_s.get_ternary_form_en(with_s.components)
        without.get_ternary_form_en(without.components)
        deepest = "deepest_formation_energy"
        assert with_s.ternary_meta[deepest] == without.ternary_meta[deepest]

    def test_the_editor_entropy_reaches_the_ternary_entry(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.add_entry("CuMg", USER_EF_EV, name=USER_LABEL, entropy=5.0 / EV_ATOM_TO_J_MOL)
        editor.apply()
        (entry,) = ternary_mod._edge_user_entries(work)
        assert entry["data"][cache.ENTROPY_KEY] == pytest.approx(5.0)

    def test_two_editor_entries_at_one_composition_keep_their_own_entropy(self, editor_cls, cu_mg):
        work = copy.deepcopy(cu_mg)
        editor = editor_cls(work)
        editor.add_entry("CuMg", USER_EF_EV, name="CuMg lo", entropy=1.0 / EV_ATOM_TO_J_MOL)
        editor.add_entry("CuMg", USER_EF_EV + 0.05, name="CuMg hi", entropy=30.0 / EV_ATOM_TO_J_MOL)
        editor.set_temperature(1500.0)
        editor.apply()
        held = {p.name: p.entropy for p in work.phases if p.non_dft == "user"}
        exported = {
            cache.non_dft_entry_label(d): d["data"][cache.ENTROPY_KEY]
            for d in ternary_mod._edge_user_entries(work)
        }
        assert exported == pytest.approx(held)
        assert exported == pytest.approx({"CuMg lo": 1.0, "CuMg hi": 30.0})

    def test_an_editor_entry_without_entropy_carries_none(self, editor_cls, cu_mg):
        work = _with_user_phase(editor_cls, cu_mg)
        (entry,) = ternary_mod._edge_user_entries(work)
        assert cache.ENTROPY_KEY not in entry["data"]


class TestTernariesLeaveOutRemovedEntries:
    @staticmethod
    def _from(removed=(), **kwargs):
        edges = _edges(mg_si_removed=removed)
        return TernaryLiquidInterpolation.from_binaries(edges, **TERNARY_KW, **kwargs)

    def test_an_edge_removal_leaves_the_ternary_hull(self, mg2si_id):
        ti = self._from([mg2si_id])
        assert ti.excluded_entries == [mg2si_id]
        assert "Mg2Si" not in _names(ti)

    def test_an_edge_without_removals_keeps_the_entry(self):
        ti = self._from()
        assert ti.excluded_entries == []
        assert "Mg2Si" in _names(ti)

    def test_a_new_api_spelling_of_the_id_matches(self, mg2si_id):
        ti = self._from([_new_api_id(mg2si_id)])
        assert ti.excluded_entries == [mg2si_id]
        assert "Mg2Si" not in _names(ti)

    def test_an_explicit_excluded_entries_wins(self, mg2si_id):
        ti = self._from([mg2si_id], excluded_entries=[])
        assert "Mg2Si" in _names(ti)

    def test_the_constructor_takes_them(self, mg2si_id):
        ti = TernaryLiquidInterpolation(list(TERNARY), excluded_entries=[mg2si_id], **TERNARY_KW)
        assert mg2si_id not in {str(e.entry_id) for e in ti.solid_hull([]).all_entries}

    def test_with_component_order_carries_them(self, mg2si_id):
        ti = self._from([mg2si_id])
        assert ti.with_component_order("Si-Al-Mg").excluded_entries == [mg2si_id]


def _edge_compounds(ti, edge) -> set:
    return {
        api.hull_entry_name(e)
        for e in ti.edge_hull(edge).stable_entries
        if len(e.composition.elements) == 2
    }


def _edge_ids(ti, edge) -> set:
    return {str(e.entry_id) for e in ti.edge_hull(edge).all_entries}


def _dashed(fig) -> list:
    return [
        t
        for t in fig.data
        if t.line is not None
        and t.line.dash == "4px,2px"
        and t.x is not None
        and t.x[0] is not None
    ]


@pytest.fixture(scope="module")
def user_edge_figures():
    ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
    drawn = [edge for edge in ti.binary_systems if edge in ti.xs_mix]
    return dict(zip(drawn, build_binary_edge_figures(ti)))


class TestEdgeFiguresDrawTheTernarysHull:
    def test_from_binaries_names_an_edge_pred_without_a_digitized_liquidus(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        assert ti.fit_or_pred == dict.fromkeys(ti.binary_systems, "pred")

    def test_from_binaries_names_an_edge_fit_when_its_binary_carries_one(self):
        edges = _edges()
        edges[1].digitized_liq = [[0.2, 1200.0], [0.5, 1300.0]]
        ti = TernaryLiquidInterpolation.from_binaries(edges, **TERNARY_KW)
        assert ti.fit_or_pred["Mg-Si"] == "fit"
        assert ti.fit_or_pred["Al-Mg"] == "pred"

    def test_an_explicit_fit_or_pred_wins(self):
        ti = TernaryLiquidInterpolation.from_binaries(
            _edges(), fit_or_pred={"Al-Mg": "fit"}, **TERNARY_KW
        )
        assert ti.fit_or_pred == {"Al-Mg": "fit"}

    def test_the_edge_hull_is_the_ternarys_hull_on_that_edge(self):
        plain = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        assert _edge_compounds(plain, "Mg-Si") == {"Mg2Si"}
        assert _edge_compounds(ti, "Mg-Si") == {DEEP_LABEL}
        assert _edge_compounds(ti, "Al-Mg") == _edge_compounds(plain, "Al-Mg")

    def test_the_edge_hull_leaves_out_the_dft_entry_an_edit_replaces(self, mg2si_id):
        shallow = _deep_user(label="Mg2Si (edited)", h_j=50000.0, replaces=[mg2si_id])
        ti = TernaryLiquidInterpolation.from_binaries(_edges(shallow), **TERNARY_KW)
        assert mg2si_id not in _edge_ids(ti, "Mg-Si")
        assert f"user:{'Mg2Si (edited)'}" in _edge_ids(ti, "Mg-Si")

    def test_the_edge_hull_keeps_an_entry_its_edge_binary_lacks(self, mg2si_id):
        # the Mg-Si edge binary's hull holds only the two elements
        ti = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        assert mg2si_id in {str(e.entry_id) for e in ti.edge_hull("Mg-Si").stable_entries}

    def test_either_spelling_and_any_component_order_give_one_edge_hull(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        expected = _edge_compounds(ti, "Mg-Si")
        assert _edge_compounds(ti, "Si-Mg") == expected
        assert _edge_compounds(ti, ("Si", "Mg")) == expected
        assert _edge_compounds(ti.with_component_order("Si-Al-Mg"), "Mg-Si") == expected

    def test_a_pickled_ternary_gives_the_same_edge_hull(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        restored = pickle.loads(pickle.dumps(ti))
        assert _edge_compounds(restored, "Mg-Si") == _edge_compounds(ti, "Mg-Si")

    @pytest.mark.parametrize("edge", ["Mg-Cu", "Mg-Mg", "Al-Mg-Si"])
    def test_an_edge_outside_the_ternary_is_refused(self, edge):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(), **TERNARY_KW)
        with pytest.raises(ValueError):
            ti.edge_hull(edge)

    def test_an_edge_figure_draws_a_user_phase_dashed_under_its_legend_entry(
        self, user_edge_figures
    ):
        fig = user_edge_figures["Mg-Si"]
        assert any("MgSi" in (t.name or "") for t in _dashed(fig))
        assert style.USER_DEFINED_LEGEND_LABEL in {t.name for t in fig.data}

    def test_an_edge_figure_without_one_marks_nothing(self, user_edge_figures):
        fig = user_edge_figures["Al-Mg"]
        assert not _dashed(fig)
        assert style.USER_DEFINED_LEGEND_LABEL not in {t.name for t in fig.data}

    def test_the_plotter_draws_the_edges_of_a_ternary_built_from_binaries(self):
        ti = TernaryLiquidInterpolation.from_binaries(_edges(_deep_user()), **TERNARY_KW)
        plotter = TLIPlotter(ti, order="given")
        plotter.get_plot("tx")
        assert len(plotter.bin_fig_list) == 3


class TestFromCacheTakesAHull:
    def test_the_phases_come_from_the_given_hull_and_the_store_is_not_read(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("the stored DFT hull was read")

        hull = _hand_built_hull()
        monkeypatch.setattr(api, "get_dft_convexhull", refuse)
        bl = BinaryLiquid.from_cache(
            "Cu-Mg", param_format="comb-exp", params=CU_MG_PARAMS, hull=hull
        )
        assert bl.dft_ch is hull
        assert {p.name for p in bl.phases if phase_mod.non_dft_kind(p)} == {"MgCu2", "Mg2Cu"}

    def test_without_one_the_stored_hull_is_read(self, cu_mg):
        assert {p.name for p in cu_mg.phases if phase_mod.non_dft_kind(p)} == set()
        assert any(str(e.entry_id) == MGCU2_ID for e in cu_mg.dft_ch.all_entries)
