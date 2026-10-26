"""Solid-energy coverage gate: the single quantitative fit/skip criterion.

A liquidus point constrains the liquid free energy only if the solid it is in equilibrium
with has a known free energy, so ``fit_parameters`` admits a system on the fraction of its
liquidus that is conjugate to a solid we cannot evaluate
(``BinaryLiquid.assess_solid_coverage`` -> ``solid_coverage.assess_solid_coverage``).
``full_comp_ss`` is provenance only.

Hf-Zr pd0 is the offline full-composition-SS system (MPDS labels '(Hf, Zr)'); with
``ss_models`` loaded it is fully supported, without them it is fully unsupported.

The frozen ground-truth verdicts over the acceptance corpus need the maintainer's per-system
corpus store and live in ``test_solid_coverage_pins.py``.
"""

import json
from pathlib import Path

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid

from ._mpds_builders import corpus_coverage_report as _report
from ._mpds_builders import coverage_pin_cases

PINS_PATH = Path(__file__).parent / "fixtures" / "solid_coverage_pins.json"


class TestSkipWithoutSsModels:
    def test_fit_skips_and_flags_init_error(self, caplog):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        result = bl.fit_parameters()
        assert result == []
        # The skip is an initialization error: drivers already branch on init_error to
        # tell a deliberate data-adequacy skip from a failed optimization.
        assert bl.init_error is True
        assert bl.full_comp_ss is True  # still recorded, but not load-bearing
        assert "Insufficient solid-phase energy information" in caplog.text
        assert "skipping fit" in caplog.text

    def test_report_attributes_the_skip_to_the_unsupported_field(self):
        """Both '(Hf, Zr)' polymorphs are unsupported; only the ht one masks anything.

        Both take the ``unknown_structure_no_ss`` fallback with no model loaded. The rt
        (alpha) field tops out at 2016.7 K, below the digitized liquidus minimum of 2128.2 K,
        so it never borders the liquid and is excluded from the mask; the ht (beta) field
        reaches 2505.3 K and alone masks the whole axis.
        """
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters()
        cov = bl.coverage_report
        assert cov.unsupported_fraction == pytest.approx(1.0, abs=1e-3)
        assert cov.ss_models == ()
        assert cov.is_insufficient()[0] is True
        unsupported = {p.name: p for p in cov.phases if not p.supported}
        assert set(unsupported) == {"(Hf, Zr) rt", "(Hf, Zr) ht"}
        assert unsupported["(Hf, Zr) ht"].reason == "unknown_structure_no_ss"
        assert unsupported["(Hf, Zr) ht"].borders_liquid is True
        assert unsupported["(Hf, Zr) rt"].reason == "subliquidus:unknown_structure_no_ss"
        assert unsupported["(Hf, Zr) rt"].borders_liquid is False
        assert cov.unsupported_ranges == [[0.0, 1.0]]


class TestProceedWithSsModels:
    def test_fit_proceeds_when_energies_are_loaded(self, caplog):
        bl = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
        )
        result = bl.fit_parameters(n_opts=1, max_iter=4)
        assert "Insufficient solid-phase energy information" not in caplog.text
        assert bl.init_error is False
        assert bl.full_comp_ss is True
        assert bl.coverage_report.unsupported_fraction == pytest.approx(0.0, abs=1e-9)
        assert isinstance(result, list)  # fit ran; contents not asserted

    def test_same_system_flips_verdict_with_the_ss_switch(self):
        """The verdict is a property of the loaded energies, not of the system alone."""
        without = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        with_ss = BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
        )
        for bl in (without, with_ss):
            bl.find_invariant_points()
            bl.assess_solid_coverage()
        assert without.coverage_report.is_insufficient()[0] is True
        assert with_ss.coverage_report.is_insufficient()[0] is False


class TestGateCanBeDisabled:
    def test_check_solid_coverage_false_bypasses_the_gate(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(n_opts=1, max_iter=2, check_solid_coverage=False)
        assert bl.init_error is False
        assert bl.coverage_report is None  # not even assessed

    def test_threshold_override_admits_a_skipped_system(self):
        """A zero floor is the only override that reaches a system with nothing surviving.

        Unlike ``check_solid_coverage=False`` it still assesses, so the report is there to
        read.
        """
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(n_opts=1, max_iter=2, coverage_thresholds={"min_supported_total": 0.0})
        assert bl.init_error is False
        assert bl.coverage_report.unsupported_fraction == pytest.approx(1.0, abs=1e-3)
        assert bl.coverage_report.largest_supported_segment == 0.0

    def test_the_field_window_override_reaches_the_report(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(
            n_opts=1,
            max_iter=2,
            coverage_thresholds={"dft_edge_tol": 0.03, "dft_structure_tol": 0.07},
        )
        assert bl.coverage_report.thresholds["dft_edge_tol"] == 0.03
        assert bl.coverage_report.thresholds["dft_structure_tol"] == 0.07

    def test_removed_threshold_keys_raise_naming_the_verdict(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        with pytest.raises(ValueError, match="use min_supported_total"):
            bl.fit_parameters(n_opts=1, max_iter=2, coverage_thresholds={"skip_frac": 1.0})


class TestChartRelocation:
    CHART_HEADER = "--- Low temperature phase mismatch ---"

    def test_find_invariant_points_no_longer_prints_chart(self, capsys):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.find_invariant_points(verbose=True)
        assert self.CHART_HEADER not in capsys.readouterr().out

    def test_fit_parameters_prints_chart_when_verbose(self, capsys):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(verbose=True)  # skips fitting (no ss_models), after the chart
        assert self.CHART_HEADER in capsys.readouterr().out


class TestSurvivingSegmentAcceptance:
    """Both directions of the surviving-liquidus criterion, on real cached systems.

    The verdict is the ``min_supported_total`` floor on the constraining liquidus total. The
    lists are pinned in BOTH directions.

    Admitted: ``ADMITTED`` (surviving branches 0.353 to 1.000 wide), Rb-Sn on its printed
    two-liquid dome (``ADMITTED_ON_A_DOME``), and Ge-W and Hg-Ni because the masked phase never
    borders the liquid (``ADMITTED_BY_SUBLIQUIDUS``). Skipped: the five zero-compound systems
    whose solid-solution fields have no loaded model, and B-Si, whose ordered field masks all
    but 0.093 (``test_skipped_on_an_ordered_field``).

    Needs the maintainer's per-system corpus store and its production omegas file; without
    both every ``ss_models`` list resolves empty.
    """

    ADMITTED = ["Mn-Pr", "Ru-Sn", "Re-Si", "Al-Ce", "Au-B", "Na-Si", "Mn-Y", "Li-Pr", "Fe-Nd"]
    SKIPPED = ["Lu-Nd", "Ho-Pr", "Er-Th", "Th-Tm", "Ho-Zr"]
    # Rb-Sn's only hull compound, RbSn, is sub-liquidus, so its supported width is tiny; it is
    # admitted on the printed L1 + L2 dome instead. Each entry is (system, n_compounds,
    # n_missing_compounds, reason fragment).
    ADMITTED_ON_A_DOME = [("Rb-Sn", 4, 3, "dome 0.385")]
    SKIPPED_ORDERED_FIELD = ["B-Si"]
    # Admitted only because a masked phase never borders the liquid. Each entry
    # is (system, phase, the phase's own tmax in K, the digitized liquidus minimum in K).
    ADMITTED_BY_SUBLIQUIDUS = [
        ("Ge-W", "W2Ge3", 1022.6, 1202.5),
        ("Hg-Ni", "HgNi", 385.1, 1215.1),
    ]

    # (n_compounds, n_missing_compounds) for each zero-compound control. Ho-Pr carries one:
    # ``Pr0.6Ho0.4 rt`` is an ordered intermetallic with a homogeneity range, so it takes the
    # compound path and is counted there.
    SKIPPED_COMPOUND_COUNTS = {
        "Lu-Nd": (0, 0),
        "Ho-Pr": (1, 1),
        "Er-Th": (0, 0),
        "Th-Tm": (0, 0),
        "Ho-Zr": (0, 0),
    }

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system", ADMITTED)
    def test_admitted(self, system, matrix_data_env):
        cov = _report(system)
        assert cov.supported_total >= cov.thresholds["min_supported_total"]
        assert cov.is_insufficient()[0] is False

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system", SKIPPED)
    def test_skipped(self, system, matrix_data_env):
        cov = _report(system)
        # An unmodelled solid-solution field masks the axis; the compound counts are pinned
        # per system.
        assert (cov.n_compounds, cov.n_missing_compounds) == self.SKIPPED_COMPOUND_COUNTS[system]
        assert cov.supported_total < cov.thresholds["min_supported_total"]
        insufficient, reason = cov.is_insufficient()
        assert insufficient is True
        assert "constraining liquidus totals" in reason

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system", SKIPPED_ORDERED_FIELD)
    def test_skipped_on_an_ordered_field(self, system, matrix_data_env):
        """B-Si is skipped on an ordered field, asserted with its missing-compound counts.

        ``Si15B290`` (x=0.0416, spacegroup 166) is an ordered intermetallic with a
        homogeneity range, so it takes the compound path. The hull holds only x=0.25, and its
        primary field (the peritectic at 0.0935 to the next invariant at 0.3469) merges with
        ``Si10B60``'s ``[0.3469, 0.9181]`` into the single mask ``0.094-0.918``, leaving 0.093
        of the axis supported.
        """
        cov = _report(system)
        assert cov.n_compounds == 3 and cov.n_missing_compounds == 2
        assert cov.supported_total < cov.thresholds["min_supported_total"]
        insufficient, reason = cov.is_insufficient()
        assert insufficient is True
        assert "constraining liquidus totals" in reason

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system,phase,phase_tmax,liq_tmin", ADMITTED_BY_SUBLIQUIDUS)
    def test_admitted_because_a_masked_phase_never_borders_the_liquid(
        self, system, phase, phase_tmax, liq_tmin, matrix_data_env
    ):
        """End to end: a phase the DFT hull is missing is admitted when its whole digitized
        field lies below the minimum digitized liquidus temperature.

        The phase stays ``supported=False`` and is excluded on ``borders_liquid``, so the
        reason string names the missing DFT phase under a ``subliquidus:`` prefix.
        """
        cov = _report(system)
        record = next(p for p in cov.phases if p.name == phase)
        assert record.supported is False
        assert record.borders_liquid is False
        assert record.reason.startswith("subliquidus:")
        assert phase_tmax < liq_tmin  # the fact the exclusion rests on
        assert cov.supported_total >= cov.thresholds["min_supported_total"]
        assert cov.is_insufficient()[0] is False

    @pytest.mark.needs_cache
    @pytest.mark.parametrize("system,n_compounds,n_missing,fragment", ADMITTED_ON_A_DOME)
    def test_admitted_on_a_two_liquid_dome(
        self, system, n_compounds, n_missing, fragment, matrix_data_env
    ):
        """Supported width below the floor, constraining total above it through a printed dome."""
        cov = _report(system)
        assert (cov.n_compounds, cov.n_missing_compounds) == (n_compounds, n_missing)
        floor = cov.thresholds["min_supported_total"]
        assert cov.supported_total < floor <= cov.constrained_total
        insufficient, reason = cov.is_insufficient()
        assert insufficient is False
        assert fragment in reason

    @pytest.mark.needs_cache
    def test_the_two_populations_do_not_overlap(self, matrix_data_env):
        """The floor sits in an empty band: no admitted system below it, none skipped above.

        Checked against the quantity the verdict reads -- the constraining total, supported
        width plus confirmed two-liquid domes -- not against the widest single segment,
        which does not gate.
        """
        admitted = min(_report(s).constrained_total for s in self.ADMITTED)
        skipped = max(
            _report(s).constrained_total for s in (self.SKIPPED + self.SKIPPED_ORDERED_FIELD)
        )
        floor = config.coverage_min_supported_total
        assert skipped < floor < admitted


class TestAttributionToASolidThatCanBeThere:
    """A stretch of liquidus is attributed to a solid whose field spans its temperature: not
    to a supported compound decomposing far below it (Bi-Pr, Co-La), and a drawn homogeneity
    range is kept over a narrower invariant-bounded field (In-Mg).
    """

    @pytest.mark.needs_cache
    @pytest.mark.parametrize(
        "system,phase,masked",
        [("Bi-Pr", "Pr5Bi3", [0.6107, 0.7801]), ("Co-La", "La2Co3", [0.2748, 0.4457])],
    )
    def test_a_supported_compound_below_the_curve_no_longer_truncates_a_mask(
        self, system, phase, masked, matrix_data_env
    ):
        cov = _report(system)
        record = next(p for p in cov.phases if p.name == phase)
        assert record.supported is True
        assert record.borders_liquid is False
        assert record.reason.startswith("subliquidus:")
        owner = next(p for p in cov.phases if p.interval[0] == pytest.approx(masked[0], abs=1e-3))
        assert owner.supported is False
        assert owner.interval[1] == pytest.approx(masked[1], abs=1e-3)

    @pytest.mark.needs_cache
    def test_a_wide_ordered_field_masks_the_width_it_is_drawn_at(self, matrix_data_env):
        """In-Mg: ``Mg0.1In0.9`` is drawn across most of the axis and is not in the hull.

        The mask reaches the field's drawn edge whatever solid solutions are loaded; the
        verdict does not, because the terminal ``(Mg)`` field's support depends on them.
        """
        cov = _report("In-Mg")
        record = next(p for p in cov.phases if p.name == "Mg0.1In0.9")
        assert record.supported is False
        assert record.interval[0] == pytest.approx(0.0661, abs=1e-3)
        assert cov.unsupported_ranges[0][0] == pytest.approx(0.0661, abs=1e-3)
        assert cov.unsupported_fraction > 0.65

    @pytest.mark.needs_cache
    def test_a_thermal_form_takes_the_peritectic_field_of_its_compound(self, matrix_data_env):
        """Li-Zn: the ht form's mask runs to the supported compound that bounds it."""
        cov = _report("Li-Zn")
        record = next(p for p in cov.phases if p.name == "Li2Zn5 ht")
        assert record.supported is False
        assert "peritectic" in record.reason
        assert cov.unsupported_ranges[-1][1] == pytest.approx(0.7592, abs=1e-3)


class TestPinnedDiagramsAreDeterministic:
    """A pin must name its diagram; ``pd_ind=None`` lets the filesystem pick one.

    ``load_mpds_data`` prefers the indexless ``<sys>.json`` and falls back to
    ``<sys>_MPDS_PD_0.json`` only when it is absent, so removing the former repoints every
    implicit load.
    """

    @pytest.fixture
    def flat_cache(self, tmp_path, monkeypatch):
        """A two-file cache whose entries differ only in which name they are stored under."""
        for name, entry in (("Cu-Mg.json", "indexless"), ("Cu-Mg_MPDS_PD_0.json", "pd0")):
            (tmp_path / name).write_text(
                json.dumps(
                    {"reference": {"entry": entry}, "chemical_elements": ["Cu", "Mg"], "shapes": []}
                ),
                encoding="utf-8",
            )
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "dir_structure", "flat")
        # Pin the mode too: this fixture IS a directory store, and leaving the mode ambient
        # would make the test depend on how the session happened to be configured (the root
        # conftest.py can swap a single-file store in process-wide).
        monkeypatch.setattr(config, "cache_mode", "directory")
        return tmp_path

    def test_explicit_index_ignores_the_indexless_sibling(self, flat_cache):
        from gliquid.mpds import load_mpds_data

        mpds_json, _ = load_mpds_data("Cu-Mg", pd_ind=0)
        assert mpds_json["reference"]["entry"] == "pd0"

    def test_implicit_index_is_shadowed_by_it(self, flat_cache):
        """The trap itself: same call, different diagram, no code change."""
        from gliquid.mpds import load_mpds_data

        mpds_json, _ = load_mpds_data("Cu-Mg", pd_ind=None)
        assert mpds_json["reference"]["entry"] == "indexless"

    def test_every_pin_case_names_its_diagram(self):
        cases = coverage_pin_cases(PINS_PATH)
        assert cases, "the pin fixture went missing"
        assert all(isinstance(expected["pd_ind"], int) for _, _, expected in cases)


def test_chart_function_renders_rows(capsys):
    from gliquid.mpds import print_phase_mismatch_chart

    print_phase_mismatch_chart([{"comp": 0.5, "name": "X"}], [0.25, 0.75])
    out = capsys.readouterr().out
    assert "MPDS:" in out and "MP:  " in out and "COMP:" in out
