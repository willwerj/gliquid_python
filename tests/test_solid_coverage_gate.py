"""Solid-energy coverage gate: the single quantitative fit/skip criterion.

A liquidus point constrains the liquid free energy only if the solid it is in equilibrium
with has a known free energy, so ``fit_parameters`` admits a system on the fraction of its
liquidus that is conjugate to a solid we cannot evaluate
(``BinaryLiquid.assess_solid_coverage`` -> ``solid_coverage.assess_solid_coverage``).
``full_comp_ss`` is provenance only.

Hf-Zr pd0 is the offline full-composition-SS system (MPDS labels '(Hf, Zr)'); with
``ss_models`` loaded it is fully supported, without them it is fully unsupported.
"""

import json

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid


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
        (alpha) field tops out below the digitized liquidus minimum, so it never borders the
        liquid and is excluded from the mask; the ht (beta) field reaches the liquidus and
        alone masks the whole axis.
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


class TestPinnedDiagramsAreDeterministic:
    """``pd_ind=None`` lets the filesystem pick the diagram, so a pin must name it.

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


def test_chart_function_renders_rows(capsys):
    from gliquid.mpds import print_phase_mismatch_chart

    print_phase_mismatch_chart([{"comp": 0.5, "name": "X"}], [0.25, 0.75])
    out = capsys.readouterr().out
    assert "MPDS:" in out and "MP:  " in out and "COMP:" in out
