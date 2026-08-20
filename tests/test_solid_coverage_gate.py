"""Solid-energy coverage gate: the single measured fit/skip criterion.

A liquidus point constrains the liquid free energy only if the solid it is in equilibrium
with has a known free energy, so ``fit_parameters`` admits a system on the measured fraction
of its liquidus that is conjugate to a solid we cannot evaluate
(``BinaryLiquid.assess_solid_coverage`` -> ``mpds.assess_solid_coverage``).

This replaces two gates that disagreed with each other:
  * a count of near-liquidus MPDS compounds missing from DFT, which campaigns disabled; and
  * a full-composition-solid-solution check that matched the *label string* ``'(A, B)'`` and
    so never saw a wide field spelled ``'(Lu)'``.
``full_comp_ss`` survives as provenance only.

Hf-Zr pd0 is the offline full-composition-SS system (MPDS labels '(Hf, Zr)'); with
``ss_models`` loaded it is fully supported, without them it is fully unsupported.

The 52 frozen ground-truth verdicts that replay this gate over the whole acceptance corpus
are maintainer machinery -- they need the workspace ``matrix_data`` store -- and live in
``tests_internal/test_solid_coverage_pins.py``. What stays here is the gate's own behaviour.
"""

import json
from pathlib import Path

import pytest

import gliquid.config as config
from gliquid.binary import BinaryLiquid

PINS_PATH = Path(__file__).parent / "fixtures" / "solid_coverage_pins.json"


class TestSkipWithoutSsModels:
    def test_fit_skips_and_flags_init_error(self, caplog):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        result = bl.fit_parameters()
        assert result == []
        # The skip is now an initialization error: drivers already branch on init_error to
        # tell a deliberate data-adequacy skip from a failed optimization.
        assert bl.init_error is True
        assert bl.full_comp_ss is True  # still recorded, no longer load-bearing
        assert "Insufficient solid-phase energy information" in caplog.text
        assert "skipping fit" in caplog.text

    def test_report_attributes_the_skip_to_the_unsupported_field(self):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters()
        cov = bl.coverage_report
        assert cov.unsupported_fraction == pytest.approx(1.0, abs=1e-3)
        assert cov.ss_models == ()
        assert cov.is_insufficient()[0] is True
        unsupported = [p for p in cov.phases if not p.supported]
        assert unsupported, "the (Hf, Zr) field should be recorded as unsupported"
        # MPDS leaves '(A, B)' shapes structurally unresolved, so this is the fallback path.
        assert all(p.reason == "unknown_structure_no_ss" for p in unsupported)


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
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(
            n_opts=1,
            max_iter=2,
            coverage_thresholds={"skip_frac": 1.0, "min_missing": 2, "missing_frac": 0.5},
        )
        assert bl.init_error is False
        assert bl.coverage_report.unsupported_fraction == pytest.approx(1.0, abs=1e-3)

    def test_retired_kwarg_warns_instead_of_silently_no_opping(self, caplog):
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)
        bl.fit_parameters(n_opts=1, max_iter=2, check_phase_mismatch=False)
        assert "'check_phase_mismatch' is retired" in caplog.text
        # and the coverage gate still fires despite the old opt-out
        assert bl.init_error is True


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


def _pin_cases():
    """(ref_mode, system, expected) for every pinned case, or nothing if pins are absent.

    ``expected`` carries a resolved ``pd_ind``: the fixture's top-level default, overridden
    per system when an entry names its own. Load-bearing -- see the note in
    ``tests_internal/test_solid_coverage_pins.py`` on why the index may not be left implicit.
    Kept here (and duplicated there) because ``TestPinnedDiagramsAreDeterministic`` asserts
    the property this reader depends on.
    """
    if not PINS_PATH.exists():
        return []
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    default_pd_ind = pins["pd_ind"]
    return [
        (ref_mode, system, {"pd_ind": default_pd_ind, **expected})
        for ref_mode, arm in pins["arms"].items()
        for system, expected in arm.items()
    ]


class TestPinnedDiagramsAreDeterministic:
    """A pin must name its diagram; ``pd_ind=None`` lets the filesystem pick one.

    ``load_mpds_data`` prefers the indexless ``<sys>.json`` and falls back to
    ``<sys>_MPDS_PD_0.json`` only when it is absent, so removing the former repoints every
    implicit load without touching a line of code. Sweeping those files out of
    ``matrix_data`` is exactly what moved the Cu-Mg pin from entry C100123 to C900864.
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
        monkeypatch.setattr(config, "data_dir", tmp_path)
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
        cases = _pin_cases()
        assert cases, "the pin fixture went missing"
        assert all(isinstance(expected["pd_ind"], int) for _, _, expected in cases)


def test_chart_function_renders_rows(capsys):
    from gliquid.mpds import print_phase_mismatch_chart

    print_phase_mismatch_chart([{"comp": 0.5, "name": "X"}], [0.25, 0.75])
    out = capsys.readouterr().out
    assert "MPDS:" in out and "MP:  " in out and "COMP:" in out
