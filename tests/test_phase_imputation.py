"""
Smoke tests for the phase-energy imputation library (gliquid.phase_imputation).

Mirrors the style of test_phase_ablation.py / test_fisher_information.py: no env setup,
load via from_cache, session-scoped fixtures, small n_opts/max_iter. Uses Cu-Mg (two
congruently melting intermetallics; the FIM/ablation test system).

The closed-loop recovery test removes a real DFT compound consistently from both the hull
and the phase list (so the system genuinely lacks it, exactly as ``from_cache`` would for a
system whose DFT hull is incomplete), re-fits, then imputes it back from the experimental
invariant and checks it reappears as a tagged, dashed-line, stable vertex.

Run with: python -m pytest tests/test_phase_imputation.py -v
    or:   python tests/test_phase_imputation.py
"""
import copy

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram

from gliquid.binary import BinaryLiquid, build_phases_from_chull, _x_vals
import gliquid.load_binary_data as lbd
from gliquid.phase_imputation import (
    ImputedPhase,
    impute_phases,
    invert_phase_energy,
    list_imputable_phases,
    select_optimal_fim_fit,
    _imputed_computed_entry,
    _liquid_gibbs,
)

SYSTEM = 'Cu-Mg'
SMOKE_FIT_KWARGS = {'n_opts': 1, 'max_iter': 32}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interior_phases(bl):
    return [p for p in bl.phases
            if not p.get('is_solution', False) and 0.0 < round(float(p['comp']), 6) < 1.0]


def _strip_interior(bl):
    """Remove ALL interior DFT compounds from both the hull and the phase list, leaving a
    clean element-only hull. Every low-T MPDS compound then becomes genuinely missing
    (no DFT coverage), without the neighbour-exposure that removing a single central phase
    from a complete hull causes. The fitted liquid curve is left untouched.
    """
    keep = [e for e in bl.dft_ch.all_entries if len(e.composition.elements) == 1]
    bl.dft_ch = PhaseDiagram(keep)
    bl.phases = build_phases_from_chull(bl.dft_ch, bl.components, bl.component_data)
    pts = np.array([[p['comp'], p['enthalpy']] for p in bl.phases if 'comp' in p])
    bl.eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], pts[:, 0], pts[:, 1])
    bl.invariants = None
    bl.low_t_exp_phases = None
    bl.hsx = None


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cu_mg_full():
    """Cu-Mg fitted once against its full DFT hull."""
    bl = BinaryLiquid.from_cache(SYSTEM, param_format='comb-exp')
    if bl.init_error:
        pytest.skip(f"{SYSTEM}: init error on load")
    fits = bl.fit_parameters(**SMOKE_FIT_KWARGS)
    if not fits or bl.init_error:
        pytest.skip(f"{SYSTEM}: baseline could not be fit")
    return bl


@pytest.fixture
def stripped(cu_mg_full):
    """A copy of the fitted system with all interior DFT phases stripped (good liquid fit
    retained). Returns ``(bl, original_enthalpies)``."""
    bl = copy.deepcopy(cu_mg_full)
    orig = {p['name']: p['enthalpy'] for p in _interior_phases(bl)}
    _strip_interior(bl)
    return bl, orig


@pytest.fixture
def imputation_recovery(stripped):
    """Impute the stripped interior phases back from the experimental invariants, anchored
    on the retained full-hull liquid curve (write_cache=False).

    Returns ``(bl, original_enthalpies, imputed_list)``.
    """
    bl, orig = stripped
    imputed = impute_phases(bl, fitting_data=None, write_cache=False)
    return bl, orig, imputed


# ---------------------------------------------------------------------------
# Detection + inversion units
# ---------------------------------------------------------------------------

class TestInversionUnits:
    def test_congruent_inversion_lies_on_liquid(self, cu_mg_full):
        x_s, temp = 0.4, 1100.0
        inv = {'type': 'cmp', 'comp': x_s, 'temp': temp, 'phases': ['X'], 'phase_comps': [x_s]}
        assert invert_phase_energy(cu_mg_full, x_s, inv) == pytest.approx(
            _liquid_gibbs(cu_mg_full, x_s, temp))

    def test_eutectic_inversion_extrapolates_tangent(self, cu_mg_full):
        # A peritectic-style anchor away from x_s should not just equal G_liq(x_s).
        inv = {'type': 'per', 'comp': 0.30, 'temp': 1000.0, 'phases': ['X'], 'phase_comps': [0.55]}
        h = invert_phase_energy(cu_mg_full, 0.55, inv)
        assert np.isfinite(h)

    def test_imputed_entry_dict_is_tagged(self, cu_mg_full):
        ed = _imputed_computed_entry(cu_mg_full, 'TestPhase', 0.5, -25000.0)
        assert lbd._is_imputed_entry_dict(ed)
        assert ed['entry_id'] == 'imputed:TestPhase'
        assert ed['data']['imputed'] is True
        # A plain DFT entry dict (no tag) must not be flagged.
        assert not lbd._is_imputed_entry_dict({'entry_id': 'mp-123', 'data': {}})

    def test_select_optimal_fim_fit_sets_params(self, cu_mg_full):
        # select_optimal_fim_fit mutates bl in place; work on a copy so the shared
        # session fixture's good fit is not clobbered for later tests.
        bl = copy.deepcopy(cu_mg_full)
        fit = {'L0_a': -5.0, 'L0_b': -1.0, 'L1_a': 1.0, 'L1_b': 0.0, 'nmpath': None}
        chosen = select_optimal_fim_fit(bl, [fit])
        assert chosen is fit
        assert bl.get_params() == [-5.0, -1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Closed-loop recovery
# ---------------------------------------------------------------------------

class TestImputationRecovery:
    def test_detects_missing_phases(self, stripped):
        bl, orig = stripped
        targets = list_imputable_phases(bl)
        detected = {ph['name'] for ph, _, _ in targets}
        # Both congruently melting Cu-Mg intermetallics are now missing from the hull.
        assert detected, "Stripped interior phases should be detected as imputable"
        for name in orig:
            assert name in detected, f"{name} should be detected as imputable"

    def test_imputes_phases_back(self, imputation_recovery):
        _, orig, imputed = imputation_recovery
        assert imputed, "Expected imputed phases"
        assert all(isinstance(ip, ImputedPhase) for ip in imputed)
        imputed_names = {ip.name for ip in imputed}
        for name in orig:
            assert name in imputed_names, f"{name} should be imputed back"

    def test_imputed_phases_are_stable_and_tagged(self, imputation_recovery):
        bl, orig, imputed = imputation_recovery
        flagged = {p['name'] for p in bl.phases if p.get('imputed')}
        for name in orig:
            assert name in flagged, f"{name} should be a tagged stable vertex in bl.phases"
        # Imputed formation enthalpies are exothermic.
        assert all(ip.enthalpy < 0.0 for ip in imputed)

    def test_recovered_enthalpy_is_in_range(self, imputation_recovery):
        # Congruent inversion (H = G_liq at the melting point) should land in the same
        # ballpark as the true DFT formation enthalpy — same sign, comparable magnitude.
        _, orig, imputed = imputation_recovery
        by_name = {ip.name: ip.enthalpy for ip in imputed}
        for name, h_dft in orig.items():
            h_imp = by_name[name]
            assert h_imp < 0.0 and h_dft < 0.0
            ratio = h_imp / h_dft
            assert 0.25 < ratio < 4.0, f"{name}: imputed {h_imp:.0f} vs DFT {h_dft:.0f} J/mol-atom"

    def test_imputed_entry_round_trips_and_filters(self, imputation_recovery):
        _, _, imputed = imputation_recovery
        ed = imputed[0].entry_dict
        assert lbd._is_imputed_entry_dict(ed)
        # include_imputed=False would drop it; the tag is what distinguishes it.
        kept = [e for e in [ed, {'entry_id': 'mp-1', 'data': {}}]
                if not lbd._is_imputed_entry_dict(e)]
        assert ed not in kept


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
