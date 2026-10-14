"""The ``gliquid.imputation`` façade: its exported names and where each one resolves.

Loading no submodule on ``import gliquid.imputation`` is checked in ``test_import_graph.py``.
"""

from importlib import import_module

import pytest

import gliquid.imputation as imputation

EXPECTED_EXPORTS = {
    "impute_phases": "gliquid.imputation.lifecycle",
    "drop_imputed_phases": "gliquid.imputation.lifecycle",
    "imputed_entry_staleness": "gliquid.imputation.lifecycle",
    "stale_imputed_entries": "gliquid.imputation.lifecycle",
    "imputed_hull": "gliquid.imputation.binary_imputation",
    "reanchor_imputed_entries": "gliquid.imputation.core",
    "imputed_entry_label": "gliquid.cache",
    "ImputationResult": "gliquid.imputation.lifecycle",
    "BinaryImputationResult": "gliquid.imputation.binary_imputation",
    "TernaryImputationResult": "gliquid.imputation.ternary_imputation",
    "list_binary_imputation_targets": "gliquid.imputation.binary_imputation",
    "impute_ternary_phase": "gliquid.imputation.ternary_imputation",
    "adjust_ternary_phase_temperature": "gliquid.imputation.ternary_imputation",
    "IMPUTED_ID_PREFIX": "gliquid.cache",
    "PROVENANCE_KEY": "gliquid.imputation.core",
    "REPLACES_KEY": "gliquid.imputation.core",
    "CLAMP_KEY": "gliquid.imputation.core",
}


def test_every_exported_name_is_the_object_its_home_module_defines():
    for name in imputation.__all__:
        assert getattr(imputation, name) is getattr(import_module(imputation._EXPORTS[name]), name)


def test_all_is_the_declared_export_list():
    assert imputation.__all__ == list(imputation._EXPORTS)
    assert imputation._EXPORTS == EXPECTED_EXPORTS
    assert list(imputation._EXPORTS) == list(EXPECTED_EXPORTS)


def test_an_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        imputation.not_an_export  # noqa: B018
