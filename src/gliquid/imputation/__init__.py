"""Phase-energy imputation for binary and ternary systems.

Each name in ``__all__`` is read from the submodule that defines it when it is accessed; importing
this package imports none of its submodules.
"""

from importlib import import_module

_EXPORTS: dict[str, str] = {
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

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
