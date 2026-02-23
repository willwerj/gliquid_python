"""
Runtime compatibility patches for the ``shap`` package.

Old XGBoost pickles (≤ v1.x) may serialize ``base_score`` as a single-element
list or array (e.g. ``[0.305]``).  SHAP ≥ 0.44 calls ``float()`` on that value
directly, which raises ``TypeError`` or ``ValueError``.  Rather than editing
site-packages (which is lost on every env rebuild), this module monkey-patches
the relevant SHAP internals at runtime.

The ``waterfall`` plot patch removes the E[f(X)] and f(x) twin-axis tick marks
from the top of waterfall plots, producing a cleaner figure.

Usage
-----
Call **once**, early in your script, before any ``shap.TreeExplainer`` is
created or any ``shap.plots.waterfall`` is called::

    from gliquid.shap_compat import apply_patches
    apply_patches()

Adding new patches
------------------
1. Write a ``def patch_<name>():`` function in this module.
2. Guard it with ``_APPLIED`` so it is idempotent.
3. Register it in :func:`apply_patches`.

The pattern used for the base-score fix (overriding ``float`` in the target
module's global namespace) works for *any* builtin that needs special handling
in a third-party module.  See :func:`patch_xgb_base_score` for the template.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

log = logging.getLogger(__name__)

_APPLIED: set[str] = set()

# ---------------------------------------------------------------------------
# Patch 1 – XGBTreeModelLoader base_score bracket fix
# ---------------------------------------------------------------------------

_builtin_float = float  # stash the real builtin


def _safe_float(value: Any) -> float:
    """Drop-in ``float()`` replacement that tolerates ``[0.305]``."""
    try:
        return _builtin_float(value)
    except (ValueError, TypeError):
        return _builtin_float(str(value).strip("[]"))


def patch_xgb_base_score() -> None:
    """Inject a bracket-tolerant ``float`` into ``shap.explainers._tree``.

    Python's ``LOAD_GLOBAL`` bytecode instruction checks the module's own
    ``__dict__`` *before* falling back to ``builtins``.  By assigning our
    ``_safe_float`` as ``float`` in the ``_tree`` module's namespace, every
    ``float(...)`` call inside that module—including the two problematic
    ``base_score`` conversions—will use our version instead of the builtin.

    The wrapper is *invisible* for well-formed inputs (the ``try`` block
    succeeds identically to the real ``float``).  It only activates when the
    native ``float()`` would have raised, in which case it strips ``[]``
    brackets and retries.
    """
    tag = "xgb_base_score"
    if tag in _APPLIED:
        return

    try:
        import shap.explainers._tree as tree_mod  # noqa: WPS433
    except ImportError:
        log.debug("shap not installed – skipping patch_xgb_base_score")
        return

    tree_mod.float = _safe_float  # type: ignore[attr-defined]
    _APPLIED.add(tag)
    log.debug("Applied shap_compat patch: %s", tag)


def unpatch_xgb_base_score() -> None:
    """Revert :func:`patch_xgb_base_score` (useful for testing)."""
    tag = "xgb_base_score"
    if tag not in _APPLIED:
        return

    import shap.explainers._tree as tree_mod  # noqa: WPS433

    if hasattr(tree_mod, "float"):
        del tree_mod.float  # reverts LOAD_GLOBAL to builtins.float
    _APPLIED.discard(tag)


# ---------------------------------------------------------------------------
# Patch 2 – Remove E[f(X)] / f(x) twin-axis tick marks from waterfall plot
# ---------------------------------------------------------------------------


def patch_waterfall_no_twin_axes() -> None:
    """Remove the E[f(X)] and f(x) twin-axis tick marks from waterfall plots.

    The default ``shap.plots.waterfall`` draws two extra twin axes at the top
    of the figure showing the base value ``E[f(X)]`` and the prediction
    ``f(x)``.  These are often unnecessary clutter.  This patch wraps the
    original function: it calls it with ``show=False``, removes the twin axes,
    and then either shows or returns the main axis as usual.
    """
    tag = "waterfall_no_twin_axes"
    if tag in _APPLIED:
        return

    try:
        import shap.plots._waterfall as waterfall_mod  # noqa: WPS433
        import shap.plots as plots_mod  # noqa: WPS433
    except ImportError:
        log.debug("shap not installed – skipping patch_waterfall_no_twin_axes")
        return

    _original_waterfall = waterfall_mod.waterfall

    @functools.wraps(_original_waterfall)
    def _patched_waterfall(shap_values, max_display=10, show=True):  # noqa: WPS430
        import matplotlib.pyplot as plt  # noqa: WPS433

        # Save interactive state; the original calls plt.ioff() when show=False
        was_interactive = plt.isinteractive()

        # Snapshot axes that exist *before* the waterfall call so we only
        # remove the twin axes that SHAP adds, not user-created subplots.
        fig = plt.gcf()
        axes_before = set(fig.get_axes())
        # The *real* main axis is the one the caller set (e.g. via plt.sca).
        main_ax = plt.gca()

        # Run the original without displaying.
        # Note: the original returns plt.gca() which ends up being ax3
        # (the second twin axis for f(x)), NOT the main plotting axis.
        _original_waterfall(shap_values, max_display=max_display, show=False)

        # Remove ALL newly created axes (both E[f(X)] and f(x) twins).
        for extra_ax in reversed(fig.get_axes()):
            if extra_ax not in axes_before:
                fig.delaxes(extra_ax)

        # Restore the main axis as current
        plt.sca(main_ax)

        # Restore interactive state
        if was_interactive:
            plt.ion()

        if show:
            plt.show()
        else:
            return main_ax

    waterfall_mod.waterfall = _patched_waterfall
    # Also patch the re-exported reference in shap.plots
    if hasattr(plots_mod, "waterfall"):
        plots_mod.waterfall = _patched_waterfall
    _APPLIED.add(tag)
    log.debug("Applied shap_compat patch: %s", tag)


def unpatch_waterfall_no_twin_axes() -> None:
    """Revert :func:`patch_waterfall_no_twin_axes` (useful for testing)."""
    tag = "waterfall_no_twin_axes"
    if tag not in _APPLIED:
        return

    try:
        # Re-importing the module won't undo our assignment, so we reload it.
        import importlib

        import shap.plots._waterfall as waterfall_mod  # noqa: WPS433
        import shap.plots as plots_mod  # noqa: WPS433

        importlib.reload(waterfall_mod)
        if hasattr(plots_mod, "waterfall"):
            plots_mod.waterfall = waterfall_mod.waterfall
    except Exception:  # noqa: BLE001
        log.debug("Could not unpatch waterfall_no_twin_axes", exc_info=True)
    _APPLIED.discard(tag)


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------

def apply_patches() -> None:
    """Apply every registered compatibility patch (idempotent)."""
    patch_xgb_base_score()
    patch_waterfall_no_twin_axes()
    # ---- add future patches here ----
