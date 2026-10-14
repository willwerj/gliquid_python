"""API façade contract (PEP-562 lazy exports in gliquid/__init__.py).

The façade IS the public surface: every name in __all__ must resolve through
`import gliquid`, and the bare import must stay light (no matplotlib/plotly).
Any change to __all__ is a breaking change needing explicit sign-off.
"""

import pytest

import gliquid

from ._fresh import ok, run_fresh


class TestFacadeExports:
    def test_every_all_name_resolves(self):
        for name in gliquid.__all__:
            assert getattr(gliquid, name) is not None, name

    def test_exports_match_declared_surface(self):
        assert set(gliquid.__all__) == {*gliquid._EXPORTS, "__version__"}

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            gliquid.not_a_real_export  # noqa: B018

    def test_dir_lists_exports(self):
        listing = dir(gliquid)
        assert "BinaryLiquid" in listing and "set_cache_dir" in listing
        assert "set_data_dir" not in listing


def test_import_gliquid_is_lazy(tmp_path):
    """A bare `import gliquid` must not drag in matplotlib or plotly."""
    code = (
        "import gliquid, sys; "
        "heavy = [m for m in ('matplotlib', 'plotly') if m in sys.modules]; "
        "assert not heavy, f'heavy imports leaked: {heavy}'; "
        "print('OK')"
    )
    ok(run_fresh(code, tmp_path))


def test_runner_frames_is_the_cache_constant():
    import gliquid.cache as gliquid_cache
    import gliquid.production_model_runner as runner

    assert runner.FRAMES is gliquid_cache.ML_FRAMES


def test_binary_reexports_only_the_released_plot_names():
    """gliquid.binary re-exports the six released plot-stack names and no private helper."""
    import gliquid.binary as binary
    import gliquid.plotting.binary_tx as binary_tx
    import gliquid.plotting.style as style

    homes = {
        "plot_tx": binary_tx,
        "record_tie_lines": binary_tx,
        "build_polymorph_transitions": binary_tx,
        "SS_FIXED_COLORS": style,
        "build_phase_color_map": style,
        "format_phase_display_name": style,
    }
    for name, home in homes.items():
        assert getattr(binary, name) is getattr(home, name), name
    for name in (
        "_pack_labels",
        "_OrientationLatch",
        "_ss_regions",
        "_PLOT_W_PX",
        "_abbreviate_phase_name",
    ):
        assert not hasattr(binary, name), name
