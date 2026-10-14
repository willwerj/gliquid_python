"""Colorbar layout of the Nelder-Mead path figure."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gliquid.plotting.binary_figs import render_nelder_mead_path  # noqa: E402

_OBJECTIVE_LABEL = "Objective Function Value"
_ITERATION_LABEL = "Nelder-Mead Iteration"
_MIN_CLEARANCE_PX = 5.0


def _path_stub(n_iters=12):
    """The slice of a BinaryLiquid that render_nelder_mead_path reads."""
    rng = np.random.default_rng(0)
    path = rng.uniform(0, 1, (3, 3, n_iters))
    path[:, :2, :] *= 3e4
    fmt = SimpleNamespace(
        orders=[0, 1], param_names=["L0_a", "L1_a"], guess_params=["L0_a", "L1_a"]
    )
    xs_mix = SimpleNamespace(format=fmt, guess_param_indices=[0, 1])
    return SimpleNamespace(nmpath=path, xs_mix=xs_mix)


@pytest.mark.parametrize(
    ("objective_range", "scientific"),
    [
        ((1, 999), False),
        ((1, 9_999), False),
        ((0.1, 0.9), False),
        ((1, 12_000), True),
        ((1, 143_000), True),
        ((1, 5e6), True),
        ((0.01, 0.05), True),
        ((1e-3, 3.5e-3), True),
        ((1e-4, 5e-4), True),
    ],
)
def test_objective_colorbar_label_clears_iteration_colorbar(objective_range, scientific):
    """Objective tick labels go scientific from five characters, keeping the label off the neighbour."""
    fig = render_nelder_mead_path(_path_stub(), plot_a_params=True, objective_range=objective_range)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colorbars = {ax.yaxis.label.get_text(): ax for ax in fig.axes[1:]}
    objective = colorbars[_OBJECTIVE_LABEL]
    iteration = colorbars[_ITERATION_LABEL]
    gap = (
        iteration.get_window_extent(renderer).x0
        - objective.yaxis.label.get_window_extent(renderer).x1
    )
    has_exponent = bool(objective.yaxis.get_offset_text().get_text())
    plt.close(fig)

    assert has_exponent is scientific
    assert gap >= _MIN_CLEARANCE_PX
