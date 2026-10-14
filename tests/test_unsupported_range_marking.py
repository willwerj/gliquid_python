"""Which fits mark ``unsupported_comp_ranges`` on the assessed liquidus of the T-x plots.

'ignore' and 'cap' treat those ranges differently in the objective, so the plot marks them;
'score' scores them like any other composition, so the liquidus is drawn in one style.
Runs offline on the Hf-Zr fixtures.
"""

import pytest

from gliquid.binary import DEFAULT_UNSUPPORTED_RANGE_MODE, BinaryLiquid, BLPlotter
from gliquid.plotting.style import ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR

RANGES = [[0.2, 0.5]]


@pytest.fixture(scope="module")
def bl():
    return BinaryLiquid.from_cache("Hf-Zr", pd_ind=0)


def _assessed_colors(fig):
    lines = (getattr(trace, "line", None) for trace in fig.data)
    colors = {getattr(line, "color", None) for line in lines}
    return colors & {ASSESSED_LIQUIDUS_COLOR, ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR}


@pytest.mark.parametrize("plot_type", ["fit+liq", "pred+liq"])
@pytest.mark.parametrize(
    ("mode", "marked"), [(None, True), ("ignore", True), ("cap", True), ("score", False)]
)
def test_the_fit_mode_decides_whether_unsupported_ranges_are_marked(bl, plot_type, mode, marked):
    bl.unsupported_comp_ranges = [list(r) for r in RANGES]
    bl.fit_unsupported_range_mode = mode

    colors = _assessed_colors(BLPlotter(bl).get_plot(plot_type))

    assert (ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR in colors) is marked
    assert ASSESSED_LIQUIDUS_COLOR in colors


def test_fit_parameters_records_the_mode_it_resolved():
    bl = BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )
    assert bl.fit_unsupported_range_mode is None

    bl.fit_parameters(n_opts=1, max_iter=4)
    assert bl.fit_unsupported_range_mode == DEFAULT_UNSUPPORTED_RANGE_MODE

    bl.fit_parameters(n_opts=1, max_iter=4, unsupported_range_mode="cap")
    assert bl.fit_unsupported_range_mode == "cap"
