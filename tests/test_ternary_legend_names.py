"""The ternary legend names an imputed phase by its standard formula unless its name reads as a
formula other than its composition."""

import numpy as np
import pandas as pd

from gliquid.plotting.ternary_surface import render_tx_surface

FLOOR = -200.0
COLOR_MAP = {"L": "#000000", "Ce12Fe34Si56": "#ff0000", "Ce2Fe2Si4": "#00aaaa"}


def test_a_formula_name_that_is_not_the_composition_keeps_its_spelling():
    solid = pd.DataFrame(
        [
            (0.25, 0.25, 900.0, "Ce12Fe34Si56"),
            (0.25, 0.50, 1000.0, "Ce2Fe2Si4"),
            (0.25, 0.25, FLOOR, "Ce12Fe34Si56"),
            (0.25, 0.50, FLOOR, "Ce2Fe2Si4"),
        ],
        columns=["x0", "x1", "T", "Phase"],
    )
    solid["x0_orig"], solid["x1_orig"] = solid["x0"], solid["x1"]
    solid["Colors"] = solid["Phase"].map(COLOR_MAP)
    liq = pd.DataFrame(
        {
            "x0": [0.0, 0.5, 1.0],
            "x1": [0.0, np.sqrt(3) / 2, 0.0],
            "T": [1200.0, 1300.0, 1400.0],
            "x0_orig": [0.0, 0.0, 1.0],
            "x1_orig": [0.0, 1.0, 0.0],
            "coexistent_phases": ["Ce2Fe2Si4", "Ce12Fe34Si56", "Ce2Fe2Si4"],
        }
    )
    fig = render_tx_surface(
        solid,
        liq,
        np.array([[0, 1, 2]]),
        COLOR_MAP,
        ["Ce", "Fe", "Si"],
        [FLOOR, 1500.0],
        imputed_phases=["Ce12Fe34Si56", "Ce2Fe2Si4"],
    )
    legend = {trace.name for trace in fig.data if trace.showlegend}
    assert legend == {
        "Ce<sub>12</sub>Fe<sub>34</sub>Si<sub>56</sub> (imputed)",
        "CeFeSi<sub>2</sub> (imputed)",
    }
