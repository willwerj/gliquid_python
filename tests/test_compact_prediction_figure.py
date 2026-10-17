"""Layout contract of ProductionModelRunner.create_compact_prediction_figure."""

import re

import numpy as np
import pytest

shap = pytest.importorskip("shap")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_close = plt.close

from gliquid.production_model_runner import (  # noqa: E402
    COMPACT_FIGURE_SIZE,
    COMPACT_REFERENCE_FEATURE,
    ProductionModelRunner,
)

SHORT = {"symmetric": [f"s{i}" for i in range(12)], "antisymmetric": [f"a{i}" for i in range(9)]}
LONG = {
    "symmetric": [f"abs_argmin_ch_enth_liq_ref_{i}" for i in range(12)],
    "antisymmetric": ["comp_skew_liq_vol_change_from_rt_solid"] + [f"diff_{i}" for i in range(8)],
}


class _Stub:
    """The four runner methods the figure calls, over seeded synthetic SHAP values."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def get_rows_for_system(self, system_name):
        return "symmetric", "antisymmetric", None

    def _prepare_row(self, row, mode):
        return row

    def _ensure_explainer(self, target):
        pass

    def get_shap_explanation(self, target, mode):
        names = self.feature_names[mode]
        rng = np.random.default_rng(len(names) + len(target))
        values = rng.normal(0, 1e4, len(names))
        values[0] = 1e6  # the first feature is always the top label
        return shap.Explanation(
            values=values,
            base_values=-5e4,
            data=rng.normal(0, 1e5, len(names)),
            feature_names=names,
        )


def _render(tmp_path, feature_names, name):
    path = tmp_path / f"{name}.svg"
    ProductionModelRunner.create_compact_prediction_figure(_Stub(feature_names), "A-B", str(path))
    head = path.read_text(encoding="utf-8")[:2000]
    size = [float(re.search(rf'{k}="([\d.]+)pt"', head).group(1)) for k in ("width", "height")]
    return path.read_text(encoding="utf-8"), size


def test_saved_size_is_fixed_whatever_the_label_lengths(tmp_path):
    _, short = _render(tmp_path, SHORT, "short")
    _, long_ = _render(tmp_path, LONG, "long")
    expected = [72 * COMPACT_FIGURE_SIZE[0], 72 * COMPACT_FIGURE_SIZE[1]]
    assert short == pytest.approx(expected, abs=0.01)
    assert long_ == pytest.approx(expected, abs=0.01)


def test_labels_show_feature_names_as_written_and_inside_the_canvas(tmp_path, monkeypatch):
    kept = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: kept.setdefault("fig", plt.gcf()))
    _render(tmp_path, LONG, "long")
    fig = kept["fig"]
    labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
    assert any(label.startswith("comp_skew_liq_vol_change_from_rt_solid = ") for label in labels)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= width and bbox.y1 <= height
    _close("all")


def test_a_label_too_wide_for_the_figure_raises(tmp_path):
    names = {"symmetric": ["x" * 120], "antisymmetric": ["y"]}
    with pytest.raises(ValueError, match="widest SHAP label"):
        _render(tmp_path, names, "too_wide")


def _axes_left(tmp_path, monkeypatch, feature_names, name):
    kept = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: kept.setdefault("fig", plt.gcf()))
    _render(tmp_path, feature_names, name)
    left = kept["fig"].subplotpars.left
    _close("all")
    return left


def test_the_label_column_is_the_reference_width_unless_a_name_is_longer(tmp_path, monkeypatch):
    reference = {"symmetric": SHORT["symmetric"], "antisymmetric": [COMPACT_REFERENCE_FEATURE]}
    short = _axes_left(tmp_path, monkeypatch, SHORT, "short")
    assert _axes_left(tmp_path, monkeypatch, reference, "reference") == short
    assert _axes_left(tmp_path, monkeypatch, LONG, "long") > short
