"""Characterization tests for the shared Plotly static-image export helpers.

Written (secondary-track S1) just before the module moved to gliquid/plotting/export.py,
gating the move. The kaleido/subprocess boundary is monkeypatched — no static-image
engine is required to run these.
"""

import subprocess
from io import StringIO

import pytest

import gliquid.plotting.export as pe


class _FakeFig:
    """Stands in for go.Figure: records inline write_image calls, serializes for the child."""

    def __init__(self):
        self.calls = []

    def write_image(self, stream, **kwargs):
        self.calls.append((stream, kwargs))

    def to_json(self):
        return '{"data": [], "layout": {}}'


class TestResolveStreamPath:
    def test_path_string_passes_through(self):
        assert pe.resolve_stream_path("out/fig.svg") == "out/fig.svg"

    def test_named_stream_resolves_to_its_path(self, tmp_path):
        target = tmp_path / "fig.svg"
        with open(target, "w", encoding="utf-8") as handle:
            assert pe.resolve_stream_path(handle) == str(target)

    def test_anonymous_stream_raises_typeerror(self):
        with pytest.raises(TypeError, match="file path or a named stream"):
            pe.resolve_stream_path(StringIO())


class TestWriteImageWithTimeout:
    def test_timeout_none_exports_inline(self):
        fig = _FakeFig()
        pe.write_image_with_timeout(fig, "fig.svg", None, scale=2)
        assert fig.calls == [("fig.svg", {"scale": 2})]

    def test_timeout_nonpositive_exports_inline(self):
        fig = _FakeFig()
        pe.write_image_with_timeout(fig, "fig.svg", 0)
        assert fig.calls == [("fig.svg", {})]

    def test_subprocess_timeout_becomes_timeouterror(self, monkeypatch):
        def _expire(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        monkeypatch.setattr(pe.subprocess, "run", _expire)
        with pytest.raises(TimeoutError, match="timed out after 5.0s"):
            pe.write_image_with_timeout(_FakeFig(), "fig.svg", 5.0)

    def test_child_failure_becomes_runtimeerror_with_stderr(self, monkeypatch):
        completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
        monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: completed)
        with pytest.raises(RuntimeError, match="boom"):
            pe.write_image_with_timeout(_FakeFig(), "fig.svg", 5.0)

    def test_success_invokes_child_on_the_stream_path(self, monkeypatch, tmp_path):
        seen = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(pe.subprocess, "run", _fake_run)
        out_path = str(tmp_path / "fig.svg")
        pe.write_image_with_timeout(_FakeFig(), out_path, 7.5)
        assert seen["timeout"] == 7.5
        assert out_path in seen["cmd"]
