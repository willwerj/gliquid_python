"""Robust Plotly static-image export shared by the plotting classes.

``write_image_with_timeout`` isolates the kaleido/orca export in a child interpreter so a hung
export raises ``TimeoutError`` instead of blocking the caller. Used by both
``gliquid.binary.BLPlotter`` and ``gliquid.ternary.TLIPlotter``.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from io import StringIO

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def resolve_stream_path(stream: str | StringIO) -> str:
    """The filesystem path Plotly should write to (a path string, or a named stream)."""
    if isinstance(stream, str):
        return stream
    if hasattr(stream, "name") and stream.name:
        return str(stream.name)
    raise TypeError("Plotly image export requires a file path or a named stream.")


def write_image_with_timeout(
    fig: go.Figure, stream: str | StringIO, timeout_s: float, **write_kwargs
) -> None:
    """Write ``fig`` to ``stream`` as a static image, bounding the export by ``timeout_s``.

    ``timeout_s`` None/<=0 exports inline (no isolation); otherwise the export runs in a child
    interpreter and a ``subprocess.TimeoutExpired`` becomes a ``TimeoutError``.
    """
    if timeout_s is None or timeout_s <= 0:
        fig.write_image(stream, **write_kwargs)
        return

    stream_path = resolve_stream_path(stream)
    with tempfile.TemporaryDirectory(prefix="gliquid_plotly_export_") as temp_dir:
        figure_payload_path = os.path.join(temp_dir, "figure_payload.json")
        kwargs_payload_path = os.path.join(temp_dir, "write_kwargs.json")

        with open(figure_payload_path, "w", encoding="utf-8") as payload_file:
            payload_file.write(fig.to_json())
        with open(kwargs_payload_path, "w", encoding="utf-8") as kwargs_file:
            json.dump(write_kwargs, kwargs_file)

        child_code = (
            "import json\n"
            "import pathlib\n"
            "import plotly.io as pio\n"
            "import sys\n"
            "figure_json = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')\n"
            "write_kwargs = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding='utf-8'))\n"
            "figure = pio.from_json(figure_json)\n"
            "figure.write_image(sys.argv[2], **write_kwargs)\n"
        )

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    figure_payload_path,
                    stream_path,
                    kwargs_payload_path,
                ],
                timeout=timeout_s,
                check=False,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Plotly image export timed out after {timeout_s:.1f}s") from exc

        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            stdout_text = (completed.stdout or "").strip()
            details = stderr_text or stdout_text or "unknown subprocess error"
            raise RuntimeError(f"Plotly subprocess export failed: {details}")


def show_figure(fig) -> None:
    """Display a produced figure: Plotly via ``fig.show``, Matplotlib via its canvas."""
    if isinstance(fig, go.Figure):
        fig.show()
        return
    import matplotlib.pyplot as plt  # lazy: keeps this module plotly-only for mpl-free callers

    if isinstance(fig, plt.Figure):
        fig.figure.show()
        return
    # Neither backend claimed it: nothing was displayed. Audible, not silent.
    logger.warning(
        "show_figure got an unsupported figure type %s; nothing was displayed.", type(fig).__name__
    )


def save_figure(
    fig,
    stream: str | StringIO,
    image_format: str = "svg",
    export_timeout_s: float = 120.0,
    label: str = "",
    **write_kwargs,
) -> None:
    """Save a produced figure to ``stream``: Plotly through the timeout-guarded subprocess
    export, Matplotlib through ``savefig``. A named StringIO's extension overrides
    ``image_format``; ``label`` names the figure in export-failure messages."""
    image_format = (
        stream.name.split(".")[-1] if isinstance(stream, StringIO) and stream.name else image_format
    )

    if isinstance(fig, go.Figure):
        merged_kwargs = {"format": image_format, **write_kwargs}
        try:
            write_image_with_timeout(fig, stream, timeout_s=export_timeout_s, **merged_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export plot '{label}' to '{stream}' with timeout={export_timeout_s:.1f}s: {exc}"
            ) from exc
        return
    import matplotlib.pyplot as plt  # lazy: keeps this module plotly-only for mpl-free callers

    if isinstance(fig, plt.Figure):
        fig.figure.savefig(stream, format=image_format)
        plt.close(fig)
        return
    # Neither backend claimed it: NOTHING was written to `stream`, and the caller would
    # otherwise learn that only from the missing file.
    logger.warning(
        "save_figure got an unsupported figure type %s for '%s'; no image was written to '%s'.",
        type(fig).__name__,
        label,
        stream,
    )
