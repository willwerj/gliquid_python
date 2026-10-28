"""No comment block in a tracked source or config file runs longer than ``MAX_BLOCK`` lines.

A comment says what the code does; a longer explanation belongs in a docstring, CONTRIBUTING.md
or CHANGELOG.md. Python files are read with ``tokenize``, so ``#`` lines inside string literals
are not comments. Config files count full-line ``#`` comments.
"""

from __future__ import annotations

import io
import subprocess
import tokenize
from pathlib import Path

import pytest

MAX_BLOCK = 3

#: Tracked non-Python files whose ``#`` comments are checked.
CONFIG_FILES = (
    "pyproject.toml",
    "tox.ini",
    ".gitignore",
    "CITATION.cff",
    ".githooks/pre-commit",
    ".github/workflows/release.yml",
    ".github/workflows/tests.yml",
)

#: ``(path, first line)`` of blocks allowed past ``MAX_BLOCK``. Keep this empty or tiny.
ALLOWED: frozenset[tuple[str, int]] = frozenset()

ROOT = Path(__file__).resolve().parent.parent


def python_comment_blocks(source: str) -> list[tuple[int, int]]:
    """``(first line, length)`` of each run of consecutive full-line comments in Python source."""
    rows = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and tok.line.strip() == tok.string.strip():
            rows.append(tok.start[0])
    return _runs(rows)


def config_comment_blocks(text: str) -> list[tuple[int, int]]:
    """``(first line, length)`` of each run of consecutive full-line ``#`` comments."""
    rows = [
        i
        for i, line in enumerate(text.splitlines(), 1)
        if line.lstrip().startswith("#") and not line.startswith("#!")
    ]
    return _runs(rows)


def _runs(rows: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for row in rows:
        if runs and row == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((row, 1))
    return runs


def _tracked_python_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.split()


def _long(blocks: list[tuple[int, int]], path: str) -> list[str]:
    return [
        f"{path}:{first} ({length} lines)"
        for first, length in blocks
        if length > MAX_BLOCK and (path, first) not in ALLOWED
    ]


def test_python_comment_blocks_are_short():
    offenders = []
    for path in _tracked_python_files():
        source = (ROOT / path).read_text(encoding="utf-8")
        offenders += _long(python_comment_blocks(source), path)
    assert not offenders, "comment blocks over the limit:\n" + "\n".join(offenders)


def test_config_comment_blocks_are_short():
    offenders = []
    for path in CONFIG_FILES:
        text = (ROOT / path).read_text(encoding="utf-8")
        offenders += _long(config_comment_blocks(text), path)
    assert not offenders, "comment blocks over the limit:\n" + "\n".join(offenders)


def test_every_listed_config_file_exists():
    assert [p for p in CONFIG_FILES if not (ROOT / p).is_file()] == []


class TestControls:
    def test_a_long_python_block_is_caught(self):
        source = "x = 1\n" + "# line\n" * (MAX_BLOCK + 1) + "y = 2\n"
        assert python_comment_blocks(source) == [(2, MAX_BLOCK + 1)]

    def test_a_long_config_block_is_caught(self):
        text = "[a]\n" + "# line\n" * (MAX_BLOCK + 1) + "b = 1\n"
        assert config_comment_blocks(text) == [(2, MAX_BLOCK + 1)]

    def test_hash_lines_inside_a_string_are_not_comments(self):
        source = 's = """\n' + "# not a comment\n" * (MAX_BLOCK + 1) + '"""\n'
        assert python_comment_blocks(source) == []

    def test_inline_comments_do_not_join_a_block(self):
        source = "# one\nx = 1  # inline\n# two\n"
        assert python_comment_blocks(source) == [(1, 1), (3, 1)]

    def test_a_blank_line_ends_a_block(self):
        assert config_comment_blocks("# a\n# b\n\n# c\n") == [(1, 2), (4, 1)]
