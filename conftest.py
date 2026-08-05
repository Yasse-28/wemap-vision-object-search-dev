"""Fail with an explanation when the pipeline submodule is missing.

`toolbox/bricks/` imports `prepare`, `indexing` and `inference` — top-level names
that only resolve because `pyproject.toml` puts `third_party/object_search` (the
wemap-vision-object-search submodule) on pytest's `pythonpath`. A fresh clone
without `--recurse-submodules` gets a bare `ModuleNotFoundError: prepare` from three
unrelated test files, which says nothing about the actual cause.

The shell-side equivalent of this guard is in `scripts/lib.sh`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SUBMODULE = Path(__file__).parent / "third_party" / "object_search"


def pytest_configure(config: pytest.Config) -> None:
    """Abort collection if the submodule is not checked out."""
    if not (SUBMODULE / "prepare").is_dir():
        raise pytest.UsageError(
            f"The pipeline submodule is not checked out at {SUBMODULE}.\n"
            "toolbox/bricks imports `prepare`, `inference` and `indexing` from it.\n"
            "Run: git submodule update --init"
        )
