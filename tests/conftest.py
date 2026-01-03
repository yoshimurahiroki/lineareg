from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the repository root is on sys.path.

    This repo contains a nested `lineareg/pyproject.toml`, so pytest may choose
    `.../econ-project/lineareg` as its rootdir. In that case, importing the
    top-level package `lineareg` fails unless the parent directory is on
    `sys.path`.
    """

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
