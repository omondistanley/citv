"""Stable relative filenames for path hypothesis exports."""
from __future__ import annotations

from pathlib import Path

PATH_HYPOTHESES_JSON_NAME = "path_hypotheses.json"


def path_hypotheses_json_path(paths_root_dir: Path) -> Path:
    return paths_root_dir / PATH_HYPOTHESES_JSON_NAME
