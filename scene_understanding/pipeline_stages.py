"""Named pipeline stage helpers (incremental migration out of process_image)."""

from __future__ import annotations

from pathlib import Path


def ensure_depth_dir(output_dir: Path) -> Path:
    d = output_dir / "depth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_scene_graph_dir(output_dir: Path) -> Path:
    d = output_dir / "scene_graph"
    d.mkdir(parents=True, exist_ok=True)
    return d
