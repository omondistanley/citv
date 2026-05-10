"""
Backward-compatible import surface for traversability helpers.

Loads ``scene_understanding/pathing/traversability.py`` via importlib so imports do not
execute ``scene_understanding/__init__.py`` (avoids circular imports with the legacy
``scene_understanding.py`` monolith).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_trav_path = Path(__file__).resolve().parent / "scene_understanding" / "pathing" / "traversability.py"
_spec = importlib.util.spec_from_file_location("_citv_pathing_traversability", _trav_path)
_trav_mod = importlib.util.module_from_spec(_spec) if _spec and _spec.loader else None
if _trav_mod is not None and _spec is not None and _spec.loader is not None:
    sys.modules[_spec.name] = _trav_mod
    _spec.loader.exec_module(_trav_mod)
    build_traversability_speed_map = _trav_mod.build_traversability_speed_map
    grid_dijkstra_path = _trav_mod.grid_dijkstra_path
    k_diverse_grid_paths = _trav_mod.k_diverse_grid_paths
    heading_from_depth_at = _trav_mod.heading_from_depth_at
else:  # pragma: no cover
    build_traversability_speed_map = None  # type: ignore[misc, assignment]
    grid_dijkstra_path = None  # type: ignore[misc, assignment]
    k_diverse_grid_paths = None  # type: ignore[misc, assignment]
    heading_from_depth_at = None  # type: ignore[misc, assignment]

__all__ = [
    "build_traversability_speed_map",
    "grid_dijkstra_path",
    "heading_from_depth_at",
    "k_diverse_grid_paths",
]
