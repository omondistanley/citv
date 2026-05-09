"""Unified export surface for path / traversability / animation helpers.

* ``build_traversability_speed_map`` and ``k_diverse_grid_paths`` are
  imported from the authoritative internal module
  ``scene_understanding.pathing.traversability``.

* The composite-image and animation-bundle helpers live in root-level
  ``citv/scene_path_*.py`` scripts.  We load them via importlib so the
  import succeeds regardless of the caller's working directory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..pathing.traversability import build_traversability_speed_map, is_soft_obstacle, k_diverse_grid_paths

_CITV_ROOT = Path(__file__).resolve().parents[2]


def _load_root_module(filename: str, name: str) -> Any:
    """Load a root-level script by absolute path and return the module."""
    p = _CITV_ROOT / filename
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _lazy(attr: str, _module_filename: str, _module_name: str):  # type: ignore[override]
    """Return a callable that defers the root-module load until first call."""
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        mod = _load_root_module(_module_filename, _module_name)
        return getattr(mod, attr)(*args, **kwargs)
    _wrapper.__name__ = attr
    return _wrapper


build_path_fields_explainer_image = _lazy(
    "build_path_fields_explainer_image",
    "scene_path_field_explainer.py",
    "_citv_scene_path_field_explainer",
)
build_path_fields_legend_payload = _lazy(
    "build_path_fields_legend_payload",
    "scene_path_field_explainer.py",
    "_citv_scene_path_field_explainer",
)
build_animation_components_bundle = _lazy(
    "build_animation_components_bundle",
    "scene_path_motion_contracts.py",
    "_citv_scene_path_motion_contracts",
)
build_insertion_bundle = _lazy(
    "build_insertion_bundle",
    "scene_path_motion_contracts.py",
    "_citv_scene_path_motion_contracts",
)
build_trajectory_bundle = _lazy(
    "build_trajectory_bundle",
    "scene_path_motion_contracts.py",
    "_citv_scene_path_motion_contracts",
)

__all__ = [
    "build_insertion_bundle",
    "build_trajectory_bundle",
    "build_animation_components_bundle",
    "build_path_fields_explainer_image",
    "build_path_fields_legend_payload",
    "build_traversability_speed_map",
    "is_soft_obstacle",
    "k_diverse_grid_paths",
]
