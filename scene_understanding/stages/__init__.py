"""Pipeline stage modules (incremental extraction from scene_understanding.py)."""

from __future__ import annotations

from . import (
    captions_export,
    depth,
    depth_mask_fusion,
    full_run,
    labelling,
    paths_export,
    preprocess,
    regions,
    relations_stage,
    scene_write,
    segmentation,
)
from .legacy_delegate import run_legacy_process_image

__all__ = [
    "run_legacy_process_image",
    "captions_export",
    "depth",
    "depth_mask_fusion",
    "full_run",
    "labelling",
    "paths_export",
    "preprocess",
    "regions",
    "relations_stage",
    "scene_write",
    "segmentation",
]
