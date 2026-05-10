"""Shared context for pipeline stages (incremental extraction from process_image)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PipelineContext:
    """Per-image immutable-ish state passed between stages."""

    image_path: Path
    output_dir: Path
    stem: str
    timestamp: str
    img_bgr: np.ndarray
    img_rgb: np.ndarray
    height: int
    width: int
    intrinsics: Dict[str, float]
    metric_depth: Optional[np.ndarray] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    region_label_map: Optional[np.ndarray] = None
    region_partition_meta: List[Dict[str, Any]] = field(default_factory=list)
    # Legacy-aligned fields (filled as staged pipeline reaches parity with process_image).
    objects_3d: List[Dict[str, Any]] = field(default_factory=list)
    sam2_metadata: Optional[Dict[str, Any]] = None
    regions_block: Optional[Dict[str, Any]] = None
    path_exports: Dict[str, Any] = field(default_factory=dict)
    mask_hierarchy: Optional[Dict[str, Any]] = None
    layers: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
