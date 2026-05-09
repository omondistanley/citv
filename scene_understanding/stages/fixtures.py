"""GPU-free test fixtures for pipeline stages (synthetic images, depth, detections)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def write_tiny_bgr_image(path: Path, size: Tuple[int, int] = (16, 16)) -> None:
    """Write a minimal valid BGR image for preprocess / I/O tests."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def flat_metric_depth(h: int, w: int, value: float = 2.5) -> np.ndarray:
    """Constant metric depth map (float32)."""
    return np.full((h, w), value, dtype=np.float32)


def fake_detection_mask(
    h: int,
    w: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    label: str = "object",
    source_model: str = "GroundedSAM2",
) -> Dict[str, Any]:
    """Single detection dict compatible with label / geometry stages."""
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[y0:y1, x0:x1] = 1
    return {
        "segmentation": seg,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "label": label,
        "gdino_conf": 0.9,
        "predicted_iou": 0.85,
        "source_model": source_model,
    }


def fake_detections_list(h: int, w: int) -> List[Dict[str, Any]]:
    """Two non-overlapping box masks for relation / count tests."""
    return [
        fake_detection_mask(h, w, 2, 8, 2, 8, label="box_a"),
        fake_detection_mask(h, w, 9, 14, 9, 14, label="box_b"),
    ]
