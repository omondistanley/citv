"""Depth pipeline helpers built around the existing metric depth estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class DepthCoordinator:
    """Handle depth inference, reuse, resizing, and persistence."""

    def __init__(self, depth_estimator, depth_scale_factor: float = 10.0):
        self.depth_estimator = depth_estimator
        self.depth_scale_factor = float(depth_scale_factor)

    @staticmethod
    def metric_depth_npy_path(output_dir: Path, image_stem: str) -> Path:
        """Canonical ``depth/{stem}_depth_metric.npy`` path under a run output directory."""
        return (output_dir / "depth").resolve() / f"{image_stem}_depth_metric.npy"

    def load_or_infer_depth(
        self,
        image_rgb: np.ndarray,
        output_dir: Path,
        image_stem: str,
        reuse_existing: bool = False,
    ) -> np.ndarray:
        """Load a saved metric depth map when requested, otherwise infer and save one."""
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_npy = output_dir / f"{image_stem}_depth_metric.npy"
        height, width = image_rgb.shape[:2]

        if reuse_existing and depth_npy.exists():
            metric_depth = np.load(str(depth_npy)).astype(np.float32)
            if metric_depth.shape[:2] != (height, width):
                metric_depth = cv2.resize(
                    metric_depth,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            return metric_depth

        backend = getattr(self.depth_estimator, "backend", None)
        if backend is None:
            raise RuntimeError(
                "Depth backend is unavailable and no reusable saved depth map was found."
            )

        raw_depth = backend.infer(image_rgb)
        depth_full = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        metric_depth = depth_full.astype(np.float32) * self.depth_scale_factor
        np.save(depth_npy, metric_depth)
        return metric_depth
