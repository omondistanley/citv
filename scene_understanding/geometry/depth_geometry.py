"""Geometry helpers for depth-backed object measurements."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


class DepthGeometry:
    """Compute robust mask depth statistics and 3D anchor coordinates."""

    def __init__(
        self,
        back_project_fn: Callable[[int, int, float, Dict[str, float]], Dict[str, float]],
        mask_erosion_kernel_size: int = 5,
        depth_adaptive_erosion: bool = True,
        depth_outlier_sigma: float = 2.0,
        depth_transparency_check: bool = True,
        depth_transparency_threshold: float = 0.15,
        depth_central_fraction: float = 0.5,
        depth_sigma_clip_scope: str = "mask",
    ):
        self._back_project = back_project_fn
        self.mask_erosion_kernel_size = int(mask_erosion_kernel_size)
        self.depth_adaptive_erosion = bool(depth_adaptive_erosion)
        self.depth_outlier_sigma = float(depth_outlier_sigma)
        self.depth_transparency_check = bool(depth_transparency_check)
        self.depth_transparency_threshold = float(depth_transparency_threshold)
        self.depth_central_fraction = float(depth_central_fraction)
        self.depth_sigma_clip_scope = str(depth_sigma_clip_scope)

    def adaptive_erosion_kernel(self, mask_bin: np.ndarray) -> int:
        """Return erosion kernel size scaled to the mask's narrowest dimension."""
        if not self.depth_adaptive_erosion or self.mask_erosion_kernel_size == 0:
            return self.mask_erosion_kernel_size
        ys, xs = np.where(mask_bin)
        if ys.size == 0:
            return 0
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        min_dim = min(bbox_h, bbox_w)
        if min_dim < 15:
            return 0
        if min_dim < 40:
            return 1
        if min_dim < 80:
            return 2
        if min_dim < 150:
            return min(3, self.mask_erosion_kernel_size)
        return self.mask_erosion_kernel_size

    def mask_depth_stats_and_3d(
        self,
        metric_depth: np.ndarray,
        intrinsics: Dict[str, float],
        mask: np.ndarray,
        detection: Optional[Dict] = None,
        use_erosion: bool = True,
        region_context: Optional[Dict] = None,
        label_map: Optional[np.ndarray] = None,
        region_index: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, float], List[int]]:
        """Compute depth stats, 3D anchor coordinates, and 2D centroid for a mask."""
        del detection
        height, width = metric_depth.shape[:2]
        mask_bin = np.asarray(mask) > 0
        if mask_bin.shape[:2] != (height, width):
            mask_bin = cv2.resize(
                mask_bin.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        if use_erosion:
            kernel_size = self.adaptive_erosion_kernel(mask_bin)
            if region_context:
                rtype = str(region_context.get("type", "")).lower()
                if rtype == "background":
                    kernel_size = 0
            if kernel_size > 0 and int(mask_bin.sum()) > 4 * kernel_size * kernel_size:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded = cv2.erode(mask_bin.astype(np.uint8), kernel, iterations=1)
                if eroded.sum() > 0:
                    mask_bin = eroded > 0

        ys, xs = np.where(mask_bin)
        depth_at_mask = metric_depth[ys, xs]
        finite_mask = np.isfinite(depth_at_mask)
        depth_at_mask = depth_at_mask[finite_mask]
        ys_f = ys[finite_mask]
        xs_f = xs[finite_mask]

        if depth_at_mask.size == 0:
            depth_stats = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "num_pixels": 0,
                "z_val": 0.0,
                "z_val_pixels": 0,
                "possibly_transparent": False,
                "depth_separation_from_background": 0.0,
            }
            coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
            centroid = [width // 2, height // 2]
            return depth_stats, coords_3d, centroid

        sigma = self.depth_outlier_sigma
        sigma_scope = str((region_context or {}).get("sigma_scope") or self.depth_sigma_clip_scope)
        if sigma > 0 and depth_at_mask.size >= 10:
            mean_depth = float(np.mean(depth_at_mask))
            std_depth = float(np.std(depth_at_mask))
            if sigma_scope == "region" and region_context:
                rs = region_context.get("depth_stats") or {}
                if rs.get("std") is not None and float(rs["std"]) > 1e-6:
                    mean_depth = float(rs.get("mean", mean_depth))
                    std_depth = float(rs["std"])
            if std_depth > 1e-6:
                inlier = np.abs(depth_at_mask - mean_depth) < sigma * std_depth
                if inlier.sum() >= 5:
                    depth_at_mask = depth_at_mask[inlier]
                    ys_f = ys_f[inlier]
                    xs_f = xs_f[inlier]

        possibly_transparent = False
        depth_separation = 0.0
        if self.depth_transparency_check and mask_bin.sum() > 0:
            try:
                kernel_5 = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(mask_bin.astype(np.uint8), kernel_5) > 0
                border_ring = dilated & ~mask_bin
                if (
                    label_map is not None
                    and int(region_index) > 0
                    and np.asarray(label_map).shape[:2] == (height, width)
                ):
                    lm = np.asarray(label_map, dtype=np.int32)
                    ring_same = border_ring & (lm == int(region_index))
                    if int(np.count_nonzero(ring_same)) >= 8:
                        border_ring = ring_same
                border_depths = metric_depth[border_ring]
                border_depths = border_depths[np.isfinite(border_depths)]
                if border_depths.size > 0 and depth_at_mask.size > 0:
                    mask_mean = float(np.mean(depth_at_mask))
                    border_mean = float(np.mean(border_depths))
                    depth_separation = abs(mask_mean - border_mean)
                    possibly_transparent = (
                        depth_separation < self.depth_transparency_threshold
                    )
            except Exception:
                pass

        weights = 1.0 / (depth_at_mask + 1e-6)
        weight_sum = float(weights.sum())
        cy_f = float(np.sum(ys_f * weights) / weight_sum)
        cx_f = float(np.sum(xs_f * weights) / weight_sum)
        dist2 = (ys_f - cy_f) ** 2 + (xs_f - cx_f) ** 2
        anchor_idx = int(np.argmin(dist2))
        cx = int(xs_f[anchor_idx])
        cy = int(ys_f[anchor_idx])

        if self.depth_central_fraction < 1.0:
            area = float(mask_bin.sum())
            radius = np.sqrt(area * self.depth_central_fraction / np.pi)
            inner_mask = dist2 <= radius**2
            inner_depths = depth_at_mask[inner_mask]
        else:
            inner_depths = depth_at_mask

        z_val_pixels = int(inner_depths.size)
        if z_val_pixels > 0:
            n_bins = max(10, min(100, z_val_pixels // 5))
            hist, edges = np.histogram(inner_depths, bins=n_bins)
            peak_bin = int(np.argmax(hist))
            z_val = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
        else:
            z_val = float(np.median(depth_at_mask))
            z_val_pixels = int(depth_at_mask.size)

        depth_stats = {
            "min": round(float(np.min(depth_at_mask)), 4),
            "max": round(float(np.max(depth_at_mask)), 4),
            "mean": round(float(np.mean(depth_at_mask)), 4),
            "median": round(float(np.median(depth_at_mask)), 4),
            "std": round(float(np.std(depth_at_mask)), 4),
            "num_pixels": int(mask_bin.sum()),
            "z_val": round(z_val, 4),
            "z_val_pixels": z_val_pixels,
            "possibly_transparent": bool(possibly_transparent),
            "depth_separation_from_background": round(depth_separation, 4),
        }
        coords_3d = self._back_project(cx, cy, z_val, intrinsics)
        return depth_stats, coords_3d, [cx, cy]
