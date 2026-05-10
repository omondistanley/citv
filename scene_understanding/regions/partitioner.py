"""Partition image into depth-coherent regions (K-means + connected components)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class RegionPartitionResult:
    """Dense region labels and per-region metadata."""

    label_map: np.ndarray  # HxW int32, 0 = void
    regions: List[Dict[str, Any]]  # one entry per region_index >= 1
    palette: List[List[int]]  # RGB (0..255) per index; palette[0] = void
    depth_model_id: str = ""


def _kmeans_1d(
    samples: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels N, centroids k) for 1D samples."""
    x = samples.astype(np.float64).ravel()
    n = x.size
    if n < k:
        k = max(1, n)
    if k <= 1:
        return np.zeros(n, dtype=np.int32), np.array([float(np.mean(x))] if n else [0.0])

    # Init centroids via percentiles + tiny jitter for stability
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centroids = np.quantile(x, qs).astype(np.float64)
    noise = rng.normal(0, 1e-4, size=k)
    centroids = centroids + noise

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist = np.abs(x[:, None] - centroids[None, :])
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask_j = labels == j
            if mask_j.any():
                centroids[j] = float(np.mean(x[mask_j]))
            else:
                centroids[j] = float(np.median(x))

    return labels, centroids


def _make_palette(n_regions: int) -> List[List[int]]:
    """Distinct RGB colours; index 0 is void (black)."""
    palette: List[List[int]] = [[0, 0, 0]]
    golden = 0.618033988749895
    h = 0.27
    for i in range(1, n_regions + 1):
        h = (h + golden) % 1.0
        s = 0.55 + 0.15 * ((i * 7) % 3)
        v = 0.75 + 0.2 * ((i * 11) % 2)
        # HSV to RGB
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s), min(1.0, v))
        palette.append([int(r * 255), int(g * 255), int(b * 255)])
    return palette


def label_map_to_bgr(label_map: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    """Encode label_map as BGR uint8 for cv2.imwrite."""
    h, w = label_map.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    flat = label_map.ravel()
    for idx in range(len(palette)):
        m = flat == idx
        if not m.any():
            continue
        rgb = palette[idx]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        tmp = out.reshape(-1, 3)
        tmp[m] = bgr
    return out


def partition_depth_regions(
    metric_depth: np.ndarray,
    k: int = 4,
    min_region_px: int = 500,
    blur_sigma: float = 0.0,
    seed: int = 42,
    depth_model_id: str = "unknown",
) -> RegionPartitionResult:
    """
    Cluster valid depth pixels with 1D K-means, then split each cluster by connected components.
    Small components are merged to void (0).
    """
    depth = np.asarray(metric_depth, dtype=np.float32)
    h, w = depth.shape[:2]
    valid = np.isfinite(depth) & (depth > 1e-6)
    if blur_sigma and blur_sigma > 0:
        d_work = depth.copy()
        d_work[~valid] = 0
        blurred = cv2.GaussianBlur(d_work, (0, 0), float(blur_sigma))
        depth_use = np.where(valid, blurred, depth)
    else:
        depth_use = depth

    rng = np.random.default_rng(int(seed))
    cluster_map = np.zeros((h, w), dtype=np.int32)  # 0..k-1 cluster id per pixel; 0 also used for invalid

    if valid.sum() < max(k * 10, 50):
        return RegionPartitionResult(
            label_map=np.zeros((h, w), dtype=np.int32),
            regions=[],
            palette=[[0, 0, 0]],
            depth_model_id=depth_model_id,
        )

    vals = depth_use[valid]
    labels_flat, _centroids = _kmeans_1d(vals, k, rng)
    cluster_map_flat = np.zeros(h * w, dtype=np.int32)
    cluster_map_flat[valid.ravel()] = labels_flat + 1  # 1..k for valid (cluster id)

    cluster_2d = cluster_map_flat.reshape(h, w)
    # cluster_2d is 0 invalid, 1..k for valid pixels' cluster

    next_region = 1
    label_map = np.zeros((h, w), dtype=np.int32)
    regions_meta: List[Dict[str, Any]] = []

    global_valid_depths = depth[valid]
    q1, q2 = np.quantile(global_valid_depths, [1.0 / 3.0, 2.0 / 3.0]) if global_valid_depths.size else (0.0, 0.0)

    for c in range(1, k + 1):
        bin_mask = (cluster_2d == c).astype(np.uint8)
        if bin_mask.sum() == 0:
            continue
        num_cc, cc_labels = cv2.connectedComponents(bin_mask)
        for comp in range(1, num_cc):
            comp_mask = (cc_labels == comp) & (cluster_2d == c)
            area = int(comp_mask.sum())
            if area < min_region_px:
                continue
            rid = next_region
            next_region += 1
            label_map[comp_mask] = rid
            zs = depth[comp_mask]
            zs = zs[np.isfinite(zs)]
            if zs.size == 0:
                continue
            ys, xs = np.where(comp_mask)
            mean_z = float(np.mean(zs))
            if mean_z <= q1:
                rtype = "foreground"
            elif mean_z <= q2:
                rtype = "midground"
            else:
                rtype = "background"

            regions_meta.append(
                {
                    "region_index": rid,
                    "id": f"region_{rid}",
                    "type": rtype,
                    "depth_cluster": c - 1,
                    "depth_band_m": [round(float(np.min(zs)), 4), round(float(np.max(zs)), 4)],
                    "bbox_px": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ],
                    "area_px": area,
                    "centroid_2d_px": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
                    "depth_stats": {
                        "min": round(float(np.min(zs)), 4),
                        "max": round(float(np.max(zs)), 4),
                        "mean": round(float(np.mean(zs)), 4),
                        "std": round(float(np.std(zs)), 4) if zs.size > 1 else 0.0,
                        "mode": round(float(np.median(zs)), 4),
                    },
                    "object_ids": [],
                }
            )

    n_reg = len(regions_meta)
    max_idx = int(label_map.max()) if n_reg else 0
    palette = _make_palette(max_idx) if max_idx > 0 else [[0, 0, 0]]
    pal = palette
    return RegionPartitionResult(
        label_map=label_map,
        regions=regions_meta,
        palette=pal,
        depth_model_id=depth_model_id,
    )


def majority_region_index(mask_bin: np.ndarray, label_map: np.ndarray) -> int:
    """Most common positive region id under mask; centroid fallback."""
    mb = np.asarray(mask_bin, dtype=bool)
    lm = np.asarray(label_map, dtype=np.int32)
    if mb.shape != lm.shape:
        mb = cv2.resize(mb.astype(np.uint8), (lm.shape[1], lm.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    vals = lm[mb]
    vals = vals[vals > 0]
    if vals.size > 0:
        return int(np.bincount(vals.astype(np.int64)).argmax())
    ys, xs = np.where(mb)
    if ys.size == 0:
        return 0
    cy = int(np.clip(np.mean(ys), 0, lm.shape[0] - 1))
    cx = int(np.clip(np.mean(xs), 0, lm.shape[1] - 1))
    return int(lm[cy, cx])
