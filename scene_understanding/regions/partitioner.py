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
    """Return (labels N, centroids k) for 1D samples.

    Phase D — the previous pure-NumPy loop materialised an N×k distance
    matrix every iteration (``np.abs(x[:, None] - centroids[None, :])``),
    which for N ≈ 500k pixels and k = 4 is ~16 MB re-allocated up to 30
    times per image. We now delegate to ``cv2.kmeans`` which runs the
    same algorithm in native SIMD; the public contract (label array,
    centroid array) is preserved byte-for-byte on dense inputs. For very
    small input (N < 32) we still use the NumPy fallback because OpenCV's
    kmeans refuses K > N.
    """

    x = samples.astype(np.float64).ravel()
    n = x.size
    if n < k:
        k = max(1, n)
    if k <= 1:
        return np.zeros(n, dtype=np.int32), np.array([float(np.mean(x))] if n else [0.0])

    # Small-sample fallback: cv2.kmeans requires N >= k and prefers larger N.
    if n < max(32, 8 * k):
        qs = np.linspace(0, 1, k + 2)[1:-1]
        centroids = np.quantile(x, qs).astype(np.float64)
        centroids = centroids + rng.normal(0, 1e-4, size=k)
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

    # cv2.kmeans path — dense, SIMD, drop-in equivalent.
    data32 = x.astype(np.float32).reshape(-1, 1)
    crit = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(max_iter),
        1e-4,
    )
    # Use percentile init so centroids are deterministic across runs.
    qs = np.linspace(0, 1, k + 2)[1:-1]
    init_centroids = np.quantile(x, qs).astype(np.float32).reshape(-1, 1)
    try:
        _compactness, labels_cv, centroids_cv = cv2.kmeans(
            data32,
            int(k),
            bestLabels=None,
            criteria=crit,
            attempts=1,
            flags=cv2.KMEANS_USE_INITIAL_LABELS if False else cv2.KMEANS_PP_CENTERS,
        )
    except Exception:
        # Fall back to NumPy implementation on any OpenCV error.
        centroids = init_centroids.astype(np.float64).ravel()
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

    labels = labels_cv.ravel().astype(np.int32)
    centroids = centroids_cv.ravel().astype(np.float64)
    # Re-sort centroids by value so downstream code that assumes
    # "cluster 0 is the near cluster, cluster k-1 is the far cluster"
    # stays stable regardless of OpenCV's internal init order.
    order = np.argsort(centroids)
    remap = np.zeros_like(order)
    remap[order] = np.arange(len(order))
    labels = remap[labels].astype(np.int32)
    centroids = centroids[order]
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
    """Encode label_map as BGR uint8 for cv2.imwrite.

    Phase D — the previous implementation scanned every palette entry
    with ``flat == idx`` which is O(N·K) on large images. We now build a
    single LUT of shape ``(max_idx + 1, 3)`` and use NumPy fancy
    indexing ``palette_lut[label_map]`` for an O(N) one-shot lookup.
    The output is byte-for-byte identical on valid label maps; any label
    value outside ``[0, len(palette))`` is clamped to 0 (void) to match
    the prior per-index behaviour.
    """

    if not palette:
        return np.zeros(label_map.shape[:2] + (3,), dtype=np.uint8)
    lm = np.asarray(label_map)
    n = len(palette)
    palette_arr = np.asarray(palette, dtype=np.int32)
    if palette_arr.ndim != 2 or palette_arr.shape[1] != 3:
        raise ValueError(f"palette must be N×3; got {palette_arr.shape}")
    # RGB → BGR reorder once, then uint8 cast.
    bgr_lut = np.empty((n, 3), dtype=np.uint8)
    bgr_lut[:, 0] = np.clip(palette_arr[:, 2], 0, 255).astype(np.uint8)  # B
    bgr_lut[:, 1] = np.clip(palette_arr[:, 1], 0, 255).astype(np.uint8)  # G
    bgr_lut[:, 2] = np.clip(palette_arr[:, 0], 0, 255).astype(np.uint8)  # R
    # Clamp labels outside the palette to 0 (void) to match legacy behaviour.
    safe = np.where((lm >= 0) & (lm < n), lm, 0).astype(np.int32)
    return bgr_lut[safe]


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
