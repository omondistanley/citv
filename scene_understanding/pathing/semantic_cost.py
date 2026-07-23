"""Precision-weighted cost-layer fusion for the FMM planner.

Each layer is a (cost, precision) pair over the image, both in [0, 1].
``precision`` says how much a layer's own cost estimate should be trusted at
each pixel -- a layer that has no evidence at a pixel should say so (low
precision) rather than silently voting for a default/free-to-traverse cost.
The final cost field is a precision-weighted average (``precision_weighted_fuse``),
not a flat sum of weighted layers, so a well-evidenced layer (e.g. an actual
object mask) dominates a low-evidence one (e.g. a bare depth gradient with no
scene-graph support) wherever they disagree.

The "scene graph" layer reuses the same per-object affordance rasters built
in ``affordance_rasters.py`` (mask-painted, open-vocabulary) rather than
re-deriving obstacle/support evidence from scratch -- ``blocker`` and
``support_surface`` are exactly the obstacle/support signal this layer needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CostLayer:
    name: str
    cost: np.ndarray       # (H, W) float32 in [0, 1]
    precision: np.ndarray  # (H, W) float32 in [0, 1]
    provenance: str = ""


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def geometry_cost_layer(img_bgr: np.ndarray, obstacle_mask: np.ndarray) -> CostLayer:
    """Image-derived layer: edges + depth-free obstacle distance-transform."""
    gray = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_cost = _minmax(cv2.magnitude(gx, gy))

    obs = np.asarray(obstacle_mask, dtype=np.uint8)
    free = (1 - obs).astype(np.uint8) * 255
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 3).astype(np.float32)
    dt_cost = 1.0 - _minmax(dt)

    depth_cost = np.zeros_like(edge_cost)  # filled by depth-aware caller if available
    cost = (edge_cost + depth_cost + dt_cost) / 3.0
    precision = cv2.GaussianBlur(np.maximum(edge_cost, dt_cost), (0, 0), 1.5)
    return CostLayer("geometry", np.clip(cost, 0, 1), np.clip(precision, 0, 1), "image_edges+free_space_dt")


def geometry_cost_layer_with_depth(img_bgr: np.ndarray, obstacle_mask: np.ndarray, metric_depth_m: np.ndarray) -> CostLayer:
    base = geometry_cost_layer(img_bgr, obstacle_mask)
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    finite = np.isfinite(dm)
    dm_f = np.where(finite, dm, 0.0)
    gx = cv2.Sobel(dm_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dm_f, cv2.CV_32F, 0, 1, ksize=3)
    depth_cost = _minmax(cv2.magnitude(gx, gy))
    depth_cost = np.where(finite, depth_cost, 0.5)
    # base.cost already averages (edge_cost + 0 + dt_cost) / 3; substitute the
    # real depth_cost for that placeholder zero to get the true 3-way average.
    cost = (base.cost * 3.0 + depth_cost) / 3.0
    precision = cv2.GaussianBlur(np.maximum(base.precision, depth_cost), (0, 0), 1.5)
    return CostLayer("geometry", np.clip(cost, 0, 1), np.clip(precision, 0, 1), "image_edges+depth_gradient+free_space_dt")


def scene_graph_cost_layer(affordance_rasters: Dict[str, np.ndarray]) -> CostLayer:
    """Obstacle-vs-support evidence, reusing the already mask-painted rasters."""
    blocker = affordance_rasters.get("blocker")
    support = affordance_rasters.get("support_surface")
    shape = blocker.shape if blocker is not None else (support.shape if support is not None else (1, 1))
    blocker = blocker if blocker is not None else np.zeros(shape, dtype=np.float32)
    support = support if support is not None else np.zeros(shape, dtype=np.float32)
    cost = np.clip(_minmax(blocker) - _minmax(support), 0.0, 1.0)
    # Precision: pixels where either channel actually carries evidence.
    contributing = np.maximum(blocker > 0, support > 0).astype(np.float32)
    precision = cv2.GaussianBlur(contributing, (0, 0), 1.0)
    uncertainty = affordance_rasters.get("uncertainty")
    if uncertainty is not None:
        precision = precision * (1.0 - 0.5 * np.clip(uncertainty, 0, 1))
    return CostLayer("scene_graph", cost, np.clip(precision, 0, 1), "object_affordance_rasters")


def agent_physics_cost_layer(
    obstacle_mask: np.ndarray,
    metric_depth_m: Optional[np.ndarray],
    intrinsics: Dict[str, float],
    agent_half_width_m: float = 0.25,
    foot_step_tolerance_m: float = 0.12,
) -> CostLayer:
    """Inflate obstacles by the agent's own footprint radius (in pixels, per-pixel
    since a fixed real-world radius covers fewer pixels far away than close up),
    plus a step-over-height gate from local depth discontinuities."""
    obs = np.asarray(obstacle_mask, dtype=np.uint8)
    h, w = obs.shape[:2]
    dt_free = cv2.distanceTransform((1 - obs).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    if metric_depth_m is not None:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        finite = np.isfinite(dm) & (dm > 1e-6)
        fx = float(intrinsics.get("fx", w))
        fy = float(intrinsics.get("fy", h))
        px_x = np.where(finite, fx * agent_half_width_m / np.maximum(dm, 1e-3), 0.0)
        px_y = np.where(finite, fy * agent_half_width_m / np.maximum(dm, 1e-3), 0.0)
        radius = (px_x + px_y) / 2.0
        med = float(np.median(radius[finite])) if finite.any() else 8.0
        radius = np.where(finite, radius, med)
    else:
        radius = np.full((h, w), 8.0, dtype=np.float32)
    radius = np.clip(radius, 1.0, max(2.0, min(h, w) * 0.25))

    inside = (dt_free <= radius).astype(np.float32)
    soft = np.clip(1.0 - (dt_free - radius) / np.maximum(radius, 1e-3), 0.0, 1.0)
    cost = np.maximum(inside, soft)

    if metric_depth_m is not None:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        finite = np.isfinite(dm)
        kernel = np.ones((3, 3), np.uint8)
        dm_free = np.where(finite, dm, 0.0).astype(np.float32)
        dil = cv2.dilate(dm_free, kernel)
        ero = cv2.erode(dm_free, kernel)
        step = np.abs(dil - ero)
        step_cost = np.clip(step / max(foot_step_tolerance_m, 1e-3), 0.0, 1.0)
        step_cost = np.where(finite, step_cost, 0.0)
        cost = np.maximum(cost, step_cost)
        precision = np.where(finite, np.maximum(inside, dt_free < radius * 2.0), 0.4).astype(np.float32)
    else:
        precision = np.clip(dt_free < radius * 2.0, 0.3, 1.0).astype(np.float32)

    return CostLayer("agent_physics", np.clip(cost, 0, 1), np.clip(precision, 0, 1), "footprint_radius+step_gate")


def depth_noise_precision_layer(metric_depth_m: Optional[np.ndarray]) -> CostLayer:
    """Precision-only layer: pixels with high local depth variance are untrustworthy."""
    if metric_depth_m is None:
        return CostLayer("depth_noise", np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), "no_depth")
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    finite = np.isfinite(dm)
    dm_f = np.where(finite, dm, 0.0)
    mean = cv2.boxFilter(dm_f, ddepth=-1, ksize=(5, 5))
    mean_sq = cv2.boxFilter(dm_f * dm_f, ddepth=-1, ksize=(5, 5))
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std01 = _minmax(np.sqrt(var))
    split = _jsd_break(std01[finite]) if finite.any() else 0.5
    precision = np.where(finite, np.clip(1.0 - std01, 0.0, 1.0), 0.0)
    precision = np.where(std01 <= split, precision, precision * 0.3)
    cost = np.zeros_like(std01)
    return CostLayer("depth_noise", cost, precision.astype(np.float32), "local_depth_variance")


def _jsd_break(values: np.ndarray, bins: int = 64) -> float:
    """Fractional threshold in [0,1] that maximizes Jensen-Shannon divergence
    between the histogram halves it splits."""
    if values.size == 0:
        return 0.5
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64) + 1e-9
    p = hist / hist.sum()
    best_split, best_jsd = 0.5, -1.0
    for i in range(1, bins - 1):
        left, right = p[:i], p[i:]
        pl, pr = left / left.sum(), right / right.sum()
        pl_full = np.zeros_like(p); pl_full[:i] = pl
        pr_full = np.zeros_like(p); pr_full[i:] = pr
        m = 0.5 * (pl_full + pr_full)

        def _kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

        jsd = 0.5 * _kl(pl_full, m) + 0.5 * _kl(pr_full, m)
        if jsd > best_jsd:
            best_jsd, best_split = jsd, edges[i]
    return float(best_split)


def precision_weighted_fuse(layers: Sequence[CostLayer]) -> Tuple[np.ndarray, np.ndarray]:
    """``C = sum(P_i * C_i) / sum(P_i)``, ``P = 1 - prod(1 - P_i)``.

    Pixels with zero precision everywhere get cost 0 (info-less, treated as
    "unknown", not as "definitely free" -- callers should still gate on the
    returned precision field before trusting cost==0 there).
    """
    shape = layers[0].cost.shape
    for layer in layers:
        if layer.cost.shape != shape:
            raise ValueError(f"cost layer '{layer.name}' shape {layer.cost.shape} != {shape}")

    weighted_sum = np.zeros(shape, dtype=np.float64)
    precision_sum = np.zeros(shape, dtype=np.float64)
    complement_product = np.ones(shape, dtype=np.float64)
    for layer in layers:
        p = layer.precision.astype(np.float64)
        weighted_sum += p * layer.cost.astype(np.float64)
        precision_sum += p
        complement_product *= (1.0 - p)

    fused_precision = 1.0 - complement_product
    has_evidence = precision_sum > 1e-9
    fused_cost = np.zeros(shape, dtype=np.float64)
    fused_cost[has_evidence] = weighted_sum[has_evidence] / precision_sum[has_evidence]
    return fused_cost.astype(np.float32), fused_precision.astype(np.float32)
