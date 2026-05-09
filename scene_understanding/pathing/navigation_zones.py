"""Soft navigation-zone maps (idle / walk / run / careful) from pipeline rasters and semantic_layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_DEFAULT_POLICY: Dict[str, Any] = {
    "channel_order": ["idle", "walk", "run", "careful"],
    "affordance_channel_prior": {
        "walkable": {"idle": 0.08, "walk": 0.42, "run": 0.45, "careful": 0.05},
        "interaction_zone": {"idle": 0.18, "walk": 0.48, "run": 0.22, "careful": 0.12},
        "obstacle": {"idle": 0.05, "walk": 0.15, "run": 0.0, "careful": 0.8},
        "far_background": {"idle": 0.02, "walk": 0.08, "run": 0.0, "careful": 0.9},
    },
    "default": {"idle": 0.15, "walk": 0.5, "run": 0.25, "careful": 0.1},
}


def load_navigation_zone_policy(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[2]
    p = root / "docs" / "navigation_zone_policy.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(_DEFAULT_POLICY)


def _affordance_prior_raster(
    lm: np.ndarray,
    regions_meta: List[Dict[str, Any]],
    region_affordances: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> np.ndarray:
    """Per-pixel 4-vector prior from region affordances (ids from pipeline, not prose)."""
    h, w = lm.shape[:2]
    pri = np.zeros((h, w, 4), dtype=np.float32)
    aff_map: Dict[str, Any] = dict(policy.get("affordance_channel_prior") or {})
    default = dict(policy.get("default") or _DEFAULT_POLICY["default"])
    rid_to_aff: Dict[str, str] = {}
    for a in region_affordances or []:
        if not isinstance(a, dict):
            continue
        rid = str(a.get("region_id", "") or "").strip()
        if rid:
            rid_to_aff[rid] = str(a.get("affordance", "interaction_zone") or "interaction_zone")
    index_to_rid: Dict[int, str] = {}
    for r in regions_meta or []:
        try:
            idx = int(r.get("region_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        rid = str(r.get("id", "") or "").strip()
        if idx > 0 and rid:
            index_to_rid[idx] = rid
    ch_names = ["idle", "walk", "run", "careful"]
    for li in np.unique(lm):
        lii = int(li)
        if lii <= 0:
            continue
        rid = index_to_rid.get(lii, "")
        aff = rid_to_aff.get(rid, "interaction_zone")
        wv = aff_map.get(aff, default)
        vec = np.array([float(wv.get(k, default.get(k, 0.25))) for k in ch_names], dtype=np.float32)
        s = float(vec.sum()) or 1.0
        vec /= s
        pri[lm == lii, :] = vec
    return pri


def build_navigation_zones(
    cost_map: np.ndarray,
    feasible: np.ndarray,
    region_label_map: np.ndarray,
    *,
    speed_map: Optional[np.ndarray] = None,
    metric_depth_m: Optional[np.ndarray] = None,
    semantic_layer: Optional[Dict[str, Any]] = None,
    regions_meta: Optional[List[Dict[str, Any]]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build (H,W,4) float soft maps: idle, walk, run, careful — scene + geometry + affordance priors.
    Outside feasible regions, channels are zero.
    """
    cm = np.clip(np.asarray(cost_map, dtype=np.float32), 0.0, 1.0)
    free = np.asarray(feasible, dtype=bool)
    lm = np.asarray(region_label_map, dtype=np.int32)
    h, w = lm.shape[:2]
    pol = policy if policy is not None else load_navigation_zone_policy()

    dt = cv2.distanceTransform((free.astype(np.uint8) * 255), cv2.DIST_L2, 5).astype(np.float32)
    dt_vals = dt[free]
    p93 = float(np.percentile(dt_vals, 93)) if dt_vals.size > 1e-6 else float(np.max(dt) or 1.0)
    n_dt = np.clip(dt / max(p93, 1e-3), 0.0, 1.0)
    c = cm
    openness = n_dt * (1.0 - 0.82 * c)
    tight = (1.0 - n_dt) * (0.35 + 0.65 * c)
    sp = None
    if speed_map is not None and np.asarray(speed_map).shape[:2] == (h, w):
        sp = np.clip(np.asarray(speed_map, dtype=np.float32), 0.0, 1.0)
    run_geo = np.clip((openness ** 1.05) * (1.0 - 0.75 * c), 0.0, 1.0)
    if sp is not None:
        run_geo = np.clip(run_geo * (0.25 + 0.75 * sp), 0.0, 1.0)
    walk_geo = np.clip(0.4 * openness + 0.55 * (1.0 - c) * n_dt + 0.15 * (1.0 - tight), 0.0, 1.0)
    idle_geo = np.clip(tight * (0.45 + 0.55 * c) + 0.12 * (1.0 - n_dt), 0.0, 1.0)
    careful_geo = np.clip(c * (0.35 + 0.65 * (1.0 - n_dt)), 0.0, 1.0)
    if metric_depth_m is not None and np.asarray(metric_depth_m).shape[:2] == (h, w):
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        gx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
        slope = cv2.magnitude(gx, gy)
        sm = float(np.percentile(slope[free], 90)) if np.any(free) else float(np.max(slope) or 1.0)
        slope_n = np.clip(slope / max(sm, 1e-4), 0.0, 1.0)
        careful_geo = np.clip(careful_geo + 0.35 * slope_n * c, 0.0, 1.0)
        run_geo = np.clip(run_geo * (1.0 - 0.4 * slope_n), 0.0, 1.0)

    geo = np.stack([idle_geo, walk_geo, run_geo, careful_geo], axis=-1).astype(np.float32)
    aff_r = _affordance_prior_raster(lm, list(regions_meta or []), list((semantic_layer or {}).get("region_affordances") or []), pol)
    fused = geo * (0.35 + 0.65 * aff_r)
    fused[~free] = 0.0
    sums = fused.sum(axis=-1, keepdims=True)
    sums = np.maximum(sums, 1e-6)
    fused = fused / sums
    fused[~free] = 0.0

    arg = np.argmax(fused, axis=-1)
    dominant = (arg + 1).astype(np.uint8)
    dominant[~free] = 0

    meta: Dict[str, Any] = {
        "schema": "citv_navigation_zones_v1",
        "version": "1.0",
        "channel_order": list(pol.get("channel_order") or ["idle", "walk", "run", "careful"]),
        "shape": {"height": h, "width": w, "channels": 4},
        "dominant_label_legend": {"0": "none", "1": "idle", "2": "walk", "3": "run", "4": "careful"},
    }
    return fused, meta


def trace_navigation_along_polyline(
    polyline_2d: List[Any],
    zones: np.ndarray,
    image_width: int,
    image_height: int,
    num_samples: int = 48,
) -> Dict[str, Any]:
    """Mean soft channel values along polyline (arc-length uniform)."""
    from scene_understanding.pathing.polyline_scene_trace import build_polyline_scene_trace

    pts: List[Tuple[float, float]] = []
    for xy in polyline_2d or []:
        if not xy or len(xy) < 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2 or zones is None or zones.ndim != 3:
        return {"sample_count": 0, "mean_channels": {}, "dominant_motion_hint": ""}
    h, w, k = zones.shape
    n = max(8, min(96, int(num_samples)))
    # reuse UV sampling pattern via cost dummy
    dummy_cost = np.zeros((h, w), dtype=np.float32)
    uv_pack = build_polyline_scene_trace(
        polyline_2d,
        cost_map=dummy_cost,
        region_label_map=None,
        metric_depth_m=None,
        image_width=int(image_width),
        image_height=int(image_height),
        num_samples=n,
    )
    samples_uv = uv_pack.get("samples_uv") or []
    ch_names = ["idle", "walk", "run", "careful"]
    acc = np.zeros((k,), dtype=np.float64)
    cnt = 0
    for s in samples_uv:
        xi = int(round(float(s.get("u", 0.0))))
        yi = int(round(float(s.get("v", 0.0))))
        if 0 <= yi < h and 0 <= xi < w:
            acc += zones[yi, xi, :].astype(np.float64)
            cnt += 1
    if cnt <= 0:
        return {"sample_count": 0, "mean_channels": {}, "dominant_motion_hint": ""}
    mean = acc / float(cnt)
    j = int(np.argmax(mean))
    return {
        "sample_count": int(cnt),
        "mean_channels": {ch_names[i]: float(mean[i]) for i in range(min(k, len(ch_names)))},
        "dominant_motion_hint": ch_names[j] if j < len(ch_names) else "",
    }


def navigation_zones_to_rgba(zones: np.ndarray, dominant: Optional[np.ndarray] = None) -> np.ndarray:
    """BGR uint8 preview: idle=cyan, walk=green, run=yellow, careful=magenta."""
    h, w = zones.shape[:2]
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    z = np.clip(zones, 0.0, 1.0)
    # weighted color mix
    bgr = (
        z[:, :, 0:1] * np.array([[[200, 220, 80]]], dtype=np.float32)
        + z[:, :, 1:2] * np.array([[[60, 200, 60]]], dtype=np.float32)
        + z[:, :, 2:3] * np.array([[[60, 200, 255]]], dtype=np.float32)
        + z[:, :, 3:4] * np.array([[[255, 60, 255]]], dtype=np.float32)
    )
    s = np.maximum(z.sum(axis=-1, keepdims=True), 1e-6)
    bgr = (bgr / s).astype(np.uint8)
    if dominant is not None:
        pass
    return bgr


__all__ = [
    "build_navigation_zones",
    "load_navigation_zone_policy",
    "trace_navigation_along_polyline",
    "navigation_zones_to_rgba",
]
