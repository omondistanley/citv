"""Sample scene evidence along path polylines (cost, depth, region labels)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _index_to_region_id(regions_meta: Optional[List[Dict[str, Any]]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not regions_meta:
        return out
    for r in regions_meta:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("region_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        rid = str(r.get("id", "") or "").strip()
        if idx > 0 and rid:
            out[idx] = rid
    return out


def _region_id_to_affordance(semantic_layer: Dict[str, Any]) -> Dict[str, str]:
    aff_by_region: Dict[str, str] = {}
    for a in semantic_layer.get("region_affordances") or []:
        if not isinstance(a, dict):
            continue
        rid = str(a.get("region_id", "") or "").strip()
        if rid:
            aff_by_region[rid] = str(a.get("affordance", "interaction_zone") or "interaction_zone")
    return aff_by_region


def build_polyline_affordance_trace(
    polyline_2d: List[Any],
    *,
    region_label_map: np.ndarray,
    regions_meta: Optional[List[Dict[str, Any]]],
    semantic_layer: Dict[str, Any],
    image_width: int,
    image_height: int,
    num_samples: int = 40,
    void_affordance: str = "void",
) -> List[Dict[str, Any]]:
    """
    Per-sample affordances along the polyline from the region label map.

    Unlike counting only ``regions_traversed`` ids, this reflects pixels the
    polyline actually crosses (aligned with corridor / walkable checks).
    """
    pts: List[Tuple[float, float]] = []
    for xy in polyline_2d or []:
        if not xy or len(xy) < 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return []

    idx_to_rid = _index_to_region_id(regions_meta)
    rid_to_aff = _region_id_to_affordance(semantic_layer)

    n = max(8, min(128, int(num_samples)))
    cum, total = _cumlen(pts)
    h = int(image_height)
    w = int(image_width)
    lm = np.asarray(region_label_map, dtype=np.int32)

    trace: List[Dict[str, Any]] = []
    for i in range(n):
        s = (i + 0.5) / float(n)
        x, y = _interp_point(pts, cum, total, s)
        xi = int(round(x))
        yi = int(round(y))
        if lm.ndim != 2 or not (0 <= yi < lm.shape[0] and 0 <= xi < lm.shape[1]):
            trace.append({"region": "", "affordance": void_affordance, "u": float(x), "v": float(y)})
            continue
        li = int(lm[yi, xi])
        if li <= 0:
            trace.append({"region": "", "affordance": void_affordance, "u": float(x), "v": float(y)})
            continue
        rid = idx_to_rid.get(li, "")
        aff = rid_to_aff.get(rid, "interaction_zone")
        trace.append({"region": rid or str(li), "affordance": aff, "u": float(x), "v": float(y)})
    return trace


def _cumlen(pts: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
    if len(pts) < 2:
        return np.array([0.0], dtype=np.float64), 0.0
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    return cum, float(cum[-1])


def _interp_point(pts: List[Tuple[float, float]], cum: np.ndarray, total: float, s: float) -> Tuple[float, float]:
    s = float(max(0.0, min(1.0, s)))
    if len(pts) < 2 or total <= 1e-9:
        return float(pts[0][0]), float(pts[0][1])
    target = s * total
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(len(pts) - 2, idx))
    seg = float(cum[idx + 1] - cum[idx]) or 1e-6
    u = (target - float(cum[idx])) / seg
    x0, y0 = float(pts[idx][0]), float(pts[idx][1])
    x1, y1 = float(pts[idx + 1][0]), float(pts[idx + 1][1])
    return x0 + u * (x1 - x0), y0 + u * (y1 - y0)


def build_polyline_scene_trace(
    polyline_2d: List[Any],
    *,
    cost_map: Optional[np.ndarray],
    region_label_map: Optional[np.ndarray],
    metric_depth_m: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    num_samples: int = 40,
    high_cost_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Return summary statistics and sparse UV samples along the polyline."""
    pts: List[Tuple[float, float]] = []
    for xy in polyline_2d or []:
        if not xy or len(xy) < 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return {
            "sample_count": 0,
            "mean_cost": None,
            "p90_cost": None,
            "depth_delta_p90_m": None,
            "high_cost_fraction": None,
            "region_label_modes": [],
            "samples_uv": [],
        }

    n = max(8, min(128, int(num_samples)))
    cum, total = _cumlen(pts)
    costs: List[float] = []
    depths: List[float] = []
    reg_ids: List[int] = []
    samples_uv: List[Dict[str, float]] = []
    h = int(image_height)
    w = int(image_width)
    cm = np.asarray(cost_map, dtype=np.float32) if cost_map is not None else None
    dm = np.asarray(metric_depth_m, dtype=np.float32) if metric_depth_m is not None else None
    lm = np.asarray(region_label_map, dtype=np.int32) if region_label_map is not None else None

    for i in range(n):
        s = (i + 0.5) / float(n)
        x, y = _interp_point(pts, cum, total, s)
        xi = int(round(x))
        yi = int(round(y))
        samples_uv.append({"u": float(x), "v": float(y)})
        if cm is not None and cm.ndim == 2 and 0 <= yi < cm.shape[0] and 0 <= xi < cm.shape[1]:
            costs.append(float(cm[yi, xi]))
        if dm is not None and dm.ndim >= 2 and 0 <= yi < dm.shape[0] and 0 <= xi < dm.shape[1]:
            d0 = float(dm[yi, xi])
            if np.isfinite(d0):
                depths.append(d0)
        if lm is not None and lm.ndim == 2 and 0 <= yi < lm.shape[0] and 0 <= xi < lm.shape[1]:
            reg_ids.append(int(lm[yi, xi]))

    mean_cost = float(np.mean(costs)) if costs else None
    p90_cost = float(np.percentile(costs, 90)) if costs else None
    hc_thr = float(high_cost_threshold)
    high_cost_fraction = float(sum(1 for c in costs if c > hc_thr) / max(1, len(costs))) if costs else None

    depth_delta_p90: Optional[float] = None
    if len(depths) >= 2:
        arr = np.asarray(depths, dtype=np.float64)
        depth_delta_p90 = float(np.percentile(np.abs(np.diff(arr)), 90))

    region_modes: List[Dict[str, Any]] = []
    if reg_ids:
        vals, counts = np.unique(np.asarray(reg_ids, dtype=np.int32), return_counts=True)
        order = np.argsort(-counts)[:5]
        for j in order:
            region_modes.append({"region_label": int(vals[j]), "count": int(counts[j])})

    return {
        "sample_count": int(n),
        "mean_cost": mean_cost,
        "p90_cost": p90_cost,
        "depth_delta_p90_m": depth_delta_p90,
        "high_cost_fraction": high_cost_fraction,
        "region_label_modes": region_modes,
        "samples_uv": samples_uv[: min(16, len(samples_uv))],
    }


__all__ = ["build_polyline_scene_trace", "build_polyline_affordance_trace"]
