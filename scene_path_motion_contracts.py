"""
Phase-1 motion contracts: insertion_path_ensemble and trajectory_hypothesis JSON
built from legacy path_hypotheses entries and mask geometry.

Kept separate from scene_understanding.py to avoid circular imports and ease testing.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scene_path_traversability import heading_from_depth_at


def _polyline_points(polyline_2d: List[Any]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for xy in polyline_2d or []:
        if not xy or len(xy) < 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except (TypeError, ValueError):
            continue
    return pts


def _sample_polyline_at_arc_length(
    pts: List[Tuple[float, float]],
    cum: np.ndarray,
    total: float,
    target: float,
    vertex_z_m: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Optional[float]]:
    target = float(max(0.0, min(total, target)))
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(len(pts) - 2, idx))
    seg_len = float(cum[idx + 1] - cum[idx]) or 1e-6
    u = (target - float(cum[idx])) / seg_len
    x0, y0 = float(pts[idx][0]), float(pts[idx][1])
    x1, y1 = float(pts[idx + 1][0]), float(pts[idx + 1][1])
    px = x0 + u * (x1 - x0)
    py = y0 + u * (y1 - y0)
    theta = math.atan2(y1 - y0, x1 - x0)
    z_out: Optional[float] = None
    if vertex_z_m is not None and len(vertex_z_m) == len(pts):
        z0 = float(vertex_z_m[idx])
        z1 = float(vertex_z_m[idx + 1])
        if math.isfinite(z0) and math.isfinite(z1) and z0 > 0.0 and z1 > 0.0:
            z_out = z0 + u * (z1 - z0)
    return px, py, theta, z_out


def polyline_to_states_t(
    polyline_2d: List[Any],
    *,
    horizon_s: float,
    rollout_steps: int,
    z_m: Optional[float],
    navigation_zones: Optional[np.ndarray] = None,
    cost_map: Optional[np.ndarray] = None,
    adaptive_bias_base: float = 0.35,
    adaptive_bias_careful: float = 0.45,
    adaptive_bias_cost: float = 0.35,
    adaptive_bias_not_run: float = 0.25,
    lateral_walk_weight: float = 0.60,
    lateral_run_weight: float = 0.40,
    lateral_careful_penalty: float = 0.55,
    lateral_amplitude_px: float = 1.60,
    lateral_phase_step: float = 0.85,
    lateral_endpoint_ramp_steps: int = 3,
    vertex_z_m: Optional[List[Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Samples along polyline with tangent headings (image px).

    When ``navigation_zones`` (H,W,4 idle/walk/run/careful) and ``cost_map`` are set,
    arc-length spacing is **non-uniform**: more samples where careful/cost are high
    (organic, scene-aware pacing along the route).
    """
    pts = _polyline_points(polyline_2d)
    if len(pts) < 2:
        return []
    rollout_steps = max(2, int(rollout_steps))
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1]) or 1e-6
    dt = float(horizon_s) / float(max(1, rollout_steps))
    states: List[Dict[str, Any]] = []
    z_arr: Optional[np.ndarray] = None
    if vertex_z_m is not None and len(vertex_z_m) == len(pts):
        z_arr = np.array([float(z) if z is not None and math.isfinite(float(z)) else float("nan") for z in vertex_z_m], dtype=np.float64)

    use_adaptive = (
        navigation_zones is not None
        and cost_map is not None
        and np.asarray(navigation_zones).ndim == 3
        and np.asarray(cost_map).ndim == 2
    )
    if not use_adaptive:
        for si in range(rollout_steps + 1):
            s = si / float(max(1, rollout_steps))
            target = s * total
            px, py, theta, zs = _sample_polyline_at_arc_length(pts, cum, total, target, z_arr)
            zm = z_m
            if zs is not None and math.isfinite(zs) and zs > 0.0:
                zm = float(zs)
            states.append({"t_s": float(si * dt), "x_px": px, "y_px": py, "theta_rad": theta, "z_m": zm})
        return states

    nz = np.asarray(navigation_zones, dtype=np.float32)
    cm = np.clip(np.asarray(cost_map, dtype=np.float32), 0.0, 1.0)
    h, w = cm.shape[:2]
    nseg = max(1, len(pts) - 1)
    weights: List[float] = []
    for i in range(nseg):
        x0, y0 = int(round(pts[i][0])), int(round(pts[i][1]))
        x1, y1 = int(round(pts[i + 1][0])), int(round(pts[i + 1][1]))
        acc = 0.0
        cnt = 0
        nsub = max(2, int(d[i]) + 1)
        for t in range(nsub):
            a = t / float(nsub - 1) if nsub > 1 else 0.0
            xi = int(round((1 - a) * x0 + a * x1))
            yi = int(round((1 - a) * y0 + a * y1))
            if 0 <= yi < h and 0 <= xi < w:
                careful = float(nz[yi, xi, 3])
                runv = float(nz[yi, xi, 2])
                cst = float(cm[yi, xi])
                acc += (
                    float(adaptive_bias_base)
                    + float(adaptive_bias_careful) * careful
                    + float(adaptive_bias_cost) * cst
                    + float(adaptive_bias_not_run) * (1.0 - runv)
                )
                cnt += 1
        weights.append((acc / max(1, cnt)) * float(d[i] + 1e-3))
    ws = np.array(weights, dtype=np.float64)
    ws = ws / (float(ws.sum()) + 1e-9)
    cum_w = np.concatenate([[0.0], np.cumsum(ws)])
    for si in range(rollout_steps + 1):
        uu = si / float(max(1, rollout_steps))
        uw = uu * float(cum_w[-1] or 1.0)
        j = int(np.searchsorted(cum_w, uw, side="right") - 1)
        j = max(0, min(nseg - 1, j))
        seg_start = float(cum[j])
        seg_end = float(cum[j + 1])
        span = max(1e-6, seg_end - seg_start)
        alpha = (uw - cum_w[j]) / max(1e-9, (cum_w[j + 1] - cum_w[j]))
        alpha = max(0.0, min(1.0, alpha))
        target = seg_start + alpha * span
        px, py, theta, zs = _sample_polyline_at_arc_length(pts, cum, total, target, z_arr)
        # Mild lateral organic variation in open/low-cost areas.
        # Keeps motion scene-aware while avoiding rigid "on-rail" movement.
        xi = int(max(0, min(w - 1, round(px))))
        yi = int(max(0, min(h - 1, round(py))))
        if 0 <= yi < h and 0 <= xi < w:
            walkv = float(nz[yi, xi, 1])
            runv = float(nz[yi, xi, 2])
            careful = float(nz[yi, xi, 3])
            cst = float(cm[yi, xi])
            openness = max(
                0.0,
                min(
                    1.0,
                    float(lateral_walk_weight) * walkv + float(lateral_run_weight) * runv - float(lateral_careful_penalty) * careful,
                ),
            )
            n_ramp = max(1, int(lateral_endpoint_ramp_steps))
            edge = min(float(si) / float(n_ramp), float(rollout_steps - si) / float(n_ramp), 1.0)
            amp = float(lateral_amplitude_px) * openness * max(0.0, 1.0 - cst) * edge
            
            # ENHANCEMENT: Harmonic blending for organic motion
            phase1 = float(lateral_phase_step) * float(si)
            phase2 = float(lateral_phase_step) * 0.43 * float(si) + 1.2
            # Blend primary stride with an out-of-phase secondary shift
            offset = amp * (math.sin(phase1) + 0.5 * math.sin(phase2)) / 1.5
            
            nx, ny = -math.sin(theta), math.cos(theta)
            px = max(0.0, min(float(w - 1), px + offset * nx))
            py = max(0.0, min(float(h - 1), py + offset * ny))
        zm = z_m
        if zs is not None and math.isfinite(zs) and zs > 0.0:
            zm = float(zs)
        states.append({"t_s": float(si * dt), "x_px": px, "y_px": py, "theta_rad": theta, "z_m": zm})
    return states


def mask_contour_to_states_t(
    mask: np.ndarray,
    *,
    horizon_s: float,
    rollout_steps: int,
    z_m: Optional[float],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """Walk along the largest exterior contour of a binary mask (intra, organic)."""
    mm = np.asarray(mask, dtype=np.uint8)
    if mm.ndim != 2:
        return []
    if int(image_width) > 0 and int(image_height) > 0 and mm.shape[:2] != (int(image_height), int(image_width)):
        mm = cv2.resize(mm, (int(image_width), int(image_height)), interpolation=cv2.INTER_NEAREST)
    mm = (mm > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 3.0:
        return []
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return []
    step = max(1, len(pts) // max(24, rollout_steps * 6))
    samp = pts[::step]
    if len(samp) < 3:
        samp = pts
    closed = np.vstack([samp, samp[:1]])
    xs, ys = closed[:, 0], closed[:, 1]
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    tot = float(cum[-1]) or 1e-6
    pts_list = [(float(closed[i, 0]), float(closed[i, 1])) for i in range(len(closed))]
    rollout_steps = max(2, int(rollout_steps))
    dt = float(horizon_s) / float(max(1, rollout_steps))
    states: List[Dict[str, Any]] = []
    for si in range(rollout_steps + 1):
        s = si / float(max(1, rollout_steps))
        target = s * tot
        px, py, theta, _zs = _sample_polyline_at_arc_length(pts_list, cum, tot, target)
        states.append({"t_s": float(si * dt), "x_px": px, "y_px": py, "theta_rad": theta, "z_m": z_m})
    return states


def _structured_relation_match(path_dict: Dict[str, Any], relations: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    src = path_dict.get("source_entity") or {}
    tgt = path_dict.get("target_entity") or {}
    sid = str(src.get("id", "") or "").strip()
    tid = str(tgt.get("id", "") or "").strip()
    out: Dict[str, Any] = {"best_predicate": "", "best_score": 0.0}
    if not sid or not tid or not relations:
        return out
    best_score = -1.0
    best_pred = ""
    for r in relations:
        if not isinstance(r, dict):
            continue
        rs = str(r.get("subject_id", "") or "").strip()
        ro = str(r.get("object_id", "") or "").strip()
        if not rs or not ro:
            continue
        sc = 0.0
        if rs == sid and ro == tid:
            sc = float(r.get("score", 0.0) or 0.0)
        elif rs == tid and ro == sid:
            sc = float(r.get("score", 0.0) or 0.0) * 0.85
        if sc > best_score:
            best_score = sc
            best_pred = str(r.get("predicate", "") or "")
    out["best_predicate"] = best_pred
    out["best_score"] = float(max(0.0, best_score))
    return out


def infer_path_family(path_dict: Dict[str, Any]) -> str:
    level = str(path_dict.get("path_level", "")).lower()
    ptype = str(path_dict.get("path_type", "")).lower()
    src = path_dict.get("source_entity") or {}
    tgt = path_dict.get("target_entity") or {}
    st = str(src.get("type", "")).lower()
    tt = str(tgt.get("type", "")).lower()
    if level == "object_region" or ptype == "object_region_fmm":
        return "object_region"
    if level == "region" and ptype in ("region_to_region", "region_fmm"):
        return "region_region"
    if level == "object" and ptype == "object_to_object":
        return "object_object"
    if level == "mask" and ptype in ("mask_contour", "mask_axis", "mask_interior_geodesic"):
        return "per_mask"
    if st and tt:
        return f"{st}_{tt}"
    return "unknown"


def _agent_class_for_path(path_dict: Dict[str, Any], cfg: Any) -> str:
    labels = []
    for ent in (path_dict.get("source_entity"), path_dict.get("target_entity")):
        if not isinstance(ent, dict):
            continue
        for k in ("display_label", "name", "label"):
            v = ent.get(k)
            if isinstance(v, str) and v.strip():
                labels.append(v.strip().lower())
    blob = " ".join(labels)
    ped = getattr(cfg, "motion_contract_agent_pedestrian_tokens", ("person", "people", "pedestrian", "man", "woman", "child")) if cfg else ()
    veh = getattr(cfg, "motion_contract_agent_vehicle_tokens", ("vehicle", "car", "truck", "bus", "bike", "bicycle")) if cfg else ()
    for t in ped:
        if t in blob:
            return "pedestrian"
    for t in veh:
        if t in blob:
            return "vehicle"
    return "generic_box"


def _relation_indices_touching(path_dict: Dict[str, Any], relations: Optional[List[Dict[str, Any]]]) -> List[str]:
    ids: List[str] = []
    for ent in (path_dict.get("source_entity"), path_dict.get("target_entity")):
        if isinstance(ent, dict):
            i = str(ent.get("id", "")).strip()
            if i:
                ids.append(i.lower())
    out: List[str] = []
    for i, r in enumerate(relations or []):
        try:
            s = json.dumps(r, ensure_ascii=False).lower()
        except Exception:
            s = str(r).lower()
        if any(x in s for x in ids):
            out.append(f"relation_index_{i}")
    return out


def legacy_path_to_insertion_ensemble(
    path_dict: Dict[str, Any],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
    path_stem: str,
    cfg: Any,
    relations: Optional[List[Dict[str, Any]]] = None,
    traversability_speed_npy_rel: str = "",
) -> Dict[str, Any]:
    """Wrap one legacy accepted path as an insertion ensemble (primary + optional geodesic hypotheses)."""
    pid = str(path_dict.get("path_id", ""))
    family = infer_path_family(path_dict)
    src = path_dict.get("source_entity") or {}
    tgt = path_dict.get("target_entity") or {}
    traversability_ref = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_cost_map.npy"
    hypothesis = {
        "hypothesis_id": f"{pid}_h0",
        "legacy_path_id": pid,
        "polyline_2d": [list(map(float, xy)) for xy in (path_dict.get("polyline_2d") or [])],
        "corridor": None,
        "topology_hint": {"regions_traversed": list(path_dict.get("regions_traversed") or [])},
        "scores": dict(path_dict.get("scores") or {}),
        "route_kind": "legacy_primary",
    }
    hypotheses_list: List[Dict[str, Any]] = [hypothesis]
    geo = path_dict.get("polyline_geodesic_2d")
    if isinstance(geo, list) and len(geo) >= 2:
        hypotheses_list.append(
            {
                "hypothesis_id": f"{pid}_h1_geodesic",
                "legacy_path_id": pid,
                "polyline_2d": [list(map(float, xy)) for xy in geo],
                "corridor": None,
                "topology_hint": {"regions_traversed": list(path_dict.get("regions_traversed") or [])},
                "scores": dict(path_dict.get("scores") or {}),
                "route_kind": "traversability_geodesic",
            }
        )
    for j, alt in enumerate(path_dict.get("polyline_geodesic_alternates_2d") or []):
        if isinstance(alt, list) and len(alt) >= 2:
            hypotheses_list.append(
                {
                    "hypothesis_id": f"{pid}_h{2 + j}_geodesic_alt",
                    "legacy_path_id": pid,
                    "polyline_2d": [list(map(float, xy)) for xy in alt],
                    "corridor": None,
                    "topology_hint": {"regions_traversed": list(path_dict.get("regions_traversed") or [])},
                    "scores": dict(path_dict.get("scores") or {}),
                    "route_kind": "traversability_geodesic_diverse",
                }
            )
    ensemble = {
        "path_family_id": pid,
        "path_family": family,
        "path_type": str(path_dict.get("path_type", "")),
        "agent_model": {
            "class": _agent_class_for_path(path_dict, cfg),
            "footprint_m": float(getattr(cfg, "motion_contract_default_footprint_m", 0.45)) if cfg else 0.45,
            "clearance_m": float(getattr(cfg, "motion_contract_default_clearance_m", 0.15)) if cfg else 0.15,
        },
        "endpoints": {
            "start": {
                "kind": str(src.get("type", "object")),
                "id": str(src.get("id", "")),
                "manifold": "centroid_uv" if str(src.get("type", "")).lower() == "object" else "anchor_uv",
                "display_label": str(src.get("display_label", "")),
            },
            "goal": {
                "kind": str(tgt.get("type", "object")),
                "id": str(tgt.get("id", "")),
                "manifold": "centroid_uv" if str(tgt.get("type", "")).lower() == "object" else "anchor_uv",
                "display_label": str(tgt.get("display_label", "")),
            },
        },
        "traversability_field_ref": traversability_ref,
        "traversability_speed_map_ref": traversability_speed_npy_rel or "",
        "hypotheses": hypotheses_list,
        "workflow_conditioning": {"relation_ids": _relation_indices_touching(path_dict, relations)},
        "notes": "adapter_from_legacy_path_hypothesis_plus_optional_geodesic",
        "regions_traversed": list(path_dict.get("regions_traversed") or []),
        "constraints_applied": dict(path_dict.get("constraints_applied") or {}),
    }
    return ensemble


def _mask_pca_tangent(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns centroid (cx, cy) and unit tangent (tx, ty) along major axis of mask pixels.
    Image coords: x right, y down.
    """
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) < 10:
        cx = float(np.mean(xs)) if len(xs) else 0.0
        cy = float(np.mean(ys)) if len(ys) else 0.0
        return cx, cy, 1.0, 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mu = pts.mean(axis=0, keepdims=True)
    x0 = pts - mu
    cov = (x0.T @ x0) / max(1.0, float(pts.shape[0] - 1))
    vals, vecs = np.linalg.eigh(cov)
    v1 = vecs[:, int(np.argmax(vals))]
    tx, ty = float(v1[0]), float(v1[1])
    n = math.hypot(tx, ty)
    if n < 1e-6:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = tx / n, ty / n
    cx = float(mu[0, 0])
    cy = float(mu[0, 1])
    return cx, cy, tx, ty


def _object_depth_m(obj: Dict[str, Any]) -> Optional[float]:
    c3d = obj.get("coordinates_3d") or obj.get("coordinates_3d_from_mask") or {}
    z = c3d.get("z", None)
    if isinstance(z, (int, float)):
        return float(z)
    ds = obj.get("depth_stats") or {}
    for k in ("median", "mean", "min", "max"):
        v = ds.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _is_mover_object(obj: Dict[str, Any], cfg: Any) -> bool:
    lab = str(obj.get("label", "") or "").strip().lower()
    movers = ("person", "vehicle", "animal")
    if lab in movers:
        return True
    if cfg and bool(getattr(cfg, "trajectory_hypotheses_include_all_objects", False)):
        return str(obj.get("entity_kind", "object")).lower() != "region"
    return False


def _relation_ids_for_object(obj_id: str, relations: Optional[List[Dict[str, Any]]]) -> List[str]:
    oid = str(obj_id).strip().lower()
    if not oid:
        return []
    out: List[str] = []
    for i, r in enumerate(relations or []):
        try:
            s = json.dumps(r, ensure_ascii=False).lower()
        except Exception:
            s = str(r).lower()
        if oid in s:
            out.append(f"relation_index_{i}")
    return out


def _counterpart_from_path(linked: Dict[str, Any], subject_id: str) -> Optional[Dict[str, Any]]:
    src = linked.get("source_entity") or {}
    tgt = linked.get("target_entity") or {}
    sid = str(subject_id or "").strip()
    if str(src.get("id", "")).strip() == sid:
        ent = tgt
    elif str(tgt.get("id", "")).strip() == sid:
        ent = src
    else:
        return None
    if not str(ent.get("id", "")).strip():
        return None
    return {
        "kind": str(ent.get("type", "object")),
        "id": str(ent.get("id", "")),
        "display_label": str(ent.get("display_label") or ent.get("name") or ent.get("id", "")),
    }


def build_trajectory_hypotheses(
    objects_3d_with_masks: List[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
    path_stem: str,
    cfg: Any,
    metric_depth_m: Optional[np.ndarray] = None,
    traversability_speed_npy_rel: str = "",
    ranked_paths: Optional[List[Dict[str, Any]]] = None,
    navigation_zones: Optional[np.ndarray] = None,
    path_cost_map: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Build short-horizon, scene-aware trajectory hypotheses per mover.

    The rollout preserves local heading semantics (PCA + optional depth heading)
    and emits multiple states (not just a single two-point instant prior).
    """
    max_n = int(getattr(cfg, "trajectory_hypotheses_max_subjects", 8)) if cfg else 8
    dt = float(getattr(cfg, "trajectory_rollout_dt_s", getattr(cfg, "trajectory_instant_dt_s", 0.04))) if cfg else 0.04
    rollout_steps = int(getattr(cfg, "trajectory_rollout_steps", 4)) if cfg else 4
    rollout_steps = max(2, rollout_steps)
    use_depth_heading = bool(getattr(cfg, "trajectory_use_depth_heading", True)) if cfg else True
    w_depth = float(getattr(cfg, "trajectory_depth_heading_blend", 0.55)) if cfg else 0.55

    rel_base = f"scene_graph/{track_dir_name}/{path_stem}_paths"
    cost_ref = f"{rel_base}/path_cost_map.npy"
    hyp_ref = f"{rel_base}/path_hypotheses.json"
    nav_zones_rel = f"{rel_base}/navigation_zones.npy" if navigation_zones is not None else ""

    objs = [o for o in objects_3d_with_masks if str(o.get("entity_kind", "object")).lower() != "region"]
    movers = [o for o in objs if _is_mover_object(o, cfg)]
    if not movers:
        movers = objs[:max_n]
    movers = movers[:max_n]

    h_img = int(image_size.get("height", 0))
    w_img = int(image_size.get("width", 0))

    # Calibrate step size from object scales when possible.
    obj_scales: List[float] = []
    for o in movers:
        m = o.get("_sam2_mask_array", None)
        if m is None:
            continue
        mm = np.asarray(m, dtype=bool)
        ys, xs = np.where(mm)
        if xs.size > 0:
            span = math.hypot(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
            if np.isfinite(span) and span > 0:
                obj_scales.append(span)
    scale_step = float(np.quantile(np.asarray(obj_scales, dtype=np.float32), 0.20)) if obj_scales else 6.0
    scale_factor = float(getattr(cfg, "trajectory_rollout_step_scale", 0.25)) if cfg else 0.25
    step_px = float(getattr(cfg, "trajectory_instant_step_px", max(2.0, scale_step * scale_factor))) if cfg else max(2.0, scale_step * scale_factor)

    out: List[Dict[str, Any]] = []
    for o in movers:
        oid = str(o.get("id", "")).strip()
        if not oid:
            continue
        m = o.get("_sam2_mask_array", None)
        if m is None:
            uv = o.get("mask_centroid_2d") or [0, 0]
            cx, cy = float(uv[0]), float(uv[1])
            tx, ty = 1.0, 0.0
        else:
            mm = np.asarray(m, dtype=bool)
            if h_img > 0 and w_img > 0 and mm.shape[:2] != (h_img, w_img):
                import cv2 as _cv2

                mm = _cv2.resize(mm.astype(np.uint8), (w_img, h_img), interpolation=_cv2.INTER_NEAREST) > 0
            cx, cy, tx, ty = _mask_pca_tangent(mm)

        theta_pca = math.atan2(ty, tx)
        theta = theta_pca
        depth_heading_used = False
        if use_depth_heading and metric_depth_m is not None:
            th_d = heading_from_depth_at(metric_depth_m, cx, cy, window=int(getattr(cfg, "trajectory_depth_heading_window", 9)) if cfg else 9)
            if th_d is not None:
                depth_heading_used = True
                w = max(0.0, min(1.0, w_depth))
                theta = math.atan2(
                    (1.0 - w) * math.sin(theta_pca) + w * math.sin(th_d),
                    (1.0 - w) * math.cos(theta_pca) + w * math.cos(th_d),
                )
        # Multi-step rollout with bounded image coordinates.
        # Stops early if the step would leave the object's own mask (prevents
        # rollouts that walk straight through walls or into other objects).
        states_t: List[Dict[str, Any]] = []
        px, py = float(cx), float(cy)
        states_t.append({"t_s": 0.0, "x_px": px, "y_px": py, "theta_rad": theta, "z_m": _object_depth_m(o)})
        for si in range(1, rollout_steps + 1):
            nx_px = max(0.0, min(float(w_img - 1), px + step_px * math.cos(theta)))
            ny_px = max(0.0, min(float(h_img - 1), py + step_px * math.sin(theta)))
            if m is not None:
                _ix, _iy = int(round(nx_px)), int(round(ny_px))
                if not (0 <= _ix < w_img and 0 <= _iy < h_img and bool(mm[_iy, _ix])):
                    break
            px, py = nx_px, ny_px
            states_t.append({"t_s": float(si * dt), "x_px": px, "y_px": py, "theta_rad": theta, "z_m": _object_depth_m(o)})
        z0 = _object_depth_m(o)

        name = ""
        for k in ("canonical_name", "name", "label"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break

        traj_id = f"traj_{image_stem}_{oid}_instant00"
        sample_rollout = {
            "sample_id": f"{traj_id}_rollout_pca",
            "sample_kind": "rollout_pca",
            "weight": 0.32,
            "states_t": states_t,
            "evidence": {
                "mask_pca_heading": True,
                "depth_heading_blend": depth_heading_used,
                "depth_available": z0 is not None,
                "rollout_steps": rollout_steps,
                "step_px": step_px,
            },
        }
        states_idle: List[Dict[str, Any]] = []
        for si in range(rollout_steps + 1):
            jx = cx + 0.2 * math.sin(si * 0.4)
            jy = cy + 0.2 * math.cos(si * 0.4)
            states_idle.append(
                {
                    "t_s": float(si * dt),
                    "x_px": float(jx),
                    "y_px": float(jy),
                    "theta_rad": theta,
                    "z_m": z0,
                }
            )
        sample_idle = {
            "sample_id": f"{traj_id}_idle",
            "sample_kind": "idle",
            "weight": 0.18,
            "states_t": states_idle,
            "evidence": {"mode": "idle_micro_jitter", "rollout_steps": rollout_steps},
        }
        samples: List[Dict[str, Any]] = [sample_idle, sample_rollout]
        linked_path: Dict[str, Any] = {}
        if ranked_paths:
            linked_path = _link_best_path_for_subject(oid, ranked_paths)
        poly = list(linked_path.get("polyline_2d") or []) if linked_path else []
        horizon = float(rollout_steps * dt)
        nz_arr = np.asarray(navigation_zones, dtype=np.float32) if navigation_zones is not None else None
        cm_arr = np.asarray(path_cost_map, dtype=np.float32) if path_cost_map is not None else None
        vertex_z_path: Optional[List[Optional[float]]] = None
        p3 = linked_path.get("polyline_3d") if linked_path else None
        pts2 = _polyline_points(poly)
        if isinstance(p3, list) and len(p3) == len(pts2) and pts2:
            zlist: List[Optional[float]] = []
            ok = True
            for item in p3:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    try:
                        zz = float(item[2])
                        zlist.append(zz if math.isfinite(zz) and zz > 0.0 else None)
                    except (TypeError, ValueError):
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok and zlist:
                vertex_z_path = zlist
        states_path = (
            polyline_to_states_t(
                poly,
                horizon_s=horizon,
                rollout_steps=rollout_steps,
                z_m=z0,
                navigation_zones=nz_arr,
                cost_map=cm_arr,
                adaptive_bias_base=float(getattr(cfg, "trajectory_adaptive_bias_base", 0.35)) if cfg else 0.35,
                adaptive_bias_careful=float(getattr(cfg, "trajectory_adaptive_bias_careful", 0.45)) if cfg else 0.45,
                adaptive_bias_cost=float(getattr(cfg, "trajectory_adaptive_bias_cost", 0.35)) if cfg else 0.35,
                adaptive_bias_not_run=float(getattr(cfg, "trajectory_adaptive_bias_not_run", 0.25)) if cfg else 0.25,
                lateral_walk_weight=float(getattr(cfg, "trajectory_lateral_walk_weight", 0.60)) if cfg else 0.60,
                lateral_run_weight=float(getattr(cfg, "trajectory_lateral_run_weight", 0.40)) if cfg else 0.40,
                lateral_careful_penalty=float(getattr(cfg, "trajectory_lateral_careful_penalty", 0.55)) if cfg else 0.55,
                lateral_amplitude_px=float(getattr(cfg, "trajectory_lateral_amplitude_px", 1.60)) if cfg else 1.60,
                lateral_phase_step=float(getattr(cfg, "trajectory_lateral_phase_step", 0.85)) if cfg else 0.85,
                lateral_endpoint_ramp_steps=int(getattr(cfg, "trajectory_lateral_endpoint_ramp_steps", 3)) if cfg else 3,
                vertex_z_m=vertex_z_path,
            )
            if poly
            else []
        )
        if states_path:
            samples.append(
                {
                    "sample_id": f"{traj_id}_inter_path",
                    "sample_kind": "inter_path",
                    "weight": 0.28,
                    "states_t": states_path,
                    "evidence": {
                        "path_id": str(linked_path.get("path_id", "")),
                        "path_type": str(linked_path.get("path_type", "")),
                        "adaptive_arc_length": bool(nz_arr is not None and cm_arr is not None),
                    },
                }
            )
        if m is not None:
            mm_b = np.asarray(m, dtype=bool)
            if h_img > 0 and w_img > 0 and mm_b.shape[:2] != (h_img, w_img):
                mm_b = cv2.resize(mm_b.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST) > 0
            # Interior geodesic traversal: BFS through eroded mask interior, then
            # arc-length sampled via polyline_to_states_t for organic pacing.
            from scene_understanding.pathing.mask_interior_path import mask_interior_geodesic_points
            int_pts = mask_interior_geodesic_points(
                mm_b, w_img, h_img,
                erode_iters=1,
                kernel_px=3,
                min_component_px=200,
            )
            if len(int_pts) >= 2:
                st_intra = polyline_to_states_t(
                    [list(p) for p in int_pts],
                    horizon_s=horizon,
                    rollout_steps=rollout_steps,
                    z_m=z0,
                    navigation_zones=nz_arr,
                    cost_map=cm_arr,
                    adaptive_bias_base=float(getattr(cfg, "trajectory_adaptive_bias_base", 0.35)) if cfg else 0.35,
                    adaptive_bias_careful=float(getattr(cfg, "trajectory_adaptive_bias_careful", 0.45)) if cfg else 0.45,
                    adaptive_bias_cost=float(getattr(cfg, "trajectory_adaptive_bias_cost", 0.35)) if cfg else 0.35,
                    adaptive_bias_not_run=float(getattr(cfg, "trajectory_adaptive_bias_not_run", 0.25)) if cfg else 0.25,
                    lateral_walk_weight=float(getattr(cfg, "trajectory_lateral_walk_weight", 0.60)) if cfg else 0.60,
                    lateral_run_weight=float(getattr(cfg, "trajectory_lateral_run_weight", 0.40)) if cfg else 0.40,
                    lateral_careful_penalty=float(getattr(cfg, "trajectory_lateral_careful_penalty", 0.55)) if cfg else 0.55,
                    lateral_amplitude_px=float(getattr(cfg, "trajectory_lateral_amplitude_px", 1.60)) if cfg else 1.60,
                    lateral_phase_step=float(getattr(cfg, "trajectory_lateral_phase_step", 0.85)) if cfg else 0.85,
                    lateral_endpoint_ramp_steps=int(getattr(cfg, "trajectory_lateral_endpoint_ramp_steps", 3)) if cfg else 3,
                )
                if st_intra and len(st_intra) >= 2:
                    samples.append(
                        {
                            "sample_id": f"{traj_id}_intra_interior",
                            "sample_kind": "intra_interior",
                            "weight": 0.22,
                            "states_t": st_intra,
                            "evidence": {
                                "mode": "mask_interior_geodesic",
                                "rollout_steps": rollout_steps,
                                "erode_iters": 1,
                                "int_pts_count": len(int_pts),
                            },
                        }
                    )
        counterpart = _counterpart_from_path(linked_path, oid) if linked_path else None
        rel_match = _structured_relation_match(linked_path, relations) if linked_path else {}
        pred = str(rel_match.get("best_predicate", "") or "")
        nzt = (linked_path.get("navigation_zone_trace") or {}) if linked_path else {}
        mc = nzt.get("mean_channels") or {}
        mr = float(mc.get("run", 0.0))
        mw = float(mc.get("walk", 0.0))
        mi = float(mc.get("idle", 0.0))
        mc_care = float(mc.get("careful", 0.0))
        if mr > max(mw, mi) + 0.06:
            tl_motion = "run"
        elif mw >= mi and mw >= mr * 0.85:
            tl_motion = "walk"
        elif mi > mw + 0.05 or mc_care > 0.55:
            tl_motion = "idle"
        else:
            tl_motion = "walk"

        cu: Dict[str, Any] = {
            "path_cost_map_npy": cost_ref,
            "path_hypotheses_json": hyp_ref,
            "relation_ids": _relation_ids_for_object(oid, relations),
        }
        if traversability_speed_npy_rel:
            cu["path_traversability_speed_npy"] = traversability_speed_npy_rel
        if nav_zones_rel:
            cu["navigation_zones_npy"] = nav_zones_rel

        conf = 0.45
        conf += 0.20 if depth_heading_used else 0.0
        conf += 0.20 if z0 is not None else 0.0
        conf += 0.15 if len(states_t) >= 3 else 0.0
        conf = max(0.0, min(1.0, conf))
        z_boost_idle, z_boost_walk, z_boost_run = 1.0, 1.0, 1.0
        if nz_arr is not None and nz_arr.ndim == 3 and h_img > 0 and w_img > 0:
            xi = int(max(0, min(w_img - 1, round(cx))))
            yi = int(max(0, min(h_img - 1, round(cy))))
            z_boost_idle = 0.45 + 0.85 * float(nz_arr[yi, xi, 0])
            z_boost_walk = 0.45 + 0.85 * float(nz_arr[yi, xi, 1])
            z_boost_run = 0.45 + 0.85 * float(nz_arr[yi, xi, 2])
        motion_mode_candidates = [
            {"motion": "idle", "confidence": float(max(0.0, min(1.0, (1.0 - conf + 0.20) * z_boost_idle)))},
            {"motion": "walk", "confidence": float(max(0.0, min(1.0, conf * z_boost_walk)))},
            {"motion": "run", "confidence": float(max(0.0, min(1.0, (conf - 0.12) * z_boost_run)))},
            {"motion": "careful", "confidence": float(max(0.0, min(1.0, mc_care * (0.55 + 0.45 * conf))))},
            {"motion": "strafe", "confidence": float(max(0.0, min(1.0, 0.22 + 0.55 * mc_care + 0.18 * (1.0 - conf))))},
            {"motion": "traverse", "confidence": float(max(0.0, min(1.0, conf * (0.38 + 0.62 * mw))))},
            {"motion": "pivot", "confidence": float(max(0.0, min(1.0, 0.18 + 0.50 * conf * (1.0 - mw))))},
            {"motion": "jump", "confidence": float(max(0.0, min(1.0, conf - 0.35)))},
            {"motion": "crawl", "confidence": float(max(0.0, min(1.0, (0.30 + 0.40 * (1.0 - conf)) * (1.0 + mc_care))))},
            {"motion": "wave", "confidence": float(max(0.0, min(1.0, 0.25 + 0.35 * (1.0 - conf))))},
            {"motion": "gesture", "confidence": float(max(0.0, min(1.0, 0.20 + 0.42 * (1.0 - conf) + (0.12 if pred else 0.0))))},
            {"motion": "sit", "confidence": float(max(0.0, min(1.0, 0.20 + 0.45 * (1.0 - conf))))},
            {"motion": "surprised", "confidence": float(max(0.0, min(1.0, 0.15 + 0.40 * (1.0 - conf))))},
            {"motion": "hop", "confidence": float(max(0.0, min(1.0, conf - 0.28)))},
        ]
        motion_mode_candidates = sorted(motion_mode_candidates, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        timeline_records: List[Dict[str, Any]] = []
        if states_t:
            t_total = float(rollout_steps * dt)
            t_idle = float(dt)
            nav_hint = str(nzt.get("dominant_motion_hint", "") or "")
            tl0: Dict[str, Any] = {
                "t0_s": 0.0,
                "t1_s": t_idle,
                "motion": "idle",
                "relation_predicate": pred,
                "navigation_hint": nav_hint,
            }
            if t_total > 2.5 * dt:
                t_cut = t_idle + max(dt, (t_total - t_idle) * 0.72)
                t_cut = min(t_cut, t_total - 0.5 * dt)
                tail_motion = "careful" if mc_care > 0.45 else ("gesture" if pred else "idle")
                timeline_records.append(tl0)
                timeline_records.append(
                    {
                        "t0_s": t_idle,
                        "t1_s": t_cut,
                        "motion": tl_motion,
                        "relation_predicate": pred,
                        "navigation_hint": nav_hint,
                    }
                )
                timeline_records.append(
                    {
                        "t0_s": t_cut,
                        "t1_s": t_total,
                        "motion": tail_motion,
                        "relation_predicate": pred,
                        "navigation_hint": nav_hint,
                    }
                )
            else:
                timeline_records.append(tl0)
                timeline_records.append(
                    {
                        "t0_s": t_idle,
                        "t1_s": t_total,
                        "motion": tl_motion,
                        "relation_predicate": pred,
                        "navigation_hint": nav_hint,
                    }
                )
        relation_ids = _relation_ids_for_object(oid, relations)
        out.append(
            {
                "trajectory_id": traj_id,
                "subject": {
                    "kind": "object",
                    "id": oid,
                    "role": "primary_actor",
                    "display_label": name or oid,
                },
                "counterpart": counterpart,
                "continues_from_path_id": str(linked_path.get("path_id", "")) if linked_path else "",
                "time_model": {"type": "short_rollout_prior", "horizon_s": float(rollout_steps * dt), "dt_s": dt},
                "state_representation": "se2_root",
                "samples": samples,
                "constraints_used": cu,
                "relation_guardrails": {"relation_ids": relation_ids, "relation_support_count": int(len(relation_ids))},
                "motion_mode_candidates": motion_mode_candidates,
                "timeline_records": timeline_records,
                "confidence": conf,
                "notes": "short_rollout_prior: PCA heading with optional depth tangent blend and calibrated step",
            }
        )
    return out


def _link_best_path_for_subject(subject_id: str, paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    sid = str(subject_id or "").strip()
    if not sid:
        return {}
    best: Dict[str, Any] = {}
    best_score = -1.0
    for p in paths:
        src = str(((p.get("source_entity") or {}).get("id", "")))
        tgt = str(((p.get("target_entity") or {}).get("id", "")))
        if sid not in (src, tgt):
            continue
        score = float((p.get("scores") or {}).get("overall_confidence", 0.0))
        if score > best_score:
            best_score = score
            best = p
    if best:
        return best
    # Fallback to top-ranked path when no direct subject match is found.
    for p in paths:
        if bool(p.get("suppressed", False)):
            continue
        return p
    return paths[0] if paths else {}


def _segment_indices_for_timeline(path_pts: List[List[float]], n_timeline: int) -> List[List[int]]:
    n_seg = max(0, len(path_pts) - 1)
    if n_seg <= 0 or n_timeline <= 0:
        return [[] for _ in range(n_timeline)]
    out: List[List[int]] = []
    for i in range(n_timeline):
        start = int(round((i / max(1, n_timeline)) * n_seg))
        end = int(round(((i + 1) / max(1, n_timeline)) * n_seg))
        end = max(start + 1, end)
        idxs = list(range(min(n_seg, start), min(n_seg, end)))
        out.append(idxs if idxs else [min(n_seg - 1, max(0, start))])
    return out


def _primary_trajectory_states(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = ("inter_path", "intra_contour", "rollout_pca", "idle")
    for kind in order:
        for s in samples:
            if str(s.get("sample_kind", "")) == kind:
                st = list(s.get("states_t") or [])
                if st:
                    return [dict(x) for x in st]
    if samples:
        st0 = samples[0].get("states_t") or []
        return [dict(x) for x in st0]
    return []


def build_animation_components_bundle(
    *,
    paths: List[Dict[str, Any]],
    trajectory_bundle: Dict[str, Any],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
) -> Dict[str, Any]:
    components: List[Dict[str, Any]] = []
    hyps = list((trajectory_bundle or {}).get("hypotheses") or [])
    for h in hyps:
        tid = str(h.get("trajectory_id", ""))
        subj = h.get("subject") or {}
        sid = str(subj.get("id", ""))
        linked_path = _link_best_path_for_subject(sid, paths)
        pid = str(linked_path.get("path_id", "")) if linked_path else ""
        path_pts = [list(map(float, xy)) for xy in (linked_path.get("polyline_2d") or [])] if linked_path else []

        samples = list(h.get("samples") or [])
        states = _primary_trajectory_states(samples)
        timeline = [dict(tl) for tl in (h.get("timeline_records") or [])]
        seg_map = _segment_indices_for_timeline(path_pts, len(timeline))
        for i, tl in enumerate(timeline):
            tl["path_segment_indices"] = seg_map[i] if i < len(seg_map) else []

        confidence = float(h.get("confidence", 0.0))
        relation_guardrails = dict(h.get("relation_guardrails") or {})
        relation_support = float(relation_guardrails.get("relation_support_count", 0.0))
        relation_bonus = min(0.15, 0.03 * relation_support)
        path_sem = float(linked_path.get("semantic_validity_score", 0.0)) if linked_path else 0.0
        naturalness = max(0.0, min(1.0, 0.55 * confidence + 0.30 * path_sem + relation_bonus))

        scene_sem = {
            "semantic_valid": bool(linked_path.get("semantic_valid", True)) if linked_path else True,
            "semantic_validity_score": path_sem,
            "path_family": str(linked_path.get("path_family", "")) if linked_path else "",
            "generation_source": str(linked_path.get("generation_source", "")) if linked_path else "",
        }

        components.append(
            {
                "component_id": f"{tid}_component",
                "trajectory_id": tid,
                "path_id_linked": pid,
                "subject": subj,
                "motion_mode_candidates": [dict(x) for x in (h.get("motion_mode_candidates") or [])],
                "timeline_records": timeline,
                "trajectory_points": states,
                "trajectory_samples": [dict(s) for s in samples],
                "counterpart": h.get("counterpart"),
                "relation_guardrails": relation_guardrails,
                "scene_semantic_constraints": scene_sem,
                "naturalness_score": float(naturalness),
                "rejection_reasons": [] if naturalness >= 0.35 else ["low_naturalness"],
            }
        )

    return {
        "schema": "citv_animation_components_bundle_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "components": components,
    }


def build_insertion_bundle(
    paths: List[Dict[str, Any]],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
    path_stem: str,
    cfg: Any,
    relations: Optional[List[Dict[str, Any]]] = None,
    traversability_speed_npy_rel: str = "",
) -> Dict[str, Any]:
    """Top-level object written to insertion_path_ensembles.json."""
    ensembles = [
        legacy_path_to_insertion_ensemble(
            p,
            image_stem,
            image_size,
            track_dir_name,
            path_stem,
            cfg,
            relations,
            traversability_speed_npy_rel=traversability_speed_npy_rel,
        )
        for p in paths
        if not p.get("suppressed")
    ]
    return {
        "schema": "citv_insertion_path_ensembles_bundle_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "ensembles": ensembles,
    }


def build_trajectory_bundle(
    objects_3d_with_masks: List[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
    path_stem: str,
    cfg: Any,
    metric_depth_m: Optional[np.ndarray] = None,
    traversability_speed_npy_rel: str = "",
    ranked_paths: Optional[List[Dict[str, Any]]] = None,
    navigation_zones: Optional[np.ndarray] = None,
    path_cost_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    hyps = build_trajectory_hypotheses(
        objects_3d_with_masks,
        relations,
        image_stem,
        image_size,
        track_dir_name,
        path_stem,
        cfg,
        metric_depth_m=metric_depth_m,
        traversability_speed_npy_rel=traversability_speed_npy_rel,
        ranked_paths=ranked_paths,
        navigation_zones=navigation_zones,
        path_cost_map=path_cost_map,
    )
    return {
        "schema": "citv_trajectory_hypotheses_bundle_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "hypotheses": hyps,
    }
