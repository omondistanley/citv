"""
Phase-1 motion contracts: insertion_path_ensemble and trajectory_hypothesis JSON
built from legacy path_hypotheses entries and mask geometry.

Kept separate from scene_understanding.py to avoid circular imports and ease testing.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scene_path_traversability import heading_from_depth_at


def infer_path_family(path_dict: Dict[str, Any]) -> str:
    level = str(path_dict.get("path_level", "")).lower()
    ptype = str(path_dict.get("path_type", "")).lower()
    src = path_dict.get("source_entity") or {}
    tgt = path_dict.get("target_entity") or {}
    st = str(src.get("type", "")).lower()
    tt = str(tgt.get("type", "")).lower()
    if level == "region" and ptype == "region_to_region":
        return "region_region"
    if level == "object" and ptype == "object_to_object":
        return "object_object"
    if level == "mask" and ptype in ("mask_contour", "mask_axis"):
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
) -> List[Dict[str, Any]]:
    """
    One instant_prior trajectory per eligible mover: centroid + heading (PCA, optionally
    blended with local depth isocontour tangent) + short offset.
    """
    max_n = int(getattr(cfg, "trajectory_hypotheses_max_subjects", 8)) if cfg else 8
    step_px = float(getattr(cfg, "trajectory_instant_step_px", 6.0)) if cfg else 6.0
    dt = float(getattr(cfg, "trajectory_instant_dt_s", 0.04)) if cfg else 0.04
    use_depth_heading = bool(getattr(cfg, "trajectory_use_depth_heading", True)) if cfg else True
    w_depth = float(getattr(cfg, "trajectory_depth_heading_blend", 0.55)) if cfg else 0.55

    rel_base = f"scene_graph/{track_dir_name}/{path_stem}_paths"
    cost_ref = f"{rel_base}/path_cost_map.npy"
    hyp_ref = f"{rel_base}/path_hypotheses.json"

    objs = [o for o in objects_3d_with_masks if str(o.get("entity_kind", "object")).lower() != "region"]
    movers = [o for o in objs if _is_mover_object(o, cfg)]
    if not movers:
        movers = objs[:max_n]
    movers = movers[:max_n]

    h_img = int(image_size.get("height", 0))
    w_img = int(image_size.get("width", 0))

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
        x1 = cx + step_px * math.cos(theta)
        y1 = cy + step_px * math.sin(theta)
        z0 = _object_depth_m(o)

        name = ""
        for k in ("canonical_name", "name", "label"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break

        traj_id = f"traj_{image_stem}_{oid}_instant00"
        sample = {
            "sample_id": f"{traj_id}_s0",
            "weight": 1.0,
            "states_t": [
                {"t_s": 0.0, "x_px": cx, "y_px": cy, "theta_rad": theta, "z_m": z0},
                {"t_s": dt, "x_px": x1, "y_px": y1, "theta_rad": theta, "z_m": z0},
            ],
            "evidence": {
                "mask_pca_heading": True,
                "depth_heading_blend": depth_heading_used,
                "depth_available": z0 is not None,
            },
        }

        cu: Dict[str, Any] = {
            "path_cost_map_npy": cost_ref,
            "path_hypotheses_json": hyp_ref,
            "relation_ids": _relation_ids_for_object(oid, relations),
        }
        if traversability_speed_npy_rel:
            cu["path_traversability_speed_npy"] = traversability_speed_npy_rel

        out.append(
            {
                "trajectory_id": traj_id,
                "subject": {
                    "kind": "object",
                    "id": oid,
                    "role": "primary_actor",
                    "display_label": name or oid,
                },
                "counterpart": None,
                "time_model": {"type": "instant_prior", "horizon_s": dt, "dt_s": dt},
                "state_representation": "se2_root",
                "samples": [sample],
                "constraints_used": cu,
                "confidence": 0.55 if depth_heading_used else 0.5,
                "notes": "instant_prior: PCA heading optionally blended with depth isocontour tangent",
            }
        )
    return out


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
    )
    return {
        "schema": "citv_trajectory_hypotheses_bundle_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "hypotheses": hyps,
    }
