"""Rasterize fused grounding evidence into soft fields for path generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _paint_mask(field: np.ndarray, mask: Any, score: float) -> None:
    if mask is None:
        return
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != field.shape or not arr.any():
        return
    field[arr] = np.maximum(field[arr], np.float32(max(0.0, min(1.0, score))))


def build_affordance_rasters(
    *,
    grounding_index: Dict[str, Any],
    objects_by_id: Dict[str, Dict[str, Any]],
    support_mask: Any,
    obstacle_mask: Any,
    metric_depth: Any,
    shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    h, w = shape
    rasters = {
        "support_surface_score": np.zeros((h, w), dtype=np.float32),
        "blocker_score": np.zeros((h, w), dtype=np.float32),
        "contact_target_score": np.zeros((h, w), dtype=np.float32),
        "occlusion_edge_score": np.zeros((h, w), dtype=np.float32),
        "portal_score": np.zeros((h, w), dtype=np.float32),
        "open_air_score": np.zeros((h, w), dtype=np.float32),
        "interior_motion_score": np.zeros((h, w), dtype=np.float32),
        "uncertainty_score": np.zeros((h, w), dtype=np.float32),
        "depth_discontinuity_score": np.zeros((h, w), dtype=np.float32),
    }
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=bool)
        if sm.shape == (h, w):
            rasters["support_surface_score"][sm] = np.maximum(rasters["support_surface_score"][sm], 0.65)
    if obstacle_mask is not None:
        om = np.asarray(obstacle_mask, dtype=bool)
        if om.shape == (h, w):
            rasters["blocker_score"][om] = np.maximum(rasters["blocker_score"][om], 0.55)
    depth = np.asarray(metric_depth, dtype=np.float32) if metric_depth is not None else None
    if depth is not None and depth.shape == (h, w):
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        disc = np.clip((gx + gy) / max(1e-3, float(np.nanpercentile(gx + gy, 95))), 0.0, 1.0)
        disc[~np.isfinite(disc)] = 0.0
        rasters["depth_discontinuity_score"] = disc.astype(np.float32)
        rasters["occlusion_edge_score"] = np.maximum(rasters["occlusion_edge_score"], rasters["depth_discontinuity_score"] * 0.45)
    for ent in list((grounding_index or {}).get("entities") or []):
        if not isinstance(ent, dict):
            continue
        obj = objects_by_id.get(str(ent.get("entity_id", "")), {})
        mask = obj.get("_sam2_mask_array")
        roles = ent.get("roles") if isinstance(ent.get("roles"), dict) else {}
        actions = ent.get("actions") if isinstance(ent.get("actions"), dict) else {}
        _paint_mask(rasters["support_surface_score"], mask, max(_float(roles.get("support")), _float(actions.get("walk")), _float(actions.get("climb"))))
        _paint_mask(rasters["blocker_score"], mask, max(_float(roles.get("hard_obstacle")), _float(roles.get("soft_obstacle")) * 0.55))
        _paint_mask(rasters["contact_target_score"], mask, max(_float(roles.get("interaction_target")), _float(actions.get("hold")), _float(actions.get("touch")), _float(actions.get("sit"))))
        _paint_mask(rasters["occlusion_edge_score"], mask, max(_float(roles.get("occluder")), _float(actions.get("peek")), _float(actions.get("hide"))))
        _paint_mask(rasters["portal_score"], mask, max(_float(roles.get("portal")), _float(actions.get("enter")), _float(actions.get("exit"))))
        _paint_mask(rasters["open_air_score"], mask, max(_float(roles.get("sky_open_air")), _float(actions.get("fly")), _float(actions.get("hover"))))
        _paint_mask(rasters["interior_motion_score"], mask, max(_float(roles.get("interior")), _float(actions.get("inside")), _float(actions.get("float"))))
        if ent.get("uncertainty"):
            _paint_mask(rasters["uncertainty_score"], mask, 0.55)
        if ent.get("contradictions"):
            _paint_mask(rasters["uncertainty_score"], mask, 0.85)
    return rasters


def save_affordance_rasters(rasters: Dict[str, np.ndarray], out_path: Path) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **rasters)
    return {
        "schema": "citv_affordance_rasters_manifest_v1",
        "file": out_path.name,
        "rasters": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": round(float(np.nanmin(arr)), 5) if arr.size else 0.0,
                "max": round(float(np.nanmax(arr)), 5) if arr.size else 0.0,
                "nonzero_px": int(np.count_nonzero(arr > 0)),
            }
            for name, arr in rasters.items()
        },
    }


__all__ = ["build_affordance_rasters", "save_affordance_rasters"]
