"""Canonical planning context for path/trajectory export stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SceneContextBundle:
    """Canonical scene context used by path/trajectory planning."""

    payload: Dict[str, Any]

    def to_json_ready(self) -> Dict[str, Any]:
        return self.payload


def build_scene_context(
    *,
    image_stem: str,
    track: str,
    img_bgr: np.ndarray,
    objects_3d_with_masks: List[Dict[str, Any]],
    regions_block: Optional[Dict[str, Any]],
    region_label_map: Optional[np.ndarray],
    region_adjacency: Optional[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]],
    metric_depth_m: Optional[np.ndarray],
    intrinsics: Optional[Dict[str, float]] = None,
) -> SceneContextBundle:
    h, w = img_bgr.shape[:2]
    obj_count = len(objects_3d_with_masks or [])
    regions = list((regions_block or {}).get("regions", []) or [])
    region_count = len(regions)
    rel_count = len(relations or [])
    depth_available = bool(metric_depth_m is not None and np.asarray(metric_depth_m).shape[:2] == (h, w))
    lm_ok = bool(region_label_map is not None and np.asarray(region_label_map).shape[:2] == (h, w))
    payload: Dict[str, Any] = {
        "schema": "citv_scene_context_v1",
        "version": "1.0",
        "image_stem": str(image_stem),
        "track": str(track),
        "image_size": {"width": int(w), "height": int(h)},
        "context_summary": {
            "objects_count": int(obj_count),
            "regions_count": int(region_count),
            "relations_count": int(rel_count),
            "depth_available": bool(depth_available),
            "region_label_map_valid": bool(lm_ok),
            "intrinsics_available": bool(isinstance(intrinsics, dict) and "fx" in intrinsics and "fy" in intrinsics),
        },
        "inputs": {
            "objects": objects_3d_with_masks or [],
            "regions_block": regions_block or {},
            "region_adjacency": region_adjacency or {},
            "relations": relations or [],
            "intrinsics": intrinsics or {},
        },
    }
    if lm_ok:
        lm = np.asarray(region_label_map, dtype=np.int32)
        payload["region_label_map_stats"] = {
            "shape": [int(h), int(w)],
            "unique_regions": int(np.unique(lm[lm > 0]).size) if lm.size else 0,
            "feasible_ratio": float((lm > 0).mean()) if lm.size else 0.0,
        }
    else:
        payload["region_label_map_stats"] = {
            "shape": [int(h), int(w)],
            "unique_regions": 0,
            "feasible_ratio": 0.0,
        }
    if depth_available:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        finite = np.isfinite(dm)
        payload["depth_stats"] = {
            "shape": [int(h), int(w)],
            "finite_ratio": float(finite.mean()) if finite.size else 0.0,
            "z_min": float(np.nanmin(dm[finite])) if finite.any() else None,
            "z_max": float(np.nanmax(dm[finite])) if finite.any() else None,
            "z_median": float(np.nanmedian(dm[finite])) if finite.any() else None,
        }
    else:
        payload["depth_stats"] = {
            "shape": [int(h), int(w)],
            "finite_ratio": 0.0,
            "z_min": None,
            "z_max": None,
            "z_median": None,
        }
    return SceneContextBundle(payload=payload)


def validate_scene_context(bundle: SceneContextBundle) -> List[str]:
    """Return validation issues (empty list means valid)."""
    p = bundle.to_json_ready()
    errs: List[str] = []
    for k in ("schema", "version", "image_stem", "track", "image_size", "context_summary", "inputs"):
        if k not in p:
            errs.append(f"missing key: {k}")
    sz = p.get("image_size", {}) if isinstance(p.get("image_size"), dict) else {}
    if not isinstance(sz.get("width"), int) or not isinstance(sz.get("height"), int):
        errs.append("image_size width/height must be integers")
    summary = p.get("context_summary", {}) if isinstance(p.get("context_summary"), dict) else {}
    for key in ("objects_count", "regions_count", "relations_count"):
        if not isinstance(summary.get(key), int):
            errs.append(f"context_summary key '{key}' must be integer")
    return errs


__all__ = ["SceneContextBundle", "build_scene_context", "validate_scene_context"]

