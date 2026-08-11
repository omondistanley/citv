"""Scene adapter for user-authored motion contracts.

This module is intentionally conservative: it preserves user geometry verbatim,
then adds scene-aware traces and validation hints. It does not try to become a
full renderer or physics engine. Renderers can consume the returned
GroundedMotionContract to scale actors by depth, mask them by occluders, bend
preview paths inside the user corridor, and explain all adaptations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scene_understanding.action_contracts.contracts import (
    GroundedMotionContract,
    MotionContract,
    SceneAdaptationReport,
    infer_manifold_type,
)

Point2D = Tuple[float, float]


def adapt_motion_contract_to_scene(
    contract: MotionContract,
    *,
    scene_graph: Optional[Dict[str, Any]] = None,
    metric_depth_m: Optional[np.ndarray] = None,
    region_label_map: Optional[np.ndarray] = None,
    object_masks: Optional[Dict[str, np.ndarray]] = None,
    sample_count: int = 48,
) -> GroundedMotionContract:
    """Enrich a user-authored contract with scene evidence.

    The returned contract keeps ``contract.user_geometry`` unchanged. Any derived
    path is written under ``grounded_geometry`` and the report says what was
    preserved/adapted/warned.
    """

    manifold = contract.manifold_type or infer_manifold_type(contract.action_text, contract.user_geometry)
    user_points = list(contract.user_geometry.points)
    sampled = _resample_points(user_points, sample_count) if len(user_points) > 1 else user_points
    depth_trace = _sample_depth(metric_depth_m, sampled)
    region_trace = _sample_regions(region_label_map, sampled)
    nearest = _nearest_scene_entities(scene_graph or {}, sampled)
    occlusion = _estimate_visibility(contract, sampled, depth_trace, object_masks or {}, scene_graph or {})
    support_trace = _support_trace_from_regions(scene_graph or {}, region_trace)
    warnings: List[str] = []
    assumptions: List[str] = []
    adapted: List[str] = []
    preserved = [
        "raw_user_geometry",
        "actor_text_or_asset",
        "action_text",
        "duration_s",
    ]

    if metric_depth_m is None:
        warnings.append("metric_depth_m unavailable; depth-aware scale and occlusion are approximate")
    else:
        adapted.append("sampled metric depth along user geometry")
    if region_label_map is None:
        warnings.append("region_label_map unavailable; support trace is unknown")
    else:
        adapted.append("sampled region/support trace along user geometry")
    if not object_masks:
        warnings.append("object masks unavailable; foreground occlusion is approximate")
    else:
        adapted.append("estimated visibility against object masks")
    if contract.source != "user_authored":
        warnings.append("contract is not marked user_authored; UI should visually separate suggestions from authored takes")

    invalid_depth_ratio = _invalid_ratio(depth_trace)
    support_unknown_ratio = _unknown_ratio(support_trace)
    visibility_min = min(occlusion.get("visibility_profile", [1.0]) or [1.0])
    if invalid_depth_ratio > 0.5:
        warnings.append("more than half of sampled points have missing/invalid depth")
    if support_unknown_ratio > 0.5 and manifold in {"centerline_path", "ribbon_path", "contact_patch"}:
        warnings.append("support surface is uncertain for a surface/contact action")
    if visibility_min < 0.25:
        warnings.append("actor may become mostly hidden by foreground scene geometry")

    if contract.policy.allow_path_bending:
        adapted.append("created adaptation corridor around user geometry for obstacle-aware preview")
    else:
        assumptions.append("path bending disabled; renderer should follow user points exactly")
    if contract.policy.required_object_ids:
        adapted.append("tracked required interaction object ids")
    if contract.policy.avoid_object_ids or contract.policy.avoid_region_ids:
        adapted.append("tracked avoid constraints for renderer/planner")

    status = "accepted_with_warnings" if warnings else "accepted"
    scores = {
        "user_geometry_preservation": 1.0,
        "depth_coverage": round(1.0 - invalid_depth_ratio, 3),
        "support_known_ratio": round(1.0 - support_unknown_ratio, 3),
        "min_visibility": round(float(visibility_min), 3),
    }
    grounded_geometry = {
        "manifold_type": manifold,
        "user_polyline_2d": _json_points(user_points),
        "adapted_polyline_2d": _json_points(sampled),
        "corridor_radius_px": contract.user_geometry.corridor_radius_px,
        "path_preservation_policy": {
            "preserve_user_geometry": contract.policy.preserve_user_geometry,
            "allow_path_bending": contract.policy.allow_path_bending,
            "max_path_deviation_px": contract.policy.max_path_deviation_px,
        },
    }
    if depth_trace:
        grounded_geometry["polyline_3d_uvz"] = [
            [float(x), float(y), float(z)] if z is not None else [float(x), float(y), None]
            for (x, y), z in zip(sampled, depth_trace)
        ]

    traces = {
        "depth_trace_m": depth_trace,
        "region_trace": region_trace,
        "support_trace": support_trace,
        "visibility_profile": occlusion.get("visibility_profile", []),
        "occluder_ids": occlusion.get("occluder_ids", []),
        "semantic_trace": _semantic_trace(contract.action_text, support_trace, region_trace),
    }
    rendering = {
        "render_layers": occlusion.get("render_layers", []),
        "alpha_profile": occlusion.get("visibility_profile", []),
        "depth_scale_hint": _depth_scale_hint(depth_trace),
        "asset_policy": {
            "actor_source": contract.actor.actor_source,
            "visual_style": contract.actor.visual_style,
            "asset_ref": contract.actor.asset_ref,
            "scene_object_id": contract.actor.scene_object_id,
            "no_hard_coded_actor_fallback": True,
        },
    }
    report = SceneAdaptationReport(
        status=status,  # type: ignore[arg-type]
        preserved=preserved,
        adapted=adapted,
        warnings=warnings,
        assumptions=assumptions,
        scores=scores,
    )
    return GroundedMotionContract(
        contract=contract,
        grounded_geometry=grounded_geometry,
        traces=traces,
        rendering=rendering,
        report=report,
        nearest_scene_entities=nearest,
        alternatives=[],
    )


def _resample_points(points: Sequence[Point2D], sample_count: int) -> List[Point2D]:
    if len(points) <= 1:
        return list(points)
    sample_count = max(2, int(sample_count))
    dists = [0.0]
    for a, b in zip(points, points[1:]):
        dists.append(dists[-1] + math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
    total = dists[-1]
    if total <= 1e-6:
        return [points[0] for _ in range(sample_count)]
    out: List[Point2D] = []
    seg = 0
    for i in range(sample_count):
        target = total * i / (sample_count - 1)
        while seg < len(dists) - 2 and dists[seg + 1] < target:
            seg += 1
        a = points[seg]
        b = points[seg + 1]
        denom = max(1e-6, dists[seg + 1] - dists[seg])
        t = (target - dists[seg]) / denom
        out.append((float(a[0]) + t * (float(b[0]) - float(a[0])), float(a[1]) + t * (float(b[1]) - float(a[1]))))
    return out


def _sample_depth(depth: Optional[np.ndarray], points: Sequence[Point2D]) -> List[Optional[float]]:
    if depth is None:
        return [None for _ in points]
    arr = np.asarray(depth, dtype=np.float32)
    h, w = arr.shape[:2]
    vals: List[Optional[float]] = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            vals.append(None)
            continue
        z = float(arr[yi, xi])
        vals.append(z if np.isfinite(z) and z > 0 else None)
    return vals


def _sample_regions(region_label_map: Optional[np.ndarray], points: Sequence[Point2D]) -> List[Optional[int]]:
    if region_label_map is None:
        return [None for _ in points]
    lm = np.asarray(region_label_map)
    h, w = lm.shape[:2]
    vals: List[Optional[int]] = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            vals.append(None)
        else:
            vals.append(int(lm[yi, xi]))
    return vals


def _support_trace_from_regions(scene_graph: Dict[str, Any], region_trace: Sequence[Optional[int]]) -> List[str]:
    regions = scene_graph.get("regions") or {}
    region_list = regions.get("regions") if isinstance(regions, dict) else regions
    by_index: Dict[int, str] = {}
    if isinstance(region_list, list):
        for r in region_list:
            if not isinstance(r, dict):
                continue
            idx = r.get("region_index") or r.get("index")
            label = r.get("semantic_label") or r.get("label") or r.get("id") or "region"
            try:
                by_index[int(idx)] = str(label)
            except Exception:
                pass
    return [by_index.get(int(v), "unknown") if v is not None else "unknown" for v in region_trace]


def _estimate_visibility(
    contract: MotionContract,
    points: Sequence[Point2D],
    depth_trace: Sequence[Optional[float]],
    object_masks: Dict[str, np.ndarray],
    scene_graph: Dict[str, Any],
) -> Dict[str, Any]:
    profile: List[float] = []
    occluders: List[List[str]] = []
    layers: List[str] = []
    for (x, y), _z in zip(points, depth_trace):
        xi, yi = int(round(x)), int(round(y))
        hit_ids: List[str] = []
        for oid, mask in object_masks.items():
            arr = np.asarray(mask)
            if arr.ndim < 2:
                continue
            h, w = arr.shape[:2]
            if 0 <= xi < w and 0 <= yi < h and bool(arr[yi, xi]):
                hit_ids.append(str(oid))
        forced = [oid for oid in contract.policy.must_render_behind_object_ids if oid in hit_ids]
        if forced:
            vis = 0.18
            layer = "behind_object"
        elif hit_ids:
            vis = 0.55
            layer = "partially_occluded"
        else:
            vis = 1.0
            layer = "in_front"
        profile.append(vis)
        occluders.append(hit_ids)
        layers.append(layer)
    return {"visibility_profile": profile, "occluder_ids": occluders, "render_layers": layers}


def _nearest_scene_entities(scene_graph: Dict[str, Any], points: Sequence[Point2D]) -> Dict[str, Any]:
    objects = scene_graph.get("objects") or []
    if not points or not isinstance(objects, list):
        return {"objects": []}
    anchors = [points[0], points[-1]] if len(points) > 1 else [points[0]]
    found: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        c = obj.get("mask_centroid_2d") or obj.get("centroid_2d")
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        d = min(math.hypot(float(c[0]) - a[0], float(c[1]) - a[1]) for a in anchors)
        found.append({"id": obj.get("id"), "label": obj.get("canonical_name") or obj.get("label"), "distance_px": round(d, 2)})
    found.sort(key=lambda x: x["distance_px"])
    return {"objects": found[:8]}


def _semantic_trace(action_text: str, support_trace: Sequence[str], region_trace: Sequence[Optional[int]]) -> List[str]:
    text = (action_text or "").lower()
    out: List[str] = []
    for support, region in zip(support_trace, region_trace):
        if support != "unknown":
            out.append(f"{support}:action_context")
        elif "fly" in text or "glide" in text:
            out.append("open_volume_uncertain")
        elif "peek" in text or "hide" in text:
            out.append("occlusion_edge_uncertain")
        else:
            out.append("semantic_context_unknown")
    return out


def _depth_scale_hint(depth_trace: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid = [z for z in depth_trace if z is not None and z > 0]
    if not valid:
        return [None for _ in depth_trace]
    ref = valid[0]
    return [round(ref / z, 4) if z is not None and z > 0 else None for z in depth_trace]


def _invalid_ratio(values: Sequence[Optional[float]]) -> float:
    if not values:
        return 1.0
    bad = sum(1 for v in values if v is None or not np.isfinite(float(v)) or float(v) <= 0)
    return bad / len(values)


def _unknown_ratio(values: Sequence[str]) -> float:
    if not values:
        return 1.0
    return sum(1 for v in values if not v or v == "unknown") / len(values)


def _json_points(points: Iterable[Point2D]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]
