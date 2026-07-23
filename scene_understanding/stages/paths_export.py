"""Path-hypotheses export stage: obstacle/support/affordance rasters → fused
FMM cost field → locomotion polylines between object pairs → metric 3D lift,
kinematic tagging, support/visibility traces, and an animation-render
contract per path.

Scope note: this covers the core point-to-point locomotion manifold (the
thing everything downstream -- Phase 2's animation tiers, Phase 3's
user-authored paths -- actually depends on). Non-line manifolds (blob/
volume/portal/effect-field/contact-patch) are a separate follow-up, not
half-implemented here.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pipeline_context import PipelineContext
from ..timing import sub_timer
from scene_understanding.pathing.affordance_rasters import build_affordance_rasters
from scene_understanding.pathing.goal_anchors import (
    ground_intercept_goal_uv,
    mask_centroid_px,
    object_footprint_mask,
    pair_passes_locomotion_gates,
)
from scene_understanding.pathing.ground_plane import build_support_mask, classify_terrain_openvocab
from scene_understanding.pathing.polyline_3d import lift_polyline_2d_to_3d, smooth_polyline_in_3d
from scene_understanding.pathing.semantic_cost import (
    agent_physics_cost_layer,
    depth_noise_precision_layer,
    geometry_cost_layer_with_depth,
    precision_weighted_fuse,
    scene_graph_cost_layer,
)
from scene_understanding.pathing.semantic_fmm import k_diverse_from_T, speed_from_cost, time_of_arrival

SCHEMA_NAME = "citv_path_hypotheses_v3"
SCHEMA_VERSION = 1

_JUMP_Z_M = 0.25
_CLIMB_Z_M = 0.12
_WALK_Z_M = 0.08
_CRAWL_MIN_RUN = 6
_CRAWL_MAX_Z_VAR = _WALK_Z_M / 4.0  # 0.02


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _paths_dir(ctx: PipelineContext) -> Path:
    d = ctx.output_dir / "scene_graph" / "staged" / f"{ctx.stem}_paths"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_obstacle_mask(rasters: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    blocker = rasters.get("blocker")
    if blocker is None:
        return np.zeros((1, 1), dtype=bool)
    return blocker >= threshold


def _largest_components_mask(mask: np.ndarray, min_area_fraction: float = 0.01) -> np.ndarray:
    """Connected-component pre-filter: drop tiny disconnected feasible
    islands (segmentation noise) that would otherwise become spurious,
    unreachable-in-practice FMM goals."""
    m = np.asarray(mask, dtype=np.uint8)
    if m.sum() == 0:
        return mask.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = m.size
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] / total >= min_area_fraction:
            keep |= (labels == i)
    return keep if keep.any() else mask.astype(bool)


def _median_filter_1d(values: List[float], window: int = 3) -> List[float]:
    if window < 3 or len(values) < window:
        return list(values)
    half = window // 2
    out = list(values)
    for i in range(half, len(values) - half):
        out[i] = float(np.median(values[i - half : i + half + 1]))
    return out


def kinematic_signatures_from_3d(
    polyline_3d_smoothed: Sequence[Sequence[float]],
    jump_z_m: float = _JUMP_Z_M,
    climb_z_m: float = _CLIMB_Z_M,
    walk_z_m: float = _WALK_Z_M,
    crawl_min_run: int = _CRAWL_MIN_RUN,
) -> List[Dict[str, Any]]:
    """Z-profile motion tagger: median-filters the vertical (Y, "up") track to
    kill monocular single-pixel spikes, then segments by per-step delta into
    walk/jump/climb/descend runs, merges consecutive same-motion runs, and
    relabels long near-flat walk runs as crawl."""
    if len(polyline_3d_smoothed) < 2:
        return []
    z = [float(p[1]) for p in polyline_3d_smoothed]  # world Y ("up") drives the tag, not camera-space depth
    z = _median_filter_1d(z, window=3)

    motions: List[str] = []
    for i in range(1, len(z)):
        dz = z[i] - z[i - 1]
        if dz > jump_z_m:
            motions.append("jump")
        elif dz > climb_z_m:
            motions.append("climb")
        elif dz < -jump_z_m or dz < -climb_z_m:
            motions.append("descend")
        else:
            motions.append("walk")

    segments: List[Dict[str, Any]] = []
    start_idx = 0
    for i in range(1, len(motions) + 1):
        if i == len(motions) or motions[i] != motions[start_idx]:
            end_idx = i  # motions[j] corresponds to step (j -> j+1): segment covers vertices start_idx..i
            dz_total = z[i] - z[start_idx] if i < len(z) else z[-1] - z[start_idx]
            segments.append({"start_idx": start_idx, "end_idx": end_idx, "motion": motions[start_idx], "dz_m": round(dz_total, 4)})
            start_idx = i

    for seg in segments:
        run_len = seg["end_idx"] - seg["start_idx"]
        if seg["motion"] == "walk" and run_len >= crawl_min_run:
            window = z[seg["start_idx"] : seg["end_idx"] + 1]
            if len(window) >= 2 and float(np.var(window)) < _CRAWL_MAX_Z_VAR:
                seg["motion"] = "crawl"
    return segments


# swimmable: only override walk/crawl -- jump/climb/descend near water can
# still be a legitimate real event (e.g. jumping in), so those are left alone.
# open_air: override ALL ground-based motions, not just walk/crawl. There is
# no ground surface at all for jump/climb/descend to mean anything over open
# air -- a "descend" segment there is just flying while losing altitude, not
# a stair-descent (confirmed via a real Tier 1 render: a bird's flight path
# stayed stuck on "descend" for its entire duration because the original
# walk/crawl-only override never touched it).
_SWIM_OVERRIDE_MOTIONS = {"walk", "crawl"}
_FLY_OVERRIDE_MOTIONS = {"walk", "crawl", "jump", "climb", "descend"}


def _apply_terrain_aware_motion(
    kinematic_sigs: List[Dict[str, Any]],
    support_trace: List[Dict[str, Any]],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Cross-references each segment's own ``support_trace`` region-type text
    (already computed per polyline sample, same index space -- see
    ``_support_trace``/``smooth_polyline_in_3d``, which never changes point
    count) against the open-vocab ``classify_terrain_openvocab`` to add
    swim/fly awareness the pure Z-profile tagger above can't see (a path
    across a pool or a gap still reads as flat "walk" from vertical delta
    alone)."""
    for seg in kinematic_sigs:
        region_types = [
            support_trace[i]["region_type"]
            for i in range(seg["start_idx"], min(seg["end_idx"] + 1, len(support_trace)))
            if i < len(support_trace)
        ]
        votes = [v for v in (classify_terrain_openvocab(t, device=device) for t in region_types) if v is not None]
        terrain_type = Counter(votes).most_common(1)[0][0] if votes else "unknown"
        seg["terrain_type"] = terrain_type
        if terrain_type == "swimmable" and seg["motion"] in _SWIM_OVERRIDE_MOTIONS:
            seg["motion"] = "swim"
        elif terrain_type == "open_air" and seg["motion"] in _FLY_OVERRIDE_MOTIONS:
            seg["motion"] = "fly"
    return kinematic_sigs


def _width_profile(
    polyline_uvz: Sequence[Sequence[float]],
    intrinsics: Dict[str, float],
    actor_width_m: float,
    h: int,
    w: int,
) -> List[float]:
    fx = float(intrinsics.get("fx", w))
    widths: List[float] = []
    lo = 3.0
    hi = max(12.0, 0.20 * min(h, w))
    for p in polyline_uvz:
        z = float(p[2]) if len(p) > 2 else -1.0
        px = fx * actor_width_m / z if z > 1e-3 else hi
        widths.append(float(np.clip(px, lo, hi)))
    return widths


def _support_trace(
    polyline_2d: Sequence[Sequence[float]],
    depth: Optional[np.ndarray],
    region_label_map: Optional[np.ndarray],
    regions_meta: List[Dict[str, Any]],
    objects: Sequence[Dict[str, Any]],
    obstacle_mask: np.ndarray,
    h: int,
    w: int,
) -> List[Dict[str, Any]]:
    region_by_id = {r.get("region_id", r.get("id")): r for r in regions_meta}
    centroids = [(o.get("id"), mask_centroid_px(o, h, w)) for o in objects]
    trace: List[Dict[str, Any]] = []
    for u, v in polyline_2d:
        xi, yi = int(round(u)), int(round(v))
        xi, yi = max(0, min(w - 1, xi)), max(0, min(h - 1, yi))
        z = float(depth[yi, xi]) if depth is not None and np.isfinite(depth[yi, xi]) else None
        region_id = int(region_label_map[yi, xi]) if region_label_map is not None else None
        region = region_by_id.get(region_id, {}) if region_id else {}
        region_type = str(region.get("region_type") or region.get("semantic_label") or "unknown")
        support_kind = "blocking" if obstacle_mask[yi, xi] else "supporting"
        nearby = sorted(
            oid for oid, c in centroids if c and (c[0] - xi) ** 2 + (c[1] - yi) ** 2 < (0.08 * max(h, w)) ** 2
        )[:5]
        trace.append({
            "u": xi, "v": yi, "z_m": z,
            "region_id": region_id, "region_type": region_type,
            "support_kind": support_kind, "nearby_object_ids": nearby,
        })
    return trace


def _visibility_profile(
    polyline_2d: Sequence[Sequence[float]],
    polyline_uvz: Sequence[Sequence[float]],
    objects: Sequence[Dict[str, Any]],
    src_id: Any,
    tgt_id: Any,
    h: int,
    w: int,
) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
    occluder_objs = [o for o in objects if o.get("id") not in (src_id, tgt_id)]
    occluder_masks = []
    for o in occluder_objs:
        m = object_footprint_mask(o, h, w)
        if m is None:
            continue
        occ_z = o.get("depth_stats", {}).get("z_val")
        occluder_masks.append((o.get("id"), m, occ_z))

    profile: List[Dict[str, Any]] = []
    alpha_profile: List[float] = []
    layers_seen: List[str] = []
    for (u, v), uvz in zip(polyline_2d, polyline_uvz):
        xi, yi = int(round(u)), int(round(v))
        xi, yi = max(0, min(w - 1, xi)), max(0, min(h - 1, yi))
        z_path = float(uvz[2]) if len(uvz) > 2 and uvz[2] > 0 else None
        # mask_hits: any object whose real mask covers this pixel, regardless
        # of depth. depth_hits: the subset actually closer to the camera
        # than the path point there -- a real occluder, not just a mask
        # that happens to share this pixel column (e.g. trees/buildings
        # under a path that's actually flying in front of/above them). A
        # mask overlap with no depth evidence on either side is left as
        # "partially_occluded" rather than assumed fully occluded.
        mask_hits = [oid for oid, mask, _occ_z in occluder_masks if mask[yi, xi]]
        depth_hits = [
            oid for oid, mask, occ_z in occluder_masks
            if mask[yi, xi] and occ_z is not None and z_path is not None and occ_z < z_path
        ]
        if depth_hits:
            layer, visible_fraction, hits = "behind_object", 0.15, depth_hits
        elif mask_hits:
            layer, visible_fraction, hits = "partially_occluded", 0.55, mask_hits
        else:
            layer, visible_fraction, hits = "in_front", 1.0, []
        layers_seen.append(layer)
        alpha_profile.append(visible_fraction)
        profile.append({
            "u": xi, "v": yi, "z_m": z_path,
            "visible_fraction": visible_fraction, "render_layer": layer, "occluder_ids": hits,
        })
    return profile, alpha_profile, sorted(set(layers_seen))


def _sample_indices(n: int, max_samples: int = 12) -> List[int]:
    if n <= max_samples:
        return list(range(n))
    return sorted(set(int(round(i)) for i in np.linspace(0, n - 1, max_samples)))


def _animation_render_contract(
    polyline_2d: Sequence[Sequence[float]],
    width_profile: List[float],
    visibility_profile: List[Dict[str, Any]],
    alpha_profile: List[float],
    kinematic_sigs: List[Dict[str, Any]],
    render_layers: List[str],
    occluder_ids: List[Any],
) -> Dict[str, Any]:
    idxs = _sample_indices(len(polyline_2d))
    preview: List[Dict[str, Any]] = []
    motion_by_idx: Dict[int, str] = {}
    for seg in kinematic_sigs:
        for i in range(seg["start_idx"], seg["end_idx"] + 1):
            motion_by_idx[i] = seg["motion"]

    for k, i in enumerate(idxs):
        u, v = polyline_2d[i]
        nxt = polyline_2d[idxs[k + 1]] if k + 1 < len(idxs) else polyline_2d[i]
        heading = float(np.degrees(np.arctan2(nxt[1] - v, nxt[0] - u))) if (nxt[0], nxt[1]) != (u, v) else 0.0
        width_px = width_profile[i] if i < len(width_profile) else (width_profile[-1] if width_profile else 12.0)
        vis = visibility_profile[i] if i < len(visibility_profile) else {"visible_fraction": 1.0, "render_layer": "in_front", "occluder_ids": [], "z_m": None}
        preview.append({
            "s": round(i / max(1, len(polyline_2d) - 1), 4),
            "position_px": [round(u, 1), round(v, 1)],
            "depth_m": vis.get("z_m"),
            "heading_deg": round(heading, 1),
            "speed_hint": 1.0,
            "scale_hint": round(float(np.clip(width_px / 12.0, 0.25, 2.5)), 3),
            "width_px": round(width_px, 2),
            "alpha": round(float(np.clip(alpha_profile[i] if i < len(alpha_profile) else 1.0, 0.15, 1.0)), 3),
            "visible_fraction": vis.get("visible_fraction", 1.0),
            "render_layer": vis.get("render_layer", "in_front"),
            "motion_label": motion_by_idx.get(i, "walk"),
            "occluder_ids": vis.get("occluder_ids", []),
        })
    return {
        "schema": "citv_animation_render_contract_v1",
        "render_primitive": "depth_tapered_corridor_plus_actor",
        "motion_labels": sorted({seg["motion"] for seg in kinematic_sigs}) or ["walk"],
        "alpha_policy": "visibility_profile",
        "width_policy": "depth_width_profile",
        "depth_scale_policy": "metric_depth_trace",
        "render_layers": render_layers,
        "occluders": sorted({oid for oid in occluder_ids if oid is not None}),
        "sample_state_preview": preview,
    }


def _manifold_acceptance(overall_confidence: float, min_confidence: float) -> Dict[str, Any]:
    accepted_conf = 0.48
    plausible_conf = min(min_confidence, 0.54)
    if overall_confidence >= accepted_conf:
        status = "accepted"
    elif overall_confidence >= plausible_conf:
        status = "low_confidence"
    else:
        status = "rejected"
    return {"acceptance_status": status, "accepted_confidence_threshold": accepted_conf, "overall_confidence": round(overall_confidence, 3)}


def build_hypothesis(
    path_id: str,
    src: Dict[str, Any],
    tgt: Dict[str, Any],
    raw_path_px: List[Tuple[int, int]],
    ctx: PipelineContext,
    cfg: Any,
    obstacle_mask: np.ndarray,
    support_mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if len(raw_path_px) < 2:
        return None
    polyline_2d = [[float(x), float(y)] for x, y in raw_path_px]
    depth = ctx.metric_depth

    polyline_uvz, snapped = lift_polyline_2d_to_3d(polyline_2d, depth, support_mask=support_mask)
    smoothed_xyz, display_polyline_2d = smooth_polyline_in_3d(polyline_uvz, ctx.intrinsics, smoothing_window=5)

    kinematic_sigs = kinematic_signatures_from_3d(smoothed_xyz)
    actor_width_m = float(_cfg_get(cfg, "path_actor_width_m", 0.55))
    width_profile = _width_profile(polyline_uvz, ctx.intrinsics, actor_width_m, ctx.height, ctx.width)
    support_trace = _support_trace(
        polyline_2d, depth, ctx.region_label_map, ctx.region_partition_meta,
        ctx.extra.get("objects", []), obstacle_mask, ctx.height, ctx.width,
    )
    kinematic_sigs = _apply_terrain_aware_motion(
        kinematic_sigs, support_trace, device=str(_cfg_get(cfg, "path_terrain_classify_device", "cpu")),
    )
    visibility_profile, alpha_profile, render_layers = _visibility_profile(
        polyline_2d, polyline_uvz, ctx.extra.get("objects", []), src.get("id"), tgt.get("id"),
        ctx.height, ctx.width,
    )
    occluder_ids = [oid for v in visibility_profile for oid in v.get("occluder_ids", [])]
    render_contract = _animation_render_contract(
        polyline_2d, width_profile, visibility_profile, alpha_profile, kinematic_sigs, render_layers, occluder_ids,
    )

    valid_depth = sum(1 for p in polyline_uvz if p[2] > 0)
    depth_validity = valid_depth / max(1, len(polyline_uvz))
    max_turn_deg = 0.0
    for i in range(1, len(polyline_2d) - 1):
        a = np.array(polyline_2d[i]) - np.array(polyline_2d[i - 1])
        b = np.array(polyline_2d[i + 1]) - np.array(polyline_2d[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        cos_t = float(np.clip(a @ b / (na * nb), -1.0, 1.0))
        max_turn_deg = max(max_turn_deg, float(np.degrees(np.arccos(cos_t))))
    overall_confidence = float(np.clip(0.5 * depth_validity + 0.5 * (1.0 - min(1.0, max_turn_deg / 180.0)), 0.0, 1.0))
    contract_status = _manifold_acceptance(overall_confidence, float(_cfg_get(cfg, "path_min_confidence", 0.55)))

    return {
        "path_id": path_id,
        "source_id": src.get("id"),
        "target_id": tgt.get("id"),
        "manifold_type": "line_path",
        "path_shape_contract": {"shape_type": "polyline", "geometry_refs": {"centerline": "polyline_2d"}},
        "polyline_2d": polyline_2d,
        "polyline_2d_raw": polyline_2d,
        "display_polyline_2d": display_polyline_2d,
        "polyline_3d": polyline_uvz,
        "polyline_3d_smoothed": smoothed_xyz,
        "display_polyline_3d": smoothed_xyz,
        "polyline_2d_support_snapped": snapped,
        "width_profile_px": width_profile,
        "support_trace": support_trace,
        "visibility_profile": visibility_profile,
        "alpha_profile": alpha_profile,
        "occlusion_trace": {
            "mean_visible_fraction": round(float(np.mean(alpha_profile)) if alpha_profile else 1.0, 3),
            "occluded_sample_fraction": round(float(np.mean([1.0 if a < 0.5 else 0.0 for a in alpha_profile])) if alpha_profile else 0.0, 3),
            "occluder_ids": sorted({oid for oid in occluder_ids if oid is not None}),
        },
        "kinematic_signatures": kinematic_sigs,
        "animation_render_contract": render_contract,
        "depth_trace_m": [
            {"s": round(i / max(1, len(polyline_uvz) - 1), 4), "u": p[0], "v": p[1], "z_m": p[2]}
            for i, p in enumerate(polyline_uvz)
        ],
        "max_turn_deg": round(max_turn_deg, 2),
        "depth_validity_fraction": round(depth_validity, 3),
        "overall_confidence": overall_confidence,
        "contract_status": contract_status,
    }


def _candidate_pairs(
    objects: List[Dict[str, Any]], relations: List[Dict[str, Any]], cfg: Any, h: int, w: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    max_candidates = int(_cfg_get(cfg, "path_max_candidates", 500))
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, src in enumerate(objects):
        for tgt in objects[i + 1 :]:
            if pair_passes_locomotion_gates(src, tgt, relations, cfg, h, w):
                pairs.append((src, tgt))
            if len(pairs) >= max_candidates:
                return pairs
    return pairs


def build_scene_cost_field(pipeline: Any, ctx: PipelineContext) -> Dict[str, Any]:
    """Support/obstacle/affordance rasters -> fused FMM cost+precision+speed
    field for this image. Split out of ``run()`` so Phase 3's custom (user
    start/end point, drawn-path) geometry can reuse the exact same fields
    the auto-generated hypotheses were built from, rather than recomputing
    (and potentially drifting from) them per custom request."""
    cfg = getattr(pipeline, "config", None)
    objects: List[Dict[str, Any]] = ctx.extra.get("objects", [])
    h, w = ctx.height, ctx.width
    depth = ctx.metric_depth
    intrinsics = ctx.intrinsics or {}
    device = str(getattr(getattr(pipeline, "depth_estimator", None), "device", "cpu"))

    # Rasters need a first-cut obstacle mask; objects' own footprint masks
    # union'd is a reasonable seed before the raster-derived one refines it.
    seed_obstacle = np.zeros((h, w), dtype=bool)
    for obj in objects:
        m = object_footprint_mask(obj, h, w)
        if m is not None:
            seed_obstacle |= m

    with sub_timer("paths_export.support_mask"):
        support_mask, support_info = build_support_mask(
            depth, intrinsics, object_mask=seed_obstacle,
            region_label_map=ctx.region_label_map,
            regions_meta={"regions": ctx.region_partition_meta},
            device=device,
        )
    with sub_timer("paths_export.affordance_rasters"):
        rasters, raster_meta = build_affordance_rasters(objects, support_mask, seed_obstacle, depth, (h, w), device=device)
        obstacle_mask = _build_obstacle_mask(rasters)
        feasible = _largest_components_mask(~obstacle_mask, min_area_fraction=0.01)

    with sub_timer("paths_export.cost_layers_fuse"):
        geo_layer = geometry_cost_layer_with_depth(ctx.img_bgr, obstacle_mask, depth)
        sg_layer = scene_graph_cost_layer(rasters)
        phys_layer = agent_physics_cost_layer(
            obstacle_mask, depth, intrinsics,
            agent_half_width_m=float(_cfg_get(cfg, "path_actor_width_m", 0.55)) / 2.0,
        )
        noise_layer = depth_noise_precision_layer(depth)
        cost, precision = precision_weighted_fuse([geo_layer, sg_layer, phys_layer, noise_layer])
        cost = np.where(feasible, cost, 1.0)
        speed = speed_from_cost(cost, precision, speed_floor=float(_cfg_get(cfg, "trav_speed_floor", 0.06)))

    return {
        "support_mask": support_mask, "support_info": support_info,
        "rasters": rasters, "raster_meta": raster_meta,
        "obstacle_mask": obstacle_mask, "feasible": feasible,
        "cost": cost, "precision": precision, "speed": speed,
    }


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    cfg = getattr(pipeline, "config", None)
    objects: List[Dict[str, Any]] = ctx.extra.get("objects", [])
    if not objects or ctx.metric_depth is None:
        return ctx

    h, w = ctx.height, ctx.width
    field = build_scene_cost_field(pipeline, ctx)
    support_mask, raster_meta, support_info = field["support_mask"], field["raster_meta"], field["support_info"]
    obstacle_mask, feasible, speed = field["obstacle_mask"], field["feasible"], field["speed"]

    # Stashed for reuse by Phase 3's custom-geometry grounding (user-drawn
    # paths / custom start-end points) so it doesn't recompute -- and can't
    # drift from -- the same cost/support fields the auto hypotheses used.
    ctx.extra["_paths_support_mask"] = support_mask
    ctx.extra["_paths_obstacle_mask"] = obstacle_mask
    ctx.extra["_paths_feasible_mask"] = feasible
    ctx.extra["_paths_speed_field"] = speed

    with sub_timer("paths_export.candidate_pairs"):
        pairs = _candidate_pairs(objects, ctx.relations, cfg, h, w)
    edge_penalty = float(_cfg_get(cfg, "path_geodesic_edge_penalty", 0.35))
    top_k_per_pair = int(_cfg_get(cfg, "path_top_k_per_pair", 3))

    anchor_cache: Dict[Any, Optional[Tuple[int, int]]] = {}

    def _anchor(obj: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        oid = obj.get("id")
        if oid in anchor_cache:
            return anchor_cache[oid]
        uv = ground_intercept_goal_uv(obj, support_mask, feasible, h, w, region_label_map=ctx.region_label_map)
        if uv is None:
            c = mask_centroid_px(obj, h, w)
            uv = (int(round(c[0])), int(round(c[1]))) if c else None
        anchor_cache[oid] = uv
        return uv

    T_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    hypotheses: List[Dict[str, Any]] = []
    fmm_solve_s = 0.0
    hypothesis_build_s = 0.0
    t_loop0 = time.time()
    for src, tgt in pairs:
        start = _anchor(src)
        goal = _anchor(tgt)
        if start is None or goal is None or start == goal:
            continue
        if goal not in T_cache:
            t0 = time.time()
            T_cache[goal] = time_of_arrival(speed, goal)
            fmm_solve_s += time.time() - t0
        T = T_cache[goal]
        if T is None:
            continue
        alt_paths = k_diverse_from_T(T, start, k=max(1, top_k_per_pair), edge_penalty=edge_penalty)
        for k, raw_path in enumerate(alt_paths[:top_k_per_pair]):
            path_id = f"staged_fmm_{src.get('id')}_to_{tgt.get('id')}_k{k:02d}"
            t0 = time.time()
            hyp = build_hypothesis(path_id, src, tgt, raw_path, ctx, cfg, obstacle_mask, support_mask)
            hypothesis_build_s += time.time() - t0
            if hyp is not None:
                hypotheses.append(hyp)
    print(f"    [Timing]   paths_export.fmm_and_hypotheses_loop: {time.time() - t_loop0:.2f}s total over {len(pairs)} pairs "
          f"(fmm_solve: {fmm_solve_s:.2f}s over {len(T_cache)} unique goals, hypothesis_build: {hypothesis_build_s:.2f}s over {len(hypotheses)} hypotheses)")

    payload = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "stem": ctx.stem,
        "hypotheses": hypotheses,
        "additive_fields": [
            "polyline_3d", "polyline_3d_smoothed", "support_trace", "visibility_profile",
            "alpha_profile", "kinematic_signatures", "animation_render_contract",
        ],
        "compat_note": "additive_fields are enrichments; polyline_2d/polyline_2d_raw are the stable contract older consumers can rely on.",
        "obstacle_and_support_meta": {**raster_meta, **support_info, "feasible_ratio": float(feasible.mean())},
    }
    out_path = _paths_dir(ctx) / "path_hypotheses.json"
    out_path.write_text(json.dumps(payload, indent=2))
    ctx.path_exports["path_hypotheses_json"] = str(out_path)
    ctx.path_exports["path_hypotheses_count"] = len(hypotheses)
    print(f"  [PathsExport] wrote {len(hypotheses)} path hypotheses -> {out_path.name}")
    return ctx
