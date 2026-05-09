"""Build final per-image paths bundle + colocated visual artifacts."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _first_number(*vals: Any) -> Optional[float]:
    for v in vals:
        try:
            x = float(v)
            return x
        except (TypeError, ValueError):
            continue
    return None


def _score_map(rows: Iterable[Dict[str, Any]], *, key_name: str = "name") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in list(rows or []):
        if not isinstance(r, dict):
            continue
        k = str(r.get(key_name, "")).strip()
        if not k:
            continue
        out[k] = _safe_float(r.get("score", 0.0), 0.0)
    return out


def _mask_mode_map(mask_row: Dict[str, Any]) -> Dict[str, float]:
    return _score_map(mask_row.get("path_modes") or [], key_name="mode")


def _infer_environment(depth_model_hint: str, scene_summary: Dict[str, Any], objects: List[Dict[str, Any]]) -> str:
    dm = str(depth_model_hint or "").lower()
    if "outdoor" in dm:
        return "outdoor"
    if "indoor" in dm:
        return "indoor"

    dom_roles = _score_map((scene_summary or {}).get("dominant_roles") or [])
    open_air = max(
        dom_roles.get("sky_open_air", 0.0),
        sum(o.get("affordance_scores", {}).get("roles", {}).get("sky_open_air", 0.0) for o in objects)
        / max(1, len(objects)),
    )
    return "outdoor" if open_air >= 0.35 else "indoor"


def _infer_has_water(scene_summary: Dict[str, Any], objects: List[Dict[str, Any]]) -> bool:
    if any(bool(o.get("is_water", False)) for o in objects):
        return True
    dom_roles = _score_map((scene_summary or {}).get("dominant_roles") or [])
    dom_actions = _score_map((scene_summary or {}).get("dominant_actions") or [])
    return max(dom_roles.get("liquid", 0.0), dom_actions.get("swim", 0.0)) >= 0.30


def _infer_people_count(objects: List[Dict[str, Any]]) -> int:
    person_tokens = {"person", "people", "human", "man", "woman", "boy", "girl", "child"}
    count = 0
    for obj in objects:
        label = str(obj.get("label", "")).lower()
        tokens = set(re.findall(r"[a-z0-9_]+", label))
        if tokens.intersection(person_tokens):
            count += 1
    return count


def _bbox_xyxy_from_any(
    bbox: Any,
    *,
    width: int,
    height: int,
    fallback_xywh: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    vals: List[float] = []
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        vals = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    elif isinstance(fallback_xywh, (list, tuple)) and len(fallback_xywh) >= 4:
        x, y, w, h = [float(fallback_xywh[0]), float(fallback_xywh[1]), float(fallback_xywh[2]), float(fallback_xywh[3])]
        vals = [x, y, x + w, y + h]
    else:
        vals = [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = vals
    # If provided as xywh, convert to xyxy.
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
    nx = [
        round((x1 / max(1.0, float(width))), 6),
        round((y1 / max(1.0, float(height))), 6),
        round((x2 / max(1.0, float(width))), 6),
        round((y2 / max(1.0, float(height))), 6),
    ]
    return px, nx


def _object_flags(roles: Dict[str, float], actions: Dict[str, float], modes: Dict[str, float]) -> Dict[str, bool]:
    walkable = max(roles.get("support", 0.0), actions.get("walk", 0.0), modes.get("on", 0.0)) >= 0.40
    climbable = max(actions.get("climb", 0.0), modes.get("above", 0.0), roles.get("soft_obstacle", 0.0)) >= 0.40
    water = max(roles.get("liquid", 0.0), actions.get("swim", 0.0), modes.get("inside", 0.0)) >= 0.35
    obstacle = max(roles.get("hard_obstacle", 0.0), roles.get("soft_obstacle", 0.0), roles.get("occluder", 0.0)) >= 0.45
    if walkable and roles.get("hard_obstacle", 0.0) < 0.55:
        obstacle = False
    return {
        "is_walkable_surface": bool(walkable),
        "is_climbable_surface": bool(climbable),
        "is_water": bool(water),
        "is_obstacle": bool(obstacle),
    }


def _build_objects(ctx: Any) -> List[Dict[str, Any]]:
    width = _safe_int(getattr(ctx, "width", 0), 0)
    height = _safe_int(getattr(ctx, "height", 0), 0)
    extra_objects = list((getattr(ctx, "extra", {}) or {}).get("objects") or [])
    extra_by_id = {str(o.get("id", "")): o for o in extra_objects if isinstance(o, dict)}

    obj_rows = list((getattr(ctx, "object_affordances", {}) or {}).get("objects") or [])
    mask_rows = list((getattr(ctx, "mask_affordances", {}) or {}).get("masks") or [])
    mask_by_obj_id = {
        str((m or {}).get("object_id") or (m or {}).get("mask_id") or ""): (m or {})
        for m in mask_rows
        if isinstance(m, dict)
    }

    out: List[Dict[str, Any]] = []
    for row in obj_rows:
        if not isinstance(row, dict):
            continue
        obj_id = str(row.get("object_id", "")).strip()
        if not obj_id:
            continue
        extra = extra_by_id.get(obj_id, {})
        mask_row = mask_by_obj_id.get(obj_id, {})

        roles = _score_map(row.get("roles") or [])
        actions = _score_map(row.get("actions") or [])
        modes = _mask_mode_map(mask_row)
        flags = _object_flags(roles, actions, modes)

        fallback_xywh = list((row.get("geometry") or {}).get("bbox_px") or [])
        bbox_px, bbox_norm = _bbox_xyxy_from_any(extra.get("bbox"), width=width, height=height, fallback_xywh=fallback_xywh)

        depth_profile = (mask_row.get("depth_profile") or {}) if isinstance(mask_row, dict) else {}
        depth_stats_obj = (extra.get("depth_stats") or {}) if isinstance(extra, dict) else {}
        mean_d = _first_number(depth_profile.get("mean_m"), depth_stats_obj.get("mean"), row.get("depth_m")) or 0.0
        min_d = _first_number(depth_profile.get("min_m"), depth_stats_obj.get("min"), mean_d) or mean_d
        max_d = _first_number(depth_profile.get("max_m"), depth_stats_obj.get("max"), mean_d) or mean_d
        med_d = _first_number(
            depth_profile.get("median_m"),
            depth_profile.get("p50_m"),
            depth_stats_obj.get("median"),
            depth_stats_obj.get("mode"),
            row.get("depth_m"),
            mean_d,
        )
        p10 = _first_number(depth_profile.get("p10_m"))
        p90 = _first_number(depth_profile.get("p90_m"))

        out.append(
            {
                "object_id": obj_id,
                "label": str(row.get("label", "")),
                "canonical_label": str(row.get("canonical_label", row.get("label", ""))),
                "bbox_xyxy_px": bbox_px,
                "bbox_xyxy_normalized": bbox_norm,
                **flags,
                "depth_stats": {
                    "unit": "m",
                    "mean_depth": round(float(mean_d), 4),
                    "median_depth": round(float(med_d), 4) if med_d is not None else None,
                    "min_depth": round(float(min_d), 4),
                    "max_depth": round(float(max_d), 4),
                    "p10_depth": round(float(p10), 4) if p10 is not None else None,
                    "p90_depth": round(float(p90), 4) if p90 is not None else None,
                },
                "affordance_scores": {
                    "roles": {k: round(float(v), 4) for k, v in roles.items()},
                    "actions": {k: round(float(v), 4) for k, v in actions.items()},
                    "path_modes": {k: round(float(v), 4) for k, v in modes.items()},
                },
                "anchors": {
                    "center_uv": list((row.get("anchors") or {}).get("center_uv") or (mask_row.get("anchors") or {}).get("center_uv") or []),
                    "approach_points": list((row.get("anchors") or {}).get("approach_points") or []),
                    "contact_points": list((row.get("anchors") or {}).get("contact_points") or []),
                    "entry_exit_points": list((row.get("anchors") or {}).get("entry_exit_points") or []),
                },
                "source_refs": {
                    "caption_object": f"scene_graph/staged/{ctx.stem}_caption_objects.json#{obj_id}",
                    "object_affordance": f"scene_graph/staged/{ctx.stem}_object_affordances.json#{obj_id}",
                    "mask_affordance": f"scene_graph/staged/{ctx.stem}_mask_affordances.json#{obj_id}",
                    "grounding_entity": f"scene_graph/staged/{ctx.stem}_paths/scene_grounding_index.json#{obj_id}",
                },
            }
        )
    out.sort(key=lambda r: str(r.get("object_id", "")))
    return out


def _path_status_counts(paths: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"accepted": 0, "plausible_uncertain": 0, "low_confidence": 0, "rejected": 0}
    for p in paths:
        st = str(p.get("acceptance_status", "")).strip()
        if st in counts:
            counts[st] += 1
    return counts


def _path_to_trajectory_id(hyp: Dict[str, Any]) -> str:
    return str(hyp.get("trajectory_id", "")).strip()


def _path_key_from_traj(hyp: Dict[str, Any]) -> str:
    ctx = hyp.get("action_context") if isinstance(hyp.get("action_context"), dict) else {}
    pid = str(ctx.get("path_id", "")).strip()
    if not pid:
        pid = str(hyp.get("continues_from_path_id", "")).strip()
    return pid


def _path_key_from_action(action_row: Dict[str, Any]) -> str:
    g = action_row.get("grounding") if isinstance(action_row.get("grounding"), dict) else {}
    return str(g.get("path_id", "")).strip()


def _copy_visuals(paths_root: Path, visuals_dir: Path) -> List[str]:
    visuals_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "path_atlas_ranked_panel_*.png",
        "path_atlas_ranked_panel_*_context.png",
        "path_atlas_ranked_panel_*_paths_trajectories.png",
        "path_trajectories.png",
        "path_trajectories_batch_*.png",
        "path_context_debug_batch_*.png",
    ]
    copied: List[str] = []
    for pattern in patterns:
        for src in sorted(paths_root.glob(pattern)):
            if not src.is_file():
                continue
            dst = visuals_dir / src.name
            try:
                shutil.copy2(src, dst)
            except Exception:
                continue
            copied.append(str(dst.name))
    return copied


def export_final_paths_bundle(
    *,
    ctx: Any,
    pipeline: Any,
    paths_root: Path,
    ranked_paths: List[Dict[str, Any]],
    traj_bundle: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Write final enriched `paths.json` + colocated atlas/trajectory assets."""
    staged_dir = paths_root.parent
    final_root = staged_dir / f"{ctx.stem}_final_paths"
    visuals_dir = final_root / "visuals"
    final_root.mkdir(parents=True, exist_ok=True)

    objects = _build_objects(ctx)
    scene_summary = (getattr(ctx, "scene_affordances", {}) or {}).get("summary") or {}
    depth_model_hint = str(getattr(getattr(pipeline, "depth_estimator", None), "backend_name", "") or "")
    status_counts = _path_status_counts(ranked_paths)

    traj_by_path: Dict[str, str] = {}
    for hyp in list((traj_bundle or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        pid = _path_key_from_traj(hyp)
        if pid and pid not in traj_by_path:
            traj_by_path[pid] = _path_to_trajectory_id(hyp)

    action_by_path: Dict[str, str] = {}
    action_rows = list((getattr(ctx, "action_hypotheses", {}) or {}).get("hypotheses") or [])
    for row in action_rows:
        if not isinstance(row, dict):
            continue
        pid = _path_key_from_action(row)
        aid = str(row.get("action_id", "")).strip()
        if pid and aid and pid not in action_by_path:
            action_by_path[pid] = aid

    per_path_dir = paths_root / "per_path"
    path_rows: List[Dict[str, Any]] = []
    for rank, p in enumerate(ranked_paths, start=1):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("path_id", "")).strip()
        if not pid:
            continue
        per_json = per_path_dir / f"{pid}.json"
        per_md = per_path_dir / f"{pid}.md"
        per_payload = _safe_load_json(per_json)
        full_rec = per_payload if per_payload else dict(p)
        scores = dict(p.get("scores") or {})
        path_rows.append(
            {
                "path_id": pid,
                "global_rank": rank,
                "path_type": str(p.get("path_type", "")),
                "manifold_type": str(p.get("manifold_type", "")),
                "action_family": str(p.get("action_family", "")),
                "acceptance_status": str(p.get("acceptance_status", "low_confidence")),
                "rejection_reasons": list(p.get("rejection_reasons") or []),
                "uncertainty_reasons": list(p.get("uncertainty_reasons") or []),
                "contradiction_reasons": list(p.get("contradiction_reasons") or []),
                "source_entity": {
                    "id": str((p.get("source_entity") or {}).get("id", "")),
                    "label": str((p.get("source_entity") or {}).get("label", "")),
                },
                "target_entity": {
                    "id": str((p.get("target_entity") or {}).get("id", "")),
                    "label": str((p.get("target_entity") or {}).get("label", "")),
                },
                "motion": {
                    "dominant_motion": str(p.get("dominant_motion", "")),
                    "recommended_motion": str((p.get("ground_object_classification") or {}).get("recommended_motion", "")),
                    "motion_hints": list(p.get("motion_hints") or []),
                    "motion_labels": list(p.get("motion_labels") or []),
                    "action_labels": list(p.get("action_labels") or []),
                },
                "scores": scores,
                "geometry": {
                    "polyline_2d": list(p.get("polyline_2d") or []),
                    "polyline_3d": list(p.get("polyline_3d") or []),
                    "display_polyline_2d": list(p.get("display_polyline_2d") or []),
                    "display_polyline_3d": list(p.get("display_polyline_3d") or []),
                    "path_shape_contract": dict(p.get("path_shape_contract") or {}),
                    "path_geometry_quality": dict(p.get("path_geometry_quality") or {}),
                    "geometry_smoothing_provenance": dict(p.get("geometry_smoothing_provenance") or {}),
                },
                "traces": {
                    "depth_trace_m": list(p.get("depth_trace_m") or []),
                    "depth_summary": dict(p.get("depth_summary") or {}),
                    "support_trace": list(p.get("support_trace") or []),
                    "semantic_trace": dict(p.get("semantic_trace") or {}),
                    "caption_trace": dict(p.get("caption_trace") or {}),
                    "visibility_profile": list(p.get("visibility_profile") or []),
                    "occlusion_trace": dict(p.get("occlusion_trace") or {}),
                    "width_profile_px": list(p.get("width_profile_px") or []),
                    "width_summary_px": dict(p.get("width_summary_px") or {}),
                },
                "grounding": {
                    "grounding_evidence": dict(p.get("grounding_evidence") or {}),
                    "ground_object_classification": dict(p.get("ground_object_classification") or {}),
                    "support_channel_means": dict(scores.get("support_channel_means") or {}),
                },
                "contracts": {
                    "trajectory_contract": dict(p.get("trajectory_contract") or {}),
                    "animation_render_contract": dict(p.get("animation_render_contract") or {}),
                },
                "links": {
                    "trajectory_id": traj_by_path.get(pid, ""),
                    "action_id": action_by_path.get(pid, ""),
                    "per_path_json": (
                        f"scene_graph/staged/{ctx.stem}_paths/per_path/{per_json.name}"
                        if per_json.exists()
                        else ""
                    ),
                    "per_path_md": (
                        f"scene_graph/staged/{ctx.stem}_paths/per_path/{per_md.name}" if per_md.exists() else ""
                    ),
                },
                "full_per_path_record": full_rec,
            }
        )

    copied_visuals = _copy_visuals(paths_root, visuals_dir)

    scene_grounding = _safe_load_json(paths_root / "scene_grounding_index.json")
    open_vocab = _safe_load_json(paths_root / "open_vocab_grounding.json")
    rasters_manifest = _safe_load_json(paths_root / "affordance_rasters_manifest.json")

    final_payload = {
        "schema": "citv_final_paths_bundle_v1",
        "version": "1.0",
        "generated_at": str(getattr(ctx, "timestamp", "")),
        "generator": {
            "pipeline": "citv_staged",
            "path_updates_doc": "docs/path_updates.md",
        },
        "images": [
            {
                "stem": str(ctx.stem),
                "scene_context": {
                    "environment": _infer_environment(depth_model_hint, scene_summary, objects),
                    "has_water": _infer_has_water(scene_summary, objects),
                    "people_count": _infer_people_count(objects),
                    "image_size": {"width": _safe_int(getattr(ctx, "width", 0), 0), "height": _safe_int(getattr(ctx, "height", 0), 0)},
                    "depth_model": depth_model_hint,
                    "dominant_actions": list(scene_summary.get("dominant_actions") or []),
                    "dominant_roles": list(scene_summary.get("dominant_roles") or []),
                    "status_counts": status_counts,
                },
                "objects": objects,
                "paths_segment": {
                    "summary": {
                        "total_paths": len(path_rows),
                        "ranking": "overall_confidence_desc",
                    },
                    "paths": path_rows,
                },
                "global_grounding": {
                    "scene_grounding_index": scene_grounding,
                    "open_vocab_grounding": open_vocab,
                    "affordance_rasters_manifest": rasters_manifest,
                },
                "artifact_refs": {
                    "path_hypotheses": f"scene_graph/staged/{ctx.stem}_paths/path_hypotheses.json",
                    "action_hypotheses": f"scene_graph/staged/{ctx.stem}_paths/action_hypotheses.json",
                    "trajectory_hypotheses": f"scene_graph/staged/{ctx.stem}_paths/trajectory_hypotheses.json",
                    "animation_components": f"scene_graph/staged/{ctx.stem}_paths/animation_components.json",
                    "animation_plan": f"scene_graph/staged/{ctx.stem}_paths/animation_plan.json",
                    "path_visual_qa": f"scene_graph/staged/{ctx.stem}_paths/path_visual_qa.json",
                },
            }
        ],
    }

    paths_json = final_root / "paths.json"
    paths_json.write_text(json.dumps(final_payload, indent=2, default=str), encoding="utf-8")

    manifest = {
        "schema": "citv_final_paths_manifest_v1",
        "stem": str(ctx.stem),
        "final_paths_json": "paths.json",
        "visuals_dir": "visuals",
        "copied_visual_files": copied_visuals,
        "source_paths_dir": str(paths_root),
    }
    manifest_json = final_root / "final_paths_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    rel_base = f"scene_graph/staged/{ctx.stem}_final_paths"
    return {
        "final_paths_dir": rel_base,
        "final_paths_json": f"{rel_base}/paths.json",
        "final_paths_manifest_json": f"{rel_base}/final_paths_manifest.json",
    }

