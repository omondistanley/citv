"""Human- and machine-readable QA bundles for batched path + trajectory scene overlays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _actions_by_path_id(action_hypotheses: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    action_rows = (
        (action_hypotheses or {}).get("hypotheses")
        or (action_hypotheses or {}).get("actions")
        or []
    )
    for action in list(action_rows):
        if not isinstance(action, dict):
            continue
        grounding = action.get("grounding") or {}
        pid = str(grounding.get("path_id", ""))
        if pid and pid not in out:
            out[pid] = action
    return out


def _traj_by_path_id(traj_bundle: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for hyp in list((traj_bundle or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        pid = str((hyp.get("action_context") or {}).get("path_id", ""))
        if not pid:
            pid = str(hyp.get("continues_from_path_id", ""))
        if pid and pid not in out:
            out[pid] = hyp
    return out


def _vertex_count(path: Dict[str, Any]) -> int:
    for key in ("polyline_2d_reprojected", "polyline_2d", "polyline_geodesic_2d"):
        raw = path.get(key)
        if isinstance(raw, list):
            return len(raw)
    return 0


def _polyline_endpoints(path: Dict[str, Any]) -> Dict[str, Any]:
    raw = path.get("polyline_2d_reprojected") or path.get("polyline_2d") or []
    if not isinstance(raw, list) or len(raw) < 2:
        return {"vertex_count": _vertex_count(path)}
    def _pair(xy: Any) -> Optional[List[float]]:
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            return None
        try:
            return [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        except (TypeError, ValueError):
            return None
    a, b = _pair(raw[0]), _pair(raw[-1])
    return {
        "vertex_count": len(raw),
        "start_xy": a,
        "end_xy": b,
    }


def _first_motion_label(path: Dict[str, Any]) -> str:
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict) and str(hint.get("motion", "")).strip():
            return str(hint.get("motion", "")).strip()
        if isinstance(hint, str) and hint.strip():
            return hint.strip()
    contract = path.get("trajectory_contract") if isinstance(path.get("trajectory_contract"), dict) else {}
    return str(contract.get("dominant_motion", "")).strip()


def _depth_summary(path: Dict[str, Any]) -> Dict[str, Any]:
    for row in list(path.get("depth_trace_m") or []):
        if isinstance(row, dict) and row.get("summary"):
            return dict(row)
    return {}


def _width_summary(path: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in list(path.get("width_profile_px") or []) if isinstance(r, dict)]
    vals: List[float] = []
    for row in rows:
        try:
            vals.append(float(row.get("width_px", 0.0)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {}
    return {
        "min_px": round(min(vals), 3),
        "max_px": round(max(vals), 3),
        "mean_px": round(sum(vals) / max(1, len(vals)), 3),
        "sample_count": len(vals),
    }


def _hint_names(rows: Any, key: str) -> List[str]:
    out: List[str] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            val = str(row.get(key, "")).strip()
        else:
            val = str(row).strip()
        if val:
            out.append(val)
    return out


def _boundary_summary(path: Dict[str, Any]) -> Dict[str, Any]:
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    if not boundary:
        return {}
    return {
        "available": bool(boundary.get("available", False)),
        "movement_scope": str(boundary.get("movement_scope", "")),
        "boundary_interaction": str(boundary.get("boundary_interaction", "")),
        "boundary_sample_fraction": boundary.get("boundary_sample_fraction", None),
        "transition_count": boundary.get("transition_count", 0),
        "max_transition_depth_delta_m": boundary.get("max_transition_depth_delta_m", None),
        "regions_sequence": list(boundary.get("regions_sequence") or [])[:8],
        "transitions": list(boundary.get("transitions") or [])[:12],
        "motion_implications": list(boundary.get("motion_implications") or [])[:8],
    }


def _ground_object_classification(path: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(path.get("ground_object_classification"), dict):
        return dict(path.get("ground_object_classification") or {})
    tc = path.get("trajectory_contract") if isinstance(path.get("trajectory_contract"), dict) else {}
    if isinstance(tc.get("ground_object_classification"), dict):
        return dict(tc.get("ground_object_classification") or {})
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    counts = dict(sem.get("support_kind_counts") or {})
    total = float(max(1.0, sum(float(v or 0.0) for v in counts.values())))
    walk = (float(counts.get("support_surface", 0.0)) + float(counts.get("floor", 0.0)) + float(counts.get("walkable", 0.0))) / total
    swim = float(counts.get("liquid", 0.0)) / total
    air = float(counts.get("open_air", 0.0)) / total
    block = (float(counts.get("blocking", 0.0)) + float(counts.get("hard_obstacle", 0.0))) / total
    rec = str(tc.get("recommended_motion") or tc.get("dominant_motion") or "").strip()
    if not rec:
        hints = list(path.get("motion_hints") or [])
        for hint in hints:
            if isinstance(hint, dict) and str(hint.get("motion", "")).strip():
                rec = str(hint.get("motion", "")).strip()
                break
            if isinstance(hint, str) and hint.strip():
                rec = hint.strip()
                break
    if not rec:
        rec = "traverse"
    labels: List[str] = []
    if walk >= 0.22:
        labels.append("walkable")
    if swim >= 0.22:
        labels.append("swimmable")
    if air >= 0.22:
        labels.append("open_air")
    if block >= 0.30:
        labels.append("obstacle_heavy")
    if not labels:
        labels.append("uncertain")
    return {
        "dominant_support_kind": max(counts, key=counts.get) if counts else "unknown",
        "walkable_fraction": round(walk, 4),
        "swimmable_fraction": round(swim, 4),
        "open_air_fraction": round(air, 4),
        "blocking_fraction": round(block, 4),
        "terrain_labels": labels,
        "recommended_motion": rec,
    }


def _summarize_path(
    path: Dict[str, Any],
    global_rank: int,
    traj_hyp: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    scores = path.get("scores") if isinstance(path.get("scores"), dict) else {}
    src = path.get("source_entity") if isinstance(path.get("source_entity"), dict) else {}
    tgt = path.get("target_entity") if isinstance(path.get("target_entity"), dict) else {}
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    src_sem = sem.get("source") if isinstance(sem.get("source"), dict) else {}
    tgt_sem = sem.get("target") if isinstance(sem.get("target"), dict) else {}
    contract = path.get("trajectory_contract") if isinstance(path.get("trajectory_contract"), dict) else {}
    shape_contract = path.get("path_shape_contract") if isinstance(path.get("path_shape_contract"), dict) else {}
    render_contract = path.get("animation_render_contract") if isinstance(path.get("animation_render_contract"), dict) else {}
    geometry_quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
    contract_status = path.get("contract_status") if isinstance(path.get("contract_status"), dict) else {}
    tid = str(traj_hyp.get("trajectory_id", "")) if traj_hyp else ""
    return {
        "global_rank": global_rank,
        "path_id": str(path.get("path_id", "")),
        "path_type": str(path.get("path_type", "")),
        "manifold_type": str(path.get("manifold_type", "")),
        "action_family": str(path.get("action_family", "")),
        "movement_scope": str(path.get("movement_scope", "")),
        "boundary_interaction": str(path.get("boundary_interaction", "")),
        "dominant_motion": _first_motion_label(path),
        "acceptance_status": str(path.get("acceptance_status", contract_status.get("status", ""))),
        "rejection_reasons": list(path.get("rejection_reasons") or contract_status.get("rejection_reasons") or []),
        "uncertainty_reasons": list(path.get("uncertainty_reasons") or contract_status.get("uncertainty_reasons") or []),
        "contradiction_reasons": list(path.get("contradiction_reasons") or contract_status.get("contradiction_reasons") or []),
        "grounding_evidence": path.get("grounding_evidence") if isinstance(path.get("grounding_evidence"), dict) else {},
        "scores": dict(scores),
        "overall_confidence": float(scores.get("overall_confidence", 0.0) or 0.0),
        "source_entity": {
            "id": str(src.get("id", "")),
            "label": str(src.get("label", "") or src.get("canonical_name", "") or src_sem.get("label", "")),
            "entity_kind": str(src.get("entity_kind", "")),
        },
        "target_entity": {
            "id": str(tgt.get("id", "")),
            "label": str(tgt.get("label", "") or tgt.get("canonical_name", "") or tgt_sem.get("label", "")),
            "entity_kind": str(tgt.get("entity_kind", "")),
        },
        "regions_traversed": list(path.get("regions_traversed") or []),
        "routing_meta": path.get("routing_meta") if isinstance(path.get("routing_meta"), dict) else {},
        "goal_generation": path.get("goal_generation") if isinstance(path.get("goal_generation"), dict) else {},
        "action_hints": path.get("action_hints"),
        "motion_hints": path.get("motion_hints"),
        "motion_labels": _hint_names(path.get("motion_hints"), "motion")[:8],
        "action_labels": (
            _hint_names(path.get("action_hints"), "action")
            + _hint_names(path.get("action_hints"), "path_mode")
        )[:12],
        "semantic_trace": path.get("semantic_trace"),
        "support_kind_counts": dict((sem or {}).get("support_kind_counts") or {}),
        "ground_object_classification": _ground_object_classification(path),
        "region_boundary_trace": _boundary_summary(path),
        "caption_trace": path.get("caption_trace"),
        "render_layers": path.get("render_layers"),
        "occlusion_trace": path.get("occlusion_trace") if isinstance(path.get("occlusion_trace"), dict) else {},
        "depth_summary": _depth_summary(path),
        "width_summary_px": _width_summary(path),
        "trajectory_contract": {
            "dominant_motion": contract.get("dominant_motion", ""),
            "shape_type": contract.get("shape_type", shape_contract.get("shape_type", "")),
            "direction_profile": contract.get("direction_profile", shape_contract.get("direction_profile", {})),
            "support_dominant": contract.get("support_dominant", ""),
            "animation_ready": contract.get("animation_ready", False),
            "mean_visible_fraction": contract.get("mean_visible_fraction", None),
            "width_mean_px": contract.get("width_mean_px", None),
        },
        "path_shape_contract": {
            "shape_type": shape_contract.get("shape_type", ""),
            "straight_line_like": shape_contract.get("straight_line_like", False),
            "straightness_ratio": shape_contract.get("straightness_ratio", None),
            "direction_profile": shape_contract.get("direction_profile", {}),
            "support_evidence": shape_contract.get("support_evidence", {}),
            "shape_justification": shape_contract.get("shape_justification", []),
            "geometry_refs": shape_contract.get("geometry_refs", {}),
            "geometry_quality": shape_contract.get("geometry_quality", {}),
            "rejection_reasons": shape_contract.get("rejection_reasons", []),
        },
        "path_geometry_quality": {
            "zigzag_score": geometry_quality.get("zigzag_score"),
            "turn_angle_p95": geometry_quality.get("turn_angle_p95"),
            "curvature_energy": geometry_quality.get("curvature_energy"),
            "vertical_shoot_score": geometry_quality.get("vertical_shoot_score"),
            "depth_jump_count": geometry_quality.get("depth_jump_count"),
            "support_snap_displacement_px": geometry_quality.get("support_snap_displacement_px"),
            "mask_boundary_crossing_count": geometry_quality.get("mask_boundary_crossing_count"),
            "terrain_transition_mismatch": geometry_quality.get("terrain_transition_mismatch"),
            "smoothability_status": geometry_quality.get("smoothability_status"),
            "geometry_rejection_reasons": list(geometry_quality.get("geometry_rejection_reasons") or []),
        },
        "geometry_smoothing_provenance": path.get("geometry_smoothing_provenance") if isinstance(path.get("geometry_smoothing_provenance"), dict) else {},
        "animation_render_contract": {
            "render_primitive": render_contract.get("render_primitive", ""),
            "alpha_policy": render_contract.get("alpha_policy", ""),
            "width_policy": render_contract.get("width_policy", ""),
            "depth_scale_policy": render_contract.get("depth_scale_policy", ""),
            "render_layers": render_contract.get("render_layers", []),
            "motion_labels": render_contract.get("motion_labels", []),
            "action_labels": render_contract.get("action_labels", []),
            "sample_state_preview_count": len(render_contract.get("sample_state_preview") or []),
        },
        "natural_direction_2d_deg": path.get("natural_direction_2d_deg"),
        "validation_notes": path.get("validation_notes") or path.get("qa_notes"),
        "polyline_summary": _polyline_endpoints(path),
        "trajectory_id": tid,
        "trajectory_has_points": bool(traj_hyp and (traj_hyp.get("trajectory_points") or traj_hyp.get("samples"))),
        "linked_action": {
            "action_id": str((action or {}).get("action_id", "")),
            "action_name": str((action or {}).get("action_name", "")),
            "action_family": str((action or {}).get("action_family", "")),
            "manifold_type": str((action or {}).get("manifold_type", "")),
        }
        if action
        else None,
    }


def export_path_visual_qa_json_and_md(
    *,
    paths_root: Path,
    stem: str,
    width: int,
    height: int,
    ranked_paths: Sequence[Dict[str, Any]],
    traj_bundle: Optional[Dict[str, Any]],
    action_hypotheses: Optional[Dict[str, Any]],
    batch_meta: Sequence[Dict[str, Any]],
    dropped_paths: Sequence[Any],
) -> None:
    paths_root.mkdir(parents=True, exist_ok=True)
    rank_by_pid: Dict[str, int] = {}
    for i, p in enumerate(ranked_paths):
        pid = str(p.get("path_id", "")).strip()
        if pid and pid not in rank_by_pid:
            rank_by_pid[pid] = i + 1

    act_map = _actions_by_path_id(action_hypotheses)
    traj_map = _traj_by_path_id(traj_bundle)

    summaries: List[Dict[str, Any]] = []
    for p in ranked_paths:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("path_id", "")).strip()
        if not pid:
            continue
        gr = rank_by_pid.get(pid, 0)
        summaries.append(_summarize_path(p, gr, traj_map.get(pid), act_map.get(pid)))

    terrain_label_counts: Dict[str, int] = {}
    support_kind_counts: Dict[str, int] = {}
    for s in summaries:
        gcls = s.get("ground_object_classification") if isinstance(s.get("ground_object_classification"), dict) else {}
        for label in list(gcls.get("terrain_labels") or []):
            label_s = str(label).strip()
            if label_s:
                terrain_label_counts[label_s] = terrain_label_counts.get(label_s, 0) + 1
        sk = str(gcls.get("dominant_support_kind", "")).strip()
        if sk:
            support_kind_counts[sk] = support_kind_counts.get(sk, 0) + 1

    payload: Dict[str, Any] = {
        "schema": "citv_path_visual_qa_v1",
        "version": "1.0",
        "stem": stem,
        "image_dimensions": [int(width), int(height)],
        "ranking": "overall_confidence_desc",
        "batch_overlays": list(batch_meta),
        "status_counts": {
            status: sum(1 for s in summaries if str(s.get("acceptance_status", "")) == status)
            for status in ("accepted", "plausible_uncertain", "low_confidence", "rejected")
        },
        "ground_object_classification_summary": {
            "terrain_label_counts": dict(sorted(terrain_label_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "dominant_support_kind_counts": dict(sorted(support_kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "paths": summaries,
        "dropped_paths": list(dropped_paths),
    }
    json_path = paths_root / "path_visual_qa.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: List[str] = [
        f"# Path visual QA — `{stem}`",
        "",
        f"Image size: **{width}×{height}**. Paths are ranked by **overall_confidence** (descending).",
        "",
        "## Batch scene overlays",
        "",
        "Each PNG shows up to 10 paths on the real image with distinct colors; trajectories linked by `path_id` are drawn on top.",
        "",
    ]
    for b in batch_meta:
        idx = b.get("batch_index", 0)
        fn = b.get("file", "")
        pids = b.get("path_ids") or []
        ranks = b.get("global_ranks") or []
        lines.append(f"- **Batch {idx}** — `{fn}` — ranks `{ranks}` — {len(pids)} path(s)")
    lines.extend(["", "## Paths (full list)", ""])
    for s in summaries:
        pid = s.get("path_id", "")
        lines.append(f"### Rank {s.get('global_rank')} — `{pid}`")
        lines.append("")
        lines.append(f"- **path_type**: {s.get('path_type')}; **manifold**: {s.get('manifold_type')}; **action_family**: {s.get('action_family')}")
        lines.append(
            f"- **status**: {s.get('acceptance_status')}; "
            f"**rejection_reasons**: {s.get('rejection_reasons') or []}"
        )
        if s.get("uncertainty_reasons") or s.get("contradiction_reasons"):
            lines.append(
                f"- **uncertainty_reasons**: {s.get('uncertainty_reasons') or []}; "
                f"**contradiction_reasons**: {s.get('contradiction_reasons') or []}"
            )
        grounding = s.get("grounding_evidence") or {}
        if grounding:
            lines.append(f"- **grounding_evidence**: {grounding}")
        if s.get("movement_scope") or s.get("boundary_interaction"):
            lines.append(
                f"- **movement_scope**: {s.get('movement_scope')}; "
                f"**boundary_interaction**: {s.get('boundary_interaction')}"
            )
        if s.get("dominant_motion"):
            lines.append(f"- **dominant_motion**: {s.get('dominant_motion')}")
        sc = s.get("scores") or {}
        if sc:
            lines.append(
                "- **scores**: "
                + ", ".join(f"{k}={v}" for k, v in sorted(sc.items(), key=lambda kv: kv[0])[:20])
            )
        lines.append(f"- **confidence**: {s.get('overall_confidence', 0.0):.3f}")
        se = s.get("source_entity") or {}
        te = s.get("target_entity") or {}
        lines.append(
            f"- **route**: `{se.get('label') or se.get('id')}` ({se.get('entity_kind')}) → "
            f"`{te.get('label') or te.get('id')}` ({te.get('entity_kind')})"
        )
        regs = s.get("regions_traversed") or []
        if regs:
            lines.append(f"- **regions_traversed**: {regs}")
        support = s.get("support_kind_counts") or {}
        if support:
            lines.append(f"- **support_kinds**: {support}")
        gcls = s.get("ground_object_classification") or {}
        if gcls:
            lines.append(f"- **ground_object_classification**: {gcls}")
        boundary = s.get("region_boundary_trace") or {}
        if boundary:
            lines.append(
                "- **region_boundary_trace**: "
                f"scope={boundary.get('movement_scope')}; "
                f"interaction={boundary.get('boundary_interaction')}; "
                f"transitions={boundary.get('transition_count')}; "
                f"boundary_fraction={boundary.get('boundary_sample_fraction')}; "
                f"implications={boundary.get('motion_implications')}"
            )
        motions = s.get("motion_labels") or []
        if motions:
            lines.append(f"- **motion_labels**: {motions}")
        actions = s.get("action_labels") or []
        if actions:
            lines.append(f"- **action_labels**: {actions}")
        rl = s.get("render_layers")
        if rl:
            lines.append(f"- **render_layers**: {rl}")
        occ = s.get("occlusion_trace") or {}
        if occ:
            lines.append(f"- **occlusion**: {occ}")
        depth = s.get("depth_summary") or {}
        width_summary = s.get("width_summary_px") or {}
        if depth or width_summary:
            lines.append(f"- **depth/width**: depth={depth}; width_px={width_summary}")
        tc = s.get("trajectory_contract") or {}
        if tc:
            lines.append(f"- **trajectory_contract**: {tc}")
        pc = s.get("path_shape_contract") or {}
        if pc:
            lines.append(f"- **path_shape_contract**: {pc}")
        rc = s.get("animation_render_contract") or {}
        if rc:
            lines.append(f"- **animation_render_contract**: {rc}")
        vn = s.get("validation_notes")
        if vn:
            lines.append(f"- **notes**: {vn}")
        rm = s.get("routing_meta") or {}
        if rm:
            lines.append(f"- **routing_meta**: {rm}")
        gg = s.get("goal_generation") or {}
        if gg:
            lines.append(f"- **goal_generation**: {gg}")
        if s.get("trajectory_id"):
            lines.append(
                f"- **trajectory**: `{s.get('trajectory_id')}` "
                f"(has_points={s.get('trajectory_has_points')})"
            )
        la = s.get("linked_action")
        if isinstance(la, dict) and (la.get("action_id") or la.get("action_name")):
            lines.append(
                f"- **action**: {la.get('action_name')} (`{la.get('action_id')}`) "
                f"family={la.get('action_family')}"
            )
        ps = s.get("polyline_summary") or {}
        if ps:
            lines.append(f"- **polyline**: {ps}")
        lines.append("")

    per_path_dir = paths_root / "per_path"
    per_path_dir.mkdir(parents=True, exist_ok=True)
    for s in summaries:
        pid = str(s.get("path_id", "")).strip()
        if not pid:
            continue
        json_out = per_path_dir / f"{pid}.json"
        md_out = per_path_dir / f"{pid}.md"
        json_out.write_text(
            json.dumps(
                {
                    "schema": "citv_path_visual_qa_per_path_v1",
                    "stem": stem,
                    "path": s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        md_lines = [
            f"# Path `{pid}`",
            "",
            f"- rank: {s.get('global_rank')}",
            f"- manifold: {s.get('manifold_type')}",
            f"- action_family: {s.get('action_family')}",
            f"- movement_scope: {s.get('movement_scope')}",
            f"- boundary_interaction: {s.get('boundary_interaction')}",
            f"- dominant_motion: {s.get('dominant_motion')}",
            f"- confidence: {float(s.get('overall_confidence', 0.0)):.3f}",
            f"- status: {s.get('acceptance_status')}",
            f"- rejection_reasons: {s.get('rejection_reasons') or []}",
            f"- uncertainty_reasons: {s.get('uncertainty_reasons') or []}",
            f"- contradiction_reasons: {s.get('contradiction_reasons') or []}",
            f"- grounding_evidence: {s.get('grounding_evidence') or {}}",
            f"- source: {((s.get('source_entity') or {}).get('label') or (s.get('source_entity') or {}).get('id') or '')}",
            f"- target: {((s.get('target_entity') or {}).get('label') or (s.get('target_entity') or {}).get('id') or '')}",
            f"- motion_labels: {s.get('motion_labels') or []}",
            f"- action_labels: {s.get('action_labels') or []}",
            f"- render_layers: {s.get('render_layers') or []}",
            f"- occlusion_trace: {s.get('occlusion_trace') or {}}",
            f"- trajectory_contract: {s.get('trajectory_contract') or {}}",
            f"- path_shape_contract: {s.get('path_shape_contract') or {}}",
            f"- animation_render_contract: {s.get('animation_render_contract') or {}}",
        ]
        md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if dropped_paths:
        lines.extend(["## Dropped / deduped paths (reference)", ""])
        for d in dropped_paths[:80]:
            lines.append(f"- {d}")
        lines.append("")

    md_path = paths_root / "path_visual_qa.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
