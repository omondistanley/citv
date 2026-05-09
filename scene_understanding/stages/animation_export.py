"""Stage: animation QA video export for the staged pipeline.

Produces MP4 QA clips at 24 fps and 120 fps by wiring the existing
``export_animation_qa`` renderer.  Prerequisite: ``paths_export`` stage must
have run and written ``path_hypotheses.json`` to the ``{stem}_paths/``
directory.  If that file is absent, this stage is a no-op.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..action_ontology import list_section, load_action_ontology, string
from ..pipeline_context import PipelineContext

_MAX_LABELLED_PATHS = 15


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _record_export_failure(ctx: PipelineContext, artifact: str, reason: str, path: str = "") -> None:
    """Record an animation-stage fallback without stopping later parity export."""
    failure = {
        "schema": "citv_artifact_failure_v1",
        "artifact": artifact,
        "path": path,
        "reason": reason,
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
    }
    ctx.extra.setdefault("artifact_failures", []).append(failure)
    ctx.extra.setdefault("artifact_warnings", []).append(reason)


def _float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _path_polyline_2d(path: Dict[str, Any]) -> List[List[float]]:
    raw = (
        path.get("display_polyline_2d")
        or path.get("polyline_2d_validated")
        or path.get("polyline_2d_reprojected")
        or path.get("polyline_2d")
        or []
    )
    pts: List[List[float]] = []
    for xy in list(raw):
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            pts.append([_float(xy[0]), _float(xy[1])])
    return pts


def _extract_kinematic_signatures(
    polyline_3d: List[List[float]],
    *,
    motion_labels: Optional[Dict[str, str]] = None,
    jump_z_thresh: float = 0.25,
    climb_z_thresh: float = 0.12,
    idle_fraction: float = 0.08,
    crawl_min_run: int = 6,
) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    if len(polyline_3d) < 2:
        return sigs
    zs = [float(pt[2]) if len(pt) > 2 else 0.0 for pt in polyline_3d]
    
    # ENHANCEMENT: Smooth monocular depth bleeding spikes via 3-sample median filter
    if len(zs) >= 3:
        smoothed_zs = [zs[0]]
        for idx in range(1, len(zs) - 1):
            smoothed_zs.append(sorted([zs[idx-1], zs[idx], zs[idx+1]])[1])
        smoothed_zs.append(zs[-1])
        zs = smoothed_zs

    labels = dict(motion_labels or {})
    flat_motion = str(labels.get("flat", "traverse"))
    jump_motion = str(labels.get("jump_up", "jump"))
    climb_motion = str(labels.get("climb_up", "climb"))
    descend_motion = str(labels.get("move_down", "descend"))
    crawl_motion = str(labels.get("crawl_low", "crawl"))
    n = len(zs)
    i = 0
    while i < n - 1:
        dz = zs[i + 1] - zs[i]
        if abs(dz) < idle_fraction:
            motion = flat_motion
        elif dz > jump_z_thresh:
            motion = jump_motion
        elif dz > climb_z_thresh:
            motion = climb_motion
        elif dz < -jump_z_thresh:
            motion = descend_motion
        elif dz < -climb_z_thresh:
            motion = descend_motion
        else:
            motion = flat_motion
        j = i + 1
        while j < n - 1:
            dz_n = zs[j + 1] - zs[j]
            if abs(dz_n) < idle_fraction:
                nm = flat_motion
            elif dz_n > jump_z_thresh:
                nm = jump_motion
            elif dz_n > climb_z_thresh:
                nm = climb_motion
            elif dz_n < -jump_z_thresh:
                nm = descend_motion
            elif dz_n < -climb_z_thresh:
                nm = descend_motion
            else:
                nm = flat_motion
            if nm != motion:
                break
            j += 1
        if motion == flat_motion and (j - i) >= crawl_min_run:
            run_zs = zs[i : j + 1]
            z_var = max(run_zs) - min(run_zs)
            if z_var < idle_fraction / 4.0:
                motion = crawl_motion
        sigs.append({"start_idx": i, "end_idx": j, "motion": motion, "dz": float(zs[j] - zs[i])})
        i = j
    return sigs


def _build_animation_plan(paths: List[Dict[str, Any]], cfg: Any) -> Dict[str, Any]:
    ontology = load_action_ontology(cfg)
    fps = int(getattr(cfg, "path_animation_fps", 24)) if cfg else 24
    speed_walk = float(getattr(cfg, "path_speed_walk_px_s", 80.0)) if cfg else 80.0
    idle_s = float(getattr(cfg, "path_duration_idle_s", 0.5)) if cfg else 0.5
    jump_s = float(getattr(cfg, "path_duration_jump_s", 0.6)) if cfg else 0.6
    climb_s = float(getattr(cfg, "path_duration_climb_s", 0.8)) if cfg else 0.8
    speed_by_motion = dict(getattr(cfg, "path_speed_by_motion", {}) or {}) if cfg else {}
    idle_motion_label = str(getattr(cfg, "path_idle_motion_label", "idle")) if cfg else "idle"
    default_tokens = _motion_policy_tokens(ontology, "default_motion", "traverse")
    default_motion = default_tokens[0] if default_tokens else "traverse"
    kin_cfg = dict(getattr(cfg, "path_kinematic_motion_labels", {}) or {}) if cfg else {}
    kinematic_labels = {
        "flat": str(kin_cfg.get("flat", default_motion)),
        "jump_up": str(kin_cfg.get("jump_up", "jump")),
        "climb_up": str(kin_cfg.get("climb_up", "climb")),
        "move_down": str(kin_cfg.get("move_down", "descend")),
        "crawl_low": str(kin_cfg.get("crawl_low", "crawl")),
    }

    ranked = sorted(
        paths,
        key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
        reverse=True,
    )
    plan_paths: List[Dict[str, Any]] = []

    for p in ranked:
        pts = [tuple(map(float, xy)) for xy in _path_polyline_2d(p)]
        if len(pts) < 2:
            continue
        length = sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )
        conf = float((p.get("scores") or {}).get("overall_confidence", 0.0))
        poly3d = p.get("polyline_3d") or []
        kin_sigs = _extract_kinematic_signatures(poly3d, motion_labels=kinematic_labels) if poly3d else []
        ground_cls = _ground_object_classification_from_path(p, ontology=ontology)
        rec_motion = str(ground_cls.get("recommended_motion", _primary_motion_from_path(p, ontology=ontology)))

        def _speed_for_motion(motion: str) -> float:
            m = str(motion or "").strip().lower()
            if m in speed_by_motion:
                return max(1.0, _float(speed_by_motion.get(m), speed_walk))
            return max(1.0, speed_walk)

        if kin_sigs:
            t = 0.0
            segments: List[Dict[str, Any]] = [{"motion": idle_motion_label, "t0_s": t, "t1_s": t + idle_s}]
            t += idle_s
            for sig in kin_sigs:
                motion = str(sig["motion"])
                if motion in {"walk", "run", "crawl"} and rec_motion:
                    motion = rec_motion
                seg_n = max(1, sig["end_idx"] - sig["start_idx"])
                seg_frac = seg_n / max(1, len(poly3d) - 1)
                if motion in {kinematic_labels["jump_up"], kinematic_labels["climb_up"], kinematic_labels["move_down"]}:
                    dur = (jump_s if motion == kinematic_labels["jump_up"] else climb_s) * max(1, seg_n // 4)
                else:
                    spd = _speed_for_motion(motion)
                    dur = float(length * seg_frac / max(1.0, spd))
                segments.append({"motion": motion, "t0_s": t, "t1_s": t + dur})
                t += dur
        else:
            motion = rec_motion or _primary_motion_from_path(p, ontology=ontology)
            speed = _speed_for_motion(motion)
            move_s = float(length / max(1.0, speed))
            t = 0.0
            segments = [
                {"motion": idle_motion_label, "t0_s": t, "t1_s": t + idle_s},
                {"motion": motion, "t0_s": t + idle_s, "t1_s": t + idle_s + move_s},
            ]
            t = t + idle_s + move_s
            if conf < 0.5 and motion == default_motion:
                segments.append({"motion": kinematic_labels["jump_up"], "t0_s": t, "t1_s": t + jump_s})
                t += jump_s

        trajectory_id = p.get("path_id", "")
        timeline = [
            {
                "time_s": seg["t0_s"],
                "motion": seg["motion"],
                "path_id": trajectory_id,
                "trajectory_id": trajectory_id,
            }
            for seg in segments
        ]

        motion_counts: Dict[str, int] = {}
        for seg in segments:
            motion_counts[seg["motion"]] = motion_counts.get(seg["motion"], 0) + 1
        total_segs = max(1, len(segments))
        all_motions: List[str] = []
        for row in _motion_candidates_from_path(p, cfg=cfg, ontology=ontology):
            m = str(row.get("motion", "")).strip()
            if m and m not in all_motions:
                all_motions.append(m)
        for seg in segments:
            m = str(seg.get("motion", "")).strip()
            if m and m not in all_motions:
                all_motions.append(m)
        if not all_motions:
            all_motions = _motion_policy_tokens(ontology, "default_motion", "traverse")
        mode_candidates = [
            {
                "motion": m,
                "score": round(
                    min(
                        1.0,
                        max(
                            float(motion_counts.get(m, 0)) / total_segs,
                            0.20 if m == rec_motion else 0.0,
                        ),
                    ),
                    3,
                ),
            }
            for m in all_motions
        ]
        mode_candidates.sort(key=lambda x: x["score"], reverse=True)

        scores = p.get("scores") or {}
        plan_paths.append(
            {
                "path_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "path_num": p.get("path_num", 0),
                "action_fit_confidence": conf,
                "assumptions": "kinematic_signature" if kin_sigs else "confidence_gated",
                "kinematic_signatures": kin_sigs,
                "motion_mode_candidates": mode_candidates,
                "confidence_breakdown": {
                    "overall": conf,
                    "geometric": float(scores.get("geometric_confidence", conf)),
                    "semantic": float(scores.get("semantic_confidence", conf)),
                    "kinematic": float(scores.get("kinematic_confidence", 0.5 if kin_sigs else 0.0)),
                },
                "relation_guardrails": list(p.get("relation_guardrails") or []),
                "path_shape_contract": dict(p.get("path_shape_contract") or {}),
                "animation_render_contract": dict(p.get("animation_render_contract") or {}),
                "direction_profile": dict((p.get("path_shape_contract") or {}).get("direction_profile") or {}),
                "acceptance_status": str(p.get("acceptance_status", "")),
                "rejection_reasons": list(p.get("rejection_reasons") or []),
                "segments": segments,
                "trajectory_points": _path_polyline_2d(p),
                "geometry_smoothing_provenance": dict(p.get("geometry_smoothing_provenance") or {}),
                "path_geometry_quality": dict(p.get("path_geometry_quality") or {}),
                "timeline_records": timeline,
                "ground_object_classification": dict(ground_cls),
                "surface_motion_recommendation": str(rec_motion),
                "fps": fps,
                "duration_s": t,
            }
        )

    return {"fps": fps, "paths": plan_paths}


def _build_path_atlas_manifest(
    paths: List[Dict[str, Any]],
    image_stem: str,
    track_dir_name: str,
    panel_size: int = 10,
) -> Dict[str, Any]:
    """Atlas manifest for animation QA, split into deterministic fixed-size panels."""
    panel_size = max(1, int(panel_size))
    entries: List[Dict[str, Any]] = []
    panel_count = max(1, int(math.ceil(len(paths) / float(panel_size)))) if paths else 1
    for i, p in enumerate(paths):
        pid = str(p.get("path_id", "")).strip()
        if not pid:
            continue
        src_id = str((p.get("source_entity") or {}).get("id", "src"))
        tgt_id = str((p.get("target_entity") or {}).get("id", "tgt"))
        panel_idx = int(i // panel_size)
        entries.append(
            {
                "path_num": i,
                "path_id": pid,
                "panel_index": panel_idx,
                "color_bgr": [80, 200, 80],
                "color_hex": "#50c850",
                "suppressed": False,
                "path_level": str(p.get("path_level", "object")),
                "path_type": str(p.get("path_type", "fmm_geodesic")),
                "route_signature": f"{src_id}_to_{tgt_id}",
                "overall_confidence": float((p.get("scores") or {}).get("overall_confidence", 0.0)),
            }
        )
    return {
        "schema": "citv_path_atlas_manifest_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "ranking_policy": {
            "rank_source": "fmm_staged",
            "path_atlas_total": len(entries),
            "path_atlas_panel_size": panel_size,
        },
        "panels": [{"index": i} for i in range(panel_count)],
        "paths": entries,
        "trajectory_colors": [],
    }


def _states_from_path(path: Dict[str, Any], *, horizon_s: float = 1.0) -> List[Dict[str, Any]]:
    pts2 = _path_polyline_2d(path)
    if len(pts2) < 2:
        return []
    pts3 = list(path.get("polyline_3d") or [])
    widths = [
        _float(r.get("width_px"), 4.0)
        for r in list(path.get("width_profile_px") or [])
        if isinstance(r, dict)
    ]
    vis_rows = [r for r in list(path.get("visibility_profile") or []) if isinstance(r, dict)]
    n = len(pts2)
    out: List[Dict[str, Any]] = []
    length_total = max(1e-6, _path_length_px(path))
    for idx, xy in enumerate(pts2):
        if idx < n - 1:
            nxt = pts2[idx + 1]
            theta = math.atan2(float(nxt[1]) - float(xy[1]), float(nxt[0]) - float(xy[0]))
            seg_len = math.hypot(float(nxt[0]) - float(xy[0]), float(nxt[1]) - float(xy[1]))
        elif out:
            theta = float(out[-1].get("theta_rad", 0.0))
            seg_len = 0.0
        else:
            theta = 0.0
            seg_len = 0.0
        z = None
        if idx < len(pts3) and isinstance(pts3[idx], (list, tuple)) and len(pts3[idx]) >= 3:
            zf = _float(pts3[idx][2], -1.0)
            z = zf if zf > 0.0 else None
        width_px = widths[min(len(widths) - 1, idx)] if widths else None
        vis = vis_rows[min(len(vis_rows) - 1, idx)] if vis_rows else {}
        visible_fraction = _float(vis.get("visible_fraction"), 1.0)
        out.append(
            {
                "t_s": float((idx / max(1, n - 1)) * horizon_s),
                "x_px": float(xy[0]),
                "y_px": float(xy[1]),
                "theta_rad": theta,
                "heading_deg": math.degrees(theta),
                "z_m": z,
                "speed_px_s": float(seg_len / max(1e-6, horizon_s / max(1, n - 1))),
                "arc_fraction": float(idx / max(1, n - 1)),
                "scale_hint": None if width_px is None else max(0.25, min(2.5, width_px / 12.0)),
                "width_px": width_px,
                "alpha": max(0.15, min(1.0, visible_fraction)),
                "visible_fraction": visible_fraction,
                "render_layer": str(vis.get("render_layer", "in_front")),
                "occluder_ids": list(vis.get("occluder_ids") or []),
                "distance_fraction_hint": float(seg_len / length_total),
            }
        )
    return out


def _path_length_px(path: Dict[str, Any]) -> float:
    pts = _path_polyline_2d(path)
    if len(pts) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )
    )


def _split_motion_terms(value: Any) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    text = text.replace("_or_", " ").replace("/", " ").replace("|", " ").replace(",", " ")
    toks = [t for t in re.split(r"[^a-z0-9_]+", text) if t]
    out: List[str] = []
    for t in toks:
        if t not in out:
            out.append(t)
    return out


def _motion_policy_tokens(ontology: Dict[str, Any], key: str, fallback: str) -> List[str]:
    m = str(string(ontology, "motion_hint_policy", key, fallback)).strip().lower()
    toks = _split_motion_terms(m)
    return toks if toks else _split_motion_terms(fallback)


def _support_action_priors(path: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, float]:
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    counts = dict(sem.get("support_kind_counts") or {})
    total = float(max(1.0, sum(_float(v, 0.0) for v in counts.values())))
    priors: Dict[str, float] = {}
    for rule in list_section(ontology, "support_kind_policy"):
        kind = str(rule.get("kind", "")).strip()
        if not kind:
            continue
        w = max(0.0, min(1.0, _float(counts.get(kind), 0.0) / total))
        if w <= 0.0:
            continue
        for action in list(rule.get("actions") or []):
            for token in _split_motion_terms(action):
                priors[token] = max(priors.get(token, 0.0), w)
    return priors


def _ground_object_classification_from_path(path: Dict[str, Any], *, ontology: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    existing = path.get("ground_object_classification")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    onto = ontology if isinstance(ontology, dict) and ontology else load_action_ontology(None)
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    counts = dict(sem.get("support_kind_counts") or {})
    total = float(max(1.0, sum(_float(v, 0.0) for v in counts.values())))
    walkable = (
        _float(counts.get("support_surface"), 0.0)
        + _float(counts.get("floor"), 0.0)
        + _float(counts.get("walkable"), 0.0)
        + _float(counts.get("portal"), 0.0) * 0.5
    ) / total
    swimmable = _float(counts.get("liquid"), 0.0) / total
    aerial = _float(counts.get("open_air"), 0.0) / total
    blocked = (
        _float(counts.get("blocking"), 0.0)
        + _float(counts.get("hard_obstacle"), 0.0)
    ) / total
    dominant = max(counts, key=counts.get) if counts else "unknown"
    action_priors = _support_action_priors(path, onto)
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict):
            mv = str(hint.get("motion", ""))
            sc = max(0.0, min(1.0, _float(hint.get("score"), 0.0)))
        else:
            mv = str(hint)
            sc = 0.4
        for token in _split_motion_terms(mv):
            action_priors[token] = max(action_priors.get(token, 0.0), sc)
    default_tokens = _motion_policy_tokens(onto, "default_motion", "traverse")
    if action_priors:
        recommended_motion = max(action_priors.items(), key=lambda kv: kv[1])[0]
    else:
        recommended_motion = default_tokens[0] if default_tokens else "traverse"
    labels: List[str] = []
    if walkable >= 0.22:
        labels.append("walkable")
    if swimmable >= 0.22:
        labels.append("swimmable")
    if aerial >= 0.22:
        labels.append("open_air")
    if blocked >= 0.30:
        labels.append("obstacle_heavy")
    if not labels:
        labels.append("uncertain")
    return {
        "dominant_support_kind": str(dominant),
        "walkable_fraction": round(float(walkable), 4),
        "swimmable_fraction": round(float(swimmable), 4),
        "open_air_fraction": round(float(aerial), 4),
        "blocking_fraction": round(float(blocked), 4),
        "labels": labels,
        "recommended_motion": str(recommended_motion),
        "support_action_priors": {str(k): round(float(v), 4) for k, v in sorted(action_priors.items(), key=lambda kv: (-kv[1], kv[0]))[:12]},
    }


def _primary_motion_from_path(path: Dict[str, Any], *, ontology: Optional[Dict[str, Any]] = None) -> str:
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict) and str(hint.get("motion", "")).strip():
            toks = _split_motion_terms(hint.get("motion", ""))
            if toks:
                return toks[0]
        if isinstance(hint, str) and hint.strip():
            toks = _split_motion_terms(hint)
            if toks:
                return toks[0]
    cls = _ground_object_classification_from_path(path, ontology=ontology)
    rec = str(cls.get("recommended_motion", "")).strip()
    if rec:
        return rec
    contract = path.get("trajectory_contract") if isinstance(path.get("trajectory_contract"), dict) else {}
    if str(contract.get("dominant_motion", "")).strip():
        toks = _split_motion_terms(contract.get("dominant_motion", ""))
        if toks:
            return toks[0]
    onto = ontology if isinstance(ontology, dict) and ontology else load_action_ontology(None)
    fallbacks = _motion_policy_tokens(onto, "default_motion", "traverse")
    return fallbacks[0] if fallbacks else "traverse"


def _motion_candidates_from_path(path: Dict[str, Any], *, cfg: Any = None, ontology: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    onto = ontology if isinstance(ontology, dict) and ontology else load_action_ontology(cfg)
    conf = max(0.05, min(1.0, _float((path.get("scores") or {}).get("overall_confidence"), 0.0)))
    rows: List[Dict[str, Any]] = []
    seen = set()
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict):
            motion = str(hint.get("motion", "")).strip()
            score = _float(hint.get("score"), conf)
        else:
            motion = str(hint).strip()
            score = conf
        for token in _split_motion_terms(motion):
            if token and token not in seen:
                rows.append({"motion": token, "confidence": max(0.05, min(1.0, score))})
                seen.add(token)
    for token, score in _support_action_priors(path, onto).items():
        if token and token not in seen:
            rows.append({"motion": token, "confidence": max(0.05, min(1.0, score))})
            seen.add(token)
    for row in list(path.get("action_hints") or []):
        if isinstance(row, dict):
            for token in _split_motion_terms(row.get("action", "")):
                if token and token not in seen:
                    rows.append({"motion": token, "confidence": max(0.05, min(1.0, _float(row.get("score"), conf * 0.65)))})
                    seen.add(token)
    cls = _ground_object_classification_from_path(path, ontology=onto)
    rec = str(cls.get("recommended_motion", "")).strip()
    for token in _split_motion_terms(rec):
        if token and token not in seen:
            rows.append({"motion": token, "confidence": max(0.15, min(1.0, conf * 0.88))})
            seen.add(token)
    default_tok = _motion_policy_tokens(onto, "default_motion", "traverse")
    for token in default_tok[:2]:
        if token and token not in seen:
            rows.append({"motion": token, "confidence": max(0.05, min(1.0, conf * 0.72))})
            seen.add(token)
    idle_label = str(getattr(cfg, "path_idle_motion_label", "idle")) if cfg is not None else "idle"
    for token in _split_motion_terms(idle_label):
        if token and token not in seen:
            rows.append({"motion": token, "confidence": 0.15})
            seen.add(token)
            break
    rows.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
    return rows[:8]


def _trajectory_path_ids(traj_bundle: Dict[str, Any]) -> set:
    out = set()
    for hyp in list((traj_bundle or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        pid = str((hyp.get("action_context") or {}).get("path_id") or hyp.get("continues_from_path_id") or "")
        if pid:
            out.add(pid)
        for sample in list(hyp.get("samples") or []):
            if not isinstance(sample, dict):
                continue
            ev = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
            pid = str(ev.get("path_id", ""))
            if pid:
                out.add(pid)
    return out


def _path_follow_trajectory(
    ctx: PipelineContext,
    path: Dict[str, Any],
    image_size: Dict[str, int],
    track_dir_name: str,
    cfg: Any,
) -> Optional[Dict[str, Any]]:
    pid = str(path.get("path_id", "")).strip()
    if not pid:
        return None
    speed_walk = float(getattr(cfg, "path_speed_walk_px_s", 80.0)) if cfg else 80.0
    length = _path_length_px(path)
    horizon_s = max(0.20, min(12.0, length / max(1.0, speed_walk)))
    states = _states_from_path(path, horizon_s=horizon_s)
    if len(states) < 2:
        return None
    conf = max(0.0, min(1.0, _float((path.get("scores") or {}).get("overall_confidence"), 0.0)))
    ontology = load_action_ontology(cfg)
    motion = _primary_motion_from_path(path, ontology=ontology)
    tid = f"traj_{ctx.stem}_{pid}_path_follow"
    return {
        "trajectory_id": tid,
        "subject": _entity_for_trajectory(path, "source_entity"),
        "counterpart": _entity_for_trajectory(path, "target_entity"),
        "continues_from_path_id": pid,
        "time_model": {
            "type": "path_follow_full_hypothesis",
            "horizon_s": horizon_s,
            "dt_s": horizon_s / max(1, len(states) - 1),
        },
        "state_representation": "se2_root",
        "samples": [
            {
                "sample_id": f"{tid}_path_follow",
                "sample_kind": "path_follow",
                "weight": max(0.10, conf),
                "states_t": states,
                "evidence": {
                    "path_id": pid,
                    "source": "accepted_path_hypothesis",
                    "path_type": str(path.get("path_type", "")),
                    "manifold_type": str(path.get("manifold_type", "")),
                    "render_layers": list(path.get("render_layers") or []),
                    "width_profile_available": bool(path.get("width_profile_px")),
                    "visibility_profile_available": bool(path.get("visibility_profile")),
                    "region_boundary_trace_available": bool(path.get("region_boundary_trace")),
                    "path_shape_contract": dict(path.get("path_shape_contract") or {}),
                    "animation_render_contract": dict(path.get("animation_render_contract") or {}),
                    "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
                    "acceptance_status": str(path.get("acceptance_status", "")),
                    "rejection_reasons": list(path.get("rejection_reasons") or []),
                    "movement_scope": str(path.get("movement_scope", "")),
                    "boundary_interaction": str(path.get("boundary_interaction", "")),
                },
            }
        ],
        "constraints_used": {
            "path_hypotheses_json": f"scene_graph/{track_dir_name}/{ctx.stem}_paths/path_hypotheses.json",
            "path_traversability_speed_npy": str(ctx.path_exports.get("traversability_speed_npy", "")),
        },
        "motion_mode_candidates": _motion_candidates_from_path(path, cfg=cfg, ontology=ontology),
        "timeline_records": [
            {
                "t0_s": 0.0,
                "t1_s": horizon_s,
                "motion": motion,
                "relation_predicate": "",
                "navigation_hint": str((path.get("navigation_zone_trace") or {}).get("dominant_motion_hint", "")),
            }
        ],
        "action_context": {
            "path_id": pid,
            "action_family": str(path.get("action_family", "")),
            "manifold_type": str(path.get("manifold_type", "")),
            "acceptance_status": str(path.get("acceptance_status", "")),
            "rejection_reasons": list(path.get("rejection_reasons") or []),
            "path_shape_contract": dict(path.get("path_shape_contract") or {}),
            "animation_render_contract": dict(path.get("animation_render_contract") or {}),
            "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
            "caption_trace_available": bool(path.get("caption_trace")),
            "support_trace_available": bool(path.get("support_trace")),
            "visibility_profile_available": bool(path.get("visibility_profile")),
            "width_profile_available": bool(path.get("width_profile_px")),
            "region_boundary_trace_available": bool(path.get("region_boundary_trace")),
            "movement_scope": str(path.get("movement_scope", "")),
            "boundary_interaction": str(path.get("boundary_interaction", "")),
            "region_boundary_trace": dict(path.get("region_boundary_trace") or {}),
            "render_layers": list(path.get("render_layers") or []),
            "occlusion_trace": dict(path.get("occlusion_trace") or {}),
        },
        "confidence": conf,
        "notes": "one path-follow trajectory emitted for accepted path coverage",
    }


def _ensure_path_follow_trajectory_coverage(
    ctx: PipelineContext,
    traj_bundle: Dict[str, Any],
    paths: List[Dict[str, Any]],
    image_size: Dict[str, int],
    track_dir_name: str,
    cfg: Any,
) -> None:
    """Ensure every accepted path_id has at least one trajectory record."""
    hyps = list(traj_bundle.get("hypotheses") or [])
    linked = _trajectory_path_ids(traj_bundle)
    added: List[str] = []
    for path in paths:
        pid = str(path.get("path_id", "")).strip()
        if not pid or pid in linked:
            continue
        hyp = _path_follow_trajectory(ctx, path, image_size, track_dir_name, cfg)
        if not hyp:
            continue
        hyps.append(hyp)
        linked.add(pid)
        added.append(pid)
    traj_bundle["hypotheses"] = hyps
    traj_bundle["path_trajectory_coverage"] = {
        "accepted_path_count": len([p for p in paths if str(p.get("path_id", "")).strip()]),
        "linked_path_count": len(linked),
        "path_follow_trajectories_added": len(added),
        "missing_path_ids": [
            str(p.get("path_id", ""))
            for p in paths
            if str(p.get("path_id", "")).strip() and str(p.get("path_id", "")).strip() not in linked
        ],
    }


def _entity_for_trajectory(path: Dict[str, Any], key: str) -> Dict[str, Any]:
    ent = dict(path.get(key) or {})
    return {
        "kind": str(ent.get("type", "object")),
        "id": str(ent.get("id", "")),
        "role": "primary_actor" if key == "source_entity" else "counterpart",
        "display_label": str(ent.get("display_label") or ent.get("label") or ent.get("name") or ent.get("id", "")),
    }


def _fallback_trajectory_bundle_from_paths(
    ctx: PipelineContext,
    paths: List[Dict[str, Any]],
    image_size: Dict[str, int],
    track_dir_name: str,
    reason: str,
) -> Dict[str, Any]:
    """Build useful trajectory hypotheses directly from enriched paths."""
    ontology = load_action_ontology(None)
    hypotheses: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        pid = str(path.get("path_id", f"path_{idx:04d}"))
        states = _states_from_path(path)
        if len(states) < 2:
            continue
        conf = _float((path.get("scores") or {}).get("overall_confidence"), 0.0)
        motion = _primary_motion_from_path(path, ontology=ontology)
        duration = float(states[-1]["t_s"]) if states else 1.0
        hypotheses.append(
            {
                "trajectory_id": f"traj_{ctx.stem}_{pid}",
                "subject": _entity_for_trajectory(path, "source_entity"),
                "counterpart": _entity_for_trajectory(path, "target_entity"),
                "continues_from_path_id": pid,
                "time_model": {"type": "path_contract_fallback", "horizon_s": duration, "dt_s": duration / max(1, len(states) - 1)},
                "state_representation": "se2_root",
                "samples": [
                    {
                        "sample_id": f"traj_{ctx.stem}_{pid}_path_follow",
                        "sample_kind": "path_follow",
                        "weight": max(0.1, conf),
                        "states_t": states,
                        "evidence": {
                            "path_id": pid,
                            "source": "animation_export_path_fallback",
                            "manifold_type": str(path.get("manifold_type", "")),
                            "render_layers": list(path.get("render_layers") or []),
                            "path_shape_contract": dict(path.get("path_shape_contract") or {}),
                            "animation_render_contract": dict(path.get("animation_render_contract") or {}),
                            "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
                            "acceptance_status": str(path.get("acceptance_status", "")),
                            "rejection_reasons": list(path.get("rejection_reasons") or []),
                        },
                    }
                ],
                "constraints_used": {
                    "path_hypotheses_json": f"scene_graph/staged/{ctx.stem}_paths/path_hypotheses.json",
                    "path_traversability_speed_npy": str(ctx.path_exports.get("traversability_speed_npy", "")),
                },
                "motion_mode_candidates": _motion_candidates_from_path(path, cfg=None, ontology=ontology),
                "timeline_records": [
                    {
                        "t0_s": 0.0,
                        "t1_s": duration,
                        "motion": motion,
                        "relation_predicate": "",
                        "navigation_hint": str((path.get("navigation_zone_trace") or {}).get("dominant_motion_hint", "")),
                    }
                ],
                "action_context": {
                    "path_id": pid,
                    "action_family": str(path.get("action_family", "")),
                    "manifold_type": str(path.get("manifold_type", "")),
                    "acceptance_status": str(path.get("acceptance_status", "")),
                    "rejection_reasons": list(path.get("rejection_reasons") or []),
                    "path_shape_contract": dict(path.get("path_shape_contract") or {}),
                    "animation_render_contract": dict(path.get("animation_render_contract") or {}),
                    "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
                    "caption_trace_available": bool(path.get("caption_trace")),
                    "support_trace_available": bool(path.get("support_trace")),
                    "visibility_profile_available": bool(path.get("visibility_profile")),
                    "width_profile_available": bool(path.get("width_profile_px")),
                    "render_layers": list(path.get("render_layers") or []),
                },
                "confidence": conf,
                "notes": reason,
            }
        )
    return {
        "schema": "citv_trajectory_hypotheses_bundle_v1",
        "version": "1.0",
        "image_stem": ctx.stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "status": "fallback_from_enriched_paths",
        "hypotheses": hypotheses,
    }


def _fallback_animation_components_from_paths(
    paths: List[Dict[str, Any]],
    trajectory_bundle: Dict[str, Any],
    image_stem: str,
    image_size: Dict[str, int],
    track_dir_name: str,
    reason: str,
) -> Dict[str, Any]:
    path_by_id = {str(p.get("path_id", "")): p for p in paths if str(p.get("path_id", "")).strip()}
    components: List[Dict[str, Any]] = []
    for hyp in list((trajectory_bundle or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        tid = str(hyp.get("trajectory_id", ""))
        pid = str(hyp.get("continues_from_path_id", ""))
        path = path_by_id.get(pid, {})
        samples = list(hyp.get("samples") or [])
        states = []
        for sample in samples:
            if isinstance(sample, dict) and sample.get("states_t"):
                states = list(sample.get("states_t") or [])
                break
        components.append(
            {
                "component_id": f"{tid or pid}_component",
                "trajectory_id": tid,
                "path_id_linked": pid,
                "subject": dict(hyp.get("subject") or {}),
                "motion_mode_candidates": list(hyp.get("motion_mode_candidates") or []),
                "timeline_records": list(hyp.get("timeline_records") or []),
                "trajectory_points": states,
                "trajectory_samples": samples,
                "counterpart": hyp.get("counterpart"),
                "relation_guardrails": dict(hyp.get("relation_guardrails") or {}),
                "scene_semantic_constraints": {
                    "semantic_trace_available": bool(path.get("semantic_trace")),
                    "caption_trace_available": bool(path.get("caption_trace")),
                    "visibility_profile_available": bool(path.get("visibility_profile")),
                },
                "path_shape_contract": dict(path.get("path_shape_contract") or {}),
                "animation_render_contract": dict(path.get("animation_render_contract") or {}),
                "path_geometry_quality": dict(path.get("path_geometry_quality") or {}),
                "geometry_smoothing_provenance": dict(path.get("geometry_smoothing_provenance") or {}),
                "acceptance_status": str(path.get("acceptance_status", "")),
                "naturalness_score": _float((path.get("scores") or {}).get("overall_confidence"), 0.0),
                "rejection_reasons": list(path.get("rejection_reasons") or []),
                "uncertainty_reasons": list(path.get("uncertainty_reasons") or []),
                "contradiction_reasons": list(path.get("contradiction_reasons") or []),
                "grounding_evidence": dict(path.get("grounding_evidence") or {}),
                "generation_note": reason,
            }
        )
    return {
        "schema": "citv_animation_components_bundle_v1",
        "version": "1.0",
        "image_stem": image_stem,
        "track": track_dir_name,
        "image_size": dict(image_size),
        "status": "fallback_from_trajectory_hypotheses",
        "components": components,
    }


def _enrich_trajectory_bundle_with_action_context(
    traj_bundle: Dict[str, Any],
    paths: List[Dict[str, Any]],
    action_hypotheses: Optional[Dict[str, Any]],
) -> None:
    """Attach additive path/action render context to trajectory hypotheses."""
    path_by_id = {str(p.get("path_id", "")): p for p in paths if str(p.get("path_id", ""))}
    action_by_path = _actions_by_path(action_hypotheses)
    for hyp in traj_bundle.get("hypotheses") or []:
        if not isinstance(hyp, dict):
            continue
        pid = str(hyp.get("continues_from_path_id", ""))
        if not pid:
            for sample in hyp.get("samples") or []:
                ev = sample.get("evidence") or {}
                pid = str(ev.get("path_id", ""))
                if pid:
                    break
        path = path_by_id.get(pid, {})
        action = action_by_path.get(pid, {})
        hyp["action_context"] = {
            "path_id": pid,
            "action_id": str(action.get("action_id", "")),
            "action_name": str(action.get("action_name", "")),
            "action_status": str(action.get("action_status", path.get("acceptance_status", ""))),
            "action_family": str(action.get("action_family", path.get("action_family", ""))),
            "manifold_type": str(action.get("manifold_type", path.get("manifold_type", ""))),
            "path_shape_contract": dict(path.get("path_shape_contract") or {}),
            "animation_render_contract": dict(path.get("animation_render_contract") or {}),
            "path_geometry_quality": dict(path.get("path_geometry_quality") or {}),
            "geometry_smoothing_provenance": dict(path.get("geometry_smoothing_provenance") or {}),
            "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
            "acceptance_status": str(path.get("acceptance_status", "")),
            "rejection_reasons": list(path.get("rejection_reasons") or []),
            "uncertainty_reasons": list(path.get("uncertainty_reasons") or []),
            "contradiction_reasons": list(path.get("contradiction_reasons") or []),
            "grounding_evidence": dict(path.get("grounding_evidence") or {}),
            "caption_trace_available": bool(path.get("caption_trace")),
            "support_trace_available": bool(path.get("support_trace")),
            "visibility_profile_available": bool(path.get("visibility_profile")),
            "width_profile_available": bool(path.get("width_profile_px")),
            "region_boundary_trace_available": bool(path.get("region_boundary_trace")),
            "movement_scope": str(path.get("movement_scope", "")),
            "boundary_interaction": str(path.get("boundary_interaction", "")),
            "region_boundary_trace": dict(path.get("region_boundary_trace") or {}),
            "render_layers": list(path.get("render_layers") or []),
            "occlusion_trace": dict(path.get("occlusion_trace") or {}),
        }
        for sample in hyp.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            evidence = dict(sample.get("evidence") or {})
            evidence.setdefault("action_context_available", bool(action or path))
            evidence.setdefault("manifold_type", hyp["action_context"].get("manifold_type", ""))
            evidence.setdefault("movement_scope", hyp["action_context"].get("movement_scope", ""))
            evidence.setdefault("boundary_interaction", hyp["action_context"].get("boundary_interaction", ""))
            evidence.setdefault("render_layers", hyp["action_context"].get("render_layers", []))
            evidence.setdefault("path_shape_contract", hyp["action_context"].get("path_shape_contract", {}))
            evidence.setdefault("animation_render_contract", hyp["action_context"].get("animation_render_contract", {}))
            evidence.setdefault("direction_profile", hyp["action_context"].get("direction_profile", {}))
            evidence.setdefault("acceptance_status", hyp["action_context"].get("acceptance_status", ""))
            evidence.setdefault("uncertainty_reasons", hyp["action_context"].get("uncertainty_reasons", []))
            evidence.setdefault("grounding_evidence", hyp["action_context"].get("grounding_evidence", {}))
            sample["evidence"] = evidence


def _enrich_animation_components_with_action_context(
    anim_components: Dict[str, Any],
    paths: List[Dict[str, Any]],
    action_hypotheses: Optional[Dict[str, Any]],
) -> None:
    path_by_id = {str(p.get("path_id", "")): p for p in paths if str(p.get("path_id", ""))}
    action_by_path = _actions_by_path(action_hypotheses)
    action_ref = ""
    if action_hypotheses:
        action_ref = "action_hypotheses.json"
    anim_components["action_hypotheses_ref"] = action_ref
    anim_components["action_context_available"] = bool(action_hypotheses)
    for comp in anim_components.get("components") or []:
        if not isinstance(comp, dict):
            continue
        pid = str(comp.get("path_id_linked", ""))
        path = path_by_id.get(pid, {})
        action = action_by_path.get(pid, {})
        comp["action_context"] = {
            "action_id": str(action.get("action_id", "")),
            "action_name": str(action.get("action_name", "")),
            "action_status": str(action.get("action_status", path.get("acceptance_status", ""))),
            "action_family": str(action.get("action_family", path.get("action_family", ""))),
            "manifold_type": str(action.get("manifold_type", path.get("manifold_type", ""))),
            "path_shape_contract": dict(path.get("path_shape_contract") or {}),
            "animation_render_contract": dict(path.get("animation_render_contract") or {}),
            "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
            "acceptance_status": str(path.get("acceptance_status", "")),
            "rejection_reasons": list(path.get("rejection_reasons") or []),
            "uncertainty_reasons": list(path.get("uncertainty_reasons") or []),
            "contradiction_reasons": list(path.get("contradiction_reasons") or []),
            "grounding_evidence": dict(path.get("grounding_evidence") or {}),
            "width_profile_px": list(path.get("width_profile_px") or [])[:32],
            "visibility_profile": list(path.get("visibility_profile") or [])[:32],
            "caption_trace": dict(path.get("caption_trace") or {}),
            "movement_scope": str(path.get("movement_scope", "")),
            "boundary_interaction": str(path.get("boundary_interaction", "")),
            "region_boundary_trace": dict(path.get("region_boundary_trace") or {}),
            "render_layers": list(path.get("render_layers") or []),
        }
        comp["path_shape_contract"] = dict(path.get("path_shape_contract") or {})
        comp["animation_render_contract"] = dict(path.get("animation_render_contract") or {})
        comp["path_geometry_quality"] = dict(path.get("path_geometry_quality") or comp.get("path_geometry_quality") or {})
        comp["geometry_smoothing_provenance"] = dict(path.get("geometry_smoothing_provenance") or comp.get("geometry_smoothing_provenance") or {})
        comp["acceptance_status"] = str(path.get("acceptance_status", ""))
        comp["rejection_reasons"] = list(path.get("rejection_reasons") or comp.get("rejection_reasons") or [])
        comp["uncertainty_reasons"] = list(path.get("uncertainty_reasons") or comp.get("uncertainty_reasons") or [])
        comp["contradiction_reasons"] = list(path.get("contradiction_reasons") or comp.get("contradiction_reasons") or [])
        comp["grounding_evidence"] = dict(path.get("grounding_evidence") or comp.get("grounding_evidence") or {})


def _write_trajectory_overlay_image(
    ctx: Any,
    traj_bundle: Dict[str, Any],
    paths: List[Dict[str, Any]],
    paths_root: Path,
) -> None:
    """Draw top ≤15 trajectory rollouts labelled on the scene image and save as PNG."""
    try:
        import cv2
    except ImportError:
        return
    if ctx.img_bgr is None or not paths:
        return

    ranked = sorted(
        paths,
        key=lambda p: float((p.get("scores") or {}).get("overall_confidence", 0.0)),
        reverse=True,
    )[:_MAX_LABELLED_PATHS]

    canvas = ctx.img_bgr.copy()
    h, w = canvas.shape[:2]

    try:
        from ..pathing.path_colors import bgr_list_with_min_hue_separation, bgr_from_stable_id_trajectory
        pid_list = [str(p.get("path_id", f"t{i}")) for i, p in enumerate(ranked)]
        colors: List[Tuple[int, int, int]] = bgr_list_with_min_hue_separation(pid_list, bgr_from_stable_id_trajectory)
    except Exception:
        colors = [(80, 180, 255)] * len(ranked)

    traj_hyps = list(traj_bundle.get("hypotheses") or [])
    traj_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for hyp in traj_hyps:
        pid_key = str((hyp.get("action_context") or {}).get("path_id", ""))
        if not pid_key:
            pid_key = str(hyp.get("continues_from_path_id", ""))
        if pid_key and hyp.get("trajectory_points"):
            traj_by_pid[pid_key] = list(hyp["trajectory_points"])

    legend_entries: List[Tuple[int, Tuple[int, int, int], float, str]] = []
    for rank_i, (path, color) in enumerate(zip(ranked, colors), start=1):
        pid = str(path.get("path_id", ""))
        conf = float((path.get("scores") or {}).get("overall_confidence", 0.0))

        traj_pts = traj_by_pid.get(pid, [])
        if traj_pts:
            pts: List[Tuple[int, int]] = [
                (max(0, min(w - 1, int(round(float(t.get("x_px", 0)))))),
                 max(0, min(h - 1, int(round(float(t.get("y_px", 0)))))))
                for t in traj_pts if isinstance(t, dict)
            ]
        else:
            raw = _path_polyline_2d(path)
            pts = [
                (max(0, min(w - 1, int(round(float(xy[0]))))),
                 max(0, min(h - 1, int(round(float(xy[1]))))))
                for xy in raw if isinstance(xy, (list, tuple)) and len(xy) >= 2
            ]

        if len(pts) < 2:
            continue

        ov = canvas.copy()
        cv2.polylines(ov, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.75, canvas, 0.25, 0.0, dst=canvas)
        cv2.arrowedLine(canvas, pts[-2], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.30)

        mid_idx = max(0, len(pts) // 2 - 1)
        mx, my = pts[mid_idx]
        label = f"T{rank_i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.rectangle(canvas, (mx - 2, my - th - 2), (mx + tw + 4, my + 3), (0, 0, 0), -1)
        cv2.putText(canvas, label, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        legend_entries.append((rank_i, color, conf, pid))

    leg_x, leg_y, leg_w, leg_line = 8, 8, 270, 20
    leg_h = 22 + len(legend_entries) * leg_line
    ov2 = canvas.copy()
    cv2.rectangle(ov2, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (15, 15, 15), -1)
    cv2.addWeighted(ov2, 0.68, canvas, 0.32, 0.0, dst=canvas)
    cv2.putText(
        canvas, f"Top {len(legend_entries)} trajectories (max {_MAX_LABELLED_PATHS})",
        (leg_x + 4, leg_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    for rank_i, color, conf, pid in legend_entries:
        ey = leg_y + 22 + (rank_i - 1) * leg_line
        cv2.line(canvas, (leg_x + 4, ey + 6), (leg_x + 22, ey + 6), color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas, f"T{rank_i} ({conf:.2f}) {pid}"[:46],
            (leg_x + 26, ey + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA,
        )

    out_path = paths_root / "trajectory_overlay.png"
    try:
        cv2.imwrite(str(out_path), canvas)
        ctx.path_exports["trajectory_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/trajectory_overlay.png"
        )
    except Exception as exc:
        print(f"  [AnimationExport] trajectory_overlay write failed: {exc}")


def _actions_by_path(action_hypotheses: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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


_TARGET_VIDEO_SECONDS = 120.0  # 2-minute target per QA video
_MIN_CANDIDATE_SECONDS = 0.4   # keep candidate clips visible but practical


class _CfgProxy:
    """Thin overlay on top of the real config for animation-specific overrides."""

    def __init__(self, base_cfg: Any, **overrides: Any) -> None:
        self._base = base_cfg
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._overrides:
            return self._overrides[name]
        if self._base is not None:
            return getattr(self._base, name)
        raise AttributeError(name)


def _plan_video_timing(n_paths: int, base_delim_s: float) -> tuple:
    """Return (candidate_seconds, delimiter_seconds, n_shown) for a 5-min budget."""
    if n_paths <= 0:
        return _TARGET_VIDEO_SECONDS, base_delim_s, 0
    # Reduce delimiter when many paths to preserve screen time for actual content.
    delim_s = base_delim_s if n_paths <= 20 else min(base_delim_s, 0.3)
    # Solve: n × cand_s + (n-1) × delim_s = 300
    cand_s = (_TARGET_VIDEO_SECONDS - (n_paths - 1) * delim_s) / n_paths
    if cand_s < _MIN_CANDIDATE_SECONDS:
        # Budget too tight — fit as many paths as possible at minimum duration.
        # n_max × (min_cand + delim) - delim = 300 → n_max = (300 + delim) / (min_cand + delim)
        n_shown = max(1, int((_TARGET_VIDEO_SECONDS + delim_s) / (_MIN_CANDIDATE_SECONDS + delim_s)))
        cand_s = (_TARGET_VIDEO_SECONDS - (n_shown - 1) * delim_s) / n_shown
    else:
        n_shown = n_paths
    return float(cand_s), float(delim_s), int(n_shown)


def _is_primary_path(path: Dict[str, Any]) -> bool:
    ptype = str(path.get("path_type", "")).strip().lower()
    plevel = str(path.get("path_level", "")).strip().lower()
    if ptype.startswith("staged_objreg_"):
        return False
    if ptype.startswith("manifold_"):
        return bool(path.get("polyline_2d") or path.get("display_polyline_2d"))
    if plevel in {"mask", "region", "object", "global"}:
        return True
    return True


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Export trajectory contracts and optional QA videos for all FMM paths."""
    cfg = getattr(pipeline, "config", None)
    qa_enabled = bool(getattr(cfg, "path_animation_qa_enabled", True) if cfg else True)

    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    paths_root = staged_dir / f"{ctx.stem}_paths"
    paths_root.mkdir(parents=True, exist_ok=True)
    hyp_path = paths_root / "path_hypotheses.json"

    if not hyp_path.exists():
        return ctx

    try:
        hyp_data = json.loads(hyp_path.read_text(encoding="utf-8"))
        paths = list(hyp_data.get("paths") or hyp_data.get("hypotheses") or [])
        if not paths:
            return ctx
    except Exception as exc:
        print(f"  [AnimationExport] path_hypotheses load failed: {exc}")
        return ctx

    objects: List[Dict[str, Any]] = list(ctx.extra.get("objects", []))
    relations: List[Dict[str, Any]] = list(ctx.relations)
    image_size = {"width": ctx.width, "height": ctx.height}
    track_dir_name = "staged"

    compliance = bool(getattr(cfg, "path_contract_compliance_mode", False)) if cfg else False
    # Compute per-path timing to fill 5 minutes with as many paths as possible.
    base_delim_s = float(getattr(cfg, "path_animation_qa_delimiter_seconds", 0.55)) if cfg else 0.55
    cand_s, delim_s, n_shown = _plan_video_timing(len(paths), base_delim_s)
    if compliance:
        n_shown = len(paths)
    elif bool(getattr(cfg, "path_qa_perf_mode", False)) if cfg else False:
        n_shown = len(paths)
        cand_s = float(getattr(cfg, "path_animation_qa_candidate_seconds", 4.0)) if cfg else 4.0
        delim_s = base_delim_s
    print(
        f"  [AnimationExport] {len(paths)} paths → showing {n_shown} "
        f"× {cand_s:.1f}s + {delim_s:.2f}s delimiter = "
        f"{n_shown * cand_s + max(0, n_shown - 1) * delim_s:.0f}s per video"
    )
    qa_modes = list(getattr(cfg, "path_animation_qa_modes", [24, 120])) if cfg else [24, 120]
    if bool(getattr(cfg, "path_animation_qa_include_120fps", True)) if cfg else True:
        if 120 not in qa_modes:
            qa_modes.append(120)
    normalized_modes: List[int] = []
    for mode in qa_modes:
        try:
            fps_mode = int(mode)
        except (TypeError, ValueError):
            continue
        if fps_mode > 0:
            normalized_modes.append(fps_mode)
    qa_modes = sorted(set(normalized_modes or [24, 120]))

    # Config proxy: override pacing while keeping panel-level candidate caps
    # bounded for robust video generation.
    render_cfg = _CfgProxy(
        cfg,
        path_animation_qa_modes=qa_modes,
        path_animation_qa_candidate_seconds=cand_s,
        path_animation_qa_delimiter_seconds=delim_s,
        path_animation_qa_max_candidates_per_panel=min(
            int(getattr(cfg, "path_atlas_panel_size", 10)) if cfg else 10,
            int(getattr(cfg, "path_scene_trajectory_batch_size", 10)) if cfg else 10,
        ),
    )

    # Build trajectory bundle. This contract is required even when heavy QA
    # rendering is disabled or fails.
    #
    # Plan §2.8: lift the over-strict subject cap and "person/vehicle/animal"
    # mover filter so every path-paired subject becomes a trajectory. Also
    # bump the max subjects to >=128 so a typical scene with 18 objects no
    # longer collapses to 7 trajectories.
    n_objects = max(1, len(objects))
    _max_subj = max(
        int(getattr(cfg, "trajectory_hypotheses_max_subjects", 8)) if cfg else 8,
        min(256, n_objects * 2),
    )
    if bool(getattr(cfg, "path_qa_perf_mode", False)) if cfg else False:
        _max_subj = max(_max_subj, min(4096, len(paths) * 4))
    traj_cfg = _CfgProxy(
        cfg,
        trajectory_hypotheses_include_all_objects=True,
        trajectory_hypotheses_max_subjects=_max_subj,
    )
    traj_bundle: Optional[Dict[str, Any]] = None
    try:
        from ..export import build_trajectory_bundle
        nav_zones = ctx.extra.get("navigation_zones")
        speed_arr = ctx.extra.get("path_traversability_speed_map")
        if speed_arr is None:
            rel_speed = str(ctx.path_exports.get("traversability_speed_npy", ""))
            if rel_speed:
                try:
                    speed_arr = np.load(ctx.output_dir / rel_speed)
                except Exception:
                    speed_arr = None
        path_cost_map = None
        if speed_arr is not None:
            try:
                path_cost_map = 1.0 - np.clip(np.asarray(speed_arr, dtype=np.float32), 0.0, 1.0)
            except Exception:
                path_cost_map = None

        traj_bundle = build_trajectory_bundle(
            objects,
            relations,
            ctx.stem,
            image_size,
            track_dir_name,
            ctx.stem,
            traj_cfg,
            metric_depth_m=ctx.metric_depth,
            traversability_speed_npy_rel=str(ctx.path_exports.get("traversability_speed_npy", "")),
            ranked_paths=paths,
            navigation_zones=nav_zones,
            path_cost_map=path_cost_map,
        )
    except Exception as exc:
        reason = f"trajectory bundle build failed: {type(exc).__name__}: {exc}"
        print(f"  [AnimationExport] {reason}")
        _record_export_failure(ctx, "trajectory_hypotheses_json", reason)
        traj_bundle = _fallback_trajectory_bundle_from_paths(ctx, paths, image_size, track_dir_name, reason)

    if not isinstance(traj_bundle, dict):
        reason = "trajectory bundle builder returned no payload"
        _record_export_failure(ctx, "trajectory_hypotheses_json", reason)
        traj_bundle = _fallback_trajectory_bundle_from_paths(ctx, paths, image_size, track_dir_name, reason)
    if paths and not list(traj_bundle.get("hypotheses") or []):
        reason = "trajectory bundle contained no hypotheses for non-empty path set"
        _record_export_failure(ctx, "trajectory_hypotheses_json", reason)
        traj_bundle = _fallback_trajectory_bundle_from_paths(ctx, paths, image_size, track_dir_name, reason)

    _ensure_path_follow_trajectory_coverage(ctx, traj_bundle, paths, image_size, track_dir_name, cfg)
    _enrich_trajectory_bundle_with_action_context(traj_bundle, paths, ctx.action_hypotheses)
    _write_json(traj_bundle, paths_root / "trajectory_hypotheses.json")
    ctx.path_exports["trajectory_hypotheses_json"] = (
        f"scene_graph/staged/{ctx.stem}_paths/trajectory_hypotheses.json"
    )
    _write_trajectory_overlay_image(ctx, traj_bundle, paths, paths_root)

    ranked_for_batches = sorted(
        [p for p in paths if _is_primary_path(p)],
        key=lambda p: float((p.get("scores") or {}).get("overall_confidence", 0.0)),
        reverse=True,
    )
    paths_rel = f"scene_graph/staged/{ctx.stem}_paths"
    batch_meta: List[Dict[str, Any]] = []
    try:
        from ..pathing.path_atlas_canvas import (
            write_scene_path_trajectories_overview,
            write_scene_path_trajectory_batches,
        )

        overview_abs = write_scene_path_trajectories_overview(
            paths_root_dir=paths_root,
            base_img_bgr=ctx.img_bgr,
            label_map=ctx.region_label_map,
            objects=list(ctx.extra.get("objects", [])),
            metric_depth_m=ctx.metric_depth,
            ranked_paths=ranked_for_batches,
            traj_bundle=traj_bundle,
            cfg=cfg,
        )
        if overview_abs:
            ctx.path_exports["path_trajectories_image"] = f"{paths_rel}/{Path(overview_abs).name}"

        written_batches, batch_meta = write_scene_path_trajectory_batches(
            paths_root_dir=paths_root,
            base_img_bgr=ctx.img_bgr,
            label_map=ctx.region_label_map,
            objects=list(ctx.extra.get("objects", [])),
            metric_depth_m=ctx.metric_depth,
            ranked_paths=ranked_for_batches,
            traj_bundle=traj_bundle,
            cfg=cfg,
        )
        for i, abs_p in enumerate(written_batches):
            ctx.path_exports[f"path_trajectories_batch_{i:03d}_image"] = (
                f"{paths_rel}/{Path(abs_p).name}"
            )
        if batch_meta:
            _write_json(
                {
                    "schema": "citv_path_trajectories_batches_v1",
                    "stem": ctx.stem,
                    "path_count": len(ranked_for_batches),
                    "batch_count": len(batch_meta),
                    "overview_image": "path_trajectories.png" if overview_abs else "",
                    "batches": batch_meta,
                },
                paths_root / "path_trajectories_batches.json",
            )
            ctx.path_exports["path_trajectories_batches_json"] = (
                f"{paths_rel}/path_trajectories_batches.json"
            )
        if written_batches:
            print(f"  [AnimationExport] path trajectory scene batches: {len(written_batches)} PNG(s)")
    except Exception as exc:
        print(f"  [AnimationExport] path_trajectories batch PNGs failed: {exc}")

    try:
        from ..pathing.path_visual_qa_export import export_path_visual_qa_json_and_md

        export_path_visual_qa_json_and_md(
            paths_root=paths_root,
            stem=ctx.stem,
            width=ctx.width,
            height=ctx.height,
            ranked_paths=ranked_for_batches,
            traj_bundle=traj_bundle,
            action_hypotheses=ctx.action_hypotheses,
            batch_meta=batch_meta,
            dropped_paths=list((ctx.extra or {}).get("path_dropped", []) or []),
        )
        ctx.path_exports["path_visual_qa_json"] = f"{paths_rel}/path_visual_qa.json"
        ctx.path_exports["path_visual_qa_md"] = f"{paths_rel}/path_visual_qa.md"
    except Exception as exc:
        print(f"  [AnimationExport] path_visual_qa json/md failed: {exc}")

    # Plan §2.9: rebuild motion contract overlay now that trajectories exist
    # so the magenta instant-prior arrows show on the QA panel.
    try:
        from ..visualization.motion_contract_overlay import write_motion_contract_overlay
        ranked_for_overlay = sorted(
            paths,
            key=lambda r: float((r.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=True,
        )
        write_motion_contract_overlay(
            ctx.img_bgr,
            ranked_for_overlay,
            traj_bundle,
            paths_root / "motion_contracts_overlay.png",
            cfg=cfg,
            support_mask=ctx.extra.get("support_mask"),
            object_affordances=ctx.object_affordances,
            metric_depth_m=ctx.metric_depth,
        )
        ctx.path_exports["motion_contracts_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/motion_contracts_overlay.png"
        )
    except Exception as exc:
        print(f"  [AnimationExport] motion_contracts_overlay failed: {exc}")

    # Build animation components independently; failures here must not erase
    # the trajectory contract written above.
    try:
        from ..export import build_animation_components_bundle

        anim_components = build_animation_components_bundle(
            paths=paths,
            trajectory_bundle=traj_bundle,
            image_stem=ctx.stem,
            image_size=image_size,
            track_dir_name=track_dir_name,
        )
    except Exception as exc:
        reason = f"animation components build failed: {type(exc).__name__}: {exc}"
        print(f"  [AnimationExport] {reason}")
        _record_export_failure(ctx, "animation_components_json", reason)
        anim_components = _fallback_animation_components_from_paths(
            paths, traj_bundle, ctx.stem, image_size, track_dir_name, reason
        )

    if paths and not list((anim_components or {}).get("components") or []):
        reason = "animation components contained no components for non-empty trajectory set"
        _record_export_failure(ctx, "animation_components_json", reason)
        anim_components = _fallback_animation_components_from_paths(
            paths, traj_bundle, ctx.stem, image_size, track_dir_name, reason
        )

    _enrich_animation_components_with_action_context(anim_components, paths, ctx.action_hypotheses)
    _write_json(anim_components, paths_root / "animation_components.json")
    ctx.path_exports["animation_components_json"] = (
        f"scene_graph/staged/{ctx.stem}_paths/animation_components.json"
    )

    # Plan §2.8: surface trajectory rejection reasons so path_diagnostics.json
    # can show *why* a path didn't yield a usable component (e.g. low_naturalness).
    try:
        rejection_log: List[Dict[str, Any]] = []
        for comp in (anim_components or {}).get("components") or []:
            reasons = list(comp.get("rejection_reasons") or [])
            if not reasons:
                continue
            rejection_log.append({
                "path_id": str(comp.get("path_id_linked", "")),
                "trajectory_id": str(comp.get("trajectory_id", "")),
                "naturalness_score": float(comp.get("naturalness_score", 0.0) or 0.0),
                "reasons": reasons,
            })
        if rejection_log:
            ctx.extra.setdefault("trajectory_rejections", []).extend(rejection_log)
    except Exception:
        pass

    # Build animation plan (kinematic segment timeline per path).
    try:
        anim_plan = _build_animation_plan(paths, cfg)
        _write_json(anim_plan, paths_root / "animation_plan.json")
        ctx.path_exports["animation_plan_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/animation_plan.json"
        )
    except Exception as exc:
        print(f"  [AnimationExport] animation_plan build failed: {exc}")

    # Build path atlas manifest with all n_shown paths in panel 0.
    try:
        panel_size = int(getattr(cfg, "path_atlas_panel_size", 10)) if cfg else 10
        atlas_manifest = _build_path_atlas_manifest(paths[:n_shown], ctx.stem, track_dir_name, panel_size=panel_size)
        _write_json(atlas_manifest, paths_root / "path_atlas_manifest.json")
        ctx.path_exports["path_atlas_manifest_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/path_atlas_manifest.json"
        )
    except Exception as exc:
        print(f"  [AnimationExport] atlas_manifest build failed: {exc}")
        return ctx

    # Run the QA video renderer with the pacing-aware config proxy.
    if qa_enabled:
        try:
            from ..visualization.animation_qa_renderer import export_animation_qa

            qa_out = export_animation_qa(
                paths_root_dir=paths_root,
                img_bgr=ctx.img_bgr,
                metric_depth_m=ctx.metric_depth,
                cfg=render_cfg,
                region_label_map=ctx.region_label_map,
            )
            paths_root_rel = f"scene_graph/staged/{ctx.stem}_paths"
            for k, v in qa_out.items():
                ctx.path_exports[k] = f"{paths_root_rel}/{v}"
            n_videos = sum(1 for k in qa_out if k.startswith("animation_qa_manifest"))
            print(f"  [AnimationExport] {n_videos} QA video mode(s) written for {ctx.stem}")
        except Exception as exc:
            reason = f"export_animation_qa failed: {type(exc).__name__}: {exc}"
            print(f"  [AnimationExport] {reason}")
            _record_export_failure(ctx, "animation_qa", reason)

    # Full path atlas panels — written after QA renderer so it may overwrite the simplified manifest.
    try:
        from ..pathing.path_atlas_export import export_trajs_upt_atlas
        paths_rel_base = f"scene_graph/staged/{ctx.stem}_paths"
        path_num_by_id = {str(p.get("path_id", "")): i for i, p in enumerate(paths)}
        atlas_out = export_trajs_upt_atlas(
            paths_root_dir=paths_root,
            path_stem=ctx.stem,
            track_dir_name=track_dir_name,
            paths_sorted=paths,
            paths_recommended=paths,
            descriptions_by_id={},
            path_num_by_id=path_num_by_id,
            min_conf=0.0,
            traj_bundle=traj_bundle,
            cfg=cfg,
            width=ctx.width,
            height=ctx.height,
            export_desc=False,
            animation_rel=f"{paths_rel_base}/animation_plan.json",
            insertion_rel="",
            trajectory_rel=f"{paths_rel_base}/trajectory_hypotheses.json",
            hyp_rel=f"{paths_rel_base}/path_hypotheses.json",
            desc_rel="",
            diag_rel="",
            semantic_rel="",
            pair_rel="",
            descriptions_md_rel="",
            cost_png_rel=str(ctx.path_exports.get("traversability_speed_png", "")),
            cost_npy_rel=str(ctx.path_exports.get("traversability_speed_npy", "")),
            trav_png_rel=str(ctx.path_exports.get("traversability_speed_png", "")),
            trav_npy_rel=str(ctx.path_exports.get("traversability_speed_npy", "")),
            img_bgr=ctx.img_bgr,
            label_map=ctx.region_label_map,
            objects=list(ctx.extra.get("objects", [])),
            metric_depth_m=ctx.metric_depth,
        )
        for k, v in atlas_out.items():
            ctx.path_exports[k] = v
        print(f"  [AnimationExport] path atlas: {len(atlas_out)} outputs written")
    except Exception as exc:
        print(f"  [AnimationExport] path atlas export failed: {exc}")

    # Motion contract overlay.
    try:
        from ..visualization.motion_contract_overlay import write_motion_contract_overlay
        write_motion_contract_overlay(
            ctx.img_bgr, paths, traj_bundle or {"hypotheses": []},
            paths_root / "motion_contract_overlay.png", cfg,
        )
        ctx.path_exports["motion_contract_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/motion_contract_overlay.png"
        )
    except Exception as exc:
        print(f"  [AnimationExport] motion_contract_overlay failed: {exc}")

    try:
        from ..pathing.path_contract_compliance import write_path_updates_compliance

        compliance_out = write_path_updates_compliance(paths_root)
        if compliance_out.get("json"):
            ctx.path_exports["path_updates_compliance_json"] = f"{paths_rel}/path_updates_compliance.json"
        if compliance_out.get("md"):
            ctx.path_exports["path_updates_compliance_md"] = f"{paths_rel}/path_updates_compliance.md"
    except Exception as exc:
        print(f"  [AnimationExport] path_updates_compliance failed: {exc}")

    try:
        from ..pathing.final_paths_export import export_final_paths_bundle

        final_out = export_final_paths_bundle(
            ctx=ctx,
            pipeline=pipeline,
            paths_root=paths_root,
            ranked_paths=ranked_for_batches,
            traj_bundle=traj_bundle,
        )
        for k, v in final_out.items():
            ctx.path_exports[k] = v
        print(f"  [AnimationExport] final paths bundle written: {final_out.get('final_paths_json', '')}")
    except Exception as exc:
        print(f"  [AnimationExport] final paths export failed: {exc}")

    return ctx
