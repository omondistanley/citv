"""Stage: action hypotheses and non-line action manifolds.

The path stage answers "where can something move?" This stage adds "what
kind of action is this motion trying to express?" and emits additive action
contracts that can point at paths, masks, contacts, volumes, portals, and
effect fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..action_ontology import dict_section, list_section, load_action_ontology, number
from ..pipeline_context import PipelineContext


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    cfg = getattr(pipeline, "config", None)
    enabled = bool(getattr(cfg, "export_action_hypotheses", True)) if cfg else True
    if not enabled:
        return ctx
    ontology = load_action_ontology(cfg)

    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    paths_root = staged_dir / f"{ctx.stem}_paths"
    paths_root.mkdir(parents=True, exist_ok=True)

    paths = _load_paths(paths_root / "path_hypotheses.json")
    actions: List[Dict[str, Any]] = []
    actions.extend(_path_actions(ctx, paths))
    actions.extend(_affordance_actions(ctx, cfg, ontology))
    actions = _rank_actions(actions)
    status_counts: Dict[str, int] = {}
    for action in actions:
        status = str(action.get("action_status", "low_confidence"))
        status_counts[status] = status_counts.get(status, 0) + 1

    bundle = {
        "schema": "citv_action_hypotheses_v2",
        "version": "2.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "image_size": {"width": int(ctx.width), "height": int(ctx.height)},
        "caption_aware": ctx.caption_evidence is not None,
        "source_refs": {
            "path_hypotheses_json": ctx.path_exports.get("path_hypotheses_json", ""),
            "scene_affordances_json": ctx.path_exports.get("scene_affordances_json", ""),
            "object_affordances_json": ctx.path_exports.get("object_affordances_json", ""),
            "mask_affordances_json": ctx.path_exports.get("mask_affordances_json", ""),
            "caption_evidence_json": ctx.path_exports.get("caption_evidence_json", ""),
        },
        "hypotheses": actions,
        "additive_fields_dropped": ["actions"],
        "compat_note": (
            "v2 removes the duplicate `actions` key. Read `hypotheses`."
        ),
        "summary": {
            "action_count": len(actions),
            "path_action_count": sum(1 for a in actions if a.get("source_type") == "path"),
            "affordance_action_count": sum(1 for a in actions if a.get("source_type") == "affordance"),
            "manifold_types": sorted(set(str(a.get("manifold_type", "")) for a in actions if a.get("manifold_type"))),
            "status_counts": status_counts,
            "accepted_count": status_counts.get("accepted", 0),
            "plausible_uncertain_count": status_counts.get("plausible_uncertain", 0),
            "low_confidence_count": status_counts.get("low_confidence", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "rejected_or_low_confidence_count": sum(
                1
                for a in actions
                if str(a.get("action_status", "")) in {"plausible_uncertain", "low_confidence", "rejected"}
            ),
        },
    }

    overlay_name = _write_action_overlay(ctx, actions, paths_root, ontology)
    if overlay_name:
        bundle["qa_overlay_image"] = f"scene_graph/staged/{ctx.stem}_paths/{overlay_name}"

    _write_json(bundle, paths_root / "action_hypotheses.json")
    ctx.action_hypotheses = bundle
    ctx.extra["action_hypotheses"] = bundle
    ctx.path_exports["action_hypotheses_json"] = (
        f"scene_graph/staged/{ctx.stem}_paths/action_hypotheses.json"
    )
    if overlay_name:
        ctx.path_exports["action_manifold_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/{overlay_name}"
        )
    print(f"  [ActionExport] {ctx.stem}: {len(actions)} action hypotheses written")
    return ctx


def _load_paths(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [dict(p) for p in (data.get("paths") or data.get("hypotheses") or []) if isinstance(p, dict)]
    except Exception:
        return []


def _path_actions(ctx: PipelineContext, paths: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        pid = str(path.get("path_id", f"path_{idx:04d}"))
        scores = dict(path.get("scores") or {})
        conf = _float(scores.get("overall_confidence"), 0.0)
        action_id = f"action_{pid}_locomotion"
        contract_status = path.get("contract_status") if isinstance(path.get("contract_status"), dict) else {}
        rejection_reasons: List[str] = list(contract_status.get("rejection_reasons") or path.get("rejection_reasons") or [])
        uncertainty_reasons: List[str] = list(contract_status.get("uncertainty_reasons") or path.get("uncertainty_reasons") or [])
        if conf < 0.18:
            rejection_reasons.append("low_path_overall_confidence")
        if not bool(path.get("polyline_3d")):
            uncertainty_reasons.append("missing_polyline_3d")
        path_status = str(path.get("acceptance_status", ""))
        status = path_status if path_status in {"accepted", "plausible_uncertain", "low_confidence", "rejected"} else _action_status(conf, rejection_reasons, uncertainty_reasons, min_accept=0.42, min_low=0.18)
        actions.append({
            "action_id": action_id,
            "source_type": "path",
            "action_status": status,
            "action_family": str(path.get("action_family", "locomotion")),
            "action_name": _path_motion_name(path),
            "manifold_type": str(path.get("manifold_type", "ribbon_path")),
            "subject": dict(path.get("source_entity") or {}),
            "target": dict(path.get("target_entity") or {}),
            "grounding": {
                "path_id": pid,
                "pixel_grounded": bool(path.get("polyline_2d")),
                "depth_grounded": bool(path.get("polyline_3d")),
                "caption_grounded": bool(path.get("caption_trace")),
                "visibility_grounded": bool(path.get("visibility_profile")),
                "local_action_evidence_confidence": _float(scores.get("local_action_evidence_confidence"), 0.0),
                "support_grounding_confidence": _float(scores.get("support_grounding_confidence"), 0.0),
                "entity_anchor_confidence": _float(scores.get("entity_anchor_confidence"), 0.0),
                "source_evidence": dict(path.get("grounding_evidence") or {}),
                "uncertainty_reasons": list(uncertainty_reasons),
                "contradiction_reasons": list(path.get("contradiction_reasons") or rejection_reasons),
            },
            "manifold": {
                "type": str(path.get("manifold_type", "ribbon_path")),
                "centerline_path": path.get("display_polyline_2d") or path.get("polyline_2d_validated") or path.get("polyline_2d", []),
                "raw_centerline_path": path.get("polyline_2d_raw") or path.get("polyline_2d", []),
                "polyline_3d": path.get("polyline_3d", []),
                "display_polyline_3d": path.get("display_polyline_3d", []),
                "width_profile_px": path.get("width_profile_px", []),
                "support_trace": path.get("support_trace", []),
                "semantic_trace": path.get("semantic_trace", []),
                "caption_trace": path.get("caption_trace", {}),
                "visibility_profile": path.get("visibility_profile", []),
                "path_shape_contract": path.get("path_shape_contract", {}),
                "path_geometry_quality": path.get("path_geometry_quality", {}),
            },
            "render_contract": path.get("animation_render_contract") or {
                "render_layers": path.get("render_layers", []),
                "occluder_ids": _occluder_ids(path),
                "alpha_policy": "visibility_profile",
                "scale_policy": "depth_width_profile",
            },
            "path_shape_contract": path.get("path_shape_contract", {}),
            "animation_render_contract": path.get("animation_render_contract", {}),
            "intent_compiler": {
                "input_text": _path_motion_name(path),
                "accepted_state": status,
                "selected_manifold": str(path.get("manifold_type", "ribbon_path")),
                "matched_evidence": {
                    "motion_hints": list(path.get("motion_hints") or [])[:8],
                    "action_hints": list(path.get("action_hints") or [])[:8],
                    "support_kind_counts": dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {}),
                    "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
                    "grounding_evidence": dict(path.get("grounding_evidence") or {}),
                },
                "assumptions": list(contract_status.get("validation_notes") or path.get("validation_notes") or []),
                "uncertainty_reasons": list(uncertainty_reasons),
                "contradiction_reasons": list(path.get("contradiction_reasons") or rejection_reasons),
                "alternatives": _action_alternatives_from_hints(path),
            },
            "open_vocab_grounding": {
                "closed_vocabulary": False,
                "seed_prompts_are_priors": True,
                "local_evidence_required": True,
                "action_text": _path_motion_name(path),
                "local_grounding_confidence": _float(scores.get("local_action_evidence_confidence"), 0.0),
            },
            "trajectory_requirements": {
                "consume_path_id": pid,
                "consume_width_profile_px": True,
                "consume_visibility_profile": True,
                "consume_depth_trace_m": bool(path.get("depth_trace_m")),
            },
            "scores": {
                "overall_confidence": conf,
                "semantic_confidence": _float(scores.get("semantic_confidence"), conf),
                "geometric_confidence": _float(scores.get("geometric_confidence"), conf),
                "caption_confidence": _float(scores.get("caption_confidence"), 0.0),
                "occlusion_confidence": _float(scores.get("occlusion_confidence"), conf),
                "depth_confidence": _float(scores.get("depth_confidence"), conf),
                "boundary_confidence": _float(scores.get("boundary_confidence"), conf),
                "trajectory_fit_confidence": _float(scores.get("trajectory_fit_confidence"), conf),
                "confidence_breakdown": {
                    "semantic": _float(scores.get("semantic_confidence"), conf),
                    "geometric": _float(scores.get("geometric_confidence"), conf),
                    "depth": _float(scores.get("depth_confidence"), conf),
                    "caption": _float(scores.get("caption_confidence"), 0.0),
                    "occlusion": _float(scores.get("occlusion_confidence"), conf),
                    "boundary": _float(scores.get("boundary_confidence"), conf),
                    "trajectory_fit": _float(scores.get("trajectory_fit_confidence"), conf),
                    "overall": conf,
                },
                "rejection_reasons": sorted(set(rejection_reasons)),
                "uncertainty_reasons": sorted(set(uncertainty_reasons)),
            },
        })
    return actions


def _path_motion_name(path: Dict[str, Any]) -> str:
    hints = list(path.get("motion_hints") or [])
    for h in hints:
        if isinstance(h, dict) and str(h.get("motion", "")).strip():
            return str(h.get("motion"))
        if isinstance(h, str) and h.strip():
            return h.strip()
    return "traverse"


def _affordance_actions(ctx: PipelineContext, cfg: Any, ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj_affs = list((ctx.object_affordances or {}).get("objects") or [])
    mask_by_oid = {
        str(m.get("object_id", "")): m
        for m in list((ctx.mask_affordances or {}).get("masks") or [])
        if isinstance(m, dict)
    }
    max_actions = int(getattr(cfg, "action_hypotheses_max_affordance_actions", 96)) if cfg else 96
    min_score = float(getattr(cfg, "action_hypotheses_min_affordance_score", 0.18)) if cfg else 0.18
    actions: List[Dict[str, Any]] = []
    for obj in obj_affs:
        oid = str(obj.get("object_id", ""))
        mask = mask_by_oid.get(oid, {})
        candidates = [a for a in obj.get("actions") or [] if _float(a.get("score"), 0.0) >= min_score]
        if not candidates:
            candidates = list(obj.get("actions") or [])[:2]
        for action in candidates[:4]:
            name = str(action.get("name", "") or "interact")
            manifold_type = _manifold_for_action(name, obj.get("roles") or [], mask.get("path_modes") or [], ontology)
            action_id = f"action_{oid}_{name}_{len(actions):04d}"
            overall = _float(action.get("score"), 0.0)
            semantic = _float(action.get("score"), 0.0)
            geometric = (
                number(ontology, "scoring", "action_mask_geometric_confidence", 0.65)
                if mask
                else number(ontology, "scoring", "action_nomask_geometric_confidence", 0.45)
            )
            caption_conf = _float((obj.get("score_sources") or {}).get("caption_confidence"), 0.0)
            rejection_reasons: List[str] = []
            if overall < min_score:
                rejection_reasons.append("below_affordance_score_threshold")
            if manifold_type == "interior_path" and not _has_interior_support(mask):
                rejection_reasons.append("insufficient_mask_interior_support")
            status = _action_status(overall, rejection_reasons, [], min_accept=max(0.28, min_score), min_low=0.12)
            actions.append({
                "action_id": action_id,
                "source_type": "affordance",
                "action_status": status,
                "action_family": _action_family(name, manifold_type, ontology),
                "action_name": name,
                "manifold_type": manifold_type,
                "subject": {"type": "open_vocab_actor", "id": "candidate_actor"},
                "target": {"type": "object", "id": oid, "label": str(obj.get("label", "object"))},
                "grounding": {
                    "object_id": oid,
                    "mask_id": str(mask.get("mask_id", oid)),
                    "region_id": str(obj.get("region_id", "")),
                    "caption_grounded": bool(obj.get("caption")),
                    "mask_grounded": bool(mask),
                    "depth_grounded": _float(obj.get("depth_m"), 0.0) > 0.0,
                },
                "manifold": _object_manifold(manifold_type, obj, mask),
                "render_contract": _render_contract_for_manifold(manifold_type, mask, ontology),
                "intent_compiler": {
                    "input_text": name,
                    "accepted_state": status,
                    "selected_manifold": manifold_type,
                    "matched_evidence": {
                        "object_id": oid,
                        "object_label": str(obj.get("label", "")),
                        "roles": list(obj.get("roles") or [])[:8],
                        "mask_path_modes": list(mask.get("path_modes") or [])[:8],
                        "evidence_terms": list(action.get("evidence_terms") or []),
                    },
                    "assumptions": ["generated_from_object_mask_affordance"],
                    "alternatives": [
                        {"action_name": str(a.get("name", "")), "score": _float(a.get("score"), 0.0)}
                        for a in list(obj.get("actions") or [])[:6]
                    ],
                },
                "open_vocab_grounding": {
                    "closed_vocabulary": False,
                    "seed_prompts_are_priors": True,
                    "action_text": name,
                    "object_dynamic_terms": list((obj.get("open_vocab_grounding") or {}).get("top_dynamic_terms") or [])[:12],
                    "local_evidence_required": True,
                },
                "trajectory_requirements": _trajectory_requirements_for_manifold(manifold_type, ontology),
                "scores": {
                    "overall_confidence": overall,
                    "semantic_confidence": semantic,
                    "geometric_confidence": geometric,
                    "caption_confidence": caption_conf,
                    "occlusion_confidence": _float(mask.get("occlusion_risk"), 0.5),
                    "depth_confidence": 1.0 if _float(obj.get("depth_m"), 0.0) > 0.0 else 0.35,
                    "boundary_confidence": _float(mask.get("boundary_quality"), 0.5),
                    "trajectory_fit_confidence": max(0.1, min(1.0, (overall + geometric) * 0.5)),
                    "confidence_breakdown": {
                        "semantic": semantic,
                        "geometric": geometric,
                        "depth": 1.0 if _float(obj.get("depth_m"), 0.0) > 0.0 else 0.35,
                        "caption": caption_conf,
                        "occlusion": _float(mask.get("occlusion_risk"), 0.5),
                        "boundary": _float(mask.get("boundary_quality"), 0.5),
                        "trajectory_fit": max(0.1, min(1.0, (overall + geometric) * 0.5)),
                        "overall": overall,
                    },
                    "rejection_reasons": sorted(set(rejection_reasons)),
                    "evidence_terms": list(action.get("evidence_terms") or []),
                },
            })
            if len(actions) >= max_actions:
                return actions
    return actions


def _action_status(confidence: float, reasons: Sequence[str], uncertainty_reasons: Sequence[str], *, min_accept: float, min_low: float) -> str:
    hard = {
        "missing_polyline_3d",
        "missing_2d_geometry",
        "unknown_only_or_unsupported_route",
        "blocking_support_dominates_route",
        "insufficient_mask_interior_support",
    }
    if set(str(r) for r in reasons).intersection(hard) and confidence < 0.55:
        return "rejected"
    if confidence >= min_accept and not reasons:
        return "plausible_uncertain" if uncertainty_reasons else "accepted"
    if confidence >= min_accept and not set(str(r) for r in reasons).intersection(hard):
        return "plausible_uncertain"
    if confidence >= min_low:
        return "low_confidence"
    return "rejected"


def _action_alternatives_from_hints(path: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict):
            name = str(hint.get("motion", "")).strip()
            score = _float(hint.get("score"), 0.0)
        else:
            name = str(hint).strip()
            score = 0.0
        if name and name not in seen:
            rows.append({"action_name": name, "score": score, "source": "motion_hint"})
            seen.add(name)
    for hint in list(path.get("action_hints") or []):
        if not isinstance(hint, dict):
            continue
        name = str(hint.get("action") or hint.get("path_mode") or "").strip()
        if name and name not in seen:
            rows.append({"action_name": name, "score": _float(hint.get("score"), 0.0), "source": str(hint.get("entity_role", ""))})
            seen.add(name)
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return rows[:8]


def _manifold_for_action(
    action_name: str,
    roles: Sequence[Dict[str, Any]],
    path_modes: Sequence[Dict[str, Any]],
    ontology: Dict[str, Any],
) -> str:
    name = action_name.lower().strip()
    role_names = {str(r.get("name", "")).lower() for r in roles}
    mode_names = {str(m.get("mode", "")).lower() for m in path_modes}
    for rule in list_section(ontology, "manifold_policy"):
        actions = {str(x).lower() for x in rule.get("actions", []) or []}
        roles_rule = {str(x).lower() for x in rule.get("roles", []) or []}
        modes_rule = {str(x).lower() for x in rule.get("modes", []) or []}
        if name in actions or role_names.intersection(roles_rule) or mode_names.intersection(modes_rule):
            return str(rule.get("manifold_type", "ribbon_path"))
    if (
        any(tok in name for tok in ("interior", "inside", "within"))
        or "interior_path" in mode_names
        or "intra_mask" in mode_names
    ):
        return "interior_path"
    return "ribbon_path"


def _action_family(action_name: str, manifold_type: str, ontology: Dict[str, Any]) -> str:
    families = dict_section(ontology, "action_family_by_manifold")
    if manifold_type in families:
        return str(families[manifold_type])
    return action_name or "interaction"


def _object_manifold(manifold_type: str, obj: Dict[str, Any], mask: Dict[str, Any]) -> Dict[str, Any]:
    anchors = dict(obj.get("anchors") or {})
    geom = dict(mask.get("geometry") or obj.get("geometry") or {})
    base = {
        "type": manifold_type,
        "object_id": str(obj.get("object_id", "")),
        "label": str(obj.get("label", "")),
        "bbox_px": geom.get("bbox_px", (obj.get("geometry") or {}).get("bbox_px", [])),
        "depth_m": _float(obj.get("depth_m"), 0.0),
        "path_modes": list(mask.get("path_modes") or []),
    }
    if manifold_type == "contact_patch":
        base["contact_patch"] = {
            "contact_points": list(anchors.get("contact_points") or []),
            "support_contact_patch": dict(anchors.get("support_contact_patch") or {}),
        }
    elif manifold_type == "occlusion_pulse":
        base["occlusion_boundary_points"] = list(anchors.get("occlusion_boundary_points") or [])
        base["pulse"] = {"visible_fraction_range": [0.0, 1.0], "edge_anchored": True}
    elif manifold_type == "portal_path":
        base["entry_exit_points"] = list(anchors.get("entry_exit_points") or [])
        base["portal_policy"] = {"alpha_fade": True, "depth_order_required": True}
    elif manifold_type == "blob_path":
        base["interior_seed_uv"] = geom.get("interior_seed_uv", [])
        base["contour_sample_px"] = geom.get("contour_sample_px", [])
        base["depth_profile"] = dict(mask.get("depth_profile") or {})
    elif manifold_type == "interior_path":
        base["interior_seed_uv"] = geom.get("interior_seed_uv", [])
        base["interior_extent"] = dict(geom.get("interior_extent") or {})
        base["interior_constraint"] = {
            "movement_scope": "intra_mask",
            "stay_inside_mask": True,
        }
    elif manifold_type == "volume_path":
        center = (anchors.get("center_uv") or geom.get("centroid_uv") or [])
        base["volume"] = {
            "center_uv": center,
            "z_anchor_m": _float(obj.get("depth_m"), 0.0),
            "vertical_policy": "above_mask_if_depth_allows",
        }
    elif manifold_type == "effect_field":
        base["effect_field"] = {
            "field_extent_px": geom.get("bbox_px", []),
            "contour_sample_px": geom.get("contour_sample_px", []),
            "supports_reflection_or_wobble": True,
        }
    else:
        base["centerline_hints"] = list(anchors.get("approach_points") or [])
    return base


def _render_contract_for_manifold(
    manifold_type: str,
    mask: Dict[str, Any],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    contracts = dict_section(ontology, "render_contracts")
    contract = dict(contracts.get(manifold_type) or contracts.get("default") or {})
    contract["mask_path_modes"] = list(mask.get("path_modes") or [])
    return contract


def _trajectory_requirements_for_manifold(manifold_type: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
    reqs = dict_section(ontology, "trajectory_requirements_by_manifold")
    return {
        "requires_path_polyline": False,
        "requires_mask_interior": False,
        "requires_volume_sampling": False,
        "requires_contact_points": False,
        "requires_visibility_profile": False,
        "requires_effect_field": False,
        **dict(reqs.get(manifold_type) or {}),
    }


def _has_interior_support(mask: Dict[str, Any]) -> bool:
    geom = mask.get("geometry") if isinstance(mask.get("geometry"), dict) else {}
    if geom.get("interior_seed_uv"):
        return True
    modes = list(mask.get("path_modes") or [])
    for m in modes:
        if isinstance(m, dict) and str(m.get("mode", "")).strip().lower() in {"blob_path", "interior_path"}:
            return True
    return False


def _rank_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions.sort(key=lambda a: _float((a.get("scores") or {}).get("overall_confidence"), 0.0), reverse=True)
    for idx, action in enumerate(actions):
        action["rank"] = idx + 1
    return actions


def _occluder_ids(path: Dict[str, Any]) -> List[str]:
    ids = set()
    for row in path.get("visibility_profile") or []:
        if not isinstance(row, dict):
            continue
        for oid in row.get("occluder_ids") or []:
            ids.add(str(oid))
    return sorted(x for x in ids if x)


def _write_action_overlay(
    ctx: PipelineContext,
    actions: List[Dict[str, Any]],
    paths_root: Path,
    ontology: Dict[str, Any],
) -> str:
    try:
        import cv2
    except Exception:
        return ""
    try:
        canvas = ctx.img_bgr.copy()
        colors = {
            str(k): tuple(int(vv) for vv in v[:3])
            for k, v in dict_section(ontology, "overlay_colors_bgr").items()
            if isinstance(v, list) and len(v) >= 3
        }
        for action in actions[:96]:
            manifold = dict(action.get("manifold") or {})
            mtype = str(action.get("manifold_type", ""))
            color = colors.get(mtype, (200, 200, 200))
            label = f"{str(action.get('action_name', 'action'))}:{mtype}"
            path_pts = manifold.get("centerline_path") or []
            pts = _np_points(path_pts)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, label[:48], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                continue
            bbox = manifold.get("bbox_px") or []
            if isinstance(bbox, list) and len(bbox) >= 4:
                x, y, w, h = [int(round(_float(v))) for v in bbox[:4]]
                cv2.rectangle(canvas, (x, y), (x + max(1, w), y + max(1, h)), color, 2, cv2.LINE_AA)
                cv2.putText(canvas, label[:48], (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            for key in ("contact_points", "occlusion_boundary_points", "entry_exit_points"):
                for uv in list(manifold.get(key) or [])[:12]:
                    if isinstance(uv, list) and len(uv) >= 2:
                        cv2.circle(canvas, (int(round(_float(uv[0]))), int(round(_float(uv[1])))), 3, color, -1, cv2.LINE_AA)
            for uv in list((manifold.get("contact_patch") or {}).get("contact_points") or [])[:12]:
                if isinstance(uv, list) and len(uv) >= 2:
                    cv2.circle(canvas, (int(round(_float(uv[0]))), int(round(_float(uv[1])))), 3, color, -1, cv2.LINE_AA)
        out = paths_root / "action_manifold_overlay.png"
        cv2.imwrite(str(out), canvas)
        return out.name
    except Exception:
        return ""


def _np_points(path_pts: Any) -> Any:
    import numpy as np

    pts = []
    for xy in path_pts or []:
        if isinstance(xy, list) and len(xy) >= 2:
            pts.append([int(round(_float(xy[0]))), int(round(_float(xy[1])))])
    return np.asarray(pts, dtype=np.int32) if pts else np.zeros((0, 2), dtype=np.int32)


def _float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        return out if out == out else default
    except (TypeError, ValueError):
        return default


__all__ = ["run"]
