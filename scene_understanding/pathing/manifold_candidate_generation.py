"""Generate grounded action/manifold candidates before final path validation."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _anchor_from_entity(ent: Dict[str, Any]) -> List[float]:
    geom = ent.get("geometry") if isinstance(ent.get("geometry"), dict) else {}
    c = geom.get("centroid_xy")
    if isinstance(c, Sequence) and len(c) >= 2:
        return [float(c[0]), float(c[1])]
    bbox = geom.get("bbox_xyxy")
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        return [float(bbox[0] + bbox[2]) * 0.5, float(bbox[1] + bbox[3]) * 0.5]
    return [0.0, 0.0]


def _short_curve(anchor: Sequence[float], *, dx: float = 18.0, dy: float = -14.0) -> List[List[float]]:
    x, y = float(anchor[0]), float(anchor[1])
    return [[x - dx, y - dy * 0.25], [x, y], [x + dx, y + dy]]


def build_manifold_candidates(
    *,
    grounding_index: Dict[str, Any],
    rasters_manifest: Dict[str, Any],
    max_candidates: int = 96,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for ent in list((grounding_index or {}).get("entities") or []):
        if not isinstance(ent, dict):
            continue
        eid = str(ent.get("entity_id", ""))
        if not eid:
            continue
        roles = ent.get("roles") if isinstance(ent.get("roles"), dict) else {}
        actions = ent.get("actions") if isinstance(ent.get("actions"), dict) else {}
        anchor = _anchor_from_entity(ent)
        local_conf = max([_float(v) for v in list(roles.values()) + list(actions.values())] or [0.0])
        source = {
            "type": "grounded_candidate",
            "id": "actor",
            "start_uv": anchor,
            "evidence_level": "local_or_scene_prior",
        }
        target = {
            "type": "object",
            "id": eid,
            "goal_uv": anchor,
            "label": str(ent.get("label", "")),
            "evidence_level": "object_mask",
        }

        def add(manifold: str, action_family: str, action_name: str, score: float, poly: List[List[float]], reason: str) -> None:
            if score <= 0.0:
                return
            candidates.append({
                "path_id": f"grounded_{manifold}_{eid}_{len(candidates):03d}",
                "path_level": "object",
                "path_type": manifold,
                "manifold_type": manifold,
                "action_family": action_family,
                "action_name": action_name,
                "source_entity": source,
                "target_entity": target,
                "polyline_2d": poly,
                "scores": {
                    "overall_confidence": round(float(min(0.86, 0.32 + 0.55 * score)), 4),
                    "grounding_candidate_confidence": round(float(score), 4),
                },
                "grounding_evidence": {
                    "schema": "citv_path_grounding_evidence_v1",
                    "candidate_source": "scene_grounding_index",
                    "entity_id": eid,
                    "entity_label": str(ent.get("label", "")),
                    "local_evidence_confidence": round(float(score), 4),
                    "global_only": False,
                    "manifold_reason": reason,
                    "uncertainty": list(ent.get("uncertainty") or []),
                    "contradictions": list(ent.get("contradictions") or []),
                },
                "routing_meta": {
                    "generation_order": "grounding_index_to_manifold_candidate",
                    "uses_affordance_rasters": bool(rasters_manifest),
                    "path_granularity": "grounded_manifold",
                },
            })

        contact = max(_float(roles.get("interaction_target")), _float(actions.get("hold")), _float(actions.get("touch")), _float(actions.get("sit")))
        add("contact_patch", "contact_interaction", "interact", contact, _short_curve(anchor, dx=10.0, dy=-8.0), "local_object_or_mask_contact")

        occ = max(_float(roles.get("occluder")), _float(actions.get("peek")), _float(actions.get("hide")))
        add("occlusion_pulse", "occlusion_interaction", "peek", occ, _short_curve(anchor, dx=16.0, dy=4.0), "local_occluder_or_depth_edge")

        open_air = max(_float(roles.get("sky_open_air")), _float(actions.get("fly")), _float(actions.get("hover")))
        add("volume_path", "volume_motion", "hover", open_air, _short_curve(anchor, dx=22.0, dy=-22.0), "local_or_region_open_air")

        support = max(_float(roles.get("support")), _float(actions.get("walk")), _float(actions.get("climb")))
        add("ribbon_path", "locomotion", "walk", support, _short_curve(anchor, dx=26.0, dy=-4.0), "local_support_or_walkable_surface")

        contour = max(_float(actions.get("inspect")), _float(actions.get("around")), _float(roles.get("boundary")))
        add("contour_path", "boundary_motion", "inspect", contour, _short_curve(anchor, dx=18.0, dy=14.0), "local_mask_boundary")

        if len(candidates) >= max_candidates:
            break
    return {
        "schema": "citv_manifold_candidates_v1",
        "candidate_count": len(candidates),
        "candidates": candidates[:max_candidates],
    }


__all__ = ["build_manifold_candidates"]
