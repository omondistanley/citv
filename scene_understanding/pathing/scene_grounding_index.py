"""Build a fused scene/object/mask/region grounding index for path generation."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .open_vocab_grounding import build_open_vocab_grounding, evidence_text


def _float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _score_map(rows: Iterable[Dict[str, Any]], key: str = "name") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get(key, "")).strip()
        if not name:
            continue
        out[name] = max(out.get(name, 0.0), _float(row.get("score"), 0.0))
    return out


def _mask_bbox(mask: Any) -> Dict[str, Any]:
    if mask is None:
        return {}
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return {}
    ys, xs = np.nonzero(arr)
    return {
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "area_px": int(arr.sum()),
        "centroid_xy": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
    }


def _object_affordance_lookup(object_affordances: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("object_id", "")): row
        for row in list((object_affordances or {}).get("objects") or [])
        if isinstance(row, dict) and str(row.get("object_id", ""))
    }


def _mask_affordance_lookup(mask_affordances: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("object_id", "") or row.get("mask_id", "")): row
        for row in list((mask_affordances or {}).get("masks") or [])
        if isinstance(row, dict) and str(row.get("object_id", "") or row.get("mask_id", ""))
    }


def _caption_lookup(caption_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = list((caption_bundle or {}).get("objects") or [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            oid = str(row.get("object_id", "") or row.get("id", ""))
            if oid:
                out[oid] = row
    return out


def _entity_record(
    obj: Dict[str, Any],
    *,
    object_aff: Dict[str, Any],
    mask_aff: Dict[str, Any],
    caption: Dict[str, Any],
) -> Dict[str, Any]:
    oid = str(obj.get("id", ""))
    roles = _score_map(object_aff.get("roles") or [])
    actions = _score_map(object_aff.get("actions") or [])
    path_modes = _score_map(object_aff.get("path_modes") or [])
    mask_roles = _score_map(mask_aff.get("roles") or [])
    mask_actions = _score_map(mask_aff.get("actions") or [])
    for k, v in mask_roles.items():
        roles[k] = max(roles.get(k, 0.0), v)
    for k, v in mask_actions.items():
        actions[k] = max(actions.get(k, 0.0), v)
    text_parts = [
        str(obj.get("label", "")),
        str(caption.get("caption", "") or caption.get("description", "")),
        evidence_text(object_aff),
        evidence_text(mask_aff),
    ]
    contradictions: List[str] = []
    uncertainty: List[str] = []
    label = str(obj.get("label", "")).strip().lower()
    if label in {"", "unknown", "set", "thin", "image", "photo", "landscape photograph"}:
        uncertainty.append("weak_or_meta_label")
    if roles.get("support", 0.0) > 0.45 and roles.get("hard_obstacle", 0.0) > 0.45:
        contradictions.append("support_and_blocker_both_high")
    return {
        "entity_id": oid,
        "entity_type": "object",
        "label": str(obj.get("label", "")),
        "region_id": str(obj.get("region_id", "")),
        "roles": roles,
        "actions": actions,
        "path_modes": path_modes,
        "geometry": _mask_bbox(obj.get("_sam2_mask_array")),
        "anchors": dict(object_aff.get("anchors") or {}),
        "evidence_text": " ".join(p for p in text_parts if p.strip()),
        "evidence_sources": {
            "object": bool(obj),
            "object_affordance": bool(object_aff),
            "mask_affordance": bool(mask_aff),
            "caption": bool(caption),
        },
        "contradictions": contradictions,
        "uncertainty": uncertainty,
    }


def build_scene_grounding_index(
    *,
    stem: str,
    objects: List[Dict[str, Any]],
    regions: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    scene_affordances: Dict[str, Any],
    object_affordances: Dict[str, Any],
    mask_affordances: Dict[str, Any],
    caption_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    obj_aff = _object_affordance_lookup(object_affordances)
    mask_aff = _mask_affordance_lookup(mask_affordances)
    captions = _caption_lookup(caption_bundle)
    entities = [
        _entity_record(
            obj,
            object_aff=obj_aff.get(str(obj.get("id", "")), {}),
            mask_aff=mask_aff.get(str(obj.get("id", "")), {}),
            caption=captions.get(str(obj.get("id", "")), {}),
        )
        for obj in objects
    ]
    evidence_rows: List[Dict[str, Any]] = []
    evidence_rows.extend(entities)
    evidence_rows.extend([r for r in regions if isinstance(r, dict)])
    evidence_rows.extend([r for r in relations if isinstance(r, dict)])
    evidence_rows.extend(list((scene_affordances or {}).get("actions") or []))
    evidence_rows.extend(list((scene_affordances or {}).get("roles") or []))
    seed_concepts = {
        "support_surface": ["support", "walk", "stand", "floor", "ground", "surface", "step", "path"],
        "blocker": ["block", "obstacle", "solid", "wall", "barrier", "foreground"],
        "contact_target": ["hold", "touch", "sit", "lean", "contact", "handle", "person"],
        "occlusion_edge": ["behind", "occlude", "peek", "hide", "foreground", "edge"],
        "portal": ["enter", "exit", "door", "opening", "through", "portal"],
        "open_air": ["sky", "air", "above", "fly", "hover", "open"],
        "interior_motion": ["inside", "within", "area", "interior", "region"],
    }
    open_vocab = build_open_vocab_grounding(evidence_rows=evidence_rows, seed_concepts=seed_concepts)
    return {
        "schema": "citv_scene_grounding_index_v1",
        "stem": stem,
        "entity_count": len(entities),
        "relation_count": len(relations or []),
        "region_count": len(regions or []),
        "entities": entities,
        "scene_priors": {
            "actions": list((scene_affordances or {}).get("actions") or [])[:32],
            "roles": list((scene_affordances or {}).get("roles") or [])[:32],
        },
        "open_vocab_summary": open_vocab,
        "policy": {
            "global_scene_evidence_is_prior_only": True,
            "local_entity_or_region_evidence_required_for_acceptance": True,
        },
    }


__all__ = ["build_scene_grounding_index"]
