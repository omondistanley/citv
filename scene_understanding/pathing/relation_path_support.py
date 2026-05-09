"""Structured relation matching for path endpoints (Pix2SG-style relation dicts)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _norm_id(x: Any) -> str:
    return str(x or "").strip()


def structured_relation_match(path_dict: Dict[str, Any], relations: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Match path source/target object ids to relation rows.

    Relations are expected to have subject_id, object_id, predicate, score (optional).
    """
    src = path_dict.get("source_entity") or {}
    tgt = path_dict.get("target_entity") or {}
    sid = _norm_id(src.get("id"))
    tid = _norm_id(tgt.get("id"))
    out = {
        "pair_hit": False,
        "direct_hit": False,
        "reverse_hit": False,
        "best_predicate": "",
        "best_score": 0.0,
        "relation_tier": "",
        "matching_relation_indices": [],
    }
    if not sid or not tid or not relations:
        return out

    best_score = -1.0
    best_pred = ""
    best_tier = ""
    indices: List[int] = []

    for i, r in enumerate(relations):
        if not isinstance(r, dict):
            continue
        rs = _norm_id(r.get("subject_id"))
        ro = _norm_id(r.get("object_id"))
        if not rs or not ro:
            continue
        if rs == sid and ro == tid:
            out["direct_hit"] = True
            out["pair_hit"] = True
            indices.append(i)
            sc = float(r.get("score", 0.0) or 0.0)
            if sc >= best_score:
                best_score = sc
                best_pred = str(r.get("predicate", "") or "")
                best_tier = str(r.get("relation_tier", "") or "")
        elif rs == tid and ro == sid:
            out["reverse_hit"] = True
            out["pair_hit"] = True
            indices.append(i)
            sc = float(r.get("score", 0.0) or 0.0) * 0.85
            if sc >= best_score:
                best_score = sc
                best_pred = str(r.get("predicate", "") or "")
                best_tier = str(r.get("relation_tier", "") or "")

    out["matching_relation_indices"] = indices
    out["best_predicate"] = best_pred
    out["best_score"] = float(max(0.0, best_score))
    out["relation_tier"] = best_tier
    return out


def relation_guardrail_structured(
    path_obj: Dict[str, Any],
    relations: Optional[List[Dict[str, Any]]],
    *,
    penalty_no_relation: float = 0.18,
) -> Dict[str, Any]:
    """Replacement for substring blob guardrail using structured subject/object ids."""
    s = path_obj.get("source_entity") or {}
    t = path_obj.get("target_entity") or {}
    st = str(s.get("type", ""))
    tt = str(t.get("type", ""))
    if st == "region" or tt == "region":
        required = False
    else:
        required = True
    m = structured_relation_match(path_obj, relations)
    pair_hit = bool(m.get("pair_hit"))
    penalty = 0.0 if (pair_hit or not required) else float(penalty_no_relation)
    return {
        "relation_pair_found": pair_hit,
        "relation_direct_hit": bool(m.get("direct_hit")),
        "relation_reverse_hit": bool(m.get("reverse_hit")),
        "relation_required": required,
        "relation_guardrail_penalty": penalty,
        "relation_best_predicate": m.get("best_predicate", ""),
        "relation_best_score": float(m.get("best_score", 0.0)),
        "relation_tier": m.get("relation_tier", ""),
    }


__all__ = ["structured_relation_match", "relation_guardrail_structured"]
