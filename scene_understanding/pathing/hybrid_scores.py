"""Hybrid semantic + geometric scores for path hypotheses (mutates path dict)."""
from __future__ import annotations

from typing import Any, Dict


def apply_hybrid_confidence_scores(
    p: Dict[str, Any],
    sem: Dict[str, Any],
    *,
    wg: float,
    ws: float,
    wr: float,
    wa: float,
) -> None:
    """
    Update *p* with semantic bundle and ``scores.hybrid_overall`` / ``overall_confidence``.

    Continuation logic (diagnostics, suppression) stays in the legacy exporter loop.
    """
    rel_score = float((p.get("scores") or {}).get("relation_consistency", 0.5))
    geom_score = float((p.get("scores") or {}).get("geometric_feasibility", 0.5))
    img_align = float((p.get("scores") or {}).get("image_alignment_score", geom_score))
    geom_score = 0.6 * geom_score + 0.4 * img_align
    action_fit = max(0.0, min(1.0, 0.5 * float(sem.get("semantic_validity_score", 0.0)) + 0.5 * rel_score))
    hybrid = (wg * geom_score) + (ws * float(sem.get("semantic_validity_score", 0.0))) + (wr * rel_score) + (wa * action_fit)
    p["semantic_valid"] = bool(sem.get("semantic_valid", False))
    p["semantic_validity_score"] = float(sem.get("semantic_validity_score", 0.0))
    p["semantic_reasons"] = list(sem.get("semantic_reasons", []))
    p["affordance_trace"] = list(sem.get("affordance_trace", []))
    p.setdefault("scores", {})
    if "is_motion_primary" not in p:
        p["is_motion_primary"] = bool(str(p.get("path_level", "")) == "object")
    mm = p.get("motion_metrics", {}) or {}
    motion_primary_score = float((p.get("scores") or {}).get("motion_primary_score", mm.get("motion_primary_score", 0.5)))
    p["scores"]["action_fit"] = float(action_fit)
    p["scores"]["hybrid_overall"] = float(hybrid)
    p["scores"]["overall_confidence"] = float(max(0.0, min(1.0, 0.7 * hybrid + 0.3 * motion_primary_score)))
