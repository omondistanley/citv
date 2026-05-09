"""Hybrid semantic + geometric scores for path hypotheses (mutates path dict)."""
from __future__ import annotations

from typing import Any, Dict


def preview_hybrid_overall(
    p: Dict[str, Any],
    sem: Dict[str, Any],
    *,
    wg: float,
    ws: float,
    wr: float,
    wa: float,
    geom_image_blend: float | None = None,
    semantic_relation_blend: float | None = None,
) -> float:
    """Same scalar as ``scores.hybrid_overall`` from ``apply_hybrid_confidence_scores`` (read-only)."""
    rel_score = float((p.get("scores") or {}).get("relation_consistency", 0.5))
    geom_score = float((p.get("scores") or {}).get("geometric_feasibility", 0.5))
    img_align = float((p.get("scores") or {}).get("image_alignment_score", geom_score))
    if geom_image_blend is None:
        delta = min(1.0, abs(geom_score - img_align))
        geom_image_blend = 0.50 + 0.30 * delta
    gib = max(0.0, min(1.0, float(geom_image_blend)))
    geom_score = (1.0 - gib) * geom_score + gib * img_align
    if semantic_relation_blend is None:
        semantic_relation_blend = 0.5
    srb = max(0.0, min(1.0, float(semantic_relation_blend)))
    action_fit = max(0.0, min(1.0, (1.0 - srb) * float(sem.get("semantic_validity_score", 0.0)) + srb * rel_score))
    return float((wg * geom_score) + (ws * float(sem.get("semantic_validity_score", 0.0))) + (wr * rel_score) + (wa * action_fit))


def apply_hybrid_confidence_scores(
    p: Dict[str, Any],
    sem: Dict[str, Any],
    *,
    wg: float,
    ws: float,
    wr: float,
    wa: float,
    geom_image_blend: float | None = None,
    semantic_relation_blend: float | None = None,
    hybrid_motion_blend: float | None = None,
) -> None:
    """
    Update *p* with semantic bundle and ``scores.hybrid_overall`` / ``overall_confidence``.

    Continuation logic (diagnostics, suppression) stays in the legacy exporter loop.
    """
    rel_score = float((p.get("scores") or {}).get("relation_consistency", 0.5))
    geom_score = float((p.get("scores") or {}).get("geometric_feasibility", 0.5))
    img_align = float((p.get("scores") or {}).get("image_alignment_score", geom_score))
    if geom_image_blend is None:
        # Stronger agreement with image alignment as geometry disagreement grows.
        delta = min(1.0, abs(geom_score - img_align))
        geom_image_blend = 0.50 + 0.30 * delta
    gib = max(0.0, min(1.0, float(geom_image_blend)))
    geom_score = (1.0 - gib) * geom_score + gib * img_align
    if semantic_relation_blend is None:
        semantic_relation_blend = 0.5
    srb = max(0.0, min(1.0, float(semantic_relation_blend)))
    action_fit = max(0.0, min(1.0, (1.0 - srb) * float(sem.get("semantic_validity_score", 0.0)) + srb * rel_score))
    hybrid = preview_hybrid_overall(
        p,
        sem,
        wg=wg,
        ws=ws,
        wr=wr,
        wa=wa,
        geom_image_blend=geom_image_blend,
        semantic_relation_blend=semantic_relation_blend,
    )
    p["semantic_valid"] = bool(sem.get("semantic_valid", False))
    p["semantic_validity_score"] = float(sem.get("semantic_validity_score", 0.0))
    p["semantic_reasons"] = list(sem.get("semantic_reasons", []))
    p["affordance_trace"] = list(sem.get("affordance_trace", []))
    p.setdefault("scores", {})
    if "is_motion_primary" not in p:
        p["is_motion_primary"] = bool(str(p.get("path_level", "")) == "object")
    mm = p.get("motion_metrics", {}) or {}
    motion_primary_score = float((p.get("scores") or {}).get("motion_primary_score", mm.get("motion_primary_score", 0.5)))
    if hybrid_motion_blend is None:
        # More motion influence for object-level travel, less for mask diagnostics.
        hybrid_motion_blend = 0.30 if str(p.get("path_level", "")) != "mask" else 0.15
    hmb = max(0.0, min(1.0, float(hybrid_motion_blend)))
    p["scores"]["action_fit"] = float(action_fit)
    p["scores"]["hybrid_overall"] = float(hybrid)
    p["scores"]["overall_confidence"] = float(max(0.0, min(1.0, (1.0 - hmb) * hybrid + hmb * motion_primary_score)))
