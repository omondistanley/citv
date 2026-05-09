"""Staged pipeline caption export: tiered caption bundles (thoughts.md Next Iteration).

Tier structure:
  1. Global scene caption   — full-image Florence-2 <MORE_DETAILED_CAPTION>
  2. Per-object captions    — per-mask Florence-2 + RAM++ labels/captions
  3. Per-region summaries   — aggregated label/depth narrative per depth region
  4. Cross-region interactions — Florence-2 enriched inter-region relation pairs
  5. Uncertainty / plausibility notes — objects with label warnings or low-confidence

All artifacts land under ``{output_dir}/scene_graph/staged/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..evidence import normalize_relation, object_id
from ..pipeline_context import PipelineContext


def _write(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Build and write all caption tiers for the staged pipeline."""
    cfg = getattr(pipeline, "config", None)
    enabled = bool(getattr(cfg, "export_hybrid_captions", True)) if cfg else True
    if not enabled:
        return ctx

    objects: List[Dict[str, Any]] = list(ctx.extra.get("objects", []))
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    stem = ctx.stem

    # ── Tier 1: Global scene caption ─────────────────────────────────────────
    global_caption, global_status = _run_global_caption(pipeline, ctx)
    tier1 = {
        "tier": "global_scene",
        "stem": stem,
        "caption": global_caption,
        "status": global_status,
        "image": str(ctx.image_path),
    }
    _write(tier1, staged_dir / f"{stem}_caption_global.json")

    # ── Tier 2: Per-object captions ──────────────────────────────────────────
    tier2_rows = _build_per_object_captions(objects)
    tier2 = {
        "tier": "per_object",
        "stem": stem,
        "count": len(tier2_rows),
        "objects": tier2_rows,
    }
    _write(tier2, staged_dir / f"{stem}_caption_objects.json")

    # ── Tier 3: Per-region summaries ─────────────────────────────────────────
    tier3_regions = _build_per_region_summaries(objects, ctx.region_partition_meta)
    tier3 = {
        "tier": "per_region",
        "stem": stem,
        "count": len(tier3_regions),
        "regions": tier3_regions,
    }
    _write(tier3, staged_dir / f"{stem}_caption_regions.json")

    # ── Tier 4: Cross-region interactions ────────────────────────────────────
    tier4_cross = _build_cross_region_interactions(ctx.relations, objects)
    tier4 = {
        "tier": "cross_region",
        "stem": stem,
        "count": len(tier4_cross),
        "interactions": tier4_cross,
    }
    _write(tier4, staged_dir / f"{stem}_caption_cross_region.json")

    # ── Tier 5: Uncertainty / plausibility notes ─────────────────────────────
    tier5_notes = _build_uncertainty_notes(objects)
    tier5 = {
        "tier": "uncertainty",
        "stem": stem,
        "count": len(tier5_notes),
        "notes": tier5_notes,
    }
    _write(tier5, staged_dir / f"{stem}_caption_uncertainty.json")

    # ── Tiered bundle manifest ────────────────────────────────────────────────
    bundle = {
        "stem": stem,
        "image": str(ctx.image_path),
        "timestamp": ctx.timestamp,
        "tiers": {
            "global_scene": f"scene_graph/staged/{stem}_caption_global.json",
            "per_object": f"scene_graph/staged/{stem}_caption_objects.json",
            "per_region": f"scene_graph/staged/{stem}_caption_regions.json",
            "cross_region": f"scene_graph/staged/{stem}_caption_cross_region.json",
            "uncertainty": f"scene_graph/staged/{stem}_caption_uncertainty.json",
        },
        "global_caption": global_caption,
        "global_status": global_status,
        "object_count": len(tier2_rows),
        "region_count": len(tier3_regions),
        "cross_region_count": len(tier4_cross),
        "uncertainty_count": len(tier5_notes),
    }
    _write(bundle, staged_dir / f"{stem}_caption_bundle.json")

    ctx.extra["caption_tiers"] = {
        "global_scene": tier1,
        "per_object": tier2,
        "per_region": tier3,
        "cross_region": tier4,
        "uncertainty": tier5,
    }
    ctx.extra["caption_bundle"] = bundle
    ctx.path_exports["caption_global_json"] = f"scene_graph/staged/{stem}_caption_global.json"
    ctx.path_exports["caption_objects_json"] = f"scene_graph/staged/{stem}_caption_objects.json"
    ctx.path_exports["caption_regions_json"] = f"scene_graph/staged/{stem}_caption_regions.json"
    ctx.path_exports["caption_cross_region_json"] = (
        f"scene_graph/staged/{stem}_caption_cross_region.json"
    )
    ctx.path_exports["caption_uncertainty_json"] = (
        f"scene_graph/staged/{stem}_caption_uncertainty.json"
    )
    ctx.path_exports["caption_bundle_json"] = f"scene_graph/staged/{stem}_caption_bundle.json"
    print(
        f"  [CaptionsExport] {stem}: global={global_status}, "
        f"{len(tier2_rows)} objects, {len(tier3_regions)} regions, "
        f"{len(tier4_cross)} cross-region, {len(tier5_notes)} uncertainty notes"
    )
    return ctx


def _run_global_caption(pipeline: Any, ctx: PipelineContext):
    """Attempt Florence-2 full-image detailed caption; return (caption_str, status_str)."""
    try:
        labellers = pipeline._get_labeling_pipeline()
        f2 = getattr(labellers, "florence2", None)
        if f2 is None or not getattr(f2, "active", False):
            return "", "florence2_unavailable"
        from PIL import Image as PILImage
        import cv2
        pil_img = PILImage.fromarray(cv2.cvtColor(ctx.img_bgr, cv2.COLOR_BGR2RGB))
        res = f2._run_task("<MORE_DETAILED_CAPTION>", pil_img)
        cap = str(res.get("<MORE_DETAILED_CAPTION>", "")).strip()
        if cap:
            return cap, "generated_local_florence"
        res2 = f2._run_task("<CAPTION>", pil_img)
        cap2 = str(res2.get("<CAPTION>", "")).strip()
        if cap2:
            return cap2, "generated_local_florence_fallback"
        return "", "empty_output"
    except Exception as exc:
        return "", f"error: {type(exc).__name__}"


def _build_per_object_captions(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for o in objects:
        src = o.get("sources") or {}
        f2_src = src.get("Florence2") or {}
        rampp_src = src.get("RAM++") or {}
        gdino_src = src.get("GroundedSAM2") or {}
        rows.append({
            "object_id": str(o.get("id", "")),
            "label": str(o.get("label", "object")),
            "canonical_label": str(o.get("canonical_label", o.get("label", "object"))),
            "label_source": str(o.get("label_source", "")),
            "label_candidates": list(o.get("label_candidates", [])),
            "rejected_labels": list(o.get("rejected_labels", [])),
            "visual_quality_attributes": list(o.get("visual_quality_attributes", [])),
            "source_agreement": float(o.get("source_agreement", 0.0) or 0.0),
            "segmentor": str(o.get("segmentor", "")),
            "gdino_label": str(gdino_src.get("label", "")),
            "gdino_conf": float(gdino_src.get("confidence", 0.0)),
            "florence_label": str(f2_src.get("label", "")),
            "florence_caption": str(f2_src.get("caption", "")),
            "rampp_label": str(rampp_src.get("label", "")),
            "rampp_tags": list(rampp_src.get("tags", [])),
            "region_id": str(o.get("region_id", "")),
            "region_index": int(o.get("region_index", 0) or 0),
            "depth_m": float((o.get("depth_stats") or {}).get("z_val", 0.0)),
            "label_warning": str(o.get("label_warning", "")),
        })
    return rows


def _build_per_region_summaries(
    objects: List[Dict[str, Any]],
    region_partition_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_region: Dict[str, List[Dict[str, Any]]] = {}
    for o in objects:
        rid = str(o.get("region_id") or f"region_{o.get('region_index', 0)}")
        by_region.setdefault(rid, []).append(o)

    summaries = []
    for r in region_partition_meta:
        rid = str(r.get("id", ""))
        objs_in = by_region.get(rid, [])
        labels = [str(o.get("label", "object")) for o in objs_in if o.get("label", "object") != "object"]
        depths = [float((o.get("depth_stats") or {}).get("z_val", 0.0)) for o in objs_in]
        summaries.append({
            "region_id": rid,
            "region_type": str(r.get("type", "")),
            "object_count": len(objs_in),
            "object_labels": labels,
            "depth_mean_m": round(sum(depths) / len(depths), 3) if depths else 0.0,
            "depth_min_m": round(min(depths), 3) if depths else 0.0,
            "depth_max_m": round(max(depths), 3) if depths else 0.0,
            "narrative": _region_narrative(rid, r, labels, depths),
        })
    # Emit objects with no region assignment
    unassigned = [o for o in objects if not o.get("region_id")]
    if unassigned:
        labels_u = [str(o.get("label", "object")) for o in unassigned]
        depths_u = [float((o.get("depth_stats") or {}).get("z_val", 0.0)) for o in unassigned]
        summaries.append({
            "region_id": "unassigned",
            "region_type": "none",
            "object_count": len(unassigned),
            "object_labels": labels_u,
            "depth_mean_m": round(sum(depths_u) / len(depths_u), 3) if depths_u else 0.0,
            "depth_min_m": round(min(depths_u), 3) if depths_u else 0.0,
            "depth_max_m": round(max(depths_u), 3) if depths_u else 0.0,
            "narrative": f"{len(unassigned)} object(s) with no region assignment: {', '.join(labels_u[:5])}.",
        })
    return summaries


def _region_narrative(rid: str, r: Dict[str, Any], labels: List[str], depths: List[float]) -> str:
    rtype = str(r.get("type", "region"))
    depth_str = f"{round(sum(depths)/len(depths), 1)} m" if depths else "unknown depth"
    label_str = ", ".join(labels[:5]) + ("…" if len(labels) > 5 else "") if labels else "no named objects"
    return f"Region {rid} ({rtype}) at {depth_str} contains: {label_str}."


def _build_cross_region_interactions(
    relations: List[Dict[str, Any]],
    objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    obj_by_id = {object_id(o): o for o in objects}
    cross = []
    for rel in relations:
        nrel = normalize_relation(rel)
        sub_id = str(nrel.get("subject_id", ""))
        obj_id = str(nrel.get("object_id", ""))
        sub = obj_by_id.get(sub_id)
        tgt = obj_by_id.get(obj_id)
        if sub is None or tgt is None:
            continue
        ri_s = int(sub.get("region_index", 0) or 0)
        ri_o = int(tgt.get("region_index", 0) or 0)
        if ri_s == ri_o or ri_s == 0 or ri_o == 0:
            continue
        cross.append({
            "subject_id": sub_id,
            "subject_label": str(sub.get("label", "object")),
            "subject_canonical_label": str(sub.get("canonical_label", sub.get("label", "object"))),
            "subject_region": str(sub.get("region_id", "")),
            "predicate": str(nrel.get("predicate", "")),
            "object_id": obj_id,
            "object_label": str(tgt.get("label", "object")),
            "object_canonical_label": str(tgt.get("canonical_label", tgt.get("label", "object"))),
            "object_region": str(tgt.get("region_id", "")),
            "relation_tier": str(nrel.get("relation_tier", "")),
            "source_layer": str(nrel.get("source_layer", "")),
            "score": float(nrel.get("score", 0.5)),
        })
    return cross


def _build_uncertainty_notes(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    notes = []
    for o in objects:
        issues = []
        lw = str(o.get("label_warning", "")).strip()
        if lw:
            issues.append(f"label_warning: {lw}")
        conf = float(o.get("conf", 1.0))
        if conf < 0.35:
            issues.append(f"low_confidence: {conf:.2f}")
        ds = o.get("depth_stats") or {}
        if ds.get("possibly_transparent"):
            issues.append("possibly_transparent")
        dp = float(o.get("depth_plausibility_score", 1.0) or 1.0)
        if dp < 0.5:
            issues.append(f"low_depth_plausibility: {dp:.2f}")
        if issues:
            notes.append({
                "object_id": str(o.get("id", "")),
                "label": str(o.get("label", "object")),
                "issues": issues,
                "region_id": str(o.get("region_id", "")),
            })
    return notes
