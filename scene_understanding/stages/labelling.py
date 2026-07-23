"""Labelling + per-mask depth geometry for staged pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from ..pipeline_context import PipelineContext
from ..timing import sub_timer


def _passes_post_filter(mask_bin: np.ndarray, det: Dict[str, Any], cfg: Any, h: int, w: int) -> str:
    """Package-side mirror of the monolith's ``_passes_post_filter`` (see
    ``scene_understanding.py`` and ``config.py`` for the ``sam2_post_filter_*``
    threshold rationale) -- kept in sync manually since the staged chain
    builds its objects list independently of the monolith's stage-3 loop.

    Returns ``"pass"`` or the name of the first gate that rejected the
    detection, so callers can report *why* real detections got dropped
    (these thresholds were explicitly flagged as unvalidated against real
    model output when written -- this return value is how that finally
    gets validated instead of guessed at)."""
    area = int(np.asarray(mask_bin).sum())
    min_area_px = int(getattr(cfg, "sam2_post_filter_min_area_px", 0)) if cfg else 0
    if area < min_area_px:
        return f"min_area_px (area={area} < {min_area_px})"
    frame_area = max(1, int(h) * int(w))
    max_area_fraction = float(getattr(cfg, "sam2_post_filter_max_area_fraction", 1.0)) if cfg else 1.0
    if area / frame_area > max_area_fraction:
        return f"max_area_fraction ({area / frame_area:.3f} > {max_area_fraction})"
    min_stability = float(getattr(cfg, "sam2_post_filter_min_stability", 0.0)) if cfg else 0.0
    stability = det.get("stability_score")
    if stability is not None and float(stability) < min_stability:
        return f"min_stability ({float(stability):.3f} < {min_stability})"
    min_pred_iou = float(getattr(cfg, "sam2_post_filter_min_pred_iou", 0.0)) if cfg else 0.0
    pred_iou = det.get("predicted_iou")
    if pred_iou is not None and float(pred_iou) < min_pred_iou:
        return f"min_pred_iou ({float(pred_iou):.3f} < {min_pred_iou})"
    min_conf = float(getattr(cfg, "grounded_sam2_min_conf_for_stage3", 0.0)) if cfg else 0.0
    conf = det.get("gdino_conf", det.get("predicted_iou"))
    if conf is not None and float(conf) < min_conf:
        return f"grounded_sam2_min_conf_for_stage3 ({float(conf):.3f} < {min_conf})"
    return "pass"


def _mask_iou(a: Any, b: Any) -> float:
    a_bin, b_bin = np.asarray(a) > 0, np.asarray(b) > 0
    if a_bin.shape != b_bin.shape:
        return 0.0
    inter = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def _combine_object_evidence(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Keep ``a``'s identity/geometry (first-seen wins, arbitrary but stable)
    but merge in ``b``'s naming evidence: more aliases, and ``b``'s
    canonical_name/label/caption take over if ``b`` was labelled with
    higher confidence."""
    merged = dict(a)
    combined_aliases = list(a.get("aliases", []))
    for alias in b.get("aliases", []):
        if alias not in combined_aliases:
            combined_aliases.append(alias)
    merged["aliases"] = combined_aliases[:8]
    if float(b.get("conf", 0.0)) > float(a.get("conf", 0.0)):
        for key in ("canonical_name", "label", "conf", "caption", "category"):
            if key in b:
                merged[key] = b[key]
    merged["merged_duplicate_ids"] = list(a.get("merged_duplicate_ids", [])) + [b.get("id")]
    return merged


def _merge_duplicate_objects(objects: List[Dict[str, Any]], iou_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Merge near-duplicate detections of the same real-world object (very
    high mutual mask IoU) into one, combining the richest available naming
    evidence -- distinct from ``regions/mask_hierarchy.py``'s containment-
    based ``object_object_part`` edges (a smaller mask nested inside a
    bigger one; genuinely different masks, just linked). This catches the
    other case: two masks covering ~the same region (e.g. GDINO matching
    the same real object under two different category strings), which
    would otherwise show up as two separate objects. This is the practical
    "AMG capability without AMG": dedup/part-of awareness from the single
    GDINO+SAM2 mask set, not a second dense mask generator."""
    consumed: set = set()
    merged_objects: List[Dict[str, Any]] = []
    for i, obj in enumerate(objects):
        if i in consumed:
            continue
        mask_i = obj.get("_sam2_mask_array")
        best = obj
        for j in range(i + 1, len(objects)):
            if j in consumed or mask_i is None:
                continue
            mask_j = objects[j].get("_sam2_mask_array")
            if mask_j is None:
                continue
            if _mask_iou(mask_i, mask_j) >= iou_threshold:
                consumed.add(j)
                best = _combine_object_evidence(best, objects[j])
        merged_objects.append(best)
    return merged_objects


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Attach labels and depth-backed geometry to each detection."""
    if ctx.metric_depth is None:
        return ctx
    cfg = getattr(pipeline, "config", None)
    label_mode = str(getattr(cfg, "post_sam_label_mode", "mask_chain") if cfg else "mask_chain").strip().lower()
    use_mask_chain = label_mode == "mask_chain" and hasattr(pipeline, "_label_mask")
    labellers = None
    with sub_timer("labelling.load_labellers"):
        if label_mode == "gdino_only":
            pass
        elif use_mask_chain:
            pipeline._load_labellers()
        else:
            labellers = pipeline._get_labeling_pipeline()
    objects: List[Dict[str, Any]] = []
    drop_reasons: List[str] = []
    lm = ctx.region_label_map
    try:
        from ..regions.partitioner import majority_region_index as _majority_ri
    except Exception:
        _majority_ri = None  # type: ignore[assignment]
    label_call_s = 0.0
    depth_stats_s = 0.0
    for idx, det in enumerate(ctx.detections):
        seg = det.get("segmentation")
        if seg is None:
            continue
        mask_bin = np.asarray(seg) > 0
        if mask_bin.sum() == 0:
            continue
        x, y, w, h = [int(v) for v in det.get("bbox", [0, 0, ctx.width, ctx.height])]
        x1, y1 = max(0, x), max(0, y)
        x2 = min(ctx.width, x + w)
        y2 = min(ctx.height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        _ri = 0
        if lm is not None and _majority_ri is not None:
            try:
                _ri = int(_majority_ri(mask_bin, lm))
            except Exception:
                _ri = 0
        amg_entry = dict(det)
        if "bbox" not in amg_entry or amg_entry.get("bbox") is None:
            amg_entry["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        t_label0 = time.time()
        # name_fields carries _label_mask()'s full open-vocab evidence-fusion
        # result (canonical_name/aliases/category/caption/florence2_caption/
        # rampp_caption/source_labels) when available (mask_chain mode) --
        # previously only `label`/`conf` were kept and everything else was
        # discarded, which is why app.py's existing preference for
        # obj.get('canonical_name') (app.py:85) never had anything to show.
        name_fields: Dict[str, Any] = {}
        if label_mode == "gdino_only":
            label_val = str(det.get("label", "object"))
            conf_val = float(det.get("gdino_conf", det.get("predicted_iou", 0.0)))
        elif use_mask_chain:
            det_out = pipeline._label_mask(ctx.img_bgr, mask_bin, amg_entry, label_map=lm, region_index=_ri)
            label_val = str(det_out.get("label", det.get("label", "object")))
            conf_val = float(det_out.get("conf", det.get("gdino_conf", 0.0)))
            name_fields = det_out
        else:
            assert labellers is not None
            crop = ctx.img_bgr[y1:y2, x1:x2].copy()
            label_payload = labellers.label_crop(crop)
            label_val = str(label_payload.get("label", det.get("label", "object")))
            conf_val = float(label_payload.get("conf", det.get("gdino_conf", 0.0)))
        this_label_s = time.time() - t_label0
        label_call_s += this_label_s
        # Real-time per-detection progress -- with Florence-2/RAM++ now
        # always consulted (mask_label_skip_secondary_when_gdino_specific
        # default flipped to False), this loop can take real, visible time
        # per mask; without a print here it silently looked "stuck" between
        # the load_labellers timing line and the loop's final summary.
        print(f"    [Labelling] {idx + 1}/{len(ctx.detections)}: '{label_val}' ({this_label_s:.2f}s)")
        filter_verdict = _passes_post_filter(mask_bin, det, cfg, ctx.height, ctx.width)
        if filter_verdict != "pass":
            drop_reasons.append(f"{label_val}: {filter_verdict}")
            continue
        t_depth0 = time.time()
        depth_stats, coords_3d, centroid = pipeline._mask_depth_stats_and_3d(
            metric_depth=ctx.metric_depth,
            K=ctx.intrinsics,
            mask=mask_bin,
            detection=det,
            use_erosion=True,
        )
        depth_stats_s += time.time() - t_depth0
        _oid = f"obj_{idx}"
        objects.append(
            {
                "id": _oid,
                "graph_id": _oid,
                "label": label_val,
                "conf": conf_val,
                "bbox": [x1, y1, x2, y2],
                "segmentor": str(det.get("source_model", "unknown")),
                "depth_stats": depth_stats,
                "coordinates_3d": coords_3d,
                "mask_centroid_2d": centroid,
                "_sam2_mask_array": mask_bin,
                # Open-vocab evidence-fusion naming (see name_fields comment above).
                "canonical_name": name_fields.get("canonical_name", label_val),
                "aliases": name_fields.get("aliases", [label_val]),
                "category": name_fields.get("category", ""),
                "caption": name_fields.get("caption", label_val),
                "florence2_label": name_fields.get("florence2_label", ""),
                "florence2_caption": name_fields.get("florence2_caption", ""),
                "rampp_label": name_fields.get("rampp_label", ""),
                "rampp_caption": name_fields.get("rampp_caption", ""),
                "rampp_tags": name_fields.get("rampp_tags", []),
                "source_labels": name_fields.get("source_labels", {}),
                "label_source": name_fields.get("source_model", "GroundingDINO"),
            }
        )
    n_det = len(ctx.detections)
    n_before_dedup = len(objects)
    objects = _merge_duplicate_objects(objects)
    if len(objects) < n_before_dedup:
        print(f"  [Labelling] merged {n_before_dedup - len(objects)} near-duplicate mask(s) (mutual IoU >= 0.85) into existing objects")
    ctx.extra["objects"] = objects
    print(f"    [Timing]   labelling.label_calls: {label_call_s:.2f}s total over {n_det} detections ({label_call_s / max(1, n_det):.3f}s/det)")
    print(f"    [Timing]   labelling.depth_stats: {depth_stats_s:.2f}s total over {n_before_dedup} kept objects")
    if drop_reasons:
        print(f"  [Labelling] {n_before_dedup}/{n_det} detections kept; {len(drop_reasons)} dropped by the post-hoc quality filter:")
        for reason in drop_reasons:
            print(f"    - {reason}")
    return ctx
