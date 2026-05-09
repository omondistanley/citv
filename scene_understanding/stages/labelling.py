"""Labelling + per-mask depth geometry for staged pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..action_ontology import load_action_ontology
from ..evidence import build_label_candidates
from ..pipeline_context import PipelineContext


_LABEL_DEPTH_PRIORS_M: Dict[str, Tuple[float, float]] = {
    "person": (0.2, 100.0),
    "people": (0.2, 100.0),
    "man": (0.2, 100.0),
    "woman": (0.2, 100.0),
    "child": (0.2, 100.0),
    "baby": (0.2, 100.0),
    "sky": (2.0, 5000.0),
    "cloud": (2.0, 5000.0),
    "road": (0.5, 200.0),
    "floor": (0.2, 80.0),
    "ground": (0.2, 500.0),
}


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Attach labels and depth-backed geometry to each detection."""
    if ctx.metric_depth is None:
        return ctx
    cfg = getattr(pipeline, "config", None)
    label_mode = str(getattr(cfg, "post_sam_label_mode", "bbox_florence") if cfg else "bbox_florence").strip().lower()
    ontology = load_action_ontology(cfg)
    labellers = None
    if label_mode == "gdino_only":
        pass
    else:
        labellers = pipeline._get_labeling_pipeline()
    objects: List[Dict[str, Any]] = []
    lm = ctx.region_label_map
    n_det = len(ctx.detections)
    try:
        from ..regions.partitioner import majority_region_index as _majority_ri
    except Exception:
        _majority_ri = None  # type: ignore[assignment]
    print(f"  [Labelling] {n_det} detections → labelling + depth geometry...")
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
        gdino_label = str(det.get("label", "object"))
        gdino_conf = float(det.get("gdino_conf", det.get("predicted_iou", 0.0)))
        if (idx + 1) % 5 == 0 or idx == 0 or idx == n_det - 1:
            print(f"  [Labelling] object {idx + 1}/{n_det}...")
        label_payload: Dict[str, Any] = {}
        if label_mode == "gdino_only":
            label_val = gdino_label
            conf_val = gdino_conf
        else:
            assert labellers is not None
            crop = ctx.img_bgr[y1:y2, x1:x2].copy()
            crop_mask = mask_bin[y1:y2, x1:x2]
            if crop.size and crop_mask.shape == crop.shape[:2]:
                crop = crop.copy()
                bg = ctx.img_bgr.mean(axis=(0, 1)).astype(np.uint8)
                crop[~crop_mask] = bg
            label_payload = labellers.label_crop(crop)
        label_fusion = build_label_candidates(
            gdino_label=gdino_label,
            gdino_conf=gdino_conf,
            florence_label=label_payload.get("label", ""),
            florence_caption=label_payload.get("caption", ""),
            rampp_label=label_payload.get("label", "") if label_payload.get("tags") else "",
            rampp_tags=list(label_payload.get("tags", []) or []),
            current_label=det.get("label", "object"),
            ontology=ontology,
        )
        if label_mode != "gdino_only":
            label_val = str(label_fusion.get("canonical_label") or label_payload.get("label", det.get("label", "object")))
            best = (label_fusion.get("label_candidates") or [{}])[0]
            conf_val = float(best.get("score", label_payload.get("conf", det.get("gdino_conf", 0.0))))
        region_context = _region_by_index(ctx.region_partition_meta).get(_ri)
        depth_stats, coords_3d, centroid = pipeline._mask_depth_stats_and_3d(
            metric_depth=ctx.metric_depth,
            K=ctx.intrinsics,
            mask=mask_bin,
            detection=det,
            use_erosion=True,
            region_context=region_context,
            label_map=lm,
            region_index=_ri,
        )
        depth_stats_raw, coords_3d_raw, centroid_raw = pipeline._mask_depth_stats_and_3d(
            metric_depth=ctx.metric_depth,
            K=ctx.intrinsics,
            mask=mask_bin,
            detection=det,
            use_erosion=False,
            region_context=region_context,
            label_map=lm,
            region_index=_ri,
        )
        _oid = f"obj_{idx}"
        region_id = f"region_{_ri}" if _ri > 0 else ""
        rel3 = _region_relative_coords(coords_3d, region_context, ctx.intrinsics, pipeline)
        depth_percentile = _region_depth_percentile(depth_stats.get("z_val", 0.0), region_context)
        plausibility, warning = _depth_plausibility(label_val, depth_stats.get("z_val", 0.0))
        objects.append(
            {
                "id": _oid,
                "graph_id": _oid,
                "label": label_val,
                "canonical_label": label_val,
                "label_source": str(label_fusion.get("label_source", "")),
                "label_candidates": list(label_fusion.get("label_candidates", [])),
                "rejected_labels": list(label_fusion.get("rejected_labels", [])),
                "visual_quality_attributes": list(label_fusion.get("visual_quality_attributes", [])),
                "source_agreement": float(label_fusion.get("source_agreement", 0.0) or 0.0),
                "caption": str(label_payload.get("caption", label_val)),
                "conf": conf_val,
                "bbox": [x1, y1, x2, y2],
                "segmentor": str(det.get("source_model", "unknown")),
                "depth_stats": depth_stats,
                "coordinates_3d": coords_3d,
                "mask_centroid_2d": centroid,
                "depth_stats_no_erosion": depth_stats_raw,
                "coordinates_3d_no_erosion": coords_3d_raw,
                "mask_centroid_2d_no_erosion": centroid_raw,
                "region_index": _ri,
                "region_id": region_id,
                "coordinates_3d_region_relative": rel3,
                "region_depth_percentile": depth_percentile,
                "depth_plausibility_score": plausibility,
                **({"label_warning": warning} if warning else {}),
                "sources": {
                    "GroundedSAM2": {
                        "label": gdino_label,
                        "confidence": gdino_conf,
                        "caption": gdino_label,
                    },
                    "Florence2": {
                        "label": str(label_payload.get("label", label_val)),
                        "caption": str(label_payload.get("caption", label_val)),
                    },
                    "RAM++": {
                        "label": str(label_payload.get("label", "")) if label_payload.get("tags") else "",
                        "caption": str(label_payload.get("caption", "")) if label_payload.get("tags") else "",
                        "tags": list(label_payload.get("tags", [])),
                    },
                    "Pix2SG": {"relations": []},
                },
                "_sam2_mask_array": mask_bin,
            }
        )
    print(f"  [Labelling] done: {len(objects)} objects labelled")
    ctx.extra["objects"] = objects
    ctx.objects_3d = objects
    _build_region_scene_context(pipeline, ctx, objects)
    return ctx


def _region_by_index(regions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(r.get("region_index", 0) or 0): r for r in regions if int(r.get("region_index", 0) or 0) > 0}


def _region_relative_coords(
    coords_3d: Dict[str, Any],
    region: Optional[Dict[str, Any]],
    intrinsics: Dict[str, float],
    pipeline: Any,
) -> Dict[str, float]:
    if not region:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    c = region.get("centroid_2d_px") or [0, 0]
    rs = region.get("depth_stats") or {}
    z = float(rs.get("mean", rs.get("mode", coords_3d.get("z", 0.0))) or 0.0)
    try:
        rc = pipeline._back_project(int(float(c[0])), int(float(c[1])), z, intrinsics)
    except Exception:
        rc = {"x": 0.0, "y": 0.0, "z": z}
    return {
        "x": round(float(coords_3d.get("x", 0.0)) - float(rc.get("x", 0.0)), 4),
        "y": round(float(coords_3d.get("y", 0.0)) - float(rc.get("y", 0.0)), 4),
        "z": round(float(coords_3d.get("z", 0.0)) - float(rc.get("z", 0.0)), 4),
    }


def _region_depth_percentile(z_val: Any, region: Optional[Dict[str, Any]]) -> float:
    if not region:
        return 0.0
    ds = region.get("depth_stats") or {}
    z = float(z_val or 0.0)
    zmin = float(ds.get("min", 0.0) or 0.0)
    zmax = float(ds.get("max", 0.0) or 0.0)
    if zmax <= zmin:
        return 0.5
    return round(float(np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)), 4)


def _depth_plausibility(label: str, z_val: Any) -> Tuple[float, str]:
    tokens = {t for t in str(label).lower().replace("_", " ").split() if t}
    z = float(z_val or 0.0)
    for token in tokens:
        if token in _LABEL_DEPTH_PRIORS_M:
            lo, hi = _LABEL_DEPTH_PRIORS_M[token]
            if lo <= z <= hi:
                return 1.0, ""
            return 0.25, "depth_prior_mismatch"
    return 1.0, ""


def _build_region_scene_context(pipeline: Any, ctx: PipelineContext, objects: List[Dict[str, Any]]) -> None:
    if ctx.region_label_map is None or not ctx.region_partition_meta:
        return
    regions = [dict(r) for r in ctx.region_partition_meta]
    by_idx = _region_by_index(regions)
    for r in regions:
        r["object_ids"] = []
        r["layer_type"] = str(r.get("type", "unassigned"))
    for obj in objects:
        ri = int(obj.get("region_index", 0) or 0)
        if ri in by_idx:
            by_idx[ri].setdefault("object_ids", []).append(str(obj.get("id", "")))

    adjacency = _build_region_adjacency(ctx.region_label_map, regions)
    ctx.regions_block = {
        "image_stem": ctx.stem,
        "width": ctx.width,
        "height": ctx.height,
        "depth_model": str(getattr(pipeline.depth_estimator, "backend_name", "")),
        "regions_json": f"scene_graph/staged/{ctx.stem}_regions.json",
        "regions_image": f"depth/{ctx.stem}_regions.png",
        "regions_overlay_image": f"scene_graph/staged/{ctx.stem}_regions_overlay.png",
        "region_adjacency_graph_json": f"scene_graph/staged/{ctx.stem}_region_adjacency_graph.json",
        "region_relations_json": f"scene_graph/staged/{ctx.stem}_region_relations.json",
        "region_relations_map_image": f"scene_graph/staged/{ctx.stem}_region_relations_map.png",
        "regions": regions,
        "adjacency": adjacency["adjacency"],
        "region_adjacency_graph": adjacency,
    }
    ctx.layers = _build_layers_payload(objects, regions)
    ctx.mask_hierarchy = _build_mask_hierarchy_payload(ctx, objects, regions)
    _write_staged_context_artifacts(ctx, objects, regions, adjacency)


def _build_region_adjacency(label_map: np.ndarray, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    lm = np.asarray(label_map, dtype=np.int32)
    pairs: Dict[Tuple[int, int], int] = {}
    for a, b in ((lm[:, :-1], lm[:, 1:]), (lm[:-1, :], lm[1:, :])):
        mask = (a > 0) & (b > 0) & (a != b)
        aa = a[mask].ravel()
        bb = b[mask].ravel()
        for x, y in zip(aa.tolist(), bb.tolist()):
            key = tuple(sorted((int(x), int(y))))
            pairs[key] = pairs.get(key, 0) + 1
    idx_to_region = _region_by_index(regions)
    adjacency: Dict[str, List[str]] = {str(r.get("id", "")): [] for r in regions}
    edges: List[Dict[str, Any]] = []
    for (ia, ib), border in sorted(pairs.items()):
        ra = idx_to_region.get(ia)
        rb = idx_to_region.get(ib)
        if not ra or not rb:
            continue
        ida = str(ra.get("id", f"region_{ia}"))
        idb = str(rb.get("id", f"region_{ib}"))
        adjacency.setdefault(ida, []).append(idb)
        adjacency.setdefault(idb, []).append(ida)
        za = float((ra.get("depth_stats") or {}).get("mean", 0.0) or 0.0)
        zb = float((rb.get("depth_stats") or {}).get("mean", 0.0) or 0.0)
        edges.append({
            "subject": ida,
            "predicate": "adjacent_to",
            "object": idb,
            "edge_type": "region_spatial_adjacency",
            "shared_border_px": int(border),
            "depth_delta_m": round(abs(za - zb), 4),
        })
    return {"kind": "region_adjacency_graph", "adjacency": adjacency, "edges": edges, "num_edges": len(edges)}


def _build_layers_payload(objects: List[Dict[str, Any]], regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    zs = [float((o.get("depth_stats") or {}).get("z_val", 0.0) or 0.0) for o in objects]
    valid = [z for z in zs if z > 0]
    q1, q2 = (np.quantile(valid, [1 / 3, 2 / 3]).tolist() if len(valid) >= 3 else [0.0, 0.0])
    bands: Dict[str, List[str]] = {"foreground": [], "midground": [], "background": [], "unassigned": []}
    for obj in objects:
        z = float((obj.get("depth_stats") or {}).get("z_val", 0.0) or 0.0)
        if z <= 0 or q1 <= 0:
            layer = "unassigned"
        elif z <= q1:
            layer = "foreground"
        elif z <= q2:
            layer = "midground"
        else:
            layer = "background"
        obj["layer_type"] = layer
        bands.setdefault(layer, []).append(str(obj.get("id", "")))
    return {
        "depth_quantiles": {"foreground_max_z": round(float(q1), 4), "midground_max_z": round(float(q2), 4)},
        "bands": [{"layer_type": k, "object_ids": v} for k, v in bands.items()],
        "regions": [{"region_id": str(r.get("id", "")), "layer_type": str(r.get("type", "unassigned"))} for r in regions],
    }


def _build_mask_hierarchy_payload(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        from ..regions.mask_hierarchy import build_mask_hierarchy
    except Exception:
        return {"edges": [], "root_object_ids": [str(o.get("id", "")) for o in objects], "num_edges": 0}
    nodes = []
    lm = np.asarray(ctx.region_label_map, dtype=np.int32)
    for r in regions:
        ridx = int(r.get("region_index", 0) or 0)
        if ridx <= 0:
            continue
        nodes.append({
            "id": str(r.get("id", f"region_{ridx}")),
            "label": str(r.get("type", "region")),
            "entity_kind": "region",
            "sam2_mask_index": ridx,
            "_sam2_mask_array": lm == ridx,
        })
    nodes.extend(objects)
    return build_mask_hierarchy(nodes)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_staged_context_artifacts(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    adjacency: Dict[str, Any],
) -> None:
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    if ctx.regions_block:
        _write_json(ctx.regions_block, staged_dir / f"{ctx.stem}_regions.json")
    _write_json(adjacency, staged_dir / f"{ctx.stem}_region_adjacency_graph.json")
    if ctx.mask_hierarchy:
        _write_json(ctx.mask_hierarchy, staged_dir / f"{ctx.stem}_mask_hierarchy.json")
    if ctx.layers:
        _write_json(ctx.layers, staged_dir / f"{ctx.stem}_layers.json")
    _write_regions_overlay(ctx, regions, adjacency, staged_dir / f"{ctx.stem}_regions_overlay.png")
    try:
        from ..visualization.layers_map import save_layers_map_bgr
        save_layers_map_bgr(ctx.img_bgr, objects, staged_dir / f"{ctx.stem}_layers.png", regions_meta=regions)
    except Exception as exc:
        print(f"  [Layers] map skipped: {exc}")


def _write_regions_overlay(
    ctx: PipelineContext,
    regions: List[Dict[str, Any]],
    adjacency: Dict[str, Any],
    out_path: Path,
) -> None:
    if ctx.region_label_map is None:
        return
    canvas = ctx.img_bgr.copy()
    lm = np.asarray(ctx.region_label_map, dtype=np.int32)
    edge = np.zeros(lm.shape, dtype=np.uint8)
    edge[:, 1:] |= (lm[:, 1:] != lm[:, :-1]).astype(np.uint8)
    edge[1:, :] |= (lm[1:, :] != lm[:-1, :]).astype(np.uint8)
    edge[lm <= 0] = 0
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1) > 0
    canvas[edge] = (0, 255, 255)
    centroids = {}
    for r in regions:
        rid = str(r.get("id", ""))
        c = r.get("centroid_2d_px") or [0, 0]
        cx = int(np.clip(float(c[0]), 0, ctx.width - 1))
        cy = int(np.clip(float(c[1]), 0, ctx.height - 1))
        centroids[rid] = (cx, cy)
        cv2.circle(canvas, (cx, cy), 5, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(canvas, rid, (cx + 7, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA)
    for e in adjacency.get("edges", []):
        a = centroids.get(str(e.get("subject", "")))
        b = centroids.get(str(e.get("object", "")))
        if a and b:
            cv2.line(canvas, a, b, (0, 180, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
