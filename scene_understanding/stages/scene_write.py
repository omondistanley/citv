"""Staged scene JSON write path (package migration artifact)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..evidence import normalize_relation
from ..pipeline_context import PipelineContext


def normalize_staged_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Shape staged object payload closer to legacy schema for parity diffs."""
    bbox = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])[:4]]
    label = str(obj.get("label", "object"))
    conf = float(obj.get("conf", 0.0))
    coords_3d = dict(obj.get("coordinates_3d", {"x": 0.0, "y": 0.0, "z": 0.0}))
    depth_stats = dict(obj.get("depth_stats", {}))
    centroid = list(obj.get("mask_centroid_2d", [0, 0]))
    segmentor = str(obj.get("segmentor", "package_staged_pipeline"))
    _gid = str(obj.get("graph_id") or obj.get("id", ""))
    raw_sources = obj.get("sources") or {}
    f2_raw = raw_sources.get("Florence2") or {}
    rampp_raw = raw_sources.get("RAM++") or {}
    gdino_raw = raw_sources.get("GroundedSAM2") or {}
    pix2sg_raw = raw_sources.get("Pix2SG") or {}
    return {
        "id": str(obj.get("id", "")),
        "graph_id": _gid,
        "label": label,
        "canonical_label": str(obj.get("canonical_label", label)),
        "label_source": str(obj.get("label_source", "")),
        "label_candidates": list(obj.get("label_candidates", [])),
        "rejected_labels": list(obj.get("rejected_labels", [])),
        "visual_quality_attributes": list(obj.get("visual_quality_attributes", [])),
        "source_agreement": float(obj.get("source_agreement", 0.0) or 0.0),
        "caption": str(obj.get("caption", label)),
        "confidence": conf,
        "conf": conf,
        "bbox": bbox,
        "segmentor": segmentor,
        "coordinates_3d": coords_3d,
        "depth_stats": depth_stats,
        "mask_centroid_2d": centroid,
        "depth_stats_no_erosion": dict(obj.get("depth_stats_no_erosion", depth_stats)),
        "coordinates_3d_no_erosion": dict(obj.get("coordinates_3d_no_erosion", coords_3d)),
        "mask_centroid_2d_no_erosion": list(obj.get("mask_centroid_2d_no_erosion", centroid)),
        "region_index": int(obj.get("region_index", 0) or 0),
        "region_id": str(obj.get("region_id", "")),
        "coordinates_3d_region_relative": dict(
            obj.get("coordinates_3d_region_relative", {"x": 0.0, "y": 0.0, "z": 0.0})
        ),
        "region_depth_percentile": float(obj.get("region_depth_percentile", 0.0) or 0.0),
        "depth_plausibility_score": float(obj.get("depth_plausibility_score", 1.0) or 1.0),
        **({"label_warning": str(obj.get("label_warning", ""))} if obj.get("label_warning") else {}),
        "layer_type": str(obj.get("layer_type", "unassigned")),
        "parent_object_id": obj.get("parent_object_id"),
        "child_object_ids": list(obj.get("child_object_ids", [])),
        "part_mask_ids": list(obj.get("part_mask_ids", [])),
        "sources": {
            "GroundedSAM2": {
                "label": gdino_raw.get("label", label),
                "confidence": float(gdino_raw.get("confidence", conf)),
                "caption": gdino_raw.get("caption", label),
            },
            "Florence2": {
                "label": f2_raw.get("label", label),
                "caption": f2_raw.get("caption", label),
            },
            "RAM++": {
                "label": rampp_raw.get("label", label),
                "caption": rampp_raw.get("caption", label),
                "tags": list(rampp_raw.get("tags", [label] if label else [])),
            },
            "Pix2SG": {"relations": list(pix2sg_raw.get("relations", []))},
        },
    }


def build_staged_scene_payload(ctx: PipelineContext) -> Dict[str, Any]:
    """Assemble the staged scene dict (metadata + objects + relations) without writing."""
    objects: List[Dict[str, Any]] = [normalize_staged_object(obj) for obj in ctx.extra.get("objects", [])]
    intrinsics = {k: float(v) for k, v in (ctx.intrinsics or {}).items()}
    stem = ctx.stem
    metadata: Dict[str, Any] = {
        "timestamp": ctx.timestamp,
        "segmentor": "package_staged_pipeline",
        "intrinsics": intrinsics,
        "models": [
            "DepthAnythingV2",
            "GroundedSAM2",
            "Florence2",
            "RAM++",
            "Pix2SG",
        ],
        "relation_sources": {"Pix2SG": {"active": True, "backend": "package_staged"}},
        "relation_debug": {
            "num_detections": len(ctx.detections),
            "num_objects": len(objects),
            "num_relations": len(ctx.relations),
        },
        "regions_enabled_staged": bool(ctx.region_partition_meta),
        # Depth
        "depth_map": f"depth/{stem}_depth_metric.npy",
        "depth_map_npz": f"depth/{stem}_depth.npz",
    }
    if ctx.regions_block is not None:
        metadata.update({
            "regions_json": f"scene_graph/staged/{stem}_regions.json",
            "regions_image": f"depth/{stem}_regions.png",
            "regions_overlay_image": f"scene_graph/staged/{stem}_regions_overlay.png",
            "region_adjacency_graph_json": f"scene_graph/staged/{stem}_region_adjacency_graph.json",
            "region_relations_json": f"scene_graph/staged/{stem}_region_relations.json",
            "region_relations_map_image": f"scene_graph/staged/{stem}_region_relations_map.png",
            "region_segmentation_image": f"scene_graph/staged/{stem}_region_segmentation.png",
            "region_sam2_segmentation_image": f"scene_graph/staged/{stem}_region_sam2_style_segmentation.png",
            "region_tinted_overlay_image": f"scene_graph/staged/{stem}_region_tinted_overlay.png",
        })
    if ctx.mask_hierarchy is not None:
        metadata["mask_hierarchy_json"] = f"scene_graph/staged/{stem}_mask_hierarchy.json"
        metadata["mask_hierarchy_detailed_json"] = f"scene_graph/staged/{stem}_mask_hierarchy_detailed.json"
        metadata["mask_hierarchy_levels_json"] = f"scene_graph/staged/{stem}_mask_hierarchy_levels.json"
        metadata["mask_hierarchy_image"] = f"scene_graph/staged/{stem}_mask_hierarchy.png"
    if ctx.layers is not None:
        metadata["layers_json"] = f"scene_graph/staged/{stem}_layers.json"
        metadata["layers_image"] = f"scene_graph/staged/{stem}_layers.png"
    # Object visualization images (populated by visualization_export stage)
    metadata["segmentation_image"] = f"scene_graph/staged/{stem}_segmentation.png"
    metadata["sam2_segmentation_image"] = f"scene_graph/staged/{stem}_segmentation.png"
    metadata["sam2_tinted_overlay_image"] = f"scene_graph/staged/{stem}_overlay.png"
    metadata["overlay_image"] = f"scene_graph/staged/{stem}_overlay.png"
    metadata["3d_viz_image"] = f"scene_graph/staged/{stem}_3d_viz.png"
    metadata["relations_map_image"] = f"scene_graph/staged/{stem}_relations_map.png"
    metadata["relations_map_objects_image"] = f"scene_graph/staged/{stem}_relations_map_objects.png"
    metadata["relations_map_regions_image"] = f"scene_graph/staged/{stem}_relations_map_regions.png"
    if ctx.sam2_metadata:
        for k, v in ctx.sam2_metadata.items():
            if k not in metadata:
                metadata[k] = v
    payload: Dict[str, Any] = {
        "metadata": metadata,
        "objects": objects,
        "relations": list(ctx.relations),
        "relations_normalized": [
            {k: v for k, v in normalize_relation(r).items() if k != "raw"}
            for r in list(ctx.relations)
        ],
    }
    if ctx.mask_hierarchy is not None:
        payload["mask_hierarchy"] = ctx.mask_hierarchy
    if ctx.layers is not None:
        payload["layers"] = ctx.layers
    if ctx.regions_block is not None:
        payload["regions"] = ctx.regions_block
    if ctx.caption_evidence is not None:
        payload["caption_evidence_summary"] = dict(ctx.caption_evidence.get("summary") or {})
    if ctx.scene_affordances is not None:
        payload["scene_affordance_summary"] = dict(ctx.scene_affordances.get("summary") or {})
    if ctx.object_affordances is not None:
        payload["object_affordance_summary"] = {
            "schema": ctx.object_affordances.get("schema"),
            "object_count": ctx.object_affordances.get("object_count", 0),
        }
    if ctx.mask_affordances is not None:
        payload["mask_affordance_summary"] = {
            "schema": ctx.mask_affordances.get("schema"),
            "mask_count": ctx.mask_affordances.get("mask_count", 0),
            "supported_path_modes": list(ctx.mask_affordances.get("supported_path_modes") or []),
        }
    if ctx.action_hypotheses is not None:
        payload["action_hypothesis_summary"] = dict(ctx.action_hypotheses.get("summary") or {})
    if ctx.path_exports:
        payload["metadata"].update({k: v for k, v in ctx.path_exports.items() if k not in payload["metadata"]})
    return payload


def write_staged_scene_json(ctx: PipelineContext) -> Path:
    """Write ``scene_graph/staged/{stem}_scene.json`` and return the path."""
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    payload = build_staged_scene_payload(ctx)
    out_path = staged_dir / f"{ctx.stem}_scene.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def export_staged_package(ctx: PipelineContext) -> Dict[str, str]:
    """Write staged JSON; return dict with ``scene_json`` absolute path string."""
    out_path = write_staged_scene_json(ctx)
    return {"scene_json": str(out_path)}
