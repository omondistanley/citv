"""Staged scene JSON write path (package migration artifact)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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
    return {
        "id": str(obj.get("id", "")),
        "graph_id": _gid,
        "label": label,
        "confidence": conf,
        "conf": conf,
        "bbox": bbox,
        "segmentor": segmentor,
        "coordinates_3d": coords_3d,
        "depth_stats": depth_stats,
        "mask_centroid_2d": centroid,
        "sources": {
            "GroundedSAM2": {"label": label, "confidence": conf, "caption": label},
            "Florence2": {"label": label, "caption": label},
            "RAM++": {"label": label, "caption": label, "tags": [label] if label else []},
            "Pix2SG": {"relations": []},
        },
    }


def build_staged_scene_payload(ctx: PipelineContext) -> Dict[str, Any]:
    """Assemble the staged scene dict (metadata + objects + relations) without writing."""
    objects: List[Dict[str, Any]] = [normalize_staged_object(obj) for obj in ctx.extra.get("objects", [])]
    intrinsics = {k: float(v) for k, v in (ctx.intrinsics or {}).items()}
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
        "depth_map": f"depth/{ctx.stem}_depth_metric.npy",
        "segmentation_image": f"scene_graph/staged/{ctx.stem}_segmentation.png",
        "sam2_segmentation_image": f"scene_graph/staged/{ctx.stem}_segmentation.png",
        "sam2_tinted_overlay_image": f"scene_graph/staged/{ctx.stem}_overlay.png",
    }
    if ctx.sam2_metadata:
        for k, v in ctx.sam2_metadata.items():
            if k not in metadata:
                metadata[k] = v
    payload: Dict[str, Any] = {
        "metadata": metadata,
        "objects": objects,
        "relations": list(ctx.relations),
    }
    if ctx.mask_hierarchy is not None:
        payload["mask_hierarchy"] = ctx.mask_hierarchy
    if ctx.layers is not None:
        payload["layers"] = ctx.layers
    if ctx.regions_block is not None:
        payload["regions"] = ctx.regions_block
    if ctx.path_exports:
        payload["metadata"].update({k: v for k, v in ctx.path_exports.items() if k not in payload["metadata"]})
    return payload


def write_staged_scene_json(ctx: PipelineContext) -> Path:
    """Write ``scene_graph/staged/{stem}_scene.json`` and return the path."""
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    payload = build_staged_scene_payload(ctx)
    out_path = staged_dir / f"{ctx.stem}_scene.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def export_staged_package(ctx: PipelineContext) -> Dict[str, str]:
    """Write staged JSON; return dict with ``scene_json`` absolute path string."""
    out_path = write_staged_scene_json(ctx)
    return {"scene_json": str(out_path)}
