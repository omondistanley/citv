"""Final staged parity exports and all-in-one scene/action bundle.

This stage is intentionally CPU-first. It does not run vision models. It
derives monolith-compatible files, manifests, and QA/failure records from the
staged artifacts already produced earlier in the pipeline.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..evidence import normalize_relation, object_id, text_blob
from ..pipeline_context import PipelineContext
from . import scene_write


_MAX_READABLE_PATHS = 15


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(ctx: PipelineContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.output_dir))
    except ValueError:
        return str(path)


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return True


def _record_parity_step_failure(
    ctx: PipelineContext,
    failures: List[Dict[str, Any]],
    warnings: List[str],
    artifact: str,
    path: Path,
    exc: Exception,
) -> None:
    reason = f"{artifact} failed: {type(exc).__name__}: {exc}"
    rec = _failure_record(ctx, artifact, path, reason)
    failures.append(rec)
    warnings.append(reason)
    try:
        _write_json(rec, path)
    except Exception:
        pass


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Write final parity artifacts, manifest, bundle, and scene JSON."""
    cfg = getattr(pipeline, "config", None)
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    paths_root = staged_dir / f"{ctx.stem}_paths"
    staged_dir.mkdir(parents=True, exist_ok=True)
    paths_root.mkdir(parents=True, exist_ok=True)

    failures: List[Dict[str, Any]] = list(ctx.extra.get("artifact_failures") or [])
    warnings: List[str] = list(ctx.extra.get("artifact_warnings") or [])

    for artifact, failure_path, fn in [
        ("core_scene_parity", staged_dir / f"{ctx.stem}_core_scene_parity_failure.json", lambda: _ensure_core_scene_files(ctx, staged_dir)),
        ("caption_parity", staged_dir / f"{ctx.stem}_caption_parity_failure.json", lambda: _write_caption_parity(ctx, staged_dir)),
        ("affordance_parity", staged_dir / f"{ctx.stem}_affordance_parity_failure.json", lambda: _ensure_affordance_files(ctx, staged_dir)),
        ("path_parity", paths_root / "path_parity_failure.json", lambda: _write_path_parity(ctx, paths_root, warnings, failures)),
        ("animation_qa_parity", paths_root / "animation_qa_parity_failure.json", lambda: _write_qa_failure_records(ctx, cfg, paths_root, failures, warnings)),
    ]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _record_parity_step_failure(ctx, failures, warnings, artifact, failure_path, exc)

    manifest_path = staged_dir / f"{ctx.stem}_artifact_manifest.json"
    bundle_path = staged_dir / f"{ctx.stem}_scene_action_bundle.json"
    scene_path = scene_write.write_staged_scene_json(ctx)
    ctx.path_exports["scene_json"] = _rel(ctx, scene_path)
    ctx.path_exports["artifact_manifest_json"] = _rel(ctx, manifest_path)
    ctx.path_exports["scene_action_bundle_json"] = _rel(ctx, bundle_path)

    # Placeholders ensure the manifest does not mark itself or the final bundle
    # as missing while it is being built.
    _write_json(
        {
            "schema": "citv_artifact_manifest_v1",
            "status": "building",
            "stem": ctx.stem,
            "timestamp": ctx.timestamp,
        },
        manifest_path,
    )
    _write_json(
        {
            "schema": "citv_scene_action_bundle_v1",
            "status": "building",
            "stem": ctx.stem,
            "timestamp": ctx.timestamp,
        },
        bundle_path,
    )

    try:
        manifest = _build_artifact_manifest(ctx, failures, warnings, cfg)
    except Exception as exc:  # noqa: BLE001
        _record_parity_step_failure(
            ctx,
            failures,
            warnings,
            "artifact_manifest",
            staged_dir / f"{ctx.stem}_artifact_manifest_failure.json",
            exc,
        )
        manifest = {
            "schema": "citv_artifact_manifest_v1",
            "version": "1.0",
            "stem": ctx.stem,
            "timestamp": ctx.timestamp,
            "records": [],
            "missing_artifacts": [],
            "failed_artifacts": failures,
            "warnings": sorted(set(str(w) for w in warnings if str(w).strip())),
            "summary": {"expected_count": 0, "present_count": 0, "missing_count": 0, "invalid_count": 0, "failed_count": len(failures)},
        }
    _write_json(manifest, manifest_path)

    try:
        bundle = _build_scene_action_bundle(ctx, manifest)
    except Exception as exc:  # noqa: BLE001
        _record_parity_step_failure(
            ctx,
            failures,
            warnings,
            "scene_action_bundle",
            staged_dir / f"{ctx.stem}_scene_action_bundle_failure.json",
            exc,
        )
        bundle = {
            "schema": "citv_scene_action_bundle_v1",
            "version": "1.0",
            "status": "fallback_after_bundle_failure",
            "stem": ctx.stem,
            "timestamp": ctx.timestamp,
            "scene_json": f"scene_graph/staged/{ctx.stem}_scene.json",
            "artifact_manifest": f"scene_graph/staged/{ctx.stem}_artifact_manifest.json",
            "failed_artifacts": failures,
        }
    _write_json(bundle, bundle_path)

    # Re-write scene JSON after manifest/bundle paths are known.
    scene_path = scene_write.write_staged_scene_json(ctx)
    ctx.path_exports["scene_json"] = _rel(ctx, scene_path)

    print(
        f"  [ParityExport] {ctx.stem}: manifest={manifest_path.name}, "
        f"bundle={bundle_path.name}, missing={len(manifest.get('missing_artifacts', []))}, "
        f"failed={len(manifest.get('failed_artifacts', []))}"
    )
    return ctx


def _ensure_core_scene_files(ctx: PipelineContext, staged_dir: Path) -> None:
    stem = ctx.stem
    objects = list(ctx.extra.get("objects") or [])

    relations_path = staged_dir / f"{stem}_relations.json"
    if not relations_path.exists():
        _write_json(list(ctx.relations), relations_path)
    ctx.path_exports["relations_json"] = _rel(ctx, relations_path)

    if ctx.layers is None:
        ctx.layers = {
            "schema": "citv_layers_v1",
            "status": "parity_fallback",
            "layers": [],
            "objects": [
                {
                    "id": str(o.get("id", "")),
                    "label": str(o.get("canonical_label") or o.get("label", "object")),
                    "layer_type": str(o.get("layer_type", "unassigned")),
                    "region_id": str(o.get("region_id", "")),
                }
                for o in objects
            ],
        }
    layers_path = staged_dir / f"{stem}_layers.json"
    if not layers_path.exists():
        _write_json(ctx.layers, layers_path)
    ctx.path_exports["layers_json"] = _rel(ctx, layers_path)

    if ctx.mask_hierarchy is None:
        ctx.mask_hierarchy = {
            "schema": "citv_mask_hierarchy_v1",
            "status": "parity_fallback",
            "nodes": [
                {
                    "id": str(o.get("id", "")),
                    "label": str(o.get("canonical_label") or o.get("label", "object")),
                    "parent_id": o.get("parent_object_id"),
                    "children": list(o.get("child_object_ids", [])),
                }
                for o in objects
            ],
        }
    hierarchy_path = staged_dir / f"{stem}_mask_hierarchy.json"
    if not hierarchy_path.exists():
        _write_json(ctx.mask_hierarchy, hierarchy_path)
    ctx.path_exports["mask_hierarchy_json"] = _rel(ctx, hierarchy_path)

    detailed_path = staged_dir / f"{stem}_mask_hierarchy_detailed.json"
    if not detailed_path.exists():
        _write_json(_hierarchy_with_depth(ctx.mask_hierarchy), detailed_path)
    ctx.path_exports["mask_hierarchy_detailed_json"] = _rel(ctx, detailed_path)

    levels_path = staged_dir / f"{stem}_mask_hierarchy_levels.json"
    if not levels_path.exists():
        _write_json(_hierarchy_levels_payload(ctx), levels_path)
    ctx.path_exports["mask_hierarchy_levels_json"] = _rel(ctx, levels_path)

    if ctx.regions_block is None:
        ctx.regions_block = {
            "schema": "citv_regions_v1",
            "status": "parity_fallback",
            "regions": list(ctx.region_partition_meta),
            "adjacency": {},
        }
    regions_path = staged_dir / f"{stem}_regions.json"
    if not regions_path.exists():
        _write_json(ctx.regions_block, regions_path)
    ctx.path_exports["regions_json"] = _rel(ctx, regions_path)

    adjacency = dict((ctx.regions_block or {}).get("adjacency") or {})
    adjacency_path = staged_dir / f"{stem}_region_adjacency_graph.json"
    if not adjacency_path.exists():
        _write_json(
            {
                "schema": "citv_region_adjacency_graph_v1",
                "status": "parity_fallback" if not adjacency else "derived_from_regions",
                "regions": list((ctx.regions_block or {}).get("regions") or []),
                "adjacency": adjacency,
            },
            adjacency_path,
        )
    ctx.path_exports["region_adjacency_graph_json"] = _rel(ctx, adjacency_path)

    region_rel_path = staged_dir / f"{stem}_region_relations.json"
    if not region_rel_path.exists():
        relations = []
        for rid, neighbours in adjacency.items():
            for nid in neighbours if isinstance(neighbours, list) else []:
                relations.append({"subject": str(rid), "predicate": "adjacent_to", "object": str(nid), "source": "region_adjacency"})
        _write_json(
            {
                "schema": "citv_region_relations_v1",
                "status": "parity_fallback" if not relations else "derived_from_region_adjacency",
                "regions": list((ctx.regions_block or {}).get("regions") or []),
                "adjacency": adjacency,
                "relations": relations,
            },
            region_rel_path,
        )
    ctx.path_exports["region_relations_json"] = _rel(ctx, region_rel_path)


def _hierarchy_with_depth(hierarchy: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(hierarchy or {})
    nodes = []
    for node in list(payload.get("nodes") or []):
        if isinstance(node, dict):
            row = dict(node)
            row.setdefault("containment_depth", 0)
            nodes.append(row)
    payload["nodes"] = nodes
    payload.setdefault("schema", "citv_mask_hierarchy_detailed_v1")
    return payload


def _hierarchy_levels_payload(ctx: PipelineContext) -> Dict[str, Any]:
    nodes = []
    hierarchy = ctx.mask_hierarchy or {}
    for node in list(hierarchy.get("nodes") or []):
        if isinstance(node, dict):
            nodes.append(
                {
                    "id": str(node.get("id", "")),
                    "label": str(node.get("label", "")),
                    "containment_depth": int(node.get("containment_depth", 0) or 0),
                }
            )
    return {"schema": "citv_mask_hierarchy_levels_v1", "levels": {"0": nodes}, "level_count": 1}


def _write_caption_parity(ctx: PipelineContext, staged_dir: Path) -> None:
    stem = ctx.stem
    tiers = dict(ctx.extra.get("caption_tiers") or {})
    bundle = dict(ctx.extra.get("caption_bundle") or {})
    tiers, bundle = _ensure_caption_tier_files(ctx, staged_dir, tiers, bundle)
    objects = list(ctx.extra.get("objects") or [])
    per_obj = list((tiers.get("per_object") or {}).get("objects") or [])
    global_tier = dict(tiers.get("global_scene") or {})
    global_caption = str(global_tier.get("caption") or bundle.get("global_caption") or "")
    global_status = str(global_tier.get("status") or bundle.get("global_status") or "derived_from_staged")

    obj_json = {
        "variant": "florence_object_captions",
        "track": "staged",
        "image_path": str(ctx.image_path),
        "status": "derived_from_staged_caption_objects",
        "objects": per_obj,
        "object_count": len(per_obj),
    }
    obj_json_path = staged_dir / f"{stem}_florence_object_captions.json"
    _write_json(obj_json, obj_json_path)
    _write_text(_caption_objects_md(stem, per_obj), staged_dir / f"{stem}_florence_object_captions.md")
    ctx.path_exports["florence_object_captions_json"] = _rel(ctx, obj_json_path)
    ctx.path_exports["florence_object_captions_md"] = f"scene_graph/staged/{stem}_florence_object_captions.md"

    scene_json = {
        "variant": "florence_only",
        "track": "staged",
        "image_path": str(ctx.image_path),
        "generated_caption": global_caption,
        "status": global_status,
        "input_files": _caption_input_files(ctx),
    }
    scene_path = staged_dir / f"{stem}_florence_scene_caption.json"
    _write_json(scene_json, scene_path)
    _write_text(f"# Florence Scene Caption - staged\n\nStatus: {global_status}\n\n{global_caption}\n", staged_dir / f"{stem}_florence_scene_caption.md")
    ctx.path_exports["florence_scene_caption_json"] = _rel(ctx, scene_path)
    ctx.path_exports["florence_scene_caption_md"] = f"scene_graph/staged/{stem}_florence_scene_caption.md"

    fusion_prompt = _fusion_prompt(ctx, objects, global_caption)
    fusion_json = {
        "variant": "fusion_only",
        "track": "staged",
        "image_path": str(ctx.image_path),
        "prompt": fusion_prompt,
        "generated_caption": global_caption,
        "status": "derived_from_staged_global_and_scene_graph",
        "input_files": _caption_input_files(ctx),
    }
    fusion_path = staged_dir / f"{stem}_fusion_scene_caption.json"
    _write_json(fusion_json, fusion_path)
    _write_text(f"# Fusion Scene Caption - staged\n\n{fusion_prompt}\n", staged_dir / f"{stem}_fusion_scene_caption.md")
    ctx.path_exports["fusion_scene_caption_json"] = _rel(ctx, fusion_path)
    ctx.path_exports["fusion_scene_caption_md"] = f"scene_graph/staged/{stem}_fusion_scene_caption.md"

    hybrid_json = {
        "variant": "hybrid",
        "track": "staged",
        "image_path": str(ctx.image_path),
        "status": "derived_from_staged_caption_tiers",
        "generated_caption": global_caption,
        "inputs": {
            "florence_object_captions": f"scene_graph/staged/{stem}_florence_object_captions.json",
            "florence_scene_caption": f"scene_graph/staged/{stem}_florence_scene_caption.json",
            "fusion_scene_caption": f"scene_graph/staged/{stem}_fusion_scene_caption.json",
        },
    }
    hybrid_path = staged_dir / f"{stem}_hybrid_scene_caption.json"
    _write_json(hybrid_json, hybrid_path)
    _write_text(
        "# Hybrid Scene Caption - staged\n\n"
        "Status: derived from staged caption tiers.\n\n"
        f"{global_caption}\n",
        staged_dir / f"{stem}_hybrid_scene_caption.md",
    )
    ctx.path_exports["hybrid_scene_caption_json"] = _rel(ctx, hybrid_path)
    ctx.path_exports["hybrid_scene_caption_md"] = f"scene_graph/staged/{stem}_hybrid_scene_caption.md"

    comparison = {
        "track": "staged",
        "variants": [
            {"name": "florence_only", "file": f"scene_graph/staged/{stem}_florence_scene_caption.json"},
            {"name": "fusion_only", "file": f"scene_graph/staged/{stem}_fusion_scene_caption.json"},
            {"name": "hybrid", "file": f"scene_graph/staged/{stem}_hybrid_scene_caption.json"},
        ],
        "scoring_template": {
            "faithfulness_to_image": None,
            "scene_graph_consistency": None,
            "relation_quality": None,
            "detail_richness": None,
        },
    }
    comparison_path = staged_dir / f"{stem}_caption_comparison.json"
    _write_json(comparison, comparison_path)
    ctx.path_exports["caption_comparison_json"] = _rel(ctx, comparison_path)

    compat_bundle = {
        "track": "staged",
        "image_path": str(ctx.image_path),
        "files": {
            "florence_object_captions_json": f"scene_graph/staged/{stem}_florence_object_captions.json",
            "florence_object_captions_md": f"scene_graph/staged/{stem}_florence_object_captions.md",
            "florence_scene_caption_json": f"scene_graph/staged/{stem}_florence_scene_caption.json",
            "florence_scene_caption_md": f"scene_graph/staged/{stem}_florence_scene_caption.md",
            "fusion_scene_caption_json": f"scene_graph/staged/{stem}_fusion_scene_caption.json",
            "fusion_scene_caption_md": f"scene_graph/staged/{stem}_fusion_scene_caption.md",
            "hybrid_scene_caption_json": f"scene_graph/staged/{stem}_hybrid_scene_caption.json",
            "hybrid_scene_caption_md": f"scene_graph/staged/{stem}_hybrid_scene_caption.md",
            "caption_comparison_json": f"scene_graph/staged/{stem}_caption_comparison.json",
        },
    }
    compat_bundle_path = staged_dir / f"{stem}_hybrid_caption_bundle.json"
    _write_json(compat_bundle, compat_bundle_path)
    ctx.path_exports["hybrid_caption_bundle_json"] = _rel(ctx, compat_bundle_path)

    prompt_bundle = {
        "track": "staged",
        "image_path": str(ctx.image_path),
        "scene_json": f"scene_graph/staged/{stem}_scene.json",
        "prompt": fusion_prompt,
        "caption_files": compat_bundle["files"],
    }
    prompt_bundle_path = staged_dir / f"{stem}_caption_prompt_bundle.json"
    _write_json(prompt_bundle, prompt_bundle_path)
    _write_text(
        "# Track Comparison Prompt - staged\n\n"
        f"- Track staged: scene=scene_graph/staged/{stem}_scene.json, "
        f"relations=scene_graph/staged/{stem}_relations.json\n\n"
        f"{fusion_prompt}\n",
        staged_dir / f"{stem}_track_comparison_prompt.md",
    )
    ctx.path_exports["caption_prompt_bundle_json"] = _rel(ctx, prompt_bundle_path)
    ctx.path_exports["track_comparison_prompt_md"] = f"scene_graph/staged/{stem}_track_comparison_prompt.md"


def _ensure_caption_tier_files(
    ctx: PipelineContext,
    staged_dir: Path,
    tiers: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stem = ctx.stem
    objects = list(ctx.extra.get("objects") or [])
    try:
        from . import captions_export as stage_captions

        per_object_rows = stage_captions._build_per_object_captions(objects)
        per_region_rows = stage_captions._build_per_region_summaries(objects, ctx.region_partition_meta)
        cross_rows = stage_captions._build_cross_region_interactions(list(ctx.relations), objects)
        uncertainty_rows = stage_captions._build_uncertainty_notes(objects)
    except Exception:
        per_object_rows = [
            {
                "object_id": str(o.get("id", "")),
                "label": str(o.get("label", "object")),
                "canonical_label": str(o.get("canonical_label", o.get("label", "object"))),
                "caption": str(o.get("caption", "")),
            }
            for o in objects
        ]
        per_region_rows = []
        cross_rows = []
        uncertainty_rows = []

    global_tier = dict(tiers.get("global_scene") or {})
    if not global_tier:
        global_tier = {
            "tier": "global_scene",
            "stem": stem,
            "caption": str(bundle.get("global_caption", "")),
            "status": str(bundle.get("global_status", "parity_empty_fallback")),
            "image": str(ctx.image_path),
        }
    tier_payloads = {
        "global_scene": global_tier,
        "per_object": dict(tiers.get("per_object") or {
            "tier": "per_object",
            "stem": stem,
            "count": len(per_object_rows),
            "objects": per_object_rows,
        }),
        "per_region": dict(tiers.get("per_region") or {
            "tier": "per_region",
            "stem": stem,
            "count": len(per_region_rows),
            "regions": per_region_rows,
        }),
        "cross_region": dict(tiers.get("cross_region") or {
            "tier": "cross_region",
            "stem": stem,
            "count": len(cross_rows),
            "interactions": cross_rows,
        }),
        "uncertainty": dict(tiers.get("uncertainty") or {
            "tier": "uncertainty",
            "stem": stem,
            "count": len(uncertainty_rows),
            "notes": uncertainty_rows,
        }),
    }
    suffix_by_tier = {
        "global_scene": "caption_global",
        "per_object": "caption_objects",
        "per_region": "caption_regions",
        "cross_region": "caption_cross_region",
        "uncertainty": "caption_uncertainty",
    }
    for tier_name, payload in tier_payloads.items():
        suffix = suffix_by_tier[tier_name]
        path = staged_dir / f"{stem}_{suffix}.json"
        if not path.exists():
            _write_json(payload, path)
        ctx.path_exports[f"{suffix}_json"] = _rel(ctx, path)

    if not bundle:
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
            "global_caption": str(global_tier.get("caption", "")),
            "global_status": str(global_tier.get("status", "parity_empty_fallback")),
            "object_count": int(tier_payloads["per_object"].get("count", 0) or 0),
            "region_count": int(tier_payloads["per_region"].get("count", 0) or 0),
            "cross_region_count": int(tier_payloads["cross_region"].get("count", 0) or 0),
            "uncertainty_count": int(tier_payloads["uncertainty"].get("count", 0) or 0),
        }
    bundle_path = staged_dir / f"{stem}_caption_bundle.json"
    if not bundle_path.exists():
        _write_json(bundle, bundle_path)
    ctx.path_exports["caption_bundle_json"] = _rel(ctx, bundle_path)
    ctx.extra["caption_tiers"] = tier_payloads
    ctx.extra["caption_bundle"] = bundle
    return tier_payloads, bundle


def _ensure_affordance_files(ctx: PipelineContext, staged_dir: Path) -> None:
    stem = ctx.stem
    if ctx.caption_evidence is None:
        ctx.caption_evidence = {
            "schema": "citv_caption_evidence_v1",
            "status": "parity_empty_fallback",
            "summary": {"objects": len(ctx.extra.get("objects") or []), "caption_aware": False},
            "records": [],
        }
    if ctx.scene_affordances is None:
        ctx.scene_affordances = {
            "schema": "citv_scene_affordances_v1",
            "status": "parity_empty_fallback",
            "summary": {},
            "affordances": [],
        }
    if ctx.object_affordances is None:
        ctx.object_affordances = {
            "schema": "citv_object_affordances_v1",
            "status": "parity_empty_fallback",
            "object_count": len(ctx.extra.get("objects") or []),
            "objects": [],
        }
    if ctx.mask_affordances is None:
        ctx.mask_affordances = {
            "schema": "citv_mask_affordances_v1",
            "status": "parity_empty_fallback",
            "mask_count": len(ctx.extra.get("objects") or []),
            "supported_path_modes": [],
            "masks": [],
        }
    for key, payload in {
        "caption_evidence": ctx.caption_evidence,
        "scene_affordances": ctx.scene_affordances,
        "object_affordances": ctx.object_affordances,
        "mask_affordances": ctx.mask_affordances,
    }.items():
        path = staged_dir / f"{stem}_{key}.json"
        if not path.exists():
            _write_json(payload, path)
        ctx.path_exports[f"{key}_json"] = _rel(ctx, path)


def _caption_input_files(ctx: PipelineContext) -> Dict[str, str]:
    stem = ctx.stem
    return {
        "scene_json": f"scene_graph/staged/{stem}_scene.json",
        "relations_json": f"scene_graph/staged/{stem}_relations.json",
        "layers_json": f"scene_graph/staged/{stem}_layers.json",
        "mask_hierarchy_json": f"scene_graph/staged/{stem}_mask_hierarchy.json",
        "segmentation_image": f"scene_graph/staged/{stem}_segmentation.png",
        "tinted_overlay_image": f"scene_graph/staged/{stem}_overlay.png",
        "relations_map_image": f"scene_graph/staged/{stem}_relations_map.png",
        "regions_json": f"scene_graph/staged/{stem}_regions.json",
        "region_segmentation_image": f"scene_graph/staged/{stem}_region_segmentation.png",
    }


def _caption_objects_md(stem: str, rows: Sequence[Dict[str, Any]]) -> str:
    lines = [f"# Florence Object Captions - staged `{stem}`", ""]
    for row in rows:
        lines.append(f"## {row.get('object_id', '')}: {row.get('canonical_label') or row.get('label', '')}")
        cap = str(row.get("florence_caption") or row.get("caption") or "")
        if cap:
            lines.append(cap)
        labels = [
            str(row.get("gdino_label", "")),
            str(row.get("florence_label", "")),
            str(row.get("rampp_label", "")),
        ]
        lines.append(f"Labels: {', '.join([x for x in labels if x])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _fusion_prompt(ctx: PipelineContext, objects: Sequence[Dict[str, Any]], global_caption: str) -> str:
    labels = ", ".join(
        sorted({str(o.get("canonical_label") or o.get("label", "")) for o in objects if str(o.get("label", "")).strip()})[:24]
    )
    relations = [
        normalize_relation(r)
        for r in list(ctx.relations)[:24]
        if normalize_relation(r).get("subject_id") and normalize_relation(r).get("object_id")
    ]
    rel_text = "; ".join(
        f"{r.get('subject_id')} {r.get('predicate')} {r.get('object_id')}" for r in relations[:12]
    )
    return (
        "Create a faithful scene caption using the staged scene graph, dense captions, "
        "object labels, relations, regions, masks, and depth evidence.\n\n"
        f"Global caption: {global_caption}\n"
        f"Object labels: {labels}\n"
        f"Relations: {rel_text}\n"
    )


def _write_path_parity(
    ctx: PipelineContext,
    paths_root: Path,
    warnings: List[str],
    failures: List[Dict[str, Any]],
) -> None:
    stem = ctx.stem
    paths = _paths(paths_root)
    trajectories = _load_json(paths_root / "trajectory_hypotheses.json", {})
    actions = _load_json(paths_root / "action_hypotheses.json", {})

    _ensure_path_contract_jsons(ctx, paths_root, paths, trajectories, actions)
    trajectories = _load_json(paths_root / "trajectory_hypotheses.json", {})
    actions = _load_json(paths_root / "action_hypotheses.json", {})
    _write_scene_context(ctx, paths_root, paths, trajectories, actions)
    _write_semantic_layer(ctx, paths_root)
    _write_path_diagnostics(ctx, paths_root, paths)
    _write_path_descriptions(ctx, paths_root, paths, trajectories, actions)
    _write_pair_proposals(ctx, paths_root, paths)
    _write_insertion_ensembles(ctx, paths_root, paths, trajectories)
    _write_cost_and_map_aliases(ctx, paths_root, paths, warnings)
    _write_per_path_images(ctx, paths_root, paths, warnings)
    _write_stage_and_triplet_context(ctx, paths_root, warnings)
    _write_traj_v2_aliases(ctx, paths_root, paths, trajectories, warnings)
    _write_atlas_fallbacks(ctx, paths_root, warnings, failures)

    # Update path_exports for the monolith-compatible files written above.
    base = f"scene_graph/staged/{stem}_paths"
    for key, name in {
        "scene_context_json": "scene_context.json",
        "semantic_layer_json": "semantic_layer.json",
        "path_hypotheses_json": "path_hypotheses.json",
        "path_descriptions_json": "path_descriptions.json",
        "path_reasoning_md": "path_reasoning.md",
        "descriptions_md": "descriptions.md",
        "path_diagnostics_json": "path_diagnostics.json",
        "path_visual_index_json": "path_visual_index.json",
        "path_pair_proposals_json": "pair_proposals.json",
        "insertion_path_ensembles_json": "insertion_path_ensembles.json",
        "trajectory_hypotheses_json": "trajectory_hypotheses.json",
        "animation_components_json": "animation_components.json",
        "animation_plan_json": "animation_plan.json",
        "action_hypotheses_json": "action_hypotheses.json",
        "path_atlas_manifest_json": "path_atlas_manifest.json",
        "path_cost_map_png": "path_cost_map.png",
        "path_cost_map_npy": "path_cost_map.npy",
        "path_fields_explainer_image": "path_fields_explainer.png",
        "path_fields_legend_json": "path_fields_legend.json",
        "path_map_all_image": f"{stem}_path_map_all.png",
        "path_map_topN_image": f"{stem}_path_map_topN.png",
        "motion_contracts_overlay_image": "motion_contracts_overlay.png",
        "context_triplets_manifest_json": "context_triplets_manifest.json",
        "trajectory_v2_manifest_json": "traj_v2/traj_v2_manifest.json",
    }.items():
        if (paths_root / name).exists():
            ctx.path_exports[key] = f"{base}/{name}"


def _paths(paths_root: Path) -> List[Dict[str, Any]]:
    data = _load_json(paths_root / "path_hypotheses.json", {})
    if isinstance(data, dict):
        return [dict(p) for p in (data.get("paths") or data.get("hypotheses") or []) if isinstance(p, dict)]
    if isinstance(data, list):
        return [dict(p) for p in data if isinstance(p, dict)]
    return []


def _ensure_path_contract_jsons(
    ctx: PipelineContext,
    paths_root: Path,
    paths: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Any],
    actions: Dict[str, Any],
) -> None:
    path_payload = {
        "schema": "citv_path_hypotheses_v3",
        "version": "3.0",
        "status": "parity_empty_fallback" if not paths else "derived_from_staged",
        "stem": ctx.stem,
        "hypotheses": list(paths),
    }
    if not (paths_root / "path_hypotheses.json").exists():
        _write_json(path_payload, paths_root / "path_hypotheses.json")
    # path_hypotheses_full.json is removed in v3.
    legacy_full = paths_root / "path_hypotheses_full.json"
    if legacy_full.exists():
        try:
            legacy_full.unlink()
        except OSError:
            pass

    if not (paths_root / "trajectory_hypotheses.json").exists():
        if paths:
            try:
                from . import animation_export

                traj_payload = animation_export._fallback_trajectory_bundle_from_paths(
                    ctx,
                    list(paths),
                    {"width": int(ctx.width), "height": int(ctx.height)},
                    "staged",
                    "parity fallback from enriched paths",
                )
            except Exception:
                traj_payload = {
                    "schema": "citv_trajectory_bundle_v1",
                    "status": "parity_empty_fallback",
                    "image_stem": ctx.stem,
                    "hypotheses": list(trajectories.get("hypotheses") or []),
                }
        else:
            traj_payload = {
                "schema": "citv_trajectory_bundle_v1",
                "status": "parity_empty_fallback",
                "image_stem": ctx.stem,
                "hypotheses": list(trajectories.get("hypotheses") or []),
            }
        _write_json(traj_payload, paths_root / "trajectory_hypotheses.json")
    if not (paths_root / "animation_components.json").exists():
        traj_payload = _load_json(paths_root / "trajectory_hypotheses.json", {})
        if paths and list((traj_payload or {}).get("hypotheses") or []):
            try:
                from . import animation_export

                comp_payload = animation_export._fallback_animation_components_from_paths(
                    list(paths),
                    traj_payload,
                    ctx.stem,
                    {"width": int(ctx.width), "height": int(ctx.height)},
                    "staged",
                    "parity fallback from trajectory hypotheses",
                )
            except Exception:
                comp_payload = {
                    "schema": "citv_animation_components_bundle_v1",
                    "status": "parity_empty_fallback",
                    "image_stem": ctx.stem,
                    "components": [],
                }
        else:
            comp_payload = {
                "schema": "citv_animation_components_bundle_v1",
                "status": "parity_empty_fallback",
                "image_stem": ctx.stem,
                "components": [],
            }
        _write_json(
            comp_payload,
            paths_root / "animation_components.json",
        )
    if not (paths_root / "animation_plan.json").exists():
        if paths:
            try:
                from . import animation_export

                plan_payload = animation_export._build_animation_plan(list(paths), getattr(ctx, "config", None))
            except Exception:
                plan_payload = {
                    "schema": "citv_animation_plan_v1",
                    "status": "parity_fallback",
                    "image_stem": ctx.stem,
                    "paths": [
                        {
                            "path_id": str(p.get("path_id", "")),
                            "trajectory_id": str(p.get("path_id", "")),
                            "trajectory_points": list(p.get("polyline_2d") or []),
                            "segments": [{"motion": "traverse", "t0_s": 0.0, "t1_s": 1.0}],
                        }
                        for p in paths
                    ],
                }
        else:
            plan_payload = {
                "schema": "citv_animation_plan_v1",
                "status": "parity_empty_fallback",
                "image_stem": ctx.stem,
                "paths": [],
            }
        _write_json(
            plan_payload,
            paths_root / "animation_plan.json",
        )
    if not (paths_root / "action_hypotheses.json").exists():
        fallback_actions = list(
            actions.get("hypotheses") or actions.get("actions") or []
        )
        if not fallback_actions and paths:
            for idx, path in enumerate(paths):
                pid = str(path.get("path_id", f"path_{idx:04d}"))
                fallback_actions.append(
                    {
                        "action_id": f"action_{pid}_parity_locomotion",
                        "source_type": "path",
                        "action_family": str(path.get("action_family", "locomotion")),
                        "action_name": "traverse",
                        "manifold_type": str(path.get("manifold_type", "ribbon_path")),
                        "subject": dict(path.get("source_entity") or {}),
                        "target": dict(path.get("target_entity") or {}),
                        "grounding": {"path_id": pid, "pixel_grounded": bool(path.get("polyline_2d"))},
                        "scores": dict(path.get("scores") or {}),
                    }
                )
        _write_json(
            {
                "schema": "citv_action_hypotheses_v2",
                "version": "2.0",
                "status": "parity_empty_fallback",
                "stem": ctx.stem,
                "hypotheses": fallback_actions,
                "summary": {"action_count": len(fallback_actions)},
            },
            paths_root / "action_hypotheses.json",
        )


def _write_scene_context(
    ctx: PipelineContext,
    paths_root: Path,
    paths: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Any],
    actions: Dict[str, Any],
) -> None:
    payload = {
        "schema": "citv_scene_context_v1",
        "stem": ctx.stem,
        "image": str(ctx.image_path),
        "objects": len(ctx.extra.get("objects") or []),
        "relations": len(ctx.relations),
        "regions": len(ctx.region_partition_meta),
        "paths": len(paths),
        "trajectories": len(trajectories.get("hypotheses") or []),
        "actions": len(actions.get("actions") or []),
        "caption_evidence_summary": dict((ctx.caption_evidence or {}).get("summary") or {}),
        "scene_affordance_summary": dict((ctx.scene_affordances or {}).get("summary") or {}),
    }
    _write_json(payload, paths_root / "scene_context.json")


def _write_semantic_layer(ctx: PipelineContext, paths_root: Path) -> None:
    scene = ctx.scene_affordances or {}
    object_aff = ctx.object_affordances or {}
    regions = list(scene.get("regions") or [])
    payload = {
        "schema": "citv_semantic_layer_staged_v1",
        "semantic_enabled": True,
        "caption_aware": True,
        "entities": [
            {
                "id": o.get("object_id"),
                "label": o.get("canonical_label") or o.get("label"),
                "roles": o.get("roles", []),
                "actions": o.get("actions", []),
                "region_id": o.get("region_id", ""),
            }
            for o in object_aff.get("objects") or []
        ],
        "actors": [
            o.get("object_id")
            for o in object_aff.get("objects") or []
            if any(str(r.get("name")) == "actor" for r in o.get("roles", []))
        ],
        "region_affordances": [
            {
                "region_id": r.get("region_id"),
                "affordance": _top_name(r.get("roles") or r.get("actions") or []),
                "actions": r.get("actions", []),
                "caption": r.get("caption", ""),
            }
            for r in regions
        ],
        "actor_intents": list((ctx.action_hypotheses or {}).get("actions") or [])[:32],
    }
    _write_json(payload, paths_root / "semantic_layer.json")


def _top_name(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(rows, key=lambda r: float(r.get("score", 0.0) or 0.0))
    return str(best.get("name", ""))


def _write_path_diagnostics(ctx: PipelineContext, paths_root: Path, paths: Sequence[Dict[str, Any]]) -> None:
    family_counts: Dict[str, int] = {}
    rows = []
    for p in paths:
        src_t = str((p.get("source_entity") or {}).get("type", "unknown"))
        tgt_t = str((p.get("target_entity") or {}).get("type", "unknown"))
        family = f"{src_t}_to_{tgt_t}"
        family_counts[family] = family_counts.get(family, 0) + 1
        rows.append({
            "path_id": p.get("path_id", ""),
            "path_level": p.get("path_level", ""),
            "path_family": family,
            "manifold_type": p.get("manifold_type", ""),
            "scores": p.get("scores", {}),
            "semantic_trace_available": bool(p.get("semantic_trace")),
            "caption_trace_available": bool(p.get("caption_trace")),
            "visibility_profile_available": bool(p.get("visibility_profile")),
            "suppressed": bool(p.get("suppressed", False)),
            "suppressed_reason": str(p.get("suppressed_reason", "")),
        })

    # Include dedupe/dropped records produced by paths_export so reviewers can
    # see *why* a candidate disappeared (Track 1 noise removal, plan §1.2).
    dropped_rows: List[Dict[str, Any]] = []
    for d in (ctx.extra.get("path_dropped") or []):
        if not isinstance(d, dict):
            continue
        dropped_rows.append({
            "path_id": str(d.get("path_id", "")),
            "dropped_reason": str(d.get("dropped_reason", "")),
            "manifold_type": d.get("manifold_type", ""),
            "source_entity_id": str((d.get("source_entity") or {}).get("id", "")),
            "target_entity_id": str((d.get("target_entity") or {}).get("id", "")),
            "scores": d.get("scores") or {},
        })

    # Include trajectory rejection reasons recorded by animation_export so the
    # 198 -> N drop-out becomes visible without crashing (plan §2.8).
    trajectory_rejections: List[Dict[str, Any]] = []
    for tr in (ctx.extra.get("trajectory_rejections") or []):
        if isinstance(tr, dict):
            trajectory_rejections.append(dict(tr))

    _write_json(
        {
            "schema": "citv_path_diagnostics_staged_v1",
            "family_coverage_report": {"family_counts": family_counts},
            "paths": rows,
            "dropped_paths": dropped_rows,
            "trajectory_rejections": trajectory_rejections,
            "summary": {
                "kept_count": len(rows),
                "dropped_count": len(dropped_rows),
                "trajectory_rejection_count": len(trajectory_rejections),
            },
        },
        paths_root / "path_diagnostics.json",
    )


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _hint_names(rows: Any, key: str, *, limit: int = 8) -> List[str]:
    out: List[str] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            val = str(row.get(key, "")).strip()
        else:
            val = str(row).strip()
        if val and val not in out:
            out.append(val)
    return out[:limit]


def _entity_display(path: Dict[str, Any], key: str) -> str:
    ent = path.get(key) if isinstance(path.get(key), dict) else {}
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    sem_ent = sem.get("source" if key == "source_entity" else "target") if isinstance(sem, dict) else {}
    label = str(ent.get("label") or ent.get("canonical_name") or (sem_ent or {}).get("label") or ent.get("id") or "")
    eid = str(ent.get("id", ""))
    return f"{label} ({eid})" if label and eid and label != eid else (label or eid or key)


def _actions_by_path_id(action_hypotheses: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    rows = (
        (action_hypotheses or {}).get("hypotheses")
        or (action_hypotheses or {}).get("actions")
        or []
    )
    for action in list(rows):
        if not isinstance(action, dict):
            continue
        pid = str((action.get("grounding") or {}).get("path_id", ""))
        if pid and pid not in out:
            out[pid] = action
    return out


def _trajectories_by_path_id(trajectories: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for hyp in list((trajectories or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        pid = str((hyp.get("action_context") or {}).get("path_id") or hyp.get("continues_from_path_id") or "")
        if not pid:
            for sample in list(hyp.get("samples") or []):
                ev = sample.get("evidence") if isinstance(sample, dict) and isinstance(sample.get("evidence"), dict) else {}
                pid = str(ev.get("path_id", ""))
                if pid:
                    break
        if pid and pid not in out:
            out[pid] = hyp
    return out


def _path_update_field_status(path: Dict[str, Any]) -> Dict[str, bool]:
    fields = [
        "polyline_2d",
        "polyline_2d_reprojected",
        "polyline_3d",
        "depth_trace_m",
        "width_profile_px",
        "support_trace",
        "semantic_trace",
        "caption_trace",
        "visibility_profile",
        "render_layers",
        "region_boundary_trace",
        "motion_hints",
        "action_hints",
        "trajectory_contract",
    ]
    return {field: bool(path.get(field)) for field in fields}


def _compact_boundary(path: Dict[str, Any]) -> Dict[str, Any]:
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    if not boundary:
        return {}
    return {
        "movement_scope": str(boundary.get("movement_scope", "")),
        "boundary_interaction": str(boundary.get("boundary_interaction", "")),
        "boundary_sample_fraction": boundary.get("boundary_sample_fraction", None),
        "transition_count": boundary.get("transition_count", 0),
        "max_transition_depth_delta_m": boundary.get("max_transition_depth_delta_m", None),
        "regions_sequence": list(boundary.get("regions_sequence") or [])[:8],
        "transitions": list(boundary.get("transitions") or [])[:12],
        "motion_implications": list(boundary.get("motion_implications") or [])[:8],
    }


def _path_description_record(
    path: Dict[str, Any],
    idx: int,
    trajectory: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    pid = str(path.get("path_id", ""))
    scores = path.get("scores") if isinstance(path.get("scores"), dict) else {}
    contract = path.get("trajectory_contract") if isinstance(path.get("trajectory_contract"), dict) else {}
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    boundary = _compact_boundary(path)
    src = _entity_display(path, "source_entity")
    tgt = _entity_display(path, "target_entity")
    motion_labels = _hint_names(path.get("motion_hints"), "motion", limit=10)
    action_labels = (
        _hint_names(path.get("action_hints"), "action", limit=10)
        + _hint_names(path.get("action_hints"), "path_mode", limit=10)
    )[:12]
    movement_scope = str(path.get("movement_scope") or boundary.get("movement_scope") or contract.get("movement_scope") or "")
    boundary_interaction = str(path.get("boundary_interaction") or boundary.get("boundary_interaction") or contract.get("boundary_interaction") or "")
    dominant_motion = str(contract.get("dominant_motion") or (motion_labels[0] if motion_labels else ""))
    summary = (
        f"Path {idx} links {src} to {tgt} as a "
        f"{path.get('manifold_type', path.get('path_type', 'path'))}."
    )
    motion_prose = (
        f"Dominant motion is `{dominant_motion or 'unspecified'}`; "
        f"movement scope is `{movement_scope or 'unknown'}` with boundary context "
        f"`{boundary_interaction or 'unclassified'}`."
    )
    evidence_prose = (
        "The record is grounded by depth, region labels, support semantics, captions, "
        "visibility/occlusion, action affordances, and trajectory linkage when available."
    )
    return {
        "schema": "citv_path_description_record_v2",
        "path_id": pid,
        "path_num": idx,
        "labels": {
            "source": src,
            "target": tgt,
            "path_type": str(path.get("path_type", "")),
            "manifold_type": str(path.get("manifold_type", "")),
            "action_family": str(path.get("action_family", "")),
            "dominant_motion": dominant_motion,
            "movement_scope": movement_scope,
            "boundary_interaction": boundary_interaction,
        },
        "prose": {
            "summary": summary,
            "motion": motion_prose,
            "evidence": evidence_prose,
            "boundary": (
                "Region-boundary evidence comes from the same region seams drawn as yellow contours. "
                "It marks support, portal, and occlusion context rather than hard obstacles."
            ) if boundary else "",
        },
        "scores": scores,
        "motion": {
            "motion_labels": motion_labels,
            "kinematic_signatures": path.get("kinematic_signatures") or [],
            "trajectory_contract": contract,
        },
        "actions": {
            "action_labels": action_labels,
            "action_hints": path.get("action_hints") or [],
            "linked_action": {
                "action_id": str((action or {}).get("action_id", "")),
                "action_name": str((action or {}).get("action_name", "")),
                "action_family": str((action or {}).get("action_family", "")),
                "manifold_type": str((action or {}).get("manifold_type", "")),
            } if action else None,
        },
        "movement": {
            "scope": movement_scope,
            "inter_region": movement_scope == "inter_region",
            "intra_region": movement_scope == "intra_region",
            "region_boundary_trace": boundary,
            "support_kind_counts": dict((sem or {}).get("support_kind_counts") or {}),
            "regions_traversed": list((sem or {}).get("regions_traversed") or path.get("regions_traversed") or []),
        },
        "grounded_inputs": {
            "full_path_record_ref": "path_hypotheses.json",
            "path_update_contract_fields": _path_update_field_status(path),
            "polyline_summary": {
                "polyline_2d_vertices": len(path.get("polyline_2d") or []),
                "polyline_3d_vertices": len(path.get("polyline_3d") or []),
                "has_reprojected_2d": bool(path.get("polyline_2d_reprojected")),
            },
            "depth_trace_summary": next((r for r in list(path.get("depth_trace_m") or []) if isinstance(r, dict) and r.get("summary")), {}),
            "caption_trace": path.get("caption_trace") or {},
            "visibility_available": bool(path.get("visibility_profile")),
            "render_layers": path.get("render_layers") or [],
        },
        "trajectory": {
            "trajectory_id": str((trajectory or {}).get("trajectory_id", "")),
            "linked": bool(trajectory),
            "has_samples": bool(trajectory and (trajectory.get("samples") or trajectory.get("trajectory_points"))),
        },
    }


def _write_path_descriptions(
    ctx: PipelineContext,
    paths_root: Path,
    paths: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Any],
    actions: Optional[Dict[str, Any]] = None,
) -> None:
    descriptions: Dict[str, Any] = {}
    ranked = _ranked(paths)
    action_by_path = _actions_by_path_id(ctx.action_hypotheses or actions)
    traj_by_path = _trajectories_by_path_id(trajectories)
    for idx, p in enumerate(ranked, start=1):
        pid = str(p.get("path_id", ""))
        descriptions[pid] = _path_description_record(
            p,
            idx,
            traj_by_path.get(pid),
            action_by_path.get(pid),
        )
    _write_json(descriptions, paths_root / "path_descriptions.json")

    md = [f"# Path hypotheses: {ctx.stem}", ""]
    md.append(
        "Each section is generated from `path_hypotheses.json`, linked action hypotheses, "
        "trajectory hypotheses, caption evidence, depth, and region-boundary traces."
    )
    md.append("")
    for idx, p in enumerate(ranked, start=1):
        pid = str(p.get("path_id", ""))
        rec = descriptions.get(pid, {})
        labels = rec.get("labels") or {}
        movement = rec.get("movement") or {}
        boundary = movement.get("region_boundary_trace") or {}
        conf = _num((p.get("scores") or {}).get("overall_confidence"), 0.0)
        md.append(f"## Path {idx}: `{pid}`")
        md.append(f"- confidence: {conf:.3f}")
        md.append(f"- labels: `{labels.get('source', '')}` → `{labels.get('target', '')}`")
        md.append(
            f"- manifold/action: `{labels.get('manifold_type', '')}` / "
            f"`{labels.get('action_family', '')}`"
        )
        md.append(f"- motion: {rec.get('prose', {}).get('motion', '')}")
        if boundary:
            md.append(
                f"- boundary evidence: scope=`{boundary.get('movement_scope', '')}`, "
                f"interaction=`{boundary.get('boundary_interaction', '')}`, "
                f"transitions={boundary.get('transition_count', 0)}, "
                f"boundary_fraction={boundary.get('boundary_sample_fraction', None)}"
            )
        actions = (rec.get("actions") or {}).get("action_labels") or []
        if actions:
            md.append(f"- action labels: {actions}")
        support = movement.get("support_kind_counts") or {}
        if support:
            md.append(f"- support labels: {support}")
        traj = rec.get("trajectory") or {}
        md.append(f"- trajectory: `{traj.get('trajectory_id', '')}` linked={traj.get('linked', False)}")
        md.append(f"- prose: {rec.get('prose', {}).get('summary', '')}")
        md.append("")
    _write_text("\n".join(md) + "\n", paths_root / "path_reasoning.md")

    desc = [f"# Path and Trajectory Descriptions: {ctx.stem}", ""]
    desc.append(f"- path_count: {len(paths)}")
    desc.append(f"- trajectory_count: {len(trajectories.get('hypotheses') or [])}")
    desc.append("")
    for h in list(trajectories.get("hypotheses") or [])[:50]:
        desc.append(f"## `{h.get('trajectory_id', '')}`")
        desc.append(f"- confidence: {float(h.get('confidence', 0.0) or 0.0):.3f}")
        desc.append(f"- action_context: {h.get('action_context', {})}")
        desc.append("")
    _write_text("\n".join(desc) + "\n", paths_root / "descriptions.md")

    visual_index = {
        "schema": "citv_path_visual_index_staged_v1",
        "paths": {
            str(p.get("path_id", "")): {
                "path_id": p.get("path_id", ""),
                "path_num": idx,
                "level": p.get("path_level", ""),
                "per_path_image": f"images/{p.get('path_level', 'object')}/path_{p.get('path_id', '')}.png",
                "description_record": "path_descriptions.json",
            }
            for idx, p in enumerate(ranked, start=1)
        },
        "scene_context_record": "scene_context.json",
    }
    _write_json(visual_index, paths_root / "path_visual_index.json")


def _write_pair_proposals(ctx: PipelineContext, paths_root: Path, paths: Sequence[Dict[str, Any]]) -> None:
    seen = set()
    pairs = []
    for p in paths:
        src = str((p.get("source_entity") or {}).get("id", ""))
        tgt = str((p.get("target_entity") or {}).get("id", ""))
        if not src or not tgt or (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        pairs.append({
            "src_id": src,
            "tgt_id": tgt,
            "proposal_score": float((p.get("scores") or {}).get("overall_confidence", 0.0) or 0.0),
            "source": "staged_path_hypotheses",
        })
    if not pairs:
        rels = [normalize_relation(r) for r in ctx.relations]
        for r in rels[:100]:
            if r.get("subject_id") and r.get("object_id"):
                pairs.append({"src_id": r["subject_id"], "tgt_id": r["object_id"], "proposal_score": r["score"], "source": "relations"})
    _write_json({"schema": "citv_pair_proposals_staged_v1", "pairs": pairs}, paths_root / "pair_proposals.json")


def _write_insertion_ensembles(
    ctx: PipelineContext,
    paths_root: Path,
    paths: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Any],
) -> None:
    bundle = {
        "schema": "citv_insertion_path_ensembles_staged_v1",
        "stem": ctx.stem,
        "ensembles": [
            {
                "ensemble_id": "top_ranked_paths",
                "path_ids": [str(p.get("path_id", "")) for p in _ranked(paths)[:_MAX_READABLE_PATHS]],
                "trajectory_ids": [str(h.get("trajectory_id", "")) for h in list(trajectories.get("hypotheses") or [])[:_MAX_READABLE_PATHS]],
            }
        ],
    }
    _write_json(bundle, paths_root / "insertion_path_ensembles.json")


def _write_cost_and_map_aliases(ctx: PipelineContext, paths_root: Path, paths: Sequence[Dict[str, Any]], warnings: List[str]) -> None:
    speed_npy = paths_root / "path_traversability_speed.npy"
    cost_npy = paths_root / "path_cost_map.npy"
    cost_png = paths_root / "path_cost_map.png"
    try:
        if speed_npy.exists():
            speed = np.load(speed_npy)
            finite = np.isfinite(speed)
            if finite.any():
                smin = float(np.nanmin(speed[finite]))
                smax = float(np.nanmax(speed[finite]))
                denom = max(1e-6, smax - smin)
                cost = 1.0 - np.clip((speed - smin) / denom, 0.0, 1.0)
            else:
                cost = np.ones_like(speed, dtype=np.float32)
        else:
            cost = np.zeros((ctx.height, ctx.width), dtype=np.float32)
        np.save(cost_npy, cost.astype(np.float32))
        _write_gray_png(cost, cost_png)
    except Exception as exc:
        warnings.append(f"path_cost_map export failed: {type(exc).__name__}: {exc}")

    overlay = paths_root / "path_overlay.png"
    traj = paths_root / "trajectory_overlay.png"
    if overlay.exists():
        _copy_if_exists(overlay, paths_root / f"{ctx.stem}_path_map_all.png")
        _copy_if_exists(overlay, paths_root / f"{ctx.stem}_path_map_topN.png")
    if traj.exists():
        _copy_if_exists(traj, paths_root / "path_fields_explainer.png")
    elif overlay.exists():
        _copy_if_exists(overlay, paths_root / "path_fields_explainer.png")
    _write_json(
        {
            "schema": "citv_path_fields_legend_v1",
            "fields": [
                "path_cost_map",
                "path_traversability_speed",
                "navigation_zones",
                "region_boundary_trace",
                "movement_scope",
                "boundary_interaction",
            ],
            "provenance": (
                "Derived from staged traversability, path overlays, navigation zones, "
                "and region_label_map seams. Region boundaries are the same signal "
                "drawn as yellow contours in QA images."
            ),
        },
        paths_root / "path_fields_legend.json",
    )
    # Monolith plural compatibility alias.
    _copy_if_exists(paths_root / "motion_contract_overlay.png", paths_root / "motion_contracts_overlay.png")


def _write_gray_png(arr: np.ndarray, path: Path) -> None:
    try:
        import cv2
    except ImportError:
        return
    data = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(data)
    if finite.any():
        mn = float(np.nanmin(data[finite]))
        mx = float(np.nanmax(data[finite]))
        norm = np.clip((data - mn) / max(1e-6, mx - mn), 0.0, 1.0)
    else:
        norm = np.zeros_like(data, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (norm * 255).astype(np.uint8))


def _write_per_path_images(ctx: PipelineContext, paths_root: Path, paths: Sequence[Dict[str, Any]], warnings: List[str]) -> None:
    try:
        import cv2
    except ImportError:
        return
    if ctx.img_bgr is None:
        return
    h, w = ctx.img_bgr.shape[:2]
    ranked = _ranked(paths)[:_MAX_READABLE_PATHS]
    stage_dirs = {str(p.get("path_level") or "object") for p in ranked} | {"region", "object", "mask"}
    for level in stage_dirs:
        (paths_root / "images" / level).mkdir(parents=True, exist_ok=True)
    for idx, p in enumerate(ranked, start=1):
        pts = _pts(p.get("polyline_2d") or [], w, h)
        if len(pts) < 2:
            continue
        level = str(p.get("path_level") or "object")
        canvas = ctx.img_bgr.copy()
        color = (80 + (idx * 37) % 150, 220 - (idx * 23) % 120, 80 + (idx * 17) % 150)
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"P{idx}", pts[len(pts)//2], cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        out = paths_root / "images" / level / f"path_{p.get('path_id', idx)}.png"
        try:
            cv2.imwrite(str(out), canvas)
        except Exception as exc:
            warnings.append(f"per-path image failed for {p.get('path_id', idx)}: {exc}")


def _write_stage_and_triplet_context(ctx: PipelineContext, paths_root: Path, warnings: List[str]) -> None:
    stages = paths_root / "images" / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(["path_overlay.png", "trajectory_overlay.png", "action_manifold_overlay.png"]):
        if (paths_root / name).exists():
            _copy_if_exists(paths_root / name, stages / f"{idx:02d}_{name}")
    triplet_dir = paths_root / "images" / "context_triplets"
    triplet_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    if (paths_root / "path_context_top5.png").exists():
        out = triplet_dir / "context_top5.png"
        _copy_if_exists(paths_root / "path_context_top5.png", out)
        entries.append({"image": "images/context_triplets/context_top5.png", "source": "path_context_top5"})
    _write_json({"schema": "citv_context_triplets_manifest_v1", "entries": entries}, paths_root / "context_triplets_manifest.json")


def _write_traj_v2_aliases(
    ctx: PipelineContext,
    paths_root: Path,
    paths: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Any],
    warnings: List[str],
) -> None:
    traj_v2 = paths_root / "traj_v2"
    traj_v2.mkdir(parents=True, exist_ok=True)
    overlay = paths_root / "trajectory_overlay.png"
    path_overlay = paths_root / "path_overlay.png"
    copies = {
        "traj_v2_all_ink.png": overlay,
        "traj_v2_all_ink_context.png": overlay if overlay.exists() else path_overlay,
        "traj_v2_topK_ink_context.png": overlay if overlay.exists() else path_overlay,
        "traj_v2_geodesic_compare.png": path_overlay,
        "traj_v2_alternates_faint.png": path_overlay,
        "traj_v2_underlay_final.png": path_overlay,
        "traj_v2_underlay_final_context.png": overlay if overlay.exists() else path_overlay,
        "traj_v2_suppressed_residual.png": path_overlay,
    }
    for name, src in copies.items():
        if src.exists():
            _copy_if_exists(src, traj_v2 / name)
    _write_json(
        {
            "schema": "citv_traj_v2_manifest_staged_v1",
            "path_count": len(paths),
            "trajectory_count": len(trajectories.get("hypotheses") or []),
            "images": sorted(p.name for p in traj_v2.glob("*.png")),
        },
        traj_v2 / "traj_v2_manifest.json",
    )


def _write_atlas_fallbacks(
    ctx: PipelineContext,
    paths_root: Path,
    warnings: List[str],
    failures: List[Dict[str, Any]],
) -> None:
    overlay = paths_root / "path_overlay.png"
    if not (paths_root / "path_atlas_manifest.json").exists():
        path_ids = [
            str(p.get("path_id", ""))
            for p in _ranked(_paths(paths_root))[:_MAX_READABLE_PATHS]
            if str(p.get("path_id", "")).strip()
        ]
        _write_json(
            {
                "schema": "citv_path_atlas_manifest_v1",
                "status": "parity_fallback",
                "stem": ctx.stem,
                "panels": [
                    {
                        "panel_index": 0,
                        "path_ids": path_ids,
                        "image": "path_atlas_ranked_panel_00.png",
                    }
                ],
                "markdown": "path_top30_atlas.md",
            },
            paths_root / "path_atlas_manifest.json",
        )
    if not (paths_root / "path_top30_atlas.md").exists():
        _write_text(
            f"# Path trajectory atlas (staged fallback): `{ctx.stem}`\n\n"
            "Atlas markdown was generated by the parity stage from staged path outputs.\n",
            paths_root / "path_top30_atlas.md",
        )
    for name in [
        "path_atlas_ranked_panel_00.png",
        "path_atlas_ranked_panel_00_context.png",
        "path_atlas_ranked_panel_00_paths_trajectories.png",
    ]:
        if not (paths_root / name).exists() and overlay.exists():
            _copy_if_exists(overlay, paths_root / name)
    if not (paths_root / "traj_atlas_line_only.png").exists() and (paths_root / "trajectory_overlay.png").exists():
        _copy_if_exists(paths_root / "trajectory_overlay.png", paths_root / "traj_atlas_line_only.png")


def _write_qa_failure_records(
    ctx: PipelineContext,
    cfg: Any,
    paths_root: Path,
    failures: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    modes = [int(x) for x in (getattr(cfg, "path_animation_qa_modes", [24, 120]) if cfg else [24, 120]) or []]
    for fps in modes:
        subdir = f"animation_qa_{fps}"
        mode_dir = paths_root / subdir
        video = mode_dir / "panel_00_paths_trajectories.mp4"
        manifest = mode_dir / "animation_qa_manifest.json"
        scores = mode_dir / "animation_qa_scores.json"
        if video.exists():
            if not manifest.exists():
                _write_json(
                    {
                        "schema": "citv_animation_qa_manifest_v2",
                        "fps": fps,
                        "panel_videos": [{"panel_index": 0, "video": f"{subdir}/panel_00_paths_trajectories.mp4"}],
                        "generated_by": "parity_export_fallback",
                    },
                    manifest,
                )
            if not scores.exists():
                _write_json({"panels": [], "generated_by": "parity_export_fallback", "fps": fps}, scores)
            ctx.path_exports[f"animation_qa_video_{fps}"] = f"scene_graph/staged/{ctx.stem}_paths/{subdir}/panel_00_paths_trajectories.mp4"
            ctx.path_exports[f"animation_qa_manifest_{fps}_json"] = f"scene_graph/staged/{ctx.stem}_paths/{subdir}/animation_qa_manifest.json"
            ctx.path_exports[f"animation_qa_scores_{fps}_json"] = f"scene_graph/staged/{ctx.stem}_paths/{subdir}/animation_qa_scores.json"
            continue
        failure_path = mode_dir / f"animation_qa_{fps}_failure.json"
        rec = _failure_record(ctx, f"animation_qa_{fps}", failure_path, f"configured QA mode {fps} did not produce a video")
        _write_json(rec, failure_path)
        failures.append(rec)
        warnings.append(rec["reason"])
        ctx.path_exports[f"animation_qa_failure_{fps}_json"] = _rel(ctx, failure_path)


def _json_list_len(payload: Any, keys: Sequence[str]) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0
    for key in keys:
        val = payload.get(key)
        if isinstance(val, list):
            return len(val)
        if isinstance(val, dict):
            return len(val)
    return 0


def _load_json_for_quality(path: Path) -> Tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid_json: {type(exc).__name__}: {exc}"


def _artifact_quality_issue(
    ctx: PipelineContext,
    key: str,
    rel_path: str,
    required: bool,
    payload_cache: Dict[str, Any],
    cfg: Any = None,
) -> str:
    """Return a short issue string when a present artifact is not useful."""
    abs_path = ctx.output_dir / rel_path
    min_mp4 = int(getattr(cfg, "artifact_manifest_mp4_min_bytes", 256) or 0) if cfg else 256
    if abs_path.suffix.lower() == ".mp4" and min_mp4 > 0 and "animation_qa" in rel_path.replace("\\", "/"):
        try:
            sz = abs_path.stat().st_size
            if sz < min_mp4:
                return f"mp4_too_small_bytes_{sz}_lt_{min_mp4}"
        except OSError:
            return "unreadable_file"
        return ""

    if not required:
        return ""
    if abs_path.suffix.lower() not in {".json", ".md", ".png", ".mp4", ".npy"}:
        return ""
    if abs_path.suffix.lower() != ".json":
        try:
            if abs_path.stat().st_size <= 0:
                return "empty_file"
        except OSError:
            return "unreadable_file"
        return ""

    payload, err = _load_json_for_quality(abs_path)
    payload_cache[rel_path] = payload
    if err:
        return err
    if payload in ({}, [], None):
        return "empty_json_payload"
    if isinstance(payload, dict) and payload.get("status") == "building" and key not in {
        "artifact_manifest_json",
        "scene_action_bundle_json",
    }:
        return "placeholder_building_payload"

    paths_root = f"scene_graph/staged/{ctx.stem}_paths"
    paths_payload = payload_cache.get(f"{paths_root}/path_hypotheses.json")
    if paths_payload is None:
        pth = ctx.output_dir / paths_root / "path_hypotheses.json"
        if pth.exists():
            paths_payload, _ = _load_json_for_quality(pth)
            payload_cache[f"{paths_root}/path_hypotheses.json"] = paths_payload
    path_count = _json_list_len(paths_payload, ("paths", "hypotheses"))

    traj_payload = payload_cache.get(f"{paths_root}/trajectory_hypotheses.json")
    if traj_payload is None:
        tpath = ctx.output_dir / paths_root / "trajectory_hypotheses.json"
        if tpath.exists():
            traj_payload, _ = _load_json_for_quality(tpath)
            payload_cache[f"{paths_root}/trajectory_hypotheses.json"] = traj_payload
    trajectory_count = _json_list_len(traj_payload, ("hypotheses", "trajectories"))

    object_count = len(ctx.extra.get("objects") or [])

    if key == "scene_json" and object_count > 0 and _json_list_len(payload, ("objects",)) == 0:
        return "scene_json_has_no_objects"
    if key == "caption_objects_json" and object_count > 0 and _json_list_len(payload, ("objects",)) == 0:
        return "caption_objects_empty"
    if key == "florence_object_captions_json" and object_count > 0 and _json_list_len(payload, ("objects",)) == 0:
        return "florence_object_captions_empty"
    if key == "object_affordances_json" and object_count > 0 and _json_list_len(payload, ("objects",)) == 0:
        return "object_affordances_empty"
    if key == "mask_affordances_json" and object_count > 0 and _json_list_len(payload, ("masks",)) == 0:
        return "mask_affordances_empty"
    if key == "path_hypotheses_json" and path_count == 0 and object_count > 0:
        return "path_hypotheses_empty"
    if key == "trajectory_hypotheses_json" and path_count > 0 and _json_list_len(payload, ("hypotheses",)) == 0:
        return "trajectory_hypotheses_empty_for_paths"
    if key == "animation_components_json" and trajectory_count > 0 and _json_list_len(payload, ("components",)) == 0:
        return "animation_components_empty_for_trajectories"
    if key == "animation_plan_json" and path_count > 0 and _json_list_len(payload, ("paths",)) == 0:
        return "animation_plan_empty_for_paths"
    if key == "action_hypotheses_json" and (path_count > 0 or object_count > 0) and _json_list_len(payload, ("actions", "hypotheses")) == 0:
        return "action_hypotheses_empty"
    if key == "pair_proposals_json" and path_count > 0 and _json_list_len(payload, ("pairs",)) == 0:
        return "pair_proposals_empty_for_paths"
    if (
        key == "path_descriptions_json"
        and path_count > 0
        and _json_list_len(payload, ("paths", "descriptions")) == 0
        and not (isinstance(payload, dict) and len(payload) > 0)
    ):
        return "path_descriptions_empty_for_paths"
    if key == "path_diagnostics_json" and path_count > 0 and _json_list_len(payload, ("paths",)) == 0:
        return "path_diagnostics_empty_for_paths"
    return ""


def _build_artifact_manifest(
    ctx: PipelineContext,
    failures: Sequence[Dict[str, Any]],
    warnings: Sequence[str],
    cfg: Any = None,
) -> Dict[str, Any]:
    expected = _expected_artifacts(ctx)
    records = []
    missing = []
    failed = list(failures)
    payload_cache: Dict[str, Any] = {}
    for key, rel_path, category, required in expected:
        abs_path = ctx.output_dir / rel_path
        failure_match = any(f.get("artifact") == key for f in failed)
        quality_issue = ""
        if abs_path.exists():
            quality_issue = _artifact_quality_issue(ctx, key, rel_path, required, payload_cache, cfg)
        status = (
            "invalid"
            if quality_issue
            else ("present" if abs_path.exists() else ("failed" if failure_match else "missing"))
        )
        rec = {
            "key": key,
            "category": category,
            "path": rel_path,
            "status": status,
            "required": bool(required),
        }
        if quality_issue:
            rec["reason"] = quality_issue
        records.append(rec)
        if status in {"missing", "invalid"}:
            missing.append(rec)
    for key, val in sorted(ctx.path_exports.items()):
        if not isinstance(val, str) or not val:
            continue
        if any(r["path"] == val for r in records):
            continue
        records.append({
            "key": key,
            "category": "registered_export",
            "path": val,
            "status": "present" if (ctx.output_dir / val).exists() else "referenced",
            "required": False,
        })
    return {
        "schema": "citv_artifact_manifest_v1",
        "version": "1.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "records": records,
        "missing_artifacts": missing,
        "failed_artifacts": failed,
        "warnings": sorted(set(str(w) for w in warnings if str(w).strip())),
        "summary": {
            "expected_count": len(expected),
            "present_count": sum(1 for r in records if r["status"] == "present"),
            "missing_count": len(missing),
            "invalid_count": sum(1 for r in records if r["status"] == "invalid"),
            "failed_count": len(failed),
        },
    }


def _expected_artifacts(ctx: PipelineContext) -> List[Tuple[str, str, str, bool]]:
    stem = ctx.stem
    base = "scene_graph/staged"
    paths = f"{base}/{stem}_paths"
    items: List[Tuple[str, str, str, bool]] = []

    def add(key: str, rel_path: str, category: str, required: bool = True) -> None:
        items.append((key, rel_path, category, required))

    for name in [
        "scene", "relations", "layers", "mask_hierarchy", "mask_hierarchy_detailed",
        "mask_hierarchy_levels", "regions", "region_adjacency_graph", "region_relations",
        "scene_action_bundle", "artifact_manifest",
    ]:
        suffix = "scene_action_bundle" if name == "scene_action_bundle" else name
        add(f"{name}_json", f"{base}/{stem}_{suffix}.json", "core_scene", True)
    for name in [
        "segmentation", "overlay", "3d_viz", "layers", "mask_hierarchy",
        "regions_overlay", "region_segmentation", "region_sam2_style_segmentation",
        "region_tinted_overlay", "relations_map", "relations_map_objects",
        "relations_map_regions", "region_relations_map",
    ]:
        add(f"{name}_png", f"{base}/{stem}_{name}.png", "core_png", False)
    for name in [
        "caption_global", "caption_objects", "caption_regions", "caption_cross_region",
        "caption_uncertainty", "caption_evidence", "caption_bundle",
        "florence_object_captions", "florence_scene_caption", "fusion_scene_caption",
        "hybrid_scene_caption", "caption_comparison", "hybrid_caption_bundle",
        "caption_prompt_bundle",
    ]:
        add(f"{name}_json", f"{base}/{stem}_{name}.json", "caption", True)
    for name in ["florence_object_captions", "florence_scene_caption", "fusion_scene_caption", "hybrid_scene_caption"]:
        add(f"{name}_md", f"{base}/{stem}_{name}.md", "caption", True)
    add("track_comparison_prompt_md", f"{base}/{stem}_track_comparison_prompt.md", "caption", True)
    for name in ["scene_affordances", "object_affordances", "mask_affordances"]:
        add(f"{name}_json", f"{base}/{stem}_{name}.json", "affordance", True)
    for name in [
        "scene_context", "semantic_layer", "path_hypotheses",
        "path_descriptions", "path_diagnostics", "path_visual_index",
        "pair_proposals", "insertion_path_ensembles", "trajectory_hypotheses",
        "animation_components", "animation_plan", "path_fields_legend",
        "action_hypotheses", "path_atlas_manifest", "context_triplets_manifest",
    ]:
        add(f"{name}_json", f"{paths}/{name}.json", "path", True)
    add("path_hypotheses_candidates_json", f"{paths}/path_hypotheses_candidates.json", "path", False)
    for name in ["path_reasoning", "descriptions", "path_top30_atlas"]:
        add(f"{name}_md", f"{paths}/{name}.md", "path", True)
    for name in [
        "action_manifold_overlay", "motion_contracts_overlay", "path_cost_map",
        "path_fields_explainer", "path_traversability_speed", "navigation_zones",
        f"{stem}_path_map_all", f"{stem}_path_map_topN", "path_overlay",
        "trajectory_overlay", "path_context_top5",
    ]:
        add(f"{name}_png", f"{paths}/{name}.png", "path_png", False)
    for name in ["path_cost_map", "path_traversability_speed", "navigation_zones"]:
        add(f"{name}_npy", f"{paths}/{name}.npy", "path_array", False)
    for fps in [24, 120]:
        add(f"animation_qa_video_{fps}", f"{paths}/animation_qa_{fps}/panel_00_paths_trajectories.mp4", "animation_qa", False)
        add(f"animation_qa_manifest_{fps}", f"{paths}/animation_qa_{fps}/animation_qa_manifest.json", "animation_qa", False)
        add(f"animation_qa_scores_{fps}", f"{paths}/animation_qa_{fps}/animation_qa_scores.json", "animation_qa", False)
    for name in [
        "traj_v2_manifest", "traj_v2_all_ink", "traj_v2_all_ink_context",
        "traj_v2_topK_ink_context", "traj_v2_geodesic_compare", "traj_v2_alternates_faint",
        "traj_v2_underlay_final", "traj_v2_underlay_final_context", "traj_v2_suppressed_residual",
    ]:
        ext = "json" if name == "traj_v2_manifest" else "png"
        add(name, f"{paths}/traj_v2/{name}.{ext}", "trajectory_v2", False)
    return items


def _build_scene_action_bundle(ctx: PipelineContext, manifest: Dict[str, Any]) -> Dict[str, Any]:
    stem = ctx.stem
    base = f"scene_graph/staged/{stem}"
    paths = f"{base}_paths"
    caption_summary = dict((ctx.caption_evidence or {}).get("summary") or {})
    return {
        "schema": "citv_scene_action_bundle_v1",
        "version": "1.0",
        "stem": stem,
        "timestamp": ctx.timestamp,
        "image": str(ctx.image_path),
        "scene_json": f"scene_graph/staged/{stem}_scene.json",
        "artifact_manifest": f"scene_graph/staged/{stem}_artifact_manifest.json",
        "captions": {
            "bundle": f"{base}_caption_bundle.json",
            "evidence": f"{base}_caption_evidence.json",
            "global": f"{base}_caption_global.json",
            "objects": f"{base}_caption_objects.json",
            "regions": f"{base}_caption_regions.json",
            "cross_region": f"{base}_caption_cross_region.json",
            "uncertainty": f"{base}_caption_uncertainty.json",
            "legacy_compat_bundle": f"{base}_hybrid_caption_bundle.json",
            "summary": caption_summary,
        },
        "caption_evidence": {"summary": caption_summary, "path": f"{base}_caption_evidence.json"},
        "labels": {
            "object_count": len(ctx.extra.get("objects") or []),
            "candidate_fusion": True,
            "open_vocab_config": "scene_understanding/resources/path_action_ontology.json",
        },
        "relations": {
            "path": f"{base}_relations.json",
            "count": len(ctx.relations),
            "normalized_count": len([r for r in ctx.relations if normalize_relation(r).get("subject_id")]),
            "cross_region": f"{base}_caption_cross_region.json",
        },
        "regions": {"path": f"{base}_regions.json", "count": len(ctx.region_partition_meta)},
        "masks": {
            "hierarchy": f"{base}_mask_hierarchy.json",
            "affordances": f"{base}_mask_affordances.json",
        },
        "affordances": {
            "scene": f"{base}_scene_affordances.json",
            "object": f"{base}_object_affordances.json",
            "mask": f"{base}_mask_affordances.json",
            "scene_summary": dict((ctx.scene_affordances or {}).get("summary") or {}),
        },
        "paths": {
            "hypotheses": f"{paths}/path_hypotheses.json",
            "diagnostics": f"{paths}/path_diagnostics.json",
            "visual_index": f"{paths}/path_visual_index.json",
            "overlay": f"{paths}/path_overlay.png",
        },
        "actions": {
            "hypotheses": f"{paths}/action_hypotheses.json",
            "overlay": f"{paths}/action_manifold_overlay.png",
            "summary": dict((ctx.action_hypotheses or {}).get("summary") or {}),
        },
        "trajectories": {
            "hypotheses": f"{paths}/trajectory_hypotheses.json",
            "overlay": f"{paths}/trajectory_overlay.png",
        },
        "animation": {
            "components": f"{paths}/animation_components.json",
            "plan": f"{paths}/animation_plan.json",
            "motion_contracts_overlay": f"{paths}/motion_contracts_overlay.png",
        },
        "qa": {
            "animation_qa_24": f"{paths}/animation_qa_24/panel_00_paths_trajectories.mp4",
            "animation_qa_120": f"{paths}/animation_qa_120/panel_00_paths_trajectories.mp4",
            "atlas_manifest": f"{paths}/path_atlas_manifest.json",
            "atlas_markdown": f"{paths}/path_top30_atlas.md",
        },
        "timing": dict(ctx.extra.get("timing") or {}),
        "warnings": list(manifest.get("warnings") or []),
        "missing_artifacts": list(manifest.get("missing_artifacts") or []),
        "failed_artifacts": list(manifest.get("failed_artifacts") or []),
    }


def _ranked(paths: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [dict(p) for p in paths],
        key=lambda p: float((p.get("scores") or {}).get("overall_confidence", 0.0) or 0.0),
        reverse=True,
    )


def _pts(raw: Sequence[Any], width: int, height: int) -> List[Tuple[int, int]]:
    pts = []
    for xy in raw:
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        try:
            x = max(0, min(width - 1, int(round(float(xy[0])))))
            y = max(0, min(height - 1, int(round(float(xy[1])))))
            pts.append((x, y))
        except Exception:
            continue
    return pts


def _failure_record(ctx: PipelineContext, artifact: str, path: Path, reason: str) -> Dict[str, Any]:
    return {
        "schema": "citv_artifact_failure_v1",
        "artifact": artifact,
        "path": _rel(ctx, path),
        "reason": reason,
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
    }
