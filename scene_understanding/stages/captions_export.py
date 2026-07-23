"""Caption bundle export: florence_object / florence_scene / fusion_scene /
hybrid caption variants + a comparison manifest, mirroring the monolith's
``_save_caption_variants_for_track``.

Adapted to the staged pipeline's flat ``scene_graph/staged/`` layout (the
monolith uses a per-track subdirectory since it supports multiple crops per
image; the staged chain processes one image as a single implicit track) and
to the artifacts the staged chain actually produces by this point
(``ctx.extra["objects"]``, ``ctx.relations``, and
``{stem}_depth_mask_A.json`` from ``depth_mask_fusion.py``) rather than the
full monolith artifact set (mask hierarchy/layers/region overlays are
separate, optional staged stages -- referenced only when present).

``florence_scene_caption`` makes a real local Florence-2 call (via the
package's ``LabelingPipeline.florence2`` wrapper) when one is available,
exactly like the monolith; everything else stays a scored/gradable
placeholder (``pending_external_generation``) since fusion/hybrid captioning
is meant to be filled in by an external reviewer or a later LLM pass, not
invented here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..pipeline_context import PipelineContext
from ..timing import sub_timer


def _md_heading(text: str) -> str:
    return f"# {text}\n\n"


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def _collect_object_caption_rows(objects: List[Dict[str, Any]], max_objects: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in objects[: max(1, int(max_objects))]:
        rows.append({
            "object_id": str(obj.get("id")),
            "label": str(obj.get("label", "object")),
            "segmentor": str(obj.get("segmentor", "unknown")),
            "florence_label": "",
            "florence_caption": "",
            "selected_caption": str(obj.get("caption", obj.get("label", ""))),
        })
    return rows


def _fusion_prompt_text(image_path: str, stem: str, input_files: Dict[str, str]) -> str:
    file_lines = "\n".join(f"- {name}: {rel}" for name, rel in input_files.items() if rel)
    return f"""You are a scene-grounded captioner and verifier.

Use ALL provided artifacts jointly:
- Original image: {image_path}
- Image stem: {stem}
{file_lines}

Goals:
1) Write a detailed caption (10-16 sentences) grounded in BOTH image evidence and scene-graph evidence.
2) Include object attributes, spatial relations, depth/layer cues, and part-whole hierarchy when supported.
3) Avoid hallucinations: do not mention entities absent from both image and graph.
4) If evidence conflicts, explicitly flag uncertainty and alternatives.
"""


def _generate_local_florence_scene_caption(pipeline: Any, ctx: PipelineContext) -> tuple[str, str]:
    try:
        from PIL import Image as PILImage

        # Reuse the legacy monolith's already-loaded pipeline.florence2 (set
        # up by _load_labellers()/_ensure_florence_for_labelling() earlier in
        # the same stage chain -- though with Florence-2 now conditional on
        # GDINO returning a generic label per mask, it may genuinely never
        # have loaded this run). Previously fell back to
        # pipeline._get_labeling_pipeline() here, which builds an entirely
        # separate LabelingPipeline that reloads BOTH Florence-2 and RAM++ --
        # a real, measurable redundant RAM++ reload, since pipeline.rampp is
        # already loaded and active by this point in the chain. Construct
        # just a Florence2Wrapper directly instead (same lazy-load pattern
        # the legacy monolith itself uses), and cache it back onto the
        # pipeline so a later call in the same run reuses it.
        florence2 = getattr(pipeline, "florence2", None)
        if florence2 is None or not getattr(florence2, "active", False):
            from scene_understanding.labeling import Florence2Wrapper

            florence2 = Florence2Wrapper(model_id=getattr(pipeline, "_florence2_model_id", "microsoft/Florence-2-large"), device=pipeline.device)
            if florence2.active:
                pipeline.florence2 = florence2
        if florence2 is None or not getattr(florence2, "active", False):
            return "", "pending_external_generation"
        pil_full = PILImage.fromarray(ctx.img_rgb)
        cap_res = florence2._run_task("<MORE_DETAILED_CAPTION>", pil_full)
        caption = str(cap_res.get("<MORE_DETAILED_CAPTION>", "")).strip()
        if caption:
            return caption, "generated_local_florence"
        cap_res2 = florence2._run_task("<CAPTION>", pil_full)
        caption = str(cap_res2.get("<CAPTION>", "")).strip()
        return (caption, "generated_local_florence_fallback_caption") if caption else ("", "pending_external_generation")
    except Exception as exc:  # pragma: no cover - best-effort local captioning
        print(f"  [CaptionsExport] Florence2 full-image scene caption failed: {exc}")
        return "", "pending_external_generation"


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    objects: List[Dict[str, Any]] = ctx.extra.get("objects", [])
    if not objects:
        return ctx

    cfg = getattr(pipeline, "config", None)
    max_objects = int(getattr(cfg, "caption_max_objects_per_track", 40)) if cfg else 40
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    stem = ctx.stem
    image_path = str(ctx.image_path)

    input_files = {
        "scene_json": f"scene_graph/staged/{stem}_scene.json",
        "depth_mask_A_json": ctx.path_exports.get("depth_mask_a_json", ""),
        "path_hypotheses_json": ctx.path_exports.get("path_hypotheses_json", ""),
        "num_relations": str(len(ctx.relations)),
    }

    # florence_object
    rows = _collect_object_caption_rows(objects, max_objects)
    florence_obj_json = {"variant": "florence_object", "image_path": image_path, "count": len(rows), "objects": rows}
    _write_json(florence_obj_json, staged_dir / f"{stem}_florence_object_captions.json")
    md_lines = [_md_heading(f"Florence Object Captions - {stem}")]
    for row in rows:
        md_lines.append(
            f"- `{row['object_id']}` | label=`{row['label']}` | florence_caption=`{row['florence_caption']}` | selected_caption=`{row['selected_caption']}`"
        )
    _write_text("\n".join(md_lines) + "\n", staged_dir / f"{stem}_florence_object_captions.md")

    # florence_scene (real local Florence-2 call, matching monolith behavior)
    with sub_timer("captions_export.florence_scene_caption"):
        caption, status = _generate_local_florence_scene_caption(pipeline, ctx)
    florence_scene_json = {
        "variant": "florence_only", "image_path": image_path,
        "generated_caption": caption, "status": status, "input_files": input_files,
    }
    _write_json(florence_scene_json, staged_dir / f"{stem}_florence_scene_caption.json")
    _write_text(
        _md_heading(f"Florence Scene Caption - {stem}") + f"Status: {status}\n\n"
        + (caption + "\n" if caption else "No local Florence caption generated. Use `*_florence_object_captions.json` as source summary.\n"),
        staged_dir / f"{stem}_florence_scene_caption.md",
    )

    # fusion_scene (prompt only -- generation deferred to an external/LLM pass)
    fusion_prompt = _fusion_prompt_text(image_path, stem, input_files)
    fusion_scene_json = {
        "variant": "fusion_only", "image_path": image_path, "prompt": fusion_prompt,
        "generated_caption": "", "status": "pending_external_generation", "input_files": input_files,
    }
    _write_json(fusion_scene_json, staged_dir / f"{stem}_fusion_scene_caption.json")
    _write_text(_md_heading(f"Fusion Scene Caption - {stem}") + fusion_prompt + "\n", staged_dir / f"{stem}_fusion_scene_caption.md")

    # hybrid (references the other two -- combining them is an external/LLM step)
    hybrid_scene_json = {
        "variant": "hybrid", "image_path": image_path, "status": "pending_external_generation",
        "inputs": {
            "florence_object_captions": f"scene_graph/staged/{stem}_florence_object_captions.json",
            "florence_scene_caption": f"scene_graph/staged/{stem}_florence_scene_caption.json",
            "fusion_scene_caption": f"scene_graph/staged/{stem}_fusion_scene_caption.json",
        },
        "generated_caption": "",
    }
    _write_json(hybrid_scene_json, staged_dir / f"{stem}_hybrid_scene_caption.json")
    _write_text(
        _md_heading(f"Hybrid Scene Caption - {stem}") + "Status: pending external generation.\n"
        + "- Combine Florence-only and fusion-only outputs for final comparison-ready caption.\n",
        staged_dir / f"{stem}_hybrid_scene_caption.md",
    )

    comparison_json = {
        "variants": [
            {"name": "florence_only", "file": f"scene_graph/staged/{stem}_florence_scene_caption.json"},
            {"name": "fusion_only", "file": f"scene_graph/staged/{stem}_fusion_scene_caption.json"},
            {"name": "hybrid", "file": f"scene_graph/staged/{stem}_hybrid_scene_caption.json"},
        ],
        "scoring_template": {
            "faithfulness_to_image": None, "scene_graph_consistency": None,
            "relation_quality": None, "detail_richness": None,
        },
    }
    _write_json(comparison_json, staged_dir / f"{stem}_caption_comparison.json")

    bundle = {
        "image_path": image_path,
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
    bundle_path = staged_dir / f"{stem}_hybrid_caption_bundle.json"
    _write_json(bundle, bundle_path)
    ctx.path_exports["caption_bundle_json"] = str(bundle_path)
    ctx.path_exports["florence_scene_caption_status"] = status
    print(f"  [CaptionsExport] wrote caption bundle ({status}) -> {bundle_path.name}")
    return ctx
