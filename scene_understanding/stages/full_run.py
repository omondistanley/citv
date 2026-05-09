"""Full staged-pipeline runner: one image → scene JSON + path artifacts.

This module is the single call site for running the entire package-native
pipeline for a batch of images.  It owns the per-image orchestration loop
and is the staged counterpart to the monolithic ``process_image`` method.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..pipeline_context import PipelineContext
from . import (
    action_export as stage_actions,
    affordances_export as stage_affordances,
    animation_export as stage_animation,
    captions_export as stage_captions,
    depth as stage_depth,
    labelling as stage_labelling,
    parity_export as stage_parity,
    paths_export as stage_paths,
    preprocess as stage_preprocess,
    regions as stage_regions,
    relations_stage as stage_relations,
    scene_write,
    segmentation as stage_segmentation,
    visualization_export as stage_viz,
)


def _record_optional_failure(ctx: PipelineContext, stage_name: str, exc: Exception) -> None:
    """Record a late-stage failure without preventing final parity exports."""
    msg = f"{type(exc).__name__}: {exc}"
    failure = {
        "schema": "citv_artifact_failure_v1",
        "artifact": stage_name,
        "path": "",
        "reason": msg,
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
    }
    ctx.extra.setdefault("artifact_failures", []).append(failure)
    ctx.extra.setdefault("artifact_warnings", []).append(f"{stage_name} stage failed: {msg}")


def _run_timed_stage(
    stage_name: str,
    fn: Any,
    pipeline: Any,
    ctx: PipelineContext,
    *,
    optional: bool = False,
) -> PipelineContext:
    started = time.perf_counter()
    try:
        out = fn(pipeline, ctx)
    except Exception as exc:  # noqa: BLE001
        if not optional:
            raise
        _record_optional_failure(ctx, stage_name, exc)
        print(f"  [StagedPipeline] optional stage {stage_name} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        out = ctx
    elapsed = time.perf_counter() - started
    out.extra.setdefault("timing", {})[stage_name] = round(elapsed, 4)
    return out


def run_single_image(
    pipeline: Any,
    image_path: str,
    output_dir: str,
    *,
    write_json: bool = True,
) -> Dict[str, Any]:
    """Run the full staged pipeline for one image.

    Returns the scene payload dict (same structure as the monolithic output).
    """
    t0 = time.perf_counter()

    ctx: PipelineContext = stage_preprocess.run(pipeline, image_path, output_dir)
    ctx.extra.setdefault("timing", {})["preprocess"] = round(time.perf_counter() - t0, 4)
    ctx = _run_timed_stage("depth", stage_depth.run, pipeline, ctx)
    ctx = _run_timed_stage("regions", stage_regions.run, pipeline, ctx)
    ctx = _run_timed_stage("segmentation", stage_segmentation.run, pipeline, ctx)
    ctx = _run_timed_stage("labelling", stage_labelling.run, pipeline, ctx)
    ctx = _run_timed_stage("relations", stage_relations.run, pipeline, ctx)

    # The stages below are additive exports. Failures should be visible in the
    # manifest and final bundle, but should not prevent the canonical scene JSON
    # from being written.
    for stage_name, stage_fn in [
        ("visualization", stage_viz.run),
        ("captions", stage_captions.run),
        ("affordances", stage_affordances.run),
        ("paths", stage_paths.run),
        ("actions", stage_actions.run),
        ("animation", stage_animation.run),
    ]:
        ctx = _run_timed_stage(stage_name, stage_fn, pipeline, ctx, optional=True)

    try:
        ctx = _run_timed_stage("parity_export", stage_parity.run, pipeline, ctx, optional=False)
    except Exception as exc:  # noqa: BLE001
        _record_optional_failure(ctx, "parity_export", exc)
        print(f"  [StagedPipeline] parity export failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if write_json:
            out_path = scene_write.write_staged_scene_json(ctx)
            print(f"  [StagedPipeline] fallback scene JSON: {out_path.name}")

    payload = scene_write.build_staged_scene_payload(ctx)
    if write_json:
        out_path = ctx.output_dir / "scene_graph" / "staged" / f"{ctx.stem}_scene.json"
        if not out_path.exists():
            out_path = scene_write.write_staged_scene_json(ctx)
        print(
            f"  [StagedPipeline] scene JSON: {out_path.name} "
            f"({time.perf_counter() - t0:.2f}s)"
        )
    return payload


def run_batch(
    pipeline: Any,
    image_paths: List[str],
    output_dir: str,
    *,
    stop_on_error: bool = False,
) -> List[Dict[str, Any]]:
    """Run the staged pipeline over a list of image paths.

    Returns a list of ``{"image": path, "result": payload | None, "error": str}``
    records in the same order as ``image_paths``.
    """
    results: List[Dict[str, Any]] = []
    for img in image_paths:
        record: Dict[str, Any] = {"image": img, "result": None, "error": ""}
        try:
            payload = run_single_image(pipeline, img, output_dir)
            record["result"] = payload
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            record["error"] = msg
            print(f"  [StagedPipeline] ERROR on {img}: {msg}")
            if stop_on_error:
                results.append(record)
                raise
        results.append(record)
    return results


def run_directory(
    pipeline: Any,
    input_dir: str,
    output_dir: str,
    *,
    extensions: Optional[List[str]] = None,
    stop_on_error: bool = False,
) -> List[Dict[str, Any]]:
    """Discover images in ``input_dir`` and run the staged pipeline on each.

    ``extensions`` defaults to ``[".jpg", ".jpeg", ".png", ".webp", ".bmp"]``.
    """
    exts = {e.lower() for e in (extensions or [".jpg", ".jpeg", ".png", ".webp", ".bmp"])}
    idir = Path(input_dir)
    image_paths = sorted(
        str(p) for p in idir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not image_paths:
        print(f"  [StagedPipeline] no images found in {idir}")
        return []
    print(f"  [StagedPipeline] found {len(image_paths)} image(s) in {idir}")
    return run_batch(pipeline, image_paths, output_dir, stop_on_error=stop_on_error)


def write_batch_manifest(
    results: List[Dict[str, Any]],
    output_dir: str,
    stem: str = "batch_manifest",
) -> Path:
    """Write a ``batch_manifest.json`` summarising a run_directory/run_batch call."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total": len(results),
        "ok": sum(1 for r in results if not r.get("error")),
        "errors": sum(1 for r in results if r.get("error")),
        "records": [
            {
                "image": r["image"],
                "error": r.get("error", ""),
                "objects": len((r.get("result") or {}).get("objects", [])),
                "relations": len((r.get("result") or {}).get("relations", [])),
            }
            for r in results
        ],
    }
    p = out / f"{stem}.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p
