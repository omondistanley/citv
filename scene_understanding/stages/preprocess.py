"""Preprocess stage: load image, optional undistort, resize, intrinsics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..pipeline_context import PipelineContext
from ..pipeline_stages import ensure_depth_dir, ensure_scene_graph_dir
from ..preprocessing import bgr_to_rgb, load_bgr_image, resize_image_if_needed


def run(pipeline: Any, image_path: str, output_dir: str) -> PipelineContext:
    """Load image, apply undistortion/resizing, and resolve intrinsics."""
    image = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = ensure_depth_dir(out_dir)
    ensure_scene_graph_dir(out_dir)
    img_bgr = load_bgr_image(image)
    if getattr(pipeline, "apply_undistortion", False):
        img_bgr = pipeline._undistort_image(img_bgr)
    img_rgb = bgr_to_rgb(img_bgr)

    cfg = getattr(pipeline, "config", None)
    max_side = int(getattr(cfg, "sam2_amg_max_image_side", 1280)) if cfg else 1280
    img_bgr, img_rgb, _scale, (width, height) = resize_image_if_needed(
        img_bgr=img_bgr,
        img_rgb=img_rgb,
        max_side=max_side,
    )
    fixed_intrinsics = getattr(pipeline, "fixed_intrinsics", None)
    if fixed_intrinsics:
        intrinsics = fixed_intrinsics
    else:
        try:
            intrinsics = pipeline._estimate_intrinsics(width, height)
        except AttributeError:
            fx = float(width) if width > 0 else 1.0
            fy = float(height) if height > 0 else 1.0
            intrinsics = {
                "fx": fx,
                "fy": fy,
                "cx": float(width) / 2.0,
                "cy": float(height) / 2.0,
            }

    return PipelineContext(
        image_path=image,
        output_dir=out_dir,
        stem=image.stem,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        img_bgr=img_bgr,
        img_rgb=img_rgb,
        height=height,
        width=width,
        intrinsics=intrinsics,
        extra={"depth_dir": depth_dir},
    )
