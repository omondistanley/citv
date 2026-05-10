"""Delegate full-image processing to the legacy monolith (parity / migration bridge)."""

from __future__ import annotations

from typing import Any


def run_legacy_process_image(pipeline: Any, image_path: str, output_dir: str) -> Any:
    """
    Call ``LegacySceneUnderstandingPipeline.process_image`` on *pipeline*.

    Use when ``scene_pipeline_mode == "legacy"`` or for golden comparisons.
    """
    from scene_understanding.pipeline import LegacySceneUnderstandingPipeline

    return LegacySceneUnderstandingPipeline.process_image(pipeline, image_path, output_dir)
