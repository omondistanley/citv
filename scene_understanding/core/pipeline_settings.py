"""Consolidated pipeline flags (read from an initialized SceneUnderstandingPipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PipelineSettings:
    """Snapshot of configuration used by orchestration and future staged helpers."""

    depth_mask_modes: List[str]
    require_any_relation_source: bool
    mask_iou_match_thresh: float
    pix2sg_mask_overlap_thresh: float
    pix2sg_depth_near_threshold: float
    depth_scale_factor: float
    depth_sigma_clip_scope: str
    regions_enabled: bool
    regions_k: int
    regions_min_region_px: int
    regions_seed: int
    export_hybrid_captions: bool
    export_caption_prompt_bundle: bool
    caption_max_objects_per_track: int
    florence2_label_enabled: bool
    rampp_enabled: bool


def attach_settings_from_pipeline(p: Any) -> None:
    """Populate ``p.pipeline_settings`` from a live pipeline instance (legacy attributes)."""
    p.pipeline_settings = PipelineSettings(
        depth_mask_modes=list(getattr(p, "depth_mask_modes", ["A"])),
        require_any_relation_source=bool(getattr(p, "require_any_relation_source", True)),
        mask_iou_match_thresh=float(getattr(p, "mask_iou_match_thresh", 0.1)),
        pix2sg_mask_overlap_thresh=float(getattr(p, "pix2sg_mask_overlap_thresh", 0.05)),
        pix2sg_depth_near_threshold=float(getattr(p, "pix2sg_depth_near_threshold", 1.0)),
        depth_scale_factor=float(getattr(p, "depth_scale_factor", 10.0)),
        depth_sigma_clip_scope=str(getattr(p, "depth_sigma_clip_scope", "mask")),
        regions_enabled=bool(getattr(p, "regions_enabled", False)),
        regions_k=int(getattr(p, "_regions_k", 4)),
        regions_min_region_px=int(getattr(p, "_regions_min_region_px", 500)),
        regions_seed=int(getattr(p, "_regions_seed", 42)),
        export_hybrid_captions=bool(getattr(p, "export_hybrid_captions", True)),
        export_caption_prompt_bundle=bool(getattr(p, "export_caption_prompt_bundle", True)),
        caption_max_objects_per_track=int(getattr(p, "caption_max_objects_per_track", 64)),
        florence2_label_enabled=bool(getattr(p, "_florence2_label_enabled", True)),
        rampp_enabled=bool(getattr(p, "_rampp_enabled", True)),
    )
