"""Main package pipeline entry point.

This wrapper keeps the package API stable while reusing the proven script-based
SceneUnderstandingPipeline implementation from the repository root.

The ``_load_legacy_module`` importlib hook remains until staged output reaches
full parity with the monolith; see staged tests and ``docs/path_context_top5_reviewer.md``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import numpy as np
import torch

from .depth.coordinator import DepthCoordinator
from .geometry.depth_geometry import DepthGeometry
from .labeling import LabelingPipeline
from .pipeline_context import PipelineContext
from .relations import RelationsPipeline
from .segmentation import SegmentationPipeline
from .stages import depth as stage_depth
from .stages import full_run as stage_full_run
from .stages import labelling as stage_labelling
from .stages import preprocess as stage_preprocess
from .stages import regions as stage_regions
from .stages import relations_stage as stage_relations
from .stages import scene_write
from .stages import segmentation as stage_segmentation


def _load_legacy_module() -> ModuleType:
    """Load root ``scene_understanding.py`` as a module (kept until staged parity replaces it)."""
    module_path = Path(__file__).resolve().parent.parent / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_scene_understanding_legacy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
LegacySceneUnderstandingPipeline = _legacy.SceneUnderstandingPipeline


def resolve_scene_pipeline_mode(config: Optional[Any] = None) -> str:
    """Return ``'legacy'`` or ``'staged'`` (env ``CITV_SCENE_PIPELINE_MODE`` overrides *config*)."""
    cfg_mode = (
        str(getattr(config, "scene_pipeline_mode", "")).strip().lower()
        if config is not None
        else ""
    )
    env_mode = str(os.getenv("CITV_SCENE_PIPELINE_MODE", "")).strip().lower()
    mode = env_mode or cfg_mode or "legacy"
    return "legacy" if mode == "legacy" else "staged"


class SceneUnderstandingPipeline(LegacySceneUnderstandingPipeline):
    """Package-exported pipeline backed by the legacy implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._segmentation_pipeline: Optional[SegmentationPipeline] = None
        self._labeling_pipeline: Optional[LabelingPipeline] = None
        self._relations_pipeline: Optional[RelationsPipeline] = None
        self.depth_coordinator = DepthCoordinator(
            self.depth_estimator,
            depth_scale_factor=getattr(self, "depth_scale_factor", 10.0),
        )
        self.depth_geometry = DepthGeometry(
            back_project_fn=self._back_project,
            mask_erosion_kernel_size=getattr(self, "mask_erosion_kernel_size", 5),
            depth_adaptive_erosion=getattr(self, "depth_adaptive_erosion", True),
            depth_outlier_sigma=getattr(self, "depth_outlier_sigma", 2.0),
            depth_transparency_check=getattr(self, "depth_transparency_check", True),
            depth_transparency_threshold=getattr(
                self,
                "depth_transparency_threshold",
                0.15,
            ),
            depth_central_fraction=getattr(self, "depth_central_fraction", 0.5),
            depth_sigma_clip_scope=getattr(self, "depth_sigma_clip_scope", "mask"),
        )

    def _adaptive_erosion_kernel(self, mask_bin):
        return self.depth_geometry.adaptive_erosion_kernel(mask_bin)

    def _mask_depth_stats_and_3d(
        self,
        metric_depth,
        K,
        mask,
        detection: Optional[dict] = None,
        use_erosion: bool = True,
        region_context: Optional[dict] = None,
        label_map: Optional[np.ndarray] = None,
        region_index: int = 0,
    ):
        return self.depth_geometry.mask_depth_stats_and_3d(
            metric_depth=metric_depth,
            intrinsics=K,
            mask=mask,
            detection=detection,
            use_erosion=use_erosion,
            region_context=region_context,
            label_map=label_map,
            region_index=region_index,
        )

    def process_image(self, image_path: str, output_dir: str):
        """
        Primary package entrypoint.

        Defaults to **legacy** monolithic execution (``PreprocessConfig.scene_pipeline_mode``).
        Use ``"staged"`` or ``CITV_SCENE_PIPELINE_MODE=staged`` for the modular package chain.
        """
        mode = self._resolve_pipeline_mode()
        if mode == "legacy":
            return super().process_image(image_path, output_dir)
        return self.process_image_staged(image_path, output_dir)

    def _run_stage_chain(self, image_path: str, output_dir: str) -> PipelineContext:
        """Execute package-native stage chain (PR2 extraction path)."""
        ctx = self._stage_preprocess(image_path, output_dir)
        ctx = self._stage_depth(ctx)
        ctx = self._stage_regions(ctx)
        ctx = self._stage_segment(ctx)
        ctx = self._stage_label_and_geometry(ctx)
        ctx = self._stage_relations(ctx)
        return ctx

    def process_image_staged(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Staged pipeline entry (package orchestration).

        **Default:** runs the same work as monolithic ``process_image`` (full
        ``scene_graph/grounded_sam2/`` tree, paths, captions, etc.) via
        :mod:`scene_understanding.stages.full_run`, so outputs match legacy mode.

        **Slim chain (tests / incremental migration):** set environment variable
        ``CITV_STAGED_MODULAR_CHAIN_ONLY=1`` to use only ``_run_stage_chain`` plus
        ``scene_graph/staged/{stem}_scene.json``.
        """
        if stage_full_run.use_modular_chain_only():
            ctx = self._run_stage_chain(image_path, output_dir)
            return self._stage_export_package(ctx)
        return stage_full_run.run_legacy_equivalent_process_image(self, image_path, output_dir)

    def _resolve_pipeline_mode(self) -> str:
        """Return normalized pipeline mode: 'legacy' (default) or 'staged'."""
        return resolve_scene_pipeline_mode(getattr(self, "config", None))

    def _stage_preprocess(self, image_path: str, output_dir: str) -> PipelineContext:
        """Load image, apply undistortion/resizing, and resolve intrinsics."""
        return stage_preprocess.run(self, image_path, output_dir)

    def _stage_depth(self, ctx: PipelineContext) -> PipelineContext:
        """Infer/reuse metric depth via DepthCoordinator."""
        return stage_depth.run(self, ctx)

    def _stage_regions(self, ctx: PipelineContext) -> PipelineContext:
        """Optional depth K-means + CC regions (writes label map + PNG under depth/)."""
        return stage_regions.run(self, ctx)

    def _stage_segment(self, ctx: PipelineContext) -> PipelineContext:
        """Generate mask detections using package segmentation pipeline."""
        return stage_segmentation.run(self, ctx)

    def _stage_label_and_geometry(self, ctx: PipelineContext) -> PipelineContext:
        """Attach labels and depth-backed geometry to each detection."""
        return stage_labelling.run(self, ctx)

    def _stage_relations(self, ctx: PipelineContext) -> PipelineContext:
        """Predict relation triplets from staged objects."""
        return stage_relations.run(self, ctx)

    def _stage_export_package(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Write a package-staged scene graph JSON under scene_graph/staged/."""
        return scene_write.export_staged_package(ctx)

    @staticmethod
    def _normalize_staged_object(obj: Dict[str, Any]) -> Dict[str, Any]:
        """Shape staged object payload closer to legacy schema for parity diffs."""
        return scene_write.normalize_staged_object(obj)

    def _get_segmentation_pipeline(self) -> SegmentationPipeline:
        if self._segmentation_pipeline is None:
            query = getattr(
                self.sam2_wrapper,
                "text_query",
                getattr(self.config, "grounding_dino_text_query", "object.") if self.config else "object.",
            )
            self._segmentation_pipeline = SegmentationPipeline(
                device=self.device if isinstance(self.device, torch.device) else torch.device(str(self.device)),
                sam2_checkpoint_path=getattr(self.config, "sam2_checkpoint_path", "sam2/checkpoints/sam2.1_hiera_large.pt") if self.config else "sam2/checkpoints/sam2.1_hiera_large.pt",
                sam2_model_cfg=getattr(self.config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l") if self.config else "configs/sam2.1/sam2.1_hiera_l",
                gdino_model_id=getattr(self.config, "grounding_dino_model", "IDEA-Research/grounding-dino-base") if self.config else "IDEA-Research/grounding-dino-base",
                gdino_box_thresh=float(getattr(self.config, "grounding_dino_box_thresh", 0.3)) if self.config else 0.3,
                gdino_text_thresh=float(getattr(self.config, "grounding_dino_text_thresh", 0.25)) if self.config else 0.25,
                text_query=query,
                use_grounded_sam2=True,
            )
        return self._segmentation_pipeline

    def _get_labeling_pipeline(self) -> LabelingPipeline:
        if self._labeling_pipeline is None:
            self._labeling_pipeline = LabelingPipeline(
                device=self.device if isinstance(self.device, torch.device) else torch.device(str(self.device)),
                florence2_model_id=getattr(self.config, "florence2_model", "microsoft/Florence-2-large") if self.config else "microsoft/Florence-2-large",
                ram_checkpoint=getattr(self.config, "rampp_checkpoint_path", None) if self.config else None,
                ram_repo_path=getattr(self.config, "rampp_repo_path", None) if self.config else None,
                ram_enabled=bool(getattr(self.config, "rampp_enabled", True)) if self.config else True,
                florence2_confidence=0.75,
                ram_confidence=float(getattr(self.config, "rampp_default_confidence", 0.7)) if self.config else 0.7,
            )
        return self._labeling_pipeline

    def _get_relations_pipeline(self) -> RelationsPipeline:
        if self._relations_pipeline is None:
            florence2 = self._get_labeling_pipeline().florence2
            self._relations_pipeline = RelationsPipeline(
                device=self.device if isinstance(self.device, torch.device) else torch.device(str(self.device)),
                triplets_dir=str(getattr(self.config, "pix2sg_triplets_dir", "pix2sg_triplets")) if self.config else "pix2sg_triplets",
                max_relations_per_object=int(getattr(self.config, "pix2sg_spatial_max_relations_per_object", 8)) if self.config else 8,
                mask_overlap_thresh=float(getattr(self.config, "pix2sg_mask_overlap_thresh", 0.05)) if self.config else 0.05,
                depth_near_threshold=float(getattr(self.config, "pix2sg_depth_near_threshold", 1.0)) if self.config else 1.0,
                depth_far_threshold=float(getattr(self.config, "pix2sg_depth_far_threshold", 3.0)) if self.config else 3.0,
                florence2_wrapper=florence2,
                relation_min_mask_overlap=float(getattr(self.config, "relation_min_mask_overlap", 0.05)) if self.config else 0.05,
                region_relation_mode=str(getattr(self.config, "region_relation_mode", "all")) if self.config else "all",
                relation_bbox_touch_margin_px=int(getattr(self.config, "relation_bbox_touch_margin_px", 2)) if self.config else 2,
            )
        return self._relations_pipeline


__all__ = ["SceneUnderstandingPipeline", "LegacySceneUnderstandingPipeline", "resolve_scene_pipeline_mode"]
