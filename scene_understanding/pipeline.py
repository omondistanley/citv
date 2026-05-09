"""Main package pipeline entry point."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .depth.coordinator import DepthCoordinator
from .geometry.depth_geometry import DepthGeometry
from .labeling import LabelingPipeline
from .pipeline_context import PipelineContext
from .preprocessing import CameraCalibration
from .relations import RelationsPipeline
from .segmentation import SegmentationPipeline
from .stages import depth as stage_depth
from .stages import labelling as stage_labelling
from .stages import preprocess as stage_preprocess
from .stages import regions as stage_regions
from .stages import relations_stage as stage_relations
from .stages import full_run, scene_write
from .stages import segmentation as stage_segmentation
from .utils import iou_xyxy



class SceneUnderstandingPipeline:
    """Package-native staged pipeline (legacy-free runtime)."""

    def __init__(self, depth_estimator: Any, config: Optional[Any] = None):
        self.depth_estimator = depth_estimator
        self.config = config
        self.device = getattr(depth_estimator, "device", torch.device("cpu"))
        self._segmentation_pipeline: Optional[SegmentationPipeline] = None
        self._labeling_pipeline: Optional[LabelingPipeline] = None
        self._relations_pipeline: Optional[RelationsPipeline] = None
        self._timing_logger: Optional[Any] = None
        # RAM++ instance shared between pre-SAM2 GDINO query refresh and labelling.
        self._rampp_shared: Optional[Any] = None

        cfg = config
        self._reuse_cached_depth = bool(getattr(cfg, "reuse_cached_depth", True)) if cfg else True
        self.apply_undistortion = bool(getattr(cfg, "apply_undistortion", True)) if cfg else True
        self.fixed_intrinsics = None

        calibration_dict: Optional[Dict[str, float]] = None
        calib_file = str(getattr(cfg, "camera_calibration_file", "") or "").strip() if cfg else ""
        if calib_file:
            try:
                calibration_dict = json.loads(Path(calib_file).read_text(encoding="utf-8"))
            except Exception as e:
                print(
                    f"[Intrinsics] calibration file load failed ({calib_file}): {e}; "
                    "falling back to FOV/explicit values."
                )
                calibration_dict = None

        self._camera_calibration = CameraCalibration(
            calibration_dict=calibration_dict,
            camera_fx=getattr(cfg, "camera_fx", None) if cfg else None,
            camera_fy=getattr(cfg, "camera_fy", None) if cfg else None,
            camera_cx=getattr(cfg, "camera_cx", None) if cfg else None,
            camera_cy=getattr(cfg, "camera_cy", None) if cfg else None,
            camera_fov_degrees=float(getattr(cfg, "camera_fov_degrees", 60.0)) if cfg else 60.0,
            apply_undistortion=self.apply_undistortion,
        )

        self.depth_coordinator = DepthCoordinator(
            self.depth_estimator,
            depth_scale_factor=float(getattr(cfg, "depth_scale_factor", 1.0)) if cfg else 1.0,
        )
        self.depth_geometry = DepthGeometry(
            back_project_fn=self._back_project,
            mask_erosion_kernel_size=int(getattr(cfg, "mask_erosion_kernel_size", 5)) if cfg else 5,
            depth_adaptive_erosion=bool(getattr(cfg, "depth_adaptive_erosion", True)) if cfg else False,
            depth_outlier_sigma=float(getattr(cfg, "depth_outlier_sigma", 2.0)) if cfg else 2.0,
            depth_transparency_check=bool(getattr(cfg, "depth_transparency_check", True)) if cfg else False,
            depth_transparency_threshold=float(getattr(cfg, "depth_transparency_threshold", 0.15)) if cfg else 0.15,
            depth_central_fraction=float(getattr(cfg, "depth_central_fraction", 0.5)) if cfg else 0.5,
            depth_sigma_clip_scope=str(getattr(cfg, "depth_sigma_clip_scope", "mask")) if cfg else "mask",
        )

    def attach_timing_logger(self, logger) -> None:
        self._timing_logger = logger

    def _estimate_intrinsics(self, width: int, height: int) -> Dict[str, float]:
        return self._camera_calibration.get_intrinsics(width, height)

    def _undistort_image(self, img_bgr: np.ndarray) -> np.ndarray:
        return self._camera_calibration.undistort_image(img_bgr)

    @staticmethod
    def _back_project(u: int, v: int, z: float, K: Dict[str, float]) -> Dict[str, float]:
        return CameraCalibration.back_project(u, v, z, K)

    @staticmethod
    def _bbox_iou_xyxy(box1, box2) -> float:
        return float(iou_xyxy(box1, box2))

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
        """Primary package entrypoint (staged chain only)."""
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
        """Staged pipeline entry (full package orchestration)."""
        _t0 = time.perf_counter()
        payload = full_run.run_single_image(self, image_path, output_dir, write_json=True)
        print(f"  [Timing] wall total: {time.perf_counter() - _t0:.3f}s")
        return {"scene_json": str(Path(output_dir) / "scene_graph" / "staged" / f"{Path(image_path).stem}_scene.json"), "payload": payload}

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
        """Normalize staged object payload shape for scene JSON export."""
        return scene_write.normalize_staged_object(obj)

    def _get_segmentation_pipeline(self) -> SegmentationPipeline:
        if self._segmentation_pipeline is None:
            query = getattr(self.config, "grounding_dino_text_query", "object.") if self.config else "object."
            _strict_seg = bool(getattr(self.config, "segmentation_production_strict", False)) if self.config else False
            amg_pred_iou = (
                float(getattr(self.config, "segmentation_production_amg_pred_iou_thresh", 0.90))
                if _strict_seg and self.config
                else float(getattr(self.config, "sam2_amg_pred_iou_thresh", 0.8))
                if self.config
                else 0.8
            )
            amg_stability = (
                float(getattr(self.config, "segmentation_production_amg_stability_score_thresh", 0.95))
                if _strict_seg and self.config
                else float(getattr(self.config, "sam2_amg_stability_score_thresh", 0.92))
                if self.config
                else 0.92
            )
            amg_min_area = (
                int(getattr(self.config, "segmentation_production_amg_min_mask_region_area", 280))
                if _strict_seg and self.config
                else int(getattr(self.config, "sam2_amg_min_mask_region_area", 100))
                if self.config
                else 100
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
                use_sam2_amg=bool(
                    getattr(self.config, "run_both_segmentors", False)
                    or getattr(self.config, "grounded_sam2_fallback_to_amg", False)
                )
                if self.config
                else False,
                amg_points_per_side=int(getattr(self.config, "sam2_amg_points_per_side", 32)) if self.config else 32,
                amg_points_per_batch=int(getattr(self.config, "sam2_amg_points_per_batch", 32)) if self.config else 32,
                amg_pred_iou_thresh=amg_pred_iou,
                amg_stability_score_thresh=amg_stability,
                amg_max_image_side=int(getattr(self.config, "sam2_amg_max_image_side", 1024)) if self.config else 1024,
                amg_crop_n_layers=int(getattr(self.config, "sam2_amg_crop_n_layers", 1)) if self.config else 1,
                amg_crop_overlap_ratio=float(getattr(self.config, "sam2_amg_crop_overlap_ratio", 0.341)) if self.config else 0.341,
                amg_crop_n_points_downscale_factor=int(getattr(self.config, "sam2_amg_crop_n_points_downscale_factor", 2)) if self.config else 2,
                amg_min_mask_region_area=amg_min_area,
                amg_use_m2m=bool(getattr(self.config, "sam2_amg_use_m2m", True)) if self.config else False,
                amg_box_nms_thresh=float(getattr(self.config, "sam2_amg_box_nms_thresh", 0.7)) if self.config else 0.7,
                iou_dedup_threshold=float(getattr(self.config, "run_both_segmentors_iou_dedup", 0.7)) if self.config else 0.7,
                amg_production_strict=_strict_seg,
                amg_part_containment_thresh=(
                    float(getattr(self.config, "segmentation_production_amg_part_containment_thresh", 0.88))
                    if self.config
                    else 0.88
                ),
                amg_part_min_area_ratio_vs_grounded=(
                    float(getattr(self.config, "segmentation_production_amg_part_min_area_ratio_vs_grounded", 0.25))
                    if self.config
                    else 0.25
                ),
            )
        return self._segmentation_pipeline

    def _ensure_rampp_for_gdino_query(self) -> Optional[Any]:
        """Lazily construct RAM++ for pre-segmentation tagging; reused by LabelingPipeline."""
        if self._rampp_shared is not None and getattr(self._rampp_shared, "active", False):
            return self._rampp_shared
        cfg = self.config
        if not cfg or not bool(getattr(cfg, "rampp_enabled", True)):
            return None
        from .labeling.ram_plus_plus import RAMPlusPlusWrapper

        device = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))
        try:
            w = RAMPlusPlusWrapper(
                device,
                checkpoint_path=getattr(cfg, "rampp_checkpoint_path", None),
                repo_path=getattr(cfg, "rampp_repo_path", None),
                image_size=int(getattr(cfg, "rampp_image_size", 384)),
                vit=str(getattr(cfg, "rampp_vit", "swin_l")),
                default_confidence=float(getattr(cfg, "rampp_default_confidence", 0.7)),
                max_tags=int(getattr(cfg, "rampp_max_tags", 8)),
                prefer_mps=bool(getattr(cfg, "rampp_prefer_mps", False)),
                mps_dtype=str(getattr(cfg, "rampp_mps_dtype", "fp16")),
                mps_watermark_fraction=float(getattr(cfg, "rampp_mps_watermark_fraction", 0.85)),
                backend=str(getattr(cfg, "rampp_backend", "rampp")),
                clip_fallback_enabled=bool(getattr(cfg, "rampp_clip_fallback_enabled", True)),
                clip_candidate_labels=list(getattr(cfg, "rampp_clip_candidate_labels", []) or []),
                quantize_dynamic_int8=bool(getattr(cfg, "rampp_quantize_dynamic_int8", True)),
            )
        except Exception as exc:
            print(f"  [Pipeline] RAM++ init for GDINO query refresh failed: {exc}")
            return None
        if not getattr(w, "active", False):
            return None
        self._rampp_shared = w
        return w

    def _get_labeling_pipeline(self) -> LabelingPipeline:
        if self._labeling_pipeline is None:
            shared_ram = self._rampp_shared
            self._labeling_pipeline = LabelingPipeline(
                device=self.device if isinstance(self.device, torch.device) else torch.device(str(self.device)),
                florence2_model_id=getattr(self.config, "florence2_model", "microsoft/Florence-2-large") if self.config else "microsoft/Florence-2-large",
                ram_checkpoint=getattr(self.config, "rampp_checkpoint_path", None) if self.config else None,
                ram_repo_path=getattr(self.config, "rampp_repo_path", None) if self.config else None,
                ram_enabled=bool(getattr(self.config, "rampp_enabled", True)) if self.config else False,
                ram_plus_plus=shared_ram if shared_ram is not None and getattr(shared_ram, "active", False) else None,
                florence2_confidence=0.75,
                ram_confidence=float(getattr(self.config, "rampp_default_confidence", 0.7)) if self.config else 0.7,
            )
        return self._labeling_pipeline

    def _get_relations_pipeline(self) -> RelationsPipeline:
        if self._relations_pipeline is None:
            florence2 = self._get_labeling_pipeline().florence2
            # florence2_relation_enabled defaults to False in config — only pass the
            # wrapper when the flag is explicitly on; otherwise Pix2SG runs spatial
            # scaffold only (fast, no per-pair Florence-2 inference).
            _f2_rel = bool(getattr(self.config, "florence2_relation_enabled", False)) if self.config else False
            self._relations_pipeline = RelationsPipeline(
                device=self.device if isinstance(self.device, torch.device) else torch.device(str(self.device)),
                triplets_dir=str(getattr(self.config, "pix2sg_triplets_dir", "pix2sg_triplets")) if self.config else "pix2sg_triplets",
                max_relations_per_object=int(getattr(self.config, "pix2sg_spatial_max_relations_per_object", 8)) if self.config else 8,
                mask_overlap_thresh=float(getattr(self.config, "pix2sg_mask_overlap_thresh", 0.05)) if self.config else 0.05,
                depth_near_threshold=float(getattr(self.config, "pix2sg_depth_near_threshold", 1.0)) if self.config else 1.0,
                depth_far_threshold=float(getattr(self.config, "pix2sg_depth_far_threshold", 3.0)) if self.config else 3.0,
                florence2_wrapper=florence2 if _f2_rel else None,
                relation_min_mask_overlap=float(getattr(self.config, "relation_min_mask_overlap", 0.05)) if self.config else 0.05,
                region_relation_mode=str(getattr(self.config, "region_relation_mode", "all")) if self.config else "all",
                relation_bbox_touch_margin_px=int(getattr(self.config, "relation_bbox_touch_margin_px", 2)) if self.config else 2,
            )
        return self._relations_pipeline

    def close(self) -> None:
        if self._segmentation_pipeline and getattr(self._segmentation_pipeline, "grounded_sam2", None):
            try:
                self._segmentation_pipeline.grounded_sam2.unload()
            except Exception:
                pass
        if self._segmentation_pipeline and getattr(self._segmentation_pipeline, "sam2_amg", None):
            try:
                self._segmentation_pipeline.sam2_amg._move_model_to_cpu()
            except Exception:
                pass
        self._segmentation_pipeline = None
        self._labeling_pipeline = None
        self._relations_pipeline = None
        self._rampp_shared = None


__all__ = ["SceneUnderstandingPipeline"]
