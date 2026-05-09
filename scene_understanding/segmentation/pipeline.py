"""
Segmentation pipeline orchestrator.

Coordinates GroundedSAM2 (GDINO + SAM2 prompted) and optional SAM2 AMG.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .grounded_sam2 import GroundedSAM2Wrapper
from .sam2_amg import SAM2AMGWrapper


class SegmentationPipeline:
    """GroundedSAM2 primary segmentation with optional AMG coverage fill."""

    def __init__(
        self,
        device: torch.device,
        sam2_checkpoint_path: str,
        sam2_model_cfg: str,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        gdino_box_thresh: float = 0.30,
        gdino_text_thresh: float = 0.25,
        text_query: str = (
            "person. animal. vehicle. furniture. appliance. food. "
            "clothing. container. tool. building. plant. electronics. object."
        ),
        use_grounded_sam2: bool = True,
        use_sam2_amg: bool = False,
        amg_points_per_side: int = 32,
        amg_points_per_batch: int = 32,
        amg_pred_iou_thresh: float = 0.8,
        amg_stability_score_thresh: float = 0.92,
        amg_max_image_side: int = 1024,
        amg_crop_n_layers: int = 1,
        amg_crop_overlap_ratio: float = 0.341,
        amg_crop_n_points_downscale_factor: int = 2,
        amg_min_mask_region_area: int = 100,
        amg_use_m2m: bool = True,
        amg_box_nms_thresh: float = 0.7,
        iou_dedup_threshold: float = 0.7,
        amg_production_strict: bool = False,
        amg_part_containment_thresh: float = 0.88,
        amg_part_min_area_ratio_vs_grounded: float = 0.25,
        **_: Any,
    ) -> None:
        self.device = device
        self.text_query = text_query
        self.grounded_sam2 = None
        self.sam2_amg = None
        self.iou_dedup_threshold = float(iou_dedup_threshold)
        self.amg_production_strict = bool(amg_production_strict)
        self.amg_part_containment_thresh = float(amg_part_containment_thresh)
        self.amg_part_min_area_ratio_vs_grounded = float(amg_part_min_area_ratio_vs_grounded)

        if use_grounded_sam2:
            self.grounded_sam2 = GroundedSAM2Wrapper(
                device=device,
                sam2_checkpoint_path=sam2_checkpoint_path,
                sam2_model_cfg=sam2_model_cfg,
                gdino_model_id=gdino_model_id,
                box_thresh=gdino_box_thresh,
                text_thresh=gdino_text_thresh,
                text_query=text_query,
            )
        if use_sam2_amg:
            self.sam2_amg = SAM2AMGWrapper(
                device=device,
                checkpoint_path=sam2_checkpoint_path,
                model_cfg=sam2_model_cfg,
                points_per_side=amg_points_per_side,
                points_per_batch=amg_points_per_batch,
                pred_iou_thresh=amg_pred_iou_thresh,
                stability_score_thresh=amg_stability_score_thresh,
                max_image_side=amg_max_image_side,
                crop_n_layers=amg_crop_n_layers,
                crop_overlap_ratio=amg_crop_overlap_ratio,
                crop_n_points_downscale_factor=amg_crop_n_points_downscale_factor,
                min_mask_region_area=amg_min_mask_region_area,
                use_m2m=amg_use_m2m,
                box_nms_thresh=amg_box_nms_thresh,
            )

    def generate(
        self,
        image_rgb: np.ndarray,
        use_primary: bool = True,
        use_secondary: bool = False,
        use_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if use_primary and self.grounded_sam2 and self.grounded_sam2.active:
            print("[SegmentationPipeline] Running GroundedSAM2...")
            results.extend(self.grounded_sam2.generate(image_rgb))
        if use_secondary and self.sam2_amg and self.sam2_amg.amg is not None:
            print("[SegmentationPipeline] Running SAM2 AMG...")
            amg_results = self.sam2_amg.generate(image_rgb)
            kept = []
            dropped_dup = 0
            dropped_part = 0
            grounded_masks = [d for d in results if str(d.get("source_model", "")) == "GroundedSAM2"]
            for ann in amg_results:
                ann = dict(ann)
                ann["source_model"] = "SAM2_AMG"
                ann.setdefault("label", "object")
                if _is_duplicate_mask(ann, results, self.iou_dedup_threshold):
                    dropped_dup += 1
                    continue
                if self.amg_production_strict and _is_part_like_relative_to_grounded(
                    ann,
                    grounded_masks,
                    containment_thresh=self.amg_part_containment_thresh,
                    min_area_ratio=self.amg_part_min_area_ratio_vs_grounded,
                ):
                    dropped_part += 1
                    continue
                kept.append(ann)
            results.extend(kept)
            print(
                f"[SegmentationPipeline] AMG kept {len(kept)}/{len(amg_results)} "
                f"(dup_drop={dropped_dup}, part_drop={dropped_part})."
            )
        if use_fallback and not results and self.sam2_amg and self.sam2_amg.amg is not None:
            print("[SegmentationPipeline] Falling back to SAM2 AMG...")
            for ann in self.sam2_amg.generate(image_rgb):
                ann = dict(ann)
                ann["source_model"] = "SAM2_AMG"
                ann.setdefault("label", "object")
                results.append(ann)
        print(f"[SegmentationPipeline] Total masks: {len(results)}")
        return results

    def update_text_query(self, query: str) -> None:
        self.text_query = query
        if self.grounded_sam2:
            self.grounded_sam2.update_text_query(query)

    @property
    def active(self) -> bool:
        return bool(self.grounded_sam2 and self.grounded_sam2.active) or bool(self.sam2_amg and self.sam2_amg.amg is not None)


def _mask_iou(a: Any, b: Any) -> float:
    ma = np.asarray(a, dtype=bool)
    mb = np.asarray(b, dtype=bool)
    if ma.shape != mb.shape or ma.size == 0 or mb.size == 0:
        return 0.0
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return float(inter / union) if union > 0 else 0.0


def _is_duplicate_mask(candidate: Dict[str, Any], existing: List[Dict[str, Any]], threshold: float) -> bool:
    cm = candidate.get("segmentation")
    if cm is None:
        return False
    for det in existing:
        em = det.get("segmentation")
        if em is not None and _mask_iou(cm, em) >= threshold:
            return True
    return False


def _is_part_like_relative_to_grounded(
    candidate: Dict[str, Any],
    grounded_existing: List[Dict[str, Any]],
    *,
    containment_thresh: float = 0.88,
    min_area_ratio: float = 0.25,
) -> bool:
    """Return True when AMG mask is likely a part-mask inside a grounded whole-object mask."""
    cm = candidate.get("segmentation")
    if cm is None or not grounded_existing:
        return False
    ma = np.asarray(cm, dtype=bool)
    if ma.size == 0 or not ma.any():
        return False
    cand_area = float(ma.sum())
    if cand_area <= 0.0:
        return False
    ct = float(np.clip(containment_thresh, 0.0, 1.0))
    ar = max(0.0, float(min_area_ratio))

    for det in grounded_existing:
        gm = det.get("segmentation")
        if gm is None:
            continue
        gb = np.asarray(gm, dtype=bool)
        if gb.shape != ma.shape or not gb.any():
            continue
        inter = float(np.logical_and(ma, gb).sum())
        if inter <= 0.0:
            continue
        contain = inter / cand_area
        g_area = float(gb.sum())
        ratio = (cand_area / g_area) if g_area > 0.0 else 0.0
        if contain >= ct and ratio < ar:
            return True
    return False
