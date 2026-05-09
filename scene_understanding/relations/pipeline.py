"""
Relations pipeline orchestrator.

Generates scene graph relation triplets using spatial and semantic methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .pix2sg import Pix2SGWrapper


class RelationsPipeline:
    """
    Relations pipeline for scene graph generation.

    Orchestrates spatial and semantic relation prediction.
    """

    def __init__(
        self,
        device: torch.device,
        triplets_dir: str = "pix2sg_triplets",
        max_relations_per_object: int = 8,
        mask_overlap_thresh: float = 0.05,
        depth_near_threshold: float = 1.0,
        depth_far_threshold: float = 3.0,
        florence2_wrapper: Optional[Any] = None,
        relation_min_mask_overlap: float = 0.05,
        region_relation_mode: str = "all",
        relation_bbox_touch_margin_px: int = 2,
    ):
        """
        Initialize relations pipeline.

        Args:
            device: torch device (cuda / cpu)
            triplets_dir: Path to precomputed triplets
            max_relations_per_object: Max relations per object
            mask_overlap_thresh: Mask overlap threshold
            depth_near_threshold: Near depth threshold
            depth_far_threshold: Far depth threshold
            florence2_wrapper: Optional Florence2 wrapper for semantic enrichment
            relation_min_mask_overlap: Mask overlap threshold for Florence2
            region_relation_mode: "all" or "intra_region_only" for relation pairs
            relation_bbox_touch_margin_px: Bbox expansion (px) for pre-gating Florence pairs
        """
        self.device = device
        self.pix2sg = Pix2SGWrapper(
            device=device,
            triplets_dir=triplets_dir,
            max_relations_per_object=max_relations_per_object,
            mask_overlap_thresh=mask_overlap_thresh,
            depth_near_threshold=depth_near_threshold,
            depth_far_threshold=depth_far_threshold,
            florence2=florence2_wrapper,
            relation_min_mask_overlap=relation_min_mask_overlap,
            region_relation_mode=region_relation_mode,
            relation_bbox_touch_margin_px=relation_bbox_touch_margin_px,
        )

    def predict_relations(
        self,
        image: np.ndarray,
        image_stem: str = "",
        detections: Optional[List[Dict[str, Any]]] = None,
        iou_func: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict relations for a set of detections.

        Args:
            image: Image array (BGR)
            image_stem: Image stem for precomputed lookup
            detections: List of detected objects
            iou_func: Optional IoU function

        Returns:
            List of relation triplets
        """
        if self.pix2sg.is_active():
            return self.pix2sg.predict(image, image_stem, detections, iou_func)
        return []

    def status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return self.pix2sg.status()

    @property
    def active(self) -> bool:
        """Check if pipeline is active."""
        return self.pix2sg.is_active()
