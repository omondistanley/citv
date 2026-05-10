"""
Segmentation pipeline orchestrator.

Coordinates GroundedSAM2 (GDINO + SAM2 prompted) only.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .grounded_sam2 import GroundedSAM2Wrapper


class SegmentationPipeline:
    """GroundedSAM2 (GDINO + SAM2) segmentation."""

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
        **_: Any,
    ) -> None:
        self.device = device
        self.text_query = text_query
        self.grounded_sam2 = None

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

    def generate(
        self,
        image_rgb: np.ndarray,
        use_primary: bool = True,
        use_secondary: bool = False,
        use_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        del use_secondary, use_fallback
        results: List[Dict[str, Any]] = []
        if use_primary and self.grounded_sam2 and self.grounded_sam2.active:
            print("[SegmentationPipeline] Running GroundedSAM2...")
            results.extend(self.grounded_sam2.generate(image_rgb))
        print(f"[SegmentationPipeline] Total masks: {len(results)}")
        return results

    def update_text_query(self, query: str) -> None:
        self.text_query = query
        if self.grounded_sam2:
            self.grounded_sam2.update_text_query(query)

    @property
    def active(self) -> bool:
        return bool(self.grounded_sam2 and self.grounded_sam2.active)
