"""
Labeling pipeline orchestrator.

Combines Florence2 (rich captions + relations) and RAM++ (tag-based) labeling approaches
for robust open-vocabulary object labeling and relation prediction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ram_plus_plus import RAMPlusPlusWrapper

import cv2
import numpy as np
import torch

from .florence2 import Florence2Wrapper
from .ram_plus_plus import RAMPlusPlusWrapper


class LabelingPipeline:
    """
    Multi-label labeling system combining Florence2 and RAM++.

    Strategy:
    - Primary: Florence2 (generates rich captions, extracts nouns)
    - Fallback: RAM++ (tag-based open-vocabulary recognition)
    - Relations: Florence2's colored overlay technique
    """

    def __init__(
        self,
        device: torch.device,
        florence2_checkpoint: Optional[str] = None,
        florence2_model_id: str = "microsoft/Florence-2-base",
        ram_checkpoint: Optional[str] = None,
        ram_repo_path: Optional[str] = None,
        ram_enabled: bool = True,
        ram_plus_plus: Optional["RAMPlusPlusWrapper"] = None,
        florence2_confidence: float = 0.75,
        ram_confidence: float = 0.70,
    ):
        """
        Initialize labeling pipeline.

        Args:
            device: torch device (cuda / cpu)
            florence2_checkpoint: Optional Florence2 checkpoint path
            florence2_model_id: Hugging Face model ID for Florence2
            ram_checkpoint: Path to RAM++ checkpoint
            ram_repo_path: Path to RAM repository
            ram_enabled: Whether to enable RAM++
            ram_plus_plus: Pre-built RAM++ instance (e.g. from pre-SAM2 GDINO query refresh)
            florence2_confidence: Default confidence for Florence2
            ram_confidence: Default confidence for RAM++
        """
        self.device = device
        # Florence2Wrapper only accepts (model_id, device); loads from HF via model_id.
        # florence2_checkpoint is reserved for future local-weights support.
        _ = florence2_checkpoint  # noqa: F841
        _ = florence2_confidence  # noqa: F841 — wrapper has no confidence ctor arg
        self.florence2 = Florence2Wrapper(florence2_model_id, device)
        self.ram_plus_plus = None

        if ram_plus_plus is not None:
            self.ram_plus_plus = ram_plus_plus
        elif ram_enabled:
            self.ram_plus_plus = RAMPlusPlusWrapper(
                device,
                checkpoint_path=ram_checkpoint,
                repo_path=ram_repo_path,
                default_confidence=ram_confidence,
            )

    def label_crop(
        self, crop_bgr: np.ndarray, use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Label a masked crop using primary (Florence2) and optional fallback (RAM++).

        Args:
            crop_bgr: Masked crop in BGR format
            use_fallback: If True, use RAM++ as fallback when Florence2 result is generic

        Returns:
            Dict with keys: label, conf, caption, tags
        """
        # Primary: Florence2
        result = self.florence2.label_crop(crop_bgr)

        # Fallback: If Florence2 label is too generic, try RAM++
        if use_fallback and self.ram_plus_plus and self.ram_plus_plus.active:
            label = result.get("label", "object")
            if label in {"object", "thing", "item"} or not label:
                ram_result = self.ram_plus_plus.label_crop(crop_bgr)
                if ram_result and ram_result.get("label", "object") not in {"object", "thing", "item"}:
                    return ram_result

        return result

    def tag_image(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Tag a full image using primary (Florence2) and optional fallback (RAM++).

        Args:
            image_rgb: RGB image (HxWx3)

        Returns:
            Dict with keys: label, conf, caption, tags
        """
        # Primary: Florence2 (expects a PIL image; *image_rgb* is HxWx3 RGB uint8)
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image_rgb)
        cap_result = self.florence2._run_task("<CAPTION>", pil_img)
        caption = cap_result.get("<CAPTION>", "")
        if not isinstance(caption, str):
            caption = str(caption)
        label = self.florence2._extract_label_from_caption(caption)

        # Fallback: If Florence2 label is too generic, try RAM++
        if self.ram_plus_plus and self.ram_plus_plus.active:
            if label in {"object", "thing", "item"} or not label:
                ram_result = self.ram_plus_plus.tag_image(image_rgb)
                if ram_result and ram_result.get("label", "object") not in {"object", "thing", "item"}:
                    return ram_result

        return {
            "label": label if label and label not in {"object", "thing", "item"} else "object",
            "conf": 0.75 if label and label not in {"object", "thing", "item"} else 0.0,
            "caption": caption,
            "tags": [label] if label and label not in {"object", "thing", "item"} else [],
        }

    def predict_relation(
        self,
        subject_mask: np.ndarray,
        obj_mask: np.ndarray,
        image_bgr: np.ndarray,
        label_sub: str = "object",
        label_obj: str = "object",
    ) -> str:
        """Predict spatial relation between subject and object.

        Uses Florence-2's coloured-overlay technique (RED=subject, BLUE=object)
        with label injection so the model focuses on the named objects.
        """
        if not self.florence2.active:
            return "near"

        return self.florence2.predict_relation(image_bgr, subject_mask, obj_mask, label_sub, label_obj)

    @property
    def active(self) -> bool:
        """Check if at least one labeling backend is active."""
        return self.florence2.active or (self.ram_plus_plus and self.ram_plus_plus.active)
