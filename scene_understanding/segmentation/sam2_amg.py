"""SAM2 AMG — canonical implementation (synced from scene_understanding legacy module)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch


class SAM2AMGWrapper:
    """
    Wrapper for SAM2 Automatic Mask Generator.
    Segments "everything" in an image; no prompts. Returns list of mask dicts.
    """
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        model_cfg: str,
        points_per_side: int = 32,
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        max_image_side: int = 1280,
        crop_n_layers: int = 1,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 200,
        use_m2m: bool = True,
        box_nms_thresh: float = 0.7,
    ):
        self.device = device
        self.amg = None
        self.max_image_side = int(max_image_side) if max_image_side else 0
        self._force_cpu = False
        self.crop_n_layers = crop_n_layers
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.use_m2m = use_m2m
        self.box_nms_thresh = box_nms_thresh
        print("Initializing SAM2 Automatic Mask Generator...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            # Checkpoint: resolve for existence check; pass path to build_sam2.
            ckpt = Path(checkpoint_path)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            if not ckpt.exists():
                print(f"SAM2 checkpoint not found at {ckpt}. Depth-mask branch disabled.")
                return
            # Config: pass as Hydra config name (e.g. "configs/sam2.1/sam2.1_hiera_l"), not a file path.
            # build_sam2 uses compose(config_name=...) so an absolute path would be mis-resolved.
            hydra_config_name = model_cfg if isinstance(model_cfg, str) else str(model_cfg)
            if hydra_config_name.endswith(".yaml"):
                hydra_config_name = hydra_config_name[:-5]
            print(f"Loading SAM2 with config: {hydra_config_name}")
            apply_pp = device.type == "cuda"
            model = build_sam2(
                hydra_config_name,
                str(ckpt),
                device=device,
                apply_postprocessing=apply_pp,
            )
            self.amg = SAM2AutomaticMaskGenerator(
                model,
                output_mode="binary_mask",
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
                use_m2m=use_m2m,
                box_nms_thresh=box_nms_thresh,
            )
            print("SAM2 AMG initialized.")
        except ImportError as e:
            print(f"SAM2 not available: {e}. Depth-mask branch disabled.")
        except Exception as e:
            print(f"Failed to load SAM2 AMG: {e}. Depth-mask branch disabled.")

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run AMG on image (HWC uint8 RGB). Returns list of dicts with segmentation, bbox (xywh), area, predicted_iou, stability_score."""
        if self.amg is None:
            return []
        try:
            if self._force_cpu:
                self._move_model_to_cpu()

            anns = self._generate_with_optional_resize(image_rgb)
            return anns if anns else []
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"SAM2 AMG generate failed: {e}")
                return []

            print("SAM2 AMG OOM on GPU. Retrying with reduced memory settings.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._move_model_to_cpu()
                self._force_cpu = True
                anns = self._generate_with_optional_resize(image_rgb)
                return anns if anns else []
            except Exception as e2:
                print(f"SAM2 AMG generate failed after CPU fallback: {e2}")
                return []
        except Exception as e:
            print(f"SAM2 AMG generate failed: {e}")
            return []

    def _move_model_to_cpu(self) -> None:
        if self.amg is None:
            return
        try:
            predictor = getattr(self.amg, "predictor", None)
            model = getattr(predictor, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass

    def _generate_with_optional_resize(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        if self.amg is None:
            return []
        h, w = image_rgb.shape[:2]
        long_side = max(h, w)
        scale = 1.0
        proc = image_rgb
        if self.max_image_side > 0 and long_side > self.max_image_side:
            scale = float(self.max_image_side) / float(long_side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            proc = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        if torch.cuda.is_available() and getattr(self.amg.predictor, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            anns = self.amg.generate(proc)
        if not anns:
            return []

        if scale == 1.0:
            return anns

        inv = 1.0 / scale
        resized_anns = []
        for ann in anns:
            ann_new = dict(ann)
            seg = ann_new.get("segmentation")
            if seg is not None:
                mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                ann_new["segmentation"] = (mask > 0)
            bbox = ann_new.get("bbox")
            if bbox is not None and len(bbox) >= 4:
                x, y, bw, bh = bbox[:4]
                ann_new["bbox"] = [float(x * inv), float(y * inv), float(bw * inv), float(bh * inv)]
            if "area" in ann_new:
                ann_new["area"] = int(np.sum(np.asarray(ann_new["segmentation"]) > 0))
            resized_anns.append(ann_new)
        return resized_anns

