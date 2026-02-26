from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class PreprocessConfig:
    device: str = "cuda"
    target_size: Tuple[int, int] = (512, 512)
    save_depth_visualizations: bool = True
    save_depth_16bit: bool = False
    depth_model: str = "depth_anything_v2_large"
    # Depth+mask (SAM2 AMG) options
    depth_mask_matching_modes: List[str] = field(default_factory=lambda: ["A", "B"])
    sam2_checkpoint_path: str = "sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_amg_points_per_side: int = 32
    sam2_amg_pred_iou_thresh: float = 0.8
    sam2_amg_stability_score_thresh: float = 0.95
    save_per_object_masks: bool = True
