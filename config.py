from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class PreprocessConfig:
    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    device: str = "cuda"
    target_size: Tuple[int, int] = (512, 512)
    save_depth_visualizations: bool = True
    save_depth_16bit: bool = False

    # -------------------------------------------------------------------------
    # Fix 5.1 — Depth model variant selection
    #
    # depth_model_variant controls which Depth Anything V2 Metric model loads:
    #   "auto"    — CLIP classifies the first image as indoor/outdoor, then
    #               the matching metric model is loaded. CLIP is immediately
    #               unloaded after classification to reclaim VRAM.
    #   "indoor"  — Always load Depth-Anything-V2-Metric-Indoor-Large-hf.
    #               Trained on NYUv2; best for rooms, offices, kitchens.
    #   "outdoor" — Always load Depth-Anything-V2-Metric-Outdoor-Large-hf.
    #               Trained on KITTI; best for streets, parks, buildings.
    #
    # depth_scale_factor MUST be 1.0 for metric models (output is already
    # meters). The old ×10 hack only applied to the relative model.
    # -------------------------------------------------------------------------
    depth_model: str = "depth_anything_v2_metric"
    depth_model_variant: str = "auto"     # "auto" | "indoor" | "outdoor"
    depth_scale_factor: float = 1.0       # 1.0 = no scaling (metric output is meters)

    # -------------------------------------------------------------------------
    # SAM2 AMG parameters
    # -------------------------------------------------------------------------
    sam2_checkpoint_path: str = "sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l"
    sam2_amg_points_per_side: int = 32
    sam2_amg_points_per_batch: int = 32
    # Slightly more permissive at AMG level so post-hoc filters have candidates
    # to work with. The two-stage approach gives finer control than a single threshold.
    sam2_amg_pred_iou_thresh: float = 0.80        # lowered from 0.88 → more candidates
    sam2_amg_stability_score_thresh: float = 0.92  # lowered from 0.97 → more candidates
    sam2_amg_max_image_side: int = 1280
    save_per_object_masks: bool = True
    save_masked_depth_npy: bool = False   # ~48 MB per object per mode — leave False
    sam2_amg_crop_n_layers: int = 1
    sam2_amg_crop_overlap_ratio: float = 0.341
    sam2_amg_crop_n_points_downscale_factor: int = 2
    sam2_amg_min_mask_region_area: int = 100   # lowered to detect small objects
    sam2_amg_use_m2m: bool = True
    sam2_amg_box_nms_thresh: float = 0.7

    # -------------------------------------------------------------------------
    # Post-hoc quality filters — ALL DISABLED
    #
    # Filtering is fully disabled so every mask is retained:
    #   small objects (buttons, coins, labels), background regions (sky, floor,
    #   wall), part-level masks (from AMG), and object-level masks (from GDINO).
    # Set thresholds to non-zero values to re-enable selective filtering.
    # -------------------------------------------------------------------------
    sam2_post_filter_min_stability: float = 0.0    # 0.0 = disabled
    sam2_post_filter_min_pred_iou: float = 0.0     # 0.0 = disabled
    sam2_post_filter_min_area_px: int = 0           # 0 = allow all sizes
    sam2_post_filter_max_area_fraction: float = 1.0 # 1.0 = allow background
    # GroundedSAM2 confidence gate — disabled (0.0 keeps all GDINO detections)
    grounded_sam2_min_conf_for_stage3: float = 0.0

    # -------------------------------------------------------------------------
    # Relation pipeline
    # -------------------------------------------------------------------------
    require_any_relation_source: bool = True
    sgsg_match_iou_thresh: float = 0.1
    pix2sg_triplets_dir: str = "pix2sg_triplets"
    pix2sg_spatial_max_relations_per_object: int = 8
    mask_iou_match_thresh: float = 0.1
    pix2sg_mask_overlap_thresh: float = 0.05
    pix2sg_depth_near_threshold: float = 1.0
    pix2sg_depth_far_threshold: float = 3.0
    depth_mask_matching_modes: List[str] = field(default_factory=lambda: ["A", "B"])

    # -------------------------------------------------------------------------
    # Fix 5.2 — Camera intrinsics via OpenCV calibration
    #
    # Priority order for intrinsics resolution (highest to lowest):
    #   1. camera_calibration_file — load a JSON file produced by
    #      tools/calibrate_camera.py (OpenCV checkerboard calibration).
    #      Provides fx, fy, cx, cy, and distortion coefficients k1,k2,p1,p2.
    #   2. camera_fx / camera_fy / camera_cx / camera_cy — explicit pixel
    #      values, e.g. read from EXIF or camera spec sheet.
    #   3. camera_fov_degrees — estimate fx from horizontal FOV.
    #      Least accurate; used only when the above are absent.
    #
    # OpenCV calibration workflow (run once per camera, outside pipeline):
    #   python tools/calibrate_camera.py \
    #       --images path/to/checkerboard_frames/ \
    #       --pattern 9x6 \
    #       --square_size 0.025 \   # metres per square
    #       --out calibration.json
    # Then set: camera_calibration_file = "calibration.json"
    #
    # Distortion coefficients (k1, k2, p1, p2) are stored in the JSON and
    # applied to undistort the image before any depth or mask processing,
    # which is critical for accurate back-projection.
    # -------------------------------------------------------------------------
    camera_calibration_file: Optional[str] = None  # path to calibration JSON
    camera_fx: Optional[float] = None
    camera_fy: Optional[float] = None
    camera_cx: Optional[float] = None
    camera_cy: Optional[float] = None
    camera_fov_degrees: float = 60.0
    # Whether to apply lens distortion correction before processing
    apply_undistortion: bool = True

    # -------------------------------------------------------------------------
    # Fix 5.7 — Depth accuracy: adaptive erosion + outlier rejection
    #
    # mask_erosion_kernel_size: base erosion kernel; actual kernel is adapted
    #   to object size (see SceneUnderstandingPipeline._adaptive_erosion_kernel).
    #   Set to 0 to disable all erosion.
    # depth_adaptive_erosion: if True, kernel is scaled to the narrowest mask
    #   dimension to avoid destroying thin objects.
    # depth_central_fraction: inner circle of mask used for z_val histogram.
    # depth_outlier_sigma: reject mask pixels beyond N standard deviations from
    #   the mask mean depth. 0 = disabled. 2.0 is recommended.
    # depth_transparency_check: compute separation between mask depth and border
    #   depth; flag objects where they are nearly equal (transparent objects).
    # depth_transparency_threshold: minimum depth separation (metres) required
    #   to classify a mask as opaque. Below this → flagged as transparent.
    # -------------------------------------------------------------------------
    mask_erosion_kernel_size: int = 5
    depth_adaptive_erosion: bool = True
    depth_central_fraction: float = 0.5
    depth_outlier_sigma: float = 2.0       # 0 = disabled; 2.0 recommended
    depth_transparency_check: bool = True
    depth_transparency_threshold: float = 0.15  # metres
    # depth_erosion_comparison: if True, compute depth_stats twice per object —
    #   once WITH adaptive erosion and once WITHOUT — so results can be compared.
    #   Both sets are stored in the scene JSON under depth_stats / depth_stats_no_erosion
    #   and coordinates_3d / coordinates_3d_no_erosion.
    depth_erosion_comparison: bool = True

    # -------------------------------------------------------------------------
    # Florence-2 labelling (Fix 5.4)
    #
    # florence2_model: HuggingFace model ID for Florence-2.
    #   "microsoft/Florence-2-large" — best accuracy (~900 MB).
    #   "microsoft/Florence-2-base"  — faster, ~500 MB.
    # florence2_task: Florence-2 task token for per-crop labelling.
    #   "<OD>" returns structured {label, bbox} — we take the highest-conf label.
    #   "<CAPTION>" returns a sentence — less structured but works without boxes.
    # florence2_label_enabled:
    #   True  -> Florence-2 participates in object labelling.
    #   False -> Florence-2 is skipped for labelling (can still be used for relations
    #            via florence2_relation_enabled).
    # -------------------------------------------------------------------------
    florence2_model: str = "microsoft/Florence-2-large"
    florence2_task: str = "<OD>"
    florence2_label_enabled: bool = True

    # -------------------------------------------------------------------------
    # Grounded-SAM2 (Fix 5.3)
    #
    # grounding_dino_model: HuggingFace model ID for Grounding DINO v2.
    #   "IDEA-Research/grounding-dino-base" — 341 MB, best precision.
    #   "IDEA-Research/grounding-dino-tiny" — 172 MB, faster.
    # grounding_dino_box_thresh: minimum detection confidence to pass a bbox
    #   to SAM2 for mask generation.
    # grounding_dino_text_thresh: token confidence threshold (GDINO-specific).
    # grounding_dino_text_query: open-vocabulary query fed to GDINO.
    #   Use broad entity categories so all scene objects are detected.
    # grounded_sam2_fallback_to_amg: if Grounding DINO fails to initialize or
    #   produces zero detections, fall back to SAM2 AMG (original behaviour).
    # -------------------------------------------------------------------------
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_thresh: float = 0.15   # lowered to detect small/weak objects
    grounding_dino_text_thresh: float = 0.15  # lowered to match
    grounding_dino_text_query: str = (
        "person. animal. vehicle. furniture. appliance. food. clothing. "
        "container. tool. building. plant. electronics. object."
    )
    grounded_sam2_fallback_to_amg: bool = True  # safety net if GDINO finds zero objects
    # run_both_segmentors: if True, run GroundedSAM2 AND SAM2 AMG together.
    #   GroundedSAM2 provides object-level masks (one per detected entity).
    #   SAM2 AMG provides part-level and small-object masks missed by GDINO.
    #   Duplicates are removed by mask IoU: if an AMG mask overlaps a GDINO mask
    #   by more than run_both_segmentors_iou_dedup, the AMG mask is dropped
    #   (the GDINO-prompted mask is higher quality for that object).
    # run_both_segmentors_iou_dedup: IoU threshold for deduplication (0.7 = 70%).
    run_both_segmentors: bool = True
    run_both_segmentors_iou_dedup: float = 0.7

    # -------------------------------------------------------------------------
    # Relation enhancement (Fix 5.6)
    #
    # florence2_relation_enabled: use Florence-2 to predict semantic relations
    #   between each overlapping object pair.
    # psgtr_enabled: load PSGTR for panoptic scene graph relations.
    # psgtr_checkpoint: path to PSGTR checkpoint (download separately).
    # univrd_enabled: load UniVRD for visual relationship detection.
    # univrd_checkpoint: path to UniVRD checkpoint.
    # relation_min_mask_overlap: minimum pixel IoU between two object masks
    #   before we attempt semantic relation prediction (avoids running VQA on
    #   every pair — most pairs have no spatial contact).
    # -------------------------------------------------------------------------
    florence2_relation_enabled: bool = True
    psgtr_enabled: bool = False           # set True once checkpoint is available
    psgtr_checkpoint: Optional[str] = None
    univrd_enabled: bool = False          # set True once checkpoint is available
    univrd_checkpoint: Optional[str] = None
    relation_min_mask_overlap: float = 0.02  # pixel IoU threshold for semantic relation attempt

    # -------------------------------------------------------------------------
    # RAM++ labelling (replaces GRiT + YOLOv8-cls fallback)
    #
    # Requires local Recognize Anything installation (module "ram") and a
    # local checkpoint file path.
    #
    # rampp_enabled: enable RAM++ fallback in object labelling.
    # rampp_checkpoint_path: local .pth checkpoint path for RAM++.
    #   Example: "recognize-anything/pretrained/ram_plus_swin_large_14m.pth"
    # rampp_repo_path: optional local repo path to append to sys.path when RAM
    #   is not installed into site-packages.
    # -------------------------------------------------------------------------
    rampp_enabled: bool = True
    rampp_checkpoint_path: Optional[str] = "checkpoints/ram_plus_swin_large_14m.pth"
    rampp_repo_path: Optional[str] = None
    rampp_image_size: int = 384
    rampp_vit: str = "swin_l"
    rampp_default_confidence: float = 0.70
    rampp_max_tags: int = 8

    # -------------------------------------------------------------------------
    # SAM3 (parallel segmentor — comparison with SAM2)
    #
    # run_sam3: if True, SAM3 is initialised alongside SAM2 and run on every
    #   image.  Results are written to scene_graph/sam3/ as a parallel track.
    #   The main _scene.json keeps SAM2 objects in "objects"; SAM3 objects are
    #   stored separately under "objects_sam3" for direct comparison.
    # sam3_only: if True, pipeline runs only depth + SAM3 (no SAM2). Use for
    #   Run 2 when comparing SAM2 vs SAM3 in two separate runs; avoids loading
    #   both segmentors in one process.
    # sam3_only_use_existing_depth: if True and sam3_only, load metric_depth
    #   from {output_dir}/depth/{stem}_depth_metric.npy when present (e.g.
    #   from Run 1); otherwise compute depth. Saves time and memory in Run 2.
    # sam3_confidence_threshold: minimum SAM3 presence score to keep a mask.
    # sam3_checkpoint_path: local path to sam3.pt; if None and
    #   sam3_load_from_hf=True the checkpoint is downloaded from HF hub.
    # sam3_text_query: text prompt fed to SAM3 (same free-form style as GDINO).
    # -------------------------------------------------------------------------
    run_sam3: bool = False          # opt-in — set True to enable SAM3 pass
    sam3_only: bool = False         # Run 2: depth + SAM3 only, no SAM2
    sam3_only_use_existing_depth: bool = False  # Reuse depth from Run 1 when sam3_only
    sam3_confidence_threshold: float = 0.3
    sam3_checkpoint_path: Optional[str] = "/home/stanleyomondi/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
    sam3_load_from_hf: bool = False  # checkpoint already cached locally
    sam3_text_query: str = (
        "person. animal. vehicle. furniture. appliance. food. "
        "clothing. container. tool. building. plant. electronics. object."
    )
