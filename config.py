from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal


@dataclass
class PreprocessConfig:
    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    device: str = "cuda"
    # Package pipeline: "legacy" = monolithic process_image; "staged" = package orchestration.
    # When "staged", full outputs match legacy (via stages/full_run → LegacySceneUnderstandingPipeline.process_image).
    # Set CITV_STAGED_MODULAR_CHAIN_ONLY=1 for the slim PR2 chain (scene_graph/staged/ JSON only).
    # Env CITV_SCENE_PIPELINE_MODE overrides this when set (see scene_understanding/pipeline.py).
    # Default remains "legacy" until you opt in to staged for production.
    scene_pipeline_mode: Literal["legacy", "staged"] = "legacy"
    # Prompting: how Grounding DINO text query is built before SAM2 (see scene_understanding.core.prompting).
    # "rampp_full" = RAM++ on the full image → dynamic GDINO query (no per-region RAM++ crops).
    query_builder_mode: Literal["inherit", "static", "rampp_full", "rampp_region_crops"] = "rampp_full"
    # Post-segmentation labelling: "gdino_only" | "bbox_florence" (package labeller) | "mask_chain" (legacy _label_mask).
    post_sam_label_mode: Literal["gdino_only", "bbox_florence", "mask_chain"] = "mask_chain"
    target_size: Tuple[int, int] = (512, 512)
    save_depth_visualizations: bool = True
    save_depth_16bit: bool = False
    # When True, load depth/{stem}_depth_metric.npy if it exists (same resolution) instead of re-inferring.
    reuse_cached_depth: bool = True
    # When True, log per-stage wall times in process_image (also set env CITV_PROFILE=1 to force on).
    scene_pipeline_profile: bool = False
    # When True and Grounding DINO already returned a specific class (not "object"), skip Florence-2 and
    # RAM++ per-mask inference for that mask (large speedup). Region/fake-amg paths still run secondaries.
    mask_label_skip_secondary_when_gdino_specific: bool = True

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
    depth_mask_matching_modes: List[str] = field(default_factory=lambda: ["A"])

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
    # depth_erosion_comparison: if True, compute depth_stats twice per object (slower).
    depth_erosion_comparison: bool = False

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
    # grounded_sam2_fallback_to_amg: removed — AMG is no longer part of the pipeline.
    # -------------------------------------------------------------------------
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_thresh: float = 0.15   # lowered to detect small/weak objects
    grounding_dino_text_thresh: float = 0.15  # lowered to match
    grounding_dino_text_query: str = (
        "person. animal. vehicle. furniture. appliance. food. clothing. "
        "container. tool. building. plant. electronics. object."
    )
    grounded_sam2_fallback_to_amg: bool = False  # ignored; AMG removed
    run_both_segmentors: bool = False  # ignored; AMG removed
    run_both_segmentors_iou_dedup: float = 0.7  # ignored
    # Track-separated SAM2 exports
    save_track_grounded_sam2: bool = True
    save_track_amg: bool = False
    save_track_combined: bool = False
    track_dir_grounded_sam2: str = "grounded_sam2"
    track_dir_amg: str = "amg"
    track_dir_combined: str = "combined"
    # Prompt bundle exports for external caption/eval pipelines
    export_caption_prompt_bundle: bool = True
    export_track_comparison_prompt: bool = True
    # Hybrid + standalone caption exports (files-only)
    export_hybrid_captions: bool = True
    export_florence_object_captions: bool = True
    export_fusion_scene_captions: bool = True
    caption_files_only: bool = True
    caption_max_objects_per_track: int = 64

    # -------------------------------------------------------------------------
    # Relation enhancement (Fix 5.6)
    #
    # florence2_relation_enabled: use Florence-2 to predict semantic relations
    #   between each overlapping object pair (O(n²) generate calls — default off; spatial scaffold only).
    # psgtr_enabled: load PSGTR for panoptic scene graph relations.
    # psgtr_checkpoint: path to PSGTR checkpoint (download separately).
    # univrd_enabled: load UniVRD for visual relationship detection.
    # univrd_checkpoint: path to UniVRD checkpoint.
    # relation_min_mask_overlap: minimum pixel IoU between two object masks
    #   before we attempt semantic relation prediction (avoids running VQA on
    #   every pair — most pairs have no spatial contact).
    # relation_bbox_touch_margin_px: axis-aligned bbox gap (px) allowed before
    #   skipping Florence entirely; if expanded boxes do not intersect, the pair
    #   cannot touch → no mask IoU / Florence call (major speedup).
    # -------------------------------------------------------------------------
    florence2_relation_enabled: bool = False
    psgtr_enabled: bool = False           # set True once checkpoint is available
    psgtr_checkpoint: Optional[str] = None
    univrd_enabled: bool = False          # set True once checkpoint is available
    univrd_checkpoint: Optional[str] = None
    relation_min_mask_overlap: float = 0.05  # stricter mask IoU before Florence (fewer calls)
    relation_bbox_touch_margin_px: int = 2  # bbox proximity gate before any mask work / Florence

    # -------------------------------------------------------------------------
    # Region-aware scene partitioning (depth K-means + connected components)
    #
    # regions_enabled: master switch; when True, writes per-track regions.json/png
    #   and adds region_id / region fields to objects and scene.json.
    # regions_k: K-means clusters on depth before CC split.
    # regions_min_region_px: drop smaller connected components.
    # regions_blur_sigma: Gaussian blur on depth before clustering (0 = off).
    # regions_seed: RNG seed for reproducible K-means init.
    # regions_use_hardlink_for_track_copies: hardlink depth/*.png into each track dir.
    # depth_sigma_clip_scope: "mask" (default) | "region" for outlier rejection stats.
    # regions_rampp_crops_enabled: RAM++ on region crops for GDINO query (expensive; default off).
    # region_relation_mode: "all" | "intra_region_only" for Pix2SG scaffold + Florence pairs.
    # append_region_layer_relations: add coarse region–region edges into relations.json.
    # regions_reject_implausible_labels: drop labels failing depth–class priors (strict).
    # -------------------------------------------------------------------------
    regions_enabled: bool = True
    regions_k: int = 4
    regions_min_region_px: int = 500
    regions_blur_sigma: float = 0.0
    regions_seed: int = 42
    regions_use_hardlink_for_track_copies: bool = True
    depth_sigma_clip_scope: str = "mask"  # "mask" | "region"
    regions_rampp_crops_enabled: bool = False
    region_relation_mode: str = "all"  # "all" | "intra_region_only"
    append_region_layer_relations: bool = True
    regions_reject_implausible_labels: bool = True

    # -------------------------------------------------------------------------
    # Region relation thresholds
    # -------------------------------------------------------------------------
    region_relation_depth_delta_m: float = 0.75       # min depth diff (m) for in_front_of / behind
    region_relation_iou_overlap: float = 0.05          # min IoU to emit "overlapping" edge
    region_relation_dilate_px: int = 2                 # bbox dilation for adjacency fallback
    region_relation_min_centroid_sep_px: int = 12      # min centroid gap for left_of/right_of/above/below

    # Phase 6/7: mask-based adjacency graph parameters
    # region_adjacency_min_border_px: minimum shared mask pixels after dilation for two
    #   regions to be considered adjacent. Lower values catch thin region borders at high
    #   resolution; raise if spurious adjacency edges appear on noisy depth maps.
    # region_adjacency_dilation_px: mask dilation radius (pixels) used when computing
    #   shared borders in both _build_region_adjacency_graph and _mask_adjacent. Should
    #   be >= region_relation_dilate_px for consistent adjacency semantics.
    region_adjacency_min_border_px: int = 10
    region_adjacency_dilation_px: int = 3

    # -------------------------------------------------------------------------
    # Visualisation map controls
    # -------------------------------------------------------------------------
    map_max_rel_per_object: int = 6                    # max relation arrows per object node
    map_max_rel_per_region: int = 4                    # max relation arrows per region node
    map_enable_split_views: bool = True                # also save objects-only and regions-only maps

    # -------------------------------------------------------------------------
    # Path hypotheses (single-image, not temporal tracking)
    #
    # Exports per-stem path hypotheses (region/object/mask) and overlay images.
    # Descriptions can be generated locally (template) or via OpenRouter.
    # -------------------------------------------------------------------------
    export_path_hypotheses: bool = True
    path_enable_region: bool = True
    path_enable_object: bool = True
    path_enable_mask: bool = True
    path_top_k_per_pair: int = 3
    path_max_candidates: int = 500
    path_min_confidence: float = 0.55
    path_invalid_pixel_ratio_max: float = 0.05
    path_max_turn_deg: float = 70.0
    path_max_depth_step_m: float = 2.0
    # tapered stroke styling (px)
    path_stroke_start_width_px: int = 8
    path_stroke_end_width_px: int = 2
    path_stroke_alpha_start: float = 0.95
    path_stroke_alpha_end: float = 0.35

    # -------------------------------------------------------------------------
    # Per-path individual PNG exports
    # -------------------------------------------------------------------------
    path_export_individual_images: bool = True
    path_individual_top_k_per_level: int = 10
    # Background choice for per-path PNGs:
    # - "transparent_mask": RGBA with alpha only (no scene clutter)
    path_individual_background: str = "transparent_mask"
    path_direction_text: bool = True
    path_label_text: bool = True
    path_text_max_chars: int = 46
    path_text_scale: float = 0.45

    # -------------------------------------------------------------------------
    # Context composites (top-N paths on scene context)
    # -------------------------------------------------------------------------
    path_export_context_composites: bool = True
    path_context_top_k: int = 5
    # Triplet context exports: render all accepted paths in batches of N
    # with short textual cues: start entity, target entity, and traversed regions.
    # Example: 39 paths with batch_size=3 -> 13 images.
    path_export_triplet_context_composites: bool = True
    path_context_triplet_size: int = 3

    # -------------------------------------------------------------------------
    # Trajectory visualization v2 (Phase 1): new files under path_v2_output_subdir
    # Black ink, thick→thin taper, last-segment direction arrow, optional geodesic.
    # -------------------------------------------------------------------------
    path_v2_visualization_enabled: bool = True
    path_v2_output_subdir: str = "trajectory_viz_v2"
    path_v2_ink_bgr: Tuple[int, int, int] = (15, 15, 15)
    path_v2_prefer_geodesic: bool = True
    # Which paths to draw: "all_valid" = full paths list; "all_recommended" = non-suppressed only.
    path_v2_draw_scope: Literal["all_valid", "all_recommended"] = "all_valid"
    path_v2_max_paths: int = 500
    path_v2_resample_points: int = 96
    path_v2_arrow_thickness_px: int = 2
    path_v2_arrow_tip_length: float = 0.22
    path_v2_context_max_boxes: int = 50

    # -------------------------------------------------------------------------
    # Trajectory visualization v2 (Phase 2): optional extra exports
    # -------------------------------------------------------------------------
    path_v2_export_topk_summary: bool = True
    path_v2_topk: int = 8
    path_v2_export_geodesic_compare: bool = True
    path_v2_export_alternates_faint: bool = True
    # Legacy polyline (polyline_2d) when comparing against geodesic on the same canvas.
    path_v2_legacy_ink_bgr: Tuple[int, int, int] = (160, 160, 160)
    path_v2_legacy_alpha_scale: float = 0.42
    # Geodesic alternate routes drawn under the primary resolved stroke.
    path_v2_alternate_ink_bgr: Tuple[int, int, int] = (120, 120, 120)
    path_v2_alternate_alpha_scale: float = 0.36
    # Halo under primary ink for readability on busy backgrounds.
    path_v2_outline_enabled: bool = False
    path_v2_outline_bgr: Tuple[int, int, int] = (250, 250, 250)
    path_v2_outline_pad_px: int = 3

    # -------------------------------------------------------------------------
    # Trajectory visualization v2 (Phase 3): ranking / pipeline underlays
    # Uses paths_sorted (recommended-ranked) when passed from path export.
    # -------------------------------------------------------------------------
    path_v3_visualization_enabled: bool = True
    path_v3_export_underlay_stack: bool = True
    path_v3_export_underlay_stack_context: bool = True
    path_v3_export_suppressed_residual: bool = True
    # Widest layer: polyline_2d only (raw route geometry) for capped candidate set.
    path_v3_all_polyline_2d_only: bool = True
    path_v3_all_alpha_scale: float = 0.12
    path_v3_all_color_bgr: Tuple[int, int, int] = (200, 200, 200)
    # Middle layer: resolved stroke (geodesic when available) for recommended paths.
    path_v3_recommended_alpha_scale: float = 0.30
    path_v3_recommended_color_bgr: Tuple[int, int, int] = (110, 110, 110)
    # Top of stack: strongest ranked paths (same ink as Phase 1 primary).
    path_v3_final_top: int = 12
    path_v3_suppressed_alpha_scale: float = 0.22
    path_v3_suppressed_color_bgr: Tuple[int, int, int] = (80, 80, 220)  # faint red-violet in BGR

    # Descriptions
    path_export_descriptions: bool = True
    path_description_backend: str = "local"  # "local" | "openrouter"
    path_description_model: str = "qwen/qwen-2.5-7b-instruct"
    path_description_max_tokens: int = 320
    path_description_temperature: float = 0.2
    openrouter_api_base: str = "https://openrouter.ai/api/v1"

    # -------------------------------------------------------------------------
    # Hybrid semantic enrichment + animation fusion
    # -------------------------------------------------------------------------
    path_semantic_enabled: bool = True
    # Optional embedding model name for future pluggable backends.
    # Current implementation uses deterministic lexical embedding fallback.
    path_semantic_embedding_model: str = "lexical-fallback-v1"
    path_semantic_llm_enabled: bool = True
    path_semantic_llm_top_k: int = 10

    # Hybrid score weights (must be non-negative; normalized at runtime)
    path_score_weight_geometric: float = 0.40
    path_score_weight_semantic: float = 0.30
    path_score_weight_relation: float = 0.20
    path_score_weight_action_fit: float = 0.10

    # Stage-wise visual exports
    path_export_stage_images: bool = True
    path_stage_levels: List[str] = field(default_factory=lambda: ["global", "region", "object", "mask"])

    # Animation fusion controls
    path_animation_enabled: bool = True
    path_animation_fps: int = 24
    path_speed_walk_px_s: float = 80.0
    path_speed_run_px_s: float = 150.0
    path_duration_idle_s: float = 0.5
    path_duration_jump_s: float = 0.6

    # Pair proposal + image-structure adherence
    path_pair_proposal_enabled: bool = True
    path_pair_top_k_targets: int = 4
    path_pair_allow_static_static: bool = True
    path_semantic_hard_filter_enabled: bool = True
    path_semantic_max_far_background_ratio: float = 0.40
    path_semantic_max_obstacle_ratio: float = 0.35
    path_semantic_min_walkable_ratio: float = 0.20

    # Image-constrained route refinement
    path_use_image_cost_refinement: bool = True
    path_cost_weight_edges: float = 0.35
    path_cost_weight_obstacle: float = 0.35
    path_cost_weight_region_prior: float = 0.15
    path_cost_weight_centering: float = 0.15
    path_refine_num_points: int = 96

    # Extended stage exports and diagnostics
    path_export_pair_proposals_json: bool = True
    path_export_diagnostics_json: bool = True

    # -------------------------------------------------------------------------
    # Motion contracts (phase 1): trajectory_hypotheses.json +
    # insertion_path_ensembles.json alongside path_hypotheses.json
    # -------------------------------------------------------------------------
    export_motion_contract_json: bool = True
    trajectory_hypotheses_max_subjects: int = 8
    trajectory_instant_step_px: float = 6.0
    trajectory_instant_dt_s: float = 0.04
    trajectory_hypotheses_include_all_objects: bool = False
    motion_contract_default_footprint_m: float = 0.45
    motion_contract_default_clearance_m: float = 0.15
    trajectory_use_depth_heading: bool = True
    trajectory_depth_heading_blend: float = 0.55
    trajectory_depth_heading_window: int = 9
    path_export_traversability_speed: bool = True
    path_use_traversability_geodesic: bool = True
    path_geodesic_replace_astar: bool = False
    path_geodesic_k_alt: int = 2
    path_geodesic_edge_penalty: float = 0.35
    path_motion_contract_overlay: bool = True
    path_motion_contract_overlay_max_paths: int = 24
    path_motion_contract_legacy_line_px: int = 2
    path_motion_contract_geodesic_line_px: int = 3
    trav_weight_image_edge: float = 0.25
    trav_weight_depth_flatness: float = 0.55
    trav_weight_image_smooth: float = 0.45
    trav_depth_grad_sigma_m: float = 0.35
    trav_speed_floor: float = 0.06
    path_export_fields_explainer: bool = True
    path_fields_explainer_panel_h: int = 380
    path_fields_explainer_max_paths: int = 4

    # -------------------------------------------------------------------------
    # Hierarchy edge controls
    # -------------------------------------------------------------------------
    hierarchy_enable_region_region_edges: bool = False      # add region-in-region containment
    hierarchy_region_region_containment_min: float = 0.97   # min ratio for region-region edges

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

