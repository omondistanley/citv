from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal


@dataclass
class PreprocessConfig:
    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    device: str = "mps"
    # When False (default), Florence-2 and RAM++ stay in memory after each image so
    # folder runs do not reload them every frame. Set True to free VRAM between images.
    # Phase B.3: flipped to False — the reload ping-pong dominated wall time on
    # folder runs (~8-15 s per image just reloading Florence). Operators who need
    # the old eager-unload behaviour on tight-memory machines can still opt back
    # in by setting this to True.
    unload_labellers_after_each_image: bool = False
    # Package pipeline is staged-only. Legacy mode requests are normalized to staged.
    scene_pipeline_mode: Literal["staged"] = "staged"
    # Prompting: how Grounding DINO text query is built before SAM2 (see scene_understanding.core.prompting).
    # Staged pipeline: runs RAM++ tag_image (or region crops) before GroundedSAM2 when rampp_enabled.
    # "inherit" = refresh GDINO text from RAM++ when rampp_enabled (default staged behavior).
    # "rampp_full" / "rampp_region_crops" = same, with region crops when regions_rampp_crops_enabled or mode forces crops.
    # "static" = use grounding_dino_text_query only.
    query_builder_mode: Literal["inherit", "static", "rampp_full", "rampp_region_crops"] = "inherit"
    # Post-segmentation labelling: "gdino_only" | "bbox_florence" (package labeller).
    # Legacy-only mask_chain has been removed with monolith decommissioning.
    post_sam_label_mode: Literal["gdino_only", "bbox_florence"] = "bbox_florence"
    target_size: Tuple[int, int] = (512, 512)
    save_depth_visualizations: bool = True
    save_depth_16bit: bool = False
    # When True, load depth/{stem}_depth_metric.npy if it exists (same resolution) instead of re-inferring.
    reuse_cached_depth: bool = True
    # When True, log per-stage wall times in process_image (also set env CITV_PROFILE=1 to force on).
    scene_pipeline_profile: bool = False
    # Directory for the append-only timing log (logs/timing_log.md + .csv).
    # The logger also embeds a `_timing` dict into each *_scene.json for per-image QA.
    timing_log_dir: str = "logs"
    # Opt-in global model precision for fp16-capable wrappers. "fp32" is the safe
    # default until Phase E QA sign-off; switching to "fp16" still requires each
    # wrapper to actually support it.
    models_dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    # Opt-in torch.compile(reduce-overhead) for SAM2 image encoder + Depth V2.
    torch_compile_enabled: bool = False
    # Cross-image parallelism for folder runs via multiprocessing.Pool. Default 1
    # keeps today's single-process behaviour; raise cautiously on 32 GB+ machines.
    pipeline_max_workers: int = 1
    # Phase H — optional MLX-VLM backend for Florence-2 on Apple Silicon.
    # ``hf`` keeps the HuggingFace path (default). ``mlx`` swaps in the
    # native MLX runtime when the ``mlx_vlm`` package is installed (see
    # scene_understanding/labeling/florence2_mlx.py for details).
    florence2_backend: Literal["hf", "mlx"] = "hf"
    # Isolated runtime for MLX Florence-2. This interpreter can carry
    # ``mlx-vlm`` + transformers 5.x without affecting the main pipeline's
    # HF dependency stack.
    florence2_mlx_python: str = ".venv/bin/python3.14"
    florence2_mlx_start_timeout_s: float = 180.0
    florence2_mlx_request_timeout_s: float = 180.0
    # Emit BOTH backends' captions during validation runs so the new MLX
    # outputs can be diffed against the HF baseline. When False (default),
    # only ``florence2_backend`` is executed.
    florence2_emit_both_backends: bool = False

    # Phase G — opt-in async stage orchestration. When True, independent
    # stages (depth, full-image caption, GDINO+SAM2) are dispatched to a
    # small thread pool so the Python glue runs in parallel with the MPS
    # queue. Default False until QA sign-off confirms byte-identical
    # outputs vs. the sequential flow.
    # Phase F — PNG compression level for *intermediate QA* PNGs (depth viz,
    # relation overlays, semantic cost heatmaps, provenance maps, cost layer
    # exports). 0 = fastest/largest, 9 = slowest/smallest, OpenCV default = 3.
    # Final deliverables keep the default.
    png_intermediate_compression: int = 1

    # Memory-pressure guard for low-VRAM systems (e.g. Apple M2 with 8 GB
    # unified memory). When True, the pipeline releases the depth model and
    # the Grounding-DINO + SAM2 wrapper *after* the segmentation stage so
    # Florence-2 + RAM++ don't have to co-exist with them during mask
    # enrichment. Prevents the 8.6 GB → 11.3 GB peak-footprint swap-thrash
    # we saw on 8 GB M2 runs (diagnosed via `sample`: >66% of wall-clock was
    # `MTLCommandBuffer waitUntilCompleted` during MPS→CPU copies caused by
    # paging). Re-inits happen lazily on the next image. Default True is
    # safe for 8 GB systems; set False on >=24 GB if you want to trade some
    # memory for the reload-skip.
    release_heavy_models_after_segmentation: bool = True
    # Region-enrichment batching knobs. ``_enrich_region_labels_from_masks``
    # previously ran 1 Florence-2 + 4 RAM++ forwards per region sequentially.
    # We now batch those forwards across all regions, chunked to respect the
    # 8 GB unified memory budget on M2. Florence-2 batched ``generate`` with
    # greedy decoding holds a per-sequence KV cache in fp16; on an 8 GB
    # unified-memory device we saw MPS swap-thrash at B=8, so the safe
    # default is B=4 (peak ~4-5 GB on top of the ~1.6 GB of weights) with
    # ``torch.mps.empty_cache()`` between chunks. RAM++ runs on CPU where
    # batches of 8-16 give the best BLAS occupancy on an M2 without
    # pressuring the unified memory shared with MPS tensors. Set to 1 to
    # disable batching (pure legacy behaviour). Increase only if you have
    # >= 16 GB unified memory.
    region_enrich_florence_batch: int = 4
    region_enrich_rampp_batch: int = 8
    # When True, the region-enrichment stage runs Florence-2 (MPS) and
    # RAM++ (CPU) CONCURRENTLY via a background thread so the two model
    # engines overlap instead of running back-to-back. This is only a
    # win on machines with enough unified memory to keep both models
    # resident and active without paging — on an 8 GB M2 we measured
    # catastrophic swap-thrash (RAM++ ran ~20× slower than serial)
    # because macOS compacts the Florence-2 MPS KV cache to disk while
    # RAM++ BLAS allocates fresh activations. Keep OFF on 8 GB, enable
    # only on ≥ 16 GB unified-memory devices or on discrete-memory
    # systems where MPS and CPU BLAS pools don't fight.
    region_enrich_parallel_labellers: bool = False
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
    sam2_amg_max_image_side: int = 1024
    save_per_object_masks: bool = True
    save_masked_depth_npy: bool = False   # ~48 MB per object per mode — leave False
    sam2_amg_crop_n_layers: int = 1
    sam2_amg_crop_overlap_ratio: float = 0.341
    sam2_amg_crop_n_points_downscale_factor: int = 2
    sam2_amg_min_mask_region_area: int = 100   # lowered to detect small objects
    sam2_amg_use_m2m: bool = True
    sam2_amg_box_nms_thresh: float = 0.7
    # Strict production profile: keep dual segmentor but bias AMG toward whole-object supplements
    # instead of tiny part fragments that GroundedSAM2 already covers.
    segmentation_production_strict: bool = False
    segmentation_production_amg_pred_iou_thresh: float = 0.90
    segmentation_production_amg_stability_score_thresh: float = 0.95
    segmentation_production_amg_min_mask_region_area: int = 280
    segmentation_production_amg_part_containment_thresh: float = 0.88
    segmentation_production_amg_part_min_area_ratio_vs_grounded: float = 0.25

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
    # Phase B.1: Florence-2 performance knobs.
    #   florence2_dtype        — "", "fp16", "bf16", or "fp32". Empty string defers
    #                            to Florence2Wrapper's device-aware default (fp16
    #                            on MPS/CUDA, fp32 on CPU). Keep empty for the
    #                            plan's output-preserving rollout.
    #   florence2_attn_impl    — "sdpa" (default), "eager" (safe fallback), or
    #                            "flash_attention_2" (CUDA-only). Wrapper already
    #                            falls back to eager when the chosen impl refuses
    #                            to load, so "sdpa" is safe to leave on.
    #   florence2_use_cache    — enable KV cache during generate. Wrapper
    #                            auto-disables once per process if the installed
    #                            transformers/Florence-2 combo raises a known
    #                            EncoderDecoderCache incompatibility.
    #   florence2_batch_size   — max images per batched generate call. 0 disables
    #                            the batched path and forces sequential calls for
    #                            QA parity runs.
    florence2_dtype: str = ""
    florence2_attn_impl: str = "sdpa"
    florence2_use_cache: bool = True
    florence2_batch_size: int = 8
    # When True, Florence-2 runs an additional <OD> task if the
    # caption-derived noun is still generic ("object"). Disable for speed.
    florence2_od_fallback_enabled: bool = False

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
    # run_both_segmentors: when True, run SAM2 AMG after GroundedSAM2 and merge
    #   non-duplicate masks (see docs/SEGMENTATION.md). Matches monolith-style coverage.
    # grounded_sam2_fallback_to_amg: when True, run AMG if GDINO yields no detections.
    # run_both_segmentors_iou_dedup: IoU threshold; AMG masks overlapping a GroundedSAM2
    #   mask above this are dropped.
    # -------------------------------------------------------------------------
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_thresh: float = 0.15   # lowered to detect small/weak objects
    grounding_dino_text_thresh: float = 0.15  # lowered to match
    grounding_dino_text_query: str = (
        "person. animal. vehicle. furniture. appliance. food. clothing. "
        "container. tool. building. plant. electronics. object."
    )
    grounded_sam2_fallback_to_amg: bool = False
    run_both_segmentors: bool = False
    run_both_segmentors_iou_dedup: float = 0.7
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
    depth_sigma_clip_scope: str = "mask"  # "mask" | "region" | "global"
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
    # If > 0, drop adjacency edges whose depth_delta_m exceeds this (meters).
    region_adjacency_max_depth_delta_m: float = 0.0

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
    # Per-level caps (0 = use path_max_candidates as cap for that tier). Prevents region fill from starving mask/cross.
    path_max_region_paths: int = 180
    path_max_object_paths: int = 220
    path_max_mask_paths: int = 120
    path_max_cross_paths: int = 200
    # Pre-dedupe snapshot for QA / calibration (same stem as path_hypotheses.json).
    export_path_hypotheses_candidates: bool = True
    # When True, run full ``_enrich_path_hypotheses`` on candidates (expensive for large N).
    path_candidates_include_enrichment: bool = False
    # Hausdorff / near-duplicate cap **per dedupe group** (see path_hypotheses_paths._pair_key).
    path_dedupe_max_per_pair: int = 5
    path_dedupe_frechet_thresh_px: float = 18.0
    # Support-aware routing: relax speed gate on ``support_mask`` pixels to reduce stair/CC splits.
    path_routing_relax_on_support: bool = True
    path_routing_support_speed_floor_mult: float = 0.72
    path_routing_support_close_px: int = 2
    # Extra bridge layer (lower floor) used only to recover cross-CC portal pairs into one FMM.
    path_routing_bridge_speed_floor_mult: float = 0.55
    path_cc_bridge_portals_enabled: bool = True
    # Region FMM beyond legacy ``fmm_ribbon_count == 0`` fallback.
    path_region_fmm_policy: Literal["legacy", "always_extra", "when_sparse"] = "when_sparse"
    path_region_fmm_min_ribbons: int = 0
    path_region_fmm_max_extra: int = 24
    path_region_fmm_max_labels_sampled: int = 12
    # Object anchors → region centroid / bottom-centroid goals (Pix2SG-assisted scoring).
    path_object_region_enabled: bool = True
    path_object_region_goal_top_k: int = 6
    # Optional explicit composite polylines when multi-segment CC routing is implemented.
    path_cc_chain_enabled: bool = False
    path_cc_chain_max_hops: int = 4
    path_export_feasible_routing_png: bool = False
    # Pre-family-cap geometric snapshot for auditing stratified caps / family trim.
    path_export_full_hypotheses: bool = True
    # Samples per path for along-polyline cost/depth/region trace in semantic scoring.
    path_polyline_semantic_samples: int = 40
    # Fuse idle/walk/run/careful soft fields from cost, regions, depth, traversability; export npy/png/json.
    path_export_navigation_zones: bool = True
    # When false, path semantic blend uses navigation_zones + structured relations (no English prototype strings).
    path_semantic_use_lexical_prototypes: bool = False
    path_min_confidence: float = 0.55
    path_invalid_pixel_ratio_max: float = 0.025
    path_max_turn_deg: float = 130.0
    path_max_depth_step_m: float = 2.0
    # Region/object polyline smoothing (organic routing).
    path_polyline_smoothing_iterations: int = 2
    path_polyline_smoothing_prev_weight: float = 0.2
    path_polyline_smoothing_self_weight: float = 0.6
    path_polyline_smoothing_next_weight: float = 0.2
    # Walkable mask: (region>0) ∧ ¬obstacles ∧ optional speed / erosion — used for validation + smoothing.
    path_walkable_use_obstacles: bool = True
    path_walkable_min_speed: float = 0.0
    # If in (0,1], keep pixels with speed >= this quantile of speeds on the pre-threshold walkable set.
    path_walkable_speed_quantile: float = 0.0
    path_walkable_erode_px: int = 0
    # Fold region affordances (semantic_layer) into path_walkable: drop far_background / obstacle labels.
    path_walkable_semantic_fusion_enabled: bool = True
    path_walkable_semantic_exclude_far_background: bool = True
    path_walkable_semantic_exclude_obstacle_regions: bool = True
    path_export_walkable_debug_png: bool = False
    # Export path_walkable.npy + path_walkable_mask.json next to path hypotheses (for QA / verify_path_rollout).
    path_export_walkable_npy: bool = True
    # Object/mask anchors: use medial-interior (distance-transform ridge) instead of nearest pixel to centroid.
    path_anchor_use_medial_interior: bool = True
    # Quadratic Bézier corner rounding after coarse / Laplacian smoothing (reduces sharp kinks).
    path_bezier_enabled: bool = True
    path_bezier_corner_deg_min: float = 22.0
    path_bezier_handle_frac: float = 0.32
    path_bezier_max_handle_px: float = 72.0
    path_bezier_samples_per_corner: int = 14
    # Snap region centroids and object endpoints toward walkable / mask interiors.
    path_anchor_snap_centroids: bool = True
    path_anchor_snap_object_endpoints: bool = True
    # Drop cross-family paths whose geometry fails _polyline_is_valid (recommended).
    path_cross_suppress_invalid_geometry: bool = True
    # Reject inter-scope polylines whose endpoints are closer than this (px) unless contour/intra.
    path_inter_min_span_px: float = 4.0
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
    path_context_top_k: int = 12
    # Triplet context exports: render all accepted paths in batches of N
    # with short textual cues: start entity, target entity, and traversed regions.
    # Example: 39 paths with batch_size=3 -> 13 images.
    path_export_triplet_context_composites: bool = True
    path_context_triplet_size: int = 3

    # -------------------------------------------------------------------------
    # Trajectory atlas (trajs-upt): ranked line-only PNG panels + manifest + MD
    # -------------------------------------------------------------------------
    path_atlas_enabled: bool = True
    path_atlas_total: int = 50
    path_atlas_panel_size: int = 10
    path_atlas_ranked_panels: int = 5
    # Rank pool: paths_sorted (recommended-first global rank) vs paths_recommended only.
    path_atlas_rank_source: Literal["paths_sorted", "paths_recommended"] = "paths_sorted"
    # Full-compliance mode: atlas/batches/animation cover all accepted paths.
    path_contract_compliance_mode: bool = True
    path_atlas_background_bgr: Tuple[int, int, int] = (26, 26, 26)
    path_atlas_prefer_geodesic: bool = True
    path_atlas_min_hue_separation: float = 0.07
    path_atlas_endpoint_dots: bool = False
    path_atlas_overlay_enabled: bool = True
    path_atlas_overlay_include_context: bool = True
    path_atlas_export_paths_trajectories_overlay: bool = True
    path_atlas_export_trajectory_line_only: bool = True
    path_atlas_trajectory_stroke_start_px: int = 5
    path_atlas_trajectory_stroke_end_px: int = 2

    # Scene batches: path_trajectories_batch_*.png (10–15 paths + linked trajectories per image).
    path_scene_trajectory_batches_enabled: bool = True
    path_scene_trajectory_batch_size: int = 10
    path_scene_batch_include_context: bool = False
    path_scene_batch_include_context_debug: bool = True

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
    path_scene_context_enabled: bool = True
    path_semantic_physical_energy_enabled: bool = True
    path_cross_family_enabled: bool = True
    path_cross_family_top_k: int = 2
    path_family_max_candidates: int = 12
    # Optional embedding model name for future pluggable backends.
    # Current implementation uses deterministic lexical embedding fallback.
    path_semantic_embedding_model: str = "lexical-fallback-v1"
    path_semantic_llm_enabled: bool = True
    path_semantic_llm_top_k: int = 10

    # Hybrid score weights (must be non-negative; normalized at runtime).
    # Raise path_score_weight_geometric to prioritize traversability / image alignment in hybrid_overall.
    path_score_weight_geometric: float = 0.40
    path_score_weight_semantic: float = 0.30
    path_score_weight_relation: float = 0.20
    path_score_weight_action_fit: float = 0.10
    path_semantic_affordance_weight: float = 0.65
    path_semantic_lexical_weight: float = 0.35
    # Affordance ratios for hard filter / naturalness: "polyline" = sample region labels under the route;
    # "regions_traversed" = legacy coarse count over region id list only (can disagree with geometry).
    path_semantic_affordance_trace_mode: Literal["polyline", "regions_traversed"] = "polyline"
    path_semantic_void_label_affordance: str = "void"
    path_semantic_affordances_traversable: Tuple[str, ...] = ("walkable", "interaction_zone")
    # Blend sim_rel with navigation zone channels (when not using lexical prototypes).
    path_semantic_score_zone_fit_weight: float = 0.55
    path_semantic_score_relation_weight: float = 0.45
    path_semantic_zone_channel_walk: float = 0.32
    path_semantic_zone_channel_run: float = 0.32
    path_semantic_zone_channel_idle: float = 0.20
    path_semantic_zone_channel_careful_inv: float = 0.16
    # After polyline mean-cost alignment: prior * score + align * (1 - mean_cost).
    path_semantic_score_polyline_prior_weight: float = 0.88
    path_semantic_score_polyline_cost_align_weight: float = 0.12
    path_semantic_score_high_cost_penalty: float = 0.04
    # Along-polyline cost trace: fraction of samples above this count as "high cost".
    path_polyline_high_cost_threshold: float = 0.65
    # Naturalness diagnostic (not a hard gate unless extended later).
    path_naturalness_weight_far_background: float = 0.35
    path_naturalness_weight_obstacle: float = 0.35
    path_naturalness_weight_relation_penalty: float = 0.15
    path_naturalness_weight_walk_deficit: float = 0.15
    # Extra energy terms in semantic-physical bookkeeping.
    path_energy_weight_depth: float = 0.15
    path_energy_weight_uncertainty: float = 0.10
    # Corridor sampling along polyline segments (post-sequence reject).
    path_corridor_samples_per_seg: int = 30

    # Scene-calibrated policy (avoid fixed thresholds in decision logic).
    path_calibration_enabled: bool = False
    path_calibration_min_conf_quantile: float = 0.55
    path_calibration_min_hybrid_quantile: float = 0.15
    path_calibration_hybrid_min_conf_cap: float = 0.56
    path_calibration_semantic_valid_quantile: float = 0.30
    path_calibration_max_far_quantile: float = 0.85
    path_calibration_max_obs_quantile: float = 0.85
    path_calibration_min_walk_quantile: float = 0.20
    path_calibration_motion_primary_quantile: float = 0.60
    path_calibration_threshold_scale: float = 1.0
    path_motion_primary_threshold: float = 0.45
    path_semantic_validity_floor: float = 0.30

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

    # -------------------------------------------------------------------------
    # Animation QA renderer (dual-FPS verification exports)
    # -------------------------------------------------------------------------
    path_animation_qa_enabled: bool = True
    # Separate run modes are supported (e.g. [24] or [120] or [24, 120]).
    path_animation_qa_modes: List[int] = field(default_factory=lambda: [24])
    path_animation_qa_candidate_seconds: float = 0.8
    path_animation_qa_render_debug_overlays: bool = True
    path_animation_qa_max_candidates_per_panel: int = 10
    path_animation_qa_output_subdir_24: str = "animation_qa_24"
    path_animation_qa_output_subdir_120: str = "animation_qa_120"
    # Character follows the ranked path polyline (arc-length), not only the short rollout polyline.
    path_animation_qa_follow_path_geometry: bool = True
    # When motion is idle, keep character fixed on path start (arc parameter s=0) instead of rollout trail.
    path_animation_qa_idle_follow_path: bool = True
    # Full-screen delimiter slate between path/trajectory candidates in stitched panel videos.
    path_animation_qa_delimiter_seconds: float = 0.12
    # When sprite sheets are missing, draw a procedural bunny so motion QA is still visible.
    path_animation_qa_procedural_bunny_fallback: bool = True
    path_animation_qa_character_base_scale: float = 0.42
    path_animation_qa_depth_scale_span: float = 0.35

    # Deterministic depth-style mapping (shared across FPS modes).
    path_animation_depth_style_enabled: bool = True
    path_animation_depth_percentile_low: float = 5.0
    path_animation_depth_percentile_high: float = 95.0
    path_animation_depth_eps: float = 1e-6
    path_animation_thickness_min_px: int = 1
    path_animation_thickness_max_px: int = 7
    path_animation_thickness_gamma: float = 1.35
    path_animation_alpha_min: float = 0.28
    path_animation_alpha_max: float = 0.95
    path_animation_alpha_gamma: float = 1.10
    path_animation_style_beta_24: float = 0.35
    path_animation_style_beta_120: float = 0.70
    path_animation_trail_samples_24: int = 6
    path_animation_trail_samples_120: int = 20
    path_animation_trail_decay: float = 0.80
    path_animation_guardrail_fallback_thickness_px: int = 1
    path_animation_guardrail_fallback_alpha: float = 0.35

    # Sprite sheet asset registry used by QA renderer.
    path_animation_sprite_sheets: Dict[str, str] = field(default_factory=lambda: {
        "idle": "assets/spritesheets/motion_idle_spritesheet_2048x2048.png",
        "walk": "assets/spritesheets/motion_walk_spritesheet_2048x2048.png",
        "run": "assets/spritesheets/motion_run_spritesheet_2048x2048.png",
        "jump": "assets/spritesheets/motion_jump_spritesheet_2048x2048.png",
        "crawl": "assets/spritesheets/motion_crawl_spritesheet_2048x2048.png",
        "wave": "assets/spritesheets/motion_wave_spritesheet_2048x2048.png",
        "sit": "assets/spritesheets/motion_sit_spritesheet_2048x2048.png",
        "surprised": "assets/spritesheets/motion_surprised_spritesheet_2048x2048.png",
        "hop": "assets/spritesheets/motion_hop_spritesheet_2048x2048.png",
    })

    # Pair proposal + image-structure adherence
    path_pair_proposal_enabled: bool = True
    path_pair_top_k_targets: int = 4
    path_pair_allow_static_static: bool = True
    # 0 = disabled. Max centroid distance (px) between src/tgt object mask centroids.
    path_pair_max_centroid_distance_px: float = 0.0
    # 0 = disabled. Else max distance = fraction * hypot(image_width, image_height).
    path_pair_max_centroid_diag_frac: float = 0.72
    # When True, require at least one relation record mentioning both object ids (or names).
    path_pair_require_relation: bool = False
    # When False, skip lexical static-static fallback that fills proposals when empty.
    path_pair_lexical_fallback_enabled: bool = False
    # Reject region/object/cross paths when mean cost_map along samples exceeds this (0 = off).
    path_sequence_reject_mean_cost_above: float = 0.78
    # Reject when fraction of samples off path_walkable exceeds this (0 = off).
    path_sequence_reject_walkable_off_ratio_above: float = 0.10
    # Mask path exports (intra diagnostics vs interior motion).
    path_mask_export_contour: bool = True
    path_mask_export_axis: bool = False
    path_mask_export_interior: bool = True
    path_mask_interior_erode_iterations: int = 2
    path_mask_interior_kernel_px: int = 3
    path_mask_interior_min_component_px: int = 400
    # Endpoints: snap toward path_walkable within max px; optional strict gate.
    path_endpoint_walkable_snap_enabled: bool = True
    path_endpoint_walkable_gate_enabled: bool = True
    path_endpoint_snap_walkable_max_px: float = 16.0
    # Descriptions: omit misleading max-turn line for polylines with fewer than 3 vertices.
    path_descriptions_omit_turn_when_short_polyline: bool = True
    path_descriptions_densify_grounded_polyline: bool = False
    path_descriptions_densify_grounded_points: int = 24
    # path_map_all.png: also draw polyline_geodesic_2d; optionally only recommended paths.
    path_map_all_draw_geodesic: bool = False
    path_map_all_geodesic_bgr: Tuple[int, int, int] = (60, 255, 120)
    path_map_all_use_recommended_only: bool = False
    # Attach polyline_3d (u,v,z_m) sampled from metric depth per vertex when depth is available.
    path_export_polyline_3d: bool = True
    path_polyline_3d_invalid_depth_value: float = -1.0
    path_semantic_hard_filter_enabled: bool = True
    path_semantic_max_far_background_ratio: float = 0.42
    path_semantic_max_obstacle_ratio: float = 0.38
    path_semantic_min_walkable_ratio: float = 0.14

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
    trajectory_rollout_steps: int = 4
    trajectory_rollout_dt_s: float = 0.04
    trajectory_rollout_step_scale: float = 0.25
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
    # Motion primary score blend (scene/data tunable; normalized at runtime).
    path_motion_primary_weight_length_efficiency: float = 0.10
    path_motion_primary_weight_border_clearance: float = 0.35
    path_motion_primary_weight_obstacle_clearance: float = 0.30
    path_motion_primary_weight_depth_alignment: float = 0.25
    # Adaptive inter-path trajectory pacing coefficients.
    trajectory_adaptive_bias_base: float = 0.35
    trajectory_adaptive_bias_careful: float = 0.45
    trajectory_adaptive_bias_cost: float = 0.35
    trajectory_adaptive_bias_not_run: float = 0.25
    # Organic lateral variation along inter-path trajectories.
    trajectory_lateral_walk_weight: float = 0.60
    trajectory_lateral_run_weight: float = 0.40
    trajectory_lateral_careful_penalty: float = 0.55
    trajectory_lateral_amplitude_px: float = 0.55
    trajectory_lateral_phase_step: float = 0.85
    # Scale lateral wiggle to 0 at trajectory endpoints (reduces hanging off anchors).
    trajectory_lateral_endpoint_ramp_steps: int = 3

    # CPU benchmark policy
    cpu_benchmark_mode: bool = False
    cpu_benchmark_target_wall_s: float = 200.0
    cpu_benchmark_disable_heavy_visuals: bool = False

    # -------------------------------------------------------------------------
    # Hierarchy edge controls
    # -------------------------------------------------------------------------
    hierarchy_enable_region_region_edges: bool = False      # add region-in-region containment
    hierarchy_region_region_containment_min: float = 0.97   # min ratio for region-region edges

    # -------------------------------------------------------------------------
    # RAM++ labelling (replaces GRiT + YOLOv8-cls fallback)
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
    # Labeller backend:
    #   "rampp" => RAM++ (highest recall, slowest on CPU)
    #   "ram"   => RAM (faster, lower memory)
    #   "clip"  => CLIP zero-shot tags only (fastest fallback)
    rampp_backend: Literal["rampp", "ram", "clip"] = "rampp"
    rampp_checkpoint_path: Optional[str] = "checkpoints/ram_plus_swin_large_14m.pth"
    rampp_repo_path: Optional[str] = None
    rampp_image_size: int = 384
    rampp_vit: str = "swin_l"
    rampp_default_confidence: float = 0.70
    rampp_max_tags: int = 8
    # Phase B.2: opt-in MPS path for RAM++. When True and the pipeline device
    # is MPS, the wrapper loads fp16 weights, enforces contiguous tensors, and
    # runs with torch.autocast under a watermark guard. Turning this on should
    # be paired with PYTORCH_ENABLE_MPS_FALLBACK=1 so any un-implemented op
    # silently dispatches to CPU instead of crashing the run.
    rampp_prefer_mps: bool = False
    rampp_mps_dtype: Literal["fp16", "fp32"] = "fp16"
    rampp_mps_watermark_fraction: float = 0.85
    # CPU throughput optimization — dynamic int8 quantization for RAM/RAM++
    # linear layers. Ignored on MPS.
    rampp_quantize_dynamic_int8: bool = True
    # If RAM/RAM++ init is unavailable for the selected backend, use CLIP
    # zero-shot labels instead of disabling tagger entirely.
    rampp_clip_fallback_enabled: bool = True
    rampp_clip_candidate_labels: List[str] = field(default_factory=lambda: [
        "person", "chair", "table", "sofa", "bed", "cabinet", "shelf", "desk",
        "lamp", "door", "window", "wall", "floor", "ceiling", "plant", "tree",
        "bag", "box", "bottle", "cup", "plate", "book", "phone", "laptop",
        "monitor", "keyboard", "television", "car", "bus", "bicycle", "road",
        "building", "sky", "animal", "dog", "cat", "object",
    ])

    # -------------------------------------------------------------------------
    # Phase C — semantic cost map + FMM pathfinding (feature-flagged rollout)
    #
    # path_cost_semantic_enabled
    #     Master switch for the semantic cost map. When False (default) the
    #     pipeline continues to emit today's legacy A*/geometric-cost outputs
    #     exactly as before. When True the scene_understanding.pathing
    #     semantic cost map is computed and exported alongside legacy outputs;
    #     the downstream planner still chooses which path to promote based on
    #     ``path_search_algorithm``.
    # path_search_algorithm
    #     Which planner runs on the (legacy or semantic) cost map:
    #       "legacy_astar" — existing A*/Dijkstra on geometric cost (default).
    #       "fmm"          — Fast Marching Method backtrace on the semantic
    #                         cost map (requires path_cost_semantic_enabled).
    #       "hier_fmm"     — Region graph + per-region FMM refinement.
    #       "multiscale_fmm" — Coarse→fine FMM pyramid for large images.
    #       "theta_star"   — Any-angle post-smoothing on the FMM backtrace.
    # path_cost_emit_layers_for_qa
    #     When True, save each semantic cost map layer (geometry, VLM,
    #     self-calibration, scene-graph, agent physics, depth noise) as a
    #     standalone PNG + .npy under the path outputs, plus
    #     ``cost_provenance.png`` and ``cost_meta.json``. Essential during the
    #     Phase C validation window; disable once legacy-vs-FMM diffs sign off.
    # path_cost_agent_profile_file
    #     Path to agent.profile.yaml. The semantic cost map's agent-physics
    #     layer refuses to build if this file is missing, malformed, or has
    #     non-physical values. No magic defaults — operators must populate
    #     this file with measured agent dimensions.
    path_cost_semantic_enabled: bool = True
    path_search_algorithm: str = "legacy_astar"
    path_cost_emit_layers_for_qa: bool = False
    path_cost_agent_profile_file: str = "agent.profile.yaml"
    # Staged fused semantic cost (geometry + scene_graph + ontology captions + agent + depth noise).
    path_use_fused_semantic_cost: bool = True
    path_fused_semantic_cost_blend: float = 0.72
    path_cost_use_vlm_layer: bool = False
    path_cost_prompts_yaml: str = "cost_map_prompts.yaml"
    path_cost_ontology_caption_layer: bool = True
    path_cost_caption_layer_precision_cap: float = 0.22
    # Multi-channel traversability speeds (ground / fluid / aerial).
    path_multi_channel_traversability: bool = True
    # Optional FMM on coarse support grid (reduces image-plane diagonal shortcuts).
    path_fmm_on_ground_manifold: bool = False
    path_ground_manifold_grid_step_px: int = 14
    # Goal repair: snap vertical-structure targets to support under bbox.
    path_goal_ground_intercept_vertical: bool = True
    path_goal_vertical_aspect_thresh: float = 1.35
    # Pair gates for object–object ribbon paths (0 = disabled on numeric thresholds).
    path_pair_min_span_px: float = 18.0
    path_pair_max_depth_delta_m: float = 0.0
    path_pair_skip_facade_facade: bool = True
    path_pair_facade_skip_requires_vertical: bool = False
    # Bézier: re-snap vertices that leave feasible mask.
    path_bezier_snap_feasible_enabled: bool = True
    path_bezier_snap_feasible_max_px: float = 8.0
    # QA perf mode: uncapped panels, optional dedupe off, atlas auto-pagination.
    path_qa_perf_mode: bool = False
    path_atlas_auto_panel_count: bool = False
    # Manifest: mark MP4 artifacts failed when file smaller than this (bytes).
    artifact_manifest_mp4_min_bytes: int = 256
    # Extra manifold hypotheses (no new models): aerial approach + obstacle contours.
    path_emit_aerial_approach_hypotheses: bool = True
    path_max_aerial_hypotheses: int = 12
    path_emit_contour_hypotheses: bool = True
    path_max_contour_hypotheses: int = 8