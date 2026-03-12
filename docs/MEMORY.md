# CITV Project Memory

## Project Overview
Scene understanding pipeline at /home/stanleyomondi/citv/
Builds 3D scene graphs from images using multiple CV models.

## Key Files
- `scene_understanding.py` — main pipeline (all wrappers + SceneUnderstandingPipeline)
- `config.py` — PreprocessConfig dataclass (all pipeline config)
- `depth.py` — DepthEstimator + SceneTypeClassifier (CLIP) + DepthAnythingV2Backend
- `tools/calibrate_camera.py` — OpenCV checkerboard calibration → calibration.json

## Model Stack (current, post-fixes 5.1–5.7)
1. **CLIP** (`openai/clip-vit-base-patch32`) — scene type classifier; loads, classifies, unloads
2. **Depth Anything V2 Metric** — indoor (`Depth-Anything-V2-Metric-Indoor-Large-hf`) or outdoor variant; selected at runtime by CLIP; outputs true meters (no scaling needed)
3. **GroundingDINO v2** (`IDEA-Research/grounding-dino-base`) — object detection with text query
4. **SAM2** (`SAM2ImagePredictor`) — mask generation prompted by GDINO boxes; AMG as fallback
5. **Florence-2** (`microsoft/Florence-2-large`) — object labelling (`<OD>`) + relation prediction (`<CAPTION>`)
6. **GRiT** — kept as last-resort label fallback; YOLO as secondary fallback
7. **Pix2SG** — spatial scaffold + Florence-2 semantic relation enrichment

## Architecture: SceneUnderstandingPipeline.process_image()
Stage 0  Load calibration JSON if `camera_calibration_file` set; undistort image via cv2.undistort()
Stage 1  Camera intrinsics: priority = calibration file > explicit fx/fy > FOV estimate (60°)
Stage 2  CLIP classifies scene → depth variant selected → Depth Anything V2 Metric → metric_depth (meters, no scaling)
Stage 3  GroundedSAM2Wrapper: GDINO detects objects → SAM2 prompted per bbox → object-level masks
         Falls back to SAM2 AMG if GDINO fails or zero detections
         amg_entry["label"] and ["gdino_conf"] set from GDINO for mask entries
Stage 3b SAM2 post-hoc filters (Fix 5.5): stability≥0.85, pred_iou≥0.82, area≥1500px, area_fraction≤0.30
         matched_A_lookup = identity dict; Mode B: _match_mask_first() uses index identity
Stage 4  _label_mask(): priority = GDINO label > Florence-2 <OD> > GRiT + YOLO fallback
         Build objects_3d via _mask_depth_stats_and_3d() with Fix 5.7:
           - Adaptive erosion kernel (scales to narrowest mask dimension)
           - Sigma-clipping: reject pixels |depth − mean| > outlier_sigma * std
           - Transparency check: mask depth vs. 3px border ring → possibly_transparent flag
Stage 5  Pix2SG.predict(objects_3d): spatial scaffold → Florence-2 _enrich_with_florence2()
         For overlapping pairs (mask IoU > 0.02): red/blue overlay → Florence-2 <CAPTION> → canonical predicate
         Relations attached via _attach_relations_by_triplets; source_layer="florence2"
Stage 6  Strip _sam2_mask_array; json.dump → {stem}_scene.json
Stage 7  Visualization with mask centroid anchors, all [M] tags

## objects_3d entry fields
id, label, conf, bbox (SAM2 bbox, viz only), coordinates_3d (always mask-native),
depth_stats (z_val, z_val_pixels, possibly_transparent, depth_separation_from_background),
mask_centroid_2d (always mask COM, depth-weighted), sam2_mask_index (always valid),
mask_matched (always True), mask_path, depth_map_path, sources{GRiT,Pix2SG,Florence2}

## Key design decisions
- GroundedSAM2 replaces SAM2 AMG as primary segmentor → object-level masks (not part-level)
- `grounded_sam2_fallback_to_amg: True` — safety net if GDINO unavailable
- Florence-2 replaces GRiT heuristic as primary labeller; GRiT kept as fallback
- depth_scale_factor = 1.0 (metric output is already meters; old ×10 hack removed)
- CLIP loads/classifies/unloads before depth model loads (~340MB VRAM freed)
- _sam2_mask_array is transient (numpy), stripped before json.dump
- Pix2SG overlap: pixel mask IoU only (no bbox IoU fallback)
- matched_A_lookup: identity dict, no IoU search needed

## Fix 5.7 depth accuracy details
- Adaptive erosion: kernel=0 if min_dim<15px, kernel=1 if <30px, kernel=2 if <60px, else config max
- `_mask_depth_stats_and_3d()` now takes `use_erosion: bool = True` param — pass False to skip erosion
- Outlier sigma: `depth_outlier_sigma = 2.0` (0 = disabled)
- Transparency: compares mean(mask_depth) vs mean(border_ring_depth); flag if separation < 0.15m
- depth_stats gains fields: `possibly_transparent` (bool), `depth_separation_from_background` (float)

## Dual erosion comparison (current)
- `depth_erosion_comparison: bool = True` in config
- Every object in scene JSON has TWO sets:
  - `depth_stats` / `coordinates_3d` / `mask_centroid_2d` → with adaptive erosion
  - `depth_stats_no_erosion` / `coordinates_3d_no_erosion` / `mask_centroid_2d_no_erosion` → without
- Same for A/B depth_mask JSON objects

## Dual segmentor (current)
- `run_both_segmentors: bool = True` — runs GroundedSAM2 AND SAM2 AMG on every image
- AMG masks deduped against GDINO masks by IoU > 0.7; non-duplicates appended to amg_masks
- Every object entry has `segmentor` field: "GroundedSAM2" or "SAM2_AMG"
- Provides: object-level (GDINO), part-level (AMG), small objects (AMG), background (AMG)

## Mask filtering — FULLY DISABLED (current)
- All thresholds set to 0 / 1.0 — every mask passes
- sam2_post_filter_min_stability: 0.0, sam2_post_filter_min_pred_iou: 0.0
- sam2_post_filter_min_area_px: 0, sam2_post_filter_max_area_fraction: 1.0
- grounded_sam2_min_conf_for_stage3: 0.0
- GDINO thresholds lowered: box_thresh=0.15, text_thresh=0.15
- AMG min_mask_region_area lowered to 100px

## Config fields (config.py PreprocessConfig) — key values
depth_model_variant: "auto"  — "auto"|"indoor"|"outdoor"
depth_scale_factor: 1.0  — metric models need no scaling
camera_calibration_file: None  — path to calibration JSON from tools/calibrate_camera.py
apply_undistortion: True
camera_fx/fy/cx/cy: Optional[float]  — explicit intrinsics (priority 2)
camera_fov_degrees: 60.0  — fallback FOV estimate (priority 3)
sam2_amg_pred_iou_thresh: 0.80  — permissive at AMG level (was 0.88)
sam2_amg_stability_score_thresh: 0.92  — permissive (was 0.97)
sam2_post_filter_min_stability: 0.85  — post-hoc strict filter (was 0.0 disabled)
sam2_post_filter_min_pred_iou: 0.82  — post-hoc strict filter (was 0.0 disabled)
sam2_post_filter_min_area_px: 1500
sam2_post_filter_max_area_fraction: 0.30
grounding_dino_model: "IDEA-Research/grounding-dino-base"
grounding_dino_box_thresh: 0.30
grounding_dino_text_thresh: 0.25
grounded_sam2_fallback_to_amg: True
florence2_model: "microsoft/Florence-2-large"
florence2_task: "<OD>"
florence2_relation_enabled: True
psgtr_enabled: False  — not yet implemented; checkpoint required
univrd_enabled: False  — not yet implemented; checkpoint required
relation_min_mask_overlap: 0.02
depth_adaptive_erosion: True
depth_outlier_sigma: 2.0
depth_transparency_check: True
depth_transparency_threshold: 0.15  — metres
mask_erosion_kernel_size: 5
depth_central_fraction: 0.5
mask_iou_match_thresh: 0.1
pix2sg_mask_overlap_thresh: 0.05
pix2sg_depth_near_threshold: 1.0
pix2sg_depth_far_threshold: 3.0

## Pending / not yet implemented
- PSGTR wrapper class (config field exists: psgtr_enabled, psgtr_checkpoint)
- UniVRD wrapper class (config field exists: univrd_enabled, univrd_checkpoint)
