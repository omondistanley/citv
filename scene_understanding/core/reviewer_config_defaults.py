"""
Default hyperparameters cited by ``docs/path_context_top5_reviewer.md``.

These literals mirror ``config.SceneUnderstandingConfig`` in the root ``config.py`` module.
When changing defaults, update **both** files so the reviewer guide stays accurate.
"""
from __future__ import annotations

# --- Preprocess / GroundedSAM2 query (subset; see config.py PreprocessConfig) ---
query_builder_mode: str = "inherit"  # static | inherit | rampp_full | rampp_region_crops
run_both_segmentors: bool = False
grounded_sam2_fallback_to_amg: bool = False
run_both_segmentors_iou_dedup: float = 0.7
segmentation_production_strict: bool = False
segmentation_production_amg_pred_iou_thresh: float = 0.90
segmentation_production_amg_stability_score_thresh: float = 0.95
segmentation_production_amg_min_mask_region_area: int = 280
segmentation_production_amg_part_containment_thresh: float = 0.88
segmentation_production_amg_part_min_area_ratio_vs_grounded: float = 0.25

# --- §8A Region partition (subset; see config.py for full class) ---
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

# --- §8B Path hypotheses (subset) ---
export_path_hypotheses: bool = True
path_enable_region: bool = True
path_enable_object: bool = True
path_enable_mask: bool = True
path_top_k_per_pair: int = 3
path_max_candidates: int = 500
export_path_hypotheses_candidates: bool = True
path_dedupe_max_per_pair: int = 5
path_routing_relax_on_support: bool = True
path_region_fmm_policy: str = "when_sparse"
path_object_region_enabled: bool = True
path_min_confidence: float = 0.55
path_invalid_pixel_ratio_max: float = 0.05
path_max_turn_deg: float = 70.0
path_max_depth_step_m: float = 2.0
path_stroke_start_width_px: int = 8
path_stroke_end_width_px: int = 2
path_stroke_alpha_start: float = 0.95
path_stroke_alpha_end: float = 0.35

path_use_image_cost_refinement: bool = True
path_cost_weight_edges: float = 0.35
path_cost_weight_obstacle: float = 0.35
path_cost_weight_region_prior: float = 0.15
path_cost_weight_centering: float = 0.15
path_refine_num_points: int = 96
path_export_traversability_speed: bool = True
path_use_traversability_geodesic: bool = True
path_geodesic_replace_astar: bool = False
path_geodesic_k_alt: int = 2
path_geodesic_edge_penalty: float = 0.35

# --- §8B2 Staged fused cost, channels, ground manifold, QA (see config.py) ---
path_use_fused_semantic_cost: bool = True
path_fused_semantic_cost_blend: float = 0.72
path_cost_use_vlm_layer: bool = False
path_cost_ontology_caption_layer: bool = True
path_cost_caption_layer_precision_cap: float = 0.22
path_multi_channel_traversability: bool = True
path_fmm_on_ground_manifold: bool = False
path_ground_manifold_grid_step_px: int = 14
path_goal_ground_intercept_vertical: bool = True
path_goal_vertical_aspect_thresh: float = 1.35
path_pair_min_span_px: float = 18.0
path_pair_max_depth_delta_m: float = 0.0
path_pair_skip_facade_facade: bool = True
path_pair_facade_skip_requires_vertical: bool = False
path_pair_require_relation: bool = False
path_bezier_snap_feasible_enabled: bool = True
path_bezier_snap_feasible_max_px: float = 8.0
path_qa_perf_mode: bool = False
path_atlas_auto_panel_count: bool = False
artifact_manifest_mp4_min_bytes: int = 256
path_emit_aerial_approach_hypotheses: bool = True
path_max_aerial_hypotheses: int = 12
path_emit_contour_hypotheses: bool = True
path_max_contour_hypotheses: int = 8

# --- §8C Traversability weights ---
trav_weight_image_edge: float = 0.25
trav_weight_depth_flatness: float = 0.55
trav_weight_image_smooth: float = 0.45
trav_depth_grad_sigma_m: float = 0.35
trav_speed_floor: float = 0.06

# --- §8D Pair proposals + hybrid weights ---
path_pair_proposal_enabled: bool = True
path_pair_top_k_targets: int = 4
path_pair_allow_static_static: bool = True
path_semantic_hard_filter_enabled: bool = True
path_semantic_max_far_background_ratio: float = 0.40
path_semantic_max_obstacle_ratio: float = 0.35
path_semantic_min_walkable_ratio: float = 0.20

path_score_weight_geometric: float = 0.40
path_score_weight_semantic: float = 0.30
path_score_weight_relation: float = 0.20
path_score_weight_action_fit: float = 0.10

# --- §8E path_context render-only ---
path_export_context_composites: bool = True
path_context_top_k: int = 12

# --- §8E2 Trajectory atlas (trajs-upt); see docs/trajs-upt.md ---
path_atlas_enabled: bool = False
path_atlas_total: int = 30
path_atlas_panel_size: int = 10
path_atlas_rank_source: str = "paths_sorted"  # "paths_sorted" | "paths_recommended"
path_contract_compliance_mode: bool = True
path_atlas_background_bgr: tuple = (26, 26, 26)
path_atlas_prefer_geodesic: bool = True
path_atlas_min_hue_separation: float = 0.07
path_atlas_endpoint_dots: bool = False
path_atlas_export_trajectory_line_only: bool = False
path_atlas_trajectory_stroke_start_px: int = 5
path_atlas_trajectory_stroke_end_px: int = 2

# --- §8E3 Scene path+trajectory batch overlays ---
path_scene_trajectory_batches_enabled: bool = True
path_scene_trajectory_batch_size: int = 10
path_scene_batch_include_context: bool = False
path_scene_batch_include_context_debug: bool = True

# --- §8F Motion contract / trajectory (subset) ---
export_motion_contract_json: bool = True
path_animation_qa_modes: list = [24]
path_animation_qa_candidate_seconds: float = 0.8
path_animation_qa_delimiter_seconds: float = 0.12
trajectory_hypotheses_max_subjects: int = 8
trajectory_instant_step_px: float = 6.0
trajectory_instant_dt_s: float = 0.04
trajectory_hypotheses_include_all_objects: bool = False
motion_contract_default_footprint_m: float = 0.45
motion_contract_default_clearance_m: float = 0.15
trajectory_use_depth_heading: bool = True
trajectory_depth_heading_blend: float = 0.55
trajectory_depth_heading_window: int = 9
path_motion_contract_overlay_max_paths: int = 24
path_motion_contract_legacy_line_px: int = 2
path_motion_contract_geodesic_line_px: int = 3
