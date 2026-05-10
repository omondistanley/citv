"""
Default hyperparameters cited by ``docs/path_context_top5_reviewer.md``.

These literals mirror ``config.SceneUnderstandingConfig`` in the root ``config.py`` module.
When changing defaults, update **both** files so the reviewer guide stays accurate.
"""
from __future__ import annotations

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
path_context_top_k: int = 5

# --- §8F Motion contract / trajectory (subset) ---
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
path_motion_contract_overlay_max_paths: int = 24
path_motion_contract_legacy_line_px: int = 2
path_motion_contract_geodesic_line_px: int = 3
