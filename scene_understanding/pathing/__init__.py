"""Path hypotheses export helpers (cost, traversability, canvas, scoring)."""

from __future__ import annotations

from scene_understanding.pathing.cost_map import astar_on_cost_map, build_path_cost_map, distance_transform_centering
from scene_understanding.pathing.export_workspace import prepare_path_hypotheses_workspace
from scene_understanding.pathing.path_hypotheses_paths import PATH_HYPOTHESES_JSON_NAME, path_hypotheses_json_path
from scene_understanding.pathing.hybrid_scores import apply_hybrid_confidence_scores
from scene_understanding.pathing.path_atlas_canvas import (
    write_path_rank_panels_line_only,
    write_trajectory_atlas_line_only,
)
from scene_understanding.pathing.path_atlas_export import export_trajs_upt_atlas
from scene_understanding.pathing.path_canvas import (
    draw_objects_boxes_bgr,
    draw_regions_contours_bgr,
    path_color_from_path_id,
    tapered_polyline_draw,
    write_path_context_top5_png,
)
from scene_understanding.pathing.path_colors import (
    bgr_from_stable_id,
    bgr_from_stable_id_trajectory,
    bgr_list_with_min_hue_separation,
    bgr_to_hex_rgb,
)
from scene_understanding.pathing.goal_anchors import (
    ground_intercept_goal_uv,
    pair_passes_locomotion_gates,
    vertical_structure_heuristic,
)
from scene_understanding.pathing.ground_route_grid import (
    build_coarse_speed_field,
    plan_coarse_support_path,
)
from scene_understanding.pathing.semantic_cost import (
    CostLayer,
    SemanticCostMap,
    layer_agent_physics,
    layer_depth_noise,
    layer_geometry,
    layer_scene_graph,
    precision_weighted_fuse,
)
from scene_understanding.pathing.staged_semantic_cost import (
    blend_traversability_with_fused_cost,
    build_staged_semantic_cost_map,
)
from scene_understanding.pathing.scene_context import (
    SceneContextBundle,
    build_scene_context,
    validate_scene_context,
)
from scene_understanding.pathing.calibration import calibrate_path_policy
from scene_understanding.pathing.semantic_fmm import (
    fmm_backtrace,
    hier_fmm_plan,
    k_diverse_fmm_paths,
    multiscale_fmm_plan,
    plan_path,
    theta_star_on_fmm,
)
from scene_understanding.pathing.semantic_validation import plan_with_validation
from scene_understanding.pathing.semantic_provenance import (
    annotate_path_with_provenance,
    save_semantic_cost_artifacts,
)
from scene_understanding.pathing.semantic_self_calibration import layer_self_calibration
from scene_understanding.pathing.semantic_vlm import (
    CLIPSegWrapper,
    CostPrompts,
    SigLIPTwoPoleScorer,
    layer_vlm,
    load_cost_prompts,
)
from scene_understanding.pathing.traversability import (
    build_traversability_speed_map,
    grid_dijkstra_path,
    heading_from_depth_at,
    is_soft_obstacle,
    k_diverse_grid_paths,
)

__all__ = [
    "blend_traversability_with_fused_cost",
    "build_coarse_speed_field",
    "build_staged_semantic_cost_map",
    "apply_hybrid_confidence_scores",
    "bgr_from_stable_id",
    "bgr_from_stable_id_trajectory",
    "bgr_list_with_min_hue_separation",
    "bgr_to_hex_rgb",
    "export_trajs_upt_atlas",
    "astar_on_cost_map",
    "build_path_cost_map",
    "build_traversability_speed_map",
    "distance_transform_centering",
    "draw_objects_boxes_bgr",
    "draw_regions_contours_bgr",
    "ground_intercept_goal_uv",
    "grid_dijkstra_path",
    "heading_from_depth_at",
    "is_soft_obstacle",
    "k_diverse_grid_paths",
    "pair_passes_locomotion_gates",
    "CostLayer",
    "SemanticCostMap",
    "CLIPSegWrapper",
    "CostPrompts",
    "SigLIPTwoPoleScorer",
    "layer_agent_physics",
    "layer_depth_noise",
    "layer_geometry",
    "layer_scene_graph",
    "layer_self_calibration",
    "layer_vlm",
    "load_cost_prompts",
    "fmm_backtrace",
    "theta_star_on_fmm",
    "multiscale_fmm_plan",
    "hier_fmm_plan",
    "k_diverse_fmm_paths",
    "plan_coarse_support_path",
    "plan_path",
    "annotate_path_with_provenance",
    "plan_with_validation",
    "save_semantic_cost_artifacts",
    "precision_weighted_fuse",
    "vertical_structure_heuristic",
    "SceneContextBundle",
    "build_scene_context",
    "validate_scene_context",
    "calibrate_path_policy",
    "PATH_HYPOTHESES_JSON_NAME",
    "path_hypotheses_json_path",
    "path_color_from_path_id",
    "prepare_path_hypotheses_workspace",
    "tapered_polyline_draw",
    "write_path_context_top5_png",
    "write_path_rank_panels_line_only",
    "write_trajectory_atlas_line_only",
]
