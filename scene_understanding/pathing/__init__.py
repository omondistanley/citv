"""Path hypotheses export helpers (cost, traversability, canvas, scoring)."""

from __future__ import annotations

from scene_understanding.pathing.affordance_rasters import (
    RASTER_CHANNELS,
    build_affordance_rasters,
    save_affordance_rasters,
)
from scene_understanding.pathing.cost_map import astar_on_cost_map, build_path_cost_map, distance_transform_centering
from scene_understanding.pathing.export_hook import invoke_path_hypotheses_export_for_track
from scene_understanding.pathing.export_workspace import prepare_path_hypotheses_workspace
from scene_understanding.pathing.goal_anchors import (
    ground_intercept_goal_uv,
    mask_bbox_px,
    mask_centroid_px,
    object_footprint_mask,
    pair_passes_locomotion_gates,
    vertical_structure_heuristic,
)
from scene_understanding.pathing.ground_plane import build_support_mask, snap_uv_down_to_support
from scene_understanding.pathing.path_hypotheses_paths import PATH_HYPOTHESES_JSON_NAME, path_hypotheses_json_path
from scene_understanding.pathing.hybrid_scores import apply_hybrid_confidence_scores
from scene_understanding.pathing.path_canvas import (
    draw_objects_boxes_bgr,
    draw_regions_contours_bgr,
    path_color_from_path_id,
    tapered_polyline_draw,
    write_path_context_top5_png,
)
from scene_understanding.pathing.polyline_3d import attach_polyline_3d_to_paths, lift_polyline_2d_to_3d, smooth_polyline_in_3d
from scene_understanding.pathing.semantic_cost import (
    CostLayer,
    agent_physics_cost_layer,
    depth_noise_precision_layer,
    geometry_cost_layer,
    geometry_cost_layer_with_depth,
    precision_weighted_fuse,
    scene_graph_cost_layer,
)
from scene_understanding.pathing.semantic_fmm import (
    has_fmm,
    k_diverse_from_T,
    k_diverse_paths,
    plan_path,
    speed_from_cost,
    time_of_arrival,
    time_of_arrival_from_speed,
)
from scene_understanding.pathing.traversability import (
    build_traversability_speed_map,
    grid_dijkstra_path,
    heading_from_depth_at,
    k_diverse_grid_paths,
)

__all__ = [
    "agent_physics_cost_layer",
    "apply_hybrid_confidence_scores",
    "astar_on_cost_map",
    "attach_polyline_3d_to_paths",
    "build_affordance_rasters",
    "build_path_cost_map",
    "build_support_mask",
    "build_traversability_speed_map",
    "CostLayer",
    "depth_noise_precision_layer",
    "distance_transform_centering",
    "draw_objects_boxes_bgr",
    "draw_regions_contours_bgr",
    "geometry_cost_layer",
    "geometry_cost_layer_with_depth",
    "ground_intercept_goal_uv",
    "grid_dijkstra_path",
    "has_fmm",
    "heading_from_depth_at",
    "invoke_path_hypotheses_export_for_track",
    "k_diverse_from_T",
    "k_diverse_grid_paths",
    "k_diverse_paths",
    "lift_polyline_2d_to_3d",
    "mask_bbox_px",
    "mask_centroid_px",
    "object_footprint_mask",
    "pair_passes_locomotion_gates",
    "PATH_HYPOTHESES_JSON_NAME",
    "path_color_from_path_id",
    "path_hypotheses_json_path",
    "plan_path",
    "precision_weighted_fuse",
    "prepare_path_hypotheses_workspace",
    "RASTER_CHANNELS",
    "save_affordance_rasters",
    "scene_graph_cost_layer",
    "smooth_polyline_in_3d",
    "snap_uv_down_to_support",
    "speed_from_cost",
    "tapered_polyline_draw",
    "time_of_arrival",
    "time_of_arrival_from_speed",
    "vertical_structure_heuristic",
    "write_path_context_top5_png",
]
