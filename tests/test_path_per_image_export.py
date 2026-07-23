import importlib.util
import sys
import json
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parent.parent


def _load_legacy_scene_understanding():
    su = REPO / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_citv_su_path_export_test", su)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_citv_su_path_export_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_rect_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_export_path_individual_pngs_top10(tmp_path):
    mod = _load_legacy_scene_understanding()
    Pipe = mod.SceneUnderstandingPipeline
    pipe = Pipe.__new__(Pipe)

    class _Cfg:
        # enable paths
        export_path_hypotheses = True
        path_enable_region = True
        path_enable_object = True
        path_enable_mask = True
        path_top_k_per_pair = 3
        path_max_candidates = 50
        path_min_confidence = 0.0
        path_invalid_pixel_ratio_max = 1.0
        path_max_turn_deg = 180.0
        path_stroke_start_width_px = 6
        path_stroke_end_width_px = 1
        path_stroke_alpha_start = 0.9
        path_stroke_alpha_end = 0.4

        # descriptions local only (no network)
        path_export_descriptions = True
        path_description_backend = "local"

        # individual images
        path_export_individual_images = True
        path_individual_top_k_per_level = 10
        path_direction_text = True
        path_label_text = True
        path_text_max_chars = 80
        path_text_scale = 0.4

        # context composites
        path_export_context_composites = True
        path_context_top_k = 5

        # semantic/hybrid/animation/stage exports
        path_semantic_enabled = True
        path_semantic_embedding_model = "lexical-fallback-v1"
        path_semantic_llm_enabled = False
        path_semantic_llm_top_k = 5
        path_score_weight_geometric = 0.4
        path_score_weight_semantic = 0.3
        path_score_weight_relation = 0.2
        path_score_weight_action_fit = 0.1
        path_export_stage_images = True
        path_stage_levels = ["global", "region", "object", "mask"]
        path_animation_enabled = True
        path_animation_fps = 24
        path_speed_walk_px_s = 80.0
        path_speed_run_px_s = 150.0
        path_duration_idle_s = 0.5
        path_duration_jump_s = 0.6
        path_pair_proposal_enabled = True
        path_pair_top_k_targets = 3
        path_pair_allow_static_static = True
        # Disabled so blank test image still yields non-suppressed paths for motion contracts.
        path_semantic_hard_filter_enabled = False
        path_semantic_max_far_background_ratio = 1.0
        path_semantic_max_obstacle_ratio = 1.0
        path_semantic_min_walkable_ratio = 0.0
        path_use_image_cost_refinement = True
        path_refine_num_points = 48
        path_export_pair_proposals_json = True
        path_export_diagnostics_json = True

        export_motion_contract_json = True
        path_export_traversability_speed = True
        path_use_traversability_geodesic = True
        path_motion_contract_overlay = True
        trajectory_use_depth_heading = True
        path_export_fields_explainer = True

        path_v2_visualization_enabled = True
        path_v2_output_subdir = "trajectory_viz_v2"
        path_v2_prefer_geodesic = True
        path_v2_draw_scope = "all_valid"
        path_v2_max_paths = 500

        path_v3_visualization_enabled = True
        path_v3_export_underlay_stack = True
        path_v3_export_underlay_stack_context = True
        path_v3_export_suppressed_residual = True

    pipe.config = _Cfg()

    h, w = 64, 64
    img_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Two side-by-side regions
    region_label_map = np.zeros((h, w), dtype=np.int32)
    region_label_map[:, : w // 2] = 1
    region_label_map[:, w // 2 :] = 2

    regions_block = {
        "regions": [
            {
                "region_index": 1,
                "id": "region_1",
                "semantic_label": "left_area",
                "centroid_2d_px": [w // 4, h // 2],
                "type": "foreground",
            },
            {
                "region_index": 2,
                "id": "region_2",
                "semantic_label": "right_area",
                "centroid_2d_px": [3 * w // 4, h // 2],
                "type": "foreground",
            },
        ]
    }

    region_adjacency = {
        "edges": [
            {
                "region_a": "region_1",
                "region_b": "region_2",
                "shared_border_px": 100,
                "depth_delta_m": 0.0,
                "depth_relation": "a_in_front",
                "edge_type": "region_spatial_adjacency",
            }
        ],
        "num_edges": 1,
        "kind": "region_adjacency_graph",
    }

    # Two objects, one per region
    mask0 = _make_rect_mask(h, w, x1=8, y1=20, x2=22, y2=42)
    mask1 = _make_rect_mask(h, w, x1=42, y1=20, x2=56, y2=42)

    objects = [
        {
            "entity_kind": "object",
            "id": "obj_0",
            "label": "person",
            "canonical_name": "person",
            "region_id": "region_1",
            "mask_centroid_2d": [15, 31],
            "_sam2_mask_array": mask0,
        },
        {
            "entity_kind": "object",
            "id": "obj_1",
            "label": "vehicle",
            "canonical_name": "vehicle",
            "region_id": "region_2",
            "mask_centroid_2d": [49, 31],
            "_sam2_mask_array": mask1,
        },
    ]

    track_dir = tmp_path / "out"
    track_dir_name = "grounded_sam2"
    path_stem = "dummy_img"

    xd = np.linspace(2.0, 8.0, w, dtype=np.float32)
    metric_depth = np.tile(xd[np.newaxis, :], (h, 1))

    pipe._export_path_hypotheses_for_track(
        img_bgr=img_bgr,
        path_stem=path_stem,
        track_dir_name=track_dir_name,
        track_dir=track_dir,
        objects_3d_with_masks=objects,
        regions_block=regions_block,
        region_label_map=region_label_map,
        region_adjacency=region_adjacency,
        relations=[],
        metric_depth_m=metric_depth,
    )

    paths_dir = track_dir / f"{path_stem}_paths"
    assert paths_dir.exists()
    assert (paths_dir / "path_hypotheses.json").exists()
    assert (paths_dir / "path_descriptions.json").exists()
    assert (paths_dir / "path_reasoning.md").exists()
    assert (paths_dir / "path_context_top5.png").exists()
    assert (paths_dir / "semantic_layer.json").exists()
    assert (paths_dir / "animation_plan.json").exists()
    assert (paths_dir / "path_visual_index.json").exists()
    assert (paths_dir / "pair_proposals.json").exists()
    assert (paths_dir / "path_diagnostics.json").exists()
    assert (paths_dir / "path_cost_map.png").exists()
    assert (paths_dir / "path_cost_map.npy").exists()
    assert (paths_dir / "path_traversability_speed.npy").exists()
    assert (paths_dir / "path_traversability_speed.png").exists()
    assert (paths_dir / "insertion_path_ensembles.json").exists()
    assert (paths_dir / "trajectory_hypotheses.json").exists()
    assert (paths_dir / "motion_contracts_overlay.png").exists()
    assert (paths_dir / "path_fields_explainer.png").exists()
    assert (paths_dir / "path_fields_legend.json").exists()

    v2_dir = paths_dir / "trajectory_viz_v2"
    assert v2_dir.is_dir()
    assert (v2_dir / "traj_v2_all_ink.png").exists()
    assert (v2_dir / "traj_v2_all_ink_context.png").exists()
    assert (v2_dir / "traj_v2_manifest.json").exists()
    assert (v2_dir / "traj_v2_topK_ink_context.png").exists()
    assert (v2_dir / "traj_v2_geodesic_compare.png").exists()
    assert (v2_dir / "traj_v2_alternates_faint.png").exists()
    assert (v2_dir / "traj_v2_underlay_final.png").exists()
    assert (v2_dir / "traj_v2_underlay_final_context.png").exists()
    assert (v2_dir / "traj_v2_suppressed_residual.png").exists()
    man = json.loads((v2_dir / "traj_v2_manifest.json").read_text())
    assert man.get("schema") == "citv_trajectory_viz_v2_manifest_v3"
    assert man.get("files", {}).get("all_ink", "").endswith("/trajectory_viz_v2/traj_v2_all_ink.png")
    assert "topk_ink_context" in man.get("files", {})
    assert "geodesic_compare" in man.get("files", {})
    assert "alternates_faint" in man.get("files", {})
    assert "underlay_final" in man.get("files", {})
    assert "underlay_final_context" in man.get("files", {})
    assert "suppressed_residual" in man.get("files", {})
    assert isinstance(man.get("phase2"), dict)
    assert man.get("phase3", {}).get("enabled") is True

    payload = json.loads((paths_dir / "path_hypotheses.json").read_text())
    assert "traversability" in payload
    assert payload["traversability"].get("speed_map_npy")
    leg = json.loads((paths_dir / "path_fields_legend.json").read_text())
    assert leg.get("schema") == "citv_path_fields_legend_v1"
    assert len(leg.get("panels", [])) >= 3
    descriptions = json.loads((paths_dir / "path_descriptions.json").read_text())
    semantic_layer = json.loads((paths_dir / "semantic_layer.json").read_text())
    animation_plan = json.loads((paths_dir / "animation_plan.json").read_text())
    visual_index = json.loads((paths_dir / "path_visual_index.json").read_text())

    assert "entities" in semantic_layer
    assert "region_affordances" in semantic_layer
    assert "paths" in animation_plan
    assert "paths" in visual_index

    ins_contract = json.loads((paths_dir / "insertion_path_ensembles.json").read_text())
    traj_contract = json.loads((paths_dir / "trajectory_hypotheses.json").read_text())
    assert ins_contract.get("schema") == "citv_insertion_path_ensembles_bundle_v1"
    assert "version" in ins_contract and "image_stem" in ins_contract and "ensembles" in ins_contract
    assert traj_contract.get("schema") == "citv_trajectory_hypotheses_bundle_v1"
    assert "version" in traj_contract and "image_stem" in traj_contract and "hypotheses" in traj_contract
    assert len(traj_contract["hypotheses"]) >= 1
    assert len(ins_contract["ensembles"]) >= 1
    legacy_by_id = {str(p.get("path_id", "")): p for p in payload.get("paths", []) if not p.get("suppressed")}
    matched = False
    for e in ins_contract["ensembles"]:
        h0 = (e.get("hypotheses") or [{}])[0]
        lid = str(h0.get("legacy_path_id", ""))
        if lid in legacy_by_id:
            assert h0.get("polyline_2d") == legacy_by_id[lid].get("polyline_2d")
            matched = True
            break
    assert matched
    families = {e.get("path_family") for e in ins_contract["ensembles"]}
    assert "object_object" in families
    assert "region_region" in families
    obj_paths = [p for p in payload.get("paths", []) if str(p.get("path_level")) == "object"]
    assert any(p.get("polyline_geodesic_2d") for p in obj_paths)
    obj_ens = next(e for e in ins_contract["ensembles"] if e.get("path_family") == "object_object")
    assert len(obj_ens.get("hypotheses") or []) >= 2

    per_level_dir = {
        "region": paths_dir / "images" / "region",
        "object": paths_dir / "images" / "object",
        "mask": paths_dir / "images" / "mask",
    }

    def _count_paths(level: str) -> int:
        return sum(1 for p in payload.get("paths", []) if str(p.get("path_level", "")).lower() == level)

    for p in payload.get("paths", []):
        assert "semantic_valid" in p
        assert "semantic_validity_score" in p
        assert "affordance_trace" in p
        assert "hybrid_overall" in (p.get("scores") or {})

    for lvl, d in per_level_dir.items():
        assert d.exists()
        imgs = sorted(list(d.glob("path_*.png")))
        assert len(imgs) == min(_count_paths(lvl), pipe.config.path_individual_top_k_per_level)
        if not imgs:
            continue

        # per-path image is overlaid on input image (BGR)
        im = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
        assert im is not None
        assert im.ndim == 3 and im.shape[2] >= 3

        # every rendered path_id must exist in descriptions
        for img_path in imgs:
            pid = img_path.stem
            assert pid.startswith("path_")
            pid = pid[len("path_") :]
            assert pid in descriptions

    stage_root = paths_dir / "images" / "stages"
    assert stage_root.exists()
    stages = ["00_pair_proposals", "01_cost_map", "02_coarse_routes", "03_refined_routes", "04_semantic_hard_filtered", "05_ranked_topk"]
    levels = pipe.config.path_stage_levels
    for stage in stages:
        for lvl in levels:
            expected = stage_root / stage / lvl / f"{path_stem}_{stage}_{lvl}.png"
            assert expected.exists()

    for pid, item in (visual_index.get("paths") or {}).items():
        per_path = item.get("per_path_image", "")
        assert isinstance(per_path, str) and per_path.endswith(".png")
        rel = per_path.split(f"{track_dir_name}/", 1)[-1]
        assert (track_dir / rel).exists()

