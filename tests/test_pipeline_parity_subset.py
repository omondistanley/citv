"""
Subset parity checks: staged scene JSON must expose a minimal metadata surface
compatible with legacy scene graph consumers (see docs/path_context_top5_reviewer.md for pipeline QA context).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import scene_write

# Overlap with legacy metadata keys used by test_scene_graph_schema / consumers.
REQUIRED_STAGED_METADATA_KEYS = {
    "timestamp",
    "segmentor",
    "intrinsics",
    "models",
    "relation_sources",
    "relation_debug",
    "depth_map",
    "segmentation_image",
    "sam2_segmentation_image",
    "sam2_tinted_overlay_image",
}


def test_build_staged_scene_payload_metadata_superset(tmp_path):
    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg",
        output_dir=tmp_path / "out",
        stem="p",
        timestamp="2099-01-01 00:00:00",
        img_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        img_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        height=8,
        width=8,
        intrinsics={"fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 4.0},
        extra={"objects": []},
    )
    payload = scene_write.build_staged_scene_payload(ctx)
    meta = payload["metadata"]
    missing = REQUIRED_STAGED_METADATA_KEYS - set(meta.keys())
    assert not missing, f"staged metadata missing keys: {missing}"
    rd = meta["relation_debug"]
    assert set(rd.keys()) >= {"num_detections", "num_objects", "num_relations"}
    assert isinstance(meta["intrinsics"], dict)
    assert meta["segmentor"] == "package_staged_pipeline"


def test_legacy_delegate_importable():
    from scene_understanding.stages import run_legacy_process_image

    assert callable(run_legacy_process_image)


def test_stage_chain_wiring_includes_new_stages():
    """Source-level regression guard: catches someone silently dropping a
    stage call from ``_run_stage_chain`` (e.g. the historical bug where
    ``paths_export`` was wired as a stage module but never actually invoked
    from the chain -- ``trajectory_hypotheses.json`` was absent despite the
    stage existing). Cheap to check without running the heavy pipeline."""
    import inspect

    import scene_understanding.pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.SceneUnderstandingPipeline._run_stage_chain)
    for call in (
        "_stage_preprocess", "_stage_depth", "_stage_regions", "_stage_segment",
        "_stage_label_and_geometry", "_stage_depth_mask_fusion", "_stage_relations",
        "_stage_hierarchy_export", "_stage_paths_export", "_stage_captions_export",
    ):
        assert call in src, f"_run_stage_chain no longer calls {call}"


def _synthetic_two_object_ctx(tmp_path):
    """Two objects on a flat floor with real per-pixel masks -- a chair and a
    table -- close enough to the fixtures used in test_scene_understanding_stages_unit.py
    but rich enough (masks, depth, relations) to exercise the new stages."""
    import numpy as np

    h, w = 96, 128
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:25, :] = 8.0  # background band

    obj_a_mask = np.zeros((h, w), dtype=bool)
    obj_a_mask[60:80, 15:35] = True
    obj_b_mask = np.zeros((h, w), dtype=bool)
    obj_b_mask[55:85, 90:115] = True
    depth[obj_a_mask] = 2.0
    depth[obj_b_mask] = 2.5

    objects = [
        {
            "id": "obj_0", "graph_id": "obj_0", "bbox": [15, 60, 35, 80],
            "canonical_name": "a wooden chair", "label": "chair",
            "_sam2_mask_array": obj_a_mask, "depth_stats": {"median": 2.0},
            "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 2.0}, "mask_centroid_2d": [25, 70],
            "segmentor": "GroundedSAM2",
        },
        {
            "id": "obj_1", "graph_id": "obj_1", "bbox": [90, 55, 115, 85],
            "canonical_name": "a dining table", "label": "table",
            "_sam2_mask_array": obj_b_mask, "depth_stats": {"median": 2.5},
            "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 2.5}, "mask_centroid_2d": [102, 70],
            "segmentor": "GroundedSAM2",
        },
    ]

    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="p",
        timestamp="2099-01-01 00:00:00", img_bgr=img, img_rgb=img, height=h, width=w,
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": w / 2.0, "cy": h / 2.0}, metric_depth=depth,
    )
    ctx.extra["objects"] = objects
    return ctx


class _FakeDepthEstimator:
    device = "cpu"


class _FakePipeline:
    config = None
    depth_estimator = _FakeDepthEstimator()


def test_paths_export_produces_non_degenerate_hypotheses(tmp_path):
    """Value-parity regression guard for the historical #1 blocker: path
    hypotheses existed only as 2-point (start/end-only) degenerate polylines
    with no 3D lift, no kinematic tags, and no animation-render contract.
    A real fixed-cost FMM solve between two floor-anchored objects should
    produce a real multi-point route with all enrichment fields populated."""
    from scene_understanding.stages import paths_export

    ctx = _synthetic_two_object_ctx(tmp_path)
    out_ctx = paths_export.run(_FakePipeline(), ctx)

    assert out_ctx.path_exports.get("path_hypotheses_count", 0) > 0
    hyp_path = Path(out_ctx.path_exports["path_hypotheses_json"])
    payload = json.loads(hyp_path.read_text())
    assert payload["hypotheses"], "no path hypotheses were written"

    hyp = payload["hypotheses"][0]
    assert len(hyp["polyline_2d"]) > 2, "path collapsed to a degenerate start/end-only polyline"
    assert len(hyp["polyline_3d"]) == len(hyp["polyline_2d"])
    assert hyp["kinematic_signatures"], "no kinematic signatures derived from the 3D Z-profile"
    assert hyp["animation_render_contract"]["sample_state_preview"]
    assert hyp["contract_status"]["acceptance_status"] in {"accepted", "low_confidence", "rejected"}
    assert "visibility_profile" in hyp and "support_trace" in hyp


def test_hierarchy_export_containment_and_relation_derived_occlusion(tmp_path):
    from scene_understanding.stages import hierarchy_export

    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    big_mask = np.zeros((h, w), dtype=bool)
    big_mask[5:55, 5:55] = True
    small_mask = np.zeros((h, w), dtype=bool)
    small_mask[15:25, 15:25] = True
    objects = [
        {"id": "obj_0", "label": "shelf", "_sam2_mask_array": big_mask, "sam2_mask_index": 0},
        {"id": "obj_1", "label": "book", "_sam2_mask_array": small_mask, "sam2_mask_index": 1},
    ]
    relations = [{"subject_id": "obj_1", "object_id": "obj_0", "predicate": "in_front_of"}]

    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="p",
        timestamp="t", img_bgr=img, img_rgb=img, height=h, width=w, intrinsics={},
        relations=relations,
    )
    ctx.extra["objects"] = objects

    out_ctx = hierarchy_export.run(_FakePipeline(), ctx)
    book, shelf = objects[1], objects[0]
    assert book["parent_object_id"] == "obj_0"
    assert book["containment_depth"] == 1
    assert shelf["containment_depth"] == 0
    assert book["occludes"] == ["obj_0"]  # relation-derived, independent of spatial containment
    assert shelf["occluded_by"] == ["obj_1"]
    assert out_ctx.mask_hierarchy is not None
    detailed = json.loads(Path(out_ctx.path_exports["mask_hierarchy_detailed_json"]).read_text())
    assert detailed["max_containment_depth"] == 1


def test_depth_mask_fusion_writes_mode_a(tmp_path):
    from scene_understanding.stages import depth_mask_fusion

    ctx = _synthetic_two_object_ctx(tmp_path)
    out_ctx = depth_mask_fusion.run(_FakePipeline(), ctx)
    payload = json.loads(Path(out_ctx.path_exports["depth_mask_a_json"]).read_text())
    assert payload["metadata"]["matching_mode"] == "A"
    assert len(payload["depth_mask"]["objects"]) == 2
    assert payload["mode_b"]["attempted"] is False


def test_captions_export_writes_full_bundle(tmp_path):
    from scene_understanding.stages import captions_export

    ctx = _synthetic_two_object_ctx(tmp_path)
    out_ctx = captions_export.run(_FakePipeline(), ctx)
    bundle = json.loads(Path(out_ctx.path_exports["caption_bundle_json"]).read_text())
    expected_files = {
        "florence_object_captions_json", "florence_scene_caption_json",
        "fusion_scene_caption_json", "hybrid_scene_caption_json", "caption_comparison_json",
    }
    assert expected_files <= set(bundle["files"].keys())


def test_pix2sg_relation_predicates_beyond_overlap():
    """SCENE_GRAPH_DEEP_DIVE.md §8 item 4: relation candidates must cover more
    than mask overlap -- containment, support/contact, and near-touch."""
    from scene_understanding.relations.pix2sg import Pix2SGWrapper

    class _Wrapper(Pix2SGWrapper):
        def __init__(self):
            self._mask_overlap_thresh = 0.05
            self._depth_far_threshold = 3.0

    w = _Wrapper()
    h_, w_ = 100, 100

    big = np.zeros((h_, w_), dtype=bool)
    big[10:90, 10:90] = True
    small = np.zeros((h_, w_), dtype=bool)
    small[40:50, 40:50] = True
    assert w._spatial_predicate_mask({"_sam2_mask_array": small}, {"_sam2_mask_array": big}) == "inside_of"
    assert w._spatial_predicate_mask({"_sam2_mask_array": big}, {"_sam2_mask_array": small}) == "contains"

    table = np.zeros((h_, w_), dtype=bool)
    table[70:80, 0:100] = True
    box = np.zeros((h_, w_), dtype=bool)
    box[55:70, 30:60] = True
    assert w._spatial_predicate_mask({"_sam2_mask_array": box}, {"_sam2_mask_array": table}) == "resting_on"

    left = np.zeros((h_, w_), dtype=bool)
    left[20:30, 10:20] = True
    right = np.zeros((h_, w_), dtype=bool)
    right[20:30, 23:33] = True
    assert w._spatial_predicate_mask({"_sam2_mask_array": left}, {"_sam2_mask_array": right}) == "touching"
