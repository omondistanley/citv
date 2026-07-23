"""Unit tests for depth region partitioning and helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_partitioner():
    p = REPO / "scene_understanding" / "regions" / "partitioner.py"
    spec = importlib.util.spec_from_file_location("_citv_regions_partitioner_test", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_citv_regions_partitioner_test"] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def test_partition_two_plateaus():
    m = _load_partitioner()
    d = np.zeros((60, 60), dtype=np.float32)
    d[:, :30] = 1.5
    d[:, 30:] = 12.0
    r = m.partition_depth_regions(d, k=2, min_region_px=80, seed=7)
    assert len(r.regions) >= 1
    assert r.label_map.shape == d.shape
    bgr = m.label_map_to_bgr(r.label_map, r.palette)
    assert bgr.shape == (60, 60, 3)


def test_majority_region_index():
    m = _load_partitioner()
    lm = np.zeros((20, 20), dtype=np.int32)
    lm[:, :10] = 1
    lm[:, 10:] = 2
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 3:8] = True
    assert m.majority_region_index(mask, lm) == 1


def test_kmeans_reproducible():
    m = _load_partitioner()
    d = np.random.RandomState(1).rand(32, 32).astype(np.float32) * 5 + 2
    a = m.partition_depth_regions(d, k=3, min_region_px=30, seed=99)
    b = m.partition_depth_regions(d, k=3, min_region_px=30, seed=99)
    assert np.array_equal(a.label_map, b.label_map)


def test_depth_geometry_region_sigma():
    p = REPO / "scene_understanding" / "geometry" / "depth_geometry.py"
    spec = importlib.util.spec_from_file_location("_citv_depth_geometry_test", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_citv_depth_geometry_test"] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    DepthGeometry = m.DepthGeometry

    def bp(u, v, z, k):
        return {"x": float(u) * z / k["fx"], "y": float(v) * z / k["fy"], "z": float(z)}

    dg = DepthGeometry(bp, depth_outlier_sigma=2.0, depth_sigma_clip_scope="mask")
    depth = np.full((20, 20), 5.0, dtype=np.float32)
    depth[5:15, 5:15] = 5.1
    depth[8, 8] = 50.0
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    k = {"fx": 500.0, "fy": 500.0, "cx": 10.0, "cy": 10.0}
    rc = {"type": "foreground", "depth_stats": {"mean": 5.1, "std": 0.05}, "sigma_scope": "region"}
    stats, _, _ = dg.mask_depth_stats_and_3d(
        depth, k, mask, region_context=rc, label_map=None, region_index=0
    )
    assert stats["max"] < 20.0


def test_layers_payload_includes_regions_summary():
    """Layers JSON includes a regions[] band when region metas are passed."""
    su = REPO / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_citv_su_layers_test", su)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Pipe = mod.SceneUnderstandingPipeline
    pipe = Pipe.__new__(Pipe)
    objs = [
        {"id": "o1", "coordinates_3d": {"z": 1.0}, "layer_type": "unassigned"},
    ]
    metas = [
        {
            "region_index": 1,
            "id": "region_1",
            "type": "foreground",
            "depth_stats": {"mean": 2.5},
            "object_ids": [],
        },
    ]
    layers = pipe._build_layers_payload(objs, metas)
    assert "regions" in layers
    assert isinstance(layers["regions"], list)
    assert len(layers["regions"]) == 1
    assert layers["regions"][0]["region_id"] == "region_1"
    assert any(e.get("entity_kind") == "region" for e in layers["ordering"])


def test_pipeline_settings_attached_after_init():
    """Legacy pipeline exposes frozen PipelineSettings snapshot on instances."""
    su = REPO / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_citv_su_settings_test", su)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Pipe = mod.SceneUnderstandingPipeline
    pipe = Pipe.__new__(Pipe)
    pipe.depth_mask_modes = ["A"]
    pipe.require_any_relation_source = True
    pipe.mask_iou_match_thresh = 0.11
    pipe.pix2sg_mask_overlap_thresh = 0.06
    pipe.pix2sg_depth_near_threshold = 1.1
    pipe.depth_scale_factor = 10.0
    pipe.depth_sigma_clip_scope = "mask"
    pipe.regions_enabled = True
    pipe._regions_k = 3
    pipe._regions_min_region_px = 400
    pipe._regions_seed = 7
    pipe.export_hybrid_captions = False
    pipe.export_caption_prompt_bundle = True
    pipe.caption_max_objects_per_track = 32
    pipe._florence2_label_enabled = True
    pipe._rampp_enabled = False
    from scene_understanding.core.pipeline_settings import attach_settings_from_pipeline

    attach_settings_from_pipeline(pipe)
    ps = pipe.pipeline_settings
    assert ps.depth_mask_modes == ["A"]
    assert ps.regions_enabled is True
    assert ps.regions_k == 3
    assert ps.export_hybrid_captions is False


def test_region_track_relpaths_keys():
    from scene_understanding.regions import region_track_relpaths

    d = region_track_relpaths("combined", "img1")
    assert d["region_segmentation_image"].endswith("img1_region_segmentation.png")
    assert "region_tinted_overlay_image" in d


def test_region_visual_objects_use_semantic_label():
    su = REPO / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_citv_su_sem_test", su)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Pipe = mod.SceneUnderstandingPipeline
    pipe = Pipe.__new__(Pipe)
    lm = np.zeros((10, 10), dtype=np.int32)
    lm[:, :5] = 1
    rmeta = [
        {
            "region_index": 1,
            "id": "region_1",
            "type": "foreground",
            "bbox_px": [0, 0, 9, 9],
            "centroid_2d_px": [2.0, 5.0],
            "semantic_label": "sky",
            "canonical_name": "sky",
            "object_ids": [],
        },
    ]
    vos = pipe._build_region_visual_objects(rmeta, lm)
    assert len(vos) == 1
    assert vos[0]["label"] == "sky"
    assert vos[0]["canonical_name"] == "sky"


# ---------------------------------------------------------------------------
# Phase 6 — Region adjacency graph
# ---------------------------------------------------------------------------

def _load_su_pipe():
    su = REPO / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_citv_su_adj_test", su)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pipe = mod.SceneUnderstandingPipeline.__new__(mod.SceneUnderstandingPipeline)
    return pipe


def test_region_adjacency_graph_detects_shared_border():
    """Two side-by-side regions must produce exactly one adjacency edge."""
    import cv2  # noqa: F401 — needed inside pipe method
    pipe = _load_su_pipe()

    # 100×100 label map: left half = region 1, right half = region 2
    lm = np.zeros((100, 100), dtype=np.int32)
    lm[:, :50] = 1
    lm[:, 50:] = 2
    metas = [
        {"region_index": 1, "id": "region_1", "depth_stats": {"mean": 1.5}},
        {"region_index": 2, "id": "region_2", "depth_stats": {"mean": 5.0}},
    ]
    result = pipe._build_region_adjacency_graph(metas, lm, min_border_px=5, dilation_px=3)
    assert result["num_edges"] == 1
    edge = result["edges"][0]
    assert edge["edge_type"] == "region_spatial_adjacency"
    assert edge["shared_border_px"] > 0
    assert set([edge["region_a"], edge["region_b"]]) == {"region_1", "region_2"}
    assert edge["depth_relation"] in ("a_in_front", "b_in_front")


def test_region_adjacency_graph_no_edge_for_non_adjacent():
    """Regions separated by a void gap must produce zero edges."""
    pipe = _load_su_pipe()

    # label map: region 1 on far left, region 2 on far right, 20px void in centre
    lm = np.zeros((100, 100), dtype=np.int32)
    lm[:, :30] = 1
    lm[:, 70:] = 2
    metas = [
        {"region_index": 1, "id": "region_1", "depth_stats": {"mean": 1.0}},
        {"region_index": 2, "id": "region_2", "depth_stats": {"mean": 4.0}},
    ]
    # dilation_px=3 means 7px kernel — still 20px gap so no overlap
    result = pipe._build_region_adjacency_graph(metas, lm, min_border_px=5, dilation_px=3)
    assert result["num_edges"] == 0


# ---------------------------------------------------------------------------
# Phase 4 — Object-only quantile pool in _build_layers_payload
# ---------------------------------------------------------------------------

def test_layers_payload_object_only_quantiles():
    """Objects at 1-3 m must not be pushed to midground/background by far region means."""
    pipe = _load_su_pipe()

    # Three objects clustered at 1–3 m
    objs = [
        {"id": "o1", "coordinates_3d": {"z": 1.0}, "layer_type": "unassigned"},
        {"id": "o2", "coordinates_3d": {"z": 2.0}, "layer_type": "unassigned"},
        {"id": "o3", "coordinates_3d": {"z": 3.0}, "layer_type": "unassigned"},
    ]
    # Regions with means at 10, 15, 20 m — formerly would shift quantiles dramatically
    metas = [
        {"region_index": 1, "id": "region_1", "depth_stats": {"mean": 10.0}, "object_ids": []},
        {"region_index": 2, "id": "region_2", "depth_stats": {"mean": 15.0}, "object_ids": []},
        {"region_index": 3, "id": "region_3", "depth_stats": {"mean": 20.0}, "object_ids": []},
    ]
    layers = pipe._build_layers_payload(objs, metas)
    # With object-only quantiles (q1≈1.67m, q2≈2.33m), the three objects should span
    # all three bands — no longer collapsed into foreground by far region depths.
    layer_types = {o["id"]: o["layer_type"] for o in objs}
    assert layer_types["o1"] == "foreground"
    assert layer_types["o3"] == "background"
    # Quantiles must be derived from [1,2,3] m — well below 10 m
    assert layers["depth_quantiles"]["foreground_max_z"] < 5.0
    assert layers["depth_quantiles"]["midground_max_z"] < 5.0


# ---------------------------------------------------------------------------
# Phase 7 — Mask-based adjacent_to in _build_region_region_relations_from_meta
# ---------------------------------------------------------------------------

def test_mask_adjacent_to_fires_for_touching_regions():
    """adjacent_to must fire when regions share a real mask border."""
    pipe = _load_su_pipe()
    pipe.config = None  # triggers defaults in _build_region_region_relations_from_meta

    lm = np.zeros((60, 60), dtype=np.int32)
    lm[:, :30] = 1
    lm[:, 30:] = 2
    metas = [
        {
            "region_index": 1, "id": "region_1", "type": "foreground",
            "depth_stats": {"mean": 1.0}, "centroid_2d_px": [15.0, 30.0],
            "bbox_px": [0, 0, 29, 59],
        },
        {
            "region_index": 2, "id": "region_2", "type": "background",
            "depth_stats": {"mean": 8.0}, "centroid_2d_px": [45.0, 30.0],
            "bbox_px": [30, 0, 59, 59],
        },
    ]
    edges = pipe._build_region_region_relations_from_meta(metas, lm)
    adj_edges = [e for e in edges if e["predicate"] == "adjacent_to"]
    assert len(adj_edges) >= 2  # symmetric pair
    # shared_border_px must be present and positive (Phase 7 enhancement)
    assert all("shared_border_px" in e for e in adj_edges)
    assert all(e["shared_border_px"] > 0 for e in adj_edges)
