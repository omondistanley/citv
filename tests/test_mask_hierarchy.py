"""Tests for mask containment hierarchy extraction."""

import numpy as np

from scene_understanding.regions.mask_hierarchy import build_mask_hierarchy, mask_area


def test_mask_area_empty_and_blob():
    assert mask_area(None) == 0
    m = np.zeros((5, 5), dtype=np.uint8)
    m[1:4, 1:4] = 1
    assert mask_area(m) == 9


def test_build_mask_hierarchy_parent_contains_child():
    parent = np.zeros((20, 20), dtype=np.uint8)
    parent[:, :] = 1
    child = np.zeros((20, 20), dtype=np.uint8)
    child[8:12, 8:12] = 1

    objects = [
        {
            "id": "p1",
            "entity_kind": "object",
            "sam2_mask_index": 0,
            "_sam2_mask_array": parent,
        },
        {
            "id": "c1",
            "entity_kind": "object",
            "sam2_mask_index": 1,
            "_sam2_mask_array": child,
        },
    ]

    out = build_mask_hierarchy(objects)
    assert out["num_edges"] == 1
    assert out["edges"][0]["parent_object_id"] == "p1"
    assert out["edges"][0]["child_object_id"] == "c1"
    assert out["edges"][0]["edge_type"] == "object_object_part"
    assert objects[1]["parent_object_id"] == "p1"
    assert "c1" in (objects[0].get("child_object_ids") or [])


def test_region_region_edges_disabled_skips_second_region_pair():
    r1 = np.zeros((10, 10), dtype=np.uint8)
    r1[:, :] = 1
    r2 = np.zeros((10, 10), dtype=np.uint8)
    r2[2:8, 2:8] = 1

    objects = [
        {"id": "reg_a", "entity_kind": "region", "sam2_mask_index": 0, "_sam2_mask_array": r1},
        {"id": "reg_b", "entity_kind": "region", "sam2_mask_index": 1, "_sam2_mask_array": r2},
    ]
    out = build_mask_hierarchy(objects, hierarchy_enable_region_region_edges=False)
    assert out["num_edges"] == 0
