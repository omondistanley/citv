"""Unit tests for stages/labelling.py's near-duplicate mask merge -- the
"AMG capability without AMG" dedup pass added alongside the naming
evidence-fusion fix (Phase 1 item 10 in the plan)."""
from __future__ import annotations

import numpy as np

from scene_understanding.stages.labelling import (
    _combine_object_evidence,
    _mask_iou,
    _merge_duplicate_objects,
)


def _mask(h: int, w: int, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_mask_iou_identical_masks_is_one():
    m = _mask(20, 20, 2, 10, 2, 10)
    assert _mask_iou(m, m) == 1.0


def test_mask_iou_disjoint_masks_is_zero():
    a = _mask(20, 20, 0, 5, 0, 5)
    b = _mask(20, 20, 15, 20, 15, 20)
    assert _mask_iou(a, b) == 0.0


def test_mask_iou_mismatched_shapes_is_zero():
    a = _mask(20, 20, 0, 5, 0, 5)
    b = _mask(10, 10, 0, 5, 0, 5)
    assert _mask_iou(a, b) == 0.0


def test_combine_object_evidence_merges_aliases_and_prefers_higher_confidence():
    a = {"id": "obj_0", "label": "furniture", "conf": 0.4, "canonical_name": "furniture",
         "caption": "furniture", "category": "furniture", "aliases": ["furniture"]}
    b = {"id": "obj_1", "label": "rocking chair", "conf": 0.8, "canonical_name": "rocking chair",
         "caption": "a wooden rocking chair", "category": "chair", "aliases": ["rocking chair", "chair"]}
    merged = _combine_object_evidence(a, b)
    # Higher-confidence source (b) wins on the naming fields.
    assert merged["canonical_name"] == "rocking chair"
    assert merged["label"] == "rocking chair"
    assert merged["conf"] == 0.8
    assert merged["caption"] == "a wooden rocking chair"
    # Aliases combine from both sources, no duplicates.
    assert merged["aliases"] == ["furniture", "rocking chair", "chair"]
    assert merged["merged_duplicate_ids"] == ["obj_1"]
    # a's own identity (id) is preserved -- b is absorbed, not the other way round.
    assert merged["id"] == "obj_0"


def test_combine_object_evidence_keeps_a_when_a_more_confident():
    a = {"id": "obj_0", "label": "rocking chair", "conf": 0.9, "canonical_name": "rocking chair",
         "caption": "a rocking chair", "aliases": ["rocking chair"]}
    b = {"id": "obj_1", "label": "furniture", "conf": 0.3, "canonical_name": "furniture",
         "caption": "furniture", "aliases": ["furniture"]}
    merged = _combine_object_evidence(a, b)
    assert merged["canonical_name"] == "rocking chair"
    assert merged["conf"] == 0.9


def test_merge_duplicate_objects_merges_near_identical_masks():
    mask_a = _mask(30, 30, 2, 20, 2, 20)
    mask_b = _mask(30, 30, 2, 21, 2, 21)  # 1px larger on each far edge -- still very high IoU
    objects = [
        {"id": "obj_0", "label": "furniture", "conf": 0.4, "canonical_name": "furniture",
         "aliases": ["furniture"], "_sam2_mask_array": mask_a},
        {"id": "obj_1", "label": "sofa", "conf": 0.85, "canonical_name": "sofa",
         "aliases": ["sofa"], "_sam2_mask_array": mask_b},
    ]
    merged = _merge_duplicate_objects(objects, iou_threshold=0.85)
    assert len(merged) == 1
    assert merged[0]["canonical_name"] == "sofa"
    assert merged[0]["merged_duplicate_ids"] == ["obj_1"]


def test_merge_duplicate_objects_leaves_distinct_objects_untouched():
    mask_a = _mask(30, 30, 0, 10, 0, 10)
    mask_b = _mask(30, 30, 20, 30, 20, 30)
    objects = [
        {"id": "obj_0", "label": "chair", "conf": 0.7, "_sam2_mask_array": mask_a},
        {"id": "obj_1", "label": "table", "conf": 0.7, "_sam2_mask_array": mask_b},
    ]
    merged = _merge_duplicate_objects(objects, iou_threshold=0.85)
    assert len(merged) == 2
    assert {o["id"] for o in merged} == {"obj_0", "obj_1"}


def test_merge_duplicate_objects_handles_missing_mask_array_gracefully():
    objects = [
        {"id": "obj_0", "label": "chair", "conf": 0.7, "_sam2_mask_array": None},
        {"id": "obj_1", "label": "table", "conf": 0.7, "_sam2_mask_array": _mask(10, 10, 0, 5, 0, 5)},
    ]
    merged = _merge_duplicate_objects(objects, iou_threshold=0.85)
    assert len(merged) == 2
