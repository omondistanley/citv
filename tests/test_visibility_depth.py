"""Unit tests for paths_export.py's depth-aware occlusion fix (Phase 2.6
item 5): _visibility_profile must not mark a path point "behind_object"
just because some other object's mask happens to share that pixel -- only
when that object is actually closer to the camera than the path point
there."""
from __future__ import annotations

import numpy as np

from scene_understanding.stages.paths_export import _visibility_profile


def _mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _obj(oid, mask, z_val):
    return {"id": oid, "_sam2_mask_array": mask, "depth_stats": {"z_val": z_val}}


H, W = 40, 40


def test_closer_occluder_marks_behind_object():
    # Occluder is closer to camera (z=1.0) than the path point (z=5.0) --
    # a genuine occluder, should mark "behind_object".
    occ_mask = _mask(H, W, 15, 25, 15, 25)
    objects = [_obj("occ_0", occ_mask, z_val=1.0)]
    polyline_2d = [[20.0, 20.0]]
    polyline_uvz = [[20.0, 20.0, 5.0]]
    profile, alpha, layers = _visibility_profile(polyline_2d, polyline_uvz, objects, "src", "tgt", H, W)
    assert profile[0]["render_layer"] == "behind_object"
    assert alpha[0] == 0.15
    assert profile[0]["occluder_ids"] == ["occ_0"]


def test_farther_object_does_not_mark_behind_object():
    # "Occluder" is actually farther (z=8.0) than the path point (z=2.0) --
    # not a real occluder even though the mask overlaps the same pixel
    # (e.g. a flying path passing in front of a distant tree).
    occ_mask = _mask(H, W, 15, 25, 15, 25)
    objects = [_obj("occ_0", occ_mask, z_val=8.0)]
    polyline_2d = [[20.0, 20.0]]
    polyline_uvz = [[20.0, 20.0, 2.0]]
    profile, alpha, layers = _visibility_profile(polyline_2d, polyline_uvz, objects, "src", "tgt", H, W)
    assert profile[0]["render_layer"] != "behind_object"
    assert alpha[0] != 0.15


def test_missing_depth_evidence_falls_back_to_partially_occluded():
    # Mask overlaps, but the occluder has no known depth -- not enough
    # evidence to claim full occlusion; falls back to the softer signal.
    occ_mask = _mask(H, W, 15, 25, 15, 25)
    objects = [_obj("occ_0", occ_mask, z_val=None)]
    polyline_2d = [[20.0, 20.0]]
    polyline_uvz = [[20.0, 20.0, 5.0]]
    profile, alpha, layers = _visibility_profile(polyline_2d, polyline_uvz, objects, "src", "tgt", H, W)
    assert profile[0]["render_layer"] == "partially_occluded"
    assert alpha[0] == 0.55
    assert profile[0]["occluder_ids"] == ["occ_0"]


def test_no_mask_overlap_is_in_front():
    occ_mask = _mask(H, W, 0, 5, 0, 5)
    objects = [_obj("occ_0", occ_mask, z_val=1.0)]
    polyline_2d = [[30.0, 30.0]]
    polyline_uvz = [[30.0, 30.0, 5.0]]
    profile, alpha, layers = _visibility_profile(polyline_2d, polyline_uvz, objects, "src", "tgt", H, W)
    assert profile[0]["render_layer"] == "in_front"
    assert alpha[0] == 1.0
    assert profile[0]["occluder_ids"] == []


def test_src_and_tgt_objects_excluded_from_occluders():
    occ_mask = _mask(H, W, 15, 25, 15, 25)
    objects = [_obj("src", occ_mask, z_val=1.0)]
    polyline_2d = [[20.0, 20.0]]
    polyline_uvz = [[20.0, 20.0, 5.0]]
    profile, alpha, layers = _visibility_profile(polyline_2d, polyline_uvz, objects, "src", "tgt", H, W)
    assert profile[0]["render_layer"] == "in_front"
