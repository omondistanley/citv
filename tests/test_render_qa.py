"""Unit tests for scene_understanding/animation/render_qa.py (Phase 2.5 item 3):
one synthetic "good" case per check plus one deliberately-broken case."""
from __future__ import annotations

import numpy as np

from scene_understanding.animation.render_qa import (
    check_background_preserved,
    check_depth_scale_monotonicity,
    check_frame_count,
    check_occlusion_order_consistency,
    run_render_qa,
)

H, W = 40, 40


def _blank(h=H, w=W):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _states(n=6, base_z=5.0, z_step=-0.5, width0=10.0, width_step=0.5, render_layer="in_front", occluder_ids=None):
    out = []
    for i in range(n):
        out.append({
            "position_px": (20.0, 20.0),
            "width_px": width0 + i * width_step,
            "z_m": base_z + i * z_step,
            "render_layer": render_layer,
            "occluder_ids": occluder_ids or [],
        })
    return out


# --- frame_count ---

def test_frame_count_ok():
    ok, msg = check_frame_count([_blank()] * 24, fps=24, duration_s=1.0)
    assert ok, msg


def test_frame_count_broken():
    ok, msg = check_frame_count([_blank()] * 10, fps=24, duration_s=1.0)
    assert not ok


# --- background_preserved ---

def test_background_preserved_ok_when_only_protected_region_changes():
    original = _blank()
    frame = original.copy()
    frame[18:22, 18:22] = 255  # inside the protected circle around position_px=(20,20)
    states = _states(n=1)
    ok, msg = check_background_preserved([frame], original, states, {})
    assert ok, msg


def test_background_preserved_broken_when_pixel_changes_outside_protected_region():
    original = _blank()
    frame = original.copy()
    frame[0:3, 0:3] = 255  # far corner, well outside the actor's protected region
    states = _states(n=1)
    ok, msg = check_background_preserved([frame], original, states, {})
    assert not ok


# --- occlusion_order_consistency ---

def test_occlusion_order_consistency_ok_when_occluder_mask_overlaps_actor():
    occ_mask = np.zeros((H, W), dtype=bool)
    occ_mask[15:25, 15:25] = True  # covers position_px=(20,20)
    states = _states(n=1, render_layer="behind_object", occluder_ids=["occ_0"])
    ok, msg = check_occlusion_order_consistency(states, {"occ_0": occ_mask}, (H, W))
    assert ok, msg


def test_occlusion_order_consistency_broken_when_occluder_mask_is_elsewhere():
    occ_mask = np.zeros((H, W), dtype=bool)
    occ_mask[0:5, 0:5] = True  # nowhere near position_px=(20,20)
    states = _states(n=1, render_layer="behind_object", occluder_ids=["occ_0"])
    ok, msg = check_occlusion_order_consistency(states, {"occ_0": occ_mask}, (H, W))
    assert not ok


def test_occlusion_order_consistency_skips_in_front_frames():
    states = _states(n=3, render_layer="in_front", occluder_ids=["occ_0"])
    ok, msg = check_occlusion_order_consistency(states, {}, (H, W))
    assert ok, msg


# --- depth_scale_monotonicity ---

def test_depth_scale_monotonicity_ok_when_farther_is_smaller():
    # z_m increases (farther) while width_px decreases (smaller) -- correct.
    states = _states(n=8, base_z=2.0, z_step=0.5, width0=20.0, width_step=-1.5)
    ok, msg = check_depth_scale_monotonicity(states)
    assert ok, msg


def test_depth_scale_monotonicity_broken_when_farther_is_larger():
    # z_m increases (farther) while width_px also increases (larger) -- inverted, wrong.
    states = _states(n=8, base_z=2.0, z_step=0.5, width0=8.0, width_step=1.5)
    ok, msg = check_depth_scale_monotonicity(states)
    assert not ok


def test_depth_scale_monotonicity_skips_flat_depth():
    states = _states(n=8, base_z=3.0, z_step=0.0, width0=10.0, width_step=0.0)
    ok, msg = check_depth_scale_monotonicity(states)
    assert ok, msg


def test_depth_scale_monotonicity_skips_when_no_depth_data():
    states = [{"position_px": (20.0, 20.0), "width_px": 10.0, "z_m": None, "render_layer": "in_front", "occluder_ids": []}]
    ok, msg = check_depth_scale_monotonicity(states)
    assert ok, msg


# --- run_render_qa integration ---

def test_run_render_qa_passes_on_a_fully_consistent_render():
    original = _blank()
    n = 6
    states = _states(n=n, base_z=2.0, z_step=0.3, width0=14.0, width_step=-0.8, render_layer="in_front")
    frames = [original.copy() for _ in range(n)]
    result = run_render_qa(frames, original, states, {}, fps=6, duration_s=1.0)
    assert result["passed"], result["failed_checks"]
    assert len(result["checks"]) == 4


def test_run_render_qa_fails_when_frame_count_is_wrong():
    original = _blank()
    states = _states(n=3)
    frames = [original.copy() for _ in range(3)]
    result = run_render_qa(frames, original, states, {}, fps=24, duration_s=1.0)
    assert not result["passed"]
    assert any("frame_count" in msg for msg in result["failed_checks"])
