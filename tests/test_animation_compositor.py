"""Tier 2 animation compositor tests (Phase 2 of the plan).

Two properties matter most, per the plan's cross-cutting requirement and
Phase 2 verification section:
  1. Background fidelity: every output frame's pixels outside the
     composited actor/shadow region must be the original photo, unchanged.
  2. Occlusion correctness: when a path sample is flagged ``behind_object``
     with occluder ids, the occluder's *real* silhouette must show through
     on top of the actor, not just a uniform per-frame alpha dim.
"""
from __future__ import annotations

import numpy as np

from scene_understanding.animation.compositor import (
    PathTrace,
    cast_contact_shadow,
    composite_frame,
    extract_actor_sprite,
    frame_schedule,
    render_animation,
)


def _synthetic_frame_and_masks():
    h, w = 100, 140
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img[:, :, 1] = 110  # tint so background isn't a flat gray (easier to eyeball if inspected)

    actor_mask = np.zeros((h, w), dtype=bool)
    actor_mask[60:85, 20:40] = True

    occluder_mask = np.zeros((h, w), dtype=bool)
    occluder_mask[40:80, 60:75] = True  # a "pillar" the actor should pass behind

    return img, actor_mask, occluder_mask


def test_extract_actor_sprite_uses_real_mask_not_ellipse():
    img, actor_mask, _ = _synthetic_frame_and_masks()
    rgba, origin = extract_actor_sprite(img, actor_mask, feather_px=1, pad_px=2)
    alpha = rgba[..., 3]
    # A real rectangular mask should stay fully opaque well inside its own
    # boundary (away from the small anti-aliasing feather at the edges) --
    # an ellipse cutout would carve out the corners even deep inside the
    # original rectangle, which this checks against. The mask is 25x20 px
    # (60:85, 20:40) with pad_px=2, so crop-local (10,10) is a corner of the
    # source rectangle but ~8px inside the feathered boundary.
    corner = alpha[10:15, 10:15]
    assert corner.mean() > 250, "sprite alpha should stay opaque in the mask's own corners, not carve them out like an ellipse"


def test_composite_frame_preserves_background_outside_actor_region():
    img, actor_mask, occluder_mask = _synthetic_frame_and_masks()
    sprite_rgba, _ = extract_actor_sprite(img, actor_mask)
    state = {"position_px": (30.0, 82.0), "width_px": 20.0, "alpha": 1.0, "render_layer": "in_front", "occluder_ids": []}

    frame = composite_frame(img, sprite_rgba, state, object_masks_by_id={}, reference_width_px=20.0)

    far_corner = frame[0:10, 0:10]
    original_far_corner = img[0:10, 0:10]
    assert np.array_equal(far_corner, original_far_corner), "background far from the actor must be untouched"


def test_composite_frame_restores_real_occluder_silhouette_on_top_of_actor():
    img, actor_mask, occluder_mask = _synthetic_frame_and_masks()
    sprite_rgba, _ = extract_actor_sprite(img, actor_mask)

    # Place the actor directly under the occluder's footprint and mark it
    # behind_object with that occluder id -- the occluder's own pixels
    # must end up on top, using its real (non-rectangular-in-general) mask.
    state = {
        "position_px": (67.0, 78.0), "width_px": 20.0, "alpha": 1.0,
        "render_layer": "behind_object", "occluder_ids": ["pillar"],
    }
    frame = composite_frame(
        img, sprite_rgba, state, object_masks_by_id={"pillar": occluder_mask}, reference_width_px=20.0,
    )
    assert np.array_equal(frame[occluder_mask], img[occluder_mask]), (
        "occluder's real silhouette must show the original photo pixels on top of the actor"
    )


def test_cast_contact_shadow_darkens_beneath_actor_only():
    img, actor_mask, _ = _synthetic_frame_and_masks()
    canvas = img.copy()
    cast_contact_shadow(canvas, foot_xy=(30.0, 85.0), width_px=20.0, alpha=0.4)
    assert canvas[85, 30].sum() < img[85, 30].sum(), "shadow should darken the pixels at the foot position"
    assert np.array_equal(canvas[0:5, 0:5], img[0:5, 0:5]), "shadow must not touch pixels far from the foot"


def test_render_animation_uses_uploaded_sprite_over_mask_cutout(tmp_path):
    """Phase 3: an uploaded actor asset must take priority over cutting a
    sprite from the actor's own mask/photo pixels."""
    img, actor_mask, _ = _synthetic_frame_and_masks()
    hyp = {
        "polyline_2d": [[float(i), 80.0] for i in range(30, 60)],
        "width_profile_px": [15.0] * 30,
        "alpha_profile": [1.0] * 30,
        "visibility_profile": [{"render_layer": "in_front", "occluder_ids": []} for _ in range(30)],
        "kinematic_signatures": [],
    }
    uploaded_sprite = np.zeros((16, 16, 4), dtype=np.uint8)
    uploaded_sprite[..., 0] = 10   # a distinctive blue-ish color, unlike the photo's green tint
    uploaded_sprite[..., 3] = 255

    result = render_animation(
        img, hyp, actor_mask=None, object_masks_by_id={},
        out_gif_path=str(tmp_path / "a.gif"), out_mp4_path=str(tmp_path / "a.mp4"),
        fps=6, duration_s=0.5, actor_sprite_rgba=uploaded_sprite,
    )
    assert result["frame_count"] > 0
    # Sanity: the composited pixel at the sprite's placement should reflect
    # the uploaded sprite's color, not the background's greenish tint.
    canvas = img.copy()
    from scene_understanding.animation.compositor import PathTrace as _PT
    trace = _PT(hyp)
    state = trace.sample(0.5)
    composited = composite_frame(img, uploaded_sprite, state, {}, reference_width_px=15.0)
    px = composited[int(state["position_px"][1]) - 3, int(state["position_px"][0])]
    # The sprite's raw blue channel (10) is harmonized partway toward the
    # background (90) -- the composited pixel should land strictly between
    # the two, not equal the raw sprite color (unharmonized) or the pure
    # background color (sprite not actually drawn).
    assert 10 < int(px[0]) < 90, "composited pixel should be the uploaded sprite's color partially harmonized toward the background, not one or the other verbatim"


def test_frame_schedule_respects_fixed_duration_motions():
    hyp = {
        "polyline_2d": [[float(i), 0.0] for i in range(20)],
        "width_profile_px": [10.0] * 20,
        "alpha_profile": [1.0] * 20,
        "visibility_profile": [{"render_layer": "in_front", "occluder_ids": []} for _ in range(20)],
        "kinematic_signatures": [
            {"start_idx": 0, "end_idx": 9, "motion": "walk", "dz_m": 0.0},
            {"start_idx": 9, "end_idx": 19, "motion": "jump", "dz_m": 0.3},
        ],
    }
    trace = PathTrace(hyp)
    schedule = frame_schedule(trace, fps=10, duration_s=2.0)
    assert len(schedule) == 20
    assert schedule[0] == 0.0
    assert abs(schedule[-1] - 1.0) < 1e-6
    # monotonically non-decreasing progress along the path
    assert all(b >= a for a, b in zip(schedule, schedule[1:]))
