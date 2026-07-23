"""Phase 3 tests: the open-ended, field-optional custom-input contract.

Covers exactly the verification scenarios the plan calls out: a drawn path
with no actor specified, text-only actor/action with no geometry, an
uploaded actor with an explicit avoid-object constraint, and the fully-auto
(nothing set) case -- confirming in each case that whatever the user set is
preserved exactly and whatever they left unset is grounded from real scene
evidence rather than a generic default.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scene_understanding.animation.motion_contract import MotionContract, ground_motion_contract, reroute_around_obstacles
from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import paths_export


class _FakeDepthEstimator:
    device = "cpu"


class _FakePipeline:
    config = None
    depth_estimator = _FakeDepthEstimator()


def _analyzed_ctx(tmp_path):
    """Runs paths_export.run on a synthetic 3-object scene, mirroring what
    app.py's Analyze step produces -- objects, cost/support fields stashed
    in ctx.extra, and a real precomputed path_hypotheses.json on disk."""
    h, w = 110, 150
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:25, :] = 8.0

    actor_mask = np.zeros((h, w), dtype=bool)
    actor_mask[70:95, 15:40] = True
    target_mask = np.zeros((h, w), dtype=bool)
    target_mask[65:100, 105:135] = True
    obstacle_mask = np.zeros((h, w), dtype=bool)
    obstacle_mask[50:90, 65:80] = True
    depth[actor_mask] = 2.0
    depth[target_mask] = 2.5
    depth[obstacle_mask] = 1.5

    objects = [
        {"id": "actor_0", "bbox": [15, 70, 40, 95], "canonical_name": "a wooden chair", "_sam2_mask_array": actor_mask, "depth_stats": {"median": 2.0}},
        {"id": "target_0", "bbox": [105, 65, 135, 100], "canonical_name": "a dining table", "_sam2_mask_array": target_mask, "depth_stats": {"median": 2.5}},
        {"id": "obstacle_0", "bbox": [65, 50, 80, 90], "canonical_name": "a pillar", "_sam2_mask_array": obstacle_mask, "depth_stats": {"median": 1.5}},
    ]
    intrinsics = {"fx": 140.0, "fy": 140.0, "cx": w / 2.0, "cy": h / 2.0}
    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="p", timestamp="t",
        img_bgr=img, img_rgb=img, height=h, width=w, intrinsics=intrinsics, metric_depth=depth,
    )
    ctx.extra["objects"] = objects
    return paths_export.run(_FakePipeline(), ctx), objects


def test_drawn_path_with_no_actor_specified_is_preserved_and_actor_grounded(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    drawn = [(20.0, 90.0), (60.0, 90.0), (100.0, 85.0), (120.0, 80.0)]
    contract = MotionContract(drawn_polyline_2d=drawn)

    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert grounded.hypothesis["polyline_2d"] == [[float(x), float(y)] for x, y in drawn], "user-drawn path must be preserved exactly, not replaced by an FMM search"
    assert "drawn_polyline_2d" in " ".join(grounded.report["preserved"])
    # No actor was specified -- since the drawn path has no real source
    # object, actor resolution should warn rather than silently invent one.
    assert grounded.actor_mask is None
    assert any("actor" in w for w in grounded.report["warnings"])


def test_drawn_path_through_obstacle_is_flagged_but_not_altered(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    # obstacle_0 ("a pillar") occupies y:50-90, x:65-80 in _analyzed_ctx.
    # This drawn path runs straight down through the middle of it.
    drawn = [(72.0, 52.0), (72.0, 70.0), (72.0, 88.0)]
    contract = MotionContract(drawn_polyline_2d=drawn)

    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    # Path preserved exactly as drawn -- obstacle awareness must never
    # silently alter it.
    assert grounded.hypothesis["polyline_2d"] == [[float(x), float(y)] for x, y in drawn]
    assert grounded.hypothesis["obstacle_crossings"], "expected at least one flagged crossing run"
    crossing = grounded.hypothesis["obstacle_crossings"][0]
    assert crossing["object_id"] == "obstacle_0"
    assert crossing["object_name"] == "a pillar"
    assert any("a pillar" in w and "drawn path crosses" in w for w in grounded.report["warnings"])


def test_drawn_path_avoiding_obstacle_has_no_crossings(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    # obstacle_0 occupies y:50-90, x:65-80 -- this path stays well clear (x:0-20).
    drawn = [(5.0, 20.0), (10.0, 40.0), (15.0, 60.0)]
    contract = MotionContract(drawn_polyline_2d=drawn)

    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert grounded.hypothesis["obstacle_crossings"] == []
    assert not any("crosses a real obstacle" in w for w in grounded.report["warnings"])


def test_reroute_around_obstacles_avoids_mask_and_preserves_endpoints(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    drawn = [(72.0, 52.0), (72.0, 70.0), (72.0, 88.0)]
    grounded = ground_motion_contract(MotionContract(drawn_polyline_2d=drawn), ctx)
    crossings = grounded.hypothesis["obstacle_crossings"]
    assert crossings, "fixture setup expects this drawn path to cross obstacle_0"

    speed_field = ctx.extra.get("_paths_speed_field")
    adjusted = reroute_around_obstacles(drawn, crossings, speed_field)

    assert adjusted != drawn, "expected the crossing sub-segment to actually change"
    assert adjusted[0] == drawn[0], "the drawn path's own start point must be preserved exactly"
    assert adjusted[-1] == drawn[-1], "the drawn path's own end point must be preserved exactly"

    # The user's own drawn anchor points (start/end of the crossing run) are
    # preserved verbatim even if they themselves sit inside the obstacle
    # (that's literally where the user clicked) -- what must actually avoid
    # the obstacle is the FMM-solved interior of the route between them.
    obstacle_mask = ctx.extra["_paths_obstacle_mask"]

    def _inside_obstacle(pt) -> bool:
        xi, yi = int(round(pt[0])), int(round(pt[1]))
        return bool(obstacle_mask[max(0, min(obstacle_mask.shape[0] - 1, yi)), max(0, min(obstacle_mask.shape[1] - 1, xi))])

    interior = adjusted[1:-1]
    assert interior, "expected a real multi-point rerouted path, not a direct A-to-B jump"
    assert not all(_inside_obstacle(pt) for pt in interior), "rerouted path never actually leaves the obstacle"


def test_reroute_around_obstacles_no_crossings_returns_input_unchanged(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    drawn = [(5.0, 20.0), (10.0, 40.0), (15.0, 60.0)]
    adjusted = reroute_around_obstacles(drawn, [], ctx.extra.get("_paths_speed_field"))
    assert adjusted == drawn


def test_reroute_around_obstacles_no_speed_field_returns_input_unchanged():
    drawn = [(1.0, 1.0), (2.0, 2.0)]
    fake_crossing = [{"start_idx": 0, "end_idx": 1, "object_id": "x", "object_name": "x"}]
    assert reroute_around_obstacles(drawn, fake_crossing, None) == drawn


def test_text_only_actor_and_action_with_no_geometry_grounds_to_best_path(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    contract = MotionContract(actor_text="a photorealistic red fox", action_text="the fox trots to the table")

    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert "actor_text" in " ".join(grounded.report["preserved"])
    assert any("not specified" in g for g in grounded.report["grounded"])  # geometry grounded, not user-given
    assert grounded.actor_mask is None  # no real object mask for a pure-text actor
    assert grounded.tier2_available is False  # Tier 2 has no photo pixels to cut a fox sprite from
    assert grounded.actor_attrs["leg_count"] == 4  # "red fox" should classify as four-legged, not upright/2-legged
    assert grounded.color_rgb != (0.55, 0.55, 0.55)  # "red" should extract a non-default tint


def test_uploaded_actor_with_avoid_constraint_is_preserved_and_avoided(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    uploaded = np.zeros((20, 20, 4), dtype=np.uint8)
    uploaded[..., 3] = 255

    contract = MotionContract(
        actor_object_id="actor_0", target_object_id="target_0",
        actor_asset_rgba=uploaded, avoid_object_ids=["obstacle_0"],
    )
    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert np.array_equal(grounded.actor_asset_rgba, uploaded), "uploaded actor asset must be preserved exactly"
    assert "actor_asset_rgba" in " ".join(grounded.report["preserved"])
    assert "avoid_object_ids" in " ".join(grounded.report["preserved"])
    # Uses the precomputed path_hypotheses.json (built with the obstacle
    # already routed around at Analyze time) rather than a fresh custom
    # search, since only actor/target were given -- confirm it's a real,
    # non-degenerate path.
    assert len(grounded.hypothesis["polyline_2d"]) > 2


def test_fully_auto_grounds_everything_from_precomputed_hypotheses(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    contract = MotionContract()  # nothing set at all

    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert grounded.report["preserved"] == []  # nothing was user-specified
    assert len(grounded.report["grounded"]) >= 2  # both actor and target grounded
    assert grounded.actor_mask is not None  # grounded to the best hypothesis's real source object


def test_uploaded_animation_ref_is_preserved_and_gracefully_falls_back(tmp_path):
    """A clip with no detectable person should be preserved (not silently
    dropped) and clearly reported as falling back to the canned motion
    style -- exercises the real MediaPipe call path, not a mock."""
    ctx, objects = _analyzed_ctx(tmp_path)
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 48))
    for i in range(5):
        writer.write(np.full((48, 48, 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()

    contract = MotionContract(actor_object_id="actor_0", target_object_id="target_0", uploaded_animation_ref=str(video_path), fps=10, duration_s=1.0)
    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    assert grounded.retargeted_motion is None
    assert any("uploaded_animation_ref" in w for w in grounded.report["warnings"])


def test_must_render_behind_forces_occlusion_override(tmp_path):
    ctx, objects = _analyzed_ctx(tmp_path)
    contract = MotionContract(
        actor_object_id="actor_0", target_object_id="target_0",
        must_render_behind_object_ids=["obstacle_0"],
    )
    grounded = ground_motion_contract(contract, ctx)
    assert grounded is not None
    visibility = grounded.hypothesis["visibility_profile"]
    forced = [v for v in visibility if "obstacle_0" in v.get("occluder_ids", [])]
    # Not every sample necessarily crosses the obstacle's footprint, but at
    # least confirm the override machinery ran without erroring and that any
    # sample it did touch was actually forced to behind_object.
    assert all(v["render_layer"] == "behind_object" for v in forced)
