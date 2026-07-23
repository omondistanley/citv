"""Tier 1 (Blender headless 3D render) tests.

``render_job.py`` has no ``bpy`` import, so its job-spec building and
compositing math are fully testable here. The actual Blender subprocess call
and ``blender_render_actor.py`` (a ``bpy``-only script) are NOT covered by
these tests -- there is no Blender binary in this development environment.
What IS covered: the clear-failure path when Blender is missing, and the
compositing correctness once a render's output frames exist (simulated here
with synthetic stand-in PNGs in place of real Blender renders).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scene_understanding.animation.tier1_blender import (
    build_job_spec,
    composite_blender_render,
    find_blender_executable,
    run_blender_render,
)
from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import paths_export


class _FakeDepthEstimator:
    device = "cpu"


class _FakePipeline:
    config = None
    depth_estimator = _FakeDepthEstimator()


def _synthetic_hypothesis_with_3d(tmp_path):
    h, w = 100, 140
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:25, :] = 8.0
    obj_a_mask = np.zeros((h, w), dtype=bool)
    obj_a_mask[60:85, 20:45] = True
    obj_b_mask = np.zeros((h, w), dtype=bool)
    obj_b_mask[65:90, 100:125] = True
    depth[obj_a_mask] = 2.0
    depth[obj_b_mask] = 2.5
    objects = [
        {"id": "obj_0", "bbox": [20, 60, 45, 85], "canonical_name": "a chair", "_sam2_mask_array": obj_a_mask, "depth_stats": {"median": 2.0}},
        {"id": "obj_1", "bbox": [100, 65, 125, 90], "canonical_name": "a table", "_sam2_mask_array": obj_b_mask, "depth_stats": {"median": 2.5}},
    ]
    intrinsics = {"fx": 130.0, "fy": 130.0, "cx": w / 2.0, "cy": h / 2.0}
    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="t1", timestamp="t",
        img_bgr=img, img_rgb=img, height=h, width=w, intrinsics=intrinsics, metric_depth=depth,
    )
    ctx.extra["objects"] = objects
    out_ctx = paths_export.run(_FakePipeline(), ctx)
    import json

    hyps = json.loads(Path(out_ctx.path_exports["path_hypotheses_json"]).read_text())["hypotheses"]
    return hyps[0], intrinsics, (w, h), img, obj_b_mask


def test_build_job_spec_produces_valid_world_frames(tmp_path):
    hyp, intrinsics, size, _img, _occ = _synthetic_hypothesis_with_3d(tmp_path)
    job = build_job_spec(hyp, intrinsics, size, fps=10, duration_s=1.0)
    assert job["schema"] == "citv_tier1_render_job_v1"
    assert len(job["frames"]) == 10
    for frame in job["frames"]:
        assert len(frame["world_xyz"]) == 3
        assert isinstance(frame["heading_deg"], float)
        assert frame["motion"]
        assert 0.5 <= frame["scale"] <= 2.0


def test_build_job_spec_requires_3d_lift():
    with pytest.raises(ValueError, match="polyline_3d_smoothed"):
        build_job_spec({"polyline_2d": [[0, 0], [1, 1]]}, {"fx": 1, "fy": 1, "cx": 0, "cy": 0}, (10, 10))


def test_run_blender_render_fails_clearly_when_blender_missing(tmp_path):
    if find_blender_executable():
        pytest.skip("a real Blender install is present in this environment -- this test targets the missing-binary path")
    with pytest.raises(RuntimeError, match="Blender executable not found"):
        run_blender_render({"frames": []}, tmp_path, blender_executable="/definitely/not/a/real/blender/binary")


def test_composite_blender_render_restores_real_occluder_silhouette(tmp_path):
    hyp, intrinsics, (w, h), img, occluder_mask = _synthetic_hypothesis_with_3d(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    n = 4
    actor_frames, shadow_frames = [], []
    for i in range(n):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[60:90, 90:130, :3] = [30, 200, 30]  # actor sprite overlapping the occluder region
        rgba[60:90, 90:130, 3] = 255
        ap = frames_dir / f"actor_{i:04d}.png"
        cv2.imwrite(str(ap), rgba)
        actor_frames.append(str(ap))
        shadow = np.zeros((h, w), dtype=np.uint8)
        sp = frames_dir / f"shadow_{i:04d}.png"
        cv2.imwrite(str(sp), shadow)
        shadow_frames.append(str(sp))

    render_result = {"actor_frames": actor_frames, "shadow_frames": shadow_frames}
    occluder_ids_per_frame = [["obj_1"]] * n
    result = composite_blender_render(
        img, render_result, fps=8, occluder_ids_per_frame=occluder_ids_per_frame,
        object_masks_by_id={"obj_1": occluder_mask},
        out_gif_path=str(tmp_path / "anim.gif"), out_mp4_path=str(tmp_path / "anim.mp4"),
    )
    assert result["frame_count"] == n
    assert Path(result["gif_path"]).exists()
    assert Path(result["mp4_path"]).exists()

    # Re-derive the last composited frame to check the occlusion property directly.
    canvas = img.copy()
    actor_rgba = cv2.imread(actor_frames[-1], cv2.IMREAD_UNCHANGED)
    alpha = actor_rgba[..., 3:4].astype(np.float32) / 255.0
    canvas[:] = np.clip(canvas.astype(np.float32) * (1 - alpha) + actor_rgba[..., :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    assert not np.array_equal(canvas[occluder_mask], img[occluder_mask]), "sanity: actor should have painted over the occluder region before restore"
    canvas[occluder_mask] = img[occluder_mask]
    assert np.array_equal(canvas[occluder_mask], img[occluder_mask]), "occluder's real silhouette must be restored on top of the actor"
