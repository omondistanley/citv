"""Phase 3 end-to-end tests through app.py's animate_scene -- the same four
verification scenarios the plan calls out, exercised through the actual
Gradio-facing function rather than the motion_contract module directly, to
catch integration bugs the unit tests wouldn't (e.g. the RGB->BGR asset
conversion, tier fallback wiring, dropdown label->id resolution).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import paths_export

import app


class _FakeDepthEstimator:
    device = "cpu"


class _FakePipeline:
    config = None
    depth_estimator = _FakeDepthEstimator()


def _analyzed_state(tmp_path):
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
    out_ctx = paths_export.run(_FakePipeline(), ctx)
    labels, id_by_label = app._object_labels(objects)
    import json

    hyps = json.loads(Path(out_ctx.path_exports["path_hypotheses_json"]).read_text())["hypotheses"]
    return {"ctx": out_ctx, "hypotheses": hyps, "id_by_label": id_by_label}, labels


def test_drawn_path_with_no_actor_fails_clearly_not_silently(tmp_path):
    state, _labels = _analyzed_state(tmp_path)
    drawn = [(20.0, 90.0), (60.0, 90.0), (100.0, 85.0), (120.0, 80.0)]
    gif, mp4, status = app.animate_scene(
        state, None, None, 10, 1.0, app.TIER_2D, drawn, "", "", None, None, [], [], [],
    )
    assert gif is None and mp4 is None
    assert "actor" in status.lower()


def test_text_only_actor_reports_tier2_unavailable(tmp_path):
    state, _labels = _analyzed_state(tmp_path)
    gif, mp4, status = app.animate_scene(
        state, None, None, 10, 1.0, app.TIER_2D, [], "a photorealistic red fox", "trots to the table", None, None, [], [], [],
    )
    assert gif is None and mp4 is None
    assert "Tier 2" in status and "unavailable" in status.lower()


def test_uploaded_actor_with_avoid_constraint_renders(tmp_path):
    state, labels = _analyzed_state(tmp_path)
    uploaded = np.zeros((20, 20, 4), dtype=np.uint8)
    uploaded[..., 3] = 255
    gif, mp4, status = app.animate_scene(
        state, labels[0], labels[1], 10, 1.0, app.TIER_2D, [], "", "", uploaded, None, [labels[2]], [], [],
    )
    assert gif is not None and Path(gif).exists()
    assert mp4 is not None and Path(mp4).exists()
    assert "Tier 2" in status


def test_fully_auto_renders_with_grounded_actor(tmp_path):
    state, _labels = _analyzed_state(tmp_path)
    gif, mp4, status = app.animate_scene(
        state, None, None, 10, 1.0, app.TIER_2D, [], "", "", None, None, [], [], [],
    )
    assert gif is not None and Path(gif).exists()
    assert "grounded" in status.lower() or "Tier 2" in status


def test_render_status_includes_qa_note(tmp_path):
    """Phase 2.5 item 3: render_qa's pass/fail summary must reach the
    Gradio status text, same pattern as the existing report_note append."""
    state, labels = _analyzed_state(tmp_path)
    gif, mp4, status = app.animate_scene(
        state, labels[0], labels[1], 10, 1.0, app.TIER_2D, [], "", "", None, None, [], [], [],
    )
    assert gif is not None and Path(gif).exists()
    assert "QA:" in status
