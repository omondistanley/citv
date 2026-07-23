"""Unit tests for app.py's drawn-path obstacle visual feedback (Phase 2.7
item 2): _segment_crosses_obstacle and _redraw_canvas's color choice, plus
item 3's auto_adjust_around_obstacles UI handler. Also covers Phase 2.8's
freehand-stroke vectorization (_vectorize_editor_stroke, on_stroke_drawn),
which replaced the old click-accumulation drawing mechanism."""
from __future__ import annotations

import cv2
import numpy as np

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import paths_export

import app


def _obstacle_mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class _FakeDepthEstimator:
    device = "cpu"


class _FakePipeline:
    config = None
    depth_estimator = _FakeDepthEstimator()


def _analyzed_ctx_with_obstacle(tmp_path):
    """Same synthetic scene shape used in test_motion_contract.py /
    test_app_animate.py: an obstacle ('a pillar') sits at y:50-90, x:65-80."""
    h, w = 110, 150
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    obstacle_mask = _obstacle_mask(h, w, 50, 90, 65, 80)
    depth[obstacle_mask] = 1.5
    objects = [{"id": "obstacle_0", "bbox": [65, 50, 80, 90], "canonical_name": "a pillar",
                "_sam2_mask_array": obstacle_mask, "depth_stats": {"median": 1.5}}]
    intrinsics = {"fx": 140.0, "fy": 140.0, "cx": w / 2.0, "cy": h / 2.0}
    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="p", timestamp="t",
        img_bgr=img, img_rgb=img, height=h, width=w, intrinsics=intrinsics, metric_depth=depth,
    )
    ctx.extra["objects"] = objects
    return paths_export.run(_FakePipeline(), ctx)


def test_segment_crosses_obstacle_true_when_line_passes_through():
    mask = _obstacle_mask(50, 50, 10, 20, 10, 20)
    assert app._segment_crosses_obstacle((5.0, 15.0), (25.0, 15.0), mask) is True


def test_segment_crosses_obstacle_false_when_line_avoids_it():
    mask = _obstacle_mask(50, 50, 10, 20, 10, 20)
    assert app._segment_crosses_obstacle((0.0, 0.0), (5.0, 5.0), mask) is False


def test_segment_crosses_obstacle_false_when_no_mask_given():
    assert app._segment_crosses_obstacle((0.0, 0.0), (49.0, 49.0), None) is False


def test_redraw_canvas_colors_crossing_segment_differently():
    h, w = 50, 50
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = _obstacle_mask(h, w, 10, 20, 10, 20)
    points = [(5.0, 15.0), (25.0, 15.0)]  # crosses the obstacle horizontally
    canvas = app._redraw_canvas(img, points, mask)
    # Somewhere along the crossing segment, the warning color should appear.
    assert (canvas.reshape(-1, 3) == np.array(app._OBSTACLE_WARNING_COLOR)).all(axis=1).any()


def test_redraw_canvas_uses_normal_color_when_no_crossing():
    h, w = 50, 50
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = _obstacle_mask(h, w, 10, 20, 10, 20)
    points = [(0.0, 0.0), (5.0, 5.0)]  # clear of the obstacle
    canvas = app._redraw_canvas(img, points, mask)
    assert not (canvas.reshape(-1, 3) == np.array(app._OBSTACLE_WARNING_COLOR)).all(axis=1).any()


def test_auto_adjust_around_obstacles_reroutes_and_reports(tmp_path):
    ctx = _analyzed_ctx_with_obstacle(tmp_path)
    state = {"ctx": ctx}
    drawn = [(72.0, 52.0), (72.0, 70.0), (72.0, 88.0)]  # straight through obstacle_0

    canvas, adjusted, status = app.auto_adjust_around_obstacles(state, drawn)
    assert canvas is not None
    assert adjusted != drawn
    assert adjusted[0] == drawn[0] and adjusted[-1] == drawn[-1]
    assert "Rerouted" in status and "1" in status


def test_auto_adjust_around_obstacles_no_crossing_leaves_path_unchanged(tmp_path):
    ctx = _analyzed_ctx_with_obstacle(tmp_path)
    state = {"ctx": ctx}
    drawn = [(5.0, 5.0), (10.0, 10.0)]  # nowhere near obstacle_0

    canvas, adjusted, status = app.auto_adjust_around_obstacles(state, drawn)
    assert adjusted == drawn
    assert "unchanged" in status


def test_auto_adjust_around_obstacles_requires_state_and_points():
    canvas, points, status = app.auto_adjust_around_obstacles(None, [(1.0, 1.0), (2.0, 2.0)])
    assert canvas is None
    assert "Draw" in status

    canvas, points, status = app.auto_adjust_around_obstacles({"ctx": object()}, [(1.0, 1.0)])
    assert canvas is None
    assert "Draw" in status


def _editor_value_from_mask(ink_mask: np.ndarray) -> dict:
    """Mimics the ``EditorValue`` dict gr.ImageEditor hands back: a single
    RGBA drawable layer whose alpha channel is the painted-ink mask."""
    h, w = ink_mask.shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[ink_mask, 0] = 235
    layer[ink_mask, 3] = 255
    return {"background": np.zeros((h, w, 3), dtype=np.uint8), "layers": [layer], "composite": None}


def test_vectorize_editor_stroke_straight_line():
    h, w = 100, 100
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask_img, (10, 50), (90, 50), 255, 3)
    editor_value = _editor_value_from_mask(mask_img > 0)

    points = app._vectorize_editor_stroke(editor_value, (h, w))
    assert len(points) >= 2
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) < 20 and max(xs) > 80, "endpoints should span the drawn line"
    assert all(abs(y - 50) <= 3 for y in ys), "a straight stroke should vectorize to a straight result"


def test_vectorize_editor_stroke_curve_is_not_collinear():
    """The core 'not forcing a polyline' regression test: a real curved
    stroke must NOT collapse to a straight line between its endpoints."""
    h, w = 150, 150
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask_img, (75, 100), (60, 60), 0, 180, 270, 255, 3)
    editor_value = _editor_value_from_mask(mask_img > 0)

    points = app._vectorize_editor_stroke(editor_value, (h, w))
    assert len(points) >= 3
    p0, p1 = np.array(points[0]), np.array(points[-1])
    chord = p1 - p0
    chord_len = float(np.hypot(*chord))
    assert chord_len > 0
    max_dev = 0.0
    for p in points:
        v = np.array(p) - p0
        proj = np.dot(v, chord) / chord_len
        perp = np.hypot(*(v - proj * chord / chord_len))
        max_dev = max(max_dev, float(perp))
    assert max_dev > 5.0, "a real curve must deviate meaningfully from the straight chord between its endpoints"


def test_vectorize_editor_stroke_ignores_small_branch():
    # A long main line with a short perpendicular branch off its MIDDLE
    # (not its end) -- this creates a real 3-endpoint skeleton (Y-shape), so
    # the BFS-diameter endpoint choice has to actually pick the two long
    # main-line ends over the shorter branch, not just default to it.
    h, w = 100, 100
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask_img, (10, 50), (90, 50), 255, 3)
    cv2.line(mask_img, (50, 50), (50, 60), 255, 3)
    editor_value = _editor_value_from_mask(mask_img > 0)

    points = app._vectorize_editor_stroke(editor_value, (h, w))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) < 20 and max(xs) > 80
    assert max(ys) < 55, "the short branch into the obstacle-facing direction must be excluded"


def test_vectorize_editor_stroke_empty_layer_returns_empty():
    h, w = 50, 50
    editor_value = _editor_value_from_mask(np.zeros((h, w), dtype=bool))
    assert app._vectorize_editor_stroke(editor_value, (h, w)) == []


def test_vectorize_editor_stroke_none_input_returns_empty():
    assert app._vectorize_editor_stroke(None, (50, 50)) == []
    assert app._vectorize_editor_stroke({"layers": []}, (50, 50)) == []


def test_vectorize_editor_stroke_uses_only_largest_component():
    h, w = 100, 100
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask_img, (10, 10), (30, 10), 255, 3)  # short, disjoint stroke
    cv2.line(mask_img, (10, 80), (90, 80), 255, 3)  # longer, disjoint stroke
    editor_value = _editor_value_from_mask(mask_img > 0)

    points = app._vectorize_editor_stroke(editor_value, (h, w))
    assert points, "expected the larger component to still produce a path"
    assert all(p[1] > 70 for p in points), "only the longer, disconnected stroke should be used"


def test_vectorize_editor_stroke_closed_loop_does_not_crash():
    h, w = 100, 100
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_img, (50, 50), 30, 255, 3)
    editor_value = _editor_value_from_mask(mask_img > 0)

    points = app._vectorize_editor_stroke(editor_value, (h, w))
    assert len(points) >= 2


def test_on_stroke_drawn_flags_obstacle_and_updates_preview(tmp_path):
    ctx = _analyzed_ctx_with_obstacle(tmp_path)
    state = {"ctx": ctx}
    mask_img = np.zeros((ctx.height, ctx.width), dtype=np.uint8)
    cv2.line(mask_img, (72, 52), (72, 88), 255, 3)  # straight through obstacle_0 ("a pillar")
    editor_value = _editor_value_from_mask(mask_img > 0)

    points, preview, status = app.on_stroke_drawn(state, editor_value)
    assert len(points) >= 2
    assert preview is not None
    assert (preview.reshape(-1, 3) == np.array(app._OBSTACLE_WARNING_COLOR)).all(axis=1).any()
    assert "a pillar" in status


def test_on_stroke_drawn_no_state_returns_empty():
    points, preview, status = app.on_stroke_drawn(None, {"layers": []})
    assert points == []
    assert preview is None
    assert status == ""
