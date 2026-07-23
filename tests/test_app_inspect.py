"""Unit tests for app.py's click-to-inspect object name feature (Phase 2.6
item 7): inspect_object_at_point hit-tests a clicked pixel against every
real object mask and reports its name."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scene_understanding.pipeline_context import PipelineContext

import app


def _ctx_with_objects(tmp_path, objects):
    h, w = 40, 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ctx = PipelineContext(
        image_path=tmp_path / "p.jpg", output_dir=tmp_path / "out", stem="p", timestamp="t",
        img_bgr=img, img_rgb=img, height=h, width=w, intrinsics={}, metric_depth=None,
    )
    ctx.extra["objects"] = objects
    return ctx


def _mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _select_event(x, y):
    return SimpleNamespace(index=(x, y))


def test_inspect_reports_canonical_name(tmp_path):
    objects = [{"id": "obj_0", "canonical_name": "a rocking chair", "category": "furniture",
                "aliases": ["a rocking chair", "chair"], "_sam2_mask_array": _mask(40, 40, 5, 20, 5, 20)}]
    ctx = _ctx_with_objects(tmp_path, objects)
    state = {"ctx": ctx}
    result = app.inspect_object_at_point(state, _select_event(10, 10))
    assert "rocking chair" in result
    assert "furniture" in result


def test_inspect_no_object_at_point(tmp_path):
    objects = [{"id": "obj_0", "canonical_name": "a lamp", "_sam2_mask_array": _mask(40, 40, 5, 10, 5, 10)}]
    ctx = _ctx_with_objects(tmp_path, objects)
    state = {"ctx": ctx}
    result = app.inspect_object_at_point(state, _select_event(35, 35))
    assert "No detected object" in result


def test_inspect_picks_smallest_overlapping_mask(tmp_path):
    # A part (small) nested inside a bigger object -- clicking should report
    # the more specific (smaller-area) hit, not the larger containing one.
    big = {"id": "big", "canonical_name": "a bookshelf", "_sam2_mask_array": _mask(40, 40, 0, 40, 0, 40)}
    small = {"id": "small", "canonical_name": "a book", "_sam2_mask_array": _mask(40, 40, 15, 20, 15, 20)}
    ctx = _ctx_with_objects(tmp_path, [big, small])
    state = {"ctx": ctx}
    result = app.inspect_object_at_point(state, _select_event(17, 17))
    assert "book" in result and "bookshelf" not in result


def test_inspect_requires_analyzed_state():
    result = app.inspect_object_at_point(None, _select_event(5, 5))
    assert "Analyze" in result


def test_inspect_falls_back_to_label_when_no_canonical_name(tmp_path):
    objects = [{"id": "obj_0", "label": "plant", "_sam2_mask_array": _mask(40, 40, 5, 20, 5, 20)}]
    ctx = _ctx_with_objects(tmp_path, objects)
    state = {"ctx": ctx}
    result = app.inspect_object_at_point(state, _select_event(10, 10))
    assert "plant" in result
