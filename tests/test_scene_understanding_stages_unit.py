"""GPU-free unit tests for extracted stage helpers and fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import fixtures as su_fixtures
from scene_understanding.stages import scene_write


def test_fixtures_flat_metric_depth_shape():
    d = su_fixtures.flat_metric_depth(10, 12, value=3.0)
    assert d.shape == (10, 12)
    assert d.dtype == np.float32
    assert float(d[0, 0]) == pytest.approx(3.0)


def test_fixtures_fake_detections_list_two_masks(tmp_path):
    su_fixtures.write_tiny_bgr_image(tmp_path / "x.png", size=(20, 20))
    dets = su_fixtures.fake_detections_list(20, 20)
    assert len(dets) == 2
    assert np.sum(dets[0]["segmentation"] > 0) > 0
    assert np.sum(dets[1]["segmentation"] > 0) > 0


def test_scene_write_normalize_staged_object_keys():
    raw = {
        "id": "obj_0",
        "graph_id": "g0",
        "label": "cup",
        "conf": 0.5,
        "bbox": [1.0, 2.0, 5.0, 8.0],
        "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 1.0},
        "depth_stats": {"mean": 1.0},
        "mask_centroid_2d": [3, 4],
        "segmentor": "GroundedSAM2",
    }
    out = scene_write.normalize_staged_object(raw)
    assert set(out["sources"].keys()) == {"GroundedSAM2", "Florence2", "RAM++", "Pix2SG"}
    assert out["bbox"] == [1, 2, 5, 8]
    assert out["confidence"] == 0.5


def test_scene_write_build_staged_scene_payload_uses_pipeline_context_fields():
    ctx = PipelineContext(
        image_path=Path("/tmp/x.jpg"),
        output_dir=Path("/tmp/out"),
        stem="x",
        timestamp="t",
        img_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        img_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        height=4,
        width=4,
        intrinsics={"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 2.0},
        detections=[{"id": "d0"}],
        relations=[{"a": 1}],
        objects_3d=[{"legacy": True}],
        sam2_metadata={"custom_meta": 1},
        path_exports={"path_hypotheses_json": "scene_graph/t/p.json"},
    )
    ctx.extra["objects"] = [
        {
            "id": "obj_0",
            "label": "thing",
            "conf": 0.25,
            "bbox": [0, 0, 2, 2],
            "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 1.0},
            "depth_stats": {},
            "mask_centroid_2d": [1, 1],
            "segmentor": "x",
        }
    ]
    payload = scene_write.build_staged_scene_payload(ctx)
    assert payload["metadata"]["relation_debug"]["num_detections"] == 1
    assert len(payload["objects"]) == 1
    assert len(payload["relations"]) == 1
    assert ctx.objects_3d == [{"legacy": True}]
    assert ctx.sam2_metadata == {"custom_meta": 1}
    assert payload["metadata"]["custom_meta"] == 1
