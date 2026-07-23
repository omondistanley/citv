"""Unit tests for path_fields_explainer (no scene_understanding import)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scene_path_field_explainer import build_path_fields_explainer_image, build_path_fields_legend_payload


class _Cfg:
    path_fields_explainer_panel_h = 120
    path_fields_explainer_max_paths = 2


def test_explainer_image_and_legend():
    h, w = 48, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (100, 80, 60)
    cost = np.random.rand(h, w).astype(np.float32) * 0.5
    speed = np.ones((h, w), dtype=np.float32) * 0.7
    paths = [
        {
            "path_id": "p1",
            "polyline_2d": [[4.0, 4.0], [30.0, 40.0]],
            "polyline_geodesic_2d": [[5.0, 5.0], [28.0, 38.0]],
        }
    ]
    out = build_path_fields_explainer_image(img, cost, speed, paths, _Cfg())
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] > 50 and out.shape[1] > 50
    leg = build_path_fields_legend_payload("stem", {"speed_mean": 0.5}, True)
    assert leg["schema"] == "citv_path_fields_legend_v1"
    assert len(leg["panels"]) == 3
