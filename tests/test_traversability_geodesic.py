"""Unit tests for traversability speed map and grid geodesic (phase 2+)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scene_path_traversability import (
    build_traversability_speed_map,
    grid_dijkstra_path,
    k_diverse_grid_paths,
    heading_from_depth_at,
)


class _Cfg:
    trav_weight_image_edge = 0.25
    trav_weight_depth_flatness = 0.55
    trav_weight_image_smooth = 0.45
    trav_depth_grad_sigma_m = 0.35
    trav_speed_floor = 0.06


def test_speed_map_and_geodesic_corner_to_corner():
    h, w = 32, 32
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (128, 128, 128)
    lm = np.ones((h, w), dtype=np.int32)
    obs = np.zeros((h, w), dtype=bool)
    depth = np.ones((h, w), dtype=np.float32) * 3.0
    depth[:, 16:] = 5.0
    sm, meta = build_traversability_speed_map(depth, lm, obs, img, _Cfg())
    assert sm.shape == (h, w)
    assert "speed_mean" in meta
    p = grid_dijkstra_path(sm, (2, 2), (28, 28))
    assert len(p) >= 2
    assert p[0] == (2, 2) and p[-1] == (28, 28)


def test_heading_from_depth():
    w, h = 20, 20
    d = np.tile(np.linspace(1.0, 3.0, w, dtype=np.float32), (h, 1))
    th = heading_from_depth_at(d, 10.0, 10.0, window=9)
    assert th is not None


def test_k_diverse_returns_multiple():
    h, w = 24, 24
    sm = np.ones((h, w), dtype=np.float32) * 0.9
    sm[10:14, 10:14] = 0.1
    paths = k_diverse_grid_paths(sm, (2, 12), (22, 12), k=2, edge_penalty=0.5)
    assert len(paths) >= 1
