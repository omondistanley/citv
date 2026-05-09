"""Phase 2 contract tests: ground plane fit, foot/support anchors, polyline snap.

Plan §2.10. These tests exercise the new pure-CPU helpers without invoking the
full pipeline so they stay fast and depend only on numpy + cv2.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

import numpy as np

from scene_understanding.pathing.ground_plane import (
    build_semantic_support_mask,
    build_support_mask,
    fit_ground_plane,
    snap_uv_down_to_support,
)
from scene_understanding.pathing.path_hypotheses_paths import (
    dedupe_by_route_signature,
    dedupe_paths,
)
from scene_understanding.pathing.polyline_3d import (
    lift_polyline_2d_to_3d,
    smooth_polyline_in_3d,
)
from scene_understanding.pathing.walkable_mask import (
    build_path_walkable_mask,
    per_actor_connected_components,
)


class _Cfg:
    path_walkable_use_obstacles = True


class GroundPlaneTests(unittest.TestCase):
    def _flat_floor_depth(self, h: int = 96, w: int = 128, plane_z: float = 2.5) -> np.ndarray:
        # Lower half is a flat floor at constant depth; upper half varies.
        d = np.full((h, w), plane_z, dtype=np.float32)
        for v in range(h // 2):
            d[v, :] = 1.5 + 0.02 * v  # Upper half: increasing depth
        return d

    def test_fit_ground_plane_finds_floor(self) -> None:
        depth = self._flat_floor_depth()
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 64.0, "cy": 48.0}
        plane = fit_ground_plane(depth, intrinsics, sample_rows=(0.5, 1.0))
        self.assertIsNotNone(plane)
        self.assertGreater(plane["inlier_count"], 100)

    def test_build_semantic_support_mask_finds_floor_region(self) -> None:
        lm = np.zeros((20, 30), dtype=np.int32)
        lm[10:, :] = 1  # bottom half labelled region 1
        regions = [
            {"region_index": 1, "id": "region_1", "type": "floor area", "semantic_label": "floor"},
            {"region_index": 2, "id": "region_2", "type": "wall", "semantic_label": "wall"},
        ]
        sm = build_semantic_support_mask(lm, regions)
        self.assertTrue(sm[15, 15])
        self.assertFalse(sm[2, 2])

    def test_build_support_mask_unions_plane_and_semantic(self) -> None:
        depth = self._flat_floor_depth()
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 64.0, "cy": 48.0}
        h, w = depth.shape
        lm = np.ones((h, w), dtype=np.int32)
        regions = [{"region_index": 1, "id": "r1", "type": "floor", "semantic_label": "floor"}]
        sm, info = build_support_mask(depth, intrinsics, lm, regions)
        self.assertGreater(info["semantic_pixel_count"], 0)
        self.assertGreater(int(sm.sum()), 0)

    def test_snap_uv_down_walks_to_support(self) -> None:
        sm = np.zeros((10, 10), dtype=bool)
        sm[7:, :] = True
        x, y = snap_uv_down_to_support((5, 2), sm)
        self.assertEqual((x, y), (5, 7))

    def test_snap_uv_down_returns_bottom_when_no_support(self) -> None:
        sm = np.zeros((10, 10), dtype=bool)
        x, y = snap_uv_down_to_support((5, 2), sm)
        self.assertEqual((x, y), (5, 9))


class PolylineSnapTests(unittest.TestCase):
    def test_polyline_snaps_to_support_for_smoother_depth(self) -> None:
        # Two-bump depth where the 'support' is a flat strip at v=8.
        h, w = 16, 16
        depth = np.full((h, w), 2.0, dtype=np.float32)
        depth[3, :] = 0.5  # spike at v=3
        depth[8, :] = 2.5  # support strip
        sm = np.zeros((h, w), dtype=bool)
        sm[8, :] = True
        poly = [[5.0, 1.0], [5.0, 3.0]]
        out = lift_polyline_2d_to_3d(poly, depth, support_mask=sm)
        self.assertEqual(len(out), 2)
        # All snapped down to the support row v=8.
        self.assertEqual(int(out[0][1]), 8)
        self.assertEqual(int(out[1][1]), 8)
        self.assertAlmostEqual(out[0][2], 2.5, places=2)

    def test_smooth_polyline_in_3d_returns_reprojected(self) -> None:
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
        poly = [[float(i), 50.0, 1.0 + 0.1 * i] for i in range(10)]
        out = smooth_polyline_in_3d(poly, intrinsics, smoothing_window=3)
        self.assertEqual(len(out["polyline_2d_reprojected"]), 10)
        self.assertEqual(len(out["polyline_3d_smoothed"]), 10)


class WalkableSupportIntersectionTests(unittest.TestCase):
    def test_walkable_intersection_with_support_mask(self) -> None:
        h, w = 20, 20
        lm = np.ones((h, w), dtype=np.int32)
        obs = np.zeros((h, w), dtype=bool)
        speed = np.ones((h, w), dtype=np.float32)
        sm = np.zeros((h, w), dtype=bool)
        sm[10:, :] = True  # only bottom half is support
        walkable, meta = build_path_walkable_mask(
            lm, obs, speed, _Cfg(),
            support_mask=sm,
        )
        self.assertEqual(meta["support_intersection_applied"], 1.0)
        self.assertTrue(walkable[15, 5])
        self.assertFalse(walkable[2, 5])

    def test_per_actor_connected_components_distinguishes_islands(self) -> None:
        h, w = 20, 30
        wk = np.zeros((h, w), dtype=bool)
        wk[5:10, 0:10] = True   # island A
        wk[5:10, 18:28] = True  # island B
        labels, per_actor = per_actor_connected_components(wk, [(2, 7), (22, 7)])
        self.assertNotEqual(per_actor[0], per_actor[1])
        self.assertGreater(per_actor[0], 0)
        self.assertGreater(per_actor[1], 0)


class DedupeReasonTests(unittest.TestCase):
    def test_dedupe_paths_keeps_best_and_records_reason(self) -> None:
        base = [[0.0 + i, 0.0] for i in range(10)]
        near = [[0.5 + i, 0.5] for i in range(10)]
        far = [[float(i), 50.0 + i] for i in range(10)]
        candidates = [
            {"path_id": "p1", "polyline_2d": base, "source_entity": {"id": "a"}, "target_entity": {"id": "b"}, "scores": {"overall_confidence": 0.9}},
            {"path_id": "p2", "polyline_2d": near, "source_entity": {"id": "a"}, "target_entity": {"id": "b"}, "scores": {"overall_confidence": 0.6}},
            {"path_id": "p3", "polyline_2d": far,  "source_entity": {"id": "a"}, "target_entity": {"id": "b"}, "scores": {"overall_confidence": 0.5}},
        ]
        kept, dropped = dedupe_paths(candidates, max_per_pair=1, frechet_thresh_px=8.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["path_id"], "p1")
        self.assertEqual(len(dropped), 2)
        for d in dropped:
            self.assertTrue(d["dropped_reason"])

    def test_dedupe_by_route_signature(self) -> None:
        records = [
            {"path_id": "a", "route_signature": "x->y"},
            {"path_id": "b", "route_signature": "x->y"},
            {"path_id": "c", "route_signature": "x->z"},
            {"path_id": "d"},  # no signature → kept
        ]
        kept, dropped = dedupe_by_route_signature(records)
        ids_kept = [r["path_id"] for r in kept]
        ids_dropped = [r["path_id"] for r in dropped]
        self.assertIn("a", ids_kept)
        self.assertIn("c", ids_kept)
        self.assertIn("d", ids_kept)
        self.assertEqual(ids_dropped, ["b"])


if __name__ == "__main__":
    unittest.main()
