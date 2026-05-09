"""Pre-SAM2 GDINO query refresh (RAM++ → SegmentationPipeline) for staged pipeline."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import numpy as np

from scene_understanding.core.prompting import refresh_gdino_query_for_staged


class _Cfg:
    query_builder_mode = "rampp_full"
    rampp_enabled = True
    rampp_max_tags = 8
    regions_rampp_crops_enabled = False


class _SegPipe:
    def __init__(self) -> None:
        self.text_query = "person. object."
        self.updated: List[str] = []

    def update_text_query(self, q: str) -> None:
        self.updated.append(q)
        self.text_query = q


class _Rampp:
    def __init__(self, tags: List[str]) -> None:
        self._tags = tags
        self.active = True

    def tag_image(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        return {"tags": list(self._tags), "label": "x"}

    def label_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        return {"tags": []}

    def label_crops(self, crops_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [{"tags": []} for _ in crops_bgr]


class StagedGdinoQueryRefreshTests(unittest.TestCase):
    def test_rampp_full_updates_query(self) -> None:
        seg = _SegPipe()
        ram = _Rampp(["stairs", "tree", "bench"])
        img = np.zeros((32, 48, 3), dtype=np.uint8)
        tags, q = refresh_gdino_query_for_staged(
            cfg=_Cfg(),
            img_rgb=img,
            region_partition_meta=[],
            width=48,
            height=32,
            seg_pipe=seg,
            rampp=ram,
        )
        self.assertEqual(tags, ["stairs", "tree", "bench"])
        self.assertTrue(q.endswith("."))
        self.assertIn("stairs", q)
        self.assertEqual(seg.updated, [q])

    def test_static_mode_no_op(self) -> None:
        class C:
            query_builder_mode = "static"
            rampp_enabled = True

        seg = _SegPipe()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        tags, q = refresh_gdino_query_for_staged(
            cfg=C(),
            img_rgb=img,
            region_partition_meta=[],
            width=8,
            height=8,
            seg_pipe=seg,
            rampp=_Rampp(["a"]),
        )
        self.assertEqual(tags, [])
        self.assertEqual(q, "person. object.")
        self.assertEqual(seg.updated, [])

    def test_inactive_rampp_returns_default(self) -> None:
        class C:
            query_builder_mode = "inherit"
            rampp_enabled = True

        seg = _SegPipe()
        ram = _Rampp([])
        ram.active = False
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        tags, q = refresh_gdino_query_for_staged(
            cfg=C(),
            img_rgb=img,
            region_partition_meta=[],
            width=8,
            height=8,
            seg_pipe=seg,
            rampp=ram,
        )
        self.assertEqual(tags, [])
        self.assertEqual(q, seg.text_query)
        self.assertEqual(seg.updated, [])


if __name__ == "__main__":
    unittest.main()
