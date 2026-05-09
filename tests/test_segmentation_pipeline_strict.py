"""Strict AMG supplement gating tests.

Ensures AMG part masks nested in grounded whole-object masks can be dropped
while larger complementary masks remain.
"""

from __future__ import annotations

import unittest

import numpy as np

from scene_understanding.segmentation.pipeline import _is_part_like_relative_to_grounded


class StrictAmgSupplementTests(unittest.TestCase):
    def test_drops_small_part_inside_grounded_object(self) -> None:
        h, w = 64, 64
        grounded = np.zeros((h, w), dtype=bool)
        grounded[10:54, 10:54] = True  # large full-object mask
        part = np.zeros((h, w), dtype=bool)
        part[24:36, 24:36] = True  # tiny inner part

        is_part = _is_part_like_relative_to_grounded(
            {"segmentation": part},
            [{"segmentation": grounded, "source_model": "GroundedSAM2"}],
            containment_thresh=0.88,
            min_area_ratio=0.25,
        )
        self.assertTrue(is_part)

    def test_keeps_large_complementary_amg_region(self) -> None:
        h, w = 64, 64
        grounded = np.zeros((h, w), dtype=bool)
        grounded[10:54, 10:54] = True
        # Comparable-scale region partially overlapping grounded mask.
        amg = np.zeros((h, w), dtype=bool)
        amg[8:56, 8:56] = True

        is_part = _is_part_like_relative_to_grounded(
            {"segmentation": amg},
            [{"segmentation": grounded, "source_model": "GroundedSAM2"}],
            containment_thresh=0.88,
            min_area_ratio=0.25,
        )
        self.assertFalse(is_part)


if __name__ == "__main__":
    unittest.main()

