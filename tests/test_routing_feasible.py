"""Tests for support-aware routing feasible masks (staged path plan §Phase 2)."""
from __future__ import annotations

import unittest

import numpy as np

from scene_understanding.pathing.routing_feasible import (
    build_feasible_base,
    build_feasible_bridge,
    build_feasible_routing,
    cc_label_at,
    connected_labels,
)


class _Cfg:
    path_routing_relax_on_support = True
    path_routing_support_speed_floor_mult = 0.72
    path_routing_support_close_px = 0
    path_routing_bridge_speed_floor_mult = 0.55


class RoutingFeasibleTests(unittest.TestCase):
    def test_routing_union_connects_support_only_low_speed(self) -> None:
        h, w = 8, 12
        lm = np.ones((h, w), dtype=np.int32)
        obs = np.zeros((h, w), dtype=bool)
        speed = np.full((h, w), 0.04, dtype=np.float32)
        speed[3:5, 2:10] = 0.12
        speed[5:7, 2:10] = 0.07
        support = np.zeros((h, w), dtype=bool)
        support[3:7, 2:10] = True
        base = build_feasible_base(lm, obs, speed, speed_floor=0.06)
        self.assertFalse(base[6, 5], "strict gate should drop low-speed tread")
        routed, variant = build_feasible_routing(base, lm, obs, speed, support, _Cfg(), speed_floor=0.06)
        self.assertEqual(variant, "routing_support")
        self.assertTrue(routed[6, 5], "support relaxation should recover stair pixel")

    def test_bridge_connects_two_cc_on_support(self) -> None:
        h, w = 10, 14
        lm = np.ones((h, w), dtype=np.int32)
        obs = np.zeros((h, w), dtype=bool)
        speed = np.full((h, w), 0.05, dtype=np.float32)
        speed[2:4, 2:5] = 0.2
        speed[7:9, 9:12] = 0.2
        speed[4:7, 5:9] = 0.06
        support = np.zeros((h, w), dtype=bool)
        support[2:4, 2:5] = True
        support[7:9, 9:12] = True
        support[4:7, 5:9] = True
        base = build_feasible_base(lm, obs, speed, speed_floor=0.06)
        routed, _ = build_feasible_routing(base, lm, obs, speed, support, _Cfg(), speed_floor=0.06)
        bridge = build_feasible_bridge(routed, lm, obs, speed, support, _Cfg(), speed_floor=0.06)
        lab, _n = connected_labels(bridge)
        a = cc_label_at(lab, (3, 3), h, w)
        b = cc_label_at(lab, (10, 8), h, w)
        self.assertGreater(a, 0)
        self.assertEqual(a, b, "bridge mask should merge disconnected support CCs")


if __name__ == "__main__":
    unittest.main()
