"""Depth-based region partitioning for CITV."""

from .export import region_track_relpaths
from .mask_hierarchy import build_mask_hierarchy, mask_area
from .partitioner import (
    RegionPartitionResult,
    label_map_to_bgr,
    majority_region_index,
    partition_depth_regions,
)

__all__ = [
    "RegionPartitionResult",
    "build_mask_hierarchy",
    "label_map_to_bgr",
    "majority_region_index",
    "mask_area",
    "partition_depth_regions",
    "region_track_relpaths",
]
