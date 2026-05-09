"""
Segmentation module: GroundedSAM2 (GDINO + SAM2) and optional SAM2 AMG wrapper (legacy).

Exports:
- GroundedSAM2Wrapper: Text-based GDINO + SAM2 prompted segmentation
- SAM2AMGWrapper: Legacy automatic mask generator (not used by default pipeline)
- SegmentationPipeline: Orchestrates segmentation backends
"""

from .grounded_sam2 import GroundedSAM2Wrapper
from .pipeline import SegmentationPipeline
from .sam2_amg import SAM2AMGWrapper

__all__ = [
    "GroundedSAM2Wrapper",
    "SAM2AMGWrapper",
    "SegmentationPipeline",
]
