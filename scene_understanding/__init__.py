"""
Scene Understanding Pipeline - Refactored Package Structure

This package contains the 3D scene graph generation pipeline.
See docs/SEGMENTATION.md, docs/DEPTH_ACCURACY.md, docs/LABELLING_AND_RELATIONS.md.

Package Structure:
  core/         - Types, constants, configuration
  preprocessing/ - Image loading, camera calibration, processing
  depth/        - Depth estimation coordination
  segmentation/ - GroundedSAM2 (GDINO + SAM2) wrappers
  labeling/     - Florence-2, RAM++, labeling orchestration
  geometry/     - 3D coordinate computation and mask geometry
  relations/    - Pix2SG, relation extraction
  output/       - Visualization and scene graph saving
  pipeline.py   - Main SceneUnderstandingPipeline orchestrator
"""

# Import from core
from .core import (
    ObjectDetection,
    Triplet,
    SceneGraph,
    CameraIntrinsics,
)

from .preprocessing import (
    load_bgr_image,
    CameraCalibration,
    resize_image_if_needed,
)

from .utils import (
    xywh_to_xyxy,
    xyxy_to_xywh,
    iou_xyxy,
    mask_iou,
)

# Import from submodules
from .labeling import (
    Florence2Wrapper,
    RAMPlusPlusWrapper,
    LabelingPipeline,
)

from .segmentation import (
    GroundedSAM2Wrapper,
    SAM2AMGWrapper,
    SegmentationPipeline,
)

from .relations import (
    Pix2SGWrapper,
    RelationsPipeline,
)

from .geometry import DepthGeometry
from .depth import DepthCoordinator
from .pipeline import SceneUnderstandingPipeline

__version__ = "2.0.0"  # Refactored version

__all__ = [
    # Core types and utilities
    "ObjectDetection",
    "Triplet",
    "SceneGraph",
    "CameraIntrinsics",
    "load_bgr_image",
    "CameraCalibration",
    "resize_image_if_needed",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "iou_xyxy",
    "mask_iou",
    # Labeling
    "Florence2Wrapper",
    "RAMPlusPlusWrapper",
    "LabelingPipeline",
    # Segmentation
    "GroundedSAM2Wrapper",
    "SAM2AMGWrapper",
    "SegmentationPipeline",
    # Relations
    "Pix2SGWrapper",
    "RelationsPipeline",
    # Depth and geometry
    "DepthGeometry",
    "DepthCoordinator",
    # Main pipeline
    "SceneUnderstandingPipeline",
]
