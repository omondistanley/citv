"""Core module: types, constants, and configuration."""

from .types import (
    BBox,
    Coordinates3D,
    DepthStats,
    DetectionMask,
    ObjectDetection,
    Triplet,
    SceneGraph,
    CameraIntrinsics,
    PipelineConfig,
)

from .pipeline_settings import (
    PipelineSettings,
    attach_settings_from_pipeline,
)
from .prompting import refresh_gdino_query_if_configured

from .constants import (
    RAMPP_GENERIC_TAGS,
    FLORENCE2_CAPTION_STOPWORDS,
    SPATIAL_PREDICATES,
    DEFAULT_GDINO_QUERY,
    DEPTH_COLORMAP,
    DEFAULT_HORIZONTAL_FOV,
    DEFAULT_VERTICAL_FOV,
    DEFAULT_MAX_IMAGE_SIDE,
    SAM2_AMG_DEFAULTS,
    GDINO_DEFAULTS,
    PIX2SG_DEFAULTS,
    DEPTH_MASK_DEFAULTS,
    RAMPP_DEFAULTS,
    FLORENCE2_DEFAULTS,
    OUTPUT_PATHS,
    FILE_PATTERNS,
)

__all__ = [
    "PipelineSettings",
    "attach_settings_from_pipeline",
    "refresh_gdino_query_if_configured",
    "BBox",
    "Coordinates3D",
    "DepthStats",
    "DetectionMask",
    "ObjectDetection",
    "Triplet",
    "SceneGraph",
    "CameraIntrinsics",
    "PipelineConfig",
    "RAMPP_GENERIC_TAGS",
    "FLORENCE2_CAPTION_STOPWORDS",
    "SPATIAL_PREDICATES",
    "DEFAULT_GDINO_QUERY",
    "DEPTH_COLORMAP",
    "DEFAULT_HORIZONTAL_FOV",
    "DEFAULT_VERTICAL_FOV",
    "DEFAULT_MAX_IMAGE_SIDE",
    "SAM2_AMG_DEFAULTS",
    "GDINO_DEFAULTS",
    "PIX2SG_DEFAULTS",
    "DEPTH_MASK_DEFAULTS",
    "RAMPP_DEFAULTS",
    "FLORENCE2_DEFAULTS",
    "OUTPUT_PATHS",
    "FILE_PATTERNS",
]
