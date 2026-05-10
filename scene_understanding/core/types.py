"""
Shared data types for the scene understanding pipeline.

The key compatibility rule is that the current truth `*_scene.json` layout stays
valid while new graph-native fields are added as a strict superset.
"""

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np


class BBox(TypedDict, total=False):
    """Bounding box in various formats."""

    xyxy: List[float]
    xywh: List[float]


class Coordinates3D(TypedDict, total=False):
    """3D coordinates in camera space."""

    x: float
    y: float
    z: float
    z_min: float
    z_max: float
    z_mean: float


class DepthStats(TypedDict, total=False):
    """Depth statistics for a masked region."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    num_pixels: int
    z_val: float
    z_val_pixels: int
    possibly_transparent: bool
    depth_separation_from_background: float


class DetectionMask(TypedDict, total=False):
    """Raw segmentation mask proposal."""

    segmentation: np.ndarray
    bbox: List[float]
    area: int
    predicted_iou: float
    stability_score: float
    label: Optional[str]
    confidence: float
    gdino_conf: float
    source_model: str


class SourceLabelPayload(TypedDict, total=False):
    """Compact per-source naming evidence."""

    label: str
    caption: str
    confidence: float
    tags: List[str]


class GroundedSAM2Source(TypedDict, total=False):
    caption: str
    label: str
    confidence: float


class Florence2Source(TypedDict, total=False):
    label: str
    caption: str


class RAMPPSource(TypedDict, total=False):
    label: str
    caption: str
    tags: List[str]


class NestedRelation(TypedDict, total=False):
    """Legacy nested relation entry under sources.Pix2SG.relations."""

    predicate: str
    target_id: str
    target_label: str
    target_caption: str
    score: float


class Pix2SGSource(TypedDict, total=False):
    relations: List[NestedRelation]


ObjectSources = TypedDict(
    "ObjectSources",
    {
        "GroundedSAM2": GroundedSAM2Source,
        "Florence2": Florence2Source,
        "RAM++": RAMPPSource,
        "Pix2SG": Pix2SGSource,
    },
    total=False,
)


class ObjectDetection(TypedDict, total=False):
    """Serialized object entry with legacy and additive fields."""

    id: str
    label: str
    confidence: float
    conf: float
    bbox: List[int]
    bbox_xyxy: List[float]
    source_model: str
    segmentor: str
    mask_centroid_2d: List[float]
    coordinates_3d: Coordinates3D
    depth_stats: DepthStats
    coordinates_3d_no_erosion: Coordinates3D
    depth_stats_no_erosion: DepthStats
    mask_centroid_2d_no_erosion: List[float]
    area_pixels: int
    sam2_mask_index: Optional[int]
    grounded_sam2_label: str
    grounded_sam2_confidence: float
    graph_id: str
    mask_matched: bool
    mask_path: Optional[str]
    depth_map_path: Optional[str]
    masked_depth_path: Optional[str]
    coordinates_3d_from_mask: Coordinates3D
    sources: ObjectSources

    # Additive graph-native fields
    name: str
    canonical_name: str
    aliases: List[str]
    category: str
    source_labels: Dict[str, SourceLabelPayload]
    layer_type: str
    parent_object_id: Optional[str]
    child_object_ids: List[str]
    part_mask_ids: List[int]
    contains: List[str]
    contained_by: List[str]
    occludes: List[str]
    occluded_by: List[str]

    # Internal only
    _sam2_mask_array: Optional[np.ndarray]


class Triplet(TypedDict, total=False):
    """Top-level exported relation entry."""

    subject_id: str
    subject_label: str
    subject_name: str
    predicate: str
    object_id: str
    object_label: str
    object_name: str
    object_caption: str
    source: str
    source_layer: str
    confidence: float
    score: float
    subject_centroid: List[int]
    object_centroid: Optional[List[int]]


class MaskHierarchyEdge(TypedDict, total=False):
    parent_object_id: str
    child_object_id: str
    parent_mask_index: Optional[int]
    child_mask_index: Optional[int]
    containment_ratio: float
    parent_overlap_ratio: float


class MaskHierarchy(TypedDict, total=False):
    edges: List[MaskHierarchyEdge]
    root_object_ids: List[str]
    num_edges: int


class LayerBand(TypedDict, total=False):
    layer_type: str
    object_ids: List[str]
    count: int
    z_min: float
    z_max: float


class LayersPayload(TypedDict, total=False):
    ordering: List[Dict[str, Any]]
    bands: List[LayerBand]
    depth_quantiles: Dict[str, float]


class SceneMetadata(TypedDict, total=False):
    timestamp: str
    segmentor: str
    intrinsics: Dict[str, float]
    models: List[str]
    rampp_tags: List[str]
    gdino_query_used: str
    relation_sources: Dict[str, Dict[str, Any]]
    relation_debug: Dict[str, Any]
    depth_map: str
    segmentation_image: str
    sam2_segmentation_image: str
    sam2_tinted_overlay_image: str

    # Additive metadata references
    relations_json: str
    mask_hierarchy_json: str
    layers_json: str
    depth_mask_A_json: str
    depth_mask_B_json: str
    relations_map_image: str


class SceneGraph(TypedDict, total=False):
    """Compatibility scene graph plus additive exports."""

    metadata: SceneMetadata
    objects: List[ObjectDetection]
    relations: List[Triplet]
    mask_hierarchy: MaskHierarchy
    layers: LayersPayload

    # Older depth-mask sidecar shape still used elsewhere
    depth: Dict[str, Any]
    segmentation: Dict[str, Any]
    summary: Dict[str, Any]


class CameraIntrinsics(TypedDict, total=False):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class PipelineConfig(TypedDict, total=False):
    """Configuration for the scene understanding pipeline."""

    sam2_amg_max_image_side: int
    gdino_box_threshold: float
    gdino_text_threshold: float
    gdino_query: str

    rampp_enabled: bool
    rampp_checkpoint_path: Optional[str]
    rampp_repo_path: Optional[str]

    depth_scale_factor: float
    depth_erosion_comparison: bool
    save_masked_depth_npy: bool

    save_per_object_masks: bool
    depth_mask_modes: List[str]

    run_both_segmentors: bool
    run_both_segmentors_iou_dedup: float

