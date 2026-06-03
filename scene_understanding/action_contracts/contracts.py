"""Data contracts for user-authored, scene-adaptive motion.

User input is authoritative; scene evidence is additive. A user-authored path,
tap point, uploaded animation, actor description, or interaction requirement must
not be silently replaced by a model-generated path. Adapters may add grounded
traces, warnings, preview corrections, and alternatives, but they must preserve
raw user geometry and report changes explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
GeometryMode = Literal["point", "start_end", "polyline", "region"]
ManifoldType = Literal[
    "centerline_path", "ribbon_path", "blob_path", "volume_path",
    "contour_path", "interior_path", "portal_path", "occlusion_pulse",
    "contact_patch", "effect_field",
]
ContractStatus = Literal[
    "accepted", "accepted_with_warnings", "low_confidence", "rejected", "accepted_surreal",
]


@dataclass(frozen=True)
class UserGeometry:
    """Immutable geometry created by the user."""

    mode: GeometryMode
    points: List[Point2D]
    source: str = "user"
    corridor_radius_px: float = 28.0
    closed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pts = [_coerce_point2d(p) for p in self.points]
        object.__setattr__(self, "points", pts)
        min_pts = {"point": 1, "start_end": 2, "polyline": 2, "region": 3}[self.mode]
        if len(pts) < min_pts:
            raise ValueError(f"{self.mode} geometry requires at least {min_pts} point(s)")
        if self.corridor_radius_px < 0:
            raise ValueError("corridor_radius_px must be non-negative")

    @property
    def start_point(self) -> Point2D:
        return self.points[0]

    @property
    def end_point(self) -> Point2D:
        return self.points[-1]

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class ActorSpec:
    """Open-vocabulary actor description."""

    actor_text: str
    actor_source: Literal["text", "uploaded_asset", "scene_object", "generated_asset", "drawn_asset"] = "text"
    asset_ref: Optional[str] = None
    scene_object_id: Optional[str] = None
    visual_style: Literal["photorealistic", "drawn", "stylized", "source_preserving"] = "source_preserving"
    physical_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = (self.actor_text or "").strip()
        if not text and not self.asset_ref and not self.scene_object_id:
            raise ValueError("ActorSpec needs actor_text, asset_ref, or scene_object_id")
        object.__setattr__(self, "actor_text", text)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class AdaptationPolicy:
    """Controls how scene evidence may adapt user input."""

    preserve_user_geometry: bool = True
    allow_path_bending: bool = True
    allow_endpoint_snap: bool = False
    allow_surreal_motion: bool = True
    max_path_deviation_px: float = 36.0
    obstacle_clearance_px: float = 8.0
    required_object_ids: List[str] = field(default_factory=list)
    avoid_object_ids: List[str] = field(default_factory=list)
    avoid_region_ids: List[str] = field(default_factory=list)
    must_render_behind_object_ids: List[str] = field(default_factory=list)
    protect_important_scene_regions: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class MotionContract:
    """User-authored action contract before scene grounding."""

    contract_id: str
    actor: ActorSpec
    action_text: str
    user_geometry: UserGeometry
    manifold_type: Optional[ManifoldType] = None
    duration_s: float = 4.0
    source: Literal["user_authored", "pipeline_suggested", "text_only"] = "user_authored"
    uploaded_animation_ref: Optional[str] = None
    policy: AdaptationPolicy = field(default_factory=AdaptationPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.contract_id or "").strip():
            raise ValueError("contract_id is required")
        text = (self.action_text or "").strip()
        if not text:
            raise ValueError("action_text is required")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        object.__setattr__(self, "action_text", text)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)
