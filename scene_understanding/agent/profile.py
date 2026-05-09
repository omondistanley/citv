"""Agent physical profile: documented measurements driving Layer E of the
semantic cost map.

Kept intentionally small: just a dataclass + strict YAML loader + a helper
that fails fast when the physics-layer is enabled but the profile is
missing / invalid. No defaults are ever silently substituted for the three
required dimensions (half_width_m, foot_tolerance_m, head_height_m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class AgentProfileError(RuntimeError):
    """Raised when the agent profile is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class AgentPhysicalProfile:
    """Documented physical measurements of the agent.

    All linear units are metres. The three required fields have no defaults:
    the caller must supply them via agent.profile.yaml. Optional fields fall
    back to physically-motivated derivations (see field comments), never
    to hand-picked magic constants.
    """

    half_width_m: float
    foot_tolerance_m: float
    head_height_m: float
    profile_id: str = "unnamed"
    source: str = "unspecified"
    measured_at: str = "unspecified"
    notes: str = ""

    pad_x_m: Optional[float] = None
    pad_y_m: Optional[float] = None
    step_m: Optional[float] = None

    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("half_width_m", self.half_width_m),
            ("foot_tolerance_m", self.foot_tolerance_m),
            ("head_height_m", self.head_height_m),
        ):
            if value is None:
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} is required but was None. "
                    "Set a real measurement in agent.profile.yaml."
                )
            if not isinstance(value, (int, float)):
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be a number, got "
                    f"{type(value).__name__}."
                )
            if math.isnan(float(value)) or math.isinf(float(value)):
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be finite, got {value!r}."
                )
            if float(value) <= 0.0:
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be > 0 (metres), "
                    f"got {value!r}."
                )

        for name, value in (
            ("pad_x_m", self.pad_x_m),
            ("pad_y_m", self.pad_y_m),
            ("step_m", self.step_m),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be a number or null, "
                    f"got {type(value).__name__}."
                )
            if math.isnan(float(value)) or math.isinf(float(value)):
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be finite when set, "
                    f"got {value!r}."
                )
            if float(value) <= 0.0:
                raise AgentProfileError(
                    f"AgentPhysicalProfile.{name} must be > 0 when set, "
                    f"got {value!r} (metres)."
                )

    def effective_pad_x_m(self) -> float:
        return float(self.pad_x_m) if self.pad_x_m is not None else float(self.half_width_m)

    def effective_pad_y_m(self) -> float:
        return float(self.pad_y_m) if self.pad_y_m is not None else float(self.half_width_m)

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "source": self.source,
            "measured_at": self.measured_at,
            "notes": self.notes,
            "half_width_m": float(self.half_width_m),
            "foot_tolerance_m": float(self.foot_tolerance_m),
            "head_height_m": float(self.head_height_m),
            "pad_x_m": None if self.pad_x_m is None else float(self.pad_x_m),
            "pad_y_m": None if self.pad_y_m is None else float(self.pad_y_m),
            "step_m": None if self.step_m is None else float(self.step_m),
        }


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # PyYAML
    except ImportError as exc:  # pragma: no cover - handled at call site
        raise AgentProfileError(
            "PyYAML is required to load agent.profile.yaml. "
            "Install via `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise AgentProfileError(f"Failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentProfileError(
            f"agent profile at {path} must be a YAML mapping at the top level."
        )
    return data


def load_agent_profile(
    yaml_path: str | Path = "agent.profile.yaml",
    *,
    required: bool = True,
) -> Optional[AgentPhysicalProfile]:
    """Load a YAML agent profile into an AgentPhysicalProfile.

    When ``required`` is False and the file is missing, returns None
    instead of raising, so callers can gracefully disable the physics layer.
    When the file exists but is malformed or incomplete, always raises
    regardless of ``required`` -- a malformed profile is never silently
    downgraded.
    """

    path = Path(yaml_path)
    if not path.exists():
        if required:
            raise AgentProfileError(
                f"agent profile not found at {path}. Either create it "
                "(see docs in agent.profile.yaml shipped with the repo) "
                "or disable path_cost_semantic_enabled in config.py."
            )
        return None

    data = _load_yaml(path)

    required_keys = ("half_width_m", "foot_tolerance_m", "head_height_m")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise AgentProfileError(
            f"agent profile at {path} is missing required keys: {missing}. "
            "These are the three dimensions Layer E of the semantic cost map "
            "depends on; fill them in with measured values."
        )

    return AgentPhysicalProfile(
        half_width_m=float(data["half_width_m"]),
        foot_tolerance_m=float(data["foot_tolerance_m"]),
        head_height_m=float(data["head_height_m"]),
        profile_id=str(data.get("id", data.get("profile_id", "unnamed"))),
        source=str(data.get("source", "unspecified")),
        measured_at=str(data.get("measured_at", "unspecified")),
        notes=str(data.get("notes", "") or ""),
        pad_x_m=(float(data["pad_x_m"]) if data.get("pad_x_m") is not None else None),
        pad_y_m=(float(data["pad_y_m"]) if data.get("pad_y_m") is not None else None),
        step_m=(float(data["step_m"]) if data.get("step_m") is not None else None),
        raw=dict(data),
    )


__all__ = [
    "AgentPhysicalProfile",
    "AgentProfileError",
    "load_agent_profile",
]
