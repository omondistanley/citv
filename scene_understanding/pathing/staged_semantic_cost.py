"""Staged-pipeline semantic cost fusion (no VLM by default).

Builds precision-weighted cost maps from geometry, scene-graph labels,
optional caption/ontology hints, agent physics, and depth-noise precision —
mirroring the monolith path but using staged ``objects`` with ``_sam2_mask_array``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..action_ontology import load_action_ontology, prompt_bank
from ..evidence import normalize_label
from .semantic_cost import (
    CostLayer,
    SemanticCostMap,
    layer_agent_physics,
    layer_depth_noise,
    layer_geometry,
    layer_scene_graph,
    precision_weighted_fuse,
)


def _objects_with_masks(
    objects: Sequence[Mapping[str, Any]],
    h: int,
    w: int,
) -> List[Dict[str, Any]]:
    """Materialise ``mask`` on each object for :func:`layer_scene_graph`."""
    out: List[Dict[str, Any]] = []
    try:
        import cv2
    except Exception:
        cv2 = None  # type: ignore[assignment]
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        raw = row.get("_sam2_mask_array")
        if raw is None:
            continue
        mm = np.asarray(raw, dtype=bool)
        if mm.shape[:2] != (h, w):
            if cv2 is not None:
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                continue
        row["mask"] = mm
        out.append(row)
    return out


def _label_matches_any_terms(label: str, terms: set) -> bool:
    label_n = normalize_label(label)
    if not label_n:
        return False
    return any(term == label_n or term in label_n or label_n in term for term in terms)


def _object_text_blob(obj: Mapping[str, Any]) -> str:
    parts = [
        str(obj.get("canonical_label") or obj.get("label") or ""),
        str(obj.get("caption") or ""),
    ]
    lc = obj.get("label_candidates") or []
    if isinstance(lc, list):
        for c in lc[:8]:
            if isinstance(c, dict):
                parts.append(str(c.get("caption") or c.get("label") or ""))
    return " ".join(parts).lower()


def layer_caption_ontology_hints(
    h: int,
    w: int,
    objects: Sequence[Mapping[str, Any]],
    *,
    ontology: Optional[Mapping[str, Any]] = None,
    precision_cap: float = 0.22,
) -> CostLayer:
    """Low-precision layer: liquid-related captions raise traversability cost on that mask.

    Uses ``role_prompts.liquid`` and ``action_prompts.swim`` token lists — no NN.
    """

    onto = ontology or load_action_ontology()
    bank = prompt_bank(onto, "role_prompts")
    liquid_terms = {normalize_label(v) for v in bank.get("liquid", []) if normalize_label(v)}
    act_bank = prompt_bank(onto, "action_prompts")
    for key in ("swim", "float"):
        liquid_terms |= {normalize_label(v) for v in act_bank.get(key, []) if normalize_label(v)}

    cost = np.zeros((h, w), dtype=np.float32)
    precision = np.zeros((h, w), dtype=np.float32)
    try:
        import cv2
    except Exception:
        cv2 = None  # type: ignore[assignment]

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        text = _object_text_blob(obj)
        if not text:
            continue
        hit = any(t and t in text for t in liquid_terms if len(t) > 2)
        if not hit:
            continue
        raw = obj.get("_sam2_mask_array")
        if raw is None:
            continue
        mm = np.asarray(raw, dtype=bool)
        if mm.shape[:2] != (h, w):
            if cv2 is not None:
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                continue
        if not mm.any():
            continue
        cost[mm] = np.maximum(cost[mm], 0.55)
        precision[mm] = np.maximum(precision[mm], float(precision_cap))

    return CostLayer(
        name="G_caption_ontology",
        cost=np.clip(cost, 0.0, 1.0).astype(np.float32),
        precision=np.clip(precision, 0.0, 1.0).astype(np.float32),
        provenance="Caption/token overlap with ontology liquid+swim lists (CPU-only, low precision).",
        diagnostics={"liquid_term_count": float(len(liquid_terms))},
    )


def _load_simple_agent_profile(cfg: Any) -> Optional[Any]:
    path = str(getattr(cfg, "path_cost_agent_profile_file", "agent.profile.yaml") or "")
    if not path:
        return None
    try:
        import yaml
        from pathlib import Path

        p = Path(path)
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None

        class SimpleProfile:
            def __init__(self, d: Dict[str, Any]) -> None:
                self._d = d

            def effective_pad_x_m(self) -> float:
                return float(self._d.get("pad_x_m", self._d.get("half_width_m", 0.3)))

            @property
            def half_width_m(self) -> float:
                return float(self._d.get("half_width_m", 0.3))

            @property
            def foot_tolerance_m(self) -> float:
                return float(self._d.get("foot_tolerance_m", 0.05))

        return SimpleProfile(data)
    except Exception:
        return None


def build_staged_semantic_cost_map(
    img_bgr: np.ndarray,
    metric_depth_m: Optional[np.ndarray],
    obstacle_mask: np.ndarray,
    objects: Sequence[Mapping[str, Any]],
    cfg: Any,
    intrinsics: Optional[Mapping[str, float]],
) -> SemanticCostMap:
    """Fuse semantic cost layers for the staged pipeline (image-grid)."""
    h, w = img_bgr.shape[:2]
    onto = load_action_ontology(cfg)

    layers: List[CostLayer] = []
    layers.append(layer_geometry(img_bgr, metric_depth_m, obstacle_mask))

    objs_masks = _objects_with_masks(objects, h, w)
    if objs_masks:
        layers.append(layer_scene_graph(h, w, objs_masks, object_label_map=None, ontology=onto))

    use_vlm = bool(getattr(cfg, "path_cost_use_vlm_layer", False)) if cfg else False
    if use_vlm:
        try:
            from pathlib import Path

            from .semantic_vlm import layer_vlm, load_cost_prompts

            py = str(getattr(cfg, "path_cost_prompts_yaml", "cost_map_prompts.yaml") or "")
            prompts = load_cost_prompts(Path(py)) if py else None
            if prompts is not None:
                layers.append(layer_vlm(img_bgr, list(objs_masks), None, prompts))
        except Exception as _vlm_exc:
            import warnings
            warnings.warn(
                f"[semantic_cost] VLM layer (CLIPSeg/SigLIP) failed and will be skipped: "
                f"{_vlm_exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if bool(getattr(cfg, "path_cost_ontology_caption_layer", True)) if cfg else True:
        layers.append(
            layer_caption_ontology_hints(
                h,
                w,
                objects,
                ontology=onto,
                precision_cap=float(getattr(cfg, "path_cost_caption_layer_precision_cap", 0.22) or 0.22),
            )
        )

    profile = _load_simple_agent_profile(cfg)
    if profile is not None and intrinsics and metric_depth_m is not None:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        if dm.shape[:2] == (h, w):
            layers.append(layer_agent_physics((h, w), obstacle_mask, dm, dict(intrinsics), profile))

    layers.append(layer_depth_noise(metric_depth_m))

    return precision_weighted_fuse(layers)


def blend_traversability_with_fused_cost(
    base_speed: np.ndarray,
    fused_cost: np.ndarray,
    *,
    blend: float = 0.72,
    speed_floor: float = 0.06,
) -> np.ndarray:
    """Modulate base traversability speed by fused semantic cost.

    ``speed = clip(base * (1 - blend * fused_cost), floor, 1)``.
    """
    b = float(np.clip(blend, 0.0, 1.0))
    fc = np.clip(np.asarray(fused_cost, dtype=np.float32), 0.0, 1.0)
    bs = np.asarray(base_speed, dtype=np.float32)
    out = bs * (1.0 - b * fc)
    return np.clip(out, float(speed_floor), 1.0).astype(np.float32)


__all__ = [
    "build_staged_semantic_cost_map",
    "blend_traversability_with_fused_cost",
    "layer_caption_ontology_hints",
]
