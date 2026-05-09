"""VLM layer for the semantic cost map (Phase C.3).

Layer B combines two zero-shot signals:

- **CLIPSeg** — per-pixel similarity to two pole-prompt lists (walkable vs.
  obstacle). Produces scene-wide dense cost evidence.
- **SigLIP** — per-object traversable vs. obstacle classification. Per-object
  scores are stamped into the image plane via the object label map so we
  keep object-level semantics while producing a dense cost.

Both wrappers are lazy: they load their weights on first use and fall back
to returning an *empty* CostLayer (precision = 0) if the model cannot be
loaded or the prompts file is malformed. The semantic cost map is always
additive — a missing VLM layer must not corrupt the fused output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scene_understanding.pathing.semantic_cost import CostLayer

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Prompts loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostPrompts:
    version: str
    clipseg_walkable: Tuple[str, ...]
    clipseg_obstacle: Tuple[str, ...]
    siglip_traversable: Tuple[str, ...]
    siglip_obstacle: Tuple[str, ...]
    clipseg_weight: float
    siglip_weight: float
    precision_floor: float
    precision_ceiling: float


def load_cost_prompts(path: str | Path) -> CostPrompts:
    """Load ``cost_map_prompts.yaml`` strictly.  Raises on any malformation."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read cost_map_prompts.yaml") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cost_map_prompts.yaml not found at {p}")
    data = yaml.safe_load(p.read_text()) or {}
    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError("cost_map_prompts.yaml missing 'version'")
    clipseg = data.get("clipseg", {}) or {}
    siglip = data.get("siglip", {}) or {}
    fusion = data.get("fusion", {}) or {}
    wk = tuple(str(x) for x in (clipseg.get("walkable_prompts") or []))
    ob = tuple(str(x) for x in (clipseg.get("obstacle_prompts") or []))
    tr = tuple(str(x) for x in (siglip.get("traversable_prompts") or []))
    so = tuple(str(x) for x in (siglip.get("obstacle_prompts") or []))
    if not wk or not ob or not tr or not so:
        raise ValueError(
            "cost_map_prompts.yaml must define non-empty clipseg.walkable_prompts, "
            "clipseg.obstacle_prompts, siglip.traversable_prompts, and "
            "siglip.obstacle_prompts"
        )
    return CostPrompts(
        version=version,
        clipseg_walkable=wk,
        clipseg_obstacle=ob,
        siglip_traversable=tr,
        siglip_obstacle=so,
        clipseg_weight=float(fusion.get("clipseg_weight", 1.0)),
        siglip_weight=float(fusion.get("siglip_weight", 1.0)),
        precision_floor=float(fusion.get("precision_floor", 0.05)),
        precision_ceiling=float(fusion.get("precision_ceiling", 0.95)),
    )


# ---------------------------------------------------------------------------
# CLIPSeg wrapper
# ---------------------------------------------------------------------------


class CLIPSegWrapper:
    """Lazy CLIPSeg zero-shot segmentation wrapper.

    ``score_prompts`` returns a ``(H, W)`` array in ``[0, 1]`` averaged over
    the supplied prompts. Returns ``None`` if the model is unavailable.
    """

    MODEL_ID = "CIDAS/clipseg-rd64-refined"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("mps" if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        self._processor: Any = None
        self._model: Any = None
        self._failed = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed or torch is None:
            return False
        try:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
            self._processor = CLIPSegProcessor.from_pretrained(self.MODEL_ID)
            self._model = CLIPSegForImageSegmentation.from_pretrained(self.MODEL_ID).to(self.device)
            self._model.eval()
            # CLIPSeg CLIP backbone expects 224×224 but the processor ships with
            # a 352×352 default — override to match the vision encoder config.
            expected = getattr(
                getattr(self._model.config, "vision_config", None), "image_size", 224
            )
            self._processor.image_processor.size = {"height": expected, "width": expected}
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[CLIPSegWrapper] load failed: {exc}. Layer B (dense) will be empty.")
            self._failed = True
            return False

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def score_prompts(self, img_bgr: np.ndarray, prompts: Sequence[str]) -> Optional[np.ndarray]:
        if cv2 is None or img_bgr is None or img_bgr.size == 0 or not prompts:
            return None
        if not self._load():
            return None
        from PIL import Image as PILImage
        h, w = img_bgr.shape[:2]
        pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            text=list(prompts),
            images=[pil] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            out = self._model(**inputs)
        except Exception as exc:
            print(f"[CLIPSegWrapper] forward failed: {exc}")
            return None
        logits = out.logits  # (n_prompts, H', W')
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        # Average over prompts, sigmoid to get [0, 1].
        sigm = torch.sigmoid(logits.float()).mean(dim=0)
        arr = sigm.detach().cpu().numpy().astype(np.float32)
        # Resize to original.
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return arr


# ---------------------------------------------------------------------------
# SigLIP two-pole object scorer
# ---------------------------------------------------------------------------


class SigLIPTwoPoleScorer:
    """Score each object crop as traversable vs. obstacle with SigLIP."""

    MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("mps" if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        self._processor: Any = None
        self._model: Any = None
        self._failed = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed or torch is None:
            return False
        try:
            from transformers import AutoProcessor, AutoModel
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self._model = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
            self._model.eval()
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[SigLIPTwoPoleScorer] load failed: {exc}. Per-object pole will be empty.")
            self._failed = True
            return False

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def score_crops(
        self,
        crops_bgr: Sequence[np.ndarray],
        traversable_prompts: Sequence[str],
        obstacle_prompts: Sequence[str],
    ) -> Optional[List[Tuple[float, float]]]:
        if not crops_bgr or not traversable_prompts or not obstacle_prompts:
            return None
        if not self._load():
            return None
        from PIL import Image as PILImage
        pil_crops = []
        for c in crops_bgr:
            if c is None or c.size == 0:
                pil_crops.append(None)
                continue
            pil_crops.append(PILImage.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
        valid_idx = [i for i, p in enumerate(pil_crops) if p is not None]
        if not valid_idx:
            return [(0.0, 0.0) for _ in pil_crops]
        all_texts = list(traversable_prompts) + list(obstacle_prompts)
        inputs = self._processor(
            text=all_texts,
            images=[pil_crops[i] for i in valid_idx],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            out = self._model(**inputs)
        except Exception as exc:
            print(f"[SigLIPTwoPoleScorer] forward failed: {exc}")
            return None
        logits = out.logits_per_image  # (n_crops, n_texts)
        n_tr = len(traversable_prompts)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        scores: List[Tuple[float, float]] = [(0.0, 0.0)] * len(pil_crops)
        for out_idx, row in zip(valid_idx, probs):
            t = float(row[:n_tr].mean())
            o = float(row[n_tr:].mean())
            scores[out_idx] = (t, o)
        return scores


# ---------------------------------------------------------------------------
# Layer B builder
# ---------------------------------------------------------------------------


def layer_vlm(
    img_bgr: np.ndarray,
    objects: Sequence[Mapping[str, Any]],
    object_label_map: Optional[np.ndarray],
    prompts: CostPrompts,
    *,
    clipseg: Optional[CLIPSegWrapper] = None,
    siglip: Optional[SigLIPTwoPoleScorer] = None,
) -> CostLayer:
    """Build the VLM evidence layer (B).

    Both sub-signals are optional. If neither model can run, the layer is
    returned with precision = 0 so the fused cost map is unaffected.
    """

    h, w = img_bgr.shape[:2]
    clipseg = clipseg or CLIPSegWrapper()
    siglip = siglip or SigLIPTwoPoleScorer()

    # ---- dense CLIPSeg scores -------------------------------------------------
    walk = clipseg.score_prompts(img_bgr, list(prompts.clipseg_walkable))
    obst = clipseg.score_prompts(img_bgr, list(prompts.clipseg_obstacle))
    if walk is not None and obst is not None:
        dense_cost = np.clip(obst - walk, 0.0, 1.0)
        dense_precision = np.clip(np.abs(obst - walk), prompts.precision_floor, prompts.precision_ceiling)
    else:
        dense_cost = np.zeros((h, w), dtype=np.float32)
        dense_precision = np.zeros((h, w), dtype=np.float32)

    # ---- per-object SigLIP stamps ---------------------------------------------
    per_obj_cost = np.zeros((h, w), dtype=np.float32)
    per_obj_precision = np.zeros((h, w), dtype=np.float32)
    crops: List[np.ndarray] = []
    obj_masks: List[Optional[np.ndarray]] = []
    lm = np.asarray(object_label_map, dtype=np.int32) if (object_label_map is not None and np.asarray(object_label_map).shape[:2] == (h, w)) else None
    for obj in objects:
        mask: Optional[np.ndarray] = None
        mid = obj.get("mask_id")
        if lm is not None and mid is not None:
            try:
                mid_i = int(mid)
            except (TypeError, ValueError):
                mid_i = None
            if mid_i and mid_i > 0:
                mask = (lm == mid_i)
        if mask is None:
            m = obj.get("mask")
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape[:2] != (h, w):
                continue
            mask = m.astype(bool)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        crop = img_bgr[y1:y2, x1:x2].copy()
        m_crop = mask[y1:y2, x1:x2]
        if cv2 is not None and m_crop.any():
            bg = img_bgr[mask].mean(axis=0).astype(np.uint8)
            crop[~m_crop] = bg
        crops.append(crop)
        obj_masks.append(mask)

    scores = siglip.score_crops(crops, list(prompts.siglip_traversable), list(prompts.siglip_obstacle)) if crops else None
    if scores is not None:
        for mask, (t, o) in zip(obj_masks, scores):
            if mask is None:
                continue
            c = max(0.0, min(1.0, o - t))
            p = max(prompts.precision_floor, min(prompts.precision_ceiling, abs(o - t)))
            per_obj_cost[mask] = np.maximum(per_obj_cost[mask], c)
            per_obj_precision[mask] = np.maximum(per_obj_precision[mask], p)

    # ---- weighted fusion of CLIPSeg + SigLIP within Layer B -------------------
    wc = prompts.clipseg_weight
    ws = prompts.siglip_weight
    wsum = max(wc + ws, 1e-6)
    wc, ws = wc / wsum, ws / wsum
    fused_cost = np.clip(wc * dense_cost + ws * per_obj_cost, 0.0, 1.0).astype(np.float32)
    fused_prec = np.clip(wc * dense_precision + ws * per_obj_precision, 0.0, 1.0).astype(np.float32)

    return CostLayer(
        name="B_vlm",
        cost=fused_cost,
        precision=fused_prec,
        provenance=(
            f"CLIPSeg dense (walkable vs. obstacle) + SigLIP per-object two-pole. "
            f"Prompt version {prompts.version}. weights=clipseg:{wc:.2f},siglip:{ws:.2f}"
        ),
        diagnostics={
            "dense_available": 1.0 if (walk is not None and obst is not None) else 0.0,
            "siglip_scored_objects": float(len(scores) if scores is not None else 0),
        },
    )


__all__ = [
    "CostPrompts",
    "CLIPSegWrapper",
    "SigLIPTwoPoleScorer",
    "load_cost_prompts",
    "layer_vlm",
]
