"""
scene_understanding.py — 3D scene graph generation pipeline.
See docs/SEGMENTATION.md, docs/DEPTH_ACCURACY.md, docs/LABELLING_AND_RELATIONS.md.
"""
# Ensure "import sam2" resolves to repo's sam2/sam2/ (not citv/sam2 as repo root) so build_sam check passes
import sys
from pathlib import Path as _Path
_script_dir = _Path(__file__).resolve().parent
_sam2_repo_root = _script_dir / "sam2"
if (_sam2_repo_root / "sam2").is_dir():
    _sp = str(_sam2_repo_root)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import cv2
import gc
import json
import numpy as np
import re
import torch
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    import transformers.safetensors_conversion as _stc

    def _disable_auto_conversion(*args, **kwargs):
        return None

    _stc.auto_conversion = _disable_auto_conversion
except Exception:
    pass

# Suppress transformers deprecation warning that has a logging format bug
# (passes FutureWarning as a format arg to a message with no %s → TypeError crash)
import logging as _logging
_logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(_logging.ERROR)

# Import existing DepthEstimator
from depth import DepthEstimator


def _load_bgr_image(path: Path) -> np.ndarray:
    """
    Load image as BGR numpy array.
    Uses OpenCV first, then PIL (with optional HEIF opener) as fallback.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is not None:
        return img_bgr

    pil_error = None
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass

        from PIL import Image
        with Image.open(path) as img_pil:
            img_rgb = np.array(img_pil.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        pil_error = e

    raise ValueError(
        f"Could not decode image: {path}. "
        "OpenCV returned None. If this is HEIF/HEIC content, convert it to JPEG/PNG "
        "or install pillow-heif for PIL decoding. "
        f"PIL fallback error: {pil_error}"
    )

# -----------------------------------------------------------------------------
# 1. RAM++ Wrapper (Recognize Anything Model++)
# -----------------------------------------------------------------------------
class RAMPlusPlusWrapper:
    """
    Optional RAM++ wrapper for open-vocabulary image tagging on masked crops.

    Requires a local Recognize Anything (RAM) installation and RAM++ checkpoint.
    If unavailable, wrapper stays inactive and returns generic labels.
    """

    _GENERIC_TAGS = {
        "object", "objects", "thing", "things", "item", "items",
        "entity", "entities", "scene", "image", "photo", "picture",
    }

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        image_size: int = 384,
        vit: str = "swin_l",
        default_confidence: float = 0.70,
        max_tags: int = 8,
    ):
        self.device = device
        self.model = None
        self.transform = None
        self.inference_fn = None
        self.active = False
        self.default_confidence = float(default_confidence)
        self.max_tags = int(max_tags)

        if repo_path:
            repo = Path(repo_path).expanduser()
            if not repo.is_absolute():
                repo = Path.cwd() / repo
            if repo.exists():
                repo_str = str(repo.resolve())
                if repo_str not in sys.path:
                    sys.path.insert(0, repo_str)

        # Compatibility shim: transformers 5.x moved/removed several APIs that
        # RAM++ depends on. Inject them back without editing any installed files.
        try:
            import sys as _sys
            import transformers.modeling_utils as _tmu
            from transformers.modeling_utils import PreTrainedModel as _PTM

            # 1. apply_chunking_to_forward moved to pytorch_utils
            if not hasattr(_tmu, "apply_chunking_to_forward"):
                from transformers.pytorch_utils import apply_chunking_to_forward as _actf
                _tmu.apply_chunking_to_forward = _actf

            # 2. find_pruneable_heads_and_indices — only used in pruning paths, not inference
            if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
                def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
                    mask = torch.ones(n_heads, head_size)
                    heads = set(heads) - already_pruned
                    for head in heads:
                        head -= sum(1 if h < head else 0 for h in already_pruned)
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index
                _tmu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

            # 3. prune_linear_layer — only used in pruning paths, not inference
            if not hasattr(_tmu, "prune_linear_layer"):
                def _prune_linear_layer(layer, index, dim=0):
                    import torch.nn as _nn
                    W = layer.weight.index_select(dim, index).clone().detach()
                    b = layer.bias.index_select(0, index).clone().detach() if layer.bias is not None else None
                    new_layer = _nn.Linear(W.size(1), W.size(0), bias=b is not None).to(layer.weight.device)
                    new_layer.weight = torch.nn.Parameter(W)
                    if b is not None:
                        new_layer.bias = torch.nn.Parameter(b)
                    return new_layer
                _tmu.prune_linear_layer = _prune_linear_layer

            # 4. all_tied_weights_keys must be a dict in transformers 5.x tie_weights().
            # Use a property with a setter so that subclasses can override via instance
            # assignment without hitting "can't set attribute" (Florence-2 does this).
            if not hasattr(_PTM, "all_tied_weights_keys"):
                def _all_tied_get(self):
                    v = self.__dict__.get("_all_tied_weights_keys_override", None)
                    if v is not None:
                        return v
                    return {k: k for k in (getattr(self, "_tied_weights_keys", None) or [])}
                def _all_tied_set(self, value):
                    self.__dict__["_all_tied_weights_keys_override"] = value
                _PTM.all_tied_weights_keys = property(_all_tied_get, _all_tied_set)

            # 5. get_head_mask removed from PreTrainedModel in transformers 5.x
            if not hasattr(_PTM, "get_head_mask"):
                def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                    if head_mask is not None:
                        if head_mask.dim() == 1:
                            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                        elif head_mask.dim() == 2:
                            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        if is_attention_chunked:
                            head_mask = head_mask.unsqueeze(-1)
                    else:
                        head_mask = [None] * num_hidden_layers
                    return head_mask
                _PTM.get_head_mask = _get_head_mask

            # 6. BertTokenizer.additional_special_tokens_ids removed — patch init_tokenizer
            #    in both ram.models.utils and ram.models.ram_plus (wildcard-imported copy).
            import ram.models.utils as _rmu
            import ram.models.ram_plus as _rmp_mod
            def _patched_init_tokenizer(text_encoder_type="bert-base-uncased"):
                from transformers import BertTokenizer as _BT
                _tok = _BT.from_pretrained(text_encoder_type)
                _tok.add_special_tokens({"bos_token": "[DEC]"})
                _tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
                _tok.enc_token_id = _tok.convert_tokens_to_ids("[ENC]")
                return _tok
            _rmu.init_tokenizer = _patched_init_tokenizer
            _sys.modules["ram.models.ram_plus"].init_tokenizer = _patched_init_tokenizer
            _sys.modules["ram.models.utils"].init_tokenizer = _patched_init_tokenizer
        except Exception as _shim_err:
            print(f"  [RAM++] transformers shim warning: {_shim_err}")

        print("Initializing RAM++...")
        try:
            from ram.models import ram_plus
            from ram import get_transform, inference_ram as inference
        except Exception as e:
            print(f"RAM++ unavailable: {e}.")
            return

        ckpt: Optional[Path] = None
        if checkpoint_path:
            ckpt = Path(checkpoint_path).expanduser()
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            if not ckpt.exists():
                print(f"RAM++ checkpoint not found at {ckpt}.")
                return
        else:
            print("RAM++ checkpoint path not configured; RAM++ labelling disabled.")
            return

        try:
            model = ram_plus(
                pretrained=str(ckpt),
                image_size=int(image_size),
                vit=str(vit),
            )
            model.to(self.device)
            model.eval()
            self.model = model
            self.transform = get_transform(image_size=int(image_size))
            self.inference_fn = inference
            self.active = True
            print(f"RAM++ ready (ckpt={ckpt}).")
        except Exception as e:
            print(f"RAM++ init failed: {e}.")
            self.active = False

    @classmethod
    def _parse_tags(cls, text: str, max_tags: int) -> List[str]:
        if not text:
            return []
        normalized = str(text).strip().lower()
        raw_parts = re.split(r"\s*\|\s*|,\s*|;\s*|\.\s*", normalized)
        tags: List[str] = []
        seen = set()
        for part in raw_parts:
            tag = " ".join(part.split()).strip()
            if not tag or tag in cls._GENERIC_TAGS:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
            if len(tags) >= max_tags:
                break
        return tags

    @staticmethod
    def _extract_english_tags(result: Any) -> str:
        if isinstance(result, dict):
            for k in ("tags", "tag_en", "english", "labels"):
                if k in result and result[k]:
                    return str(result[k])
        if isinstance(result, (list, tuple)):
            if len(result) >= 1 and result[0]:
                return str(result[0])
            return ""
        return str(result) if result is not None else ""

    def tag_image(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Run RAM++ on a full RGB image (numpy HxWx3). Returns: label, conf, caption, tags.
        Same as label_crop but accepts RGB directly.
        """
        if image_rgb is None or image_rgb.size == 0:
            return {"label": "object", "conf": 0.0, "caption": "object", "tags": []}
        crop_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.label_crop(crop_bgr)

    def label_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run RAM++ on a masked crop. Returns: label, conf, caption, tags.
        """
        if (
            not self.active
            or self.model is None
            or self.transform is None
            or self.inference_fn is None
            or crop_bgr is None
            or crop_bgr.size == 0
        ):
            return {"label": "object", "conf": 0.0, "caption": "object", "tags": []}

        try:
            from PIL import Image as PILImage

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(crop_rgb)
            image = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                result = self.inference_fn(image, self.model)

            tag_text = self._extract_english_tags(result)
            tags = self._parse_tags(tag_text, self.max_tags)
            label = tags[0] if tags else "object"
            caption = tag_text if tag_text else label
            conf = self.default_confidence if label != "object" else 0.0
            return {"label": label, "conf": conf, "caption": caption, "tags": tags}
        except Exception as e:
            print(f"  [RAM++] label_crop failed: {e}")
            return {"label": "object", "conf": 0.0, "caption": "object", "tags": []}



# -----------------------------------------------------------------------------
# 6. Pix2SG Wrapper (Pixel-to-Scene Graph)
# -----------------------------------------------------------------------------
class Pix2SGWrapper:
    """
    Spatial relation scaffold + Florence-2 semantic enrichment.
    Layer 1: pixel IoU / depth-axis / centroid direction.
    Layer 2: Florence-2 RED/BLUE colour-overlay captions for overlapping pairs.
    See docs/LABELLING_AND_RELATIONS.md for formulas and predicate table.
    """
    def __init__(
        self,
        device: torch.device,
        triplets_dir: str = "pix2sg_triplets",
        max_relations_per_object: int = 8,
        mask_overlap_thresh: float = 0.05,
        depth_near_threshold: float = 1.0,
        depth_far_threshold: float = 3.0,
        florence2: Optional["Florence2Wrapper"] = None,
        relation_min_mask_overlap: float = 0.02,
    ):
        self.device = device
        print("Initializing Pix2SG...")
        self.model = None
        self.triplets_dir = Path(triplets_dir)
        self.max_relations_per_object = max(1, int(max_relations_per_object))
        self._mask_overlap_thresh = float(mask_overlap_thresh)
        self._depth_near_threshold = float(depth_near_threshold)
        self._depth_far_threshold = float(depth_far_threshold)
        # Fix 5.6: Florence-2 semantic enrichment
        self._florence2 = florence2
        self._relation_min_mask_overlap = float(relation_min_mask_overlap)
        self.backend = "spatial_scaffold"
        self.active = True
        self.disabled_reason = ""
        if self.triplets_dir.exists():
            self.backend = "precomputed_triplets"
            print(f"Pix2SG precomputed triplets backend enabled: {self.triplets_dir.resolve()}")
        else:
            self.disabled_reason = (
                f"No precomputed triplets dir at {self.triplets_dir.resolve()}. "
                "Using spatial scaffold backend."
            )
            print(f"Pix2SG notice: {self.disabled_reason}")

    def is_active(self) -> bool:
        return bool(self.active)

    def status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active(),
            "backend": self.backend,
            "reason": self.disabled_reason,
        }

    def _load_precomputed_triplets(self, image_stem: str) -> List[Dict[str, Any]]:
        if self.backend != "precomputed_triplets":
            return []
        json_path = self.triplets_dir / f"{image_stem}.json"
        if not json_path.exists():
            return []
        try:
            with open(json_path, "r") as f:
                payload = json.load(f)
            triplets = payload.get("triplets", payload)
            if not isinstance(triplets, list):
                return []
            cleaned: List[Dict[str, Any]] = []
            for t in triplets:
                if not isinstance(t, dict):
                    continue
                sub = str(t.get("sub", "")).strip()
                pred = str(t.get("pred", "")).strip()
                obj = str(t.get("obj", "")).strip()
                if not (sub and pred and obj):
                    continue
                cleaned.append({
                    "sub": sub.lower(),
                    "pred": pred.lower(),
                    "obj": obj.lower(),
                    "score": float(t.get("score", 1.0)),
                    "sub_id": t.get("sub_id"),
                    "obj_id": t.get("obj_id"),
                })
            return cleaned
        except Exception as e:
            print(f"Pix2SG precomputed triplets parse failed ({json_path}): {e}")
            return []

    @staticmethod
    def _center(box: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _get_centroid(obj: Dict[str, Any]) -> Tuple[float, float]:
        """Return (cx, cy) preferring mask_centroid_2d over bbox center.

        When an object was matched to a SAM2 mask, mask_centroid_2d holds the
        centre-of-mass of the actual foreground pixels — more accurate than the
        midpoint of the bounding box, especially for non-rectangular objects.
        """
        mc = obj.get("mask_centroid_2d")
        if mc and len(mc) == 2:
            return float(mc[0]), float(mc[1])
        box = obj.get("bbox", [])
        if len(box) >= 4:
            return Pix2SGWrapper._center(box)
        return 0.0, 0.0

    @staticmethod
    def _spatial_predicate_bbox(
        box_a: List[float],
        box_b: List[float],
        image_w: int,
        image_h: int,
        iou_func,
    ) -> str:
        """Bbox-based spatial predicate — kept as fallback when masks are unavailable."""
        iou = float(iou_func(box_a, box_b))
        if iou >= 0.1:
            return "overlapping"
        ax, ay = Pix2SGWrapper._center(box_a)
        bx, by = Pix2SGWrapper._center(box_b)
        dx = bx - ax
        dy = by - ay
        if abs(dx) >= abs(dy):
            return "left_of" if dx > 0 else "right_of"
        return "above" if dy > 0 else "below"

    def _spatial_predicate_mask(self, sub: Dict[str, Any], obj: Dict[str, Any]) -> str:
        """Return spatial predicate using pixel-mask IoU and depth-weighted centroids."""
        sub_mask = sub.get("_sam2_mask_array")
        obj_mask = obj.get("_sam2_mask_array")

        # --- Overlap via pixel mask intersection (all objects have masks) ---
        if sub_mask is not None and obj_mask is not None:
            sub_m = np.asarray(sub_mask) > 0
            obj_m = np.asarray(obj_mask) > 0
            # Resize to match if sizes differ (e.g. if masks came from different resolutions)
            if sub_m.shape != obj_m.shape:
                obj_m = (cv2.resize(obj_m.astype(np.uint8), (sub_m.shape[1], sub_m.shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0)
            inter = int(np.logical_and(sub_m, obj_m).sum())
            union = int(np.logical_or(sub_m, obj_m).sum())
            if union > 0 and (inter / (union + 1e-8)) >= self._mask_overlap_thresh:
                return "overlapping"

        sub_z = sub.get("coordinates_3d", {}).get("z")
        obj_z = obj.get("coordinates_3d", {}).get("z")
        if sub_z is not None and obj_z is not None:
            depth_diff = abs(float(obj_z) - float(sub_z))
            if depth_diff >= self._depth_far_threshold:
                return "in_front_of" if float(sub_z) < float(obj_z) else "behind"

        sx, sy = self._get_centroid(sub)
        ox, oy = self._get_centroid(obj)
        dx, dy = ox - sx, oy - sy
        if abs(dx) >= abs(dy):
            return "left_of" if dx > 0 else "right_of"
        return "above" if dy > 0 else "below"

    def _build_spatial_scaffold_triplets(
        self,
        detections: List[Dict[str, Any]],
        image_h: int,
        image_w: int,
        iou_func,
    ) -> List[Dict[str, Any]]:
        if len(detections) < 2:
            return []
        triplets: List[Dict[str, Any]] = []
        for i, sub in enumerate(detections):
            sub_label = str(sub.get("label", "object")).lower()
            sub_id = sub.get("graph_id", sub.get("id"))
            sub_box = sub.get("bbox", [])
            if len(sub_box) < 4:
                continue
            scored_neighbors: List[Tuple[float, int]] = []
            # Use mask centroid for neighbour distance sorting (falls back to bbox center)
            sx, sy = self._get_centroid(sub)
            for j, obj in enumerate(detections):
                if i == j:
                    continue
                obj_box = obj.get("bbox", [])
                if len(obj_box) < 4:
                    continue
                ox, oy = self._get_centroid(obj)
                dist = float(np.hypot(ox - sx, oy - sy))
                scored_neighbors.append((dist, j))
            scored_neighbors.sort(key=lambda x: x[0])
            for _, j in scored_neighbors[: self.max_relations_per_object]:
                obj = detections[j]
                obj_label = str(obj.get("label", "object")).lower()
                obj_id = obj.get("graph_id", obj.get("id"))
                obj_box = obj.get("bbox", [])
                # Use mask-native predicate when at least one object has a mask;
                # otherwise fall back to the original bbox-based predicate.
                if sub.get("_sam2_mask_array") is not None or obj.get("_sam2_mask_array") is not None:
                    pred = self._spatial_predicate_mask(sub, obj)
                else:
                    pred = self._spatial_predicate_bbox(sub_box, obj_box, image_w, image_h, iou_func)
                if pred == "overlapping":
                    score = 0.85
                elif pred in {"in_front_of", "behind"}:
                    score = 0.75
                else:
                    score = 0.70
                triplets.append({
                    "sub": sub_label,
                    "pred": pred,
                    "obj": obj_label,
                    "sub_id": sub_id,
                    "obj_id": obj_id,
                    "score": score,
                })
        return triplets

    def predict(
        self,
        image: np.ndarray,
        image_stem: str = "",
        detections: Optional[List[Dict[str, Any]]] = None,
        iou_func=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate relation triplets.

        Layer 1: Spatial scaffold (always).
        Layer 2: Florence-2 semantic enrichment (when self._florence2 is set).
          - Iterates over pairs where masks overlap (IoU > relation_min_mask_overlap).
          - Queries Florence-2 with a color-coded crop of the pair.
          - If Florence-2 returns a known predicate, ADDS it as a separate
            triplet alongside the spatial one (both are kept — they describe
            different relation types for the same pair).
        Layer 3: Precomputed triplets (skips layers 1+2 when found).
        """
        if not self.is_active():
            return []
        if image_stem:
            precomputed = self._load_precomputed_triplets(image_stem)
            if precomputed:
                return precomputed
        if detections is None:
            return []
        h, w = image.shape[:2]
        if iou_func is None:
            iou_func = lambda _a, _b: 0.0

        # Layer 1: spatial scaffold
        triplets = self._build_spatial_scaffold_triplets(detections, h, w, iou_func)

        # Layer 2: Florence-2 semantic enrichment
        if self._florence2 is not None and self._florence2.active and len(detections) >= 2:
            florence_triplets = self._enrich_with_florence2(image, detections)
            triplets.extend(florence_triplets)

        return triplets

    def _enrich_with_florence2(
        self,
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For every pair of objects whose masks overlap above relation_min_mask_overlap,
        query Florence-2 for a semantic predicate.

        Only runs on spatially-proximate pairs (mask IoU > threshold) to avoid
        calling Florence-2 on every N^2 pair, which would be prohibitively slow.

        Returns additional triplets with source_layer="florence2".
        """
        extra: List[Dict[str, Any]] = []
        n = len(detections)
        for i in range(n):
            sub = detections[i]
            sub_mask = sub.get("_sam2_mask_array")
            if sub_mask is None:
                continue
            sub_m = np.asarray(sub_mask) > 0
            sub_label = str(sub.get("label", "object"))
            sub_id = sub.get("graph_id") or sub.get("id")

            for j in range(n):
                if i == j:
                    continue
                obj = detections[j]
                obj_mask = obj.get("_sam2_mask_array")
                if obj_mask is None:
                    continue
                obj_m = np.asarray(obj_mask) > 0
                if sub_m.shape != obj_m.shape:
                    obj_m = cv2.resize(
                        obj_m.astype(np.uint8),
                        (sub_m.shape[1], sub_m.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                inter = int(np.logical_and(sub_m, obj_m).sum())
                union = int(np.logical_or(sub_m, obj_m).sum())
                if union == 0 or (inter / union) < self._relation_min_mask_overlap:
                    continue

                obj_label = str(obj.get("label", "object"))
                obj_id = obj.get("graph_id") or obj.get("id")

                pred = self._florence2.predict_relation(
                    image_bgr, sub_m, obj_m, sub_label, obj_label
                )
                if pred is not None:
                    extra.append({
                        "sub": sub_label.lower(),
                        "pred": pred,
                        "obj": obj_label.lower(),
                        "sub_id": sub_id,
                        "obj_id": obj_id,
                        "score": 0.75,
                        "source_layer": "florence2",
                    })
        return extra

class Florence2Wrapper:
    """
    Florence-2 wrapper for object labelling (<OD>) and relation prediction (<CAPTION>).
    See docs/LABELLING_AND_RELATIONS.md for label priority and colour-overlay method.
    """

    def __init__(self, model_id: str = "microsoft/Florence-2-large", device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.active = False

        print(f"Initializing Florence-2 ({model_id})...")
        try:
            # Florence-2's processing_florence2.py line 89 accesses
            # tokenizer.additional_special_tokens on a TokenizersBackend object.
            # transformers >= 4.45 moved this attribute resolution into __getattr__
            # which raises AttributeError for unknown attributes. Patch the class
            # before loading the processor so the attribute exists.
            try:
                from transformers.tokenization_utils_tokenizers import TokenizersBackend
                if not hasattr(TokenizersBackend, "additional_special_tokens"):
                    TokenizersBackend.additional_special_tokens = property(
                        lambda self: []
                    )
            except Exception:
                pass

            from transformers import AutoProcessor, AutoModelForCausalLM
            # use_fast=False avoids the fast tokenizer backend that triggers the
            # additional_special_tokens AttributeError on tokenizers >= 0.20.
            _dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            _proc_kwargs = dict(trust_remote_code=True)
            try:
                self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False, **_proc_kwargs)
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(model_id, **_proc_kwargs)

            # attn_implementation="eager" disables SDPA dispatch, which avoids
            # the _supports_sdpa property being called before language_model is
            # initialised (transformers 5.x + Florence-2 custom code incompatibility).
            # The cached modeling_florence2.py is also patched for the two known bugs:
            #   1. dpr linspace uses device="cpu" to avoid meta-tensor .item() error
            #   2. _supports_sdpa property guards against uninitialised language_model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=_dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)
            self.model.eval()
            self.active = True
            print(f"Florence-2 ready ({model_id}).")
        except Exception as e:
            print(f"Florence-2 init failed: {e}.")

    def _run_task(self, task: str, pil_image, extra_text: str = "") -> Any:
        """Run a single Florence-2 task on a PIL image. Returns parsed result."""
        if not self.active:
            return {}
        try:
            prompt = task + extra_text
            inputs = self.processor(
                text=prompt, images=pil_image, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.model.dtype == torch.float16:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=False,  # transformers 5.x EncoderDecoderCache is incompatible with Florence-2 custom code
                )
            text_out = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                text_out,
                task=task,
                image_size=(pil_image.width, pil_image.height),
            )
            return parsed
        except Exception as e:
            print(f"  [Florence2] task={task} failed: {e}")
            return {}

    # Stopwords to skip when extracting a noun label from a Florence-2 caption.
    # Articles, prepositions, common adjectives, and meta-words ("image", "photo")
    # are all non-informative for an object label.
    _CAPTION_STOPWORDS = {
        # articles / determiners
        "a", "an", "the", "some", "one", "two", "three",
        # prepositions / conjunctions
        "with", "on", "of", "in", "at", "by", "and", "or", "from", "to",
        "for", "as", "up", "out", "into", "over", "under", "about", "around",
        # verbs / auxiliaries
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "can", "may", "will", "appears", "seems", "showing", "shows", "shown",
        # pronouns
        "this", "that", "these", "those", "it", "its", "there", "their",
        # meta / photographic words
        "image", "photo", "picture", "view", "close", "shot",
        # positional / descriptive (these are adjectives, not nouns)
        "side", "top", "front", "back", "left", "right", "center", "middle",
        # common adjectives that precede the actual noun
        "red", "blue", "green", "yellow", "white", "black", "brown", "grey",
        "gray", "orange", "purple", "pink", "dark", "light", "bright",
        "large", "small", "big", "little", "tiny", "tall", "short", "long",
        "old", "new", "open", "closed", "empty", "full", "flat", "round",
        "square", "wooden", "metal", "plastic", "glass", "stone", "brick",
        "single", "double", "multiple", "various", "different", "same",
        # filler adverbs
        "very", "quite", "just", "also", "well",
    }

    @classmethod
    def _extract_label_from_caption(cls, caption: str) -> str:
        """
        Extract the first meaningful noun from a Florence-2 caption.
        Skips stopwords and punctuation. Returns "object" if nothing useful found.
        """
        if not caption or not isinstance(caption, str):
            return "object"
        for w in caption.lower().split():
            w_clean = w.strip(".,;:!?\"'()")
            if w_clean.isalpha() and len(w_clean) > 2 and w_clean not in cls._CAPTION_STOPWORDS:
                return w_clean
        return "object"

    def label_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Label a BGR crop using Florence-2.

        Primary: <MORE_DETAILED_CAPTION> — on a tight single-object crop this
        reliably produces sentences like "a wooden dining chair with padded seat"
        from which we extract the first meaningful noun ("chair").

        Secondary: <OD> — if caption extraction still yields "object", run
        detection on the crop and pick the label from the largest bbox.

        Returns dict: label (str), conf (float), caption (str).
        """
        if not self.active or crop_bgr is None or crop_bgr.size == 0:
            return {"label": "object", "conf": 0.0, "caption": "object"}

        from PIL import Image as PILImage
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = PILImage.fromarray(crop_rgb)

        # --- Primary: rich caption → noun extraction ---
        cap_result = self._run_task("<MORE_DETAILED_CAPTION>", pil_crop)
        caption = cap_result.get("<MORE_DETAILED_CAPTION>", "")
        if not isinstance(caption, str):
            caption = str(caption)

        label = self._extract_label_from_caption(caption)
        conf = 0.75

        # --- Secondary: structured <OD> when caption gave nothing useful ---
        if label == "object":
            od_result = self._run_task("<OD>", pil_crop)
            od_data = od_result.get("<OD>", {})
            od_labels = od_data.get("labels", [])
            od_bboxes = od_data.get("bboxes", [])
            if od_labels:
                best_area = -1
                for lbl, box in zip(od_labels, od_bboxes):
                    if len(box) >= 4:
                        area = abs(box[2] - box[0]) * abs(box[3] - box[1])
                        if area > best_area:
                            best_area = area
                            label = str(lbl).strip().lower()
                if label != "object":
                    conf = 0.80

        if not caption:
            caption = label

        return {"label": label, "conf": conf, "caption": caption}

    def predict_relation(
        self,
        full_img_bgr: np.ndarray,
        mask_sub: np.ndarray,
        mask_obj: np.ndarray,
        label_sub: str,
        label_obj: str,
    ) -> Optional[str]:
        """Predict relation via RED/BLUE colour overlay + Florence-2 caption. Returns predicate or None."""
        if not self.active:
            return None
        try:
            from PIL import Image as PILImage

            h, w = full_img_bgr.shape[:2]

            # Resize masks to image dims
            sub_m = cv2.resize(mask_sub.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            obj_m = cv2.resize(mask_obj.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Union bbox
            union_m = sub_m | obj_m
            ys, xs = np.where(union_m)
            if ys.size == 0:
                return None
            x1, x2 = max(0, int(xs.min()) - 10), min(w, int(xs.max()) + 10)
            y1, y2 = max(0, int(ys.min()) - 10), min(h, int(ys.max()) + 10)

            crop = full_img_bgr[y1:y2, x1:x2].copy()
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Color overlay: subject=red tint, object=blue tint
            sub_crop = sub_m[y1:y2, x1:x2]
            obj_crop = obj_m[y1:y2, x1:x2]
            alpha = 0.45
            crop_rgb[sub_crop, 0] = np.clip(crop_rgb[sub_crop, 0] * (1 - alpha) + 255 * alpha, 0, 255)
            crop_rgb[obj_crop, 2] = np.clip(crop_rgb[obj_crop, 2] * (1 - alpha) + 255 * alpha, 0, 255)

            pil_crop = PILImage.fromarray(crop_rgb.astype(np.uint8))

            result = self._run_task("<MORE_DETAILED_CAPTION>", pil_crop)
            raw = result.get("<MORE_DETAILED_CAPTION>", "")
            if not isinstance(raw, str) or not raw.strip():
                return None

            # Map common relation phrases to canonical predicates
            return self._parse_relation_phrase(raw.lower().strip())
        except Exception as e:
            print(f"  [Florence2] predict_relation failed: {e}")
            return None

    @staticmethod
    def _parse_relation_phrase(text: str) -> Optional[str]:
        """Map free-form Florence-2 caption to a canonical predicate, or None if no match."""
        PHRASE_MAP = [
            # Spatial
            (["on top of", "resting on", "placed on", "sitting on", "standing on", "lying on"], "on"),
            (["under", "below", "beneath"], "under"),
            (["next to", "beside", "adjacent", "alongside"], "is_next_to"),
            (["in front of", "in front"], "in_front_of"),
            (["behind", "in back of"], "behind"),
            (["inside", "within", "contained in", "in the"], "inside_of"),
            (["hanging from", "suspended from"], "hangs_from"),
            (["leaning on", "leaning against"], "leans_on"),
            (["at", "located at"], "at"),
            # Functional / action
            (["holding", "carrying", "gripping", "grabbing"], "holds"),
            (["wearing", "dressed in"], "wears"),
            (["riding", "mounted on"], "rides"),
            (["eating", "consuming"], "eats"),
            (["drinking"], "drinks"),
            (["reading"], "reads"),
            (["using", "operating"], "uses"),
            (["looking at", "gazing at"], "looks_at"),
            (["talking on", "speaking on"], "talks_on_phone"),
            (["playing"], "plays"),
            (["kicking"], "kicks"),
            (["catching"], "catches"),
            (["cutting"], "cuts"),
        ]
        for phrases, predicate in PHRASE_MAP:
            if any(p in text for p in phrases):
                return predicate
        return None


# See docs/SEGMENTATION.md for GroundedSAM2 vs AMG architecture and fallback logic.
class GroundedSAM2Wrapper:
    """
    Grounding DINO v2 + SAM2 prompted mode for object-level instance segmentation.

    Returns the same list-of-dicts format as SAM2AMGWrapper.generate() so it
    is a drop-in replacement.  Extra keys added per entry:
      - "label": semantic class from GDINO (str)
      - "gdino_conf": GDINO detection confidence (float)
    """

    def __init__(
        self,
        device: torch.device,
        sam2_checkpoint_path: str,
        sam2_model_cfg: str,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        box_thresh: float = 0.30,
        text_thresh: float = 0.25,
        text_query: str = "person. animal. vehicle. furniture. appliance. food. clothing. container. tool. building. plant. electronics. object.",
        max_image_side: int = 1280,
        fallback_to_amg: bool = True,
        # AMG fallback params
        points_per_side: int = 32,
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 1000,
        use_m2m: bool = True,
        box_nms_thresh: float = 0.7,
    ):
        self.device = device
        self.text_query = text_query
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.max_image_side = max_image_side
        self.fallback_to_amg = fallback_to_amg
        self._gdino = None
        self._gdino_processor = None
        self._sam2_predictor = None
        self._amg_fallback = None
        self.active = False

        # 1. Load Grounding DINO
        print(f"Initializing Grounding DINO ({gdino_model_id})...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
                gdino_model_id
            ).to(device)
            self._gdino.eval()
            print("Grounding DINO ready.")
        except Exception as e:
            print(f"Grounding DINO init failed: {e}.")
            if not fallback_to_amg:
                raise
            print("Will use SAM2 AMG fallback.")

        # 2. Load SAM2 predictor (single-image prompted mode)
        print("Initializing SAM2 predictor (prompted mode)...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            ckpt = Path(sam2_checkpoint_path)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            hydra_cfg = sam2_model_cfg
            if hydra_cfg.endswith(".yaml"):
                hydra_cfg = hydra_cfg[:-5]
            model = build_sam2(hydra_cfg, str(ckpt))
            model.to(device)
            self._sam2_predictor = SAM2ImagePredictor(model)
            print("SAM2 predictor (prompted) ready.")
            self.active = True
        except Exception as e:
            print(f"SAM2 predictor init failed: {e}.")
            self.active = False

        # 3. AMG fallback (original SAM2AMGWrapper)
        if fallback_to_amg:
            print("Initializing SAM2 AMG as fallback...")
            self._amg_fallback = SAM2AMGWrapper(
                device=device,
                checkpoint_path=sam2_checkpoint_path,
                model_cfg=sam2_model_cfg,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
                use_m2m=use_m2m,
                box_nms_thresh=box_nms_thresh,
            )

    def _detect_objects(self, image_rgb: np.ndarray) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Run Grounding DINO on image_rgb.

        Returns:
            boxes:  list of [x1,y1,x2,y2] in pixel coords
            labels: list of class label strings
            scores: list of detection confidence floats
        """
        if self._gdino is None:
            return [], [], []

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_rgb)
        inputs = self._gdino_processor(
            images=pil_img,
            text=self.text_query,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._gdino(**inputs)

        h, w = image_rgb.shape[:2]
        # transformers API changed `box_threshold` -> `threshold` across versions.
        # Try the current signature first, then fall back for older releases.
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[(h, w)],
            )[0]
        except TypeError as e:
            if "unexpected keyword argument 'threshold'" not in str(e):
                raise
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[(h, w)],
            )[0]

        boxes = results["boxes"].cpu().numpy().tolist()    # [[x1,y1,x2,y2], ...]
        scores = results["scores"].cpu().numpy().tolist()  # [float, ...]
        labels = results["labels"]                         # [str, ...]

        # Clamp to image bounds
        clipped = []
        for box in boxes:
            clipped.append([
                max(0.0, box[0]), max(0.0, box[1]),
                min(float(w), box[2]), min(float(h), box[3]),
            ])
        return clipped, list(labels), scores

    def _boxes_to_masks(
        self, image_rgb: np.ndarray, boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Run SAM2 predictor once on the image with all detected boxes as prompts.
        Returns one binary mask per box (HxW bool arrays).

        SAM2 prompted mode processes all boxes in a single forward pass,
        which is more efficient than one pass per box.
        """
        if self._sam2_predictor is None or not boxes:
            return []
        try:
            self._sam2_predictor.set_image(image_rgb)
            boxes_np = np.array(boxes, dtype=np.float32)  # (N,4) xyxy
            with torch.inference_mode():
                masks, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_np,
                    multimask_output=False,  # one mask per box
                )
            # masks shape: (N, 1, H, W) or (N, H, W) depending on SAM2 version
            result = []
            for i in range(len(boxes)):
                m = masks[i]
                if m.ndim == 3:
                    m = m[0]  # take first (only) mask
                result.append(m.astype(bool))
            return result
        except Exception as e:
            print(f"  [GroundedSAM2] SAM2 predict failed: {e}")
            return [np.zeros(image_rgb.shape[:2], dtype=bool)] * len(boxes)

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Main entry point — same API as SAM2AMGWrapper.generate().

        Returns list of dicts with keys:
          segmentation (bool HxW), bbox ([x,y,w,h] xywh), area (int),
          predicted_iou (float), stability_score (float),
          label (str), gdino_conf (float)

        Execution flow:
          1. Optionally resize if image > max_image_side.
          2. Run Grounding DINO → boxes + labels + scores.
          3. If zero detections and fallback_to_amg → use SAM2 AMG.
          4. Run SAM2 predictor with all boxes → masks.
          5. Build result dicts in SAM2AMGWrapper format + label/gdino_conf.
        """
        h, w = image_rgb.shape[:2]
        scale = 1.0
        proc = image_rgb
        if self.max_image_side > 0 and max(h, w) > self.max_image_side:
            scale = self.max_image_side / max(h, w)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            proc = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        boxes, labels, scores = self._detect_objects(proc)
        print(f"  [GroundedSAM2] GDINO detected {len(boxes)} objects.")

        # Fallback if GDINO gave nothing
        if not boxes and self.fallback_to_amg and self._amg_fallback is not None:
            print("  [GroundedSAM2] Zero detections → falling back to SAM2 AMG.")
            return self._amg_fallback.generate(image_rgb)

        if not boxes:
            return []

        # Scale boxes back to original image coords if we resized
        if scale != 1.0:
            inv = 1.0 / scale
            boxes = [[b[0]*inv, b[1]*inv, b[2]*inv, b[3]*inv] for b in boxes]
            proc = image_rgb  # use original for SAM2

        masks = self._boxes_to_masks(image_rgb, boxes)

        results = []
        for i, (box_xyxy, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            x1, y1, x2, y2 = box_xyxy
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = int(mask.sum()) if mask is not None else int(bw * bh)
            results.append({
                "segmentation": mask,
                "bbox": [float(x1), float(y1), float(bw), float(bh)],  # xywh
                "area": area,
                "predicted_iou": float(score),   # use GDINO score as proxy
                "stability_score": float(score),  # proxy
                "quality_source": "gdino_conf_proxy",
                "label": str(label).strip().lower(),
                "gdino_conf": float(score),
                "source_model": "GroundedSAM2",
            })
        return results

    def update_text_query(self, query: str) -> None:
        """Update the GDINO text query (e.g. from RAM++ dynamic vocabulary)."""
        self.text_query = query


# -----------------------------------------------------------------------------
# 7. SAM2 Automatic Mask Generator Wrapper
# (kept as fallback for GroundedSAM2Wrapper)
# -----------------------------------------------------------------------------
class SAM2AMGWrapper:
    """
    Wrapper for SAM2 Automatic Mask Generator.
    Segments "everything" in an image; no prompts. Returns list of mask dicts.
    """
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        model_cfg: str,
        points_per_side: int = 32,
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        max_image_side: int = 1280,
        crop_n_layers: int = 1,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 200,
        use_m2m: bool = True,
        box_nms_thresh: float = 0.7,
    ):
        self.device = device
        self.amg = None
        self.max_image_side = int(max_image_side) if max_image_side else 0
        self._force_cpu = False
        self.crop_n_layers = crop_n_layers
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.use_m2m = use_m2m
        self.box_nms_thresh = box_nms_thresh
        print("Initializing SAM2 Automatic Mask Generator...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            # Checkpoint: resolve for existence check; pass path to build_sam2.
            ckpt = Path(checkpoint_path)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            if not ckpt.exists():
                print(f"SAM2 checkpoint not found at {ckpt}. Depth-mask branch disabled.")
                return
            # Config: pass as Hydra config name (e.g. "configs/sam2.1/sam2.1_hiera_l"), not a file path.
            # build_sam2 uses compose(config_name=...) so an absolute path would be mis-resolved.
            hydra_config_name = model_cfg if isinstance(model_cfg, str) else str(model_cfg)
            if hydra_config_name.endswith(".yaml"):
                hydra_config_name = hydra_config_name[:-5]
            print(f"Loading SAM2 with config: {hydra_config_name}")
            model = build_sam2(hydra_config_name, str(ckpt))

            model.to(device)
            self.amg = SAM2AutomaticMaskGenerator(
                model,
                output_mode="binary_mask",
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
                use_m2m=use_m2m,
                box_nms_thresh=box_nms_thresh,
            )
            print("SAM2 AMG initialized.")
        except ImportError as e:
            print(f"SAM2 not available: {e}. Depth-mask branch disabled.")
        except Exception as e:
            print(f"Failed to load SAM2 AMG: {e}. Depth-mask branch disabled.")

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run AMG on image (HWC uint8 RGB). Returns list of dicts with segmentation, bbox (xywh), area, predicted_iou, stability_score."""
        if self.amg is None:
            return []
        try:
            if self._force_cpu:
                self._move_model_to_cpu()

            anns = self._generate_with_optional_resize(image_rgb)
            return anns if anns else []
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"SAM2 AMG generate failed: {e}")
                return []

            print("SAM2 AMG OOM on GPU. Retrying with reduced memory settings.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._move_model_to_cpu()
                self._force_cpu = True
                anns = self._generate_with_optional_resize(image_rgb)
                return anns if anns else []
            except Exception as e2:
                print(f"SAM2 AMG generate failed after CPU fallback: {e2}")
                return []
        except Exception as e:
            print(f"SAM2 AMG generate failed: {e}")
            return []

    def _move_model_to_cpu(self) -> None:
        if self.amg is None:
            return
        try:
            predictor = getattr(self.amg, "predictor", None)
            model = getattr(predictor, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass

    def _generate_with_optional_resize(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        if self.amg is None:
            return []
        h, w = image_rgb.shape[:2]
        long_side = max(h, w)
        scale = 1.0
        proc = image_rgb
        if self.max_image_side > 0 and long_side > self.max_image_side:
            scale = float(self.max_image_side) / float(long_side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            proc = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        if torch.cuda.is_available() and getattr(self.amg.predictor, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            anns = self.amg.generate(proc)
        if not anns:
            return []

        if scale == 1.0:
            return anns

        inv = 1.0 / scale
        resized_anns = []
        for ann in anns:
            ann_new = dict(ann)
            seg = ann_new.get("segmentation")
            if seg is not None:
                mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                ann_new["segmentation"] = (mask > 0)
            bbox = ann_new.get("bbox")
            if bbox is not None and len(bbox) >= 4:
                x, y, bw, bh = bbox[:4]
                ann_new["bbox"] = [float(x * inv), float(y * inv), float(bw * inv), float(bh * inv)]
            if "area" in ann_new:
                ann_new["area"] = int(np.sum(np.asarray(ann_new["segmentation"]) > 0))
            resized_anns.append(ann_new)
        return resized_anns

# -----------------------------------------------------------------------------
# 8. SAM3 Wrapper
# Mirrors the GroundedSAM2Wrapper.generate() output format so it can slot into
# the same depth / labelling / relation pipeline as a parallel segmentor.
# Each mask dict gets "segmentor": "SAM3" for traceability.
# -----------------------------------------------------------------------------
class SAM3Wrapper:
    """
    SAM3 text-prompted segmentor.

    Uses ``build_sam3_image_model`` + ``Sam3Processor`` from the local
    ``sam3/`` repo (already cloned alongside this project).

    Output format matches GroundedSAM2Wrapper.generate():
      segmentation (bool HxW), bbox ([x,y,w,h] xywh), area (int),
      predicted_iou (float), stability_score (float),
      label (str), gdino_conf (float — here: SAM3 score),
      source_model (str — "SAM3")
    """

    def __init__(
        self,
        device: torch.device,
        text_query: str = (
            "person. animal. vehicle. furniture. appliance. food. "
            "clothing. container. tool. building. plant. electronics. object."
        ),
        confidence_threshold: float = 0.3,
        checkpoint_path: Optional[str] = None,
        load_from_hf: bool = True,
    ):
        self.device = device
        self.text_query = text_query
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None
        self.active = False

        print("Initializing SAM3...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            device_str = str(device) if hasattr(device, "__str__") else "cuda"
            if hasattr(device, "type"):
                device_str = device.type
            self._model = build_sam3_image_model(
                device=device_str,
                eval_mode=True,
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_hf,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            self._processor = Sam3Processor(
                self._model,
                device=device_str,
                confidence_threshold=confidence_threshold,
            )
            self.active = True
            print("SAM3 ready.")
        except Exception as e:
            print(f"SAM3 init failed (will skip SAM3 pass): {e}")
            self.active = False

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run SAM3 on an RGB image with the configured text query.

        Returns list of dicts in GroundedSAM2Wrapper format so the rest of the
        pipeline (depth stats, labelling, relations) treats them identically.
        Masks are full-resolution bool arrays (HxW).
        """
        if not self.active or self._processor is None:
            return []

        h, w = image_rgb.shape[:2]
        results = []
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_rgb)
            state = self._processor.set_image(pil_img)

            # Run one prompt per category phrase for finer recall, then
            # aggregate. Alternatively run a single compound prompt — we do
            # the single-prompt path here for efficiency.
            state = self._processor.set_text_prompt(
                prompt=self.text_query, state=state
            )

            masks_bool = state.get("masks")   # (N,1,H,W) bool tensor
            boxes = state.get("boxes")         # (N,4) float tensor [x0,y0,x1,y1]
            scores = state.get("scores")       # (N,) float tensor

            if masks_bool is None or len(masks_bool) == 0:
                print("  [SAM3] No masks returned.")
                return []

            masks_bool = masks_bool.cpu().numpy()   # (N,1,H,W)
            boxes_np   = boxes.cpu().numpy()        # (N,4)
            scores_np  = scores.cpu().numpy()       # (N,)

            for i in range(len(masks_bool)):
                m = masks_bool[i]
                if m.ndim == 3:
                    m = m[0]              # (H,W)
                mask = m.astype(bool)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                area = int(mask.sum())
                if area == 0:
                    continue

                # Convert xyxy box → xywh (SAM2 convention used elsewhere)
                x1, y1, x2, y2 = (
                    float(boxes_np[i, 0]), float(boxes_np[i, 1]),
                    float(boxes_np[i, 2]), float(boxes_np[i, 3]),
                )
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                score = float(scores_np[i])

                results.append({
                    "segmentation": mask,
                    "bbox": [x1, y1, bw, bh],   # xywh
                    "area": area,
                    "predicted_iou": score,
                    "stability_score": score,    # SAM3 has no separate stability; use score
                    "label": "object",           # SAM3 doesn't return per-mask labels; downstream labeller fills this
                    "gdino_conf": score,
                    "source_model": "SAM3",
                })

            print(f"  [SAM3] {len(results)} masks generated.")
        except Exception as e:
            print(f"  [SAM3] generate() failed: {e}")

        return results

    def unload(self) -> None:
        """Delete model weights and free VRAM so SAM2 can run next."""
        self._model = None
        self._processor = None
        self.active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[SAM3] Model unloaded, VRAM freed.")


# -----------------------------------------------------------------------------
# Sentinel for SAM3-only mode (no SAM2 loaded)
# -----------------------------------------------------------------------------
class _Sam2OnlySentinel:
    """Placeholder for sam2_wrapper when sam3_only=True; avoids loading SAM2."""

    active = False
    _amg_fallback = None

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        return []


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
class SceneUnderstandingPipeline:
    def __init__(
        self,
        depth_estimator: DepthEstimator,
        intrinsics: Optional[Dict] = None,
        depth_mask_modes: Optional[List[str]] = None,
        config: Optional[Any] = None,
    ):
        """
        Args:
            depth_estimator: Initialized instance from depth.py
            intrinsics: Dict {'fx', 'fy', 'cx', 'cy'}. If None, will be auto-estimated.
            depth_mask_modes: List of "A", "B" for matching (A=detection-first, B=mask-first). Default ["A", "B"].
            config: Optional PreprocessConfig for SAM2 paths and AMG params. If None, depth-mask uses defaults.
        """
        self.depth_estimator = depth_estimator
        self.device = depth_estimator.device
        self.fixed_intrinsics = intrinsics
        if config is not None and hasattr(config, "depth_mask_matching_modes"):
            self.depth_mask_modes = list(config.depth_mask_matching_modes)
        else:
            self.depth_mask_modes = depth_mask_modes if depth_mask_modes is not None else ["A", "B"]
        self.config = config
        self.require_any_relation_source = (
            bool(getattr(config, "require_any_relation_source", True)) if config is not None else True
        )
        self.mask_iou_match_thresh = (
            float(getattr(config, "mask_iou_match_thresh", 0.1)) if config is not None else 0.1
        )
        self.pix2sg_mask_overlap_thresh = (
            float(getattr(config, "pix2sg_mask_overlap_thresh", 0.05)) if config is not None else 0.05
        )
        self.pix2sg_depth_near_threshold = (
            float(getattr(config, "pix2sg_depth_near_threshold", 1.0)) if config is not None else 1.0
        )
        # Camera intrinsics
        self.camera_fx = getattr(config, "camera_fx", None) if config is not None else None
        self.camera_fy = getattr(config, "camera_fy", None) if config is not None else None
        self.camera_cx = getattr(config, "camera_cx", None) if config is not None else None
        self.camera_cy = getattr(config, "camera_cy", None) if config is not None else None
        self.camera_fov_degrees = float(getattr(config, "camera_fov_degrees", 60.0)) if config is not None else 60.0
        # Depth accuracy
        self.mask_erosion_kernel_size = int(getattr(config, "mask_erosion_kernel_size", 5)) if config is not None else 5
        self.depth_central_fraction = float(getattr(config, "depth_central_fraction", 0.5)) if config is not None else 0.5
        self.depth_scale_factor = float(getattr(config, "depth_scale_factor", 10.0)) if config is not None else 10.0
        # SAM2 post-hoc filter
        self.sam2_post_filter_min_stability = float(getattr(config, "sam2_post_filter_min_stability", 0.0)) if config is not None else 0.0
        self.sam2_post_filter_min_pred_iou = float(getattr(config, "sam2_post_filter_min_pred_iou", 0.0)) if config is not None else 0.0
        self.sam2_post_filter_min_area_px = int(getattr(config, "sam2_post_filter_min_area_px", 1000)) if config is not None else 1000
        self.sam2_post_filter_max_area_fraction = float(getattr(config, "sam2_post_filter_max_area_fraction", 0.35)) if config is not None else 0.35
        self.grounded_sam2_min_conf_for_stage3 = (
            float(
                getattr(
                    config,
                    "grounded_sam2_min_conf_for_stage3",
                    getattr(config, "grounding_dino_box_thresh", 0.25),
                )
            )
            if config is not None
            else 0.25
        )
        pix2sg_triplets_dir = (
            str(getattr(config, "pix2sg_triplets_dir", "pix2sg_triplets")) if config is not None else "pix2sg_triplets"
        )
        pix2sg_max_relations_per_object = (
            int(getattr(config, "pix2sg_spatial_max_relations_per_object", 8)) if config is not None else 8
        )
        pix2sg_depth_far_threshold = (
            float(getattr(config, "pix2sg_depth_far_threshold", 3.0)) if config is not None else 3.0
        )

        # Fix 5.7 — depth accuracy params
        self.depth_adaptive_erosion = bool(getattr(config, "depth_adaptive_erosion", True)) if config is not None else True
        self.depth_outlier_sigma = float(getattr(config, "depth_outlier_sigma", 2.0)) if config is not None else 2.0
        self.depth_transparency_check = bool(getattr(config, "depth_transparency_check", True)) if config is not None else True
        self.depth_transparency_threshold = float(getattr(config, "depth_transparency_threshold", 0.15)) if config is not None else 0.15

        # Fix 5.2 — calibration file
        self._calibration: Optional[Dict] = None
        cal_file = getattr(config, "camera_calibration_file", None) if config is not None else None
        if cal_file:
            self._calibration = self._load_calibration(cal_file)
        self.apply_undistortion = bool(getattr(config, "apply_undistortion", True)) if config is not None else True

        # Labellers — lazy-loaded on first use, unloaded after Stage 5 to save VRAM
        self._florence2_model_id = getattr(config, "florence2_model", "microsoft/Florence-2-large") if config is not None else "microsoft/Florence-2-large"
        self._florence2_label_enabled = bool(getattr(config, "florence2_label_enabled", True)) if config is not None else True
        self._rampp_enabled = bool(getattr(config, "rampp_enabled", True)) if config is not None else True
        self._rampp_checkpoint_path = getattr(config, "rampp_checkpoint_path", None) if config is not None else None
        self._rampp_repo_path = getattr(config, "rampp_repo_path", None) if config is not None else None
        self._rampp_image_size = int(getattr(config, "rampp_image_size", 384)) if config is not None else 384
        self._rampp_vit = str(getattr(config, "rampp_vit", "swin_l")) if config is not None else "swin_l"
        self._rampp_default_conf = float(getattr(config, "rampp_default_confidence", 0.70)) if config is not None else 0.70
        self._rampp_max_tags = int(getattr(config, "rampp_max_tags", 8)) if config is not None else 8
        self.florence2: Optional[Florence2Wrapper] = None
        self.rampp: Optional[RAMPlusPlusWrapper] = None
        print("Labellers (Florence-2, RAM++) will load on first image.")

        # Fix 5.6 — Pix2SG with Florence-2 enrichment
        relation_min_mask_overlap = float(getattr(config, "relation_min_mask_overlap", 0.02)) if config is not None else 0.02
        self.pix2sg = Pix2SGWrapper(
            self.device,
            triplets_dir=pix2sg_triplets_dir,
            max_relations_per_object=pix2sg_max_relations_per_object,
            mask_overlap_thresh=self.pix2sg_mask_overlap_thresh,
            depth_near_threshold=self.pix2sg_depth_near_threshold,
            depth_far_threshold=pix2sg_depth_far_threshold,
            florence2=None,  # injected lazily at process_image() time after load
            relation_min_mask_overlap=relation_min_mask_overlap,
        )
        self._relation_source_status = self._collect_relation_source_status()
        self._print_relation_source_status()
        self._assert_relation_sources_or_fail()

        # SAM3-only mode: skip SAM2 entirely (Run 2 for SAM2 vs SAM3 comparison)
        self._sam3_only = bool(getattr(config, "sam3_only", False)) if config is not None else False
        self._sam3_only_use_existing_depth = (
            bool(getattr(config, "sam3_only_use_existing_depth", False)) if config is not None else False
        )

        if self._sam3_only:
            self.sam2_wrapper = _Sam2OnlySentinel()
            self._run_sam3 = True
            print("SAM3-only mode: SAM2 skipped; only depth + SAM3 will run.")
        else:
            # Fix 5.3 — GroundedSAM2 (replaces SAM2AMGWrapper)
            if self.config is not None:
                ckpt = getattr(self.config, "sam2_checkpoint_path", "sam2/checkpoints/sam2.1_hiera_large.pt")
                cfg = getattr(self.config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l")
                gdino_model = getattr(self.config, "grounding_dino_model", "IDEA-Research/grounding-dino-base")
                gdino_box_thresh = float(getattr(self.config, "grounding_dino_box_thresh", 0.30))
                gdino_text_thresh = float(getattr(self.config, "grounding_dino_text_thresh", 0.25))
                gdino_query = getattr(self.config, "grounding_dino_text_query",
                    "person. man. woman. child. animal. dog. cat. car. truck. bicycle. motorcycle. bus. "
                    "chair. table. desk. sofa. bed. shelf. cabinet. door. window. floor. wall. ceiling. "
                    "bottle. cup. bowl. plate. glass. fork. knife. spoon. pot. pan. "
                    "laptop. phone. keyboard. monitor. television. remote. camera. "
                    "bag. backpack. suitcase. box. basket. "
                    "book. paper. pen. clock. lamp. mirror. painting. "
                    "tree. plant. flower. grass. sky. road. building. sign.")
                fallback_amg = bool(getattr(self.config, "grounded_sam2_fallback_to_amg", True))
                pts = getattr(self.config, "sam2_amg_points_per_side", 32)
                ppb = getattr(self.config, "sam2_amg_points_per_batch", 32)
                iou = getattr(self.config, "sam2_amg_pred_iou_thresh", 0.80)
                stab = getattr(self.config, "sam2_amg_stability_score_thresh", 0.92)
                max_side = getattr(self.config, "sam2_amg_max_image_side", 1280)
                min_region = int(getattr(self.config, "sam2_amg_min_mask_region_area", 1000))
                use_m2m = bool(getattr(self.config, "sam2_amg_use_m2m", True))
                box_nms = float(getattr(self.config, "sam2_amg_box_nms_thresh", 0.7))
                self.sam2_wrapper = GroundedSAM2Wrapper(
                    device=self.device,
                    sam2_checkpoint_path=ckpt,
                    sam2_model_cfg=cfg,
                    gdino_model_id=gdino_model,
                    box_thresh=gdino_box_thresh,
                    text_thresh=gdino_text_thresh,
                    text_query=gdino_query,
                    max_image_side=max_side,
                    fallback_to_amg=fallback_amg,
                    points_per_side=pts,
                    points_per_batch=ppb,
                    pred_iou_thresh=iou,
                    stability_score_thresh=stab,
                    min_mask_region_area=min_region,
                    use_m2m=use_m2m,
                    box_nms_thresh=box_nms,
                )
            else:
                self.sam2_wrapper = GroundedSAM2Wrapper(
                    device=self.device,
                    sam2_checkpoint_path="sam2/checkpoints/sam2.1_hiera_large.pt",
                    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l",
                )
            # Validate SAM2 predictor or AMG is available
            if not self.sam2_wrapper.active and getattr(self.sam2_wrapper, "_amg_fallback", None) is None:
                raise RuntimeError(
                    "Both Grounded-SAM2 and SAM2 AMG fallback failed to initialize."
                )

            # SAM3 — optional segmentor; weights loaded lazily per-image to save VRAM
            self._run_sam3 = bool(getattr(config, "run_sam3", False)) if config is not None else False

        self.sam3_wrapper: Optional[SAM3Wrapper] = None
        # Store config params — actual model load happens in process_image() after SAM2 unloads
        if self._run_sam3:
            self._sam3_text_query = getattr(
                config, "sam3_text_query",
                "person. animal. vehicle. furniture. appliance. food. "
                "clothing. container. tool. building. plant. electronics. object.",
            ) if config is not None else (
                "person. animal. vehicle. furniture. appliance. food. "
                "clothing. container. tool. building. plant. electronics. object."
            )
            self._sam3_conf = float(getattr(config, "sam3_confidence_threshold", 0.3)) if config is not None else 0.3
            self._sam3_ckpt = getattr(config, "sam3_checkpoint_path", None) if config is not None else None
            self._sam3_hf = bool(getattr(config, "sam3_load_from_hf", True)) if config is not None else True
            print("SAM3 enabled — weights will load per-image after SAM2 completes.")

    def _collect_relation_source_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Pix2SG": self.pix2sg.status(),
        }

    def _print_relation_source_status(self) -> None:
        print("=== Relation Source Diagnostics ===")
        for name, status in self._relation_source_status.items():
            state = "ACTIVE" if status.get("active") else "INACTIVE"
            backend = status.get("backend", "unknown")
            reason = status.get("reason", "")
            print(f"{name}: {state} (backend={backend})")
            if reason:
                print(f"  reason: {reason}")

    def _assert_relation_sources_or_fail(self) -> None:
        if not self.require_any_relation_source:
            return
        if any(s.get("active") for s in self._relation_source_status.values()):
            return
        details = "; ".join(
            f"{name}: {status.get('reason', 'inactive')}"
            for name, status in self._relation_source_status.items()
        )
        raise RuntimeError(
            "No active relation source is available. "
            f"Disable strict check via require_any_relation_source=False, or fix dependencies. Details: {details}"
        )

    def _load_labellers(self) -> None:
        """Load Florence-2 and RAM++ into VRAM. Called once per image before Stage 4."""
        need_florence = self._florence2_label_enabled or bool(getattr(self.config, "florence2_relation_enabled", True))
        if need_florence and (self.florence2 is None or not self.florence2.active):
            self.florence2 = Florence2Wrapper(model_id=self._florence2_model_id, device=self.device)

        if self._rampp_enabled and (self.rampp is None or not self.rampp.active):
            self.rampp = RAMPlusPlusWrapper(
                device=self.device,
                checkpoint_path=self._rampp_checkpoint_path,
                repo_path=self._rampp_repo_path,
                image_size=self._rampp_image_size,
                vit=self._rampp_vit,
                default_confidence=self._rampp_default_conf,
                max_tags=self._rampp_max_tags,
            )

        # Inject loaded Florence-2 into Pix2SG so relations use it
        if bool(getattr(self.config, "florence2_relation_enabled", True)) and self.florence2 is not None and self.florence2.active:
            self.pix2sg._florence2 = self.florence2
        else:
            self.pix2sg._florence2 = None

    def _unload_labellers(self) -> None:
        """Unload Florence-2 and RAM++ to free VRAM after Stage 5."""
        if self.florence2 is not None:
            self.florence2.model = None
            self.florence2.processor = None
            self.florence2.active = False
        self.florence2 = None
        if self.rampp is not None:
            self.rampp.model = None
            self.rampp.transform = None
            self.rampp.inference_fn = None
            self.rampp.active = False
        self.rampp = None
        self.pix2sg._florence2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  [Labellers] Florence-2 / RAM++ unloaded, VRAM freed.")

    @staticmethod
    def _load_calibration(cal_file: str) -> Optional[Dict]:
        """
        Fix 5.2: Load OpenCV camera calibration JSON produced by
        tools/calibrate_camera.py.

        Expected JSON structure:
          {
            "fx": float, "fy": float, "cx": float, "cy": float,
            "k1": float, "k2": float, "p1": float, "p2": float,
            "image_size": [w, h]
          }

        Why calibration matters:
          A checkerboard calibration with 20+ images gives focal length
          accurate to <0.5% and principal point to <2 px.  The FOV estimate
          can be off by 10-30% for non-standard lenses or cropped sensors,
          directly corrupting X/Y in coordinates_3d.
          Distortion coefficients (k1,k2,p1,p2) correct barrel/pincushion
          distortion; without undistortion, depth back-projection assumes a
          perfect pinhole which is violated by real lenses.
        """
        try:
            with open(cal_file, "r") as f:
                cal = json.load(f)
            required = {"fx", "fy", "cx", "cy"}
            if not required.issubset(cal.keys()):
                print(f"[Calibration] Missing keys in {cal_file}. Need {required}.")
                return None
            print(f"[Calibration] Loaded from {cal_file}: "
                  f"fx={cal['fx']:.1f} fy={cal['fy']:.1f} "
                  f"cx={cal['cx']:.1f} cy={cal['cy']:.1f}")
            return cal
        except Exception as e:
            print(f"[Calibration] Failed to load {cal_file}: {e}")
            return None

    def _undistort_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Fix 5.2: Apply lens distortion correction using calibration coefficients.

        Uses OpenCV cv2.undistort() with the loaded k1,k2,p1,p2 coefficients.
        If no calibration is loaded or undistortion is disabled, returns the
        image unchanged.

        When to enable:
          - DSLR / mirrorless cameras: usually mild distortion (k1 ≈ -0.05)
          - Wide-angle / fisheye lenses: strong distortion (k1 ≈ -0.3 or worse)
          - Smartphone cameras at 0.5× zoom: significant barrel distortion
        """
        if self._calibration is None or not self.apply_undistortion:
            return img_bgr
        try:
            cal = self._calibration
            h, w = img_bgr.shape[:2]
            K_mat = np.array([
                [cal["fx"], 0.0,       cal["cx"]],
                [0.0,       cal["fy"], cal["cy"]],
                [0.0,       0.0,       1.0],
            ], dtype=np.float64)
            dist_coeffs = np.array([
                cal.get("k1", 0.0), cal.get("k2", 0.0),
                cal.get("p1", 0.0), cal.get("p2", 0.0),
            ], dtype=np.float64)
            return cv2.undistort(img_bgr, K_mat, dist_coeffs)
        except Exception as e:
            print(f"[Undistort] Failed: {e}. Using original image.")
            return img_bgr

    def _estimate_intrinsics(self, width: int, height: int) -> Dict[str, float]:
        """
        Fix 5.2: Return camera intrinsics with priority order:
          1. Calibration file (OpenCV checkerboard calibration — most accurate)
          2. Explicit camera_fx / camera_fy / camera_cx / camera_cy in config
          3. FOV-based estimate (least accurate; error can be 10-30%)

        The returned dict is used for all back-projection in Stage 4.
        """
        # Priority 1: calibration file
        if self._calibration is not None:
            cal = self._calibration
            return {
                "fx": float(cal["fx"]),
                "fy": float(cal["fy"]),
                "cx": float(cal.get("cx", width / 2)),
                "cy": float(cal.get("cy", height / 2)),
            }
        # Priority 2: explicit values
        if self.camera_fx is not None:
            return {
                "fx": float(self.camera_fx),
                "fy": float(self.camera_fy if self.camera_fy is not None else self.camera_fx),
                "cx": float(self.camera_cx if self.camera_cx is not None else width / 2),
                "cy": float(self.camera_cy if self.camera_cy is not None else height / 2),
            }
        # Priority 3: FOV estimate
        f_x = (width / 2) / np.tan(np.deg2rad(self.camera_fov_degrees) / 2)
        print(f"  [Intrinsics] Using FOV estimate ({self.camera_fov_degrees}°): fx=fy={f_x:.1f}")
        return {"fx": f_x, "fy": f_x, "cx": width / 2, "cy": height / 2}

    def _back_project(self, u: int, v: int, z: float, K: Dict[str, float]) -> Dict[str, float]:
        x = (u - K['cx']) * z / K['fx']
        y = (v - K['cy']) * z / K['fy']
        return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3)}

    @staticmethod
    def _bbox_iou_xyxy(box1: List[float], box2: List[float]) -> float:
        """IoU of two boxes in xyxy format [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    @staticmethod
    def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
        x, y, w, h = bbox_xywh[:4]
        return [x, y, x + w, y + h]

    def _match_mask_first(
        self,
        amg_masks: List[Dict],
        detections: List[Dict],
        iou_thresh: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Option B: Identity mapping — detection i corresponds to amg_masks[i].

        Since all_detections is built directly from amg_masks (one entry per mask),
        the match is exact: mask index == detection index. The iou_thresh parameter
        is kept for API compatibility but is not used.
        """
        out = []
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            amg_xyxy = self._xywh_to_xyxy(amg.get("bbox", [0, 0, 0, 0]))
            det = detections[idx] if idx < len(detections) else None
            out.append({
                "mask": mask,
                "sam2_mask_index": idx,
                "detection": det,
                "mask_bbox_xyxy": amg_xyxy,
            })
        return out

    def _adaptive_erosion_kernel(self, mask_bin: np.ndarray) -> int:
        """Return erosion kernel size scaled to the mask's narrowest dimension. 0 = skip erosion."""
        if not self.depth_adaptive_erosion or self.mask_erosion_kernel_size == 0:
            return self.mask_erosion_kernel_size
        ys, xs = np.where(mask_bin)
        if ys.size == 0:
            return 0
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        min_dim = min(bbox_h, bbox_w)
        # Scale table: fraction of narrowest dimension used as kernel
        if min_dim < 15:
            return 0    # very thin object — skip erosion entirely
        elif min_dim < 40:
            return 1
        elif min_dim < 80:
            return 2
        elif min_dim < 150:
            return min(3, self.mask_erosion_kernel_size)
        else:
            return self.mask_erosion_kernel_size

    def _mask_depth_stats_and_3d(
        self,
        metric_depth: np.ndarray,
        K: Dict[str, float],
        mask: np.ndarray,
        detection: Optional[Dict] = None,
        use_erosion: bool = True,
    ) -> tuple:
        """
        Compute depth stats and 3D coords from mask pixels.
        use_erosion=False skips adaptive erosion (for comparison stats).
        Returns (depth_stats_dict, coordinates_3d, mask_centroid_2d).
        See docs/DEPTH_ACCURACY.md for all formulas.
        """
        h, w = metric_depth.shape[:2]
        mask_bin = (np.asarray(mask) > 0)
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_bin > 0)

        if use_erosion:
            kernel_size = self._adaptive_erosion_kernel(mask_bin)
            if kernel_size > 0 and int(mask_bin.sum()) > 4 * kernel_size * kernel_size:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded = cv2.erode(mask_bin.astype(np.uint8), kernel, iterations=1)
                if eroded.sum() > 0:
                    mask_bin = (eroded > 0)

        ys, xs = np.where(mask_bin)
        depth_at_mask = metric_depth[ys, xs]
        finite_mask = np.isfinite(depth_at_mask)
        depth_at_mask = depth_at_mask[finite_mask]
        ys_f = ys[finite_mask]
        xs_f = xs[finite_mask]

        if depth_at_mask.size == 0:
            depth_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0,
                           "num_pixels": 0, "z_val_pixels": 0,
                           "possibly_transparent": False, "depth_separation_from_background": 0.0}
            coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
            centroid = [w // 2, h // 2]
            return depth_stats, coords_3d, centroid

        # Sigma-clipping: reject |depth_i - mean| > sigma * std  (see docs/DEPTH_ACCURACY.md)
        sigma = self.depth_outlier_sigma
        if sigma > 0 and depth_at_mask.size >= 10:
            mean_d = float(np.mean(depth_at_mask))
            std_d = float(np.std(depth_at_mask))
            if std_d > 1e-6:
                inlier = np.abs(depth_at_mask - mean_d) < sigma * std_d
                if inlier.sum() >= 5:
                    depth_at_mask = depth_at_mask[inlier]
                    ys_f = ys_f[inlier]
                    xs_f = xs_f[inlier]

        # Transparency detection via 5px border ring (see docs/DEPTH_ACCURACY.md)
        possibly_transparent = False
        depth_separation = 0.0
        if self.depth_transparency_check and mask_bin.sum() > 0:
            try:
                kernel_5 = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(mask_bin.astype(np.uint8), kernel_5) > 0
                border_ring = dilated & ~mask_bin
                border_depths = metric_depth[border_ring]
                border_depths = border_depths[np.isfinite(border_depths)]
                if border_depths.size > 0 and depth_at_mask.size > 0:
                    mask_mean = float(np.mean(depth_at_mask))
                    border_mean = float(np.mean(border_depths))
                    depth_separation = abs(mask_mean - border_mean)
                    possibly_transparent = depth_separation < self.depth_transparency_threshold
            except Exception:
                pass

        # Depth-weighted centroid: w_i = 1/(depth_i + ε)  (see docs/DEPTH_ACCURACY.md)
        weights = 1.0 / (depth_at_mask + 1e-6)
        w_sum = float(weights.sum())
        cy_f = float(np.sum(ys_f * weights) / w_sum)
        cx_f = float(np.sum(xs_f * weights) / w_sum)

        # Nearest real mask pixel to the weighted centroid (no holes in anchor)
        dist2 = (ys_f - cy_f) ** 2 + (xs_f - cx_f) ** 2
        anchor_idx = int(np.argmin(dist2))
        cx = int(xs_f[anchor_idx])
        cy = int(ys_f[anchor_idx])

        # z_val: histogram mode over inner-circle pixels  (see docs/DEPTH_ACCURACY.md)
        central_frac = self.depth_central_fraction
        if central_frac < 1.0:
            area = float(mask_bin.sum())
            radius = np.sqrt(area * central_frac / np.pi)
            inner_mask = dist2 <= radius ** 2
            inner_depths = depth_at_mask[inner_mask]
        else:
            inner_depths = depth_at_mask
        z_val_pixels = int(inner_depths.size)
        if z_val_pixels > 0:
            n_bins = max(10, min(100, z_val_pixels // 5))
            hist, edges = np.histogram(inner_depths, bins=n_bins)
            peak_bin = int(np.argmax(hist))
            z_val = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
        else:
            z_val = float(np.median(depth_at_mask))
            z_val_pixels = int(depth_at_mask.size)

        depth_stats = {
            "min": round(float(np.min(depth_at_mask)), 4),
            "max": round(float(np.max(depth_at_mask)), 4),
            "mean": round(float(np.mean(depth_at_mask)), 4),
            "median": round(float(np.median(depth_at_mask)), 4),
            "std": round(float(np.std(depth_at_mask)), 4),
            "num_pixels": int(mask_bin.sum()),
            "z_val": round(z_val, 4),
            "z_val_pixels": z_val_pixels,
            # Fix 5.7c: transparency diagnostics
            "possibly_transparent": bool(possibly_transparent),
            "depth_separation_from_background": round(depth_separation, 4),
        }
        coords_3d = self._back_project(cx, cy, z_val, K)
        return depth_stats, coords_3d, [cx, cy]

    def _label_mask(
        self,
        img_bgr: np.ndarray,
        mask_bin: np.ndarray,
        amg_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Label a mask via priority chain: GDINO → Florence-2 (optional) → RAM++.
        """
        h_img, w_img = img_bgr.shape[:2]
        x, y, bw, bh = amg_entry.get("bbox", [0, 0, w_img, h_img])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img, int(x + bw)), min(h_img, int(y + bh))

        if x2 <= x1 or y2 <= y1:
            return {
                "label": "object",
                "conf": 0.0,
                "caption": "object",
                "source_model": "fallback",
                "florence2_label": "",
                "florence2_caption": "",
                "rampp_label": "",
                "rampp_caption": "",
                "rampp_tags": [],
            }

        crop = img_bgr[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        mask_resized = cv2.resize(
            mask_bin.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        bg_mean = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
        crop_filled = crop.copy()
        crop_filled[~mask_resized] = bg_mean

        f2_label, f2_caption = "object", "object"
        if (
            self._florence2_label_enabled
            and self.florence2 is not None
            and self.florence2.active
        ):
            f2_result = self.florence2.label_crop(crop_filled)
            f2_label = str(f2_result.get("label", "object")).strip().lower() or "object"
            f2_caption = str(f2_result.get("caption", "object"))

        rampp_label, rampp_caption, rampp_tags = "object", "object", []

        # Priority 1: GDINO label (wins if specific — not "object")
        gdino_label = str(amg_entry.get("label", "object")).strip().lower()
        gdino_conf = float(amg_entry.get("gdino_conf", 0.0))
        if gdino_label and gdino_label != "object":
            return {
                "label": gdino_label,
                "conf": gdino_conf,
                "caption": gdino_label,
                "source_model": "GroundingDINO",
                "florence2_label": f2_label,
                "florence2_caption": f2_caption,
                "rampp_label": rampp_label,
                "rampp_caption": rampp_caption,
                "rampp_tags": rampp_tags,
            }

        # Priority 2: Florence-2
        if f2_label != "object":
            return {
                "label": f2_label,
                "conf": 0.75,
                "caption": f2_caption,
                "source_model": "Florence-2",
                "florence2_label": f2_label,
                "florence2_caption": f2_caption,
                "rampp_label": rampp_label,
                "rampp_caption": rampp_caption,
                "rampp_tags": rampp_tags,
            }

        # Priority 3: RAM++
        rampp_conf = 0.0
        if self.rampp is not None and self.rampp.active:
            rampp_result = self.rampp.label_crop(crop_filled)
            rampp_label = str(rampp_result.get("label", "object")).strip().lower() or "object"
            rampp_caption = str(rampp_result.get("caption", "object"))
            rampp_tags = list(rampp_result.get("tags", []))
            rampp_conf = float(rampp_result.get("conf", 0.0))

        if rampp_label != "object":
            return {
                "label": rampp_label,
                "conf": rampp_conf,
                "caption": rampp_caption,
                "source_model": "RAM++",
                "florence2_label": f2_label,
                "florence2_caption": f2_caption,
                "rampp_label": rampp_label,
                "rampp_caption": rampp_caption,
                "rampp_tags": rampp_tags,
            }

        return {
            "label": "object",
            "conf": 0.0,
            "caption": f2_caption if f2_caption else "object",
            "source_model": "fallback",
            "florence2_label": f2_label,
            "florence2_caption": f2_caption,
            "rampp_label": rampp_label,
            "rampp_caption": rampp_caption,
            "rampp_tags": rampp_tags,
        }

    def _attach_relations_by_triplets(
        self,
        objects_3d: List[Dict[str, Any]],
        triplets: List[Dict[str, Any]],
        source_name: str,
    ) -> Dict[str, int]:
        """Attach triplets to fused objects using IDs first, then label matching."""
        stats = {
            "input_triplets": int(len(triplets)),
            "attached": 0,
            "subject_id_matched": 0,
            "subject_label_matched": 0,
            "target_id_matched": 0,
            "target_label_matched": 0,
            "external_targets": 0,
            "unmatched_subjects": 0,
        }
        if not triplets:
            return stats

        id_to_obj = {str(o.get("id")): o for o in objects_3d}

        def _find_by_label(label: str) -> Optional[Dict[str, Any]]:
            needle = str(label).strip().lower()
            if not needle:
                return None
            for obj in objects_3d:
                obj_label = str(obj.get("label", "")).lower()
                if needle in obj_label or obj_label in needle:
                    return obj
            return None

        for triplet in triplets:
            source_obj = None
            sub_id = triplet.get("sub_id")
            if sub_id is not None:
                source_obj = id_to_obj.get(str(sub_id))
                if source_obj is not None:
                    stats["subject_id_matched"] += 1
            if source_obj is None:
                source_obj = _find_by_label(triplet.get("sub", ""))
                if source_obj is not None:
                    stats["subject_label_matched"] += 1
            if source_obj is None:
                stats["unmatched_subjects"] += 1
                continue

            target_id = None
            raw_obj_id = triplet.get("obj_id")
            if raw_obj_id is not None:
                target = id_to_obj.get(str(raw_obj_id))
                if target is not None:
                    target_id = target.get("id")
                    stats["target_id_matched"] += 1
            if target_id is None:
                target = _find_by_label(triplet.get("obj", ""))
                if target is not None:
                    target_id = target.get("id")
                    stats["target_label_matched"] += 1
                else:
                    target_label = str(triplet.get("obj", "unknown")).strip().lower() or "unknown"
                    target_id = f"external_{target_label}"
                    stats["external_targets"] += 1

            # Resolve target label + caption for relation enrichment
            _target_obj = id_to_obj.get(str(target_id)) if target_id and not str(target_id).startswith("external_") else None
            if _target_obj is not None:
                _target_label = str(_target_obj.get("label", "unknown"))
                _target_src = _target_obj.get("sources", {})
                _target_caption = (
                    _target_src.get("GroundedSAM2", {}).get("caption")
                    or _target_src.get("RAM++", {}).get("caption")
                    or _target_src.get("Florence2", {}).get("caption")
                    or ""
                )
            else:
                _target_label = str(target_id).replace("external_", "") if target_id else "unknown"
                _target_caption = ""

            relation_entry = {
                "predicate": str(triplet.get("pred", "related_to")),
                "target_id": target_id,
                "target_label": _target_label,
                "target_caption": _target_caption,
            }
            if "score" in triplet:
                relation_entry["score"] = round(float(triplet["score"]), 4)

            source_obj["sources"].setdefault(source_name, {"relations": []})
            source_obj["sources"][source_name]["relations"].append(relation_entry)
            stats["attached"] += 1

        return stats

    def _save_depth_map_image(self, metric_depth: np.ndarray, path: Path) -> None:
        """Save full depth as colormap PNG."""
        d = metric_depth.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min < 1e-8:
            vis = np.zeros((*d.shape, 3), dtype=np.uint8)
        else:
            vis = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(path), vis)

    @staticmethod
    def _mask_colour(seed: int) -> tuple:
        """Deterministic BGR colour from integer seed."""
        rng = np.random.RandomState(seed)
        r, g, b = rng.randint(60, 230, 3)
        return (int(b), int(g), int(r))  # BGR

    @staticmethod
    def _draw_label(canvas_bgr: np.ndarray, text: str, cx: int, cy: int, mask_area: int) -> None:
        """Draw a label string with a dark pill background at (cx, cy)."""
        if not text:
            return
        # Scale font with mask area (clamp between 0.35 and 0.7)
        scale = float(np.clip(np.sqrt(mask_area) / 250.0, 0.35, 0.70))
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = 3
        x0 = max(0, cx - tw // 2 - pad)
        y0 = max(0, cy - th - pad)
        x1 = min(canvas_bgr.shape[1] - 1, cx + tw // 2 + pad)
        y1 = min(canvas_bgr.shape[0] - 1, cy + baseline + pad)
        # Dark semi-transparent pill
        roi = canvas_bgr[y0:y1, x0:x1].astype(np.float32)
        roi[:] = roi * 0.35
        canvas_bgr[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        cv2.putText(canvas_bgr, text, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _save_labelled_segmentation(
        self,
        objects_3d: List[Dict],
        path: Path,
    ) -> None:
        """
        Coloured segment map (one colour per object) with label text at each
        mask centroid. Uses _sam2_mask_array from objects_3d (still present
        before strip). Falls back to bbox if centroid unavailable.
        """
        if not objects_3d:
            return
        # Derive canvas size from first valid mask
        h, w = 0, 0
        for obj in objects_3d:
            m = obj.get("_sam2_mask_array")
            if m is not None:
                h, w = np.asarray(m).shape[:2]
                break
        if h == 0 or w == 0:
            return

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i, obj in enumerate(objects_3d):
            mask = obj.get("_sam2_mask_array")
            if mask is None:
                continue
            mask = np.asarray(mask)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            bin_mask = mask > 0
            colour = self._mask_colour(i)
            canvas[bin_mask] = colour[::-1]  # store as RGB then convert at save
            # Contour in slightly brighter shade
            contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bright = tuple(min(255, c + 60) for c in colour[::-1])
            cv2.drawContours(canvas, contours, -1, bright, 1)

        # Draw labels after all fills so text is on top
        for i, obj in enumerate(objects_3d):
            label = str(obj.get("label", "object"))
            mc = obj.get("mask_centroid_2d")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            cx = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
            cy = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
            mask = obj.get("_sam2_mask_array")
            area = int(np.sum(np.asarray(mask) > 0)) if mask is not None else 1000
            # canvas is BGR already (colour stored as BGR above via colour flip)
            self._draw_label(canvas, label, cx, cy, area)

        cv2.imwrite(str(path), canvas)

    def _save_labelled_tinted_overlay(
        self,
        objects_3d: List[Dict],
        image_rgb: np.ndarray,
        path: Path,
        alpha: float = 0.45,
    ) -> None:
        """
        Original photo with each mask as a semi-transparent colour tint and
        label text drawn at the mask centroid.
        """
        if not objects_3d or image_rgb is None:
            return
        h, w = image_rgb.shape[:2]
        out = image_rgb.copy().astype(np.float32)

        for i, obj in enumerate(objects_3d):
            mask = obj.get("_sam2_mask_array")
            if mask is None:
                continue
            mask = np.asarray(mask)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            bin_mask = mask > 0
            colour_bgr = self._mask_colour(i)
            colour_rgb = np.array([colour_bgr[2], colour_bgr[1], colour_bgr[0]], dtype=np.float32)
            out[bin_mask] = out[bin_mask] * (1 - alpha) + colour_rgb * alpha

        out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Draw labels on top of tints
        for i, obj in enumerate(objects_3d):
            label = str(obj.get("label", "object"))
            mc = obj.get("mask_centroid_2d")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            cx = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
            cy = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
            mask = obj.get("_sam2_mask_array")
            area = int(np.sum(np.asarray(mask) > 0)) if mask is not None else 1000
            self._draw_label(out_bgr, label, cx, cy, area)

        cv2.imwrite(str(path), out_bgr)

    def _save_sam2_outputs(
        self,
        amg_masks: List[Dict[str, Any]],
        h: int,
        w: int,
        out_dir: Path,
        path_stem: str,
        image_path: str,
        timestamp: str,
        image_rgb: np.ndarray = None,
    ) -> Dict[str, str]:
        """
        Save SAM2 outputs independently of depth-mask outputs.
        Returns relative paths from out_dir for metadata wiring.
        """
        sam2_dir = out_dir
        # Placeholder segmentation saved before labelling; overwritten after Stage 4
        # with the labelled version. We still record its path here for metadata.
        return {
            "sam2_segmentation_image_path": f"scene_graph/{path_stem}_sam2_segmentation.png",
            "sam2_tinted_overlay_image_path": f"scene_graph/{path_stem}_sam2_tinted_overlay.png",
        }

    def _save_sam3_outputs(
        self,
        sam3_masks: List[Dict[str, Any]],
        h: int,
        w: int,
        out_dir: Path,
        path_stem: str,
        image_path: str,
        timestamp: str,
        image_rgb: np.ndarray = None,
    ) -> Dict[str, str]:
        """
        Save SAM3 mask outputs to out_dir/sam3/ — mirrors _save_sam2_outputs.
        Returns relative paths for metadata wiring.
        """
        sam3_dir = out_dir / "sam3"
        sam3_dir.mkdir(parents=True, exist_ok=True)

        mask_records: List[Dict[str, Any]] = []
        for idx, m_dict in enumerate(sam3_masks):
            seg = m_dict.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            mask_records.append({
                "mask_index": idx,
                "mask_path": None,
                "bbox_xywh": [float(v) for v in m_dict.get("bbox", [0, 0, 0, 0])],
                "area": int(m_dict.get("area", int(np.sum(mask > 0)))),
                "predicted_iou": float(m_dict.get("predicted_iou", 0.0)),
                "stability_score": float(m_dict.get("stability_score", 0.0)),
                "label": str(m_dict.get("label", "object")),
                "sam3_score": float(m_dict.get("gdino_conf", 0.0)),
            })

        sam3_json = {
            "metadata": {
                "image_path": image_path,
                "image_stem": path_stem,
                "timestamp": timestamp,
                "image_size": [w, h],
                "model": "SAM3",
                "mode": "text_prompted",
            },
            "summary": {
                "num_masks": len(mask_records),
                "segmentation_map_image_path": f"sam3/{path_stem}_sam3_segmentation.png",
                "masks_dir": "sam3/masks",
            },
            "masks": mask_records,
        }

        sam3_json_path = sam3_dir / f"{path_stem}_sam3_masks.json"
        with open(sam3_json_path, "w") as f:
            json.dump(sam3_json, f, indent=2)

        return {
            "sam3_json_path": f"sam3/{path_stem}_sam3_masks.json",
            "sam3_segmentation_image_path": f"sam3/{path_stem}_sam3_segmentation.png",
            "sam3_tinted_overlay_image_path": f"sam3/{path_stem}_sam3_tinted_overlay.png",
        }

    def _save_depth_mask_mapping_image(
        self,
        metric_depth: np.ndarray,
        matched_objects: List[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Save depth colormap only where matched masks are; rest black."""
        h, w = metric_depth.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)
        for obj in matched_objects:
            m = obj.get("mask")
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask |= (m > 0)
        d = metric_depth.astype(np.float32)
        if np.any(combined_mask):
            d_min, d_max = float(d[combined_mask].min()), float(d[combined_mask].max())
        else:
            d_min, d_max = 0.0, 1.0
        if not np.any(combined_mask):
            vis = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            if d_max - d_min < 1e-8:
                d_max = d_min + 1.0
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            norm = np.clip(norm, 0, 255)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            vis[combined_mask] = colored[combined_mask]
        cv2.imwrite(str(path), vis)

    def _build_depth_mask_json(
        self,
        image_path: str,
        path_stem: str,
        timestamp: str,
        image_size: List[int],
        matching_mode: str,
        depth_map_path: str,
        depth_map_image_path: str,
        depth_global_min: float,
        depth_global_max: float,
        depth_global_mean: float,
        segmentation_map_image_path: str,
        num_auto_masks: int,
        mapping_image_path: str,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the depth+mask details JSON structure."""
        return {
            "metadata": {
                "image_path": image_path,
                "image_stem": path_stem,
                "timestamp": timestamp,
                "image_size": image_size,
                "matching_mode": matching_mode,
            },
            "depth": {
                "depth_map_path": depth_map_path,
                "depth_map_image_path": depth_map_image_path,
                "depth_min": depth_global_min,
                "depth_max": depth_global_max,
                "depth_mean": depth_global_mean,
            },
            "segmentation": {
                "model": "SAM2",
                "mode": "automatic_mask_generator",
                "segmentation_map_image_path": segmentation_map_image_path,
                "num_auto_masks": num_auto_masks,
                "match_strategy": "iou_with_detection_bbox",
            },
            "depth_mask": {
                "mapping_image_path": mapping_image_path,
                "objects": objects,
            },
        }

    def process_image(self, image_path: str, output_dir: str):
        path = Path(image_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Processing scene understanding for: {path.name}")
        
        img_bgr = _load_bgr_image(path)

        img_bgr = self._undistort_image(img_bgr)  # must run before depth/segmentation (see docs/CAMERA_CALIBRATION.md)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Downscale large images to keep RAM usage bounded.
        max_side = int(getattr(self.config, "sam2_amg_max_image_side", 1280)) if self.config else 1280
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
            print(f"  Resized to {w}x{h} (max_side={max_side})")

        # Intrinsics: calibration file > explicit values > FOV estimate (see docs/CAMERA_CALIBRATION.md)
        K = self.fixed_intrinsics if self.fixed_intrinsics else self._estimate_intrinsics(w, h)

        # Depth: compute or reuse from Run 1 when sam3_only + sam3_only_use_existing_depth
        depth_dir = out / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        if self._sam3_only and self._sam3_only_use_existing_depth:
            depth_npy = depth_dir / f"{path.stem}_depth_metric.npy"
            if depth_npy.exists():
                metric_depth = np.load(str(depth_npy)).astype(np.float32)
                if metric_depth.shape[:2] != (h, w):
                    metric_depth = cv2.resize(metric_depth, (w, h), interpolation=cv2.INTER_NEAREST)
                print(f"  [SAM3-only] Reusing depth from {depth_npy}")
            else:
                backend = getattr(self.depth_estimator, "backend", None)
                if backend is None:
                    raise FileNotFoundError(
                        f"Depth file {depth_npy} not found and depth backend was already unloaded (SAM3-only mode). "
                        "Run the pipeline without --sam3-only first to generate depth for all images, then re-run with --sam3-only --use-existing-depth."
                    )
                raw_depth = backend.infer(img_rgb)
                depth_full = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_NEAREST)
                metric_depth = depth_full * self.depth_scale_factor
                np.save(depth_dir / f"{path.stem}_depth_metric.npy", metric_depth)
        else:
            raw_depth = self.depth_estimator.backend.infer(img_rgb)  # float32 metres
            depth_full = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            metric_depth = depth_full * self.depth_scale_factor  # 1.0 for metric models
            np.save(depth_dir / f"{path.stem}_depth_metric.npy", metric_depth)

        if self._sam3_only:
            K_serializable = {k: float(v) for k, v in K.items()} if K else None
            scene_graph_dir = out / "scene_graph"
            scene_graph_dir.mkdir(parents=True, exist_ok=True)
            save_per_object_masks = False

        # 3. Object-level segmentation + labelling (see docs/SEGMENTATION.md, docs/LABELLING_AND_RELATIONS.md)
        if not self._sam3_only:
            scene_graph_dir = out / "scene_graph"
            scene_graph_dir.mkdir(parents=True, exist_ok=True)
            sam2_sg_dir = scene_graph_dir
            depth_mask_dir = sam2_sg_dir / "depth_mask"
            masks_dir = sam2_sg_dir / "masks"
            save_per_object_masks = False
            save_masked_depth_npy = getattr(self.config, "save_masked_depth_npy", False) if self.config else False

            self._save_depth_map_image(metric_depth, sam2_sg_dir / f"{path.stem}_depth_map.png")
            depth_map_image_rel = f"scene_graph/{path.stem}_depth_map.png"
            depth_map_npy_rel = f"depth/{path.stem}_depth_metric.npy"
            depth_global_min = float(np.min(metric_depth))
            depth_global_max = float(np.max(metric_depth))
            depth_global_mean = float(np.mean(metric_depth))

            # RAM++ dynamic vocabulary: tag the full image, build per-image GDINO query.
            # Load RAM++ now (before SAM2 generate) so the dynamic query is ready.
            # Florence-2 is deferred until Stage 4 to keep VRAM free during SAM2 inference.
            _rampp_tags_for_metadata: list = []
            _gdino_query_used: str = self.sam2_wrapper.text_query
            if self._rampp_enabled:
                if self.rampp is None or not self.rampp.active:
                    self.rampp = RAMPlusPlusWrapper(
                        device=self.device,
                        checkpoint_path=self._rampp_checkpoint_path,
                        repo_path=self._rampp_repo_path,
                        image_size=self._rampp_image_size,
                        vit=self._rampp_vit,
                        default_confidence=self._rampp_default_conf,
                        max_tags=self._rampp_max_tags,
                    )
                if self.rampp is not None and self.rampp.active:
                    _tag_result = self.rampp.tag_image(img_rgb)
                    _rampp_tags_for_metadata = list(_tag_result.get("tags", []))
                    if _rampp_tags_for_metadata:
                        # Build GDINO query: period-separated nouns from RAM++ tags
                        _dynamic_query = ". ".join(_rampp_tags_for_metadata) + "."
                        self.sam2_wrapper.update_text_query(_dynamic_query)
                        _gdino_query_used = _dynamic_query
                        print(f"  [RAM++] Tags: {', '.join(_rampp_tags_for_metadata)}")
                        print(f"  [RAM++] GDINO query updated ({len(_rampp_tags_for_metadata)} tags)")
                    else:
                        print("  [RAM++] No tags returned — using default GDINO query")

            amg_masks = self.sam2_wrapper.generate(img_rgb)

            if (
                getattr(self.config, "run_both_segmentors", False)
                and getattr(self.sam2_wrapper, "_amg_fallback", None) is not None
            ):
                amg_only = self.sam2_wrapper._amg_fallback.generate(img_rgb)
                iou_dedup = float(getattr(self.config, "run_both_segmentors_iou_dedup", 0.7))
                gdino_bins = [np.asarray(m["segmentation"]) > 0 for m in amg_masks]
                extra = []
                for amg_m in amg_only:
                    seg_amg = amg_m.get("segmentation")
                    if seg_amg is None:
                        continue
                    seg_bin = np.asarray(seg_amg) > 0
                    if seg_bin.sum() == 0:
                        continue
                    duplicate = False
                    for gbin in gdino_bins:
                        g = gbin
                        if g.shape != seg_bin.shape:
                            g = cv2.resize(g.astype(np.uint8), (seg_bin.shape[1], seg_bin.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        inter = int(np.logical_and(seg_bin, g).sum())
                        union = int(np.logical_or(seg_bin, g).sum())
                        if union > 0 and inter / union >= iou_dedup:
                            duplicate = True
                            break
                    if not duplicate:
                        amg_m["source_model"] = "SAM2_AMG"
                        extra.append(amg_m)
                n_gdino = len(amg_masks)
                amg_masks = amg_masks + extra
                print(f"  [DualSegmentor] GroundedSAM2={n_gdino}, AMG-extra(non-dup)={len(extra)}, total={len(amg_masks)}")

            seg_map_rel = f"scene_graph/{path.stem}_segmentation.png"
            sam2_paths = self._save_sam2_outputs(
                amg_masks=amg_masks,
                h=h,
                w=w,
                out_dir=sam2_sg_dir,
                path_stem=path.stem,
                image_path=str(path.resolve()),
                timestamp=timestamp,
                image_rgb=img_rgb,
            )

            # SAM3 runs after SAM2 (sequentially) — defer to after SAM2 pipeline completes
            sam3_masks: List[Dict[str, Any]] = []
            sam3_paths: Dict[str, str] = {}

            all_detections = []
            for i, amg in enumerate(amg_masks):
                seg = amg.get("segmentation")
                mask_bin = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                det = self._label_mask(img_bgr, mask_bin, amg)
                det["graph_id"] = f"obj_{i}_GroundedSAM2"
                det["sam2_mask_index"] = int(i)
                det["grounded_sam2_label"] = str(amg.get("label", det.get("label", "object"))).strip().lower()
                det["grounded_sam2_confidence"] = float(amg.get("gdino_conf", amg.get("predicted_iou", 0.0)))
                det["bbox"] = self._xywh_to_xyxy(amg.get("bbox", [0, 0, w, h]))  # SAM2 bbox, viz only
                det["segmentor"] = str(amg.get("source_model", "GroundedSAM2"))
                all_detections.append(det)

            print(f"Stage 3: {len(all_detections)} mask-objects labelled (no filtering — all masks kept)")
            # Free temporary GPU cache from per-object labelling calls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            matched_A_lookup: Dict[str, Dict[str, Any]] = {}
            for det in all_detections:
                idx = int(det.get("sam2_mask_index", -1))
                if 0 <= idx < len(amg_masks):
                    seg = amg_masks[idx].get("segmentation")
                    mask = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                else:
                    idx = -1
                    mask = np.zeros((h, w), dtype=bool)
                matched_A_lookup[det["graph_id"]] = {
                    "detection": det,
                    "mask": mask,
                    "sam2_mask_index": idx,
                    "mask_bbox_xyxy": det["bbox"],
                }

            for mode in self.depth_mask_modes:
                if mode == "A":
                    matched = [matched_A_lookup[det["graph_id"]] for det in all_detections]
                elif mode == "B":
                    matched = self._match_mask_first(
                        amg_masks, all_detections, iou_thresh=self.mask_iou_match_thresh
                    ) if amg_masks else []
                else:
                    continue
                json_objects = []
                for i, mobj in enumerate(matched):
                    det = mobj.get("detection")
                    mask = mobj.get("mask")
                    if mask is None:
                        continue
                    depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d(
                        metric_depth, K, mask, det, use_erosion=True)
                    _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                    if _do_erosion_cmp:
                        depth_stats_raw, coords_3d_raw, centroid_raw = self._mask_depth_stats_and_3d(
                            metric_depth, K, mask, det, use_erosion=False)
                    else:
                        depth_stats_raw, coords_3d_raw, centroid_raw = None, None, None
                    if det:
                        obj_id = f"obj_{i}_{det.get('source_model', 'det')}"
                        label = det.get("label", "unknown")
                        bbox = det.get("bbox", [])
                        source_model = det.get("source_model", "")
                    else:
                        obj_id = f"obj_{i}_mask"
                        label = "unlabeled"
                        bbox = list(mobj.get("mask_bbox_xyxy", [0, 0, 0, 0]))
                        source_model = ""
                    mask_path_rel = None
                    masked_depth_path_rel = None
                    obj_entry = {
                        "id": obj_id,
                        "label": label,
                        "bbox": bbox,
                        "source_model": source_model,
                        "segmentor": str(det.get("segmentor", source_model)) if det else source_model,
                        "sam2_mask_index": mobj.get("sam2_mask_index", -1),
                        "mask_path": mask_path_rel,
                        "masked_depth_path": masked_depth_path_rel,
                        # With adaptive erosion (default, more accurate for large objects)
                        "depth_stats": depth_stats,
                        "coordinates_3d_from_mask": coords_3d,
                        "mask_centroid_2d": centroid,
                        # Without erosion (raw boundary pixels included — for comparison)
                        "depth_stats_no_erosion": depth_stats_raw,
                        "coordinates_3d_no_erosion": coords_3d_raw,
                        "mask_centroid_2d_no_erosion": centroid_raw,
                    }
                    json_objects.append(obj_entry)
                mapping_path = sam2_sg_dir / f"{path.stem}_depth_mask_mapping_{mode}.png"
                self._save_depth_mask_mapping_image(metric_depth, matched, mapping_path)
                mapping_rel = f"scene_graph/{path.stem}_depth_mask_mapping_{mode}.png"
                dm_json = self._build_depth_mask_json(
                    image_path=str(path.resolve()),
                    path_stem=path.stem,
                    timestamp=timestamp,
                    image_size=[w, h],
                    matching_mode=mode,
                    depth_map_path=depth_map_npy_rel,
                    depth_map_image_path=depth_map_image_rel,
                    depth_global_min=depth_global_min,
                    depth_global_max=depth_global_max,
                    depth_global_mean=depth_global_mean,
                    segmentation_map_image_path=seg_map_rel,
                    num_auto_masks=len(amg_masks),
                    mapping_image_path=mapping_rel,
                    objects=json_objects,
                )

            # 4. Integration & Projection (mask-native)
            # Load labellers now — they were deferred to keep VRAM free during SAM2 inference.
            self._load_labellers()
            # Every object is built directly from a SAM2 mask — no bbox fallback exists.
            # coordinates_3d, depth_stats, and mask_centroid_2d are all pixel-native:
            #   z = median depth over mask foreground pixels (robust to noise)
            #   (cx, cy) = centre-of-mass of mask pixels
            # _sam2_mask_array is stored as a transient field for downstream Pix2SG;
            # it is stripped before json.dump in Stage 6.
            objects_3d = []

            # Per-object depth PNGs go directly into scene_graph/
            objects_dir = sam2_sg_dir

            for i, det in enumerate(all_detections):
                bbox = det['bbox']
                src = det.get('source_model', 'Unknown')
                graph_id = det.get("graph_id", f"obj_{i}_{src}")
                confidence = round(float(det.get("conf", 0.0)), 4)
                gdino_confidence = round(float(det.get("grounded_sam2_confidence", confidence)), 4)
                grounded_label = str(det.get("grounded_sam2_label", det.get("label", "object"))).strip().lower()
                bbox_int = [int(round(v)) for v in bbox[:4]]

                mobj = matched_A_lookup[graph_id]  # always present — identity mapping
                mask = mobj["mask"]
                # With adaptive erosion (default — boundary bleed removed)
                depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d(
                    metric_depth, K, mask, det, use_erosion=True
                )
                # Without erosion — raw boundary pixels included (comparison set)
                _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                if _do_erosion_cmp:
                    depth_stats_raw, coords_3d_raw, centroid_raw = self._mask_depth_stats_and_3d(
                        metric_depth, K, mask, det, use_erosion=False
                    )
                else:
                    depth_stats_raw, coords_3d_raw, centroid_raw = None, None, None

                sam2_mask_index = mobj["sam2_mask_index"]
                mask_path_rel: Optional[str] = None
                mask_matched = True

                obj_depth_filename = None

                obj_entry = {
                    "id": graph_id,
                    "label": str(det.get("label", "object")).strip().lower(),
                    "confidence": confidence,
                    "conf": confidence,                  # backward compatibility
                    "bbox": bbox_int,
                    "segmentor": str(det.get("segmentor", src)),  # GroundedSAM2 or SAM2_AMG
                    # --- With adaptive erosion (default, boundary-bleed removed) ---
                    "coordinates_3d": coords_3d,
                    "depth_stats": depth_stats,
                    "mask_centroid_2d": centroid,
                    # --- Without erosion (raw — for comparison) ---
                    "coordinates_3d_no_erosion": coords_3d_raw,
                    "depth_stats_no_erosion": depth_stats_raw,
                    "mask_centroid_2d_no_erosion": centroid_raw,
                    "sam2_mask_index": sam2_mask_index,  # index into amg_masks; -1 if unmatched
                    "mask_matched": mask_matched,        # True = mask-native coords
                    "mask_path": mask_path_rel,
                    "depth_map_path": None,
                    "sources": {
                        "GroundedSAM2": {
                            "caption": str(det.get("caption", grounded_label)),
                            "label": grounded_label,
                            "confidence": gdino_confidence,
                        },
                        "Florence2": {
                            "label": str(det.get("florence2_label", "")),
                            "caption": str(det.get("florence2_caption", "")),
                        },
                        "RAM++": {
                            "label": str(det.get("rampp_label", "")),
                            "caption": str(det.get("rampp_caption", "")),
                            "tags": list(det.get("rampp_tags", [])),
                        },
                        "Pix2SG": {"relations": []},
                    },
                    "_sam2_mask_array": mask,            # TRANSIENT — stripped before JSON save
                }
                objects_3d.append(obj_entry)

            # 5. Relation Matching
            # Pix2SG runs here (after Stage 4) so objects_3d entries carry
            # _sam2_mask_array and mask_centroid_2d, enabling mask-native spatial
            # predicates in _build_spatial_scaffold_triplets.
            pix2sg_out = self.pix2sg.predict(
                img_bgr,
                image_stem=path.stem,
                detections=objects_3d,
                iou_func=self._bbox_iou_xyxy,
            )
            if self.pix2sg.is_active():
                print(
                    "Pix2SG produced "
                    f"{len(pix2sg_out)} raw triplets (backend={self.pix2sg.status().get('backend', 'unknown')})."
                )
            else:
                print(f"Pix2SG inactive: {self.pix2sg.status().get('reason', 'unknown reason')}")

            pix2sg_stats = self._attach_relations_by_triplets(
                objects_3d,
                pix2sg_out,
                "Pix2SG",
            )
            print(
                "Relation attach stats: "
                f"Pix2SG(attached={pix2sg_stats['attached']}/{pix2sg_stats['input_triplets']}, "
                f"sub_id={pix2sg_stats['subject_id_matched']}, sub_label={pix2sg_stats['subject_label_matched']})"
            )

            # SGTR — semantic relations (OIv6 30 predicates).
            if (
                self.require_any_relation_source
                and len(all_detections) >= 2
                and pix2sg_stats["input_triplets"] == 0
            ):
                relation_status = self._collect_relation_source_status()
                details = "; ".join(
                    f"{name}: active={status.get('active')} backend={status.get('backend')} reason={status.get('reason', '')}"
                    for name, status in relation_status.items()
                )
                raise RuntimeError(
                    "No relation triplets were produced by Pix2SG despite multiple detections. "
                    f"Diagnostics: {details}"
                )

            # Unload labellers — relations are done, SAM3 loads next, VRAM needed.
            _f2_was_active = self.florence2 is not None and self.florence2.active
            _rampp_was_active = self.rampp is not None and self.rampp.active
            self._unload_labellers()

            # 6. Save SAM2 scene graph JSON (into scene_graph/sam2/)
            if K:
                K_serializable = {k: float(v) for k, v in K.items()}
            else:
                K_serializable = None

            models_used_sam2 = ["GroundedSAM2"]
            if _f2_was_active:
                models_used_sam2.append("Florence-2")
            if _rampp_was_active:
                models_used_sam2.append("RAM++")
            if self.pix2sg.is_active():
                models_used_sam2.append("Pix2SG")

            sam2_metadata = {
                "timestamp": timestamp,
                "segmentor": "SAM2",
                "intrinsics": K_serializable,
                "models": models_used_sam2,
                "rampp_tags": _rampp_tags_for_metadata,
                "gdino_query_used": _gdino_query_used,
                "relation_sources": self._collect_relation_source_status(),
                "relation_debug": {
                    "pix2sg": pix2sg_stats,
                    "mask_iou_match_thresh": float(self.mask_iou_match_thresh),
                    "pix2sg_mask_overlap_thresh": float(self.pix2sg_mask_overlap_thresh),
                    "pix2sg_depth_near_threshold": float(self.pix2sg_depth_near_threshold),
                    "raw_triplets": {"pix2sg": int(len(pix2sg_out))},
                    "num_detected_objects": int(len(all_detections)),
                    "num_mask_matched": int(sum(1 for o in objects_3d if o.get("mask_matched"))),
                },
                "depth_map": depth_map_image_rel,
                "segmentation_image": seg_map_rel,
            }
            sam2_metadata["sam2_segmentation_image"] = sam2_paths["sam2_segmentation_image_path"]
            sam2_metadata["sam2_tinted_overlay_image"] = sam2_paths["sam2_tinted_overlay_image_path"]

            # Save labelled visualizations while _sam2_mask_array is still present
            self._save_labelled_segmentation(
                objects_3d,
                sam2_sg_dir / f"{path.stem}_sam2_segmentation.png",
            )
            self._save_labelled_tinted_overlay(
                objects_3d,
                img_rgb,
                sam2_sg_dir / f"{path.stem}_sam2_tinted_overlay.png",
            )

            for obj in objects_3d:
                obj.pop("_sam2_mask_array", None)
            sam2_scene_output = {"metadata": sam2_metadata, "objects": objects_3d}
            with open(sam2_sg_dir / f"{path.stem}_scene.json", "w") as f:
                json.dump(sam2_scene_output, f, indent=2)
            print(f"SAM2 scene graph saved: scene_graph/{path.stem}_scene.json")

            # 7. SAM2 Visualization (into scene_graph/sam2/)
            viz_sam2 = img_bgr.copy()
            for obj in objects_3d:
                bbox = obj["bbox"]
                color = (0, 255, 0)
                label = f"{obj['label']} [M]"
                cv2.rectangle(viz_sam2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(viz_sam2, label, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                mc = obj.get("mask_centroid_2d")
                if mc and len(mc) == 2 and obj.get("mask_matched"):
                    cv2.circle(viz_sam2, (int(mc[0]), int(mc[1])), 3, color, -1)
                cx_a = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
                cy_a = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
                for source in ["Pix2SG", "SGTR"]:
                    if source in obj.get("sources", {}):
                        for rel in obj["sources"][source].get("relations", []):
                            target_id = rel["target_id"]
                            if isinstance(target_id, str) and target_id.startswith("external_"):
                                continue
                            target = next((o for o in objects_3d if o["id"] == target_id), None)
                            if target:
                                bbox_b = target["bbox"]
                                mc_b = target.get("mask_centroid_2d")
                                cx_b = int(mc_b[0]) if mc_b and len(mc_b) == 2 else (int(bbox_b[0]) + int(bbox_b[2])) // 2
                                cy_b = int(mc_b[1]) if mc_b and len(mc_b) == 2 else (int(bbox_b[1]) + int(bbox_b[3])) // 2
                                cv2.line(viz_sam2, (cx_a, cy_a), (cx_b, cy_b), (0, 255, 255), 1)
                                cv2.putText(viz_sam2, rel["predicate"],
                                            ((cx_a + cx_b) // 2, (cy_a + cy_b) // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_3d_viz.png"), viz_sam2)

        # 8. Sequential SAM3 pass — loads weights here (after SAM2 finishes),
        #    runs inference, then immediately unloads to free VRAM.
        objects_3d_sam3: List[Dict[str, Any]] = []
        if self._run_sam3:
            if not self._sam3_only:
                print("="*60)
                print("Stage SAM3: Freeing SAM2 inference VRAM before SAM3 load...")
                del amg_masks, all_detections
            # Always unload depth backend before SAM3 loads — depth inference is already done by this point.
            # Frees ~1.8 GB VRAM on T4; without this SAM3 (3.45 GB) + Florence-2 (2.3 GB) cause OOM.
            if getattr(self.depth_estimator, "backend", None) is not None:
                print("Stage SAM3: Unloading depth backend to free VRAM before SAM3 load...")
                self.depth_estimator.unload_backend()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Lazy-load SAM3 weights now (after SAM2 inference tensors freed)
            if self.sam3_wrapper is None or not self.sam3_wrapper.active:
                print("Stage SAM3: Loading SAM3 weights...")
                self.sam3_wrapper = SAM3Wrapper(
                    device=self.device,
                    text_query=self._sam3_text_query,
                    confidence_threshold=self._sam3_conf,
                    checkpoint_path=self._sam3_ckpt,
                    load_from_hf=self._sam3_hf,
                )

            sam3_masks = []
            if self.sam3_wrapper.active:
                print("Stage SAM3: Running SAM3 segmentation pass (sequential)...")
                sam3_masks = self.sam3_wrapper.generate(img_rgb)
                # Unload immediately after inference to reclaim ~3.5GB VRAM
                self.sam3_wrapper.unload()
                self.sam3_wrapper = None

            sam3_paths = self._save_sam3_outputs(
                sam3_masks=sam3_masks,
                h=h,
                w=w,
                out_dir=out,
                path_stem=path.stem,
                image_path=str(path.resolve()),
                timestamp=timestamp,
                image_rgb=img_rgb,
            ) if sam3_masks else {}

            if sam3_masks:
                # SAM3 sub-directory mirrors SAM2 structure
                sam3_sg_dir = scene_graph_dir / "sam3"
                sam3_sg_dir.mkdir(parents=True, exist_ok=True)
                sam3_objects_dir = sam3_sg_dir / "objects"
                sam3_objects_dir.mkdir(parents=True, exist_ok=True)
                sam3_masks_out_dir = sam3_sg_dir / "masks"
                sam3_masks_out_dir.mkdir(parents=True, exist_ok=True)

                print(f"Stage SAM3: building 3D object entries for {len(sam3_masks)} SAM3 masks...")
                _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                self._load_labellers()

                for i, sam3_m in enumerate(sam3_masks):
                    seg = sam3_m.get("segmentation")
                    mask = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)

                    det_s3 = self._label_mask(img_bgr, mask, sam3_m)
                    det_s3["graph_id"] = f"sam3_obj_{i}"
                    det_s3["sam2_mask_index"] = int(i)
                    det_s3["grounded_sam2_label"] = str(sam3_m.get("label", det_s3.get("label", "object"))).strip().lower()
                    det_s3["grounded_sam2_confidence"] = float(sam3_m.get("gdino_conf", sam3_m.get("predicted_iou", 0.0)))
                    det_s3["bbox"] = self._xywh_to_xyxy(sam3_m.get("bbox", [0, 0, w, h]))
                    det_s3["segmentor"] = "SAM3"

                    graph_id_s3 = det_s3["graph_id"]
                    confidence_s3 = round(float(det_s3.get("conf", 0.0)), 4)
                    gdino_conf_s3 = round(float(det_s3.get("grounded_sam2_confidence", confidence_s3)), 4)
                    grounded_label_s3 = det_s3.get("grounded_sam2_label", det_s3.get("label", "object"))
                    bbox_s3 = [int(round(v)) for v in det_s3["bbox"][:4]]

                    depth_stats_s3, coords_3d_s3, centroid_s3 = self._mask_depth_stats_and_3d(
                        metric_depth, K, mask, det_s3, use_erosion=True)
                    if _do_erosion_cmp:
                        depth_stats_raw_s3, coords_3d_raw_s3, centroid_raw_s3 = self._mask_depth_stats_and_3d(
                            metric_depth, K, mask, det_s3, use_erosion=False)
                    else:
                        depth_stats_raw_s3, coords_3d_raw_s3, centroid_raw_s3 = None, None, None

                    # Per-object mask PNG
                    mask_path_rel_s3 = None

                    # Per-object depth visualization
                    x1s = max(0, bbox_s3[0]); y1s = max(0, bbox_s3[1])
                    x2s = min(w, bbox_s3[2]); y2s = min(h, bbox_s3[3])
                    obj_depth_fn_s3 = f"{path.stem}_sam3_obj_{i}_depth.png"
                    object_depth_s3 = metric_depth[y1s:y2s, x1s:x2s]
                    if object_depth_s3.size > 0:
                        dm, dx = object_depth_s3.min(), object_depth_s3.max()
                        nd = ((object_depth_s3 - dm) / (dx - dm) * 255).astype(np.uint8) if dx - dm > 1e-6 else np.zeros_like(object_depth_s3, dtype=np.uint8)
                        cv2.imwrite(str(sam3_objects_dir / obj_depth_fn_s3),
                                    cv2.applyColorMap(nd, cv2.COLORMAP_INFERNO))

                    obj_entry_s3 = {
                        "id": graph_id_s3,
                        "label": str(det_s3.get("label", "object")).strip().lower(),
                        "confidence": confidence_s3,
                        "conf": confidence_s3,
                        "bbox": bbox_s3,
                        "segmentor": "SAM3",
                        "coordinates_3d": coords_3d_s3,
                        "depth_stats": depth_stats_s3,
                        "mask_centroid_2d": centroid_s3,
                        "coordinates_3d_no_erosion": coords_3d_raw_s3,
                        "depth_stats_no_erosion": depth_stats_raw_s3,
                        "mask_centroid_2d_no_erosion": centroid_raw_s3,
                        "sam3_mask_index": i,
                        "mask_matched": True,
                        "mask_path": mask_path_rel_s3,
                        "depth_map_path": f"scene_graph/sam3/objects/{obj_depth_fn_s3}",
                        "sources": {
                            "SAM3": {
                                "caption": str(det_s3.get("caption", grounded_label_s3)),
                                "label": grounded_label_s3,
                                "confidence": gdino_conf_s3,
                            },
                            "Pix2SG": {"relations": []},
                        },
                        "_sam2_mask_array": mask,   # TRANSIENT
                    }
                    objects_3d_sam3.append(obj_entry_s3)

                # Depth map copy into sam3 dir (same depth, different segmentor)
                self._save_depth_map_image(metric_depth, sam3_sg_dir / f"{path.stem}_depth_map.png")

                # Pix2SG relations on SAM3 objects
                if objects_3d_sam3:
                    pix2sg_out_s3 = self.pix2sg.predict(
                        img_bgr,
                        image_stem=path.stem,
                        detections=objects_3d_sam3,
                        iou_func=self._bbox_iou_xyxy,
                    )
                    pix2sg_stats_sam3 = self._attach_relations_by_triplets(
                        objects_3d_sam3, pix2sg_out_s3, "Pix2SG"
                    )
                    print(
                        f"SAM3 Pix2SG relations: attached={pix2sg_stats_sam3['attached']}/"
                        f"{pix2sg_stats_sam3['input_triplets']}"
                    )

                self._unload_labellers()

                # SAM3 labelled visualizations (before mask strip)
                self._save_labelled_segmentation(
                    objects_3d_sam3,
                    sam3_sg_dir / f"{path.stem}_sam3_segmentation.png",
                )
                self._save_labelled_tinted_overlay(
                    objects_3d_sam3,
                    img_rgb,
                    sam3_sg_dir / f"{path.stem}_sam3_tinted_overlay.png",
                )

                # SAM3 3d_viz (mirroring SAM2 viz style, orange boxes to distinguish)
                viz_sam3 = img_bgr.copy()
                for obj in objects_3d_sam3:
                    obj.pop("_sam2_mask_array", None)
                    bbox = obj["bbox"]
                    color = (255, 128, 0)   # orange-ish in BGR — distinct from SAM2 green
                    label = f"{obj['label']} [S3]"
                    cv2.rectangle(viz_sam3, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(viz_sam3, label, (int(bbox[0]), int(bbox[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    mc = obj.get("mask_centroid_2d")
                    if mc and len(mc) == 2 and obj.get("mask_matched"):
                        cv2.circle(viz_sam3, (int(mc[0]), int(mc[1])), 3, color, -1)
                    cx_a = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
                    cy_a = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
                    for source in ["Pix2SG"]:
                        if source in obj.get("sources", {}):
                            for rel in obj["sources"][source].get("relations", []):
                                target_id = rel["target_id"]
                                if isinstance(target_id, str) and target_id.startswith("external_"):
                                    continue
                                target = next((o for o in objects_3d_sam3 if o["id"] == target_id), None)
                                if target:
                                    bbox_b = target["bbox"]
                                    mc_b = target.get("mask_centroid_2d")
                                    cx_b = int(mc_b[0]) if mc_b and len(mc_b) == 2 else (int(bbox_b[0]) + int(bbox_b[2])) // 2
                                    cy_b = int(mc_b[1]) if mc_b and len(mc_b) == 2 else (int(bbox_b[1]) + int(bbox_b[3])) // 2
                                    cv2.line(viz_sam3, (cx_a, cy_a), (cx_b, cy_b), (0, 200, 255), 1)
                                    cv2.putText(viz_sam3, rel["predicate"],
                                                ((cx_a + cx_b) // 2, (cy_a + cy_b) // 2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                cv2.imwrite(str(sam3_sg_dir / f"{path.stem}_3d_viz.png"), viz_sam3)

                # SAM3 scene JSON
                sam3_scene_output = {
                    "metadata": {
                        "timestamp": timestamp,
                        "segmentor": "SAM3",
                        "intrinsics": K_serializable,
                        "models": ["SAM3", "Florence-2", "Pix2SG"],
                        "depth_map": f"scene_graph/sam3/{path.stem}_depth_map.png",
                        "segmentation_image": f"scene_graph/sam3/{path.stem}_sam3_segmentation.png",
                        "sam3_raw_json": sam3_paths.get("sam3_json_path", ""),
                        "num_objects": len(objects_3d_sam3),
                    },
                    "objects": objects_3d_sam3,
                }
                with open(sam3_sg_dir / f"{path.stem}_scene.json", "w") as f:
                    json.dump(sam3_scene_output, f, indent=2)
                print(f"SAM3 scene graph saved: scene_graph/sam3/{path.stem}_scene.json")

        print(f"Results saved to {out}")

        # Release large arrays so memory is available before the next image
        if not self._sam3_only:
            try:
                del amg_masks, all_detections
            except NameError:
                pass
        del img_bgr, img_rgb, metric_depth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    import argparse
    from config import PreprocessConfig

    parser = argparse.ArgumentParser(description="Run scene understanding pipeline (SAM2 and/or SAM3).")
    parser.add_argument("--input_dir", type=str, default="images", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output_scene", help="Output directory for scene graphs")
    parser.add_argument("--sam3-only", action="store_true", help="Run only depth + SAM3 (no SAM2). Use for Run 2 when comparing SAM2 vs SAM3.")
    parser.add_argument("--use-existing-depth", action="store_true", help="When --sam3-only, load depth from output_dir/depth/ when present (from Run 1).")
    parser.add_argument("--run-both", action="store_true", help="Run SAM2 and SAM3 in one process (requires ~16GB+ VRAM). Default is SAM2-only to avoid OOM.")
    args = parser.parse_args()

    print("Testing pipeline initialization...")
    cfg = PreprocessConfig()
    cfg.sam3_only = args.sam3_only
    cfg.sam3_only_use_existing_depth = args.use_existing_depth
    # Default: SAM2-only so VM does not OOM when loading SAM3 after SAM2. Use --sam3-only for Run 2, or --run-both if you have enough VRAM.
    if not cfg.sam3_only:
        cfg.run_sam3 = args.run_both
    if cfg.sam3_only:
        print("Mode: SAM3-only (depth + SAM3). Run 1 first without --sam3-only to get SAM2 results.")
    elif not cfg.run_sam3:
        print("Mode: SAM2-only (depth + SAM2). Run again with --sam3-only [--use-existing-depth] to get SAM3 results.")
    else:
        print("Mode: SAM2 + SAM3 in one process (--run-both). Requires sufficient VRAM.")
    images_dir = Path(args.input_dir)
    output_dir = args.output_dir
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

    try:
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir.resolve()}")

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No supported images found in {images_dir.resolve()} "
                f"for extensions: {sorted(supported_exts)}"
            )

        # Pass first image so CLIP can classify indoor/outdoor before loading
        # the depth model. Without this, 'auto' always defaults to indoor.
        depth_estimator = DepthEstimator(cfg, first_image=image_paths[0])
        pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)

        print(f"Found {len(image_paths)} images in {images_dir.resolve()}")
        for img_path in image_paths:
            print(f"Processing image: {img_path.name}")
            try:
                pipeline.process_image(str(img_path), str(output_dir))
            except ValueError as e:
                print(f"Skipping {img_path.name}: {e}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
