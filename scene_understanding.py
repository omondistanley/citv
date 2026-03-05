"""
scene_understanding.py
Unified 3D Scene Graph Generation using Depth Estimator and Specific Scene Graph Models.
Integrates Depth Anything V2 with SAM2, GRiT, and Pix2SG.
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
# GRiT Imports & Setup
# -----------------------------------------------------------------------------
import sys
import os

# Add GRiT paths so imports work correctly
# We need to add the project root and the specific CenterNet2 path
grit_root = Path("GRiT")
centernet_path = grit_root / "third_party/CenterNet2/projects/CenterNet2/"

# Add to sys.path if not already present
if str(centernet_path.resolve()) not in sys.path:
    sys.path.insert(0, str(centernet_path.resolve()))
if str(grit_root.resolve()) not in sys.path:
    sys.path.insert(0, str(grit_root.resolve()))

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data.detection_utils import read_image
    from centernet.config import add_centernet_config
    from grit.config import add_grit_config
except ImportError:
    print("Warning: Could not import GRiT modules. Ensure detectron2 and GRiT requirements are installed.")
    # Define dummy functions to prevent NameError if import fails
    def get_cfg(): return None
    def add_centernet_config(cfg): pass
    def add_grit_config(cfg): pass
    DefaultPredictor = None

# -----------------------------------------------------------------------------
# 1. GRiT Wrapper (Generative Region-to-Text Transformer)
# -----------------------------------------------------------------------------
class GRiTWrapper:
    """
    Wrapper for GRiT.
    Provides: Dense captions, Bounding boxes, Object labels.
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing GRiT...")
        
        # Check if GRiT imports succeeded
        if get_cfg() is None:
            print("GRiT initialization skipped: Dependencies not found.")
            self.predictor = None
            return

        # Paths to config and weights
        self.config_file = "GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml"
        self.weights_file = "GRiT/models/grit_b_densecap_objectdet.pth"

        if not os.path.exists(self.config_file) or not os.path.exists(self.weights_file):
            print(f"Error: GRiT config or weights not found at {self.config_file} / {self.weights_file}")
            self.predictor = None
            return

        # Initialize Config
        self.cfg = self._setup_cfg()
        if self.cfg:
            self.predictor = DefaultPredictor(self.cfg)
        else:
            self.predictor = None

    def _setup_cfg(self):
        cfg = get_cfg()
        if cfg is None:
            return None
            
        if self.device.type == 'cpu':
            cfg.MODEL.DEVICE = "cpu"
        
        add_centernet_config(cfg)
        add_grit_config(cfg)
        cfg.merge_from_file(self.config_file)
        
        # Set weights and confidence
        cfg.MODEL.WEIGHTS = self.weights_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        
        # Specific GRiT settings for DenseCap
        cfg.MODEL.TEST_TASK = 'DenseCap'
        cfg.MODEL.BEAM_SIZE = 1
        cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
        cfg.USE_ACT_CHECKPOINT = False
        
        cfg.freeze()
        return cfg

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs GRiT inference on a BGR image (OpenCV format).
        Returns list of objects with label, confidence, bbox, and caption.
        """
        if self.predictor is None:
            return []
        
        # Predictor handles BGR -> RGB conversion internally if configured, 
        # but Detectron2 DefaultPredictor expects BGR (OpenCV format) by default.
        predictions = self.predictor(image)
        
        instances = predictions["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        # GRiT stores captions in 'pred_object_descriptions'
        # Note: Depending on the specific GRiT version, this might be a list or object
        if hasattr(instances, 'pred_object_descriptions'):
            captions = instances.pred_object_descriptions.data
        else:
            captions = ["unknown"] * len(boxes)

        results = []
        for i in range(len(boxes)):
            box = boxes[i].astype(int).tolist() # [x1, y1, x2, y2]
            score = float(scores[i])
            caption = captions[i]
            
            # Simple heuristic to get a short "label" from the longer caption
            label = caption.split()[-1] if caption else "object"
            
            # Ensure box coordinates are integers
            box = [int(x) for x in box]
            
            results.append({
                "label": label, 
                "conf": score, 
                "bbox": box, 
                "caption": caption,
                "source_model": "GRiT"
            })
            
        return results

# -----------------------------------------------------------------------------
# 1b. YOLO Classification Wrapper (class-aware labelling per masked crop)
# -----------------------------------------------------------------------------
class YOLOClassifierWrapper:
    """
    Runs YOLOv8 classification on a masked image crop to produce a semantic class label.
    Gives SAM2 masks a proper class from the ImageNet-1k taxonomy (1000 classes).
    Falls back gracefully if ultralytics is not installed.
    """
    def __init__(self, model_name: str = "yolov8x-cls.pt", conf_thresh: float = 0.3):
        self.conf_thresh = conf_thresh
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"YOLOClassifier ({model_name}) ready.")
        except Exception as e:
            print(f"YOLOClassifier init failed: {e}. Will fall back to GRiT labels.")

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Run classification on a BGR crop. Returns (class_name, confidence).
        Returns ("object", 0.0) if model unavailable or confidence below threshold.
        """
        if self.model is None or crop_bgr.size == 0:
            return "object", 0.0
        try:
            results = self.model(crop_bgr, verbose=False)
            top1_conf = float(results[0].probs.top1conf)
            if top1_conf >= self.conf_thresh:
                top1_cls = results[0].names[results[0].probs.top1]
                return top1_cls, top1_conf
        except Exception:
            pass
        return "object", 0.0



# -----------------------------------------------------------------------------
# 6. Pix2SG Wrapper (Pixel-to-Scene Graph)
# -----------------------------------------------------------------------------
class Pix2SGWrapper:
    """
    Wrapper for Pix2SG spatial scaffold + Florence-2 semantic enrichment.

    Fix 5.6 — Three-layer relation prediction:
      Layer 1 (Spatial scaffold, always active):
        Pixel mask IoU → overlapping
        Depth difference → in_front_of / behind
        Mask centroid direction → left_of / right_of / above / below

      Layer 2 (Florence-2 semantic, when florence2 instance provided):
        For object pairs with overlapping masks (IoU > relation_min_mask_overlap),
        query Florence-2 with a color-coded crop:
          "Describe the relationship between the red [sub] and the blue [obj]."
        Maps free-form response to canonical OIv6-style predicates.
        This adds action/functional predicates (holds, wears, rides, eats …)
        that pure spatial reasoning cannot infer.

      Layer 3 (Precomputed triplets, when pix2sg_triplets_dir exists):
        Loads a pre-generated {stem}.json and skips both layers above.
        Useful for offline/batch processing or hand-curated triplets.

    Why this is better than SGTR alone (Fix 5.6 rationale):
      SGTR requires cvpods C++ extensions, runs on full images (no masks),
      and matches bboxes back to SAM2 objects via IoU — introducing matching
      error. Florence-2 works directly on masked regions, requires no C++
      extension, and shares the model already loaded for Fix 5.4 (zero extra
      VRAM cost for the relation step).
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
        """Mask-native spatial predicate using pixel overlap and mask centroids.

        Overlap detection: uses pixel mask intersection/union rather than bbox IoU.
        Bbox IoU fires `overlapping` whenever two padded rectangles share area —
        even for objects far apart in 3D whose pixels don't touch. Pixel
        intersection/union measures actual foreground contact.

        Directional predicates: uses mask_centroid_2d (centre-of-mass of the
        object's foreground pixels) rather than the bbox midpoint, giving more
        accurate left/right/above/below classification for asymmetric objects.

        Falls back to bbox IoU for overlap detection and bbox center for
        direction when one or both objects lack a matched mask.
        """
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

        # --- Depth-gated direction: if objects are far apart in depth, the dominant
        # relation is depth-axis (in_front_of / behind), not 2D image direction.
        sub_z = sub.get("coordinates_3d", {}).get("z")
        obj_z = obj.get("coordinates_3d", {}).get("z")
        if sub_z is not None and obj_z is not None:
            depth_diff = abs(float(obj_z) - float(sub_z))
            if depth_diff >= self._depth_far_threshold:
                return "in_front_of" if float(sub_z) < float(obj_z) else "behind"

        # --- Directional from mask centroid (same 2D plane) ---
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

# -----------------------------------------------------------------------------
# Fix 5.4: Florence-2 Wrapper
# Replaces GRiT's unreliable "last-word" heuristic with structured OD output.
#
# Why Florence-2 instead of GRiT:
#   - GRiT generates free-form captions; extracting a label requires heuristics
#     (last word, first noun) that fail ~30% of the time.
#   - Florence-2 with task "<OD>" returns structured {label, bbox} dicts directly.
#     We pick the highest-confidence label from the crop — no parsing needed.
#   - Florence-2 also supports "<DETAILED_CAPTION>" for rich descriptions and
#     "<REGION_RELATION>" for pairwise relation queries (used in Fix 5.6).
#   - Single model for both labelling and relation prediction: reduces total
#     VRAM footprint vs. having GRiT + a separate VQA model.
#
# Memory strategy:
#   Florence-2-large: ~900 MB VRAM.  Florence-2-base: ~500 MB VRAM.
#   The model is loaded once and kept resident — re-loading per image is slow.
#   It is shared between the labelling step (Fix 5.4) and the relation step
#   (Fix 5.6) via the same wrapper instance.
# -----------------------------------------------------------------------------
class Florence2Wrapper:
    """
    Microsoft Florence-2 wrapper for per-mask object labelling and pairwise
    semantic relation prediction.

    Tasks used:
      "<OD>"              → structured {label, bbox} per crop → best label
      "<DETAILED_CAPTION>"→ rich sentence description of crop (stored as caption)
      "<REGION_RELATION>" → "What is the relationship between <region1> and <region2>?"
                            Used in Fix 5.6 for mask-pair semantic relations.
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
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True, use_fast=False
                )
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            self.active = True
            print(f"Florence-2 ready ({model_id}).")
        except Exception as e:
            print(f"Florence-2 init failed: {e}. Falling back to GRiT label extraction.")

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

    def label_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Label a BGR crop image using Florence-2 <OD> task.

        Returns dict with keys: label (str), conf (float), caption (str).

        How it works:
          1. Convert crop to PIL RGB.
          2. Run <OD> task → {"<OD>": {"labels": [...], "bboxes": [...]}}
             Florence-2 OD on a single-object crop typically returns 1-3 items.
             We pick the label with the largest bbox area (= dominant object).
          3. Run <DETAILED_CAPTION> for the rich caption stored in scene JSON.
          4. If OD returns nothing, fall back to <CAPTION> for label extraction.
        """
        if not self.active or crop_bgr is None or crop_bgr.size == 0:
            return {"label": "object", "conf": 0.0, "caption": "object"}

        from PIL import Image as PILImage
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = PILImage.fromarray(crop_rgb)
        cw, ch = pil_crop.width, pil_crop.height

        # --- Object Detection task: gives structured labels ---
        od_result = self._run_task("<OD>", pil_crop)
        od_data = od_result.get("<OD>", {})
        labels = od_data.get("labels", [])
        bboxes = od_data.get("bboxes", [])

        label = "object"
        conf = 0.5  # Florence-2 OD doesn't provide per-box confidence in base output

        if labels:
            # Pick label from the largest bounding box — the dominant object in the crop
            best_area = -1
            for lbl, box in zip(labels, bboxes):
                if len(box) >= 4:
                    area = abs(box[2] - box[0]) * abs(box[3] - box[1])
                    if area > best_area:
                        best_area = area
                        label = str(lbl).strip().lower()
            conf = 0.80  # Florence-2 OD is reliable; assign high fixed confidence

        # --- Detailed caption for scene JSON ---
        cap_result = self._run_task("<DETAILED_CAPTION>", pil_crop)
        caption = cap_result.get("<DETAILED_CAPTION>", label)
        if not isinstance(caption, str):
            caption = str(caption)

        # Fallback: if OD gave nothing, extract first noun from caption
        if label == "object" and caption and caption != "object":
            words = caption.lower().split()
            stopwords = {"a", "an", "the", "with", "on", "of", "in", "at", "by", "and", "or", "is", "are"}
            meaningful = [w for w in words if w.isalpha() and w not in stopwords]
            if meaningful:
                label = meaningful[0]

        return {"label": label, "conf": conf, "caption": caption}

    def predict_relation(
        self,
        full_img_bgr: np.ndarray,
        mask_sub: np.ndarray,
        mask_obj: np.ndarray,
        label_sub: str,
        label_obj: str,
    ) -> Optional[str]:
        """
        Predict semantic relation between two masked regions using Florence-2.

        Strategy (Fix 5.6):
          1. Compute union bbox of both masks.
          2. Crop the full image to that union region.
          3. Overlay masks with distinct colors (red=subject, blue=object).
          4. Ask Florence-2: "<CAPTION> Describe the relationship between the
             red [label_sub] and the blue [label_obj]."
          5. Parse the response to extract a predicate.

        Why union bbox + color overlay:
          - Florence-2 doesn't have a native "region-pair relation" task, but
            it understands color references in captions very well.
          - Coloring the masks makes the two regions unambiguous.
          - Cropping to union reduces context noise from unrelated objects.

        Returns: predicate string or None if prediction fails/is uninformative.
        """
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

            prompt_text = (
                f" Describe the relationship between the red {label_sub} "
                f"and the blue {label_obj} in one short phrase."
            )
            result = self._run_task("<CAPTION>", pil_crop, extra_text=prompt_text)
            raw = result.get("<CAPTION>", "")
            if not isinstance(raw, str) or not raw.strip():
                return None

            # Map common relation phrases to canonical predicates
            return self._parse_relation_phrase(raw.lower().strip())
        except Exception as e:
            print(f"  [Florence2] predict_relation failed: {e}")
            return None

    @staticmethod
    def _parse_relation_phrase(text: str) -> Optional[str]:
        """
        Map Florence-2 free-form relation description to a canonical predicate.

        The mapping covers spatial, functional, and action predicates.
        Returns None if no clear predicate is found (avoids noise triplets).
        """
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


# -----------------------------------------------------------------------------
# Fix 5.3: Grounded-SAM2 Wrapper
# Replaces SAM2 AMG (class-agnostic, part-level) with object-level segmentation.
#
# Architecture:
#   GroundingDINO v2  →  open-vocabulary object detection (bboxes + labels)
#         ↓
#   SAM2 (prompted)   →  one high-quality mask per detected bbox
#         ↓
#   Result: object-level binary masks with semantic labels already attached
#
# Why this fixes the core accuracy problem (5.3):
#   SAM2 AMG places a dense point grid and segments EVERY coherent region,
#   producing part-level masks (car door, car wheel, car window = 3 masks for
#   one car).  Grounding DINO detects at the ENTITY level: "car" = one bbox.
#   SAM2 then generates one mask for that one entity.  This gives you the
#   correct object-level instance segmentation your pipeline requires.
#
# Text query strategy:
#   A broad category query ("person. vehicle. furniture. ...") covers all
#   common scene graph objects without overfitting to a fixed class list.
#   GDINO's open-vocabulary design handles unseen objects gracefully.
#
# Fallback:
#   If GDINO fails to import or produces zero detections, the wrapper falls
#   back to SAM2 AMG (original behavior) so the pipeline never fails silently.
# -----------------------------------------------------------------------------
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

        # Fix 5.4 — Florence-2 (shared for labelling + relations)
        florence2_model_id = getattr(config, "florence2_model", "microsoft/Florence-2-large") if config is not None else "microsoft/Florence-2-large"
        self.florence2 = Florence2Wrapper(model_id=florence2_model_id, device=self.device)

        # GRiT kept as secondary fallback if Florence-2 fails
        self.grit = GRiTWrapper(self.device)
        yolo_cls_model = getattr(config, "yolo_cls_model", "yolov8x-cls.pt") if config is not None else "yolov8x-cls.pt"
        yolo_cls_thresh = float(getattr(config, "yolo_cls_conf_thresh", 0.30)) if config is not None else 0.30
        self.yolo_cls = YOLOClassifierWrapper(yolo_cls_model, yolo_cls_thresh)

        # Fix 5.6 — Pix2SG with Florence-2 enrichment
        relation_min_mask_overlap = float(getattr(config, "relation_min_mask_overlap", 0.02)) if config is not None else 0.02
        self.pix2sg = Pix2SGWrapper(
            self.device,
            triplets_dir=pix2sg_triplets_dir,
            max_relations_per_object=pix2sg_max_relations_per_object,
            mask_overlap_thresh=self.pix2sg_mask_overlap_thresh,
            depth_near_threshold=self.pix2sg_depth_near_threshold,
            depth_far_threshold=pix2sg_depth_far_threshold,
            florence2=self.florence2 if (config is not None and getattr(config, "florence2_relation_enabled", True)) else None,
            relation_min_mask_overlap=relation_min_mask_overlap,
        )
        self._relation_source_status = self._collect_relation_source_status()
        self._print_relation_source_status()
        self._assert_relation_sources_or_fail()

        # Fix 5.3 — GroundedSAM2 (replaces SAM2AMGWrapper)
        if self.config is not None:
            ckpt = getattr(self.config, "sam2_checkpoint_path", "sam2/checkpoints/sam2.1_hiera_large.pt")
            cfg = getattr(self.config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l")
            gdino_model = getattr(self.config, "grounding_dino_model", "IDEA-Research/grounding-dino-base")
            gdino_box_thresh = float(getattr(self.config, "grounding_dino_box_thresh", 0.30))
            gdino_text_thresh = float(getattr(self.config, "grounding_dino_text_thresh", 0.25))
            gdino_query = getattr(self.config, "grounding_dino_text_query",
                "person. animal. vehicle. furniture. appliance. food. clothing. container. tool. building. plant. electronics. object.")
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
        """
        Fix 5.7: Compute erosion kernel size adapted to the object's narrowest dimension.

        Problem with fixed kernel:
          A 5px erosion on a 3px-wide pole destroys the entire mask.
          A 5px erosion on a large sofa (300px wide) is insufficient.

        Strategy:
          - Compute the bounding box of the mask.
          - Narrowest dimension = min(bbox_w, bbox_h).
          - Scale kernel: 0 for very thin objects, up to max for large objects.
          - Cap at config's mask_erosion_kernel_size.

        Returns kernel size (int); 0 means skip erosion.
        """
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
        Compute depth stats and 3D from mask pixels only.

        Fix 5.7 improvements over original:
          1. Adaptive erosion (kernel sized to object narrowest dim).
             Controlled by use_erosion — pass False to skip erosion entirely,
             producing raw depth stats for comparison against the eroded version.
          2. Depth outlier rejection (sigma-clipping removes background bleed).
          3. Transparency check (compares mask depth to surrounding border).

        Args:
            use_erosion: If True (default), apply adaptive erosion before depth
                         extraction. If False, skip erosion — produces
                         depth_stats_no_erosion / coordinates_3d_no_erosion.

        Returns (depth_stats_dict, coordinates_3d_from_mask, mask_centroid_2d).
        """
        h, w = metric_depth.shape[:2]
        mask_bin = (np.asarray(mask) > 0)
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_bin > 0)

        # --- Fix 5.7a: Adaptive erosion ---
        # Removes boundary pixels where the depth network blends fg+bg depths.
        # Kernel is scaled to the object's narrowest dimension so thin objects
        # (poles, wires) are not destroyed by an oversized kernel.
        # Skipped entirely when use_erosion=False (comparison / no-erosion run).
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

        # --- Fix 5.7b: Depth outlier rejection (sigma clipping) ---
        # Background pixels that bleed through transparent surfaces or
        # thin mask boundaries have depth values far from the object's true depth.
        # Reject pixels beyond N sigma from the mask mean.
        #
        # Example: a glass vase mask contains 200 pixels at ~0.8m (vase surface)
        # and 50 pixels at ~2.5m (background seen through glass). The mean is
        # ~1.0m, std ~0.4m. With sigma=2.0, the 2.5m pixels are rejected (>2σ).
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

        # --- Fix 5.7c: Transparency detection ---
        # Transparent objects (glass, water, plastic) produce masks where the
        # depth inside the mask matches the background depth because the depth
        # model sees through them. Detect this by comparing mask depth to a
        # 3px dilated border ring around the mask.
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

        # --- Depth-weighted centroid ---
        # Closer pixels get higher weight (1/depth) → centroid pulled toward
        # the nearest visible surface face of the object.
        weights = 1.0 / (depth_at_mask + 1e-6)
        w_sum = float(weights.sum())
        cy_f = float(np.sum(ys_f * weights) / w_sum)
        cx_f = float(np.sum(xs_f * weights) / w_sum)

        # Nearest real mask pixel to the weighted centroid (no holes in anchor)
        dist2 = (ys_f - cy_f) ** 2 + (xs_f - cx_f) ** 2
        anchor_idx = int(np.argmin(dist2))
        cx = int(xs_f[anchor_idx])
        cy = int(ys_f[anchor_idx])

        # --- z_val: histogram mode over inner-circle pixels ---
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
        Fix 5.4: Label a SAM2/GroundedSAM2 mask region.

        Priority order:
          1. If GroundedSAM2 already attached a label (gdino_conf key present),
             use it directly — GDINO is the most semantically accurate source.
          2. Florence-2 <OD> on the masked crop — structured label extraction.
          3. GRiT + YOLO fallback (original behavior) if Florence-2 unavailable.

        Why Florence-2 beats GRiT here (Fix 5.4):
          GRiT generates free-form text ("a wooden chair with armrests") and
          the original code took the LAST word ("armrests") — correct only by
          accident.  Florence-2 <OD> returns structured {"labels": ["chair"]}
          directly; no parsing heuristic needed.

        Mean-fill background is preserved for all crop-based models: replacing
        out-of-mask pixels with the image mean colour preserves natural image
        statistics (brightness, color distribution) so both Florence-2 and GRiT
        behave as trained — black zeros create an unnatural distribution.
        """
        # Priority 1: GDINO label already attached by GroundedSAM2Wrapper
        gdino_label = amg_entry.get("label")
        gdino_conf = amg_entry.get("gdino_conf", 0.0)
        if gdino_label and gdino_label != "object" and gdino_conf >= 0.30:
            return {
                "label": gdino_label,
                "conf": gdino_conf,
                "caption": gdino_label,
                "source_model": "GroundingDINO",
            }

        # Build masked crop for Florence-2 / GRiT
        h_img, w_img = img_bgr.shape[:2]
        x, y, bw, bh = amg_entry.get("bbox", [0, 0, w_img, h_img])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img, int(x + bw)), min(h_img, int(y + bh))
        if x2 <= x1 or y2 <= y1:
            return {"label": "object", "conf": 0.0, "caption": "object", "source_model": "fallback"}

        crop = img_bgr[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        mask_resized = cv2.resize(
            mask_bin.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        bg_mean = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
        crop_filled = crop.copy()
        crop_filled[~mask_resized] = bg_mean

        # Priority 2: Florence-2 (Fix 5.4)
        if self.florence2 is not None and self.florence2.active:
            f2_result = self.florence2.label_crop(crop_filled)
            if f2_result.get("label", "object") != "object":
                f2_result["source_model"] = "Florence-2"
                return f2_result
            # Florence-2 returned "object" — fall through to GRiT
            caption = f2_result.get("caption", "object")
        else:
            caption = "object"

        # Priority 3: GRiT + YOLO fallback
        yolo_label, yolo_conf = self.yolo_cls.classify(crop_filled)
        results = self.grit.predict(crop_filled)
        if results:
            best = max(results, key=lambda r: r.get("conf", 0.0))
            grit_label = best.get("label", "object")
            grit_conf = best.get("conf", 0.0)
            caption = best.get("caption", caption)
        else:
            grit_label, grit_conf = "object", 0.0

        if yolo_conf >= self.yolo_cls.conf_thresh and yolo_label != "object":
            label, conf = yolo_label, yolo_conf
        else:
            label, conf = grit_label, grit_conf

        return {"label": label, "conf": conf, "caption": caption, "source_model": "GRiT+YOLO"}

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
                    or _target_src.get("GRiT", {}).get("caption")
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

    def _save_sam2_segmentation_image(self, amg_masks: List[Dict], h: int, w: int, path: Path) -> None:
        """Save RGB image with one color per AMG mask (all masks)."""
        out = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(42)
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))
            out[mask > 0] = color
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    def _save_sam2_outputs(
        self,
        amg_masks: List[Dict[str, Any]],
        h: int,
        w: int,
        out_dir: Path,
        path_stem: str,
        image_path: str,
        timestamp: str,
    ) -> Dict[str, str]:
        """
        Save SAM2 outputs independently of depth-mask outputs.
        Returns relative paths from out_dir for metadata wiring.
        """
        sam2_dir = out_dir / "sam2"
        sam2_masks_dir = sam2_dir / "masks"
        sam2_dir.mkdir(parents=True, exist_ok=True)
        sam2_masks_dir.mkdir(parents=True, exist_ok=True)

        seg_png_path = sam2_dir / f"{path_stem}_sam2_segmentation.png"
        self._save_sam2_segmentation_image(amg_masks, h, w, seg_png_path)

        mask_records: List[Dict[str, Any]] = []
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            mask_path = sam2_masks_dir / f"{path_stem}_sam2_mask_{idx:04d}.png"
            cv2.imwrite(str(mask_path), mask_uint8)
            mask_records.append({
                "mask_index": idx,
                "mask_path": f"sam2/masks/{path_stem}_sam2_mask_{idx:04d}.png",
                "bbox_xywh": [float(v) for v in amg.get("bbox", [0, 0, 0, 0])],
                "area": int(amg.get("area", int(np.sum(mask > 0)))),
                "predicted_iou": float(amg.get("predicted_iou", 0.0)),
                "stability_score": float(amg.get("stability_score", 0.0)),
            })

        sam2_json = {
            "metadata": {
                "image_path": image_path,
                "image_stem": path_stem,
                "timestamp": timestamp,
                "image_size": [w, h],
                "model": "SAM2",
                "mode": "automatic_mask_generator",
            },
            "summary": {
                "num_masks": len(mask_records),
                "segmentation_map_image_path": f"sam2/{path_stem}_sam2_segmentation.png",
                "masks_dir": "sam2/masks",
            },
            "masks": mask_records,
        }

        sam2_json_path = sam2_dir / f"{path_stem}_sam2_masks.json"
        with open(sam2_json_path, "w") as f:
            json.dump(sam2_json, f, indent=2)

        return {
            "sam2_json_path": f"sam2/{path_stem}_sam2_masks.json",
            "sam2_segmentation_image_path": f"sam2/{path_stem}_sam2_segmentation.png",
            "sam2_masks_dir": "sam2/masks",
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

        # Fix 5.2: Apply lens distortion correction before any processing.
        # Undistortion must happen first: depth estimation, SAM2 segmentation,
        # and back-projection all assume a pinhole camera model. Uncorrected
        # barrel/pincushion distortion causes edge objects to back-project to
        # wrong 3D positions proportional to their distance from image center.
        img_bgr = self._undistort_image(img_bgr)

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

        # 1. Intrinsics (Fix 5.2: calibration file > explicit values > FOV estimate)
        K = self.fixed_intrinsics if self.fixed_intrinsics else self._estimate_intrinsics(w, h)

        # 2. Depth Estimation
        # Fix 5.1: backend.infer() now returns RAW METRIC METERS from the
        # Depth-Anything-V2-Metric model. depth_scale_factor is 1.0 in config
        # so metric_depth = raw_meters * 1.0 = true metric depth.
        # The old ×10 hack is no longer applied.
        raw_depth = self.depth_estimator.backend.infer(img_rgb)
        depth_full = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_NEAREST)
        metric_depth = depth_full * self.depth_scale_factor  # 1.0 for metric models

        # Save Depth
        depth_dir = out / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_dir / f"{path.stem}_depth_metric.npy", metric_depth)

        # 3. Object-level segmentation + labelling
        # Fix 5.3: GroundedSAM2Wrapper runs Grounding DINO for object-level bbox
        # detection, then SAM2 prompted mode for one clean mask per object.
        # Fix 5.4: Labels come from GDINO (if available) or Florence-2 OD on crop.

        # 3b. SAM2 AMG + Depth-Mask outputs (depth map image, SAM2 seg, depth+mask mapping + JSON per mode)
        scene_graph_dir = out / "scene_graph"
        scene_graph_dir.mkdir(parents=True, exist_ok=True)
        depth_mask_dir = scene_graph_dir / "depth_mask"
        masks_dir = scene_graph_dir / "masks"
        save_per_object_masks = getattr(self.config, "save_per_object_masks", True) if self.config else True
        save_masked_depth_npy = getattr(self.config, "save_masked_depth_npy", False) if self.config else False

        self._save_depth_map_image(metric_depth, scene_graph_dir / f"{path.stem}_depth_map.png")
        depth_map_image_rel = f"scene_graph/{path.stem}_depth_map.png"
        depth_map_npy_rel = f"depth/{path.stem}_depth_metric.npy"
        depth_global_min = float(np.min(metric_depth))
        depth_global_max = float(np.max(metric_depth))
        depth_global_mean = float(np.mean(metric_depth))

        amg_masks = self.sam2_wrapper.generate(img_rgb)

        # Dual-segmentor: run SAM2 AMG alongside GroundedSAM2 to capture
        # part-level and small-object masks that GDINO may miss.
        # AMG masks that overlap a GDINO mask by > iou_dedup are dropped
        # (the GDINO-prompted mask is higher quality for that object).
        # Remaining AMG masks cover: parts, small objects, background regions.
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

        seg_map_rel = f"scene_graph/{path.stem}_sam2_segmentation.png"
        sam2_paths = self._save_sam2_outputs(
            amg_masks=amg_masks,
            h=h,
            w=w,
            out_dir=out,
            path_stem=path.stem,
            image_path=str(path.resolve()),
            timestamp=timestamp,
        )
        if amg_masks:
            self._save_sam2_segmentation_image(amg_masks, h, w, scene_graph_dir / f"{path.stem}_sam2_segmentation.png")

        # Build all_detections: one entry per SAM2/AMG mask — NO filtering applied.
        # All masks are kept: small objects, background regions, part-level masks
        # (from AMG), and object-level masks (from GroundedSAM2).
        # graph_id format: obj_{mask_index}_GroundedSAM2 to keep object IDs stable.
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
        # Free GPU memory accumulated across many GRiT inference calls
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


        # Pix2SG is deferred to after Stage 4 so objects_3d entries (carrying
        # _sam2_mask_array and mask_centroid_2d) can be passed instead of all_detections.

        # A-mode mapping keeps object IDs anchored to the original SAM2 mask index.
        # This is robust even when some masks were filtered out in Stage 3.
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
                mask_path_rel = f"scene_graph/masks/{path.stem}_obj_{i}_mask_{mode}.png"
                masked_depth_path_rel = f"scene_graph/masks/{path.stem}_obj_{i}_masked_depth_{mode}.npy"
                if save_per_object_masks:
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_uint8 = (np.asarray(mask).astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
                    if mask_uint8.shape[:2] != (h, w):
                        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(masks_dir / f"{path.stem}_obj_{i}_mask_{mode}.png"), mask_uint8)
                    if save_masked_depth_npy:
                        mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (h, w) else mask.astype(np.float32)
                        masked_depth = metric_depth.astype(np.float32) * (mask_resized > 0)
                        np.save(masks_dir / f"{path.stem}_obj_{i}_masked_depth_{mode}.npy", masked_depth)
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
            mapping_path = scene_graph_dir / f"{path.stem}_depth_mask_mapping_{mode}.png"
            self._save_depth_mask_mapping_image(metric_depth, matched, mapping_path)
            mapping_rel = f"scene_graph/{path.stem}_depth_mask_mapping_{mode}.png"
            depth_mask_dir.mkdir(parents=True, exist_ok=True)
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
            with open(depth_mask_dir / f"{path.stem}_depth_mask_{mode}.json", "w") as f:
                json.dump(dm_json, f, indent=2)

        # 4. Integration & Projection (mask-native)
        # Every object is built directly from a SAM2 mask — no bbox fallback exists.
        # coordinates_3d, depth_stats, and mask_centroid_2d are all pixel-native:
        #   z = median depth over mask foreground pixels (robust to noise)
        #   (cx, cy) = centre-of-mass of mask pixels
        # _sam2_mask_array is stored as a transient field for downstream Pix2SG;
        # it is stripped before json.dump in Stage 6.
        objects_3d = []

        # Ensure objects directory exists
        objects_dir = scene_graph_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

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
            mask_path_rel: Optional[str] = f"scene_graph/masks/{path.stem}_obj_{i}_mask_A.png"
            mask_matched = True

            # Per-object depth visualization — use mask bbox (SAM2's own bbox)
            x1, y1, x2, y2 = (
                max(0, int(bbox[0])), max(0, int(bbox[1])),
                min(w, int(bbox[2])), min(h, int(bbox[3])),
            )
            object_depth_map = metric_depth[y1:y2, x1:x2]
            obj_depth_filename = f"{path.stem}_obj_{i}_{src}_depth.png"
            obj_depth_path = objects_dir / obj_depth_filename

            if object_depth_map.size > 0:
                d_min, d_max = object_depth_map.min(), object_depth_map.max()
                if d_max - d_min > 1e-6:
                    norm_obj_depth = ((object_depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    norm_obj_depth = np.zeros_like(object_depth_map, dtype=np.uint8)
                norm_obj_depth = cv2.applyColorMap(norm_obj_depth, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(obj_depth_path), norm_obj_depth)

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
                "depth_map_path": f"objects/{obj_depth_filename}",
                "sources": {
                    "GroundedSAM2": {
                        "caption": str(det.get("caption", grounded_label)),
                        "label": grounded_label,
                        "confidence": gdino_confidence,
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

        # 6. Save Combined Scene Graph
        scene_graph_dir = out / "scene_graph"
        scene_graph_dir.mkdir(parents=True, exist_ok=True)

        # Convert Intrinsics values to native python types for JSON serialization
        if K:
            K_serializable = {k: float(v) for k, v in K.items()}
        else:
            K_serializable = None

        models_used = ["GroundedSAM2", "Florence-2"]
        if self.grit.predictor is not None:
            models_used.append("GRiT")  # fallback active
        if self.pix2sg.is_active():
            models_used.append("Pix2SG")

        metadata = {
            "timestamp": timestamp,
            "intrinsics": K_serializable,
            "models": models_used,
            "relation_sources": self._collect_relation_source_status(),
            "relation_debug": {
                "pix2sg": pix2sg_stats,
                "mask_iou_match_thresh": float(self.mask_iou_match_thresh),
                "pix2sg_mask_overlap_thresh": float(self.pix2sg_mask_overlap_thresh),
                "pix2sg_depth_near_threshold": float(self.pix2sg_depth_near_threshold),
                "raw_triplets": {
                    "pix2sg": int(len(pix2sg_out)),
                },
                "num_detected_objects": int(len(all_detections)),
                "num_mask_matched": int(sum(1 for o in objects_3d if o.get("mask_matched"))),
            },
        }
        if "A" in self.depth_mask_modes:
            metadata["depth_mask_json_A"] = f"scene_graph/depth_mask/{path.stem}_depth_mask_A.json"
        if "B" in self.depth_mask_modes:
            metadata["depth_mask_json_B"] = f"scene_graph/depth_mask/{path.stem}_depth_mask_B.json"
        metadata["sam2_json"] = sam2_paths["sam2_json_path"]
        metadata["sam2_segmentation_image"] = sam2_paths["sam2_segmentation_image_path"]
        metadata["sam2_masks_dir"] = sam2_paths["sam2_masks_dir"]
        # Strip transient numpy mask arrays before JSON serialization.
        # _sam2_mask_array is not JSON-serializable; stripping here is safe because
        # Pix2SG spatial predicates have already consumed it above.
        for obj in objects_3d:
            obj.pop("_sam2_mask_array", None)
        json_output = {"metadata": metadata, "objects": objects_3d}
        with open(scene_graph_dir / f"{path.stem}_scene.json", 'w') as f:
            json.dump(json_output, f, indent=2)

        # 7. Visualization
        viz = img_bgr.copy()
        for obj in objects_3d:
            x, y, z = obj['coordinates_3d'].values()
            bbox = obj['bbox']
            # All objects are GRiT-labelled mask-native — uniform color
            color = (0, 255, 0)

            label = f"{obj['label']} [M]"
            thickness = 2
            cv2.rectangle(viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
            cv2.putText(viz, label, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw mask centroid dot for matched objects — shows the actual 3D anchor point
            mc = obj.get("mask_centroid_2d")
            if mc and len(mc) == 2 and obj.get("mask_matched"):
                cv2.circle(viz, (int(mc[0]), int(mc[1])), 3, color, -1)

            # Relation line anchor: prefer mask centroid over bbox center
            if mc and len(mc) == 2:
                cx_a, cy_a = int(mc[0]), int(mc[1])
            else:
                cx_a, cy_a = (int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2

            # Draw Relations
            for source in ["Pix2SG", "SGTR"]:
                if source in obj["sources"]:
                    for rel in obj["sources"][source]["relations"]:
                        target_id = rel['target_id']
                        # Skip external targets — no location to draw to
                        if isinstance(target_id, str) and target_id.startswith("external_"):
                            continue
                        target = next((o for o in objects_3d if o['id'] == target_id), None)
                        if target:
                            bbox_b = target['bbox']
                            mc_b = target.get("mask_centroid_2d")
                            if mc_b and len(mc_b) == 2:
                                cx_b, cy_b = int(mc_b[0]), int(mc_b[1])
                            else:
                                cx_b = (int(bbox_b[0]) + int(bbox_b[2])) // 2
                                cy_b = (int(bbox_b[1]) + int(bbox_b[3])) // 2
                            line_color = (0, 255, 255)
                            cv2.line(viz, (int(cx_a), int(cy_a)), (int(cx_b), int(cy_b)), line_color, 1)
                            mx = (int(cx_a) + int(cx_b)) // 2
                            my = (int(cy_a) + int(cy_b)) // 2
                            cv2.putText(viz, rel['predicate'], (int(mx), int(my)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

        cv2.imwrite(str(scene_graph_dir / f"{path.stem}_3d_viz.png"), viz)
        print(f"Results saved to {out}")

        # Release large arrays so memory is available before the next image
        del img_bgr, img_rgb, metric_depth, amg_masks, all_detections
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    from config import PreprocessConfig
    
    print("Testing pipeline initialization...")
    cfg = PreprocessConfig()
    images_dir = Path("images")
    output_dir = "output_scene"
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
                pipeline.process_image(str(img_path), output_dir)
            except ValueError as e:
                print(f"Skipping {img_path.name}: {e}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
