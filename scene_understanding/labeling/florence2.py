"""Florence-2 wrapper — canonical implementation (synced from scene_understanding legacy module)."""
from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


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
                torch_dtype=_dtype,
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
                # Exclude attention_mask: the processor returns a [1,1,14,14] patch-grid
                # mask but modeling_florence2 expects [1,1,seq,seq]. Dropping it is
                # safe for single-image inference (no padding, mask is all-ones).
                _gen_inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
                generated = self.model.generate(
                    **_gen_inputs,
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
