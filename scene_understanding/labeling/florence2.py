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

    # Process-wide circuit breaker. The Florence-2 weights are ~2GB; if the
    # first init fails (e.g. missing flash_attn, incompatible transformers
    # version), re-entering __init__ for every region burns minutes of
    # wall-clock and memory. Once we've failed once, every subsequent wrapper
    # short-circuits to an inactive instance so callers simply skip Florence.
    _INIT_FAILED: bool = False
    _INIT_FAIL_REASON: str = ""

    # Phase B.1 / B.6: task-specific generation budgets + prompt-token cache.
    # Keys are Florence-2 task tokens; values are the caps the plan mandates:
    # short budgets for structured tasks (<OD>, <CAPTION>, relations) and the
    # full 256 retained for <MORE_DETAILED_CAPTION> where rich text is the
    # whole point. Output preservation: longer tasks never see smaller caps.
    _TASK_MAX_NEW_TOKENS = {
        "<OD>": 32,
        "<CAPTION>": 32,
        "<DETAILED_CAPTION>": 128,
        "<MORE_DETAILED_CAPTION>": 256,
        "<DENSE_REGION_CAPTION>": 128,
        "<CAPTION_TO_PHRASE_GROUNDING>": 64,
    }

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: torch.device = None,
        *,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        use_cache: Optional[bool] = None,
        od_fallback_enabled: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.active = False
        self._model_id = model_id
        # Dtype selection — fp16 is the right default on MPS / CUDA, fp32 on CPU.
        _requested_dtype = (dtype or "").strip().lower()
        if _requested_dtype in ("fp16", "float16", "half"):
            self._dtype = torch.float16
        elif _requested_dtype in ("bf16", "bfloat16"):
            self._dtype = torch.bfloat16
        elif _requested_dtype in ("fp32", "float32"):
            self._dtype = torch.float32
        else:
            # Default: fp16 on accelerator devices, fp32 on CPU to avoid
            # numerical-op fallbacks that silently hurt CPU runs.
            self._dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        # Generation-side plumbing.
        self._use_cache = True if use_cache is None else bool(use_cache)
        # When False, skip the secondary <OD> pass in label_crop(s) and keep
        # caption-derived labels only. This is a strict performance/latency
        # switch; the primary caption pass is unchanged.
        self._od_fallback_enabled = bool(od_fallback_enabled)
        # Attention backend selection.
        #   - CUDA: prefer sdpa (or flash_attention_2 if user requested).
        #   - MPS/CPU: default to ``eager``. SDPA on MPS routes through
        #     Florence-2's custom modeling code which tries to import
        #     ``flash_attn`` unconditionally, and failing that raises the
        #     infamous "flash_attn.__spec__ is None" — which triggered a
        #     per-region retry loop that ate minutes of wall-clock. Staying in
        #     eager on non-CUDA devices avoids the whole mess and keeps
        #     throughput predictable.
        import os as _os
        _env_attn = str(_os.getenv("FLORENCE2_ATTN_IMPL", "")).strip().lower()
        if attn_implementation is not None:
            self._attn_impl = attn_implementation
        elif _env_attn in ("sdpa", "eager", "flash_attention_2"):
            self._attn_impl = _env_attn
        elif self.device.type == "cuda":
            self._attn_impl = "sdpa"
        else:
            self._attn_impl = "eager"
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}
        # Phase B.9 — per-(object_id, task) result memo. Callers can opt into
        # memoisation by passing ``cache_key=object_id`` on single-image calls;
        # cleared via :meth:`reset_image_caches` at the end of each frame.
        self._result_memo: Dict[tuple, Any] = {}

        # Circuit breaker — if a previous construction in this process failed,
        # silently return an inactive wrapper rather than paying the 5–10 s
        # weight-download cost per region.
        if Florence2Wrapper._INIT_FAILED:
            return

        print(f"Initializing Florence-2 ({model_id}, dtype={self._dtype}, attn={self._attn_impl})...")
        try:
            try:
                from transformers.tokenization_utils_tokenizers import TokenizersBackend
                if not hasattr(TokenizersBackend, "additional_special_tokens"):
                    TokenizersBackend.additional_special_tokens = property(
                        lambda self: []
                    )
            except Exception:
                pass

            # Provide a *proper* flash_attn shim iff something downstream is
            # going to import it. ``types.ModuleType`` alone leaves ``__spec__
            # == None`` which newer transformers versions treat as "broken
            # package". Attach a real ``ModuleSpec`` so the module-system
            # checks succeed. Only install when we actually need sdpa/flash
            # paths — eager attention doesn't touch flash_attn.
            if self._attn_impl in ("sdpa", "flash_attention_2"):
                try:
                    import importlib, importlib.util, importlib.machinery
                    if importlib.util.find_spec("flash_attn") is None:
                        import types
                        _shim = types.ModuleType("flash_attn")
                        _shim.__version__ = "0.0.0"
                        _shim.__spec__ = importlib.machinery.ModuleSpec(
                            "flash_attn", loader=None
                        )
                        sys.modules.setdefault("flash_attn", _shim)
                except Exception:
                    pass

            from transformers import AutoProcessor, AutoModelForCausalLM
            _proc_kwargs = dict(trust_remote_code=True)
            try:
                self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False, **_proc_kwargs)
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(model_id, **_proc_kwargs)

            # Try the requested attn implementation first; fall back to eager
            # if SDPA is refused by the Florence-2 custom code (known issue on
            # transformers >=5 for some model revisions).
            _attn_for_load = self._attn_impl
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    attn_implementation=_attn_for_load,
                ).to(self.device)
            except Exception as sdpa_err:
                if _attn_for_load != "eager":
                    print(f"  [Florence2] {_attn_for_load} load failed ({sdpa_err}); falling back to eager.")
                    self._attn_impl = "eager"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=self._dtype,
                        trust_remote_code=True,
                        attn_implementation="eager",
                    ).to(self.device)
                else:
                    raise
            self.model.eval()
            self.active = True
            print(f"Florence-2 ready ({model_id}, attn={self._attn_impl}, use_cache={self._use_cache}).")
        except Exception as e:
            Florence2Wrapper._INIT_FAILED = True
            Florence2Wrapper._INIT_FAIL_REASON = f"{e}"
            print(
                f"Florence-2 init failed: {e}. "
                "Disabling Florence-2 for the rest of this run to avoid per-region retry storm."
            )

    def _max_new_tokens_for(self, task: str) -> int:
        return int(self._TASK_MAX_NEW_TOKENS.get(task, 256))

    def _run_task(self, task: str, pil_image, extra_text: str = "") -> Any:
        """Run a single Florence-2 task on a PIL image. Returns parsed result.

        Internally delegates to :meth:`_run_task_batched` with a single-image
        batch so the sequential and batched code paths share one generation.
        """
        if not self.active:
            return {}
        out = self._run_task_batched(task, [pil_image], extra_text=extra_text)
        return out[0] if out else {}

    def _run_task_batched(
        self,
        task: str,
        pil_images: List[Any],
        extra_text: str = "",
    ) -> List[Any]:
        """Run one Florence-2 task on a list of PIL images in a single generate call.

        Keeps output-identity with the sequential path because generation is
        greedy (``do_sample=False``, ``num_beams=1``). Per-image
        ``post_process_generation`` preserves the existing result structure.
        """
        if not self.active or not pil_images:
            return [{} for _ in (pil_images or [])]
        try:
            prompt = task + extra_text
            prompts = [prompt] * len(pil_images)
            inputs = self.processor(
                text=prompts, images=pil_images, return_tensors="pt", padding=True
            )
            inputs = {k: (v.to(self.device).contiguous() if hasattr(v, "to") else v) for k, v in inputs.items()}
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != self._dtype:
                try:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
                except Exception:
                    pass
            max_new = self._max_new_tokens_for(task)
            with torch.no_grad():
                _gen_inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
                try:
                    generated = self.model.generate(
                        **_gen_inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        num_beams=1,
                        use_cache=self._use_cache,
                    )
                except (TypeError, AttributeError, KeyError) as kv_err:
                    # Some transformers/Florence-2 combos break with use_cache=True
                    # due to EncoderDecoderCache shape mismatches. Fall back once
                    # per process and remember.
                    if self._use_cache:
                        print(
                            f"  [Florence2] use_cache=True incompatible ({kv_err}); "
                            "falling back to use_cache=False for this process."
                        )
                        self._use_cache = False
                        generated = self.model.generate(
                            **_gen_inputs,
                            max_new_tokens=max_new,
                            do_sample=False,
                            num_beams=1,
                            use_cache=False,
                        )
                    else:
                        raise
            decoded = self.processor.batch_decode(generated, skip_special_tokens=False)
            out: List[Any] = []
            for text_out, pil_img in zip(decoded, pil_images):
                try:
                    parsed = self.processor.post_process_generation(
                        text_out,
                        task=task,
                        image_size=(pil_img.width, pil_img.height),
                    )
                except Exception as e:
                    print(f"  [Florence2] post_process_generation failed for task={task}: {e}")
                    parsed = {}
                out.append(parsed)
            return out
        except Exception as e:
            print(f"  [Florence2] task={task} batched failed (n={len(pil_images)}): {e}")
            return [{} for _ in pil_images]

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
        """Extract a multi-word noun phrase from a Florence-2 caption.

        Accumulates consecutive non-stopword alpha tokens (up to 3) to produce
        phrases like "coffee mug" or "dining table" instead of just "coffee".
        Stops at the first stopword encountered after the phrase has started.
        """
        if not caption or not isinstance(caption, str):
            return "object"
        tokens = caption.lower().split()
        phrase_tokens: List[str] = []
        for w in tokens:
            w_clean = w.strip(".,;:!?\"'()")
            if w_clean.isalpha() and len(w_clean) > 2 and w_clean not in cls._CAPTION_STOPWORDS:
                phrase_tokens.append(w_clean)
                if len(phrase_tokens) >= 3:
                    break
            elif phrase_tokens:
                break
        return " ".join(phrase_tokens) if phrase_tokens else "object"

    def label_crops(self, crops_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batched Florence-2 labelling across multiple BGR crops.

        Uses a single batched generate call for the primary
        ``<MORE_DETAILED_CAPTION>`` task, then falls through to ``<OD>`` only
        for the crops where caption extraction returned the generic "object"
        token. Output format matches ``label_crop`` per element.
        """
        if not self.active or not crops_bgr:
            return [
                {"label": "object", "conf": 0.0, "caption": "object"} for _ in (crops_bgr or [])
            ]
        from PIL import Image as PILImage
        pil_crops: List[Any] = []
        keep_idx: List[int] = []
        default_result = {"label": "object", "conf": 0.0, "caption": "object"}
        results: List[Dict[str, Any]] = [dict(default_result) for _ in crops_bgr]
        for i, crop_bgr in enumerate(crops_bgr):
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crops.append(PILImage.fromarray(rgb))
            keep_idx.append(i)
        if not pil_crops:
            return results

        cap_parsed = self._run_task_batched("<MORE_DETAILED_CAPTION>", pil_crops)
        # First pass — extract labels from captions.
        fallback_crops: List[Any] = []
        fallback_slots: List[int] = []
        for slot, parsed in enumerate(cap_parsed):
            caption = parsed.get("<MORE_DETAILED_CAPTION>", "") if isinstance(parsed, dict) else ""
            if not isinstance(caption, str):
                caption = str(caption)
            label = self._extract_label_from_caption(caption)
            conf = 0.75
            if label == "object":
                fallback_crops.append(pil_crops[slot])
                fallback_slots.append(slot)
            if not caption:
                caption = label
            out_idx = keep_idx[slot]
            results[out_idx] = {"label": label, "conf": conf, "caption": caption}

        # Second pass — structured <OD> only for crops whose caption yielded
        # "object". Batched generate keeps the Florence-2 forward fixed at 2x
        # rather than n+k.
        if fallback_crops and self._od_fallback_enabled:
            od_parsed_batch = self._run_task_batched("<OD>", fallback_crops)
            for slot, od_parsed in zip(fallback_slots, od_parsed_batch):
                od_data = od_parsed.get("<OD>", {}) if isinstance(od_parsed, dict) else {}
                od_labels = od_data.get("labels", [])
                od_bboxes = od_data.get("bboxes", [])
                label = "object"
                conf = 0.75
                if od_labels:
                    best_area = -1.0
                    for lbl, box in zip(od_labels, od_bboxes):
                        if len(box) >= 4:
                            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
                            if area > best_area:
                                best_area = area
                                label = str(lbl).strip().lower()
                    if label != "object":
                        conf = 0.80
                out_idx = keep_idx[slot]
                # Preserve caption already set (from first pass) — label/conf overwrite.
                results[out_idx] = {
                    **results[out_idx],
                    "label": label,
                    "conf": conf,
                }
        return results

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
        if label == "object" and self._od_fallback_enabled:
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

    def reset_image_caches(self) -> None:
        """Clear per-image memo caches.

        Called by the pipeline at the end of each frame so memoised
        per-(object_id, task) results from one image cannot leak into the
        next; the tokenizer-level prompt cache lives on ``_prompt_cache`` and
        is intentionally preserved across frames.
        """
        try:
            self._result_memo.clear()
        except Exception:
            pass

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

            # Burn subject/object labels as text onto the crop so Florence-2 can
            # read them as image context. Florence-2 rejects extra text appended to
            # task tokens, so label injection must be visual rather than textual.
            try:
                from PIL import ImageDraw as _IDraw
                labelled_crop = pil_crop.copy()
                _draw = _IDraw.Draw(labelled_crop)
                _sub_txt = f"RED: {label_sub or 'object'}"
                _obj_txt = f"BLUE: {label_obj or 'object'}"
                _draw.text((4, 4), _sub_txt, fill=(255, 60, 60))
                _draw.text((4, 20), _obj_txt, fill=(60, 60, 255))
            except Exception:
                labelled_crop = pil_crop

            result = self._run_task("<MORE_DETAILED_CAPTION>", labelled_crop)
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
