"""RAM++ wrapper — canonical implementation (synced from scene_understanding legacy module)."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch


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
        *,
        prefer_mps: bool = False,
        mps_dtype: str = "fp16",
        mps_watermark_fraction: float = 0.85,
        backend: str = "rampp",
        clip_fallback_enabled: bool = True,
        clip_candidate_labels: Optional[List[str]] = None,
        quantize_dynamic_int8: bool = True,
    ):
        self.device = device
        # Phase B.2: opt-in MPS path. Historically RAM++ runs on CPU when the
        # pipeline device is MPS because the Swin backbone triggered silent
        # fallbacks and unpredictable crashes. With prefer_mps=True we try MPS
        # in fp16 behind a watermark guard + sanity probe, falling back to CPU
        # on the first symptom. Setting PYTORCH_ENABLE_MPS_FALLBACK=1 lets
        # unsupported ops dispatch to CPU silently instead of raising.
        self._prefer_mps = bool(prefer_mps) and device.type == "mps"
        self._mps_dtype = str(mps_dtype or "fp16").lower()
        self._mps_watermark_fraction = float(mps_watermark_fraction)
        self._ram_device = torch.device("cpu") if device.type == "mps" else device
        self._autocast_dtype: Optional[torch.dtype] = None
        self._backend = str(backend or "rampp").strip().lower()
        if self._backend not in {"rampp", "ram", "clip"}:
            self._backend = "rampp"
        self._clip_fallback_enabled = bool(clip_fallback_enabled)
        self._clip_candidate_labels = list(clip_candidate_labels or [])
        self._quantize_dynamic_int8 = bool(quantize_dynamic_int8)
        self._runtime_backend_mode = self._backend
        self.model = None
        self.transform = None
        self.inference_fn = None
        self.active = False
        self.default_confidence = float(default_confidence)
        self.max_tags = int(max_tags)
        self._clip_model = None
        self._clip_processor = None
        self._clip_model_id = "openai/clip-vit-base-patch32"

        if repo_path:
            repo = Path(repo_path).expanduser()
            if not repo.is_absolute():
                repo = Path.cwd() / repo
            if repo.exists():
                repo_str = str(repo.resolve())
                if repo_str not in sys.path:
                    sys.path.insert(0, repo_str)

        ckpt: Optional[Path] = None
        if self._backend != "clip":
            if checkpoint_path:
                ckpt = Path(checkpoint_path).expanduser()
                if not ckpt.is_absolute():
                    ckpt = Path.cwd() / ckpt
                if not ckpt.exists():
                    print(f"RAM checkpoint not found at {ckpt}.")
                    if self._clip_fallback_enabled:
                        self._enable_clip_fallback(reason="missing checkpoint")
                    return
            else:
                print("RAM checkpoint path not configured.")
                if self._clip_fallback_enabled:
                    self._enable_clip_fallback(reason="checkpoint not configured")
                return
            _sz = int(ckpt.stat().st_size)
            if _sz < 512:
                print(
                    f"RAM checkpoint missing or empty ({ckpt}, {_sz} bytes). "
                    "Download a full .pth from the Recognize Anything model zoo."
                )
                if self._clip_fallback_enabled:
                    self._enable_clip_fallback(reason="empty checkpoint")
                return
            try:
                with open(ckpt, "rb") as _bf:
                    _head = _bf.read(64)
                if _head.startswith(b"version https://git-lfs"):
                    print(
                        f"RAM checkpoint at {ckpt} is a Git LFS pointer, not weights. "
                        "Run `git lfs pull` in the repo that owns this file, or download the full .pth."
                    )
                    if self._clip_fallback_enabled:
                        self._enable_clip_fallback(reason="checkpoint is LFS pointer")
                    return
            except OSError as _e:
                print(f"RAM checkpoint unreadable ({ckpt}): {_e}.")
                if self._clip_fallback_enabled:
                    self._enable_clip_fallback(reason="checkpoint unreadable")
                return

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

        if self._backend == "clip":
            self._enable_clip_fallback(reason="configured backend=clip")
            return

        _backend_name = "RAM++" if self._backend == "rampp" else "RAM"
        print(f"Initializing {_backend_name}...")
        try:
            from ram.models import ram as _ram_base, ram_plus as _ram_plus
            from ram import get_transform, inference_ram as inference
        except Exception as e:
            print(f"{_backend_name} unavailable: {e}.")
            if self._clip_fallback_enabled:
                self._enable_clip_fallback(reason=f"{_backend_name} import failure")
            return

        try:
            model_ctor = _ram_plus if self._backend == "rampp" else _ram_base
            model = model_ctor(
                pretrained=str(ckpt),
                image_size=int(image_size),
                vit=str(vit),
            )
            model.eval()

            # Decide target device with watermark guard before moving weights.
            target_device = self._select_ram_device(model, image_size=int(image_size))
            if target_device.type == "mps":
                # fp16 weights keep the Swin backbone under ~350 MB and cut MPS
                # kernel dispatch time roughly in half. Ensure tensors are
                # contiguous — RAM++'s patch-embed linear is known to error on
                # non-contiguous fp16 inputs.
                _target_dtype = torch.float16 if self._mps_dtype in ("fp16", "half") else torch.float32
                try:
                    model = model.to(target_device, dtype=_target_dtype)
                except Exception as _mps_err:
                    print(f"  [RAM++] MPS load failed ({_mps_err}); falling back to CPU.")
                    target_device = torch.device("cpu")
                    model = model.to(target_device)
                else:
                    # Force parameter contiguity once, up-front.
                    with torch.no_grad():
                        for p in model.parameters():
                            if not p.is_contiguous():
                                p.data = p.data.contiguous()
                    self._autocast_dtype = _target_dtype
            else:
                model = model.to(target_device)
                if target_device.type == "cpu" and self._quantize_dynamic_int8:
                    try:
                        import torch.nn as _nn
                        model = torch.quantization.quantize_dynamic(
                            model,
                            {_nn.Linear},
                            dtype=torch.qint8,
                        )
                        print(f"  [{_backend_name}] dynamic int8 quantization enabled (Linear layers).")
                    except Exception as _qerr:
                        print(f"  [{_backend_name}] int8 quantization skipped: {_qerr}")

            self._ram_device = target_device
            self.model = model
            self.transform = get_transform(image_size=int(image_size))
            self.inference_fn = inference
            self._runtime_backend_mode = self._backend

            # Sanity probe: forward a tiny dummy crop to trigger every lazy init
            # once. If this raises, fall back to CPU and re-probe; only then
            # mark the wrapper active.
            if not self._sanity_probe(image_size=int(image_size)):
                if target_device.type == "mps":
                    print("  [RAM++] MPS sanity probe failed; retrying on CPU.")
                    try:
                        self.model = self.model.to(torch.device("cpu"), dtype=torch.float32)
                    except Exception:
                        pass
                    self._ram_device = torch.device("cpu")
                    self._autocast_dtype = None
                    if not self._sanity_probe(image_size=int(image_size)):
                        raise RuntimeError(f"{_backend_name} sanity probe failed on both MPS and CPU.")
                else:
                    raise RuntimeError(f"{_backend_name} sanity probe failed on target device.")

            self.active = True
            if self._ram_device.type == "mps":
                print(
                    f"{_backend_name} ready (ckpt={ckpt}, device=mps, dtype={self._mps_dtype}). "
                    "Set PYTORCH_ENABLE_MPS_FALLBACK=1 if any op raises to silently use CPU."
                )
            elif self.device.type == "mps" and self._ram_device.type == "cpu":
                print(f"{_backend_name} ready (ckpt={ckpt}, inference on CPU for MPS compatibility).")
            else:
                print(f"{_backend_name} ready (ckpt={ckpt}).")
        except Exception as e:
            msg = str(e).strip() or repr(e)
            print(f"{_backend_name} init failed: {type(e).__name__}: {msg}")
            self.active = False
            if self._clip_fallback_enabled:
                self._enable_clip_fallback(reason=f"{_backend_name} init failure")

    def _select_ram_device(self, model: torch.nn.Module, image_size: int) -> torch.device:
        """Choose MPS vs CPU using a watermark guard on unified memory.

        Returns mps only when the caller requested it AND the MPS memory
        headroom is above ``mps_watermark_fraction`` of the recommended max.
        """
        if not self._prefer_mps:
            return torch.device("cpu") if self.device.type == "mps" else self.device
        if not hasattr(torch, "mps"):
            return torch.device("cpu")
        try:
            recommended = float(torch.mps.recommended_max_memory())
            currently_allocated = float(torch.mps.current_allocated_memory())
        except Exception:
            return torch.device("cpu")
        if recommended <= 0.0:
            return torch.device("cpu")
        # Estimate RAM++ resident footprint: ~param_count * 4 bytes for fp32,
        # halved for fp16. Budget extra for activations during a 384x384 fwd.
        try:
            param_bytes = sum(p.numel() for p in model.parameters()) * (2 if self._mps_dtype in ("fp16", "half") else 4)
        except Exception:
            param_bytes = 0
        activation_bytes = int(3 * image_size * image_size * 4 * 8)
        reserve = param_bytes + activation_bytes
        available = recommended - currently_allocated
        if reserve <= 0 or available <= 0:
            return torch.device("cpu")
        # Require that post-load headroom stays under the watermark.
        post_load_frac = (currently_allocated + reserve) / recommended
        if post_load_frac > self._mps_watermark_fraction:
            print(
                f"  [RAM++] MPS watermark guard: projected usage {post_load_frac:.2%} "
                f"> {self._mps_watermark_fraction:.0%}; staying on CPU."
            )
            return torch.device("cpu")
        return torch.device("mps")

    def _sanity_probe(self, image_size: int) -> bool:
        """Run a single forward on a zero tensor; returns False on any raise.

        This catches the common class of MPS failures (missing ops, shape
        broadcasts) before the first real crop reaches the wrapper.
        """
        if self.model is None or self.inference_fn is None or self.transform is None:
            return False
        try:
            probe = torch.zeros(1, 3, int(image_size), int(image_size), device=self._ram_device)
            if self._ram_device.type == "mps" and self._autocast_dtype == torch.float16:
                probe = probe.to(torch.float16).contiguous()
            with torch.no_grad():
                if self._ram_device.type == "mps" and self._autocast_dtype is not None:
                    with torch.autocast(device_type="mps", dtype=self._autocast_dtype):
                        _ = self.inference_fn(probe, self.model)
                else:
                    _ = self.inference_fn(probe, self.model)
            return True
        except Exception as e:
            print(f"  [RAM++] sanity probe raised: {type(e).__name__}: {e}")
            return False

    def _enable_clip_fallback(self, reason: str = "") -> None:
        """Enable CLIP zero-shot tagging fallback for CPU-only reliability."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_id)
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_id).to(torch.device("cpu"))
            self._clip_model.eval()
            self._runtime_backend_mode = "clip"
            self.active = True
            _msg = f"  [RAM] CLIP fallback active ({self._clip_model_id})"
            if reason:
                _msg += f" — {reason}"
            print(_msg)
        except Exception as e:
            self._clip_model = None
            self._clip_processor = None
            self.active = False
            print(f"  [RAM] CLIP fallback unavailable: {e}")

    def _clip_label_crops(self, crops_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        default_result = {"label": "object", "conf": 0.0, "caption": "object", "tags": []}
        if self._clip_model is None or self._clip_processor is None:
            return [dict(default_result) for _ in (crops_bgr or [])]
        labels = [str(x).strip().lower() for x in (self._clip_candidate_labels or []) if str(x).strip()]
        if not labels:
            labels = ["person", "chair", "table", "sofa", "plant", "bag", "book", "car", "building", "object"]
        try:
            from PIL import Image as PILImage
            pil_images: List[Any] = []
            slots: List[int] = []
            out = [dict(default_result) for _ in crops_bgr]
            for i, crop in enumerate(crops_bgr):
                if crop is None or crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_images.append(PILImage.fromarray(rgb))
                slots.append(i)
            if not pil_images:
                return out
            text = [f"a photo of a {lbl}" for lbl in labels]
            inputs = self._clip_processor(
                text=text,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=-1)
            for row_idx, slot in enumerate(slots):
                _p = probs[row_idx]
                best = int(torch.argmax(_p).item())
                conf = float(_p[best].item())
                label = labels[best]
                topk = min(self.max_tags, len(labels))
                top_vals, top_idx = torch.topk(_p, k=topk)
                tags = [labels[int(j)] for j in top_idx.tolist() if labels[int(j)] != "object"]
                out[slot] = {
                    "label": label,
                    "conf": conf,
                    "caption": f"clip_zero_shot:{label}",
                    "tags": tags,
                }
            return out
        except Exception as e:
            print(f"  [RAM] CLIP zero-shot failed: {e}")
            return [dict(default_result) for _ in (crops_bgr or [])]

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

    def label_crops(self, crops_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batched RAM++ labelling across many crops.

        Stacks transformed tensors into a single ``(B, 3, H, W)`` forward pass.
        Falls back to per-crop calls for rows whose transform raised (unusable
        crop, colour channel mismatch, etc.).
        """
        default_result = {"label": "object", "conf": 0.0, "caption": "object", "tags": []}
        if not self.active or not crops_bgr:
            return [dict(default_result) for _ in (crops_bgr or [])]
        if self._runtime_backend_mode == "clip":
            return self._clip_label_crops(crops_bgr)
        try:
            from PIL import Image as PILImage
        except Exception:
            return [self.label_crop(c) for c in crops_bgr]
        tensors: List[torch.Tensor] = []
        slot_to_input: List[int] = []
        results: List[Dict[str, Any]] = [dict(default_result) for _ in crops_bgr]
        for idx, crop_bgr in enumerate(crops_bgr):
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            try:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)
                t = self.transform(pil_img)
                tensors.append(t)
                slot_to_input.append(idx)
            except Exception:
                # Fallback to single-crop path for this row only.
                results[idx] = self.label_crop(crop_bgr)
        if not tensors:
            return results
        try:
            batch = torch.stack(tensors, dim=0).to(self._ram_device)
            if self._ram_device.type == "mps" and self._autocast_dtype == torch.float16:
                batch = batch.to(torch.float16).contiguous()
            elif not batch.is_contiguous():
                batch = batch.contiguous()
            with torch.no_grad():
                if self._ram_device.type == "mps" and self._autocast_dtype is not None:
                    with torch.autocast(device_type="mps", dtype=self._autocast_dtype):
                        batch_result = self.inference_fn(batch, self.model)
                else:
                    batch_result = self.inference_fn(batch, self.model)
        except Exception as e:
            # Batched path failed — fall back to per-crop so the caller still
            # gets label data rather than an empty block.
            print(f"  [RAM++] label_crops batched forward failed: {e}; falling back per-crop.")
            for slot, idx in enumerate(slot_to_input):
                results[idx] = self.label_crop(crops_bgr[idx])
            return results

        # RAM++ inference can return either a single string (join of tags) or
        # a list of strings — normalise both.
        if isinstance(batch_result, (list, tuple)) and batch_result and isinstance(batch_result[0], (list, tuple)):
            tag_texts = [
                (self._extract_english_tags(row) if not isinstance(row, str) else row)
                for row in batch_result[0]
            ]
        elif isinstance(batch_result, (list, tuple)) and all(isinstance(x, str) for x in batch_result):
            tag_texts = list(batch_result)
        elif isinstance(batch_result, str):
            tag_texts = [batch_result] * len(slot_to_input)
        else:
            tag_texts = [self._extract_english_tags(batch_result)] * len(slot_to_input)

        if len(tag_texts) != len(slot_to_input):
            # The model did not fan out per-example; degrade to per-crop safely.
            for slot, idx in enumerate(slot_to_input):
                results[idx] = self.label_crop(crops_bgr[idx])
            return results

        for slot, idx in enumerate(slot_to_input):
            tag_text = tag_texts[slot]
            tags = self._parse_tags(tag_text, self.max_tags)
            label = tags[0] if tags else "object"
            caption = tag_text if tag_text else label
            conf = self.default_confidence if label != "object" else 0.0
            results[idx] = {"label": label, "conf": conf, "caption": caption, "tags": tags}
        return results

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
        if self._runtime_backend_mode == "clip":
            return self._clip_label_crops([crop_bgr])[0]

        try:
            from PIL import Image as PILImage

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(crop_rgb)
            image = self.transform(pil_img).unsqueeze(0).to(self._ram_device)
            if self._ram_device.type == "mps" and self._autocast_dtype == torch.float16:
                image = image.to(torch.float16).contiguous()
            elif not image.is_contiguous():
                image = image.contiguous()

            with torch.no_grad():
                if self._ram_device.type == "mps" and self._autocast_dtype is not None:
                    with torch.autocast(device_type="mps", dtype=self._autocast_dtype):
                        result = self.inference_fn(image, self.model)
                else:
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
