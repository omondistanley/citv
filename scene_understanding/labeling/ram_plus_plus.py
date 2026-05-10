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
