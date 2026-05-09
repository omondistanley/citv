"""Isolated MLX Florence-2 worker process.

Protocol: JSON-lines over stdin/stdout.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import traceback
from typing import Any, Dict, List

from PIL import Image


_CAPTION_STOPWORDS = {
    "a", "an", "the", "some", "one", "two", "three",
    "with", "on", "of", "in", "at", "by", "and", "or", "from", "to",
    "for", "as", "up", "out", "into", "over", "under", "about", "around",
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "can", "may", "will", "appears", "seems", "showing", "shows", "shown",
    "this", "that", "these", "those", "it", "its", "there", "their",
    "image", "photo", "picture", "view", "close", "shot",
    "side", "top", "front", "back", "left", "right", "center", "middle",
    "red", "blue", "green", "yellow", "white", "black", "brown", "grey",
    "gray", "orange", "purple", "pink", "dark", "light", "bright",
    "large", "small", "big", "little", "tiny", "tall", "short", "long",
    "old", "new", "open", "closed", "empty", "full", "flat", "round",
    "square", "wooden", "metal", "plastic", "glass", "stone", "brick",
    "single", "double", "multiple", "various", "different", "same",
    "very", "quite", "just", "also", "well",
}


def _max_tokens_for(task_prompt: str) -> int:
    table = {
        "<OD>": 32,
        "<CAPTION>": 32,
        "<DETAILED_CAPTION>": 128,
        "<MORE_DETAILED_CAPTION>": 256,
        "<DENSE_REGION_CAPTION>": 128,
        "<CAPTION_TO_PHRASE_GROUNDING>": 64,
    }
    return int(table.get(task_prompt, 128))


def _extract_label_from_caption(caption: str) -> str:
    if not caption:
        return "object"
    for w in str(caption).lower().split():
        w_clean = w.strip(".,;:!?\"'()")
        if w_clean.isalpha() and len(w_clean) > 2 and w_clean not in _CAPTION_STOPWORDS:
            return w_clean
    return "object"


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _od_fallback_label(text: str) -> str:
    for tok in str(text or "").replace("|", ",").split(","):
        t = tok.strip().lower()
        if t.isalpha() and len(t) > 2 and t not in _CAPTION_STOPWORDS:
            return t
    return "object"


def _gen_to_text(out: Any) -> str:
    txt = getattr(out, "text", None)
    if txt is not None:
        return str(txt).strip()
    return str(out).strip()


def _reply(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="fp16")
    args = parser.parse_args()

    try:
        # Compatibility shim for remote Florence-2 configs on newer
        # transformers builds: some model cards reference
        # ``forced_bos_token_id`` even when the field is not materialized on
        # the config instance. Falling back to ``None`` keeps behaviour aligned
        # with older transformers that defaulted this field.
        from transformers.configuration_utils import PretrainedConfig

        _orig_getattribute = PretrainedConfig.__getattribute__

        def _patched_getattribute(self, key):
            if key == "forced_bos_token_id":
                try:
                    return _orig_getattribute(self, key)
                except AttributeError:
                    return None
            return _orig_getattribute(self, key)

        PretrainedConfig.__getattribute__ = _patched_getattribute

        from mlx_vlm import generate, load

        model, processor = load(args.model_id)
    except Exception as exc:
        _reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    _reply({"ready": True})

    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        req_id = int(req.get("id", -1))
        op = str(req.get("op", ""))

        try:
            if op == "shutdown":
                _reply({"id": req_id, "ok": True, "result": "bye"})
                break

            if op == "run_task":
                task = str(req.get("task", "<CAPTION>"))
                text_input = str(req.get("text_input", "") or "")
                prompt = task if not text_input else f"{task} {text_input}"
                pil = _decode_image(str(req.get("image_b64", "")))
                out = generate(
                    model,
                    processor,
                    prompt=prompt,
                    image=pil,
                    max_tokens=int(req.get("max_tokens", _max_tokens_for(task))),
                    temp=0.0,
                )
                _reply({"id": req_id, "ok": True, "result": _gen_to_text(out)})
                continue

            if op == "label_crops":
                images_b64 = req.get("images_b64", []) or []
                use_od = bool(req.get("od_fallback_enabled", True))
                results: List[Dict[str, Any]] = []
                for image_b64 in images_b64:
                    pil = _decode_image(str(image_b64))
                    cap = _gen_to_text(
                        generate(
                            model,
                            processor,
                            prompt="<MORE_DETAILED_CAPTION>",
                            image=pil,
                            max_tokens=_max_tokens_for("<MORE_DETAILED_CAPTION>"),
                            temp=0.0,
                        )
                    )
                    label = _extract_label_from_caption(cap)
                    conf = 0.75
                    if label == "object" and use_od:
                        od_out = _gen_to_text(
                            generate(
                                model,
                                processor,
                                prompt="<OD>",
                                image=pil,
                                max_tokens=_max_tokens_for("<OD>"),
                                temp=0.0,
                            )
                        )
                        od_label = _od_fallback_label(od_out)
                        if od_label != "object":
                            label = od_label
                            conf = 0.80
                    if not cap:
                        cap = label
                    results.append({"label": label, "conf": conf, "caption": cap})
                _reply({"id": req_id, "ok": True, "result": results})
                continue

            if op == "health":
                _reply({"id": req_id, "ok": True, "result": "ok"})
                continue

            _reply({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            _reply(
                {
                    "id": req_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

