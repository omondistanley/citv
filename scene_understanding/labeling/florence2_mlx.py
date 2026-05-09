"""Florence-2 MLX backend via isolated subprocess runtime.

This wrapper intentionally does *not* import ``mlx_vlm`` in the main
pipeline process. Instead it launches a dedicated Python runtime
(``python3.14`` by default) that owns the MLX stack and communicates via
JSON lines over stdin/stdout.

Why: the main pipeline currently pins a transformers version for the HF
path that is incompatible with MLX-VLM's newer requirements. Isolating
MLX in a sidecar process lets both stacks coexist without dependency
conflicts.
"""
from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage


class Florence2MLXWrapper:
    """MLX-VLM backend with an HF-compatible surface."""

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        *,
        dtype: str = "fp16",
        od_fallback_enabled: bool = True,
        python_executable: Optional[str] = None,
        start_timeout_s: float = 120.0,
        request_timeout_s: float = 180.0,
    ) -> None:
        self._model_id = model_id
        self._dtype = dtype
        self._python_executable = str(
            python_executable
            or os.getenv("FLORENCE2_MLX_PYTHON", "").strip()
            or "python3.14"
        )
        self._start_timeout_s = float(start_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._worker: Optional[subprocess.Popen] = None
        self._rpc_id = 0
        self._model: Optional[object] = None
        self._processor: Optional[object] = None
        self.active: bool = False
        self.available: bool = False
        self._result_memo: Dict[tuple, Any] = {}
        self._od_fallback_enabled = bool(od_fallback_enabled)
        self._ensure_loaded()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, value):
        self._processor = value
        if value is None:
            self._shutdown_worker()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self.active:
            return True
        return self._start_worker()

    def _start_worker(self) -> bool:
        worker_script = Path(__file__).with_name("florence2_mlx_worker.py")
        cmd = [
            self._python_executable,
            "-u",
            str(worker_script),
            "--model-id",
            self._model_id,
            "--dtype",
            str(self._dtype or "fp16"),
        ]
        try:
            self._worker = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(
                f"[Florence2MLX] failed to start worker with '{self._python_executable}': {exc}"
            )
            self.available = False
            return False

        deadline = time.monotonic() + max(1.0, self._start_timeout_s)
        ready_line = ""
        while time.monotonic() < deadline:
            if self._worker.poll() is not None:
                break
            if self._worker.stdout is None:
                break
            rlist, _, _ = select.select([self._worker.stdout], [], [], 0.2)
            if not rlist:
                continue
            ready_line = self._worker.stdout.readline().strip()
            if ready_line:
                break
        try:
            payload = json.loads(ready_line) if ready_line else {}
        except Exception:
            payload = {}
        if payload.get("ready"):
            self.active = True
            self.available = True
            self._model = object()
            self._processor = object()
            return True

        err = payload.get("error", "")
        if self._worker is not None and self._worker.stderr is not None:
            try:
                _rlist, _, _ = select.select([self._worker.stderr], [], [], 0.05)
                if _rlist:
                    _line = self._worker.stderr.readline().strip()
                    if _line:
                        err = f"{err} | {_line}" if err else _line
            except Exception:
                pass
        print(f"[Florence2MLX] failed to load {self._model_id}: {err or 'worker did not report ready'}")
        self._shutdown_worker()
        self.active = False
        self.available = False
        return False

    def _shutdown_worker(self) -> None:
        proc = self._worker
        self._worker = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                req = {"id": -1, "op": "shutdown"}
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._model = None
        self._processor = None

    def _encode_image_b64(self, image: Any) -> str:
        pil = self._image_to_pil(image).convert("RGB")
        arr = np.array(pil)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return ""
        return base64.b64encode(enc.tobytes()).decode("ascii")

    def _rpc(self, op: str, payload: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if not self._ensure_loaded():
            return {}
        if self._worker is None or self._worker.poll() is not None:
            self.active = False
            self.available = False
            return {}
        self._rpc_id += 1
        req = {"id": self._rpc_id, "op": op}
        req.update(payload or {})
        try:
            assert self._worker.stdin is not None
            self._worker.stdin.write(json.dumps(req) + "\n")
            self._worker.stdin.flush()
        except Exception as exc:
            print(f"[Florence2MLX] worker send failed: {exc}")
            self._shutdown_worker()
            self.active = False
            self.available = False
            return {}
        _timeout = float(timeout_s if timeout_s is not None else self._request_timeout_s)
        deadline = time.monotonic() + max(1.0, _timeout)
        while time.monotonic() < deadline:
            if self._worker is None or self._worker.poll() is not None:
                break
            if self._worker.stdout is None:
                break
            rlist, _, _ = select.select([self._worker.stdout], [], [], 0.2)
            if not rlist:
                continue
            line = self._worker.stdout.readline()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if int(msg.get("id", -999999)) != self._rpc_id:
                continue
            if bool(msg.get("ok", False)):
                return msg
            print(f"[Florence2MLX] worker error: {msg.get('error', 'unknown')}")
            return {}
        print(f"[Florence2MLX] worker timeout on op={op}")
        self._shutdown_worker()
        self.active = False
        self.available = False
        return {}

    def reset_image_caches(self) -> None:
        try:
            self._result_memo.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _image_to_pil(self, image: Any) -> PILImage.Image:
        if isinstance(image, PILImage.Image):
            return image
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return PILImage.fromarray(arr)
        if arr.ndim == 2:
            return PILImage.fromarray(arr).convert("RGB")
        raise ValueError(f"Unsupported image dtype/shape for MLX: {arr.shape}/{arr.dtype}")

    def run_task(
        self,
        task_prompt: str,
        image: Any,
        *,
        text_input: Optional[str] = None,
    ) -> str:
        image_b64 = self._encode_image_b64(image)
        if not image_b64:
            return ""
        resp = self._rpc(
            "run_task",
            {
                "task": str(task_prompt),
                "text_input": str(text_input or ""),
                "image_b64": image_b64,
                "max_tokens": int(_max_tokens_for(task_prompt)),
            },
        )
        return str(resp.get("result", "") or "")

    def _run_task(self, task: str, pil_image, extra_text: str = "") -> Dict[str, Any]:
        raw = self.run_task(task, pil_image, text_input=extra_text if extra_text else None)
        if not raw:
            return {}
        return {task: raw}

    # ------------------------------------------------------------------
    # Compatibility-friendly helpers
    # ------------------------------------------------------------------
    def caption(
        self,
        image: Any,
        *,
        task: str = "<CAPTION>",
    ) -> str:
        """Return a caption for ``image`` (``<CAPTION>`` by default)."""
        return self.run_task(task, image)

    def label_object(
        self,
        image_crop: Any,
    ) -> str:
        """Compatibility shim for per-object labelling."""
        return self.run_task("<OD>", image_crop)

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

    @classmethod
    def _extract_label_from_caption(cls, caption: str) -> str:
        if not caption or not isinstance(caption, str):
            return "object"
        for w in caption.lower().split():
            w_clean = w.strip(".,;:!?\"'()")
            if w_clean.isalpha() and len(w_clean) > 2 and w_clean not in cls._CAPTION_STOPWORDS:
                return w_clean
        return "object"

    def label_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        if not self.active or crop_bgr is None or crop_bgr.size == 0:
            return {"label": "object", "conf": 0.0, "caption": "object"}
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = PILImage.fromarray(crop_rgb)
        cap_result = self._run_task("<MORE_DETAILED_CAPTION>", pil_crop)
        caption = str(cap_result.get("<MORE_DETAILED_CAPTION>", "") or "")
        label = self._extract_label_from_caption(caption)
        conf = 0.75

        if label == "object" and self._od_fallback_enabled:
            od_result = self._run_task("<OD>", pil_crop)
            od_raw = str(od_result.get("<OD>", "") or "")
            od_tokens = [t.strip().lower() for t in od_raw.replace("|", ",").split(",") if t.strip()]
            for tok in od_tokens:
                if tok.isalpha() and len(tok) > 2 and tok not in self._CAPTION_STOPWORDS:
                    label = tok
                    conf = 0.80
                    break

        if not caption:
            caption = label
        return {"label": label, "conf": conf, "caption": caption}

    def label_crops(self, crops_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not self.active or not crops_bgr:
            return [{"label": "object", "conf": 0.0, "caption": "object"} for _ in (crops_bgr or [])]
        payload_images: List[str] = []
        slots: List[int] = []
        out: List[Dict[str, Any]] = [
            {"label": "object", "conf": 0.0, "caption": "object"} for _ in crops_bgr
        ]
        for i, crop_bgr in enumerate(crops_bgr):
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            ok, enc = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            payload_images.append(base64.b64encode(enc.tobytes()).decode("ascii"))
            slots.append(i)
        if not payload_images:
            return out
        resp = self._rpc(
            "label_crops",
            {
                "images_b64": payload_images,
                "od_fallback_enabled": bool(self._od_fallback_enabled),
            },
        )
        results = resp.get("result", [])
        if not isinstance(results, list):
            return out
        for i, obj in zip(slots, results):
            if isinstance(obj, dict):
                out[i] = {
                    "label": str(obj.get("label", "object")),
                    "conf": float(obj.get("conf", 0.0) or 0.0),
                    "caption": str(obj.get("caption", "object")),
                }
        return out


def _max_tokens_for(task_prompt: str) -> int:
    """Mirror ``Florence2Wrapper._TASK_MAX_NEW_TOKENS`` so both backends
    share the same truncation budget and produce comparable outputs
    during validation."""
    table = {
        "<OD>": 32,
        "<CAPTION>": 32,
        "<DETAILED_CAPTION>": 128,
        "<MORE_DETAILED_CAPTION>": 256,
        "<DENSE_REGION_CAPTION>": 128,
        "<CAPTION_TO_PHRASE_GROUNDING>": 64,
    }
    return int(table.get(task_prompt, 128))


__all__ = ["Florence2MLXWrapper"]
