"""Local backend for the CITV Motion Composer UI.

This dependency-light server serves the static UI and exposes a JSON grounding
endpoint that calls the Python scene adapter:

    POST /api/motion/ground

Run from the repository root:

    python ui/motion-composer/server.py --host 127.0.0.1 --port 8088

Then open:

    http://127.0.0.1:8088/ui/motion-composer/
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from scene_understanding.action_contracts import (
    adapt_motion_contract_to_scene,
    motion_contract_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BODY_BYTES = 50 * 1024 * 1024


class MotionComposerHandler(SimpleHTTPRequestHandler):
    """Static-file handler plus JSON API routes."""

    server_version = "CITVMotionComposer/0.1"

    def __init__(self, *args: Any, directory: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        if self.path == "/api/motion/ground":
            self._handle_ground_motion()
            return
        self._send_json({"error": f"Unknown endpoint: {self.path}"}, status=404)

    def _handle_ground_motion(self) -> None:
        try:
            payload = self._read_json_body()
            contract_payload = payload.get("motion_contract") or payload.get("contract") or payload
            scene_graph = payload.get("scene_graph") or payload.get("scene") or None
            depth = _load_optional_array(payload.get("metric_depth_m"), payload.get("metric_depth_path"))
            region_map = _load_optional_array(payload.get("region_label_map"), payload.get("region_label_map_path"))
            object_masks = _load_object_masks(payload.get("object_masks"), payload.get("object_mask_paths"))
            sample_count = int(payload.get("sample_count", 48))
            contract = motion_contract_from_json(contract_payload)
            grounded = adapt_motion_contract_to_scene(
                contract,
                scene_graph=scene_graph,
                metric_depth_m=depth,
                region_label_map=region_map,
                object_masks=object_masks,
                sample_count=sample_count,
            )
            result = grounded.to_json()
            result.setdefault("report", {}).setdefault("adapted", []).append("grounded by local Python API")
            self._send_json(result)
        except Exception as exc:  # pragma: no cover - returned to browser for local debugging
            self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"Request body too large: {length} bytes")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:  # noqa: D401 - inherited method
        guessed = super().guess_type(path)
        if guessed == "application/octet-stream":
            return mimetypes.guess_type(path)[0] or guessed
        return guessed


def _load_optional_array(value: Any, path_value: Optional[str]) -> Optional[np.ndarray]:
    if value is not None:
        return np.asarray(value)
    if not path_value:
        return None
    path = _safe_repo_path(path_value)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        first_key = next(iter(data.files))
        return data[first_key]
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text()))
    raise ValueError(f"Unsupported array path: {path}")


def _load_object_masks(mask_values: Any, mask_paths: Any) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    if isinstance(mask_values, dict):
        for key, value in mask_values.items():
            masks[str(key)] = np.asarray(value).astype(bool)
    if isinstance(mask_paths, dict):
        for key, path_value in mask_paths.items():
            arr = _load_optional_array(None, str(path_value))
            if arr is not None:
                masks[str(key)] = np.asarray(arr).astype(bool)
    return masks


def _safe_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if os.path.commonpath([str(REPO_ROOT), str(resolved)]) != str(REPO_ROOT):
        raise ValueError(f"Path escapes repository root: {path_value}")
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the CITV Motion Composer UI and local grounding API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), MotionComposerHandler)
    print(f"Serving CITV Motion Composer at http://{args.host}:{args.port}/ui/motion-composer/")
    print(f"Grounding API: http://{args.host}:{args.port}/api/motion/ground")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
