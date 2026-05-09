"""Per-image / per-stage wall-clock timing instrumentation.

Provides:

- ``StageTimer``: a context manager that records one (image, stage) timing
  with MPS synchronization so the wall-clock captures GPU-dispatched work
  correctly, and optionally empties the MPS cache on exit.

- ``RunTimingLogger``: accumulates rows across all images in a folder run
  and emits both ``logs/timing_log.md`` (append-only, human-readable) and
  ``logs/timing_log.csv`` (machine-readable). Each image's timings are also
  returned as a dict ready to embed under the ``_timing`` key of the scene
  JSON.

The logger is deliberately dependency-light (stdlib + torch optional) so it
can be imported from any stage without dragging heavy CV deps.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - torch is present in this repo
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _short_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def _p95(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    idx = max(0, min(len(xs) - 1, int(round(0.95 * (len(xs) - 1)))))
    return xs[idx]


def _is_mps_device(device: str | None) -> bool:
    if not device:
        return False
    if torch is None:  # pragma: no cover
        return False
    try:
        return torch.device(device).type == "mps" and torch.backends.mps.is_available()
    except Exception:
        return False


def _mps_sync_and_empty(do_empty_cache: bool) -> None:
    if torch is None:
        return
    try:
        if not torch.backends.mps.is_available():
            return
    except Exception:
        return
    try:
        torch.mps.synchronize()  # type: ignore[attr-defined]
    except Exception:
        pass
    if do_empty_cache:
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------


@dataclass
class _TimingRecord:
    run_id: str
    image_key: str
    stage: str
    wall_ms: float
    ts_iso: str


class StageTimer:
    """Context manager for one (image, stage) wall-clock span.

    On ``__exit__`` (assuming no exception), records the elapsed ms into the
    owning :class:`RunTimingLogger`. When the logger's device is MPS, the
    exit path calls ``torch.mps.synchronize()`` before stopping the clock so
    GPU work is actually waited on (and optionally frees the MPS cache to
    flatten peak memory between stages).
    """

    __slots__ = (
        "_logger",
        "_image_key",
        "_stage",
        "_t0",
        "_mps_sync",
        "_empty_cache",
        "_recorded",
    )

    def __init__(
        self,
        logger: "RunTimingLogger",
        image_key: str,
        stage: str,
        *,
        mps_sync: bool = True,
        empty_cache: bool = False,
    ) -> None:
        self._logger = logger
        self._image_key = image_key
        self._stage = stage
        self._t0 = 0.0
        self._mps_sync = bool(mps_sync) and logger.device_is_mps
        self._empty_cache = bool(empty_cache) and logger.device_is_mps
        self._recorded = False

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        if self._recorded:
            return False
        if self._mps_sync:
            _mps_sync_and_empty(self._empty_cache)
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        self._logger.record(self._image_key, self._stage, wall_ms)
        self._recorded = True
        return False  # never swallow exceptions


# ---------------------------------------------------------------------------
# RunTimingLogger
# ---------------------------------------------------------------------------


class RunTimingLogger:
    """Accumulate timing rows for a folder run and emit .md + .csv logs.

    Usage pattern::

        logger = RunTimingLogger(
            log_dir="logs",
            device="mps",
            models_dtype="fp32",
            workers=1,
            config_hash=hash_config(config),
        )
        for image_path in images:
            with logger.stage(image_path, "depth", empty_cache=True):
                ...
            scene_json["_timing"] = logger.image_timings(image_path)
        logger.finalize()
    """

    _MD_NAME = "timing_log.md"
    _CSV_NAME = "timing_log.csv"

    def __init__(
        self,
        log_dir: str | os.PathLike,
        *,
        device: Optional[str] = None,
        models_dtype: str = "fp32",
        workers: int = 1,
        config_hash: Optional[str] = None,
        run_id: Optional[str] = None,
        git_sha: Optional[str] = None,
        extra_header_notes: Optional[List[str]] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or "cpu"
        self.device_is_mps = _is_mps_device(self.device)
        self.models_dtype = models_dtype
        self.workers = int(workers)
        self.config_hash = config_hash or "unknown"
        self.run_id = run_id or _now_iso()
        self.git_sha = git_sha or _short_git_sha()
        self.extra_header_notes = list(extra_header_notes or [])
        self._lock = threading.Lock()
        self._records: List[_TimingRecord] = []
        self._md_path = self.log_dir / self._MD_NAME
        self._csv_path = self.log_dir / self._CSV_NAME
        self._finalized = False

        if self.device_is_mps and torch is not None:
            try:
                rec_max = float(torch.mps.recommended_max_memory())  # type: ignore[attr-defined]
                cur_alloc = float(torch.mps.current_allocated_memory())  # type: ignore[attr-defined]
                pct = 100.0 * cur_alloc / rec_max if rec_max > 0 else 0.0
                self.extra_header_notes.append(
                    f"MPS memory at run start: {cur_alloc / (1024**3):.2f} GB "
                    f"of {rec_max / (1024**3):.2f} GB recommended ({pct:.1f}%)"
                )
                if pct > 80.0:
                    self.extra_header_notes.append(
                        "WARNING: MPS memory pressure >80% of recommended "
                        "at run start; fp16 / fp32 QA deltas may be explained "
                        "by this headroom shortfall."
                    )
            except Exception:
                pass

    # -------------------------- recording ----------------------------------

    def stage(
        self,
        image_key: str,
        stage: str,
        *,
        mps_sync: bool = True,
        empty_cache: bool = False,
    ) -> StageTimer:
        return StageTimer(
            self,
            _image_key_to_str(image_key),
            stage,
            mps_sync=mps_sync,
            empty_cache=empty_cache,
        )

    def record(self, image_key: str, stage: str, wall_ms: float) -> None:
        with self._lock:
            self._records.append(
                _TimingRecord(
                    run_id=self.run_id,
                    image_key=_image_key_to_str(image_key),
                    stage=str(stage),
                    wall_ms=float(wall_ms),
                    ts_iso=_now_iso(),
                )
            )

    # -------------------------- queries ------------------------------------

    def image_timings(self, image_key: str) -> Dict[str, float]:
        """Return ``{<stage>_ms: float, ...}`` for one image (plus total_ms)."""

        key = _image_key_to_str(image_key)
        with self._lock:
            rows = [r for r in self._records if r.image_key == key]
        out: Dict[str, float] = {}
        total = 0.0
        for r in rows:
            field_name = f"{r.stage}_ms"
            out[field_name] = round(out.get(field_name, 0.0) + r.wall_ms, 3)
            total += r.wall_ms
        out["total_ms"] = round(total, 3)
        return out

    def all_images(self) -> List[str]:
        with self._lock:
            return sorted({r.image_key for r in self._records})

    def all_stages(self) -> List[str]:
        with self._lock:
            return sorted({r.stage for r in self._records})

    # -------------------------- finalize -----------------------------------

    def finalize(self) -> Tuple[Path, Path]:
        if self._finalized:
            return self._md_path, self._csv_path
        self._append_md_section()
        self._append_csv_rows()
        self._finalized = True
        return self._md_path, self._csv_path

    def _append_md_section(self) -> None:
        with self._lock:
            images = sorted({r.image_key for r in self._records})
            stages = sorted({r.stage for r in self._records})
            per_image: Dict[str, Dict[str, float]] = {img: {} for img in images}
            for r in self._records:
                per_image[r.image_key][r.stage] = per_image[r.image_key].get(r.stage, 0.0) + r.wall_ms

        lines: List[str] = []
        header_exists = self._md_path.exists() and self._md_path.stat().st_size > 0
        if not header_exists:
            lines.append("# CITV Pipeline Timing Log")
            lines.append("")
            lines.append(
                "Every folder run appends one new section below. Each row is one image; "
                "the final block lists mean / median / p95 per stage across all images "
                "in the run."
            )
            lines.append("")

        lines.append(
            f"## Run {self.run_id} git={self.git_sha} "
            f"config_hash={self.config_hash} images={len(images)}"
        )
        lines.append(
            f"Device: {self.device}  Models dtype: {self.models_dtype}  "
            f"Workers: {self.workers}"
        )
        for note in self.extra_header_notes:
            lines.append(f"- {note}")
        lines.append("")

        if stages and images:
            header = ["image"] + [f"{s}_ms" for s in stages] + ["total_ms"]
            sep = ["---"] + ["---:"] * (len(header) - 1)
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(sep) + " |")
            for img in images:
                row_vals = per_image.get(img, {})
                total_ms = sum(row_vals.values())
                row = [img]
                for s in stages:
                    row.append(f"{row_vals.get(s, 0.0):.1f}")
                row.append(f"{total_ms:.1f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            lines.append("Stage aggregates (ms): mean / median / p95")
            for s in stages:
                per_image_stage = [per_image[img].get(s, 0.0) for img in images]
                lines.append(
                    f"- {s}: {mean(per_image_stage):.1f} / "
                    f"{median(per_image_stage):.1f} / "
                    f"{_p95(per_image_stage):.1f}"
                )
            lines.append("")

        with self._md_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            fh.write("\n")

    def _append_csv_rows(self) -> None:
        write_header = not (self._csv_path.exists() and self._csv_path.stat().st_size > 0)
        with self._csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(
                    [
                        "run_id",
                        "ts_iso",
                        "image",
                        "stage",
                        "wall_ms",
                        "device",
                        "models_dtype",
                        "workers",
                        "config_hash",
                        "git_sha",
                    ]
                )
            with self._lock:
                rows = list(self._records)
            for r in rows:
                writer.writerow(
                    [
                        r.run_id,
                        r.ts_iso,
                        r.image_key,
                        r.stage,
                        f"{r.wall_ms:.3f}",
                        self.device,
                        self.models_dtype,
                        self.workers,
                        self.config_hash,
                        self.git_sha,
                    ]
                )
            with self._lock:
                image_totals: Dict[str, float] = {}
                for r in self._records:
                    image_totals[r.image_key] = image_totals.get(r.image_key, 0.0) + r.wall_ms
            for image_key, total in image_totals.items():
                writer.writerow(
                    [
                        self.run_id,
                        _now_iso(),
                        image_key,
                        "_total",
                        f"{total:.3f}",
                        self.device,
                        self.models_dtype,
                        self.workers,
                        self.config_hash,
                        self.git_sha,
                    ]
                )


# ---------------------------------------------------------------------------
# No-op fallback so callers can always `with logger.stage(...):` safely even
# when timing is disabled. Kept API-compatible with RunTimingLogger.
# ---------------------------------------------------------------------------


class NullTimingLogger:
    """Drop-in no-op replacement when timing instrumentation is disabled."""

    device_is_mps = False

    def stage(self, image_key: str, stage: str, **_: object) -> "NullTimingLogger":
        return self

    def __enter__(self) -> "NullTimingLogger":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def record(self, *_: object, **__: object) -> None:
        return None

    def image_timings(self, _image_key: str) -> Dict[str, float]:
        return {}

    def all_images(self) -> List[str]:
        return []

    def all_stages(self) -> List[str]:
        return []

    def finalize(self) -> Tuple[Optional[Path], Optional[Path]]:
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_key_to_str(image_key: object) -> str:
    if isinstance(image_key, Path):
        return image_key.name
    if isinstance(image_key, str):
        return Path(image_key).name if os.sep in image_key or "/" in image_key else image_key
    return str(image_key)


def hash_config(config: object) -> str:
    """Stable hash of a dataclass-like config for the timing log header."""

    try:
        if hasattr(config, "__dict__"):
            payload = {k: _safe_coerce(v) for k, v in vars(config).items()}
        elif isinstance(config, dict):
            payload = {k: _safe_coerce(v) for k, v in config.items()}
        else:
            payload = {"repr": repr(config)}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(config).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def _safe_coerce(value: object) -> object:
    try:
        json.dumps(value, default=str)
    except Exception:
        return str(value)
    return value


__all__ = [
    "StageTimer",
    "RunTimingLogger",
    "NullTimingLogger",
    "hash_config",
]
