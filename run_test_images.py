"""Benchmark runner: process images_test/ and log per-image per-stage timing.

Usage (from citv/):
    venv/bin/python3 run_test_images.py

Writes timing output to logs/timing_log.md and logs/timing_log.csv.
Prints a per-image wall-time summary and flags any image over 100 s.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PreprocessConfig  # noqa: E402

# Enable per-stage timing
cfg = PreprocessConfig()
cfg.scene_pipeline_profile = True  # type: ignore[assignment]

_pipe_mod = importlib.import_module("scene_understanding.pipeline")
SceneUnderstandingPipeline = _pipe_mod.SceneUnderstandingPipeline

DepthEstimator = importlib.import_module("depth").DepthEstimator

from scene_understanding.observability import RunTimingLogger, hash_config  # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────────
IMAGES_DIR = ROOT / "images_test"
OUTPUT_DIR = ROOT / "output_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
image_paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
if not image_paths:
    print(f"No images found in {IMAGES_DIR}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"CITV benchmark: {len(image_paths)} images in {IMAGES_DIR.name}/")
print(f"Output  : {OUTPUT_DIR}")
print(f"Device  : {cfg.device}  dtype: {cfg.models_dtype}")
print(f"{'='*60}\n")

# ── timing logger ────────────────────────────────────────────────────────────
timing_log_dir = ROOT / "logs"
run_logger = RunTimingLogger(
    log_dir=timing_log_dir,
    device=str(cfg.device),
    models_dtype=str(cfg.models_dtype),
    workers=1,
    config_hash=hash_config(cfg),
)

# ── pipeline init ─────────────────────────────────────────────────────────────
t_init = time.perf_counter()
depth_estimator = DepthEstimator(cfg, first_image=image_paths[0])
pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)
pipeline.attach_timing_logger(run_logger)
print(f"Pipeline initialised in {time.perf_counter() - t_init:.1f}s\n")

# ── process each image ────────────────────────────────────────────────────────
TARGET_S = 100.0
summary: list[dict] = []

for idx, img_path in enumerate(image_paths, 1):
    print(f"[{idx}/{len(image_paths)}] {img_path.name}")
    pipeline._has_more_images_queued = idx < len(image_paths)
    t0 = time.perf_counter()
    try:
        pipeline.process_image(str(img_path), str(OUTPUT_DIR))
        elapsed = time.perf_counter() - t0
        status = "OK" if elapsed < TARGET_S else "SLOW"
        flag = "  ✓" if elapsed < TARGET_S else f"  ✗ OVER {TARGET_S:.0f}s"
        print(f"    total: {elapsed:.1f}s{flag}\n")
        summary.append({"name": img_path.name, "elapsed": elapsed, "ok": elapsed < TARGET_S})
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        import traceback
        print(f"    ERROR after {elapsed:.1f}s: {exc}")
        traceback.print_exc()
        summary.append({"name": img_path.name, "elapsed": elapsed, "ok": False, "error": str(exc)})

pipeline._has_more_images_queued = False

# ── finalise timing log ───────────────────────────────────────────────────────
try:
    md_path, csv_path = run_logger.finalize()
    print(f"\nTiming log  : {md_path}")
    print(f"Timing CSV  : {csv_path}")
except Exception as te:
    print(f"\n[Timing] finalize failed: {te}")

# ── print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'IMAGE':<30} {'TIME (s)':>10}  {'STATUS'}")
print(f"{'-'*60}")
for row in summary:
    status = "OK" if row["ok"] else ("SLOW" if "error" not in row else "ERROR")
    print(f"{row['name']:<30} {row['elapsed']:>10.1f}  {status}")

total = sum(r["elapsed"] for r in summary)
n_ok = sum(1 for r in summary if r["ok"])
print(f"{'-'*60}")
print(f"{'TOTAL':<30} {total:>10.1f}  {n_ok}/{len(summary)} under {TARGET_S:.0f}s")
print(f"{'='*60}\n")

# Print existing timing log tail for per-stage breakdown
try:
    lines = Path(md_path).read_text().split("\n")
    # find the last run section
    last_run_start = max((i for i, l in enumerate(lines) if l.startswith("## Run")), default=None)
    if last_run_start is not None:
        print("Per-stage breakdown (from timing_log.md):")
        print("\n".join(lines[last_run_start:last_run_start + 120]))
except Exception:
    pass
