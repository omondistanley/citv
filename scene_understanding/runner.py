"""Package-owned runner for full scene-understanding folder execution.

This module is a parity-preserving extraction of the monolithic script's
`if __name__ == "__main__"` flow so command execution can live under the
`scene_understanding/` package while outputs remain unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from config import PreprocessConfig
from depth import DepthEstimator
from scene_understanding.pipeline import SceneUnderstandingPipeline


def _pipeline_worker_entrypoint(worker_args: tuple) -> list:
    """Spawn-safe worker for staged-only folder chunk processing."""
    cfg, image_paths, output_dir = worker_args
    results = []
    try:
        if not image_paths:
            return results
        depth_estimator = DepthEstimator(cfg, first_image=image_paths[0])
        pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)
        for p in image_paths:
            try:
                img_out_dir = Path(output_dir) / Path(p).stem
                pipeline.process_image(str(p), str(img_out_dir))
                results.append((str(p), True, "ok"))
            except Exception as e:
                results.append((str(p), False, str(e)))
    except Exception as e:
        for p in image_paths or []:
            results.append((str(p), False, f"worker init failed: {e}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scene understanding pipeline (depth + Grounded SAM2).")
    parser.add_argument("--input_dir", type=str, default="images", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output_scene", help="Output directory for scene graphs")
    args = parser.parse_args()

    print("Initializing pipeline (depth + Grounded SAM2)...")
    cfg = PreprocessConfig()
    if bool(getattr(cfg, "cpu_benchmark_mode", False)):
        cfg.device = "cpu"  # type: ignore[assignment]
        cfg.models_dtype = "fp32"  # type: ignore[assignment]
        cfg.rampp_prefer_mps = False  # type: ignore[assignment]
        print("CPU benchmark mode enabled: forcing device=cpu, dtype=fp32, no MPS preference.")
    print("  Output: scene_graph/staged/{stem}_scene.json")
    images_dir = Path(args.input_dir)
    output_dir = args.output_dir
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

    try:
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir.resolve()}")

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts
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

        # Attach a per-run timing logger so logs/timing_log.md + .csv get
        # appended one section per folder run, and each scene JSON gains
        # its _timing block. Output-preserving.
        try:
            from scene_understanding.observability import RunTimingLogger, hash_config

            _timing_log_dir = Path(getattr(cfg, "timing_log_dir", "logs"))
            _run_logger = RunTimingLogger(
                log_dir=_timing_log_dir,
                device=str(getattr(cfg, "device", "cpu")),
                models_dtype=str(getattr(cfg, "models_dtype", "fp32")),
                workers=int(getattr(cfg, "pipeline_max_workers", 1)),
                config_hash=hash_config(cfg),
            )
            if hasattr(pipeline, "attach_timing_logger"):
                pipeline.attach_timing_logger(_run_logger)
        except Exception as _te:
            print(f"  [Timing] logger setup skipped: {_te}")
            _run_logger = None

        print(f"Found {len(image_paths)} images in {images_dir.resolve()}")

        # Cross-image parallelism via multiprocessing.Pool when
        # ``cfg.pipeline_max_workers > 1``.
        _max_workers = int(getattr(cfg, "pipeline_max_workers", 1) or 1)
        if _max_workers > 1 and len(image_paths) > 1:
            try:
                pipeline.close()
            except Exception:
                pass
            del pipeline
            del depth_estimator
            import gc as _gc

            _gc.collect()

            from multiprocessing import get_context

            ctx = get_context("spawn")
            chunk_count = min(_max_workers, len(image_paths))
            chunks: List[List[Path]] = [[] for _ in range(chunk_count)]
            for _i, _p in enumerate(image_paths):
                chunks[_i % chunk_count].append(_p)
            worker_args = [(cfg, chunk, output_dir) for chunk in chunks if chunk]
            print(f"  [Multiproc] dispatching {len(image_paths)} images across {len(worker_args)} workers")
            with ctx.Pool(processes=len(worker_args)) as pool:
                worker_results = pool.map(_pipeline_worker_entrypoint, worker_args)
            ok = 0
            fail = 0
            for batch in worker_results:
                for item in batch:
                    if item[0] == "__timing__":
                        print(f"  [Timing][worker] {item[2]}")
                        continue
                    if item[1]:
                        ok += 1
                    else:
                        fail += 1
                        print(f"  [Multiproc] {item[0]} failed: {item[2]}")
            print(f"  [Multiproc] done - ok={ok} fail={fail}")
        else:
            _n_imgs = len(image_paths)
            for _img_idx, img_path in enumerate(image_paths):
                pipeline._has_more_images_queued = _img_idx < _n_imgs - 1
                print(f"Processing image: {img_path.name} ({_img_idx + 1}/{_n_imgs})")
                try:
                    img_out_dir = Path(output_dir) / img_path.stem
                    pipeline.process_image(str(img_path), str(img_out_dir))
                except ValueError as e:
                    print(f"Skipping {img_path.name}: {e}")
            pipeline._has_more_images_queued = False

        if _max_workers <= 1 and _run_logger is not None:
            try:
                md_path, csv_path = _run_logger.finalize()
                print(f"  [Timing] wrote {md_path} and {csv_path}")
            except Exception as _te:
                print(f"  [Timing] finalize failed: {_te}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
