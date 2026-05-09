"""Async-style stage orchestration helpers (Phase G).

Goal
====
The legacy ``process_image`` flow runs depth estimation, the full-image
caption, and GDINO+SAM2 prompted segmentation strictly in sequence even
though they are read-only on the input image and produce independent
artefacts. On a single Apple Silicon device they cannot truly execute in
parallel (MPS has one queue), but a surprising fraction of each stage's
wall-clock is spent in Python glue code (tokenisation, image resizing,
BGR↔RGB conversion, HF post-processing, disk writes). Running them on
``asyncio.run_in_executor`` threads lets the device pipeline stay busy
while the other stages do their CPU work.

Design
======
* Pure infrastructure — this module does not touch the pipeline. It
  exposes :func:`run_stages_parallel` that takes a mapping of
  ``{stage_name: (callable, args, kwargs)}`` and returns a
  ``{stage_name: result}`` dict. Exceptions are re-raised after all
  stages have been awaited so partial failures don't leave resources
  leaking.
* Thread pool — we use ``concurrent.futures.ThreadPoolExecutor`` because
  all three stages release the GIL inside C extensions (PyTorch, OpenCV,
  HuggingFace) and releasing the GIL is what matters for overlap. A
  pool size of 3 covers the documented stages with one worker each.
* Disable with ``config.pipeline_async_stages_enabled = False`` (default
  off until QA sign-off) so the existing sequential behaviour is a hard
  contract.

The caller is responsible for ensuring the passed callables are
thread-safe. In practice the three supported stages are:
  * ``depth.DepthEstimator.estimate_metric`` — calls PyTorch forward on
    MPS. Safe because the model weights are read-only post-init and the
    ``torch.no_grad`` wrapper is re-entrant.
  * ``Florence2Wrapper.caption`` — calls HF ``generate``. Safe when run
    against a pre-loaded model, but only one such call should be
    inflight at a time per wrapper (enforced by a per-wrapper lock).
  * ``GroundedSAM2Wrapper.generate`` — GDINO + SAM2 prompted. Safe when
    ``_maybe_set_image`` caches the encoder embedding for this image.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from typing import Any, Callable, Dict, Mapping, Tuple


StageSpec = Tuple[Callable[..., Any], tuple, dict]


def run_stages_parallel(
    stages: Mapping[str, StageSpec],
    *,
    max_workers: int = 3,
) -> Dict[str, Any]:
    """Run ``stages`` concurrently on a small thread pool and collect results.

    Parameters
    ----------
    stages:
        Mapping of ``{name: (fn, args, kwargs)}``. ``name`` is an opaque
        identifier used in both the result dict and error messages.
    max_workers:
        Upper bound on concurrent worker threads. Defaults to 3 which
        matches the three documented pipeline stages (depth, caption,
        segmentation); larger values offer no speedup because all three
        stages funnel through the same MPS queue.

    Returns
    -------
    dict
        ``{name: result}`` for every stage. If any stage raised, the
        first exception is re-raised after every stage has finished so
        callers don't see half-produced state.
    """
    if not stages:
        return {}

    n = min(max_workers, max(1, len(stages)))
    futures_by_name: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="citv-stage") as pool:
        for name, (fn, args, kwargs) in stages.items():
            futures_by_name[name] = pool.submit(fn, *(args or ()), **(kwargs or {}))
        done, not_done = wait(
            list(futures_by_name.values()), return_when=FIRST_EXCEPTION
        )
        # Drain remaining futures so their exceptions surface cleanly and
        # their CPU work doesn't bleed past the parent's scope.
        for fut in not_done:
            try:
                fut.result()
            except Exception:
                pass

    results: Dict[str, Any] = {}
    first_exc: Exception | None = None
    for name, fut in futures_by_name.items():
        try:
            results[name] = fut.result()
        except Exception as exc:  # noqa: BLE001
            if first_exc is None:
                first_exc = exc
            results[name] = None
    if first_exc is not None:
        raise first_exc
    return results


__all__ = ["run_stages_parallel", "StageSpec"]
