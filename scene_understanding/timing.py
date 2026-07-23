"""Shared sub-step timing helper for the staged pipeline chain.

``pipeline.py``'s ``_run_stage_chain`` already times and prints each
top-level stage (``[Timing] stage: Xs``). This gives the stages that do
substantial internal work (repeated model calls, per-pair FMM solves) a
matching, indented ``[Timing]   stage.substep: Xs`` line so a real run's
console output shows where time goes inside a stage, not just across
stages.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def sub_timer(label: str) -> Iterator[None]:
    t0 = time.time()
    yield
    print(f"    [Timing]   {label}: {time.time() - t0:.2f}s")
