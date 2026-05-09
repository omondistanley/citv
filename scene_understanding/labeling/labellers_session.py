"""Explicit Florence-2 / RAM++ VRAM lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


@contextmanager
def labellers_session(pipeline: Any) -> Generator[None, None, None]:
    """Load labellers, yield, then unload (matches `_load_labellers` / `_unload_labellers`)."""
    pipeline._load_labellers()
    try:
        yield
    finally:
        pipeline._unload_labellers()
