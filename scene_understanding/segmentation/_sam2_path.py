"""Ensure the sam2 package (citv/sam2/sam2/) is importable from any cwd.

When the user runs Python from citv/, the directory citv/sam2/ (the cloned
sam2 repo) is on sys.path and shadows the real package at citv/sam2/sam2/.
SAM2's own build_sam.py detects this and raises RuntimeError.

The fix: insert citv/sam2/ (the repo root) at the front of sys.path BEFORE
any sam2 import.  Because citv/sam2/ has no __init__.py, Python will look
inside it and find citv/sam2/sam2/__init__.py as the `sam2` package, which
is exactly what we want.

Call ensure_sam2_importable() once before any `from sam2.xxx import yyy`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# citv/scene_understanding/segmentation/ → citv/
_CITV_ROOT = Path(__file__).resolve().parents[2]
_SAM2_REPO_ROOT = _CITV_ROOT / "sam2"


def ensure_sam2_importable() -> None:
    """Prepend citv/sam2/ to sys.path so sam2 resolves to the inner package."""
    repo = _SAM2_REPO_ROOT
    if not repo.is_dir():
        return
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    # If an incorrect sam2 module was already cached (resolved to the repo
    # directory rather than the inner package), evict it so the next import
    # picks up the corrected path.
    stale = sam2_mod = sys.modules.get("sam2")
    if stale is not None:
        pkg_path = getattr(stale, "__path__", None)
        if pkg_path:
            inner = Path(list(pkg_path)[0])
            # Wrong if the cached package dir is citv/sam2/ (not citv/sam2/sam2/)
            if inner == repo:
                for key in list(sys.modules.keys()):
                    if key == "sam2" or key.startswith("sam2."):
                        del sys.modules[key]
