"""Numba-accelerated numerical kernels used by the pipeline hot paths.

All public functions have a NumPy fallback so the import succeeds even
when Numba is not installed.  Callers should never introspect whether
Numba is active; compile decisions are made once at import time based on
``os.environ.get("CITV_DISABLE_NUMBA")`` and the presence of ``numba``.
"""

from scene_understanding.numerics.kernels import (
    astar_shortest_path,
    dijkstra_shortest_path,
    masked_sigma_clip,
    numba_available,
    unproject_pixels_to_xyz,
)

__all__ = [
    "astar_shortest_path",
    "dijkstra_shortest_path",
    "masked_sigma_clip",
    "numba_available",
    "unproject_pixels_to_xyz",
]
