"""
Relations module: Scene graph relation prediction.

Exports:
- Pix2SGWrapper: Spatial scaffold + Florence2 semantic enrichment
- RelationsPipeline: Orchestrates relation prediction
"""

from .pipeline import RelationsPipeline
from .pix2sg import Pix2SGWrapper

__all__ = [
    "Pix2SGWrapper",
    "RelationsPipeline",
]
