"""
Labeling module: Open-vocabulary object labeling via Florence2 and RAM++.

Exports:
- Florence2Wrapper: Rich caption-based labeling and spatial relation prediction
- RAMPlusPlusWrapper: Tag-based open-vocabulary labeling (fallback)
- LabelingPipeline: Orchestrates both approaches
"""

from .florence2 import Florence2Wrapper
from .florence2_mlx import Florence2MLXWrapper
from .labellers_session import labellers_session
from .pipeline import LabelingPipeline
from .ram_plus_plus import RAMPlusPlusWrapper

__all__ = [
    "Florence2Wrapper",
    "Florence2MLXWrapper",
    "RAMPlusPlusWrapper",
    "LabelingPipeline",
    "labellers_session",
]
