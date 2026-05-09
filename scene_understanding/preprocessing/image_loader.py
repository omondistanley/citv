"""Image loading utilities with multi-format support."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_bgr_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image as BGR numpy array.
    Uses OpenCV first, then PIL (with optional HEIF opener) as fallback.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as HxWx3 BGR numpy array (uint8)
        
    Raises:
        ValueError: If image cannot be decoded
    """
    path = Path(path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is not None:
        return img_bgr

    pil_error = None
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass

        from PIL import Image
        with Image.open(path) as img_pil:
            img_rgb = np.array(img_pil.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        pil_error = e

    raise ValueError(
        f"Could not decode image: {path}. "
        "OpenCV returned None. If this is HEIF/HEIC content, convert it to JPEG/PNG "
        "or install pillow-heif for PIL decoding. "
        f"PIL fallback error: {pil_error}"
    )
