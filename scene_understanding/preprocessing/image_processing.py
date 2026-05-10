"""Image processing utilities: resizing, color conversion, etc."""

from typing import Tuple, List
import cv2
import numpy as np


def resize_image_if_needed(
    img_bgr: np.ndarray,
    img_rgb: np.ndarray,
    max_side: int = 1280,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Downscale large images to keep RAM usage bounded.
    
    Args:
        img_bgr: Image in BGR format
        img_rgb: Image in RGB format (for consistency)
        max_side: Maximum side length (will downscale if larger)
        
    Returns:
        Tuple of (resized_bgr, resized_rgb, scale_factor, new_size)
        where scale_factor is new_size / original_size
        and new_size is (new_w, new_h)
    """
    h, w = img_bgr.shape[:2]
    
    if max_side <= 0 or max(h, w) <= max_side:
        return img_bgr, img_rgb, 1.0, (w, h)
    
    scale = max_side / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_rgb_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print(f"  Resized to {new_w}x{new_h} (max_side={max_side})")
    
    return img_bgr_resized, img_rgb_resized, scale, (new_w, new_h)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rescale_bbox(
    bbox: List[float],
    scale: float,
    format: str = "xywh",
) -> List[float]:
    """
    Rescale bounding box by a scale factor.
    
    Args:
        bbox: Bounding box coordinates
        scale: Scale factor (e.g., 0.5 for half-size)
        format: "xywh" or "xyxy"
        
    Returns:
        Scaled bounding box in same format
    """
    if format == "xywh":
        x, y, w, h = bbox[:4]
        return [x * scale, y * scale, w * scale, h * scale]
    elif format == "xyxy":
        x1, y1, x2, y2 = bbox[:4]
        return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    else:
        raise ValueError(f"Unknown format: {format}")
