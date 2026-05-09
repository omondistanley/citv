"""Utility functions used across the pipeline."""

from typing import List
import numpy as np


def xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """
    Convert bounding box from xywh to xyxy format.
    
    Args:
        bbox_xywh: [x, y, width, height]
        
    Returns:
        [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
    """
    x, y, w, h = bbox_xywh[:4]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    """
    Convert bounding box from xyxy to xywh format.
    
    Args:
        bbox_xyxy: [x1, y1, x2, y2]
        
    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return [x1, y1, x2 - x1, y2 - y1]


def iou_xyxy(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two boxes in xyxy format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    
    return inter / (union + 1e-8)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.
    
    Args:
        mask1: Binary mask (HxW bool or uint8)
        mask2: Binary mask (HxW bool or uint8)
        
    Returns:
        IoU score between 0 and 1
    """
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)
    
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 0.0
    return inter / union
