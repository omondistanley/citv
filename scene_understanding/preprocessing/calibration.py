"""Camera calibration and intrinsic parameter handling."""

from typing import Dict, Optional, Any
import numpy as np
import cv2


class CameraCalibration:
    """
    Manages camera intrinsic parameters and distortion coefficients.
    Supports calibration file loading, FOV-based estimation, and lens undistortion.
    """
    
    def __init__(
        self,
        calibration_dict: Optional[Dict[str, float]] = None,
        camera_fx: Optional[float] = None,
        camera_fy: Optional[float] = None,
        camera_cx: Optional[float] = None,
        camera_cy: Optional[float] = None,
        camera_fov_degrees: float = 71.0,
        apply_undistortion: bool = True,
    ):
        """
        Initialize camera calibration.
        
        Priority for intrinsics:
            1. calibration_dict (most accurate)
            2. explicit fx/fy/cx/cy parameters
            3. FOV-based estimate (least accurate)
        
        Args:
            calibration_dict: Dict with fx, fy, cx, cy, k1, k2, p1, p2 (from OpenCV calibration)
            camera_fx/fy/cx/cy: Explicit intrinsic parameters
            camera_fov_degrees: Horizontal FOV for fallback estimation
            apply_undistortion: Whether to apply lens distortion correction
        """
        self._calibration = calibration_dict
        self.camera_fx = camera_fx
        self.camera_fy = camera_fy
        self.camera_cx = camera_cx
        self.camera_cy = camera_cy
        self.camera_fov_degrees = float(camera_fov_degrees)
        self.apply_undistortion = bool(apply_undistortion)
    
    def get_intrinsics(self, width: int, height: int) -> Dict[str, float]:
        """
        Get camera intrinsics K with priority order:
          1. Calibration file (OpenCV checkerboard calibration — most accurate)
          2. Explicit camera parameters
          3. FOV-based estimate (least accurate; error can be 10-30%)

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Dict with keys: fx, fy, cx, cy
        """
        # Priority 1: calibration file
        if self._calibration is not None:
            cal = self._calibration
            return {
                "fx": float(cal["fx"]),
                "fy": float(cal["fy"]),
                "cx": float(cal.get("cx", width / 2)),
                "cy": float(cal.get("cy", height / 2)),
            }
        
        # Priority 2: explicit values
        if self.camera_fx is not None:
            return {
                "fx": float(self.camera_fx),
                "fy": float(self.camera_fy if self.camera_fy is not None else self.camera_fx),
                "cx": float(self.camera_cx if self.camera_cx is not None else width / 2),
                "cy": float(self.camera_cy if self.camera_cy is not None else height / 2),
            }
        
        # Priority 3: FOV estimate
        f_x = (width / 2) / np.tan(np.deg2rad(self.camera_fov_degrees) / 2)
        print(f"  [Intrinsics] Using FOV estimate ({self.camera_fov_degrees}°): fx=fy={f_x:.1f}")
        return {"fx": f_x, "fy": f_x, "cx": width / 2, "cy": height / 2}
    
    def undistort_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Apply lens distortion correction using calibration coefficients.
        Uses OpenCV cv2.undistort() with loaded k1,k2,p1,p2 coefficients.

        Returns image unchanged if no calibration is loaded or undistortion disabled.

        Args:
            img_bgr: Image as HxWx3 BGR array

        Returns:
            Undistorted image (or original if no calibration)
        """
        if self._calibration is None or not self.apply_undistortion:
            return img_bgr
        try:
            cal = self._calibration
            h, w = img_bgr.shape[:2]
            K_mat = np.array([
                [cal["fx"], 0.0,       cal["cx"]],
                [0.0,       cal["fy"], cal["cy"]],
                [0.0,       0.0,       1.0],
            ], dtype=np.float64)
            dist_coeffs = np.array([
                cal.get("k1", 0.0), cal.get("k2", 0.0),
                cal.get("p1", 0.0), cal.get("p2", 0.0),
            ], dtype=np.float64)
            return cv2.undistort(img_bgr, K_mat, dist_coeffs)
        except Exception as e:
            print(f"[Undistort] Failed: {e}. Using original image.")
            return img_bgr
    
    @staticmethod
    def back_project(u: int, v: int, z: float, K: Dict[str, float]) -> Dict[str, float]:
        """
        Back-project a pixel to 3D camera coordinates.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            z: Depth value (meters)
            K: Camera intrinsics dict with keys: fx, fy, cx, cy
            
        Returns:
            Dict with x, y, z in camera space (meters)
        """
        x = (u - K['cx']) * z / K['fx']
        y = (v - K['cy']) * z / K['fy']
        return {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "z": round(float(z), 3)
        }
