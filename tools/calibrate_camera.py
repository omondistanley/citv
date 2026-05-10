"""
tools/calibrate_camera.py
Fix 5.2: OpenCV camera calibration using a printed checkerboard pattern.

Produces a calibration.json compatible with PreprocessConfig.camera_calibration_file.

Usage:
    python tools/calibrate_camera.py \
        --images path/to/checkerboard_frames/ \
        --pattern 9x6 \
        --square_size 0.025 \
        --out calibration.json

Arguments:
    --images       Directory of JPG/PNG images of the checkerboard, OR a single video.
    --pattern      Inner corners of the checkerboard (columns x rows), e.g. "9x6".
                   A 10x7 printed grid has 9x6 INNER corners.
    --square_size  Physical size of one square in METRES (e.g. 0.025 for 25mm).
                   Only affects the scale of translation vectors; fx/fy are independent.
    --out          Output JSON path (default: calibration.json).
    --visualize    Show corner detection for each image (useful for debugging).

What is a checkerboard calibration?
    OpenCV's cv2.calibrateCamera() solves for the camera matrix K and distortion
    coefficients D by finding the same set of 3D points (checkerboard corners at
    known spacings) in multiple 2D images from different angles. With 20+ images:
      - fx, fy accurate to < 0.5%
      - cx, cy accurate to < 2 pixels
      - k1, k2 (radial distortion), p1, p2 (tangential distortion) accurate enough
        to reduce residual reprojection error to < 0.5 pixels.

    For reference, a 60-degree FOV estimate gives fx error of ~10-30% depending
    on the actual lens — calibration eliminates this error completely.

How to print the checkerboard:
    1. Download any standard OpenCV calibration checkerboard PDF (search
       "opencv calibration checkerboard 9x6").
    2. Print at 100% scale (no "fit to page") on A4 or Letter paper.
    3. Measure the printed square size with a ruler → use that as --square_size.
    4. Tape the printout to a rigid flat surface (clipboard, hardcover book).

How to take calibration images:
    - Shoot 20-50 images of the checkerboard from different angles and positions.
    - Cover: straight-on, tilted left/right/up/down, at near/far distances.
    - Include corners of the image frame in some shots (distortion is strongest there).
    - Keep the checkerboard fully visible in every image.
    - Use the same zoom level and focus distance as your actual pipeline images.
    - Do NOT change focal length between calibration and use.

Output JSON structure:
    {
        "fx": 1234.5,      # focal length x in pixels
        "fy": 1236.1,      # focal length y in pixels
        "cx": 640.3,       # principal point x
        "cy": 360.1,       # principal point y
        "k1": -0.12,       # radial distortion coefficient 1
        "k2": 0.08,        # radial distortion coefficient 2
        "p1": 0.001,       # tangential distortion coefficient 1
        "p2": -0.002,      # tangential distortion coefficient 2
        "image_size": [1280, 720],  # [width, height] of calibration images
        "rms_reprojection_error": 0.42,  # lower is better; < 1.0 is acceptable
        "num_images_used": 23,
        "square_size_m": 0.025,
        "pattern": "9x6"
    }

Using the output:
    In config.py:
        camera_calibration_file = "calibration.json"
        apply_undistortion = True
    The pipeline will automatically load these values and apply cv2.undistort()
    before any depth or mask processing.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV camera calibration from checkerboard images.")
    p.add_argument("--images", required=True,
                   help="Directory of calibration images or path to a video file.")
    p.add_argument("--pattern", default="9x6",
                   help="Inner corners: 'cols x rows', e.g. '9x6'. Default: 9x6.")
    p.add_argument("--square_size", type=float, default=0.025,
                   help="Physical size of one square in metres. Default: 0.025 (25mm).")
    p.add_argument("--out", default="calibration.json",
                   help="Output JSON path. Default: calibration.json.")
    p.add_argument("--visualize", action="store_true",
                   help="Show corner detection for each image.")
    return p.parse_args()


def collect_frames(images_path: str):
    """Return list of BGR frames from a directory or video."""
    p = Path(images_path)
    frames = []
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if not paths:
            print(f"No images found in {p}")
            sys.exit(1)
        for fp in paths:
            img = cv2.imread(str(fp))
            if img is not None:
                frames.append(img)
        print(f"Loaded {len(frames)} images from {p}")
    elif p.is_file():
        cap = cv2.VideoCapture(str(p))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Sample every 10th frame from video to avoid redundancy
            if frame_idx % 10 == 0:
                frames.append(frame)
            frame_idx += 1
        cap.release()
        print(f"Sampled {len(frames)} frames from video {p}")
    else:
        print(f"Path not found: {p}")
        sys.exit(1)
    return frames


def calibrate(frames, pattern_size, square_size_m, visualize=False):
    """
    Run OpenCV checkerboard calibration.

    Returns (camera_matrix, dist_coeffs, rms_error, image_size) or raises.

    cv2.findChessboardCorners finds the inner corner grid in each image.
    cv2.cornerSubPix refines to sub-pixel accuracy.
    cv2.calibrateCamera solves for K and D using all detected corner sets.
    """
    cols, rows = pattern_size
    # 3D object points: z=0 plane, corners at (col*sq, row*sq, 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points = []  # 3D points in world space
    img_points = []  # 2D points in image space
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    n_found = 0
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            # Sub-pixel refinement for higher accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            n_found += 1

            if visualize:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, (cols, rows), corners_refined, ret)
                cv2.imshow(f"Corners [{i+1}/{len(frames)}]", vis)
                key = cv2.waitKey(300)
                if key == 27:  # ESC to stop visualization
                    visualize = False
        else:
            print(f"  Frame {i+1}: corners NOT found (check lighting, angle, full visibility)")

    if visualize:
        cv2.destroyAllWindows()

    print(f"\nCorners found in {n_found}/{len(frames)} frames.")
    if n_found < 10:
        print("WARNING: Fewer than 10 frames with corners detected. "
              "Calibration will be unreliable. Capture more images.")
    if n_found == 0:
        raise RuntimeError(
            "No checkerboard corners found in any frame. "
            "Check: correct --pattern value, good lighting, fully visible board."
        )

    print("Running cv2.calibrateCamera()...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return camera_matrix, dist_coeffs, rms, image_size, n_found


def main():
    args = parse_args()

    # Parse pattern string "9x6" → (9, 6)
    try:
        cols_str, rows_str = args.pattern.lower().split("x")
        pattern_size = (int(cols_str), int(rows_str))
    except Exception:
        print(f"Invalid --pattern '{args.pattern}'. Expected format: 'cols x rows', e.g. '9x6'.")
        sys.exit(1)

    frames = collect_frames(args.images)

    try:
        K, D, rms, image_size, n_used = calibrate(
            frames, pattern_size, args.square_size, args.visualize
        )
    except RuntimeError as e:
        print(f"Calibration failed: {e}")
        sys.exit(1)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    k1 = float(D[0, 0])
    k2 = float(D[0, 1])
    p1 = float(D[0, 2])
    p2 = float(D[0, 3])

    result = {
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "k1": round(k1, 6),
        "k2": round(k2, 6),
        "p1": round(p1, 6),
        "p2": round(p2, 6),
        "image_size": list(image_size),
        "rms_reprojection_error": round(float(rms), 4),
        "num_images_used": n_used,
        "square_size_m": args.square_size,
        "pattern": args.pattern,
    }

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Calibration Results ===")
    print(f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"k1={k1:.5f}  k2={k2:.5f}  p1={p1:.5f}  p2={p2:.5f}")
    print(f"RMS reprojection error: {rms:.4f} px  (< 0.5 = excellent, < 1.0 = good)")
    print(f"Saved to: {out_path.resolve()}")
    print()
    print("Next steps:")
    print(f"  1. Set in config.py:  camera_calibration_file = '{out_path}'")
    print( "  2. Set:              apply_undistortion = True")
    print( "  3. Set:              depth_scale_factor = 1.0  (metric model)")
    if rms > 1.0:
        print()
        print("WARNING: RMS error > 1.0 px. Consider:")
        print("  - Adding more calibration images (aim for 30+)")
        print("  - Ensuring the board is fully visible in all images")
        print("  - Checking that printed squares are truly square")


if __name__ == "__main__":
    main()
