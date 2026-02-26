"""
Depth estimation and segmentation module.
Provides depth estimation using Depth Anything V2 Large
and semantic segmentation using Mask2Former/DeepLabV3.
"""
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import torch
from PIL import Image
import cv2
from config import PreprocessConfig


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] range."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


def _resize_to_target(depth: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize depth map to target size (width, height)."""
    if depth.shape[:2] == (target_size[1], target_size[0]):
        return depth
    return cv2.resize(
        depth.astype(np.float32),
        target_size,
        interpolation=cv2.INTER_LINEAR,
    )


class DepthAnythingV2Backend:
    """Depth Anything V2 Large via HuggingFace transformers."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        # Force using the Large model
        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        
        print(f"Loading Depth Anything V2 Large: {model_id}")
        from transformers import pipeline
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device.type == "cuda" else -1,
        )

    def infer(self, img_rgb: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(img_rgb)
        out = self.pipe(pil_img)
        depth = out["depth"]
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        elif hasattr(depth, "numpy"):
            depth = depth.numpy()
        else:
            depth = np.array(depth)
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth = _resize_to_target(depth, self.config.target_size)
        return _normalize_depth(depth)


def _save_depth_details(
    depths: List[np.ndarray],
    out_dir: Path,
    model_prefix: str,
    target_size: Tuple[int, int],
) -> None:
    """Persist per-frame min, max, mean and metadata alongside depth maps."""
    details = {
        "model_prefix": model_prefix,
        "target_size": list(target_size),
        "num_frames": len(depths),
        "frames": [],
    }
    for i, d in enumerate(depths):
        d = np.asarray(d, dtype=np.float64)
        details["frames"].append({
            "frame": i,
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "mean": float(np.mean(d)),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "depth_details.json", "w") as f:
        json.dump(details, f, indent=2)


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2 Large."""

    def __init__(self, config: PreprocessConfig, model_name: Optional[str] = None):
        # model_name argument is kept for compatibility but ignored/logged
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        
        try:
            self.backend = DepthAnythingV2Backend(config)
            self.backend_name = "DepthAnythingV2-Large"
            print(f"Depth estimator initialized: {self.backend_name}")
        except Exception as e:
            print(f"Failed to load Depth Anything V2 Large: {e}")
            raise e

    def _save_depth_maps(
        self,
        depth: np.ndarray,
        out_dir: Path,
        idx_or_name: Union[int, str],
        prefix: str = "depth",
    ) -> None:
        """Save depth map as .npy and optionally PNG visualizations."""
        if isinstance(idx_or_name, int):
            stem = f"{prefix}_{idx_or_name:06d}"
        else:
            stem = f"{prefix}_{idx_or_name}"
            
        np.save(out_dir / f"{stem}.npy", depth.astype(np.float32))

        if getattr(self.config, "save_depth_visualizations", True):
            vis = (depth * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(out_dir / f"{stem}.png"), vis)

        if getattr(self.config, "save_depth_16bit", False):
            depth_16 = (depth * 65535).astype(np.uint16)
            cv2.imwrite(str(out_dir / f"{stem}_16bit.png"), depth_16)

    def estimate_depth(
        self,
        input_path: str,
        output_dir: str,
        model_prefix: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Estimate depth for a single image or all frames in a directory.

        Args:
            input_path: Path to a single image file or directory containing frame images
            output_dir: Directory to save depth maps
            model_prefix: If set, save under output_dir/model_prefix/ with prefix in filenames

        Returns:
            List of depth maps as numpy arrays
        """
        input_p = Path(input_path)
        frames = []
        if input_p.is_file():
            frames = [input_p]
        else:
            extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
            for ext in extensions:
                frames.extend(input_p.glob(ext))
            frames = sorted(frames)
            
        if not frames:
            print(f"No images found at {input_path}")
            return []

        base_out = Path(output_dir)
        out_dir = base_out / model_prefix if model_prefix else base_out
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_prefix or "depth"
        depths = []

        with torch.no_grad():
            for i, f in enumerate(tqdm(frames, desc="Estimating depth")):
                img_bgr = cv2.imread(str(f))
                if img_bgr is None:
                    print(f"Could not read image: {f}")
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                depth = self.backend.infer(img_rgb)
                # For single file, use the filename stem, else use index
                idx_or_name = f.stem if input_p.is_file() else i
                self._save_depth_maps(depth, out_dir, idx_or_name, prefix=prefix)
                depths.append(depth)

        if model_prefix and depths:
            _save_depth_details(depths, out_dir, model_prefix, self.config.target_size)
        return depths

    def temporal_filter_depth(
        self,
        depths: List[np.ndarray],
        flows: List[np.ndarray],
        output_dir: str,
    ) -> List[np.ndarray]:
        """
        Apply temporal filtering to depth maps using optical flow.

        Args:
            depths: List of depth maps
            flows: List of optical flow maps
            output_dir: Directory to save filtered depth maps

        Returns:
            List of filtered depth maps
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not depths:
            return []

        alpha = 0.6
        filtered = [depths[0]]
        for i in range(1, len(depths)):
            d = alpha * depths[i] + (1 - alpha) * filtered[-1]
            filtered.append(d)

        for i, d in enumerate(filtered):
            np.save(out_dir / f"depth_filtered_{i:06d}.npy", d.astype(np.float32))
            if getattr(self.config, "save_depth_visualizations", True):
                vis = (d * 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(out_dir / f"depth_filtered_{i:06d}.png"), vis)
            if getattr(self.config, "save_depth_16bit", False):
                depth_16 = (d * 65535).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"depth_filtered_{i:06d}_16bit.png"), depth_16)

        return filtered


class Segmentor:
    """
    Semantic and dynamic segmentation.
    Uses Mask2Former or DeepLabV3 with fallback to motion-based masks.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model = None
        self.processor = None
        self.preproc = None

        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade")
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-ade"
            ).to(self.device)
            print("Mask2Former loaded for segmentation.")
        except Exception:
            try:
                import torchvision
                from torchvision import transforms
                seg = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True).to(self.device)
                seg.eval()
                self.model = seg
                self.preproc = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.config.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                print("torchvision DeepLabV3 loaded for semantic segmentation.")
            except Exception as e:
                print(f"No segmentation model available: {e}. Falling back to motion-based masks.")

    def segment_frames(
        self,
        input_path: str,
        output_dir: str,
        flow_dir: Optional[str] = None
    ):
        """
        Segment frames using semantic segmentation or fallback strategies.

        Args:
            input_path: Path to a single image file or directory containing frame images
            output_dir: Directory to save segmentation masks
            flow_dir: Optional directory containing optical flow files (for motion fallback)
        """
        input_p = Path(input_path)
        frames = []
        if input_p.is_file():
            frames = [input_p]
        else:
            extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
            for ext in extensions:
                frames.extend(input_p.glob(ext))
            frames = sorted(frames)
            
        if not frames:
            print(f"No images found for segmentation at {input_path}")
            return

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            print("Primary segmentation model missing.")
            
            # Check if flow directory is provided and valid
            if flow_dir and Path(flow_dir).exists() and not input_p.is_file():
                print("Using motion-based fallback.")
                for i, f in enumerate(tqdm(frames, desc="Segmenting (motion fallback)")):
                    img = cv2.imread(str(f))
                    if img is None: continue
                    img_res = cv2.resize(img, self.config.target_size)
                    
                    # Try to find corresponding flow file
                    flow_path = Path(flow_dir) / f"flow_fwd_{i:06d}.npy"
                    if flow_path.exists():
                        flow = np.load(flow_path)
                        mag = np.linalg.norm(flow, axis=2)
                    else:
                        mag = np.zeros(self.config.target_size[::-1], dtype=np.float32)
                        
                    dyn_mask = (mag > 1.5).astype("uint8") * 255
                    stem = f.stem
                    cv2.imwrite(str(out_dir / f"dynamic_{stem}.png"), dyn_mask)
                    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                    sem_mask = (gray > 50).astype("uint8") * 255
                    cv2.imwrite(str(out_dir / f"semantic_{stem}.png"), sem_mask)
            else:
                # Fallback for single image or missing flow
                print("No optical flow available or single image. Using static thresholding fallback.")
                for i, f in enumerate(tqdm(frames, desc="Segmenting (static fallback)")):
                    img = cv2.imread(str(f))
                    if img is None: continue
                    img_res = cv2.resize(img, self.config.target_size)
                    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                    
                    # Simple heuristic: Otsu's thresholding
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, sem_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    stem = f.stem
                    cv2.imwrite(str(out_dir / f"semantic_{stem}.png"), sem_mask)
            return

        try:
            from transformers import Mask2FormerForUniversalSegmentation
            if isinstance(self.model, Mask2FormerForUniversalSegmentation):
                for i, f in enumerate(tqdm(frames, desc="Mask2Former segmentation")):
                    try:
                        img = Image.open(f).convert("RGB")
                    except Exception as e:
                        print(f"Could not open image {f}: {e}")
                        continue
                        
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    pred_masks = outputs.pred_masks.cpu().numpy()
                    combined = (pred_masks.sum(0) > 0).astype("uint8") * 255
                    combined_resized = cv2.resize(combined, self.config.target_size)
                    
                    stem = f.stem
                    cv2.imwrite(str(out_dir / f"semantic_{stem}.png"), combined_resized)
                return
        except ImportError:
            pass

        for i, f in enumerate(tqdm(frames, desc="DeepLabV3 segmentation")):
            img = cv2.imread(str(f))
            if img is None: continue
            inp = self.preproc(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(inp)["out"][0]
            seg_map = out.argmax(0).cpu().numpy().astype("uint8")
            seg_resized = cv2.resize(seg_map, self.config.target_size, interpolation=cv2.INTER_NEAREST)
            bin_mask = (seg_resized > 0).astype("uint8") * 255
            
            stem = f.stem
            cv2.imwrite(str(out_dir / f"semantic_{stem}.png"), bin_mask)
