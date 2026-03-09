"""
Depth estimation module using Depth Anything V2 Metric.
CLIP classifies the scene (indoor/outdoor) to select the correct variant.
See docs/DEPTH_ESTIMATION.md for model details, formulas, and VRAM lifecycle.
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
    """Normalize depth to [0, 1] range. Only valid for relative (non-metric) models."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


def _resize_to_target(depth: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize depth map to (width, height) using INTER_NEAREST to preserve metric values."""
    if depth.shape[:2] == (target_size[1], target_size[0]):
        return depth
    return cv2.resize(depth.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)


class SceneTypeClassifier:
    """
    CLIP-based zero-shot indoor/outdoor classifier.
    Loads, classifies, then immediately unloads to free ~340 MB VRAM.
    See docs/DEPTH_ESTIMATION.md for the softmax formula and prompt design.
    """

    _INDOOR_PROMPTS = [
        "a photo of an indoor room",
        "an interior space with furniture",
        "inside a building with walls and ceiling",
    ]
    _OUTDOOR_PROMPTS = [
        "a photo taken outside with sky or open space",
        "an outdoor scene with trees, streets or buildings",
        "a landscape or street scene outside",
    ]

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

    def _load(self) -> bool:
        """Load CLIP. Returns True if successful."""
        if self._loaded:
            return True
        try:
            from transformers import CLIPProcessor, CLIPModel
            model_id = "openai/clip-vit-base-patch32"
            print(f"  [SceneClassifier] Loading CLIP ({model_id}) for scene type detection...")
            self._processor = CLIPProcessor.from_pretrained(model_id)
            self._model = CLIPModel.from_pretrained(model_id).to(self.device)
            self._model.eval()
            self._loaded = True
            return True
        except Exception as e:
            print(f"  [SceneClassifier] CLIP load failed: {e}. Will use default scene type.")
            return False

    def unload(self) -> None:
        """Free CLIP from GPU memory. Call immediately after classify()."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [SceneClassifier] CLIP unloaded, VRAM reclaimed.")

    def classify(self, pil_image: Image.Image) -> str:
        """Classify image as 'indoor' or 'outdoor' via CLIP softmax. Returns scene type string."""
        if not self._load():
            return "indoor"  # safe default; indoor model has finer near-range detail

        try:
            all_texts = self._INDOOR_PROMPTS + self._OUTDOOR_PROMPTS
            inputs = self._processor(
                text=all_texts,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits = outputs.logits_per_image[0]  # shape (num_texts,)
            probs = logits.softmax(dim=0).cpu().numpy()

            n_indoor = len(self._INDOOR_PROMPTS)
            indoor_score = float(probs[:n_indoor].mean())
            outdoor_score = float(probs[n_indoor:].mean())

            scene_type = "indoor" if indoor_score >= outdoor_score else "outdoor"
            print(
                f"  [SceneClassifier] indoor={indoor_score:.3f}  "
                f"outdoor={outdoor_score:.3f}  → {scene_type}"
            )
            return scene_type
        except Exception as e:
            print(f"  [SceneClassifier] classify() failed: {e}. Defaulting to indoor.")
            return "indoor"


class DepthAnythingV2Backend:
    """
    Depth Anything V2 Metric via HuggingFace (indoor NYUv2 or outdoor KITTI variant).
    infer() returns raw float32 metres — no normalisation applied.
    See docs/DEPTH_ESTIMATION.md for variant selection rationale and output format.
    """

    _MODEL_IDS = {
        "indoor":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    }

    def __init__(self, config: PreprocessConfig, scene_type: str = "indoor"):
        self.config = config
        self.scene_type = scene_type if scene_type in self._MODEL_IDS else "indoor"
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        model_id = self._MODEL_IDS[self.scene_type]
        print(f"Loading Depth Anything V2 Metric ({self.scene_type}): {model_id}")

        # Suppress safetensors auto-conversion noise
        try:
            import transformers.safetensors_conversion as _stc
            def _disable_auto_conversion(*args, **kwargs):
                return None
            _stc.auto_conversion = _disable_auto_conversion
        except Exception:
            pass

        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device.type == "cuda" else -1,
        )

    def infer(self, img_rgb: np.ndarray) -> np.ndarray:
        """Run depth estimation. Returns float32 array in metres at native model resolution."""
        pil_img = Image.fromarray(img_rgb)
        out = self.pipe(pil_img)
        # Use "predicted_depth" (float32 metres), not "depth" (uint8 uint8 visualisation).
        depth = out["predicted_depth"]

        # Convert to numpy
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        elif hasattr(depth, "numpy"):
            depth = depth.numpy()
        else:
            depth = np.array(depth)

        if depth.ndim == 3:
            depth = depth.squeeze()

        depth = depth.astype(np.float32)
        return depth  # native model resolution; resized in process_image()


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
    """
    Monocular metric depth estimation. Scene type (indoor/outdoor) is determined
    once at construction via CLIP, then the matching Depth Anything V2 backend loads.
    See docs/DEPTH_ESTIMATION.md for the full lifecycle.
    """

    def __init__(
        self,
        config: PreprocessConfig,
        model_name: Optional[str] = None,
        first_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    ):
        """
        Args:
            config: Pipeline configuration.
            model_name: Ignored (kept for API compatibility).
            first_image: Optional image used for auto scene classification.
                         If config.depth_model_variant == 'auto' and this is
                         None, falls back to 'indoor'.
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )

        variant = getattr(config, "depth_model_variant", "auto")

        if variant == "auto":
            scene_type = self._auto_classify(first_image)
        elif variant in ("indoor", "outdoor"):
            scene_type = variant
            print(f"[DepthEstimator] Using configured scene type: {scene_type}")
        else:
            scene_type = "indoor"
            print(f"[DepthEstimator] Unknown variant '{variant}', defaulting to indoor.")

        try:
            self.backend = DepthAnythingV2Backend(config, scene_type=scene_type)
            self.backend_name = f"DepthAnythingV2-Metric-{scene_type.capitalize()}"
            print(f"Depth estimator initialized: {self.backend_name}")
        except Exception as e:
            print(f"Failed to load Depth Anything V2 Metric: {e}")
            raise e

    def unload_backend(self) -> None:
        """Free the depth model from GPU/RAM. Use in SAM3-only mode before loading SAM3 to avoid OOM."""
        if getattr(self, "backend", None) is not None:
            backend = self.backend
            if getattr(backend, "pipe", None) is not None:
                backend.pipe = None  # release reference so the pipeline can be gc'd
            self.backend = None
            del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("  [DepthEstimator] Backend unloaded, VRAM reclaimed.")

    def _auto_classify(
        self,
        image: Optional[Union[str, Path, Image.Image, np.ndarray]],
    ) -> str:
        """Run CLIP, classify scene, unload CLIP, return 'indoor' or 'outdoor'."""
        if image is None:
            print("[DepthEstimator] No image for auto scene classification. Defaulting to indoor.")
            return "indoor"

        classifier = SceneTypeClassifier(self.device)

        # Convert to PIL regardless of input type
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                      if image.shape[2] == 3 else image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            print("[DepthEstimator] Unrecognized image type for classification. Defaulting to indoor.")
            return "indoor"

        scene_type = classifier.classify(pil_img)
        classifier.unload()  # frees ~340 MB before depth model loads
        return scene_type

    def _save_depth_maps(
        self,
        depth: np.ndarray,
        out_dir: Path,
        idx_or_name: Union[int, str],
        prefix: str = "depth",
    ) -> None:
        """Save depth as .npy (metres) and optional PNG visualisation / 16-bit PNG."""
        if isinstance(idx_or_name, int):
            stem = f"{prefix}_{idx_or_name:06d}"
        else:
            stem = f"{prefix}_{idx_or_name}"

        np.save(out_dir / f"{stem}.npy", depth.astype(np.float32))

        if getattr(self.config, "save_depth_visualizations", True):
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min < 1e-8:
                vis = np.zeros_like(depth, dtype=np.uint8)
            else:
                vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(out_dir / f"{stem}.png"), vis)

        if getattr(self.config, "save_depth_16bit", False):
            max_range = 20.0 if getattr(self.config, "depth_model_variant", "auto") != "outdoor" else 80.0
            depth_16 = np.clip(depth / max_range * 65535, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out_dir / f"{stem}_16bit.png"), depth_16)

    def estimate_depth(
        self,
        input_path: str,
        output_dir: str,
        model_prefix: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Estimate metric depth for a single image or all frames in a directory.

        Returns:
            List of float32 depth arrays in METERS.
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
            for i, f in enumerate(tqdm(frames, desc="Estimating metric depth")):
                img_bgr = cv2.imread(str(f))
                if img_bgr is None:
                    print(f"Could not read image: {f}")
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                depth = self.backend.infer(img_rgb)
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
        Apply temporal EMA filtering to depth maps using optical flow.
        alpha=0.6 blends current and previous frame for temporal consistency.
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
                d_min, d_max = d.min(), d.max()
                vis = ((d - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(out_dir / f"depth_filtered_{i:06d}.png"), vis)
            if getattr(self.config, "save_depth_16bit", False):
                max_range = 20.0
                depth_16 = np.clip(d / max_range * 65535, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(out_dir / f"depth_filtered_{i:06d}_16bit.png"), depth_16)

        return filtered
