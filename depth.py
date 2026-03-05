"""
Depth estimation and segmentation module.
Provides depth estimation using Depth Anything V2 Metric (Indoor or Outdoor)
and semantic segmentation using Mask2Former/DeepLabV3.

Fix 5.1: CLIP-based automatic scene classification selects the correct
Depth Anything V2 Metric variant at runtime.
  - Indoor scenes → Depth-Anything-V2-Metric-Indoor-Large-hf (NYUv2-trained)
  - Outdoor scenes → Depth-Anything-V2-Metric-Outdoor-Large-hf (KITTI-trained)
  - "auto" mode: uses CLIP zero-shot to classify before loading depth model,
    then unloads CLIP immediately to reclaim VRAM.
  - The metric models output true meters; depth_scale_factor should be 1.0.
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
    """Normalize depth to [0, 1] range.
    NOTE: Only call this for the relative (non-metric) model variant.
    Metric models output meters directly — normalizing destroys that information.
    """
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


def _resize_to_target(depth: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize depth map to target size (width, height).
    Uses INTER_NEAREST to avoid blending real depth values at boundaries —
    critical for mask-level depth extraction later in the pipeline.
    """
    if depth.shape[:2] == (target_size[1], target_size[0]):
        return depth
    return cv2.resize(
        depth.astype(np.float32),
        target_size,
        # INTER_NEAREST: no interpolation, preserves actual metric values.
        # INTER_LINEAR would blend fg+bg depths at object edges, corrupting
        # the mask-level depth stats computed in Stage 4.
        interpolation=cv2.INTER_NEAREST,
    )


# ---------------------------------------------------------------------------
# Fix 5.1: CLIP-based scene type classifier
# ---------------------------------------------------------------------------
class SceneTypeClassifier:
    """
    Lightweight CLIP-based zero-shot classifier that decides whether an input
    image is an INDOOR or OUTDOOR scene, so the correct Depth Anything V2
    Metric variant can be loaded.

    Memory strategy:
      - CLIP ViT-B/32 weighs ~340 MB on GPU.
      - We load it, classify, then immediately call unload() to free that VRAM
        before loading the (much larger) depth model.
      - If CLIP is unavailable we fall back to the configured default.

    Why this matters:
      - Depth-Anything-V2-Metric-Indoor is trained on NYUv2 (indoor depth up
        to ~10 m with fine near-range detail).
      - Depth-Anything-V2-Metric-Outdoor is trained on KITTI (outdoor depth
        up to ~80 m, coarser near-range).
      - Using the wrong variant introduces systematic scale and shape errors
        that cascade into wrong Z coordinates and broken depth-gated relations.
    """

    # CLIP prompts for indoor/outdoor classification.
    # Phrased as full sentences so CLIP's contrastive training aligns well.
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
        """
        Explicitly unload CLIP from GPU/CPU memory.
        Call this immediately after classify() so VRAM is free before the
        depth model loads. On a 24 GB card this reclaims ~340 MB, which
        matters when you later load SAM2 (850 MB) + Florence-2 (1.5 GB) +
        depth model (1.8 GB) simultaneously.
        """
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
        """
        Classify a PIL image as 'indoor' or 'outdoor'.

        How it works:
          1. Encode the image and both prompt sets with CLIP.
          2. Compute softmax similarity between image and each prompt.
          3. Average scores within each group (indoor vs outdoor).
          4. Return the group with the higher average score.

        Returns: 'indoor' or 'outdoor'
        """
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

            # logits_per_image: shape (1, num_texts)
            logits = outputs.logits_per_image[0]  # (num_texts,)
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


# ---------------------------------------------------------------------------
# Fix 5.1: Metric depth backend with runtime variant selection
# ---------------------------------------------------------------------------
class DepthAnythingV2Backend:
    """
    Depth Anything V2 Metric Large via HuggingFace transformers.

    Fix 5.1 changes vs. original:
    - model_id is chosen at init time based on scene_type ('indoor'/'outdoor').
    - infer() returns RAW METRIC METERS — it does NOT call _normalize_depth().
      The ×10 depth_scale_factor hack in config.py is bypassed for metric models.
    - _resize_to_target() still uses INTER_NEAREST (no boundary blending).

    Model IDs:
      indoor  → depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
                 Trained on NYUv2; depth range ~0.1 m – 10 m.
                 Best for: rooms, furniture, people, kitchen, office.
      outdoor → depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
                 Trained on KITTI; depth range ~0.5 m – 80 m.
                 Best for: streets, vehicles, buildings, parks, landscapes.
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
        """
        Run metric depth estimation.

        Returns:
            float32 array of metric depth in METERS, same spatial size as
            config.target_size (resized with INTER_NEAREST).

        Critical implementation note:
            We do NOT call _normalize_depth() here. The metric models output
            values already in meters (e.g. 0.3 m – 8.5 m for an indoor room).
            Normalizing would collapse these to [0, 1] and destroy all metric
            information. The depth_scale_factor in config should be 1.0 for
            metric models — it is applied in process_image() and will multiply
            real meters by 1.0 (no-op) rather than the old ×10 hack.
        """
        pil_img = Image.fromarray(img_rgb)
        out = self.pipe(pil_img)

        # IMPORTANT: out["depth"] is a PIL Image normalised to uint8 0-255 —
        # it is NOT metric depth. The raw metric tensor (in true metres) lives
        # in out["predicted_depth"] as a torch.Tensor of shape (H, W).
        # Always use "predicted_depth" for any metric computation.
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

        # Return at native model resolution. process_image() in
        # scene_understanding.py resizes this to match the input image with
        # cv2.resize(..., INTER_NEAREST) — one resize is better than two.
        # Do NOT call _resize_to_target here; that would force a fixed 512×512
        # square before the second resize, destroying aspect ratio and resolution.
        return depth


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
    Monocular metric depth estimation using Depth Anything V2 Metric.

    Fix 5.1: scene_type is determined at construction time.
    If config.depth_model_variant == 'auto', a CLIP-based classifier runs
    on the first image to decide indoor vs outdoor, then CLIP is immediately
    unloaded to free VRAM before the depth model loads.

    Variant selection is sticky: once determined, the same backend is used
    for all subsequent images (re-classification per image would require
    reloading the depth model each time, which is prohibitively slow).
    If your dataset mixes indoor and outdoor images, set depth_model_variant
    explicitly to the dominant scene type rather than 'auto'.
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

    def _auto_classify(
        self,
        image: Optional[Union[str, Path, Image.Image, np.ndarray]],
    ) -> str:
        """
        Run CLIP scene classifier on image, then immediately unload CLIP.
        Falls back to 'indoor' if image is None or CLIP is unavailable.

        Memory lifecycle:
          1. CLIP loads (~340 MB VRAM).
          2. Single forward pass classifies the scene.
          3. CLIP.unload() frees all VRAM before depth model loads.
          4. Depth model loads into freshly freed memory.
        """
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

        # Classify, then immediately free CLIP VRAM
        scene_type = classifier.classify(pil_img)
        classifier.unload()   # <-- critical: frees ~340 MB before depth model loads
        return scene_type

    def _save_depth_maps(
        self,
        depth: np.ndarray,
        out_dir: Path,
        idx_or_name: Union[int, str],
        prefix: str = "depth",
    ) -> None:
        """Save depth map as .npy and optionally PNG visualizations.

        For metric depth the .npy stores raw meter values.
        The PNG visualization normalizes to [0,255] for display only —
        it does NOT affect the stored metric values.
        """
        if isinstance(idx_or_name, int):
            stem = f"{prefix}_{idx_or_name:06d}"
        else:
            stem = f"{prefix}_{idx_or_name}"

        np.save(out_dir / f"{stem}.npy", depth.astype(np.float32))

        if getattr(self.config, "save_depth_visualizations", True):
            # Normalize for visualization only (metric values preserved in .npy)
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min < 1e-8:
                vis = np.zeros_like(depth, dtype=np.uint8)
            else:
                vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(out_dir / f"{stem}.png"), vis)

        if getattr(self.config, "save_depth_16bit", False):
            # 16-bit PNG: scale meters to [0, 65535] with max_range=20m indoor / 80m outdoor
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


class Segmentor:
    """
    Semantic and dynamic segmentation.
    Uses Mask2Former or DeepLabV3 with fallback to motion-based masks.
    (Unchanged from original — this class is not used in the main pipeline.)
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
                print(f"No segmentation model available: {e}.")

    def segment_frames(
        self,
        input_path: str,
        output_dir: str,
        flow_dir: Optional[str] = None
    ):
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
            if flow_dir and Path(flow_dir).exists() and not input_p.is_file():
                print("Using motion-based fallback.")
                for i, f in enumerate(tqdm(frames, desc="Segmenting (motion fallback)")):
                    img = cv2.imread(str(f))
                    if img is None: continue
                    img_res = cv2.resize(img, self.config.target_size)
                    flow_path = Path(flow_dir) / f"flow_fwd_{i:06d}.npy"
                    if flow_path.exists():
                        flow = np.load(flow_path)
                        mag = np.linalg.norm(flow, axis=2)
                    else:
                        mag = np.zeros(self.config.target_size[::-1], dtype=np.float32)
                    dyn_mask = (mag > 1.5).astype("uint8") * 255
                    cv2.imwrite(str(out_dir / f"dynamic_{f.stem}.png"), dyn_mask)
                    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                    sem_mask = (gray > 50).astype("uint8") * 255
                    cv2.imwrite(str(out_dir / f"semantic_{f.stem}.png"), sem_mask)
            else:
                for i, f in enumerate(tqdm(frames, desc="Segmenting (static fallback)")):
                    img = cv2.imread(str(f))
                    if img is None: continue
                    img_res = cv2.resize(img, self.config.target_size)
                    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, sem_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imwrite(str(out_dir / f"semantic_{f.stem}.png"), sem_mask)
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
                    cv2.imwrite(str(out_dir / f"semantic_{f.stem}.png"), combined_resized)
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
            cv2.imwrite(str(out_dir / f"semantic_{f.stem}.png"), bin_mask)
