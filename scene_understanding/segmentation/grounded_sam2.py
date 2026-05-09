"""Grounded SAM2 — canonical implementation (synced from scene_understanding legacy module)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


class GroundedSAM2Wrapper:
    """
    Grounding DINO v2 + SAM2 prompted mode for object-level instance segmentation.

    Returns the same list-of-dicts format as the former SAM2AMGWrapper so it
    remains a drop-in replacement for callers. Extra keys per entry:
      - "label": semantic class from GDINO (str)
      - "gdino_conf": GDINO detection confidence (float)
    """

    def __init__(
        self,
        device: torch.device,
        sam2_checkpoint_path: str,
        sam2_model_cfg: str,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        box_thresh: float = 0.30,
        text_thresh: float = 0.25,
        text_query: str = "person. animal. vehicle. furniture. appliance. food. clothing. container. tool. building. plant. electronics. object.",
        max_image_side: int = 1280,
        torch_compile_enabled: bool = False,
        **_: Any,
    ) -> None:
        """
        Args:
            device: torch device (cuda / cpu)
            sam2_checkpoint_path: Path to SAM2 checkpoint
            sam2_model_cfg: SAM2 Hydra config
            gdino_model_id: GDINO model ID
            box_thresh: GDINO box threshold
            text_thresh: GDINO text threshold
            text_query: Open-vocabulary query for GDINO
            max_image_side: Downscale long edge before GDINO/SAM2 if larger
            **_: Ignored legacy kwargs (AMG / fallback) removed from the pipeline.
        """
        self.device = device
        self.text_query = text_query
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.max_image_side = max_image_side
        self._torch_compile_enabled = bool(torch_compile_enabled)
        self._gdino = None
        self._gdino_processor = None
        self._sam2_predictor = None
        self.active = False
        # Phase B.4 caches.
        # GDINO text tokens are deterministic for a fixed query; cache across
        # images so every frame skips the tokenizer call (few ms per image but
        # adds up for folder runs and keeps allocations stable on MPS).
        self._text_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        # SAM2 image-embedding identity marker: set_image recomputes the
        # encoder when the payload changes. We identify the last encoded
        # image by (id, shape, dtype, a 16-byte payload hash). When another
        # call with the same image identity arrives, skip re-encoding.
        self._sam2_last_image_key: Any = None

        print(f"Initializing Grounding DINO ({gdino_model_id})...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
                gdino_model_id
            ).to(device)
            self._gdino.eval()
            print("Grounding DINO ready.")
        except Exception as e:
            print(f"Grounding DINO init failed: {e}.")
            raise

        print("Initializing SAM2 predictor (prompted mode)...")
        try:
            from ._sam2_path import ensure_sam2_importable
            ensure_sam2_importable()
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            ckpt = Path(sam2_checkpoint_path)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            hydra_cfg = sam2_model_cfg
            if hydra_cfg.endswith(".yaml"):
                hydra_cfg = hydra_cfg[:-5]
            # Meta's build_sam2 defaults to device="cuda" and moves the model internally.
            # Pass our pipeline device so CPU/MPS (e.g. Apple Silicon) works without editing sam2.
            apply_pp = device.type == "cuda"
            model = build_sam2(
                hydra_cfg,
                str(ckpt),
                device=device,
                apply_postprocessing=apply_pp,
            )
            self._sam2_predictor = SAM2ImagePredictor(model)
            # Phase E — opt-in torch.compile on the SAM2 image encoder. We
            # only compile the encoder (not the mask decoder) because the
            # encoder is called once per unique image and dominates cost,
            # while the decoder is called per-prompt and stays eager for
            # latency predictability. Gate via config.torch_compile_enabled.
            if self._torch_compile_enabled:
                try:
                    import torch
                    encoder = getattr(model, "image_encoder", None)
                    if encoder is not None:
                        compiled = torch.compile(encoder, mode="reduce-overhead", fullgraph=False)
                        model.image_encoder = compiled
                        print("SAM2: torch.compile(reduce-overhead) enabled on image encoder.")
                except Exception as _compile_exc:
                    print(f"SAM2 torch.compile failed ({_compile_exc.__class__.__name__}); continuing eager.")
            print("SAM2 predictor (prompted) ready.")
            self.active = True
        except Exception as e:
            print(f"SAM2 predictor init failed: {e}.")
            self.active = False

    def _prepare_gdino_inputs(self, pil_img) -> Dict[str, Any]:
        """Build GDINO processor inputs, caching the text half across calls.

        Falls back to the standard ``self._gdino_processor(images=, text=)``
        call when the underlying processor does not expose the component
        ``tokenizer`` / ``image_processor`` pair expected by the cache path.
        """
        tokenizer = getattr(self._gdino_processor, "tokenizer", None)
        image_processor = getattr(self._gdino_processor, "image_processor", None)
        if tokenizer is None or image_processor is None:
            return self._gdino_processor(
                images=pil_img,
                text=self.text_query,
                return_tensors="pt",
            )
        text_inputs = self._text_cache.get(self.text_query)
        if text_inputs is None:
            tok = tokenizer(self.text_query, return_tensors="pt")
            text_inputs = {k: v for k, v in tok.items()}
            self._text_cache[self.text_query] = text_inputs
        image_inputs = image_processor(pil_img, return_tensors="pt")
        return {
            **{k: v for k, v in image_inputs.items()},
            **text_inputs,
        }

    def _maybe_set_image(self, image_rgb: np.ndarray) -> None:
        """Run SAM2 ``set_image`` only when the image payload actually changed.

        The SAM2ImagePredictor caches image embeddings internally after
        ``set_image``. We short-circuit on repeat calls with the same ndarray
        identity + shape + dtype + a tiny head/tail fingerprint so downstream
        prompt-only callers (box + point + mask prompts) all share a single
        encoder pass per image.
        """
        key = (
            id(image_rgb),
            image_rgb.shape,
            image_rgb.dtype.str,
            int(image_rgb.size),
            int(image_rgb[0, 0, 0]) if image_rgb.ndim == 3 and image_rgb.size else 0,
            int(image_rgb[-1, -1, -1]) if image_rgb.ndim == 3 and image_rgb.size else 0,
        )
        if key == self._sam2_last_image_key:
            return
        self._sam2_predictor.set_image(image_rgb)
        self._sam2_last_image_key = key

    def _detect_objects(self, image_rgb: np.ndarray) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Run Grounding DINO on image_rgb.

        Returns:
            boxes:  list of [x1,y1,x2,y2] in pixel coords
            labels: list of class label strings
            scores: list of detection confidence floats
        """
        if self._gdino is None:
            return [], [], []

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_rgb)
        inputs = self._prepare_gdino_inputs(pil_img)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self._gdino(**inputs)

        h, w = image_rgb.shape[:2]
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[(h, w)],
            )[0]
        except TypeError as e:
            if "unexpected keyword argument 'threshold'" not in str(e):
                raise
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[(h, w)],
            )[0]

        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results["labels"]

        clipped = []
        for box in boxes:
            clipped.append([
                max(0.0, box[0]), max(0.0, box[1]),
                min(float(w), box[2]), min(float(h), box[3]),
            ])
        return clipped, list(labels), scores

    def _boxes_to_masks(
        self,
        image_rgb: np.ndarray,
        boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """Run SAM2 predictor once with all boxes as prompts; one binary mask per box."""
        if self._sam2_predictor is None or not boxes:
            return []
        try:
            self._maybe_set_image(image_rgb)
            boxes_np = np.array(boxes, dtype=np.float32)
            with torch.inference_mode():
                masks, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_np,
                    multimask_output=False,
                )
            result = []
            for i in range(len(boxes)):
                m = masks[i]
                if m.ndim == 3:
                    m = m[0]
                result.append(m.astype(bool))
            return result
        except Exception as e:
            print(f"  [GroundedSAM2] SAM2 predict failed: {e}")
            return [np.zeros(image_rgb.shape[:2], dtype=bool)] * len(boxes)

    def generate(
        self,
        image_rgb: np.ndarray,
        *,
        pyramid: Any = None,
        gdino_level: str = "gdino",
    ) -> List[Dict[str, Any]]:
        """Run GDINO → SAM2 masks. Returns [] when GDINO finds no boxes.

        When ``pyramid`` is provided, the GDINO-sized view is taken from the
        shared :class:`ImagePyramid` instead of calling ``cv2.resize`` here.
        """
        h, w = image_rgb.shape[:2]
        scale = 1.0
        proc = image_rgb
        if pyramid is not None:
            try:
                proc = pyramid.rgb(gdino_level)
                scale = float(pyramid.scale_for(gdino_level))
            except Exception:
                proc = image_rgb
                scale = 1.0
        elif self.max_image_side > 0 and max(h, w) > self.max_image_side:
            scale = self.max_image_side / max(h, w)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            proc = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        boxes, labels, scores = self._detect_objects(proc)
        print(f"  [GroundedSAM2] GDINO detected {len(boxes)} objects.")

        if not boxes:
            return []

        if scale != 1.0 and scale > 0.0:
            inv = 1.0 / scale
            boxes = [[b[0]*inv, b[1]*inv, b[2]*inv, b[3]*inv] for b in boxes]

        masks = self._boxes_to_masks(image_rgb, boxes)

        results = []
        for i, (box_xyxy, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            x1, y1, x2, y2 = box_xyxy
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = int(mask.sum()) if mask is not None else int(bw * bh)
            results.append({
                "segmentation": mask,
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "area": area,
                "predicted_iou": float(score),
                "stability_score": float(score),
                "quality_source": "gdino_conf_proxy",
                "label": str(label).strip().lower(),
                "gdino_conf": float(score),
                "source_model": "GroundedSAM2",
            })
        return results

    def update_text_query(self, query: str) -> None:
        """Update the GDINO text query (e.g. from RAM++ dynamic vocabulary).

        The text-embedding cache is query-keyed so previously tokenized
        queries remain cached if the pipeline cycles back to them.
        """
        self.text_query = query

    def unload(self) -> None:
        """Free GDINO + SAM2 weights and drop caches.

        Called after the segmentation stage on memory-constrained devices
        (e.g. Apple M2 with 8 GB unified memory) to avoid co-residency with
        Florence-2 / RAM++ during mask-enrichment. On the next call to
        :meth:`generate` the pipeline must re-construct the wrapper; the
        scene_understanding orchestrator is responsible for that lifecycle.
        """
        import gc

        self._gdino = None
        self._gdino_processor = None
        self._sam2_predictor = None
        self._text_cache = {}
        self._sam2_last_image_key = None
        self.active = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        gc.collect()
