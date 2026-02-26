"""
scene_understanding.py
Unified 3D Scene Graph Generation using Depth Estimator and Specific Scene Graph Models.
Integrates Depth Anything V2 with GRiT, SGSG, Pix2SG, Faster R-CNN, DETR, and Pix2Seq.
"""
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import existing DepthEstimator
from depth import DepthEstimator

# -----------------------------------------------------------------------------
# GRiT Imports & Setup
# -----------------------------------------------------------------------------
import sys
import os

# Add GRiT paths so imports work correctly
# We need to add the project root and the specific CenterNet2 path
grit_root = Path("GRiT")
centernet_path = grit_root / "third_party/CenterNet2/projects/CenterNet2/"

# Add to sys.path if not already present
if str(centernet_path.resolve()) not in sys.path:
    sys.path.insert(0, str(centernet_path.resolve()))
if str(grit_root.resolve()) not in sys.path:
    sys.path.insert(0, str(grit_root.resolve()))

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data.detection_utils import read_image
    from centernet.config import add_centernet_config
    from grit.config import add_grit_config
except ImportError:
    print("Warning: Could not import GRiT modules. Ensure detectron2 and GRiT requirements are installed.")
    # Define dummy functions to prevent NameError if import fails
    def get_cfg(): return None
    def add_centernet_config(cfg): pass
    def add_grit_config(cfg): pass
    DefaultPredictor = None

# -----------------------------------------------------------------------------
# Additional Model Imports (Faster R-CNN, DETR, Pix2Seq)
# -----------------------------------------------------------------------------
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
except ImportError:
    print("Warning: Could not import torchvision detection models.")

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
except ImportError:
    print("Warning: Could not import transformers DETR models.")

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. GRiT Wrapper (Generative Region-to-Text Transformer)
# -----------------------------------------------------------------------------
class GRiTWrapper:
    """
    Wrapper for GRiT.
    Provides: Dense captions, Bounding boxes, Object labels.
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing GRiT...")
        
        # Check if GRiT imports succeeded
        if get_cfg() is None:
            print("GRiT initialization skipped: Dependencies not found.")
            self.predictor = None
            return

        # Paths to config and weights
        self.config_file = "GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml"
        self.weights_file = "GRiT/models/grit_b_densecap_objectdet.pth"

        if not os.path.exists(self.config_file) or not os.path.exists(self.weights_file):
            print(f"Error: GRiT config or weights not found at {self.config_file} / {self.weights_file}")
            self.predictor = None
            return

        # Initialize Config
        self.cfg = self._setup_cfg()
        if self.cfg:
            self.predictor = DefaultPredictor(self.cfg)
        else:
            self.predictor = None

    def _setup_cfg(self):
        cfg = get_cfg()
        if cfg is None:
            return None
            
        if self.device.type == 'cpu':
            cfg.MODEL.DEVICE = "cpu"
        
        add_centernet_config(cfg)
        add_grit_config(cfg)
        cfg.merge_from_file(self.config_file)
        
        # Set weights and confidence
        cfg.MODEL.WEIGHTS = self.weights_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        
        # Specific GRiT settings for DenseCap
        cfg.MODEL.TEST_TASK = 'DenseCap'
        cfg.MODEL.BEAM_SIZE = 1
        cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
        cfg.USE_ACT_CHECKPOINT = False
        
        cfg.freeze()
        return cfg

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs GRiT inference on a BGR image (OpenCV format).
        Returns list of objects with label, confidence, bbox, and caption.
        """
        if self.predictor is None:
            print("GRiT predictor not initialized.")
            return []
        
        # Predictor handles BGR -> RGB conversion internally if configured, 
        # but Detectron2 DefaultPredictor expects BGR (OpenCV format) by default.
        predictions = self.predictor(image)
        
        instances = predictions["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        # GRiT stores captions in 'pred_object_descriptions'
        # Note: Depending on the specific GRiT version, this might be a list or object
        if hasattr(instances, 'pred_object_descriptions'):
            captions = instances.pred_object_descriptions.data
        else:
            captions = ["unknown"] * len(boxes)

        results = []
        for i in range(len(boxes)):
            box = boxes[i].astype(int).tolist() # [x1, y1, x2, y2]
            score = float(scores[i])
            caption = captions[i]
            
            # Simple heuristic to get a short "label" from the longer caption
            label = caption.split()[-1] if caption else "object"
            
            # Ensure box coordinates are integers
            box = [int(x) for x in box]
            
            results.append({
                "label": label, 
                "conf": score, 
                "bbox": box, 
                "caption": caption,
                "source_model": "GRiT"
            })
            
        return results

# -----------------------------------------------------------------------------
# 2. Faster R-CNN Wrapper
# -----------------------------------------------------------------------------
class FasterRCNNWrapper:
    """
    Wrapper for Faster R-CNN (Torchvision).
    Provides: Bounding boxes, Object labels.
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing Faster R-CNN...")
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
            self.model.eval()
            self.categories = weights.meta["categories"]
        except Exception as e:
            print(f"Error initializing Faster R-CNN: {e}")
            self.model = None

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        
        # Convert BGR (OpenCV) to RGB and normalize to tensor [0, 1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)[0]

        results = []
        # Filter by confidence > 0.5
        threshold = 0.5
        boxes = outputs['boxes'][outputs['scores'] > threshold].cpu().numpy()
        scores = outputs['scores'][outputs['scores'] > threshold].cpu().numpy()
        labels = outputs['labels'][outputs['scores'] > threshold].cpu().numpy()

        for i in range(len(boxes)):
            box = boxes[i].astype(int).tolist()
            score = float(scores[i])
            label_idx = labels[i]
            label = self.categories[label_idx] if label_idx < len(self.categories) else f"class_{label_idx}"
            
            # Ensure box coordinates are integers for drawing
            box = [int(x) for x in box]
            
            results.append({
                "label": label,
                "conf": score,
                "bbox": box,
                "caption": label, # Faster R-CNN doesn't produce captions
                "source_model": "FasterRCNN"
            })
        return results

# -----------------------------------------------------------------------------
# 3. DETR Wrapper
# -----------------------------------------------------------------------------
class DETRWrapper:
    """
    Wrapper for DETR (Transformers).
    Provides: Bounding boxes, Object labels.
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing DETR...")
        try:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error initializing DETR: {e}")
            self.model = None

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        # Convert BGR to RGB (PIL Image expected by processor)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits)
        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(self.device)
        results_detr = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        results = []
        for score, label, box in zip(results_detr["scores"], results_detr["labels"], results_detr["boxes"]):
            box = box.cpu().numpy().astype(int).tolist()
            score = float(score)
            label_str = self.model.config.id2label[label.item()]
            
            # Ensure box coordinates are integers
            box = [int(x) for x in box]
            
            results.append({
                "label": label_str,
                "conf": score,
                "bbox": box,
                "caption": label_str,
                "source_model": "DETR"
            })
        return results

# -----------------------------------------------------------------------------
# 4. Pix2Seq / OWL-ViT Wrapper
# -----------------------------------------------------------------------------
import tensorflow as tf

class Pix2SeqWrapper:
    """
    Wrapper for Pix2Seq object detection with OWL-ViT fallback.
    If official Pix2Seq checkpoint is missing, falls back to OWL-ViT (Hugging Face).
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.model_type = "none"
        print("Initializing Pix2Seq/OWL-ViT...")
        
        # 1. Try Official Pix2Seq
        pix2seq_path = Path("pix2seq")
        if str(pix2seq_path.resolve()) not in sys.path:
            sys.path.insert(0, str(pix2seq_path.resolve()))
            
        try:
            from pix2seq.models import model as model_lib
            from pix2seq.tasks import task as task_lib
            from pix2seq.configs import config_det_finetune
            
            self.config = config_det_finetune.get_config()
            ckpt_path = pix2seq_path / "checkpoints/vit_b_640x640"
            
            # Check if checkpoint exists
            if ckpt_path.exists() and list(ckpt_path.glob("checkpoint")):
                print(f"Found Pix2Seq checkpoint at {ckpt_path}")
                # TF Setup
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        print(e)

                print("Building Pix2Seq model...")
                self.model = model_lib.ModelRegistry.lookup(self.config.model.name)(self.config)
                
                checkpoint = tf.train.Checkpoint(model=self.model)
                latest_ckpt = tf.train.latest_checkpoint(str(ckpt_path))
                if latest_ckpt:
                    print(f"Restoring Pix2Seq from {latest_ckpt}")
                    checkpoint.restore(latest_ckpt).expect_partial()
                    self.task = task_lib.TaskRegistry.lookup(self.config.task.name)(self.config)
                    self.model_type = "pix2seq"
                    return
            else:
                print("Pix2Seq checkpoint not found. Falling back to OWL-ViT (Hugging Face)...")

        except Exception as e:
            print(f"Error initializing Pix2Seq: {e}. Falling back to OWL-ViT...")

        # 2. Fallback to OWL-ViT (Hugging Face)
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            print("Loading OWL-ViT (google/owlvit-base-patch32)...")
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
            self.model.eval()
            self.model_type = "owlvit"
            
            # Load COCO classes for open-vocabulary detection
            # We use a standard list of 80 COCO classes
            self.texts = [
                ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                 "hair drier", "toothbrush"]
            ]
            
        except Exception as e:
            print(f"Error initializing OWL-ViT: {e}")
            self.model = None
            self.model_type = "none"

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.model_type == "none":
            return []
            
        if self.model_type == "pix2seq":
            return self._predict_pix2seq(image)
        elif self.model_type == "owlvit":
            return self._predict_owlvit(image)
        return []

    def _predict_pix2seq(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            example = {
                'image': tf.convert_to_tensor(img_rgb, dtype=tf.uint8),
                'image/id': tf.constant(0, dtype=tf.int64),
                'orig_image_size': tf.constant([h, w], dtype=tf.int32),
                'bbox': tf.zeros([0, 4], dtype=tf.float32),
                'label': tf.zeros([0], dtype=tf.int64),
                'area': tf.zeros([0], dtype=tf.float32),
                'is_crowd': tf.zeros([0], dtype=tf.bool),
            }
            
            for t in self.task.eval_transforms:
                example = t.process_example(example)
                
            batched_example = {k: tf.expand_dims(v, 0) for k, v in example.items()}
            _, pred_seq, logits = self.task.infer(self.model, (batched_example['image'], None, batched_example))
            results = self.task.postprocess_tpu(batched_example, pred_seq, logits, training=False)
            (_, _, _, pred_bboxes_rescaled, pred_classes, scores, _, _, _, _, _) = results
             
            boxes = pred_bboxes_rescaled.numpy()[0] # yxyx
            classes = pred_classes.numpy()[0]
            confidences = scores.numpy()[0]
            
            final_results = []
            category_names = self.task._category_names
            
            for i in range(len(boxes)):
                score = float(confidences[i])
                if score < 0.5: continue
                cls_id = int(classes[i])
                if cls_id == 0: continue
                
                label = category_names.get(cls_id, f"class_{cls_id}")
                y1, x1, y2, x2 = boxes[i]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                final_results.append({
                    "label": label,
                    "conf": score,
                    "bbox": bbox,
                    "caption": label,
                    "source_model": "Pix2Seq"
                })
            return final_results
        except Exception as e:
            print(f"Pix2Seq inference error: {e}")
            return []

    def _predict_owlvit(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            inputs = self.processor(text=self.texts, images=img_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([[h, w]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
            
            final_results = []
            # results["boxes"] are [x1, y1, x2, y2] format for OWL-ViT
            for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score < 0.2: continue # Use lower threshold for open-vocab
                
                box = [int(i) for i in box.tolist()]
                label = self.texts[0][label_idx]
                
                final_results.append({
                    "label": label,
                    "conf": float(score),
                    "bbox": box,
                    "caption": label,
                    "source_model": "OWL-ViT" # Label as OWL-ViT to distinguish
                })
            return final_results
        except Exception as e:
            print(f"OWL-ViT inference error: {e}")
            return []

# -----------------------------------------------------------------------------
# 5. SGSG Wrapper (Spatial Scene Graph Generation - SGTR+)
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

# Add SGTR and cvpods to path
sgtr_root = Path("SGTR")
cvpods_root = Path("cvpods")

if str(sgtr_root.resolve()) not in sys.path:
    sys.path.insert(0, str(sgtr_root.resolve()))
if str(cvpods_root.resolve()) not in sys.path:
    sys.path.insert(0, str(cvpods_root.resolve()))

class SGSGWrapper:
    """
    Wrapper for SGSG using SGTR+ (End-to-End Scene Graph Generation with Transformer).
    Provides: 3D-aware Relationship Triplets (Subject-Predicate-Object).
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing SGSG (SGTR+)...")
        self.model = None
        self.cfg = None
        
        try:
            # Import SGTR+ specific modules
            # Add SGTR/playground to sys.path to allow direct imports
            playground_path = sgtr_root / "playground"
            if str(playground_path.resolve()) not in sys.path:
                sys.path.insert(0, str(playground_path.resolve()))

            # Explicitly import the config file to avoid module path issues
            import importlib.util
            config_path = playground_path / "sgg/detr.res101.c5.one_stage_rel_tfmer/config_vg_sgtr.py"
            spec = importlib.util.spec_from_file_location("config_vg_sgtr", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = config_module.config

            # Import the net module similarly or via sys.path trickery
            # The net module uses relative imports, so we need to be careful.
            # Best approach: append the parent dir of 'sgg' to path? No, 'sgg' is inside 'playground'
            
            # Since we added 'playground' to sys.path, we should be able to import 'sgg....'
            # But the previous error said "No module named 'playground.sgg.detr'" when we tried that.
            # This suggests 'playground' is NOT a package (missing __init__.py)? We added it.
            
            # Let's try importing from 'sgg' package assuming 'playground' is in path
            from sgg.detr.res101.c5.one_stage_rel_tfmer.net import OneStageEncDecVRD
            from cvpods.checkpoint import DetectionCheckpointer
            
            self.cfg = config
            self.cfg.MODEL.DEVICE = str(self.device)
            # Patch: Remove unsupported 'res5' key if it causes issues or ensure ResNet backbone is compatible
            # SGTR config might expect specific keys for ResNet
            
            # Load Model
            self.model = OneStageEncDecVRD(self.cfg)
            self.model.to(self.device)
            
            # Load Checkpoint
            weights_path = "sgtr_vg_new_pth/model_0095999.pth" 
            if os.path.exists(weights_path):
                 print(f"Loading SGTR+ weights from {weights_path}")
                 DetectionCheckpointer(self.model).resume_or_load(weights_path, resume=False)
            else:
                 print(f"Warning: SGTR+ weights not found at {weights_path}. Model initialized with random weights.")

            self.model.eval()
            
        except ImportError as e:
            print(f"Error initializing SGTR+: {e}. Ensure 'cvpods' and 'SGTR' are in your path.")
        except Exception as e:
             print(f"Error loading SGTR+ model: {e}")

    def predict(self, image: np.ndarray) -> List[Dict[str, str]]:
        if self.model is None:
            return []
            
        try:
            # Prepare Input
            # cvpods expects BGR image (OpenCV format) transformed to tensor
            # The preprocessing is typically handled inside the model or a mapper
            # Here we do a basic manual preprocessing matching standard detection pipelines
            
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            
            with torch.no_grad():
                # SGTR+ Inference
                # Returns list of dicts with 'instances' and 'relationships'
                outputs = self.model(inputs)[0]
                
            # Parse Outputs
            relationships = outputs["relationships"] # cvpods Relationships object
            
            # Extract Triplets
            # relationships.rel_pair_tensor: [N, 2] indices into instances
            # relationships.pred_rel_classs: [N] predicate labels
            
            pred_rels = relationships.pred_rel_classs.cpu().numpy()
            rel_pairs = relationships.rel_pair_tensor.cpu().numpy()
            
            # We need the entity labels to map indices to names
            instances = outputs["instances"]
            pred_classes = instances.pred_classes.cpu().numpy()
            
            # Placeholder for label mapping (Visual Genome classes)
            # In a real setup, you'd load the specific VG class names used by the model
            # For now, we return indices or generic placeholders if mapping isn't loaded
            
            triplets = []
            for i in range(len(pred_rels)):
                sub_idx = rel_pairs[i][0]
                obj_idx = rel_pairs[i][1]
                
                # Get class IDs
                sub_cls = pred_classes[sub_idx]
                obj_cls = pred_classes[obj_idx]
                pred_cls = pred_rels[i]
                
                # Convert to string (requires label map, using IDs for now)
                # You should load the actual VG label map 'vg_classes.json' and 'vg_predicates.json'
                triplet = {
                    "sub": f"class_{sub_cls}", 
                    "pred": f"rel_{pred_cls}", 
                    "obj": f"class_{obj_cls}"
                }
                triplets.append(triplet)
                
            return triplets

        except Exception as e:
            print(f"SGTR+ Inference Error: {e}")
            return []

# -----------------------------------------------------------------------------
# 6. Pix2SG Wrapper (Pixel-to-Scene Graph)
# -----------------------------------------------------------------------------
class Pix2SGWrapper:
    """
    Wrapper for Pix2SG.
    Provides: Semantic/Abstract Relationship Triplets.
    """
    def __init__(self, device: torch.device):
        self.device = device
        print("Initializing Pix2SG...")
        self.model = None

    def predict(self, image: np.ndarray) -> List[Dict[str, str]]:
        if self.model is None:
            return []
        return []

# -----------------------------------------------------------------------------
# 7. SAM2 Automatic Mask Generator Wrapper
# -----------------------------------------------------------------------------
class SAM2AMGWrapper:
    """
    Wrapper for SAM2 Automatic Mask Generator.
    Segments "everything" in an image; no prompts. Returns list of mask dicts.
    """
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        model_cfg: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
    ):
        self.device = device
        self.amg = None
        print("Initializing SAM2 Automatic Mask Generator...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            ckpt = Path(checkpoint_path)
            cfg = Path(model_cfg)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            if not cfg.is_absolute():
                cfg = Path.cwd() / cfg
            if not ckpt.exists():
                print(f"SAM2 checkpoint not found at {ckpt}. Depth-mask branch disabled.")
                return
            if not cfg.exists():
                print(f"SAM2 config not found at {cfg}. Depth-mask branch disabled.")
                return
            model = build_sam2(str(cfg), str(ckpt))
            model.to(device)
            self.amg = SAM2AutomaticMaskGenerator(
                model,
                output_mode="binary_mask",
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            print("SAM2 AMG initialized.")
        except ImportError as e:
            print(f"SAM2 not available: {e}. Depth-mask branch disabled.")
        except Exception as e:
            print(f"Failed to load SAM2 AMG: {e}. Depth-mask branch disabled.")

    def generate(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run AMG on image (HWC uint8 RGB). Returns list of dicts with segmentation, bbox (xywh), area, predicted_iou, stability_score."""
        if self.amg is None:
            return []
        try:
            with torch.inference_mode():
                anns = self.amg.generate(image_rgb)
            return anns if anns else []
        except Exception as e:
            print(f"SAM2 AMG generate failed: {e}")
            return []

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
class SceneUnderstandingPipeline:
    def __init__(
        self,
        depth_estimator: DepthEstimator,
        intrinsics: Optional[Dict] = None,
        depth_mask_modes: Optional[List[str]] = None,
        config: Optional[Any] = None,
    ):
        """
        Args:
            depth_estimator: Initialized instance from depth.py
            intrinsics: Dict {'fx', 'fy', 'cx', 'cy'}. If None, will be auto-estimated.
            depth_mask_modes: List of "A", "B" for matching (A=detection-first, B=mask-first). Default ["A", "B"].
            config: Optional PreprocessConfig for SAM2 paths and AMG params. If None, depth-mask uses defaults.
        """
        self.depth_estimator = depth_estimator
        self.device = depth_estimator.device
        self.fixed_intrinsics = intrinsics
        if config is not None and hasattr(config, "depth_mask_matching_modes"):
            self.depth_mask_modes = list(config.depth_mask_matching_modes)
        else:
            self.depth_mask_modes = depth_mask_modes if depth_mask_modes is not None else ["A", "B"]
        self.config = config

        # Initialize Specific Models
        self.grit = GRiTWrapper(self.device)
        self.faster_rcnn = FasterRCNNWrapper(self.device)
        self.detr = DETRWrapper(self.device)
        self.pix2seq = Pix2SeqWrapper(self.device)
        self.sgsg = SGSGWrapper(self.device)
        self.pix2sg = Pix2SGWrapper(self.device)

        # SAM2 AMG (optional)
        self.sam2_wrapper = None
        if self.config is not None:
            ckpt = getattr(self.config, "sam2_checkpoint_path", "sam2/checkpoints/sam2.1_hiera_large.pt")
            cfg = getattr(self.config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
            pts = getattr(self.config, "sam2_amg_points_per_side", 32)
            iou = getattr(self.config, "sam2_amg_pred_iou_thresh", 0.8)
            stab = getattr(self.config, "sam2_amg_stability_score_thresh", 0.95)
            self.sam2_wrapper = SAM2AMGWrapper(self.device, ckpt, cfg, points_per_side=pts, pred_iou_thresh=iou, stability_score_thresh=stab)
        else:
            self.sam2_wrapper = SAM2AMGWrapper(
                self.device,
                "sam2/checkpoints/sam2.1_hiera_large.pt",
                "configs/sam2.1/sam2.1_hiera_l.yaml",
            )

    def _estimate_intrinsics(self, width: int, height: int) -> Dict[str, float]:
        """Estimate intrinsics assuming a 60-degree FOV."""
        fov_degrees = 60.0
        f_x = (width / 2) / np.tan(np.deg2rad(fov_degrees) / 2)
        return {"fx": f_x, "fy": f_x, "cx": width / 2, "cy": height / 2}

    def _back_project(self, u: int, v: int, z: float, K: Dict[str, float]) -> Dict[str, float]:
        x = (u - K['cx']) * z / K['fx']
        y = (v - K['cy']) * z / K['fy']
        return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3)}

    @staticmethod
    def _bbox_iou_xyxy(box1: List[float], box2: List[float]) -> float:
        """IoU of two boxes in xyxy format [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    @staticmethod
    def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
        x, y, w, h = bbox_xywh[:4]
        return [x, y, x + w, y + h]

    def _match_detection_first(
        self,
        detections: List[Dict],
        amg_masks: List[Dict],
        iou_thresh: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Option A: For each detection, assign best AMG mask by bbox IoU. Returns list of matched objects."""
        out = []
        for det in detections:
            det_xyxy = det["bbox"]
            best_iou = iou_thresh
            best_idx = -1
            best_mask = None
            for idx, amg in enumerate(amg_masks):
                seg = amg.get("segmentation")
                if seg is None:
                    continue
                amg_xywh = amg.get("bbox", [0, 0, 0, 0])
                amg_xyxy = self._xywh_to_xyxy(amg_xywh)
                iou = self._bbox_iou_xyxy(det_xyxy, amg_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    best_mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            if best_idx >= 0 and best_mask is not None:
                out.append({
                    "detection": det,
                    "mask": best_mask,
                    "sam2_mask_index": best_idx,
                    "mask_bbox_xyxy": self._xywh_to_xyxy(amg_masks[best_idx]["bbox"]),
                })
        return out

    def _match_mask_first(
        self,
        amg_masks: List[Dict],
        detections: List[Dict],
        iou_thresh: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Option B: For each AMG mask, assign best detection by bbox IoU. Returns list of matched objects (mask + optional detection)."""
        out = []
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            amg_xywh = amg.get("bbox", [0, 0, 0, 0])
            amg_xyxy = self._xywh_to_xyxy(amg_xywh)
            best_iou = iou_thresh
            best_det = None
            for det in detections:
                iou = self._bbox_iou_xyxy(det["bbox"], amg_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            out.append({
                "mask": mask,
                "sam2_mask_index": idx,
                "detection": best_det,
                "mask_bbox_xyxy": amg_xyxy,
            })
        return out

    def _mask_depth_stats_and_3d(
        self,
        metric_depth: np.ndarray,
        K: Dict[str, float],
        mask: np.ndarray,
        detection: Optional[Dict] = None,
    ) -> tuple:
        """
        Compute depth stats and 3D from mask region only.
        Returns (depth_stats_dict, coordinates_3d_from_mask, mask_centroid_2d).
        """
        h, w = metric_depth.shape[:2]
        mask_bin = (np.asarray(mask) > 0)
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_bin > 0)
        depth_at_mask = metric_depth[mask_bin]
        depth_at_mask = depth_at_mask[np.isfinite(depth_at_mask)]
        if depth_at_mask.size == 0:
            depth_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "num_pixels": 0}
            coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
            centroid = [w // 2, h // 2]
            return depth_stats, coords_3d, centroid
        depth_stats = {
            "min": round(float(np.min(depth_at_mask)), 4),
            "max": round(float(np.max(depth_at_mask)), 4),
            "mean": round(float(np.mean(depth_at_mask)), 4),
            "median": round(float(np.median(depth_at_mask)), 4),
            "std": round(float(np.std(depth_at_mask)), 4),
            "num_pixels": int(np.sum(mask_bin)),
        }
        ys, xs = np.where(mask_bin)
        cy = int(round(float(np.mean(ys))))
        cx = int(round(float(np.mean(xs))))
        cy = max(0, min(h - 1, cy))
        cx = max(0, min(w - 1, cx))
        z_val = float(np.median(depth_at_mask))
        coords_3d = self._back_project(cx, cy, z_val, K)
        return depth_stats, coords_3d, [int(cx), int(cy)]

    def _match_relations(self, current_obj: Dict, triplets: List[Dict], source_name: str, all_objs: List[Dict]):
        """Matches extracted triplets to the spatial objects detected."""
        current_label = current_obj['label'].lower()
        
        for triplet in triplets:
            # Check if this triplet's subject matches our current object
            if triplet['sub'] in current_label or current_label in triplet['sub']:
                target_label = triplet['obj']
                target_id = None
                
                # Find target object in our detected list
                for other in all_objs:
                    if target_label in other['label'].lower() or other['label'].lower() in target_label:
                        target_id = other['id']
                        break
                
                relation_entry = {
                    "predicate": triplet['pred'],
                    "target_id": target_id if target_id else f"external_{target_label}"
                }
                
                if source_name not in current_obj["sources"]:
                    current_obj["sources"][source_name] = {"relations": []}
                
                current_obj["sources"][source_name]["relations"].append(relation_entry)

    def _save_model_output(self, image: np.ndarray, detections: List[Dict], output_dir: Path, filename_stem: str, model_name: str):
        """Helper to save individual model results."""
        if not detections:
            return
            
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(model_dir / f"{filename_stem}_{model_name}.json", 'w') as f:
            json.dump(detections, f, indent=2)
            
        # Save Viz
        viz = image.copy()
        for det in detections:
            bbox = det['bbox']
            label = f"{det['label']} {det.get('conf', 0):.2f}"
            cv2.rectangle(viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(viz, label, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(str(model_dir / f"{filename_stem}_{model_name}_viz.png"), viz)

    def _save_depth_map_image(self, metric_depth: np.ndarray, path: Path) -> None:
        """Save full depth as colormap PNG."""
        d = metric_depth.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min < 1e-8:
            vis = np.zeros((*d.shape, 3), dtype=np.uint8)
        else:
            vis = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(path), vis)

    def _save_sam2_segmentation_image(self, amg_masks: List[Dict], h: int, w: int, path: Path) -> None:
        """Save RGB image with one color per AMG mask (all masks)."""
        out = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(42)
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))
            out[mask > 0] = color
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    def _save_depth_mask_mapping_image(
        self,
        metric_depth: np.ndarray,
        matched_objects: List[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Save depth colormap only where matched masks are; rest black."""
        h, w = metric_depth.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)
        for obj in matched_objects:
            m = obj.get("mask")
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask |= (m > 0)
        d = metric_depth.astype(np.float32)
        if np.any(combined_mask):
            d_min, d_max = float(d[combined_mask].min()), float(d[combined_mask].max())
        else:
            d_min, d_max = 0.0, 1.0
        if not np.any(combined_mask):
            vis = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            if d_max - d_min < 1e-8:
                d_max = d_min + 1.0
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            norm = np.clip(norm, 0, 255)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            vis[combined_mask] = colored[combined_mask]
        cv2.imwrite(str(path), vis)

    def _build_depth_mask_json(
        self,
        image_path: str,
        path_stem: str,
        timestamp: str,
        image_size: List[int],
        matching_mode: str,
        depth_map_path: str,
        depth_map_image_path: str,
        depth_global_min: float,
        depth_global_max: float,
        depth_global_mean: float,
        segmentation_map_image_path: str,
        num_auto_masks: int,
        mapping_image_path: str,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the depth+mask details JSON structure."""
        return {
            "metadata": {
                "image_path": image_path,
                "image_stem": path_stem,
                "timestamp": timestamp,
                "image_size": image_size,
                "matching_mode": matching_mode,
            },
            "depth": {
                "depth_map_path": depth_map_path,
                "depth_map_image_path": depth_map_image_path,
                "depth_min": depth_global_min,
                "depth_max": depth_global_max,
                "depth_mean": depth_global_mean,
            },
            "segmentation": {
                "model": "SAM2",
                "mode": "automatic_mask_generator",
                "segmentation_map_image_path": segmentation_map_image_path,
                "num_auto_masks": num_auto_masks,
                "match_strategy": "iou_with_detection_bbox",
            },
            "depth_mask": {
                "mapping_image_path": mapping_image_path,
                "objects": objects,
            },
        }

    def process_image(self, image_path: str, output_dir: str):
        path = Path(image_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Processing scene understanding for: {path.name}")
        
        img_bgr = cv2.imread(str(path))
        if img_bgr is None: return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # 1. Intrinsics
        K = self.fixed_intrinsics if self.fixed_intrinsics else self._estimate_intrinsics(w, h)

        # 2. Depth Estimation (Depth Anything V2)
        norm_depth_small = self.depth_estimator.backend.infer(img_rgb)
        norm_depth = cv2.resize(norm_depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        metric_scale = 10.0 # Adjust based on Metric model vs Relative model usage
        metric_depth = norm_depth * metric_scale
        
        # Save Depth
        depth_dir = out / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_dir / f"{path.stem}_depth_metric.npy", metric_depth)

        # 3. Object Detection & Scene Graph Generation
        # Run all detection models
        all_detections = []
        
        # GRiT
        grit_out = self.grit.predict(img_bgr)
        self._save_model_output(img_bgr, grit_out, out, path.stem, "grit")
        all_detections.extend(grit_out)
        
        # Faster R-CNN
        frcnn_out = self.faster_rcnn.predict(img_bgr)
        self._save_model_output(img_bgr, frcnn_out, out, path.stem, "faster_rcnn")
        all_detections.extend(frcnn_out)
        
        # DETR
        detr_out = self.detr.predict(img_bgr)
        self._save_model_output(img_bgr, detr_out, out, path.stem, "detr")
        all_detections.extend(detr_out)
        
        # Pix2Seq
        pix2seq_out = self.pix2seq.predict(img_bgr)
        self._save_model_output(img_bgr, pix2seq_out, out, path.stem, "pix2seq")
        all_detections.extend(pix2seq_out)

        # Relations
        sgsg_out = self.sgsg.predict(img_bgr)
        
        # Log discovered relations
        if sgsg_out:
            print(f"SGSG found {len(sgsg_out)} relations.")
        else:
            print("SGSG found no relations.")
            
        pix2sg_out = self.pix2sg.predict(img_bgr)

        # 3b. SAM2 AMG + Depth-Mask outputs (depth map image, SAM2 seg, depth+mask mapping + JSON per mode)
        scene_graph_dir = out / "scene_graph"
        scene_graph_dir.mkdir(parents=True, exist_ok=True)
        depth_mask_dir = scene_graph_dir / "depth_mask"
        masks_dir = scene_graph_dir / "masks"
        save_per_object_masks = getattr(self.config, "save_per_object_masks", True) if self.config else True

        self._save_depth_map_image(metric_depth, scene_graph_dir / f"{path.stem}_depth_map.png")
        depth_map_image_rel = f"scene_graph/{path.stem}_depth_map.png"
        depth_map_npy_rel = f"depth/{path.stem}_depth_metric.npy"
        depth_global_min = float(np.min(metric_depth))
        depth_global_max = float(np.max(metric_depth))
        depth_global_mean = float(np.mean(metric_depth))

        amg_masks = []
        if self.sam2_wrapper is not None and getattr(self.sam2_wrapper, "amg", None) is not None:
            amg_masks = self.sam2_wrapper.generate(img_rgb)
        seg_map_rel = f"scene_graph/{path.stem}_sam2_segmentation.png"
        if amg_masks:
            self._save_sam2_segmentation_image(amg_masks, h, w, scene_graph_dir / f"{path.stem}_sam2_segmentation.png")

        for mode in self.depth_mask_modes:
            if mode == "A":
                matched = self._match_detection_first(all_detections, amg_masks) if amg_masks else []
            elif mode == "B":
                matched = self._match_mask_first(amg_masks, all_detections) if amg_masks else []
            else:
                continue
            json_objects = []
            for i, mobj in enumerate(matched):
                det = mobj.get("detection")
                mask = mobj.get("mask")
                if mask is None:
                    continue
                depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d(metric_depth, K, mask, det)
                if det:
                    obj_id = f"obj_{i}_{det.get('source_model', 'det')}"
                    label = det.get("label", "unknown")
                    bbox = det.get("bbox", [])
                    source_model = det.get("source_model", "")
                else:
                    obj_id = f"obj_{i}_mask"
                    label = "unlabeled"
                    bbox = list(mobj.get("mask_bbox_xyxy", [0, 0, 0, 0]))
                    source_model = ""
                mask_path_rel = f"scene_graph/masks/{path.stem}_obj_{i}_mask_{mode}.png"
                masked_depth_path_rel = f"scene_graph/masks/{path.stem}_obj_{i}_masked_depth_{mode}.npy"
                if save_per_object_masks:
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_uint8 = (np.asarray(mask).astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
                    if mask_uint8.shape[:2] != (h, w):
                        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(masks_dir / f"{path.stem}_obj_{i}_mask_{mode}.png"), mask_uint8)
                    mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (h, w) else mask.astype(np.float32)
                    masked_depth = metric_depth.astype(np.float32) * (mask_resized > 0)
                    np.save(masks_dir / f"{path.stem}_obj_{i}_masked_depth_{mode}.npy", masked_depth)
                obj_entry = {
                    "id": obj_id,
                    "label": label,
                    "bbox": bbox,
                    "source_model": source_model,
                    "sam2_mask_index": mobj.get("sam2_mask_index", -1),
                    "mask_path": mask_path_rel,
                    "masked_depth_path": masked_depth_path_rel,
                    "depth_stats": depth_stats,
                    "coordinates_3d_from_mask": coords_3d,
                    "mask_centroid_2d": centroid,
                }
                json_objects.append(obj_entry)
            mapping_path = scene_graph_dir / f"{path.stem}_depth_mask_mapping_{mode}.png"
            self._save_depth_mask_mapping_image(metric_depth, matched, mapping_path)
            mapping_rel = f"scene_graph/{path.stem}_depth_mask_mapping_{mode}.png"
            depth_mask_dir.mkdir(parents=True, exist_ok=True)
            dm_json = self._build_depth_mask_json(
                image_path=str(path.resolve()),
                path_stem=path.stem,
                timestamp=timestamp,
                image_size=[w, h],
                matching_mode=mode,
                depth_map_path=depth_map_npy_rel,
                depth_map_image_path=depth_map_image_rel,
                depth_global_min=depth_global_min,
                depth_global_max=depth_global_max,
                depth_global_mean=depth_global_mean,
                segmentation_map_image_path=seg_map_rel,
                num_auto_masks=len(amg_masks),
                mapping_image_path=mapping_rel,
                objects=json_objects,
            )
            with open(depth_mask_dir / f"{path.stem}_depth_mask_{mode}.json", "w") as f:
                json.dump(dm_json, f, indent=2)

        # 4. Integration & Projection
        objects_3d = []
        
        # Ensure objects directory exists
        objects_dir = scene_graph_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        
        for i, det in enumerate(all_detections):
            bbox = det['bbox']
            cx, cy = (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2
            cx, cy = max(0, min(w-1, cx)), max(0, min(h-1, cy))
            
            z_val = metric_depth[cy, cx]
            xyz = self._back_project(cx, cy, z_val, K)
            
            # Identify source
            src = det.get('source_model', 'Unknown')
            
            # --- Extract specific object depth map ---
            x1, y1, x2, y2 = bbox
            x1, x2 = max(0, int(x1)), min(w, int(x2))
            y1, y2 = max(0, int(y1)), min(h, int(y2))
            
            object_depth_map = metric_depth[y1:y2, x1:x2]
            
            obj_depth_filename = f"{path.stem}_obj_{i}_{src}_depth.png"
            obj_depth_path = objects_dir / obj_depth_filename
            
            if object_depth_map.size > 0:
                # Normalize for visualization
                d_min, d_max = object_depth_map.min(), object_depth_map.max()
                if d_max - d_min > 1e-6:
                    norm_obj_depth = ((object_depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    norm_obj_depth = np.zeros_like(object_depth_map, dtype=np.uint8)
                
                norm_obj_depth = cv2.applyColorMap(norm_obj_depth, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(obj_depth_path), norm_obj_depth)
            
            obj_entry = {
                "id": f"obj_{i}_{src}",
                "label": det['label'],
                "conf": det['conf'],
                "bbox": bbox,
                "coordinates_3d": xyz,
                "depth_map_path": f"objects/{obj_depth_filename}",
                "sources": {
                    src: {"caption": det['caption']},
                    "SGSG": {"relations": []},
                    "Pix2SG": {"relations": []}
                }
            }
            objects_3d.append(obj_entry)

        # 5. Relation Matching
        for obj in objects_3d:
            self._match_relations(obj, sgsg_out, "SGSG", objects_3d)
            self._match_relations(obj, pix2sg_out, "Pix2SG", objects_3d)

        # 6. Save Combined Scene Graph
        scene_graph_dir = out / "scene_graph"
        scene_graph_dir.mkdir(parents=True, exist_ok=True)

        # Convert Intrinsics values to native python types for JSON serialization
        if K:
            K_serializable = {k: float(v) for k, v in K.items()}
        else:
            K_serializable = None

        metadata = {
            "timestamp": timestamp,
            "intrinsics": K_serializable,
            "models": ["GRiT", "FasterRCNN", "DETR", "Pix2Seq", "SGSG", "Pix2SG"],
        }
        if "A" in self.depth_mask_modes:
            metadata["depth_mask_json_A"] = f"scene_graph/depth_mask/{path.stem}_depth_mask_A.json"
        if "B" in self.depth_mask_modes:
            metadata["depth_mask_json_B"] = f"scene_graph/depth_mask/{path.stem}_depth_mask_B.json"
        json_output = {"metadata": metadata, "objects": objects_3d}
        with open(scene_graph_dir / f"{path.stem}_scene.json", 'w') as f:
            json.dump(json_output, f, indent=2)

        # 7. Visualization
        viz = img_bgr.copy()
        for obj in objects_3d:
            x, y, z = obj['coordinates_3d'].values()
            bbox = obj['bbox']
            # Color based on source?
            # GRiT: Green, FasterRCNN: Blue, DETR: Red, etc.
            if "GRiT" in obj["sources"]: color = (0, 255, 0)
            elif "FasterRCNN" in obj["sources"]: color = (255, 0, 0)
            elif "DETR" in obj["sources"]: color = (0, 0, 255)
            else: color = (255, 255, 0)
            
            label = f"{obj['label']} ({list(obj['sources'].keys())[0]})"
            cv2.rectangle(viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(viz, label, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw Relations
            cx_a, cy_a = (int(bbox[0])+int(bbox[2]))//2, (int(bbox[1])+int(bbox[3]))//2
            for source in ["SGSG", "Pix2SG"]:
                if source in obj["sources"]:
                    for rel in obj["sources"][source]["relations"]:
                        # Find target object
                        target = None
                        target_id = rel['target_id']
                        
                        # Handle external targets (not in objects_3d list)
                        if isinstance(target_id, str) and target_id.startswith("external_"):
                             # Can't draw line to unknown location
                             continue
                             
                        # Find internal target
                        target = next((o for o in objects_3d if o['id'] == target_id), None)
                        
                        if target:
                            bbox_b = target['bbox']
                            cx_b, cy_b = (int(bbox_b[0])+int(bbox_b[2]))//2, (int(bbox_b[1])+int(bbox_b[3]))//2
                            line_color = (0, 255, 255)
                            cv2.line(viz, (int(cx_a), int(cy_a)), (int(cx_b), int(cy_b)), line_color, 1)
                            mx, my = (int(cx_a) + int(cx_b))//2, (int(cy_a) + int(cy_b))//2
                            cv2.putText(viz, rel['predicate'], (int(mx), int(my)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

        cv2.imwrite(str(scene_graph_dir / f"{path.stem}_3d_viz.png"), viz)
        print(f"Results saved to {out}")

if __name__ == "__main__":
    from config import PreprocessConfig
    
    # Simple test with a dummy image
    print("Testing pipeline initialization...")
    cfg = PreprocessConfig()
    
    try:
        depth_estimator = DepthEstimator(cfg)
        pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)
        
        # Test image
        test_img_path = "IMG_3062.png" 
        if os.path.exists(test_img_path):
            print(f"Processing test image: {test_img_path}")
            pipeline.process_image(test_img_path, "output_test")
        else:
            # Fallback to dummy
            print("Test image not found, using dummy blank image")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite("dummy_test.png", img)
            pipeline.process_image("dummy_test.png", "output_test")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()