import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
import urllib.request
from config import CONFIDENCE_THRESHOLD, POSE_BASE_THRESHOLDS, MODEL_CONFIGS

# Optional imports with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from transformers import AutoFeatureExtractor, DetrForObjectDetection
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoFeatureExtractor = None
    DetrForObjectDetection = None

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTDetector:
    """Enhanced ViT Detector with GPU optimization and improved error handling"""

    def __init__(self,
                 model_name: str = "facebook/detr-resnet-50",
                 device: Optional["torch.device"] = None, # type: ignore
                 enable_gpu: bool = True,
                 half_precision: bool = False):

        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.error(
                "ViT Detector requires torch and transformers libraries")
            raise ImportError(
                "Missing required libraries: torch and/or transformers")

        # Device setup
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if enable_gpu and torch.cuda.is_available() else "cpu")

        self.half_precision = half_precision and self.device.type == 'cuda'

        logger.info(f"Initializing ViT Detector on {self.device}")

        try:
            # Load model and processor
            self.model = DetrForObjectDetection.from_pretrained(
                model_name).to(self.device)
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)

            # Enable half precision if requested and supported
            if self.half_precision:
                self.model = self.model.half()
                logger.info("ViT Detector: Half precision enabled")

            self.model.eval()

            # Warm up the model
            self._warmup()

            logger.info("ViT Detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ViT Detector: {e}")
            raise

    def _warmup(self):
        """Warm up the model for optimal performance"""
        try:
            dummy_input = np.random.randint(
                0, 255, (640, 480, 3), dtype=np.uint8)
            _ = self.detect(dummy_input)
            logger.info("ViT Detector warm-up completed")
        except Exception as e:
            logger.warning(f"ViT Detector warm-up failed: {e}")

    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """Enhanced detection with GPU optimization and error handling"""
        if confidence_threshold is None:
            confidence_threshold = CONFIDENCE_THRESHOLD

        try:
            # Validate input
            if frame is None or frame.size == 0:
                return []

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process inputs
            inputs = self.processor(images=frame_rgb, return_tensors="pt")

            # Move to device and apply half precision if enabled
            if self.half_precision:
                inputs = {k: v.to(self.device).half() if v.dtype.is_floating_point else v.to(self.device)
                          for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-processing
            target_sizes = torch.tensor([frame_rgb.shape[:2]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            # Extract person detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == 1:  # COCO label 1 = "person"
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()

                    # Validate bounding box
                    if self._is_valid_bbox([x1, y1, x2, y2], frame.shape):
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(score.cpu().numpy()),
                            "class_id": "person",
                            "model": "vit"
                        })

            return detections

        except Exception as e:
            logger.error(f"ViT Detection error: {e}")
            return []

    def _is_valid_bbox(self, bbox: List[float], frame_shape: Tuple[int, int, int]) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        return (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2 - x1) * (y2 - y1) > 100)


class YOLOv11Detector:
    """Enhanced YOLOv11 Detector with GPU optimization and robust error handling"""

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 device: Optional["torch.device"] = None, # type: ignore
                 half: bool = False,
                 enable_gpu: bool = True,
                 input_size: int = 640):

        # Device setup
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if enable_gpu and torch.cuda.is_available() else "cpu")

        self.half_precision = half and self.device.type == 'cuda'
        self.input_size = input_size

        logger.info(f"Initializing YOLOv11 Detector on {self.device}")

        try:
            # Load model
            self._load_model(model_path)

            # Warm up the model
            self._warmup()

            logger.info("YOLOv11 Detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize YOLOv11 Detector: {e}")
            raise

    def _load_model(self, model_path: str):
        """Load YOLOv11 model with proper error handling"""
        if not HAS_ULTRALYTICS:
            logger.error("YOLOv11 Detector requires ultralytics library")
            raise ImportError("Missing required library: ultralytics")

        if not os.path.exists(model_path):
            # Try common YOLOv11 model names
            common_models = ["yolov8n.pt", "yolov8s.pt",
                             "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
            for model_name in common_models:
                if os.path.exists(model_name):
                    model_path = model_name
                    logger.info(f"Using alternative model: {model_name}")
                    break
            else:
                # Download default model if none found
                logger.warning(
                    "No YOLOv11 model found, downloading yolov8n.pt")
                try:
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        "yolov8n.pt"
                    )
                    model_path = "yolov8n.pt"
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    raise

        try:
            # Try loading as YOLOv11/ultralytics model
            self.model = YOLO(model_path)
            if HAS_TORCH:
                self.model.to(self.device)
            self.is_ultralytics = True

            if self.half_precision and HAS_TORCH:
                self.model.model = self.model.model.half()
                logger.info("YOLOv11: Half precision enabled")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _warmup(self):
        """Warm up the model for optimal performance"""
        try:
            dummy_input = np.random.randint(
                0, 255, (640, 480, 3), dtype=np.uint8)
            _ = self.detect(dummy_input)
            logger.info("YOLOv11 Detector warm-up completed")
        except Exception as e:
            logger.warning(f"YOLOv11 Detector warm-up failed: {e}")

    def detect(self,
               image: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> List[Dict]:
        """Enhanced detection with GPU optimization"""
        try:
            # Validate input
            if image is None or image.size == 0:
                return []

            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                half=self.half_precision,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box data
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # Filter for person class (class 0 in COCO)
                        if cls == 0 and conf >= conf_threshold:
                            x1, y1, x2, y2 = map(int, xyxy)

                            if self._is_valid_bbox([x1, y1, x2, y2], image.shape):
                                detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": conf,
                                    "class_id": "person",
                                    "model": "yolov11"
                                })

            return detections

        except Exception as e:
            logger.error(f"YOLOv11 Detection error: {e}")
            return []

    def _is_valid_bbox(self, bbox: List[int], frame_shape: Tuple[int, int, int]) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        return (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2 - x1) * (y2 - y1) > 100)


class MediaPipeDetector:
    """MediaPipe pose detection for human detection"""

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        if not HAS_MEDIAPIPE:
            logger.error("MediaPipe Detector requires mediapipe library")
            raise ImportError("Missing required library: mediapipe")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        logger.info("Initializing MediaPipe Detector")

        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

            logger.info("MediaPipe Detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Detector: {e}")
            raise

    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """Detect persons using MediaPipe pose estimation"""
        if confidence_threshold is None:
            confidence_threshold = self.min_detection_confidence

        try:
            # Validate input
            if frame is None or frame.size == 0:
                return []

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = self.pose.process(frame_rgb)

            detections = []
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Calculate bounding box from pose landmarks
                h, w = frame.shape[:2]

                # Get all visible landmarks
                visible_landmarks = [
                    (lm.x * w, lm.y * h) for lm in landmarks
                    if lm.visibility > confidence_threshold
                ]

                if len(visible_landmarks) > 5:  # Need sufficient landmarks
                    # Calculate bounding box
                    x_coords = [pt[0] for pt in visible_landmarks]
                    y_coords = [pt[1] for pt in visible_landmarks]

                    x1 = max(0, int(min(x_coords) - 20))
                    y1 = max(0, int(min(y_coords) - 20))
                    x2 = min(w, int(max(x_coords) + 20))
                    y2 = min(h, int(max(y_coords) + 20))

                    # Calculate average confidence
                    avg_confidence = np.mean([
                        lm.visibility for lm in landmarks
                        if lm.visibility > confidence_threshold
                    ])

                    if self._is_valid_bbox([x1, y1, x2, y2], frame.shape):
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(avg_confidence),
                            "class_id": "person",
                            "model": "mediapipe",
                            "landmarks": [(lm.x * w, lm.y * h, lm.visibility) for lm in landmarks]
                        })

            return detections

        except Exception as e:
            logger.error(f"MediaPipe Detection error: {e}")
            return []

    def _is_valid_bbox(self, bbox: List[int], frame_shape: Tuple[int, int, int]) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        return (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2 - x1) * (y2 - y1) > 500)

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
