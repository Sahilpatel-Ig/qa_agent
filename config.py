import os
from typing import Dict, Any

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Pose detection thresholds
POSE_BASE_THRESHOLDS = {
    "visibility": 0.5,
    "presence": 0.5
}

# Model configurations
MODEL_CONFIGS = {
    "yolov11": {
        "model_path": "yolov8n.pt",
        "confidence": 0.5,
        "iou": 0.45,
        "input_size": 640
    },
    "vit": {
        "model_name": "facebook/detr-resnet-50",
        "confidence": 0.6
    },
    "mediapipe": {
        "confidence": 0.5,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5
    }
}

# DeepSORT configuration
DEEPSORT_CONFIG = {
    "max_dist": 0.2,
    "min_confidence": 0.3,
    "nms_max_overlap": 1.0,
    "max_iou_distance": 0.7,
    "max_age": 70,
    "n_init": 3,
    "nn_budget": 100
}

# Video processing
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
MAX_VIDEO_SIZE_MB = 500

# Colors for visualization (BGR format)
COLORS = {
    "custom_yolo": (0, 255, 0),      # Green for custom YOLO
    "missed_detection": (0, 0, 255),  # Red for missed detections
    "vit_detection": (255, 0, 0),     # Blue for ViT detections
    "mediapipe_detection": (0, 255, 255),  # Yellow for MediaPipe
    "tracking_id": (255, 255, 255)    # White for tracking IDs
}

# Gemini API configuration
GEMINI_MODEL = "gemini-2.5-flash"
# Replace with your actual API key
GEMINI_API_KEY = "AIzaSyDHun1ooMRjs68Hem7Giz363dJuvFMKfaw"

# Performance settings
USE_GPU = True
HALF_PRECISION = True
BATCH_SIZE = 4
