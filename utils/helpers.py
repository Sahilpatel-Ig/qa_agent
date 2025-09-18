import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import tempfile
import shutil

logger = logging.getLogger(__name__)


def validate_video_file(file_path: str) -> Dict[str, Any]:
    """Validate video file and return metadata"""
    
    result = {
        'valid': False,
        'error': None,
        'metadata': {}
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result['error'] = "File does not exist"
            return result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            result['error'] = "File is empty"
            return result
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            result['error'] = "Cannot open video file"
            return result
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Validate basic properties
        if width <= 0 or height <= 0:
            result['error'] = "Invalid video dimensions"
            return result
        
        if fps <= 0:
            result['error'] = "Invalid frame rate"
            return result
        
        # Check file format
        file_ext = Path(file_path).suffix.lower()
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        if file_ext not in supported_formats:
            result['error'] = f"Unsupported format: {file_ext}"
            return result
        
        # Set metadata
        result['metadata'] = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'file_size': file_size,
            'format': file_ext,
            'file_path': file_path
        }
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
    
    return result


def parse_custom_yolo_results(results_data: Any) -> List[List[Dict]]:
    """Parse custom YOLO results from various input formats"""
    
    try:
        # Handle different input formats
        if isinstance(results_data, str):
            # JSON string
            data = json.loads(results_data)
        elif isinstance(results_data, (list, dict)):
            # Already parsed data
            data = results_data
        else:
            raise ValueError("Unsupported data format")
        
        # Convert to standardized format
        parsed_results = []
        
        if isinstance(data, list):
            # List of frames
            for frame_data in data:
                frame_detections = []
                
                if isinstance(frame_data, list):
                    # List of detections for this frame
                    for det in frame_data:
                        frame_detections.append(standardize_detection(det))
                elif isinstance(frame_data, dict):
                    # Single detection or frame info
                    if 'detections' in frame_data:
                        for det in frame_data['detections']:
                            frame_detections.append(standardize_detection(det))
                    else:
                        frame_detections.append(standardize_detection(frame_data))
                
                parsed_results.append(frame_detections)
        
        elif isinstance(data, dict):
            # Single frame or grouped data
            if 'frames' in data:
                for frame_data in data['frames']:
                    frame_detections = []
                    for det in frame_data.get('detections', []):
                        frame_detections.append(standardize_detection(det))
                    parsed_results.append(frame_detections)
            else:
                # Single frame
                frame_detections = []
                for det in data.get('detections', []):
                    frame_detections.append(standardize_detection(det))
                parsed_results.append(frame_detections)
        
        return parsed_results
        
    except Exception as e:
        logger.error(f"Error parsing custom YOLO results: {e}")
        return []


def standardize_detection(detection: Dict) -> Dict:
    """Standardize detection format"""
    
    standardized = {
        'bbox': [0, 0, 0, 0],
        'confidence': 0.0,
        'class_id': 'person',
        'model': 'custom_yolo'
    }
    
    try:
        # Handle different bbox formats
        if 'bbox' in detection:
            bbox = detection['bbox']
        elif 'box' in detection:
            bbox = detection['box']
        elif all(k in detection for k in ['x1', 'y1', 'x2', 'y2']):
            bbox = [detection['x1'], detection['y1'], detection['x2'], detection['y2']]
        elif all(k in detection for k in ['x', 'y', 'w', 'h']):
            # Convert from xywh to xyxy
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            bbox = [x, y, x + w, y + h]
        else:
            raise ValueError("No valid bbox format found")
        
        # Ensure bbox is list of integers
        standardized['bbox'] = [int(x) for x in bbox]
        
        # Handle confidence
        if 'confidence' in detection:
            standardized['confidence'] = float(detection['confidence'])
        elif 'conf' in detection:
            standardized['confidence'] = float(detection['conf'])
        elif 'score' in detection:
            standardized['confidence'] = float(detection['score'])
        
        # Handle class
        if 'class_id' in detection:
            standardized['class_id'] = detection['class_id']
        elif 'class' in detection:
            standardized['class_id'] = detection['class']
        elif 'label' in detection:
            standardized['class_id'] = detection['label']
        
        # Handle model name
        if 'model' in detection:
            standardized['model'] = detection['model']
        
    except Exception as e:
        logger.warning(f"Error standardizing detection: {e}")
    
    return standardized


def save_results_to_json(results: Dict, output_path: str) -> bool:
    """Save analysis results to JSON file"""
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = convert_to_json_serializable(results)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        return False


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format"""
    
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, 'dict'):  # Pydantic models
        return obj.dict()
    else:
        return obj


def create_temp_file(suffix: str = ".mp4") -> str:
    """Create temporary file and return path"""
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name


def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {file_path}: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string"""
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"


def calculate_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """Calculate center point of bounding box"""
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def calculate_bbox_area(bbox: List[int]) -> int:
    """Calculate area of bounding box"""
    
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def validate_detection_format(detection: Dict) -> bool:
    """Validate detection dictionary format"""
    
    required_fields = ['bbox', 'confidence', 'class_id', 'model']
    
    # Check required fields
    if not all(field in detection for field in required_fields):
        return False
    
    # Validate bbox
    bbox = detection['bbox']
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    
    # Validate bbox values
    x1, y1, x2, y2 = bbox
    if not (isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and
            isinstance(x2, (int, float)) and isinstance(y2, (int, float))):
        return False
    
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Validate confidence
    confidence = detection['confidence']
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        return False
    
    return True


def merge_detection_lists(detection_lists: List[List[Dict]]) -> List[Dict]:
    """Merge multiple detection lists into single list"""
    
    merged = []
    for detection_list in detection_lists:
        if isinstance(detection_list, list):
            merged.extend(detection_list)
    
    return merged


def filter_detections_by_confidence(detections: List[Dict], 
                                   min_confidence: float) -> List[Dict]:
    """Filter detections by minimum confidence threshold"""
    
    return [
        det for det in detections 
        if det.get('confidence', 0) >= min_confidence
    ]


def get_video_thumbnail(video_path: str, 
                       timestamp: float = 0.0,
                       output_size: Tuple[int, int] = (320, 240)) -> Optional[np.ndarray]:
    """Extract thumbnail from video at specified timestamp"""
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Seek to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Resize if needed
        if output_size:
            frame = cv2.resize(frame, output_size)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {e}")
        return None
