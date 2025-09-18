import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .detectors import ViTDetector, YOLOv11Detector, MediaPipeDetector
from .tracking import SimpleTracker
from config import MODEL_CONFIGS, COLORS

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Ensemble detector combining multiple detection models"""
    
    def __init__(self, 
                 enable_yolo: bool = True,
                 enable_vit: bool = True,
                 enable_mediapipe: bool = True,
                 enable_tracking: bool = True):
        
        self.enable_yolo = enable_yolo
        self.enable_vit = enable_vit
        self.enable_mediapipe = enable_mediapipe
        self.enable_tracking = enable_tracking
        
        self.detectors = {}
        self.tracker = None
        
        logger.info("Initializing Ensemble Detector")
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Initialize tracker
        if enable_tracking:
            self.tracker = SimpleTracker(
                max_disappeared=30,  # Allow tracks to persist longer
                max_distance=150.0,  # Increase matching distance
                min_hits=1  # Reduce minimum hits requirement
            )
        
        logger.info("Ensemble Detector initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize all requested detectors"""
        try:
            if self.enable_yolo:
                try:
                    config = MODEL_CONFIGS['yolov11']
                    self.detectors['yolo'] = YOLOv11Detector(
                        model_path=config['model_path'],
                        input_size=config['input_size']
                    )
                    logger.info("YOLOv11 detector initialized")
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to initialize YOLOv11 detector: {e}")
                    self.enable_yolo = False
            
            if self.enable_vit:
                try:
                    config = MODEL_CONFIGS['vit']
                    self.detectors['vit'] = ViTDetector(
                        model_name=config['model_name']
                    )
                    logger.info("ViT detector initialized")
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to initialize ViT detector: {e}")
                    self.enable_vit = False
            
            if self.enable_mediapipe:
                try:
                    config = MODEL_CONFIGS['mediapipe']
                    self.detectors['mediapipe'] = MediaPipeDetector(
                        min_detection_confidence=config['detection_confidence'],
                        min_tracking_confidence=config['tracking_confidence']
                    )
                    logger.info("MediaPipe detector initialized")
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to initialize MediaPipe detector: {e}")
                    self.enable_mediapipe = False
                
        except Exception as e:
            logger.error(f"Error initializing detectors: {e}")
            # Don't raise, continue with whatever detectors we have
    
    def detect_all(self, frame: np.ndarray, custom_yolo_detections: List[Dict] = None) -> Dict:
        """Run all detectors on a frame and return comprehensive results"""
        results = {
            'frame_shape': frame.shape,
            'detections': {
                'custom_yolo': custom_yolo_detections or [],
                'ensemble_yolo': [],
                'vit': [],
                'mediapipe': []
            },
            'tracking': [],
            'missed_detections': [],
            'additional_detections': [],
            'ensemble_detections': []
        }
        
        try:
            # Run each detector
            if 'yolo' in self.detectors:
                detections = self.detectors['yolo'].detect(
                    frame, 
                    conf_threshold=MODEL_CONFIGS['yolov11']['confidence']
                )
                results['detections']['ensemble_yolo'] = detections
            
            if 'vit' in self.detectors:
                detections = self.detectors['vit'].detect(
                    frame, 
                    confidence_threshold=MODEL_CONFIGS['vit']['confidence']
                )
                results['detections']['vit'] = detections
            
            if 'mediapipe' in self.detectors:
                detections = self.detectors['mediapipe'].detect(
                    frame, 
                    confidence_threshold=MODEL_CONFIGS['mediapipe']['confidence']
                )
                results['detections']['mediapipe'] = detections
            
            # Combine all ensemble detections
            all_ensemble_detections = []
            for model_name, detections in results['detections'].items():
                if model_name != 'custom_yolo':
                    all_ensemble_detections.extend(detections)
            
            # Refine bounding boxes and apply NMS
            refined_detections = self._refine_bboxes(all_ensemble_detections)
            results['ensemble_detections'] = self._apply_nms(refined_detections, iou_threshold=0.3)
            
            # Compare with custom YOLO detections
            if custom_yolo_detections:
                missed, additional = self._compare_detections(
                    custom_yolo_detections, 
                    results['ensemble_detections']
                )
                results['missed_detections'] = missed
                results['additional_detections'] = additional
            
            # Apply tracking if enabled
            if self.tracker:
                # Combine custom YOLO and ensemble detections for tracking
                all_detections = (custom_yolo_detections or []) + results['ensemble_detections']
                results['tracking'] = self.tracker.update(all_detections)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            return results
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_detections(self, 
                           custom_detections: List[Dict], 
                           ensemble_detections: List[Dict],
                           iou_threshold: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """Compare custom YOLO detections with ensemble detections"""
        missed_detections = []
        additional_detections = []
        
        # Find ensemble detections not covered by custom YOLO (missed detections)
        for ensemble_det in ensemble_detections:
            is_covered = False
            for custom_det in custom_detections:
                iou = self._calculate_iou(ensemble_det['bbox'], custom_det['bbox'])
                if iou > iou_threshold:
                    is_covered = True
                    break
            
            if not is_covered:
                missed_detections.append(ensemble_det)
        
        # Find custom YOLO detections not covered by ensemble (additional detections)
        for custom_det in custom_detections:
            is_covered = False
            for ensemble_det in ensemble_detections:
                iou = self._calculate_iou(custom_det['bbox'], ensemble_det['bbox'])
                if iou > iou_threshold:
                    is_covered = True
                    break
            
            if not is_covered:
                additional_detections.append(custom_det)
        
        return missed_detections, additional_detections
    
    def _refine_bboxes(self, detections: List[Dict]) -> List[Dict]:
        """Refine bounding boxes to better separate people and detect small persons"""
        refined_detections = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Split large bounding boxes that likely contain multiple people
            if width > 150 and height > 200 and area > 40000:
                # Split horizontally if very wide
                if width / height > 2.0:
                    mid_x = x1 + width // 2
                    # Left half
                    refined_detections.append({
                        **det,
                        'bbox': [x1, y1, mid_x, y2],
                        'confidence': det['confidence'] * 0.8
                    })
                    # Right half
                    refined_detections.append({
                        **det,
                        'bbox': [mid_x, y1, x2, y2],
                        'confidence': det['confidence'] * 0.8
                    })
                else:
                    refined_detections.append(det)
            else:
                refined_detections.append(det)
        
        return refined_detections
    

    
    def get_detection_stats(self, results: Dict) -> Dict:
        """Calculate detection statistics"""
        stats = {
            'custom_yolo_count': len(results['detections']['custom_yolo']),
            'ensemble_yolo_count': len(results['detections']['ensemble_yolo']),
            'vit_count': len(results['detections']['vit']),
            'mediapipe_count': len(results['detections']['mediapipe']),
            'total_ensemble_count': len(results['ensemble_detections']),
            'missed_count': len(results['missed_detections']),
            'additional_count': len(results['additional_detections']),
            'tracking_count': len(results['tracking']),
            'avg_confidence': {}
        }
        
        # Calculate average confidence for each model
        for model_name, detections in results['detections'].items():
            if detections:
                avg_conf = np.mean([det['confidence'] for det in detections])
                stats['avg_confidence'][model_name] = float(avg_conf)
            else:
                stats['avg_confidence'][model_name] = 0.0
        
        return stats
    
    def reset_tracking(self):
        """Reset tracking state"""
        if self.tracker:
            self.tracker.reset()
    
    def cleanup(self):
        """Cleanup resources"""
        if 'mediapipe' in self.detectors:
            del self.detectors['mediapipe']
        logger.info("Ensemble detector cleanup completed")
