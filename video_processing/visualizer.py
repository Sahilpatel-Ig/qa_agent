import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config import COLORS

logger = logging.getLogger(__name__)


class VideoVisualizer:
    """Video visualization utilities for detection results"""
    
    def __init__(self):
        self.colors = COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.text_thickness = 1
        
    def draw_detections(self, frame: np.ndarray, detection_results: Dict) -> np.ndarray:
        """Draw all detections on frame with color coding"""
        
        try:
            # Create a copy to avoid modifying original
            vis_frame = frame.copy()
            
            # Draw custom YOLO detections (green)
            custom_detections = detection_results.get('detections', {}).get('custom_yolo', [])
            for det in custom_detections:
                vis_frame = self._draw_single_detection(
                    vis_frame, det, self.colors['custom_yolo'], "YOLO"
                )
            
            # Draw missed detections (red)
            missed_detections = detection_results.get('missed_detections', [])
            for det in missed_detections:
                vis_frame = self._draw_single_detection(
                    vis_frame, det, self.colors['missed_detection'], f"MISSED-{det.get('model', 'UNK').upper()}"
                )
            
            # ViT and MediaPipe detections are still processed but not visualized
            # (Detection continues in background for ensemble analysis)
            
            # Draw tracking IDs
            tracking_results = detection_results.get('tracking', [])
            for track in tracking_results:
                vis_frame = self._draw_tracking_id(vis_frame, track)
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return frame
    
    def _draw_single_detection(self, 
                              frame: np.ndarray, 
                              detection: Dict, 
                              color: Tuple[int, int, int],
                              label_prefix: str) -> np.ndarray:
        """Draw a single detection with bounding box and label"""
        
        try:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Prepare label
            label = f"{label_prefix}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.text_thickness
            )
            
            # Draw label background
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            cv2.rectangle(
                frame,
                (x1, label_y - label_height - 5),
                (x1 + label_width + 5, label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.text_thickness
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing single detection: {e}")
            return frame
    
    def _draw_tracking_id(self, frame: np.ndarray, track: Dict) -> np.ndarray:
        """Draw tracking ID on detection"""
        
        try:
            bbox = track['bbox']
            track_id = track['track_id']
            x1, y1, x2, y2 = bbox
            
            # Draw tracking ID in center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            id_label = f"ID:{track_id}"
            
            # Draw ID background circle
            cv2.circle(frame, (center_x, center_y), 15, self.colors['tracking_id'], -1)
            cv2.circle(frame, (center_x, center_y), 15, (0, 0, 0), 1)
            
            # Draw ID text
            (text_width, text_height), _ = cv2.getTextSize(
                str(track_id), self.font, 0.4, 1
            )
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            cv2.putText(
                frame,
                str(track_id),
                (text_x, text_y),
                self.font,
                0.4,
                (0, 0, 0),  # Black text
                1
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing tracking ID: {e}")
            return frame
    
    def add_frame_info(self, frame: np.ndarray, frame_result: Dict) -> np.ndarray:
        """Add frame information overlay"""
        
        try:
            # Get frame info
            frame_number = frame_result.get('frame_number', 0)
            metrics = frame_result.get('metrics', {})
            ai_analysis = frame_result.get('ai_analysis', {})
            
            # Prepare info text
            info_lines = [
                f"Frame: {frame_number}",
                f"Custom YOLO: {metrics.get('custom_yolo_count', 0)}",
                f"Ensemble: {metrics.get('total_ensemble_count', 0)}",
                f"Missed: {metrics.get('missed_count', 0)}",
                f"Tracking: {metrics.get('tracking_count', 0)}"
            ]
            
            # Add AI analysis info if available
            if ai_analysis:
                risk_level = ai_analysis.get('risk_level', 'unknown')
                severity = ai_analysis.get('missed_detection_severity', 0)
                info_lines.extend([
                    f"Risk: {risk_level.upper()}",
                    f"Severity: {severity}/5"
                ])
            
            # Note about hidden detections
            info_lines.append("(ViT/MP: Hidden)")
            
            # Draw info panel
            panel_height = len(info_lines) * 25 + 20
            panel_width = 250
            
            # Draw background panel
            cv2.rectangle(
                frame,
                (10, 10),
                (panel_width, panel_height),
                (0, 0, 0),  # Black background
                -1
            )
            cv2.rectangle(
                frame,
                (10, 10),
                (panel_width, panel_height),
                (255, 255, 255),  # White border
                1
            )
            
            # Draw info text
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(
                    frame,
                    line,
                    (20, y_pos),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),  # White text
                    self.text_thickness
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding frame info: {e}")
            return frame
    
    def create_comparison_view(self, 
                              frame: np.ndarray,
                              custom_detections: List[Dict],
                              ensemble_detections: List[Dict]) -> np.ndarray:
        """Create side-by-side comparison view"""
        
        try:
            height, width = frame.shape[:2]
            
            # Create side-by-side frame
            comparison_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Left side: Custom YOLO only
            left_frame = frame.copy()
            for det in custom_detections:
                left_frame = self._draw_single_detection(
                    left_frame, det, self.colors['custom_yolo'], "YOLO"
                )
            
            # Right side: Only show YOLO and missed detections
            right_frame = frame.copy()
            for det in ensemble_detections:
                model = det.get('model', 'unknown')
                # Only show YOLO detections, hide ViT and MediaPipe
                if model not in ['vit', 'mediapipe']:
                    color = self.colors['custom_yolo']
                    label = model.upper()
                    right_frame = self._draw_single_detection(
                        right_frame, det, color, label
                    )
            
            # Combine frames
            comparison_frame[:, :width] = left_frame
            comparison_frame[:, width:] = right_frame
            
            # Add labels
            cv2.putText(
                comparison_frame,
                "Custom YOLO",
                (20, 30),
                self.font,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                comparison_frame,
                "Ensemble Models",
                (width + 20, 30),
                self.font,
                0.8,
                (255, 255, 255),
                2
            )
            
            return comparison_frame
            
        except Exception as e:
            logger.error(f"Error creating comparison view: {e}")
            return frame
    
    def draw_confidence_histogram(self, 
                                 detection_results: Dict,
                                 width: int = 400,
                                 height: int = 200) -> np.ndarray:
        """Draw confidence score histogram"""
        
        try:
            # Create blank image
            hist_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Collect all confidence scores
            all_confidences = []
            for model_detections in detection_results.get('detections', {}).values():
                for det in model_detections:
                    all_confidences.append(det['confidence'])
            
            if not all_confidences:
                cv2.putText(
                    hist_img,
                    "No detections",
                    (width//2 - 50, height//2),
                    self.font,
                    0.6,
                    (255, 255, 255),
                    1
                )
                return hist_img
            
            # Create histogram
            bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
            hist, _ = np.histogram(all_confidences, bins=bins)
            
            # Normalize histogram
            if hist.max() > 0:
                hist = hist.astype(float) / hist.max()
            
            # Draw bars
            bar_width = width // len(hist)
            for i, val in enumerate(hist):
                bar_height = int(val * (height - 40))
                x1 = i * bar_width
                x2 = (i + 1) * bar_width
                y1 = height - 20
                y2 = y1 - bar_height
                
                cv2.rectangle(hist_img, (x1, y2), (x2, y1), (0, 255, 0), -1)
                cv2.rectangle(hist_img, (x1, y2), (x2, y1), (255, 255, 255), 1)
            
            # Add labels
            cv2.putText(
                hist_img,
                "Confidence Distribution",
                (10, 20),
                self.font,
                0.5,
                (255, 255, 255),
                1
            )
            
            return hist_img
            
        except Exception as e:
            logger.error(f"Error creating confidence histogram: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)
