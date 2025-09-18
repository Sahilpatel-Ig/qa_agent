import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object"""
    id: int
    bbox: List[int]
    confidence: float
    class_id: str
    model: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: deque = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=30)
    
    def update(self, bbox: List[int], confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
    
    def predict(self):
        """Predict next position (simple linear prediction)"""
        if len(self.history) < 2:
            return self.bbox
        
        # Simple linear prediction based on last two positions
        last_bbox = list(self.history[-1])
        prev_bbox = list(self.history[-2])
        
        dx = last_bbox[0] - prev_bbox[0]
        dy = last_bbox[1] - prev_bbox[1]
        
        predicted_bbox = [
            last_bbox[0] + dx,
            last_bbox[1] + dy,
            last_bbox[2] + dx,
            last_bbox[3] + dy
        ]
        
        return predicted_bbox


class SimpleTracker:
    """Simplified object tracker for person detection"""
    
    def __init__(self, 
                 max_disappeared: int = 10,
                 max_distance: float = 100.0,
                 min_hits: int = 3):
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.dormant_tracks: Dict[int, Track] = {}  # Store disappeared tracks
        
        logger.info("Simple Tracker initialized")
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections"""
        if not detections:
            # Age existing tracks
            self._age_tracks()
            return self._get_confirmed_tracks()
        
        # Convert detections to standardized format
        detection_boxes = []
        for det in detections:
            detection_boxes.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'model': det['model']
            })
        
        # Predict current positions for existing tracks
        for track in self.tracks.values():
            track.time_since_update += 1
        
        # Associate detections to tracks
        matched_detections, unmatched_detections = self._associate_detections(detection_boxes)
        
        # Update matched tracks
        for track_id, detection in matched_detections:
            self.tracks[track_id].update(
                detection['bbox'],
                detection['confidence']
            )
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            # Check if this detection matches a dormant track
            reused_track_id = self._check_dormant_tracks(detection)
            if reused_track_id:
                self._reactivate_track(reused_track_id, detection)
            else:
                self._create_track(detection)
        
        # Remove old tracks
        self._age_tracks()
        
        return self._get_confirmed_tracks()
    
    def _associate_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, Dict]], List[Dict]]:
        """Associate detections to existing tracks using IoU"""
        if not self.tracks:
            return [], detections
        
        # Calculate IoU and distance matrix for better matching
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        distance_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id].predict()
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, detection['bbox'])
                distance_matrix[i, j] = self._calculate_center_distance(track_bbox, detection['bbox'])
        
        # Improved matching using both IoU and distance
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        # Create combined score (IoU + distance penalty)
        match_pairs = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                iou = iou_matrix[i, j]
                distance = distance_matrix[i, j]
                
                # Combined score: prioritize IoU but penalize large distances
                if iou > 0.05 or distance < self.max_distance:
                    score = iou - (distance / 1000.0)  # Distance penalty
                    match_pairs.append((score, i, j, iou, distance))
        
        match_pairs.sort(reverse=True)
        
        for score, track_idx, det_idx, iou, distance in match_pairs:
            if track_idx not in used_tracks and det_idx not in used_detections:
                if iou > 0.05 or distance < self.max_distance:
                    matched_pairs.append((track_ids[track_idx], detections[det_idx]))
                    used_tracks.add(track_idx)
                    used_detections.add(det_idx)
        
        # Get unmatched detections
        unmatched_detections = [
            detections[i] for i in range(len(detections)) 
            if i not in used_detections
        ]
        
        return matched_pairs, unmatched_detections
    
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
    
    def _calculate_center_distance(self, box1: List[int], box2: List[int]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def _create_track(self, detection: Dict):
        """Create new track from detection"""
        track = Track(
            id=self.next_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            class_id=detection['class_id'],
            model=detection['model'],
            age=1,
            hits=1,
            time_since_update=0
        )
        
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _age_tracks(self):
        """Move old tracks to dormant storage"""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            # Move to dormant tracks instead of deleting
            self.dormant_tracks[track_id] = self.tracks[track_id]
            del self.tracks[track_id]
    
    def _get_confirmed_tracks(self) -> List[Dict]:
        """Get tracks that have been confirmed (hit minimum number of times)"""
        confirmed_tracks = []
        
        for track in self.tracks.values():
            if track.hits >= self.min_hits or track.age <= 10:  # Longer grace period
                confirmed_tracks.append({
                    'track_id': track.id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'class_id': track.class_id,
                    'model': track.model,
                    'age': track.age,
                    'hits': track.hits
                })
        
        return confirmed_tracks
    
    def get_track_history(self, track_id: int) -> Optional[List[List[int]]]:
        """Get position history for a specific track"""
        if track_id in self.tracks:
            return list(self.tracks[track_id].history)
        return None
    
    def _check_dormant_tracks(self, detection: Dict) -> Optional[int]:
        """Check if detection matches any dormant track"""
        best_match_id = None
        best_iou = 0.0
        
        for track_id, dormant_track in self.dormant_tracks.items():
            iou = self._calculate_iou(detection['bbox'], dormant_track.bbox)
            if iou > best_iou and iou > 0.2:  # Lower threshold for dormant matching
                best_iou = iou
                best_match_id = track_id
        
        return best_match_id
    
    def _reactivate_track(self, track_id: int, detection: Dict):
        """Reactivate a dormant track"""
        dormant_track = self.dormant_tracks[track_id]
        dormant_track.update(detection['bbox'], detection['confidence'])
        dormant_track.time_since_update = 0
        
        # Move back to active tracks
        self.tracks[track_id] = dormant_track
        del self.dormant_tracks[track_id]
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.dormant_tracks.clear()
        self.next_id = 1
        logger.info("Tracker reset")
