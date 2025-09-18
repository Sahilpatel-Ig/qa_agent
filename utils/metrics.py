import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for detection analysis"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_frame_metrics(self, detection_results: Dict) -> Dict:
        """Calculate metrics for a single frame"""
        
        try:
            metrics = {}
            
            # Basic counts
            detections = detection_results.get('detections', {})
            metrics['custom_yolo_count'] = len(detections.get('custom_yolo', []))
            metrics['ensemble_yolo_count'] = len(detections.get('ensemble_yolo', []))
            metrics['vit_count'] = len(detections.get('vit', []))
            metrics['mediapipe_count'] = len(detections.get('mediapipe', []))
            metrics['total_ensemble_count'] = len(detection_results.get('ensemble_detections', []))
            metrics['missed_count'] = len(detection_results.get('missed_detections', []))
            metrics['additional_count'] = len(detection_results.get('additional_detections', []))
            metrics['tracking_count'] = len(detection_results.get('tracking', []))
            
            # Confidence statistics
            metrics['confidence_stats'] = self._calculate_confidence_stats(detections)
            
            # Detection density (detections per area)
            frame_shape = detection_results.get('frame_shape', (480, 640, 3))
            frame_area = frame_shape[0] * frame_shape[1]
            metrics['detection_density'] = metrics['total_ensemble_count'] / frame_area
            
            # Coverage metrics
            metrics['coverage_metrics'] = self._calculate_coverage_metrics(detection_results)
            
            # Quality scores
            metrics['quality_scores'] = self._calculate_quality_scores(detection_results)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating frame metrics: {e}")
            return {}
    
    def calculate_video_metrics(self, frame_results: List[Dict]) -> Dict:
        """Calculate aggregated metrics for entire video"""
        
        try:
            video_metrics = {
                'total_frames': len(frame_results),
                'frames_with_errors': 0,
                'detection_summary': defaultdict(list),
                'confidence_summary': defaultdict(list),
                'temporal_consistency': {},
                'performance_metrics': {},
                'quality_assessment': {}
            }
            
            # Aggregate frame-level metrics
            for frame_result in frame_results:
                if 'error' in frame_result:
                    video_metrics['frames_with_errors'] += 1
                    continue
                
                metrics = frame_result.get('metrics', {})
                
                # Collect detection counts
                for key in ['custom_yolo_count', 'ensemble_yolo_count', 'vit_count', 
                           'mediapipe_count', 'total_ensemble_count', 'missed_count', 
                           'additional_count', 'tracking_count']:
                    if key in metrics:
                        video_metrics['detection_summary'][key].append(metrics[key])
                
                # Collect confidence scores
                confidence_stats = metrics.get('confidence_stats', {})
                for model, stats in confidence_stats.items():
                    if 'mean' in stats:
                        video_metrics['confidence_summary'][model].append(stats['mean'])
            
            # Calculate summary statistics
            video_metrics['detection_statistics'] = self._calculate_summary_stats(
                video_metrics['detection_summary']
            )
            video_metrics['confidence_statistics'] = self._calculate_summary_stats(
                video_metrics['confidence_summary']
            )
            
            # Calculate temporal consistency
            video_metrics['temporal_consistency'] = self._calculate_temporal_consistency(
                frame_results
            )
            
            # Calculate performance metrics
            video_metrics['performance_metrics'] = self._calculate_performance_metrics(
                video_metrics['detection_summary']
            )
            
            # Calculate overall quality assessment
            video_metrics['quality_assessment'] = self._calculate_quality_assessment(
                video_metrics
            )
            
            return video_metrics
            
        except Exception as e:
            logger.error(f"Error calculating video metrics: {e}")
            return {}
    
    def _calculate_confidence_stats(self, detections: Dict) -> Dict:
        """Calculate confidence statistics for each model"""
        
        confidence_stats = {}
        
        for model_name, model_detections in detections.items():
            if not model_detections:
                confidence_stats[model_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
                continue
            
            confidences = [det['confidence'] for det in model_detections]
            
            confidence_stats[model_name] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'count': len(confidences)
            }
        
        return confidence_stats
    
    def _calculate_coverage_metrics(self, detection_results: Dict) -> Dict:
        """Calculate spatial coverage metrics"""
        
        coverage_metrics = {
            'total_area_covered': 0.0,
            'average_detection_size': 0.0,
            'coverage_overlap': 0.0
        }
        
        try:
            frame_shape = detection_results.get('frame_shape', (480, 640, 3))
            frame_area = frame_shape[0] * frame_shape[1]
            
            all_detections = []
            for detections in detection_results.get('detections', {}).values():
                all_detections.extend(detections)
            
            if not all_detections:
                return coverage_metrics
            
            # Calculate total area covered by detections
            total_detection_area = 0
            detection_sizes = []
            
            for det in all_detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                total_detection_area += area
                detection_sizes.append(area)
            
            coverage_metrics['total_area_covered'] = total_detection_area / frame_area
            coverage_metrics['average_detection_size'] = np.mean(detection_sizes) if detection_sizes else 0
            
            # Calculate overlap (simplified)
            coverage_metrics['coverage_overlap'] = self._calculate_bbox_overlap(all_detections)
            
        except Exception as e:
            logger.error(f"Error calculating coverage metrics: {e}")
        
        return coverage_metrics
    
    def _calculate_bbox_overlap(self, detections: List[Dict]) -> float:
        """Calculate average overlap between bounding boxes"""
        
        if len(detections) < 2:
            return 0.0
        
        total_overlap = 0.0
        comparisons = 0
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                overlap = self._calculate_iou(det1['bbox'], det2['bbox'])
                total_overlap += overlap
                comparisons += 1
        
        return total_overlap / comparisons if comparisons > 0 else 0.0
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
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
    
    def _calculate_quality_scores(self, detection_results: Dict) -> Dict:
        """Calculate quality scores for the frame"""
        
        quality_scores = {
            'detection_consistency': 0.0,
            'confidence_quality': 0.0,
            'coverage_quality': 0.0,
            'overall_quality': 0.0
        }
        
        try:
            # Detection consistency score
            total_ensemble = len(detection_results.get('ensemble_detections', []))
            missed = len(detection_results.get('missed_detections', []))
            
            if total_ensemble > 0:
                quality_scores['detection_consistency'] = 1.0 - (missed / total_ensemble)
            else:
                quality_scores['detection_consistency'] = 1.0
            
            # Confidence quality score
            all_confidences = []
            for detections in detection_results.get('detections', {}).values():
                all_confidences.extend([det['confidence'] for det in detections])
            
            if all_confidences:
                quality_scores['confidence_quality'] = np.mean(all_confidences)
            
            # Coverage quality (simplified)
            coverage_metrics = detection_results.get('coverage_metrics', {})
            quality_scores['coverage_quality'] = min(1.0, coverage_metrics.get('total_area_covered', 0) * 2)
            
            # Overall quality (weighted average)
            weights = [0.4, 0.4, 0.2]  # consistency, confidence, coverage
            scores = [
                quality_scores['detection_consistency'],
                quality_scores['confidence_quality'],
                quality_scores['coverage_quality']
            ]
            quality_scores['overall_quality'] = np.average(scores, weights=weights)
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
        
        return quality_scores
    
    def _calculate_summary_stats(self, data_dict: Dict[str, List]) -> Dict:
        """Calculate summary statistics for aggregated data"""
        
        summary = {}
        
        for key, values in data_dict.items():
            if not values:
                summary[key] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0,
                    'count': 0
                }
                continue
            
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'total': float(np.sum(values)),
                'count': len(values)
            }
        
        return summary
    
    def _calculate_temporal_consistency(self, frame_results: List[Dict]) -> Dict:
        """Calculate temporal consistency metrics"""
        
        consistency_metrics = {
            'detection_variance': 0.0,
            'tracking_stability': 0.0,
            'confidence_stability': 0.0
        }
        
        try:
            # Extract detection counts over time
            detection_counts = []
            tracking_counts = []
            avg_confidences = []
            
            for frame_result in frame_results:
                if 'error' in frame_result:
                    continue
                
                metrics = frame_result.get('metrics', {})
                detection_counts.append(metrics.get('total_ensemble_count', 0))
                tracking_counts.append(metrics.get('tracking_count', 0))
                
                # Calculate average confidence for frame
                confidence_stats = metrics.get('confidence_stats', {})
                frame_confidences = []
                for model_stats in confidence_stats.values():
                    if 'mean' in model_stats:
                        frame_confidences.append(model_stats['mean'])
                
                if frame_confidences:
                    avg_confidences.append(np.mean(frame_confidences))
                else:
                    avg_confidences.append(0.0)
            
            # Calculate variance measures
            if detection_counts:
                consistency_metrics['detection_variance'] = float(np.var(detection_counts))
            
            if tracking_counts:
                # Tracking stability = 1 - normalized variance
                if np.mean(tracking_counts) > 0:
                    normalized_var = np.var(tracking_counts) / np.mean(tracking_counts)
                    consistency_metrics['tracking_stability'] = max(0.0, 1.0 - normalized_var)
            
            if avg_confidences:
                consistency_metrics['confidence_stability'] = 1.0 - float(np.var(avg_confidences))
            
        except Exception as e:
            logger.error(f"Error calculating temporal consistency: {e}")
        
        return consistency_metrics
    
    def _calculate_performance_metrics(self, detection_summary: Dict) -> Dict:
        """Calculate performance metrics"""
        
        performance = {
            'detection_rate': 0.0,
            'miss_rate': 0.0,
            'precision_estimate': 0.0,
            'recall_estimate': 0.0
        }
        
        try:
            custom_total = sum(detection_summary.get('custom_yolo_count', []))
            ensemble_total = sum(detection_summary.get('total_ensemble_count', []))
            missed_total = sum(detection_summary.get('missed_count', []))
            
            # Detection rate
            if custom_total > 0:
                performance['detection_rate'] = (custom_total - missed_total) / custom_total
            
            # Miss rate
            if ensemble_total > 0:
                performance['miss_rate'] = missed_total / ensemble_total
            
            # Precision estimate (simplified)
            if custom_total > 0:
                performance['precision_estimate'] = min(1.0, custom_total / max(1, ensemble_total))
            
            # Recall estimate (simplified)
            if ensemble_total > 0:
                performance['recall_estimate'] = custom_total / ensemble_total
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        
        return performance
    
    def _calculate_quality_assessment(self, video_metrics: Dict) -> Dict:
        """Calculate overall quality assessment"""
        
        assessment = {
            'overall_score': 0.0,
            'reliability_score': 0.0,
            'consistency_score': 0.0,
            'confidence_score': 0.0,
            'risk_level': 'medium',
            'recommendations': []
        }
        
        try:
            performance = video_metrics.get('performance_metrics', {})
            temporal = video_metrics.get('temporal_consistency', {})
            
            # Calculate component scores
            detection_rate = performance.get('detection_rate', 0.5)
            miss_rate = performance.get('miss_rate', 0.5)
            tracking_stability = temporal.get('tracking_stability', 0.5)
            confidence_stability = temporal.get('confidence_stability', 0.5)
            
            # Reliability score
            assessment['reliability_score'] = (detection_rate + (1 - miss_rate)) / 2
            
            # Consistency score
            assessment['consistency_score'] = (tracking_stability + confidence_stability) / 2
            
            # Confidence score (from average confidence)
            confidence_stats = video_metrics.get('confidence_statistics', {})
            avg_confidences = []
            for model_stats in confidence_stats.values():
                if isinstance(model_stats, dict) and 'mean' in model_stats:
                    avg_confidences.append(model_stats['mean'])
            
            if avg_confidences:
                assessment['confidence_score'] = np.mean(avg_confidences)
            
            # Overall score (weighted average)
            weights = [0.4, 0.3, 0.3]  # reliability, consistency, confidence
            scores = [
                assessment['reliability_score'],
                assessment['consistency_score'],
                assessment['confidence_score']
            ]
            assessment['overall_score'] = np.average(scores, weights=weights)
            
            # Risk level assessment
            if assessment['overall_score'] >= 0.8:
                assessment['risk_level'] = 'low'
            elif assessment['overall_score'] >= 0.6:
                assessment['risk_level'] = 'medium'
            elif assessment['overall_score'] >= 0.4:
                assessment['risk_level'] = 'high'
            else:
                assessment['risk_level'] = 'critical'
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_recommendations(assessment, video_metrics)
            
        except Exception as e:
            logger.error(f"Error calculating quality assessment: {e}")
        
        return assessment
    
    def _generate_recommendations(self, assessment: Dict, video_metrics: Dict) -> List[str]:
        """Generate actionable recommendations based on assessment"""
        
        recommendations = []
        
        try:
            overall_score = assessment.get('overall_score', 0.5)
            reliability_score = assessment.get('reliability_score', 0.5)
            consistency_score = assessment.get('consistency_score', 0.5)
            confidence_score = assessment.get('confidence_score', 0.5)
            
            # Low overall score
            if overall_score < 0.6:
                recommendations.append("Overall detection quality is below acceptable threshold. Consider system review.")
            
            # Low reliability
            if reliability_score < 0.7:
                recommendations.append("High miss rate detected. Consider adjusting detection thresholds or retraining models.")
            
            # Low consistency
            if consistency_score < 0.7:
                recommendations.append("Temporal inconsistency detected. Review tracking parameters and video quality.")
            
            # Low confidence
            if confidence_score < 0.6:
                recommendations.append("Low detection confidence. Consider improving lighting conditions or model fine-tuning.")
            
            # High miss rate
            performance = video_metrics.get('performance_metrics', {})
            miss_rate = performance.get('miss_rate', 0)
            if miss_rate > 0.2:
                recommendations.append("Significant number of missed detections. Implement ensemble voting or lower thresholds.")
            
            # General recommendations
            if not recommendations:
                recommendations.append("System performing within acceptable parameters. Continue monitoring.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error.")
        
        return recommendations
