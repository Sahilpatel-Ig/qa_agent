import cv2
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
import logging
import tempfile
import os
from pathlib import Path
import json
from detection.ensemble import EnsembleDetector
from ai_agent.gemini_agent import GeminiAgent
from ai_agent.langchain_tools import DetectionAnalysisTools
from .visualizer import VideoVisualizer
from utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing class for drowning detection analysis"""

    def __init__(self,
                 enable_ai_analysis: bool = True,
                 output_dir: str = "output"):

        self.enable_ai_analysis = enable_ai_analysis
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.ensemble_detector = EnsembleDetector()
        self.visualizer = VideoVisualizer()
        self.metrics_calculator = MetricsCalculator()

        # Tracking ID persistence
        self.track_id_map = {}  # Maps detection to persistent track ID

        # Initialize AI components if enabled
        if enable_ai_analysis:
            try:
                self.gemini_agent = GeminiAgent()
                self.analysis_tools = DetectionAnalysisTools(self.gemini_agent)
                logger.info("AI analysis components initialized")
            except (ImportError, Exception) as e:
                logger.warning(f"AI analysis disabled due to error: {e}")
                self.enable_ai_analysis = False
                self.gemini_agent = None
                self.analysis_tools = None

        logger.info("Video Processor initialized successfully")

    def process_video(self,
                      video_path: str,
                      custom_yolo_results: Optional[List[Dict]] = None,
                      progress_callback: Optional[Callable] = None) -> Dict:
        """Process video with comprehensive analysis"""

        logger.info(f"Starting video processing: {video_path}")

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            video_info = self._get_video_info(cap)
            logger.info(f"Video info: {video_info}")



            # Process frames
            frame_results = []
            frame_count = 0
            total_frames = int(video_info['total_frames'])

            # Reset ensemble detector tracking and ID mapping
            self.ensemble_detector.reset_tracking()
            self.track_id_map.clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Store original frame for detection
                original_frame = frame.copy()
                # Get custom YOLO detections for this frame
                frame_custom_detections = []
                if custom_yolo_results and frame_count < len(custom_yolo_results):
                    frame_custom_detections = custom_yolo_results[frame_count]

                # Process frame
                frame_result = self._process_frame(
                    original_frame,
                    frame_count,
                    frame_custom_detections
                )

                frame_results.append(frame_result)
                frame_count += 1

                # Update progress
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)

                # Optional: Limit processing for testing
                # if frame_count >= 100:  # Process only first 100 frames
                #     break

            cap.release()

            # Generate comprehensive analysis
            analysis_result = self._generate_analysis(
                frame_results,
                video_info,
                video_path
            )

            logger.info(
                f"Video processing completed. Processed {frame_count} frames")
            return analysis_result

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            if 'cap' in locals():
                cap.release()
            raise

    def _get_video_info(self, cap: cv2.VideoCapture) -> Dict:
        """Extract video metadata"""
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }



    def _process_frame(self,
                       frame: np.ndarray,
                       frame_number: int,
                       custom_detections: List[Dict]) -> Dict:
        """Process single frame with all detection models"""

        try:
            # Run ensemble detection on original frame
            detection_results = self.ensemble_detector.detect_all(
                frame,
                custom_yolo_detections=custom_detections
            )

            # Apply persistent tracking IDs
            detection_results = self._apply_persistent_tracking(
                detection_results)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_frame_metrics(
                detection_results)

            # AI analysis (if enabled)
            ai_analysis = None
            if self.enable_ai_analysis and self.gemini_agent:
                try:
                    ai_analysis = self.gemini_agent.analyze_detection_quality(
                        detection_results,
                        frame_number,
                        "drowning detection"
                    )
                except Exception as e:
                    logger.warning(
                        f"AI analysis failed for frame {frame_number}: {e}")

            # Compile frame result
            frame_result = {
                'frame_number': frame_number,
                'frame_shape': frame.shape,
                'detection_results': detection_results,
                'metrics': metrics,
                'ai_analysis': ai_analysis.dict() if ai_analysis else None,
                'timestamp': frame_number / 30.0  # Assume 30 FPS if not available
            }

            return frame_result

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return {
                'frame_number': frame_number,
                'error': str(e),
                'detection_results': {},
                'metrics': {},
                'ai_analysis': None
            }

    def _generate_analysis(self,
                           frame_results: List[Dict],
                           video_info: Dict,
                           video_path: str) -> Dict:
        """Generate comprehensive video analysis"""

        try:
            # Calculate overall metrics
            overall_metrics = self.metrics_calculator.calculate_video_metrics(
                frame_results)

            # Generate AI report (if enabled)
            ai_report = None
            langchain_analysis = None
            if self.enable_ai_analysis and self.gemini_agent:
                try:
                    ai_report = self.gemini_agent.generate_qa_report(
                        frame_results,
                        video_info
                    )

                    # Also run LangChain analysis
                    if self.analysis_tools:
                        langchain_analysis = self.analysis_tools.analyze_video_quality(
                            frame_results
                        )

                except Exception as e:
                    logger.warning(f"AI report generation failed: {e}")
                    ai_report = f"AI analysis failed: {str(e)}"
                    langchain_analysis = "LangChain analysis unavailable"

            # Compile final analysis
            analysis = {
                'video_info': video_info,
                'video_path': video_path,
                'processing_summary': {
                    'total_frames_processed': len(frame_results),
                    'frames_with_errors': sum(1 for fr in frame_results if 'error' in fr),
                    'processing_time': 0,  # Would track actual processing time
                },
                'frame_results': frame_results,
                'overall_metrics': overall_metrics,
                'ai_analysis': {
                    'gemini_report': ai_report,
                    'langchain_analysis': langchain_analysis if self.enable_ai_analysis else None
                },
                'export_paths': {}
            }

            # Save analysis to file
            analysis_path = self.output_dir / \
                f"analysis_{Path(video_path).stem}.json"
            with open(analysis_path, 'w') as f:
                # Create a copy without the large frame_results for JSON export
                json_analysis = analysis.copy()
                json_analysis['frame_results'] = f"Processed {len(frame_results)} frames"
                json.dump(json_analysis, f, indent=2, default=str)

            analysis['export_paths']['analysis_json'] = str(analysis_path)

            return analysis

        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return {
                'error': str(e),
                'video_info': video_info,
                'frame_results': frame_results
            }

    def _apply_persistent_tracking(self, detection_results: Dict) -> Dict:
        """Apply persistent tracking IDs to maintain person identity throughout video"""
        try:
            # Get tracking results
            tracking_results = detection_results.get('tracking', [])

            if not tracking_results:
                return detection_results

            # Update all detection types with persistent track IDs
            for detection_type in ['custom_yolo', 'ensemble_yolo', 'vit', 'mediapipe']:
                if detection_type in detection_results.get('detections', {}):
                    detections = detection_results['detections'][detection_type]
                    detection_results['detections'][detection_type] = self._assign_track_ids_to_detections(
                        detections, tracking_results
                    )

            # Update ensemble detections
            if 'ensemble_detections' in detection_results:
                detection_results['ensemble_detections'] = self._assign_track_ids_to_detections(
                    detection_results['ensemble_detections'], tracking_results
                )

            # Update missed and additional detections
            if 'missed_detections' in detection_results:
                detection_results['missed_detections'] = self._assign_track_ids_to_detections(
                    detection_results['missed_detections'], tracking_results
                )

            if 'additional_detections' in detection_results:
                detection_results['additional_detections'] = self._assign_track_ids_to_detections(
                    detection_results['additional_detections'], tracking_results
                )

            return detection_results

        except Exception as e:
            logger.warning(f"Error applying persistent tracking: {e}")
            return detection_results

    def _assign_track_ids_to_detections(self, detections: List[Dict], tracking_results: List[Dict]) -> List[Dict]:
        """Assign track IDs to detections based on spatial overlap"""
        if not detections or not tracking_results:
            return detections

        updated_detections = []

        for detection in detections:
            best_track_id = None
            best_iou = 0.0

            # Find best matching track based on IoU
            for track in tracking_results:
                iou = self._calculate_detection_iou(
                    detection['bbox'], track['bbox'])
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_track_id = track['track_id']

            # Add track ID to detection
            detection_copy = detection.copy()
            detection_copy['track_id'] = best_track_id
            updated_detections.append(detection_copy)

        return updated_detections

    def _calculate_detection_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            x1_2, y1_2, x2_2, y2_2 = box2[:4]

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
        except Exception:
            return 0.0

    def export_enhanced_video(self,
                              video_path: str,
                              analysis_results: Dict,
                              output_path: Optional[str] = None) -> str:
        """Export video with enhanced visualizations"""

        if not output_path:
            output_path = self.output_dir / \
                f"enhanced_{Path(video_path).stem}.mp4"

        try:
            logger.info(f"Exporting enhanced video to: {output_path}")

            # Open original video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer with better codec
            # Use XVID codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            # Check if video writer initialized properly
            if not out.isOpened():
                logger.error("Failed to initialize video writer")
                raise RuntimeError("Could not initialize video writer")

            frame_count = 0
            frame_results = analysis_results.get('frame_results', [])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get detection results for this frame
                if frame_count < len(frame_results):
                    frame_result = frame_results[frame_count]
                    detection_results = frame_result.get(
                        'detection_results', {})

                    # Add visualizations
                    enhanced_frame = self.visualizer.draw_detections(
                        frame, detection_results)
                    enhanced_frame = self.visualizer.add_frame_info(
                        enhanced_frame, frame_result)
                else:
                    enhanced_frame = frame

                out.write(enhanced_frame)
                frame_count += 1

            cap.release()
            out.release()

            logger.info(f"Enhanced video exported successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error exporting enhanced video: {e}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            raise

    def process_batch(self,
                      video_paths: List[str],
                      progress_callback: Optional[Callable] = None) -> Dict:
        """Process multiple videos in batch"""

        batch_results = {}
        total_videos = len(video_paths)

        for i, video_path in enumerate(video_paths):
            try:
                logger.info(
                    f"Processing video {i+1}/{total_videos}: {video_path}")

                # Individual video progress callback
                def video_progress(progress, frame_num, total_frames):
                    overall_progress = (i + progress) / total_videos
                    if progress_callback:
                        progress_callback(
                            overall_progress, f"Video {i+1}/{total_videos}, Frame {frame_num}/{total_frames}")

                # Process video
                result = self.process_video(
                    video_path, progress_callback=video_progress)
                batch_results[video_path] = result

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                batch_results[video_path] = {'error': str(e)}

        return batch_results

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.ensemble_detector.cleanup()
            logger.info("Video processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
