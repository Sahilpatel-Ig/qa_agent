import json
import logging
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from config import GEMINI_MODEL, GEMINI_API_KEY

# Optional imports for AI functionality
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

logger = logging.getLogger(__name__)


class DetectionAnalysis(BaseModel):
    """Pydantic model for detection analysis results"""
    missed_detection_severity: int  # 1-5 scale
    confidence_assessment: float  # 0-1 scale
    reasoning: str
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high", "critical"


class GeminiAgent:
    """Gemini-powered AI agent for detection quality analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not HAS_GEMINI:
            logger.error("Gemini Agent requires google-generativeai library")
            raise ImportError("Missing required library: google-generativeai")
            
        self.api_key = api_key or GEMINI_API_KEY
        
        if not self.api_key or self.api_key == "default_key":
            raise ValueError("Valid Gemini API key is required")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            logger.info("Gemini Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Agent: {e}")
            raise
    
    def analyze_detection_quality(self, 
                                detection_results: Dict,
                                frame_number: int,
                                video_context: str = "drowning detection") -> DetectionAnalysis:
        """Analyze detection quality and provide insights"""
        
        try:
            # Prepare analysis data
            analysis_data = self._prepare_analysis_data(detection_results, frame_number, video_context)
            
            # Create system prompt
            system_prompt = self._create_analysis_prompt()
            
            # Generate analysis
            prompt = f"{system_prompt}\n\nAnalysis data: {json.dumps(analysis_data, indent=2)}"
            response = self.model.generate_content(prompt)
            
            if response.text:
                analysis_json = json.loads(response.text)
                return DetectionAnalysis(**analysis_json)
            else:
                return self._get_default_analysis()
                
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return self._get_default_analysis()
    
    def _prepare_analysis_data(self, 
                              detection_results: Dict, 
                              frame_number: int,
                              video_context: str) -> Dict:
        """Prepare structured data for analysis"""
        
        stats = self._calculate_detection_stats(detection_results)
        
        analysis_data = {
            "frame_number": frame_number,
            "video_context": video_context,
            "detection_counts": {
                "custom_yolo": len(detection_results['detections']['custom_yolo']),
                "ensemble_models": len(detection_results['ensemble_detections']),
                "missed_detections": len(detection_results['missed_detections']),
                "additional_detections": len(detection_results['additional_detections']),
                "tracking_active": len(detection_results['tracking'])
            },
            "confidence_scores": {
                model: stats['avg_confidence'].get(model, 0.0)
                for model in ['custom_yolo', 'ensemble_yolo', 'vit', 'mediapipe']
            },
            "missed_detection_details": [
                {
                    "model": det.get('model', 'unknown'),
                    "confidence": det.get('confidence', 0.0),
                    "bbox_area": self._calculate_bbox_area(det['bbox'])
                }
                for det in detection_results['missed_detections']
            ],
            "detection_consistency": self._analyze_detection_consistency(detection_results)
        }
        
        return analysis_data
    
    def _create_analysis_prompt(self) -> str:
        """Create system prompt for detection analysis"""
        return """
You are an expert AI agent specializing in computer vision quality assurance for drowning detection systems.
Your role is to analyze detection results and provide insights on missed detections and system performance.

Analyze the provided detection data and assess:

1. MISSED DETECTION SEVERITY (1-5 scale):
   - 1: Minor miss, low impact
   - 2: Noticeable miss, moderate impact
   - 3: Significant miss, concerning
   - 4: Major miss, high risk
   - 5: Critical miss, immediate attention required

2. CONFIDENCE ASSESSMENT (0-1 scale):
   - Overall confidence in the detection system performance
   - Consider detection consistency across models

3. REASONING:
   - Explain your analysis
   - Identify patterns in missed detections
   - Assess model performance differences

4. RECOMMENDATIONS:
   - Specific actionable recommendations
   - Model tuning suggestions
   - System improvements

5. RISK LEVEL:
   - "low": System performing well
   - "medium": Some concerns, monitoring needed
   - "high": Significant issues, intervention recommended
   - "critical": Immediate action required

Consider the context of drowning detection where missing a person could be life-threatening.
Be thorough but concise in your analysis.
"""
    
    def _calculate_detection_stats(self, detection_results: Dict) -> Dict:
        """Calculate detailed detection statistics"""
        stats = {
            'total_detections': 0,
            'avg_confidence': {},
            'model_agreement': 0.0
        }
        
        # Count total detections
        for model_name, detections in detection_results['detections'].items():
            stats['total_detections'] += len(detections)
            
            # Calculate average confidence per model
            if detections:
                avg_conf = sum(det['confidence'] for det in detections) / len(detections)
                stats['avg_confidence'][model_name] = avg_conf
            else:
                stats['avg_confidence'][model_name] = 0.0
        
        return stats
    
    def _calculate_bbox_area(self, bbox: List[int]) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _analyze_detection_consistency(self, detection_results: Dict) -> Dict:
        """Analyze consistency between different detection models"""
        consistency_metrics = {
            "cross_model_agreement": 0.0,
            "confidence_variance": 0.0,
            "detection_density": 0.0
        }
        
        # Calculate cross-model agreement
        total_detections = sum(
            len(detections) for detections in detection_results['detections'].values()
        )
        
        if total_detections > 0:
            unique_detections = len(detection_results['ensemble_detections'])
            consistency_metrics["cross_model_agreement"] = unique_detections / total_detections
        
        # Calculate confidence variance
        all_confidences = []
        for detections in detection_results['detections'].values():
            all_confidences.extend([det['confidence'] for det in detections])
        
        if all_confidences:
            import numpy as np
            consistency_metrics["confidence_variance"] = float(np.var(all_confidences))
        
        return consistency_metrics
    
    def _get_default_analysis(self) -> DetectionAnalysis:
        """Return default analysis when Gemini analysis fails"""
        return DetectionAnalysis(
            missed_detection_severity=3,
            confidence_assessment=0.5,
            reasoning="Analysis unavailable due to API error. Default assessment provided.",
            recommendations=["Review detection parameters", "Check model performance"],
            risk_level="medium"
        )
    
    def generate_qa_report(self, 
                          video_analysis_results: List[Dict],
                          video_metadata: Dict) -> str:
        """Generate comprehensive QA report for entire video"""
        
        try:
            # Prepare report data
            report_data = self._prepare_report_data(video_analysis_results, video_metadata)
            
            # Create report prompt
            prompt = f"""
Generate a comprehensive Quality Assurance report for drowning detection video analysis.

Video Metadata:
{json.dumps(video_metadata, indent=2)}

Analysis Results Summary:
{json.dumps(report_data, indent=2)}

Please provide:
1. Executive Summary
2. Detection Performance Overview
3. Critical Issues Identified
4. Model Comparison Analysis
5. Recommendations for Improvement
6. Risk Assessment
7. Action Items

Make the report professional, detailed, and actionable for QA teams.
"""
            
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            
            return response.text or "Report generation failed"
            
        except Exception as e:
            logger.error(f"Error generating QA report: {e}")
            return f"Report generation failed: {str(e)}"
    
    def _prepare_report_data(self, 
                           video_analysis_results: List[Dict],
                           video_metadata: Dict) -> Dict:
        """Prepare aggregated data for report generation"""
        
        total_frames = len(video_analysis_results)
        total_missed = sum(len(result.get('missed_detections', [])) for result in video_analysis_results)
        total_custom_detections = sum(len(result.get('detections', {}).get('custom_yolo', [])) for result in video_analysis_results)
        
        report_data = {
            "video_info": video_metadata,
            "total_frames_analyzed": total_frames,
            "total_custom_detections": total_custom_detections,
            "total_missed_detections": total_missed,
            "miss_rate": total_missed / max(total_custom_detections, 1),
            "frames_with_misses": sum(1 for result in video_analysis_results if result.get('missed_detections', [])),
            "average_detections_per_frame": total_custom_detections / max(total_frames, 1),
            "critical_frames": [
                i for i, result in enumerate(video_analysis_results)
                if len(result.get('missed_detections', [])) > 2
            ]
        }
        
        return report_data
