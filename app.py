import streamlit as st
import cv2
import os
import json
import tempfile
import traceback
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional

# Import our custom modules
from video_processing.processor import VideoProcessor
from detection.ensemble import EnsembleDetector
from ai_agent.gemini_agent import GeminiAgent
from utils.helpers import (
    validate_video_file,
    parse_custom_yolo_results,
    format_duration,
    format_file_size,
    create_temp_file,
    cleanup_temp_files
)

from utils.metrics import MetricsCalculator
from config import MODEL_CONFIGS, COLORS, VIDEO_FORMATS, MAX_VIDEO_SIZE_MB

# Page configuration
st.set_page_config(
    page_title="   D.R.O.N.A.",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []


def main():
    """Main application function"""

    # Title and description
    st.title("🏊 Drowning Risk Observation & Next-gen Quality Assurance")
    st.markdown("""
    **Self-learning AI QA agent** that analyzes drowning detection videos using multiple models 
    and DeepSORT tracking to identify missed human detections.
    """)

    # Sidebar configuration
    setup_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Video Analysis",
        "📊 Results Dashboard",
        "🤖 AI Insights",
        "📋 Export & Reports"
    ])

    with tab1:
        video_analysis_tab()

    with tab2:
        results_dashboard_tab()

    with tab3:
        ai_insights_tab()

    with tab4:
        export_reports_tab()

    # Cleanup on app end
    cleanup_session()


def setup_sidebar():
    """Setup sidebar with configuration options"""

    st.sidebar.header("⚙️ Configuration")

    # Model configuration
    st.sidebar.subheader("Detection Models")
    enable_yolo = st.sidebar.checkbox("Enable YOLOv11", value=True)
    enable_vit = st.sidebar.checkbox("Enable ViT/DETR", value=True)
    enable_mediapipe = st.sidebar.checkbox("Enable MediaPipe", value=True)
    enable_tracking = st.sidebar.checkbox(
        "Enable DeepSORT Tracking", value=True)

    # Confidence thresholds
    st.sidebar.subheader("Confidence Thresholds")
    yolo_confidence = st.sidebar.slider("YOLO Confidence", 0.1, 1.0,
                                        MODEL_CONFIGS['yolov11']['confidence'], 0.05)
    vit_confidence = st.sidebar.slider("ViT Confidence", 0.1, 1.0,
                                       MODEL_CONFIGS['vit']['confidence'], 0.05)
    mediapipe_confidence = st.sidebar.slider("MediaPipe Confidence", 0.1, 1.0,
                                             MODEL_CONFIGS['mediapipe']['confidence'], 0.05)



    # AI Analysis
    st.sidebar.subheader("AI Analysis")
    enable_ai_analysis = st.sidebar.checkbox(
        "Enable Gemini AI Analysis", value=True)

    # Performance settings
    st.sidebar.subheader("Performance")
    use_gpu = st.sidebar.checkbox("Use GPU Acceleration", value=True)
    half_precision = st.sidebar.checkbox("Half Precision (FP16)", value=True)

    # Store settings in session state
    st.session_state.model_config = {
        'enable_yolo': enable_yolo,
        'enable_vit': enable_vit,
        'enable_mediapipe': enable_mediapipe,
        'enable_tracking': enable_tracking,
        'yolo_confidence': yolo_confidence,
        'vit_confidence': vit_confidence,
        'mediapipe_confidence': mediapipe_confidence,

        'enable_ai_analysis': enable_ai_analysis,
        'use_gpu': use_gpu,
        'half_precision': half_precision
    }


def video_analysis_tab():
    """Video analysis and processing tab"""

    st.header("Video Analysis")

    # Video upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
            help=f"Supported formats: {', '.join(VIDEO_FORMATS)}. Max size: {MAX_VIDEO_SIZE_MB}MB"
        )

        if uploaded_video is not None:
            # Display video info
            file_size = len(uploaded_video.getvalue())
            st.info(
                f"**File:** {uploaded_video.name} | **Size:** {format_file_size(file_size)}")

            # Save uploaded video to temp file
            temp_video_path = create_temp_file(suffix=".mp4")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getvalue())

            st.session_state.temp_files.append(temp_video_path)

            # Validate video
            validation_result = validate_video_file(temp_video_path)

            if validation_result['valid']:
                metadata = validation_result['metadata']
                st.success("✅ Video validation successful")

                # Display video metadata
                with st.expander("Video Information", expanded=True):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Resolution",
                                  f"{metadata['width']}x{metadata['height']}")
                        st.metric("Duration", format_duration(
                            metadata['duration']))
                    with col_b:
                        st.metric("Frame Rate", f"{metadata['fps']:.1f} FPS")
                        st.metric("Total Frames",
                                  f"{metadata['frame_count']:,}")
                    with col_c:
                        st.metric("Format", metadata['format'].upper())
                        st.metric("File Size", format_file_size(
                            metadata['file_size']))

                st.session_state.video_path = temp_video_path
                st.session_state.video_metadata = metadata

            else:
                st.error(
                    f"❌ Video validation failed: {validation_result['error']}")
                return

    with col2:
        st.subheader("Custom YOLO Results")
        st.markdown("Upload your custom YOLO detection results (optional)")

        # Option to upload custom YOLO results
        upload_option = st.radio(
            "YOLO Results Input",
            ["None", "Upload JSON File", "Paste JSON"]
        )

        custom_yolo_results = None

        if upload_option == "Upload JSON File":
            uploaded_json = st.file_uploader(
                "Upload YOLO results JSON",
                type=['json'],
                help="JSON file containing per-frame detection results"
            )

            if uploaded_json is not None:
                try:
                    json_data = json.load(uploaded_json)
                    custom_yolo_results = parse_custom_yolo_results(json_data)
                    st.success(
                        f"✅ Loaded {len(custom_yolo_results)} frames of YOLO results")
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {str(e)}")

        elif upload_option == "Paste JSON":
            json_text = st.text_area(
                "Paste JSON data",
                height=200,
                placeholder='[{"bbox": [x1, y1, x2, y2], "confidence": 0.9, "class_id": "person"}]'
            )

            if json_text.strip():
                try:
                    json_data = json.loads(json_text)
                    custom_yolo_results = parse_custom_yolo_results(json_data)
                    st.success(
                        f"✅ Parsed {len(custom_yolo_results)} frames of YOLO results")
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {str(e)}")

        st.session_state.custom_yolo_results = custom_yolo_results

    # Processing section
    st.subheader("Start Analysis")

    if 'video_path' in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                start_video_processing()

        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_session()

        with col3:
            if st.session_state.processing_complete and st.session_state.analysis_results:
                st.success(
                    "✅ Analysis Complete! Check other tabs for results.")
    else:
        st.info("👆 Please upload a video file to begin analysis")


def start_video_processing():
    """Start the video processing pipeline"""

    try:
        st.session_state.processing_complete = False

        # Create progress containers
        progress_container = st.container()
        status_container = st.container()

        with status_container:
            st.info("🔄 Initializing video processor...")

        # Initialize video processor
        config = st.session_state.model_config
        processor = VideoProcessor(
            enable_ai_analysis=config['enable_ai_analysis']
        )

        with status_container:
            st.info("🔄 Starting video analysis...")

        # Progress callback
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        def progress_callback(progress, frame_num, total_frames):
            progress_bar.progress(progress)
            status_text.text(
                f"Processing frame {frame_num}/{total_frames} ({progress:.1%})")

        # Process video
        with st.spinner("Processing video... This may take several minutes."):
            results = processor.process_video(
                st.session_state.video_path,
                custom_yolo_results=st.session_state.custom_yolo_results,
                progress_callback=progress_callback
            )

        # Store results
        st.session_state.analysis_results = results
        st.session_state.processing_complete = True

        # Automatically generate enhanced video output
        with status_container:
            with st.spinner("🎥 Generating QA output video..."):
                try:
                    enhanced_path = processor.export_enhanced_video(
                        st.session_state.video_path,
                        results
                    )
                    st.session_state.enhanced_video_path = enhanced_path
                    st.success(
                        "✅ Video analysis completed and QA output video generated!")
                    st.info(f"📁 Enhanced video saved to: {enhanced_path}")
                except Exception as e:
                    st.warning(
                        f"⚠️ Analysis completed but video export failed: {str(e)}")
                    st.success("✅ Video analysis completed successfully!")

        # Cleanup processor
        processor.cleanup()

        # Auto-switch to results tab
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.error("**Error Details:**")
        st.code(traceback.format_exc())


def results_dashboard_tab():
    """Results and metrics dashboard"""

    st.header("Results Dashboard")

    if not st.session_state.processing_complete or not st.session_state.analysis_results:
        st.info("📊 No analysis results available. Please run video analysis first.")
        return

    results = st.session_state.analysis_results

    # Summary metrics
    st.subheader("📈 Analysis Summary")

    summary = results.get('processing_summary', {})
    overall_metrics = results.get('overall_metrics', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Frames Processed",
            f"{summary.get('total_frames_processed', 0):,}",
            delta=None
        )

    with col2:
        st.metric(
            "Processing Errors",
            summary.get('frames_with_errors', 0),
            delta=None,
            delta_color="inverse"
        )

    with col3:
        detection_stats = overall_metrics.get('detection_statistics', {})
        total_detections = detection_stats.get(
            'total_ensemble_count', {}).get('total', 0)
        st.metric(
            "Total Detections",
            f"{total_detections:,}",
            delta=None
        )

    with col4:
        missed_detections = detection_stats.get(
            'missed_count', {}).get('total', 0)
        st.metric(
            "Missed Detections",
            f"{missed_detections:,}",
            delta=None,
            delta_color="inverse"
        )

    # Quality assessment
    quality_assessment = overall_metrics.get('quality_assessment', {})

    if quality_assessment:
        st.subheader("🎯 Quality Assessment")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Quality scores radar chart
            categories = ['Overall Score', 'Reliability',
                          'Consistency', 'Confidence']
            values = [
                quality_assessment.get('overall_score', 0) * 100,
                quality_assessment.get('reliability_score', 0) * 100,
                quality_assessment.get('consistency_score', 0) * 100,
                quality_assessment.get('confidence_score', 0) * 100
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Quality Scores'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Quality Assessment Radar Chart"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            risk_level = quality_assessment.get('risk_level', 'medium')
            risk_colors = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }

            st.metric(
                "Risk Level",
                f"{risk_colors.get(risk_level, '⚪')} {risk_level.upper()}"
            )

            st.metric(
                "Overall Score",
                f"{quality_assessment.get('overall_score', 0):.2%}"
            )

            # Recommendations
            recommendations = quality_assessment.get('recommendations', [])
            if recommendations:
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    st.write(f"{i}. {rec}")

    # Detection statistics over time
    st.subheader("📊 Detection Statistics")

    frame_results = results.get('frame_results', [])
    if frame_results:
        # Prepare data for plotting
        frame_data = []
        for frame_result in frame_results:
            if 'error' not in frame_result:
                metrics = frame_result.get('metrics', {})
                frame_data.append({
                    'Frame': frame_result.get('frame_number', 0),
                    'Custom YOLO': metrics.get('custom_yolo_count', 0),
                    'Ensemble': metrics.get('total_ensemble_count', 0),
                    'Missed': metrics.get('missed_count', 0),
                    'Tracking': metrics.get('tracking_count', 0)
                })

        if frame_data:
            df = pd.DataFrame(frame_data)

            # Line chart of detections over time
            fig = px.line(
                df,
                x='Frame',
                y=['Custom YOLO', 'Ensemble', 'Missed', 'Tracking'],
                title="Detection Counts Over Time",
                labels={'value': 'Detection Count',
                        'variable': 'Detection Type'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics table
            st.subheader("📋 Detection Summary Statistics")
            summary_df = df.describe()
            st.dataframe(summary_df, use_container_width=True)


def ai_insights_tab():
    """AI insights and analysis tab"""

    st.header("🤖 AI Insights")

    if not st.session_state.processing_complete or not st.session_state.analysis_results:
        st.info("🤖 No AI analysis available. Please run video analysis first.")
        return

    results = st.session_state.analysis_results
    ai_analysis = results.get('ai_analysis', {})

    # Gemini AI Report
    st.subheader("🧠 Gemini AI Analysis Report")

    gemini_report = ai_analysis.get('gemini_report')
    if gemini_report and gemini_report != "AI analysis failed":
        st.markdown(gemini_report)
    else:
        st.warning("⚠️ Gemini AI analysis was not available or failed.")
        if gemini_report:
            st.error(f"Error: {gemini_report}")

    # LangChain Analysis
    st.subheader("🔗 LangChain Agent Analysis")

    langchain_analysis = ai_analysis.get('langchain_analysis')
    if langchain_analysis:
        st.markdown(langchain_analysis)
    else:
        st.warning("⚠️ LangChain analysis was not available.")

    # Frame-level AI insights
    st.subheader("🎯 Frame-level AI Insights")

    frame_results = results.get('frame_results', [])
    critical_frames = []

    for frame_result in frame_results:
        ai_frame_analysis = frame_result.get('ai_analysis')
        if ai_frame_analysis:
            severity = ai_frame_analysis.get('missed_detection_severity', 0)
            risk_level = ai_frame_analysis.get('risk_level', 'medium')

            # Consider frames with high severity or high/critical risk
            if severity >= 4 or risk_level in ['high', 'critical']:
                critical_frames.append({
                    'Frame': frame_result.get('frame_number', 0),
                    'Severity': severity,
                    'Risk Level': risk_level,
                    'Confidence Assessment': ai_frame_analysis.get('confidence_assessment', 0),
                    'Reasoning': ai_frame_analysis.get('reasoning', 'N/A')
                })

    if critical_frames:
        st.warning(
            f"⚠️ Found {len(critical_frames)} critical frames requiring attention")

        # Display critical frames table
        critical_df = pd.DataFrame(critical_frames)
        st.dataframe(
            critical_df.style.highlight_max(
                subset=['Severity'], color='lightcoral'),
            use_container_width=True
        )

        # Show details for most critical frame
        if critical_frames:
            most_critical = max(critical_frames, key=lambda x: x['Severity'])

            with st.expander(f"🚨 Most Critical Frame: {most_critical['Frame']}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Severity Level",
                              f"{most_critical['Severity']}/5")
                    st.metric("Risk Level",
                              most_critical['Risk Level'].upper())

                with col2:
                    st.metric("Confidence Assessment",
                              f"{most_critical['Confidence Assessment']:.2%}")

                st.write("**AI Reasoning:**")
                st.write(most_critical['Reasoning'])
    else:
        st.success("✅ No critical frames detected by AI analysis")

    # AI Configuration Status
    with st.expander("🔧 AI Configuration Status"):
        config = st.session_state.model_config

        col1, col2 = st.columns(2)

        with col1:
            st.write("**AI Analysis Settings:**")
            st.write(
                f"- Gemini AI: {'✅ Enabled' if config.get('enable_ai_analysis') else '❌ Disabled'}")
            st.write(
                f"- LangChain Tools: {'✅ Available' if langchain_analysis else '❌ Not Available'}")

        with col2:
            st.write("**API Status:**")
            gemini_key = os.getenv("GEMINI_API_KEY", "default_key")
            if gemini_key != "default_key":
                st.write("- Gemini API Key: ✅ Configured")
            else:
                st.write("- Gemini API Key: ⚠️ Using Default (Set GEMINI_API_KEY)")


def export_reports_tab():
    """Export and reports tab"""

    st.header("📋 Export & Reports")

    if not st.session_state.processing_complete or not st.session_state.analysis_results:
        st.info("📋 No results to export. Please run video analysis first.")
        return

    results = st.session_state.analysis_results

    # Export options
    st.subheader("📤 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Available Exports:**")

        # JSON Report
        if st.button("📄 Download JSON Report", use_container_width=True):
            download_json_report(results)

        # CSV Statistics
        if st.button("📊 Download CSV Statistics", use_container_width=True):
            download_csv_statistics(results)

        # Enhanced Video
        if hasattr(st.session_state, 'enhanced_video_path') and st.session_state.enhanced_video_path:
            st.success("✅ QA Output video already generated!")
            if st.button("🎥 Download Enhanced Video", use_container_width=True):
                download_enhanced_video()
        else:
            if st.button("🎥 Generate Enhanced Video", use_container_width=True):
                with st.spinner("Generating enhanced video with QA annotations..."):
                    generate_enhanced_video(results)

    with col2:
        st.write("**Export Information:**")
        st.info("""
        - **JSON Report**: Complete analysis results in JSON format
        - **CSV Statistics**: Frame-by-frame metrics for further analysis
        - **Enhanced Video**: Original video with detection overlays
        """)

    # Quick report preview
    st.subheader("📖 Quick Report Preview")

    with st.expander("Analysis Summary", expanded=True):
        video_info = results.get('video_info', {})
        processing_summary = results.get('processing_summary', {})

        st.write("**Video Information:**")
        st.write(
            f"- Duration: {format_duration(video_info.get('duration', 0))}")
        st.write(
            f"- Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
        st.write(f"- Frame Rate: {video_info.get('fps', 0):.1f} FPS")
        st.write(f"- Total Frames: {video_info.get('total_frames', 0):,}")

        st.write("**Processing Results:**")
        st.write(
            f"- Frames Processed: {processing_summary.get('total_frames_processed', 0):,}")
        st.write(
            f"- Processing Errors: {processing_summary.get('frames_with_errors', 0)}")

        overall_metrics = results.get('overall_metrics', {})
        quality_assessment = overall_metrics.get('quality_assessment', {})

        if quality_assessment:
            st.write("**Quality Assessment:**")
            st.write(
                f"- Overall Score: {quality_assessment.get('overall_score', 0):.2%}")
            st.write(
                f"- Risk Level: {quality_assessment.get('risk_level', 'unknown').upper()}")
            st.write(
                f"- Reliability Score: {quality_assessment.get('reliability_score', 0):.2%}")


def download_json_report(results: Dict):
    """Generate and download JSON report"""
    try:
        # Create a clean copy for export (remove large data)
        export_results = results.copy()
        export_results['frame_results'] = f"Processed {len(results.get('frame_results', []))} frames"

        json_str = json.dumps(export_results, indent=2, default=str)

        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"drowning_detection_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        st.success("✅ JSON report ready for download!")

    except Exception as e:
        st.error(f"❌ Error generating JSON report: {str(e)}")


def download_csv_statistics(results: Dict):
    """Generate and download CSV statistics"""
    try:
        frame_results = results.get('frame_results', [])

        # Prepare CSV data
        csv_data = []
        for frame_result in frame_results:
            if 'error' not in frame_result:
                metrics = frame_result.get('metrics', {})
                ai_analysis = frame_result.get('ai_analysis', {})

                row = {
                    'frame_number': frame_result.get('frame_number', 0),
                    'timestamp': frame_result.get('timestamp', 0),
                    'custom_yolo_count': metrics.get('custom_yolo_count', 0),
                    'ensemble_yolo_count': metrics.get('ensemble_yolo_count', 0),
                    'vit_count': metrics.get('vit_count', 0),
                    'mediapipe_count': metrics.get('mediapipe_count', 0),
                    'total_ensemble_count': metrics.get('total_ensemble_count', 0),
                    'missed_count': metrics.get('missed_count', 0),
                    'tracking_count': metrics.get('tracking_count', 0),
                    'ai_severity': ai_analysis.get('missed_detection_severity', 0) if ai_analysis else 0,
                    'ai_risk_level': ai_analysis.get('risk_level', 'unknown') if ai_analysis else 'unknown',
                    'ai_confidence_assessment': ai_analysis.get('confidence_assessment', 0) if ai_analysis else 0
                }
                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)

            st.download_button(
                label="📊 Download CSV Statistics",
                data=csv_str,
                file_name=f"drowning_detection_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            st.success("✅ CSV statistics ready for download!")
        else:
            st.warning("⚠️ No frame data available for CSV export")

    except Exception as e:
        st.error(f"❌ Error generating CSV statistics: {str(e)}")


def download_enhanced_video():
    """Download the already generated enhanced video"""
    try:
        enhanced_path = st.session_state.enhanced_video_path

        if not os.path.exists(enhanced_path):
            st.error("❌ Enhanced video file not found")
            return

        # Read enhanced video for download
        with open(enhanced_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="🎥 Download QA Output Video",
            data=video_bytes,
            file_name=f"qa_output_drowning_detection_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            mime="video/mp4"
        )

        st.success("✅ QA output video ready for download!")

    except Exception as e:
        st.error(f"❌ Error preparing video download: {str(e)}")


def generate_enhanced_video(results: Dict):
    """Generate enhanced video with detection overlays (fallback)"""
    try:
        if 'video_path' not in st.session_state:
            st.error("❌ Original video not available")
            return

        with st.spinner("🎥 Generating enhanced video... This may take several minutes."):
            # Initialize video processor
            processor = VideoProcessor()

            # Generate enhanced video
            enhanced_path = processor.export_enhanced_video(
                st.session_state.video_path,
                results
            )

            # Store path for future downloads
            st.session_state.enhanced_video_path = enhanced_path

            # Cleanup processor
            processor.cleanup()

            download_enhanced_video()

    except Exception as e:
        st.error(f"❌ Error generating enhanced video: {str(e)}")


def reset_session():
    """Reset session state"""

    # Cleanup temp files
    cleanup_session()

    # Reset session state
    st.session_state.processing_complete = False
    st.session_state.analysis_results = None

    if 'video_path' in st.session_state:
        del st.session_state.video_path
    if 'video_metadata' in st.session_state:
        del st.session_state.video_metadata
    if 'custom_yolo_results' in st.session_state:
        del st.session_state.custom_yolo_results

    st.success("✅ Session reset complete")
    st.rerun()


def cleanup_session():
    """Cleanup temporary files and resources"""

    if 'temp_files' in st.session_state:
        cleanup_temp_files(st.session_state.temp_files)
        st.session_state.temp_files = []


if __name__ == "__main__":
    main()
