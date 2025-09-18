# Overview

This is a drowning detection quality assurance (QA) system that uses an ensemble of computer vision models combined with AI-powered analysis. The application processes video files to detect people in drowning scenarios, compares results across multiple detection models, and provides intelligent feedback on detection quality using a Gemini-powered AI agent. Built as a Streamlit web application, it serves as a comprehensive tool for evaluating and improving drowning detection systems.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Single-page application with file upload, real-time processing visualization, and interactive results dashboard
- **Session State Management**: Maintains processing state, analysis results, and temporary file references across user interactions
- **Interactive Visualizations**: Uses Plotly for metrics dashboards and OpenCV for annotated video playback

## Backend Architecture
- **Ensemble Detection System**: Combines multiple computer vision models (YOLOv11, Vision Transformer/DETR, MediaPipe) for robust person detection
- **Video Processing Pipeline**: Frame-by-frame analysis with configurable processing options and progress tracking
- **AI Analysis Layer**: Gemini-powered agent provides intelligent quality assessment and recommendations
- **Metrics Calculation Engine**: Comprehensive performance metrics including confidence statistics, detection density, and coverage analysis

## Core Components

### Detection Models
- **YOLOv11 Detector**: Primary object detection using custom-trained YOLO model
- **Vision Transformer (DETR)**: Facebook's DETR-ResNet-50 for enhanced person detection
- **MediaPipe**: Google's pose detection for human presence validation
- **Ensemble Logic**: Combines results from all models with confidence weighting and non-maximum suppression

### Tracking System
- **Simple Tracker**: Custom implementation with linear motion prediction and track management
- **Track Management**: Handles object lifecycle, ID assignment, and trajectory history

### AI Analysis
- **Gemini Integration**: Uses Google's Gemini 2.5 Flash model for detection quality analysis
- **LangChain Tools**: Structured agent tools for comparative analysis and recommendations
- **Structured Output**: Pydantic models ensure consistent analysis format with severity scoring

## Data Flow
1. **Video Upload**: Validates format, size, and codec compatibility
2. **Frame Processing**: Sequential frame extraction and parallel model inference
3. **Result Fusion**: Ensemble logic combines detections with confidence weighting
4. **AI Analysis**: Gemini agent evaluates each frame's detection quality
5. **Visualization**: Annotated output with color-coded detections and metrics overlay

## Configuration Management
- **Model Configs**: Centralized configuration for all detection models including thresholds and parameters
- **Processing Settings**: Configurable confidence thresholds, IoU parameters, and tracking settings
- **Visualization Settings**: Color schemes and annotation parameters

# External Dependencies

## AI Services
- **Google Gemini API**: Primary AI analysis engine for detection quality assessment
- **Gemini 2.5 Flash Model**: Specific model used for fast, accurate analysis

## Machine Learning Frameworks
- **PyTorch**: Core deep learning framework for DETR and custom models
- **Transformers (HuggingFace)**: Vision Transformer implementation and model loading
- **MediaPipe**: Google's pose detection and human analysis framework
- **YOLOv11**: Custom object detection model (assumed to be locally implemented)

## Computer Vision
- **OpenCV**: Video processing, frame manipulation, and visualization
- **NumPy**: Numerical computations and array operations

## Web Framework
- **Streamlit**: Complete web application framework with built-in components
- **Plotly**: Interactive visualization library for metrics dashboards

## Data Processing
- **Pandas**: Data manipulation for metrics analysis
- **Pydantic**: Data validation and structured output modeling
- **LangChain**: AI agent orchestration and tool management

## Video Processing
- **Supported Formats**: MP4, AVI, MOV, MKV, WMV, FLV
- **Codec Requirements**: Standard video codecs supported by OpenCV
- **Size Limitations**: Maximum 500MB file size for processing

## Development Tools
- **Logging**: Python's built-in logging framework for debugging and monitoring
- **Pathlib**: Modern path handling for cross-platform compatibility
- **Tempfile**: Secure temporary file management for video processing