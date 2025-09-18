#!/usr/bin/env python3
"""
Flask web-based ROI selector
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import json
import base64
from datetime import datetime
import os

app = Flask(__name__)

# Global variables
video_path = None
current_frame = None
roi_points = []


def extract_frame(video_path, frame_number):
    """Extract frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def frame_to_base64(frame):
    """Convert frame to base64 for web display"""
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    return render_template('roi_selector.html')


@app.route('/load_video', methods=['POST'])
def load_video():
    global video_path, current_frame

    data = request.json
    video_path = data['video_path']
    frame_number = data.get('frame_number', 0)

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'})

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Load frame
    current_frame = extract_frame(video_path, frame_number)
    if current_frame is None:
        return jsonify({'error': 'Could not load frame'})

    frame_base64 = frame_to_base64(current_frame)

    return jsonify({
        'success': True,
        'frame': frame_base64,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'current_frame': frame_number
    })


@app.route('/change_frame', methods=['POST'])
def change_frame():
    global current_frame

    data = request.json
    frame_number = data['frame_number']

    current_frame = extract_frame(video_path, frame_number)
    if current_frame is None:
        return jsonify({'error': 'Could not load frame'})

    frame_base64 = frame_to_base64(current_frame)

    return jsonify({
        'success': True,
        'frame': frame_base64,
        'current_frame': frame_number
    })


@app.route('/add_point', methods=['POST'])
def add_point():
    global roi_points

    data = request.json
    x = int(data['x'])
    y = int(data['y'])

    roi_points.append([x, y])

    return jsonify({
        'success': True,
        'points': roi_points,
        'point_count': len(roi_points)
    })


@app.route('/clear_points', methods=['POST'])
def clear_points():
    global roi_points
    roi_points = []

    return jsonify({
        'success': True,
        'points': roi_points
    })


@app.route('/save_roi', methods=['POST'])
def save_roi():
    global roi_points, video_path

    if len(roi_points) < 3:
        return jsonify({'error': 'Need at least 3 points'})

    # Get video name from path
    video_name = os.path.basename(video_path)

    # Load existing ROI data or create new
    roi_file = "roi_polygon.json"
    try:
        with open(roi_file, 'r') as f:
            all_roi_data = json.load(f)
        # Handle old format - convert to new format
        if "roi_points" in all_roi_data:
            all_roi_data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        all_roi_data = {}

    # Add/update ROI for this video
    all_roi_data[video_name] = {
        "roi_points": roi_points,
        "point_count": len(roi_points),
        "created_at": datetime.now().isoformat(),
        "video_path": video_path
    }

    # Save updated data
    with open(roi_file, 'w') as f:
        json.dump(all_roi_data, f, indent=2)

    return jsonify({
        'success': True,
        'message': f'ROI saved for {video_name} with {len(roi_points)} points'
    })


if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    # Create HTML template
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>ROI Selector</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .controls { margin: 20px 0; }
        .controls input, .controls button { margin: 5px; padding: 8px; }
        #videoFrame { border: 2px solid #ccc; max-width: 100%; display: block; }
        #canvas { border: 2px solid #ccc; cursor: crosshair; position: absolute; top: 0; left: 0; z-index: 10; pointer-events: auto; }
        #imageContainer { position: relative; display: inline-block; margin-bottom: 20px; }
        .controls { position: relative; z-index: 20; background: white; padding: 10px; }
        .points-list { margin: 20px 0; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pool ROI Selector</h1>
        
        <div class="controls">
            <input type="text" id="videoPath" placeholder="Enter video path" style="width: 400px;" 
                   value="/home/jetson/QA Automation/Drowning_detection_tests/AI_Agent/Input/drown7 - Copy - Trim_annotated_1.mp4">
            <input type="number" id="frameNumber" placeholder="Frame number" value="0" min="0">
            <button onclick="loadVideo()">Load Video</button>
        </div>
        
        <div class="controls" id="frameControls" style="display: none;">
            <button onclick="changeFrame(-10)">-10 Frames</button>
            <button onclick="changeFrame(-1)">-1 Frame</button>
            <input type="range" id="frameSlider" min="0" max="100" value="0" onchange="setFrame(this.value)">
            <button onclick="changeFrame(1)">+1 Frame</button>
            <button onclick="changeFrame(10)">+10 Frames</button>
            <span id="frameInfo"></span>
        </div>
        
        <div id="imageContainer">
            <img id="videoFrame" style="display: none;">
            <canvas id="canvas" onclick="addPoint(event)" style="display: none;"></canvas>
        </div>
        
        <div class="controls" id="roiControls" style="display: none; clear: both; margin-top: 10px;">
            <button onclick="clearPoints()" style="z-index: 30; position: relative;">Clear Points</button>
            <button onclick="saveROI()" id="saveBtn" disabled style="z-index: 30; position: relative;">Save ROI</button>
            <span id="pointCount">Points: 0</span>
        </div>
        
        <div id="pointsList" class="points-list"></div>
        <div id="status"></div>
    </div>

    <script>
        let currentFrame = 0;
        let totalFrames = 0;
        let points = [];
        let canvas, ctx;
        
        function initCanvas() {
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
        }
        
        function drawROI() {
            if (!ctx || points.length === 0) {
                if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = 'red';
            points.forEach(function(point, index) {
                ctx.beginPath();
                ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText(index + 1, point[0] - 3, point[1] + 4);
                ctx.fillStyle = 'red';
            });
            
            if (points.length >= 2) {
                ctx.strokeStyle = 'lime';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i][0], points[i][1]);
                }
                
                if (points.length >= 3) {
                    ctx.closePath();
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                    ctx.fill();
                }
                
                ctx.stroke();
            }
        }
        
        function showStatus(message, isError) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
        }
        
        function loadVideo() {
            const videoPath = document.getElementById('videoPath').value;
            const frameNumber = parseInt(document.getElementById('frameNumber').value) || 0;
            
            fetch('/load_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_path: videoPath, frame_number: frameNumber})
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    const img = document.getElementById('videoFrame');
                    img.src = data.frame;
                    img.style.display = 'block';
                    
                    canvas = document.getElementById('canvas');
                    canvas.width = data.width;
                    canvas.height = data.height;
                    canvas.style.display = 'block';
                    initCanvas();
                    
                    document.getElementById('frameControls').style.display = 'block';
                    document.getElementById('roiControls').style.display = 'block';
                    
                    currentFrame = data.current_frame;
                    totalFrames = data.total_frames;
                    
                    document.getElementById('frameSlider').max = totalFrames - 1;
                    document.getElementById('frameSlider').value = currentFrame;
                    document.getElementById('frameInfo').textContent = 'Frame ' + (currentFrame + 1) + ' of ' + totalFrames + ' (' + data.width + 'x' + data.height + ')';
                    
                    showStatus('Video loaded successfully! Frame ' + (currentFrame + 1) + ' of ' + totalFrames, false);
                } else {
                    showStatus(data.error, true);
                }
            });
        }
        
        function changeFrame(delta) {
            const newFrame = Math.max(0, Math.min(totalFrames - 1, currentFrame + delta));
            setFrame(newFrame);
        }
        
        function setFrame(frameNumber) {
            fetch('/change_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frame_number: parseInt(frameNumber)})
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    document.getElementById('videoFrame').src = data.frame;
                    currentFrame = data.current_frame;
                    document.getElementById('frameSlider').value = currentFrame;
                    document.getElementById('frameInfo').textContent = 'Frame ' + (currentFrame + 1) + ' of ' + totalFrames;
                    drawROI();
                } else {
                    showStatus(data.error, true);
                }
            });
        }
        
        function addPoint(event) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.round(event.clientX - rect.left);
            const y = Math.round(event.clientY - rect.top);
            
            fetch('/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: x, y: y})
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    points = data.points;
                    document.getElementById('pointCount').textContent = 'Points: ' + data.point_count;
                    document.getElementById('saveBtn').disabled = data.point_count < 3;
                    
                    drawROI();
                    updatePointsList();
                    showStatus('Point ' + data.point_count + ' added at (' + x + ', ' + y + ')', false);
                }
            });
        }
        
        function clearPoints() {
            fetch('/clear_points', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    points = [];
                    document.getElementById('pointCount').textContent = 'Points: 0';
                    document.getElementById('saveBtn').disabled = true;
                    document.getElementById('pointsList').innerHTML = '';
                    
                    drawROI();
                    showStatus('All points cleared', false);
                }
            });
        }
        
        function updatePointsList() {
            const listDiv = document.getElementById('pointsList');
            if (points.length === 0) {
                listDiv.innerHTML = '';
                return;
            }
            
            let html = '<h3>ROI Points:</h3><ul>';
            points.forEach(function(point, index) {
                html += '<li>Point ' + (index + 1) + ': (' + point[0] + ', ' + point[1] + ')</li>';
            });
            html += '</ul>';
            listDiv.innerHTML = html;
        }
        
        function saveROI() {
            fetch('/save_roi', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    showStatus(data.message, false);
                } else {
                    showStatus(data.error, true);
                }
            });
        }
    </script>
</body>
</html>'''

    # Write HTML template to file
    with open('templates/roi_selector.html', 'w') as f:
        f.write(html_template)

    print("Starting ROI Selector on http://127.0.0.1:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
