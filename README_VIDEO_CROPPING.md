# Video Cropping Service

A robust video cropping service that automatically generates vertical (9:16) clips from horizontal source videos using AI-powered face tracking and speaker detection.

## Overview

This service provides intelligent video cropping with support for:

### **Use Cases:**

1. **Solo Speaker**
   - One person on camera throughout
   - Crop to follow their face/mouth movements
   - Perfect for tutorials, vlogs, presentations

2. **Interview**
   - Two participants (host + guest)
   - Dynamically switch focus to whoever is speaking
   - Ideal for podcasts, interviews, panel discussions

### **Key Features:**

- **MediaPipe Face Detection** - Primary face tracking with high accuracy
- **MTCNN Fallback** - Secondary face detection for challenging scenarios
- **PyAnnote Speaker Diarization** - Audio-based speaker identification for interviews
- **Temporal Smoothing** - Reduces jitter and provides stable crops
- **Multiple Aspect Ratios** - 9:16, 1:1, 4:5, 3:4 support
- **Async Processing** - Handle multiple requests simultaneously
- **Comprehensive Logging** - Detailed progress tracking and error reporting

## Installation

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **FFmpeg** installed on system
3. **HuggingFace Token** (optional, for interview mode)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export HUGGINGFACE_TOKEN="your_token_here"
```

### Required Dependencies

```txt
mediapipe==0.10.18
opencv-python==4.11.0.86
numpy>=1.21.0
ffmpeg-python>=0.2.0
fastapi>=0.115.0
```

### Optional Dependencies (for enhanced features)

```txt
pyannote.audio==3.1.1
torch>=1.13.0
torchaudio>=0.13.0
mtcnn==0.1.1
scenedetect==0.6.6
```

## API Endpoints

### 1. Analyze Video Mode

**POST /crop/analyze**

Analyze video to determine optimal cropping mode without processing.

```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "confidence_threshold": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "detected_mode": "solo",
    "confidence": 0.85,
    "recommended_settings": {
      "mode": "solo",
      "target_aspect_ratio": "9:16",
      "enable_scene_detection": true
    }
  },
  "supported_modes": ["auto", "solo", "interview", "fallback"],
  "supported_aspect_ratios": ["9:16", "1:1", "4:5", "3:4"]
}
```

### 2. Crop Video from URL

**POST /crop/crop**

Process video from URL with intelligent face tracking.

```json
{
  "video_url": "https://youtube.com/watch?v=example",
  "mode": "auto",
  "target_aspect_ratio": "9:16",
  "output_resolution": "1080x1920",
  "confidence_threshold": 0.7,
  "enable_scene_detection": true,
  "smoothing_window": 30,
  "padding_ratio": 0.1
}
```

**Response:**
```json
{
  "success": true,
  "task_id": "crop_abc12345",
  "message": "Video cropping started",
  "status_endpoint": "/crop/status/crop_abc12345",
  "estimated_time": "3-15 minutes depending on video length and mode"
}
```

### 3. Upload and Crop Video

**POST /crop/crop-upload**

Upload video file and process with intelligent cropping.

**Form Data:**
- `file`: Video file (MP4, AVI, MOV, MKV, WebM)
- `mode`: "auto", "solo", "interview", "fallback"
- `target_aspect_ratio`: "9:16", "1:1", "4:5", "3:4"
- `output_resolution`: "WIDTHxHEIGHT" (e.g., "1080x1920")
- `confidence_threshold`: 0.1-1.0
- `enable_scene_detection`: true/false
- `smoothing_window`: Number of frames for smoothing
- `padding_ratio`: Extra padding around face (0.0-0.5)

### 4. Check Task Status

**GET /crop/status/{task_id}**

Monitor processing progress and get results.

**Response:**
```json
{
  "task_id": "crop_abc12345",
  "status": "completed",
  "progress": 100,
  "current_step": "completed",
  "message": "Video cropping completed successfully",
  "result": {
    "success": true,
    "output_path": "/path/to/cropped_video.mp4",
    "file_size_mb": 45.2,
    "mode_used": "solo",
    "target_aspect_ratio": "9:16",
    "output_resolution": "1080x1920"
  },
  "download_endpoint": "/crop/download/crop_abc12345",
  "processing_time_seconds": 180
}
```

### 5. Download Cropped Video

**GET /crop/download/{task_id}**

Download the processed video file.

Returns video file with headers:
- `Content-Type: video/mp4`
- `Content-Disposition: attachment; filename="cropped_abc12345.mp4"`

### 6. Health Check

**GET /crop/health**

Check service status and available features.

```json
{
  "status": "ok",
  "service": "video_cropper",
  "dependencies": {
    "mediapipe": true,
    "opencv": true,
    "audio_processing": true,
    "video_processing": true
  },
  "active_tasks": 3,
  "supported_modes": ["auto", "solo", "interview", "fallback"],
  "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"]
}
```

## Processing Modes

### Auto Mode (Recommended)
Automatically detects the best mode based on video analysis:
- Analyzes face count distribution
- Detects speaker patterns
- Chooses optimal processing strategy

### Solo Mode
For single-speaker videos:
- **Face Detection**: MediaPipe + MTCNN fallback
- **Tracking**: OpenCV CSRT tracker for smooth motion
- **Fallback**: Center crop when face detection fails
- **Best for**: Tutorials, vlogs, presentations, solo content

### Interview Mode
For multi-speaker scenarios:
- **Speaker Diarization**: PyAnnote Audio for speaker identification
- **Face Association**: Maps speakers to face positions
- **Dynamic Switching**: Focuses on active speaker
- **Temporal Smoothing**: Reduces jarring transitions
- **Best for**: Interviews, podcasts, panel discussions

### Fallback Mode
Simple center crop when face detection fails:
- **No AI Processing**: Basic geometric center crop
- **Reliable**: Always works regardless of content
- **Fast**: Minimal processing time
- **Best for**: Content without clear faces, backup option

## Configuration Options

### Aspect Ratios

| Ratio | Dimensions | Use Case |
|-------|------------|----------|
| 9:16  | 1080x1920  | TikTok, YouTube Shorts, Instagram Reels |
| 1:1   | 1080x1080  | Instagram Posts, Facebook |
| 4:5   | 1080x1350  | Instagram Feed |
| 3:4   | 1080x1440  | Pinterest, Alternative Vertical |

### Quality Settings

| Resolution | Description | File Size | Processing Time |
|------------|-------------|-----------|----------------|
| 720x1280   | HD          | Small     | Fast |
| 1080x1920  | Full HD     | Medium    | Medium |
| 1440x2560  | 2K          | Large     | Slow |

### Performance Tuning

```json
{
  "confidence_threshold": 0.7,      // Higher = more selective face detection
  "smoothing_window": 30,           // Higher = smoother but less responsive
  "padding_ratio": 0.1,             // Higher = more padding around face
  "enable_scene_detection": true    // Handles video cuts and transitions
}
```

## Usage Examples

### Python Client Example

```python
import requests
import time

# 1. Start cropping job
response = requests.post("http://localhost:8000/crop/crop", json={
    "video_url": "https://youtube.com/watch?v=example",
    "mode": "auto",
    "target_aspect_ratio": "9:16",
    "confidence_threshold": 0.8
})

task_id = response.json()["task_id"]
print(f"Started task: {task_id}")

# 2. Monitor progress
while True:
    status = requests.get(f"http://localhost:8000/crop/status/{task_id}")
    data = status.json()
    
    print(f"Progress: {data['progress']}% - {data['message']}")
    
    if data["status"] == "completed":
        print("✅ Processing completed!")
        break
    elif data["status"] == "failed":
        print(f"❌ Processing failed: {data['error']}")
        break
    
    time.sleep(5)

# 3. Download result
if data["status"] == "completed":
    video_response = requests.get(f"http://localhost:8000/crop/download/{task_id}")
    
    with open("cropped_video.mp4", "wb") as f:
        f.write(video_response.content)
    
    print("📹 Video downloaded successfully!")
```

### cURL Examples

```bash
# Analyze video mode
curl -X POST "http://localhost:8000/crop/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://youtube.com/watch?v=example"}'

# Start cropping job
curl -X POST "http://localhost:8000/crop/crop" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtube.com/watch?v=example",
    "mode": "solo",
    "target_aspect_ratio": "9:16"
  }'

# Check status
curl "http://localhost:8000/crop/status/crop_abc12345"

# Download result
curl -O "http://localhost:8000/crop/download/crop_abc12345"
```

### Upload Example

```bash
curl -X POST "http://localhost:8000/crop/crop-upload" \
  -F "file=@video.mp4" \
  -F "mode=auto" \
  -F "target_aspect_ratio=9:16" \
  -F "output_resolution=1080x1920"
```

## Performance & Limitations

### Performance Metrics
- **Solo Mode**: 2-5x video length processing time
- **Interview Mode**: 3-8x video length processing time
- **Fallback Mode**: 1-2x video length processing time
- **Memory Usage**: 500MB-2GB depending on resolution

### Limitations
1. **Face Detection Accuracy**: Requires clear, well-lit faces
2. **Speaker Diarization**: Needs clear audio for interview mode
3. **Processing Time**: Can be significant for long videos
4. **GPU Acceleration**: Not currently implemented (CPU only)

### Optimization Tips
1. Use **auto mode** for best results
2. **Lower resolution** for faster processing
3. **Solo mode override** for single-speaker content
4. **Good lighting** improves face detection accuracy
5. **Clear audio** enhances interview mode performance

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install mediapipe opencv-python

# For interview mode
pip install pyannote.audio torch torchaudio
```

**2. Face Detection Failures**
- Increase `confidence_threshold` for better accuracy
- Ensure good lighting in source video
- Use `mode="fallback"` for low-quality videos

**3. Audio Processing Issues**
- Verify FFmpeg installation
- Check audio track exists in video
- Use solo mode for videos without clear speech

**4. HuggingFace Token Issues**
```bash
# Set environment variable
export HUGGINGFACE_TOKEN="your_token_here"

# Or pass in request
# Note: Interview mode requires valid HF token
```

### Performance Issues

**For Speed:**
```json
{
  "mode": "fallback",
  "output_resolution": "720x1280",
  "smoothing_window": 15
}
```

**For Quality:**
```json
{
  "mode": "auto",
  "confidence_threshold": 0.8,
  "smoothing_window": 45,
  "enable_scene_detection": true
}
```

## Testing

Run the test suite to verify installation:

```bash
python test_crop_service.py
```

Expected output:
```
🎉 All tests passed! Video cropping service is ready to use.
```

## License & Attribution

Based on the ClipsAI open-source project with custom adaptations for face tracking and speaker detection. Uses MediaPipe (Apache 2.0), PyAnnote.audio (MIT), and other open-source libraries.

---

**Built for the ClipsAI project** - Production-ready video cropping with intelligent speaker detection and tracking. 