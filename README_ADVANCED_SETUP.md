# Advanced Vertical Cropping Setup Guide

This guide will help you set up the advanced vertical cropping features that use AI-powered speaker diarization and face detection.

## 🚀 Quick Setup

### 1. Install HuggingFace CLI and Authenticate

```bash
# Install HuggingFace CLI (if not already installed)
pip install huggingface_hub[cli]

# Login with your HuggingFace account
huggingface-cli login
```

When prompted, enter your HuggingFace token. Get one from: https://huggingface.co/settings/tokens

### 2. Accept Model License Agreements

Visit these pages and accept the license agreements:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Test Model Access

```bash
# Run the setup script to verify everything works
python setup_hf_auth.py
```

### 4. Start the Server

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🎯 Advanced Features Available

Once set up, you'll have access to:

### 📁 File Upload Endpoint
- **Endpoint**: `POST /workflow/upload-vertical-crop-advanced`
- **Features**: Direct file upload with AI processing
- **Supported formats**: MP4, AVI, MOV, MKV, WebM
- **Max file size**: 500MB

### 🎤 AI-Powered Processing
- **Speaker Diarization**: Uses pyannote.audio to identify who's speaking when
- **Scene Detection**: PySceneDetect for smart transitions
- **Face Detection**: MediaPipe + MTCNN for accurate face tracking
- **Smart Cropping**: Intelligent center-of-attention detection

### 📺 YouTube URL Endpoint
- **Endpoint**: `POST /workflow/create-vertical-crop-advanced` 
- **Features**: Download from YouTube and apply advanced processing

## 🔧 Authentication Methods

The system supports multiple authentication methods:

### Method 1: HuggingFace CLI (Recommended)
```bash
huggingface-cli login
```
This is the most secure and convenient method.

### Method 2: Environment Variable
Create a `.env` file:
```bash
cp env.example .env
# Edit .env and set your token:
HF_TOKEN=your_actual_token_here
```

### Method 3: Form Parameter
You can optionally provide the token in API requests, but this is less secure.

## 🧪 Testing

### Test Model Access
```python
from pyannote.audio import Model, Inference, Pipeline

# Test segmentation model
model = Model.from_pretrained("pyannote/segmentation-3.0")
inference = Inference(model)

# Test speaker diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

### Test API Endpoints
```bash
# Test HF connection
curl -X POST "http://localhost:8000/workflow/test-hf-connection" \
     -F "hf_token=optional_token_here"

# Test file upload (replace with your file)
curl -X POST "http://localhost:8000/workflow/upload-vertical-crop-advanced" \
     -F "file=@your_video.mp4" \
     -F "target_aspect_ratio=9:16"
```

## 🔍 Troubleshooting

### Common Issues

1. **"Model not found" errors**
   - Make sure you've accepted the license agreements
   - Run `huggingface-cli login` again
   - Check your internet connection

2. **"Authentication failed" errors**
   - Verify your HuggingFace token is valid
   - Make sure you have access to the models
   - Try logging out and back in: `huggingface-cli logout && huggingface-cli login`

3. **"Pipeline not initialized" errors**
   - The system will fall back to face-detection only mode
   - Check the server logs for detailed error messages
   - Ensure all dependencies are installed

### Fallback Mode

If speaker diarization fails, the system automatically falls back to:
- Face detection only (MediaPipe + MTCNN)
- Scene detection (PySceneDetect)
- Basic center-weighted cropping

This ensures your videos are still processed even if AI features aren't available.

## 📊 Performance Tips

- **GPU Support**: The system automatically uses GPU if available for AI models
- **Concurrent Processing**: Configure `MAX_CONCURRENT_TASKS` in your .env file
- **Memory Usage**: Large videos may require significant RAM during processing
- **Processing Time**: Expect 5-15 minutes for advanced processing depending on video length

## 🛡️ Security Notes

- Never commit your `.env` file or HuggingFace tokens to version control
- Use HuggingFace CLI authentication for production environments
- Tokens in API requests are logged (use sparingly)
- Consider using read-only tokens for model access

## 📋 API Documentation

### Upload Endpoint Parameters
- `file`: Video file (required)
- `target_aspect_ratio`: e.g., "9:16" (default)
- `use_speaker_detection`: boolean (default: true)
- `hf_token`: optional (uses CLI auth if not provided)
- `output_filename`: optional custom filename

### Response Format
```json
{
  "success": true,
  "task_id": "adv_crop_abc12345",
  "message": "Advanced vertical cropping started successfully",
  "processing_features": [
    "Pyannote Speaker Diarization",
    "PySceneDetect Scene Changes",
    "MediaPipe + MTCNN Face Detection"
  ],
  "status_endpoint": "/workflow/advanced-task-status/abc12345",
  "download_endpoint": "/workflow/download-advanced-result/abc12345"
}
```

## 🎬 Example Usage

```python
import requests

# Upload and process a video
with open("my_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/workflow/upload-vertical-crop-advanced",
        files={"file": f},
        data={
            "target_aspect_ratio": "9:16",
            "use_speaker_detection": True
        }
    )

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/workflow/advanced-task-status/{task_id}")

# Download result when ready
if status.json()["status"] == "completed":
    result = requests.get(f"http://localhost:8000/workflow/download-advanced-result/{task_id}")
``` 