from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
import tempfile
import os

# Import our services
from app.services.youtube import (
    get_video_id, download_video, cut_clips, cut_clips_vertical, DownloadError,
    get_video_info, get_available_formats, youtube_service
)
from app.services.transcript import fetch_youtube_transcript, extract_full_transcript
from app.services.gemini import analyze_transcript_with_gemini
from app.services.vertical_crop import crop_video_to_vertical

router = APIRouter()

class ProcessVideoRequest(BaseModel):
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
    create_vertical: Optional[bool] = False  # Create vertical (9:16) clips
    smoothing_strength: Optional[str] = "very_high"  # low, medium, high, very_high

class VideoInfoRequest(BaseModel):
    youtube_url: str

@router.post("/video-info")
async def get_video_information(request: VideoInfoRequest):
    """
    Get detailed video information including available formats
    """
    try:
        print(f"🔍 Getting video info for: {request.youtube_url}")
        
        # Get video info
        video_info = get_video_info(request.youtube_url)
        
        # Get available formats
        formats = get_available_formats(request.youtube_url)
        
        return {
            "success": True,
            "video_info": video_info,
            "available_formats": formats[:10],  # Top 10 formats
            "supported_qualities": ["best", "8k", "4k", "1440p", "1080p", "720p"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get video info: {str(e)}")

@router.post("/process-complete")
async def process_video_complete(request: ProcessVideoRequest):
    """
    Complete video processing workflow with quality selection:
    1. Extract transcript from YouTube URL
    2. Analyze with Gemini AI to find viral segments  
    3. Download video in specified quality (supports up to 8K)
    4. Cut video into segments based on Gemini analysis
    
    Returns paths to created clips and full analysis
    """
    url = request.youtube_url
    quality = request.quality or "best"
    
    try:
        print(f"\n🚀 Starting complete workflow for: {url}")
        print(f"🎯 Quality setting: {quality}")
        
        # Step 0: Get video info first
        print(f"\n📋 Step 0: Getting video information...")
        video_info = get_video_info(url)
        print(f"📺 Title: {video_info['title']}")
        print(f"⏱️ Duration: {video_info['duration']} sec")
        
        # Step 1: Extract video ID and get transcript
        print(f"\n📝 Step 1: Extracting transcript...")
        video_id = video_info['id']
        
        raw_transcript_data = fetch_youtube_transcript(video_id)
        transcript_result = extract_full_transcript(raw_transcript_data)
        
        if isinstance(transcript_result, dict) and 'error' in transcript_result:
            raise HTTPException(status_code=400, detail=f"Transcript error: {transcript_result['error']}")
        
        print(f"✅ Transcript extracted: {len(transcript_result.get('transcript', ''))} characters")
        
        # Step 2: Analyze with Gemini AI
        print(f"\n🤖 Step 2: Analyzing with Gemini AI...")
        gemini_analysis = await analyze_transcript_with_gemini(transcript_result)
        
        if not gemini_analysis.get("gemini_analysis", {}).get("viral_segments"):
            raise HTTPException(status_code=400, detail="No viral segments found in Gemini analysis")
        
        viral_segments = gemini_analysis["gemini_analysis"]["viral_segments"]
        print(f"✅ Gemini analysis complete: {len(viral_segments)} segments found")
        
        # Step 3: Download video in specified quality
        print(f"\n📥 Step 3: Downloading video in {quality} quality...")
        try:
            video_path = download_video(url, quality)
            print(f"✅ Video downloaded: {video_path}")
            
            # Get file size info
            file_size_mb = video_path.stat().st_size / (1024*1024)
            print(f"📁 File size: {file_size_mb:.1f} MB")
            
        except DownloadError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
        
        # Step 4: Cut video into clips
        print(f"\n✂️ Step 4: Cutting video into clips...")
        try:
            if request.create_vertical:
                print(f"📱 Creating vertical clips in native resolution")
                print(f"🎛️ Smoothing level: {request.smoothing_strength}")
                clip_paths = cut_clips_vertical(
                    video_path, 
                    gemini_analysis, 
                    smoothing_strength=request.smoothing_strength
                )
            else:
                print(f"📺 Creating standard horizontal clips")
                clip_paths = cut_clips(video_path, gemini_analysis)
            print(f"✅ Clips created: {len(clip_paths)} files")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clip cutting failed: {str(e)}")
        
        # Prepare response
        result = {
            "success": True,
            "workflow_steps": {
                "video_info_extraction": True,
                "transcript_extraction": True,
                "gemini_analysis": True, 
                "video_download": True,
                "clip_cutting": True
            },
            "video_info": {
                "id": video_info['id'],
                "title": video_info['title'],
                "duration": video_info['duration'],
                "uploader": video_info.get('uploader'),
                "view_count": video_info.get('view_count'),
                "category": transcript_result.get("category"),
                "description": video_info.get('description', '')[:200] + "..." if video_info.get('description') else "",
                "transcript_length": len(transcript_result.get("transcript", "")),
                "timecodes_count": len(transcript_result.get("timecodes", []))
            },
            "download_info": {
                "quality_requested": quality,
                "file_size_mb": round(file_size_mb, 1),
                "file_path": str(video_path)
            },
            "analysis_results": {
                "viral_segments_found": len(viral_segments),
                "segments": [
                    {
                        "title": seg.get("title"),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": seg.get("duration")
                    }
                    for seg in viral_segments
                ]
            },
            "files_created": {
                "source_video": str(video_path),
                "clips_created": len(clip_paths),
                "clip_paths": [str(p) for p in clip_paths],
                "clip_type": "vertical" if request.create_vertical else "horizontal",
                "resolution": "native" if request.create_vertical else "original"
            }
        }
        
        print(f"\n🎉 Workflow completed successfully!")
        print(f"📊 Summary: {len(viral_segments)} segments → {len(clip_paths)} clips")
        print(f"💾 Total file size: {file_size_mb:.1f} MB")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

@router.post("/download-only")
async def download_only(request: ProcessVideoRequest):
    """
    Download video only in specified quality without processing
    """
    url = request.youtube_url
    quality = request.quality or "best"
    
    try:
        print(f"\n📥 Downloading video: {url}")
        print(f"🎯 Quality: {quality}")
        
        # Get video info first
        video_info = get_video_info(url)
        
        # Download video
        video_path = download_video(url, quality)
        file_size_mb = video_path.stat().st_size / (1024*1024)
        
        return {
            "success": True,
            "video_info": video_info,
            "download_info": {
                "quality_requested": quality,
                "file_size_mb": round(file_size_mb, 1),
                "file_path": str(video_path)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

@router.post("/analyze-only")
async def analyze_only(request: ProcessVideoRequest):
    """
    Only extract transcript and analyze with Gemini (no download/cutting)
    """
    url = request.youtube_url
    
    try:
        print(f"\n🔍 Analyzing video: {url}")
        
        # Step 1: Extract transcript
        video_id = get_video_id(url)
        raw_transcript_data = fetch_youtube_transcript(video_id)
        transcript_result = extract_full_transcript(raw_transcript_data)
        
        if isinstance(transcript_result, dict) and 'error' in transcript_result:
            raise HTTPException(status_code=400, detail=f"Transcript error: {transcript_result['error']}")
        
        # Step 2: Analyze with Gemini
        gemini_analysis = await analyze_transcript_with_gemini(transcript_result)
        
        return {
            "success": True,
            "video_info": {
                "id": transcript_result.get("id"),
                "title": transcript_result.get("title"),
                "category": transcript_result.get("category"),
                "transcript_length": len(transcript_result.get("transcript", ""))
            },
            "analysis": gemini_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

class VerticalCropRequest(BaseModel):
    """Request for creating vertical crops from existing video"""
    video_path: str
    output_path: Optional[str] = None
    use_speaker_detection: Optional[bool] = True
    smoothing_strength: Optional[str] = "very_high"  # low, medium, high, very_high

@router.post("/create-vertical-crop")
async def create_vertical_crop(request: VerticalCropRequest):
    """
    Create a vertically cropped version of an existing video file
    """
    try:
        source_path = Path(request.video_path)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source video not found")
        
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_path = source_path.with_name(f"{source_path.stem}_vertical.mp4")
        
        print(f"🎬 Creating vertical crop for: {source_path}")
        print(f"💾 Saving to: {output_path}")
        print(f"🎛️ Smoothing: {request.smoothing_strength}")

        success = crop_video_to_vertical(
            source_path,
            output_path,
            use_speaker_detection=request.use_speaker_detection,
            smoothing_strength=request.smoothing_strength
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create vertical crop")
            
        return {
            "success": True,
            "output_path": str(output_path)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vertical crop: {str(e)}")

# Temporary file upload directory
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_path: Optional[str] = None
    
@router.post("/test-upload-vertical")
async def test_upload_vertical(
    file: UploadFile = File(...),
    use_speaker_detection: bool = True,
    smoothing_strength: str = "very_high"
):
    """
    Test endpoint for uploading a video and creating a vertical crop from it.
    This demonstrates the vertical cropping functionality on any video file.
    """
    try:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix, dir=TEMP_UPLOADS_DIR) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = Path(tmp.name)
        
        print(f"📄 Uploaded file saved to temporary path: {temp_path}")

        # Define the output path for the cropped video
        output_dir = Path("temp_vertical")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{temp_path.stem}_vertical_{smoothing_strength}.mp4"
        
        print(f"🚀 Starting vertical crop process...")
        print(f"   - Speaker detection: {use_speaker_detection}")
        print(f"   - Smoothing: {smoothing_strength}")

        # Call the vertical cropping service
        success = crop_video_to_vertical(
            input_path=temp_path,
            output_path=output_path,
            use_speaker_detection=use_speaker_detection,
            smoothing_strength=smoothing_strength
        )

        if not success:
            # Clean up the temp file even on failure
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Vertical cropping failed during processing")

        print(f"✅ Vertical crop successful! Output at: {output_path}")
        
        # Return a downloadable link to the file
        return FileResponse(
            path=output_path, 
            media_type='video/mp4', 
            filename=output_path.name
        )

    except HTTPException:
        # Re-raise HTTP exceptions to return proper status codes
        raise
    except Exception as e:
        print(f"❌ An error occurred during vertical crop test: {e}")
        # Clean up temp file on any other exception
        if 'temp_path' in locals() and temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/test-upload-info")
async def test_upload_info(file: UploadFile = File(...)):
    """
    Test endpoint to upload a video and get its info via ffprobe
    """
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 files are supported"
        )
    
    try:
        import cv2
        
        # Import filename sanitization function
        from app.services.youtube import _sanitize_filename
        
        # Sanitize the filename to prevent encoding issues
        safe_filename = _sanitize_filename(file.filename or "upload.mp4")
        
        # Save file temporarily
        upload_dir = Path("temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        temp_path = upload_dir / f"info_{safe_filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze video with OpenCV
        cap = cv2.VideoCapture(str(temp_path))
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file. File may be corrupted.")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        orientation = "landscape" if aspect_ratio > 1 else "portrait" if aspect_ratio < 1 else "square"
        
        # File size
        file_size_mb = len(content) / (1024*1024)
        
        # Cleanup
        temp_path.unlink()
        
        return {
            "success": True,
            "file_info": {
                "filename": file.filename,
                "file_size_mb": round(file_size_mb, 2),
                "duration_seconds": round(duration, 2),
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "resolution": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "orientation": orientation
                }
            },
            "vertical_crop_suitability": {
                "current_format": f"{width}x{height}",
                "is_horizontal": orientation == "landscape",
                "recommended_for_cropping": orientation == "landscape" and width >= 1080,
                "notes": [
                    "Horizontal videos work best for vertical cropping",
                    "Minimum recommended width: 1080px",
                    "Portrait videos may not crop well"
                ]
            }
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="OpenCV not available for video analysis"
        )
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"File analysis failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint to confirm the API is running
    """
    print("🏥 Health check OK")
    return {"status": "ok"} 