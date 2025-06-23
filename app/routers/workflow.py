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

router = APIRouter()

class ProcessVideoRequest(BaseModel):
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
    create_vertical: Optional[bool] = False  # Create vertical (9:16) clips
    vertical_resolution: Optional[str] = "shorts_hd"  # shorts_hd, shorts_fhd, tiktok
    smoothing_strength: Optional[str] = "medium"  # low, medium, high - control head movement smoothing

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
                print(f"📱 Creating vertical clips in {request.vertical_resolution} resolution")
                print(f"🎛️ Smoothing level: {request.smoothing_strength}")
                clip_paths = cut_clips_vertical(
                    video_path, 
                    gemini_analysis, 
                    request.vertical_resolution,
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
                "resolution": request.vertical_resolution if request.create_vertical else "original"
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
    resolution: Optional[str] = "shorts_hd"  # shorts_hd, shorts_fhd, tiktok
    use_speaker_detection: Optional[bool] = True
    smoothing_strength: Optional[str] = "medium"  # low, medium, high - control head movement smoothing

@router.post("/create-vertical-crop")
async def create_vertical_crop(request: VerticalCropRequest):
    """
    Create a vertical (9:16) crop from an existing video file
    """
    try:
        # Import inside function to handle potential import errors
        from app.services.vertical_crop import crop_video_to_vertical, get_available_resolutions
        
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
        
        # Generate output path if not provided
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            # Create output in vertical subfolder
            clips_dir = Path("clips") / "vertical"
            clips_dir.mkdir(parents=True, exist_ok=True)
            base_name = video_path.stem
            output_path = clips_dir / f"{base_name}_vertical.mp4"
        
        # Get available resolutions
        resolutions = get_available_resolutions()
        if request.resolution not in resolutions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid resolution. Available: {list(resolutions.keys())}"
            )
        
        print(f"📱 Creating vertical crop:")
        print(f"   Input: {video_path}")
        print(f"   Output: {output_path}")
        print(f"   Resolution: {request.resolution}")
        print(f"   Speaker detection: {request.use_speaker_detection}")
        print(f"   Smoothing: {request.smoothing_strength}")
        
        # Create vertical crop
        success = crop_video_to_vertical(
            input_path=video_path,
            output_path=output_path,
            resolution=request.resolution,
            use_speaker_detection=request.use_speaker_detection,
            smoothing_strength=request.smoothing_strength
        )
        
        if success and output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024*1024)
            target_size = resolutions[request.resolution]
            
            return {
                "success": True,
                "input_path": str(video_path),
                "output_path": str(output_path),
                "resolution": request.resolution,
                "target_size": f"{target_size[0]}x{target_size[1]}",
                "file_size_mb": round(file_size_mb, 1),
                "speaker_detection_used": request.use_speaker_detection
            }
        else:
            raise HTTPException(status_code=500, detail="Vertical crop creation failed")
    
    except ImportError:
        raise HTTPException(
            status_code=503, 
            detail="Vertical cropping service not available. Install required dependencies."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertical crop failed: {str(e)}")

@router.get("/vertical-resolutions")
async def get_vertical_resolutions():
    """
    Get available vertical video resolutions for cropping
    """
    try:
        from app.services.vertical_crop import get_available_resolutions
        
        resolutions = get_available_resolutions()
        
        return {
            "success": True,
            "available_resolutions": {
                name: {
                    "width": size[0],
                    "height": size[1],
                    "aspect_ratio": "9:16",
                    "description": {
                        "shorts_hd": "YouTube Shorts HD",
                        "shorts_fhd": "YouTube Shorts Full HD", 
                        "tiktok": "TikTok format"
                    }.get(name, name)
                }
                for name, size in resolutions.items()
            },
            "default": "shorts_hd"
        }
        
    except ImportError:
                 raise HTTPException(
             status_code=503,
             detail="Vertical cropping service not available"
         )

@router.post("/test-upload-vertical")
async def test_upload_vertical(
    file: UploadFile = File(...),
    resolution: str = "shorts_hd",
    use_speaker_detection: bool = True,
    smoothing_strength: str = "medium"  # low, medium, high - control head movement smoothing
):
    """
    🧪 TEST ENDPOINT: Upload an MP4 file and convert it to vertical format
    
    Enhanced with motion smoothing to prevent jerky head movements!
    
    Args:
        file: MP4 video file to upload
        resolution: Target resolution (shorts_hd, shorts_fhd, tiktok)
        use_speaker_detection: Whether to use AI speaker detection
        smoothing_strength: Motion smoothing level:
            - "low": Minimal smoothing, more responsive (alpha=0.4, max_jump=50px)
            - "medium": Balanced smoothing (alpha=0.2, max_jump=30px) 
            - "high": Maximum smoothing, very stable (alpha=0.1, max_jump=20px)
    
    Returns:
        Converted vertical video file for download
    """
    
    # Validate smoothing strength
    if smoothing_strength not in ["low", "medium", "high"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid smoothing_strength. Must be 'low', 'medium', or 'high'"
        )
    
    # Validate file type
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400, 
            detail="Only MP4 files are supported. Please upload an .mp4 file."
        )
    
    # Import filename sanitization function
    from app.services.youtube import _sanitize_filename
    
    # Check if vertical cropping is available
    try:
        from app.services.vertical_crop import crop_video_to_vertical, get_available_resolutions
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Install required dependencies: opencv-python, pydub, webrtcvad"
        )
    
    # Validate resolution
    available_resolutions = get_available_resolutions()
    if resolution not in available_resolutions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution '{resolution}'. Available: {list(available_resolutions.keys())}"
        )
    
    # Create temporary directories
    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    vertical_dir = Path("temp_vertical")
    vertical_dir.mkdir(exist_ok=True)
    
    temp_input_path = None
    temp_output_path = None
    
    try:
        # Sanitize the filename to prevent encoding issues
        safe_filename = _sanitize_filename(file.filename or "upload.mp4")
        
        print(f"\n🧪 TEST UPLOAD: Converting {file.filename} to vertical format")
        print(f"📱 Target resolution: {resolution}")
        print(f"🤖 Speaker detection: {use_speaker_detection}")
        print(f"🎛️ Motion smoothing: {smoothing_strength}")
        print(f"🔧 Safe filename: {safe_filename}")
        
        # Save uploaded file temporarily with sanitized name
        temp_input_path = upload_dir / f"upload_{safe_filename}"
        
        with open(temp_input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size_mb = len(content) / (1024*1024)
        print(f"📁 Uploaded file size: {file_size_mb:.1f} MB")
        
        # Check if file is valid
        if not temp_input_path.exists() or temp_input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted")
        
        # Generate output filename using sanitized name
        base_name = Path(safe_filename).stem
        output_filename = f"{base_name}_vertical_{resolution}.mp4"
        temp_output_path = vertical_dir / output_filename
        
        print(f"🔄 Converting to vertical format...")
        print(f"   Input: {temp_input_path.name}")
        print(f"   Output: {output_filename}")
        
        # Convert to vertical format
        success = crop_video_to_vertical(
            input_path=temp_input_path,
            output_path=temp_output_path,
            resolution=resolution,
            use_speaker_detection=use_speaker_detection,
            smoothing_strength=smoothing_strength
        )
        
        if not success or not temp_output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Vertical conversion failed. Check if the uploaded file is a valid MP4 video."
            )
        
        # Get output file info
        output_size_mb = temp_output_path.stat().st_size / (1024*1024)
        target_size = available_resolutions[resolution]
        
        print(f"✅ Conversion successful!")
        print(f"📊 Output size: {output_size_mb:.1f} MB")
        print(f"🎯 Resolution: {target_size[0]}x{target_size[1]}")
        
        # Return the converted file
        return FileResponse(
            path=str(temp_output_path),
            filename=output_filename,
            media_type="video/mp4",
            headers={
                "X-Original-Filename": file.filename,  # Keep original for reference
                "X-Safe-Filename": safe_filename,      # Show sanitized version
                "X-Conversion-Resolution": resolution,
                "X-Target-Size": f"{target_size[0]}x{target_size[1]}",
                "X-Speaker-Detection": str(use_speaker_detection),
                "X-Smoothing-Strength": smoothing_strength,
                "X-Input-Size-MB": f"{file_size_mb:.1f}",
                "X-Output-Size-MB": f"{output_size_mb:.1f}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload conversion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files (optional - you might want to keep them for debugging)
        # Uncomment these lines if you want automatic cleanup:
        
        # if temp_input_path and temp_input_path.exists():
        #     temp_input_path.unlink()
        #     print(f"🧹 Cleaned up input file: {temp_input_path.name}")
        
        # Note: We don't delete the output file here because FileResponse needs it
        # You might want to implement a cleanup task that runs periodically
        pass

@router.post("/test-upload-info")
async def test_upload_info(file: UploadFile = File(...)):
    """
    🧪 TEST ENDPOINT: Get information about an uploaded MP4 file
    
    This endpoint analyzes the uploaded file and provides technical details
    without converting it. Useful for debugging and validation.
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
    """Health check for workflow service"""
    try:
        from app.services.vertical_crop import get_available_resolutions
        vertical_crop_available = True
        vertical_resolutions = list(get_available_resolutions().keys())
    except ImportError:
        vertical_crop_available = False
        vertical_resolutions = []
    
    return {
        "status": "healthy",
        "service": "workflow",
        "capabilities": [
            "video_info_extraction",
            "transcript_extraction",
            "gemini_analysis", 
            "hq_video_download",  # Updated capability
            "8k_video_support",   # New capability
            "clip_cutting",
            "vertical_cropping" if vertical_crop_available else "vertical_cropping_unavailable"
        ],
        "supported_qualities": ["best", "8k", "4k", "1440p", "1080p", "720p"],
        "vertical_cropping": {
            "available": vertical_crop_available,
            "resolutions": vertical_resolutions
        }
    } 