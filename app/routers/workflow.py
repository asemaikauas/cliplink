from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Form, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import datetime
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our services
from app.services.youtube import (
    get_video_id, download_video, cut_clips, DownloadError,
    get_video_info, get_available_formats, youtube_service
)
from app.services.transcript import fetch_youtube_transcript, extract_full_transcript
from app.services.gemini import analyze_transcript_with_gemini

# Import the new crop router if available
try:
    from app.routers.crop import router as crop_router
    CROP_ROUTER_AVAILABLE = True
except ImportError:
    CROP_ROUTER_AVAILABLE = False

# Intelligent cropper imports removed

# Helper functions
async def download_video_temp(video_url: str) -> Path:
    """Download video to temporary location"""
    try:
        # Use the existing youtube service to download
        video_path = await _run_blocking_task(download_video, video_url, "best")
        return video_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

# Add placeholder imports for authentication (will need to be implemented)
# For now, we'll create simple placeholder functions
def get_current_active_user():
    """Placeholder for user authentication dependency"""
    return {"id": 1, "username": "test_user"}

class User:
    """Placeholder User model"""
    def __init__(self, id: int, username: str):
        self.id = id
        self.username = username

router = APIRouter()

# Directory constants
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Global task management for complete workflow processing
workflow_tasks: Dict[str, Dict] = {}
workflow_task_lock = threading.Lock()

# Thread pool for CPU-intensive tasks
workflow_executor = ThreadPoolExecutor(max_workers=6)  # Adjust based on your server capacity

class ProcessVideoRequest(BaseModel):
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
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

class AsyncProcessVideoRequest(BaseModel):
    """Request for async complete video processing"""
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
    smoothing_strength: Optional[str] = "very_high"  # low, medium, high, very_high
    priority: Optional[str] = "normal"  # low, normal, high
    notify_webhook: Optional[str] = None  # Optional webhook URL for completion notification

async def _run_blocking_task(func, *args, **kwargs):
    """Run blocking functions in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(workflow_executor, func, *args, **kwargs)

def _update_workflow_progress(task_id: str, step: str, progress: int, message: str, data: Optional[Dict] = None):
    """Update workflow task progress with thread safety"""
    with workflow_task_lock:
        if task_id in workflow_tasks:
            workflow_tasks[task_id].update({
                "current_step": step,
                "progress": progress,
                "message": message,
                "updated_at": datetime.now()
            })
            if data:
                workflow_tasks[task_id].update(data)

async def _process_video_workflow_async(
    task_id: str,
    youtube_url: str,
    quality: str,
    smoothing_strength: str
):
    """
    Async implementation of the complete video processing workflow
    """
    try:
        _update_workflow_progress(task_id, "init", 5, f"Starting workflow for: {youtube_url}")
        
        # Step 1: Get video info (5-10%)
        _update_workflow_progress(task_id, "video_info", 5, "Getting video information...")
        video_info = await _run_blocking_task(get_video_info, youtube_url)
        
        _update_workflow_progress(
            task_id, "video_info", 10, 
            f"Video info retrieved: {video_info['title']}", 
            {"video_info": video_info}
        )
        
        # Step 2: Extract transcript (10-25%)
        _update_workflow_progress(task_id, "transcript", 10, "Extracting transcript...")
        video_id = video_info['id']
        
        raw_transcript_data = await _run_blocking_task(fetch_youtube_transcript, video_id)
        transcript_result = await _run_blocking_task(extract_full_transcript, raw_transcript_data)
        
        if isinstance(transcript_result, dict) and 'error' in transcript_result:
            raise Exception(f"Transcript error: {transcript_result['error']}")
        
        _update_workflow_progress(
            task_id, "transcript", 25, 
            f"Transcript extracted: {len(transcript_result.get('transcript', ''))} characters",
            {"transcript_result": transcript_result}
        )
        
        # Step 3: Gemini Analysis (25-40%)
        _update_workflow_progress(task_id, "analysis", 25, "Analyzing with Gemini AI...")
        gemini_analysis = await analyze_transcript_with_gemini(transcript_result)
        
        if not gemini_analysis.get("gemini_analysis", {}).get("viral_segments"):
            raise Exception("No viral segments found in Gemini analysis")
        
        viral_segments = gemini_analysis["gemini_analysis"]["viral_segments"]
        _update_workflow_progress(
            task_id, "analysis", 40, 
            f"Gemini analysis complete: {len(viral_segments)} segments found",
            {"gemini_analysis": gemini_analysis}
        )
        
        # Step 4: Download video (40-60%)
        _update_workflow_progress(task_id, "download", 40, f"Downloading video in {quality} quality...")
        
        try:
            video_path = await _run_blocking_task(download_video, youtube_url, quality)
            file_size_mb = video_path.stat().st_size / (1024*1024)
            
            _update_workflow_progress(
                task_id, "download", 60, 
                f"Video downloaded: {file_size_mb:.1f} MB",
                {
                    "video_path": str(video_path),
                    "file_size_mb": file_size_mb
                }
            )
        except DownloadError as e:
            raise Exception(f"Download failed: {str(e)}")
        
        # Step 5: Cut clips (60-95%)
        _update_workflow_progress(task_id, "cutting", 60, "Cutting video into clips...")
        
        try:
            _update_workflow_progress(task_id, "cutting", 65, "Creating standard horizontal clips...")
            clip_paths = await _run_blocking_task(cut_clips, video_path, gemini_analysis)
            
            _update_workflow_progress(
                task_id, "cutting", 95, 
                f"Clips created: {len(clip_paths)} files",
                {"clip_paths": [str(p) for p in clip_paths]}
            )
        except Exception as e:
            raise Exception(f"Clip cutting failed: {str(e)}")
        
        # Step 6: Finalize (95-100%)
        _update_workflow_progress(task_id, "finalizing", 95, "Finalizing results...")
        
        # Prepare final result
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
                "clip_type": "horizontal",
                "resolution": "original"
            }
        }
        
        # Mark as completed
        with workflow_task_lock:
            workflow_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": f"Workflow completed successfully! {len(viral_segments)} segments → {len(clip_paths)} clips",
                "result": result,
                "completed_at": datetime.now()
            })
        
        return result
        
    except Exception as e:
        # Mark as failed
        with workflow_task_lock:
            workflow_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": f"Workflow failed: {str(e)}",
                "completed_at": datetime.now()
            })
        raise e

@router.post("/process-complete-async")
async def process_video_complete_async(request: AsyncProcessVideoRequest):
    """
    Async complete video processing workflow with progress tracking:
    1. Extract transcript from YouTube URL
    2. Analyze with Gemini AI to find viral segments  
    3. Download video in specified quality (supports up to 8K)
    4. Cut video into segments based on Gemini analysis
    
    Returns immediately with task_id for status polling
    """
    try:
        # Generate unique task ID
        task_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Initialize task tracking
        with workflow_task_lock:
            workflow_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "created_at": datetime.now(),
                "youtube_url": request.youtube_url,
                "quality": request.quality or "best",
                "smoothing_strength": request.smoothing_strength or "very_high",
                "priority": request.priority or "normal",
                "notify_webhook": request.notify_webhook,
                "current_step": "queued",
                "message": "Workflow queued for processing",
                "error": None
            }
        
        print(f"🚀 Async workflow {task_id} queued: {request.youtube_url}")
        print(f"🎯 Settings: quality={request.quality}, smoothing={request.smoothing_strength}")
        
        # Start async processing (don't await - let it run in background)
        asyncio.create_task(_process_video_workflow_async(
            task_id,
            request.youtube_url,
            request.quality or "best",
            request.smoothing_strength or "very_high"
        ))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Complete video processing workflow started",
            "youtube_url": request.youtube_url,
            "settings": {
                "quality": request.quality or "best",
                "smoothing_strength": request.smoothing_strength or "very_high"
            },
            "status_endpoint": f"/workflow/workflow-status/{task_id}",
            "estimated_time": "5-20 minutes depending on video length and quality"
        }
        
    except Exception as e:
        print(f"❌ Failed to start async workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.get("/workflow-status/{task_id}")
async def get_workflow_status(task_id: str):
    """
    Get the status and progress of a complete workflow task
    """
    with workflow_task_lock:
        task = workflow_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Workflow task {task_id} not found")
    
    # Calculate processing time
    created_at = task.get("created_at")
    updated_at = task.get("updated_at", created_at)
    completed_at = task.get("completed_at")
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "current_step": task["current_step"],
        "message": task["message"],
        "youtube_url": task["youtube_url"],
        "settings": {
            "quality": task["quality"],
            "smoothing_strength": task["smoothing_strength"]
        },
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }
    
    if completed_at:
        response["completed_at"] = completed_at.isoformat()
        processing_time = (completed_at - created_at).total_seconds()
        response["processing_time_seconds"] = round(processing_time, 2)
        response["processing_time_formatted"] = f"{int(processing_time // 60)}:{int(processing_time % 60):02d}"
    
    if task.get("error"):
        response["error"] = task["error"]
    
    # Add partial results if available
    if "video_info" in task:
        response["video_info"] = task["video_info"]
    if "file_size_mb" in task:
        response["download_info"] = {
            "file_size_mb": task["file_size_mb"],
            "video_path": task.get("video_path")
        }
    if "clip_paths" in task:
        response["clips_info"] = {
            "clips_created": len(task["clip_paths"]),
            "clip_paths": task["clip_paths"]
        }
    
    # Add complete result if finished
    if task["status"] == "completed" and "result" in task:
        response["result"] = task["result"]
        response["download_endpoint"] = f"/workflow/download-workflow-result/{task_id}"
    
    return response

@router.get("/download-workflow-result/{task_id}")
async def download_workflow_result(task_id: str, clip_index: Optional[int] = None):
    """
    Download results from a completed workflow task
    If clip_index is specified, download that specific clip, otherwise return the source video
    """
    with workflow_task_lock:
        task = workflow_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Workflow task {task_id} not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Workflow task {task_id} is not completed yet")
    
    if "result" not in task:
        raise HTTPException(status_code=404, detail="Result data not found")
    
    result = task["result"]
    
    if clip_index is not None:
        # Download specific clip
        clip_paths = result["files_created"]["clip_paths"]
        if clip_index >= len(clip_paths):
            raise HTTPException(status_code=404, detail=f"Clip index {clip_index} not found")
        
        file_path = Path(clip_paths[clip_index])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Clip file not found")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='video/mp4'
        )
    else:
        # Download source video
        video_path = Path(result["download_info"]["file_path"])
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Source video file not found")
        
        return FileResponse(
            path=str(video_path),
            filename=video_path.name,
            media_type='video/mp4'
        )

@router.get("/workflow-tasks")
async def list_workflow_tasks():
    """
    List all workflow processing tasks
    """
    with workflow_task_lock:
        tasks = {tid: task.copy() for tid, task in workflow_tasks.items()}
    
    # Format response
    formatted_tasks = {}
    for task_id, task in tasks.items():
        processing_time = 0
        if task.get("updated_at") and task.get("created_at"):
            processing_time = (task["updated_at"] - task["created_at"]).total_seconds()
        elif task.get("completed_at") and task.get("created_at"):
            processing_time = (task["completed_at"] - task["created_at"]).total_seconds()
        
        formatted_tasks[task_id] = {
            "status": task["status"],
            "progress": task["progress"],
            "current_step": task["current_step"],
            "message": task["message"],
            "youtube_url": task["youtube_url"],
            "settings": {
                "quality": task["quality"],
                "smoothing_strength": task["smoothing_strength"]
            },
            "created_at": task["created_at"].isoformat(),
            "processing_time_seconds": round(processing_time, 1)
        }
    
    return {
        "workflow_tasks": formatted_tasks,
        "total_tasks": len(tasks),
        "queued": len([t for t in tasks.values() if t["status"] == "queued"]),
        "processing": len([t for t in tasks.values() if t["status"] == "processing"]),
        "completed": len([t for t in tasks.values() if t["status"] == "completed"]),
        "failed": len([t for t in tasks.values() if t["status"] == "failed"])
    }

@router.post("/cleanup-workflow-tasks")
async def cleanup_workflow_tasks(max_age_hours: int = 24):
    """
    Clean up completed workflow tasks older than specified hours
    """
    current_time = datetime.now()
    to_remove = []
    
    with workflow_task_lock:
        for task_id, task in workflow_tasks.items():
            if task["status"] in ["completed", "failed"]:
                completed_at = task.get("completed_at", task.get("created_at"))
                if completed_at and (current_time - completed_at).total_seconds() > max_age_hours * 3600:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del workflow_tasks[task_id]
    
    return {
        "success": True,
        "cleaned_tasks": len(to_remove),
        "remaining_tasks": len(workflow_tasks),
        "message": f"Cleaned up {len(to_remove)} workflow tasks older than {max_age_hours} hours"
    }

# Keep the original synchronous endpoint for backward compatibility
@router.post("/process-complete")
async def process_video_complete(request: ProcessVideoRequest):
    """
    Complete video processing workflow (LEGACY - consider using /process-complete-async)
    For backward compatibility - this is still async but waits for completion
    """
    # Convert to async request and wait for completion
    async_request = AsyncProcessVideoRequest(**request.dict())
    
    # Start the async workflow
    response = await process_video_complete_async(async_request)
    task_id = response["task_id"]
    
    # Poll for completion (with timeout)
    max_wait_time = 1800  # 30 minutes max
    poll_interval = 2  # Check every 2 seconds
    waited_time = 0
    
    while waited_time < max_wait_time:
        await asyncio.sleep(poll_interval)
        waited_time += poll_interval
        
        status_response = await get_workflow_status(task_id)
        
        if status_response["status"] == "completed":
            return status_response["result"]
        elif status_response["status"] == "failed":
            raise HTTPException(status_code=500, detail=status_response.get("error", "Workflow failed"))
    
    # If we reach here, it's a timeout
    raise HTTPException(status_code=408, detail="Workflow timeout - use /process-complete-async for long-running tasks")

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

@router.get("/health")
async def health_check():
    """
    Health check endpoint to confirm the API is running
    """
    print("🏥 Health check OK")
    return {"status": "ok"}

# Vertical cropping endpoints removed

# Legacy vertical cropping endpoints removed

# Advanced vertical cropping upload endpoint removed

# HuggingFace test endpoint removed

# Intelligent cropping endpoints and classes removed

# Intelligent crop analyze endpoint removed

# Intelligent crop endpoint removed

# Intelligent crop upload endpoint removed

# All intelligent cropping config endpoints removed 