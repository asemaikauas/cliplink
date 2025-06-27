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
    get_video_id, download_video, cut_clips, cut_clips_vertical, cut_clips_vertical_async, DownloadError,
    get_video_info, get_available_formats, youtube_service
)
from app.services.transcript import fetch_youtube_transcript, extract_full_transcript
from app.services.gemini import analyze_transcript_with_gemini
from app.services.vertical_crop import crop_video_to_vertical
from app.services.vertical_crop_async import (
    crop_video_to_vertical_async,
    async_vertical_crop_service,
    get_crop_task_status,
    list_crop_tasks,
    cleanup_old_crop_tasks,
    # Advanced functions
    initialize_advanced_service,
    crop_video_advanced,
    get_advanced_crop_task_status
)

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

class AsyncProcessVideoRequest(BaseModel):
    """Request for async complete video processing"""
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
    create_vertical: Optional[bool] = False  # Create vertical (9:16) clips
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
    create_vertical: bool,
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
            if create_vertical:
                _update_workflow_progress(task_id, "cutting", 65, f"Creating vertical clips with {smoothing_strength} smoothing...")
                clip_paths = await cut_clips_vertical_async(
                    video_path, 
                    gemini_analysis, 
                    smoothing_strength
                )
            else:
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
                "clip_type": "vertical" if create_vertical else "horizontal",
                "resolution": "native" if create_vertical else "original"
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
                "create_vertical": request.create_vertical,
                "smoothing_strength": request.smoothing_strength,
                "priority": request.priority or "normal",
                "notify_webhook": request.notify_webhook,
                "current_step": "queued",
                "message": "Workflow queued for processing",
                "error": None
            }
        
        print(f"🚀 Async workflow {task_id} queued: {request.youtube_url}")
        print(f"🎯 Settings: quality={request.quality}, vertical={request.create_vertical}, smoothing={request.smoothing_strength}")
        
        # Start async processing (don't await - let it run in background)
        asyncio.create_task(_process_video_workflow_async(
            task_id,
            request.youtube_url,
            request.quality or "best",
            request.create_vertical or False,
            request.smoothing_strength or "very_high"
        ))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Complete video processing workflow started",
            "youtube_url": request.youtube_url,
            "settings": {
                "quality": request.quality or "best",
                "create_vertical": request.create_vertical or False,
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
            "create_vertical": task["create_vertical"],
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
                "create_vertical": task["create_vertical"]
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

class VerticalCropRequest(BaseModel):
    """Request for creating vertical crops from existing video"""
    video_path: str
    output_path: Optional[str] = None
    use_speaker_detection: Optional[bool] = True
    smoothing_strength: Optional[str] = "very_high"  # low, medium, high, very_high

class AsyncVerticalCropRequest(BaseModel):
    """Request for async vertical crop processing"""
    video_path: str
    output_path: Optional[str] = None
    use_speaker_detection: Optional[bool] = True
    smoothing_strength: Optional[str] = "very_high"
    priority: Optional[str] = "normal"  # low, normal, high

@router.post("/create-vertical-crop-async")
async def create_vertical_crop_async(request: AsyncVerticalCropRequest):
    """
    Create vertical crop asynchronously with progress tracking
    Returns immediately with task_id for status polling
    """
    try:
        input_path = Path(request.video_path)
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
        
        # Generate output path if not provided
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_dir = Path("temp_vertical") / "clips"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_vertical.mp4"
        
        print(f"🚀 Starting async vertical crop: {input_path.name}")
        print(f"📱 Output: {output_path}")
        print(f"🎛️ Smoothing: {request.smoothing_strength}")
        print(f"🔊 Speaker detection: {request.use_speaker_detection}")
        
        # Start async processing
        result = await crop_video_to_vertical_async(
            input_path=input_path,
            output_path=output_path,
            use_speaker_detection=request.use_speaker_detection,
            smoothing_strength=request.smoothing_strength
        )
        task_id = result["task_id"]
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Vertical crop processing started",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status_endpoint": f"/workflow/task-status/{task_id}",
            "estimated_time": "2-10 minutes depending on video length"
        }
        
    except Exception as e:
        print(f"❌ Async vertical crop failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status and progress of an async task
    """
    task_status = await async_vertical_crop_service.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Add time information
    created_at = task_status.get("created_at")
    updated_at = task_status.get("updated_at", created_at)
    completed_at = task_status.get("completed_at")
    
    response = {
        "task_id": task_id,
        "status": task_status["status"],
        "progress": task_status["progress"],
        "message": task_status["message"],
        "input_path": task_status["input_path"],
        "output_path": task_status["output_path"],
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }
    
    if completed_at:
        response["completed_at"] = completed_at.isoformat()
        
        # Calculate processing time
        if created_at:
            processing_time = (completed_at - created_at).total_seconds()
            response["processing_time_seconds"] = round(processing_time, 2)
    
    if task_status.get("error"):
        response["error"] = task_status["error"]
    
    # Add file info if completed
    if task_status["status"] == "completed":
        output_path = Path(task_status["output_path"])
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024*1024)
            response["output_file_size_mb"] = round(file_size_mb, 2)
            response["download_endpoint"] = f"/workflow/download-result/{task_id}"
    
    return response

@router.get("/download-result/{task_id}")
async def download_task_result(task_id: str):
    """
    Download the result file of a completed task
    """
    task_status = await async_vertical_crop_service.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_status["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed yet")
    
    output_path = Path(task_status["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=str(output_path),
        filename=output_path.name,
        media_type='video/mp4'
    )

@router.get("/active-tasks")
async def list_active_tasks():
    """
    List all active processing tasks
    """
    tasks = await async_vertical_crop_service.list_active_tasks()
    
    # Format response
    formatted_tasks = {}
    for task_id, task in tasks.items():
        formatted_tasks[task_id] = {
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"].isoformat(),
            "input_file": Path(task["input_path"]).name,
            "processing_time": (
                (task.get("updated_at", task["created_at"]) - task["created_at"]).total_seconds()
                if task.get("updated_at") else 0
            )
        }
    
    return {
        "active_tasks": formatted_tasks,
        "total_tasks": len(tasks),
        "queued": len([t for t in tasks.values() if t["status"] == "queued"]),
        "processing": len([t for t in tasks.values() if t["status"] == "processing"]),
        "completed": len([t for t in tasks.values() if t["status"] == "completed"]),
        "failed": len([t for t in tasks.values() if t["status"] == "failed"])
    }

@router.post("/cleanup-tasks")
async def cleanup_completed_tasks(max_age_hours: int = 24):
    """
    Clean up completed tasks older than specified hours
    """
    before_count = len(await async_vertical_crop_service.list_active_tasks())
    await async_vertical_crop_service.cleanup_completed_tasks(max_age_hours)
    after_count = len(await async_vertical_crop_service.list_active_tasks())
    
    cleaned_count = before_count - after_count
    
    return {
        "success": True,
        "cleaned_tasks": cleaned_count,
        "remaining_tasks": after_count,
        "message": f"Cleaned up {cleaned_count} tasks older than {max_age_hours} hours"
    }

# Update the existing create_vertical_crop endpoint to support both sync and async
@router.post("/create-vertical-crop")
async def create_vertical_crop(request: VerticalCropRequest, async_processing: bool = False):
    """
    Create vertical crop - supports both sync and async processing
    """
    if async_processing:
        # Redirect to async endpoint
        async_request = AsyncVerticalCropRequest(**request.dict())
        return await create_vertical_crop_async(async_request)
    
    # ... existing synchronous code ...

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

@router.post("/create-vertical-crop-advanced")
async def create_vertical_crop_advanced(
    request: Request,
    background_tasks: BackgroundTasks,
    input_video_url: str = Form(...),
    output_filename: str = Form(None),
    target_aspect_ratio: str = Form("9:16"),
    use_speaker_detection: bool = Form(True),
    hf_token: str = Form(None),  # Optional - will use env var if not provided
    current_user: User = Depends(get_current_active_user)
):
    """
    Create vertical crop using advanced AI speaker tracking (pyannote + MediaPipe + scene detection)
    Requires HuggingFace token for speaker diarization
    """
    try:
        # Parse aspect ratio
        aspect_parts = target_aspect_ratio.split(":")
        if len(aspect_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid aspect ratio format. Use 'width:height' (e.g., '9:16')")
        
        target_ratio = (int(aspect_parts[0]), int(aspect_parts[1]))
        
        # Generate unique filename for output
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"advanced_vertical_crop_{timestamp}.mp4"
        
        # Ensure filename has .mp4 extension
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        
        # Setup paths
        input_path = await download_video_temp(input_video_url)
        temp_vertical_dir = Path("temp_vertical")
        temp_vertical_dir.mkdir(exist_ok=True)
        output_path = temp_vertical_dir / output_filename
        
        # Initialize advanced service with HF token
        initialize_advanced_service(hf_token)
        
        # Start advanced vertical cropping
        result = await crop_video_advanced(
            input_path=input_path,
            output_path=output_path,
            hf_token=hf_token,
            target_aspect_ratio=target_ratio
        )
        
        if result["success"]:
            # Clean up input file
            if input_path.exists():
                os.remove(input_path)
            
            return {
                "success": True,
                "task_id": result["task_id"],
                "message": "Advanced vertical cropping started successfully",
                "output_path": str(output_path),
                "features_used": [
                    "Pyannote Speaker Diarization",
                    "PySceneDetect Scene Changes", 
                    "MediaPipe + MTCNN Face Detection",
                    "Advanced Transition Management"
                ]
            }
        else:
            # Clean up on failure
            if input_path.exists():
                os.remove(input_path)
            
            raise HTTPException(
                status_code=500, 
                detail=f"Advanced vertical cropping failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        print(f"❌ Advanced vertical crop endpoint error: {str(e)}")
        
        # Clean up on error
        if 'input_path' in locals() and input_path.exists():
            os.remove(input_path)
        
        raise HTTPException(status_code=500, detail=f"Advanced processing failed: {str(e)}")

@router.get("/advanced-task-status/{task_id}")
async def get_advanced_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get status of advanced vertical cropping task"""
    try:
        status = await get_advanced_crop_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Advanced task not found")
        
        return {
            "success": True,
            "task_status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting advanced task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

@router.get("/download-advanced-result/{task_id}")
async def download_advanced_result(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Download the result of advanced vertical cropping task"""
    try:
        # Get task status to find output file
        status = await get_advanced_crop_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Advanced task not found")
        
        if status["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Task not completed. Current status: {status['status']}")
        
        output_path = status.get("output_path")
        if not output_path or not Path(output_path).exists():
            raise HTTPException(status_code=404, detail="Advanced processed video file not found")
        
        # Return file for download
        return FileResponse(
            path=output_path,
            media_type='video/mp4',
            filename=f"advanced_vertical_{task_id}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error downloading advanced result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@router.post("/upload-vertical-crop-advanced")
async def upload_vertical_crop_advanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_filename: str = Form(None),
    target_aspect_ratio: str = Form("9:16"),
    use_speaker_detection: bool = Form(True),
    hf_token: str = Form(None),  # Optional - will use env var if not provided
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload a video file and create vertical crop using advanced AI speaker tracking
    Same as create-vertical-crop-advanced but with direct file upload
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload MP4, AVI, MOV, MKV, or WebM files."
            )
        
        # Parse aspect ratio
        aspect_parts = target_aspect_ratio.split(":")
        if len(aspect_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid aspect ratio format. Use 'width:height' (e.g., '9:16')")
        
        target_ratio = (int(aspect_parts[0]), int(aspect_parts[1]))
        
        # Generate unique filename for output
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(file.filename).stem
            output_filename = f"advanced_upload_{base_name}_{timestamp}.mp4"
        
        # Ensure filename has .mp4 extension
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        
        # Save uploaded file temporarily
        temp_uploads_dir = Path("temp_uploads")
        temp_uploads_dir.mkdir(exist_ok=True)
        
        # Sanitize filename
        from app.services.youtube import _sanitize_filename
        safe_filename = _sanitize_filename(file.filename or "upload.mp4")
        temp_input_path = temp_uploads_dir / f"upload_{uuid.uuid4().hex[:8]}_{safe_filename}"
        
        # Save uploaded file
        print(f"📁 Saving uploaded file: {file.filename} ({file.size} bytes)")
        with open(temp_input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate file size (optional - adjust as needed)
        file_size_mb = len(content) / (1024 * 1024)
        max_size_mb = 500  # 500 MB limit
        if file_size_mb > max_size_mb:
            # Clean up
            if temp_input_path.exists():
                os.remove(temp_input_path)
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f} MB). Maximum size is {max_size_mb} MB."
            )
        
        # Setup output path
        temp_vertical_dir = Path("temp_vertical")
        temp_vertical_dir.mkdir(exist_ok=True)
        output_path = temp_vertical_dir / output_filename
        
        print(f"🚀 Starting advanced vertical crop for uploaded file:")
        print(f"   📹 Input: {file.filename} ({file_size_mb:.1f} MB)")
        print(f"   📱 Output: {output_filename}")
        print(f"   🎯 Aspect ratio: {target_aspect_ratio}")
        print(f"   🔊 Speaker detection: {use_speaker_detection}")
        
        # Initialize advanced service (will use HF CLI auth if no token provided)
        initialize_advanced_service(hf_token)
        
        # Start advanced vertical cropping
        result = await crop_video_advanced(
            input_path=temp_input_path,
            output_path=output_path,
            target_aspect_ratio=target_ratio,
            hf_token=hf_token
        )
        
        # Clean up input file in background after processing starts
        def cleanup_input_file():
            try:
                if temp_input_path.exists():
                    os.remove(temp_input_path)
                    print(f"🧹 Cleaned up temporary input file: {temp_input_path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up input file: {e}")
        
        background_tasks.add_task(cleanup_input_file)
        
        if result["success"]:
            return {
                "success": True,
                "task_id": result["task_id"],
                "message": "Advanced vertical cropping started successfully for uploaded file",
                "uploaded_file": {
                    "filename": file.filename,
                    "size_mb": round(file_size_mb, 2)
                },
                "output_path": str(output_path),
                "processing_features": [
                    "Pyannote Speaker Diarization",
                    "PySceneDetect Scene Changes", 
                    "MediaPipe + MTCNN Face Detection",
                    "Advanced Transition Management"
                ],
                "status_endpoint": f"/workflow/advanced-task-status/{result['task_id']}",
                "download_endpoint": f"/workflow/download-advanced-result/{result['task_id']}",
                "estimated_time": "5-15 minutes depending on video length"
            }
        else:
            # Clean up on failure
            if temp_input_path.exists():
                os.remove(temp_input_path)
            
            raise HTTPException(
                status_code=500, 
                detail=f"Advanced vertical cropping failed: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload vertical crop endpoint error: {str(e)}")
        
        # Clean up on error
        if 'temp_input_path' in locals() and temp_input_path.exists():
            os.remove(temp_input_path)
        
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@router.post("/test-hf-connection")
async def test_hf_connection(
    hf_token: str = Form(None),  # Optional - will use env var if not provided
    current_user: User = Depends(get_current_active_user)
):
    """Test HuggingFace token and pyannote access"""
    try:
        # Use environment token if none provided
        token = hf_token or os.getenv('HF_TOKEN')
        if not token:
            return {
                "success": False,
                "error": "No HuggingFace token provided and HF_TOKEN not set in environment",
                "message": "Please provide a token or set HF_TOKEN in your .env file"
            }
        
        print(f"🔍 Testing HuggingFace connection with token: {token[:10]}...")
        
        from huggingface_hub import HfApi
        
        # Test HF API connection
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"✅ HuggingFace API connection successful for user: {user_info.get('name', 'Unknown')}")
        
        # Test pyannote model access
        try:
            print("🔍 Testing pyannote model access...")
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            pipeline_status = "✅ Accessible"
            print("✅ Pyannote model access successful")
        except Exception as e:
            pipeline_status = f"❌ Error: {str(e)}"
            print(f"❌ Pyannote model access failed: {e}")
        
        return {
            "success": True,
            "hf_user": user_info.get("name", "Unknown"),
            "token_valid": True,
            "pyannote_access": pipeline_status,
            "required_models": [
                "pyannote/speaker-diarization-3.1",
                "pyannote/segmentation-3.0", 
                "pyannote/embedding"
            ],
            "recommendations": [
                "Ensure you have accepted the license agreement for pyannote models on HuggingFace",
                "Check that your token has read access to the models",
                "Verify your internet connection for model download"
            ]
        }
        
    except Exception as e:
        print(f"❌ HuggingFace connection test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "HuggingFace token validation failed. Please check your token and model access.",
            "troubleshooting": [
                "Verify your HuggingFace token is correct",
                "Ensure you have accepted the license agreements for pyannote models",
                "Check your internet connection",
                "Make sure your token has read permissions"
            ]
        } 