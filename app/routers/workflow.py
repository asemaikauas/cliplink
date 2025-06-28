from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
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
    async_vertical_crop_service
)

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
    """Request for async vertical crop processing with smart scene detection"""
    video_path: str
    output_path: Optional[str] = None
    use_speaker_detection: Optional[bool] = True
    use_smart_scene_detection: Optional[bool] = True
    scene_content_threshold: Optional[float] = 30.0
    scene_fade_threshold: Optional[float] = 8.0
    scene_min_length: Optional[int] = 15
    ignore_micro_cuts: Optional[bool] = True
    micro_cut_threshold: Optional[int] = 10
    smoothing_strength: Optional[str] = "very_high"
    priority: Optional[str] = "normal"  # low, normal, high

@router.post("/create-vertical-crop-async")
async def create_vertical_crop_async(request: AsyncVerticalCropRequest):
    """
    Create vertical crop asynchronously with smart scene detection and progress tracking
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
            smart_suffix = "_smart" if request.use_smart_scene_detection else "_legacy"
            output_path = output_dir / f"{input_path.stem}_vertical{smart_suffix}.mp4"
        
        print(f"🚀 Starting smart async vertical crop: {input_path.name}")
        print(f"📱 Output: {output_path}")
        print(f"🎛️ Smoothing: {request.smoothing_strength}")
        print(f"🔊 Speaker detection: {request.use_speaker_detection}")
        print(f"🎬 Smart scene detection: {request.use_smart_scene_detection}")
        if request.use_smart_scene_detection:
            print(f"   📊 Content threshold: {request.scene_content_threshold}")
            print(f"   🌅 Fade threshold: {request.scene_fade_threshold}")
            print(f"   ⏱️ Min scene length: {request.scene_min_length}")
            print(f"   🔍 Ignore micro-cuts: {request.ignore_micro_cuts} (< {request.micro_cut_threshold} frames)")
        
        # Start smart async processing
        result = await crop_video_to_vertical_async(
            input_path=input_path,
            output_path=output_path,
            use_speaker_detection=request.use_speaker_detection,
            use_smart_scene_detection=request.use_smart_scene_detection,
            scene_content_threshold=request.scene_content_threshold,
            scene_fade_threshold=request.scene_fade_threshold,
            scene_min_length=request.scene_min_length,
            ignore_micro_cuts=request.ignore_micro_cuts,
            micro_cut_threshold=request.micro_cut_threshold,
            smoothing_strength=request.smoothing_strength
        )
        task_id = result["task_id"]
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Smart vertical crop processing started with PySceneDetect",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "smart_features": {
                "scene_detection_enabled": request.use_smart_scene_detection,
                "speaker_detection_enabled": request.use_speaker_detection,
                "scene_content_threshold": request.scene_content_threshold,
                "scene_fade_threshold": request.scene_fade_threshold,
                "micro_cut_filtering": request.ignore_micro_cuts,
                "smoothing_strength": request.smoothing_strength
            },
            "status_endpoint": f"/workflow/task-status/{task_id}",
            "estimated_time": "2-10 minutes depending on video length and scene complexity"
        }
        
    except Exception as e:
        print(f"❌ Smart async vertical crop failed: {str(e)}")
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
    use_smart_scene_detection: bool = True,
    enable_group_conversation_framing: bool = False,
    scene_content_threshold: float = 30.0,
    scene_fade_threshold: float = 8.0,
    scene_min_length: int = 15,
    ignore_micro_cuts: bool = True,
    micro_cut_threshold: int = 10,
    smoothing_strength: str = "very_high",
    async_processing: bool = True
):
    """
    Test endpoint for uploading a video and creating a smart vertical crop with scene detection.
    
    This demonstrates the NEW intelligent scene-aware cropping system where PySceneDetect
    acts as the "brain" telling your smoothing when to reset for perfect responsiveness
    on every real cut while maintaining smooth tracking in stable scenes.
    
    Args:
        file: Video file to upload and process
        use_speaker_detection: Whether to use speaker detection for smart cropping
        use_smart_scene_detection: Whether to use smart scene detection for crop resets
        enable_group_conversation_framing: Enable split-screen layout for 2-speaker conversations (top/bottom)
        scene_content_threshold: Sensitivity for hard cuts (higher = less sensitive, 30.0 = default)
        scene_fade_threshold: Sensitivity for gradual transitions/fades (8.0 = default)
        scene_min_length: Minimum scene length in frames to avoid micro-cuts (15 = default)
        ignore_micro_cuts: Whether to ignore very short scenes (True = recommended)
        micro_cut_threshold: Threshold for micro-cut detection in frames (10 = default)
        smoothing_strength: Motion smoothing level ("very_high" = most stable)
        async_processing: Whether to use async processing (True = recommended for better performance)
    """
    try:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix, dir=TEMP_UPLOADS_DIR) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = Path(tmp.name)
        
        print(f"📄 Uploaded file saved to temporary path: {temp_path}")
        print(f"📁 File size: {temp_path.stat().st_size / (1024*1024):.2f} MB")

        # Define the output path for the smart cropped video
        output_dir = Path("temp_vertical")
        output_dir.mkdir(exist_ok=True)
        
        # Include scene detection info in filename for clarity
        scene_suffix = "_smart" if use_smart_scene_detection else "_legacy"
        output_path = output_dir / f"{temp_path.stem}_vertical{scene_suffix}_{smoothing_strength}.mp4"
        
        print(f"🎬 Starting SMART vertical crop process...")
        print(f"   🔊 Speaker detection: {use_speaker_detection}")
        print(f"   🎬 Smart scene detection: {use_smart_scene_detection}")
        if use_smart_scene_detection:
            print(f"   📊 Scene content threshold: {scene_content_threshold}")
            print(f"   🌅 Scene fade threshold: {scene_fade_threshold}")
            print(f"   ⏱️ Min scene length: {scene_min_length} frames")
            print(f"   🔍 Ignore micro-cuts: {ignore_micro_cuts} (< {micro_cut_threshold} frames)")
        print(f"   🎛️ Smoothing: {smoothing_strength}")
        print(f"   ⚡ Async processing: {async_processing}")

        if async_processing:
            # Use the NEW smart async processing system
            result = await crop_video_to_vertical_async(
                input_path=temp_path,
                output_path=output_path,
                use_speaker_detection=use_speaker_detection,
                use_smart_scene_detection=use_smart_scene_detection,
                enable_group_conversation_framing=enable_group_conversation_framing,
                scene_content_threshold=scene_content_threshold,
                scene_fade_threshold=scene_fade_threshold,
                scene_min_length=scene_min_length,
                ignore_micro_cuts=ignore_micro_cuts,
                micro_cut_threshold=micro_cut_threshold,
                smoothing_strength=smoothing_strength
            )
            
            task_id = result["task_id"]
            
            print(f"🚀 Smart vertical crop started with task ID: {task_id}")
            
            # Clean up temp file (the async process has its own copy)
            if temp_path.exists():
                os.remove(temp_path)
            
            return {
                "success": True,
                "processing_type": "smart_async",
                "task_id": task_id,
                "message": "Smart vertical crop processing started with PySceneDetect",
                "uploaded_file": file.filename,
                "output_path": str(output_path),
                "smart_features": {
                    "scene_detection_enabled": use_smart_scene_detection,
                    "speaker_detection_enabled": use_speaker_detection,
                    "group_conversation_framing_enabled": enable_group_conversation_framing,
                    "scene_content_threshold": scene_content_threshold,
                    "scene_fade_threshold": scene_fade_threshold,
                    "micro_cut_filtering": ignore_micro_cuts,
                    "smoothing_strength": smoothing_strength
                },
                "status_endpoint": f"/workflow/task-status/{task_id}",
                "download_endpoint": f"/workflow/download-result/{task_id}",
                "estimated_processing_time": "2-10 minutes depending on video length and scene complexity",
                "scene_detection_info": "Video will be pre-analyzed for scene boundaries before processing",
                "expected_results": {
                    "scenes_detected": "Will be available in status after scene analysis",
                    "smart_resets": "Number of intelligent crop resets at scene cuts",
                    "cut_boundaries": "Frame numbers where scene changes occur"
                }
            }
        
        else:
            # Fallback to synchronous processing (legacy mode)
            print("⚠️ Using legacy synchronous processing - consider using async_processing=True")
            
            # Import the old synchronous function for backward compatibility
            from app.services.vertical_crop import crop_video_to_vertical as crop_sync
            
            success = crop_sync(
                input_path=temp_path,
                output_path=output_path,
                use_speaker_detection=use_speaker_detection,
                smoothing_strength=smoothing_strength
            )

            if not success:
                # Clean up the temp file even on failure
                os.remove(temp_path)
                raise HTTPException(status_code=500, detail="Vertical cropping failed during processing")

            print(f"✅ Legacy vertical crop successful! Output at: {output_path}")
            
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)
            
            # Return a downloadable link to the file (legacy mode)
            return FileResponse(
                path=output_path, 
                media_type='video/mp4', 
                filename=output_path.name
            )

    except HTTPException:
        # Re-raise HTTP exceptions to return proper status codes
        raise
    except Exception as e:
        print(f"❌ An error occurred during smart vertical crop test: {e}")
        # Clean up temp file on any other exception
        if 'temp_path' in locals() and temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/test-upload-vertical-legacy")
async def test_upload_vertical_legacy(
    file: UploadFile = File(...),
    use_speaker_detection: bool = True,
    smoothing_strength: str = "very_high"
):
    """
    LEGACY test endpoint for uploading a video and creating a vertical crop (old system).
    Use /test-upload-vertical with async_processing=True for the new smart scene detection system.
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
        output_path = output_dir / f"{temp_path.stem}_vertical_legacy_{smoothing_strength}.mp4"
        
        print(f"🚀 Starting LEGACY vertical crop process...")
        print(f"   - Speaker detection: {use_speaker_detection}")
        print(f"   - Smoothing: {smoothing_strength}")

        # Call the OLD vertical cropping service
        from app.services.vertical_crop import crop_video_to_vertical as crop_sync
        
        success = crop_sync(
            input_path=temp_path,
            output_path=output_path,
            use_speaker_detection=use_speaker_detection,
            smoothing_strength=smoothing_strength
        )

        if not success:
            # Clean up the temp file even on failure
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Vertical cropping failed during processing")

        print(f"✅ Legacy vertical crop successful! Output at: {output_path}")
        
        # Clean up temp file
        if temp_path.exists():
            os.remove(temp_path)
        
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
        print(f"❌ An error occurred during legacy vertical crop test: {e}")
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