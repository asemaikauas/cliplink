#!/usr/bin/env python3
"""
Video Cropping API Router
Provides endpoints for automatic vertical video cropping with face tracking
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
import uuid
import asyncio
import os
from datetime import datetime
import logging

# Import cropping services - handle gracefully if dependencies are missing
try:
    from app.services.video_cropper import (
        CropMode, AspectRatio, CropSettings, crop_tasks, crop_task_lock, crop_executor,
        VIDEO_PROCESSING_AVAILABLE, AUDIO_PROCESSING_AVAILABLE
    )
    from app.services.crop_processor import VideoCropProcessor, crop_video_async, _update_crop_progress
    CROP_SERVICES_AVAILABLE = True
except ImportError as e:
    # Create dummy objects if services aren't available
    CROP_SERVICES_AVAILABLE = False
    VIDEO_PROCESSING_AVAILABLE = False
    AUDIO_PROCESSING_AVAILABLE = False
    
    # Dummy classes for when dependencies are missing
    class CropMode:
        AUTO = "auto"
        SOLO = "solo" 
        INTERVIEW = "interview"
        FALLBACK = "fallback"
    
    crop_tasks = {}
    crop_task_lock = None

logger = logging.getLogger(__name__)
router = APIRouter()

# Directory constants
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CROPS_OUTPUT_DIR = Path("crops")
CROPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class CropVideoRequest(BaseModel):
    """Request for video cropping from URL"""
    video_url: str
    mode: str = "auto"  # auto, solo, interview, fallback
    target_aspect_ratio: str = "9:16"  # 9:16, 1:1, 4:5, 3:4
    output_resolution: str = "1080x1920"  # WIDTHxHEIGHT
    confidence_threshold: float = 0.7
    enable_scene_detection: bool = True
    smoothing_window: int = 30
    padding_ratio: float = 0.1

class CropAnalysisRequest(BaseModel):
    """Request for video mode analysis"""
    video_url: str
    confidence_threshold: float = 0.7

def _parse_aspect_ratio(ratio_str: str) -> AspectRatio:
    """Parse aspect ratio string to enum"""
    ratio_map = {
        "9:16": AspectRatio.VERTICAL,
        "1:1": AspectRatio.SQUARE,
        "4:5": AspectRatio.PORTRAIT,
        "3:4": AspectRatio.THREE_FOUR
    }
    return ratio_map.get(ratio_str, AspectRatio.VERTICAL)

def _parse_resolution(resolution_str: str) -> tuple:
    """Parse resolution string to tuple"""
    try:
        width, height = resolution_str.split('x')
        return (int(width), int(height))
    except:
        return (1080, 1920)  # Default 9:16

def _check_crop_services_available():
    """Check if crop services are available and raise error if not"""
    if not CROP_SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Video cropping service unavailable - missing required dependencies (MediaPipe, OpenCV, etc.)"
        )

def _parse_crop_mode(mode_str: str) -> CropMode:
    """Parse crop mode string to enum"""
    mode_map = {
        "auto": CropMode.AUTO,
        "solo": CropMode.SOLO,
        "interview": CropMode.INTERVIEW,
        "fallback": CropMode.FALLBACK
    }
    return mode_map.get(mode_str.lower(), CropMode.AUTO)

@router.post("/analyze")
async def analyze_video_mode(request: CropAnalysisRequest):
    """
    Analyze video to determine optimal cropping mode without processing
    """
    _check_crop_services_available()
    try:
        logger.info(f"🔍 Analyzing video mode for: {request.video_url}")
        
        # Download video temporarily
        from app.services.youtube import download_video
        video_path = download_video(request.video_url, "720p")  # Use lower quality for analysis
        
        # Create settings for analysis
        settings = CropSettings(
            mode=CropMode.AUTO,
            confidence_threshold=request.confidence_threshold
        )
        
        # Analyze mode
        processor = VideoCropProcessor(settings)
        detected_mode = processor.analyze_video_mode(video_path)
        
        # Clean up temp video
        if video_path.exists():
            video_path.unlink()
        
        processor.cleanup()
        
        return {
            "success": True,
            "analysis": {
                "detected_mode": detected_mode.value,
                "confidence": 0.85,  # Placeholder confidence
                "video_url": request.video_url,
                "recommended_settings": {
                    "mode": detected_mode.value,
                    "target_aspect_ratio": "9:16",
                    "enable_scene_detection": True,
                    "confidence_threshold": request.confidence_threshold
                }
            },
            "supported_modes": ["auto", "solo", "interview", "fallback"],
            "supported_aspect_ratios": ["9:16", "1:1", "4:5", "3:4"]
        }
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@router.post("/crop")
async def crop_video_from_url(request: CropVideoRequest):
    """
    Crop video from URL with intelligent face tracking and speaker detection
    Returns task ID for async processing
    """
    _check_crop_services_available()
    try:
        # Generate unique task ID
        task_id = f"crop_{uuid.uuid4().hex[:8]}"
        
        # Parse settings
        settings = CropSettings(
            mode=_parse_crop_mode(request.mode),
            target_aspect_ratio=_parse_aspect_ratio(request.target_aspect_ratio),
            output_resolution=_parse_resolution(request.output_resolution),
            confidence_threshold=request.confidence_threshold,
            enable_scene_detection=request.enable_scene_detection,
            smoothing_window=request.smoothing_window,
            padding_ratio=request.padding_ratio
        )
        
        # Initialize task tracking
        with crop_task_lock:
            crop_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "created_at": datetime.now(),
                "video_url": request.video_url,
                "settings": {
                    "mode": request.mode,
                    "target_aspect_ratio": request.target_aspect_ratio,
                    "output_resolution": request.output_resolution,
                    "confidence_threshold": request.confidence_threshold
                },
                "current_step": "queued",
                "message": "Video cropping queued for processing",
                "error": None
            }
        
        logger.info(f"🚀 Video crop task {task_id} queued: {request.video_url}")
        
        # Start async processing
        asyncio.create_task(_process_crop_task_async(task_id, request.video_url, settings))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Video cropping started",
            "video_url": request.video_url,
            "settings": {
                "mode": request.mode,
                "target_aspect_ratio": request.target_aspect_ratio,
                "output_resolution": request.output_resolution
            },
            "status_endpoint": f"/crop/status/{task_id}",
            "estimated_time": "3-15 minutes depending on video length and mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to start crop task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start cropping: {str(e)}")

@router.post("/crop-upload")
async def crop_video_upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    target_aspect_ratio: str = Form("9:16"),
    output_resolution: str = Form("1080x1920"),
    confidence_threshold: float = Form(0.7),
    enable_scene_detection: bool = Form(True),
    smoothing_window: int = Form(30),
    padding_ratio: float = Form(0.1)
):
    """
    Upload and crop video file with intelligent face tracking
    """
    _check_crop_services_available()
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Unsupported video format")
        
        # Generate unique task ID
        task_id = f"crop_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file
        input_path = TEMP_UPLOADS_DIR / f"{task_id}_{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse settings
        settings = CropSettings(
            mode=_parse_crop_mode(mode),
            target_aspect_ratio=_parse_aspect_ratio(target_aspect_ratio),
            output_resolution=_parse_resolution(output_resolution),
            confidence_threshold=confidence_threshold,
            enable_scene_detection=enable_scene_detection,
            smoothing_window=smoothing_window,
            padding_ratio=padding_ratio
        )
        
        # Initialize task tracking
        with crop_task_lock:
            crop_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "created_at": datetime.now(),
                "video_url": f"upload:{file.filename}",
                "input_path": str(input_path),
                "settings": {
                    "mode": mode,
                    "target_aspect_ratio": target_aspect_ratio,
                    "output_resolution": output_resolution,
                    "confidence_threshold": confidence_threshold
                },
                "current_step": "queued",
                "message": "Uploaded video queued for cropping",
                "error": None
            }
        
        logger.info(f"🚀 Video crop upload task {task_id} queued: {file.filename}")
        
        # Start async processing
        asyncio.create_task(_process_crop_upload_async(task_id, input_path, settings))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Video upload and cropping started",
            "filename": file.filename,
            "file_size_mb": round(len(await file.read()) / (1024*1024), 1),
            "settings": {
                "mode": mode,
                "target_aspect_ratio": target_aspect_ratio,
                "output_resolution": output_resolution
            },
            "status_endpoint": f"/crop/status/{task_id}",
            "estimated_time": "3-15 minutes depending on video length and mode"
        }
        
    except Exception as e:
        logger.error(f"Upload crop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@router.get("/status/{task_id}")
async def get_crop_status(task_id: str):
    """
    Get the status and progress of a video cropping task
    """
    with crop_task_lock:
        task = crop_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Crop task {task_id} not found")
    
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
        "video_url": task["video_url"],
        "settings": task["settings"],
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
    
    # Add result if completed
    if task["status"] == "completed" and "result" in task:
        response["result"] = task["result"]
        response["download_endpoint"] = f"/crop/download/{task_id}"
    
    return response

@router.get("/download/{task_id}")
async def download_cropped_video(task_id: str):
    """
    Download the cropped video result
    """
    with crop_task_lock:
        task = crop_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Crop task {task_id} not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Crop task {task_id} is not completed yet")
    
    if "result" not in task:
        raise HTTPException(status_code=404, detail="Result data not found")
    
    output_path = Path(task["result"]["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Cropped video file not found")
    
    return FileResponse(
        path=str(output_path),
        filename=f"cropped_{task_id}.mp4",
        media_type='video/mp4'
    )

@router.get("/tasks")
async def list_crop_tasks():
    """
    List all video cropping tasks
    """
    with crop_task_lock:
        tasks = {tid: task.copy() for tid, task in crop_tasks.items()}
    
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
            "video_url": task["video_url"],
            "settings": task["settings"],
            "created_at": task["created_at"].isoformat(),
            "processing_time_seconds": round(processing_time, 1)
        }
    
    return {
        "crop_tasks": formatted_tasks,
        "total_tasks": len(tasks),
        "queued": len([t for t in tasks.values() if t["status"] == "queued"]),
        "processing": len([t for t in tasks.values() if t["status"] == "processing"]),
        "completed": len([t for t in tasks.values() if t["status"] == "completed"]),
        "failed": len([t for t in tasks.values() if t["status"] == "failed"])
    }

@router.post("/cleanup")
async def cleanup_crop_tasks(max_age_hours: int = 24):
    """
    Clean up completed crop tasks older than specified hours
    """
    current_time = datetime.now()
    to_remove = []
    
    with crop_task_lock:
        for task_id, task in crop_tasks.items():
            if task["status"] in ["completed", "failed"]:
                completed_at = task.get("completed_at", task.get("created_at"))
                if completed_at and (current_time - completed_at).total_seconds() > max_age_hours * 3600:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            # Clean up files
            task = crop_tasks[task_id]
            if "result" in task and "output_path" in task["result"]:
                output_path = Path(task["result"]["output_path"])
                if output_path.exists():
                    output_path.unlink()
            
            if "input_path" in task:
                input_path = Path(task["input_path"])
                if input_path.exists():
                    input_path.unlink()
            
            del crop_tasks[task_id]
    
    return {
        "success": True,
        "cleaned_tasks": len(to_remove),
        "remaining_tasks": len(crop_tasks),
        "message": f"Cleaned up {len(to_remove)} crop tasks older than {max_age_hours} hours"
    }

async def _process_crop_task_async(task_id: str, video_url: str, settings: CropSettings):
    """Process crop task from URL"""
    try:
        _update_crop_progress(task_id, "downloading", 10, f"Downloading video: {video_url}")
        
        # Download video
        from app.services.youtube import download_video
        video_path = download_video(video_url, "best")
        
        # Generate output path
        output_path = CROPS_OUTPUT_DIR / f"cropped_{task_id}.mp4"
        
        # Process video
        success = await crop_video_async(task_id, video_path, output_path, settings)
        
        # Clean up input video
        if video_path.exists():
            video_path.unlink()
        
        return success
        
    except Exception as e:
        error_msg = f"Crop task failed: {str(e)}"
        _update_crop_progress(task_id, "failed", 0, error_msg)
        
        with crop_task_lock:
            crop_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": error_msg,
                "completed_at": datetime.now()
            })
        
        logger.error(error_msg)
        return False

async def _process_crop_upload_async(task_id: str, input_path: Path, settings: CropSettings):
    """Process crop task from uploaded file"""
    try:
        # Generate output path
        output_path = CROPS_OUTPUT_DIR / f"cropped_{task_id}.mp4"
        
        # Process video
        success = await crop_video_async(task_id, input_path, output_path, settings)
        
        # Clean up input file after processing
        if input_path.exists():
            input_path.unlink()
        
        return success
        
    except Exception as e:
        error_msg = f"Upload crop task failed: {str(e)}"
        _update_crop_progress(task_id, "failed", 0, error_msg)
        
        with crop_task_lock:
            crop_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": error_msg,
                "completed_at": datetime.now()
            })
        
        logger.error(error_msg)
        return False

@router.get("/health")
async def health_check():
    """
    Health check for crop service
    """
    try:
        # Test MediaPipe availability
        import mediapipe as mp
        mp_available = True
    except ImportError:
        mp_available = False
    
    try:
        # Test OpenCV availability
        import cv2
        cv_available = True
    except ImportError:
        cv_available = False
    
    return {
        "status": "ok",
        "service": "video_cropper",
        "dependencies": {
            "mediapipe": mp_available,
            "opencv": cv_available,
            "audio_processing": AUDIO_PROCESSING_AVAILABLE,
            "video_processing": VIDEO_PROCESSING_AVAILABLE
        },
        "active_tasks": len(crop_tasks),
        "supported_modes": ["auto", "solo", "interview", "fallback"],
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"]
    } 