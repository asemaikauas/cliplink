from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
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
import time
from sqlalchemy.ext.asyncio import AsyncSession

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
# Add imports for subtitle processing
from app.services.subs import convert_groq_to_subtitles
from app.services.burn_in import burn_subtitles_to_video
from app.services.groq_client import transcribe

# Import authentication and database
from ..auth import get_current_user
from ..database import get_db
from ..models import User, Video, VideoStatus, Clip

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

class ComprehensiveWorkflowRequest(BaseModel):
    """Request for comprehensive workflow: transcript → gemini → download → vertical crop → burn subtitles with speech sync"""
    youtube_url: str
    quality: Optional[str] = "best"  # best, 8k, 4k, 1440p, 1080p, 720p
    create_vertical: Optional[bool] = True  # Create vertical (9:16) clips (default True for comprehensive workflow)
    smoothing_strength: Optional[str] = "very_high"  # low, medium, high, very_high
    burn_subtitles: Optional[bool] = True  # Whether to burn subtitles into videos (always uses speech synchronization)
    font_size: Optional[int] = 15  # Font size for subtitles (12-120)
    export_codec: Optional[str] = "h264"  # Video codec (h264, h265)
    priority: Optional[str] = "normal"  # low, normal, high
    notify_webhook: Optional[str] = None  # Optional webhook URL for completion notification

async def _run_blocking_task(func, *args, **kwargs):
    """Run blocking functions in thread pool"""
    loop = asyncio.get_event_loop()
    # Create a wrapper function that handles keyword arguments
    def wrapper():
        return func(*args, **kwargs)
    return await loop.run_in_executor(workflow_executor, wrapper)

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

# NEW: Fully parallel processing function for a single segment
async def _process_single_viral_segment_parallel(
    segment_index: int,
    segment_data: Dict[str, Any],
    source_video_path: Path,
    task_id: str,
    create_vertical: bool,
    smoothing_strength: str,
    burn_subtitles: bool,
    font_size: int,
    export_codec: str
) -> Dict[str, Any]:
    """
    Processes a single viral segment in its own parallel task.
    This includes cutting, vertical cropping, and subtitle burning.
    """
    try:
        start_time_total = time.time()
        
        # Unpack segment data
        start_time = segment_data.get("start")
        end_time = segment_data.get("end")
        title = segment_data.get("title", f"Segment_{segment_index+1}")
        
        if start_time is None or end_time is None:
            return {"success": False, "error": "Missing start or end time", "clip_path": None}
        
        # --- 1. Cut Clip ---
        from app.services.youtube import _sanitize_filename, create_clip_with_direct_ffmpeg
        safe_title = _sanitize_filename(title)
        
        # Define clip paths
        base_dir = source_video_path.parent
        clips_dir = base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        # We always cut a temporary horizontal clip first
        temp_horizontal_clip_path = clips_dir / f"temp_{safe_title}_{segment_index+1}.mp4"
        
        print(f"🚀 [Segment {segment_index+1}] Starting processing: '{title}'")
        print(f"   - Cutting segment: {start_time}s - {end_time}s")
        
        if not create_clip_with_direct_ffmpeg(source_video_path, start_time, end_time, temp_horizontal_clip_path):
            raise Exception("Failed to cut video segment using ffmpeg.")
        
        processing_clip_path = temp_horizontal_clip_path
        
        # --- 2. Vertical Cropping (if enabled) ---
        if create_vertical:
            print(f"   - Applying vertical crop with '{smoothing_strength}' smoothing...")
            vertical_clip_path = clips_dir / f"{safe_title}_vertical.mp4"
            
            from app.services.vertical_crop_async import crop_video_to_vertical_async
            
            crop_result = await crop_video_to_vertical_async(
                input_path=temp_horizontal_clip_path,
                output_path=vertical_clip_path,
                use_speaker_detection=True,
                use_smart_scene_detection=True,
                smoothing_strength=smoothing_strength
            )
            
            if not crop_result.get("success"):
                raise Exception(f"Vertical cropping failed: {crop_result.get('error')}")
            
            processing_clip_path = vertical_clip_path
            # Clean up the temp horizontal clip now that we have the vertical one
            if temp_horizontal_clip_path.exists():
                temp_horizontal_clip_path.unlink()
        
        subtitled_clip_path = None
        # --- 3. Subtitle Generation & Burning (if enabled) ---
        if burn_subtitles:
            print(f"   - Generating and burning subtitles...")
            
            # a. Extract audio from the final (cropped) clip
            from pydub import AudioSegment
            temp_audio_path = clips_dir / f"temp_audio_{safe_title}_{segment_index+1}.wav"
            audio = AudioSegment.from_file(str(processing_clip_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_audio_path, format="wav")
            
            # b. Transcribe the audio
            from app.services.groq_client import transcribe
            transcription_result = transcribe(
                file_path=str(temp_audio_path),
                apply_vad=True,
                task_id=f"{task_id}_clip_{segment_index}"
            )
            
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            
            if not transcription_result or not transcription_result.get("segments"):
                print(f"⚠️ [Segment {segment_index+1}] No transcription found. Skipping subtitle burn.")
            else:
                # c. Convert transcription to SRT
                from app.services.subs import convert_groq_to_subtitles
                subtitles_dir = clips_dir / "subtitles"
                subtitles_dir.mkdir(exist_ok=True)
                
                srt_path, _ = await _run_blocking_task(
                    convert_groq_to_subtitles,
                    groq_segments=transcription_result["segments"],
                    word_timestamps=transcription_result.get("word_timestamps", []),
                    output_dir=str(subtitles_dir),
                    filename_base=f"clip_{segment_index+1}_{safe_title}",
                    speech_sync_mode=True
                )
                
                # d. Burn subtitles
                if srt_path and Path(srt_path).exists():
                    from app.services.burn_in import burn_subtitles_to_video
                    subtitled_clip_path = clips_dir / f"subtitled_{processing_clip_path.name}"
                    
                    await _run_blocking_task(
                        burn_subtitles_to_video,
                        video_path=str(processing_clip_path),
                        srt_path=srt_path,
                        output_path=str(subtitled_clip_path),
                        font_size=font_size,
                        export_codec=export_codec
                    )
                    
                    if subtitled_clip_path.exists():
                        # We have a new subtitled clip, remove the non-subtitled one
                        processing_clip_path.unlink()
                    else:
                        subtitled_clip_path = None # Burn-in failed
                else:
                    print(f"⚠️ [Segment {segment_index+1}] SRT file generation failed. Skipping burn.")

        final_clip_path = subtitled_clip_path if subtitled_clip_path else processing_clip_path

        total_time = time.time() - start_time_total
        print(f"✅ [Segment {segment_index+1}] Finished processing in {total_time:.2f}s. Final file: {final_clip_path.name}")
        
        return {
            "success": True, 
            "clip_path": str(final_clip_path),
            "has_subtitles": "subtitled_" in final_clip_path.name,
            "processing_time": total_time
        }
        
    except Exception as e:
        import traceback
        print(f"❌ [Segment {segment_index+1}] Processing failed: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": str(e), "clip_path": None}

async def _process_video_workflow_async(
    task_id: str,
    youtube_url: str,
    quality: str,
    create_vertical: bool,
    smoothing_strength: str,
    burn_subtitles: bool = False,
    font_size: int = 15,
    export_codec: str = "h264"
):
    """
    Async implementation of the complete video processing workflow
    """
    try:
        print(f"🚀 Starting comprehensive workflow with settings:")
        print(f"   📺 URL: {youtube_url}")
        print(f"   📹 Quality: {quality}")
        print(f"   📱 Create vertical: {create_vertical}")
        print(f"   🔥 Burn subtitles: {burn_subtitles}")
        print(f"   🎨 Font size: {font_size}px")
        print(f"   🎯 Speech synchronization: ENABLED (word-level timestamps)")
        print(f"   🎛️ VAD filtering: ENABLED (with retry logic)")
        print(f"   🎬 Export codec: {export_codec}")
        
        _update_workflow_progress(task_id, "init", 5, f"Starting comprehensive workflow for: {youtube_url}")
        
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
        
        # --- REFACTORED Steps 5, 6, 7: Parallel Processing of Viral Segments ---
        _update_workflow_progress(task_id, "parallel_processing", 60, f"🚀 Starting parallel processing for {len(viral_segments)} viral segments...")
        
        parallel_start_time = time.time()
        
        tasks = []
        for i, segment in enumerate(viral_segments):
            task = _process_single_viral_segment_parallel(
                segment_index=i,
                segment_data=segment,
                source_video_path=video_path,
                task_id=task_id,
                create_vertical=create_vertical,
                smoothing_strength=smoothing_strength,
                burn_subtitles=burn_subtitles,
                font_size=font_size,
                export_codec=export_codec,
            )
            tasks.append(task)
            
        # Run all processing tasks concurrently and wait for them to complete
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        parallel_duration = time.time() - parallel_start_time
        print(f"🎬 All parallel processing finished in {parallel_duration:.2f} seconds.")

        # Collect results
        final_clip_paths = []
        original_clip_paths_for_result = [] # Keep track of original paths before subtitling for the result
        successful_clips = 0
        failed_clips = 0
        
        for result in processed_results:
            if isinstance(result, Exception) or not result.get("success"):
                failed_clips += 1
                continue
            
            final_clip_paths.append(result["clip_path"])
            successful_clips += 1
        
        # This is a bit tricky, the original paths are now intermediate.
        # For simplicity, let's just use the final paths for both in the result for now.
        original_clip_paths_for_result = final_clip_paths

        _update_workflow_progress(
            task_id, "parallel_processing", 95, 
            f"✅ Parallel processing complete: {successful_clips} clips created, {failed_clips} failed.",
            {"clip_paths": final_clip_paths}
        )
        
        # Step 8: Finalize (95-100%)
        _update_workflow_progress(task_id, "finalizing", 95, "Finalizing comprehensive workflow results...")
        
        subtitled_count = len([p for p in final_clip_paths if 'subtitled_' in p])
        result = {
            "success": True,
            "workflow_type": "comprehensive",
            "workflow_steps": {
                "video_info_extraction": True,
                "transcript_extraction": True,
                "gemini_analysis": True, 
                "video_download": True,
                "subtitle_generation": burn_subtitles and subtitled_count > 0,
                "clip_cutting": True,
                "subtitle_burning": burn_subtitles and subtitled_count > 0
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
            "subtitle_info": {
                "subtitle_style": "speech_synchronized" if burn_subtitles else None,
                "subtitle_approach": "per_clip_generation_with_word_timestamps" if burn_subtitles else None,
                "speech_synchronization": True if burn_subtitles else None,
                "vad_filtering": True if burn_subtitles else None,
                "clips_with_subtitles": subtitled_count,
                "total_clips": len(final_clip_paths),
                "subtitle_success_rate": f"{(subtitled_count/len(final_clip_paths)*100):.1f}%" if len(final_clip_paths) > 0 else "0%",
                "font_size": font_size if burn_subtitles else None,
                "export_codec": export_codec if burn_subtitles else None
            } if burn_subtitles else None,
            "files_created": {
                "source_video": str(video_path),
                "clips_created": len(final_clip_paths),
                "original_clip_paths": original_clip_paths_for_result,
                "subtitled_clips_created": subtitled_count,
                "final_clip_paths": final_clip_paths,
                "clip_type": "vertical" if create_vertical else "horizontal",
                "has_subtitles": burn_subtitles and subtitled_count > 0,
                "subtitle_files_location": str(Path(final_clip_paths[0]).parent / "subtitles") if len(final_clip_paths) > 0 and burn_subtitles else None
            }
        }
        
        # Mark as completed
        with workflow_task_lock:
            if burn_subtitles and subtitled_count > 0:
                message = f"Comprehensive workflow completed! {len(viral_segments)} segments → {successful_clips} clips → {subtitled_count} clips with subtitles"
            elif burn_subtitles:
                message = f"Workflow completed! {len(viral_segments)} segments → {successful_clips} clips (subtitle processing failed or not needed on all)"
            else:
                message = f"Workflow completed! {len(viral_segments)} segments → {successful_clips} clips"
            
            workflow_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": message,
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
                "message": f"Comprehensive workflow failed: {str(e)}",
                "completed_at": datetime.now()
            })
        raise e

@router.post("/process-comprehensive-async")
async def process_comprehensive_workflow_async(request: ComprehensiveWorkflowRequest):
    """
    🚀 COMPREHENSIVE async workflow that combines EVERYTHING:
    
    1. 📄 Extract transcript from YouTube URL
    2. 🤖 Analyze with Gemini AI to find viral segments  
    3. 📥 Download video in specified quality (supports up to 8K)
    4. ✂️ Cut video into segments based on Gemini analysis with vertical cropping
    5. 📝 Generate subtitles with TRUE SPEECH SYNCHRONIZATION (word-level timestamps)
    6. 🔥 Burn subtitles directly into the final clips with perfect timing
    
    🎯 ADVANCED SUBTITLE FEATURES:
    - Word-level timestamp synchronization for perfect speech alignment
    - VAD filtering with intelligent retry logic
    - Environment-configurable parameters
    - Multiple fallback strategies for maximum reliability
    
    This is the ultimate all-in-one endpoint that takes a YouTube URL and produces 
    ready-to-upload short clips with professional-quality burned-in subtitles!
    
    Returns immediately with task_id for status polling.
    """
    try:
        # Generate unique task ID
        task_id = f"comprehensive_{uuid.uuid4().hex[:8]}"
        
        # Validate font size
        if request.font_size and (request.font_size < 12 or request.font_size > 120):
            raise HTTPException(status_code=400, detail="Font size must be between 12 and 120")
        
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
                "burn_subtitles": request.burn_subtitles,
                "font_size": request.font_size,
                "export_codec": request.export_codec,
                "speech_synchronization": True,  # Always enabled
                "vad_filtering": True,  # Always enabled
                "priority": request.priority or "normal",
                "notify_webhook": request.notify_webhook,
                "current_step": "queued",
                "message": "Comprehensive workflow queued for processing",
                "error": None,
                "workflow_type": "comprehensive"
            }
        
        print(f"🚀 Comprehensive workflow {task_id} queued: {request.youtube_url}")
        print(f"🎯 Settings: quality={request.quality}, vertical={request.create_vertical}, subtitles={request.burn_subtitles}")
        print(f"🎬 Subtitle settings: speech_sync=True, vad_filtering=True, size={request.font_size}px, codec={request.export_codec}")
        
        # Start async processing (don't await - let it run in background)
        # Note: Using the optimized workflow function with speech synchronization
        asyncio.create_task(_process_video_workflow_async(
            task_id,
            request.youtube_url,
            request.quality or "best",
            request.create_vertical or True,
            request.smoothing_strength or "very_high",
            request.burn_subtitles or False,
            request.font_size or 15,
            request.export_codec or "h264"
        ))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "🎬 Comprehensive workflow started! This will create clips with burned-in subtitles.",
            "youtube_url": request.youtube_url,
            "workflow_type": "comprehensive",
            "settings": {
                "quality": request.quality or "best",
                "create_vertical": request.create_vertical or True,
                "smoothing_strength": request.smoothing_strength or "very_high",
                "burn_subtitles": request.burn_subtitles or True,
                "speech_synchronization": True,
                "vad_filtering": True,
                "font_size": request.font_size or 15,
                "export_codec": request.export_codec or "h264"
            },
            "workflow_steps": [
                "1. Video info extraction",
                "2. Transcript extraction", 
                "3. Gemini AI analysis",
                "4. Video download",
                "5. Vertical clip cutting",
                "6. Per-clip speech-synchronized subtitle generation",
                "7. Professional subtitle burning with word-level timing",
                "8. Final processing"
            ],
            "status_endpoint": f"/workflow/workflow-status/{task_id}",
            "estimated_time": "10-30 minutes depending on video length and quality"
        }
        
    except Exception as e:
        print(f"❌ Failed to start comprehensive workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start comprehensive workflow: {str(e)}")

@router.get("/status/{task_id}")
async def get_workflow_status(task_id: str, current_user: User = Depends(get_current_user)):
    """
    Get the current status of a workflow task
    
    Returns the task status, progress, and other metadata.
    """
    try:
        with workflow_task_lock:
            task_info = workflow_tasks.get(task_id)
        
        if not task_info:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        # Convert datetime objects to ISO strings for JSON serialization
        response_data = task_info.copy()
        
        # Handle datetime conversion
        for key in ['created_at', 'updated_at', 'completed_at']:
            if key in response_data and response_data[key]:
                if isinstance(response_data[key], datetime):
                    response_data[key] = response_data[key].isoformat()
        
        # Map status values to match frontend expectations
        status_mapping = {
            'queued': 'pending',
            'processing': 'processing',
            'completed': 'done',
            'failed': 'failed'
        }
        
        if 'status' in response_data:
            response_data['status'] = status_mapping.get(response_data['status'], response_data['status'])
        
        # Ensure we have the required fields for the frontend
        if 'stage' not in response_data:
            response_data['stage'] = response_data.get('current_step', 'unknown')
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.post("/process-video")
async def process_video_authenticated(
    request: ProcessVideoRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    Authenticated endpoint to process YouTube videos and create clips
    
    This creates a Video record in the database and processes it asynchronously.
    """
    try:
        # Extract YouTube video ID
        video_id = get_video_id(request.youtube_url)
        
        # Get video info first
        video_info = get_video_info(request.youtube_url)
        
        # Create Video record in database
        from sqlalchemy import select
        
        # Check if user already has this video
        existing_video_query = select(Video).where(
            Video.user_id == current_user.id,
            Video.youtube_id == video_id
        )
        existing_video_result = await db.execute(existing_video_query)
        existing_video = existing_video_result.scalar_one_or_none()
        
        if existing_video and existing_video.status == VideoStatus.PROCESSING:
            raise HTTPException(
                status_code=400,
                detail="You already have this video being processed. Please wait for it to complete."
            )
        
        # Create or update video record
        if existing_video:
            video_record = existing_video
            video_record.status = VideoStatus.PROCESSING
        else:
            video_record = Video(
                user_id=current_user.id,
                youtube_id=video_id,
                title=video_info.get('title', 'Unknown Title'),
                status=VideoStatus.PROCESSING
            )
            db.add(video_record)
        
        await db.commit()
        await db.refresh(video_record)
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task in workflow_tasks
        with workflow_task_lock:
            workflow_tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0,
                "stage": "initializing",
                "message": "Video processing request received",
                "video_id": str(video_record.id),
                "user_id": str(current_user.id),
                "youtube_url": request.youtube_url,
                "created_at": datetime.now().isoformat(),
                "error": None,
                "clip_paths": []
            }
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(
                _process_video_with_db_updates,
                task_id=task_id,
                video_record_id=str(video_record.id),
                youtube_url=request.youtube_url,
                quality=request.quality or "best",
                create_vertical=request.create_vertical or False,
                smoothing_strength=request.smoothing_strength or "very_high"
            )
        else:
            # Fallback: start in thread pool
            workflow_executor.submit(
                asyncio.run,
                _process_video_with_db_updates(
                    task_id=task_id,
                    video_record_id=str(video_record.id),
                    youtube_url=request.youtube_url,
                    quality=request.quality or "best",
                    create_vertical=request.create_vertical or False,
                    smoothing_strength=request.smoothing_strength or "very_high"
                )
            )
        
        return {
            "success": True,
            "task_id": task_id,
            "video_id": str(video_record.id),
            "message": "Video processing started successfully",
            "status_url": f"/workflow/status/{task_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Update video status to failed if we created one
        try:
            if 'video_record' in locals():
                video_record.status = VideoStatus.FAILED
                await db.commit()
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video processing: {str(e)}"
        )


async def _process_video_with_db_updates(
    task_id: str,
    video_record_id: str,
    youtube_url: str,
    quality: str,
    create_vertical: bool,
    smoothing_strength: str
):
    """
    Process video and update database records
    """
    from ..database import get_db_session
    
    db = get_db_session()
    
    try:
        # Update task status
        _update_workflow_progress(task_id, "starting", 5, "Starting video processing...")
        
        # Call the existing comprehensive workflow
        await _process_video_workflow_async(
            task_id=task_id,
            youtube_url=youtube_url,
            quality=quality,
            create_vertical=create_vertical,
            smoothing_strength=smoothing_strength,
            burn_subtitles=False,
            font_size=15,
            export_codec="h264"
        )
        
        # Get the final task result
        with workflow_task_lock:
            task_result = workflow_tasks.get(task_id, {})
        
        if task_result.get("status") == "done":
            # Update video status to done
            from sqlalchemy import select
            query = select(Video).where(Video.id == video_record_id)
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            
            if video_record:
                video_record.status = VideoStatus.DONE
                
                # Create clip records if we have clip paths
                clip_paths = task_result.get("clip_paths", [])
                for i, clip_path in enumerate(clip_paths):
                    if isinstance(clip_path, Path):
                        clip_path = str(clip_path)
                    
                    # You'll need to extract timing info from the analysis
                    # For now, using placeholder values
                    clip_record = Clip(
                        video_id=video_record.id,
                        s3_url=clip_path,  # In production, upload to S3 first
                        start_time=0.0,    # Extract from analysis
                        end_time=60.0,     # Extract from analysis  
                        duration=60.0      # Extract from analysis
                    )
                    db.add(clip_record)
                
                await db.commit()
        else:
            # Update video status to failed
            from sqlalchemy import select
            query = select(Video).where(Video.id == video_record_id)
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            
            if video_record:
                video_record.status = VideoStatus.FAILED
                await db.commit()
        
    except Exception as e:
        # Update video status to failed
        try:
            from sqlalchemy import select
            query = select(Video).where(Video.id == video_record_id)
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            
            if video_record:
                video_record.status = VideoStatus.FAILED
                await db.commit()
        except:
            pass
        
        # Update task status
        with workflow_task_lock:
            if task_id in workflow_tasks:
                workflow_tasks[task_id]["status"] = "failed"
                workflow_tasks[task_id]["error"] = str(e)
        
        print(f"❌ Error in video processing task {task_id}: {str(e)}")
    
    finally:
        await db.close() 