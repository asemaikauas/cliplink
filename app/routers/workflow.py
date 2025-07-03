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
# Add imports for subtitle processing
from app.services.subs import convert_groq_to_subtitles
from app.services.burn_in import burn_subtitles_to_video
from app.services.groq_client import transcribe

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
        
        # Step 5: Cut clips with vertical cropping (60-75%)
        _update_workflow_progress(task_id, "cutting", 60, "✂️ Cutting and processing video clips...")
        
        try:
            if create_vertical:
                _update_workflow_progress(task_id, "cutting", 62, f"Creating vertical clips directly with {smoothing_strength} smoothing...")
                
                # Use the more efficient vertical cutting approach that combines cutting and cropping
                from app.services.vertical_crop_async import crop_video_to_vertical_async
                
                # Process each viral segment directly to vertical clips
                clip_paths = []
                
                for i, segment in enumerate(viral_segments):
                    progress = 62 + (i / len(viral_segments)) * 13  # 62-75%
                    _update_workflow_progress(
                        task_id, "cutting", int(progress), 
                        f"Creating vertical clip {i+1}/{len(viral_segments)}: {segment.get('title', f'Segment {i+1}')}"
                    )
                    
                    start_time = segment.get("start")
                    end_time = segment.get("end")
                    title = segment.get("title", f"Segment_{i+1}")
                    
                    if start_time is None or end_time is None:
                        continue
                    
                    # Create safe filename
                    from app.services.youtube import _sanitize_filename
                    safe_title = _sanitize_filename(title)
                    
                    # Create output directory
                    clips_dir = video_path.parent / "clips" / "vertical"
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create temporary segment first
                    temp_segment_path = clips_dir / f"temp_{safe_title}_{i+1}.mp4"
                    
                    # Cut the segment using direct ffmpeg
                    from app.services.youtube import create_clip_with_direct_ffmpeg
                    if not create_clip_with_direct_ffmpeg(video_path, start_time, end_time, temp_segment_path):
                        print(f"⚠️ Failed to cut segment {i+1}: {title}")
                        continue
                    
                    # Apply vertical cropping to this segment
                    vertical_clip_path = clips_dir / f"{safe_title}_vertical.mp4"
                    
                    try:
                        crop_result = await crop_video_to_vertical_async(
                            input_path=temp_segment_path,
                            output_path=vertical_clip_path,
                            use_speaker_detection=True,
                            use_smart_scene_detection=True,
                            smoothing_strength=smoothing_strength
                        )
                        
                        if crop_result.get("success") and vertical_clip_path.exists():
                            clip_paths.append(vertical_clip_path)
                            print(f"✅ Vertical clip created: {vertical_clip_path.name}")
                        else:
                            print(f"⚠️ Failed to create vertical clip for segment {i+1}")
                    except Exception as e:
                        print(f"❌ Error creating vertical clip {i+1}: {str(e)}")
                    finally:
                        # Clean up temporary segment immediately
                        if temp_segment_path.exists():
                            temp_segment_path.unlink()
                
            else:
                _update_workflow_progress(task_id, "cutting", 62, "Creating standard horizontal clips...")
                clip_paths = await _run_blocking_task(cut_clips, video_path, gemini_analysis)
            
            _update_workflow_progress(
                task_id, "cutting", 75, 
                f"✅ Clips created: {len(clip_paths)} files",
                {"clip_paths": [str(p) for p in clip_paths]}
            )
        except Exception as e:
            raise Exception(f"Clip cutting failed: {str(e)}")
        
        # Step 6 & 7: Generate subtitles and burn them per clip (75-95%) - if enabled
        subtitled_clips = clip_paths  # Default to original clips
        
        if burn_subtitles:
            print(f"📝 Subtitle generation ENABLED - generating subtitles for {len(clip_paths)} individual clips...")
            _update_workflow_progress(task_id, "per_clip_subtitles", 75, "📝 Generating subtitles for each clip individually...")
            
            try:
                subtitled_clips = []
                total_subtitle_segments = 0
                
                for i, clip_path in enumerate(clip_paths):
                    # Progress: 75% + (i/clips * 20%) = 75-95%
                    base_progress = 75 + (i / len(clip_paths)) * 20
                    
                    _update_workflow_progress(
                        task_id, "per_clip_subtitles", int(base_progress), 
                        f"📝 Processing clip {i+1}/{len(clip_paths)}: {clip_path.name}"
                    )
                    
                    try:
                        # Step 6a: Generate subtitles for this specific clip
                        print(f"📝 Generating subtitles for clip: {clip_path.name}")
                        
                        # Extract audio from this clip (same logic as /subtitles endpoint)
                        from pydub import AudioSegment
                        temp_audio_path = f"temp_clip_audio_{task_id[:8]}_{i}.wav"
                        
                        print(f"🎵 Extracting audio from clip {i+1} for transcription...")
                        audio = AudioSegment.from_file(str(clip_path))
                        # Convert to standard format for Groq (16kHz, mono, WAV)
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(temp_audio_path, format="wav")
                        
                        # Get audio duration for logging
                        duration_s = len(audio) / 1000.0
                        print(f"✅ Audio extracted: {duration_s:.1f}s")
                        
                        # Transcribe this clip's audio with VAD enabled and retry logic
                        print(f"🎤 Starting transcription with Groq Whisper large-v3 (VAD enabled)...")
                        
                        transcription_result = await _run_blocking_task(
                            transcribe,
                            file_path=temp_audio_path,
                            apply_vad=True,  # VAD enabled for better quality
                            task_id=f"{task_id}_clip_{i}"
                        )
                        
                        # Retry logic from /subtitles endpoint for better results
                        if len(transcription_result["segments"]) == 0:
                            print(f"🔄 No segments found with VAD for clip {i+1}, retrying without VAD...")
                            transcription_result = await _run_blocking_task(
                                transcribe,
                                file_path=temp_audio_path,
                                apply_vad=False,
                                task_id=f"{task_id}_clip_{i}_retry"
                            )
                        elif len(transcription_result["segments"]) < 3:
                            print(f"🔄 Few segments with VAD for clip {i+1} ({len(transcription_result['segments'])}), trying without VAD...")
                            retry_result = await _run_blocking_task(
                                transcribe,
                                file_path=temp_audio_path,
                                apply_vad=False,
                                task_id=f"{task_id}_clip_{i}_retry_few"
                            )
                            # Use the result with more segments
                            if len(retry_result["segments"]) > len(transcription_result["segments"]):
                                print(f"✅ Better result without VAD: {len(retry_result['segments'])} vs {len(transcription_result['segments'])} segments")
                                transcription_result = retry_result
                        
                        # Clean up temp audio
                        if Path(temp_audio_path).exists():
                            Path(temp_audio_path).unlink()
                        
                        if not transcription_result or not transcription_result.get("segments"):
                            print(f"⚠️ No transcription segments for clip {i+1}, skipping subtitles")
                            subtitled_clips.append(clip_path)
                            continue
                        
                        print(f"✅ Transcription complete: {len(transcription_result['segments'])} segments, language: {transcription_result['language']}")
                        
                        # Generate subtitle files for this clip (same logic as /subtitles endpoint)
                        clip_subtitle_dir = clip_path.parent / "subtitles"
                        clip_subtitle_dir.mkdir(exist_ok=True)
                        
                        print(f"📝 Generating SRT and VTT subtitle files for clip {i+1}...")
                        
                        # Configure subtitle processing parameters (same as /subtitles endpoint)
                        import os
                        max_chars_per_line = int(os.getenv("SUBTITLE_MAX_CHARS_PER_LINE", 50))
                        max_lines = int(os.getenv("SUBTITLE_MAX_LINES", 2))
                        merge_gap_threshold = int(os.getenv("SUBTITLE_MERGE_GAP_MS", 200))
                        
                        # CapCut-style parameters from environment
                        capcut_mode = os.getenv("SUBTITLE_CAPCUT_MODE", "true").lower() == "true"
                        min_word_duration = int(os.getenv("CAPCUT_MIN_WORD_DURATION_MS", 800))
                        max_word_duration = int(os.getenv("CAPCUT_MAX_WORD_DURATION_MS", 1500))
                        word_overlap = int(os.getenv("CAPCUT_WORD_OVERLAP_MS", 150))
                        
                        # ENABLE SPEECH SYNC BY DEFAULT (key feature!)
                        speech_sync = True  # Always enable word-level timestamps for best quality
                        
                        # Determine subtitle mode with speech sync priority
                        if speech_sync:
                            mode_name = "Speech-synchronized"
                            actual_capcut_mode = False  # Speech sync takes priority
                        elif capcut_mode:
                            mode_name = "CapCut adaptive-word"
                            actual_capcut_mode = True
                        else:
                            mode_name = "Traditional"
                            actual_capcut_mode = False
                        
                        print(f"🎬 Subtitle mode for clip {i+1}: {mode_name} style")
                        
                        # Get word timestamps for speech sync
                        word_timestamps = transcription_result.get("word_timestamps", []) if speech_sync else None
                        if speech_sync and word_timestamps:
                            print(f"🎯 Using {len(word_timestamps)} word timestamps for speech sync")
                        elif speech_sync:
                            print(f"⚠️ Speech sync requested but no word timestamps available for clip {i+1}, falling back to CapCut mode")
                            actual_capcut_mode = True
                        
                        clip_srt_path, clip_vtt_path = await _run_blocking_task(
                            convert_groq_to_subtitles,
                            groq_segments=transcription_result["segments"],
                            output_dir=str(clip_subtitle_dir),
                            filename_base=f"clip_{i+1}_{clip_path.stem}",
                            max_chars_per_line=max_chars_per_line,
                            max_lines=max_lines,
                            merge_gap_threshold_ms=merge_gap_threshold,
                            capcut_mode=actual_capcut_mode,
                            speech_sync_mode=speech_sync,
                            word_timestamps=word_timestamps,
                            min_word_duration_ms=min_word_duration,
                            max_word_duration_ms=max_word_duration,
                            word_overlap_ms=word_overlap
                        )
                        
                        if not clip_srt_path or not Path(clip_srt_path).exists():
                            print(f"⚠️ Failed to generate SRT for clip {i+1}, using original clip")
                            subtitled_clips.append(clip_path)
                            continue
                        
                        print(f"✅ Subtitles generated for clip {i+1}: {len(transcription_result['segments'])} segments")
                        total_subtitle_segments += len(transcription_result["segments"])
                        
                        # Step 6b: Burn subtitles into this clip
                        print(f"🔥 Burning subtitles into clip {i+1}...")
                        
                        # Create output path for subtitled clip
                        subtitled_path = clip_path.parent / f"subtitled_{clip_path.name}"
                        
                        # Burn clip-specific subtitles
                        await _run_blocking_task(
                            burn_subtitles_to_video,
                            video_path=str(clip_path),
                            srt_path=clip_srt_path,
                            output_path=str(subtitled_path),
                            font_size=font_size,
                            export_codec=export_codec,
                            task_id=f"{task_id}_burn_{i}"
                        )
                        
                        if subtitled_path.exists():
                            subtitled_clips.append(subtitled_path)
                            print(f"✅ Subtitled clip created: {subtitled_path.name}")
                        else:
                            print(f"⚠️ Failed to create subtitled clip: {subtitled_path}")
                            # Fall back to original clip if subtitle burning fails
                            subtitled_clips.append(clip_path)
                            
                    except Exception as e:
                        print(f"⚠️ Failed to process subtitles for clip {i+1}: {str(e)}")
                        # Fall back to original clip if anything fails
                        subtitled_clips.append(clip_path)
                        continue
                
                subtitled_count = len([p for p in subtitled_clips if 'subtitled_' in str(p)])
                _update_workflow_progress(
                    task_id, "per_clip_subtitles", 95, 
                    f"✅ Per-clip subtitle processing complete: {subtitled_count}/{len(clip_paths)} clips with subtitles",
                    {
                        "subtitled_clips": [str(p) for p in subtitled_clips],
                        "total_subtitle_segments": total_subtitle_segments
                    }
                )
                
                print(f"✅ Per-clip subtitle processing completed successfully!")
                print(f"   🎬 Clips processed: {len(clip_paths)}")
                print(f"   🔥 Subtitled clips created: {subtitled_count}")
                print(f"   📝 Total subtitle segments: {total_subtitle_segments}")
                
            except Exception as e:
                # If subtitle processing fails, continue with original clips
                print(f"⚠️ Per-clip subtitle processing failed: {str(e)}")
                import traceback
                print(f"🔍 Full per-clip subtitle error traceback: {traceback.format_exc()}")
                subtitled_clips = clip_paths
                _update_workflow_progress(
                    task_id, "per_clip_subtitles", 95, 
                    f"❌ Per-clip subtitle processing failed, using original clips: {str(e)}",
                    {"subtitled_clips": [str(p) for p in clip_paths]}
                )
        else:
            _update_workflow_progress(task_id, "per_clip_subtitles", 95, "Skipping per-clip subtitle generation (disabled)")
        
        # Step 8: Finalize (95-100%)
        _update_workflow_progress(task_id, "finalizing", 95, "Finalizing comprehensive workflow results...")
        
        # Prepare final result
        subtitled_count = len([p for p in subtitled_clips if 'subtitled_' in str(p)])
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
                "subtitle_style": "speech_synchronized" if burn_subtitles else None,  # Always use speech sync
                "subtitle_approach": "per_clip_generation_with_word_timestamps" if burn_subtitles else None,
                "speech_synchronization": True if burn_subtitles else None,
                "vad_filtering": True if burn_subtitles else None,
                "clips_with_subtitles": subtitled_count,
                "total_clips": len(clip_paths),
                "subtitle_success_rate": f"{(subtitled_count/len(clip_paths)*100):.1f}%" if len(clip_paths) > 0 else "0%",
                "font_size": font_size if burn_subtitles else None,
                "export_codec": export_codec if burn_subtitles else None
            } if burn_subtitles else None,
            "files_created": {
                "source_video": str(video_path),
                "clips_created": len(clip_paths),
                "original_clip_paths": [str(p) for p in clip_paths],
                "subtitled_clips_created": subtitled_count,
                "final_clip_paths": [str(p) for p in subtitled_clips],
                "clip_type": "vertical" if create_vertical else "horizontal",
                "has_subtitles": burn_subtitles and subtitled_count > 0,
                "subtitle_files_location": str(clip_paths[0].parent / "subtitles") if len(clip_paths) > 0 and burn_subtitles else None
            }
        }
        
        # Mark as completed
        with workflow_task_lock:
            if burn_subtitles and subtitled_count > 0:
                message = f"Comprehensive workflow completed! {len(viral_segments)} segments → {len(clip_paths)} clips → {subtitled_count} clips with subtitles"
            elif burn_subtitles:
                message = f"Workflow completed! {len(viral_segments)} segments → {len(clip_paths)} clips (subtitle processing failed)"
            else:
                message = f"Workflow completed! {len(viral_segments)} segments → {len(clip_paths)} clips"
            
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