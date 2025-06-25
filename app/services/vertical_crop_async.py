"""
Asynchronous vertical cropping service for creating YouTube Shorts (9:16 aspect ratio)
Designed to handle multiple concurrent requests without blocking
"""

import asyncio
import cv2
import numpy as np
import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncVerticalCropService:
    """
    Asynchronous service for creating vertical (9:16) crops of videos with intelligent speaker tracking
    Supports concurrent processing of multiple requests
    """
    
    def __init__(self, max_workers: int = 4, max_concurrent_tasks: int = 10):
        # Thread pool for CPU-intensive tasks
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Process pool for very heavy operations
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, max_workers))
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize VAD for voice activity detection
        try:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 0-3
            logger.info("✅ Voice Activity Detection initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize VAD: {e}")
            self.vad = None
        
        # Try to load OpenCV DNN model for face detection
        self.face_net = self._load_face_detection_model()
        
        logger.info(f"🚀 AsyncVerticalCropService initialized with {max_workers} workers, max {max_concurrent_tasks} concurrent tasks")
    
    def _load_face_detection_model(self):
        """Load OpenCV DNN model for face detection (thread-safe)"""
        try:
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            
            if Path(prototxt_path).exists() and Path(model_path).exists():
                net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logger.info("✅ Face detection model loaded")
                return net
            else:
                logger.warning("⚠️ Face detection models not found. Using center-crop fallback.")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Could not load face detection model: {e}")
            return None
    
    async def _run_cpu_bound_task(self, func, *args, **kwargs):
        """Run CPU-bound task in thread executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
    
    async def _run_heavy_task(self, func, *args, **kwargs):
        """Run very heavy task in process executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_executor, func, *args, **kwargs)
    
    def _create_task_id(self) -> str:
        """Generate unique task ID"""
        return f"crop_{uuid.uuid4().hex[:8]}"
    
    def _update_task_status(self, task_id: str, status: str, progress: int = 0, message: str = "", data: Optional[Dict] = None):
        """Thread-safe task status update"""
        with self.task_lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].update({
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "updated_at": datetime.now().isoformat()
                })
                if data:
                    self.active_tasks[task_id].update(data)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID"""
        with self.task_lock:
            return self.active_tasks.get(task_id, None)
    
    async def list_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all active tasks"""
        with self.task_lock:
            return self.active_tasks.copy()
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self.task_lock:
            tasks_to_remove = []
            for task_id, task_info in self.active_tasks.items():
                if task_info.get("status") in ["completed", "failed"]:
                    created_time = task_info.get("created_at", datetime.now()).timestamp()
                    if created_time < cutoff_time:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.active_tasks[task_id]
            
            logger.info(f"🧹 Cleaned up {len(tasks_to_remove)} old tasks")
    
    def _detect_faces_sync(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Synchronous face detection for thread executor"""
        if self.face_net is None:
            return []
        
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    
                    # Check for invalid values before casting
                    if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                        continue
                    
                    x, y, x1, y1 = box.astype("int")
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, min(w, x))
                    y = max(0, min(h, y))
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    
                    # Ensure valid bounding box (x1 > x and y1 > y)
                    if x1 > x and y1 > y:
                        faces.append((x, y, x1, y1))
            
            return faces
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    async def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Async face detection"""
        return await self._run_cpu_bound_task(self._detect_faces_sync, frame)
    
    def _detect_voice_activity_sync(self, audio_frame: bytes) -> bool:
        """Synchronous voice activity detection"""
        if self.vad is None:
            return True
        
        try:
            return self.vad.is_speech(audio_frame, 16000)
        except Exception as e:
            logger.error(f"Voice activity detection error: {e}")
            return True
    
    async def detect_voice_activity(self, audio_frame: bytes) -> bool:
        """Async voice activity detection"""
        return await self._run_cpu_bound_task(self._detect_voice_activity_sync, audio_frame)
    
    async def find_active_speaker(
        self, 
        frame: np.ndarray, 
        audio_frame: Optional[bytes] = None,
        previous_crop_center: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """Async active speaker detection"""
        faces = await self.detect_faces(frame)
        
        if not faces:
            return None
        
        if len(faces) == 1:
            return faces[0]
        
        # Multiple faces: use heuristics
        h, w = frame.shape[:2]
        best_face = None
        best_score = 0
        
        # Check voice activity
        has_voice_activity = True
        if audio_frame:
            has_voice_activity = await self.detect_voice_activity(audio_frame)
        
        for face in faces:
            x, y, x1, y1 = face
            face_width = x1 - x
            face_height = y1 - y
            
            # Score calculations
            size_score = (face_width * face_height) / (w * h)
            
            face_center_x = (x + x1) / 2
            face_center_y = (y + y1) / 2
            center_score = 1.0 - (abs(face_center_x - w/2) / (w/2))
            
            # Stability score
            stability_score = 0
            if previous_crop_center:
                prev_x, prev_y = previous_crop_center
                distance = np.sqrt((face_center_x - prev_x)**2 + (face_center_y - prev_y)**2)
                max_distance = np.sqrt(w**2 + h**2) / 3
                stability_score = max(0, 1.0 - distance / max_distance)
            
            total_score = (
                size_score * 0.35 + 
                center_score * 0.25 + 
                stability_score * 0.4
            )
            
            if has_voice_activity:
                total_score *= 1.15
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def _smooth_crop_center(
        self, 
        new_center: Tuple[int, int], 
        previous_crop_center: Optional[Tuple[int, int]],
        recent_centers: List[Tuple[int, int]],
        smoothing_config: Dict[str, Any]
    ) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """Smooth crop center calculation"""
        smoothing_factor = smoothing_config["smoothing_factor"]
        max_jump_distance = smoothing_config["max_jump_distance"]
        stability_frames = smoothing_config["stability_frames"]
        
        if previous_crop_center is None:
            return new_center, [new_center]
        
        # Add new center to history
        recent_centers = recent_centers.copy()
        recent_centers.append(new_center)
        if len(recent_centers) > stability_frames:
            recent_centers.pop(0)
        
        # Calculate average
        avg_x = sum(center[0] for center in recent_centers) / len(recent_centers)
        avg_y = sum(center[1] for center in recent_centers) / len(recent_centers)
        averaged_center = (int(avg_x), int(avg_y))
        
        prev_x, prev_y = previous_crop_center
        new_x, new_y = averaged_center
        
        # Limit jump distance
        distance = np.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2)
        
        if distance > max_jump_distance:
            direction_x = (new_x - prev_x) / distance if distance > 0 else 0
            direction_y = (new_y - prev_y) / distance if distance > 0 else 0
            
            new_x = prev_x + direction_x * max_jump_distance
            new_y = prev_y + direction_y * max_jump_distance
        
        # Apply exponential smoothing
        smoothed_x = int(prev_x * smoothing_factor + new_x * (1 - smoothing_factor))
        smoothed_y = int(prev_y * smoothing_factor + new_y * (1 - smoothing_factor))
        
        return (smoothed_x, smoothed_y), recent_centers
    
    def _crop_frame_to_vertical(
        self, 
        frame: np.ndarray, 
        speaker_box: Optional[Tuple[int, int, int, int]],
        target_size: Tuple[int, int],
        crop_center: Optional[Tuple[int, int]] = None,
        padding_factor: float = 1.5
    ) -> np.ndarray:
        """Synchronous frame cropping for thread executor"""
        h, w = frame.shape[:2]
        target_width, target_height = target_size
        target_aspect = target_width / target_height
        
        # Determine crop center
        if crop_center:
            crop_center_x, crop_center_y = crop_center
        elif speaker_box:
            x, y, x1, y1 = speaker_box
            face_center_x = (x + x1) // 2
            face_center_y = (y + y1) // 2
            face_height = y1 - y
            padding_y = int(face_height * padding_factor) // 2
            crop_center_x = face_center_x
            crop_center_y = max(face_center_y - padding_y, face_center_y)
        else:
            crop_center_x = w // 2
            crop_center_y = h // 2
        
        # Calculate crop dimensions
        if w / h > target_aspect:
            crop_height = h
            crop_width = int(h * target_aspect)
        else:
            crop_width = w
            crop_height = int(w / target_aspect)
        
        # Calculate crop boundaries
        left = max(0, crop_center_x - crop_width // 2)
        right = min(w, left + crop_width)
        top = max(0, crop_center_y - crop_height // 2)
        bottom = min(h, top + crop_height)
        
        # Adjust if needed
        if right - left < crop_width:
            if left == 0:
                right = min(w, crop_width)
            else:
                left = max(0, w - crop_width)
        
        if bottom - top < crop_height:
            if top == 0:
                bottom = min(h, crop_height)
            else:
                top = max(0, h - crop_height)
        
        # Perform crop
        cropped = frame[top:bottom, left:right]
        
        # Resize to target
        if cropped.shape[:2] != (target_height, target_width):
            cropped = cv2.resize(cropped, target_size)
        
        return cropped
    
    async def crop_frame_to_vertical(
        self, 
        frame: np.ndarray, 
        speaker_box: Optional[Tuple[int, int, int, int]],
        target_size: Tuple[int, int],
        crop_center: Optional[Tuple[int, int]] = None,
        padding_factor: float = 1.5
    ) -> np.ndarray:
        """Async frame cropping"""
        return await self._run_cpu_bound_task(
            self._crop_frame_to_vertical,
            frame, speaker_box, target_size, crop_center, padding_factor
        )
    
    def _extract_audio_sync(self, video_path: Path) -> Optional[bytes]:
        """Synchronous audio extraction"""
        try:
            temp_audio_path = f"temp_audio_vad_{uuid.uuid4().hex[:8]}.wav"
            audio = AudioSegment.from_file(str(video_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_audio_path, format="wav")
            
            with wave.open(temp_audio_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
            
            os.remove(temp_audio_path)
            return audio_data
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    async def extract_audio_for_vad(self, video_path: Path) -> Optional[bytes]:
        """Async audio extraction"""
        return await self._run_cpu_bound_task(self._extract_audio_sync, video_path)
    
    def _process_audio_frames(self, audio_data: bytes, sample_rate: int = 16000, frame_duration_ms: int = 30):
        """Generator for audio frame processing"""
        if not audio_data:
            return
        
        n = int(sample_rate * frame_duration_ms / 1000) * 2
        offset = 0
        while offset + n <= len(audio_data):
            frame = audio_data[offset:offset + n]
            offset += n
            yield frame
    
    async def create_vertical_crop_async(
        self, 
        input_video_path: Path, 
        output_video_path: Path,
        use_speaker_detection: bool = True,
        smoothing_strength: str = "very_high",
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create vertical crop asynchronously with progress tracking
        """
        if not task_id:
            task_id = self._create_task_id()
        
        # Check concurrent task limit
        with self.task_lock:
            active_count = len([t for t in self.active_tasks.values() if t["status"] == "processing"])
            if active_count >= self.max_concurrent_tasks:
                return {
                    "success": False,
                    "error": f"Maximum concurrent tasks ({self.max_concurrent_tasks}) reached",
                    "task_id": task_id
                }
        
        # Initialize task tracking
        with self.task_lock:
            self.active_tasks[task_id] = {
                "task_id": task_id,
                "status": "initializing",
                "progress": 0,
                "message": "Initializing video processing...",
                "created_at": datetime.now(),
                "input_path": str(input_video_path),
                "output_path": str(output_video_path),
                "use_speaker_detection": use_speaker_detection,
                "smoothing_strength": smoothing_strength
            }
        
        try:
            # Get video properties
            self._update_task_status(task_id, "processing", 5, "Reading video properties...")
            
            cap = cv2.VideoCapture(str(input_video_path))
            if not cap.isOpened():
                raise Exception(f"Could not open video: {input_video_path}")
            
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Calculate target size
            target_height = original_height
            target_width = int(original_height * (9 / 16))
            if target_width % 2 != 0:
                target_width += 1
            target_size = (target_width, target_height)
            
            # Configure smoothing
            smoothing_configs = {
                "low": {"smoothing_factor": 0.3, "max_jump_distance": 80, "stability_frames": 3},
                "medium": {"smoothing_factor": 0.75, "max_jump_distance": 50, "stability_frames": 5},
                "high": {"smoothing_factor": 0.9, "max_jump_distance": 25, "stability_frames": 8},
                "very_high": {"smoothing_factor": 0.95, "max_jump_distance": 15, "stability_frames": 12}
            }
            
            smoothing_config = smoothing_configs.get(smoothing_strength, smoothing_configs["medium"])
            
            self._update_task_status(
                task_id, "processing", 10, 
                f"Video: {target_size[0]}x{target_size[1]}, {fps}fps, {total_frames} frames"
            )
            
            # Extract audio if needed
            audio_data = None
            if use_speaker_detection and self.vad:
                self._update_task_status(task_id, "processing", 15, "Extracting audio for voice detection...")
                audio_data = await self.extract_audio_for_vad(input_video_path)
            
            # Process video
            self._update_task_status(task_id, "processing", 20, "Starting video processing...")
            
            # Run the heavy video processing in a separate task
            result = await self._process_video_frames(
                task_id, input_video_path, output_video_path, 
                target_size, smoothing_config, audio_data,
                use_speaker_detection, fps, total_frames
            )
            
            if result["success"]:
                self._update_task_status(
                    task_id, "completed", 100, 
                    f"Video processing completed! Output: {output_video_path}",
                    {"output_path": str(output_video_path), "file_size_mb": result.get("file_size_mb", 0)}
                )
            else:
                self._update_task_status(
                    task_id, "failed", 0, 
                    f"Processing failed: {result.get('error', 'Unknown error')}"
                )
            
            return {
                "success": result["success"],
                "task_id": task_id,
                "output_path": str(output_video_path) if result["success"] else None,
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"❌ Async vertical crop failed for task {task_id}: {str(e)}")
            self._update_task_status(task_id, "failed", 0, f"Error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _process_video_frames(
        self,
        task_id: str,
        input_video_path: Path,
        output_video_path: Path,
        target_size: Tuple[int, int],
        smoothing_config: Dict[str, Any],
        audio_data: Optional[bytes],
        use_speaker_detection: bool,
        fps: int,
        total_frames: int
    ) -> Dict[str, Any]:
        """Process video frames with async/await pattern"""
        try:
            # Setup temp video path
            temp_video_path = output_video_path.with_name(f"{output_video_path.stem}_temp_{task_id}.mp4")
            
            # Video processing state
            previous_crop_center = None
            recent_centers = []
            
            # Setup audio generator
            audio_generator = None
            if audio_data:
                audio_generator = self._process_audio_frames(audio_data)
            
            # Open video
            cap = cv2.VideoCapture(str(input_video_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, target_size)
            
            frame_count = 0
            last_progress_update = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get audio frame
                audio_frame = None
                if audio_generator:
                    try:
                        audio_frame = next(audio_generator)
                    except StopIteration:
                        audio_frame = None
                
                # Find active speaker
                speaker_box = None
                if use_speaker_detection:
                    speaker_box = await self.find_active_speaker(frame, audio_frame, previous_crop_center)
                
                # Calculate crop center with smoothing
                if speaker_box:
                    x, y, x1, y1 = speaker_box
                    raw_center = ((x + x1) // 2, (y + y1) // 2)
                else:
                    h, w = frame.shape[:2]
                    raw_center = (w // 2, h // 2)
                
                # Apply smoothing
                crop_center, recent_centers = self._smooth_crop_center(
                    raw_center, previous_crop_center, recent_centers, smoothing_config
                )
                previous_crop_center = crop_center
                
                # Crop frame
                cropped_frame = await self.crop_frame_to_vertical(
                    frame, speaker_box, target_size, crop_center
                )
                
                # Write frame
                out.write(cropped_frame)
                
                frame_count += 1
                
                # Update progress (every 2 seconds to avoid spam)
                if frame_count - last_progress_update >= (fps * 2):
                    progress = 20 + int((frame_count / total_frames) * 60)  # 20-80% for video processing
                    self._update_task_status(
                        task_id, "processing", progress,
                        f"Processing frames: {frame_count}/{total_frames} ({progress-20:.1f}%)"
                    )
                    last_progress_update = frame_count
                    
                    # Yield control to allow other tasks to run
                    await asyncio.sleep(0)
            
            cap.release()
            out.release()
            
            # Add audio back
            self._update_task_status(task_id, "processing", 85, "Adding audio to video...")
            
            success = await self._add_audio_to_video(temp_video_path, input_video_path, output_video_path)
            
            # Calculate file size
            file_size_mb = 0
            if output_video_path.exists():
                file_size_mb = output_video_path.stat().st_size / (1024 * 1024)
            
            # Cleanup temp file
            if temp_video_path.exists():
                os.remove(temp_video_path)
            
            return {
                "success": success,
                "file_size_mb": round(file_size_mb, 2),
                "frames_processed": frame_count
            }
            
        except Exception as e:
            logger.error(f"Video processing error for task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _add_audio_to_video(
        self, 
        temp_video_path: Path, 
        input_video_path: Path, 
        output_video_path: Path
    ) -> bool:
        """Add audio to video using ffmpeg (async)"""
        try:
            # Check if original has audio
            with VideoFileClip(str(input_video_path)) as original_clip:
                if original_clip.audio is None:
                    temp_video_path.rename(output_video_path)
                    return True
            
            # Use ffmpeg to merge audio
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_video_path),
                '-i', str(input_video_path),
                '-c:v', 'copy', '-c:a', 'copy',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', str(output_video_path), '-y'
            ]
            
            # Run ffmpeg asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                logger.error(f"ffmpeg error: {stderr.decode()}")
                # Fallback: rename temp file
                if temp_video_path.exists():
                    temp_video_path.rename(output_video_path)
                return False
                
        except Exception as e:
            logger.error(f"Audio merge error: {e}")
            if temp_video_path.exists():
                temp_video_path.rename(output_video_path)
            return False

# Global async service instance
async_vertical_crop_service = AsyncVerticalCropService()

# Convenience functions
async def crop_video_to_vertical_async(
    input_path: Path,
    output_path: Path,
    use_speaker_detection: bool = True,
    smoothing_strength: str = "very_high",
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async convenience function to crop video to vertical format
    
    Returns:
        Dict with success, task_id, output_path, error keys
    """
    return await async_vertical_crop_service.create_vertical_crop_async(
        input_path, output_path, use_speaker_detection, smoothing_strength, task_id
    )

async def get_crop_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get status of a cropping task"""
    return await async_vertical_crop_service.get_task_status(task_id)

async def list_crop_tasks() -> Dict[str, Dict[str, Any]]:
    """List all active cropping tasks"""
    return await async_vertical_crop_service.list_active_tasks()

async def cleanup_old_crop_tasks(max_age_hours: int = 24):
    """Clean up old completed tasks"""
    await async_vertical_crop_service.cleanup_completed_tasks(max_age_hours) 