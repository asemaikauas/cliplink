"""
Advanced Vertical Cropping Service - ClipsAI Inspired
Utilizes state-of-the-art speaker diarization, scene detection, and face tracking
"""

import asyncio
import cv2
import numpy as np
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torchaudio
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import scenedetect
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import mediapipe as mp
from mtcnn import MTCNN
import librosa
from huggingface_hub import hf_hub_download
from transformers import pipeline

from moviepy import VideoFileClip
from pydub import AudioSegment
import subprocess

class AdvancedSpeakerTracker:
    """
    Advanced speaker tracking using pyannote.audio for speaker diarization
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        # Get HF token from environment if not provided
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            print("❌ No HuggingFace token found. Please set HF_TOKEN in your .env file")
            print("   Get your token from: https://huggingface.co/settings/tokens")
            
        self.pipeline = None
        self.speaker_embeddings = {}
        self.initialization_attempted = False
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize pyannote speaker diarization pipeline"""
        if self.initialization_attempted:
            return
            
        self.initialization_attempted = True
        try:
            print("🔍 Initializing pyannote speaker diarization pipeline...")
            
            # Use HuggingFace CLI authentication (preferred method)
            # If token is provided, use it; otherwise rely on HF CLI login
            kwargs = {}
            if self.hf_token and self.hf_token != 'your_huggingface_token_here':
                print(f"   Using provided token: {self.hf_token[:10]}...")
                kwargs['use_auth_token'] = self.hf_token
            else:
                print("   Using HuggingFace CLI authentication...")
            
            # Use the latest speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **kwargs
            )
            
            # Send pipeline to GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("🚀 Speaker diarization pipeline loaded on GPU")
            else:
                print("🚀 Speaker diarization pipeline loaded on CPU")
                
        except Exception as e:
            print(f"❌ Failed to initialize speaker diarization: {e}")
            print(f"ℹ️  This might be due to:")
            print(f"   • HuggingFace authentication not set up (run: huggingface-cli login)")
            print(f"   • Missing license acceptance for pyannote models")
            print(f"   • Network connectivity issues")
            print(f"   • Model access restrictions")
            print(f"🔄 Advanced cropping will proceed with face-detection only")
            self.pipeline = None
    
    def retry_initialization(self, new_token: str = None):
        """Retry pipeline initialization with optional new token"""
        if new_token:
            self.hf_token = new_token
        self.initialization_attempted = False
        self.initialize_pipeline()
    
    def test_model_access(self):
        """Test access to pyannote models"""
        try:
            from pyannote.audio import Model, Inference
            
            # Test segmentation model
            print("📥 Testing pyannote/segmentation-3.0...")
            model = Model.from_pretrained("pyannote/segmentation-3.0")
            inference = Inference(model)
            print("✅ Segmentation model accessible")
            
            # Test speaker diarization
            print("📥 Testing pyannote/speaker-diarization-3.1...")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            print("✅ Speaker diarization pipeline accessible")
            
            return True
            
        except Exception as e:
            print(f"❌ Model access test failed: {e}")
            return False
    
    async def analyze_speakers(self, audio_path: Path) -> Annotation:
        """
        Perform speaker diarization on audio file
        Returns pyannote Annotation with speaker segments
        """
        if not self.pipeline:
            print("⚠️ Speaker diarization pipeline not available, returning empty annotation")
            return Annotation()
        
        try:
            print(f"🎤 Starting speaker diarization analysis on: {audio_path}")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Create input dict for pyannote
            audio_input = {
                "waveform": waveform,
                "sample_rate": 16000
            }
            
            # Perform diarization
            diarization = self.pipeline(audio_input)
            
            print(f"✅ Speaker diarization completed: {len(list(diarization.itertracks()))} speaker segments")
            return diarization
            
        except Exception as e:
            print(f"❌ Speaker diarization failed: {e}")
            print("🔄 Falling back to face-detection only mode")
            return Annotation()
    
    def get_active_speaker_at_time(self, diarization: Annotation, timestamp: float) -> Optional[str]:
        """Get the active speaker at a specific timestamp"""
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                return speaker
        return None


class AdvancedSceneDetector:
    """
    Scene change detection using PySceneDetect
    """
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
    
    async def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        Detect scene changes in video
        Returns list of (start_time, end_time) tuples
        """
        try:
            # Detect scenes using content detector
            scene_list = detect(str(video_path), ContentDetector(threshold=self.threshold))
            
            # Convert to time tuples
            scenes = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scenes.append((start_time, end_time))
            
            print(f"✅ Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            print(f"❌ Scene detection failed: {e}")
            return [(0.0, float('inf'))]  # Single scene fallback


class AdvancedFaceDetector:
    """
    Advanced face detection using MediaPipe and MTCNN
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.mp_face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short range, 1 for full range
            min_detection_confidence=0.5
        )
        
        # Initialize MTCNN for backup
        try:
            self.mtcnn = MTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.8])
            print("✅ MTCNN face detector initialized")
        except Exception as e:
            print(f"⚠️ MTCNN initialization failed: {e}")
            self.mtcnn = None
        
        print("✅ MediaPipe face detector initialized")
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w = frame.shape[:2]
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    x1, y1 = x + width, y + height
                    
                    # Ensure coordinates are valid
                    x, y = max(0, x), max(0, y)
                    x1, y1 = min(w, x1), min(h, y1)
                    
                    if x1 > x and y1 > y:
                        face_center_x = (x + x1) / 2
                        position = "left" if face_center_x < w / 2 else "right"
                        
                        faces.append({
                            "box": (x, y, x1, y1),
                            "position": position,
                            "confidence": detection.score[0],
                            "id": f"mp_{position}_{i}",
                            "landmarks": self._extract_landmarks(detection, w, h)
                        })
            
            return faces
            
        except Exception as e:
            print(f"MediaPipe face detection error: {e}")
            return []
    
    def detect_faces_mtcnn(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN as backup"""
        if not self.mtcnn:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.mtcnn.detect_faces(rgb_frame)
            
            faces = []
            h, w = frame.shape[:2]
            
            for i, detection in enumerate(detections):
                if detection['confidence'] > 0.5:
                    bbox = detection['box']
                    x, y, width, height = bbox
                    x1, y1 = x + width, y + height
                    
                    # Ensure coordinates are valid
                    x, y = max(0, x), max(0, y)
                    x1, y1 = min(w, x1), min(h, y1)
                    
                    if x1 > x and y1 > y:
                        face_center_x = (x + x1) / 2
                        position = "left" if face_center_x < w / 2 else "right"
                        
                        faces.append({
                            "box": (x, y, x1, y1),
                            "position": position,
                            "confidence": detection['confidence'],
                            "id": f"mtcnn_{position}_{i}",
                            "landmarks": detection.get('keypoints', {})
                        })
            
            return faces
            
        except Exception as e:
            print(f"MTCNN face detection error: {e}")
            return []
    
    def _extract_landmarks(self, detection, width: int, height: int) -> Dict[str, Tuple[int, int]]:
        """Extract facial landmarks from MediaPipe detection"""
        landmarks = {}
        if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_keypoints'):
            for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                x = int(keypoint.x * width)
                y = int(keypoint.y * height)
                landmarks[f"point_{i}"] = (x, y)
        return landmarks
    
    async def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Main face detection method - tries MediaPipe first, then MTCNN"""
        faces = self.detect_faces_mediapipe(frame)
        
        # If MediaPipe fails or finds no faces, try MTCNN
        if not faces and self.mtcnn:
            faces = self.detect_faces_mtcnn(frame)
        
        return faces


class AdvancedTransitionManager:
    """
    Intelligent transition management with scene awareness
    """
    
    def __init__(self, stability_frames: int = 8, hold_frames: int = 60, scene_change_reset: bool = True):
        self.stability_frames = stability_frames
        self.hold_frames = hold_frames
        self.scene_change_reset = scene_change_reset
        
        self.current_speaker = None
        self.candidate_speaker = None
        self.last_known_speaker = None
        self.candidate_frames = 0
        self.silent_frames = 0
        self.current_scene_start = 0.0
        
        # Speaker confidence tracking
        self.speaker_confidence_history = []
        self.confidence_window = 10
    
    def reset_for_scene_change(self, scene_start_time: float):
        """Reset transition state when scene changes"""
        if self.scene_change_reset:
            self.current_scene_start = scene_start_time
            self.current_speaker = None
            self.candidate_speaker = None
            self.candidate_frames = 0
            self.silent_frames = 0
            self.speaker_confidence_history = []
            print(f"🎬 Scene change at {scene_start_time:.2f}s - resetting speaker tracking")
    
    def get_stable_target(
        self, 
        active_speaker: Optional[str], 
        confidence: float = 1.0,
        timestamp: float = 0.0
    ) -> Optional[str]:
        """Enhanced stable target selection with confidence scoring"""
        
        # Track confidence history
        self.speaker_confidence_history.append({
            'speaker': active_speaker,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        if len(self.speaker_confidence_history) > self.confidence_window:
            self.speaker_confidence_history.pop(0)
        
        # Case 1: Active speaker detected
        if active_speaker and confidence > 0.6:
            self.silent_frames = 0
            
            # Calculate average confidence for this speaker
            recent_confidences = [
                h['confidence'] for h in self.speaker_confidence_history[-5:]
                if h['speaker'] == active_speaker
            ]
            avg_confidence = np.mean(recent_confidences) if recent_confidences else confidence
            
            # Higher confidence threshold for speaker changes
            if active_speaker != self.current_speaker:
                if active_speaker == self.candidate_speaker:
                    self.candidate_frames += 1
                else:
                    self.candidate_speaker = active_speaker
                    self.candidate_frames = 1
                
                # Require higher stability for speaker changes
                required_frames = max(self.stability_frames, int(self.stability_frames * (2.0 - avg_confidence)))
                
                if self.candidate_frames >= required_frames:
                    self.current_speaker = self.candidate_speaker
                    self.last_known_speaker = self.current_speaker
                    print(f"🎤 Speaker changed to {self.current_speaker} (confidence: {avg_confidence:.2f})")
        
        # Case 2: No clear speaker (silence or low confidence)
        else:
            self.silent_frames += 1
            self.candidate_speaker = None
            self.candidate_frames = 0
            
            # Return to wide shot after extended silence
            if self.silent_frames >= self.hold_frames:
                self.current_speaker = None
        
        # Hold on last known speaker during brief silence
        if self.current_speaker is None and self.last_known_speaker:
            return self.last_known_speaker
        
        return self.current_speaker


class AdvancedVerticalCropService:
    """
    Advanced vertical cropping service using state-of-the-art AI tools
    """
    
    def __init__(self, hf_token: Optional[str] = None, max_workers: int = 4, max_concurrent_tasks: int = 10):
        # Get HF token from environment if not provided
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        if self.hf_token:
            print(f"🔧 Creating AdvancedVerticalCropService with token: {self.hf_token[:10]}...")
        else:
            print("⚠️ Creating AdvancedVerticalCropService without HF token (face-detection only)")
        
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize components
        print("🎤 Initializing speaker tracker...")
        self.speaker_tracker = AdvancedSpeakerTracker(self.hf_token)
        
        print("🎬 Initializing scene detector...")
        self.scene_detector = AdvancedSceneDetector()
        
        print("👤 Initializing face detector...")
        self.face_detector = AdvancedFaceDetector()
        
        # Threading
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, max_workers))
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()
        
        print(f"🚀 AdvancedVerticalCropService initialized successfully")
    
    async def _run_cpu_bound_task(self, func, *args, **kwargs):
        """Run CPU-bound task in thread executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
    
    def _create_task_id(self) -> str:
        """Generate unique task ID"""
        return f"adv_crop_{uuid.uuid4().hex[:8]}"
    
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
    
    async def create_advanced_vertical_crop(
        self,
        input_video_path: Path,
        output_video_path: Path,
        target_aspect_ratio: Tuple[int, int] = (9, 16),
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create vertical crop using advanced AI techniques
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
                "message": "Initializing advanced video processing...",
                "created_at": datetime.now(),
                "input_path": str(input_video_path),
                "output_path": str(output_video_path)
            }
        
        try:
            # Step 1: Extract audio for speaker diarization
            self._update_task_status(task_id, "processing", 10, "Extracting audio for speaker diarization...")
            temp_audio_path = Path(f"temp_audio_{task_id}.wav")
            await self._extract_audio_for_diarization(input_video_path, temp_audio_path)
            
            # Step 2: Perform speaker diarization
            self._update_task_status(task_id, "processing", 25, "Analyzing speakers with AI...")
            try:
                diarization = await self.speaker_tracker.analyze_speakers(temp_audio_path)
                if not list(diarization.itertracks()):
                    print("⚠️ No speaker segments found, using face-detection only mode")
            except Exception as e:
                print(f"⚠️ Speaker diarization unavailable: {e}")
                print("🔄 Proceeding with face-detection only mode")
                diarization = Annotation()
            
            # Step 3: Detect scene changes
            self._update_task_status(task_id, "processing", 40, "Detecting scene changes...")
            scenes = await self.scene_detector.detect_scenes(input_video_path)
            
            # Step 4: Process video with advanced tracking
            self._update_task_status(task_id, "processing", 50, "Processing video with advanced AI tracking...")
            result = await self._process_video_advanced(
                task_id, input_video_path, output_video_path,
                diarization, scenes, target_aspect_ratio
            )
            
            # Cleanup
            if temp_audio_path.exists():
                os.remove(temp_audio_path)
            
            if result["success"]:
                self._update_task_status(
                    task_id, "completed", 100,
                    f"Advanced processing completed! Output: {output_video_path}",
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
            print(f"❌ Advanced vertical crop failed for task {task_id}: {str(e)}")
            self._update_task_status(task_id, "failed", 0, f"Error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _extract_audio_for_diarization(self, video_path: Path, audio_path: Path):
        """Extract high-quality audio for speaker diarization"""
        try:
            # Use ffmpeg for high-quality extraction
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                str(audio_path),
                '-y'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Audio extraction failed: {stderr.decode()}")
                
        except Exception as e:
            print(f"Audio extraction error: {e}")
            raise
    
    async def _process_video_advanced(
        self,
        task_id: str,
        input_video_path: Path,
        output_video_path: Path,
        diarization: Annotation,
        scenes: List[Tuple[float, float]],
        target_aspect_ratio: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process video with advanced AI tracking"""
        try:
            # Setup temp video path
            temp_video_path = output_video_path.with_name(f"{output_video_path.stem}_temp_{task_id}.mp4")
            
            # Open video
            cap = cv2.VideoCapture(str(input_video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # Calculate target dimensions
            target_width = int(original_height * target_aspect_ratio[0] / target_aspect_ratio[1])
            target_height = original_height
            target_size = (target_width, target_height)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, target_size)
            
            # Initialize transition manager
            transition_manager = AdvancedTransitionManager()
            
            # Processing state
            frame_count = 0
            last_progress_update = 0
            current_scene_idx = 0
            
            # Smoothing state
            previous_crop_center = None
            recent_centers = []
            smoothing_factor = 0.85
            max_jump_distance = 30
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current timestamp
                timestamp = frame_count / fps
                
                # Check for scene changes
                if current_scene_idx < len(scenes) - 1:
                    if timestamp >= scenes[current_scene_idx + 1][0]:
                        current_scene_idx += 1
                        transition_manager.reset_for_scene_change(timestamp)
                
                # Get active speaker from diarization
                active_speaker = self.speaker_tracker.get_active_speaker_at_time(diarization, timestamp)
                
                # Detect faces in current frame
                faces = await self.face_detector.detect_faces(frame)
                
                # Find face that matches active speaker
                target_face = self._match_speaker_to_face(active_speaker, faces, timestamp)
                
                # Get stable target from transition manager
                confidence = target_face.get('confidence', 0.0) if target_face else 0.0
                stable_speaker = transition_manager.get_stable_target(
                    active_speaker, confidence, timestamp
                )
                
                # Calculate crop center
                if target_face and stable_speaker == active_speaker:
                    x, y, x1, y1 = target_face['box']
                    raw_center = ((x + x1) // 2, (y + y1) // 2)
                else:
                    # Default to center when no clear speaker
                    h, w = frame.shape[:2]
                    raw_center = (w // 2, h // 2)
                
                # Apply smoothing
                crop_center = self._smooth_crop_center(
                    raw_center, previous_crop_center, recent_centers,
                    smoothing_factor, max_jump_distance
                )
                previous_crop_center = crop_center
                
                # Crop frame
                cropped_frame = self._crop_frame_to_target_ratio(
                    frame, target_size, crop_center
                )
                
                # Write frame
                out.write(cropped_frame)
                
                frame_count += 1
                
                # Update progress
                if frame_count - last_progress_update >= (fps * 2):
                    progress = 50 + int((frame_count / total_frames) * 35)  # 50-85%
                    self._update_task_status(
                        task_id, "processing", progress,
                        f"Processing frames: {frame_count}/{total_frames} ({progress-50:.1f}%)"
                    )
                    last_progress_update = frame_count
                    await asyncio.sleep(0)  # Yield control
            
            cap.release()
            out.release()
            
            # Add audio back
            self._update_task_status(task_id, "processing", 90, "Adding audio to video...")
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
            print(f"Advanced video processing error for task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _match_speaker_to_face(
        self, 
        active_speaker: Optional[str], 
        faces: List[Dict[str, Any]], 
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Match active speaker to detected face"""
        if not active_speaker or not faces:
            return None
        
        # Simple heuristic: return highest confidence face
        # In production, you might use speaker embedding matching
        if faces:
            return max(faces, key=lambda f: f.get('confidence', 0))
        
        return None
    
    def _smooth_crop_center(
        self,
        new_center: Tuple[int, int],
        previous_center: Optional[Tuple[int, int]],
        recent_centers: List[Tuple[int, int]],
        smoothing_factor: float,
        max_jump_distance: int
    ) -> Tuple[int, int]:
        """Apply smoothing to crop center"""
        if previous_center is None:
            return new_center
        
        # Add to history
        recent_centers.append(new_center)
        if len(recent_centers) > 8:
            recent_centers.pop(0)
        
        # Calculate average
        avg_x = sum(c[0] for c in recent_centers) / len(recent_centers)
        avg_y = sum(c[1] for c in recent_centers) / len(recent_centers)
        averaged_center = (int(avg_x), int(avg_y))
        
        prev_x, prev_y = previous_center
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
        
        return (smoothed_x, smoothed_y)
    
    def _crop_frame_to_target_ratio(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int],
        crop_center: Tuple[int, int]
    ) -> np.ndarray:
        """Crop frame to target aspect ratio centered on specific point"""
        h, w = frame.shape[:2]
        target_width, target_height = target_size
        
        crop_center_x, crop_center_y = crop_center
        
        # Calculate crop boundaries
        left = max(0, crop_center_x - target_width // 2)
        right = min(w, left + target_width)
        top = max(0, crop_center_y - target_height // 2)
        bottom = min(h, top + target_height)
        
        # Adjust if hitting boundaries
        if right - left < target_width:
            if left == 0:
                right = min(w, target_width)
            else:
                left = max(0, w - target_width)
        
        if bottom - top < target_height:
            if top == 0:
                bottom = min(h, target_height)
            else:
                top = max(0, h - target_height)
        
        # Perform crop
        cropped = frame[top:bottom, left:right]
        
        # Resize if needed
        if cropped.shape[:2] != (target_height, target_width):
            cropped = cv2.resize(cropped, target_size)
        
        return cropped
    
    async def _add_audio_to_video(
        self,
        temp_video_path: Path,
        input_video_path: Path,
        output_video_path: Path
    ) -> bool:
        """Add audio to video using ffmpeg"""
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
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                print(f"ffmpeg error: {stderr.decode()}")
                if temp_video_path.exists():
                    temp_video_path.rename(output_video_path)
                return False
                
        except Exception as e:
            print(f"Audio merge error: {e}")
            if temp_video_path.exists():
                temp_video_path.rename(output_video_path)
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID"""
        with self.task_lock:
            return self.active_tasks.get(task_id, None)
    
    async def list_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all active tasks"""
        with self.task_lock:
            return self.active_tasks.copy()


# Basic Async Vertical Crop Service (for backward compatibility)
class AsyncVerticalCropService:
    """Basic async vertical crop service for backward compatibility"""
    
    def __init__(self, max_workers: int = 4, max_concurrent_tasks: int = 10):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()
        print(f"🚀 AsyncVerticalCropService initialized with {max_workers} workers")
    
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
                    "updated_at": datetime.now()
                })
                if data:
                    self.active_tasks[task_id].update(data)
    
    async def create_vertical_crop_async(
        self,
        input_path: Path,
        output_path: Path,
        use_speaker_detection: bool = True,
        smoothing_strength: str = "very_high"
    ) -> Dict[str, Any]:
        """Create vertical crop asynchronously"""
        task_id = self._create_task_id()
        
        # Initialize task tracking
        with self.task_lock:
            self.active_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "message": "Processing queued",
                "created_at": datetime.now(),
                "input_path": str(input_path),
                "output_path": str(output_path)
            }
        
        # Start processing in background
        asyncio.create_task(self._process_vertical_crop(
            task_id, input_path, output_path, use_speaker_detection, smoothing_strength
        ))
        
        return {"task_id": task_id, "status": "queued"}
    
    async def _process_vertical_crop(
        self,
        task_id: str,
        input_path: Path,
        output_path: Path,
        use_speaker_detection: bool,
        smoothing_strength: str
    ):
        """Process vertical crop in background"""
        try:
            self._update_task_status(task_id, "processing", 10, "Starting vertical crop processing...")
            
            # Import the basic vertical crop function
            from app.services.vertical_crop import crop_video_to_vertical
            
            # Run in thread executor
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.thread_executor,
                crop_video_to_vertical,
                input_path,
                output_path,
                use_speaker_detection,
                smoothing_strength
            )
            
            if success:
                file_size_mb = 0
                if output_path.exists():
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                
                self._update_task_status(
                    task_id, "completed", 100,
                    f"Vertical crop completed successfully",
                    {
                        "output_path": str(output_path),
                        "file_size_mb": round(file_size_mb, 2),
                        "completed_at": datetime.now()
                    }
                )
            else:
                self._update_task_status(task_id, "failed", 0, "Vertical crop processing failed")
                
        except Exception as e:
            self._update_task_status(task_id, "failed", 0, f"Error: {str(e)}")
    
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
        current_time = datetime.now()
        to_remove = []
        
        with self.task_lock:
            for task_id, task in self.active_tasks.items():
                if task["status"] in ["completed", "failed"]:
                    completed_at = task.get("completed_at", task.get("created_at"))
                    if completed_at and (current_time - completed_at).total_seconds() > max_age_hours * 3600:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.active_tasks[task_id]


# Global service instances
advanced_vertical_crop_service = None
async_vertical_crop_service = AsyncVerticalCropService()

def initialize_advanced_service(hf_token: Optional[str] = None):
    """Initialize the advanced service with HuggingFace token (from env if not provided)"""
    global advanced_vertical_crop_service
    token = hf_token or os.getenv('HF_TOKEN')
    if token:
        print(f"🔧 Initializing global advanced service with HF token: {token[:10]}...")
    else:
        print("🔧 Initializing global advanced service without HF token (face-detection only)")
    advanced_vertical_crop_service = AdvancedVerticalCropService(token)
    print(f"✅ Advanced service initialized successfully")

# Backward compatibility functions
async def crop_video_to_vertical_async(
    input_path: Path,
    output_path: Path,
    use_speaker_detection: bool = True,
    smoothing_strength: str = "very_high"
) -> Dict[str, Any]:
    """Async vertical crop for backward compatibility"""
    return await async_vertical_crop_service.create_vertical_crop_async(
        input_path, output_path, use_speaker_detection, smoothing_strength
    )

async def get_crop_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get crop task status for backward compatibility"""
    return await async_vertical_crop_service.get_task_status(task_id)

async def list_crop_tasks() -> Dict[str, Dict[str, Any]]:
    """List crop tasks for backward compatibility"""
    return await async_vertical_crop_service.list_active_tasks()

async def cleanup_old_crop_tasks(max_age_hours: int = 24):
    """Cleanup old crop tasks for backward compatibility"""
    await async_vertical_crop_service.cleanup_completed_tasks(max_age_hours)

# Advanced functions
async def crop_video_advanced(
    input_path: Path,
    output_path: Path,
    target_aspect_ratio: Tuple[int, int] = (9, 16),
    task_id: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Advanced video cropping with AI speaker tracking
    HF token will be loaded from environment variables if not provided
    """
    global advanced_vertical_crop_service
    
    # Initialize service if not already done or if a specific token is provided
    if not advanced_vertical_crop_service or hf_token:
        token = hf_token or os.getenv('HF_TOKEN')
        if token:
            print(f"🔧 Initializing advanced service with HF token: {token[:10]}...")
        else:
            print("🔧 Initializing advanced service without HF token (face-detection only)")
        advanced_vertical_crop_service = AdvancedVerticalCropService(token)
    
    return await advanced_vertical_crop_service.create_advanced_vertical_crop(
        input_path, output_path, target_aspect_ratio, task_id
    )

async def get_advanced_crop_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get status of an advanced cropping task"""
    if not advanced_vertical_crop_service:
        return None
    return await advanced_vertical_crop_service.get_task_status(task_id)
