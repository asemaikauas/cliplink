#!/usr/bin/env python3
"""
Video Cropping Service - Automatically generates vertical (9:16) clips from horizontal videos
Supports both solo speaker and interview modes with intelligent speaker detection

Based on ClipsAI reference implementation with custom adaptations
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import json
from datetime import datetime, timedelta
import subprocess
import uuid

# Core ML libraries
import mediapipe as mp
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import webrtcvad

# Audio processing
try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import ffmpeg
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("PyAnnote audio processing not available - interview mode will be limited")

# Video processing
try:
    from moviepy import VideoFileClip
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    logging.error("Video processing libraries not available")

# Setup logging
logger = logging.getLogger(__name__)

class CropMode(Enum):
    """Video cropping modes"""
    AUTO = "auto"
    SOLO = "solo"
    INTERVIEW = "interview"
    FALLBACK = "fallback"

class AspectRatio(Enum):
    """Supported aspect ratios"""
    VERTICAL = (9, 16)  # 9:16 for TikTok/Shorts
    SQUARE = (1, 1)     # 1:1 for Instagram
    PORTRAIT = (4, 5)   # 4:5 for Instagram
    THREE_FOUR = (3, 4) # 3:4 alternative

@dataclass
class FaceDetection:
    """Face detection result"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Optional[Dict] = None

@dataclass
class SpeakerSegment:
    """Speaker segment with timing info"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float

@dataclass
class CropSettings:
    """Video cropping configuration"""
    mode: CropMode = CropMode.AUTO
    target_aspect_ratio: AspectRatio = AspectRatio.VERTICAL
    output_resolution: Tuple[int, int] = (1080, 1920)  # 9:16 at 1080p
    confidence_threshold: float = 0.7
    enable_scene_detection: bool = True
    smoothing_window: int = 30  # frames for temporal smoothing
    padding_ratio: float = 0.1  # extra padding around face
    fallback_crop_center: bool = True

# Task management for async processing
crop_tasks: Dict[str, Dict] = {}
crop_task_lock = threading.Lock()
crop_executor = ThreadPoolExecutor(max_workers=4)

class AdvancedFaceDetector:
    """Advanced face detection using MediaPipe with MTCNN fallback"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = None
        self.face_mesh = None
        self.mtcnn_detector = None
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize MediaPipe and MTCNN detectors"""
        try:
            # MediaPipe Face Detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            
            # MediaPipe Face Mesh for mouth detection
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("✅ MediaPipe face detection initialized")
            
        except Exception as e:
            logger.error(f"❌ MediaPipe initialization failed: {e}")
        
        # MTCNN fallback
        try:
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.8])
            logger.info("✅ MTCNN fallback detector initialized")
        except ImportError:
            logger.warning("⚠️ MTCNN not available - no fallback face detection")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe with MTCNN fallback"""
        faces = []
        
        # Try MediaPipe first
        if self.face_detection:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        confidence = detection.score[0]
                        
                        # Convert to absolute coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        faces.append(FaceDetection(x, y, width, height, confidence))
                
                return faces
                
            except Exception as e:
                logger.warning(f"MediaPipe detection failed: {e}")
        
        # Fallback to MTCNN
        if self.mtcnn_detector and not faces:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.mtcnn_detector.detect_faces(rgb_frame)
                
                for detection in detections:
                    bbox = detection['box']
                    confidence = detection['confidence']
                    
                    faces.append(FaceDetection(
                        x=bbox[0], y=bbox[1], 
                        width=bbox[2], height=bbox[3], 
                        confidence=confidence
                    ))
                
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}")
        
        return faces
    
    def get_mouth_activity(self, frame: np.ndarray, face_box: FaceDetection) -> float:
        """Detect mouth movement for speaker identification"""
        if not self.face_mesh:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Use mouth landmarks to detect opening
                # This is a simplified implementation
                landmarks = results.multi_face_landmarks[0]
                
                # Get mouth corner and top/bottom lip landmarks
                mouth_top = landmarks.landmark[13]
                mouth_bottom = landmarks.landmark[14]
                
                # Calculate mouth opening (simplified)
                mouth_opening = abs(mouth_top.y - mouth_bottom.y)
                return mouth_opening
                
        except Exception as e:
            logger.warning(f"Mouth activity detection failed: {e}")
        
        return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        if self.face_detection:
            self.face_detection.close()
        if self.face_mesh:
            self.face_mesh.close()

class SpeakerDiarizer:
    """Speaker diarization using PyAnnote Audio"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.pipeline = None
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize PyAnnote pipeline"""
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Audio processing not available - speaker diarization disabled")
            return
        
        if not self.hf_token:
            logger.warning("No HuggingFace token - speaker diarization disabled")
            return
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            logger.info("✅ PyAnnote speaker diarization initialized")
        except Exception as e:
            logger.error(f"❌ PyAnnote initialization failed: {e}")
    
    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video for processing"""
        audio_path = video_path.with_suffix('.wav')
        
        try:
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise
    
    def diarize_speakers(self, audio_path: Path) -> List[SpeakerSegment]:
        """Perform speaker diarization"""
        if not self.pipeline:
            return []
        
        try:
            # Run diarization
            diarization = self.pipeline(str(audio_path))
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    start_time=turn.start,
                    end_time=turn.end,
                    speaker_id=speaker,
                    confidence=1.0  # PyAnnote doesn't provide confidence
                ))
            
            logger.info(f"✅ Detected {len(segments)} speaker segments")
            return segments
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return []

class SceneDetector:
    """Scene change detection using PySceneDetect"""
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
    
    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """Detect scene changes in video"""
        try:
            # Create video manager and scene manager
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()
            
            # Add content detector
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            
            # Set downscale factor for faster processing
            video_manager.set_downscale_factor()
            
            # Start video manager
            video_manager.start()
            
            # Perform scene detection
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            # Convert to time ranges
            scenes = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scenes.append((start_time, end_time))
            
            logger.info(f"✅ Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return [(0.0, 0.0)]  # Return full video as single scene

class VideoCropperCore:
    """Core video cropping logic"""
    
    def __init__(self, settings: CropSettings):
        self.settings = settings
        self.face_detector = AdvancedFaceDetector()
        self.speaker_diarizer = SpeakerDiarizer()
        self.scene_detector = SceneDetector()
        
        # Tracking state
        self.face_trackers = {}
        self.current_speaker = None
        self.speaker_segments = []
        self.scene_changes = []
        
        # Smoothing buffers
        self.crop_history = []
        self.speaker_history = []
    
    def analyze_video_mode(self, video_path: Path) -> CropMode:
        """Analyze video to determine optimal cropping mode"""
        try:
            logger.info(f"🔍 Analyzing video mode for: {video_path.name}")
            
            # Extract short sample for analysis (first 30 seconds)
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_frames = int(fps * 30)  # 30 seconds
            
            face_counts = []
            frame_count = 0
            
            while frame_count < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every 30th frame for efficiency
                if frame_count % 30 == 0:
                    faces = self.face_detector.detect_faces(frame)
                    face_counts.append(len(faces))
                
                frame_count += 1
            
            cap.release()
            
            # Analyze face count distribution
            if not face_counts:
                return CropMode.FALLBACK
            
            avg_faces = np.mean(face_counts)
            max_faces = max(face_counts)
            
            logger.info(f"   Average faces: {avg_faces:.1f}, Max faces: {max_faces}")
            
            # Decision logic
            if avg_faces < 0.5:
                return CropMode.FALLBACK
            elif avg_faces <= 1.2 and max_faces <= 2:
                return CropMode.SOLO
            elif avg_faces > 1.2 or max_faces > 2:
                return CropMode.INTERVIEW
            else:
                return CropMode.SOLO
                
        except Exception as e:
            logger.error(f"Mode analysis failed: {e}")
            return CropMode.FALLBACK
    
    def process_solo_mode(self, video_path: Path, output_path: Path) -> bool:
        """Process video in solo speaker mode"""
        try:
            logger.info("🎯 Processing in SOLO mode - tracking single speaker")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_width, out_height = self.settings.output_resolution
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
            
            # Face tracker
            tracker = cv2.TrackerCSRT_create()
            tracker_initialized = False
            last_face_box = None
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # Detect faces periodically or when tracker fails
                if frame_num % 30 == 0 or not tracker_initialized:
                    faces = self.face_detector.detect_faces(frame)
                    
                    if faces:
                        # Use the most confident face
                        best_face = max(faces, key=lambda f: f.confidence)
                        
                        if best_face.confidence > self.settings.confidence_threshold:
                            # Initialize/reinitialize tracker
                            tracker = cv2.TrackerCSRT_create()
                            bbox = (best_face.x, best_face.y, best_face.width, best_face.height)
                            tracker.init(frame, bbox)
                            tracker_initialized = True
                            last_face_box = best_face
                
                # Update tracker
                crop_box = None
                if tracker_initialized:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w_box, h_box = [int(v) for v in bbox]
                        crop_box = FaceDetection(x, y, w_box, h_box, 1.0)
                    else:
                        tracker_initialized = False
                
                # Fallback to last known position or center
                if crop_box is None:
                    if last_face_box:
                        crop_box = last_face_box
                    else:
                        # Center crop
                        crop_box = FaceDetection(
                            x=w//4, y=h//4, 
                            width=w//2, height=h//2, 
                            confidence=0.5
                        )
                
                # Calculate crop region with padding
                crop_region = self._calculate_crop_region(
                    crop_box, (w, h), self.settings.target_aspect_ratio.value
                )
                
                # Apply temporal smoothing
                crop_region = self._smooth_crop_region(crop_region)
                
                # Crop and resize frame
                cropped_frame = self._apply_crop(frame, crop_region, (out_width, out_height))
                out.write(cropped_frame)
                
                frame_num += 1
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"   Processing: {progress:.1f}% ({frame_num}/{total_frames})")
            
            cap.release()
            out.release()
            
            logger.info("✅ Solo mode processing completed")
            return True
            
        except Exception as e:
            logger.error(f"Solo mode processing failed: {e}")
            return False
    
    def process_interview_mode(self, video_path: Path, output_path: Path) -> bool:
        """Process video in interview mode with speaker switching"""
        try:
            logger.info("🎯 Processing in INTERVIEW mode - tracking multiple speakers")
            
            # First, extract audio and perform speaker diarization
            audio_path = self.speaker_diarizer.extract_audio(video_path)
            speaker_segments = self.speaker_diarizer.diarize_speakers(audio_path)
            
            # Clean up audio file
            if audio_path.exists():
                audio_path.unlink()
            
            if not speaker_segments:
                logger.warning("No speaker segments found - falling back to solo mode")
                return self.process_solo_mode(video_path, output_path)
            
            # Process video with speaker switching
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_width, out_height = self.settings.output_resolution
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
            
            # Face trackers for multiple speakers
            face_trackers = {}
            speaker_faces = {}
            current_speaker = None
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_num / fps
                h, w = frame.shape[:2]
                
                # Determine active speaker at this time
                active_speaker = self._get_active_speaker(current_time, speaker_segments)
                
                # Detect faces periodically
                if frame_num % 30 == 0:
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Update speaker-face associations
                    if len(faces) >= 2:
                        # Assume left speaker is host, right is guest
                        faces_sorted = sorted(faces, key=lambda f: f.x)
                        speaker_faces['SPEAKER_00'] = faces_sorted[0]  # Left speaker
                        speaker_faces['SPEAKER_01'] = faces_sorted[1]  # Right speaker
                    elif len(faces) == 1:
                        # Single face - assign to active speaker
                        if active_speaker:
                            speaker_faces[active_speaker] = faces[0]
                
                # Get target face based on active speaker
                target_face = None
                if active_speaker and active_speaker in speaker_faces:
                    target_face = speaker_faces[active_speaker]
                elif speaker_faces:
                    # Fallback to any available face
                    target_face = next(iter(speaker_faces.values()))
                
                # Calculate crop region
                if target_face:
                    crop_region = self._calculate_crop_region(
                        target_face, (w, h), self.settings.target_aspect_ratio.value
                    )
                else:
                    # Center crop fallback
                    crop_region = self._calculate_center_crop((w, h), self.settings.target_aspect_ratio.value)
                
                # Apply temporal smoothing
                crop_region = self._smooth_crop_region(crop_region)
                
                # Crop and resize frame
                cropped_frame = self._apply_crop(frame, crop_region, (out_width, out_height))
                out.write(cropped_frame)
                
                frame_num += 1
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"   Processing: {progress:.1f}% ({frame_num}/{total_frames})")
            
            cap.release()
            out.release()
            
            logger.info("✅ Interview mode processing completed")
            return True
            
        except Exception as e:
            logger.error(f"Interview mode processing failed: {e}")
            return False
    
    def process_fallback_mode(self, video_path: Path, output_path: Path) -> bool:
        """Process video with center crop fallback"""
        try:
            logger.info("🎯 Processing in FALLBACK mode - center crop")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_width, out_height = self.settings.output_resolution
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # Calculate center crop
                crop_region = self._calculate_center_crop((w, h), self.settings.target_aspect_ratio.value)
                
                # Crop and resize frame
                cropped_frame = self._apply_crop(frame, crop_region, (out_width, out_height))
                out.write(cropped_frame)
                
                frame_num += 1
                if frame_num % 200 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"   Processing: {progress:.1f}% ({frame_num}/{total_frames})")
            
            cap.release()
            out.release()
            
            logger.info("✅ Fallback mode processing completed")
            return True
            
        except Exception as e:
            logger.error(f"Fallback mode processing failed: {e}")
            return False
    
    def _get_active_speaker(self, current_time: float, segments: List[SpeakerSegment]) -> Optional[str]:
        """Get active speaker at given time"""
        for segment in segments:
            if segment.start_time <= current_time <= segment.end_time:
                return segment.speaker_id
        return None
    
    def _calculate_crop_region(self, face: FaceDetection, frame_size: Tuple[int, int], aspect_ratio: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Calculate crop region based on face detection"""
        frame_w, frame_h = frame_size
        target_w, target_h = aspect_ratio
        
        # Add padding around face
        padding = int(min(face.width, face.height) * self.settings.padding_ratio)
        
        # Calculate face center
        face_center_x = face.x + face.width // 2
        face_center_y = face.y + face.height // 2
        
        # Calculate crop dimensions maintaining aspect ratio
        if frame_w / frame_h > target_w / target_h:
            # Frame is wider - fit height, crop width
            crop_h = frame_h
            crop_w = int(crop_h * target_w / target_h)
        else:
            # Frame is taller - fit width, crop height
            crop_w = frame_w
            crop_h = int(crop_w * target_h / target_w)
        
        # Center crop around face
        crop_x = max(0, min(face_center_x - crop_w // 2, frame_w - crop_w))
        crop_y = max(0, min(face_center_y - crop_h // 2, frame_h - crop_h))
        
        return (crop_x, crop_y, crop_w, crop_h)
    
    def _calculate_center_crop(self, frame_size: Tuple[int, int], aspect_ratio: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Calculate center crop region"""
        frame_w, frame_h = frame_size
        target_w, target_h = aspect_ratio
        
        # Calculate crop dimensions maintaining aspect ratio
        if frame_w / frame_h > target_w / target_h:
            # Frame is wider - fit height, crop width
            crop_h = frame_h
            crop_w = int(crop_h * target_w / target_h)
        else:
            # Frame is taller - fit width, crop height
            crop_w = frame_w
            crop_h = int(crop_w * target_h / target_w)
        
        # Center the crop
        crop_x = (frame_w - crop_w) // 2
        crop_y = (frame_h - crop_h) // 2
        
        return (crop_x, crop_y, crop_w, crop_h)
    
    def _smooth_crop_region(self, crop_region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to reduce jitter"""
        self.crop_history.append(crop_region)
        
        # Keep only recent history
        if len(self.crop_history) > self.settings.smoothing_window:
            self.crop_history.pop(0)
        
        if len(self.crop_history) < 3:
            return crop_region
        
        # Apply moving average
        avg_x = int(np.mean([c[0] for c in self.crop_history[-5:]]))
        avg_y = int(np.mean([c[1] for c in self.crop_history[-5:]]))
        avg_w = int(np.mean([c[2] for c in self.crop_history[-5:]]))
        avg_h = int(np.mean([c[3] for c in self.crop_history[-5:]]))
        
        return (avg_x, avg_y, avg_w, avg_h)
    
    def _apply_crop(self, frame: np.ndarray, crop_region: Tuple[int, int, int, int], output_size: Tuple[int, int]) -> np.ndarray:
        """Apply crop and resize to frame"""
        crop_x, crop_y, crop_w, crop_h = crop_region
        out_w, out_h = output_size
        
        # Crop frame
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Resize to target resolution
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def cleanup(self):
        """Clean up resources"""
        self.face_detector.cleanup()

# Task management for async processing
crop_tasks: Dict[str, Dict] = {}
crop_task_lock = threading.Lock()
crop_executor = ThreadPoolExecutor(max_workers=4)

def _update_crop_progress(task_id: str, step: str, progress: int, message: str, data: Optional[Dict] = None):
    """Update crop task progress with thread safety"""
    with crop_task_lock:
        if task_id in crop_tasks:
            crop_tasks[task_id].update({
                "current_step": step,
                "progress": progress,
                "message": message,
                "updated_at": datetime.now()
            })
            if data:
                crop_tasks[task_id].update(data)

async def crop_video_async(
    task_id: str,
    input_path: Path,
    output_path: Path,
    settings: CropSettings
) -> bool:
    """Async video cropping with progress tracking"""
    try:
        _update_crop_progress(task_id, "init", 5, f"Starting video crop: {input_path.name}")
        
        # Initialize cropper
        cropper = VideoCropperCore(settings)
        
        # Analyze mode if auto
        if settings.mode == CropMode.AUTO:
            _update_crop_progress(task_id, "analysis", 10, "Analyzing video mode...")
            detected_mode = cropper.analyze_video_mode(input_path)
            settings.mode = detected_mode
            _update_crop_progress(task_id, "analysis", 20, f"Detected mode: {detected_mode.value}")
        
        # Process based on mode
        _update_crop_progress(task_id, "processing", 25, f"Processing in {settings.mode.value} mode...")
        
        success = False
        if settings.mode == CropMode.SOLO:
            success = cropper.process_solo_mode(input_path, output_path)
        elif settings.mode == CropMode.INTERVIEW:
            success = cropper.process_interview_mode(input_path, output_path)
        else:
            success = cropper.process_fallback_mode(input_path, output_path)
        
        if success:
            _update_crop_progress(task_id, "completed", 100, "Video cropping completed successfully")
            
            # Get output file size
            file_size_mb = output_path.stat().st_size / (1024*1024) if output_path.exists() else 0
            
            with crop_task_lock:
                crop_tasks[task_id].update({
                    "status": "completed",
                    "result": {
                        "success": True,
                        "output_path": str(output_path),
                        "file_size_mb": round(file_size_mb, 1),
                        "mode_used": settings.mode.value,
                        "target_aspect_ratio": f"{settings.target_aspect_ratio.value[0]}:{settings.target_aspect_ratio.value[1]}",
                        "output_resolution": f"{settings.output_resolution[0]}x{settings.output_resolution[1]}"
                    },
                    "completed_at": datetime.now()
                })
        else:
            raise Exception("Video processing failed")
        
        # Cleanup
        cropper.cleanup()
        return success
        
    except Exception as e:
        error_msg = f"Video cropping failed: {str(e)}"
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