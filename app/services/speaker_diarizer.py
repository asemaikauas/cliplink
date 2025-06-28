#!/usr/bin/env python3
"""
Speaker Diarization for Interview Mode Video Cropping
Uses PyAnnote Audio for speaker identification and segmentation
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
import subprocess

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

from .video_cropper import SpeakerSegment

logger = logging.getLogger(__name__)

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