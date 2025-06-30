"""Groq Whisper client wrapper with VAD pre-filtering."""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()
import groq
from pydub import AudioSegment
from pydub.silence import detect_silence

from app.exceptions import TranscriptionError, VADError


logger = logging.getLogger(__name__)


class GroqClient:
    """Groq Whisper large-v3 client with VAD pre-filtering."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise TranscriptionError("GROQ_API_KEY not found in environment variables")
        
        self.client = groq.Groq(api_key=self.api_key)
        self.model = "whisper-large-v3"
    
    def _apply_vad_filtering(
        self, 
        audio_path: str, 
        silence_threshold: int = -45,
        min_silence_duration: int = 3000
    ) -> str:
        """Apply Voice Activity Detection to remove silent stretches.
        
        Args:
            audio_path: Path to input audio file
            silence_threshold: Silence threshold in dB (default: -45 dB)
            min_silence_duration: Minimum silence duration in ms (default: 3000 ms)
            
        Returns:
            Path to processed audio file with silence removed
            
        Raises:
            VADError: If VAD processing fails
        """
        try:
            logger.info(f"Applying VAD filtering to {audio_path}")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Detect silent segments
            silent_segments = detect_silence(
                audio,
                min_silence_len=min_silence_duration,
                silence_thresh=silence_threshold
            )
            
            logger.info(f"Found {len(silent_segments)} silent segments to remove")
            
            # Remove silent segments (in reverse order to maintain indices)
            for start, end in reversed(silent_segments):
                logger.debug(f"Removing silence from {start}ms to {end}ms")
                audio = audio[:start] + audio[end:]
            
            # Export processed audio
            output_path = audio_path.replace(".wav", "_vad_filtered.wav")
            if output_path == audio_path:  # If not a .wav file, add suffix
                output_path = f"{audio_path}_vad_filtered.wav"
            
            audio.export(output_path, format="wav")
            
            logger.info(f"VAD filtered audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise VADError(f"VAD filtering failed: {str(e)}")
    
    def transcribe(
        self,
        file_path: str,
        apply_vad: bool = True,
        language: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file using Groq Whisper large-v3.
        
        Args:
            file_path: Path to audio/video file
            apply_vad: Whether to apply VAD pre-filtering
            language: Language code (optional, auto-detect if None)
            task_id: Task ID for logging
            
        Returns:
            Dictionary containing:
                - segments: List of transcription segments with word-level timestamps
                - language: Detected language code
                - cost_usd: Estimated cost in USD
                - latency_ms: Processing latency in milliseconds
                
        Raises:
            TranscriptionError: If transcription fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting transcription for {file_path} (task_id: {task_id})")
            
            # Apply VAD filtering if requested
            processed_file_path = file_path
            if apply_vad:
                processed_file_path = self._apply_vad_filtering(file_path)
            
            # Open and transcribe the audio file
            with open(processed_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language=language
                )
            
            # Debug: Print what we got from Groq
            print(f"\n🔍 DEBUG - Groq Response for task {task_id}:")
            print(f"   Has segments attr: {hasattr(transcription, 'segments')}")
            print(f"   Segments is None: {getattr(transcription, 'segments', 'NO_ATTR') is None}")
            print(f"   Has text attr: {hasattr(transcription, 'text')}")
            print(f"   Text content: {getattr(transcription, 'text', 'NO_TEXT')[:100] if hasattr(transcription, 'text') else 'NO_TEXT'}...")
            print(f"   Language: {getattr(transcription, 'language', 'NO_LANGUAGE')}")
            if hasattr(transcription, 'segments') and transcription.segments:
                print(f"   Segments count: {len(transcription.segments)}")
                print(f"   First segment: {transcription.segments[0] if transcription.segments else 'NO_SEGMENTS'}")
            
            # Validate the transcription response
            if not hasattr(transcription, 'segments'):
                logger.warning(f"Transcription response missing segments attribute (task_id: {task_id})")
                segments = []
            elif transcription.segments is None:
                logger.warning(f"Transcription segments is None (task_id: {task_id})")
                segments = []
            else:
                segments = transcription.segments
            
            # Validate language
            language_code = getattr(transcription, 'language', 'unknown')
            if not language_code:
                language_code = 'unknown'
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Estimate cost (Groq pricing: ~$0.111 per hour of audio)
            file_size_mb = os.path.getsize(processed_file_path) / (1024 * 1024)
            estimated_duration_hours = file_size_mb / 10  # Rough estimate
            cost_usd = estimated_duration_hours * 0.111
            
            # Print transcribed text to console
            full_text = ""
            if segments:
                full_text = " ".join([segment.text.strip() for segment in segments if hasattr(segment, 'text')])
            elif hasattr(transcription, 'text') and transcription.text:
                full_text = transcription.text
            
            if full_text:
                print("\n" + "="*80)
                print(f"🎤 TRANSCRIBED TEXT (Task: {task_id})")
                print("="*80)
                print(full_text)
                print("="*80 + "\n")
            else:
                print(f"\n⚠️  No transcribed text found for task {task_id}\n")
            
            logger.info(
                f"Transcription completed (task_id: {task_id}) - "
                f"language: {language_code}, "
                f"segments: {len(segments)}, "
                f"latency: {latency_ms}ms, "
                f"estimated_cost: ${cost_usd:.4f}"
            )
            
            # Validate that we have some content
            if len(segments) == 0:
                logger.warning(f"No transcription segments found (task_id: {task_id})")
                # Check if there's text content in the response
                if hasattr(transcription, 'text') and transcription.text:
                    logger.info(f"Found transcription text but no segments, creating fallback segment")
                    # Create a simple segment from the full text
                    from types import SimpleNamespace
                    fallback_segment = SimpleNamespace()
                    fallback_segment.start = 0.0
                    fallback_segment.end = 30.0  # Longer default duration
                    fallback_segment.text = transcription.text
                    segments = [fallback_segment]
                    
                    # Try to split into smaller segments based on sentences
                    import re
                    sentences = re.split(r'[.!?]+', transcription.text)
                    if len(sentences) > 1:
                        segments = []
                        duration_per_sentence = 30.0 / len(sentences)
                        for i, sentence in enumerate(sentences):
                            if sentence.strip():
                                segment = SimpleNamespace()
                                segment.start = i * duration_per_sentence
                                segment.end = (i + 1) * duration_per_sentence
                                segment.text = sentence.strip()
                                segments.append(segment)
                        logger.info(f"Split text into {len(segments)} sentence-based segments")
                else:
                    logger.warning(f"No transcription content found at all (task_id: {task_id})")
            
            # Clean up VAD-filtered file if created
            if apply_vad and processed_file_path != file_path:
                try:
                    os.remove(processed_file_path)
                except OSError:
                    logger.warning(f"Failed to clean up VAD-filtered file: {processed_file_path}")
            
            return {
                "segments": segments,
                "language": language_code,
                "cost_usd": round(cost_usd, 4),
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"Transcription failed (task_id: {task_id}): {str(e)}")
            raise TranscriptionError(f"Transcription failed: {str(e)}", task_id=task_id)


def transcribe(
    file_path: str,
    apply_vad: bool = True,
    language: Optional[str] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to transcribe audio file.
    
    Args:
        file_path: Path to audio/video file
        apply_vad: Whether to apply VAD pre-filtering
        language: Language code (optional, auto-detect if None)
        task_id: Task ID for logging
        
    Returns:
        Dictionary containing segments, language, cost_usd, latency_ms
    """
    client = GroqClient()
    return client.transcribe(file_path, apply_vad, language, task_id) 