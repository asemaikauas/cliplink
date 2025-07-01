"""Subtitle post-processor for converting Groq segments to SRT and VTT."""

import re
import logging
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from app.exceptions import SubtitleFormatError


logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """A subtitle segment with text and timing."""
    start_time: float
    end_time: float
    text: str
    
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


class SubtitleProcessor:
    """Process transcription segments into subtitle formats."""
    
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        merge_gap_threshold_ms: int = 200
    ):
        """Initialize subtitle processor.
        
        Args:
            max_chars_per_line: Maximum characters per subtitle line
            max_lines: Maximum number of lines per subtitle
            merge_gap_threshold_ms: Merge segments with gaps smaller than this (ms)
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.merge_gap_threshold_ms = merge_gap_threshold_ms
    
    def _merge_micro_gaps(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Merge segments with micro-gaps smaller than threshold.
        
        Args:
            segments: List of subtitle segments
            
        Returns:
            List of merged subtitle segments
        """
        if not segments:
            return segments
        
        merged = [segments[0]]
        merge_threshold_s = self.merge_gap_threshold_ms / 1000.0
        
        for current in segments[1:]:
            previous = merged[-1]
            gap = current.start_time - previous.end_time
            
            # If gap is smaller than threshold, merge segments
            if gap <= merge_threshold_s:
                logger.debug(
                    f"Merging segments with {gap*1000:.1f}ms gap: "
                    f"'{previous.text}' + '{current.text}'"
                )
                merged[-1] = SubtitleSegment(
                    start_time=previous.start_time,
                    end_time=current.end_time,
                    text=f"{previous.text} {current.text}"
                )
            else:
                merged.append(current)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged
    
    def _wrap_text(self, text: str) -> List[str]:
        """Wrap text to meet line length and count constraints.
        
        Args:
            text: Text to wrap
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed line length
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                # If we've reached max lines, truncate
                if len(lines) >= self.max_lines:
                    break
        
        # Add the last line
        if current_line and len(lines) < self.max_lines:
            lines.append(current_line)
        
        return lines
    
    def _format_time_srt(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string for SRT
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_time_vtt(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string for VTT
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def _format_time_simple(self, seconds: float) -> str:
        """Format time in a simple readable format (MM:SS).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (MM:SS or HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _groq_segments_to_subtitle_segments(self, groq_segments: List[Any]) -> List[SubtitleSegment]:
        """Convert Groq transcription segments to subtitle segments.
        
        Args:
            groq_segments: Groq transcription segments (can be dicts or objects)
            
        Returns:
            List of subtitle segments
        """
        subtitle_segments = []
        
        for segment in groq_segments:
            # Handle both dictionary and object formats
            if isinstance(segment, dict):
                # Dictionary format: {'text': '...', 'start': 0.0, 'end': 3.5}
                text = segment.get('text', '').strip()
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)
            else:
                # Object format: segment.text, segment.start, segment.end
                text = getattr(segment, 'text', '').strip()
                start_time = getattr(segment, 'start', 0.0)
                end_time = getattr(segment, 'end', 0.0)
            
            if not text:
                continue
            
            subtitle_segments.append(SubtitleSegment(
                start_time=start_time,
                end_time=end_time,
                text=text
            ))
        
        return subtitle_segments
    
    def process_segments(self, groq_segments: List[Any]) -> List[SubtitleSegment]:
        """Process Groq segments into optimized subtitle segments.
        
        Args:
            groq_segments: Raw Groq transcription segments
            
        Returns:
            Processed subtitle segments
        """
        try:
            logger.info(f"Processing {len(groq_segments)} Groq segments")
            
            # Convert to subtitle segments
            subtitle_segments = self._groq_segments_to_subtitle_segments(groq_segments)
            
            # Merge micro-gaps
            merged_segments = self._merge_micro_gaps(subtitle_segments)
            
            # Wrap text for each segment
            final_segments = []
            for segment in merged_segments:
                wrapped_lines = self._wrap_text(segment.text)
                if wrapped_lines:
                    final_segments.append(SubtitleSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text="\n".join(wrapped_lines)
                    ))
            
            logger.info(f"Final processed segments: {len(final_segments)}")
            
            # Print subtitles to console
            self._print_subtitles_to_console(final_segments)
            
            return final_segments
            
        except Exception as e:
            raise SubtitleFormatError(f"Failed to process segments: {str(e)}")
    
    def _print_subtitles_to_console(self, segments: List[SubtitleSegment]) -> None:
        """Print formatted subtitles to console.
        
        Args:
            segments: List of subtitle segments to print
        """
        if not segments:
            print("\n⚠️  No subtitle segments to display\n")
            sys.stdout.flush()
            return
        
        print("\n" + "="*80)
        print(f"📝 GENERATED SUBTITLES ({len(segments)} segments)")
        print("="*80)
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_time_simple(segment.start_time)
            end_time = self._format_time_simple(segment.end_time)
            duration = segment.end_time - segment.start_time
            
            print(f"\n[{i:3d}] {start_time} --> {end_time} ({duration:.1f}s)")
            
            # Handle multi-line text
            lines = segment.text.split('\n')
            for line in lines:
                print(f"      {line}")
        
        print("\n" + "="*80 + "\n")
        sys.stdout.flush()
    
    def generate_srt(self, segments: List[SubtitleSegment]) -> str:
        """Generate SRT subtitle content.
        
        Args:
            segments: List of subtitle segments
            
        Returns:
            SRT content as string
        """
        try:
            srt_content = []
            
            for i, segment in enumerate(segments, 1):
                start_time = self._format_time_srt(segment.start_time)
                end_time = self._format_time_srt(segment.end_time)
                
                srt_content.extend([
                    str(i),
                    f"{start_time} --> {end_time}",
                    segment.text,
                    ""  # Empty line separator
                ])
            
            return "\n".join(srt_content)
            
        except Exception as e:
            raise SubtitleFormatError(f"Failed to generate SRT: {str(e)}")
    
    def generate_vtt(self, segments: List[SubtitleSegment]) -> str:
        """Generate VTT subtitle content.
        
        Args:
            segments: List of subtitle segments
            
        Returns:
            VTT content as string
        """
        try:
            vtt_content = ["WEBVTT", ""]
            
            for segment in segments:
                start_time = self._format_time_vtt(segment.start_time)
                end_time = self._format_time_vtt(segment.end_time)
                
                vtt_content.extend([
                    f"{start_time} --> {end_time}",
                    segment.text,
                    ""  # Empty line separator
                ])
            
            return "\n".join(vtt_content)
            
        except Exception as e:
            raise SubtitleFormatError(f"Failed to generate VTT: {str(e)}")
    
    def save_subtitles(
        self,
        segments: List[SubtitleSegment],
        output_dir: str,
        filename_base: str
    ) -> Tuple[str, str]:
        """Save subtitle files in both SRT and VTT formats.
        
        Args:
            segments: List of subtitle segments
            output_dir: Output directory path
            filename_base: Base filename without extension
            
        Returns:
            Tuple of (srt_path, vtt_path)
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate content
            srt_content = self.generate_srt(segments)
            vtt_content = self.generate_vtt(segments)
            
            # Save files
            srt_path = output_path / f"{filename_base}.srt"
            vtt_path = output_path / f"{filename_base}.vtt"
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(vtt_content)
            
            logger.info(f"Saved subtitles: {srt_path}, {vtt_path}")
            return str(srt_path), str(vtt_path)
            
        except Exception as e:
            raise SubtitleFormatError(f"Failed to save subtitles: {str(e)}")


def convert_groq_to_subtitles(
    groq_segments: List[Any],
    output_dir: str,
    filename_base: str,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    merge_gap_threshold_ms: int = 200
) -> Tuple[str, str]:
    """Convenience function to convert Groq segments to subtitle files.
    
    Args:
        groq_segments: Groq transcription segments
        output_dir: Output directory path
        filename_base: Base filename without extension
        max_chars_per_line: Maximum characters per subtitle line
        max_lines: Maximum number of lines per subtitle
        merge_gap_threshold_ms: Merge segments with gaps smaller than this (ms)
        
    Returns:
        Tuple of (srt_path, vtt_path)
    """
    processor = SubtitleProcessor(max_chars_per_line, max_lines, merge_gap_threshold_ms)
    segments = processor.process_segments(groq_segments)
    return processor.save_subtitles(segments, output_dir, filename_base) 