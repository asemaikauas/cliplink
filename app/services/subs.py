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
        max_chars_per_line: int = 50,  # Legacy parameter - not used in CapCut mode
        max_lines: int = 2,  # Legacy parameter - not used in CapCut mode  
        merge_gap_threshold_ms: int = 200,
        capcut_mode: bool = True,  # Enable CapCut-style punch words
        min_word_duration_ms: int = 600,  # Minimum display time per word chunk
        max_word_duration_ms: int = 1200,  # Maximum display time per word chunk
        word_overlap_ms: int = 200  # Overlap between word chunks for smooth flow
    ):
        """Initialize subtitle processor.
        
        Args:
            max_chars_per_line: Maximum characters per subtitle line (legacy)
            max_lines: Maximum number of lines per subtitle (legacy)
            merge_gap_threshold_ms: Merge segments with gaps smaller than this (ms)
            capcut_mode: Enable CapCut-style 1-3 word punch subtitles
            min_word_duration_ms: Minimum display time per word chunk (ms)
            max_word_duration_ms: Maximum display time per word chunk (ms)  
            word_overlap_ms: Overlap between word chunks for smooth transitions (ms)
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.merge_gap_threshold_ms = merge_gap_threshold_ms
        self.capcut_mode = capcut_mode
        self.min_word_duration_ms = min_word_duration_ms
        self.max_word_duration_ms = max_word_duration_ms
        self.word_overlap_ms = word_overlap_ms
    
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
    
    def _wrap_text(self, text: str) -> Tuple[List[str], str]:
        """Wrap text to meet line length and count constraints.
        
        Args:
            text: Text to wrap
            
        Returns:
            Tuple of (wrapped_lines, remaining_text)
        """
        words = text.split()
        lines = []
        current_line = ""
        remaining_words = []
        
        for i, word in enumerate(words):
            # Check if adding this word would exceed line length
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                # If we've reached max lines, save remaining words starting from NEXT word
                if len(lines) >= self.max_lines:
                    # Current word goes in current_line, remaining starts from next word
                    remaining_words = words[i + 1:]  # Start from NEXT word, not current
                    break
        
        # Add the last line if we haven't exceeded max lines
        if current_line and len(lines) < self.max_lines:
            lines.append(current_line)
        elif current_line and len(lines) >= self.max_lines:
            # If we have a current line but reached max lines, add it to remaining
            remaining_words = [current_line] + remaining_words
        
        remaining_text = " ".join(remaining_words) if remaining_words else ""
        
        return lines, remaining_text
    
    def _create_capcut_word_chunks(self, text: str, start_time: float, end_time: float) -> List[SubtitleSegment]:
        """Create CapCut-style word chunks with millisecond precision timing.
        
        Args:
            text: Text to chunk into punch words
            start_time: Original segment start time (seconds)
            end_time: Original segment end time (seconds)
            
        Returns:
            List of word chunk segments with overlapping timing
        """
        words = text.split()
        if not words:
            return []
        
        duration_s = end_time - start_time
        duration_ms = duration_s * 1000
        
        # Create 1-3 word chunks with smarter grouping
        chunks = []
        i = 0
        while i < len(words):
            remaining_words = len(words) - i
            
            # Smart chunk size selection
            if remaining_words >= 5:
                # Prefer 3-word chunks when we have plenty of words
                chunk_size = 3
            elif remaining_words == 4:
                # Split 4 words into 2+2
                chunk_size = 2
            elif remaining_words == 3:
                # Keep 3 words together
                chunk_size = 3
            elif remaining_words == 2:
                # Keep 2 words together
                chunk_size = 2
            else:
                # Single word
                chunk_size = 1
            
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            i += chunk_size
        
        if not chunks:
            return []
        
        logger.debug(f"CapCut chunking: '{text}' -> {len(chunks)} chunks: {chunks}")
        
        # Calculate timing for each chunk
        segments = []
        overlap_s = self.word_overlap_ms / 1000.0
        min_duration_s = self.min_word_duration_ms / 1000.0
        max_duration_s = self.max_word_duration_ms / 1000.0
        
        # Calculate base duration per chunk
        if len(chunks) == 1:
            # Single chunk gets full duration
            chunk_duration = min(max_duration_s, duration_s)
            chunk_start = start_time
            chunk_end = start_time + chunk_duration
        else:
            # Multiple chunks with overlap
            # Total time available for chunks (including overlaps)
            total_available_time = duration_s + (len(chunks) - 1) * overlap_s
            base_duration_per_chunk = total_available_time / len(chunks)
            
            # Ensure duration is within bounds
            base_duration_per_chunk = max(min_duration_s, min(max_duration_s, base_duration_per_chunk))
            
            current_start = start_time
            for i, chunk_text in enumerate(chunks):
                chunk_start = current_start
                chunk_end = chunk_start + base_duration_per_chunk
                
                # Ensure last chunk doesn't exceed original end time by too much
                if i == len(chunks) - 1:
                    chunk_end = min(chunk_end, end_time + 0.5)  # Allow 500ms overshoot for last chunk
                
                segments.append(SubtitleSegment(
                    start_time=chunk_start,
                    end_time=chunk_end,
                    text=chunk_text
                ))
                
                # Next chunk starts with overlap
                current_start = chunk_start + base_duration_per_chunk - overlap_s
        
        # Handle single chunk case
        if len(chunks) == 1:
            segments.append(SubtitleSegment(
                start_time=chunk_start,
                end_time=chunk_end,
                text=chunks[0]
            ))
        
        logger.debug(f"CapCut timing: {len(segments)} segments with {overlap_s*1000:.0f}ms overlaps")
        
        return segments
    
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
        """Format time in a simple readable format with milliseconds for CapCut mode.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (MM:SS.mmm for CapCut, MM:SS for traditional)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        if self.capcut_mode:
            # Show milliseconds for CapCut mode
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
            else:
                return f"{minutes:02d}:{secs:02d}.{millisecs:03d}"
        else:
            # Traditional format without milliseconds
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
            
            # Process segments based on mode
            final_segments = []
            
            if self.capcut_mode:
                # CapCut-style: Create punch word chunks with overlapping timing
                logger.info(f"🎬 Processing in CapCut mode: creating punch-word chunks")
                
                for segment in merged_segments:
                    word_chunks = self._create_capcut_word_chunks(
                        text=segment.text,
                        start_time=segment.start_time,
                        end_time=segment.end_time
                    )
                    final_segments.extend(word_chunks)
                    
                    logger.debug(f"CapCut segment '{segment.text[:30]}...' -> {len(word_chunks)} word chunks")
                
            else:
                # Legacy mode: Traditional text wrapping with splitting
                logger.info(f"📝 Processing in Legacy mode: traditional subtitle wrapping")
                
                for segment in merged_segments:
                    current_text = segment.text
                    current_start = segment.start_time
                    segment_duration = segment.end_time - segment.start_time
                    
                    # Keep processing until all text is handled
                    iteration = 0
                    while current_text.strip():
                        iteration += 1
                        wrapped_lines, remaining_text = self._wrap_text(current_text)
                        
                        if wrapped_lines:
                            # Calculate duration for this sub-segment
                            if remaining_text.strip():
                                # If there's remaining text, this is a partial segment
                                # Estimate duration based on character proportion
                                chars_used = sum(len(line) for line in wrapped_lines)
                                total_chars = len(segment.text)
                                duration_fraction = chars_used / total_chars if total_chars > 0 else 1.0
                                sub_duration = segment_duration * duration_fraction
                                current_end = current_start + sub_duration
                                
                                logger.debug(f"Split segment {iteration}: '{' '.join(wrapped_lines)[:50]}...' + remaining: '{remaining_text[:30]}...'")
                            else:
                                # Last segment gets remaining time
                                current_end = segment.end_time
                            
                            final_segments.append(SubtitleSegment(
                                start_time=current_start,
                                end_time=current_end,
                                text="\n".join(wrapped_lines)
                            ))
                            
                            # Update for next iteration
                            current_text = remaining_text
                            current_start = current_end
                        else:
                            # No lines could be wrapped (shouldn't happen)
                            logger.warning(f"Could not wrap text: '{current_text[:50]}...'")
                            break
                        
                        # Safety check to prevent infinite loops
                        if iteration > 10:
                            logger.warning(f"Text wrapping reached maximum iterations for segment, truncating remaining: '{current_text[:50]}...'")
                            break
            
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
    max_chars_per_line: int = 50,  # Legacy parameter
    max_lines: int = 2,  # Legacy parameter
    merge_gap_threshold_ms: int = 200,
    capcut_mode: bool = True,  # Enable CapCut-style punch words by default
    min_word_duration_ms: int = 600,  # CapCut: Min display time per word chunk
    max_word_duration_ms: int = 1200,  # CapCut: Max display time per word chunk
    word_overlap_ms: int = 200  # CapCut: Overlap between word chunks
) -> Tuple[str, str]:
    """Convenience function to convert Groq segments to subtitle files.
    
    Args:
        groq_segments: Groq transcription segments
        output_dir: Output directory path
        filename_base: Base filename without extension
        max_chars_per_line: Maximum characters per subtitle line (legacy mode only)
        max_lines: Maximum number of lines per subtitle (legacy mode only)
        merge_gap_threshold_ms: Merge segments with gaps smaller than this (ms)
        capcut_mode: Enable CapCut-style 1-3 word punch subtitles with overlaps
        min_word_duration_ms: Minimum display time per word chunk (CapCut mode)
        max_word_duration_ms: Maximum display time per word chunk (CapCut mode)
        word_overlap_ms: Overlap between word chunks for smooth transitions (CapCut mode)
        
    Returns:
        Tuple of (srt_path, vtt_path)
    """
    processor = SubtitleProcessor(
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        merge_gap_threshold_ms=merge_gap_threshold_ms,
        capcut_mode=capcut_mode,
        min_word_duration_ms=min_word_duration_ms,
        max_word_duration_ms=max_word_duration_ms,
        word_overlap_ms=word_overlap_ms
    )
    segments = processor.process_segments(groq_segments)
    return processor.save_subtitles(segments, output_dir, filename_base) 