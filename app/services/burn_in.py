"""Subtitle burn-in renderer using FFmpeg."""

import os
import subprocess
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from app.exceptions import BurnInError


logger = logging.getLogger(__name__)


class BurnInRenderer:
    """FFmpeg-based subtitle burn-in renderer."""
    
    def __init__(self):
        """Initialize burn-in renderer."""
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise BurnInError("FFmpeg not found or not working properly")
            logger.info("FFmpeg verified successfully")
        except FileNotFoundError:
            raise BurnInError("FFmpeg not found. Please install FFmpeg.")
        except subprocess.TimeoutExpired:
            raise BurnInError("FFmpeg verification timed out")
        except Exception as e:
            raise BurnInError(f"FFmpeg verification failed: {str(e)}")
    
    def _build_force_style(
        self,
        font_size_pct: float = 4.5,
        font_name: str = "Inter SemiBold",
        primary_colour: str = "&Hffffff&",
        back_colour: str = "&H66000000&",
        alignment: int = 2,
        margin_v: int = 40
    ) -> str:
        """Build ASS subtitle force_style string.
        
        Args:
            font_size_pct: Font size as percentage of video height
            font_name: Font family name
            primary_colour: Primary text color in ASS format
            back_colour: Background color in ASS format
            alignment: Text alignment (2 = bottom center)
            margin_v: Bottom margin in pixels
            
        Returns:
            Force style string for FFmpeg
        """
        # Convert percentage to dynamic font size calculation
        # We'll use a filter to calculate font size based on video height
        font_size_expr = f"h*{font_size_pct/100}"
        
        force_style = (
            f"Fontname={font_name},"
            f"Fontsize={font_size_expr},"
            f"PrimaryColour={primary_colour},"
            f"BackColour={back_colour},"
            f"Alignment={alignment},"
            f"MarginV={margin_v}"
        )
        
        return force_style
    
    def burn_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font_size_pct: float = 4.5,
        export_codec: str = "h264",
        crf: int = 18,
        task_id: Optional[str] = None
    ) -> str:
        """Burn subtitles into video using FFmpeg.
        
        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path for output video file
            font_size_pct: Font size as percentage of video height
            export_codec: Video codec (h264, h265, etc.)
            crf: Constant Rate Factor for video quality (lower = higher quality)
            task_id: Task ID for logging
            
        Returns:
            Path to the output video file
            
        Raises:
            BurnInError: If burn-in process fails
        """
        try:
            logger.info(f"Starting subtitle burn-in (task_id: {task_id})")
            logger.info(f"Input video: {video_path}")
            logger.info(f"SRT file: {srt_path}")
            logger.info(f"Output: {output_path}")
            
            # Verify input files exist
            if not os.path.exists(video_path):
                raise BurnInError(f"Input video not found: {video_path}")
            if not os.path.exists(srt_path):
                raise BurnInError(f"SRT file not found: {srt_path}")
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Build force style for subtitles
            force_style = self._build_force_style(font_size_pct=font_size_pct)
            
            # Build FFmpeg command
            # Use libx264 for h264, libx265 for h265
            video_codec = "libx264" if export_codec == "h264" else f"lib{export_codec}"
            
            # Escape the SRT path for FFmpeg (handle spaces and special characters)
            escaped_srt_path = srt_path.replace("\\", "\\\\").replace(":", "\\:")
            
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"subtitles='{escaped_srt_path}':force_style='{force_style}'",
                "-c:v", video_codec,
                "-crf", str(crf),
                "-c:a", "copy",  # Copy audio without re-encoding
                "-y",  # Overwrite output file
                output_path
            ]
            
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Execute FFmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for long videos
            )
            
            if result.returncode != 0:
                error_msg = f"FFmpeg failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                raise BurnInError(error_msg)
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise BurnInError("Output file was not created")
            
            output_size = os.path.getsize(output_path)
            logger.info(
                f"Subtitle burn-in completed (task_id: {task_id}) - "
                f"output: {output_path} ({output_size / (1024*1024):.1f} MB)"
            )
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise BurnInError("FFmpeg process timed out (>1 hour)")
        except Exception as e:
            logger.error(f"Subtitle burn-in failed (task_id: {task_id}): {str(e)}")
            raise BurnInError(f"Subtitle burn-in failed: {str(e)}", task_id=task_id)
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
            
        Raises:
            BurnInError: If getting video info fails
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise BurnInError(f"FFprobe failed: {result.stderr}")
            
            import json
            return json.loads(result.stdout)
            
        except Exception as e:
            raise BurnInError(f"Failed to get video info: {str(e)}")


def burn_subtitles_to_video(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size_pct: float = 4.5,
    export_codec: str = "h264",
    crf: int = 18,
    task_id: Optional[str] = None
) -> str:
    """Convenience function to burn subtitles into video.
    
    Args:
        video_path: Path to input video file
        srt_path: Path to SRT subtitle file
        output_path: Path for output video file
        font_size_pct: Font size as percentage of video height
        export_codec: Video codec (h264, h265, etc.)
        crf: Constant Rate Factor for video quality
        task_id: Task ID for logging
        
    Returns:
        Path to the output video file
    """
    renderer = BurnInRenderer()
    return renderer.burn_subtitles(
        video_path, srt_path, output_path, font_size_pct, export_codec, crf, task_id
    ) 