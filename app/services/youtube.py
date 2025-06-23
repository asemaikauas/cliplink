import yt_dlp
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from moviepy import VideoFileClip

# Import vertical cropping service
try:
    from .vertical_crop import crop_video_to_vertical, get_available_resolutions
    VERTICAL_CROP_AVAILABLE = True
    print("✅ Vertical cropping service loaded")
except ImportError as e:
    VERTICAL_CROP_AVAILABLE = False
    print(f"⚠️ Vertical cropping not available: {e}")

# Configure MoviePy to use the system ffmpeg if available
try:
    import imageio_ffmpeg
    # Try to set ffmpeg path explicitly
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg_path:
        os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
        print(f"🎬 Using ffmpeg: {ffmpeg_path}")
    else:
        print("⚠️ Could not find ffmpeg via imageio-ffmpeg, trying system PATH")
except Exception as e:
    print(f"⚠️ FFmpeg configuration warning: {e}")
    print("🔧 MoviePy will try to use system ffmpeg")

def create_clip_with_direct_ffmpeg(video_path: Path, start: float, end: float, output_path: Path) -> bool:
    """
    Fallback function to create clips using direct ffmpeg calls with proper error handling.
    This addresses the 'NoneType' object has no attribute 'stdout' issue.
    """
    try:
        cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error',
            '-ss', str(start),
            '-i', str(video_path),
            '-t', str(end - start),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            str(output_path),
            '-y'  # Overwrite output file
        ]
        
        # Use subprocess.run with proper error handling (as suggested in the original query)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ FFmpeg stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg timeout after 300 seconds")
        return False
    except Exception as e:
        print(f"❌ FFmpeg exception: {str(e)}")
        return False

class DownloadError(Exception):
    """Custom exception for video download errors"""
    pass

class YouTubeService:
    """
    Enhanced YouTube service with high-quality video download support including 8K
    """
    def __init__(self, downloads_dir: str = "downloads"):
        self.downloads_dir = Path(downloads_dir)
        self.downloads_dir.mkdir(exist_ok=True)
        
        self.base_ydl_opts = {
            'writeinfojson': False,
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'prefer_ffmpeg': True,
            'audioformat': 'best',
            'audioquality': '0',  # Best audio quality
            'ignoreerrors': False,
            'no_warnings': False,
            'extractflat': False,
            'http_chunk_size': 10485760,  # 10MB chunks for large files
            'retries': 10,
            'fragment_retries': 10,
        }

    def get_video_info(self, url: str) -> Dict:
        """
        Extract detailed video information from YouTube URL
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'upload_date': info.get('upload_date'),
                    'uploader': info.get('uploader'),
                    'description': info.get('description'),
                    'is_live': info.get('is_live', False),
                    'availability': info.get('availability')
                }
        except Exception as e:
            raise DownloadError(f"Failed to extract video info: {str(e)}")

    def get_available_formats(self, url: str) -> List[Dict]:
        """
        Get list of available video formats sorted by quality
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                video_formats = []
                for fmt in formats:
                    if fmt.get('vcodec') != 'none':  # Only video formats
                        video_formats.append({
                            'format_id': fmt.get('format_id'),
                            'ext': fmt.get('ext'),
                            'resolution': fmt.get('resolution', 'unknown'),
                            'height': fmt.get('height', 0),
                            'width': fmt.get('width', 0),
                            'fps': fmt.get('fps', 0),
                            'filesize': fmt.get('filesize', 0),
                            'vcodec': fmt.get('vcodec'),
                            'acodec': fmt.get('acodec'),
                            'format_note': fmt.get('format_note', '')
                        })
                
                # Sort by height (quality)
                video_formats.sort(key=lambda x: x['height'] or 0, reverse=True)
                return video_formats
                
        except Exception as e:
            raise DownloadError(f"Failed to get available formats: {str(e)}")

    def download_ultra_hq_video(self, url: str, max_height: Optional[int] = None, 
                                force_8k: bool = False) -> Path:
        """
        Download video in ultra high quality (including 8K support)
        
        Args:
            url: YouTube video URL
            max_height: Maximum video height (e.g., 1080, 1440, 2160, 4320)
            force_8k: Force 8K download if available
        """
        # Get video info first
        info = self.get_video_info(url)
        video_id = info['id']
        title = info['title']
        
        if info['is_live']:
            raise DownloadError("Cannot download live streams")
        
        print(f"📺 Скачиваю: {title}")
        print(f"🆔 ID: {video_id}")
        
        # Configure format selector based on requirements
        if force_8k:
            print("🎯 Режим 8K активирован!")
            format_selector = (
                'bestvideo[height>=4320]+bestaudio[ext=m4a]/'  # 8K priority
                'bestvideo[height>=2160]+bestaudio[ext=m4a]/'  # 4K fallback
                'bestvideo[filesize>500M]+bestaudio[ext=m4a]/'  # Large files
                'bestvideo+bestaudio'
            )
            outtmpl = str(self.downloads_dir / '8K-%(title)s-%(id)s.%(ext)s')
            
        elif max_height:
            print(f"🎯 Максимальное разрешение: {max_height}p")
            format_selector = (
                f'bestvideo[height<={max_height}]+bestaudio[ext=m4a]/'
                f'bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/'
                f'best[height<={max_height}]'
            )
            outtmpl = str(self.downloads_dir / f'{max_height}p-%(title)s-%(id)s.%(ext)s')
            
        else:
            print("🎯 Максимальное доступное качество")
            format_selector = (
                'bestvideo[height>=2160]+bestaudio[ext=m4a]/'  # 4K+ priority
                'bestvideo+bestaudio[ext=m4a]/'  # Best video + m4a audio
                'bestvideo[ext=mp4]+bestaudio[ext=m4a]/'  # MP4 video + m4a audio
                'bestvideo+bestaudio/'  # Best video + best audio
                'best[height>=2160]/'  # Best 4K+
                'best'  # Fallback
            )
            outtmpl = str(self.downloads_dir / 'UHQ-%(title)s-%(id)s.%(ext)s')
        
        # Setup download options
        ydl_opts = self.base_ydl_opts.copy()
        ydl_opts.update({
            'format': format_selector,
            'outtmpl': outtmpl,
            'overwrites': True
        })
        
        return self._download_with_options(url, ydl_opts, video_id)

    def download_video(self, url: str, quality: str = "best") -> Path:
        """
        Main download function with quality options
        
        Args:
            url: YouTube video URL
            quality: "best", "8k", "4k", "1440p", "1080p", "720p"
        """
        quality_map = {
            "8k": {"force_8k": True},
            "4k": {"max_height": 2160},
            "1440p": {"max_height": 1440},
            "1080p": {"max_height": 1080},
            "720p": {"max_height": 720},
            "best": {}
        }
        
        if quality not in quality_map:
            quality = "best"
        
        return self.download_ultra_hq_video(url, **quality_map[quality])

    def _download_with_options(self, url: str, opts: Dict, video_id: str) -> Path:
        """
        Download video with specified options
        """
        try:
            # Check if file already exists to prevent duplicate downloads
            existing_file = self._find_downloaded_file(video_id)
            if existing_file and existing_file.exists():
                print(f"✅ Файл уже существует: {existing_file.name}")
                return existing_file.absolute()
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                print(f"📥 Начинаю скачивание...")
                
                # Get format info before download
                info = ydl.extract_info(url, download=False)
                
                # Show selected format info based on format selector
                if info and 'formats' in info:
                    selected_format = info.get('format', 'unknown')
                    resolution = info.get('resolution', 'unknown')
                    print(f"🎯 Выбранный формат: {resolution} ({selected_format})")
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                video_file = self._find_downloaded_file(video_id)
                
                if not video_file or not video_file.exists():
                    raise DownloadError(f"Downloaded file not found for video {video_id}")
                
                file_size_mb = video_file.stat().st_size / (1024*1024)
                print(f"✅ Видео скачано: {video_file.name}")
                print(f"📁 Размер файла: {file_size_mb:.1f} MB")
                
                return video_file.absolute()
                
        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                raise DownloadError(f"Video unavailable: {url}")
            elif "Private video" in error_msg:
                raise DownloadError(f"Private video cannot be downloaded: {url}")
            elif "copyright" in error_msg.lower():
                raise DownloadError(f"Video is copyright protected: {url}")
            else:
                raise DownloadError(f"Failed to download video: {error_msg}")
        except Exception as e:
            raise DownloadError(f"Unexpected error downloading video: {str(e)}")

    def _find_downloaded_file(self, video_id: str) -> Optional[Path]:
        """
        Find downloaded file by video ID
        """
        possible_extensions = ['.mp4', '.webm', '.mkv', '.m4v', '.avi']
        
        for ext in possible_extensions:
            pattern = f"*{video_id}*{ext}"
            files = list(self.downloads_dir.glob(pattern))
            if files:
                return files[0]
        return None

youtube_service = YouTubeService()

# Backward compatibility functions for existing FastAPI endpoints
def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    info = youtube_service.get_video_info(url)
    return info['id']

def download_video(url: str, quality: str = "best") -> Path:
    """
    Download video with specified quality
    Supported qualities: best, 8k, 4k, 1440p, 1080p, 720p
    """
    return youtube_service.download_video(url, quality)

def get_video_info(url: str) -> Dict:
    """Get detailed video information"""
    return youtube_service.get_video_info(url)

def get_available_formats(url: str) -> List[Dict]:
    """Get available video formats"""
    return youtube_service.get_available_formats(url)

def cut_clips(video_path: Path, analysis: Dict) -> List[Path]:
    """
    Cut clips from video based on Gemini analysis using MoviePy 2.2.1 API.
    analysis comes from gemini.py and has structure:
    {
      "gemini_analysis": {
        "viral_segments": [
          {
            "title": "...",
            "start": 85,
            "end": 129,
            "duration": 44,
            "subtitles": {...}
          }
        ]
      }
    }
    """
    clips_dir = Path("clips")
    clips_dir.mkdir(exist_ok=True)
    
    # Extract viral segments from analysis
    gemini_analysis = analysis.get("gemini_analysis", {})
    viral_segments = gemini_analysis.get("viral_segments", [])
    
    if not viral_segments:
        logging.warning("No viral segments found in analysis")
        return []
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"✂️ Нарезаю {len(viral_segments)} клипов из {video_path.name}")
    
    # Load video with MoviePy 2.2.1
    try:
        video = VideoFileClip(str(video_path))
        video_duration = video.duration
        print(f"📹 Длительность видео: {video_duration:.1f} сек")
    except Exception as e:
        raise Exception(f"Failed to load video with MoviePy: {str(e)}")
    
    created_clips = []
    skipped_clips = []
    failed_clips = []
    
    for i, segment in enumerate(viral_segments):
        clip_id = i + 1
        title = segment.get('title', f'Segment_{clip_id}')
        start = segment.get('start', 0)
        end = segment.get('end', start + 60)  # Default 60 sec if no end
        
        print(f"\n--- Клип {clip_id}: {title} ---")
        print(f"⏰ Время: {start} - {end} сек ({end - start} сек)")
        
        # Enhanced validation with detailed logging
        if start >= end:
            reason = f"Invalid timing: start ({start}) >= end ({end})"
            print(f"⚠️ ПРОПУСКАЮ клип {clip_id}: {reason}")
            skipped_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
            continue
            
        if start < 0:
            reason = f"Invalid start time: {start} < 0"
            print(f"⚠️ ПРОПУСКАЮ клип {clip_id}: {reason}")
            skipped_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
            continue
            
        if end > video_duration:
            # Adjust end time instead of skipping
            original_end = end
            end = video_duration
            print(f"⚠️ КОРРЕКТИРУЮ клип {clip_id}: end {original_end} -> {end} (video duration)")
        
        try:
            # Create safe filename
            safe_title = _sanitize_filename(title)
            clip_filename = f"{clip_id:02d}_{safe_title}.mp4"
            clip_path = clips_dir / clip_filename
            
            print(f"📁 Сохраняю как: {clip_filename}")
            
            # Cut segment with MoviePy 2.2.1 API
            print(f"✂️ Нарезаю сегмент {start:.1f}-{end:.1f}...")
            try:
                segment_clip = video.subclipped(start, end)
            except Exception as e:
                if "'NoneType' object has no attribute 'stdout'" in str(e):
                    print(f"⚠️ MoviePy subclip failed for clip {clip_id}, trying direct ffmpeg fallback...")
                    
                    # Try direct ffmpeg approach without MoviePy
                    if create_clip_with_direct_ffmpeg(video_path, start, end, clip_path):
                        if clip_path.exists():
                            file_size = clip_path.stat().st_size
                            if file_size > 0:
                                file_size_mb = file_size / (1024*1024)
                                print(f"✅ Клип создан (via direct ffmpeg): {clip_path.name} ({file_size_mb:.1f} MB)")
                                created_clips.append(clip_path.absolute())
                                continue
                            else:
                                clip_path.unlink()  # Delete empty file
                    
                    reason = f"Both MoviePy subclip and direct ffmpeg failed: {str(e)}"
                    print(f"❌ ОШИБКА клип {clip_id}: {reason}")
                    failed_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
                    continue
                else:
                    raise e
            
            # Write video file with enhanced error handling
            print(f"💾 Записываю видеофайл...")
            try:
                segment_clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    logger=None
                )
            except Exception as e:
                if "'NoneType' object has no attribute 'stdout'" in str(e):
                    print(f"⚠️ MoviePy failed for clip {clip_id}, trying direct ffmpeg fallback...")
                    segment_clip.close()
                    
                    # Try direct ffmpeg approach
                    if create_clip_with_direct_ffmpeg(video_path, start, end, clip_path):
                        if clip_path.exists():
                            file_size = clip_path.stat().st_size
                            if file_size > 0:
                                file_size_mb = file_size / (1024*1024)
                                print(f"✅ Клип создан (via direct ffmpeg): {clip_path.name} ({file_size_mb:.1f} MB)")
                                created_clips.append(clip_path.absolute())
                                continue
                            else:
                                clip_path.unlink()  # Delete empty file
                    
                    reason = f"Both MoviePy and direct ffmpeg failed: {str(e)}"
                    print(f"❌ ОШИБКА клип {clip_id}: {reason}")
                    failed_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
                    continue
                else:
                    raise e
            
            # Close clip to free memory
            segment_clip.close()
            
            # Verify file creation and size
            if clip_path.exists():
                file_size = clip_path.stat().st_size
                if file_size == 0:
                    reason = "Generated file is empty (0 bytes)"
                    print(f"❌ ОШИБКА клип {clip_id}: {reason}")
                    clip_path.unlink()  # Delete empty file
                    failed_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
                else:
                    file_size_mb = file_size / (1024*1024)
                    print(f"✅ Клип создан: {clip_path.name} ({file_size_mb:.1f} MB)")
                    created_clips.append(clip_path.absolute())
            else:
                reason = "File was not created by MoviePy"
                print(f"❌ ОШИБКА клип {clip_id}: {reason}")
                failed_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
                
        except Exception as e:
            reason = f"Exception during processing: {str(e)}"
            print(f"❌ ОШИБКА клип {clip_id}: {reason}")
            failed_clips.append({"clip_id": clip_id, "title": title, "reason": reason})
            
            # Try to clean up any partial files
            if 'clip_path' in locals() and clip_path.exists():
                try:
                    clip_path.unlink()
                    print(f"🧹 Удален частичный файл: {clip_path.name}")
                except:
                    pass
            continue
    
    # Close main video
    video.close()
    
    # Detailed summary
    print(f"\n🎉 РЕЗУЛЬТАТ НАРЕЗКИ:")
    print(f"✅ Создано клипов: {len(created_clips)}")
    print(f"⚠️ Пропущено клипов: {len(skipped_clips)}")
    print(f"❌ Ошибок при создании: {len(failed_clips)}")
    print(f"📊 Общий итог: {len(created_clips)}/{len(viral_segments)}")
    
    # Log details of skipped/failed clips
    if skipped_clips:
        print(f"\n⚠️ ПРОПУЩЕННЫЕ КЛИПЫ:")
        for skip in skipped_clips:
            print(f"  - Клип {skip['clip_id']} ({skip['title']}): {skip['reason']}")
    
    if failed_clips:
        print(f"\n❌ НЕУДАЧНЫЕ КЛИПЫ:")
        for fail in failed_clips:
            print(f"  - Клип {fail['clip_id']} ({fail['title']}): {fail['reason']}")
    
    return created_clips


def cut_clips_vertical(video_path: Path, analysis: Dict, resolution: str = "shorts_hd", smoothing_strength: str = "medium") -> List[Path]:
    """
    Cut clips from video and convert them to vertical format (9:16) for YouTube Shorts.
    This combines the clip cutting functionality with vertical cropping.
    Enhanced with motion smoothing to prevent jerky head movements.
    
    Args:
        video_path: Path to the source video
        analysis: Gemini analysis with viral segments
        resolution: Target resolution ("shorts_hd", "shorts_fhd", "tiktok")
        smoothing_strength: Motion smoothing level ("low", "medium", "high")
    
    Returns:
        List of paths to created vertical clips
    """
    if not VERTICAL_CROP_AVAILABLE:
        print("❌ Vertical cropping not available. Using standard clipping.")
        return cut_clips(video_path, analysis)
    
    clips_dir = Path("clips")
    clips_dir.mkdir(exist_ok=True)
    
    # Create vertical clips directory
    vertical_clips_dir = clips_dir / "vertical"
    vertical_clips_dir.mkdir(exist_ok=True)
    
    # Extract viral segments from analysis
    gemini_analysis = analysis.get("gemini_analysis", {})
    viral_segments = gemini_analysis.get("viral_segments", [])
    
    if not viral_segments:
        logging.warning("No viral segments found in analysis")
        return []
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"🎬 Creating {len(viral_segments)} vertical clips from {video_path.name}")
    print(f"📱 Target resolution: {resolution}")
    
    created_clips = []
    skipped_clips = []
    failed_clips = []
    
    # Get resolution info
    resolutions = get_available_resolutions()
    target_size = resolutions.get(resolution, resolutions["shorts_hd"])
    print(f"🎯 Output size: {target_size[0]}x{target_size[1]}")
    
    # First, create regular clips using existing functionality
    print("\n📹 Step 1: Creating horizontal clips...")
    horizontal_clips = cut_clips(video_path, analysis)
    
    print(f"\n🔄 Step 2: Converting {len(horizontal_clips)} clips to vertical format...")
    
    for i, horizontal_clip_path in enumerate(horizontal_clips):
        clip_id = i + 1
        
        try:
            # Get the base filename without extension
            base_name = horizontal_clip_path.stem
            vertical_clip_filename = f"{base_name}_vertical.mp4"
            vertical_clip_path = vertical_clips_dir / vertical_clip_filename
            
            print(f"\n--- Converting Clip {clip_id}: {base_name} ---")
            print(f"📂 Input: {horizontal_clip_path.name}")
            print(f"📱 Output: {vertical_clip_filename}")
            
            # Convert to vertical format
            success = crop_video_to_vertical(
                input_path=horizontal_clip_path,
                output_path=vertical_clip_path,
                resolution=resolution,
                use_speaker_detection=True,
                smoothing_strength=smoothing_strength
            )
            
            if success and vertical_clip_path.exists():
                file_size = vertical_clip_path.stat().st_size
                if file_size > 0:
                    file_size_mb = file_size / (1024*1024)
                    print(f"✅ Vertical clip created: {vertical_clip_filename} ({file_size_mb:.1f} MB)")
                    created_clips.append(vertical_clip_path.absolute())
                    
                    # Optionally remove horizontal clip to save space
                    # horizontal_clip_path.unlink()
                    
                else:
                    reason = "Generated vertical file is empty (0 bytes)"
                    print(f"❌ ОШИБКА clip {clip_id}: {reason}")
                    vertical_clip_path.unlink()  # Delete empty file
                    failed_clips.append({"clip_id": clip_id, "title": base_name, "reason": reason})
            else:
                reason = "Vertical cropping failed"
                print(f"❌ ОШИБКА clip {clip_id}: {reason}")
                failed_clips.append({"clip_id": clip_id, "title": base_name, "reason": reason})
                
        except Exception as e:
            reason = f"Exception during vertical conversion: {str(e)}"
            print(f"❌ ОШИБКА clip {clip_id}: {reason}")
            failed_clips.append({"clip_id": clip_id, "title": "unknown", "reason": reason})
            continue
    
    # Detailed summary
    print(f"\n🎉 VERTICAL CONVERSION RESULTS:")
    print(f"✅ Created vertical clips: {len(created_clips)}")
    print(f"📊 Success rate: {len(created_clips)}/{len(horizontal_clips)}")
    print(f"🎯 Resolution: {target_size[0]}x{target_size[1]} ({resolution})")
    
    if failed_clips:
        print(f"\n❌ FAILED CONVERSIONS:")
        for fail in failed_clips:
            print(f"  - Clip {fail['clip_id']} ({fail['title']}): {fail['reason']}")
    
    print(f"\n📁 Vertical clips saved to: {vertical_clips_dir}")
    
    return created_clips


def check_video_quality(video_path: Path) -> Dict:
    """
    Быстрая проверка качества скачанного видео
    Возвращает основную информацию о качестве
    """
    try:
        from moviepy import VideoFileClip
        
        if not video_path.exists():
            return {"error": f"Файл не найден: {video_path}"}
        
        # Получить базовую информацию
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        # Попробовать загрузить видео с MoviePy
        try:
            with VideoFileClip(str(video_path)) as video:
                width, height = video.size
                duration = video.duration
                fps = video.fps
                
                # Определить качество
                if height >= 4320:
                    quality_level = "8K Ultra"
                elif height >= 2160:
                    quality_level = "4K Very High"
                elif height >= 1440:
                    quality_level = "2K High"
                elif height >= 1080:
                    quality_level = "Full HD"
                elif height >= 720:
                    quality_level = "HD"
                else:
                    quality_level = "Low"
                
                return {
                    "file_name": video_path.name,
                    "file_size_mb": round(file_size_mb, 1),
                    "resolution": f"{width}x{height}",
                    "quality_level": quality_level,
                    "duration_minutes": round(duration / 60, 1),
                    "fps": round(fps, 1) if fps else "N/A",
                    "mb_per_minute": round(file_size_mb / (duration / 60), 1) if duration > 0 else 0,
                    "status": "success"
                }
        except Exception as e:
            return {
                "error": f"Ошибка при анализе видео: {str(e)}",
                "file_name": video_path.name,
                "file_size_mb": round(file_size_mb, 1)
            }
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

def _sanitize_filename(filename: str) -> str:
    """
    Create a safe filename by removing invalid characters
    Keeps only letters, numbers, hyphens, and underscores
    """
    import re
    
    # Remove emojis and special Unicode characters first
    filename = re.sub(r'[^\w\s\-\.]', '', filename, flags=re.UNICODE)
    
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    
    # Keep only alphanumeric, hyphens, underscores, and dots
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > 50:
        filename = filename[:50].rstrip('_')
    
    # Ensure filename is not empty
    if not filename:
        filename = "clip"
    
    return filename 