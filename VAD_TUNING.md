# VAD (Voice Activity Detection) Tuning Guide

## Problem: Subtitles Being Cut Off

If you're experiencing missing or cut-off subtitles, there are two main causes:

1. **VAD Filtering**: Voice Activity Detection removes "silent" segments from audio before transcription, but sometimes it incorrectly removes speech.
2. **Text Truncation**: The subtitle formatter was truncating long sentences instead of splitting them into multiple segments. **This has been fixed!**

## Quick Fixes

### 1. Disable VAD (Easiest)
When calling the subtitle API, set `disable_vad=true`:
```bash
curl -X POST "http://localhost:8000/subtitles" \
  -F "video_file=@your_video.mp4" \
  -F "disable_vad=true"
```

### 2. Tune VAD Parameters (Recommended)
Set environment variables to make VAD less aggressive:

```bash
# More conservative silence threshold (quieter sounds won't be removed)
export VAD_SILENCE_THRESHOLD=-60  # Default: -55dB (lower = more conservative)

# Longer silence duration required before removal
export VAD_MIN_SILENCE_DURATION=7000  # Default: 5000ms (higher = more conservative)
```

## Understanding VAD Parameters

### `VAD_SILENCE_THRESHOLD`
- **What it does**: Audio below this volume (in dB) is considered "silence"
- **Default**: -55dB
- **More conservative**: -60dB to -65dB (keeps quieter speech)
- **More aggressive**: -45dB to -40dB (removes more content)

### `VAD_MIN_SILENCE_DURATION`
- **What it does**: How long silence must last before it's removed (in milliseconds)
- **Default**: 5000ms (5 seconds)
- **More conservative**: 7000-10000ms (only removes longer pauses)
- **More aggressive**: 2000-3000ms (removes shorter pauses)

## Automatic Retry Logic

The system now automatically retries transcription without VAD if:
1. No segments are found with VAD enabled
2. Very few segments (< 3) are found with VAD enabled

## Monitoring VAD Impact

Check the logs for VAD warnings:
```
VAD will remove 45.2s of silence from 180.0s audio (25.1%)
⚠️ VAD is removing >50% of audio content! Consider disabling VAD or adjusting parameters.
```

If VAD is removing >30% of your audio, consider:
1. Disabling VAD entirely
2. Using more conservative parameters
3. Checking if your audio has very quiet speech

## Environment Variables Template

Create a `.env` file in your backend directory:
```bash
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# VAD Tuning (adjust these to fix cut-off subtitles)
VAD_SILENCE_THRESHOLD=-60    # More conservative than default -55
VAD_MIN_SILENCE_DURATION=7000  # More conservative than default 5000

# Subtitle Formatting (NEW: prevent text truncation)
SUBTITLE_MAX_CHARS_PER_LINE=50   # Default: 50 (increased from 42)
SUBTITLE_MAX_LINES=2             # Default: 2 lines per subtitle
SUBTITLE_MERGE_GAP_MS=200        # Default: 200ms gap merging

# Example: Very conservative settings for quiet speech
# VAD_SILENCE_THRESHOLD=-65
# VAD_MIN_SILENCE_DURATION=10000

# Example: Longer subtitles for dense content
# SUBTITLE_MAX_CHARS_PER_LINE=60
# SUBTITLE_MAX_LINES=3

# Example: Disable VAD for continuous speech (podcasts, etc.)
# Just use disable_vad=true in API calls instead
```

## When to Use Each Approach

| Content Type | Recommendation |
|--------------|----------------|
| **Podcasts/Interviews** | Disable VAD (`disable_vad=true`) |
| **Lectures** | Conservative VAD (`-60dB`, `7000ms`) |
| **Music with vocals** | Disable VAD |
| **Noisy environments** | Standard or aggressive VAD |
| **Clean studio audio** | Standard VAD settings |

## Testing Your Settings

1. Process a sample video with your settings
2. Check the logs for VAD removal percentage
3. Verify all expected subtitles are present
4. Adjust parameters if needed

## Troubleshooting

**Still missing subtitles?**
- Try `disable_vad=true` first
- Check if audio quality is very poor
- Verify the speech is actually audible to humans

**Subtitles cutting off mid-sentence?**
- **This should now be fixed!** Long text is split into multiple segments instead of truncated
- If still happening, increase `SUBTITLE_MAX_CHARS_PER_LINE` or `SUBTITLE_MAX_LINES`
- Run the test: `python backend/test_subtitle_fix.py` to verify the fix

**Too much background noise in transcription?**
- Use more aggressive VAD settings
- Consider audio preprocessing

**Processing too slow?**
- VAD speeds up transcription by removing silence
- Balance between speed and accuracy based on your content

**Want to test the fix?**
```bash
cd backend
python test_subtitle_fix.py
``` 