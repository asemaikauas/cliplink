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

# Subtitle Formatting
SUBTITLE_MAX_CHARS_PER_LINE=50   # Default: 50 (legacy mode only)
SUBTITLE_MAX_LINES=2             # Default: 2 lines per subtitle (legacy mode only)
SUBTITLE_MERGE_GAP_MS=200        # Default: 200ms gap merging

# CapCut-Style Punch Words (NEW: default mode)
SUBTITLE_CAPCUT_MODE=true        # Enable CapCut-style 1-3 word chunks
CAPCUT_MIN_WORD_DURATION_MS=800  # Minimum display time per word chunk (800ms = better readability)
CAPCUT_MAX_WORD_DURATION_MS=1500 # Maximum display time per word chunk (1500ms = smoother flow)
CAPCUT_WORD_OVERLAP_MS=150       # Overlap between chunks for smooth flow (150ms = cleaner transitions)

# Example: Very conservative settings for quiet speech
# VAD_SILENCE_THRESHOLD=-65
# VAD_MIN_SILENCE_DURATION=10000

# Example: Longer subtitles for dense content
# SUBTITLE_MAX_CHARS_PER_LINE=60
# SUBTITLE_MAX_LINES=3

# Example: Disable VAD for continuous speech (podcasts, etc.)
# Just use disable_vad=true in API calls instead
```

## CapCut vs Traditional Subtitle Modes

### **CapCut Mode (Default)** - Punch Words
```
[1] 00:00.000 --> 00:00.850  "you know an"
[2] 00:00.700 --> 00:01.600  "agent, a chat"  
[3] 00:01.450 --> 00:02.450  "tool on the"
[4] 00:02.300 --> 00:03.300  "side to say"
[5] 00:03.150 --> 00:04.150  "hey, you know"
```
✅ **Perfect for**: Social media, short clips, viral content  
✅ **Features**: Millisecond precision, 2-3 words per chunk, overlapping timing  

### **Traditional Mode** - Full Sentences  
```
[1] 00:00 --> 00:04  "you know, an agent, a chat tool on the side to say, hey, you know, this is how you can learn"
[2] 00:04 --> 00:07  "coding. This is kind of how you can fix your bugs."
```
✅ **Perfect for**: Long-form content, lectures, documentaries  
✅ **Features**: Complete sentences, no overlaps  

## When to Use Each Approach

| Content Type | VAD Setting | Subtitle Mode |
|--------------|-------------|---------------|
| **TikTok/Shorts** | Disable VAD | CapCut punch words |
| **Podcasts/Interviews** | Disable VAD | Traditional full sentences |
| **Lectures** | Conservative VAD | Traditional full sentences |
| **Music with vocals** | Disable VAD | CapCut punch words |
| **Viral clips** | Disable VAD | CapCut punch words |
| **Documentaries** | Standard VAD | Traditional full sentences |

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