#!/usr/bin/env python3
"""Test script for speech synchronization functionality."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.subs import SubtitleProcessor


def create_mock_word_timestamps():
    """Create mock word timestamps that simulate Groq's response."""
    words = [
        {"word": "Hello", "start": 0.5, "end": 0.8},
        {"word": "world,", "start": 0.9, "end": 1.2},
        {"word": "this", "start": 1.5, "end": 1.7},
        {"word": "is", "start": 1.8, "end": 1.9},
        {"word": "a", "start": 2.0, "end": 2.1},
        {"word": "test", "start": 2.2, "end": 2.5},
        {"word": "of", "start": 2.6, "end": 2.8},
        {"word": "speech", "start": 3.0, "end": 3.4},
        {"word": "synchronization.", "start": 3.5, "end": 4.2},
        
        # Simulate a pause
        {"word": "The", "start": 5.0, "end": 5.2},
        {"word": "quick", "start": 5.3, "end": 5.6},
        {"word": "brown", "start": 5.7, "end": 6.0},
        {"word": "fox", "start": 6.1, "end": 6.4},
        {"word": "jumps", "start": 6.5, "end": 6.9},
        {"word": "over", "start": 7.0, "end": 7.3},
        {"word": "the", "start": 7.4, "end": 7.5},
        {"word": "lazy", "start": 7.6, "end": 7.9},
        {"word": "dog.", "start": 8.0, "end": 8.5},
        
        # Another pause
        {"word": "This", "start": 10.0, "end": 10.3},
        {"word": "should", "start": 10.4, "end": 10.7},
        {"word": "create", "start": 10.8, "end": 11.2},
        {"word": "natural", "start": 11.3, "end": 11.8},
        {"word": "chunks", "start": 11.9, "end": 12.4},
        {"word": "based", "start": 12.5, "end": 12.8},
        {"word": "on", "start": 12.9, "end": 13.1},
        {"word": "actual", "start": 13.2, "end": 13.6},
        {"word": "speech", "start": 13.7, "end": 14.1},
        {"word": "timing.", "start": 14.2, "end": 14.8},
    ]
    return words


def test_speech_sync_mode():
    """Test speech synchronization mode."""
    print("🎯 Testing Speech Synchronization Mode")
    print("=" * 50)
    
    # Create mock word timestamps
    word_timestamps = create_mock_word_timestamps()
    print(f"Created {len(word_timestamps)} mock word timestamps")
    
    # Test speech sync processor
    processor = SubtitleProcessor(
        speech_sync_mode=True,
        min_word_duration_ms=600,  # 0.6s minimum
        max_word_duration_ms=2000,  # 2s maximum
    )
    
    # Process with speech sync (no groq_segments needed for speech sync)
    segments = processor.process_segments([], word_timestamps=word_timestamps)
    
    print(f"\n✅ Generated {len(segments)} speech-synchronized segments:")
    print("-" * 50)
    
    for i, segment in enumerate(segments, 1):
        duration = segment.duration()
        print(f"{i:2d}. [{segment.start_time:6.3f}s - {segment.end_time:6.3f}s] "
              f"({duration:5.2f}s) '{segment.text}'")
    
    # Test saving to files
    with tempfile.TemporaryDirectory() as temp_dir:
        srt_path, vtt_path = processor.save_subtitles(segments, temp_dir, "speech_sync_test")
        
        print(f"\n📁 Files created:")
        print(f"   SRT: {srt_path}")
        print(f"   VTT: {vtt_path}")
        
        # Show SRT content
        print(f"\n📝 SRT Content:")
        print("-" * 30)
        with open(srt_path, 'r', encoding='utf-8') as f:
            print(f.read())


def test_comparison_modes():
    """Compare different subtitle modes."""
    print("\n\n🔄 Comparing Subtitle Modes")
    print("=" * 50)
    
    word_timestamps = create_mock_word_timestamps()
    
    # Create mock groq segments from words (for comparison)
    full_text = " ".join([w["word"] for w in word_timestamps])
    mock_segment = SimpleNamespace()
    mock_segment.text = full_text
    mock_segment.start = word_timestamps[0]["start"]
    mock_segment.end = word_timestamps[-1]["end"]
    groq_segments = [mock_segment]
    
    modes = [
        ("Traditional", False, False),
        ("CapCut", True, False),
        ("Speech-Sync", False, True),
    ]
    
    for mode_name, capcut_mode, speech_sync_mode in modes:
        print(f"\n🎬 {mode_name} Mode:")
        print("-" * 30)
        
        processor = SubtitleProcessor(
            capcut_mode=capcut_mode,
            speech_sync_mode=speech_sync_mode,
            min_word_duration_ms=600,
            max_word_duration_ms=2000,
        )
        
        if speech_sync_mode:
            segments = processor.process_segments([], word_timestamps=word_timestamps)
        else:
            segments = processor.process_segments(groq_segments)
        
        print(f"Generated {len(segments)} segments:")
        for i, segment in enumerate(segments[:5], 1):  # Show first 5
            duration = segment.duration()
            print(f"  {i}. [{segment.start_time:6.3f}s - {segment.end_time:6.3f}s] "
                  f"({duration:5.2f}s) '{segment.text[:40]}{'...' if len(segment.text) > 40 else ''}'")
        
        if len(segments) > 5:
            print(f"  ... and {len(segments) - 5} more segments")


if __name__ == "__main__":
    try:
        test_speech_sync_mode()
        test_comparison_modes()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 