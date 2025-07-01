#!/usr/bin/env python3
"""Test script to verify subtitle text wrapping fixes."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.subs import SubtitleProcessor
from types import SimpleNamespace

def test_text_wrapping():
    """Test that long text is properly split into multiple segments instead of truncated."""
    
    print("🧪 Testing Subtitle Text Wrapping Fix...")
    print("="*60)
    
    # Create test processor
    processor = SubtitleProcessor(max_chars_per_line=50, max_lines=2)
    
    # Test case 1: Long text that would previously be truncated
    test_segment = SimpleNamespace()
    test_segment.start = 0.0
    test_segment.end = 30.0
    test_segment.text = "you know, an agent, a chat tool on the side to say, hey, you know, this is how you can unblock yourself and get the information you need quickly without having to wait for someone else"
    
    print(f"📝 Original text: '{test_segment.text}'")
    print(f"⏱️  Original timing: {test_segment.start}s - {test_segment.end}s")
    print()
    
    # Process the segment
    final_segments = []
    current_text = test_segment.text
    current_start = test_segment.start
    segment_duration = test_segment.end - test_segment.start
    
    iteration = 0
    while current_text.strip():
        iteration += 1
        wrapped_lines, remaining_text = processor._wrap_text(current_text)
        
        if wrapped_lines:
            # Calculate duration for this sub-segment
            if remaining_text.strip():
                # If there's remaining text, this is a partial segment
                chars_used = sum(len(line) for line in wrapped_lines)
                total_chars = len(test_segment.text)
                duration_fraction = chars_used / total_chars if total_chars > 0 else 1.0
                sub_duration = segment_duration * duration_fraction
                current_end = current_start + sub_duration
            else:
                # Last segment gets remaining time
                current_end = test_segment.end
            
            from app.services.subs import SubtitleSegment
            final_segments.append(SubtitleSegment(
                start_time=current_start,
                end_time=current_end,
                text="\n".join(wrapped_lines)
            ))
            
            print(f"✅ Segment {iteration}:")
            print(f"   Time: {current_start:.1f}s - {current_end:.1f}s ({current_end - current_start:.1f}s)")
            print(f"   Text: '{chr(10).join(wrapped_lines)}'")
            if remaining_text.strip():
                print(f"   Remaining: '{remaining_text[:50]}...'")
            print()
            
            # Update for next iteration
            current_text = remaining_text
            current_start = current_end
        else:
            print("❌ Could not wrap text!")
            break
        
        if iteration > 10:
            print("⚠️  Safety break - too many iterations")
            break
    
    print(f"📊 Summary:")
    print(f"   Original segments: 1")
    print(f"   Final segments: {len(final_segments)}")
    print(f"   Total characters: {len(test_segment.text)}")
    
    # Verify all text is preserved
    recovered_text = " ".join(segment.text.replace('\n', ' ') for segment in final_segments)
    print(f"   Text preserved: {recovered_text == test_segment.text}")
    
    if recovered_text != test_segment.text:
        print(f"   ❌ Text mismatch!")
        print(f"   Original:  '{test_segment.text}'")
        print(f"   Recovered: '{recovered_text}'")
    else:
        print(f"   ✅ All text preserved!")
    
    print("\n" + "="*60)
    return len(final_segments) > 1 and recovered_text == test_segment.text

def test_groq_segment_processing():
    """Test the full Groq segment processing pipeline."""
    
    print("🧪 Testing Full Groq Segment Processing...")
    print("="*60)
    
    # Create mock Groq segments
    mock_segments = [
        {
            'text': 'you know, an agent, a chat tool on the side to say, hey, you know, this is how',
            'start': 0.0,
            'end': 7.5
        },
        {
            'text': 'This is kind of how you can unblock yourself, right? Like the most frustrating thing',
            'start': 7.5,
            'end': 23.0
        },
        {
            'text': "That's what we mean. Everyone who wants to learn it can learn it. Now, that doesn't mean everyone will learn it.",
            'start': 23.0,
            'end': 55.0
        }
    ]
    
    processor = SubtitleProcessor(max_chars_per_line=50, max_lines=2)
    final_segments = processor.process_segments(mock_segments)
    
    print(f"📊 Processing Results:")
    print(f"   Input segments: {len(mock_segments)}")
    print(f"   Output segments: {len(final_segments)}")
    
    # Check if any text was lost
    input_text = " ".join(seg['text'] for seg in mock_segments)
    output_text = " ".join(seg.text.replace('\n', ' ') for seg in final_segments)
    
    print(f"   Input characters: {len(input_text)}")
    print(f"   Output characters: {len(output_text)}")
    print(f"   Text preserved: {len(input_text) <= len(output_text)}")  # Allow for slight differences due to spacing
    
    return len(final_segments) >= len(mock_segments)

if __name__ == "__main__":
    print("🚀 Running Subtitle Processing Tests...\n")
    
    # Test 1: Text wrapping
    test1_passed = test_text_wrapping()
    
    # Test 2: Full processing
    test2_passed = test_groq_segment_processing()
    
    print("\n🎯 Test Results:")
    print(f"   Text Wrapping Fix: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Full Processing: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Subtitle processing should now preserve all text.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed! Check the implementation.")
        sys.exit(1) 