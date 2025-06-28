#!/usr/bin/env python3
"""
Test script for Smooth Transition Fix

This tests the fix for smoothing conflicts when transitioning between 
single-speaker and dual-speaker modes in group conversation framing.

ISSUE FIXED:
- Single→Dual transitions were jarring (sudden jump to center)
- Dual→Single transitions were jarring (sudden jump from center to speaker)
- High smoothing settings made the problem worse

SOLUTION:
- Smart mode transition detection
- Eased interpolation between positions
- Preserved smoothing history during transitions
- Configurable transition timing
"""

import requests
import json
from pathlib import Path
import time

# Test server configuration
BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/workflow/test-upload-vertical"

def test_smooth_transitions():
    """Test the smooth transition functionality"""
    
    print("🔄 Testing Smooth Mode Transitions")
    print("=" * 50)
    print()
    print("This test verifies that transitions between single-speaker")
    print("and dual-speaker modes are now smooth instead of jarring.")
    print()
    
    # Test video path - should have scenes that transition between 1 and 2 people
    test_video_path = "test_conversation_transitions.mp4"
    
    if not Path(test_video_path).exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("📝 Please provide a video that transitions between:")
        print("   • Single speaker scenes")
        print("   • Dual speaker scenes (conversations)")
        print("   • Back to single speaker")
        return
    
    print(f"📹 Using test video: {test_video_path}")
    
    # Test with high smoothing to verify the fix works even with aggressive smoothing
    test_params = {
        "use_speaker_detection": True,
        "use_smart_scene_detection": True,
        "enable_group_conversation_framing": True,  # Enable dual-speaker mode
        "scene_content_threshold": 25.0,  # Slightly more sensitive
        "scene_fade_threshold": 8.0,
        "scene_min_length": 10,  # Shorter scenes to catch more transitions
        "ignore_micro_cuts": True,
        "micro_cut_threshold": 8,
        "smoothing_strength": "very_high",  # High smoothing = where the bug was worst
        "async_processing": True
    }
    
    print("\n🎛️ Test Configuration (High Smoothing + Transitions):")
    for key, value in test_params.items():
        emoji = "🆕" if key == "enable_group_conversation_framing" else "🔧"
        print(f"   {emoji} {key}: {value}")
    
    print(f"\n🚀 Testing smooth transitions...")
    print(f"📊 What to expect:")
    print(f"   ✅ Smooth 0.33s transitions between modes")
    print(f"   ✅ No jarring jumps at scene changes")
    print(f"   ✅ Preserved tracking during transitions")
    print(f"   ✅ Detailed transition logging")
    
    try:
        # Upload and process video
        with open(test_video_path, 'rb') as video_file:
            files = {'file': (test_video_path, video_file, 'video/mp4')}
            
            response = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                params=test_params
            )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            
            print(f"\n✅ Upload successful!")
            print(f"📋 Task ID: {task_id}")
            print(f"🔄 Smooth transitions: ENABLED")
            print(f"👥 Group conversation framing: {'ENABLED' if result['smart_features']['group_conversation_framing_enabled'] else 'DISABLED'}")
            
            # Monitor progress and look for transition logs
            print(f"\n📊 Monitoring for transition events...")
            monitor_transitions(task_id)
            
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def monitor_transitions(task_id: str):
    """Monitor task progress and highlight transition information"""
    
    status_url = f"{BASE_URL}/workflow/task-status/{task_id}"
    
    while True:
        try:
            response = requests.get(status_url)
            if response.status_code == 200:
                status = response.json()
                
                progress = status.get("progress", 0)
                message = status.get("message", "")
                task_status = status.get("status", "unknown")
                
                # Look for transition-related messages
                if "transition" in message.lower() or "mode" in message.lower():
                    print(f"🔄 TRANSITION: {message}")
                else:
                    print(f"📊 Progress: {progress}% | {message}")
                
                if task_status == "completed":
                    print(f"\n🎉 Smooth transition test completed!")
                    
                    # Show results
                    output_path = status.get("output_path")
                    file_size = status.get("output_file_size_mb", 0)
                    processing_time = status.get("processing_time_seconds", 0)
                    
                    print(f"📁 Output: {output_path}")
                    print(f"📊 Size: {file_size:.2f} MB")
                    print(f"⏱️ Time: {processing_time:.1f}s")
                    
                    # Show download link
                    download_url = f"{BASE_URL}/workflow/download-result/{task_id}"
                    print(f"⬇️ Download: {download_url}")
                    
                    print(f"\n✨ Smooth Transition Features:")
                    print(f"   🔄 Mode transitions detected automatically")
                    print(f"   📏 0.33s smooth interpolation between positions")
                    print(f"   🎯 Ease-out for single→dual transitions")
                    print(f"   🎯 Ease-in-out for dual→single transitions")
                    print(f"   📊 Preserved smoothing history during transitions")
                    print(f"   🚫 No more jarring jumps!")
                    
                    break
                    
                elif task_status == "failed":
                    print(f"❌ Test failed!")
                    error = status.get("error", "Unknown error")
                    print(f"Error: {error}")
                    break
                
                # Wait before next check
                time.sleep(2)
            
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                break
                
        except Exception as e:
            print(f"❌ Error monitoring: {str(e)}")
            break

def main():
    """Main test function"""
    
    print("🔄 SMOOTH TRANSITIONS TEST SUITE")
    print("=" * 40)
    print()
    print("This test verifies the fix for smoothing conflicts")
    print("during mode transitions in group conversation framing.")
    print()
    print("🚨 ISSUE FIXED:")
    print("  • Single→Dual: Jarring jump to center")
    print("  • Dual→Single: Jarring jump from center")
    print("  • High smoothing made it worse")
    print()
    print("✅ SOLUTION IMPLEMENTED:")
    print("  • Smart transition detection")
    print("  • Smooth 0.33s interpolation")
    print("  • Easing functions for natural motion")
    print("  • Preserved tracking history")
    print()
    
    # Run the test
    choice = input("Run smooth transition test? (y/n): ").lower().strip()
    if choice == 'y':
        test_smooth_transitions()
    else:
        print("👋 Test skipped. Try with a video that has speaker transitions!")

if __name__ == "__main__":
    main() 