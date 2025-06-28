#!/usr/bin/env python3
"""
Test script for Group Conversation Framing feature

This script demonstrates the new dual-speaker split-screen layout functionality
that automatically detects when there are exactly 2 speakers in a scene and 
creates a split-screen layout with:
- Top half: First speaker
- Bottom half: Second speaker

Perfect for conversation scenes, interviews, debates, and panel discussions!
"""

import requests
import json
from pathlib import Path
import time

# Test server configuration
BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/workflow/test-upload-vertical"

def test_group_conversation_api():
    """Test the group conversation framing via API"""
    
    print("👥 Testing Group Conversation Framing Feature")
    print("=" * 60)
    
    # Test video path (you can replace this with your own conversation video)
    test_video_path = "test_conversation.mp4"  # Replace with actual path
    
    if not Path(test_video_path).exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("📝 Please place a conversation video (with 2 people) at test_conversation.mp4")
        return
    
    print(f"📹 Using test video: {test_video_path}")
    
    # Test parameters for group conversation
    test_params = {
        "use_speaker_detection": True,
        "use_smart_scene_detection": True,
        "enable_group_conversation_framing": True,  # 🆕 NEW FEATURE!
        "scene_content_threshold": 30.0,
        "scene_fade_threshold": 8.0,
        "scene_min_length": 15,
        "ignore_micro_cuts": True,
        "micro_cut_threshold": 10,
        "smoothing_strength": "very_high",
        "async_processing": True
    }
    
    print("\n🎛️ Test Parameters:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n🚀 Starting group conversation framing test...")
    
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
            
            print(f"✅ Upload successful!")
            print(f"📋 Task ID: {task_id}")
            print(f"🎯 Group conversation framing: {'ENABLED' if result['smart_features']['group_conversation_framing_enabled'] else 'DISABLED'}")
            
            # Monitor progress
            print(f"\n📊 Monitoring progress...")
            monitor_task_progress(task_id)
            
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def monitor_task_progress(task_id: str):
    """Monitor task progress and show results"""
    
    status_url = f"{BASE_URL}/workflow/task-status/{task_id}"
    
    while True:
        try:
            response = requests.get(status_url)
            if response.status_code == 200:
                status = response.json()
                
                progress = status.get("progress", 0)
                message = status.get("message", "")
                task_status = status.get("status", "unknown")
                
                print(f"🔄 Progress: {progress}% | Status: {task_status} | {message}")
                
                if task_status == "completed":
                    print(f"\n🎉 Processing completed successfully!")
                    
                    # Show results
                    output_path = status.get("output_path")
                    file_size = status.get("output_file_size_mb", 0)
                    processing_time = status.get("processing_time_seconds", 0)
                    
                    print(f"📁 Output file: {output_path}")
                    print(f"📊 File size: {file_size:.2f} MB")
                    print(f"⏱️ Processing time: {processing_time:.1f} seconds")
                    
                    # Show download link
                    download_url = f"{BASE_URL}/workflow/download-result/{task_id}"
                    print(f"⬇️ Download: {download_url}")
                    
                    print(f"\n👥 Group Conversation Features Applied:")
                    print(f"   • When 2 faces detected → Split-screen layout")
                    print(f"   • Top half: First speaker")
                    print(f"   • Bottom half: Second speaker")
                    print(f"   • Subtle divider line between speakers")
                    print(f"   • Smart framing for each speaker region")
                    
                    break
                    
                elif task_status == "failed":
                    print(f"❌ Processing failed!")
                    error = status.get("error", "Unknown error")
                    print(f"Error: {error}")
                    break
                
                # Wait before next check
                time.sleep(3)
            
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                break
                
        except Exception as e:
            print(f"❌ Error monitoring progress: {str(e)}")
            break

def main():
    """Main test function"""
    
    print("🎬 GROUP CONVERSATION FRAMING TEST SUITE")
    print("=" * 50)
    print()
    print("This feature automatically creates split-screen layouts when")
    print("exactly 2 speakers are detected in conversation scenes:")
    print()
    print("✨ Benefits:")
    print("  • Perfect for interviews, debates, panel discussions")
    print("  • Shows both speakers simultaneously")
    print("  • Smart individual framing for each speaker")
    print("  • Seamless fallback to single-speaker mode")
    print("  • Works with all existing scene detection features")
    print()
    print("📋 Test Scenarios:")
    print("  1. Interview Style - Standard 2-person conversation")
    print("  2. Debate Format - Dynamic speaker changes")
    print("  3. Panel Discussion - Multiple people (fallback mode)")
    print("  4. Single Speaker - Traditional mode for comparison")
    print()
    
    # Run the actual test
    choice = input("Run API test? (y/n): ").lower().strip()
    if choice == 'y':
        test_group_conversation_api()
    else:
        print("👋 Test skipped. Try again when you have a conversation video ready!")

if __name__ == "__main__":
    main() 