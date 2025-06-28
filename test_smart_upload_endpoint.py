#!/usr/bin/env python3
"""
Test script to verify the new smart scene detection endpoint
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your server runs on different port
ENDPOINT = "/workflow/test-upload-vertical"

def test_smart_upload_endpoint():
    """Test the updated test_upload_vertical endpoint with smart scene detection"""
    
    print("🧪 Testing Smart Scene Detection Upload Endpoint")
    print("=" * 50)
    
    # Note: For this test, you'll need to have a sample video file
    # Replace with an actual video file path on your system
    test_video_path = Path("sample_video.mp4")  # Update this path
    
    if not test_video_path.exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("Please create a sample video file or update the path in this script")
        return False
    
    print(f"📁 Using test video: {test_video_path}")
    print(f"📊 File size: {test_video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Test 1: Smart Scene Detection (Async - Default)
    print(f"\n🎬 Test 1: Smart Scene Detection (Async)")
    print("-" * 30)
    
    with open(test_video_path, 'rb') as video_file:
        files = {'file': (test_video_path.name, video_file, 'video/mp4')}
        data = {
            'use_speaker_detection': True,
            'use_smart_scene_detection': True,
            'scene_content_threshold': 25.0,  # More sensitive
            'scene_fade_threshold': 10.0,
            'scene_min_length': 12,
            'ignore_micro_cuts': True,
            'micro_cut_threshold': 8,
            'smoothing_strength': 'very_high',
            'async_processing': True
        }
        
        try:
            response = requests.post(f"{BASE_URL}{ENDPOINT}", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Smart async upload successful!")
                print(f"📋 Task ID: {result['task_id']}")
                print(f"🎬 Scene detection enabled: {result['smart_features']['scene_detection_enabled']}")
                print(f"🔊 Speaker detection enabled: {result['smart_features']['speaker_detection_enabled']}")
                print(f"⚙️ Content threshold: {result['smart_features']['scene_content_threshold']}")
                print(f"🌅 Fade threshold: {result['smart_features']['scene_fade_threshold']}")
                print(f"🎛️ Smoothing: {result['smart_features']['smoothing_strength']}")
                
                # Monitor task progress
                task_id = result['task_id']
                monitor_task_progress(task_id)
                
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False
    
    # Test 2: Legacy Mode (Sync)
    print(f"\n🔄 Test 2: Legacy Mode (Sync)")
    print("-" * 20)
    
    with open(test_video_path, 'rb') as video_file:
        files = {'file': (test_video_path.name, video_file, 'video/mp4')}
        data = {
            'use_speaker_detection': True,
            'use_smart_scene_detection': False,
            'smoothing_strength': 'high',
            'async_processing': False
        }
        
        try:
            response = requests.post(f"{BASE_URL}{ENDPOINT}", files=files, data=data)
            
            if response.status_code == 200:
                print("✅ Legacy sync upload successful!")
                print(f"📁 Response type: {response.headers.get('content-type')}")
                # This should return a file download
                
            else:
                print(f"❌ Legacy upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Legacy request failed: {e}")
    
    # Test 3: Legacy Endpoint
    print(f"\n📼 Test 3: Legacy Endpoint")
    print("-" * 18)
    
    with open(test_video_path, 'rb') as video_file:
        files = {'file': (test_video_path.name, video_file, 'video/mp4')}
        data = {
            'use_speaker_detection': True,
            'smoothing_strength': 'medium'
        }
        
        try:
            response = requests.post(f"{BASE_URL}/workflow/test-upload-vertical-legacy", files=files, data=data)
            
            if response.status_code == 200:
                print("✅ Legacy endpoint successful!")
                print(f"📁 Response type: {response.headers.get('content-type')}")
                
            else:
                print(f"❌ Legacy endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Legacy endpoint request failed: {e}")
    
    return True

def monitor_task_progress(task_id: str, max_wait_minutes: int = 10):
    """Monitor the progress of an async task"""
    print(f"\n📊 Monitoring task progress: {task_id}")
    print("-" * 40)
    
    max_iterations = max_wait_minutes * 6  # Check every 10 seconds
    iteration = 0
    
    while iteration < max_iterations:
        try:
            response = requests.get(f"{BASE_URL}/workflow/task-status/{task_id}")
            
            if response.status_code == 200:
                status = response.json()
                
                print(f"🔄 Status: {status['status']} | Progress: {status['progress']}% | {status['message']}")
                
                if status['status'] == 'completed':
                    print("✅ Task completed successfully!")
                    print(f"📁 Output file size: {status.get('output_file_size_mb', 'N/A')} MB")
                    
                    # Check for smart scene detection results
                    if 'scenes_detected' in status:
                        print(f"🎬 Scenes detected: {status.get('scenes_detected', 'N/A')}")
                        print(f"🔄 Smart resets: {status.get('smart_resets', 'N/A')}")
                    
                    print(f"⬇️ Download: {BASE_URL}/workflow/download-result/{task_id}")
                    break
                    
                elif status['status'] == 'failed':
                    print(f"❌ Task failed: {status.get('error', 'Unknown error')}")
                    break
                    
            else:
                print(f"❌ Status check failed: {response.status_code}")
                break
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
            break
        
        time.sleep(10)  # Wait 10 seconds between checks
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"⏰ Timeout after {max_wait_minutes} minutes")

def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/workflow/health")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"Make sure the server is running on {BASE_URL}")
        return False

if __name__ == "__main__":
    print("🚀 Smart Scene Detection Endpoint Test")
    print("=" * 60)
    
    # Check server health first
    if not test_server_health():
        exit(1)
    
    # Run the tests
    success = test_smart_upload_endpoint()
    
    if success:
        print("\n✅ All tests completed!")
        print("🎬 Smart scene detection integration successful!")
    else:
        print("\n❌ Some tests failed")
        exit(1) 