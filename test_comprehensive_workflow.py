#!/usr/bin/env python3
"""
Test script for the comprehensive workflow endpoint that combines:
- Transcript extraction
- Gemini AI analysis  
- Video download
- Vertical cropping
- Subtitle burning

This endpoint provides everything in one API call!
"""

import requests
import time
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your test video

def test_comprehensive_workflow():
    """Test the comprehensive workflow endpoint"""
    
    print("🚀 Testing Comprehensive Workflow Endpoint")
    print("=" * 60)
    
    # Step 1: Start the comprehensive workflow
    print(f"🎬 Starting comprehensive workflow for: {YOUTUBE_URL}")
    
    payload = {
        "youtube_url": YOUTUBE_URL,
        "quality": "1080p",           # Video quality: best, 8k, 4k, 1440p, 1080p, 720p
        "create_vertical": True,       # Create vertical 9:16 clips
        "smoothing_strength": "very_high",  # Motion smoothing: low, medium, high, very_high
        "burn_subtitles": True,        # Burn subtitles into clips (always uses speech synchronization)
        "font_size": 16,              # Subtitle font size (12-120)
        "export_codec": "h264",       # Video codec: h264, h265
        "priority": "normal"          # Priority: low, normal, high
        # Note: Speech synchronization and VAD filtering are automatically enabled for best quality
    }
    
    response = requests.post(f"{API_BASE_URL}/workflow/process-comprehensive-async", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to start workflow: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    
    print(f"✅ Workflow started successfully!")
    print(f"📋 Task ID: {task_id}")
    print(f"⏱️ Estimated time: {result['estimated_time']}")
    print(f"🔗 Status URL: {API_BASE_URL}{result['status_endpoint']}")
    print()
    
    print("🎯 Workflow Steps:")
    for i, step in enumerate(result["workflow_steps"], 1):
        print(f"  {i}. {step}")
    print()
    
    # Step 2: Monitor progress
    print("📊 Monitoring progress...")
    print("-" * 40)
    
    last_progress = 0
    start_time = time.time()
    
    while True:
        # Get status
        status_response = requests.get(f"{API_BASE_URL}/workflow/workflow-status/{task_id}")
        
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.status_code}")
            break
        
        status = status_response.json()
        current_progress = status["progress"]
        current_step = status["current_step"]
        message = status["message"]
        
        # Show progress updates
        if current_progress > last_progress:
            elapsed = time.time() - start_time
            print(f"[{elapsed:6.1f}s] {current_progress:3d}% | {current_step:15s} | {message}")
            last_progress = current_progress
        
        # Check if completed
        if status["status"] == "completed":
            print()
            print("🎉 Comprehensive workflow completed successfully!")
            print("=" * 60)
            
            # Show results
            result = status["result"]
            show_results(result)
            break
        
        elif status["status"] == "failed":
            print()
            print("❌ Workflow failed!")
            print(f"Error: {status.get('error', 'Unknown error')}")
            break
        
        # Wait before next check
        time.sleep(3)

def show_results(result: Dict[str, Any]):
    """Display the comprehensive workflow results"""
    
    print("\n📋 COMPREHENSIVE WORKFLOW RESULTS")
    print("=" * 60)
    
    # Video info
    video_info = result["video_info"]
    print(f"🎬 Video: {video_info['title']}")
    print(f"⏱️  Duration: {video_info['duration']} seconds")
    print(f"👤 Uploader: {video_info.get('uploader', 'Unknown')}")
    print(f"📊 Views: {video_info.get('view_count', 'Unknown'):,}")
    
    # Download info
    download_info = result["download_info"]
    print(f"\n📥 Download: {download_info['quality_requested']} quality, {download_info['file_size_mb']} MB")
    
    # Analysis results
    analysis = result["analysis_results"]
    print(f"\n🤖 AI Analysis: {analysis['viral_segments_found']} viral segments found")
    
    # Subtitle info
    subtitle_info = result["subtitle_info"]
    if subtitle_info:
        print(f"\n📝 Subtitles: {subtitle_info['clips_with_subtitles']}/{subtitle_info['total_clips']} clips ({subtitle_info['subtitle_success_rate']})")
        print(f"🎯 Speech Sync: {subtitle_info['speech_synchronization']} (word-level timestamps)")
        print(f"🎛️ VAD Filtering: {subtitle_info['vad_filtering']} (with retry logic)")
        print(f"🔧 Approach: {subtitle_info['subtitle_approach']}")
    else:
        print(f"\n📝 Subtitles: Disabled")
    
    # Files created
    files = result["files_created"]
    print(f"\n📁 Files Created:")
    print(f"   📺 Source video: {files['source_video']}")
    print(f"   ✂️  Original clips: {files['clips_created']}")
    print(f"   🔥 Subtitled clips: {files['subtitled_clips_created']}")
    print(f"   📱 Clip type: {files['clip_type']}")
    if files['subtitle_files_location']:
        print(f"   📁 Subtitle files: {files['subtitle_files_location']}")
    else:
        print(f"   📄 No subtitle files generated")
    
    # Show segments
    print(f"\n🎯 Viral Segments Found:")
    for i, segment in enumerate(analysis['segments'], 1):
        print(f"   {i}. {segment['title']} ({segment['start']}-{segment['end']}s, {segment['duration']}s)")
    
    print(f"\n✅ All files ready for upload to TikTok/YouTube Shorts/Instagram Reels!")

def test_status_endpoint():
    """Test just the status endpoint with a sample task ID"""
    print("\n🔍 Testing status endpoint...")
    
    # This will return 404 since we're using a fake task ID, but shows the endpoint works
    fake_task_id = "comprehensive_12345678"
    response = requests.get(f"{API_BASE_URL}/workflow/workflow-status/{fake_task_id}")
    
    print(f"Status response: {response.status_code}")
    if response.status_code == 404:
        print("✅ Status endpoint working (404 expected for fake task ID)")

if __name__ == "__main__":
    print("🎬 ClipLink Comprehensive Workflow Test")
    print("This will test the complete pipeline from YouTube URL to ready-to-upload clips with subtitles")
    print()
    
    # Test the API
    try:
        # Test API health
        health_response = requests.get(f"{API_BASE_URL}/workflow/health")
        if health_response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            exit(1)
        
        # Test comprehensive workflow
        test_comprehensive_workflow()
        
        # Test status endpoint
        test_status_endpoint()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
        print("Run: cd backend && python -m uvicorn app.main:app --reload")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}") 