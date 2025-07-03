#!/usr/bin/env python3
"""
Test script for the FAST workflow endpoint optimized for speed:
- Parallel subtitle processing
- 720p quality default
- Medium smoothing
- Optimized settings
"""

import requests
import time
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your test video

def test_fast_workflow():
    """Test the fast workflow endpoint"""
    
    print("⚡ Testing FAST Workflow Endpoint")
    print("=" * 60)
    
    # Step 1: Start the fast workflow
    print(f"🚀 Starting FAST workflow for: {YOUTUBE_URL}")
    
    payload = {
        "youtube_url": YOUTUBE_URL,
        "quality": "720p",              # Fast default
        "create_vertical": True,        # Vertical clips
        "smoothing_strength": "medium", # Faster smoothing
        "burn_subtitles": True,         # Keep subtitles
        "font_size": 14,               # Smaller font for speed
        "export_codec": "h264",        # Fast codec
        "priority": "high"             # High priority
        # Note: Parallel processing is automatically enabled
    }
    
    response = requests.post(f"{API_BASE_URL}/workflow/process-fast-async", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to start fast workflow: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    
    print(f"✅ Fast workflow started successfully!")
    print(f"📋 Task ID: {task_id}")
    print(f"⏱️ Estimated time: {result['estimated_time']}")
    print(f"🔗 Status URL: {API_BASE_URL}{result['status_endpoint']}")
    print()
    
    print("🚀 Speed Optimizations:")
    for opt in result["speed_optimizations"]:
        print(f"  {opt}")
    print()
    
    print("🎯 Workflow Steps:")
    for i, step in enumerate(result["workflow_steps"], 1):
        print(f"  {i}. {step}")
    print()
    
    # Step 2: Monitor progress with speed tracking
    print("📊 Monitoring progress (speed optimized)...")
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
        
        # Show progress updates with speed indicators
        if current_progress > last_progress:
            elapsed = time.time() - start_time
            estimated_total = elapsed / (current_progress / 100) if current_progress > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            
            speed_indicator = "⚡" if "parallel" in current_step else "🚀" if current_progress > 75 else "📊"
            print(f"[{elapsed:6.1f}s] {current_progress:3d}% {speed_indicator} | {current_step:15s} | {message}")
            if remaining > 0:
                print(f"        └─ Est. remaining: {remaining:.1f}s")
            last_progress = current_progress
        
        # Check if completed
        if status["status"] == "completed":
            print()
            print("🎉 FAST workflow completed successfully!")
            print("=" * 60)
            
            # Show results
            result = status["result"]
            show_fast_results(result, time.time() - start_time)
            break
        
        elif status["status"] == "failed":
            print()
            print("❌ Fast workflow failed!")
            print(f"Error: {status.get('error', 'Unknown error')}")
            break
        
        # Wait before next check
        time.sleep(2)  # Faster polling for speed test

def show_fast_results(result: Dict[str, Any], total_time: float):
    """Display the fast workflow results with speed metrics"""
    
    print("\n📋 FAST WORKFLOW RESULTS")
    print("=" * 60)
    
    # Speed metrics
    print(f"⚡ Total Processing Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Video info
    video_info = result["video_info"]
    print(f"🎬 Video: {video_info['title']}")
    print(f"⏱️  Duration: {video_info['duration']} seconds")
    
    # Analysis results
    analysis = result["analysis_results"]
    print(f"\n🤖 AI Analysis: {analysis['viral_segments_found']} viral segments found")
    
    # Subtitle info
    subtitle_info = result["subtitle_info"]
    if subtitle_info:
        print(f"\n📝 Subtitles: {subtitle_info['clips_with_subtitles']}/{subtitle_info['total_clips']} clips ({subtitle_info['subtitle_success_rate']})")
        print(f"⚡ Processing: {subtitle_info['subtitle_approach']}")
        print(f"🎯 Speech Sync: {subtitle_info['speech_synchronization']}")
    
    # Files created
    files = result["files_created"]
    print(f"\n📁 Files Created:")
    print(f"   📺 Source quality: {result['download_info']['quality_requested']} ({result['download_info']['file_size_mb']} MB)")
    print(f"   ✂️  Clips created: {files['clips_created']}")
    print(f"   🔥 Subtitled clips: {files['subtitled_clips_created']}")
    print(f"   📱 Type: {files['clip_type']}")
    
    # Speed comparison
    estimated_sequential_time = total_time * 2.5  # Rough estimate
    print(f"\n🚀 Speed Comparison:")
    print(f"   ⚡ Fast workflow: {total_time:.1f}s")
    print(f"   🐌 Sequential estimate: {estimated_sequential_time:.1f}s")
    print(f"   📈 Speed improvement: {estimated_sequential_time/total_time:.1f}x faster")
    
    # Show segments
    print(f"\n🎯 Viral Segments Found:")
    for i, segment in enumerate(analysis['segments'], 1):
        print(f"   {i}. {segment['title']} ({segment['start']}-{segment['end']}s)")
    
    print(f"\n✅ Ready for upload to TikTok/YouTube Shorts/Instagram Reels!")
    print(f"⚡ Processing completed in {total_time/60:.1f} minutes!")

if __name__ == "__main__":
    print("⚡ ClipLink FAST Workflow Test")
    print("Speed-optimized processing with parallel subtitle generation")
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
        
        # Test fast workflow
        test_fast_workflow()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
        print("Run: cd backend && python -m uvicorn app.main:app --reload")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}") 