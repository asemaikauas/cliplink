#!/usr/bin/env python3
"""
Test script for high-quality YouTube download functionality
"""

import requests
import json
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_URL = "https://youtu.be/S9BuExKqw3k?si=hiIjxLNj2j-ecyqS"

def test_video_info():
    """Test video info endpoint"""
    print("🔍 Testing video info endpoint...")
    
    response = requests.post(f"{BASE_URL}/workflow/video-info", 
                           json={"youtube_url": TEST_VIDEO_URL})
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Video info retrieved successfully")
        print(f"📺 Title: {data['video_info']['title']}")
        print(f"⏱️ Duration: {data['video_info']['duration']} sec")
        print(f"👁️ Views: {data['video_info'].get('view_count', 'N/A')}")
        print(f"🎯 Available qualities: {data['supported_qualities']}")
        print(f"📊 Top formats: {len(data['available_formats'])} available")
        return True
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return False

def test_download_only(quality="1080p"):
    """Test download only endpoint"""
    print(f"\n📥 Testing download-only endpoint with {quality} quality...")
    
    response = requests.post(f"{BASE_URL}/workflow/download-only", 
                           json={
                               "youtube_url": TEST_VIDEO_URL,
                               "quality": quality
                           })
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Download completed successfully")
        print(f"📁 File size: {data['download_info']['file_size_mb']} MB")
        print(f"📂 File path: {data['download_info']['file_path']}")
        print(f"🎯 Quality requested: {data['download_info']['quality_requested']}")
        return True
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return False

def test_complete_workflow(quality="best"):
    """Test complete workflow with quality"""
    print(f"\n🚀 Testing complete workflow with {quality} quality...")
    
    response = requests.post(f"{BASE_URL}/workflow/process-complete", 
                           json={
                               "youtube_url": TEST_VIDEO_URL,
                               "quality": quality
                           })
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Complete workflow finished successfully")
        print(f"📺 Video: {data['video_info']['title']}")
        print(f"📁 Download size: {data['download_info']['file_size_mb']} MB")
        print(f"🧠 Segments found: {data['analysis_results']['viral_segments_found']}")
        print(f"✂️ Clips created: {data['files_created']['clips_created']}")
        
        # Show segments
        for i, seg in enumerate(data['analysis_results']['segments'][:3]):
            print(f"  {i+1}. {seg['title']} ({seg['start']}s - {seg['end']}s)")
        
        return True
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return False

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    
    response = requests.get(f"{BASE_URL}/workflow/health")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Service is healthy")
        print(f"🛠️ Capabilities: {', '.join(data['capabilities'])}")
        print(f"🎯 Qualities: {', '.join(data['supported_qualities'])}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting high-quality download tests...")
    print("=" * 60)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/workflow/health", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ Server is not running! Please start the FastAPI server first:")
        print("   cd backend && python -m uvicorn app.main:app --reload --port 8000")
        return
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    # Test 2: Video info
    results.append(("Video Info", test_video_info()))
    
    # Test 3: Download only (1080p)
    results.append(("Download 1080p", test_download_only("1080p")))
    
    # Test 4: Download only (4K) - uncomment if you want to test
    # results.append(("Download 4K", test_download_only("4k")))
    
    # Test 5: Complete workflow (best quality)
    # results.append(("Complete Workflow", test_complete_workflow("best")))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 