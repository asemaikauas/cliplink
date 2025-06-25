#!/usr/bin/env python3
"""
Test script to verify Gemini API error handling and retry logic
"""

import asyncio
import json
from app.services.gemini import analyze_transcript_with_gemini

async def test_gemini_api():
    """Test the Gemini API with improved error handling"""
    
    # Sample transcript data for testing
    test_data = {
        "title": "Test Video Title",
        "description": "This is a test video description",
        "category": "Education",
        "transcript": "This is a sample transcript for testing the Gemini API functionality. It should be long enough to demonstrate the viral segment analysis capabilities.",
        "timecodes": [
            {"start": 0, "duration": 5, "text": "This is a sample transcript"},
            {"start": 5, "duration": 10, "text": "for testing the Gemini API functionality"},
            {"start": 15, "duration": 8, "text": "It should be long enough to demonstrate"},
            {"start": 23, "duration": 12, "text": "the viral segment analysis capabilities"}
        ]
    }
    
    print("🧪 Testing Gemini API with improved error handling...")
    print(f"📝 Test data: {len(test_data['transcript'])} characters")
    
    try:
        result = await analyze_transcript_with_gemini(test_data)
        
        if result and "gemini_analysis" in result:
            analysis = result["gemini_analysis"]
            if "viral_segments" in analysis:
                print(f"✅ Success! Found {len(analysis['viral_segments'])} viral segments")
                for i, segment in enumerate(analysis['viral_segments'], 1):
                    print(f"   {i}. {segment.get('title', 'No title')} ({segment.get('start', 0)}-{segment.get('end', 0)}s)")
            else:
                print(f"⚠️ No viral segments found in response")
                print(f"   Response: {json.dumps(analysis, indent=2)}")
        else:
            print(f"❌ Invalid response format")
            print(f"   Result: {json.dumps(result, indent=2)}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    print("🚀 Starting Gemini API test...")
    success = asyncio.run(test_gemini_api())
    
    if success:
        print("\n✅ All tests passed! Gemini API error handling is working correctly.")
    else:
        print("\n❌ Tests failed! Please check your GEMINI_API_KEY and network connection.") 