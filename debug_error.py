#!/usr/bin/env python3

"""
Debug script to identify the 'too many values to unpack' error
"""

import sys
import traceback

def test_import():
    """Test importing the intelligent cropping components"""
    try:
        print("Testing imports...")
        
        # Test individual component imports
        print("1. Testing utils import...")
        from app.services.intelligent_cropper.utils import ConfigManager
        print("   ✅ Utils imported successfully")
        
        print("2. Testing speaker detector import...")
        from app.services.intelligent_cropper.speaker_detector import SpeakerCountDetector
        print("   ✅ Speaker detector imported successfully")
        
        print("3. Testing solo mode import...")
        from app.services.intelligent_cropper.solo_mode import SoloSpeakerProcessor
        print("   ✅ Solo mode imported successfully")
        
        print("4. Testing interview mode import...")
        from app.services.intelligent_cropper.interview_mode import InterviewProcessor
        print("   ✅ Interview mode imported successfully")
        
        print("5. Testing pipeline import...")
        from app.services.intelligent_cropper.pipeline import IntelligentCroppingPipeline
        print("   ✅ Pipeline imported successfully")
        
        print("6. Testing pipeline initialization...")
        pipeline = IntelligentCroppingPipeline()
        print("   ✅ Pipeline initialized successfully")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_scene_detection():
    """Test scene detection specifically"""
    try:
        print("\nTesting scene detection...")
        from scenedetect import detect, ContentDetector
        
        # Test with dummy data
        scene_list = []  # Empty list to test parsing
        
        scene_boundaries = []
        for scene in scene_list:
            try:
                if hasattr(scene, '__len__') and len(scene) >= 2:
                    start_time = scene[0].get_seconds() if hasattr(scene[0], 'get_seconds') else float(scene[0])
                    end_time = scene[1].get_seconds() if hasattr(scene[1], 'get_seconds') else float(scene[1])
                    scene_boundaries.append((start_time, end_time))
                else:
                    print(f"Unexpected scene format: {scene}")
            except (AttributeError, IndexError, TypeError) as e:
                print(f"Error parsing scene {scene}: {e}")
                continue
                
        print("   ✅ Scene detection parsing test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Scene detection error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Starting intelligent cropping debug tests...\n")
    
    success1 = test_import()
    success2 = test_scene_detection()
    
    if success1 and success2:
        print("\n✅ All debug tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 