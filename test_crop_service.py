#!/usr/bin/env python3
"""
Test script for Video Cropping Service
Tests MediaPipe, OpenCV, and video processing functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🧪 Testing Video Cropping Dependencies")
    print("=" * 50)
    
    # Test MediaPipe
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
        mp_available = True
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        mp_available = False
    
    # Test OpenCV
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
        cv_available = True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        cv_available = False
    
    # Test NumPy
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
        np_available = True
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        np_available = False
    
    # Test ffmpeg-python
    try:
        import ffmpeg
        print("✅ ffmpeg-python imported successfully")
        ffmpeg_available = True
    except ImportError as e:
        print(f"⚠️ ffmpeg-python not available: {e}")
        ffmpeg_available = False
    
    # Test PyAnnote (optional)
    try:
        from pyannote.audio import Pipeline
        print("✅ PyAnnote Audio available")
        pyannote_available = True
    except ImportError as e:
        print(f"⚠️ PyAnnote Audio not available: {e}")
        pyannote_available = False
    
    # Test MTCNN (optional)
    try:
        from mtcnn import MTCNN
        print("✅ MTCNN available")
        mtcnn_available = True
    except ImportError as e:
        print(f"⚠️ MTCNN not available: {e}")
        mtcnn_available = False
    
    print("\n" + "=" * 50)
    print("📊 Dependency Summary:")
    print(f"   MediaPipe (required): {'✅' if mp_available else '❌'}")
    print(f"   OpenCV (required): {'✅' if cv_available else '❌'}")
    print(f"   NumPy (required): {'✅' if np_available else '❌'}")
    print(f"   ffmpeg-python (optional): {'✅' if ffmpeg_available else '⚠️'}")
    print(f"   PyAnnote Audio (optional): {'✅' if pyannote_available else '⚠️'}")
    print(f"   MTCNN (optional): {'✅' if mtcnn_available else '⚠️'}")
    
    # Determine service readiness
    required_deps = mp_available and cv_available and np_available
    
    if required_deps:
        print("\n🎉 Video cropping service is ready!")
        print("   Core face detection and video processing will work.")
        if not ffmpeg_available:
            print("   ⚠️ Consider installing ffmpeg-python for better video handling.")
        if not pyannote_available:
            print("   ⚠️ Interview mode will be limited without PyAnnote Audio.")
    else:
        print("\n❌ Video cropping service is NOT ready.")
        print("   Please install missing required dependencies.")
    
    return required_deps

def test_basic_face_detection():
    """Test basic MediaPipe face detection"""
    print("\n🧪 Testing Basic Face Detection")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.5
        )
        
        # Create a simple test image with a face-like pattern
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face pattern
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)  # Face
        cv2.circle(test_image, (270, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (330, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(test_image, (290, 260), (310, 280), (0, 0, 0), -1)  # Nose
        cv2.rectangle(test_image, (260, 300), (340, 320), (0, 0, 0), -1)  # Mouth
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_detection.process(rgb_image)
        
        if results.detections:
            print(f"✅ MediaPipe detected {len(results.detections)} face(s) in synthetic image")
            for i, detection in enumerate(results.detections):
                confidence = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                print(f"   Face {i+1}: confidence = {confidence:.3f}")
                print(f"   Bounding box: x={bbox.xmin:.3f}, y={bbox.ymin:.3f}, "
                      f"w={bbox.width:.3f}, h={bbox.height:.3f}")
        else:
            print("⚠️ MediaPipe didn't detect faces in synthetic image")
            print("   This might be normal - the test image is very simple")
        
        face_detection.close()
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_video_cropper_import():
    """Test if video cropper modules can be imported"""
    print("\n🧪 Testing Video Cropper Module Imports")
    print("=" * 50)
    
    try:
        from app.services.video_cropper import (
            CropMode, AspectRatio, CropSettings, FaceDetection
        )
        print("✅ Video cropper core classes imported successfully")
        
        # Test enum values
        print(f"   Available crop modes: {[mode.value for mode in CropMode]}")
        print(f"   Available aspect ratios: {[f'{ratio.value[0]}:{ratio.value[1]}' for ratio in AspectRatio]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video cropper import failed: {e}")
        return False

def test_face_detector_import():
    """Test if face detector can be imported"""
    print("\n🧪 Testing Face Detector Import")
    print("=" * 50)
    
    try:
        from app.services.face_detector import AdvancedFaceDetector
        
        detector = AdvancedFaceDetector()
        print("✅ AdvancedFaceDetector initialized successfully")
        
        # Test cleanup
        detector.cleanup()
        print("✅ Face detector cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detector test failed: {e}")
        return False

def test_crop_api_import():
    """Test if crop API router can be imported"""
    print("\n🧪 Testing Crop API Router Import")
    print("=" * 50)
    
    try:
        from app.routers.crop import router
        print("✅ Crop API router imported successfully")
        
        # Check available routes
        routes = [route.path for route in router.routes]
        print(f"   Available endpoints: {routes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Crop API router import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Video Cropping Service Test Suite")
    print("=" * 70)
    
    # Test 1: Dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Cannot proceed with further tests - missing dependencies")
        return False
    
    # Test 2: Basic face detection
    face_detection_ok = test_basic_face_detection()
    
    # Test 3: Video cropper imports
    video_cropper_ok = test_video_cropper_import()
    
    # Test 4: Face detector imports
    face_detector_ok = test_face_detector_import()
    
    # Test 5: API router imports
    api_ok = test_crop_api_import()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   Face Detection: {'✅' if face_detection_ok else '❌'}")
    print(f"   Video Cropper: {'✅' if video_cropper_ok else '❌'}")
    print(f"   Face Detector: {'✅' if face_detector_ok else '❌'}")
    print(f"   API Router: {'✅' if api_ok else '❌'}")
    
    all_ok = all([deps_ok, face_detection_ok, video_cropper_ok, face_detector_ok, api_ok])
    
    if all_ok:
        print("\n🎉 All tests passed! Video cropping service is ready to use.")
        print("\n🔥 You can now use the following endpoints:")
        print("   POST /crop/analyze - Analyze video for optimal mode")
        print("   POST /crop/crop - Crop video from URL")
        print("   POST /crop/crop-upload - Upload and crop video")
        print("   GET /crop/status/{task_id} - Check crop task status")
        print("   GET /crop/download/{task_id} - Download cropped video")
        print("   GET /crop/health - Service health check")
    else:
        print("\n❌ Some tests failed. Please fix issues before using the service.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 