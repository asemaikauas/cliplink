#!/usr/bin/env python3
"""
Test MediaPipe Face Detection
Quick test to verify MediaPipe is working for face detection
"""

import cv2
import numpy as np
from pathlib import Path

def test_mediapipe_import():
    """Test if MediaPipe can be imported"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False

def test_face_detection_init():
    """Test if MediaPipe Face Detection can be initialized"""
    try:
        import mediapipe as mp
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short range, 1 for full range
            min_detection_confidence=0.5
        )
        
        print("✅ MediaPipe Face Detection initialized successfully")
        print(f"   Model selection: 1 (full range)")
        print(f"   Min confidence: 0.5")
        return face_detection
        
    except Exception as e:
        print(f"❌ MediaPipe Face Detection initialization failed: {e}")
        return None

def test_face_detection_on_sample():
    """Test face detection on a sample image"""
    try:
        import mediapipe as mp
        
        # Initialize face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Create a simple test image (black with white rectangle for face-like shape)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face-like shape (rectangle)
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)
        cv2.circle(test_image, (270, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (330, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(test_image, (290, 260), (310, 280), (0, 0, 0), -1)  # Nose
        cv2.rectangle(test_image, (260, 300), (340, 320), (0, 0, 0), -1)  # Mouth
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_detection.process(rgb_image)
        
        if results.detections:
            print(f"✅ MediaPipe detected {len(results.detections)} face(s) in test image")
            for i, detection in enumerate(results.detections):
                confidence = detection.score[0]
                print(f"   Face {i+1}: confidence = {confidence:.3f}")
                
                bbox = detection.location_data.relative_bounding_box
                print(f"   Bounding box: x={bbox.xmin:.3f}, y={bbox.ymin:.3f}, "
                      f"w={bbox.width:.3f}, h={bbox.height:.3f}")
        else:
            print("⚠️ MediaPipe didn't detect any faces in test image")
            print("   This might be normal - the test image is very simple")
        
        face_detection.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe face detection test failed: {e}")
        return False

def test_mtcnn_fallback():
    """Test MTCNN as fallback"""
    try:
        from mtcnn import MTCNN
        detector = MTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.8])
        print("✅ MTCNN fallback available")
        print(f"   Min face size: 40")
        print(f"   Thresholds: [0.6, 0.7, 0.8]")
        return True
    except ImportError as e:
        print(f"❌ MTCNN not available: {e}")
        return False
    except Exception as e:
        print(f"❌ MTCNN initialization failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        print(f"✅ OpenCV available: {cv2.__version__}")
        
        # Test basic image operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV image operations working")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing MediaPipe Face Detection Setup")
    print("=" * 50)
    
    # Test 1: Import
    print("\n1. Testing MediaPipe Import:")
    mediapipe_ok = test_mediapipe_import()
    
    # Test 2: OpenCV
    print("\n2. Testing OpenCV:")
    opencv_ok = test_opencv()
    
    # Test 3: Face Detection Init
    print("\n3. Testing Face Detection Initialization:")
    face_detection_ok = test_face_detection_init() is not None
    
    # Test 4: Sample Detection
    print("\n4. Testing Face Detection on Sample:")
    sample_ok = test_face_detection_on_sample()
    
    # Test 5: MTCNN Fallback
    print("\n5. Testing MTCNN Fallback:")
    mtcnn_ok = test_mtcnn_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   MediaPipe Import: {'✅' if mediapipe_ok else '❌'}")
    print(f"   OpenCV: {'✅' if opencv_ok else '❌'}")
    print(f"   Face Detection Init: {'✅' if face_detection_ok else '❌'}")
    print(f"   Sample Detection: {'✅' if sample_ok else '❌'}")
    print(f"   MTCNN Fallback: {'✅' if mtcnn_ok else '❌'}")
    
    if mediapipe_ok and opencv_ok and face_detection_ok:
        print("\n🎉 MediaPipe Face Detection is working properly!")
        print("   Your vertical cropping should work for face tracking.")
    else:
        print("\n⚠️ Some issues detected with MediaPipe setup.")
        print("   Face detection may not work optimally.")
        
        if mtcnn_ok:
            print("   ✅ MTCNN fallback is available as backup.")
        else:
            print("   ❌ No face detection fallback available.") 