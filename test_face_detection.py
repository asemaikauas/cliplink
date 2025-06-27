#!/usr/bin/env python3
"""
Face Detection Diagnostic Script
Tests all face detection components to identify issues
"""

import cv2
import numpy as np
import os
from pathlib import Path

def test_opencv():
    """Test OpenCV installation and basic functionality"""
    print("🔍 Testing OpenCV...")
    try:
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functions
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV color conversion works")
        
        # Test video capture (will fail but should not crash)
        cap = cv2.VideoCapture()
        print("✅ OpenCV VideoCapture class available")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face detection"""
    print("\n🔍 Testing MediaPipe Face Detection...")
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe imported successfully")
        
        # Initialize face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe FaceDetection initialized")
        
        # Test with a dummy image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a simple "face-like" pattern
        cv2.rectangle(test_image, (250, 150), (390, 330), (255, 255, 255), -1)  # face
        cv2.circle(test_image, (290, 200), 10, (0, 0, 0), -1)  # left eye
        cv2.circle(test_image, (350, 200), 10, (0, 0, 0), -1)  # right eye
        cv2.circle(test_image, (320, 250), 5, (0, 0, 0), -1)   # nose
        
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if results.detections:
            print(f"✅ MediaPipe detected {len(results.detections)} face(s) in test image")
        else:
            print("⚠️ MediaPipe didn't detect faces in test image (this is normal for synthetic image)")
        
        face_detection.close()
        return True
        
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_mtcnn():
    """Test MTCNN face detection"""
    print("\n🔍 Testing MTCNN Face Detection...")
    try:
        from mtcnn import MTCNN
        print("✅ MTCNN imported successfully")
        
        # Initialize MTCNN
        detector = MTCNN(min_face_size=40, thresholds=[0.6, 0.7, 0.8])
        print("✅ MTCNN detector initialized")
        
        # Test with a dummy image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        try:
            detections = detector.detect_faces(rgb_image)
            print(f"✅ MTCNN processed test image, found {len(detections)} faces")
        except Exception as e:
            print(f"⚠️ MTCNN detection test failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ MTCNN import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MTCNN test failed: {e}")
        return False

def test_dnn_face_detection():
    """Test OpenCV DNN face detection models"""
    print("\n🔍 Testing OpenCV DNN Face Detection...")
    
    prototxt_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    if not Path(prototxt_path).exists():
        print(f"❌ Prototxt file not found: {prototxt_path}")
        return False
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("✅ OpenCV DNN model loaded successfully")
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(test_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        print(f"✅ DNN inference completed, output shape: {detections.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DNN face detection test failed: {e}")
        return False

def test_face_detection_service():
    """Test our AdvancedFaceDetector class"""
    print("\n🔍 Testing AdvancedFaceDetector Service...")
    try:
        from app.services.vertical_crop_async import AdvancedFaceDetector
        
        face_detector = AdvancedFaceDetector()
        print("✅ AdvancedFaceDetector initialized")
        
        # Create a test image with a simple face pattern
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a simple rectangular "face"
        cv2.rectangle(test_image, (250, 150), (390, 330), (128, 128, 128), -1)
        
        # Test both detection methods
        print("   Testing MediaPipe detection...")
        mp_faces = face_detector.detect_faces_mediapipe(test_image)
        print(f"   MediaPipe found {len(mp_faces)} faces")
        
        print("   Testing MTCNN detection...")
        mtcnn_faces = face_detector.detect_faces_mtcnn(test_image)
        print(f"   MTCNN found {len(mtcnn_faces)} faces")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedFaceDetector test failed: {e}")
        return False

def test_with_real_image():
    """Test face detection with a real image if available"""
    print("\n🔍 Looking for test images...")
    
    # Look for any image files in common locations
    test_locations = [
        "temp_uploads",
        "downloads", 
        "clips",
        "."
    ]
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for location in test_locations:
        if Path(location).exists():
            for file in Path(location).iterdir():
                if file.suffix.lower() in image_extensions:
                    print(f"📸 Found test image: {file}")
                    
                    try:
                        image = cv2.imread(str(file))
                        if image is not None:
                            print(f"   Image shape: {image.shape}")
                            
                            # Test MediaPipe
                            import mediapipe as mp
                            mp_face_detection = mp.solutions.face_detection
                            face_detection = mp_face_detection.FaceDetection(model_selection=1)
                            
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = face_detection.process(rgb_image)
                            
                            if results.detections:
                                print(f"   ✅ MediaPipe found {len(results.detections)} face(s)")
                            else:
                                print("   ⚠️ MediaPipe found no faces")
                            
                            face_detection.close()
                            return True
                            
                    except Exception as e:
                        print(f"   ❌ Error processing {file}: {e}")
    
    print("📸 No test images found")
    return False

def main():
    """Run all face detection tests"""
    print("🔍 Face Detection Diagnostic Tool")
    print("=" * 50)
    
    results = {}
    
    # Test basic dependencies
    results['opencv'] = test_opencv()
    results['mediapipe'] = test_mediapipe()
    results['mtcnn'] = test_mtcnn()
    results['dnn_models'] = test_dnn_face_detection()
    results['service'] = test_face_detection_service()
    
    # Test with real image
    test_with_real_image()
    
    # Summary
    print("\n📋 Test Results Summary:")
    print("-" * 30)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.upper():<15} {status}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not results['opencv']:
        print("❌ Install OpenCV: pip install opencv-python")
    
    if not results['mediapipe']:
        print("❌ Install MediaPipe: pip install mediapipe")
    
    if not results['mtcnn']:
        print("❌ Install MTCNN: pip install mtcnn")
    
    if not results['dnn_models']:
        print("❌ Download DNN models:")
        print("   • deploy.prototxt")
        print("   • res10_300x300_ssd_iter_140000_fp16.caffemodel")
    
    if all(results.values()):
        print("✅ All face detection components are working!")
    else:
        print("⚠️ Some components need attention (see above)")

if __name__ == "__main__":
    main() 