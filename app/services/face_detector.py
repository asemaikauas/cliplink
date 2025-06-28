#!/usr/bin/env python3
"""
Advanced Face Detection for Video Cropping
Uses MediaPipe as primary detector with MTCNN fallback
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
import mediapipe as mp

# Optional MTCNN fallback
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

from .video_cropper import FaceDetection

logger = logging.getLogger(__name__)

class AdvancedFaceDetector:
    """Advanced face detection using MediaPipe with MTCNN fallback"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = None
        self.face_mesh = None
        self.mtcnn_detector = None
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize MediaPipe and MTCNN detectors"""
        try:
            # MediaPipe Face Detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            
            # MediaPipe Face Mesh for mouth detection
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("✅ MediaPipe face detection initialized")
            
        except Exception as e:
            logger.error(f"❌ MediaPipe initialization failed: {e}")
        
        # MTCNN fallback
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn_detector = MTCNN(min_face_size=40)
                logger.info("✅ MTCNN fallback detector initialized")
            except Exception as e:
                logger.warning(f"MTCNN initialization failed: {e}")
        else:
            logger.warning("⚠️ MTCNN not available - no fallback face detection")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe with MTCNN fallback"""
        faces = []
        
        # Try MediaPipe first
        if self.face_detection:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        confidence = detection.score[0]
                        
                        # Convert to absolute coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        faces.append(FaceDetection(x, y, width, height, confidence))
                
                return faces
                
            except Exception as e:
                logger.warning(f"MediaPipe detection failed: {e}")
        
        # Fallback to MTCNN
        if self.mtcnn_detector and not faces:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.mtcnn_detector.detect_faces(rgb_frame)
                
                for detection in detections:
                    bbox = detection['box']
                    confidence = detection['confidence']
                    
                    faces.append(FaceDetection(
                        x=bbox[0], y=bbox[1], 
                        width=bbox[2], height=bbox[3], 
                        confidence=confidence
                    ))
                
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}")
        
        return faces
    
    def get_mouth_activity(self, frame: np.ndarray, face_box: FaceDetection) -> float:
        """Detect mouth movement for speaker identification"""
        if not self.face_mesh:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Use mouth landmarks to detect opening
                landmarks = results.multi_face_landmarks[0]
                
                # Get mouth corner and top/bottom lip landmarks
                mouth_top = landmarks.landmark[13]
                mouth_bottom = landmarks.landmark[14]
                
                # Calculate mouth opening (simplified)
                mouth_opening = abs(mouth_top.y - mouth_bottom.y)
                return mouth_opening
                
        except Exception as e:
            logger.warning(f"Mouth activity detection failed: {e}")
        
        return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        if self.face_detection:
            self.face_detection.close()
        if self.face_mesh:
            self.face_mesh.close() 