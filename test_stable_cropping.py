#!/usr/bin/env python3
"""
Test script for improved vertical cropping stability with multiple faces
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.vertical_crop import VerticalCropService

def create_test_frames_with_multiple_faces():
    """
    Create test frames with simulated multiple faces to test stability
    """
    frames = []
    width, height = 1920, 1080
    
    # Create 100 test frames
    for i in range(100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Face 1: Main speaker (moves slightly)
        face1_x = 600 + int(5 * np.sin(i * 0.1))  # Small oscillation
        face1_y = 300 + int(3 * np.cos(i * 0.15))
        cv2.rectangle(frame, (face1_x, face1_y), (face1_x + 120, face1_y + 150), (100, 150, 200), -1)
        cv2.putText(frame, "Speaker 1", (face1_x, face1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Face 2: Secondary person (larger movements)
        face2_x = 1200 + int(20 * np.sin(i * 0.2))  # Larger oscillation
        face2_y = 350 + int(15 * np.cos(i * 0.25))
        cv2.rectangle(frame, (face2_x, face2_y), (face2_x + 100, face2_y + 130), (150, 100, 200), -1)
        cv2.putText(frame, "Person 2", (face2_x, face2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sometimes add a third face to test switching
        if i > 30 and i < 70:
            face3_x = 400 + int(10 * np.sin(i * 0.3))
            face3_y = 400 + int(8 * np.cos(i * 0.2))
            cv2.rectangle(frame, (face3_x, face3_y), (face3_x + 90, face3_y + 120), (200, 100, 150), -1)
            cv2.putText(frame, "Person 3", (face3_x, face3_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames

def test_cropping_stability():
    """
    Test the stability of the vertical cropping service
    """
    print("🧪 Testing vertical cropping stability with multiple faces...")
    
    # Create test frames
    test_frames = create_test_frames_with_multiple_faces()
    print(f"✅ Created {len(test_frames)} test frames")
    
    # Initialize cropping service
    crop_service = VerticalCropService()
    print("✅ Initialized VerticalCropService")
    
    # Process frames and track crop centers
    crop_centers = []
    speaker_switches = 0
    previous_speaker = None
    
    for i, frame in enumerate(test_frames):
        # Find active speaker
        speaker_box = crop_service.find_active_speaker(frame)
        
        if speaker_box:
            # Calculate crop center that would be used
            face_center_x = (speaker_box[0] + speaker_box[2]) // 2
            face_center_y = (speaker_box[1] + speaker_box[3]) // 2
            
            # Apply smoothing to get actual crop center
            crop_center = crop_service._smooth_crop_center((face_center_x, face_center_y))
            crop_centers.append(crop_center)
            
            # Track speaker switches
            current_speaker = f"{speaker_box[0]}-{speaker_box[1]}"  # Simple ID based on position
            if previous_speaker and previous_speaker != current_speaker:
                speaker_switches += 1
            previous_speaker = current_speaker
            
            if i % 20 == 0:
                print(f"Frame {i:3d}: Crop center at ({crop_center[0]:4d}, {crop_center[1]:4d}), "
                      f"Speaker box: ({speaker_box[0]:4d}, {speaker_box[1]:4d}, {speaker_box[2]:4d}, {speaker_box[3]:4d})")
        else:
            crop_centers.append(None)
            if i % 20 == 0:
                print(f"Frame {i:3d}: No speaker detected")
    
    # Analyze stability
    valid_centers = [c for c in crop_centers if c is not None]
    if len(valid_centers) > 1:
        # Calculate movement between frames
        movements = []
        for i in range(1, len(valid_centers)):
            if crop_centers[i] and crop_centers[i-1]:
                dx = crop_centers[i][0] - crop_centers[i-1][0]
                dy = crop_centers[i][1] - crop_centers[i-1][1]
                movement = np.sqrt(dx*dx + dy*dy)
                movements.append(movement)
        
        avg_movement = np.mean(movements) if movements else 0
        max_movement = np.max(movements) if movements else 0
        
        print(f"\n📊 Stability Analysis:")
        print(f"   • Total frames processed: {len(test_frames)}")
        print(f"   • Frames with detected speaker: {len(valid_centers)}")
        print(f"   • Speaker switches: {speaker_switches}")
        print(f"   • Average frame-to-frame movement: {avg_movement:.1f} pixels")
        print(f"   • Maximum frame-to-frame movement: {max_movement:.1f} pixels")
        
        # Evaluation
        if avg_movement < 5.0 and max_movement < 20.0 and speaker_switches < 5:
            print("✅ EXCELLENT: Cropping is very stable!")
        elif avg_movement < 10.0 and max_movement < 40.0 and speaker_switches < 10:
            print("✅ GOOD: Cropping is reasonably stable")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Cropping shows instability")
    
    print("\n🎯 Test completed!")

def main():
    """
    Main test function
    """
    print("🚀 Starting stability test for vertical cropping...")
    
    try:
        test_cropping_stability()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 