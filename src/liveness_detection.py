"""
Liveness detection module
"""
import cv2
import numpy as np
import time

class LivenessDetector:
    """Motion-based liveness detection"""
    
    def __init__(self):
        """Initialize liveness detector"""
        pass
    
    def detect_motion(self, num_frames=15, motion_threshold=800):
        """
        Detect motion-based liveness
        
        Args:
            num_frames: Number of frames to analyze
            motion_threshold: Minimum motion to consider live
        
        Returns:
            is_live: Boolean indicating if liveness detected
            avg_motion: Average motion value
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return False, 0
        
        previous_frame = None
        total_motion = 0
        frame_count = 0
        
        try:
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if previous_frame is not None:
                    # Compute difference
                    frame_diff = cv2.absdiff(previous_frame, gray)
                    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                    
                    # Calculate motion
                    motion = np.sum(thresh)
                    total_motion += motion
                    frame_count += 1
                
                previous_frame = gray
                time.sleep(0.05)
        
        finally:
            cap.release()
        
        avg_motion = total_motion / frame_count if frame_count > 0 else 0
        is_live = avg_motion > motion_threshold
        
        return is_live, avg_motion