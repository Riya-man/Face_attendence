"""
Main attendance system module
"""
import cv2
import time
from PIL import Image
from datetime import datetime, timedelta

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .database import AttendanceDatabase
from .liveness_detection import LivenessDetector

class AttendanceSystem:
    """Complete face attendance system"""
    
    def __init__(self, db_path, device=None):
        """Initialize attendance system"""
        self.detector = FaceDetector(device)
        self.recognizer = FaceRecognizer(device)
        self.database = AttendanceDatabase(db_path)
        self.liveness = LivenessDetector()
    
    def register_user(self, name, employee_id, department, num_images=5, delay=2):
        """
        Register new user
        
        Returns:
            user_id or None if failed
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None, "Could not open webcam"
        
        captured_faces = []
        
        try:
            for i in range(num_images):
                time.sleep(delay)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Extract face
                face_tensor = self.detector.extract_face(frame)
                
                if face_tensor is None:
                    continue
                
                captured_faces.append(face_tensor)
        
        finally:
            cap.release()
        
        if len(captured_faces) < 3:
            return None, f"Only captured {len(captured_faces)} images, need at least 3"
        
        # Extract embeddings
        embeddings = self.recognizer.extract_embeddings_batch(captured_faces)
        avg_embedding = self.recognizer.average_embeddings(embeddings)
        
        # Save to database
        user_id = self.database.add_user(name, employee_id, department, avg_embedding, len(captured_faces))
        
        if user_id:
            return user_id, "Success"
        return None, "User already exists"
    
    def mark_attendance(self, liveness_check=True, threshold=0.7):
        """
        Mark attendance with face recognition
        
        Returns:
            result dict or None
        """
        # Liveness check
        if liveness_check:
            is_live, motion = self.liveness.detect_motion()
            if not is_live:
                return None, "Liveness check failed"
        
        # Capture face
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "Failed to capture image"
        
        # Detect face
        boxes, probs = self.detector.detect_face(frame)
        
        if boxes is None or len(boxes) == 0:
            return None, "No face detected"
        
        # Extract face and embedding
        face_tensor = self.detector.extract_face(frame)
        
        if face_tensor is None:
            return None, "Failed to extract face"
        
        embedding = self.recognizer.extract_embedding(face_tensor)
        
        # Match with database
        matched_user = self.database.find_matching_user(embedding, threshold)
        
        if matched_user is None:
            return None, "User not recognized"
        
        # Determine punch type
        last_attendance = self.database.get_last_attendance(matched_user['user_id'])
        
        if last_attendance is None:
            punch_type = 'IN'
        else:
            last_time = datetime.strptime(last_attendance['timestamp'], '%Y-%m-%d %H:%M:%S')
            time_diff = datetime.now() - last_time
            
            if time_diff < timedelta(minutes=1):
                return None, "Duplicate punch attempt"
            
            punch_type = 'OUT' if last_attendance['punch_type'] == 'IN' else 'IN'
        
        # Mark attendance
        attendance_id = self.database.mark_attendance(
            matched_user['user_id'],
            punch_type,
            matched_user['similarity']
        )
        
        return {
            'attendance_id': attendance_id,
            'user': matched_user,
            'punch_type': punch_type,
            'timestamp': datetime.now(),
            'frame': frame
        }, "Success"