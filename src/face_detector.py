"""
Face detection module using MTCNN
"""
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    """Face detection using MTCNN"""
    
    def __init__(self, device=None):
        """Initialize MTCNN face detector"""
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False
        )
    
    def detect_face(self, image, return_landmarks=False):
        """
        Detect face in image
        
        Args:
            image: PIL Image or numpy array
            return_landmarks: Whether to return facial landmarks
        
        Returns:
            boxes, probs, landmarks (if requested)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if return_landmarks:
            return boxes, probs, landmarks
        return boxes, probs
    
    def extract_face(self, image):
        """
        Extract aligned face tensor from image
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            face_tensor: Aligned face as tensor (3, 160, 160)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        face_tensor = self.mtcnn(image)
        return face_tensor
    
    def is_face_detected(self, image, min_confidence=0.9):
        """
        Check if face is detected with minimum confidence
        
        Args:
            image: PIL Image or numpy array
            min_confidence: Minimum detection confidence
        
        Returns:
            bool: True if face detected with sufficient confidence
        """
        boxes, probs = self.detect_face(image)
        
        if boxes is None or len(boxes) == 0:
            return False
        
        return probs[0] >= min_confidence