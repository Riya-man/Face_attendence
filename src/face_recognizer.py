"""
Face recognition module using FaceNet
"""
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

class FaceRecognizer:
    """Face recognition using FaceNet"""
    
    def __init__(self, device=None):
        """Initialize FaceNet model"""
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)
    
    def extract_embedding(self, face_tensor):
        """
        Extract 512-dimensional embedding from face tensor
        
        Args:
            face_tensor: Aligned face tensor from MTCNN
        
        Returns:
            embedding: numpy array of shape (512,)
        """
        if face_tensor is None:
            return None
        
        # Add batch dimension if needed
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def extract_embeddings_batch(self, face_tensors):
        """
        Extract embeddings from multiple face tensors
        
        Args:
            face_tensors: List of face tensors
        
        Returns:
            embeddings: List of numpy arrays
        """
        embeddings = []
        
        for face_tensor in face_tensors:
            embedding = self.extract_embedding(face_tensor)
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings
    
    def average_embeddings(self, embeddings):
        """
        Average multiple embeddings
        
        Args:
            embeddings: List of embeddings
        
        Returns:
            avg_embedding: Averaged embedding
        """
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def calculate_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: 'cosine' or 'euclidean'
        
        Returns:
            similarity: Similarity score
        """
        if metric == 'cosine':
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == 'euclidean':
            similarity = np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
        
        return similarity
    
    def compare_faces(self, embedding1, embedding2, threshold=0.7):
        """
        Compare two faces and determine if they match
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Matching threshold for cosine similarity
        
        Returns:
            is_match: Boolean indicating if faces match
            similarity: Similarity score
        """
        similarity = self.calculate_similarity(embedding1, embedding2, 'cosine')
        is_match = similarity >= threshold
        
        return is_match, similarity