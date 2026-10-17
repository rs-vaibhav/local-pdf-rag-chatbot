"""
Embedding Model Module

Generates vector embeddings for text chunks using sentence transformers.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path


class EmbeddingModel:
    """Class for generating text embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, 'get_embedding_dimension'):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]
    
    def cosine_similarity(self, embeddings1: np.ndarray, 
                         embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
        
        Returns:
            Cosine similarity matrix
        """
        # Normalize embeddings
        norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-9)
        norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-9)
        
        # Compute cosine similarity
        similarity = np.dot(norm1, norm2.T)
        
        return similarity
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], 
                       output_path: str) -> None:
        """
        Save embeddings and corresponding texts to disk.
        
        Args:
            embeddings: Embedding array
            texts: List of original texts
            output_path: Path to save the file
        """
        data = {
            'embeddings': embeddings,
            'texts': texts,
            'model_name': self.model_name
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_embeddings(self, input_path: str) -> tuple:
        """
        Load embeddings and texts from disk.
        
        Args:
            input_path: Path to the saved file
        
        Returns:
            Tuple of (embeddings, texts, model_name)
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['texts'], data['model_name']
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.model.max_seq_length
        }


def generate_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2',
                       show_progress: bool = True) -> np.ndarray:
    """
    Convenience function to generate embeddings.
    
    Args:
        texts: List of texts to embed
        model_name: Model name to use
        show_progress: Whether to show progress bar
    
    Returns:
        Numpy array of embeddings
    """
    embedder = EmbeddingModel(model_name=model_name)
    return embedder.encode(texts, show_progress=show_progress)
