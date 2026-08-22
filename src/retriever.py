"""
Retriever Module

Implements semantic retrieval with cross-encoder reranking.
"""

import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
from .embeddings import EmbeddingModel


class RetrieverWithRerank:
    """Class for semantic retrieval with cross-encoder reranking."""
    
    def __init__(self, embeddings: np.ndarray, chunks: List[str],
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the retriever.
        
        Args:
            embeddings: Pre-computed chunk embeddings
            chunks: List of text chunks
            embedding_model: Name of embedding model
            reranker_model: Name of cross-encoder reranker model
        """
        self.embeddings = embeddings
        self.chunks = chunks
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.reranker = CrossEncoder(reranker_model)
    
    def search(self, query: str, top_k: int = 5, 
               rerank_top_k: int = 10) -> List[Dict]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            rerank_top_k: Number of results to rerank
        
        Returns:
            List of dictionaries with chunk text and scores
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # Compute cosine similarity
        similarities = self.embedder.cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top-k indices before reranking
        initial_top_k_indices = np.argsort(similarities)[::-1][:rerank_top_k]
        
        # Prepare pairs for reranking
        query_chunk_pairs = [
            [query, self.chunks[idx]] 
            for idx in initial_top_k_indices
        ]
        
        # Rerank using cross-encoder
        rerank_scores = self.reranker.predict(query_chunk_pairs)
        
        # Sort by rerank scores
        reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in reranked_indices:
            original_idx = initial_top_k_indices[idx]
            results.append({
                'text': self.chunks[original_idx],
                'similarity_score': float(similarities[original_idx]),
                'rerank_score': float(rerank_scores[idx]),
                'chunk_id': int(original_idx)
            })
        
        return results
    
    def search_without_rerank(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search without reranking (faster but less accurate).
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of dictionaries with chunk text and scores
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # Compute cosine similarity
        similarities = self.embedder.cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.chunks[idx],
                'similarity_score': float(similarities[idx]),
                'chunk_id': int(idx)
            })
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Search for multiple queries in batch.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
        
        Returns:
            List of result lists, one per query
        """
        all_results = []
        
        for query in queries:
            results = self.search(query, top_k=top_k)
            all_results.append(results)
        
        return all_results


def retrieve_relevant_chunks(query: str, chunks: List[str], 
                            embeddings: np.ndarray,
                            top_k: int = 5) -> List[Dict]:
    """
    Convenience function to retrieve relevant chunks.
    
    Args:
        query: Search query
        chunks: List of text chunks
        embeddings: Pre-computed embeddings
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with scores
    """
    retriever = RetrieverWithRerank(embeddings=embeddings, chunks=chunks)
    return retriever.search(query, top_k=top_k)
