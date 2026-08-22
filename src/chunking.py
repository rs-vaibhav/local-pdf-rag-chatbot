"""
Text Chunking Module

Implements recursive text chunking strategies for RAG.
"""

from typing import List, Optional
import re


class RecursiveChunker:
    """Class for recursively chunking text into smaller segments."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive strategy.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List of text chunks
        """
        # Try splitting by paragraphs first
        chunks = self._split_by_separator(text, '\n\n')
        
        # If chunks are still too large, split by sentences
        chunks = self._further_split(chunks, ['.', '!', '?'])
        
        # If still too large, split by words
        chunks = self._further_split(chunks, [' '])
        
        # Finally, enforce hard chunk size limit
        chunks = self._enforce_chunk_size(chunks)
        
        return chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a given separator."""
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk += part + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part + separator
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _further_split(self, chunks: List[str], separators: List[str]) -> List[str]:
        """Further split chunks that exceed chunk_size."""
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                # Try to split by separators
                split_success = False
                for sep in separators:
                    sub_chunks = self._split_by_separator(chunk, sep)
                    if any(len(c) <= self.chunk_size for c in sub_chunks):
                        result.extend(sub_chunks)
                        split_success = True
                        break
                
                # If splitting failed, keep the original chunk
                if not split_success:
                    result.append(chunk)
        
        return result
    
    def _enforce_chunk_size(self, chunks: List[str]) -> List[str]:
        """Enforce maximum chunk size with overlap."""
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                # Hard split with overlap
                start = 0
                while start < len(chunk):
                    end = start + self.chunk_size
                    result.append(chunk[start:end])
                    start = end - self.chunk_overlap
        
        return result
    
    def split_with_metadata(self, text: str, source: str = "") -> List[dict]:
        """
        Split text and include metadata for each chunk.
        
        Args:
            text: Input text to chunk
            source: Source identifier (e.g., filename)
        
        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = self.split(text)
        
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'chunk_id': i,
                'source': source,
                'start_char': sum(len(c) for c in chunks[:i]),
                'end_char': sum(len(c) for c in chunks[:i+1]),
                'total_chunks': len(chunks)
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    """
    Convenience function to chunk text.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.split(text)
