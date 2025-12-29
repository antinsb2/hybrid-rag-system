"""
Simple vector store with linear search baseline.
"""

from typing import List, Tuple, Optional
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from similarity search."""
    text: str
    score: float
    metadata: dict
    index: int


class VectorStore:
    """
    Simple vector store with linear (brute force) search.
    Good baseline for comparison.
    """
    
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
        self.dimension = None
    
    def add(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        metadata: List[dict]
    ):
        """
        Add vectors to store.
        
        Args:
            embeddings: Array of embeddings [n, dim]
            texts: List of text chunks
            metadata: List of metadata dicts
        """
        if len(embeddings) != len(texts) or len(embeddings) != len(metadata):
            raise ValueError("embeddings, texts, and metadata must have same length")
        
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        elif embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        
        self.vectors.extend(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar vectors using linear scan.
        
        Args:
            query_embedding: Query vector [dim]
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        if len(self.vectors) == 0:
            return []
        
        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        
        # Compute similarities (cosine similarity via dot product for normalized vectors)
        vectors_array = np.array(self.vectors)
        scores = np.dot(vectors_array, query_embedding)
        
        # Filter by min score
        valid_indices = np.where(scores >= min_score)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get top k
        top_indices = valid_indices[np.argsort(scores[valid_indices])[-top_k:][::-1]]
        
        results = [
            SearchResult(
                text=self.texts[idx],
                score=float(scores[idx]),
                metadata=self.metadata[idx],
                index=int(idx)
            )
            for idx in top_indices
        ]
        
        return results
    
    def save(self, file_path: str):
        """Save store to disk."""
        data = {
            'vectors': self.vectors,
            'texts': self.texts,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, file_path: str):
        """Load store from disk."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectors = data['vectors']
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.dimension = data['dimension']
    
    def get_stats(self) -> dict:
        """Get store statistics."""
        return {
            'num_vectors': len(self.vectors),
            'dimension': self.dimension,
            'memory_mb': sum(v.nbytes for v in self.vectors) / 1024 / 1024 if self.vectors else 0
        }
