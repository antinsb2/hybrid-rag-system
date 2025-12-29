"""
HNSW (Hierarchical Navigable Small World) index for fast approximate search.
"""

from typing import List, Optional
import numpy as np
import hnswlib
from pathlib import Path

from .vector_store import SearchResult


class HNSWIndex:
    """
    Fast approximate nearest neighbor search using HNSW.
    Much faster than linear search for large datasets.
    """
    
    def __init__(
        self, 
        dimension: int,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16
    ):
        """
        Args:
            dimension: Vector dimension
            max_elements: Maximum number of vectors
            ef_construction: Construction time/accuracy tradeoff (higher = better but slower)
            M: Number of bi-directional links (higher = better recall but more memory)
        """
        self.dimension = dimension
        self.max_elements = max_elements
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        
        # Set query time accuracy
        self.index.set_ef(50)  # Higher = more accurate but slower
        
        # Store metadata
        self.texts = []
        self.metadata = []
        self.current_count = 0
    
    def add(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        metadata: List[dict]
    ):
        """
        Add vectors to index.
        
        Args:
            embeddings: Array of embeddings [n, dim]
            texts: List of text chunks
            metadata: List of metadata dicts
        """
        if len(embeddings) != len(texts) or len(embeddings) != len(metadata):
            raise ValueError("embeddings, texts, and metadata must have same length")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        
        n = len(embeddings)
        
        if self.current_count + n > self.max_elements:
            # Resize index
            new_max = max(self.max_elements * 2, self.current_count + n)
            self.index.resize_index(new_max)
            self.max_elements = new_max
        
        # Add to HNSW index
        ids = np.arange(self.current_count, self.current_count + n)
        self.index.add_items(embeddings, ids)
        
        # Store text and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        self.current_count += n
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector [dim]
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        if self.current_count == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        labels, distances = self.index.knn_query(query_embedding, k=min(top_k, self.current_count))
        
        # Convert distances to similarities (cosine distance -> similarity)
        similarities = 1 - distances[0]
        
        results = [
            SearchResult(
                text=self.texts[label],
                score=float(sim),
                metadata=self.metadata[label],
                index=int(label)
            )
            for label, sim in zip(labels[0], similarities)
        ]
        
        return results
    
    def save(self, directory: str):
        """Save index and metadata."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save HNSW index
        self.index.save_index(str(path / "index.bin"))
        
        # Save metadata
        import pickle
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'current_count': self.current_count,
                'max_elements': self.max_elements
            }, f)
    
    def load(self, directory: str):
        """Load index and metadata."""
        path = Path(directory)
        
        # Load metadata first
        import pickle
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.dimension = data['dimension']
        self.current_count = data['current_count']
        self.max_elements = data['max_elements']
        
        # Load HNSW index
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.index.load_index(str(path / "index.bin"), max_elements=self.max_elements)
        self.index.set_ef(50)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'num_vectors': self.current_count,
            'dimension': self.dimension,
            'max_elements': self.max_elements,
            'index_type': 'HNSW'
        }
