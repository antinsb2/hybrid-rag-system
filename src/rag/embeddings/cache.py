"""
Embedding cache to avoid recomputing embeddings.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings.
    Uses text hash as key to avoid recomputing.
    """
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        
        self.cache_file = self.cache_dir / "embeddings.pkl"
        self._load_cache()
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash of text for cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.memory_cache = pickle.load(f)
                print(f"Loaded cache with {len(self.memory_cache)} entries")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                self.memory_cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
            print(f"Saved cache with {len(self.memory_cache)} entries")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Embedding if found, None otherwise
        """
        key = self._compute_hash(text)
        
        if key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[key]
        
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding to store
        """
        key = self._compute_hash(text)
        self.memory_cache[key] = embedding
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.memory_cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self):
        """Clear all cache."""
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
        if self.cache_file.exists():
            self.cache_file.unlink()
