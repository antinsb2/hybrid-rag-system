"""
Sparse retrieval using BM25.
"""

from typing import List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.bm25 import BM25
from rag.retrieval.types import RetrievalResult


class SparseRetriever:
    """
    Sparse retrieval system using BM25.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: BM25 term frequency parameter
            b: BM25 length normalization parameter
        """
        self.bm25 = BM25(k1=k1, b=b)
        self.is_indexed = False
    
    def index(self, texts: List[str], metadata: List[dict] = None):
        """
        Index documents for sparse retrieval.
        
        Args:
            texts: Document texts
            metadata: Document metadata
        """
        self.bm25.add_documents(texts, metadata)
        self.is_indexed = True
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold
            
        Returns:
            List of retrieval results
        """
        if not self.is_indexed:
            raise RuntimeError("No documents indexed. Call index() first.")
        
        return self.bm25.search(query, top_k=top_k, min_score=min_score)
    
    def save(self, filepath: str):
        """Save retriever state."""
        self.bm25.save(filepath)
    
    def load(self, filepath: str):
        """Load retriever state."""
        self.bm25.load(filepath)
        self.is_indexed = True
    
    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            'is_indexed': self.is_indexed,
            'bm25': self.bm25.get_stats()
        }
