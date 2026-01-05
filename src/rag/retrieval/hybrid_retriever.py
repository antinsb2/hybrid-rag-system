"""
Hybrid retriever combining dense and sparse retrieval.
"""

from typing import List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.types import RetrievalResult
from rag.retrieval.fusion import ReciprocalRankFusion, WeightedFusion


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse approaches.
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_method: str = "rrf"
    ):
        """
        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            fusion_method: Fusion strategy ("rrf", "weighted", "simple")
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of final results
            dense_weight: Weight for dense scores (for weighted fusion)
            rrf_k: RRF constant (for RRF fusion)
            
        Returns:
            Fused retrieval results
        """
        # Get results from both retrievers (get more than needed for fusion)
        retrieve_k = top_k * 2
        
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieve_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieve_k)
        
        # Fuse results
        if self.fusion_method == "rrf":
            fused = ReciprocalRankFusion.fuse(dense_results, sparse_results, k=rrf_k)
        elif self.fusion_method == "weighted":
            fused = WeightedFusion.fuse(dense_results, sparse_results, alpha=dense_weight)
        else:  # simple
            from rag.retrieval.fusion import SimpleCombination
            fused = SimpleCombination.fuse(dense_results, sparse_results)
        
        # Return top k
        return fused[:top_k]
    
    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            'fusion_method': self.fusion_method,
            'dense': self.dense_retriever.get_stats(),
            'sparse': self.sparse_retriever.get_stats()
        }
