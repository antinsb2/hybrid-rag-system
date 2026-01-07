"""
Re-ranking using cross-encoder models.
"""

from typing import List
from sentence_transformers import CrossEncoder
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.types import RetrievalResult


class CrossEncoderReranker:
    """
    Re-rank results using a cross-encoder model.
    
    Cross-encoders jointly encode query + document for better accuracy
    than bi-encoders (which encode separately).
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Args:
            model_name: Cross-encoder model from sentence-transformers
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        
        print(f"Loaded cross-encoder: {model_name}")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return (None = all)
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, result.text] for result in results]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Create new results with cross-encoder scores
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(RetrievalResult(
                text=result.text,
                score=float(score),
                metadata={**result.metadata, "original_score": result.score},
                rank=result.rank
            ))
        
        # Sort by new scores
        reranked.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked, 1):
            result.rank = i
        
        # Return top k
        if top_k:
            return reranked[:top_k]
        
        return reranked
    
    def rerank_batch(
        self,
        queries: List[str],
        results_list: List[List[RetrievalResult]],
        top_k: int = None
    ) -> List[List[RetrievalResult]]:
        """
        Re-rank multiple result sets.
        
        Args:
            queries: List of queries
            results_list: List of result lists
            top_k: Number of results per query
            
        Returns:
            List of re-ranked result lists
        """
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_list)
        ]


class TwoStageRetrieval:
    """
    Two-stage retrieval: fast first-stage + accurate re-ranking.
    """
    
    def __init__(
        self,
        first_stage_retriever,
        reranker: CrossEncoderReranker,
        first_stage_k: int = 50
    ):
        """
        Args:
            first_stage_retriever: Dense, sparse, or hybrid retriever
            reranker: Cross-encoder reranker
            first_stage_k: How many to retrieve before re-ranking
        """
        self.first_stage = first_stage_retriever
        self.reranker = reranker
        self.first_stage_k = first_stage_k
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve with two-stage approach.
        
        Args:
            query: Search query
            top_k: Final number of results
            
        Returns:
            Re-ranked results
        """
        # Stage 1: Fast retrieval (get more candidates)
        candidates = self.first_stage.retrieve(query, top_k=self.first_stage_k)
        
        # Stage 2: Accurate re-ranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        
        return reranked
