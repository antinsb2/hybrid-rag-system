"""
Fusion algorithms for combining dense and sparse retrieval results.
"""

from typing import List, Dict
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.types import RetrievalResult


class FusionStrategy:
    """Base class for fusion strategies."""
    
    @staticmethod
    def fuse(
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Combine results from dense and sparse retrieval."""
        raise NotImplementedError


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF).
    
    RRF(d) = Σ 1/(k + rank_i(d))
    
    where k is a constant (typically 60) and rank_i(d) is the rank
    of document d in result list i.
    """
    
    @staticmethod
    def fuse(
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine using RRF.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: RRF constant (default 60)
            
        Returns:
            Fused and re-ranked results
        """
        # Create document index by text (assuming text is unique identifier)
        doc_scores = defaultdict(float)
        doc_data = {}
        
        # Process dense results
        for result in dense_results:
            rrf_score = 1.0 / (k + result.rank)
            doc_scores[result.text] += rrf_score
            if result.text not in doc_data:
                doc_data[result.text] = result
        
        # Process sparse results
        for result in sparse_results:
            rrf_score = 1.0 / (k + result.rank)
            doc_scores[result.text] += rrf_score
            if result.text not in doc_data:
                doc_data[result.text] = result
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for rank, (text, score) in enumerate(sorted_docs, 1):
            original = doc_data[text]
            fused_results.append(RetrievalResult(
                text=text,
                score=score,
                metadata=original.metadata,
                rank=rank
            ))
        
        return fused_results


class WeightedFusion(FusionStrategy):
    """
    Weighted combination of dense and sparse scores.
    
    Score(d) = α * dense_score(d) + (1-α) * sparse_score(d)
    """
    
    @staticmethod
    def fuse(
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Combine using weighted scores.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            alpha: Weight for dense scores (0-1)
            
        Returns:
            Fused and re-ranked results
        """
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            
            scores = {r.text: r.score for r in results}
            min_score = min(scores.values())
            max_score = max(scores.values())
            
            if max_score == min_score:
                return {text: 0.5 for text in scores}
            
            return {
                text: (score - min_score) / (max_score - min_score)
                for text, score in scores.items()
            }
        
        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)
        
        # Get all unique documents
        all_docs = set(dense_normalized.keys()) | set(sparse_normalized.keys())
        
        # Compute weighted scores
        doc_scores = {}
        doc_data = {}
        
        for result in dense_results:
            doc_data[result.text] = result
        for result in sparse_results:
            if result.text not in doc_data:
                doc_data[result.text] = result
        
        for text in all_docs:
            dense_score = dense_normalized.get(text, 0.0)
            sparse_score = sparse_normalized.get(text, 0.0)
            
            doc_scores[text] = alpha * dense_score + (1 - alpha) * sparse_score
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for rank, (text, score) in enumerate(sorted_docs, 1):
            original = doc_data[text]
            fused_results.append(RetrievalResult(
                text=text,
                score=score,
                metadata=original.metadata,
                rank=rank
            ))
        
        return fused_results


class SimpleCombination(FusionStrategy):
    """
    Simple combination: take top results from both, deduplicate, sort by original scores.
    """
    
    @staticmethod
    def fuse(
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Simple combination strategy.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Combined results
        """
        # Collect all results
        seen_texts = set()
        combined = []
        
        # Add dense results first (prioritize semantic matching)
        for result in dense_results:
            if result.text not in seen_texts:
                seen_texts.add(result.text)
                combined.append(result)
        
        # Add sparse results that aren't already included
        for result in sparse_results:
            if result.text not in seen_texts:
                seen_texts.add(result.text)
                combined.append(result)
        
        # Re-rank
        for rank, result in enumerate(combined, 1):
            result.rank = rank
        
        return combined
