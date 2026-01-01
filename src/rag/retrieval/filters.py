"""
Filtering and post-processing for retrieval results.
"""

from typing import List, Callable, Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.dense_retriever import RetrievalResult


class ResultFilter:
    """
    Filter retrieval results based on various criteria.
    """
    
    @staticmethod
    def by_score(
        results: List[RetrievalResult],
        min_score: float
    ) -> List[RetrievalResult]:
        """
        Filter by minimum similarity score.
        
        Args:
            results: List of results
            min_score: Minimum score threshold
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.score >= min_score]
    
    @staticmethod
    def by_metadata(
        results: List[RetrievalResult],
        key: str,
        value: any
    ) -> List[RetrievalResult]:
        """
        Filter by metadata field.
        
        Args:
            results: List of results
            key: Metadata key
            value: Expected value
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.metadata.get(key) == value]
    
    @staticmethod
    def by_source(
        results: List[RetrievalResult],
        sources: List[str]
    ) -> List[RetrievalResult]:
        """
        Filter by document source.
        
        Args:
            results: List of results
            sources: Allowed sources
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.metadata.get('source') in sources]
    
    @staticmethod
    def deduplicate(
        results: List[RetrievalResult],
        similarity_threshold: float = 0.95
    ) -> List[RetrievalResult]:
        """
        Remove near-duplicate results.
        
        Args:
            results: List of results
            similarity_threshold: Cosine similarity threshold for duplicates
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        unique_results = [results[0]]
        
        for result in results[1:]:
            # Check if too similar to any existing result
            is_duplicate = False
            
            for existing in unique_results:
                # Simple text-based deduplication
                text_similarity = _text_similarity(result.text, existing.text)
                if text_similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    @staticmethod
    def by_custom(
        results: List[RetrievalResult],
        filter_fn: Callable[[RetrievalResult], bool]
    ) -> List[RetrievalResult]:
        """
        Filter by custom function.
        
        Args:
            results: List of results
            filter_fn: Function that returns True to keep result
            
        Returns:
            Filtered results
        """
        return [r for r in results if filter_fn(r)]


def _text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity based on character overlap.
    """
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Jaccard similarity on character trigrams
    def get_trigrams(text):
        return set(text[i:i+3] for i in range(len(text)-2))
    
    t1 = get_trigrams(text1)
    t2 = get_trigrams(text2)
    
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    
    return intersection / union if union > 0 else 0.0


class ResultRanker:
    """
    Re-rank retrieval results to improve quality.
    """
    
    @staticmethod
    def by_score(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Sort by similarity score (already done by retriever, but explicit).
        """
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(sorted_results, 1):
            result.rank = i
        
        return sorted_results
    
    @staticmethod
    def boost_by_metadata(
        results: List[RetrievalResult],
        key: str,
        boost_value: any,
        boost_factor: float = 1.2
    ) -> List[RetrievalResult]:
        """
        Boost scores for results matching metadata criteria.
        
        Args:
            results: List of results
            key: Metadata key
            boost_value: Value to boost
            boost_factor: Multiplier for score
            
        Returns:
            Re-ranked results
        """
        for result in results:
            if result.metadata.get(key) == boost_value:
                result.score *= boost_factor
        
        return ResultRanker.by_score(results)
    
    @staticmethod
    def boost_by_recency(
        results: List[RetrievalResult],
        date_key: str = 'created',
        boost_factor: float = 1.1
    ) -> List[RetrievalResult]:
        """
        Boost more recent documents.
        
        Args:
            results: List of results
            date_key: Metadata key for date
            boost_factor: Multiplier for recent docs
            
        Returns:
            Re-ranked results
        """
        now = datetime.now()
        
        for result in results:
            date_str = result.metadata.get(date_key)
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(date_str)
                    days_old = (now - doc_date).days
                    
                    # Boost if less than 30 days old
                    if days_old < 30:
                        result.score *= boost_factor
                except:
                    pass
        
        return ResultRanker.by_score(results)
