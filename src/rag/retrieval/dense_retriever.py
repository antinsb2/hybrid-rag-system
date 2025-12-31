"""
Dense retrieval using embeddings.
"""

from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.indexing import VectorStore, HNSWIndex, SearchResult
from rag.retrieval.query_processor import QueryProcessor


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    text: str
    score: float
    metadata: dict
    rank: int


class DenseRetriever:
    """
    Dense retrieval system combining query processing and vector search.
    """
    
    def __init__(
        self,
        index: Union[VectorStore, HNSWIndex],
        use_query_expansion: bool = False
    ):
        """
        Args:
            index: Vector index to search
            use_query_expansion: Whether to expand queries
        """
        self.index = index
        self.query_processor = QueryProcessor()
        self.use_query_expansion = use_query_expansion
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        # Process query
        if self.use_query_expansion:
            # Use query expansion
            expanded = self.query_processor.expand_query(query, num_expansions=3)
            query_embeddings = self.query_processor.process_batch(expanded)
            
            # Average embeddings
            query_embedding = np.mean(query_embeddings, axis=0)
        else:
            query_embedding = self.query_processor.process(query)
        
        # Search index
        if isinstance(self.index, VectorStore):
            results = self.index.search(query_embedding, top_k=top_k, min_score=min_score)
        else:  # HNSWIndex
            results = self.index.search(query_embedding, top_k=top_k)
            # Filter by min_score
            results = [r for r in results if r.score >= min_score]
        
        # Convert to RetrievalResult with rank
        retrieval_results = [
            RetrievalResult(
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                rank=i+1
            )
            for i, r in enumerate(results)
        ]
        
        return retrieval_results
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        return [self.retrieve(q, top_k=top_k) for q in queries]
    
    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "index_type": type(self.index).__name__,
            "index_stats": self.index.get_stats(),
            "query_expansion": self.use_query_expansion
        }
