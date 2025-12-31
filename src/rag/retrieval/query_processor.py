"""
Query processing for retrieval.
"""

from typing import List, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.embeddings import EmbeddingModel


class QueryProcessor:
    """
    Process user queries for retrieval.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Embedding model to use
        """
        self.model = EmbeddingModel(model_name)
    
    def process(self, query: str) -> np.ndarray:
        """
        Process a single query.
        
        Args:
            query: User query text
            
        Returns:
            Query embedding
        """
        # Basic preprocessing
        query = query.strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Convert to embedding
        embedding = self.model.encode(query, normalize=True)[0]
        
        return embedding
    
    def process_batch(self, queries: List[str]) -> np.ndarray:
        """
        Process multiple queries at once.
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of query embeddings [n_queries, dimension]
        """
        queries = [q.strip() for q in queries if q.strip()]
        
        if not queries:
            raise ValueError("No valid queries provided")
        
        embeddings = self.model.encode(queries, normalize=True, batch_size=32)
        
        return embeddings
    
    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Generate query variations for better retrieval.
        
        Args:
            query: Original query
            num_expansions: Number of variations to generate
            
        Returns:
            List of query variations including original
        """
        # Simple expansion strategies
        expansions = [query]
        
        # Add question variations
        if not query.endswith('?'):
            expansions.append(query + '?')
        
        # Add "how to" variation if not present
        if not query.lower().startswith(('how', 'what', 'why', 'when', 'where')):
            expansions.append(f"How to {query}")
        
        return expansions[:num_expansions]
