"""
Tests for retrieval components.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import QueryProcessor


def test_query_processor():
    """Test query processing."""
    processor = QueryProcessor()
    
    query = "How do I configure SSL?"
    embedding = processor.process(query)
    
    assert embedding.shape == (384,)
    assert embedding.dtype == 'float32'
    
    print(f"✅ Query processor: embedding shape {embedding.shape}")


def test_batch_processing():
    """Test batch query processing."""
    processor = QueryProcessor()
    
    queries = [
        "What is machine learning?",
        "How to train a model?",
        "Python programming basics"
    ]
    
    embeddings = processor.process_batch(queries)
    
    assert embeddings.shape == (3, 384)
    print(f"✅ Batch processing: {embeddings.shape}")


def test_query_expansion():
    """Test query expansion."""
    processor = QueryProcessor()
    
    query = "configure SSL"
    expansions = processor.expand_query(query, num_expansions=3)
    
    assert len(expansions) <= 3
    assert query in expansions
    
    print(f"✅ Query expansion: {expansions}")


if __name__ == "__main__":
    test_query_processor()
    test_batch_processing()
    test_query_expansion()
    print("\n✅ All retrieval tests passed!")
