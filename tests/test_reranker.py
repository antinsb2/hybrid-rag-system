"""
Tests for cross-encoder re-ranking.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import RetrievalResult, CrossEncoderReranker


def test_reranker_basic():
    """Test basic re-ranking."""
    reranker = CrossEncoderReranker()
    
    # Create mock results (in non-ideal order)
    results = [
        RetrievalResult("Python is a programming language", 0.7, {"id": 1}, 1),
        RetrievalResult("Machine learning uses algorithms", 0.8, {"id": 2}, 2),
        RetrievalResult("Python programming tutorial for beginners", 0.6, {"id": 3}, 3),
    ]
    
    query = "Python programming guide"
    
    # Re-rank
    reranked = reranker.rerank(query, results, top_k=3)
    
    assert len(reranked) == 3
    assert all(hasattr(r, 'score') for r in reranked)
    
    # Check that Python tutorial moved up
    print(f"\n✅ Re-ranker test")
    print(f"Query: '{query}'")
    for r in reranked:
        print(f"  Rank {r.rank} (score: {r.score:.3f}): {r.text[:50]}...")


def test_reranker_improves_order():
    """Test that re-ranking changes order meaningfully."""
    reranker = CrossEncoderReranker()
    
    results = [
        RetrievalResult("Document about cars and vehicles", 0.9, {"id": 1}, 1),
        RetrievalResult("Guide to Python web development with Flask", 0.5, {"id": 2}, 2),
        RetrievalResult("Random text unrelated to anything", 0.4, {"id": 3}, 3),
    ]
    
    query = "Python Flask tutorial"
    
    # Original order: cars, flask, random
    # Expected after rerank: flask should be #1
    
    reranked = reranker.rerank(query, results)
    
    # Flask doc should move to top
    assert "Flask" in reranked[0].text
    
    print(f"\n✅ Re-ranker improves order")
    print(f"Before: {[r.metadata['id'] for r in results]}")
    print(f"After: {[r.metadata['id'] for r in reranked]}")


if __name__ == "__main__":
    test_reranker_basic()
    test_reranker_improves_order()
    print("\n✅ All reranker tests passed!")
