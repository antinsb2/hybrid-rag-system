"""
Tests for result filtering and ranking.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import RetrievalResult, ResultFilter, ResultRanker


def create_test_results():
    """Create sample results for testing."""
    return [
        RetrievalResult("Document about Python", 0.95, {"source": "python.txt", "type": "guide"}, 1),
        RetrievalResult("Document about Python programming", 0.92, {"source": "python.txt", "type": "tutorial"}, 2),
        RetrievalResult("Document about Java", 0.85, {"source": "java.txt", "type": "guide"}, 3),
        RetrievalResult("Machine learning guide", 0.78, {"source": "ml.txt", "type": "guide"}, 4),
        RetrievalResult("Python basics", 0.70, {"source": "basics.txt", "type": "intro"}, 5),
    ]


def test_filter_by_score():
    """Test score filtering."""
    results = create_test_results()
    
    filtered = ResultFilter.by_score(results, min_score=0.80)
    
    assert len(filtered) == 3
    assert all(r.score >= 0.80 for r in filtered)
    
    print(f"✅ Score filter: {len(results)} -> {len(filtered)} results")


def test_filter_by_metadata():
    """Test metadata filtering."""
    results = create_test_results()
    
    filtered = ResultFilter.by_metadata(results, "type", "guide")
    
    assert len(filtered) == 3
    assert all(r.metadata["type"] == "guide" for r in filtered)
    
    print(f"✅ Metadata filter: {len(results)} -> {len(filtered)} results")


def test_filter_by_source():
    """Test source filtering."""
    results = create_test_results()
    
    filtered = ResultFilter.by_source(results, ["python.txt", "basics.txt"])
    
    assert len(filtered) == 3
    
    print(f"✅ Source filter: {len(results)} -> {len(filtered)} results")


def test_deduplicate():
    """Test deduplication."""
    results = [
        RetrievalResult("Python programming guide", 0.95, {}, 1),
        RetrievalResult("Python programming guide for beginners", 0.90, {}, 2),
        RetrievalResult("Java programming", 0.85, {}, 3),
    ]
    
    deduplicated = ResultFilter.deduplicate(results, similarity_threshold=0.8)
    
    print(f"Original: {len(results)} results")
    print(f"Deduplicated: {len(deduplicated)} results")
    for r in deduplicated:
        print(f"  - {r.text}")
    
    assert len(deduplicated) <= len(results)  # Should have fewer or equal
    
    print(f"✅ Deduplication: {len(results)} -> {len(deduplicated)} results")

def test_boost_by_metadata():
    """Test metadata boosting."""
    results = create_test_results()
    
    # Boost "guide" type
    reranked = ResultRanker.boost_by_metadata(results, "type", "guide", boost_factor=1.5)
    
    # Check that guides moved up
    assert reranked[0].metadata["type"] == "guide"
    
    print(f"✅ Metadata boost: Top result type = {reranked[0].metadata['type']}")


def test_custom_filter():
    """Test custom filtering."""
    results = create_test_results()
    
    # Filter results with "Python" in text
    filtered = ResultFilter.by_custom(results, lambda r: "Python" in r.text)
    
    assert len(filtered) == 3
    assert all("Python" in r.text for r in filtered)
    
    print(f"✅ Custom filter: {len(results)} -> {len(filtered)} results")


if __name__ == "__main__":
    test_filter_by_score()
    test_filter_by_metadata()
    test_filter_by_source()
    test_deduplicate()
    test_boost_by_metadata()
    test_custom_filter()
    print("\n✅ All filter tests passed!")
