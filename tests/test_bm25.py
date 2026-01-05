"""
Tests for BM25 sparse retrieval.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval.bm25 import BM25


def test_bm25_basic():
    """Test basic BM25 functionality."""
    bm25 = BM25()
    
    docs = [
        "Python is a programming language",
        "Machine learning uses Python",
        "Java is also a programming language",
        "Deep learning is a subset of machine learning"
    ]
    
    bm25.add_documents(docs)
    
    assert bm25.num_docs == 4
    assert bm25.avg_doc_length > 0
    
    print(f"✅ BM25 basic: {bm25.get_stats()}")


def test_bm25_search():
    """Test BM25 search."""
    bm25 = BM25()
    
    docs = [
        "Python programming language tutorial",
        "Learn Python for data science",
        "Java programming fundamentals",
        "Machine learning with Python",
        "Deep learning neural networks"
    ]
    
    metadata = [{"id": i} for i in range(len(docs))]
    bm25.add_documents(docs, metadata)
    
    # Search for Python
    results = bm25.search("Python programming", top_k=3)
    
    assert len(results) > 0
    assert "Python" in results[0].text
    
    print(f"\n✅ BM25 search test")
    print(f"Query: 'Python programming'")
    for r in results:
        print(f"  Rank {r.rank} (score: {r.score:.2f}): {r.text}")


def test_bm25_keyword_matching():
    """Test that BM25 finds exact keywords."""
    bm25 = BM25()
    
    docs = [
        "Python version 3.10 released",
        "Machine learning algorithms",
        "Python 3.10 new features"
    ]
    
    bm25.add_documents(docs)
    
    # Search for exact version
    results = bm25.search("Python 3.10", top_k=2)
    
    assert len(results) == 2
    assert all("3.10" in r.text or "310" in r.text for r in results)
    
    print(f"\n✅ Keyword matching: Found {len(results)} docs with 'Python 3.10'")


def test_bm25_persistence():
    """Test save/load."""
    import tempfile
    
    bm25 = BM25()
    docs = ["Document one", "Document two", "Document three"]
    bm25.add_documents(docs)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        bm25.save(temp_path)
        
        # Load into new instance
        bm25_loaded = BM25()
        bm25_loaded.load(temp_path)
        
        assert bm25_loaded.num_docs == 3
        
        # Test search works
        results = bm25_loaded.search("document", top_k=2)
        assert len(results) == 2
        
        print(f"\n✅ Persistence test passed")
        
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    test_bm25_basic()
    test_bm25_search()
    test_bm25_keyword_matching()
    test_bm25_persistence()
    print("\n✅ All BM25 tests passed!")
