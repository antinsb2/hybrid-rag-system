"""
Tests for vector indexing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
from rag.indexing import VectorStore


def test_vector_store_basic():
    """Test basic vector store operations."""
    store = VectorStore()
    
    # Create test data
    embeddings = np.random.rand(100, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
    texts = [f"Document {i}" for i in range(100)]
    metadata = [{"id": i} for i in range(100)]
    
    # Add to store
    store.add(embeddings, texts, metadata)
    
    assert store.get_stats()['num_vectors'] == 100
    assert store.get_stats()['dimension'] == 384
    
    print(f"✅ Vector store basic: {store.get_stats()}")


def test_vector_store_search():
    """Test search functionality."""
    store = VectorStore()
    
    # Create test data
    embeddings = np.random.rand(100, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    texts = [f"Document {i}" for i in range(100)]
    metadata = [{"id": i} for i in range(100)]
    
    store.add(embeddings, texts, metadata)
    
    # Search with first embedding
    query = embeddings[0]
    results = store.search(query, top_k=5)
    
    assert len(results) == 5
    assert results[0].text == "Document 0"  # Should find itself
    assert results[0].score > 0.99  # Should be nearly 1.0
    
    print(f"✅ Search test: Found {len(results)} results, top score: {results[0].score:.3f}")


def test_vector_store_persistence():
    """Test save/load."""
    import tempfile
    
    store = VectorStore()
    
    embeddings = np.random.rand(50, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    texts = [f"Doc {i}" for i in range(50)]
    metadata = [{"id": i} for i in range(50)]
    
    store.add(embeddings, texts, metadata)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        store.save(temp_path)
        
        # Load into new store
        new_store = VectorStore()
        new_store.load(temp_path)
        
        assert new_store.get_stats()['num_vectors'] == 50
        
        # Test search works
        results = new_store.search(embeddings[0], top_k=3)
        assert len(results) == 3
        
        print("✅ Persistence test passed")
        
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    test_vector_store_basic()
    test_vector_store_search()
    test_vector_store_persistence()
    print("\n✅ All indexing tests passed!")
