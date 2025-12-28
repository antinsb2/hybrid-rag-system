"""
Tests for embedding model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.embeddings import EmbeddingModel
import numpy as np


def test_embedding_generation():
    """Test basic embedding generation."""
    model = EmbeddingModel()
    
    text = "This is a test sentence."
    embedding = model.encode(text)
    
    assert embedding.shape == (1, model.dimension)
    assert isinstance(embedding, np.ndarray)
    
    print(f"✅ Single embedding: shape {embedding.shape}")


def test_batch_embedding():
    """Test batch embedding generation."""
    model = EmbeddingModel()
    
    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence."
    ]
    
    embeddings = model.encode(texts, batch_size=2)
    
    assert embeddings.shape == (3, model.dimension)
    print(f"✅ Batch embeddings: shape {embeddings.shape}")


def test_similarity():
    """Test similarity computation."""
    model = EmbeddingModel()
    
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    text3 = "Python programming language."
    
    emb1 = model.encode(text1)[0]
    emb2 = model.encode(text2)[0]
    emb3 = model.encode(text3)[0]
    
    sim_similar = model.similarity(emb1, emb2)
    sim_different = model.similarity(emb1, emb3)
    
    assert sim_similar > sim_different
    print(f"✅ Similarity test: similar={sim_similar:.3f}, different={sim_different:.3f}")


def test_model_info():
    """Test model info retrieval."""
    model = EmbeddingModel()
    info = model.get_info()
    
    assert "model_name" in info
    assert "device" in info
    assert "dimension" in info
    
    print(f"✅ Model info: {info}")


if __name__ == "__main__":
    test_embedding_generation()
    test_batch_embedding()
    test_similarity()
    test_model_info()
    print("\n✅ All embedding tests passed!")

def test_cache():
    """Test embedding cache."""
    from rag.embeddings import EmbeddingCache
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        text = "Test sentence for caching."
        embedding = np.random.rand(384)
        
        # Store in cache
        cache.put(text, embedding)
        
        # Retrieve from cache
        cached = cache.get(text)
        assert cached is not None
        assert np.allclose(cached, embedding)
        
        # Check stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
        
        print(f"✅ Cache test passed: {stats}")


def test_pipeline():
    """Test complete embedding pipeline."""
    from rag.embeddings import EmbeddingPipeline
    import tempfile
    
    # Create temp text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document. " * 100)
        temp_path = f.name
    
    try:
        pipeline = EmbeddingPipeline(chunk_size=50, chunk_overlap=10, use_cache=False)
        chunks = pipeline.process_document(temp_path)
        
        assert len(chunks) > 0
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'embedding') for c in chunks)
        assert all(c.embedding.shape[0] == pipeline.model.dimension for c in chunks)
        
        print(f"✅ Pipeline test: {len(chunks)} chunks created")
        print(f"   Stats: {pipeline.get_stats()}")
        
    finally:
        Path(temp_path).unlink()

