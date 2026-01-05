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


def test_dense_retriever():
    """Test complete dense retrieval."""
    from rag.retrieval import DenseRetriever
    from rag.indexing import VectorStore
    from rag.embeddings import EmbeddingModel
    
    # Create test index
    model = EmbeddingModel()
    
    docs = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Deep learning is a subset of ML",
        "Natural language processing",
        "Computer vision and image recognition"
    ]
    
    embeddings = model.encode(docs)
    metadata = [{"id": i, "source": f"doc_{i}"} for i in range(len(docs))]
    
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    
    # Create retriever
    retriever = DenseRetriever(store)
    
    # Test retrieval
    query = "What is machine learning?"
    results = retriever.retrieve(query, top_k=3)
    
    assert len(results) == 3
    assert all(hasattr(r, 'text') for r in results)
    assert all(hasattr(r, 'score') for r in results)
    assert all(hasattr(r, 'rank') for r in results)
    
    print(f"\n✅ Dense retriever test")
    print(f"Query: '{query}'")
    for r in results:
        print(f"  Rank {r.rank}: {r.text[:50]}... (score: {r.score:.3f})")


def test_query_expansion_retrieval():
    """Test retrieval with query expansion."""
    from rag.retrieval import DenseRetriever
    from rag.indexing import VectorStore
    from rag.embeddings import EmbeddingModel
    
    model = EmbeddingModel()
    
    docs = [
        "How to configure SSL certificates",
        "SSL setup guide for servers",
        "Security configuration best practices"
    ]
    
    embeddings = model.encode(docs)
    metadata = [{"id": i} for i in range(len(docs))]
    
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    
    # Without expansion
    retriever_no_exp = DenseRetriever(store, use_query_expansion=False)
    results_no_exp = retriever_no_exp.retrieve("SSL setup", top_k=2)
    
    # With expansion
    retriever_exp = DenseRetriever(store, use_query_expansion=True)
    results_exp = retriever_exp.retrieve("SSL setup", top_k=2)
    
    print(f"\n✅ Query expansion comparison")
    print(f"Without expansion: {len(results_no_exp)} results")
    print(f"With expansion: {len(results_exp)} results")


def test_sparse_retriever():
    """Test sparse retrieval."""
    from rag.retrieval import SparseRetriever
    
    retriever = SparseRetriever()
    
    docs = [
        "Python 3.10 features include pattern matching",
        "Machine learning with Python and TensorFlow",
        "Java Spring Boot tutorial",
        "Python data science libraries"
    ]
    
    metadata = [{"id": i, "source": f"doc_{i}"} for i in range(len(docs))]
    
    retriever.index(docs, metadata)
    
    # Test retrieval
    results = retriever.retrieve("Python 3.10", top_k=3)
    
    assert len(results) > 0
    assert results[0].text.startswith("Python 3.10")
    
    print(f"\n✅ Sparse retriever test")
    print(f"Stats: {retriever.get_stats()}")
    for r in results:
        print(f"  Rank {r.rank}: {r.text[:50]}...")


def test_hybrid_retriever():
    """Test hybrid retrieval."""
    from rag.retrieval import HybridRetriever, DenseRetriever, SparseRetriever
    from rag.indexing import VectorStore
    from rag.embeddings import EmbeddingModel
    
    docs = [
        "Python 3.10 new features",
        "Machine learning with Python",
        "Deep learning tutorials",
        "Python programming guide"
    ]
    
    metadata = [{"id": i} for i in range(len(docs))]
    
    # Setup dense
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    dense = DenseRetriever(store)
    
    # Setup sparse
    sparse = SparseRetriever()
    sparse.index(docs, metadata)
    
    # Create hybrid
    hybrid = HybridRetriever(dense, sparse, fusion_method="rrf")
    
    # Test
    results = hybrid.retrieve("Python 3.10", top_k=3)
    
    assert len(results) > 0
    assert results[0].text  # Has content
    
    print(f"\n✅ Hybrid retriever test")
    print(f"Query: 'Python 3.10'")
    for r in results:
        print(f"  Rank {r.rank}: {r.text}")
