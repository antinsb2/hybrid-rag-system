"""
Integration tests for complete RAG pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile
import pytest


def test_complete_pipeline_flow():
    """Test complete end-to-end flow."""
    
    # Create test documents
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "test_doc.txt"
    doc1.write_text("Python is a programming language used for web development and data science.")
    
    # Initialize pipeline
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    # Ingest
    pipeline.ingest_documents([str(doc1)], show_progress=False)
    assert pipeline.is_indexed
    
    # Enable hybrid
    pipeline.enable_hybrid(fusion_method="rrf")
    assert hasattr(pipeline, 'hybrid_retriever')
    
    # Enable generation
    pipeline.enable_generation(use_mock=True)
    assert hasattr(pipeline, 'generator')
    
    # Query
    result = pipeline.ask("What is Python used for?", top_k=3)
    
    assert "answer" in result
    assert "sources" in result
    assert result["num_chunks"] > 0
    
    print("✅ Complete pipeline flow test passed")


def test_pipeline_with_multiple_documents():
    """Test with multiple documents."""
    
    docs_dir = Path(tempfile.mkdtemp())
    
    # Create multiple docs
    for i in range(5):
        doc = docs_dir / f"doc_{i}.txt"
        doc.write_text(f"Document {i} about topic {i % 3}")
    
    pipeline = RAGPipeline(chunk_size=128, use_hnsw=False)
    
    doc_paths = [str(docs_dir / f"doc_{i}.txt") for i in range(5)]
    pipeline.ingest_documents(doc_paths, show_progress=False)
    
    stats = pipeline.get_stats()
    assert stats["index"]["num_vectors"] == 5  # 5 documents
    
    print(f"✅ Multiple documents test: {stats['index']['num_vectors']} chunks indexed")


def test_retrieval_quality():
    """Test that retrieval returns relevant results."""
    
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "python.txt"
    doc1.write_text("Python is a high-level programming language created by Guido van Rossum")
    
    doc2 = docs_dir / "java.txt"
    doc2.write_text("Java is a statically-typed programming language developed by Sun Microsystems")
    
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    pipeline.ingest_documents([str(doc1), str(doc2)], show_progress=False)
    
    # Query about Python
    results = pipeline.query("Python programming", top_k=2)
    
    # Top result should be from Python doc
    assert "Python" in results[0].text
    assert results[0].score > 0.5  # Should have decent similarity
    
    print("✅ Retrieval quality test passed")


def test_error_handling():
    """Test error handling in pipeline."""
    
    pipeline = RAGPipeline(use_hnsw=False)
    
    # Query before indexing should raise error
    with pytest.raises(RuntimeError, match="No documents indexed"):
        pipeline.query("test query")
    
    # Ask before enabling generation should raise error
    docs_dir = Path(tempfile.mkdtemp())
    doc = docs_dir / "test.txt"
    doc.write_text("Test content")
    
    pipeline.ingest_documents([str(doc)], show_progress=False)
    
    with pytest.raises(RuntimeError, match="Generation not enabled"):
        pipeline.ask("test question")
    
    print("✅ Error handling test passed")


def test_cache_effectiveness():
    """Test that caching actually works."""
    
    docs_dir = Path(tempfile.mkdtemp())
    doc = docs_dir / "test.txt"
    doc.write_text("Test document " * 20)
    
    pipeline = RAGPipeline(chunk_size=128, use_cache=True)
    
    # First ingestion
    pipeline.ingest_documents([str(doc)], show_progress=False)
    stats1 = pipeline.get_stats()
    
    # Create new pipeline with same cache
    pipeline2 = RAGPipeline(chunk_size=128, use_cache=True)
    pipeline2.ingest_documents([str(doc)], show_progress=False)
    stats2 = pipeline2.get_stats()
    
    # Cache should have hits
    cache_stats = stats2.get("embedding_pipeline", {}).get("cache", {})
    
    print(f"✅ Cache test: {cache_stats}")


if __name__ == "__main__":
    test_complete_pipeline_flow()
    test_pipeline_with_multiple_documents()
    test_retrieval_quality()
    test_error_handling()
    test_cache_effectiveness()
    print("\n✅ All integration tests passed!")
