"""
Advanced retrieval with filtering and ranking.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def main():
    """Demonstrate advanced retrieval features."""
    
    # Create sample documents with metadata
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "python_2023.txt"
    doc1.write_text("""
    Python 3.11 was released in 2023 with performance improvements.
    The new version includes faster startup and better error messages.
    Type hints have been enhanced for better static analysis.
    """)
    
    doc2 = docs_dir / "python_2022.txt"
    doc2.write_text("""
    Python 3.10 introduced pattern matching in 2022.
    Structural pattern matching makes code more readable.
    Union types got simplified syntax with the pipe operator.
    """)
    
    doc3 = docs_dir / "java_guide.txt"
    doc3.write_text("""
    Java is a statically-typed programming language.
    It runs on the Java Virtual Machine (JVM).
    Java is widely used in enterprise applications.
    """)
    
    # Create pipeline
    print("Creating RAG pipeline...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    # Ingest
    documents = [str(doc1), str(doc2), str(doc3)]
    pipeline.ingest_documents(documents, show_progress=False)
    
    print("\n" + "="*60)
    print("Test 1: Basic Retrieval")
    print("="*60)
    
    results = pipeline.query("Python features", top_k=3)
    for r in results:
        print(f"Rank {r.rank} (score: {r.score:.3f}): {r.text[:60]}...")
    
    print("\n" + "="*60)
    print("Test 2: Filter by Source")
    print("="*60)
    
    # Only get Python 2023 results
    results = pipeline.query_with_filters(
        "Python features",
        top_k=3,
        sources=["python_2023.txt"]
    )
    
    for r in results:
        print(f"Source: {r.metadata['source']}")
        print(f"  {r.text[:80]}...")
    
    print("\n" + "="*60)
    print("Test 3: Minimum Score Filter")
    print("="*60)
    
    results = pipeline.query_with_filters(
        "programming language",
        top_k=5,
        min_score=0.3
    )
    
    print(f"Found {len(results)} results with score >= 0.3")
    for r in results:
        print(f"  Score {r.score:.3f}: {r.text[:50]}...")
    
    print("\n" + "="*60)
    print("Test 4: Deduplication")
    print("="*60)
    
    results_with_dup = pipeline.query("Python", top_k=5, deduplicate=False)
    results_no_dup = pipeline.query_with_filters("Python", top_k=5, deduplicate=True)
    
    print(f"Without deduplication: {len(results_with_dup)} results")
    print(f"With deduplication: {len(results_no_dup)} results")


if __name__ == "__main__":
    main()
