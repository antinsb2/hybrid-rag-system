"""
Basic retrieval example.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def main():
    """Demonstrate basic RAG pipeline."""
    
    # Create sample documents
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "python_guide.txt"
    doc1.write_text("""
    Python is a high-level programming language. It is known for its simplicity and readability.
    Python is widely used in machine learning, data science, and web development.
    The language supports multiple programming paradigms including procedural and object-oriented programming.
    """)
    
    doc2 = docs_dir / "ml_intro.txt"
    doc2.write_text("""
    Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.
    Common machine learning algorithms include decision trees, neural networks, and support vector machines.
    Deep learning is a specialized form of machine learning using neural networks with many layers.
    """)
    
    doc3 = docs_dir / "web_dev.txt"
    doc3.write_text("""
    Web development involves building websites and web applications.
    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.
    Backend development handles server-side logic and databases.
    """)
    
    # Create pipeline
    print("Creating RAG pipeline...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)  # Small chunks for demo
    
    # Ingest documents
    documents = [str(doc1), str(doc2), str(doc3)]
    pipeline.ingest_documents(documents, show_progress=False)
    
    # Test queries
    queries = [
        "What is Python used for?",
        "Tell me about machine learning",
        "What is frontend development?"
    ]
    
    print("\n" + "="*60)
    print("Testing Retrieval")
    print("="*60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.query(query, top_k=3)
        
        for result in results:
            print(f"  Rank {result.rank} (score: {result.score:.3f})")
            print(f"    {result.text[:100]}...")
            print(f"    Source: {result.metadata.get('source', 'unknown')}")
    
    # Show stats
    print("\n" + "="*60)
    print("Pipeline Statistics")
    print("="*60)
    stats = pipeline.get_stats()
    print(f"Index type: {stats['index']['index_type'] if 'index' in stats else 'None'}")
    print(f"Indexed vectors: {stats['index']['num_vectors'] if 'index' in stats else 0}")
    print(f"Embedding cache: {stats['embedding_pipeline']['cache']}")


if __name__ == "__main__":
    main()
