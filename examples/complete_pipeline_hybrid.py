"""
Complete pipeline with hybrid retrieval.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def main():
    """Demonstrate complete hybrid RAG pipeline."""
    
    # Create sample documents
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "python_310.txt"
    doc1.write_text("""
    Python 3.10 was released on October 4, 2021. The major new features include:
    - Structural pattern matching with match/case statements
    - Better error messages with precise line numbers
    - Type unions using the | operator instead of Union
    - New zip() strict parameter for safety
    """)
    
    doc2 = docs_dir / "ml_basics.txt"
    doc2.write_text("""
    Machine learning is a subset of artificial intelligence. Common algorithms include:
    - Decision trees for classification
    - Neural networks for complex patterns
    - Support vector machines for classification
    - K-means for clustering
    Training requires labeled datasets and computational resources.
    """)
    
    doc3 = docs_dir / "deep_learning.txt"
    doc3.write_text("""
    Deep learning uses neural networks with multiple layers. Key architectures:
    - Convolutional Neural Networks (CNNs) for images
    - Recurrent Neural Networks (RNNs) for sequences
    - Transformers for natural language processing
    GPT-3 has 175 billion parameters and was trained on massive text corpora.
    """)
    
    doc4 = docs_dir / "frameworks.txt"
    doc4.write_text("""
    Popular deep learning frameworks include:
    - TensorFlow 2.x with Keras integration
    - PyTorch for research and production
    - JAX for high-performance computing
    - FastAI for rapid prototyping
    """)
    
    # Create pipeline
    print("Creating hybrid RAG pipeline...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    # Ingest documents
    documents = [str(doc1), str(doc2), str(doc3), str(doc4)]
    pipeline.ingest_documents(documents, show_progress=False)
    
    # Enable hybrid retrieval
    pipeline.enable_hybrid(fusion_method="rrf")
    
    # Test queries
    queries = [
        "Python 3.10 features",           # Exact version match
        "neural network architectures",   # Semantic concept
        "175 billion parameters",         # Exact number
        "machine learning algorithms"     # General concept
    ]
    
    print("\n" + "="*70)
    print("DENSE vs HYBRID COMPARISON")
    print("="*70)
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print('-'*70)
        
        # Dense only
        dense_results = pipeline.query(query, top_k=3)
        print("\nDENSE ONLY:")
        for r in dense_results:
            print(f"  {r.rank}. [{r.score:.3f}] {r.text[:60]}...")
        
        # Hybrid
        hybrid_results = pipeline.query_hybrid(query, top_k=3)
        print("\nHYBRID (RRF):")
        for r in hybrid_results:
            print(f"  {r.rank}. [{r.score:.4f}] {r.text[:60]}...")
    
    print("\n" + "="*70)
    print("Pipeline Statistics:")
    print("="*70)
    stats = pipeline.get_stats()
    print(f"Documents indexed: {stats.get('index', {}).get('num_vectors', 0)}")
    print(f"Hybrid enabled: {hasattr(pipeline, 'hybrid_retriever')}")
    if hasattr(pipeline, 'hybrid_retriever'):
        print(f"Fusion method: {pipeline.hybrid_retriever.fusion_method}")


if __name__ == "__main__":
    main()
