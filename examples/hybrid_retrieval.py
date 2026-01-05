"""
Demonstrate hybrid retrieval.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore


def main():
    """Demonstrate hybrid retrieval advantages."""
    
    # Sample documents
    docs = [
        "Python 3.11 was released in October 2022",
        "Machine learning models require large datasets",
        "Deep learning uses neural networks with multiple layers",
        "TensorFlow 2.x introduced eager execution by default",
        "Natural language processing analyzes human language",
        "PyTorch is popular for research and production",
        "Transformers revolutionized NLP starting in 2017",
        "GPT-3 has 175 billion parameters"
    ]
    
    metadata = [{"id": i, "source": f"doc_{i}.txt"} for i in range(len(docs))]
    
    print("Setting up hybrid retrieval system...")
    
    # Setup dense retriever
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    
    dense_store = VectorStore()
    dense_store.add(embeddings, docs, metadata)
    dense_retriever = DenseRetriever(dense_store)
    
    # Setup sparse retriever
    sparse_retriever = SparseRetriever()
    sparse_retriever.index(docs, metadata)
    
    # Create hybrid retrievers with different fusion methods
    hybrid_rrf = HybridRetriever(dense_retriever, sparse_retriever, fusion_method="rrf")
    hybrid_weighted = HybridRetriever(dense_retriever, sparse_retriever, fusion_method="weighted")
    
    # Test queries
    queries = [
        "Python 3.11 release",      # Benefits from exact keyword match
        "neural network training",   # Benefits from semantic understanding
        "175 billion parameters"     # Benefits from exact numbers
    ]
    
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL DEMONSTRATION")
    print("="*70)
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print('='*70)
        
        # Dense only
        dense_results = dense_retriever.retrieve(query, top_k=3)
        print("\nDENSE ONLY (Semantic):")
        for r in dense_results:
            print(f"  {r.rank}. [{r.score:.3f}] {r.text}")
        
        # Sparse only
        sparse_results = sparse_retriever.retrieve(query, top_k=3)
        print("\nSPARSE ONLY (Keywords):")
        for r in sparse_results:
            print(f"  {r.rank}. [{r.score:.2f}] {r.text}")
        
        # Hybrid RRF
        hybrid_results = hybrid_rrf.retrieve(query, top_k=3)
        print("\nHYBRID (RRF):")
        for r in hybrid_results:
            print(f"  {r.rank}. [{r.score:.4f}] {r.text}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("• Dense: Good for conceptual matches")
    print("• Sparse: Good for exact terms and numbers")
    print("• Hybrid: Combines strengths of both approaches")
    print("• RRF: Works well when scores aren't comparable")
    print("="*70)


if __name__ == "__main__":
    main()
