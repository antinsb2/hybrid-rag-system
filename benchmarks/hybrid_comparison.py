"""
Benchmark hybrid vs single-method retrieval.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore
import time


def main():
    """Compare retrieval quality and speed."""
    
    # Create larger test dataset
    docs = [
        f"Document {i} about machine learning and artificial intelligence"
        if i % 2 == 0
        else f"Python 3.{i % 10} programming tutorial and examples"
        for i in range(100)
    ]
    
    metadata = [{"id": i} for i in range(len(docs))]
    
    print("Setting up retrievers...")
    
    # Dense
    model = EmbeddingModel()
    embeddings = model.encode(docs, show_progress=True)
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    dense = DenseRetriever(store)
    
    # Sparse
    sparse = SparseRetriever()
    sparse.index(docs, metadata)
    
    # Hybrid
    hybrid = HybridRetriever(dense, sparse, fusion_method="rrf")
    
    # Test queries
    queries = [
        "Python 3.5 tutorial",
        "machine learning AI",
        "programming examples",
        "artificial intelligence"
    ]
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    for retriever_name, retriever in [
        ("Dense", dense),
        ("Sparse", sparse),
        ("Hybrid", hybrid)
    ]:
        print(f"\n{retriever_name} Retrieval:")
        
        times = []
        for query in queries:
            start = time.time()
            results = retriever.retrieve(query, top_k=5)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"  Avg query time: {avg_time:.2f}ms")
        print(f"  Results per query: {len(results)}")


if __name__ == "__main__":
    main()
