"""
Compare dense and sparse retrieval.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore


def main():
    """Compare dense and sparse retrieval."""
    
    # Sample documents
    docs = [
        "Python 3.10 was released in October 2021",
        "Machine learning models require training data",
        "Deep learning uses neural networks with many layers",
        "Python is great for data science and ML",
        "Natural language processing analyzes text",
        "TensorFlow 2.0 introduced eager execution",
        "PyTorch is popular for research",
        "Transformers revolutionized NLP in 2017"
    ]
    
    metadata = [{"id": i, "source": f"doc_{i}.txt"} for i in range(len(docs))]
    
    print("Setting up retrievers...")
    
    # Dense retriever
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    
    dense_store = VectorStore()
    dense_store.add(embeddings, docs, metadata)
    dense_retriever = DenseRetriever(dense_store)
    
    # Sparse retriever
    sparse_retriever = SparseRetriever()
    sparse_retriever.index(docs, metadata)
    
    # Test queries
    queries = [
        "Python 3.10 release date",  # Exact keyword match
        "neural network training",    # Semantic similarity
        "text analysis NLP"           # Mixed
    ]
    
    print("\n" + "="*70)
    print("DENSE vs SPARSE RETRIEVAL COMPARISON")
    print("="*70)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        # Dense results
        dense_results = dense_retriever.retrieve(query, top_k=3)
        print("\nDENSE (Semantic):")
        for r in dense_results:
            print(f"  {r.rank}. [{r.score:.3f}] {r.text}")
        
        # Sparse results
        sparse_results = sparse_retriever.retrieve(query, top_k=3)
        print("\nSPARSE (Keywords):")
        for r in sparse_results:
            print(f"  {r.rank}. [{r.score:.2f}] {r.text}")
    
    print("\n" + "="*70)
    print("OBSERVATIONS:")
    print("="*70)
    print("Dense: Good for concept matching (ML → machine learning)")
    print("Sparse: Good for exact terms (Python 3.10 → Python 3.10)")
    print("Hybrid: Best of both approaches!")
    print("="*70)


if __name__ == "__main__":
    main()
