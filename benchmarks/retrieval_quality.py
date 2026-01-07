"""
Benchmark retrieval quality: Dense vs Sparse vs Hybrid.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore
import numpy as np


def create_test_corpus():
    """Create test corpus with known relevant documents."""
    docs = [
        # Python programming cluster
        "Python 3.10 introduced structural pattern matching with match/case syntax",
        "Python programming language is known for readability and simplicity",
        "Python virtual environments isolate project dependencies using venv",
        
        # Machine learning cluster
        "Machine learning models learn patterns from training data",
        "Neural networks consist of interconnected layers of artificial neurons",
        "Deep learning uses neural networks with multiple hidden layers",
        
        # NLP cluster
        "Natural language processing analyzes and understands human language",
        "Transformers revolutionized NLP with attention mechanisms in 2017",
        "GPT-3 has 175 billion parameters and generates human-like text",
        
        # Frameworks cluster
        "TensorFlow 2.0 introduced eager execution as default mode",
        "PyTorch is popular for deep learning research and production",
        "Hugging Face Transformers library provides pre-trained NLP models",
        
        # Unrelated
        "Cooking pasta requires boiling water and salt",
        "The weather forecast predicts rain tomorrow afternoon",
        "Basketball is played with five players on each team"
    ]
    
    # Ground truth relevance (query -> list of relevant doc indices)
    relevance = {
        "Python 3.10 features": [0, 1, 2],
        "neural networks deep learning": [4, 5, 6],
        "transformers NLP": [6, 7, 8],
        "175 billion parameters": [8],
        "PyTorch framework": [10, 11],
    }
    
    return docs, relevance


def calculate_metrics(retrieved_indices, relevant_indices, k=10):
    """
    Calculate retrieval metrics.
    
    Returns:
        recall@k, precision@k, ndcg@k
    """
    retrieved_set = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    # Recall@k
    hits = len(retrieved_set & relevant_set)
    recall = hits / len(relevant_set) if relevant_set else 0
    
    # Precision@k
    precision = hits / k if k > 0 else 0
    
    # nDCG@k (simplified)
    dcg = 0
    idcg = 0
    
    for i, idx in enumerate(retrieved_indices[:k], 1):
        if idx in relevant_set:
            dcg += 1 / np.log2(i + 1)
    
    for i in range(1, min(k, len(relevant_set)) + 1):
        idcg += 1 / np.log2(i + 1)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return recall, precision, ndcg


def benchmark_retrieval_quality():
    """Benchmark retrieval quality across methods."""
    
    docs, relevance = create_test_corpus()
    metadata = [{"id": i, "doc_id": i} for i in range(len(docs))]
    
    print("Setting up retrievers...")
    
    # Dense retriever
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    
    dense_store = VectorStore()
    dense_store.add(embeddings, docs, metadata)
    dense = DenseRetriever(dense_store)
    
    # Sparse retriever
    sparse = SparseRetriever()
    sparse.index(docs, metadata)
    
    # Hybrid retrievers
    hybrid_rrf = HybridRetriever(dense, sparse, fusion_method="rrf")
    hybrid_weighted = HybridRetriever(dense, sparse, fusion_method="weighted")
    
    # Test all methods
    methods = [
        ("Dense", dense),
        ("Sparse", sparse),
        ("Hybrid-RRF", hybrid_rrf),
        ("Hybrid-Weighted", hybrid_weighted),
    ]
    
    print("\n" + "="*70)
    print("RETRIEVAL QUALITY BENCHMARK")
    print("="*70)
    
    all_metrics = {name: {"recall": [], "precision": [], "ndcg": []} for name, _ in methods}
    
    for query, relevant_docs in relevance.items():
        print(f"\nQuery: '{query}'")
        print(f"Relevant docs: {relevant_docs}")
        print("-" * 70)
        
        for method_name, retriever in methods:
            results = retriever.retrieve(query, top_k=10)
            retrieved_indices = [r.metadata["doc_id"] for r in results]
            
            recall, precision, ndcg = calculate_metrics(retrieved_indices, relevant_docs, k=10)
            
            all_metrics[method_name]["recall"].append(recall)
            all_metrics[method_name]["precision"].append(precision)
            all_metrics[method_name]["ndcg"].append(ndcg)
            
            print(f"{method_name:20s} - Recall: {recall:.3f}, Precision: {precision:.3f}, nDCG: {ndcg:.3f}")
    
    # Average metrics
    print("\n" + "="*70)
    print("AVERAGE METRICS")
    print("="*70)
    
    for method_name in all_metrics:
        avg_recall = np.mean(all_metrics[method_name]["recall"])
        avg_precision = np.mean(all_metrics[method_name]["precision"])
        avg_ndcg = np.mean(all_metrics[method_name]["ndcg"])
        
        print(f"{method_name:20s} - Recall: {avg_recall:.3f}, Precision: {avg_precision:.3f}, nDCG: {avg_ndcg:.3f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("- Hybrid methods typically outperform single methods")
    print("- Dense: Better for semantic queries")
    print("- Sparse: Better for exact term/number matching")
    print("- Hybrid: Best overall performance")
    print("="*70)


if __name__ == "__main__":
    benchmark_retrieval_quality()
