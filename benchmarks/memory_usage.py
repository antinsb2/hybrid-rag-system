"""
Analyze memory usage of different components.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore
import numpy as np
import tracemalloc


def get_size_mb(obj):
    """Estimate object size in MB."""
    import sys
    size = sys.getsizeof(obj)
    
    if isinstance(obj, (list, tuple)):
        size += sum(get_size_mb(item) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size_mb(k) + get_size_mb(v) for k, v in obj.items())
    
    return size / (1024 * 1024)


def benchmark_memory():
    """Benchmark memory usage."""
    
    print("="*70)
    print("MEMORY USAGE ANALYSIS")
    print("="*70)
    
    scales = [100, 500, 1000, 5000]
    
    for n_docs in scales:
        print(f"\n{'='*70}")
        print(f"Dataset: {n_docs} documents")
        print('='*70)
        
        docs = [f"Document {i} " + " ".join([f"word{j}" for j in range(50)]) for i in range(n_docs)]
        metadata = [{"id": i} for i in range(n_docs)]
        
        # Measure dense retriever memory
        tracemalloc.start()
        
        model = EmbeddingModel()
        embeddings = model.encode(docs, show_progress=False)
        
        store = VectorStore()
        store.add(embeddings, docs, metadata)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        dense_memory = peak / (1024 * 1024)  # MB
        
        # Measure sparse retriever memory
        tracemalloc.start()
        
        sparse = SparseRetriever()
        sparse.index(docs, metadata)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        sparse_memory = peak / (1024 * 1024)  # MB
        
        # Calculate per-document memory
        dense_per_doc = dense_memory / n_docs
        sparse_per_doc = sparse_memory / n_docs
        
        print(f"\nDense Retriever:")
        print(f"  Total memory: {dense_memory:.1f} MB")
        print(f"  Per document: {dense_per_doc:.3f} MB")
        print(f"  Embeddings: {embeddings.nbytes / (1024*1024):.1f} MB")
        
        print(f"\nSparse Retriever:")
        print(f"  Total memory: {sparse_memory:.1f} MB")
        print(f"  Per document: {sparse_per_doc:.3f} MB")
        print(f"  Index terms: {len(sparse.bm25.inverted_index)}")
        
        print(f"\nMemory ratio (Dense/Sparse): {dense_memory/sparse_memory:.2f}x")


def analyze_component_memory():
    """Break down memory by component."""
    
    print("\n" + "="*70)
    print("COMPONENT MEMORY BREAKDOWN (1000 docs)")
    print("="*70)
    
    n_docs = 1000
    docs = [f"Sample document {i} with various content" for i in range(n_docs)]
    
    # Embeddings only
    model = EmbeddingModel()
    embeddings = model.encode(docs, show_progress=False)
    emb_memory = embeddings.nbytes / (1024 * 1024)
    
    # Text storage
    text_memory = sum(len(doc.encode('utf-8')) for doc in docs) / (1024 * 1024)
    
    # Metadata
    metadata = [{"id": i, "source": f"doc_{i}.txt"} for i in range(n_docs)]
    meta_memory = get_size_mb(metadata)
    
    print(f"\nMemory breakdown:")
    print(f"  Embeddings: {emb_memory:.1f} MB ({emb_memory/emb_memory*100:.0f}%)")
    print(f"  Text: {text_memory:.1f} MB ({text_memory/emb_memory*100:.0f}%)")
    print(f"  Metadata: {meta_memory:.1f} MB ({meta_memory/emb_memory*100:.0f}%)")
    print(f"  Total: {emb_memory + text_memory + meta_memory:.1f} MB")


if __name__ == "__main__":
    benchmark_memory()
    analyze_component_memory()
    
    print("\n" + "="*70)
    print("INSIGHTS:")
    print("- Dense retrieval: Higher memory (stores embeddings)")
    print("- Sparse retrieval: Lower memory (stores terms)")
    print("- Embeddings dominate memory usage")
    print("- Trade-off: Memory vs semantic understanding")
    print("="*70)
