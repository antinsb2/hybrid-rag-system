"""
Benchmark retrieval performance: Linear vs HNSW.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import time
from rag.indexing import VectorStore, HNSWIndex


def create_test_data(n_vectors, dimension):
    """Create normalized random vectors."""
    vectors = np.random.rand(n_vectors, dimension).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    texts = [f"Document {i}" for i in range(n_vectors)]
    metadata = [{"id": i} for i in range(n_vectors)]
    return vectors, texts, metadata


def benchmark_linear_search():
    """Benchmark linear search."""
    print("="*60)
    print("Linear Search Benchmark")
    print("="*60)
    
    sizes = [1000, 5000, 10000]
    dimension = 384
    
    for size in sizes:
        vectors, texts, metadata = create_test_data(size, dimension)
        
        # Build index
        store = VectorStore()
        start = time.time()
        store.add(vectors, texts, metadata)
        build_time = time.time() - start
        
        # Search
        query = vectors[0]
        times = []
        
        for _ in range(100):
            start = time.time()
            results = store.search(query, top_k=10)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        p95_time = np.percentile(times, 95) * 1000
        
        print(f"\nDataset size: {size}")
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Avg query time: {avg_time:.2f}ms")
        print(f"  P95 query time: {p95_time:.2f}ms")
        print(f"  Memory: {store.get_stats()['memory_mb']:.1f}MB")


def benchmark_hnsw_search():
    """Benchmark HNSW search."""
    print("\n" + "="*60)
    print("HNSW Search Benchmark")
    print("="*60)
    
    sizes = [1000, 5000, 10000, 50000]
    dimension = 384
    
    for size in sizes:
        vectors, texts, metadata = create_test_data(size, dimension)
        
        # Build index
        index = HNSWIndex(dimension=dimension, max_elements=size)
        start = time.time()
        index.add(vectors, texts, metadata)
        build_time = time.time() - start
        
        # Search
        query = vectors[0]
        times = []
        
        for _ in range(100):
            start = time.time()
            results = index.search(query, top_k=10)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        p95_time = np.percentile(times, 95) * 1000
        
        print(f"\nDataset size: {size}")
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Avg query time: {avg_time:.2f}ms")
        print(f"  P95 query time: {p95_time:.2f}ms")


def compare_accuracy():
    """Compare recall between linear and HNSW."""
    print("\n" + "="*60)
    print("Accuracy Comparison (Recall@10)")
    print("="*60)
    
    size = 5000
    dimension = 384
    
    vectors, texts, metadata = create_test_data(size, dimension)
    
    # Build both indexes
    linear = VectorStore()
    linear.add(vectors, texts, metadata)
    
    hnsw = HNSWIndex(dimension=dimension, max_elements=size)
    hnsw.add(vectors, texts, metadata)
    
    # Test on 100 queries
    n_queries = 100
    recalls = []
    
    for i in range(n_queries):
        query = vectors[i]
        
        # Get ground truth from linear
        linear_results = linear.search(query, top_k=10)
        linear_ids = set(r.index for r in linear_results)
        
        # Get HNSW results
        hnsw_results = hnsw.search(query, top_k=10)
        hnsw_ids = set(r.index for r in hnsw_results)
        
        # Calculate recall
        recall = len(linear_ids & hnsw_ids) / len(linear_ids)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"\nAverage Recall@10: {avg_recall:.3f}")
    print(f"Min Recall: {min(recalls):.3f}")
    print(f"Max Recall: {max(recalls):.3f}")


if __name__ == "__main__":
    benchmark_linear_search()
    benchmark_hnsw_search()
    compare_accuracy()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Linear: Exact but slow (O(n))")
    print("- HNSW: Fast but approximate (~95% recall)")
    print("- Use Linear for <10K vectors")
    print("- Use HNSW for >10K vectors")
    print("="*60)
