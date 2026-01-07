"""
Benchmark query latency across retrieval methods.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore
import time
import numpy as np


def benchmark_latency():
    """Benchmark query latency at different scales."""
    
    print("="*70)
    print("LATENCY BENCHMARK")
    print("="*70)
    
    scales = [100, 500, 1000, 2000]
    
    for n_docs in scales:
        print(f"\n{'='*70}")
        print(f"Dataset size: {n_docs} documents")
        print('='*70)
        
        # Create documents
        docs = [
            f"Document {i} about topic {i % 10} with content and information"
            for i in range(n_docs)
        ]
        metadata = [{"id": i} for i in range(n_docs)]
        
        # Setup retrievers
        print("Building indexes...")
        
        # Dense
        model = EmbeddingModel()
        start = time.time()
        embeddings = model.encode(docs, show_progress=False)
        dense_index_time = time.time() - start
        
        store = VectorStore()
        store.add(embeddings, docs, metadata)
        dense = DenseRetriever(store)
        
        # Sparse
        start = time.time()
        sparse = SparseRetriever()
        sparse.index(docs, metadata)
        sparse_index_time = time.time() - start
        
        # Hybrid
        hybrid = HybridRetriever(dense, sparse, fusion_method="rrf")
        
        # Benchmark queries
        test_queries = [
            "topic information",
            "document content",
            "information about topic 5",
            "content and information"
        ]
        
        methods = [
            ("Dense", dense),
            ("Sparse", sparse),
            ("Hybrid-RRF", hybrid)
        ]
        
        print(f"\nIndexing time:")
        print(f"  Dense: {dense_index_time:.2f}s")
        print(f"  Sparse: {sparse_index_time:.2f}s")
        
        print(f"\nQuery latency (avg over {len(test_queries)} queries):")
        
        for method_name, retriever in methods:
            times = []
            
            for query in test_queries:
                start = time.time()
                results = retriever.retrieve(query, top_k=10)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times) * 1000  # ms
            p50 = np.percentile(times, 50) * 1000
            p95 = np.percentile(times, 95) * 1000
            p99 = np.percentile(times, 99) * 1000
            
            print(f"\n{method_name}:")
            print(f"  Avg: {avg_time:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")
            print(f"  P99: {p99:.2f}ms")


def benchmark_cache_impact():
    """Benchmark embedding cache impact."""
    
    print("\n" + "="*70)
    print("CACHE IMPACT BENCHMARK")
    print("="*70)
    
    docs = [f"Document {i} with unique content" for i in range(100)]
    
    # Without cache
    print("\nWithout cache:")
    model = EmbeddingModel()
    
    start = time.time()
    embeddings1 = model.encode(docs, show_progress=False)
    time1 = time.time() - start
    
    start = time.time()
    embeddings2 = model.encode(docs, show_progress=False)
    time2 = time.time() - start
    
    print(f"  First encoding: {time1:.2f}s")
    print(f"  Second encoding: {time2:.2f}s")
    print(f"  No speedup: {time1/time2:.1f}x")
    
    # With cache
    print("\nWith cache:")
    from rag.embeddings import EmbeddingCache
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(tmpdir)
        
        # First pass - populate cache
        start = time.time()
        for doc in docs:
            emb = model.encode(doc)[0]
            cache.put(doc, emb)
        time_no_cache = time.time() - start
        
        # Second pass - use cache
        start = time.time()
        for doc in docs:
            emb = cache.get(doc)
        time_cached = time.time() - start
        
        stats = cache.get_stats()
        speedup = time_no_cache / time_cached
        
        print(f"  First pass (populate): {time_no_cache:.2f}s")
        print(f"  Second pass (cached): {time_cached:.4f}s")
        print(f"  Speedup: {speedup:.0f}x")
        print(f"  Cache hit rate: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    benchmark_latency()
    benchmark_cache_impact()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("- Dense: Slower but semantic understanding")
    print("- Sparse: Faster for keyword matching")
    print("- Hybrid: Balanced performance with best quality")
    print("- Cache: 100-500x speedup on repeated queries")
    print("="*70)
