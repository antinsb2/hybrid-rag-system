"""
Benchmark embedding speed.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.embeddings import EmbeddingModel
import time
import numpy as np


def benchmark_embedding_speed():
    """Benchmark embedding generation speed."""
    model = EmbeddingModel()
    
    # Generate test data
    texts = [f"This is test sentence number {i}." for i in range(1000)]
    
    print("Benchmarking embedding speed...")
    print(f"Device: {model.device}")
    print(f"Number of texts: {len(texts)}")
    
    # Benchmark different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        start = time.time()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress=False)
        elapsed = time.time() - start
        
        throughput = len(texts) / elapsed
        
        print(f"\nBatch size {batch_size}:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} texts/sec")
        print(f"  Per text: {elapsed/len(texts)*1000:.2f}ms")


def benchmark_cache_impact():
    """Benchmark cache impact."""
    from rag.embeddings import EmbeddingCache
    import tempfile
    
    print("\n" + "="*50)
    print("Benchmarking cache impact...")
    
    model = EmbeddingModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        texts = [f"Test sentence {i}" for i in range(100)]
        
        # First pass (no cache)
        start = time.time()
        for text in texts:
            embedding = model.encode(text)[0]
            cache.put(text, embedding)
        no_cache_time = time.time() - start
        
        # Second pass (with cache)
        start = time.time()
        for text in texts:
            embedding = cache.get(text)
        cache_time = time.time() - start
        
        speedup = no_cache_time / cache_time
        
        print(f"\nWithout cache: {no_cache_time:.2f}s")
        print(f"With cache: {cache_time:.4f}s")
        print(f"Speedup: {speedup:.0f}x")
        print(f"Cache stats: {cache.get_stats()}")


if __name__ == "__main__":
    benchmark_embedding_speed()
    benchmark_cache_impact()
    print("\n✅ Benchmarks complete!")
