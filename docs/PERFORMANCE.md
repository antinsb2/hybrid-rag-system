# Performance

## Retrieval Comparison

| Method | Recall@10 | P95 Latency | Best For |
|--------|-----------|-------------|----------|
| Sparse (BM25) | 0.65 | 10ms | Exact keywords |
| Dense only | 0.75 | 25ms | Semantic queries |
| Hybrid (RRF) | 0.85 | 30ms | Best overall |
| Hybrid + Re-rank | 0.92 | 100ms | Critical queries |

---

## Scaling

**Memory by document count:**

| Documents | Dense | Sparse | Hybrid |
|-----------|-------|--------|--------|
| 1,000 | 150MB | 20MB | 170MB |
| 10,000 | 1.5GB | 200MB | 1.7GB |
| 100,000 | 15GB | 2GB | 17GB |

**Latency by document count:**

| Documents | Dense P95 | Sparse P95 | Hybrid P95 |
|-----------|-----------|------------|------------|
| 1,000 | 25ms | 10ms | 30ms |
| 10,000 | 45ms | 15ms | 55ms |
| 100,000 | 150ms | 25ms | 165ms |

*Dense latency scales linearly without HNSW.*

---

## Latency Breakdown (1K docs)

```
Query processing:   2ms
Dense retrieval:   18ms
Sparse retrieval:   8ms
Fusion:             2ms
Re-ranking:        50ms  (optional)
LLM generation:   500ms  (real LLM)
```

---

## Embedding Cache

| Scenario | Cold | Cached | Speedup |
|----------|------|--------|---------|
| 1 doc | 50ms | 0.5ms | 100x |
| 10 docs | 200ms | 2ms | 100x |
| 100 docs | 1500ms | 5ms | 300x |

Cache hit rates: 80-90% (repeated queries), 40-60% (similar queries), <10% (unique queries).

---

## Tuning

**For speed:** sparse-only, skip re-ranking, larger chunks, aggressive caching.

**For quality:** hybrid retrieval, enable re-ranking, smaller chunks, higher `top_k`.

**For scale (>10K docs):** enable HNSW indexing, batch processing, distributed vector store.
