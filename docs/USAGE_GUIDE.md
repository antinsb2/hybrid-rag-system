# Usage Guide

## Basic Usage

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_size=512)
pipeline.ingest_documents(['data/guide.pdf', 'data/manual.docx'])
pipeline.enable_hybrid(fusion_method="rrf")
pipeline.enable_generation(use_mock=True)

result = pipeline.ask("How do I configure SSL?")
print(result["answer"])
```

---

## Retrieval Options

**Retrieve without generating an answer:**
```python
results = pipeline.query("Python features", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.text[:100]}...")
```

**With re-ranking:**
```python
pipeline.enable_reranking()
results = pipeline.query_with_rerank("complex question", top_k=10, candidates_k=50)
```

**With filters:**
```python
# Filter by source file
results = pipeline.query_with_filters("Python guide", sources=["python.pdf"], top_k=5)

# Filter by metadata
results = pipeline.query_with_filters("recent updates", metadata_filters={"year": 2023}, top_k=5)
```

---

## Fusion Methods

```python
pipeline.enable_hybrid(fusion_method="rrf")       # Reciprocal Rank Fusion (default)
pipeline.enable_hybrid(fusion_method="weighted")   # Tunable α balance
pipeline.enable_hybrid(fusion_method="simple")     # Fast deduplication
```

---

## Chunk Size

| Size | Tradeoff |
|------|----------|
| 256 tokens | More precise, less context |
| 512 tokens | Balanced (recommended) |
| 1024 tokens | More context, less precise |

---

## When to Use Each Method

| Scenario | Recommended |
|----------|------------|
| Conceptual queries | Dense only |
| Exact terms / version numbers | Sparse only |
| Mixed queries (most cases) | Hybrid |
| Top-3 quality critical | Hybrid + re-ranking |
| High throughput (1000+ qps) | Skip re-ranking |

---

## API

```bash
python run_server.py
```

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to use Python?", "top_k": 3}'

# Metrics
curl http://localhost:8000/metrics
```

---

## Troubleshooting

**Slow queries** — check cache hit rate, reduce `chunk_size`, skip re-ranking, use dense-only.

**Low quality** — increase `top_k`, enable re-ranking, switch to hybrid, tune chunk size.

**High memory** — use larger chunks (fewer total), use sparse-only, clear embedding cache.
