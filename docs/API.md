# API Reference

Start the server:
```bash
python run_server.py
# Runs on http://localhost:8000
```

---

## Endpoints

### GET /health
```json
{
  "status": "healthy",
  "indexed_documents": 150,
  "capabilities": ["document_ingestion", "hybrid_retrieval", "query", "answer_generation"]
}
```

### GET /stats
```json
{
  "is_indexed": true,
  "num_vectors": 150,
  "embedding_dimension": 384,
  "cache_stats": { "size": 150, "hits": 45, "misses": 150, "hit_rate": 0.23 }
}
```

### GET /metrics
```json
{
  "uptime_seconds": 3600,
  "total_requests": 1250,
  "total_queries": 800,
  "total_errors": 5,
  "query_latency": { "avg_ms": 45.2, "p50_ms": 38.5, "p95_ms": 89.3, "p99_ms": 125.7 },
  "cache": { "hits": 320, "misses": 480, "hit_rate": 0.4 }
}
```

### GET /metrics/detailed
Same as `/metrics` plus `requests_per_minute`, `error_rate`, `avg_chunks_per_query`.

### POST /ingest
```json
// Request
{ "file_paths": ["doc1.pdf", "doc2.txt"], "chunk_size": 512, "chunk_overlap": 50 }

// Response
{ "message": "Ingesting 2 documents", "status": "processing" }
```

### POST /query
```json
// Request
{ "query": "How do I configure SSL?", "top_k": 5, "min_score": 0.3, "use_reranking": false }

// Response
{
  "answer": "To configure SSL...",
  "sources": [{ "text": "SSL configuration requires...", "score": 0.87, "source": "ssl_guide.pdf", "rank": 1 }],
  "num_chunks": 3,
  "query": "How do I configure SSL?"
}
```

### POST /query/retrieve-only
```json
// Request
{ "query": "What is Flask?", "top_k": 5, "min_score": 0.3 }

// Response
{
  "query": "What is Flask?",
  "results": [{ "text": "Flask is a web framework...", "score": 0.92, "rank": 1, "metadata": {"source": "flask.txt"} }],
  "num_results": 5
}
```

---

## Errors

| Code | Meaning |
|------|---------|
| 400 | Invalid input or no documents indexed |
| 404 | File not found during ingestion |
| 422 | Request schema mismatch |
| 500 | Server error |
