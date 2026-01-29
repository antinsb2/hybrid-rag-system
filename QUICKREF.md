# Quick Reference

Fast reference for common operations.

## Setup (One-time)
```bash
pip install -r requirements.txt
```

## Basic Usage
```python
from rag.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Index documents
pipeline.ingest_documents(['doc1.pdf', 'doc2.txt'])

# Enable features
pipeline.enable_hybrid()
pipeline.enable_generation(use_mock=True)

# Ask questions
result = pipeline.ask("Your question?")
print(result["answer"])
```

## Common Patterns

### Retrieval Only (No Answer)
```python
results = pipeline.query("search query", top_k=10)
for r in results:
    print(f"{r.score:.3f}: {r.text[:100]}")
```

### With Re-ranking
```python
pipeline.enable_reranking()
results = pipeline.query_with_rerank("question", top_k=10, candidates_k=50)
```

### Filter by Source
```python
results = pipeline.query_with_filters(
    "query",
    sources=["specific_doc.pdf"],
    top_k=5
)
```

## API Usage

### Start Server
```bash
python run_server.py
```

### Query API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Flask?", "top_k": 3}'
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Metrics
```bash
curl http://localhost:8000/metrics
```

## Docker
```bash
# Build
docker build -t hybrid-rag .

# Run
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

## Testing
```bash
# All tests
cd tests && python run_all_tests.py

# Specific test
python tests/test_integration.py

# Benchmarks
cd benchmarks && python retrieval_quality.py
```

## Configuration

Edit `.env` file:
```
USE_MOCK_LLM=true
CHUNK_SIZE=512
ENABLE_RERANKING=false
LOG_LEVEL=INFO
```

## Troubleshooting

**Slow queries?**
- Check cache hit rate
- Reduce top_k
- Skip re-ranking
- Use sparse-only

**Low quality?**
- Enable hybrid
- Add re-ranking
- Increase top_k
- Reduce chunk_size

**Out of memory?**
- Increase chunk_size
- Clear cache
- Use sparse-only
- Reduce document count

---

For detailed documentation, see [docs/](docs/)
