# Usage Guide

Complete guide for using the Hybrid RAG System.

## Quick Start

### 1. Installation
```bash
git clone https://github.com/antinsb2/hybrid-rag-system.git
cd hybrid-rag-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Documents

Place your documents in a directory:
```
data/
├── guide1.pdf
├── manual.docx
└── notes.txt
```

### 3. Basic Usage
```python
from rag.pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(chunk_size=512)

# Ingest documents
pipeline.ingest_documents(['data/guide1.pdf', 'data/manual.docx'])

# Enable features
pipeline.enable_hybrid(fusion_method="rrf")
pipeline.enable_generation(use_mock=True)

# Ask questions
result = pipeline.ask("How do I configure SSL?")
print(result["answer"])
```

## Advanced Usage

### Custom Chunk Size
```python
# Smaller chunks for more precision
pipeline = RAGPipeline(chunk_size=256, chunk_overlap=30)

# Larger chunks for more context
pipeline = RAGPipeline(chunk_size=1024, chunk_overlap=100)
```

### Different Fusion Methods
```python
# Reciprocal Rank Fusion (balanced)
pipeline.enable_hybrid(fusion_method="rrf")

# Weighted combination (tune with alpha)
pipeline.enable_hybrid(fusion_method="weighted")
```

### Retrieval Without Generation
```python
# Just get relevant chunks
results = pipeline.query("Python features", top_k=5)

for result in results:
    print(f"[{result.score:.3f}] {result.text[:100]}...")
```

### With Re-ranking
```python
# Enable re-ranking for better quality
pipeline.enable_reranking()

# Query with re-ranking (slower but more accurate)
result = pipeline.query_with_rerank("complex question", top_k=10, candidates_k=50)
```

### Filtering Results
```python
# Filter by source
results = pipeline.query_with_filters(
    "Python guide",
    sources=["python.pdf"],
    top_k=5
)

# Filter by metadata
results = pipeline.query_with_filters(
    "recent updates",
    metadata_filters={"year": 2023},
    top_k=5
)
```

## API Usage

### Start Server
```bash
python run_server.py
```

### Query via API
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What is Flask?",
    "top_k": 5,
    "min_score": 0.3
})

result = response.json()
print(result["answer"])
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to use Python?",
    "top_k": 3
  }'

# Metrics
curl http://localhost:8000/metrics
```

## Best Practices

### Chunk Size Selection

- **Small (256 tokens)**: Precise retrieval, less context
- **Medium (512 tokens)**: Balanced (recommended)
- **Large (1024 tokens)**: More context, less precise

### When to Use Hybrid

**Use hybrid when:**
- Queries mix concepts and exact terms
- Need best possible quality
- Can afford ~20ms extra latency

**Use dense-only when:**
- Purely conceptual queries
- Speed is critical
- Lower memory footprint needed

**Use sparse-only when:**
- Exact keyword matching required
- Very fast responses needed
- Limited compute resources

### Re-ranking Trade-offs

**Enable re-ranking when:**
- Top-3 results quality matters most
- User-facing search (UX critical)
- Can afford 50-100ms latency

**Skip re-ranking when:**
- High throughput needed (1000+ qps)
- Latency budget is tight
- Returning many results (top-50+)

## Troubleshooting

### Slow Queries

- Check cache hit rate (should be >50% for repeated queries)
- Reduce chunk_size if many chunks
- Use dense-only instead of hybrid
- Skip re-ranking

### Low Quality Results

- Increase top_k (get more candidates)
- Enable re-ranking
- Use hybrid instead of single method
- Tune chunk_size and overlap

### High Memory Usage

- Reduce chunk_size (fewer, larger chunks)
- Use sparse-only retrieval
- Clear embedding cache periodically

## Performance Tuning

### Optimize for Speed
```python
pipeline = RAGPipeline(
    chunk_size=512,
    use_hnsw=False,  # Linear is faster for <10K docs
    use_cache=True
)

# Dense-only, no re-ranking
results = pipeline.query("question", top_k=5)
```

### Optimize for Quality
```python
pipeline = RAGPipeline(chunk_size=256)  # Smaller chunks
pipeline.enable_hybrid(fusion_method="rrf")
pipeline.enable_reranking()

# Two-stage with re-ranking
results = pipeline.query_with_rerank("question", top_k=10, candidates_k=100)
```

### Optimize for Scale
```python
# Would use HNSW when available
pipeline = RAGPipeline(
    chunk_size=512,
    use_hnsw=True,  # Faster for >10K docs
    use_cache=True
)
```
