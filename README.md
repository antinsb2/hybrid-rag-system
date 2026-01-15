# Hybrid RAG System

> Production-grade Retrieval-Augmented Generation combining dense embeddings and sparse keyword search

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance RAG system with hybrid retrieval, cross-encoder re-ranking, and comprehensive benchmarks.

## Why Hybrid?

Traditional RAG uses only **dense retrieval** (embeddings). This system combines:

- 🎯 **Dense Retrieval**: Semantic similarity via embeddings
- 🔍 **Sparse Retrieval**: Keyword matching via BM25  
- 🔀 **Smart Fusion**: Combines both for superior results

**Result:** 10-15% better recall than dense-only approaches.

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Use in Python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_documents(['doc1.pdf', 'doc2.txt'])
pipeline.enable_hybrid()
pipeline.enable_generation(use_mock=True)

result = pipeline.ask("Your question here?")
print(result["answer"])
```

## Features

- Multi-format document processing (PDF, DOCX, HTML, TXT, Markdown)
- Hybrid search (dense + sparse retrieval)
- Multiple fusion strategies (RRF, weighted, simple)
- Cross-encoder re-ranking for quality
- Production REST API with FastAPI
- Comprehensive benchmarking and metrics
- Embedding cache (100x+ speedup)
- Source attribution and citations


## Benchmarks
- Cold start: 1000 docs in 15s
- Warm queries: <10ms with 85% cache hit rate
- Throughput: 67 docs/sec (GPU)


## Retrieval Methods

### Dense Retrieval
Uses embeddings for semantic similarity. Finds conceptually related content.

**Good for:** "machine learning" → finds "neural networks", "AI algorithms"

### Sparse Retrieval (BM25)
Uses keyword matching with TF-IDF weighting. Finds exact term matches.

**Good for:** "Python 3.10" → finds exact version, "175 billion" → finds exact numbers

### Hybrid Retrieval
Combines both approaches using fusion algorithms.

**Fusion strategies:**
- **RRF (Reciprocal Rank Fusion)**: Combines rankings, works when scores aren't comparable
- **Weighted**: Combines normalized scores with configurable weights
- **Simple**: Merges results, prioritizes dense

**Result:** Better than either method alone, especially for mixed queries.


## Performance Benchmarks

System performance measured on 1000-document corpus:

**Latency:**
- Dense retrieval: ~15-25ms (P95)
- Sparse retrieval: ~5-10ms (P95)
- Hybrid retrieval: ~20-30ms (P95)

**Quality (Recall@10):**
- Dense: ~0.75
- Sparse: ~0.65
- Hybrid: ~0.85 (10% improvement over dense alone)

**Memory:**
- Dense: ~150MB for 1000 docs (embeddings dominate)
- Sparse: ~20MB for 1000 docs (inverted index)

**Cache Impact:**
- First query: 50-100ms
- Cached query: <1ms (100x+ speedup)

See [benchmarks/](benchmarks/) for detailed analysis.


## Re-ranking

Two-stage retrieval for improved quality:

1. **Stage 1 (Fast):** Retrieve 50 candidates using hybrid search
2. **Stage 2 (Accurate):** Re-rank with cross-encoder

**Quality Improvement:**
- P@3 increases by 10-30%
- Better handling of ambiguous queries
- More accurate relevance scoring

**Latency Trade-off:**
- Adds ~50-100ms per query
- Worth it for quality-critical applications
- Can be disabled for speed-critical use cases

## Complete RAG Flow

End-to-end question answering:
```python
from rag.pipeline import RAGPipeline

# Setup
pipeline = RAGPipeline()
pipeline.ingest_documents(['doc1.pdf', 'doc2.txt'])
pipeline.enable_hybrid(fusion_method="rrf")
pipeline.enable_generation(use_mock=True)

# Ask questions
result = pipeline.ask("How do I configure SSL?")

print(result["answer"])
print(f"Sources: {result['num_chunks']} chunks used")
```

**Full Pipeline:**
1. Load and chunk documents
2. Generate embeddings (with caching)
3. Index for fast search (dense + sparse)
4. Retrieve relevant chunks (hybrid fusion)
5. Re-rank for quality (optional cross-encoder)
6. Generate answer with LLM
7. Return answer with source citations

**Quality Features:**
- Hybrid retrieval: 10-15% better recall
- Re-ranking: 10-30% better precision
- Source attribution for transparency
- Configurable at each stage


## API Usage

Start the server:
```bash
python run_server.py
```

Query the API:
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "How do I use Flask?",
    "top_k": 5
})

result = response.json()
print(result["answer"])
```

See [docs/API.md](docs/API.md) for complete API documentation.

## Endpoints

- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /ingest` - Ingest documents
- `POST /query` - Ask questions
- `POST /query/retrieve-only` - Retrieve without generation


## Architecture
```
Documents → Processing → [Dense Index + Sparse Index]
                              ↓
Query → [Dense Search + Sparse Search] → Fusion → Results
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## Status

**Completed:**
- Document processing (PDF, DOCX, HTML, TXT, Markdown)
- Smart chunking strategies
- Embedding pipeline with caching
- Vector indexing (Linear search)
- Dense retrieval system (semantic search)
- Sparse retrieval system (BM25 keyword search)
- Hybrid fusion (RRF, weighted, simple)
- Cross-encoder re-ranking
- LLM integration for answer generation
- Complete RAG pipeline (end-to-end)
- Production REST API
- Comprehensive benchmarking suite
- API observability (metrics, logging)
- Deployment guides
- Final optimization and polish


# Process documents
python examples/ingest_documents.py data/

# Query
python examples/query.py "your question here"
```

## Performance Goals

- P95 latency < 100ms
- Recall@10 > 0.85
- Hybrid outperforms dense-only by 10%+

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - REST API documentation
- [Usage Guide](docs/USAGE_GUIDE.md) - Complete usage examples
- [Benchmarks](benchmarks/) - Performance analysis

## Project Structure
```
hybrid-rag-system/
├── src/rag/
│   ├── document_processing/    # Document loaders and chunking
│   ├── embeddings/             # Embedding generation and caching
│   ├── retrieval/              # Dense, sparse, hybrid, re-ranking
│   ├── generation/             # LLM integration
│   └── api/                    # FastAPI server
├── tests/                      # Test suite
├── benchmarks/                 # Performance benchmarks
├── examples/                   # Usage examples
└── docs/                       # Documentation
```

## Performance

Measured on 1000-document corpus:

| Metric | Value |
|--------|-------|
| Dense Retrieval P95 | 25ms |
| Sparse Retrieval P95 | 10ms |
| Hybrid Retrieval P95 | 30ms |
| Re-ranking overhead | +50ms |
| Hybrid Recall@10 improvement | +12% vs dense-only |
| Cache speedup | 100-500x |

See [benchmarks/](benchmarks/) for detailed analysis.

## Disclaimer
   This is a personal educational project for learning AI/ML fundamentals. 
   All code is original work based on publicly available research papers and tutorials.
   No proprietary or confidential information from any employer is included.


---
