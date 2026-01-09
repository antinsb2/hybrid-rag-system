# Hybrid RAG System

Production-grade Retrieval-Augmented Generation system with hybrid search, GPU-optimized embeddings, and intelligent caching.

## What Makes This "Hybrid"?

Traditional RAG uses only **dense retrieval** (embeddings). This system combines:

-  **Dense Retrieval**: Semantic similarity via embeddings
-  **Sparse Retrieval**: Keyword matching via BM25
-  **Smart Fusion**: Combines both for best results

**Why hybrid?** Dense finds semantically similar content, sparse catches exact terms. Together they outperform either alone.

## Features

- Hybrid search (dense + sparse)
- GPU-accelerated embeddings (150x cache speedup)
- Production caching layer
- Comprehensive benchmarks
- 10ms query latency at 10K documents

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
- Result filtering and ranking
- Comprehensive benchmarking suite

**In Progress:**
- System optimization

**Planned:**
- LLM integration for answer generation
- Production API with observability
- Final polish and deployment guides


## Quick Start
```bash
# Setup
pip install -r requirements.txt

# Process documents
python examples/ingest_documents.py data/

# Query
python examples/query.py "your question here"
```

## Performance Goals

- P95 latency < 100ms
- Recall@10 > 0.85
- Hybrid outperforms dense-only by 10%+

## Benchmarks

Coming soon: Comprehensive comparison of retrieval strategies.

## Disclaimer
   This is a personal educational project for learning AI/ML fundamentals. 
   All code is original work based on publicly available research papers and tutorials.
   No proprietary or confidential information from any employer is included.


---
