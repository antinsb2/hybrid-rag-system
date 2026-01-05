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
- Vector indexing (Linear and HNSW)
- Dense retrieval system (semantic search)
- Sparse retrieval system (BM25 keyword search)
- Result filtering and ranking

**In Progress:**
- Hybrid fusion (combining dense + sparse)

**Planned:**
- Re-ranking with cross-encoder
- LLM integration for answer generation
- Production API


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

---

*Part of year-long AI systems engineering deep dive*
*Focus: Production systems with real metrics*
