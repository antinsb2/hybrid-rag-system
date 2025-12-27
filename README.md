# Hybrid RAG System

High-performance Retrieval-Augmented Generation combining dense embeddings and sparse keyword search for superior retrieval quality.

## What Makes This "Hybrid"?

Traditional RAG uses only **dense retrieval** (embeddings). This system combines:

-  **Dense Retrieval**: Semantic similarity via embeddings
-  **Sparse Retrieval**: Keyword matching via BM25
-  **Smart Fusion**: Combines both for best results

**Why hybrid?** Dense finds semantically similar content, sparse catches exact terms. Together they outperform either alone.

## Features

- Multi-format document processing (PDF, DOCX, HTML, Markdown, TXT)
- Hybrid search with configurable fusion strategies
- Custom vector indexing with HNSW
- Comprehensive benchmarks (dense vs sparse vs hybrid)
- Production-ready API with observability
- Fine-tunable embeddings for domain adaptation

## Architecture
```
Documents → Processing → [Dense Index + Sparse Index]
                              ↓
Query → [Dense Search + Sparse Search] → Fusion → Results
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## Plan for next 3-4 weeks

** Document processing and chunking
** Embedding pipeline and vector indexing
** Sparse retrieval and hybrid fusion
** API, benchmarks, optimization

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
