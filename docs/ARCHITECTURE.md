# Hybrid RAG System Architecture

## Overview
Production-grade Retrieval-Augmented Generation system combining dense and sparse retrieval methods for superior search quality.

## What is Hybrid RAG?

**Hybrid = Dense + Sparse Retrieval**

- **Dense Retrieval**: Uses embeddings for semantic similarity
  - Finds conceptually similar content
  - Example: "ML model" matches "neural network"
  - Tech: Sentence transformers, vector similarity

- **Sparse Retrieval**: Uses keyword matching (BM25)
  - Finds exact term matches
  - Example: "Python 3.10" matches exact phrase
  - Tech: BM25 algorithm, inverted index

- **Hybrid**: Combines both for best results
  - Dense catches semantic matches
  - Sparse catches exact terms
  - Fusion algorithm merges results

## System Components

### 1. Document Processing
- Multi-format support (PDF, DOCX, HTML, Markdown, TXT)
- Metadata extraction (author, date, source)
- Smart chunking strategies:
  - Token-based chunking with overlap
  - Sentence-aware chunking
  - Structure-aware (preserve headers, lists)

### 2. Embedding Pipeline
- Sentence transformers for dense retrieval
- Fine-tuning on domain data
- Batch processing with rate limiting
- Embedding cache for performance

### 3. Hybrid Search Engine
- **Dense retrieval**: Embedding similarity search
- **Sparse retrieval**: BM25 term matching
- Re-ranking with cross-encoder
- Fusion strategies (RRF, weighted)

### 4. Vector Store
- Custom HNSW implementation
- Qdrant integration for comparison
- Metadata filtering
- Persistence and versioning

### 5. API Layer
- FastAPI endpoints
- Streaming responses
- Rate limiting
- Observability (metrics, logging)

## Data Flow
```
Documents → Chunking → Embedding → Indexing
                ↓
Query → [Dense Search + Sparse Search] → Fusion → Re-rank → Results → LLM
```

## Current Implementation Status

### Retrieval Components (Completed)

**Dense Retrieval:**
- Sentence transformers for embeddings
- Vector similarity search (cosine)
- Linear and HNSW indexing options
- Query expansion support

**Sparse Retrieval:**
- BM25 algorithm from scratch
- Inverted index for efficient term lookup
- Configurable k1 and b parameters
- Term frequency analysis

**Hybrid Fusion:**
- Reciprocal Rank Fusion (RRF)
- Weighted score combination
- Simple merge strategy
- Automatic result deduplication

**Quality Features:**
- Result filtering (score, source, metadata)
- Ranking with boosting
- Deduplication
- Persistence for all components


## Performance Targets
- Latency: P95 < 100ms (retrieval only)
- Throughput: 1000+ queries/sec
- Recall@10: > 0.85 (hybrid should beat pure dense)
- Cost: < $0.01 per 1000 queries

## Tech Stack
- Python 3.10+
- PyTorch, Transformers
- Sentence-Transformers (dense retrieval)
- BM25 implementation (sparse retrieval)
- Qdrant (vector database)
- FastAPI (API)
- Pytest (testing)

## Benchmarking Strategy

Compare performance across:
- Dense-only retrieval
- Sparse-only retrieval (BM25)
- Hybrid fusion
- Different fusion strategies

Metrics: nDCG, Recall@K, MRR, latency, cost

### Vector Indexing

**Linear Search (Baseline)**
- Brute force comparison
- 100% accurate
- O(n) complexity
- Good for <10K vectors

**HNSW (Production)**
- Graph-based approximate search
- ~95% recall
- O(log n) complexity
- Good for millions of vectors
- 100x+ faster than linear

**Performance:**
- Linear: ~10ms for 10K vectors
- HNSW: ~0.1ms for 1M vectors

- ## Re-ranking Stage

**Cross-Encoder Re-ranking:**
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Jointly encodes query + document (more accurate than bi-encoder)
- Applied to top-50 candidates from hybrid retrieval
- Returns top-10 after re-scoring

**Two-Stage Pipeline:**
```
Query → Hybrid Retrieval (50 candidates) → Cross-Encoder Re-rank (10 results)
```

**Performance:**
- Quality: 10-30% better P@3
- Latency: +50-100ms overhead
- Memory: Minimal (model loaded once)

**When to use:**
- Quality matters more than speed
- Small result sets (top-10)
- User-facing search (better UX)

**When to skip:**
- High-throughput scenarios
- Large result sets (top-100+)
- Latency-critical applications
