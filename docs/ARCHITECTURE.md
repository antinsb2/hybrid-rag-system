# Architecture

## Data Flow

```
Documents → Chunking → Embedding → Indexing
                ↓
Query → [Dense Search + Sparse Search] → Fusion → Re-rank → LLM → Answer
```

---

## Components

### Document Processing
- Multi-format loaders: PDF, DOCX, HTML, Markdown, TXT
- Token-based chunking with overlap (default: 512 tokens, 50 overlap)
- Metadata extraction (source, author, date)

### Embedding Pipeline
- Sentence Transformers (default: `all-MiniLM-L6-v2`)
- Batch processing
- Disk-based embedding cache

### Retrieval

**Dense** — cosine similarity over embeddings. Finds semantically similar content even with different wording.

**Sparse** — BM25 keyword matching. Finds exact terms, version numbers, specific identifiers.

**Hybrid** — runs both, merges results via fusion. Catches what either method alone would miss.

### Fusion Strategies
- **RRF (default)** — `1 / (k + rank)` — robust, doesn't require score normalization
- **Weighted** — `α * dense + (1-α) * sparse` — tunable balance
- **Simple** — deduplicates and prioritizes dense results

### Vector Indexing

| Type | When to use | Complexity |
|------|------------|------------|
| Linear (baseline) | < 10K documents | O(n), 100% accurate |
| HNSW | > 10K documents | O(log n), ~95% recall |

### Re-ranking
Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly scores query + document pairs. Applied to top-50 candidates, returns top-10.

- Quality: +10-30% P@3
- Latency: +50-100ms
- Use when quality matters more than speed

### Generation
- LLM providers: OpenAI, Anthropic, or mock
- Filters chunks below `min_score` threshold
- Returns answer + source citations

---

## Tech Stack

Python 3.9+, PyTorch, Sentence Transformers, hnswlib, FastAPI, Pydantic, Docker
