# Hybrid RAG System

Production-grade Retrieval-Augmented Generation combining dense embeddings and sparse keyword search for 12-15% better recall than dense-only approaches.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

```bash
git clone https://github.com/antinsb2/hybrid-rag-system.git
cd hybrid-rag-system
pip install -r requirements.txt
```

**Python:**
```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_documents(['your_doc.pdf'])
pipeline.enable_hybrid()
pipeline.enable_generation(use_mock=True)

result = pipeline.ask("Your question?")
print(result["answer"])
```

**API:**
```bash
python run_server.py
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Flask?"}'
```

---

## How It Works

```
Documents → Processing → [Dense Index + Sparse Index]
                                  ↓
Query → [Dense + Sparse] → Fusion → Re-rank → LLM → Answer
```

- **Dense retrieval** — semantic search via Sentence Transformers
- **Sparse retrieval** — exact keyword matching via BM25
- **Fusion** — Reciprocal Rank Fusion (RRF), weighted, or simple combination
- **Re-ranking** — cross-encoder for a quality boost on top results

---

## Performance

Measured on a 1000-document corpus:

| Method | Latency (P95) | Recall@10 |
|--------|--------------|-----------|
| Sparse (BM25) | ~10ms | — |
| Dense | ~25ms | baseline |
| Hybrid | ~30ms | +12% |
| Hybrid + Re-rank | ~100ms | +20% P@3 |

- Memory: ~150MB per 1000 documents
- Embedding cache: 100-500x speedup on repeated queries

---

## Features

- Multi-format document loading: PDF, DOCX, HTML, Markdown, TXT
- Token-aware chunking with overlap
- HNSW indexing for large-scale search
- OpenAI, Anthropic, or mock LLM integration
- FastAPI REST API with request logging and metrics
- Docker support

---

## Testing

```bash
cd tests
python run_all_tests.py        # all tests
python test_integration.py     # integration only
python test_api.py             # API only
```

13 test suites, 50+ test cases covering unit, integration, and API layers.

---

## Benchmarks

```bash
cd benchmarks
python retrieval_quality.py
python latency_benchmark.py
python memory_usage.py
python generate_report.py
```

---

## Docker

```bash
docker build -t hybrid-rag-system .
docker-compose up -d
curl http://localhost:8000/health
```

---

## Docs

- [Usage Guide](docs/USAGE_GUIDE.md)
- [API Reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Deployment](docs/DEPLOYMENT.md)

---

## Stack

Python 3.9+, PyTorch, Sentence Transformers, FastAPI, hnswlib, Pydantic, Docker

---

MIT License — [Antin Selvaraj](https://github.com/antinsb2)
