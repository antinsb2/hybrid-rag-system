# Hybrid RAG System

> Production-grade Retrieval-Augmented Generation combining dense embeddings and sparse keyword search

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance RAG system with hybrid retrieval, cross-encoder re-ranking, and comprehensive benchmarks. Built from first principles to understand production AI systems.

---

## 🎯 Why This Exists

Most RAG implementations use only dense retrieval. This system demonstrates production-grade hybrid search combining:

- **Dense Retrieval** → Semantic understanding
- **Sparse Retrieval** → Exact keyword matching  
- **Intelligent Fusion** → Best of both worlds

**Result:** 12-15% better recall than dense-only approaches with comprehensive benchmarks to prove it.

---

## ⚡ Quick Start
```bash
# Install
git clone https://github.com/antinsb2/hybrid-rag-system.git
cd hybrid-rag-system
pip install -r requirements.txt

# Run
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_documents(['your_doc.pdf'])
pipeline.enable_hybrid()
pipeline.enable_generation(use_mock=True)

result = pipeline.ask("Your question?")
print(result["answer"])
```

**Or via API:**
```bash
python run_server.py
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is Flask?"}'
```

---

## 🚀 Features

### Retrieval Methods

| Method | Use Case | Latency | Quality |
|--------|----------|---------|---------|
| Dense | Semantic queries | ~25ms | Good |
| Sparse (BM25) | Exact keywords | ~10ms | Good |
| Hybrid | Best overall | ~30ms | Excellent |
| + Re-ranking | Highest quality | ~100ms | Best |

### Core Capabilities

✅ **Multi-format Processing** - PDF, DOCX, HTML, Markdown, TXT  
✅ **Smart Chunking** - Token-based and sentence-aware  
✅ **Embedding Cache** - 100-500x speedup on repeated queries  
✅ **Hybrid Search** - Three fusion algorithms (RRF, weighted, simple)  
✅ **Re-ranking** - Cross-encoder for top results  
✅ **LLM Integration** - OpenAI, Anthropic, or mock  
✅ **Production API** - FastAPI with observability  
✅ **Comprehensive Benchmarks** - Quality, latency, memory  

---

## 📊 Performance

Measured on 1000-document corpus:

**Quality Metrics:**
- Hybrid Recall@10: **0.85** (+12% vs dense-only)
- With re-ranking P@3: **+20% improvement**
- Cache hit rate: **40-80%** (workload dependent)

**Latency (P95):**
- Retrieval only: **30ms**
- With re-ranking: **100ms**
- End-to-end: **<150ms**

**Memory:**
- Per 1000 documents: **~150MB**
- Embedding cache: **~0.5MB** per 1000 items

See [benchmarks/](benchmarks/) for detailed analysis.

---

## 🏗️ Architecture
```
Documents → Processing → [Dense Index + Sparse Index]
                              ↓
Query → [Dense + Sparse] → Fusion → Re-rank → LLM → Answer
```

**Key Components:**
- **Document Processor**: Multi-format loaders, smart chunking
- **Embedding Pipeline**: Sentence transformers, caching layer
- **Hybrid Retriever**: Dense (embeddings) + Sparse (BM25) + Fusion
- **Re-ranker**: Cross-encoder for quality boost
- **Generator**: LLM integration with citations

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

---

## 📖 Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Complete examples and patterns
- **[API Reference](docs/API.md)** - REST endpoint documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Docker and cloud deployment
- **[Configuration](docs/CONFIGURATION.md)** - Environment variables and settings
- **[Architecture](docs/ARCHITECTURE.md)** - System design deep dive

---

## 🧪 Testing
```bash
# Run all tests
cd tests
python run_all_tests.py

# Run specific test suite
python test_integration.py
python test_api.py
```

**Test Coverage:**
- 13 test suites
- 50+ test cases
- Integration, unit, and API tests
- Benchmarking suite

---

## 🐳 Docker Deployment
```bash
# Build
docker build -t hybrid-rag-system .

# Run
docker-compose up -d

# Access
curl http://localhost:8000/health
```

---

## 💡 Use Cases

**Enterprise Knowledge Base**
- Index company documentation
- Answer employee questions
- Cite specific sources

**Technical Documentation Search**
- API documentation retrieval
- Code example finding
- Best practice recommendations

**Research Assistant**
- Paper summarization
- Cross-reference finding
- Literature review support

---

## 🎓 Technical Highlights

**What makes this implementation unique:**

1. **Complete hybrid approach** - Most implementations do dense-only
2. **Comprehensive benchmarks** - Quantified improvements with metrics
3. **Production-ready** - API, logging, metrics, error handling
4. **Well-documented** - Architecture decisions and trade-offs explained
5. **Tested** - 50+ tests covering all components

**Built from scratch to understand:**
- How embeddings work in production
- Trade-offs between retrieval methods
- When to use re-ranking
- Real-world performance characteristics

---

## 🔧 Technology Stack

**Core:**
- Python 3.9+
- PyTorch
- Sentence Transformers
- FastAPI

**Retrieval:**
- Dense: Sentence transformers embeddings
- Sparse: BM25 algorithm
- Re-ranking: Cross-encoder models

**Infrastructure:**
- Docker for deployment
- Pydantic for validation
- Uvicorn for serving

---

## 📈 Benchmarks

Run benchmarks yourself:
```bash
cd benchmarks
python retrieval_quality.py
python latency_benchmark.py
python memory_usage.py
python generate_report.py
```

---

## 🤝 Contributing

This is a learning project built to understand production RAG systems. 

**Areas for enhancement:**
- HNSW indexing for better scale
- Additional embedding models
- More fusion strategies
- Streaming API responses
- Authentication layer

---

## 📝 License

MIT License - see LICENSE file

---

## 👤 Author

**Antin Selvaraj**

- GitHub: [@antinsb2](https://github.com/antinsb2)

---

## 🌟 Acknowledgments

**Inspired by:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- Production RAG systems at scale
- Open source sentence-transformers library

**Built to learn:**
- Production AI system architecture
- Hybrid retrieval strategies
- Performance optimization
- Real-world trade-offs

---

*If you find this useful, please ⭐ star the repo!*

---
