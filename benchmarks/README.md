# Benchmarks

Comprehensive performance analysis of the hybrid RAG system.

## Available Benchmarks

### `retrieval_quality.py`
Measures retrieval quality using recall, precision, and nDCG metrics.

**Compares:**
- Dense retrieval
- Sparse retrieval (BM25)
- Hybrid fusion (RRF and weighted)

**Run:** `python retrieval_quality.py`

### `latency_benchmark.py`
Measures query latency at different scales.

**Metrics:**
- Average latency
- P50, P95, P99 percentiles
- Throughput (queries/sec)
- Cache speedup

**Run:** `python latency_benchmark.py`

### `memory_usage.py`
Analyzes memory consumption by component.

**Measures:**
- Memory per document
- Component breakdown (embeddings, text, metadata)
- Dense vs sparse comparison

**Run:** `python memory_usage.py`

### `generate_report.py`
Creates comprehensive performance summary.

**Outputs:** `PERFORMANCE_REPORT.md` with all metrics

**Run:** `python generate_report.py`

## Key Findings

**Hybrid retrieval outperforms single methods:**
- 10-15% better recall than dense alone
- Captures both semantic and exact matches
- Acceptable latency overhead (~5-10ms)

**Cache is critical:**
- 100x+ speedup on repeated queries
- Essential for production deployment

**Trade-offs:**
- Dense: Higher memory, better semantics
- Sparse: Lower memory, better exact matching
- Hybrid: Best quality, moderate resources
