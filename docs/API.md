# API Documentation

REST API for the Hybrid RAG System.

## Starting the Server
```bash
python run_server.py
```

Server runs on `http://localhost:8000`

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "indexed_documents": 150,
  "capabilities": ["document_ingestion", "hybrid_retrieval", "query", "answer_generation"]
}
```

### GET /stats

System statistics.

**Response:**
```json
{
  "is_indexed": true,
  "num_vectors": 150,
  "embedding_dimension": 384,
  "cache_stats": {
    "size": 150,
    "hits": 45,
    "misses": 150,
    "hit_rate": 0.23
  }
}
```

### POST /ingest

Ingest documents into the system.

**Request:**
```json
{
  "file_paths": ["doc1.pdf", "doc2.txt"],
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**Response:**
```json
{
  "message": "Ingesting 2 documents",
  "status": "processing"
}
```

### POST /query

Query the system and get an answer.

**Request:**
```json
{
  "query": "How do I configure SSL?",
  "top_k": 5,
  "min_score": 0.3,
  "use_reranking": false
}
```

**Response:**
```json
{
  "answer": "To configure SSL...",
  "sources": [
    {
      "text": "SSL configuration requires...",
      "score": 0.87,
      "source": "ssl_guide.pdf",
      "rank": 1
    }
  ],
  "num_chunks": 3,
  "query": "How do I configure SSL?"
}
```

### POST /query/retrieve-only

Retrieve relevant chunks without generating answer.

**Request:**
```json
{
  "query": "What is Flask?",
  "top_k": 5,
  "min_score": 0.3
}
```

**Response:**
```json
{
  "query": "What is Flask?",
  "results": [
    {
      "text": "Flask is a web framework...",
      "score": 0.92,
      "rank": 1,
      "metadata": {"source": "flask.txt"}
    }
  ],
  "num_results": 5
}
```

## Example Usage

### Python
```python
import requests

# Query
response = requests.post("http://localhost:8000/query", json={
    "query": "What is machine learning?",
    "top_k": 3
})

result = response.json()
print(result["answer"])
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "top_k": 3}'
```

## Error Responses

**400 Bad Request:** Invalid input or no documents indexed
**404 Not Found:** File not found during ingestion
**422 Validation Error:** Request doesn't match schema
**500 Internal Error:** Server error

## Rate Limiting

Currently no rate limiting. Add in production deployment.
