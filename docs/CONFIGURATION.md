# Configuration Guide

## Environment Variables

All settings can be configured via environment variables or `.env` file.

### API Settings

- `API_HOST`: Host to bind (default: 0.0.0.0)
- `API_PORT`: Port to listen (default: 8000)
- `API_RELOAD`: Auto-reload on code changes (default: false)

### Pipeline Settings

- `CHUNK_SIZE`: Text chunk size in tokens (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)

### LLM Settings

- `USE_MOCK_LLM`: Use mock LLM for testing (default: true)
- `LLM_PROVIDER`: openai or anthropic (default: openai)
- `LLM_MODEL`: Model name (default: gpt-3.5-turbo)
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key

### Retrieval Settings

- `USE_HYBRID`: Enable hybrid retrieval (default: true)
- `FUSION_METHOD`: rrf, weighted, or simple (default: rrf)
- `ENABLE_RERANKING`: Use cross-encoder re-ranking (default: false)

### Performance Settings

- `MAX_QUERY_LENGTH`: Maximum query characters (default: 1000)
- `MAX_RESULTS`: Maximum results per query (default: 20)
- `DEFAULT_TOP_K`: Default number of results (default: 5)

### Logging

- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `LOG_FILE`: Log file path (default: logs/api.log)

## Configuration File

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Production Settings

Recommended for production:
```
API_RELOAD=false
USE_MOCK_LLM=false
LLM_PROVIDER=openai
ENABLE_RERANKING=true
LOG_LEVEL=WARNING
```

## Development Settings

Recommended for development:
```
API_RELOAD=true
USE_MOCK_LLM=true
ENABLE_RERANKING=false
LOG_LEVEL=DEBUG
```
