# Configuration

All settings can be set via environment variables or a `.env` file.

```bash
cp .env.example .env
```

---

## Variables

### API
| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Host to bind |
| `API_PORT` | `8000` | Port to listen |
| `API_RELOAD` | `false` | Auto-reload on code changes |

### Pipeline
| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

### LLM
| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_LLM` | `true` | Use mock LLM (no API cost) |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |

### Retrieval
| Variable | Default | Description |
|----------|---------|-------------|
| `USE_HYBRID` | `true` | Enable hybrid retrieval |
| `FUSION_METHOD` | `rrf` | `rrf`, `weighted`, or `simple` |
| `ENABLE_RERANKING` | `false` | Enable cross-encoder re-ranking |

### Query
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_QUERY_LENGTH` | `1000` | Max query characters |
| `MAX_RESULTS` | `20` | Max results per query |
| `DEFAULT_TOP_K` | `5` | Default number of results |

### Logging
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `logs/api.log` | Log file path |

---

## Recommended Settings

**Development:**
```
API_RELOAD=true
USE_MOCK_LLM=true
ENABLE_RERANKING=false
LOG_LEVEL=DEBUG
```

**Production:**
```
API_RELOAD=false
USE_MOCK_LLM=false
LLM_PROVIDER=openai
ENABLE_RERANKING=true
LOG_LEVEL=WARNING
```
