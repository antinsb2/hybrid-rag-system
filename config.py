"""
Production configuration.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings from environment variables.
    """
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Pipeline settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    use_mock_llm: bool = True
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Retrieval settings
    use_hybrid: bool = True
    fusion_method: str = "rrf"
    enable_reranking: bool = False
    
    # Performance settings
    max_query_length: int = 1000
    max_results: int = 20
    default_top_k: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/api.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
```

Create example `.env.example`:
```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# Pipeline Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Settings
USE_MOCK_LLM=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Retrieval Settings
USE_HYBRID=true
FUSION_METHOD=rrf
ENABLE_RERANKING=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log
