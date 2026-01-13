"""
API request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class IngestRequest(BaseModel):
    """Request to ingest documents."""
    file_paths: List[str] = Field(..., description="List of document file paths")
    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks")


class QueryRequest(BaseModel):
    """Request to query the system."""
    query: str = Field(..., description="User question", min_length=1)
    top_k: int = Field(5, description="Number of results", ge=1, le=20)
    min_score: float = Field(0.3, description="Minimum relevance score", ge=0.0, le=1.0)
    use_reranking: bool = Field(False, description="Use cross-encoder re-ranking")


class Source(BaseModel):
    """Source information."""
    text: str
    score: float
    source: str
    rank: int


class QueryResponse(BaseModel):
    """Response from query."""
    answer: str
    sources: List[Source]
    num_chunks: int
    query: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    indexed_documents: int
    capabilities: List[str]


class StatsResponse(BaseModel):
    """System statistics."""
    is_indexed: bool
    num_vectors: int
    embedding_dimension: int
    cache_stats: Optional[dict] = None
