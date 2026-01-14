"""
FastAPI server for RAG system.
"""


from rag.api.logging_config import logger
from rag.api.middleware import RequestLoggingMiddleware
from rag.api.metrics import metrics
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.pipeline import RAGPipeline
from rag.api.models import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    Source,
    HealthResponse,
    StatsResponse
)

# Global pipeline instance
pipeline = None

# Create FastAPI app
app = FastAPI(
    title="Hybrid RAG System API",
    description="Production RAG with hybrid retrieval and re-ranking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting RAG API server...")
    pipeline = RAGPipeline(chunk_size=512, use_hnsw=False)
    print("✅ RAG Pipeline initialized")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    capabilities = ["document_ingestion", "hybrid_retrieval"]
    
    if pipeline and pipeline.is_indexed:
        capabilities.append("query")
    
    if pipeline and hasattr(pipeline, 'hybrid_retriever'):
        capabilities.append("hybrid_search")
    
    if pipeline and hasattr(pipeline, 'generator'):
        capabilities.append("answer_generation")
    
    return HealthResponse(
        status="healthy",
        indexed_documents=pipeline.index.get_stats()["num_vectors"] if pipeline and pipeline.is_indexed else 0,
        capabilities=capabilities
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    stats = pipeline.get_stats()
    
    return StatsResponse(
        is_indexed=pipeline.is_indexed,
        num_vectors=stats.get("index", {}).get("num_vectors", 0),
        embedding_dimension=stats.get("embedding_pipeline", {}).get("model", {}).get("dimension", 0),
        cache_stats=stats.get("embedding_pipeline", {}).get("cache")
    )

@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return metrics.get_summary()


@app.post("/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents into the system.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Validate files exist
    for file_path in request.file_paths:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Ingest in background
    def ingest_task():
        pipeline.ingest_documents(request.file_paths, show_progress=False)
        pipeline.enable_hybrid(fusion_method="rrf")
        pipeline.enable_generation(use_mock=True)
    
    background_tasks.add_task(ingest_task)
    
    return {
        "message": f"Ingesting {len(request.file_paths)} documents",
        "status": "processing"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    """

    metrics.record_request("/query")
    start_time = time.time()

    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.is_indexed:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    if not hasattr(pipeline, 'generator'):
        raise HTTPException(status_code=400, detail="Generation not enabled")
    
    try:
        # Generate answer
        result = pipeline.ask(
            request.query,
            top_k=request.top_k
        )
        
        # Convert to response model
        sources = [
            Source(
                text=s["text"],
                score=s["score"],
                source=s["source"],
                rank=s["rank"]
            )
            for s in result["sources"]
        ]
        
        metrics.record_query(time.time() - start_time)
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            num_chunks=result["num_chunks"],
            query=request.query
        )
    
    except Exception as e:
        metrics.record_error(type(e).__name__)
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/retrieve-only")
async def retrieve_only(request: QueryRequest):
    """
    Retrieve relevant chunks without generating answer.
    """
    if not pipeline or not pipeline.is_indexed:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    try:
        # Use hybrid if available
        if hasattr(pipeline, 'hybrid_retriever'):
            results = pipeline.query_hybrid(request.query, top_k=request.top_k)
        else:
            results = pipeline.query(request.query, top_k=request.top_k)
        
        return {
            "query": request.query,
            "results": [
                {
                    "text": r.text,
                    "score": r.score,
                    "rank": r.rank,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "num_results": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Hybrid RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "ingest": "POST /ingest",
            "query": "POST /query",
            "retrieve_only": "POST /query/retrieve-only"
        }
    }
