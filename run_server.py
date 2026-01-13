"""
Run the RAG API server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.rag.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
