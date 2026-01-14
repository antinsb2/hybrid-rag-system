"""
Run the RAG API server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "rag.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
