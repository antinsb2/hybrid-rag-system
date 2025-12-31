"""
End-to-end RAG pipeline.
"""

from typing import List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag.document_processing.loaders import DocumentLoader
from rag.document_processing.chunking import TextChunker
from rag.embeddings import EmbeddingPipeline
from rag.indexing import HNSWIndex
from rag.retrieval import DenseRetriever, RetrievalResult


class RAGPipeline:
    """
    Complete RAG pipeline: ingest documents -> retrieve relevant chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_cache: bool = True,
        use_hnsw: bool = True
    ):
        """
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_cache: Whether to cache embeddings
            use_hnsw: Use HNSW index (faster) vs linear
        """
        self.embedding_pipeline = EmbeddingPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_cache=use_cache
        )
        
        self.use_hnsw = use_hnsw
        self.index = None
        self.retriever = None
        self.is_indexed = False
    
    def ingest_documents(self, file_paths: List[str], show_progress: bool = True):
        """
        Ingest documents into the system.
        
        Args:
            file_paths: List of document paths
            show_progress: Show progress bar
        """
        print(f"Ingesting {len(file_paths)} documents...")
        
        # Process documents
        embedded_chunks = self.embedding_pipeline.process_documents(
            file_paths,
            show_progress=show_progress
        )
        
        print(f"Processed {len(embedded_chunks)} chunks")
        
        # Build index
        embeddings = [chunk.embedding for chunk in embedded_chunks]
        texts = [chunk.text for chunk in embedded_chunks]
        metadata = [chunk.metadata for chunk in embedded_chunks]
        
        dimension = embedded_chunks[0].embedding.shape[0]
        
        if self.use_hnsw:
            self.index = HNSWIndex(
                dimension=dimension,
                max_elements=len(embeddings) * 2  # Room for growth
            )
        else:
            from rag.indexing import VectorStore
            self.index = VectorStore()
        
        self.index.add(embeddings, texts, metadata)
        
        # Create retriever
        self.retriever = DenseRetriever(self.index)
        self.is_indexed = True
        
        print(f"✅ Indexing complete: {self.index.get_stats()}")
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Query the system.
        
        Args:
            query: User query
            top_k: Number of results
            min_score: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        if not self.is_indexed:
            raise RuntimeError("No documents indexed. Call ingest_documents() first.")
        
        return self.retriever.retrieve(query, top_k=top_k, min_score=min_score)
    
    def save(self, directory: str):
        """Save pipeline state."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.use_hnsw:
            self.index.save(str(path / "index"))
        else:
            self.index.save(str(path / "index.pkl"))
        
        # Save cache
        self.embedding_pipeline.save_cache()
        
        print(f"✅ Pipeline saved to {directory}")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "embedding_pipeline": self.embedding_pipeline.get_stats(),
            "is_indexed": self.is_indexed
        }
        
        if self.index:
            stats["index"] = self.index.get_stats()
        
        return stats
