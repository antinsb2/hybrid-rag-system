"""
End-to-end embedding pipeline: documents -> chunks -> embeddings.
"""

from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.document_processing.loaders import DocumentLoader
from rag.document_processing.chunking import TextChunker
from rag.embeddings.embedding_model import EmbeddingModel
from rag.embeddings.cache import EmbeddingCache


@dataclass
class EmbeddedChunk:
    """Chunk with its embedding."""
    text: str
    embedding: np.ndarray
    metadata: dict


class EmbeddingPipeline:
    """
    Complete pipeline: load documents -> chunk -> embed.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_cache: bool = True,
        cache_dir: str = ".cache/embeddings"
    ):
        """
        Args:
            model_name: Embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_cache: Whether to use embedding cache
            cache_dir: Cache directory
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.model = EmbeddingModel(model_name)
        
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)
        else:
            self.cache = None
    
    def process_document(self, file_path: str) -> List[EmbeddedChunk]:
        """
        Process a single document.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of embedded chunks
        """
        # Load document
        doc = self.loader.load(file_path)
        
        # Chunk text
        chunks = self.chunker.chunk_by_tokens(doc.content)
        
        # Embed chunks
        embedded_chunks = []
        
        for chunk in chunks:
            # Check cache
            if self.use_cache:
                cached_emb = self.cache.get(chunk)
                if cached_emb is not None:
                    embedding = cached_emb
                else:
                    embedding = self.model.encode(chunk)[0]
                    self.cache.put(chunk, embedding)
            else:
                embedding = self.model.encode(chunk)[0]
            
            embedded_chunks.append(EmbeddedChunk(
                text=chunk,
                embedding=embedding,
                metadata={**doc.metadata, "chunk_index": len(embedded_chunks)}
            ))
        
        return embedded_chunks
    
    def process_documents(
        self, 
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[EmbeddedChunk]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of document paths
            show_progress: Show progress bar
            
        Returns:
            List of all embedded chunks
        """
        all_chunks = []
        
        iterator = tqdm(file_paths) if show_progress else file_paths
        
        for file_path in iterator:
            if show_progress:
                iterator.set_description(f"Processing {Path(file_path).name}")
            
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return all_chunks
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "model": self.model.get_info(),
            "chunker": {
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.chunk_overlap
            }
        }
        
        if self.use_cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def save_cache(self):
        """Save cache to disk."""
        if self.use_cache:
            self.cache.save_cache()
