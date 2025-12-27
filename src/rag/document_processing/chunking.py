"""
Text chunking strategies for RAG.
"""

from typing import List
import tiktoken


class TextChunker:
    """Chunk text into smaller pieces for processing."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """
        Chunk text by token count with overlap.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            start = end - self.chunk_overlap
            
            if end >= len(tokens):
                break
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk by sentences, respecting token limits.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
