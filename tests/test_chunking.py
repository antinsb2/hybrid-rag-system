"""
Tests for chunking strategies.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.document_processing.chunking import TextChunker


def test_token_chunking():
    """Test token-based chunking."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    text = "This is a test sentence. " * 100
    chunks = chunker.chunk_by_tokens(text)
    
    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)
    
    print(f"✅ Token chunking: {len(chunks)} chunks created")


def test_sentence_chunking():
    """Test sentence-based chunking."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    text = "First sentence. Second sentence. Third sentence. " * 20
    chunks = chunker.chunk_by_sentences(text)
    
    assert len(chunks) > 0
    assert all('.' in chunk for chunk in chunks)
    
    print(f"✅ Sentence chunking: {len(chunks)} chunks created")


if __name__ == "__main__":
    test_token_chunking()
    test_sentence_chunking()
    print("\n✅ All chunking tests passed!")
