"""
Tests for RAG generator.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.generation import MockLLM, RAGGenerator
from rag.retrieval import DenseRetriever, RetrievalResult
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorStore


def test_rag_generator():
    """Test complete RAG generation."""
    
    # Setup retriever
    docs = [
        "Flask is a lightweight Python web framework",
        "Django is a full-featured Python framework",
        "FastAPI is a modern Python API framework"
    ]
    
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    metadata = [{"id": i} for i in range(len(docs))]
    
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    retriever = DenseRetriever(store)
    
    # Create RAG generator with mock
    llm = MockLLM()
    rag = RAGGenerator(retriever, llm)
    
    # Generate answer
    result = rag.generate_answer("What is Flask?", top_k=2)
    
    assert "answer" in result
    assert "sources" in result
    assert result["num_chunks"] > 0
    
    print(f"✅ RAG generator test")
    print(f"Query: What is Flask?")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Sources used: {result['num_chunks']}")


def test_citations():
    """Test answer with citations."""
    
    docs = ["Python programming", "Web development", "Data science"]
    
    model = EmbeddingModel()
    embeddings = model.encode(docs)
    metadata = [{"source": f"doc_{i}.txt"} for i in range(len(docs))]
    
    store = VectorStore()
    store.add(embeddings, docs, metadata)
    retriever = DenseRetriever(store)
    
    llm = MockLLM()
    rag = RAGGenerator(retriever, llm)
    
    result = rag.generate_with_citations("Tell me about Python", top_k=2)
    
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    
    print(f"\n✅ Citations test")
    print(f"Answer: {result['answer'][:80]}...")
    print(f"Number of sources: {len(result['sources'])}")


if __name__ == "__main__":
    test_rag_generator()
    test_citations()
    print("\n✅ All RAG generator tests passed!")
