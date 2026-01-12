"""
Complete RAG generation: retrieve + generate answer.
"""

from typing import List, Optional, Union

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.generation.llm_client import LLMClient
from rag.generation.mock_llm import MockLLM
from rag.retrieval.types import RetrievalResult


class RAGGenerator:
    """
    Complete RAG: retrieval + answer generation.
    """
    
    def __init__(
        self,
        retriever,
        llm: Union[LLMClient, MockLLM],
        include_sources: bool = True
    ):
        """
        Args:
            retriever: Any retriever (dense, sparse, hybrid, or with reranking)
            llm: LLM client or mock
            include_sources: Whether to include source citations
        """
        self.retriever = retriever
        self.llm = llm
        self.include_sources = include_sources
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 500,
        min_score: float = 0.3
    ) -> dict:
        """
        Generate answer for query.
        
        Args:
            query: User question
            top_k: Number of context chunks to use
            max_tokens: Maximum answer length
            min_score: Minimum relevance score for chunks
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Filter by score
        results = [r for r in results if r.score >= min_score]
        
        if not results:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "num_chunks": 0
            }
        
        # Extract context
        context_chunks = [r.text for r in results]
        
        # Generate answer
        answer = self.llm.generate_with_context(
            query=query,
            context_chunks=context_chunks,
            max_tokens=max_tokens
        )
        
        # Prepare sources
        sources = []
        if self.include_sources:
            sources = [
                {
                    "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                    "score": r.score,
                    "source": r.metadata.get("source", "unknown"),
                    "rank": r.rank
                }
                for r in results
            ]
        
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(results),
            "query": query
        }
    
    def generate_with_citations(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 500
    ) -> dict:
        """
        Generate answer with inline citations.
        
        Args:
            query: User question
            top_k: Number of context chunks
            max_tokens: Maximum answer length
            
        Returns:
            Dict with answer and sources
        """
        # Retrieve
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }
        
        # Build context with citation markers
        context_with_citations = "\n\n".join([
            f"[Source {i+1}] {r.text}"
            for i, r in enumerate(results)
        ])
        
        prompt = f"""Answer the question using the provided sources. Include citation numbers [1], [2], etc. when referencing information.

Sources:
{context_with_citations}

Question: {query}

Answer with citations:"""
        
        answer = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        
        sources = [
            {
                "id": i + 1,
                "text": r.text,
                "source": r.metadata.get("source", "unknown"),
                "score": r.score
            }
            for i, r in enumerate(results)
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(results)
        }
