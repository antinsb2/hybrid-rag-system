"""
LLM client for answer generation.
"""

from typing import List, Optional
import os


class LLMClient:
    """
    Client for LLM API calls.
    Supports OpenAI and Anthropic.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None
    ):
        """
        Args:
            provider: "openai" or "anthropic"
            model: Model name (defaults to gpt-3.5-turbo or claude-3-sonnet)
            api_key: API key (or set via environment variable)
        """
        self.provider = provider.lower()
        
        if model is None:
            model = "gpt-3.5-turbo" if self.provider == "openai" else "claude-3-5-sonnet-20241022"
        
        self.model = model
        
        # Get API key
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not provided for {provider}")
        
        self.api_key = api_key
        
        # Initialize client
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        max_tokens: int = 500
    ) -> str:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User question
            context_chunks: Retrieved document chunks
            max_tokens: Maximum response tokens
            
        Returns:
            Generated answer
        """
        # Build prompt with context
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Answer the following question using only the provided context. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.3)
