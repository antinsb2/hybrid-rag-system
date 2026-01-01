"""
Data types for retrieval.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    text: str
    score: float
    metadata: dict
    rank: int
