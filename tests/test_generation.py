"""
Tests for answer generation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.generation import MockLLM


def test_mock_llm():
    """Test mock LLM."""
    llm = MockLLM()
    
    response = llm.generate("What is Python?")
    assert len(response) > 0
    
    print(f"✅ Mock LLM generate: {response[:80]}...")


def test_mock_with_context():
    """Test context-based generation."""
    llm = MockLLM()
    
    query = "What is Flask?"
    context = [
        "Flask is a Python web framework",
        "Flask is lightweight and easy to use"
    ]
    
    response = llm.generate_with_context(query, context)
    
    assert len(response) > 0
    assert "2" in response  # Should mention 2 chunks
    
    print(f"\n✅ Mock LLM with context:")
    print(response)


if __name__ == "__main__":
    test_mock_llm()
    test_mock_with_context()
    print("\n✅ All generation tests passed!")
