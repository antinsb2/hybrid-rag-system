"""
Tests for error handling.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.api.error_handling import validate_query, handle_file_errors, retry_with_backoff
import pytest


def test_validate_query():
    """Test query validation."""
    
    # Valid query
    assert validate_query("  What is Python?  ") == "What is Python?"
    
    # Empty query
    with pytest.raises(ValueError, match="empty"):
        validate_query("")
    
    # Too short
    with pytest.raises(ValueError, match="too short"):
        validate_query("ab")
    
    # Too long
    with pytest.raises(ValueError, match="too long"):
        validate_query("x" * 1001)
    
    print("✅ Query validation tests passed")


def test_file_validation():
    """Test file validation."""
    import tempfile
    
    # Valid file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    try:
        handle_file_errors(temp_path)  # Should not raise
        print("✅ Valid file check passed")
    finally:
        Path(temp_path).unlink()
    
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        handle_file_errors("/nonexistent/file.txt")
    
    print("✅ File validation tests passed")


def test_retry_decorator():
    """Test retry with backoff."""
    
    attempt_count = 0
    
    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < 3:
            raise RetryableError("Temporary failure")
        
        return "success"
    
    from rag.api.error_handling import RetryableError
    
    result = flaky_function()
    
    assert result == "success"
    assert attempt_count == 3  # Failed twice, succeeded on third
    
    print(f"✅ Retry test: succeeded after {attempt_count} attempts")


if __name__ == "__main__":
    test_validate_query()
    test_file_validation()
    test_retry_decorator()
    print("\n✅ All error handling tests passed!")
