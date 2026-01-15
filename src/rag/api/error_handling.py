"""
Error handling and retry logic for API.
"""

from functools import wraps
import time
import logging
from typing import Callable, Any

logger = logging.getLogger("rag_api")


class RetryableError(Exception):
    """Error that can be retried."""
    pass


class NonRetryableError(Exception):
    """Error that should not be retried."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except NonRetryableError:
                    # Don't retry these
                    raise
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} retries failed: {str(e)}")
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for failing components.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception if circuit is open
        """
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is open - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in half_open
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            
            return result
        
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


def validate_query(query: str) -> str:
    """
    Validate and sanitize user query.
    
    Args:
        query: User query string
        
    Returns:
        Sanitized query
        
    Raises:
        ValueError if query is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    # Check length
    if len(query) > 1000:
        raise ValueError("Query too long (max 1000 characters)")
    
    if len(query) < 3:
        raise ValueError("Query too short (min 3 characters)")
    
    return query


def handle_file_errors(file_path: str) -> None:
    """
    Validate file exists and is readable.
    
    Args:
        file_path: Path to file
        
    Raises:
        FileNotFoundError or PermissionError
    """
    from pathlib import Path
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    
    # Check if readable
    try:
        with open(path, 'rb') as f:
            f.read(1)
    except PermissionError:
        raise PermissionError(f"Cannot read file: {file_path}")
