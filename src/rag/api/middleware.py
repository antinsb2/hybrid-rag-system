"""
Middleware for API request/response logging.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rag_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all API requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration:.3f}s"
            )
            
            # Add custom header with duration
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"error={str(e)} duration={duration:.3f}s"
            )
            raise
