"""
Metrics collection for API monitoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import time
from collections import defaultdict, deque


@dataclass
class Metrics:
    """
    Track API metrics.
    """
    
    # Request counts
    total_requests: int = 0
    requests_by_endpoint: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Query metrics
    total_queries: int = 0
    total_retrievals: int = 0
    
    # Latency tracking (keep last 1000)
    query_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    retrieval_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error tracking
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Start time
    start_time: float = field(default_factory=time.time)
    
    def record_request(self, endpoint: str):
        """Record an API request."""
        self.total_requests += 1
        self.requests_by_endpoint[endpoint] += 1
    
    def record_query(self, latency: float):
        """Record a query."""
        self.total_queries += 1
        self.query_latencies.append(latency)
    
    def record_retrieval(self, latency: float):
        """Record a retrieval operation."""
        self.total_retrievals += 1
        self.retrieval_latencies.append(latency)
    
    def record_error(self, error_type: str):
        """Record an error."""
        self.total_errors += 1
        self.errors_by_type[error_type] += 1
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        import numpy as np
        
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "requests_by_endpoint": dict(self.requests_by_endpoint),
            "total_queries": self.total_queries,
            "total_retrievals": self.total_retrievals,
            "total_errors": self.total_errors,
        }
        
        # Query latency stats
        if self.query_latencies:
            latencies = list(self.query_latencies)
            summary["query_latency"] = {
                "avg_ms": np.mean(latencies) * 1000,
                "p50_ms": np.percentile(latencies, 50) * 1000,
                "p95_ms": np.percentile(latencies, 95) * 1000,
                "p99_ms": np.percentile(latencies, 99) * 1000,
            }
        
        # Retrieval latency stats
        if self.retrieval_latencies:
            latencies = list(self.retrieval_latencies)
            summary["retrieval_latency"] = {
                "avg_ms": np.mean(latencies) * 1000,
                "p95_ms": np.percentile(latencies, 95) * 1000,
            }
        
        # Cache stats
        total_cache = self.cache_hits + self.cache_misses
        if total_cache > 0:
            summary["cache"] = {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / total_cache
            }
        
        return summary


# Global metrics instance
metrics = Metrics()
