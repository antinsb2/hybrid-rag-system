"""
Tests for fusion algorithms.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.retrieval import RetrievalResult, ReciprocalRankFusion, WeightedFusion, SimpleCombination


def create_test_results():
    """Create sample results for testing."""
    dense = [
        RetrievalResult("Doc A about ML", 0.9, {"id": "a"}, 1),
        RetrievalResult("Doc B about AI", 0.8, {"id": "b"}, 2),
        RetrievalResult("Doc C about Python", 0.7, {"id": "c"}, 3),
    ]
    
    sparse = [
        RetrievalResult("Doc C about Python", 15.2, {"id": "c"}, 1),
        RetrievalResult("Doc D about ML", 12.5, {"id": "d"}, 2),
        RetrievalResult("Doc A about ML", 10.1, {"id": "a"}, 3),
    ]
    
    return dense, sparse


def test_rrf():
    """Test Reciprocal Rank Fusion."""
    dense, sparse = create_test_results()
    
    fused = ReciprocalRankFusion.fuse(dense, sparse, k=60)
    
    assert len(fused) == 4  # All unique docs
    
    # Doc A and C should rank high (appear in both)
    top_ids = [r.metadata["id"] for r in fused[:2]]
    assert "a" in top_ids or "c" in top_ids
    
    print(f"✅ RRF test")
    for r in fused:
        print(f"  Rank {r.rank}: Doc {r.metadata['id']} (score: {r.score:.4f})")


def test_weighted_fusion():
    """Test weighted fusion."""
    dense, sparse = create_test_results()
    
    # Equal weight
    fused = WeightedFusion.fuse(dense, sparse, alpha=0.5)
    
    assert len(fused) == 4
    
    print(f"\n✅ Weighted fusion (α=0.5)")
    for r in fused:
        print(f"  Rank {r.rank}: Doc {r.metadata['id']} (score: {r.score:.4f})")
    
    # Dense-heavy
    fused_dense = WeightedFusion.fuse(dense, sparse, alpha=0.8)
    
    print(f"\n✅ Weighted fusion (α=0.8, dense-heavy)")
    for r in fused_dense:
        print(f"  Rank {r.rank}: Doc {r.metadata['id']} (score: {r.score:.4f})")


def test_simple_combination():
    """Test simple combination."""
    dense, sparse = create_test_results()
    
    fused = SimpleCombination.fuse(dense, sparse)
    
    assert len(fused) == 4
    
    print(f"\n✅ Simple combination")
    for r in fused:
        print(f"  Rank {r.rank}: Doc {r.metadata['id']}")


if __name__ == "__main__":
    test_rrf()
    test_weighted_fusion()
    test_simple_combination()
    print("\n✅ All fusion tests passed!")
