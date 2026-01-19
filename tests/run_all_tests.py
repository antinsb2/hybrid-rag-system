"""
Run all tests and generate summary.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all test files and report results."""
    
    test_dir = Path(__file__).parent
    test_files = [
        "test_loaders.py",
        "test_chunking.py",
        "test_embeddings.py",
        "test_indexing.py",
        "test_bm25.py",
        "test_retrieval.py",
        "test_fusion.py",
        "test_filters.py",
        "test_reranker.py",
        "test_generation.py",
        "test_api.py",
        "test_error_handling.py",
        "test_integration.py"
    ]
    
    print("="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    results = {}
    
    for test_file in test_files:
        test_path = test_dir / test_file
        
        if not test_path.exists():
            print(f"\n⚠️  {test_file}: NOT FOUND")
            results[test_file] = "SKIP"
            continue
        
        print(f"\n📝 Running {test_file}...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=test_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {test_file}: PASSED")
                results[test_file] = "PASS"
            else:
                print(f"❌ {test_file}: FAILED")
                print(result.stderr[:200])
                results[test_file] = "FAIL"
        
        except subprocess.TimeoutExpired:
            print(f"⏱️  {test_file}: TIMEOUT")
            results[test_file] = "TIMEOUT"
        
        except Exception as e:
            print(f"💥 {test_file}: ERROR - {str(e)}")
            results[test_file] = "ERROR"
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r == "PASS")
    failed = sum(1 for r in results.values() if r == "FAIL")
    skipped = sum(1 for r in results.values() if r == "SKIP")
    errors = sum(1 for r in results.values() if r == "ERROR")
    
    total = len(results)
    
    print(f"\nTotal: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"💥 Errors: {errors}")
    
    if failed == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
