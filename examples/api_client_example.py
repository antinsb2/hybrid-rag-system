"""
Example API client usage.
"""

import requests
import json


def main():
    """Demonstrate API usage."""
    
    base_url = "http://localhost:8000"
    
    print("="*70)
    print("RAG API CLIENT EXAMPLE")
    print("="*70)
    print("\nNOTE: Start server first with: python run_server.py")
    print("="*70)
    
    # Health check
    print("\n1. Health Check:")
    response = requests.get(f"{base_url}/health")
    print(json.dumps(response.json(), indent=2))
    
    # Stats
    print("\n2. System Stats:")
    response = requests.get(f"{base_url}/stats")
    print(json.dumps(response.json(), indent=2))
    
    # Query (will fail if no documents indexed)
    print("\n3. Query Example:")
    query_data = {
        "query": "What is Python?",
        "top_k": 3,
        "min_score": 0.3
    }
    
    try:
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"\nAnswer: {result['answer'][:200]}...")
            print(f"Sources used: {result['num_chunks']}")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Retrieve only
    print("\n4. Retrieve Only (no answer generation):")
    try:
        response = requests.post(f"{base_url}/query/retrieve-only", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"Found {result['num_results']} results")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
