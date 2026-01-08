"""
Demonstrate re-ranking improvements.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def main():
    """Show how re-ranking improves results."""
    
    # Create test documents
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "flask_intro.txt"
    doc1.write_text("""
    Flask is a lightweight Python web framework. Getting started with Flask:
    1. Install Flask using pip install flask
    2. Create a basic app with app = Flask(__name__)
    3. Define routes using @app.route decorators
    4. Run with app.run()
    """)
    
    doc2 = docs_dir / "django_guide.txt"
    doc2.write_text("""
    Django is a full-featured Python web framework for rapid development.
    It includes an ORM, admin interface, and built-in security features.
    Django follows the MTV (Model-Template-View) pattern.
    """)
    
    doc3 = docs_dir / "python_basics.txt"
    doc3.write_text("""
    Python is a high-level programming language. Basic concepts include:
    variables, functions, classes, and modules. Python uses indentation
    for code blocks and has dynamic typing.
    """)
    
    doc4 = docs_dir / "flask_advanced.txt"
    doc4.write_text("""
    Advanced Flask topics: blueprints for modular applications, SQLAlchemy
    for database integration, Flask-Login for authentication, and deploying
    Flask apps with Gunicorn and Nginx.
    """)
    
    doc5 = docs_dir / "web_concepts.txt"
    doc5.write_text("""
    Web development concepts: HTTP methods (GET, POST, PUT, DELETE),
    RESTful APIs, JSON data format, authentication and authorization,
    and frontend-backend communication.
    """)
    
    # Create pipeline
    print("Creating pipeline with re-ranking...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    # Ingest
    documents = [str(doc1), str(doc2), str(doc3), str(doc4), str(doc5)]
    pipeline.ingest_documents(documents, show_progress=False)
    
    # Enable hybrid
    pipeline.enable_hybrid(fusion_method="rrf")
    
    # Enable re-ranking
    pipeline.enable_reranking()
    
    # Test query
    query = "How do I get started with Flask web framework?"
    
    print("\n" + "="*70)
    print(f"Query: '{query}'")
    print("="*70)
    
    # Without re-ranking
    print("\nWITHOUT RE-RANKING (Hybrid only):")
    hybrid_results = pipeline.query_hybrid(query, top_k=5)
    for r in hybrid_results:
        print(f"  {r.rank}. [{r.score:.4f}] {r.text[:70]}...")
    
    # With re-ranking
    print("\nWITH RE-RANKING:")
    reranked_results = pipeline.query_with_rerank(query, top_k=5, candidates_k=10)
    for r in reranked_results:
        orig_score = r.metadata.get('original_score', 0)
        print(f"  {r.rank}. [{r.score:.4f}] (was {orig_score:.4f}) {r.text[:70]}...")
    
    print("\n" + "="*70)
    print("OBSERVATION:")
    print("- Re-ranking uses query-document interaction")
    print("- More accurate than separate encoding")
    print("- Typically improves top-3 precision by 10-20%")
    print("- Trade-off: Higher latency (~50-100ms extra)")
    print("="*70)


if __name__ == "__main__":
    main()
