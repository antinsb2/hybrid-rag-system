"""
Complete RAG system demonstration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def main():
    """Demonstrate complete RAG pipeline."""
    
    # Create knowledge base
    docs_dir = Path(tempfile.mkdtemp())
    
    doc1 = docs_dir / "flask_basics.txt"
    doc1.write_text("""
    Flask is a micro web framework written in Python. It is designed to make
    getting started quick and easy, with the ability to scale up to complex
    applications. Flask offers suggestions, but doesn't enforce any dependencies
    or project layout.
    
    To get started with Flask:
    1. Install with: pip install flask
    2. Create app: app = Flask(__name__)
    3. Define routes: @app.route('/')
    4. Run server: app.run()
    """)
    
    doc2 = docs_dir / "flask_routing.txt"
    doc2.write_text("""
    Flask routing maps URLs to Python functions. Use the @app.route() decorator
    to bind functions to URLs. You can capture URL parameters using <variable_name>
    syntax. Flask supports GET, POST, PUT, DELETE methods. Dynamic routes allow
    flexible URL patterns.
    """)
    
    doc3 = docs_dir / "flask_templates.txt"
    doc3.write_text("""
    Flask uses Jinja2 templating engine. Templates are HTML files with special
    syntax for variables and logic. Use render_template() to render templates.
    Templates support inheritance, macros, and filters. Store templates in a
    templates/ directory by default.
    """)
    
    print("="*70)
    print("COMPLETE RAG SYSTEM DEMO")
    print("="*70)
    
    # Create pipeline
    print("\n1. Creating pipeline...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    # Ingest documents
    print("2. Ingesting documents...")
    documents = [str(doc1), str(doc2), str(doc3)]
    pipeline.ingest_documents(documents, show_progress=False)
    
    # Enable hybrid retrieval
    print("3. Enabling hybrid retrieval...")
    pipeline.enable_hybrid(fusion_method="rrf")
    
    # Enable generation
    print("4. Enabling answer generation...")
    pipeline.enable_generation(use_mock=True)
    
    # Ask questions
    questions = [
        "How do I get started with Flask?",
        "How does Flask routing work?",
        "What templating engine does Flask use?"
    ]
    
    print("\n" + "="*70)
    print("ASKING QUESTIONS")
    print("="*70)
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print('='*70)
        
        result = pipeline.ask(question, top_k=3)
        
        print(f"\nAnswer:")
        print(result["answer"])
        
        print(f"\nSources used ({result['num_chunks']} chunks):")
        for i, source in enumerate(result["sources"], 1):
            print(f"  [{i}] Score: {source['score']:.3f}")
            print(f"      {source['text'][:80]}...")
    
    # Test with citations
    print("\n" + "="*70)
    print("ANSWER WITH CITATIONS")
    print("="*70)
    
    result = pipeline.ask_with_citations(
        "What are the main features of Flask?",
        top_k=3
    )
    
    print(f"\nAnswer:")
    print(result["answer"])
    
    print(f"\nSources:")
    for source in result["sources"]:
        print(f"  [{source['id']}] {source['source']}: {source['text'][:60]}...")
    
    print("\n" + "="*70)
    print("✅ COMPLETE RAG SYSTEM WORKING!")
    print("="*70)


if __name__ == "__main__":
    main()
