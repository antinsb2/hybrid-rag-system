"""
Quick 2-minute demo of the RAG system.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.pipeline import RAGPipeline
import tempfile


def quick_demo():
    """
    Demonstrate RAG in under 2 minutes.
    """
    
    print("="*60)
    print("HYBRID RAG SYSTEM - QUICK DEMO")
    print("="*60)
    
    # Create a sample document
    docs_dir = Path(tempfile.mkdtemp())
    doc = docs_dir / "sample.txt"
    doc.write_text("""
    Flask is a lightweight web framework for Python. It's designed to be
    simple and easy to use, making it perfect for small applications and
    APIs. Flask doesn't enforce any dependencies or project structure,
    giving developers flexibility.
    
    To get started with Flask, install it using pip install flask, create
    a basic app with app = Flask(__name__), define routes with decorators,
    and run with app.run(). Flask uses Jinja2 for templating.
    """)
    
    print("\n1️⃣  Creating RAG pipeline...")
    pipeline = RAGPipeline(chunk_size=256, use_hnsw=False)
    
    print("2️⃣  Indexing document...")
    pipeline.ingest_documents([str(doc)], show_progress=False)
    
    print("3️⃣  Enabling hybrid retrieval...")
    pipeline.enable_hybrid(fusion_method="rrf")
    
    print("4️⃣  Enabling answer generation...")
    pipeline.enable_generation(use_mock=True)
    
    print("\n" + "="*60)
    print("ASKING QUESTIONS")
    print("="*60)
    
    questions = [
        "What is Flask?",
        "How do I get started with Flask?",
        "What templating engine does Flask use?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-"*60)
        
        result = pipeline.ask(question, top_k=2)
        
        print(f"💬 Answer: {result['answer'][:200]}...")
        print(f"📚 Used {result['num_chunks']} source chunks")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\nThis demonstrated:")
    print("  • Document ingestion and chunking")
    print("  • Hybrid retrieval (dense + sparse)")
    print("  • Answer generation with context")
    print("  • Source attribution")
    print("\nFull system ready for production use!")


if __name__ == "__main__":
    quick_demo()
