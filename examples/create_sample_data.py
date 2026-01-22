"""
Create sample dataset for testing the RAG system.
"""

from pathlib import Path
import tempfile


def create_sample_documents(output_dir: str = None):
    """
    Create sample documents for RAG testing.
    
    Args:
        output_dir: Directory to save documents (creates temp if None)
        
    Returns:
        Path to directory with sample documents
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    docs_dir = Path(output_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Python guide
    (docs_dir / "python_guide.txt").write_text("""
Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity
and readability. Created by Guido van Rossum in 1991, Python emphasizes code
readability with significant whitespace.

Key Features:
- Dynamic typing and automatic memory management
- Comprehensive standard library
- Support for multiple programming paradigms
- Extensive third-party packages via PyPI

Python is widely used in:
- Web development (Django, Flask, FastAPI)
- Data science and machine learning (NumPy, Pandas, scikit-learn)
- Automation and scripting
- Scientific computing

Getting Started:
1. Install Python from python.org
2. Learn basic syntax: variables, functions, classes
3. Explore the standard library
4. Install packages with pip
5. Build projects to practice
    """)
    
    # Machine Learning intro
    (docs_dir / "ml_intro.txt").write_text("""
Machine Learning Introduction

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without explicit programming.

Types of Machine Learning:
1. Supervised Learning - Learn from labeled data
   - Classification: Categorize data points
   - Regression: Predict continuous values
   
2. Unsupervised Learning - Find patterns in unlabeled data
   - Clustering: Group similar items
   - Dimensionality reduction: Simplify data
   
3. Reinforcement Learning - Learn through trial and error
   - Agent interacts with environment
   - Receives rewards or penalties

Common Algorithms:
- Decision trees and random forests
- Support vector machines
- Neural networks and deep learning
- K-means clustering
- Linear and logistic regression

Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles
    """)
    
    # Flask tutorial
    (docs_dir / "flask_tutorial.txt").write_text("""
Flask Web Framework Tutorial

Flask is a micro web framework for Python. It's lightweight, flexible, and
easy to get started with while being powerful enough for complex applications.

Installation:
pip install flask

Basic Application:
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

Core Concepts:
1. Routing - Map URLs to Python functions
2. Templates - Use Jinja2 for dynamic HTML
3. Request handling - Access form data, query parameters
4. Response - Return HTML, JSON, or redirects

Advanced Features:
- Blueprints for modular applications
- SQLAlchemy integration for databases
- Flask-Login for user authentication
- RESTful API development
- Error handling and custom error pages

Best Practices:
- Use virtual environments
- Keep secret keys in environment variables
- Use blueprints for larger apps
- Implement proper error handling
- Add logging for debugging
    """)
    
    # Data Science guide
    (docs_dir / "data_science.txt").write_text("""
Data Science with Python

Data science combines statistics, programming, and domain knowledge to extract
insights from data. Python is the leading language for data science due to its
powerful libraries and ease of use.

Essential Libraries:
1. NumPy - Numerical computing and arrays
2. Pandas - Data manipulation and analysis
3. Matplotlib/Seaborn - Data visualization
4. Scikit-learn - Machine learning algorithms
5. Jupyter - Interactive notebooks

Data Science Workflow:
1. Data Collection - Gather from databases, APIs, files
2. Data Cleaning - Handle missing values, outliers
3. Exploratory Analysis - Understand patterns, distributions
4. Feature Engineering - Create meaningful features
5. Modeling - Apply ML algorithms
6. Evaluation - Assess model performance
7. Deployment - Put models into production

Common Tasks:
- Predictive modeling
- Customer segmentation
- Time series forecasting
- A/B testing analysis
- Anomaly detection

Tools and Platforms:
- Jupyter notebooks for exploration
- Git for version control
- Docker for reproducibility
- Cloud platforms (AWS, GCP, Azure)
    """)
    
    # NLP basics
    (docs_dir / "nlp_basics.txt").write_text("""
Natural Language Processing Basics

Natural Language Processing (NLP) enables computers to understand, interpret,
and generate human language. It's a crucial technology for chatbots, translation,
sentiment analysis, and information extraction.

Key Concepts:
1. Tokenization - Split text into words or subwords
2. Embeddings - Represent words as vectors
3. Part-of-speech tagging - Identify grammatical roles
4. Named entity recognition - Find names, places, organizations
5. Sentiment analysis - Determine emotional tone

Common Tasks:
- Text classification (spam detection, topic categorization)
- Machine translation (English to French)
- Question answering systems
- Text summarization
- Information extraction

Modern NLP:
- Transformers architecture (BERT, GPT)
- Attention mechanisms
- Transfer learning with pre-trained models
- Few-shot and zero-shot learning

Python Libraries:
- NLTK - Natural Language Toolkit
- spaCy - Industrial-strength NLP
- Transformers (Hugging Face) - Pre-trained models
- Gensim - Topic modeling
    """)
    
    print(f"✅ Created 5 sample documents in {docs_dir}")
    print("\nDocuments:")
    for doc in docs_dir.glob("*.txt"):
        size = doc.stat().st_size
        print(f"  - {doc.name} ({size} bytes)")
    
    return str(docs_dir)


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/samples"
    path = create_sample_documents(output_dir)
    
    print(f"\n📁 Sample documents ready at: {path}")
    print("\nUse with:")
    print(f"  pipeline.ingest_documents(['{path}/python_guide.txt', ...])")
