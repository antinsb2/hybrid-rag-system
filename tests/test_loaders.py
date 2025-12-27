"""
Tests for document loaders.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.document_processing.loaders import DocumentLoader, TextLoader


def test_text_loader():
    """Test text file loading."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nWith multiple lines.")
        temp_path = f.name
    
    try:
        loader = TextLoader()
        doc = loader.load(temp_path)
        
        assert doc.content == "This is a test document.\nWith multiple lines."
        assert doc.metadata['type'] == 'text'
        assert doc.source == temp_path
        
        print("✅ Text loader test passed")
    finally:
        Path(temp_path).unlink()


def test_document_loader_routing():
    """Test that DocumentLoader routes correctly."""
    loader = DocumentLoader()
    
    assert '.pdf' in loader.loaders
    assert '.docx' in loader.loaders
    assert '.html' in loader.loaders
    assert '.txt' in loader.loaders
    
    print("✅ Loader routing test passed")


if __name__ == "__main__":
    test_text_loader()
    test_document_loader_routing()
    print("\n✅ All loader tests passed!")
