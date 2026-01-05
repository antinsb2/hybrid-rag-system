"""
BM25 sparse retrieval implementation.
"""

from typing import List, Dict
import math
from collections import Counter
import pickle
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval.types import RetrievalResult


class BM25:
    """
    BM25 (Best Matching 25) algorithm for sparse retrieval.
    
    Uses term frequency and document frequency for keyword matching.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        
        # Document storage
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        
        # Inverted index: term -> [(doc_id, term_freq), ...]
        self.inverted_index: Dict[str, List[tuple]] = {}
        
        # Document frequency: term -> num_docs_containing_term
        self.doc_freq: Dict[str, int] = {}
        
        self.num_docs = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (lowercase + split).
        """
        # Remove punctuation and lowercase
        text = text.lower()
        tokens = []
        
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def add_documents(self, texts: List[str], metadata: List[dict] = None):
        """
        Add documents to the index.
        
        Args:
            texts: List of document texts
            metadata: List of metadata dicts
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        for doc_id, (text, meta) in enumerate(zip(texts, metadata), start=self.num_docs):
            # Tokenize
            tokens = self._tokenize(text)
            
            # Store document
            self.documents.append({
                'text': text,
                'metadata': meta,
                'tokens': tokens,
                'doc_id': doc_id
            })
            
            # Document length
            self.doc_lengths.append(len(tokens))
            
            # Update inverted index
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    self.doc_freq[term] = 0
                
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freq[term] += 1
        
        # Update statistics
        self.num_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
    
    def _bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Compute BM25 score for a document.
        
        BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        
        where:
        - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        - f(qi, D) = frequency of term qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - N = total number of documents
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        # Get term frequencies for this document
        doc_tokens = self.documents[doc_id]['tokens']
        term_freqs = Counter(doc_tokens)
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            # Term frequency in document
            tf = term_freqs.get(term, 0)
            
            # Document frequency
            df = self.doc_freq[term]
            
            # IDF calculation
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum BM25 score
            
        Returns:
            List of retrieval results
        """
        if self.num_docs == 0:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Get candidate documents (documents containing any query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, _ in self.inverted_index[term]:
                    candidate_docs.add(doc_id)
        
        if not candidate_docs:
            return []
        
        # Score candidates
        scores = []
        for doc_id in candidate_docs:
            score = self._bm25_score(query_terms, doc_id)
            if score >= min_score:
                scores.append((doc_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            doc = self.documents[doc_id]
            results.append(RetrievalResult(
                text=doc['text'],
                score=score,
                metadata=doc['metadata'],
                rank=rank
            ))
        
        return results
    
    def save(self, filepath: str):
        """Save BM25 index."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'inverted_index': self.inverted_index,
            'doc_freq': self.doc_freq,
            'num_docs': self.num_docs
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load BM25 index."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.k1 = data['k1']
        self.b = data['b']
        self.documents = data['documents']
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_length = data['avg_doc_length']
        self.inverted_index = data['inverted_index']
        self.doc_freq = data['doc_freq']
        self.num_docs = data['num_docs']
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'num_docs': self.num_docs,
            'num_terms': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length,
            'k1': self.k1,
            'b': self.b
        }
