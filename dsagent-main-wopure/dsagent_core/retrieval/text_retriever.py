"""
Text-based experience retrieval using BM25 algorithm.

This module provides BM25-based retrieval of textual experiences,
independent of any specific agent framework.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import math

from dsagent_core.retrieval.base import (
    ExperienceRetriever,
    ExperienceEntry,
    RetrievalResult
)


class BM25Retriever:
    """
    Simplified BM25 implementation for document retrieval.
    
    BM25 is a probabilistic retrieval function that ranks documents
    based on query term frequency and document length.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 parameters.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = Counter()
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.N = 0
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 model on a corpus of documents.
        
        Args:
            corpus: List of document strings
        """
        self.corpus = corpus
        self.N = len(corpus)
        
        # Tokenize and calculate document frequencies
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0
        
        # Calculate document frequencies
        for doc_tokens in tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Calculate IDF
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and lowercasing"""
        return text.lower().split()
    
    def _score_document(self, query_tokens: List[str], doc_index: int) -> float:
        """
        Calculate BM25 score for a document given query tokens.
        
        Args:
            query_tokens: List of query tokens
            doc_index: Index of document in corpus
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = self._tokenize(self.corpus[doc_index])
        doc_len = self.doc_len[doc_index]
        
        # Calculate term frequency for document
        term_freq = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = term_freq.get(token, 0)
            idf = self.idf[token]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[tuple[int, float]]:
        """
        Search for most relevant documents.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for i in range(self.N):
            score = self._score_document(query_tokens, i)
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class TextExperienceRetriever(ExperienceRetriever):
    """
    Text-based experience retrieval using BM25 algorithm.
    
    This retriever loads experiences from a JSON file and uses BM25
    to find the most relevant experiences for a given query.
    """
    
    def __init__(
        self,
        experience_path: Optional[Path] = None,
        top_k: int = 5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        min_score_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize text experience retriever.
        
        Args:
            experience_path: Path to JSON file containing experiences
            top_k: Number of top experiences to retrieve
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            min_score_threshold: Minimum BM25 score threshold
            **kwargs: Additional configuration
        """
        super().__init__(experience_path=experience_path, top_k=top_k, **kwargs)
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.experiences: List[ExperienceEntry] = []
        self.min_score_threshold = min_score_threshold
        
        if experience_path:
            self.load_experiences(experience_path)
    
    def load_experiences(self, path: Optional[Path] = None) -> int:
        """
        Load experiences from JSON file.
        
        Expected JSON format:
        [
            {
                "task": "description of task",
                "solution": "description of solution",
                "metadata": {...}  // optional
            },
            ...
        ]
        
        Args:
            path: Path to JSON file
            
        Returns:
            Number of experiences loaded
        """
        file_path = path or self.experience_path
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"Experience file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.experiences = []
        corpus = []
        
        for idx, item in enumerate(data):
            # Support different JSON formats
            if "task" in item and "solution" in item:
                content = f"{item['task']}\n{item['solution']}"
                metadata = item.get("metadata", {})
            elif "content" in item:
                content = item["content"]
                metadata = item.get("metadata", {})
            else:
                # Fallback: use entire item as content
                content = json.dumps(item, ensure_ascii=False)
                metadata = {}
            
            exp = ExperienceEntry(
                content=content,
                metadata=metadata,
                id=str(idx)
            )
            self.experiences.append(exp)
            corpus.append(content)
        
        # Fit BM25 model
        self.bm25.fit(corpus)
        self._is_initialized = True
        
        return len(self.experiences)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant experiences using BM25.
        
        Args:
            query: Query string
            top_k: Number of results (overrides default)
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult with matched experiences
        """
        if not self.is_initialized:
            raise RuntimeError("Retriever not initialized. Call load_experiences() first.")
        
        start_time = time.time()
        k = top_k or self.top_k
        
        # Get BM25 scores
        results = self.bm25.search(query, top_k=k * 2)  # Get more for filtering
        
        # Create result experiences
        matched_experiences = []
        for idx, score in results:
            if score < self.min_score_threshold:
                continue
            
            # Handle metadata copy (metadata can be dict, str, or None)
            original_metadata = self.experiences[idx].metadata
            if isinstance(original_metadata, dict):
                metadata_copy = original_metadata.copy()
            else:
                metadata_copy = original_metadata
            
            exp = ExperienceEntry(
                content=self.experiences[idx].content,
                metadata=metadata_copy,
                score=score,
                id=self.experiences[idx].id
            )
            
            # Apply filters if specified
            if filters:
                if not self._matches_filters(exp, filters):
                    continue
            
            matched_experiences.append(exp)
            
            if len(matched_experiences) >= k:
                break
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            experiences=matched_experiences,
            query=query,
            retrieval_time=retrieval_time,
            total_candidates=len(self.experiences)
        )
    
    def _matches_filters(self, exp: ExperienceEntry, filters: Dict[str, Any]) -> bool:
        """Check if experience matches all specified filters"""
        for key, value in filters.items():
            if key not in exp.metadata:
                return False
            if exp.metadata[key] != value:
                return False
        return True
    
    def add_experience(self, experience: ExperienceEntry) -> bool:
        """
        Add a new experience to the knowledge base.
        
        Note: This requires re-fitting the BM25 model.
        
        Args:
            experience: Experience to add
            
        Returns:
            True if successfully added
        """
        if not experience.id:
            experience.id = str(len(self.experiences))
        
        self.experiences.append(experience)
        
        # Re-fit BM25 model
        corpus = [exp.content for exp in self.experiences]
        self.bm25.fit(corpus)
        
        return True
    
    def save_experiences(self, path: Optional[Path] = None) -> bool:
        """
        Save experiences to JSON file.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successfully saved
        """
        file_path = path or self.experience_path
        if not file_path:
            raise ValueError("No path specified for saving")
        
        data = []
        for exp in self.experiences:
            # Parse content back to task/solution if possible
            if "\n" in exp.content:
                parts = exp.content.split("\n", 1)
                item = {
                    "task": parts[0],
                    "solution": parts[1] if len(parts) > 1 else "",
                    "metadata": exp.metadata
                }
            else:
                item = {
                    "content": exp.content,
                    "metadata": exp.metadata
                }
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience base"""
        return {
            "total_experiences": len(self.experiences),
            "is_initialized": self.is_initialized,
            "avg_content_length": sum(len(exp.content) for exp in self.experiences) / len(self.experiences) if self.experiences else 0,
            "bm25_params": {
                "k1": self.bm25.k1,
                "b": self.bm25.b
            }
        }
