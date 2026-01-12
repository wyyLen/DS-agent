"""
Base classes and interfaces for experience retrieval.

This module provides framework-agnostic abstractions for retrieving
relevant experiences from a knowledge base.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ExperienceEntry:
    """
    Represents a single experience entry in the knowledge base.
    
    Attributes:
        content: The main content/description of the experience
        metadata: Additional metadata (e.g., task type, domain, success rate)
        score: Relevance score (set during retrieval)
        id: Unique identifier for the experience
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceEntry":
        """Create from dictionary representation"""
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0),
            id=data.get("id")
        )


@dataclass
class RetrievalResult:
    """
    Result of an experience retrieval operation.
    
    Attributes:
        experiences: List of retrieved experience entries, sorted by relevance
        query: The original query string
        retrieval_time: Time taken for retrieval in seconds
        total_candidates: Total number of experiences considered
    """
    experiences: List[ExperienceEntry]
    query: str
    retrieval_time: float = 0.0
    total_candidates: int = 0
    
    def get_top_k(self, k: int) -> List[ExperienceEntry]:
        """Get top k most relevant experiences"""
        return self.experiences[:k]
    
    def filter_by_score(self, min_score: float) -> List[ExperienceEntry]:
        """Filter experiences by minimum score threshold"""
        return [exp for exp in self.experiences if exp.score >= min_score]


class ExperienceRetriever(ABC):
    """
    Abstract base class for experience retrieval systems.
    
    This interface allows different retrieval implementations (BM25, semantic search,
    graph matching, etc.) to be used interchangeably.
    """
    
    def __init__(
        self,
        experience_path: Optional[Path] = None,
        top_k: int = 5,
        **kwargs
    ):
        """
        Initialize the retriever.
        
        Args:
            experience_path: Path to the experience knowledge base file
            top_k: Number of top experiences to retrieve
            **kwargs: Additional configuration options
        """
        self.experience_path = experience_path
        self.top_k = top_k
        self.config = kwargs
        self._is_initialized = False
    
    @abstractmethod
    def load_experiences(self, path: Optional[Path] = None) -> int:
        """
        Load experiences from a file or database.
        
        Args:
            path: Path to load from (overrides initialization path)
            
        Returns:
            Number of experiences loaded
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant experiences for a given query.
        
        Args:
            query: The query string or object
            top_k: Number of results to return (overrides default)
            filters: Optional filters to apply (e.g., task_type, domain)
            
        Returns:
            RetrievalResult containing matched experiences
        """
        pass
    
    @abstractmethod
    def add_experience(self, experience: ExperienceEntry) -> bool:
        """
        Add a new experience to the knowledge base.
        
        Args:
            experience: The experience entry to add
            
        Returns:
            True if successfully added
        """
        pass
    
    def save_experiences(self, path: Optional[Path] = None) -> bool:
        """
        Save the current experience base to disk.
        
        Args:
            path: Path to save to (overrides initialization path)
            
        Returns:
            True if successfully saved
        """
        raise NotImplementedError("save_experiences not implemented")
    
    @property
    def is_initialized(self) -> bool:
        """Check if retriever has been initialized with experiences"""
        return self._is_initialized
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience base"""
        return {
            "total_experiences": 0,
            "is_initialized": self.is_initialized
        }
