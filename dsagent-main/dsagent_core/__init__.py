"""
DSAgent Core - Framework-Agnostic Data Science Agent Mechanisms

This package provides reusable components for building intelligent data science agents:
- Text Experience Retrieval: BM25-based experience matching
- Workflow Experience Retrieval: Graph-based workflow matching  
- Tree Search: LATS (Language Agent Tree Search) for autonomous exploration

These components are designed to be framework-agnostic and can integrate with
MetaGPT, LangChain, AutoGen, or custom agent frameworks.
"""

from dsagent_core.retrieval.base import (
    ExperienceRetriever,
    RetrievalResult,
    ExperienceEntry
)
from dsagent_core.retrieval.text_retriever import TextExperienceRetriever
from dsagent_core.retrieval.workflow_retriever import WorkflowExperienceRetriever
from dsagent_core.search.tree_search import TreeSearchEngine, SearchNode

__version__ = "0.1.0"

__all__ = [
    "ExperienceRetriever",
    "RetrievalResult",
    "ExperienceEntry",
    "TextExperienceRetriever",
    "WorkflowExperienceRetriever",
    "TreeSearchEngine",
    "SearchNode",
]
