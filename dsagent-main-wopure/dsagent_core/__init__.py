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
from dsagent_core.search.lats_core import LATSCore, LATSNode

try:
    from dsagent_core.adapters import (
        AutoGenAdapter,
        AUTOGEN_AVAILABLE,
        MetaGPTLATSAdapter,
        METAGPT_LATS_AVAILABLE,
        AutoGenLATSAdapter,
        AUTOGEN_LATS_AVAILABLE,
        create_autogen_lats,
    )
except ImportError:
    AutoGenAdapter = None
    AUTOGEN_AVAILABLE = False
    MetaGPTLATSAdapter = None
    METAGPT_LATS_AVAILABLE = False
    AutoGenLATSAdapter = None
    AUTOGEN_LATS_AVAILABLE = False
    create_autogen_lats = None

__version__ = "0.2.0"

__all__ = [
    # Retrieval
    "ExperienceRetriever",
    "RetrievalResult",
    "ExperienceEntry",
    "TextExperienceRetriever",
    "WorkflowExperienceRetriever",
    # Search
    "TreeSearchEngine",
    "SearchNode",
    "LATSCore",
    "LATSNode",
    # Adapters (optional)
    "AutoGenAdapter",
    "AUTOGEN_AVAILABLE",
    "MetaGPTLATSAdapter",
    "METAGPT_LATS_AVAILABLE",
    "AutoGenLATSAdapter",
    "AUTOGEN_LATS_AVAILABLE",
    "create_autogen_lats",
]
