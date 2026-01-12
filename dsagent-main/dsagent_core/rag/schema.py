"""Custom RAG schemas for DSAgent."""
from pydantic import Field

from metagpt.rag.schema import IndexRetrieverConfig


class MixtureRetrieverConfig(IndexRetrieverConfig):
    """Config for Mixture-based retrievers."""

    dimensions: int = Field(default=1536, description="Dimensionality of the vectors for FAISS index construction.")
