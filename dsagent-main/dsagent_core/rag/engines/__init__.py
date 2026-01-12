"""RAG engines package."""

from metagpt.rag.engines import SimpleEngine as CustomEngine  # Alias for backward compatibility
from dsagent_core.rag.engines.customMixture import CustomMixtureEngine
from dsagent_core.rag.engines.customWorkflowGM import CustomWorkflowGMEngine
from dsagent_core.rag.engines.customSolutionSamplesGenerate import SolutionSpaceGenerateEngine
from dsagent_core.rag.engines.customEmbeddingComparisonEngine import CustomEmbeddingComparisonEngine

__all__ = [
    "CustomEngine",  # SimpleEngine alias
    "CustomEmbeddingComparisonEngine",
    "CustomMixtureEngine",
    "CustomWorkflowGMEngine",
    "SolutionSpaceGenerateEngine"
]
