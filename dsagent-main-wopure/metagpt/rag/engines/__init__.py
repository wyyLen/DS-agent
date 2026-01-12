"""Engines init"""

from metagpt.rag.engines.simple import SimpleEngine
from metagpt.rag.engines.flare import FLAREEngine
from metagpt.rag.engines.customMixture import CustomMixtureEngine

# Alias for backward compatibility
CustomEngine = CustomMixtureEngine

__all__ = ["SimpleEngine", "FLAREEngine", "CustomMixtureEngine", "CustomEngine"]
