"""
DataInterpreter role wrapper for direct MetaGPT usage.

This module provides a thin wrapper around MetaGPT's DataInterpreter
to maintain consistent interface with DSAgent architecture.
"""

from metagpt.roles.di.data_interpreter import DataInterpreter as MetaGPTDataInterpreter

# Re-export MetaGPT's DataInterpreter directly
DataInterpreter = MetaGPTDataInterpreter

__all__ = ["DataInterpreter"]
