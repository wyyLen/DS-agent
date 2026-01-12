"""
DSAgent Core Actions - Independent action modules.

These actions don't depend on MetaGPT and can be used standalone.
"""

from dsagent_core.actions.execute_code import (
    IndependentCodeExecutor,
    execute_code
)

__all__ = [
    "IndependentCodeExecutor",
    "execute_code"
]
