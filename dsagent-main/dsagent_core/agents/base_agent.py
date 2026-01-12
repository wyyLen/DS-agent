"""
Base Agent - Framework-agnostic agent interface.

This provides a common interface that can be implemented using any LLM framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for DSAgent."""
    agent_id: str
    model: str = "qwen-plus-2025-12-01"
    temperature: float = 0.7
    max_tokens: int = 8192
    use_rag: bool = True
    use_reflection: bool = True
    text_exp_path: Optional[Path] = None
    workflow_exp_path: Optional[Path] = None


class BaseAgent(ABC):
    """
    Base class for all DSAgent implementations.
    
    This provides a framework-agnostic interface. Concrete implementations
    can use MetaGPT, AutoGen, LangChain, or any other framework.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self._active = False
    
    @abstractmethod
    async def acquire(self) -> bool:
        """
        Acquire agent for exclusive use.
        
        Returns:
            True if acquired, False if busy
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release agent and clear state."""
        pass
    
    @abstractmethod
    async def process_stream(
        self,
        query: str,
        uploaded_files: Optional[list] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming output.
        
        Args:
            query: User query
            uploaded_files: Optional uploaded file paths
            **kwargs: Additional parameters
            
        Yields:
            Dictionaries with response chunks
        """
        pass
    
    @abstractmethod
    def clear_state(self):
        """Clear agent's internal state."""
        pass


class CodeExecutor(ABC):
    """Abstract interface for code execution."""
    
    @abstractmethod
    async def run(self, code: str) -> tuple[str, bool]:
        """
        Execute code and return result.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (output, success)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset execution environment."""
        pass
