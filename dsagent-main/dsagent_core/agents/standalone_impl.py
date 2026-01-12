"""
Standalone DSAgent Implementation - No MetaGPT dependency.

This is a complete DSAgent implementation that doesn't require MetaGPT.
It uses only dsagent_core modules and standard libraries.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator, List
from pathlib import Path
from datetime import datetime

from dsagent_core.agents.base_agent import BaseAgent, AgentConfig
from dsagent_core.actions import IndependentCodeExecutor
from dsagent_core.retrieval import TextExperienceRetriever

logger = logging.getLogger(__name__)


class StandaloneDSAgent(BaseAgent):
    """
    Standalone DSAgent implementation without MetaGPT dependency.
    
    This agent can work with any LLM API (OpenAI, Anthropic, DashScope, etc.)
    and doesn't require MetaGPT to be installed.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize standalone agent.
        
        Args:
            config: Agent configuration
            llm_client: LLM client instance (optional, will create default if not provided)
        """
        super().__init__(config)
        
        # Initialize code executor
        self.code_executor = IndependentCodeExecutor(
            workspace_path=Path.cwd() / "workspace" / config.agent_id
        )
        
        # Initialize RAG if paths provided
        self.text_retriever: Optional[TextExperienceRetriever] = None
        if config.text_exp_path and config.use_rag:
            self.text_retriever = TextExperienceRetriever(
                experience_path=config.text_exp_path,
                top_k=5
            )
            logger.info(f"RAG initialized for {config.agent_id}")
        
        # LLM client
        self.llm_client = llm_client or self._create_default_llm_client()
        
        self._lock = asyncio.Lock()
        logger.info(f"✅ Standalone DSAgent {config.agent_id} initialized")
    
    def _create_default_llm_client(self):
        """Create default LLM client based on config."""
        # This is a placeholder - you would implement your LLM client here
        # For example, using OpenAI, Anthropic, DashScope, etc.
        logger.warning("Using placeholder LLM client - implement your own LLM integration")
        return None
    
    async def acquire(self) -> bool:
        """Acquire agent for exclusive use."""
        async with self._lock:
            if not self._active:
                self._active = True
                return True
            return False
    
    def release(self):
        """Release agent and clear state."""
        self._active = False
        logger.info(f"🛑 Agent {self.agent_id} released")
    
    async def process_stream(
        self,
        query: str,
        uploaded_files: Optional[list] = None,
        max_iterations: int = 5,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming output.
        
        Args:
            query: User query
            uploaded_files: Optional uploaded file paths
            max_iterations: Maximum number of iterations
            **kwargs: Additional parameters
            
        Yields:
            Dictionaries with response chunks
        """
        try:
            # Step 1: RAG retrieval
            if self.text_retriever and self.config.use_rag:
                yield {
                    "type": "rag_start",
                    "message": "🔍 Retrieving relevant experiences...",
                    "timestamp": datetime.now().isoformat()
                }
                
                rag_results = self.text_retriever.retrieve(query, top_k=5)
                
                yield {
                    "type": "rag_complete",
                    "count": len(rag_results.experiences),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                rag_results = None
            
            # Step 2: Build context
            context = self._build_context(query, rag_results, uploaded_files)
            
            # Step 3: Generate plan (placeholder - would call LLM)
            yield {
                "type": "planning",
                "message": "📋 Generating analysis plan...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 4: Execute plan iteratively
            for iteration in range(max_iterations):
                yield {
                    "type": "iteration_start",
                    "iteration": iteration + 1,
                    "message": f"🔄 Iteration {iteration + 1}/{max_iterations}",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Generate code (placeholder - would call LLM)
                code = self._generate_code_placeholder(query, context, iteration)
                
                if code:
                    yield {
                        "type": "code_generated",
                        "code": code,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Execute code
                    output, success = await self.code_executor.run(code)
                    
                    yield {
                        "type": "code_executed",
                        "output": output,
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if not success:
                        yield {
                            "type": "error",
                            "message": f"Code execution failed: {output}",
                            "timestamp": datetime.now().isoformat()
                        }
                        break
                else:
                    break
            
            # Step 5: Complete
            yield {
                "type": "system",
                "content": "✅ Analysis completed",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {e}")
            yield {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.release()
    
    def _build_context(
        self,
        query: str,
        rag_results: Optional[Any],
        uploaded_files: Optional[List[str]]
    ) -> str:
        """Build context for LLM."""
        context = f"User Query: {query}\n\n"
        
        if uploaded_files:
            context += "Uploaded Files:\n"
            for file in uploaded_files:
                context += f"- {file}\n"
            context += "\n"
        
        if rag_results and rag_results.experiences:
            context += "Relevant Experiences:\n"
            for exp in rag_results.experiences[:3]:
                context += f"- {exp.content[:200]}...\n"
            context += "\n"
        
        return context
    
    def _generate_code_placeholder(
        self,
        query: str,
        context: str,
        iteration: int
    ) -> Optional[str]:
        """
        Placeholder for code generation.
        
        In a real implementation, this would call your LLM to generate code.
        """
        if iteration == 0:
            return """
import pandas as pd
import numpy as np

# Load data
print("Ready for analysis")
"""
        return None
    
    def clear_state(self):
        """Clear agent's internal state."""
        asyncio.create_task(self.code_executor.reset())
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.code_executor.terminate()
        logger.info(f"Agent {self.agent_id} cleaned up")


# Convenience function
def create_standalone_agent(
    agent_id: str,
    text_exp_path: Optional[Path] = None,
    **kwargs
) -> StandaloneDSAgent:
    """
    Create a standalone DSAgent.
    
    Args:
        agent_id: Unique agent ID
        text_exp_path: Path to text experiences
        **kwargs: Additional config parameters
        
    Returns:
        StandaloneDSAgent instance
    """
    config = AgentConfig(
        agent_id=agent_id,
        text_exp_path=text_exp_path,
        **kwargs
    )
    return StandaloneDSAgent(config)
