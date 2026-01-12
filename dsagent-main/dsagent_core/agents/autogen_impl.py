"""
AutoGen-based DSAgent implementation.

This wraps AutoGen agents to conform to the BaseAgent interface.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from pathlib import Path

from dsagent_core.agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class AutoGenDSAgent(BaseAgent):
    """
    DSAgent implementation using AutoGen framework.
    
    This is a wrapper around PureAutoGenDSAgent that provides
    the standard BaseAgent interface.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize AutoGen-based agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        try:
            # Import the pure AutoGen implementation
            import sys
            from pathlib import Path
            
            # Add agent_service to path if not already there
            agent_service_path = Path(__file__).parent.parent.parent / "examples" / "ds_agent" / "agent_service"
            if str(agent_service_path) not in sys.path:
                sys.path.insert(0, str(agent_service_path))
            
            from autogen_agent_service_pure import PureAutoGenDSAgent
            
            # Get API key from environment
            import os
            api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable")
            
            # Create AutoGen agent with configuration
            self.autogen_agent = PureAutoGenDSAgent(
                agent_id=config.agent_id,
                text_exp_path=config.text_exp_path,
                workflow_exp_path=config.workflow_exp_path,
                llm_config={
                    "model": config.model,
                    "api_key": api_key,
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
            )
            
            logger.info(f"✅ AutoGen DSAgent {config.agent_id} initialized")
            
        except ImportError as e:
            raise RuntimeError(
                f"AutoGen implementation not available: {e}. "
                "Make sure autogen_agent_service_pure.py is accessible."
            )
    
    async def acquire(self) -> bool:
        """Acquire agent for exclusive use."""
        return await self.autogen_agent.acquire()
    
    def release(self):
        """Release agent and clear state."""
        self.autogen_agent.release()
        logger.info(f"🛑 Agent {self.agent_id} released")
    
    async def process_stream(
        self,
        query: str,
        uploaded_files: Optional[list] = None,
        mode: str = "react",
        max_turns: int = 10,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming output.
        
        Args:
            query: User query
            uploaded_files: Optional uploaded file paths
            mode: Execution mode ('react' or 'lats')
            max_turns: Maximum conversation turns
            **kwargs: Additional parameters
            
        Yields:
            Dictionaries with response chunks
        """
        try:
            # AutoGen's process_stream already returns dicts
            async for chunk in self.autogen_agent.process_stream(
                query=query,
                uploaded_files=uploaded_files,
                mode=mode,
                max_turns=max_turns
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"AutoGen Agent {self.agent_id} error: {e}")
            from datetime import datetime
            yield {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_state(self):
        """Clear agent's internal state."""
        # AutoGen agent clears state on release
        self.autogen_agent.release()
