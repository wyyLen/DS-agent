"""
Refactored Agent Service using DSAgent Core Factory Pattern.

This service provider no longer directly imports MetaGPT or AutoGen.
Instead, it uses the AgentFactory to create agents dynamically.
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Union
from pathlib import Path

from fastapi import HTTPException

# Use DSAgent Core factory instead of direct imports
from dsagent_core.agents import AgentFactory, BaseAgent, AgentConfig

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 框架选择: 'metagpt' 或 'autogen'
AGENT_FRAMEWORK = os.getenv('AGENT_FRAMEWORK', 'metagpt').lower()


# ------------------ SSE格式化工具 ------------------
class StreamFormatter:
    @staticmethod
    def format_data(data: dict) -> str:
        """直接返回JSON数据，不添加SSE格式的头信息"""
        json_data = json.dumps(data, ensure_ascii=False)
        return json_data + "\n"


# ------------------ API服务封装 ------------------
class AgentServiceProvider:
    """
    Agent service provider using factory pattern.
    
    No direct dependency on MetaGPT or AutoGen - uses AgentFactory instead.
    """
    
    def __init__(self, initial_agent_counts: dict = {"ds": 1, "lats": 1}, framework: str = None):
        self.framework = framework or AGENT_FRAMEWORK
        self.agents_pool: Dict[str, Dict[str, BaseAgent]] = {
            "ds": {},
            "lats": {}
        }
        
        # Check framework availability
        available_frameworks = AgentFactory.list_available_frameworks()
        
        logger.info(f"=" * 60)
        logger.info(f"Agent Framework: {self.framework.upper()}")
        logger.info(f"Available frameworks: {available_frameworks}")
        
        if not available_frameworks.get(self.framework, False):
            logger.warning(f"{self.framework} not available")
            # Find first available framework
            for fw, avail in available_frameworks.items():
                if avail:
                    logger.info(f"Falling back to {fw}")
                    self.framework = fw
                    break
            else:
                raise RuntimeError("No agent frameworks available!")
        
        logger.info(f"=" * 60)
        
        self._init_agent_pool(initial_agent_counts)
    
    def _init_agent_pool(self, initial_agent_counts: dict):
        """Initialize agent pool using factory."""
        # Setup experience paths
        text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
        workflow_exp_path = Path("examples/data/exp_bank/workflow_exp2_clean_new.json")
        
        for mode, count in initial_agent_counts.items():
            # Skip LATS for AutoGen (not implemented)
            if mode == "lats" and self.framework == "autogen":
                logger.info("Skipping LATS initialization for AutoGen (not implemented)")
                continue
            
            for _ in range(count):
                agent_id = f"{mode.upper()}-{len(self.agents_pool[mode]) + 1}-{int(time.time())}"
                
                try:
                    # Use factory to create agent
                    agent = AgentFactory.create_agent(
                        agent_id=agent_id,
                        framework=self.framework,
                        agent_type=mode,
                        text_exp_path=text_exp_path if text_exp_path.exists() else None,
                        workflow_exp_path=workflow_exp_path if workflow_exp_path.exists() else None
                    )
                    
                    self.agents_pool[mode][agent_id] = agent
                    logger.info(f"✅ Initialized {self.framework} {mode} agent {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {mode} agent: {e}")
    
    async def get_idle_agent(self, mode: str = "ds") -> BaseAgent:
        """Get an idle agent from the pool."""
        if mode not in self.agents_pool:
            raise HTTPException(400, detail=f"Invalid agent mode: {mode}")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            for agent in self.agents_pool[mode].values():
                if await agent.acquire():
                    return agent
            await asyncio.sleep(0.1)
        
        raise HTTPException(503, detail=f"No available {mode} agents")
    
    async def stream_generator(
        self,
        agent: BaseAgent,
        query: str,
        session_id: str = None,
        uploaded_files: list = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from agent.
        
        Args:
            agent: Agent instance
            query: User query
            session_id: Optional session ID
            uploaded_files: Optional uploaded files
            **kwargs: Additional parameters
            
        Yields:
            Formatted JSON strings
        """
        try:
            # Stream from agent (framework-agnostic)
            async for chunk in agent.process_stream(
                query=query,
                uploaded_files=uploaded_files,
                **kwargs
            ):
                yield StreamFormatter.format_data(chunk)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield StreamFormatter.format_data(error_chunk)


# Global service provider instance
_service_provider: AgentServiceProvider = None


def get_service_provider() -> AgentServiceProvider:
    """Get or create global service provider."""
    global _service_provider
    if _service_provider is None:
        _service_provider = AgentServiceProvider()
    return _service_provider
