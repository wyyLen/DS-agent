"""
Agent Factory - Create agents based on framework selection.

This provides a unified interface to create agents using different frameworks.
"""

import logging
from typing import Optional, Literal
from pathlib import Path

from dsagent_core.agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

FrameworkType = Literal["metagpt", "autogen"]


class AgentFactory:
    """Factory for creating DSAgent instances with different frameworks."""
    
    @staticmethod
    def create_agent(
        agent_id: str,
        framework: FrameworkType = "metagpt",
        agent_type: str = "ds",  # 'ds' or 'lats'
        **config_kwargs
    ) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_id: Unique agent identifier
            framework: Framework to use ('metagpt' or 'autogen')
            agent_type: Type of agent ('ds' or 'lats')
            **config_kwargs: Additional configuration parameters
            
        Returns:
            BaseAgent instance
            
        Raises:
            ValueError: If framework or agent_type is invalid
            RuntimeError: If framework is not available
        """
        # Create agent configuration
        config = AgentConfig(
            agent_id=agent_id,
            model=config_kwargs.get("model", "qwen-plus-2025-12-01"),
            temperature=config_kwargs.get("temperature", 0.7),
            max_tokens=config_kwargs.get("max_tokens", 8192),
            use_rag=config_kwargs.get("use_rag", True),
            use_reflection=config_kwargs.get("use_reflection", True),
            text_exp_path=config_kwargs.get("text_exp_path"),
            workflow_exp_path=config_kwargs.get("workflow_exp_path")
        )
        
        logger.info(f"Creating {framework.upper()} {agent_type.upper()} agent: {agent_id}")
        
        if framework == "metagpt":
            return AgentFactory._create_metagpt_agent(agent_type, config)
        elif framework == "autogen":
            return AgentFactory._create_autogen_agent(agent_type, config)
        else:
            raise ValueError(f"Unknown framework: {framework}. Choose 'metagpt' or 'autogen'")
    
    @staticmethod
    def _create_metagpt_agent(agent_type: str, config: AgentConfig) -> BaseAgent:
        """Create MetaGPT-based agent."""
        try:
            from dsagent_core.agents.metagpt_impl import MetaGPTDSAgent, MetaGPTLATSAgent
            
            if agent_type == "ds":
                return MetaGPTDSAgent(config)
            elif agent_type == "lats":
                return MetaGPTLATSAgent(config)
            else:
                raise ValueError(f"Unknown MetaGPT agent type: {agent_type}")
                
        except ImportError as e:
            raise RuntimeError(
                f"MetaGPT framework not available: {e}\n"
                "Install with: pip install metagpt"
            )
    
    @staticmethod
    def _create_autogen_agent(agent_type: str, config: AgentConfig) -> BaseAgent:
        """Create AutoGen-based agent."""
        try:
            from dsagent_core.agents.autogen_impl import AutoGenDSAgent
            
            if agent_type == "ds":
                return AutoGenDSAgent(config)
            elif agent_type == "lats":
                raise NotImplementedError("LATS is not yet implemented for AutoGen")
            else:
                raise ValueError(f"Unknown AutoGen agent type: {agent_type}")
                
        except ImportError as e:
            raise RuntimeError(
                f"AutoGen framework not available: {e}\n"
                "Install with: pip install autogen-agentchat autogen-core"
            )
    
    @staticmethod
    def list_available_frameworks() -> dict:
        """
        Check which frameworks are available.
        
        Returns:
            Dictionary with framework availability status
        """
        available = {}
        
        # Check MetaGPT
        try:
            import metagpt
            from metagpt.roles.di.data_interpreter import DataInterpreter
            available["metagpt"] = True
        except ImportError:
            available["metagpt"] = False
        
        # Check AutoGen
        try:
            from autogen_agentchat.agents import AssistantAgent
            available["autogen"] = True
        except ImportError:
            available["autogen"] = False
        
        return available


# Convenience function
def create_agent(
    agent_id: str,
    framework: str = "metagpt",
    agent_type: str = "ds",
    **kwargs
) -> BaseAgent:
    """
    Convenience function to create an agent.
    
    Args:
        agent_id: Unique agent identifier
        framework: Framework to use ('metagpt' or 'autogen')
        agent_type: Type of agent ('ds' or 'lats')
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseAgent instance
    """
    return AgentFactory.create_agent(agent_id, framework, agent_type, **kwargs)
