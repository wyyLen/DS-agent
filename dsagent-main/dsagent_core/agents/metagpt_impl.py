"""
MetaGPT-based DSAgent implementation.

This wraps MetaGPT's DataInterpreter to conform to the BaseAgent interface.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from pathlib import Path

# Import provider module first to register all LLM providers (including DASHSCOPE)
import metagpt.provider

from dsagent_core.agents.base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class MetaGPTDSAgent(BaseAgent):
    """
    DSAgent implementation using MetaGPT framework.
    
    This is a wrapper around MetaGPT's DataInterpreter that provides
    the standard BaseAgent interface.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize MetaGPT-based agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        try:
            # Import DSAgentStream (providers already registered at module level)
            from dsagent_core.roles.ds_agent_stream import DSAgentStream
            from metagpt.llm import LLM
            from metagpt.context import Context
            from metagpt.config2 import Config
            
            # Create LLM first to ensure provider is registered
            llm_instance = LLM()
            
            # Create a Context and set both _llm and private_llm to avoid config validation
            ctx = Context()
            ctx._llm = llm_instance
            
            # Create custom DSAgent with use_rag=False first to avoid validator issues
            self.metagpt_agent = DSAgentStream(
                name=f"DSAgent_{config.agent_id}",
                use_reflection=True,
                use_rag=False,  # Start with False to skip validator LLM creation
                use_kaggle_exp=False,
                use_exp_extractor=False,
                context=ctx
            )
            
            # Now set private_llm to ensure it uses the correct LLM
            self.metagpt_agent.private_llm = llm_instance
            
            # Manually initialize RAG after LLM is set
            try:
                import json
                from metagpt.rag.schema import FAISSRetrieverConfig
                from dsagent_core.rag.engines import CustomEngine
                from dsagent_core.const import EXP_PLAN
                from dsagent_core.roles.ds_agent_stream import get_rag_engine_llm
                from metagpt.configs.llm_config import LLMConfig, LLMType
                
                with open(EXP_PLAN, 'r', encoding='utf-8') as file:
                    exp_data = json.load(file)
                logger.info(f"Loading exp_bank with {len(exp_data)} entries for RAG")
                
                # Temporarily override context config to use the existing LLM instance
                # This prevents SimpleEngine from creating a new LLM with DASHSCOPE
                original_config_llm = ctx.config.llm
                ctx.config.llm = LLMConfig(
                    api_type=LLMType.OPENAI,  # Use a registered provider
                    model="gpt-3.5-turbo",
                    api_key="dummy"  # Won't be used since we pass llm explicitly
                )
                
                try:
                    self.metagpt_agent.rag_engine = CustomEngine.from_docs(
                        input_files=[EXP_PLAN],
                        retriever_configs=[FAISSRetrieverConfig(similarity_top_k=2)],
                        ranker_configs=[],
                        llm=get_rag_engine_llm(model_infer=llm_instance),
                    )
                    self.metagpt_agent.use_rag = True
                    logger.info("✅ RAG engine initialized successfully")
                finally:
                    # Restore original config
                    ctx.config.llm = original_config_llm
                    
            except Exception as e:
                logger.warning(f"⚠️ RAG initialization failed: {e}. Continuing without RAG.")
                self.metagpt_agent.use_rag = False
            
            self._lock = asyncio.Lock()
            rag_status = "with RAG" if self.metagpt_agent.use_rag else "without RAG"
            logger.info(f"✅ MetaGPT DSAgent {config.agent_id} initialized {rag_status}")
            
        except ImportError as e:
            raise RuntimeError(
                f"MetaGPT not available: {e}. "
                "Install with: pip install metagpt"
            )
    
    async def acquire(self) -> bool:
        """Acquire agent for exclusive use."""
        async with self._lock:
            if not self._active:
                self._active = True
                return True
            return False
    
    def release(self):
        """Release agent and clear state."""
        self.metagpt_agent.clear_content()
        self._active = False
        logger.info(f"🛑 Agent {self.agent_id} released")
    
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
            **kwargs: Additional parameters (e.g., mode='react' or 'lats')
            
        Yields:
            Dictionaries with response chunks
        """
        try:
            from datetime import datetime
            
            # DSAgentStream's stream_run returns text chunks directly
            sequence = 0
            async for chunk in self.metagpt_agent.stream_run(with_message=query):
                sequence += 1
                yield {
                    "type": "text_chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat(),
                    "sequence": sequence
                }
            
            yield {
                "type": "system",
                "content": "Processing completed",
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
    
    def clear_state(self):
        """Clear agent's internal state."""
        # DataInterpreter doesn't have clear_content(), no-op
        pass


class MetaGPTLATSAgent(BaseAgent):
    """
    LATS Agent implementation using MetaGPT framework.
    
    Wraps MetaGPT's LanguageAgentTreeSearchStream.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize MetaGPT LATS agent."""
        super().__init__(config)
        
        try:
            from dsagent_core.roles.lats_react_stream import LanguageAgentTreeSearchStream
            from metagpt.llm import LLM
            
            self.metagpt_agent = LanguageAgentTreeSearchStream(
                name=f"LATSAgent_{config.agent_id}",
                llm=LLM()
            )
            
            self._lock = asyncio.Lock()
            logger.info(f"✅ MetaGPT LATS Agent {config.agent_id} initialized")
            
        except ImportError as e:
            raise RuntimeError(f"MetaGPT not available: {e}")
    
    async def acquire(self) -> bool:
        """Acquire agent for exclusive use."""
        async with self._lock:
            if not self._active:
                self._active = True
                return True
            return False
    
    def release(self):
        """Release agent and clear state."""
        self.metagpt_agent.root = None
        self.metagpt_agent.all_nodes = []
        self.metagpt_agent.failed_trajectories = []
        self.metagpt_agent.terminal_nodes = []
        self._active = False
    
    async def process_stream(
        self,
        query: str,
        uploaded_files: Optional[list] = None,
        iterations: int = 10,
        n_generate_sample: int = 2,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process query using LATS tree search."""
        try:
            from datetime import datetime
            
            self.metagpt_agent.goal = query
            sequence = 0
            
            async for chunk in self.metagpt_agent.enhance_run(
                iterations=iterations,
                n_generate_sample=n_generate_sample
            ):
                sequence += 1
                yield {
                    "type": "text_chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat(),
                    "sequence": sequence
                }
            
            yield {
                "type": "system",
                "content": "LATS processing completed",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LATS Agent {self.agent_id} error: {e}")
            yield {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.release()
    
    def clear_state(self):
        """Clear LATS agent's search tree."""
        self.metagpt_agent.root = None
        self.metagpt_agent.all_nodes = []
        self.metagpt_agent.failed_trajectories = []
        self.metagpt_agent.terminal_nodes = []
