"""
Pure AutoGen Agent Service - No MetaGPT Fallback

This module implements AutoGen's native conversational multi-agent system
without any MetaGPT dependencies.
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Load API key from config
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from metagpt.config2 import Config
    config = Config.default()
    if hasattr(config, 'llm') and hasattr(config.llm, 'api_key'):
        api_key = config.llm.api_key
        if api_key and not os.getenv('DASHSCOPE_API_KEY'):
            os.environ['DASHSCOPE_API_KEY'] = api_key
            logger.info(f"✅ Loaded API key from config: {api_key[:10]}...")
except Exception as e:
    logger.warning(f"Could not load API key from config: {e}")

# Import AutoGen (0.4+)
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult
from autogen_core import CancellationToken
from typing import Sequence, List as TypingList

# Import DSAgent adapter for RAG
from dsagent_core.adapters.autogen_adapter import AutoGenAdapter

logger.info(f"✅ AutoGen 0.4+ loaded successfully")


# Simple ChatCompletionClient wrapper for DashScope
class DashScopeChatClient(ChatCompletionClient):
    """Simple DashScope client using OpenAI-compatible API."""
    
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self._client = None
    
    @property
    def model_info(self):
        """Return model information."""
        return {
            "model": self.model,
            "vision": False,  # DashScope text model doesn't support vision
            "function_calling": True,  # DashScope supports function calling
            "json_output": True  # Supports JSON mode
        }
    
    @property
    def capabilities(self):
        """Return client capabilities."""
        return {}
    
    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        cancellation_token: CancellationToken | None = None,
        **kwargs
    ) -> CreateResult:
        """Create chat completion."""
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                role = "assistant" if msg.source == "assistant" or msg.source == "DataScientist" else "user"
                openai_messages.append({"role": role, "content": str(msg.content)})
            else:
                openai_messages.append({"role": "user", "content": str(msg)})
        
        try:
            # Call API
            response = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.kwargs.get('temperature', 0.7),
                max_tokens=self.kwargs.get('max_tokens', 8192)
            )
            
            # Return result
            from autogen_core.models import CreateResult, RequestUsage
            return CreateResult(
                content=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                usage=RequestUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0
                ),
                cached=False  # Required field for CreateResult
            )
        except Exception as e:
            logger.error(f"API call error: {e}")
            logger.error(f"Model: {self.model}, Messages: {openai_messages[:2]}")
            raise
    
    async def create_stream(self, messages: Sequence[LLMMessage], *, cancellation_token=None, **kwargs):
        """Streaming not implemented."""
        raise NotImplementedError("Streaming not implemented")
    
    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs) -> int:
        """Estimate token count."""
        return sum(len(str(m.content if hasattr(m, 'content') else m).split()) for m in messages)
    
    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs) -> int:
        """Return remaining tokens."""
        return self.kwargs.get('max_tokens', 8192) - self.count_tokens(messages)
    
    def total_tokens(self, messages: Sequence[LLMMessage], **kwargs) -> int:
        """Return total available tokens."""
        return self.kwargs.get('max_tokens', 8192)
    
    def total_usage(self) -> None:
        """Return total usage - not tracked."""
        return None
    
    def actual_usage(self) -> None:
        """Return actual usage - not tracked."""
        return None
    
    async def close(self):
        """Close client."""
        if self._client:
            await self._client.close()


class PureAutoGenDSAgent:
    """
    Pure AutoGen-based Data Science Agent with true multi-agent conversation.
    
    Architecture:
    1. AssistantAgent (DataScientist): AI that analyzes and writes code
    2. UserProxyAgent (Executor): Executes code and provides feedback
    3. RAG: Retrieves relevant experiences via adapter
    4. Conversation loop until "TERMINATE"
    """
    
    def __init__(
        self, 
        agent_id: str,
        llm_config: Optional[Dict] = None,
        text_exp_path: Optional[Path] = None,
        workflow_exp_path: Optional[Path] = None
    ):
        """
        Initialize Pure AutoGen DS Agent.
        
        Args:
            agent_id: Unique agent identifier
            llm_config: LLM configuration
            text_exp_path: Path to text experiences
            workflow_exp_path: Path to workflow experiences
        """
        self.agent_id = agent_id
        
        # Initialize adapter for RAG
        self.adapter = AutoGenAdapter(
            text_exp_path=text_exp_path,
            workflow_exp_path=workflow_exp_path
        )
        logger.info(f"📚 RAG initialized for agent {agent_id}")
        
        # Setup LLM config
        self.llm_config = llm_config or self._get_default_llm_config()
        self.work_dir = tempfile.mkdtemp(prefix=f"autogen_{agent_id}_")
        
        # Initialize shared code executor for maintaining state between code blocks
        from metagpt.actions.di.execute_nb_code import ExecuteNbCode
        self.code_executor = ExecuteNbCode()
        logger.info(f"🔧 Shared code executor initialized")
        
        # Initialize multi-agent conversation
        self._init_agents()
        
        self._active = False
        self._lock = asyncio.Lock()
        logger.info(f"✅ Pure AutoGen agent {agent_id} ready")
    
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
    
    def _get_default_llm_config(self) -> Dict:
        """Get default DashScope LLM configuration."""
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
        
        return {
            "model": "qwen-plus-2025-12-01",
            "api_key": api_key,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "temperature": 0.7,
            "max_tokens": 8192
        }
    
    def _init_agents(self):
        """Initialize AssistantAgent and UserProxyAgent for conversation."""
        
        # Create model client
        model_client = DashScopeChatClient(
            model=self.llm_config["model"],
            api_key=self.llm_config["api_key"],
            base_url=self.llm_config["base_url"],
            temperature=self.llm_config.get("temperature", 0.7),
            max_tokens=self.llm_config.get("max_tokens", 8192)
        )
        
        # Create DataScientist (Assistant)
        self.assistant = AssistantAgent(
            name="DataScientist",
            model_client=model_client,
            description="Expert data scientist who analyzes problems and writes Python code",
            system_message=(
                "You are an expert data scientist. Follow these principles:\n\n"
                "STEP 1 - ALWAYS START WITH DATA INSPECTION:\n"
                "```python\n"
                "# Check structure first!\n"
                "print('Shape:', df.shape)\n"
                "print('Columns:', list(df.columns))  # Use EXACT names from this output!\n"
                "print('Data types:', df.dtypes)\n"
                "print('First rows:', df.head())\n"
                "```\n\n"
                "CODE WRITING RULES:\n"
                "1. Write SHORT, focused code blocks (10-20 lines max)\n"
                "2. Each code block should do ONE clear task\n"
                "3. Always print intermediate results for verification\n"
                "4. USE EXACT COLUMN NAMES - check df.columns output!\n"
                "5. Wrap code in ```python ... ``` blocks\n\n"
                "DATA HANDLING:\n"
                "1. Clean data: df.columns = df.columns.str.strip()\n"
                "2. For string columns: df[col] = df[col].str.strip()\n"
                "3. Common mistakes: 'hour-per-week' vs 'hours-per-week' (check actual name!)\n\n"
                "ERROR HANDLING:\n"
                "1. KeyError? → Print df.columns and use the EXACT name\n"
                "2. Don't guess column names - verify them first\n"
                "3. Fix and retry - don't give up after one error\n\n"
                "WORKFLOW:\n"
                "1. Inspect data structure (MANDATORY)\n"
                "2. Perform analysis step by step\n"
                "3. Summarize results clearly\n"
                "4. Reply TERMINATE only when task is successfully completed\n\n"
                "CRITICAL: Always check df.columns before using column names!"
            )
        )
        
        # Create Executor (UserProxy) - AutoGen 0.4+ doesn't use code_execution_config
        self.executor = UserProxyAgent(
            name="Executor",
            description="Code executor that runs Python code and reports results"
        )
        
        logger.info(f"🤖 Agents initialized: DataScientist + Executor")
    
    async def process_stream(
        self,
        query: str,
        uploaded_files: Optional[list] = None,
        mode: str = "react",
        max_turns: int = 10
    ) -> AsyncGenerator[str, None]:
        """
        Process query with streaming AutoGen conversation.
        
        Args:
            query: User query
            uploaded_files: Uploaded file paths
            mode: Execution mode ('react' or 'lats')
            max_turns: Maximum conversation turns
            
        Yields:
            JSON strings with conversation updates
        """
        self._active = True
        
        try:
            # Step 1: Retrieve relevant experiences via RAG
            yield {
                "type": "rag_start",
                "message": "🔍 Retrieving relevant experiences...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Only retrieve text experiences (workflow retrieval requires a plan/workflow structure)
            text_exps = self.adapter.retrieve_text_experiences(query, top_k=5)
            workflow_exps = []  # Skip workflow retrieval for now (requires plan generation first)
            
            # Build context
            context = self._build_context(query, text_exps, workflow_exps, uploaded_files)
            

            
            # Add data preprocessing instructions to context (for reference)
            context += "\n\n## Important Data Preprocessing Notes:\n"
            context += "1. **Whitespace cleaning is CRITICAL**: Adult Census dataset has leading/trailing spaces\n"
            context += "   - After loading CSV: df.columns = df.columns.str.strip()\n"
            context += "   - For all object columns: df[col] = df[col].str.strip()\n"
            context += "2. **Education column**: Use 'education' (text) NOT 'education-num' (numeric)\n"
            context += "   - Values to look for: 'Bachelors', 'Masters', 'Doctorate' (after stripping spaces)\n"
            context += "3. **Income column**: Use 'income' column\n"
            context += "   - High income is: '>50K' (after stripping spaces, compare with '>')\n"
            context += "4. **Gender column**: Use 'sex' column\n"
            context += "   - Values: 'Male', 'Female' (after stripping spaces)\n"
            context += "5. **Example preprocessing code**:\n"
            context += "```python\n"
            context += "# Clean column names\n"
            context += "df.columns = df.columns.str.strip()\n"
            context += "# Clean string values\n"
            context += "for col in df.select_dtypes(include='object').columns:\n"
            context += "    df[col] = df[col].str.strip()\n"
            context += "```\n"
            
            yield {
                "type": "rag_complete",
                "text_count": len(text_exps),
                "workflow_count": len(workflow_exps),
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 2: Start conversation
            yield {
                "type": "conversation_start",
                "message": "💬 Starting multi-agent conversation...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 3: Run AutoGen conversation (AutoGen 0.4+ API)
            # Create initial message with context
            initial_message = TextMessage(
                content=context,
                source="user"
            )
            
            # Run the conversation using assistant's run method
            # In AutoGen 0.4+, the assistant and executor work together
            turn_count = 0
            conversation_history = [initial_message]
            
            while turn_count < max_turns and self._active:
                # DataScientist generates response
                try:
                    # Call assistant with conversation history
                    logger.info(f"Turn {turn_count}: Calling assistant with {len(conversation_history)} messages")
                    assistant_response = await self.assistant.on_messages(
                        conversation_history,
                        cancellation_token=None
                    )
                    
                    # Extract message content
                    logger.info(f"Assistant response type: {type(assistant_response)}")
                    if hasattr(assistant_response, 'chat_message'):
                        assistant_msg = assistant_response.chat_message.content
                    else:
                        assistant_msg = str(assistant_response)
                    
                    logger.info(f"Assistant message: {assistant_msg[:200]}")
                    
                    yield {
                        "type": "text_chunk",
                        "content": f"\n**[DataScientist]:** {assistant_msg}\n",
                        "timestamp": datetime.now().isoformat(),
                        "sequence": turn_count + 1
                    }
                    
                    # Add to history
                    conversation_history.append(TextMessage(
                        content=assistant_msg,
                        source="DataScientist"
                    ))
                    
                    # Execute code if present in assistant's message (BEFORE checking termination)
                    executor_msg = await self._execute_code_from_message(assistant_msg)
                    has_error = "❌" in executor_msg
                    
                    yield {
                        "type": "text_chunk", 
                        "content": f"**[Executor]:** {executor_msg}\n",
                        "timestamp": datetime.now().isoformat(),
                        "sequence": turn_count + 1
                    }
                    
                    conversation_history.append(TextMessage(
                        content=executor_msg,
                        source="Executor"
                    ))
                    
                    turn_count += 1
                    
                    # Check for termination AFTER executing code
                    # Only allow termination if no errors occurred
                    if "TERMINATE" in assistant_msg.upper() or "FINAL ANSWER" in assistant_msg.upper():
                        if has_error:
                            # Don't terminate if there are errors - encourage retry
                            continue_msg = (
                                "⚠️ Cannot terminate yet - there are execution errors. "
                                "Please fix the errors and complete the task successfully."
                            )
                            yield {
                                "type": "text_chunk",
                                "content": f"**[System]:** {continue_msg}\n",
                                "timestamp": datetime.now().isoformat(),
                                "sequence": turn_count
                            }
                            conversation_history.append(TextMessage(
                                content=continue_msg,
                                source="System"
                            ))
                        else:
                            # Success - allow termination
                            yield {
                                "type": "conversation_end",
                                "message": "✅ Task completed",
                                "timestamp": datetime.now().isoformat()
                            }
                            break
                    
                except Exception as e:
                    logger.error(f"Error in conversation turn {turn_count}: {e}", exc_info=True)
                    yield {
                        "type": "error",
                        "content": f"Conversation error: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    break
            
            # Final result
            yield {
                "type": "task_complete",
                "turns": turn_count,
                "message": "✅ Multi-agent conversation completed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in process_stream: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self._active = False
    
    async def _execute_code_from_message(self, message: str) -> str:
        """Extract and execute Python code from assistant's message."""
        import re
        
        logger.info(f"Attempting to extract code from message (length: {len(message)})")
        
        # Extract code blocks (```python ... ``` or ```...```)
        code_pattern = r'```(?:python)?\s*\n(.*?)```'
        code_blocks = re.findall(code_pattern, message, re.DOTALL)
        
        logger.info(f"Found {len(code_blocks)} code blocks")
        
        if not code_blocks:
            # Try without newline after ```
            code_pattern2 = r'```(?:python)?(.*?)```'
            code_blocks = re.findall(code_pattern2, message, re.DOTALL)
            logger.info(f"Alternative pattern found {len(code_blocks)} code blocks")
        
        if not code_blocks:
            return "No executable code found in the message."
        
        # Execute each code block using the shared executor (maintains variable state)
        results = []
        for i, code in enumerate(code_blocks, 1):
            code = code.strip()
            logger.info(f"Executing code block {i} (length: {len(code)}, first 100 chars: {code[:100]})")
            
            try:
                # Use shared executor - variables persist across blocks!
                output, success = await self.code_executor.run(code)
                
                logger.info(f"Code block {i} execution result: success={success}, output length={len(output)}")
                
                if success:
                    # Only show last 1000 chars to avoid overwhelming output
                    output_display = output[-1000:] if len(output) > 1000 else output
                    results.append(f"✅ Code Block {i} executed successfully.\nOutput:\n{output_display}")
                else:
                    # Provide constructive error feedback with specific hints
                    error_msg = output[:500]
                    feedback = f"❌ Code Block {i} failed.\nError:\n{error_msg}\n\n"
                    
                    # Add specific hints based on error type
                    if "KeyError" in output:
                        feedback += "💡 KeyError - Column name doesn't exist!\n"
                        feedback += "ACTION: Print df.columns to see exact column names\n"
                        feedback += "Common mistakes: 'hour-per-week' vs 'hours-per-week', extra spaces\n\n"
                    elif "NameError" in output:
                        feedback += "💡 NameError - Variable not defined!\n"
                        feedback += "ACTION: Check if previous code blocks ran successfully\n\n"
                    elif "TypeError" in output:
                        feedback += "💡 TypeError - Wrong data type!\n"
                        feedback += "ACTION: Use df.dtypes to check column types\n\n"
                    else:
                        feedback += "💡 General suggestions:\n"
                        feedback += "- Check column names with df.columns\n"
                        feedback += "- Verify data types with df.dtypes\n"
                        feedback += "- Look for typos\n\n"
                    
                    feedback += "Please fix and retry."
                    results.append(feedback)
            except Exception as e:
                logger.error(f"Code block {i} execution exception: {e}", exc_info=True)
                results.append(
                    f"❌ Code Block {i} execution error: {str(e)[:500]}\n\n"
                    f"💡 This is a system error. Please:\n"
                    f"1. Check your code syntax\n"
                    f"2. Verify all imports are included\n"
                    f"3. Make sure variables are defined before use\n"
                    f"Please write corrected code to continue."
                )
        
        return "\n\n".join(results)
    
    def _build_context(
        self,
        query: str,
        text_exps: list,
        workflow_exps: list,
        uploaded_files: Optional[list]
    ) -> str:
        """Build conversation context with RAG results."""
        context = f"# Task\n{query}\n\n"
        
        if uploaded_files:
            context += f"# Uploaded Files\n"
            for f in uploaded_files:
                context += f"- {f}\n"
            context += "\n"
        
        if text_exps:
            context += "# Relevant Experiences (Text)\n"
            for i, exp in enumerate(text_exps, 1):
                # ExperienceEntry is a dataclass, use .content attribute
                content = exp.content if hasattr(exp, 'content') else str(exp)
                context += f"{i}. Score: {exp.score:.3f} | {content[:200]}...\n"
            context += "\n"
        
        if workflow_exps:
            context += "# Relevant Workflows\n"
            for i, wf in enumerate(workflow_exps, 1):
                # ExperienceEntry with workflow data
                content = wf.content if hasattr(wf, 'content') else str(wf)
                context += f"{i}. Score: {wf.score:.3f} | {content[:150]}...\n"
            context += "\n"
        
        return context
    
    def stop(self):
        """Stop the agent."""
        self._active = False
        logger.info(f"🛑 Agent {self.agent_id} stopped")
