"""
AutoGen LATS Adapter

Adapts AutoGen's capabilities to work with the framework-agnostic LATS core.
This enables tree search for AutoGen agents.
"""

import asyncio
import tempfile
import os
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from dsagent_core.search.lats_core import (
    LATSCore,
    LATSNode,
    CodeExecutor,
    ThoughtGenerator,
    ActionGenerator,
    StateEvaluator
)


class AutoGenCodeExecutor(CodeExecutor):
    """Code executor using Jupyter for AutoGen."""
    
    def __init__(self):
        self.executor = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of executor."""
        if not self._initialized:
            from metagpt.actions import ExecuteNbCode
            self.executor = ExecuteNbCode()
            self._initialized = True
    
    async def execute(self, code: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute code using Jupyter notebook."""
        try:
            await self._ensure_initialized()
            
            # Add timeout and error handling
            import asyncio
            result = await asyncio.wait_for(
                self.executor.run(code=code),
                timeout=60.0  # 60 second timeout
            )
            success = result.get('is_success', False)
            output = result.get('output', '')
            
            # Limit output length
            if len(output) > 5000:
                output = output[:5000] + "\n... (output truncated)"
            
            return success, output
        except asyncio.TimeoutError:
            return False, "Execution timeout (60s)"
        except Exception as e:
            import traceback
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            print(f"⚠️  Code execution failed: {error_msg}")
            return False, error_msg
    
    async def terminate(self):
        """Terminate the executor."""
        if self.executor and hasattr(self.executor, 'nb') and self.executor.nb:
            try:
                await self.executor.terminate()
            except Exception as e:
                print(f"⚠️  Error terminating executor: {e}")


class AutoGenThoughtGenerator(ThoughtGenerator):
    """Thought generator using AutoGen's LLM."""
    
    def __init__(self, model_client, system_message: Optional[str] = None):
        """
        Initialize thought generator.
        
        Args:
            model_client: AutoGen model client
            system_message: System message for thought generation
        """
        self.model_client = model_client
        self.system_message = system_message or self._default_system_message()
    
    def _default_system_message(self) -> str:
        return """You are a data science expert. Generate the next step to solve the problem.

Output JSON format:
{
    "thought": "Your reasoning and plan for this step",
    "task_type": "data_preprocess|eda|feature_engineering|model_train|other|finish"
}

Choose task_type='finish' only when the problem is completely solved."""
    
    async def generate(
        self,
        node: LATSNode,
        context: Dict[str, Any],
        n_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate thoughts using AutoGen's LLM."""
        import json
        from autogen_core.models import LLMMessage, SystemMessage, UserMessage
        
        goal = context.get('goal', '')
        trajectory = node.get_trajectory()
        
        messages = [
            SystemMessage(content=self.system_message),
            UserMessage(
                content=f"Problem: {goal}\n\nCurrent progress:\n{trajectory}\n\nGenerate the next step:",
                source="user"
            )
        ]
        
        thoughts = []
        for _ in range(n_samples):
            try:
                result = await self.model_client.create(messages)
                thought_str = result.content
                
                # Parse JSON
                # Try to extract JSON from markdown code blocks
                if '```json' in thought_str:
                    thought_str = thought_str.split('```json')[1].split('```')[0].strip()
                elif '```' in thought_str:
                    thought_str = thought_str.split('```')[1].split('```')[0].strip()
                
                thought = json.loads(thought_str)
                thoughts.append(thought)
            except Exception as e:
                # Fallback
                thoughts.append({
                    'thought': f"Continue analysis (parse error: {e})",
                    'task_type': 'other'
                })
        
        return thoughts


class AutoGenActionGenerator(ActionGenerator):
    """Action generator using AutoGen's LLM."""
    
    def __init__(self, model_client, system_message: Optional[str] = None):
        """
        Initialize action generator.
        
        Args:
            model_client: AutoGen model client
            system_message: System message for code generation
        """
        self.model_client = model_client
        self.system_message = system_message or self._default_system_message()
    
    def _default_system_message(self) -> str:
        return """You are a data science coding expert. Generate Python code to accomplish the given task.

Requirements:
- Write clean, executable Python code
- Use pandas, numpy, sklearn, matplotlib as needed
- Handle errors gracefully
- Print results clearly

Output only the Python code, no explanations."""
    
    async def generate(
        self,
        thought: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate code from thought using AutoGen's LLM."""
        from autogen_core.models import LLMMessage, SystemMessage, UserMessage
        
        thought_str = thought.get('thought', '')
        task_type = thought.get('task_type', 'other')
        
        messages = [
            SystemMessage(content=self.system_message),
            UserMessage(
                content=f"Task: {thought_str}\nTask Type: {task_type}\n\nGenerate Python code:",
                source="user"
            )
        ]
        
        try:
            result = await self.model_client.create(messages)
            code = result.content
            
            # Extract code from markdown if present
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].split('```')[0].strip()
            
            return code
        except Exception as e:
            return f"# Error generating code: {e}\nprint('Code generation failed')"


class AutoGenStateEvaluator(StateEvaluator):
    """State evaluator using AutoGen's LLM."""
    
    def __init__(self, model_client, system_message: Optional[str] = None):
        """
        Initialize state evaluator.
        
        Args:
            model_client: AutoGen model client
            system_message: System message for evaluation
        """
        self.model_client = model_client
        self.system_message = system_message or self._default_system_message()
    
    def _default_system_message(self) -> str:
        return """You are an evaluator for data science solutions. Rate the quality of the solution trajectory.

Scoring criteria (0-10):
- 0-3: Poor, wrong approach or major errors
- 4-6: Partial solution, some progress made
- 7-8: Good solution, minor issues
- 9-10: Excellent, complete solution

Output only a single number (0-10)."""
    
    async def evaluate(
        self,
        trajectory: str,
        is_terminal: bool,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate trajectory using AutoGen's LLM."""
        from autogen_core.models import LLMMessage, SystemMessage, UserMessage
        
        goal = context.get('goal', '')
        
        prompt = f"""Problem: {goal}

Solution trajectory:
{trajectory}

Is this a complete solution? {is_terminal}

Rate this solution (0-10):"""
        
        messages = [
            SystemMessage(content=self.system_message),
            UserMessage(content=prompt, source="user")
        ]
        
        try:
            result = await self.model_client.create(messages)
            score_str = result.content.strip()
            
            # Extract number
            import re
            match = re.search(r'\d+', score_str)
            if match:
                score = float(match.group())
                return min(max(score, 0), 10)  # Clamp to 0-10
            else:
                return 5.0  # Default middle score
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 5.0


class AutoGenLATSAdapter:
    """
    AutoGen LATS Adapter - enables tree search for AutoGen agents.
    
    This adapter wraps the framework-agnostic LATS core with AutoGen-specific
    implementations, allowing AutoGen to use the LATS algorithm.
    """
    
    def __init__(
        self,
        model_client,
        max_depth: int = 10,
        high_reward_threshold: float = 7.0,
        thought_system_message: Optional[str] = None,
        action_system_message: Optional[str] = None,
        eval_system_message: Optional[str] = None
    ):
        """
        Initialize AutoGen LATS adapter.
        
        Args:
            model_client: AutoGen ChatCompletionClient
            max_depth: Maximum tree depth
            high_reward_threshold: Threshold for early termination
            thought_system_message: Custom system message for thoughts
            action_system_message: Custom system message for actions
            eval_system_message: Custom system message for evaluation
        """
        self.model_client = model_client
        
        # Create AutoGen-specific implementations
        self.code_executor = AutoGenCodeExecutor()
        self.thought_generator = AutoGenThoughtGenerator(
            model_client,
            thought_system_message
        )
        self.action_generator = AutoGenActionGenerator(
            model_client,
            action_system_message
        )
        self.state_evaluator = AutoGenStateEvaluator(
            model_client,
            eval_system_message
        )
        
        # Initialize core LATS engine
        self.lats_core = LATSCore(
            thought_generator=self.thought_generator,
            action_generator=self.action_generator,
            code_executor=self.code_executor,
            state_evaluator=self.state_evaluator,
            max_depth=max_depth,
            high_reward_threshold=high_reward_threshold
        )
        
        self.goal: str = ""
    
    async def run(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Tuple[LATSNode, List[LATSNode]]:
        """
        Run LATS search.
        
        Args:
            goal: Problem to solve
            iterations: Number of search iterations
            n_generate_sample: Number of samples per expansion
            
        Returns:
            Tuple of (best_node, all_nodes)
        """
        print("\n🚀 AutoGen LATS Adapter - Starting tree search")
        self.goal = goal
        
        best_node, all_nodes = await self.lats_core.search(
            goal=goal,
            iterations=iterations,
            n_generate_sample=n_generate_sample
        )
        
        return best_node, all_nodes
    
    async def run_and_format(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Dict[str, Any]:
        """
        Run LATS and return formatted result.
        
        Args:
            goal: Problem to solve
            iterations: Number of search iterations
            n_generate_sample: Number of samples per expansion
            
        Returns:
            Dictionary with solution details
        """
        best_node, all_nodes = await self.run(goal, iterations, n_generate_sample)
        
        solution_path = self.lats_core.get_solution_path(best_node)
        
        return {
            'goal': goal,
            'best_reward': best_node.reward,
            'depth': best_node.depth,
            'nodes_explored': len(all_nodes),
            'solution_steps': [
                {
                    'thought': step.get('thought', ''),
                    'action': step.get('action', ''),
                    'observation': step.get('observation', '')
                }
                for step in solution_path
            ],
            'final_output': best_node.state.get('observation', '')
        }
    
    def get_solution_code(self, node: LATSNode) -> str:
        """
        Extract all code from solution path.
        
        Args:
            node: Solution node
            
        Returns:
            Combined code string
        """
        solution_path = self.lats_core.get_solution_path(node)
        code_blocks = [
            step.get('action', '')
            for step in solution_path
            if step.get('action')
        ]
        return '\n\n'.join(code_blocks)
    
    async def cleanup(self):
        """Clean up resources."""
        await self.lats_core.cleanup()
    
    @property
    def root(self):
        """Get root node."""
        return self.lats_core.root
    
    @property
    def all_nodes(self):
        """Get all nodes."""
        return self.lats_core.all_nodes


def create_autogen_lats(
    api_key: str,
    model: str = "qwen-plus",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    **kwargs
) -> AutoGenLATSAdapter:
    """
    Convenience function to create AutoGen LATS adapter with DashScope.
    
    Args:
        api_key: DashScope API key
        model: Model name
        base_url: API base URL
        **kwargs: Additional arguments for AutoGenLATSAdapter
        
    Returns:
        Configured AutoGen LATS adapter
        
    Example:
        >>> lats = create_autogen_lats(
        ...     api_key="your-key",
        ...     model="qwen-plus"
        ... )
        >>> result = await lats.run_and_format(
        ...     goal="Analyze the data and build a model",
        ...     iterations=10
        ... )
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from examples.ds_agent.agent_service.autogen_agent_service_pure import DashScopeChatClient
    
    model_client = DashScopeChatClient(
        model=model,
        api_key=api_key,
        base_url=base_url
    )
    
    return AutoGenLATSAdapter(model_client, **kwargs)
