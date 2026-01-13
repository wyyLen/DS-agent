"""
MetaGPT LATS Adapter

Adapts MetaGPT's capabilities to work with the framework-agnostic LATS core.
"""

import asyncio
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


class MetaGPTCodeExecutor(CodeExecutor):
    """Code executor using MetaGPT's ExecuteNbCode."""
    
    def __init__(self):
        from metagpt.actions import ExecuteNbCode
        self.executor = ExecuteNbCode()
    
    async def execute(self, code: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute code using Jupyter notebook."""
        try:
            result = await self.executor.run(code=code)
            success = result.get('is_success', False)
            output = result.get('output', '')
            return success, output
        except Exception as e:
            return False, str(e)
    
    async def terminate(self):
        """Terminate the executor."""
        if hasattr(self.executor, 'nb') and self.executor.nb:
            await self.executor.terminate()


class MetaGPTThoughtGenerator(ThoughtGenerator):
    """Thought generator using MetaGPT's LLM."""
    
    def __init__(self, use_exp_driven: bool = True):
        from metagpt.actions.lats.lats_react import GenerateAction
        self.generate_action = GenerateAction()
        self.use_exp_driven = use_exp_driven
    
    async def generate(
        self,
        node: LATSNode,
        context: Dict[str, Any],
        n_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate thoughts using MetaGPT's GenerateAction."""
        goal = context.get('goal', '')
        prompt = self._build_prompt(node, goal)
        
        thoughts = []
        for _ in range(n_samples):
            thought_str = await self.generate_action.generate_thought(
                prompt,
                depth=node.depth + 1
            )
            
            # Parse thought
            import json
            try:
                if isinstance(thought_str, str):
                    thought = json.loads(thought_str)
                else:
                    thought = thought_str
                thoughts.append(thought)
            except:
                # Fallback
                thoughts.append({
                    'thought': str(thought_str),
                    'task_type': 'other'
                })
        
        return thoughts
    
    def _build_prompt(self, node: LATSNode, goal: str) -> str:
        """Build prompt from node trajectory."""
        trajectory = []
        current = node
        while current:
            if current.state.get('thought'):
                trajectory.append(f"Thought: {current.state['thought']}")
            if current.state.get('action'):
                trajectory.append(f"Action: {current.state['action']}")
            if current.state.get('observation'):
                trajectory.append(f"Observation: {current.state['observation']}")
            current = current.parent
        
        return goal + '\n' + '\n'.join(reversed(trajectory))


class MetaGPTActionGenerator(ActionGenerator):
    """Action generator using MetaGPT's GenerateAction."""
    
    def __init__(self):
        from metagpt.actions.lats.lats_react import GenerateAction
        self.generate_action = GenerateAction()
    
    async def generate(
        self,
        thought: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate code from thought."""
        thought_str = thought.get('thought', '')
        task_type = thought.get('task_type', 'other')
        
        # Use MetaGPT's action generation
        code = await self.generate_action.generate_action(
            thought_str,
            task_type=task_type
        )
        
        return code


class MetaGPTStateEvaluator(StateEvaluator):
    """State evaluator using MetaGPT's evaluation."""
    
    def __init__(self):
        from metagpt.actions.lats.lats_react import GenerateAction
        self.generate_action = GenerateAction()
    
    async def evaluate(
        self,
        trajectory: str,
        is_terminal: bool,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate trajectory using MetaGPT's evaluation."""
        goal = context.get('goal', '')
        
        if is_terminal:
            score = await self.generate_action.evaluate_terminal_trajectory(
                goal,
                trajectory,
                n_evaluate_sample=1
            )
        else:
            score = await self.generate_action.evaluate_current_trajectory(
                goal,
                trajectory,
                n_evaluate_sample=1
            )
        
        return float(score)


class MetaGPTLATSAdapter:
    """
    Adapter that wraps the LATS core with MetaGPT implementations.
    
    This provides a drop-in replacement for MetaGPT's LanguageAgentTreeSearch
    while using the framework-agnostic core.
    """
    
    def __init__(
        self,
        use_exp_driven_search: bool = True,
        max_depth: int = 10,
        high_reward_threshold: float = 7.0
    ):
        """
        Initialize MetaGPT LATS adapter.
        
        Args:
            use_exp_driven_search: Enable experience-driven search
            max_depth: Maximum tree depth
            high_reward_threshold: Threshold for early termination
        """
        # Create MetaGPT-specific implementations
        self.code_executor = MetaGPTCodeExecutor()
        self.thought_generator = MetaGPTThoughtGenerator(use_exp_driven_search)
        self.action_generator = MetaGPTActionGenerator()
        self.state_evaluator = MetaGPTStateEvaluator()
        
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
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Tuple[LATSNode, List[LATSNode]]:
        """
        Run LATS search.
        
        Args:
            iterations: Number of search iterations
            n_generate_sample: Number of samples per expansion
            
        Returns:
            Tuple of (best_node, all_nodes)
        """
        if not self.goal:
            raise ValueError("Goal must be set before running")
        
        print("\n🚀 MetaGPT LATS Adapter - Starting tree search")
        
        best_node, all_nodes = await self.lats_core.search(
            goal=self.goal,
            iterations=iterations,
            n_generate_sample=n_generate_sample
        )
        
        return best_node, all_nodes
    
    async def enhance_run(
        self,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> str:
        """
        Run LATS and return formatted conclusion.
        
        Args:
            iterations: Number of search iterations
            n_generate_sample: Number of samples per expansion
            
        Returns:
            Formatted conclusion string
        """
        best_node, all_nodes = await self.run(iterations, n_generate_sample)
        
        # Clean up executors
        for node in all_nodes:
            if hasattr(node, 'execute_code'):
                await node.execute_code.terminate()
        
        # Generate conclusion
        from metagpt.actions.ds_agent.conclude_res import Conclusion
        
        solution_path = self.lats_core.get_solution_path(best_node)
        tasks_with_res = [
            {
                "task_instruction": step.get('thought', ''),
                "task_res": step.get('observation', '')
            }
            for step in solution_path
        ]
        
        conclusion = await Conclusion().run(
            final_goal=self.goal,
            tasks_res=tasks_with_res
        )
        
        return conclusion
    
    @property
    def root(self):
        """Get root node (for compatibility)."""
        return self.lats_core.root
    
    @root.setter
    def root(self, value):
        """Set root node (for compatibility)."""
        self.lats_core.root = value
    
    @property
    def all_nodes(self):
        """Get all nodes (for compatibility)."""
        return self.lats_core.all_nodes
    
    @all_nodes.setter
    def all_nodes(self, value):
        """Set all nodes (for compatibility)."""
        self.lats_core.all_nodes = value
    
    @property
    def failed_trajectories(self):
        """Get failed trajectories (for compatibility)."""
        return self.lats_core.failed_trajectories
    
    @failed_trajectories.setter
    def failed_trajectories(self, value):
        """Set failed trajectories (for compatibility)."""
        self.lats_core.failed_trajectories = value
    
    def calculate_total_cost(self) -> Tuple[int, int]:
        """
        Calculate total LLM cost.
        
        Returns:
            Tuple of (prompt_tokens, completion_tokens)
        """
        prompt_tokens = 0
        completion_tokens = 0
        
        # Sum up costs from all action generators
        if hasattr(self.thought_generator.generate_action, 'llm'):
            llm = self.thought_generator.generate_action.llm
            if hasattr(llm, 'cost_manager'):
                prompt_tokens += llm.cost_manager.total_prompt_tokens
                completion_tokens += llm.cost_manager.total_completion_tokens
        
        return prompt_tokens, completion_tokens
