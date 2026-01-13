"""
Language Agent Tree Search (LATS) Core Implementation

Framework-agnostic implementation of LATS algorithm for data science tasks.
Can be used with MetaGPT, AutoGen, or any other agent framework.
"""

import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


@dataclass
class LATSNode:
    """
    Node in the LATS search tree.
    
    Represents a state in the problem-solving process with:
    - thought: The reasoning/plan for this step
    - action: The code/action to execute
    - observation: The result/feedback from execution
    """
    state: Dict[str, Any]
    question: str
    parent: Optional["LATSNode"] = None
    children: List["LATSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    is_success: bool = False
    is_terminal: bool = False
    reward: float = 0.0
    exhausted: bool = False  # All children explored
    
    def __post_init__(self):
        if self.parent:
            self.depth = self.parent.depth + 1
        if not self.state:
            self.state = {'thought': '', 'action': '', 'observation': ''}
    
    def uct(self, exploration_weight: float = 1.4) -> float:
        """Calculate UCT (Upper Confidence bound for Trees) score."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_weight * np.sqrt(
                2 * np.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def get_trajectory(self) -> str:
        """Get the full trajectory from root to this node."""
        trajectory = []
        node = self
        while node:
            segment = []
            if node.state.get('thought'):
                segment.append(f"Thought: {node.state['thought']}")
            if node.state.get('action'):
                segment.append(f"Action: {node.state['action']}")
            if node.state.get('observation'):
                segment.append(f"Observation: {node.state['observation']}")
            if segment:
                trajectory.append('\n'.join(segment))
            node = node.parent
        return '\n\n'.join(reversed(trajectory))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'state': self.state,
            'depth': self.depth,
            'visits': self.visits,
            'value': self.value,
            'reward': self.reward,
            'is_terminal': self.is_terminal,
            'is_success': self.is_success,
            'children_count': len(self.children)
        }


class CodeExecutor(ABC):
    """Abstract interface for executing code."""
    
    @abstractmethod
    async def execute(self, code: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Execute code and return success status and output.
        
        Args:
            code: Code to execute
            context: Execution context
            
        Returns:
            Tuple of (success, output/error)
        """
        pass
    
    @abstractmethod
    async def terminate(self):
        """Clean up resources."""
        pass


class ThoughtGenerator(ABC):
    """Abstract interface for generating thoughts/plans."""
    
    @abstractmethod
    async def generate(
        self,
        node: LATSNode,
        context: Dict[str, Any],
        n_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate thoughts/plans from current node.
        
        Args:
            node: Current node
            context: Generation context
            n_samples: Number of samples to generate
            
        Returns:
            List of thought dictionaries with 'thought' and 'task_type'
        """
        pass


class ActionGenerator(ABC):
    """Abstract interface for generating actions/code."""
    
    @abstractmethod
    async def generate(
        self,
        thought: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate action/code from thought.
        
        Args:
            thought: Thought dictionary
            context: Generation context
            
        Returns:
            Generated code/action
        """
        pass


class StateEvaluator(ABC):
    """Abstract interface for evaluating states."""
    
    @abstractmethod
    async def evaluate(
        self,
        trajectory: str,
        is_terminal: bool,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate a trajectory and return a score.
        
        Args:
            trajectory: Full trajectory string
            is_terminal: Whether this is a terminal state
            context: Evaluation context
            
        Returns:
            Evaluation score (0-10)
        """
        pass


class LATSCore:
    """
    Core LATS algorithm implementation - framework agnostic.
    
    This class implements the Language Agent Tree Search algorithm
    without dependencies on specific agent frameworks.
    """
    
    def __init__(
        self,
        thought_generator: ThoughtGenerator,
        action_generator: ActionGenerator,
        code_executor: CodeExecutor,
        state_evaluator: StateEvaluator,
        max_depth: int = 10,
        exploration_weight: float = 1.4,
        high_reward_threshold: float = 7.0
    ):
        """
        Initialize LATS core engine.
        
        Args:
            thought_generator: Generator for thoughts/plans
            action_generator: Generator for actions/code
            code_executor: Executor for code
            state_evaluator: Evaluator for states
            max_depth: Maximum tree depth
            exploration_weight: UCT exploration weight
            high_reward_threshold: Threshold for early termination
        """
        self.thought_generator = thought_generator
        self.action_generator = action_generator
        self.code_executor = code_executor
        self.state_evaluator = state_evaluator
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.high_reward_threshold = high_reward_threshold
        
        self.root: Optional[LATSNode] = None
        self.all_nodes: List[LATSNode] = []
        self.failed_trajectories: List[Dict[str, Any]] = []
    
    async def search(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[LATSNode, List[LATSNode]]:
        """
        Run LATS search algorithm.
        
        Args:
            goal: The problem/goal to solve
            iterations: Number of search iterations
            n_generate_sample: Number of samples per expansion
            context: Additional context
            
        Returns:
            Tuple of (best_node, all_nodes)
        """
        print("=" * 60)
        print("🌲 LATS CORE ACTIVATED - Framework-agnostic tree search")
        print(f"   Goal: {goal[:80]}...")
        print(f"   Iterations: {iterations}, Samples: {n_generate_sample}")
        print("=" * 60)
        
        context = context or {}
        context['goal'] = goal
        
        # Initialize root node
        self.root = LATSNode(
            state={'thought': '', 'action': '', 'observation': ''},
            question=goal
        )
        self.all_nodes = [self.root]
        
        for iteration in range(iterations):
            print(f"🔍 LATS Iteration {iteration + 1}/{iterations}")
            
            # Check for high-reward terminal nodes
            if iteration > 0:
                best_terminal = self._find_best_terminal_node()
                if best_terminal and best_terminal.reward >= self.high_reward_threshold:
                    print(f"✅ Found high-reward solution (reward={best_terminal.reward:.2f})")
                    return best_terminal, self.all_nodes
            
            # Select node using UCT
            node = self._select_node(self.root)
            
            if node is None:
                print("⚠️ No more nodes to explore")
                break
            
            # Skip terminal nodes with low reward
            if node.is_terminal and node.reward < self.high_reward_threshold:
                continue
            
            # Expand node
            await self._expand_node(node, context, n_generate_sample)
            
            # Evaluate children
            if node.children:
                await self._evaluate_children(node, context)
                
                # Backpropagate best child value
                best_child = max(node.children, key=lambda c: c.value)
                self._backpropagate(best_child, best_child.value)
        
        # Return best node found
        best_node = max(self.all_nodes, key=lambda n: n.reward, default=self.root)
        return best_node, self.all_nodes
    
    def _select_node(self, node: LATSNode) -> Optional[LATSNode]:
        """Select a node to expand using UCT."""
        if node.is_terminal:
            return None
        
        if not node.children:
            return node
        
        # Select child with highest UCT score
        unexplored = [c for c in node.children if c.visits == 0]
        if unexplored:
            return unexplored[0]
        
        best_child = max(
            [c for c in node.children if not c.exhausted],
            key=lambda c: c.uct(self.exploration_weight),
            default=None
        )
        
        if best_child:
            return self._select_node(best_child)
        
        node.exhausted = True
        return None
    
    async def _expand_node(
        self,
        node: LATSNode,
        context: Dict[str, Any],
        n_samples: int
    ):
        """Expand a node by generating and executing actions."""
        print(f"  📝 Expanding node at depth {node.depth}")
        
        try:
            # Generate thoughts
            thoughts = await self.thought_generator.generate(node, context, n_samples)
        except Exception as e:
            print(f"  ⚠️  Error generating thoughts: {e}")
            return
        
        unique_children = {}
        
        for i, thought in enumerate(thoughts):
            try:
                print(f"  🔄 Processing thought {i+1}/{len(thoughts)}")
                
                # Generate action
                action_code = await self.action_generator.generate(thought, context)
                
                # Execute action with error handling
                try:
                    is_success, observation = await self.code_executor.execute(
                        action_code,
                        context
                    )
                except Exception as exec_error:
                    print(f"  ⚠️  Execution error: {exec_error}")
                    is_success = False
                    observation = f"Execution failed: {str(exec_error)}"
                
                # Create child node
                child_state = {
                    'thought': thought,
                    'action': action_code,
                    'observation': observation
                }
                
                child = LATSNode(
                    state=child_state,
                    question=node.question,
                    parent=node,
                    is_success=is_success
                )
                
                child.is_terminal = (
                    thought.get('task_type') == 'finish' or
                    not is_success or
                    child.depth >= self.max_depth
                )
                
                # Avoid duplicates
                key = f"{child.depth}_{thought.get('thought', '')[:50]}"
                if key not in unique_children:
                    unique_children[key] = child
                    node.children.append(child)
                    self.all_nodes.append(child)
            except Exception as e:
                print(f"  ⚠️  Error processing thought: {e}")
                continue
        
        print(f"  ✅ Generated {len(node.children)} children")
    
    async def _evaluate_children(
        self,
        node: LATSNode,
        context: Dict[str, Any]
    ):
        """Evaluate all children of a node."""
        for child in node.children:
            trajectory = child.get_trajectory()
            score = await self.state_evaluator.evaluate(
                trajectory,
                child.is_terminal,
                context
            )
            child.value = score
            child.reward = score
    
    def _backpropagate(self, node: LATSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current:
            current.visits += 1
            if current.visits == 1:
                current.value = value
            else:
                # Apply penalty for low-reward terminals
                if current.is_terminal and current.reward < 5:
                    penalty = -1 * (current.depth / self.max_depth)
                    current.value = (current.value * (current.visits - 1) + penalty) / current.visits
                else:
                    current.value = (current.value * (current.visits - 1) + value) / current.visits
            
            if not current.is_success:
                current.value = 0
            
            current = current.parent
    
    def _find_best_terminal_node(self) -> Optional[LATSNode]:
        """Find the best terminal node."""
        terminals = [n for n in self.all_nodes if n.is_terminal]
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.reward)
    
    def get_solution_path(self, node: LATSNode) -> List[Dict[str, Any]]:
        """Get the solution path from root to given node."""
        path = []
        current = node
        while current and current.parent:  # Skip root
            path.append(current.state)
            current = current.parent
        return list(reversed(path))
    
    async def cleanup(self):
        """Clean up resources."""
        await self.code_executor.terminate()
