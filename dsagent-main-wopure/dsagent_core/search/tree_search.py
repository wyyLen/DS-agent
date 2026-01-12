"""
Language Agent Tree Search (LATS) implementation.

This module provides a framework-agnostic tree search mechanism
for autonomous exploration of solution spaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import math


class NodeStatus(Enum):
    """Status of a search node"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class SearchNode:
    """
    Represents a node in the search tree.
    
    Attributes:
        state: Current state representation (e.g., plan, code, observation)
        parent: Parent node in the tree
        children: Child nodes
        depth: Depth in the tree (root is 0)
        visits: Number of times this node has been visited
        value: Accumulated value/reward
        status: Current status of the node
        metadata: Additional metadata
    """
    state: Any
    parent: Optional["SearchNode"] = None
    children: List["SearchNode"] = field(default_factory=list)
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: "SearchNode"):
        """Add a child node"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_path_to_root(self) -> List["SearchNode"]:
        """Get path from this node to root"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (success or failed)"""
        return self.status in [NodeStatus.SUCCESS, NodeStatus.FAILED]
    
    def get_ucb_score(self, exploration_weight: float = 1.414) -> float:
        """
        Calculate Upper Confidence Bound (UCB) score for selection.
        
        Args:
            exploration_weight: Exploration constant (sqrt(2) by default)
            
        Returns:
            UCB score
        """
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration


class ActionGenerator(ABC):
    """Abstract interface for generating possible actions from a state"""
    
    @abstractmethod
    def generate_actions(
        self,
        state: Any,
        context: Dict[str, Any]
    ) -> List[Any]:
        """
        Generate possible actions from the current state.
        
        Args:
            state: Current state
            context: Additional context information
            
        Returns:
            List of possible actions/next states
        """
        pass


class StateEvaluator(ABC):
    """Abstract interface for evaluating states"""
    
    @abstractmethod
    def evaluate(
        self,
        state: Any,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate a state and return a score.
        
        Args:
            state: State to evaluate
            context: Additional context information
            
        Returns:
            Evaluation score (higher is better)
        """
        pass


class TerminationChecker(ABC):
    """Abstract interface for checking if a state is terminal"""
    
    @abstractmethod
    def is_terminal(
        self,
        state: Any,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if the state is terminal (goal reached or failed).
        
        Args:
            state: State to check
            context: Additional context information
            
        Returns:
            True if terminal
        """
        pass


class TreeSearchEngine:
    """
    Tree search engine for autonomous exploration.
    
    Implements a Monte Carlo Tree Search (MCTS) variant with
    customizable action generation, evaluation, and termination.
    """
    
    def __init__(
        self,
        action_generator: ActionGenerator,
        state_evaluator: StateEvaluator,
        termination_checker: TerminationChecker,
        max_depth: int = 10,
        max_iterations: int = 100,
        exploration_weight: float = 1.414,
        **kwargs
    ):
        """
        Initialize tree search engine.
        
        Args:
            action_generator: Generator for possible actions
            state_evaluator: Evaluator for states
            termination_checker: Checker for terminal states
            max_depth: Maximum depth of search tree
            max_iterations: Maximum number of search iterations
            exploration_weight: UCB exploration weight
            **kwargs: Additional configuration
        """
        self.action_generator = action_generator
        self.state_evaluator = state_evaluator
        self.termination_checker = termination_checker
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.config = kwargs
    
    def search(
        self,
        initial_state: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchNode:
        """
        Perform tree search from initial state.
        
        Args:
            initial_state: Starting state
            context: Additional context information
            
        Returns:
            Best leaf node found
        """
        ctx = context or {}
        root = SearchNode(state=initial_state, depth=0)
        
        for iteration in range(self.max_iterations):
            # 1. Selection: Select most promising node
            node = self._select(root)
            
            # 2. Expansion: Expand if not terminal
            if not node.is_terminal() and node.depth < self.max_depth:
                node = self._expand(node, ctx)
            
            # 3. Simulation/Evaluation: Evaluate the node
            value = self._evaluate(node, ctx)
            
            # 4. Backpropagation: Update ancestors
            self._backpropagate(node, value)
            
            # Check for early termination
            if self._should_terminate(root, ctx):
                break
        
        # Return best leaf node
        return self._get_best_leaf(root)
    
    def _select(self, node: SearchNode) -> SearchNode:
        """
        Select most promising node using UCB.
        
        Args:
            node: Current node
            
        Returns:
            Selected leaf or expandable node
        """
        current = node
        
        while not current.is_leaf() and not current.is_terminal():
            # Select child with highest UCB score
            current = max(
                current.children,
                key=lambda c: c.get_ucb_score(self.exploration_weight)
            )
        
        return current
    
    def _expand(
        self,
        node: SearchNode,
        context: Dict[str, Any]
    ) -> SearchNode:
        """
        Expand a node by generating children.
        
        Args:
            node: Node to expand
            context: Context information
            
        Returns:
            One of the newly created children
        """
        # Check if terminal
        if self.termination_checker.is_terminal(node.state, context):
            node.status = NodeStatus.SUCCESS
            return node
        
        # Generate possible actions
        actions = self.action_generator.generate_actions(node.state, context)
        
        if not actions:
            node.status = NodeStatus.FAILED
            return node
        
        # Create child nodes
        for action in actions:
            child = SearchNode(state=action, parent=node)
            node.add_child(child)
        
        # Return first child for evaluation
        return node.children[0] if node.children else node
    
    def _evaluate(
        self,
        node: SearchNode,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate a node.
        
        Args:
            node: Node to evaluate
            context: Context information
            
        Returns:
            Evaluation score
        """
        return self.state_evaluator.evaluate(node.state, context)
    
    def _backpropagate(self, node: SearchNode, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            node: Starting node
            value: Value to propagate
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    def _should_terminate(
        self,
        root: SearchNode,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if search should terminate early.
        
        Args:
            root: Root node
            context: Context information
            
        Returns:
            True if should terminate
        """
        # Find if we have a highly confident solution
        best_leaf = self._get_best_leaf(root)
        if best_leaf.is_terminal() and best_leaf.status == NodeStatus.SUCCESS:
            avg_value = best_leaf.value / best_leaf.visits if best_leaf.visits > 0 else 0
            if avg_value > 0.9:  # High confidence threshold
                return True
        
        return False
    
    def _get_best_leaf(self, root: SearchNode) -> SearchNode:
        """
        Get best leaf node from tree.
        
        Args:
            root: Root node
            
        Returns:
            Best leaf node
        """
        def collect_leaves(node: SearchNode) -> List[SearchNode]:
            if node.is_leaf():
                return [node]
            leaves = []
            for child in node.children:
                leaves.extend(collect_leaves(child))
            return leaves
        
        leaves = collect_leaves(root)
        if not leaves:
            return root
        
        # Return leaf with highest average value
        return max(
            leaves,
            key=lambda n: (n.value / n.visits) if n.visits > 0 else float('-inf')
        )
    
    def get_best_path(self, root: SearchNode) -> List[SearchNode]:
        """
        Get the best path from root to a leaf.
        
        Args:
            root: Root node
            
        Returns:
            List of nodes representing the best path
        """
        best_leaf = self._get_best_leaf(root)
        return best_leaf.get_path_to_root()
    
    def get_statistics(self, root: SearchNode) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Args:
            root: Root node
            
        Returns:
            Dictionary of statistics
        """
        def count_nodes(node: SearchNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)
        
        def count_by_status(node: SearchNode) -> Dict[NodeStatus, int]:
            counts = {status: 0 for status in NodeStatus}
            
            def traverse(n: SearchNode):
                counts[n.status] += 1
                for child in n.children:
                    traverse(child)
            
            traverse(node)
            return counts
        
        status_counts = count_by_status(root)
        
        return {
            "total_nodes": count_nodes(root),
            "max_depth_reached": self._get_max_depth(root),
            "status_distribution": {k.value: v for k, v in status_counts.items()},
            "root_visits": root.visits,
            "root_value": root.value
        }
    
    def _get_max_depth(self, node: SearchNode) -> int:
        """Get maximum depth in tree"""
        if node.is_leaf():
            return node.depth
        return max(self._get_max_depth(child) for child in node.children)
