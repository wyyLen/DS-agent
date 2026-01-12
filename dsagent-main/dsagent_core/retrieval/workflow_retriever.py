"""
Workflow-based experience retrieval using graph matching.

This module provides graph-based matching of workflow structures,
allowing retrieval of similar solution patterns.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from dsagent_core.retrieval.base import (
    ExperienceRetriever,
    ExperienceEntry,
    RetrievalResult
)


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow"""
    task_id: str
    instruction: str
    task_type: str
    dependent_task_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "task_type": self.task_type,
            "dependent_task_ids": self.dependent_task_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTask":
        return cls(
            task_id=data.get("task_id", ""),
            instruction=data.get("instruction", ""),
            task_type=data.get("task_type", "other"),
            dependent_task_ids=data.get("dependent_task_ids", [])
        )


@dataclass
class Workflow:
    """Represents a complete workflow as a directed acyclic graph (DAG)"""
    tasks: List[WorkflowTask]
    
    def get_task_by_id(self, task_id: str) -> Optional[WorkflowTask]:
        """Get task by its ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_dependencies(self) -> Dict[str, List[str]]:
        """Get dependency map: task_id -> list of dependent task ids"""
        return {task.task_id: task.dependent_task_ids for task in self.tasks}
    
    def get_task_types(self) -> List[str]:
        """Get list of task types in order"""
        return [task.task_type for task in self.tasks]
    
    def to_dict(self) -> List[Dict[str, Any]]:
        return [task.to_dict() for task in self.tasks]
    
    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> "Workflow":
        tasks = [WorkflowTask.from_dict(task_data) for task_data in data]
        return cls(tasks=tasks)


class WorkflowMatcher:
    """
    Matches workflows based on structure and task types.
    
    Uses a graph edit distance-inspired approach to measure
    similarity between workflow DAGs.
    """
    
    def __init__(
        self,
        type_weight: float = 0.7,
        structure_weight: float = 0.3
    ):
        """
        Initialize matcher with scoring weights.
        
        Args:
            type_weight: Weight for task type similarity
            structure_weight: Weight for graph structure similarity
        """
        self.type_weight = type_weight
        self.structure_weight = structure_weight
    
    def calculate_similarity(
        self,
        workflow1: Workflow,
        workflow2: Workflow
    ) -> float:
        """
        Calculate similarity score between two workflows.
        
        Args:
            workflow1: First workflow
            workflow2: Second workflow
            
        Returns:
            Similarity score between 0 and 1
        """
        # Task type similarity
        types1 = workflow1.get_task_types()
        types2 = workflow2.get_task_types()
        type_sim = self._calculate_sequence_similarity(types1, types2)
        
        # Structure similarity (based on dependencies)
        struct_sim = self._calculate_structure_similarity(
            workflow1.get_dependencies(),
            workflow2.get_dependencies()
        )
        
        # Combined score
        total_score = (
            self.type_weight * type_sim +
            self.structure_weight * struct_sim
        )
        
        return total_score
    
    def _calculate_sequence_similarity(
        self,
        seq1: List[str],
        seq2: List[str]
    ) -> float:
        """
        Calculate similarity between two sequences using longest common subsequence.
        
        Returns value between 0 and 1.
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Simple LCS-based similarity
        lcs_length = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        return lcs_length / max_len if max_len > 0 else 0.0
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_structure_similarity(
        self,
        deps1: Dict[str, List[str]],
        deps2: Dict[str, List[str]]
    ) -> float:
        """
        Calculate structural similarity based on dependency patterns.
        
        Returns value between 0 and 1.
        """
        if not deps1 or not deps2:
            return 0.0
        
        # Calculate degree distributions
        in_degrees1 = self._calculate_in_degrees(deps1)
        in_degrees2 = self._calculate_in_degrees(deps2)
        
        # Compare degree distributions
        max_degree = max(
            max(in_degrees1.values()) if in_degrees1 else 0,
            max(in_degrees2.values()) if in_degrees2 else 0
        )
        
        if max_degree == 0:
            return 1.0
        
        # Simple degree distribution similarity
        degree_diff = sum(
            abs(in_degrees1.get(i, 0) - in_degrees2.get(i, 0))
            for i in range(max_degree + 1)
        )
        
        max_possible_diff = len(deps1) + len(deps2)
        
        return 1.0 - (degree_diff / max_possible_diff) if max_possible_diff > 0 else 0.0
    
    def _calculate_in_degrees(self, deps: Dict[str, List[str]]) -> Dict[int, int]:
        """Calculate in-degree distribution of the DAG"""
        in_degree_count = defaultdict(int)
        in_degrees = defaultdict(int)
        
        # Count in-degrees for each node
        for node, dependencies in deps.items():
            for dep in dependencies:
                in_degrees[dep] += 1
        
        # Count frequency of each in-degree
        for degree in in_degrees.values():
            in_degree_count[degree] += 1
        
        # Also count nodes with in-degree 0
        nodes_with_zero_degree = len(deps) - len(in_degrees)
        if nodes_with_zero_degree > 0:
            in_degree_count[0] = nodes_with_zero_degree
        
        return dict(in_degree_count)


class WorkflowExperienceRetriever(ExperienceRetriever):
    """
    Workflow-based experience retrieval using graph matching.
    
    This retriever loads workflow experiences and matches them based
    on structural and semantic similarity.
    """
    
    def __init__(
        self,
        experience_path: Optional[Path] = None,
        top_k: int = 5,
        type_weight: float = 0.7,
        structure_weight: float = 0.3,
        min_similarity_threshold: float = 0.3,
        **kwargs
    ):
        """
        Initialize workflow experience retriever.
        
        Args:
            experience_path: Path to JSON file containing workflow experiences
            top_k: Number of top experiences to retrieve
            type_weight: Weight for task type matching
            structure_weight: Weight for structure matching
            min_similarity_threshold: Minimum similarity score threshold
            **kwargs: Additional configuration
        """
        super().__init__(experience_path=experience_path, top_k=top_k, **kwargs)
        self.matcher = WorkflowMatcher(
            type_weight=type_weight,
            structure_weight=structure_weight
        )
        self.workflows: List[Tuple[Workflow, ExperienceEntry]] = []
        self.min_similarity_threshold = min_similarity_threshold
        self._is_initialized = False
        
        # Load experiences if path provided
        if experience_path:
            self.load_experiences(experience_path)
    
    @property
    def experiences(self) -> List[ExperienceEntry]:
        """Get all loaded workflow experiences."""
        return [exp for _, exp in self.workflows]

    
    def load_experiences(self, path: Optional[Path] = None) -> int:
        """
        Load workflow experiences from JSON file.
        
        Expected JSON format:
        [
            {
                "workflow": [
                    {
                        "task_id": "1",
                        "instruction": "...",
                        "task_type": "pda",
                        "dependent_task_ids": []
                    },
                    ...
                ],
                "exp": "description of experience"
            },
            ...
        ]
        
        Args:
            path: Path to JSON file
            
        Returns:
            Number of workflow experiences loaded
        """
        file_path = path or self.experience_path
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"Experience file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.workflows = []
        
        for idx, item in enumerate(data):
            if "workflow" not in item:
                continue
            
            workflow = Workflow.from_dict(item["workflow"])
            
            # Create experience entry
            exp_content = item.get("exp", "")
            metadata = {
                "task": item.get("task", ""),
                "num_tasks": len(workflow.tasks),
                "task_types": workflow.get_task_types()
            }
            
            exp = ExperienceEntry(
                content=exp_content,
                metadata=metadata,
                id=str(idx)
            )
            
            self.workflows.append((workflow, exp))
        
        self._is_initialized = True
        return len(self.workflows)
    
    def retrieve(
        self,
        query: Any,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve similar workflows.
        
        Args:
            query: Either a Workflow object or a workflow dict/list
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult with matched workflows
        """
        if not self.is_initialized:
            raise RuntimeError("Retriever not initialized. Call load_experiences() first.")
        
        start_time = time.time()
        k = top_k or self.top_k
        
        # Convert query to Workflow if needed
        if isinstance(query, dict):
            query_workflow = Workflow.from_dict(query.get("workflow", []))
        elif isinstance(query, list):
            query_workflow = Workflow.from_dict(query)
        elif isinstance(query, Workflow):
            query_workflow = query
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")
        
        # Calculate similarities
        scored_experiences = []
        for workflow, exp in self.workflows:
            similarity = self.matcher.calculate_similarity(query_workflow, workflow)
            
            if similarity < self.min_similarity_threshold:
                continue
            
            # Create result experience with score
            result_exp = ExperienceEntry(
                content=exp.content,
                metadata=exp.metadata.copy(),
                score=similarity,
                id=exp.id
            )
            
            # Apply filters if specified
            if filters:
                if not self._matches_filters(result_exp, filters):
                    continue
            
            scored_experiences.append(result_exp)
        
        # Sort by score descending
        scored_experiences.sort(key=lambda x: x.score, reverse=True)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            experiences=scored_experiences[:k],
            query=str(query_workflow.to_dict()),
            retrieval_time=retrieval_time,
            total_candidates=len(self.workflows)
        )
    
    def _matches_filters(self, exp: ExperienceEntry, filters: Dict[str, Any]) -> bool:
        """Check if experience matches all specified filters"""
        for key, value in filters.items():
            if key not in exp.metadata:
                return False
            if exp.metadata[key] != value:
                return False
        return True
    
    def add_experience(self, experience: ExperienceEntry) -> bool:
        """
        Add a new workflow experience.
        
        Note: experience.metadata must contain "workflow" key with workflow data.
        
        Args:
            experience: Experience to add
            
        Returns:
            True if successfully added
        """
        if "workflow" not in experience.metadata:
            raise ValueError("Experience metadata must contain 'workflow' key")
        
        workflow = Workflow.from_dict(experience.metadata["workflow"])
        
        if not experience.id:
            experience.id = str(len(self.workflows))
        
        self.workflows.append((workflow, experience))
        
        return True
    
    def save_experiences(self, path: Optional[Path] = None) -> bool:
        """
        Save workflow experiences to JSON file.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successfully saved
        """
        file_path = path or self.experience_path
        if not file_path:
            raise ValueError("No path specified for saving")
        
        data = []
        for workflow, exp in self.workflows:
            item = {
                "workflow": workflow.to_dict(),
                "exp": exp.content,
                "task": exp.metadata.get("task", "")
            }
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the workflow experience base"""
        if not self.workflows:
            return {
                "total_workflows": 0,
                "is_initialized": self.is_initialized
            }
        
        task_counts = [len(wf.tasks) for wf, _ in self.workflows]
        all_task_types = []
        for wf, _ in self.workflows:
            all_task_types.extend(wf.get_task_types())
        
        from collections import Counter
        type_distribution = Counter(all_task_types)
        
        return {
            "total_workflows": len(self.workflows),
            "is_initialized": self.is_initialized,
            "avg_tasks_per_workflow": sum(task_counts) / len(task_counts),
            "min_tasks": min(task_counts),
            "max_tasks": max(task_counts),
            "task_type_distribution": dict(type_distribution),
            "matcher_params": {
                "type_weight": self.matcher.type_weight,
                "structure_weight": self.matcher.structure_weight
            }
        }
