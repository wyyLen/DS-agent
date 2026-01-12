"""
MetaGPT Adapter - Connect DSAgent Core to MetaGPT framework.

This adapter allows using DSAgent's retrieval and search mechanisms
within MetaGPT agents.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from metagpt.schema import Message, Plan, Task
    from metagpt.llm import LLM
    METAGPT_AVAILABLE = True
except ImportError:
    METAGPT_AVAILABLE = False
    Message = Any
    Plan = Any
    Task = Any
    LLM = Any

from dsagent_core.retrieval import (
    TextExperienceRetriever,
    WorkflowExperienceRetriever,
    RetrievalResult
)
from dsagent_core.search import TreeSearchEngine, SearchNode, ActionGenerator, StateEvaluator, TerminationChecker


class MetaGPTAdapter:
    """
    Adapter for integrating DSAgent Core with MetaGPT framework.
    
    This class provides convenient methods to use text retrieval,
    workflow retrieval, and tree search within MetaGPT agents.
    """
    
    def __init__(
        self,
        text_exp_path: Optional[Path] = None,
        workflow_exp_path: Optional[Path] = None,
        llm: Optional[Any] = None
    ):
        """
        Initialize MetaGPT adapter.
        
        Args:
            text_exp_path: Path to text experience JSON file
            workflow_exp_path: Path to workflow experience JSON file
            llm: MetaGPT LLM instance (optional)
        """
        if not METAGPT_AVAILABLE:
            raise ImportError(
                "MetaGPT is not installed. "
                "Install it with: pip install metagpt"
            )
        
        self.llm = llm
        self.text_retriever: Optional[TextExperienceRetriever] = None
        self.workflow_retriever: Optional[WorkflowExperienceRetriever] = None
        
        # Initialize retrievers if paths provided
        if text_exp_path:
            self.init_text_retriever(text_exp_path)
        
        if workflow_exp_path:
            self.init_workflow_retriever(workflow_exp_path)
    
    def init_text_retriever(
        self,
        path: Path,
        top_k: int = 5,
        **kwargs
    ) -> TextExperienceRetriever:
        """
        Initialize text experience retriever.
        
        Args:
            path: Path to text experience JSON
            top_k: Number of top results
            **kwargs: Additional retriever parameters
            
        Returns:
            Initialized retriever
        """
        self.text_retriever = TextExperienceRetriever(
            experience_path=path,
            top_k=top_k,
            **kwargs
        )
        return self.text_retriever
    
    def init_workflow_retriever(
        self,
        path: Path,
        top_k: int = 5,
        **kwargs
    ) -> WorkflowExperienceRetriever:
        """
        Initialize workflow experience retriever.
        
        Args:
            path: Path to workflow experience JSON
            top_k: Number of top results
            **kwargs: Additional retriever parameters
            
        Returns:
            Initialized retriever
        """
        self.workflow_retriever = WorkflowExperienceRetriever(
            experience_path=path,
            top_k=top_k,
            **kwargs
        )
        return self.workflow_retriever
    
    def retrieve_text_experiences(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve text experiences.
        
        Args:
            query: Query string
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            Retrieval result
        """
        if not self.text_retriever:
            raise RuntimeError("Text retriever not initialized")
        
        return self.text_retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )
    
    def retrieve_workflow_experiences(
        self,
        plan: Plan,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve similar workflow experiences given a MetaGPT Plan.
        
        Args:
            plan: MetaGPT Plan object
            top_k: Number of results
            
        Returns:
            Retrieval result
        """
        if not self.workflow_retriever:
            raise RuntimeError("Workflow retriever not initialized")
        
        # Convert MetaGPT Plan to workflow format
        workflow_dict = self._plan_to_workflow(plan)
        
        return self.workflow_retriever.retrieve(
            query=workflow_dict,
            top_k=top_k
        )
    
    def _plan_to_workflow(self, plan: Plan) -> List[Dict[str, Any]]:
        """
        Convert MetaGPT Plan to workflow format.
        
        Args:
            plan: MetaGPT Plan object
            
        Returns:
            Workflow as list of task dicts
        """
        workflow = []
        
        for task in plan.tasks:
            task_dict = {
                "task_id": str(task.task_id),
                "instruction": task.instruction,
                "task_type": task.task_type if hasattr(task, 'task_type') else "other",
                "dependent_task_ids": [str(tid) for tid in task.dependent_task_ids]
            }
            workflow.append(task_dict)
        
        return workflow
    
    def create_tree_search_engine(
        self,
        action_generator: ActionGenerator,
        state_evaluator: StateEvaluator,
        termination_checker: TerminationChecker,
        **kwargs
    ) -> TreeSearchEngine:
        """
        Create a tree search engine.
        
        Args:
            action_generator: Action generator implementation
            state_evaluator: State evaluator implementation
            termination_checker: Termination checker implementation
            **kwargs: Additional search parameters
            
        Returns:
            Configured tree search engine
        """
        return TreeSearchEngine(
            action_generator=action_generator,
            state_evaluator=state_evaluator,
            termination_checker=termination_checker,
            **kwargs
        )
    
    def format_experiences_for_prompt(
        self,
        result: RetrievalResult,
        max_experiences: int = 3
    ) -> str:
        """
        Format retrieved experiences for inclusion in LLM prompts.
        
        Args:
            result: Retrieval result
            max_experiences: Maximum number of experiences to include
            
        Returns:
            Formatted string for prompt
        """
        if not result.experiences:
            return "No relevant experiences found."
        
        formatted = "Relevant experiences from knowledge base:\n\n"
        
        for i, exp in enumerate(result.experiences[:max_experiences], 1):
            formatted += f"Experience {i} (relevance: {exp.score:.2f}):\n"
            formatted += f"{exp.content}\n\n"
        
        return formatted
    
    def add_text_experience_from_message(
        self,
        message: Message,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a MetaGPT Message as a text experience.
        
        Args:
            message: MetaGPT Message object
            metadata: Optional metadata
            
        Returns:
            True if successfully added
        """
        if not self.text_retriever:
            raise RuntimeError("Text retriever not initialized")
        
        from dsagent_core.retrieval.base import ExperienceEntry
        
        exp = ExperienceEntry(
            content=message.content,
            metadata=metadata or {}
        )
        
        return self.text_retriever.add_experience(exp)
    
    def save_all_experiences(self) -> Dict[str, bool]:
        """
        Save all experience bases to disk.
        
        Returns:
            Dictionary with save status for each retriever
        """
        results = {}
        
        if self.text_retriever:
            try:
                results["text"] = self.text_retriever.save_experiences()
            except Exception as e:
                results["text"] = False
                results["text_error"] = str(e)
        
        if self.workflow_retriever:
            try:
                results["workflow"] = self.workflow_retriever.save_experiences()
            except Exception as e:
                results["workflow"] = False
                results["workflow_error"] = str(e)
        
        return results


# Convenience function for quick setup
def create_dsagent_metagpt_adapter(
    text_exp_path: Optional[Path] = None,
    workflow_exp_path: Optional[Path] = None,
    auto_load: bool = True
) -> MetaGPTAdapter:
    """
    Convenience function to create and configure a MetaGPT adapter.
    
    Args:
        text_exp_path: Path to text experiences
        workflow_exp_path: Path to workflow experiences
        auto_load: Whether to automatically load experiences
        
    Returns:
        Configured adapter
    """
    return MetaGPTAdapter(
        text_exp_path=text_exp_path if auto_load else None,
        workflow_exp_path=workflow_exp_path if auto_load else None
    )
