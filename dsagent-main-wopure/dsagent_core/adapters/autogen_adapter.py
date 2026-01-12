"""
AutoGen Adapter - Connect DSAgent Core to AutoGen framework.

This adapter allows using DSAgent's retrieval and search mechanisms
within AutoGen agents.

Supports both AutoGen 0.2.x (pyautogen) and AutoGen 0.4+ (autogen-agentchat) APIs.
"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Try new AutoGen 0.4+ API first
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import Response
    AUTOGEN_VERSION = "0.4+"
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Fallback to older pyautogen API
    try:
        from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
        AUTOGEN_VERSION = "0.2.x"
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_VERSION = None
        AUTOGEN_AVAILABLE = False
        # Define dummy types
        AssistantAgent = Any
        Response = Any
        ConversableAgent = Any
        UserProxyAgent = Any

from dsagent_core.retrieval import (
    TextExperienceRetriever,
    WorkflowExperienceRetriever,
    RetrievalResult
)
from dsagent_core.search import TreeSearchEngine, SearchNode, ActionGenerator, StateEvaluator, TerminationChecker


class AutoGenAdapter:
    """
    Adapter for integrating DSAgent Core with AutoGen framework.
    
    This class provides methods to use text retrieval, workflow retrieval,
    and tree search within AutoGen agents.
    """
    
    def __init__(
        self,
        text_exp_path: Optional[Path] = None,
        workflow_exp_path: Optional[Path] = None
    ):
        """
        Initialize AutoGen adapter.
        
        Args:
            text_exp_path: Path to text experience JSON file
            workflow_exp_path: Path to workflow experience JSON file
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. "
                "Install it with: pip install pyautogen"
            )
        
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
    
    def retrieve_text_experiences(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve text experiences based on query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        if not self.text_retriever:
            return []
        
        result = self.text_retriever.retrieve(query=query, top_k=top_k)
        return result.experiences if result else []
    
    def create_retrieval_function(self) -> Callable:
        """
        Create a function that can be registered with AutoGen agents.
        
        Returns:
            Function for experience retrieval
        """
        def retrieve_experience(query: str, top_k: int = 3) -> str:
            """
            Retrieve relevant experiences based on query.
            
            Args:
                query: Search query
                top_k: Number of results to return
                
            Returns:
                Formatted experience string
            """
            if not self.text_retriever:
                return "Text retriever not initialized."
            
            result = self.text_retriever.retrieve(query=query, top_k=top_k)
            
            if not result.experiences:
                return "No relevant experiences found."
            
            formatted = "Retrieved experiences:\n\n"
            for i, exp in enumerate(result.experiences, 1):
                formatted += f"{i}. (Score: {exp.score:.2f})\n{exp.content}\n\n"
            
            return formatted
        
        return retrieve_experience
    
    def register_with_agent(
        self,
        agent: Any,
        function_name: str = "retrieve_experience"
    ):
        """
        Register retrieval function with an AutoGen agent.
        
        Note: This method is designed for AutoGen 0.2.x (pyautogen).
        For AutoGen 0.4+, you may need to use different registration methods.
        
        Args:
            agent: AutoGen agent to register with
            function_name: Name for the registered function
        """
        if AUTOGEN_VERSION == "0.4+":
            print("Warning: register_with_agent is designed for AutoGen 0.2.x")
            print("For AutoGen 0.4+, consider using tools/functions directly in agent config")
            return
        
        retrieve_func = self.create_retrieval_function()
        
        # Register as a callable function (0.2.x API)
        if hasattr(agent, 'register_function'):
            agent.register_function(
                function_map={
                    function_name: retrieve_func
                }
            )
    
    def create_rag_assistant(
        self,
        name: str = "DSAgent_Assistant",
        llm_config: Optional[Dict] = None,
        system_message: Optional[str] = None
    ) -> Any:
        """
        Create an AutoGen AssistantAgent with RAG capabilities.
        
        Note: API differs between AutoGen versions.
        - For 0.2.x (pyautogen): Returns configured AssistantAgent
        - For 0.4+: You may need to configure agents differently
        
        Args:
            name: Agent name
            llm_config: LLM configuration
            system_message: System message for the agent
            
        Returns:
            Configured AssistantAgent (version-dependent)
        """
        if system_message is None:
            system_message = (
                "You are a data science assistant with access to past experiences. "
                "When solving problems, use the retrieve_experience function to find "
                "relevant solutions from the knowledge base. Analyze the retrieved "
                "experiences and adapt them to the current task."
            )
        
        if AUTOGEN_VERSION == "0.4+":
            print("Warning: create_rag_assistant for AutoGen 0.4+ may need adjustments")
            print("Please refer to AutoGen 0.4+ documentation for agent configuration")
            # Return None or basic agent without full RAG setup
            return None
        
        # For AutoGen 0.2.x
        assistant = AssistantAgent(
            name=name,
            llm_config=llm_config,
            system_message=system_message
        )
        
        # Register retrieval function
        self.register_with_agent(assistant)
        
        return assistant
    
    def format_workflow_for_autogen(
        self,
        workflow: List[Dict[str, Any]]
    ) -> str:
        """
        Format workflow for AutoGen message.
        
        Args:
            workflow: Workflow as list of task dicts
            
        Returns:
            Formatted workflow string
        """
        if not workflow:
            return "Empty workflow."
        
        formatted = "Workflow Plan:\n\n"
        
        for task in workflow:
            task_id = task.get("task_id", "?")
            instruction = task.get("instruction", "")
            task_type = task.get("task_type", "other")
            deps = task.get("dependent_task_ids", [])
            
            formatted += f"Task {task_id} ({task_type}):\n"
            formatted += f"  Instruction: {instruction}\n"
            if deps:
                formatted += f"  Dependencies: {', '.join(deps)}\n"
            formatted += "\n"
        
        return formatted
    
    def retrieve_and_format_for_message(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        Retrieve experiences and format for AutoGen message.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            Formatted string for message
        """
        if not self.text_retriever:
            return "Retriever not initialized."
        
        result = self.text_retriever.retrieve(query=query, top_k=top_k)
        return self._format_retrieval_result(result)
    
    def _format_retrieval_result(self, result: RetrievalResult) -> str:
        """Format retrieval result for display."""
        if not result.experiences:
            return "No relevant experiences found."
        
        formatted = "📚 Retrieved Experiences:\n\n"
        
        for i, exp in enumerate(result.experiences, 1):
            formatted += f"**Experience {i}** (Relevance: {exp.score:.2f})\n"
            formatted += f"{exp.content}\n"
            formatted += "-" * 50 + "\n\n"
        
        return formatted
    
    def create_conversable_agent_with_rag(
        self,
        name: str,
        llm_config: Dict,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 10
    ) -> Any:
        """
        Create a ConversableAgent with RAG capabilities.
        
        Note: This method is designed for AutoGen 0.2.x (pyautogen).
        For AutoGen 0.4+, agent APIs have changed significantly.
        
        Args:
            name: Agent name
            llm_config: LLM configuration
            human_input_mode: When to ask for human input (0.2.x only)
            max_consecutive_auto_reply: Max auto replies (0.2.x only)
            
        Returns:
            Configured agent (version-dependent, may be None for 0.4+)
        """
        if AUTOGEN_VERSION == "0.4+":
            print("Warning: create_conversable_agent_with_rag is for AutoGen 0.2.x")
            print("For AutoGen 0.4+, please refer to the new agent APIs")
            return None
        
        # For AutoGen 0.2.x only
        if AUTOGEN_VERSION == "0.2.x":
            agent = ConversableAgent(
                name=name,
                llm_config=llm_config,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply,
                system_message=(
                    "You have access to a knowledge base of past experiences. "
                    "Use retrieve_experience(query) to search for relevant solutions."
                )
            )
            
            self.register_with_agent(agent)
            return agent
        
        return None
    
    def add_experience_from_conversation(
        self,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add AutoGen conversation messages as experiences.
        
        Args:
            messages: List of AutoGen message dicts
            metadata: Optional metadata
            
        Returns:
            True if successfully added
        """
        if not self.text_retriever:
            raise RuntimeError("Text retriever not initialized")
        
        from dsagent_core.retrieval.base import ExperienceEntry
        
        # Combine messages into experience
        content = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        exp = ExperienceEntry(
            content=content,
            metadata=metadata or {}
        )
        
        return self.text_retriever.add_experience(exp)
    
    def retrieve_workflow_experiences(
        self,
        workflow: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve similar workflow experiences.
        
        Args:
            workflow: Workflow as list of task dicts
            top_k: Number of results
            
        Returns:
            Retrieval result
        """
        if not self.workflow_retriever:
            raise RuntimeError("Workflow retriever not initialized")
        
        return self.workflow_retriever.retrieve(
            query=workflow,
            top_k=top_k
        )
    
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
def create_dsagent_autogen_adapter(
    text_exp_path: Optional[Path] = None,
    workflow_exp_path: Optional[Path] = None
) -> AutoGenAdapter:
    """
    Convenience function to create and configure an AutoGen adapter.
    
    Args:
        text_exp_path: Path to text experiences
        workflow_exp_path: Path to workflow experiences
        
    Returns:
        Configured adapter
    
    Example:
        >>> adapter = create_dsagent_autogen_adapter(
        ...     text_exp_path=Path("data/exp_bank/plan_exp.json")
        ... )
        >>> assistant = adapter.create_rag_assistant(
        ...     llm_config={"model": "gpt-4"}
        ... )
    """
    return AutoGenAdapter(
        text_exp_path=text_exp_path,
        workflow_exp_path=workflow_exp_path
    )
