"""
Example: Using DSAgent Core with AutoGen

This example demonstrates how to use DSAgent's retrieval capabilities
within AutoGen agents.
"""

from pathlib import Path
from dsagent_core.adapters.autogen_adapter import create_dsagent_autogen_adapter

# AutoGen imports (conditional)
try:
    from autogen_agentchat.agents import AssistantAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from autogen import AssistantAgent, UserProxyAgent
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False
        print("AutoGen not installed. Install with: pip install pyautogen")


def example_basic_rag():
    """Basic RAG example with AutoGen."""
    
    if not AUTOGEN_AVAILABLE:
        print("Skipping: AutoGen not available")
        return
    
    # 1. Create adapter with experience path
    adapter = create_dsagent_autogen_adapter(
        text_exp_path=Path("examples/data/exp_bank/plan_exp.json")
    )
    
    # 2. Configure LLM (use your API key)
    llm_config = {
        "model": "gpt-4",
        "api_key": "your-api-key-here",
        "temperature": 0.7
    }
    
    # 3. Create RAG-enabled assistant
    assistant = adapter.create_rag_assistant(
        name="DataScience_Assistant",
        llm_config=llm_config
    )
    
    # 4. Create user proxy
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    
    # 5. Start conversation with retrieval
    user_proxy.initiate_chat(
        assistant,
        message=(
            "I need to analyze correlation between two variables. "
            "Please retrieve_experience('correlation analysis') first, "
            "then guide me through the process."
        )
    )


def example_custom_retrieval():
    """Example with custom retrieval function."""
    
    from pathlib import Path
    from dsagent_core.adapters.autogen_adapter import AutoGenAdapter
    
    # Create adapter
    adapter = AutoGenAdapter(
        text_exp_path=Path("examples/data/exp_bank/plan_exp.json")
    )
    
    # Retrieve experiences manually
    query = "how to handle missing values in dataset"
    experiences = adapter.retrieve_and_format_for_message(query, top_k=3)
    
    print(experiences)


def example_multi_agent_with_rag():
    """Example with multiple agents sharing knowledge."""
    
    if not AUTOGEN_AVAILABLE:
        print("Skipping: AutoGen not available")
        return
    
    adapter = create_dsagent_autogen_adapter(
        text_exp_path=Path("examples/data/exp_bank/plan_exp.json")
    )
    
    llm_config = {
        "model": "gpt-4",
        "api_key": "your-api-key-here"
    }
    
    # Planner agent with RAG
    planner = adapter.create_conversable_agent_with_rag(
        name="Planner",
        llm_config=llm_config
    )
    
    # Coder agent with RAG
    coder = adapter.create_conversable_agent_with_rag(
        name="Coder",
        llm_config=llm_config
    )
    
    # User proxy
    user = UserProxyAgent(
        name="User",
        human_input_mode="TERMINATE",
        code_execution_config={"work_dir": "coding"}
    )
    
    # Multi-agent conversation
    user.initiate_chat(
        planner,
        message="Create a data analysis plan for customer churn prediction"
    )


def example_workflow_retrieval():
    """Example using workflow retrieval."""
    
    from dsagent_core.adapters.autogen_adapter import AutoGenAdapter
    
    adapter = AutoGenAdapter(
        workflow_exp_path=Path("examples/data/exp_bank/workflow_exp2_clean_new.json")
    )
    
    # Define a sample workflow
    sample_workflow = [
        {
            "task_id": "1",
            "instruction": "Load dataset",
            "task_type": "data_loading",
            "dependent_task_ids": []
        },
        {
            "task_id": "2", 
            "instruction": "Handle missing values",
            "task_type": "data_preprocessing",
            "dependent_task_ids": ["1"]
        },
        {
            "task_id": "3",
            "instruction": "Train model",
            "task_type": "machine_learning",
            "dependent_task_ids": ["2"]
        }
    ]
    
    # Retrieve similar workflows
    result = adapter.retrieve_workflow_experiences(
        workflow=sample_workflow,
        top_k=3
    )
    
    # Format for AutoGen
    formatted = adapter.format_workflow_for_autogen(sample_workflow)
    print(formatted)
    
    print(f"\nFound {len(result.experiences)} similar workflows")
    for i, exp in enumerate(result.experiences, 1):
        print(f"\nSimilar Workflow {i} (Score: {exp.score:.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("DSAgent Core + AutoGen Integration Examples")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Example 1: Manual Text Retrieval")
    print("=" * 60)
    try:
        example_custom_retrieval()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example 2: Workflow Retrieval")
    print("=" * 60)
    try:
        example_workflow_retrieval()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example 3: Basic RAG (requires API key)")
    print("=" * 60)
    print("Uncomment example_basic_rag() to test with your API key")
    # example_basic_rag()  # Uncomment with valid API key
    
    print("\n" + "=" * 60)
    print("Example 4: Multi-Agent RAG (requires API key)")
    print("=" * 60)
    print("Uncomment example_multi_agent_with_rag() to test with your API key")
    # example_multi_agent_with_rag()  # Uncomment with valid API key
