"""
Tests for AutoGen adapter integration.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsagent_core.adapters.autogen_adapter import (
    AutoGenAdapter,
    create_dsagent_autogen_adapter,
    AUTOGEN_AVAILABLE
)


@pytest.fixture
def text_exp_path():
    """Path to text experience file."""
    return Path("data/exp_bank/plan_exp.json")


@pytest.fixture
def workflow_exp_path():
    """Path to workflow experience file."""
    return Path("data/exp_bank/workflow_exp.json")


def test_autogen_availability():
    """Test if AutoGen is available."""
    if not AUTOGEN_AVAILABLE:
        pytest.skip("AutoGen not installed")
    
    assert AUTOGEN_AVAILABLE is True


def test_adapter_creation(text_exp_path):
    """Test adapter creation."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    adapter = AutoGenAdapter(text_exp_path=text_exp_path)
    
    assert adapter is not None
    assert adapter.text_retriever is not None


def test_convenience_function(text_exp_path):
    """Test convenience function."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    adapter = create_dsagent_autogen_adapter(text_exp_path=text_exp_path)
    
    assert adapter is not None
    assert isinstance(adapter, AutoGenAdapter)


def test_text_retrieval(text_exp_path):
    """Test text experience retrieval."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    adapter = AutoGenAdapter(text_exp_path=text_exp_path)
    
    # Test retrieval
    result = adapter.retrieve_and_format_for_message("correlation analysis", top_k=2)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_workflow_retrieval(workflow_exp_path):
    """Test workflow retrieval."""
    if not workflow_exp_path.exists():
        pytest.skip(f"Workflow file not found: {workflow_exp_path}")
    
    adapter = AutoGenAdapter(workflow_exp_path=workflow_exp_path)
    
    # Sample workflow
    workflow = [
        {
            "task_id": "1",
            "instruction": "Load data",
            "task_type": "data_loading",
            "dependent_task_ids": []
        }
    ]
    
    result = adapter.retrieve_workflow_experiences(workflow, top_k=2)
    
    assert result is not None
    assert hasattr(result, 'experiences')


def test_retrieval_function_creation(text_exp_path):
    """Test creating retrieval function."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    adapter = AutoGenAdapter(text_exp_path=text_exp_path)
    func = adapter.create_retrieval_function()
    
    assert func is not None
    assert callable(func)
    
    # Test function execution
    result = func("data preprocessing", top_k=1)
    assert isinstance(result, str)


@pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="AutoGen not installed")
def test_rag_assistant_creation(text_exp_path):
    """Test RAG assistant creation."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    adapter = AutoGenAdapter(text_exp_path=text_exp_path)
    
    # Mock LLM config (won't actually call API in test)
    llm_config = {
        "model": "gpt-4",
        "api_key": "test-key"
    }
    
    assistant = adapter.create_rag_assistant(
        name="TestAssistant",
        llm_config=llm_config
    )
    
    assert assistant is not None
    assert assistant.name == "TestAssistant"


def test_workflow_formatting(workflow_exp_path):
    """Test workflow formatting for AutoGen."""
    adapter = AutoGenAdapter()
    
    workflow = [
        {
            "task_id": "1",
            "instruction": "Task 1",
            "task_type": "type1",
            "dependent_task_ids": []
        },
        {
            "task_id": "2",
            "instruction": "Task 2",
            "task_type": "type2",
            "dependent_task_ids": ["1"]
        }
    ]
    
    formatted = adapter.format_workflow_for_autogen(workflow)
    
    assert formatted is not None
    assert "Task 1" in formatted
    assert "Task 2" in formatted
    assert "type1" in formatted


def test_experience_save(text_exp_path, tmp_path):
    """Test saving experiences."""
    if not text_exp_path.exists():
        pytest.skip(f"Experience file not found: {text_exp_path}")
    
    # Use temporary path for test
    temp_exp_path = tmp_path / "test_exp.json"
    
    adapter = AutoGenAdapter()
    adapter.init_text_retriever(text_exp_path)
    
    # This will test the save functionality
    results = adapter.save_all_experiences()
    
    assert "text" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
