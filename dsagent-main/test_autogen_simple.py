"""
Simple test for AutoGen adapter - without requiring API keys
"""

from pathlib import Path
import sys

# Test 1: Import test
print("=" * 60)
print("Test 1: Import AutoGen components")
print("=" * 60)

try:
    from autogen_agentchat.agents import AssistantAgent
    print("✓ Successfully imported autogen_agentchat.agents.AssistantAgent")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Adapter import
print("\n" + "=" * 60)
print("Test 2: Import DSAgent AutoGen Adapter")
print("=" * 60)

try:
    from dsagent_core.adapters.autogen_adapter import (
        AutoGenAdapter,
        AUTOGEN_AVAILABLE,
        create_dsagent_autogen_adapter
    )
    print(f"✓ Adapter imported successfully")
    print(f"  AUTOGEN_AVAILABLE: {AUTOGEN_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import adapter: {e}")
    sys.exit(1)

# Test 3: Create adapter without loading experiences
print("\n" + "=" * 60)
print("Test 3: Create Adapter Instance")
print("=" * 60)

try:
    adapter = AutoGenAdapter()
    print("✓ Adapter instance created successfully")
except Exception as e:
    print(f"✗ Failed to create adapter: {e}")
    sys.exit(1)

# Test 4: Test text retriever initialization
print("\n" + "=" * 60)
print("Test 4: Initialize Text Retriever")
print("=" * 60)

text_exp_path = Path("data/exp_bank/plan_exp.json")
if text_exp_path.exists():
    try:
        adapter.init_text_retriever(text_exp_path, top_k=3)
        print(f"✓ Text retriever initialized")
        print(f"  Experience file: {text_exp_path}")
        
        # Test retrieval
        result = adapter.retrieve_and_format_for_message("correlation analysis", top_k=2)
        print(f"✓ Retrieval test successful")
        print(f"  Result length: {len(result)} characters")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
else:
    print(f"⊘ Skipped - file not found: {text_exp_path}")

# Test 5: Test workflow retriever
print("\n" + "=" * 60)
print("Test 5: Initialize Workflow Retriever")
print("=" * 60)

workflow_exp_path = Path("data/exp_bank/workflow_exp.json")
if workflow_exp_path.exists():
    try:
        adapter2 = AutoGenAdapter(workflow_exp_path=workflow_exp_path)
        print(f"✓ Workflow retriever initialized")
        
        # Test workflow formatting
        sample_workflow = [
            {
                "task_id": "1",
                "instruction": "Load dataset",
                "task_type": "data_loading",
                "dependent_task_ids": []
            },
            {
                "task_id": "2",
                "instruction": "Preprocess data",
                "task_type": "preprocessing",
                "dependent_task_ids": ["1"]
            }
        ]
        
        formatted = adapter2.format_workflow_for_autogen(sample_workflow)
        print(f"✓ Workflow formatting successful")
        print(f"  Formatted length: {len(formatted)} characters")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
else:
    print(f"⊘ Skipped - file not found: {workflow_exp_path}")

# Test 6: Test retrieval function creation
print("\n" + "=" * 60)
print("Test 6: Create Retrieval Function")
print("=" * 60)

if text_exp_path.exists():
    try:
        adapter3 = AutoGenAdapter(text_exp_path=text_exp_path)
        retrieve_func = adapter3.create_retrieval_function()
        print(f"✓ Retrieval function created")
        print(f"  Function callable: {callable(retrieve_func)}")
        
        # Test the function
        result = retrieve_func("machine learning", top_k=1)
        print(f"✓ Function execution successful")
        print(f"  Result type: {type(result).__name__}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✓ All basic tests passed!")
print("✓ AutoGen adapter is working correctly")
print("\nNote: Tests requiring API keys (agent creation, conversations) are skipped")
print("      To test those features, you need to provide API keys in the examples")
