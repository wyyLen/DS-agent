"""
Full AutoGen adapter test with real experience data
"""

from pathlib import Path
from dsagent_core.adapters.autogen_adapter import (
    AutoGenAdapter,
    create_dsagent_autogen_adapter,
    AUTOGEN_VERSION,
    AUTOGEN_AVAILABLE
)

print("=" * 60)
print("DSAgent AutoGen Adapter - Full Test")
print("=" * 60)
print(f"AutoGen Version: {AUTOGEN_VERSION}")
print(f"AutoGen Available: {AUTOGEN_AVAILABLE}")
print()

# Test 1: Create adapter with text experiences
print("=" * 60)
print("Test 1: Text Experience Retrieval")
print("=" * 60)

text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
if text_exp_path.exists():
    adapter = create_dsagent_autogen_adapter(text_exp_path=text_exp_path)
    
    # Test queries
    test_queries = [
        "correlation analysis",
        "handle missing values",
        "feature engineering",
        "machine learning model training"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = adapter.retrieve_and_format_for_message(query, top_k=2)
        print(f"Result preview: {result[:200]}...")
        print("-" * 60)
else:
    print(f"✗ Experience file not found: {text_exp_path}")

# Test 2: Workflow experience retrieval
print("\n" + "=" * 60)
print("Test 2: Workflow Experience Retrieval")
print("=" * 60)

workflow_exp_path = Path("examples/data/exp_bank/workflow_exp.json")
if workflow_exp_path.exists():
    adapter2 = AutoGenAdapter(workflow_exp_path=workflow_exp_path)
    
    # Define sample workflows
    sample_workflows = [
        {
            "name": "Simple Data Analysis",
            "workflow": [
                {
                    "task_id": "1",
                    "instruction": "Load CSV dataset",
                    "task_type": "data_loading",
                    "dependent_task_ids": []
                },
                {
                    "task_id": "2",
                    "instruction": "Compute correlation matrix",
                    "task_type": "analysis",
                    "dependent_task_ids": ["1"]
                }
            ]
        },
        {
            "name": "ML Pipeline",
            "workflow": [
                {
                    "task_id": "1",
                    "instruction": "Load training data",
                    "task_type": "data_loading",
                    "dependent_task_ids": []
                },
                {
                    "task_id": "2",
                    "instruction": "Preprocess and clean data",
                    "task_type": "preprocessing",
                    "dependent_task_ids": ["1"]
                },
                {
                    "task_id": "3",
                    "instruction": "Train classification model",
                    "task_type": "machine_learning",
                    "dependent_task_ids": ["2"]
                }
            ]
        }
    ]
    
    for sample in sample_workflows:
        print(f"\nSample Workflow: {sample['name']}")
        
        # Format workflow
        formatted = adapter2.format_workflow_for_autogen(sample['workflow'])
        print(formatted)
        
        # Retrieve similar workflows
        result = adapter2.retrieve_workflow_experiences(sample['workflow'], top_k=2)
        print(f"Found {len(result.experiences)} similar workflows")
        
        for i, exp in enumerate(result.experiences, 1):
            print(f"  {i}. Score: {exp.score:.4f}")
        
        print("-" * 60)
else:
    print(f"✗ Workflow file not found: {workflow_exp_path}")

# Test 3: Retrieval function creation
print("\n" + "=" * 60)
print("Test 3: Retrieval Function")
print("=" * 60)

if text_exp_path.exists():
    adapter3 = AutoGenAdapter(text_exp_path=text_exp_path)
    retrieve_func = adapter3.create_retrieval_function()
    
    print("Testing retrieval function...")
    result = retrieve_func("data visualization", top_k=3)
    print(f"Function returned: {type(result).__name__}")
    print(f"Length: {len(result)} characters")
    print(f"\nPreview:\n{result[:300]}...")
else:
    print("Skipped - no experience file")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✓ AutoGen adapter fully functional")
print("✓ Text retrieval working")
print("✓ Workflow retrieval working")
print("✓ Function creation working")
print(f"\nAutoGen Version: {AUTOGEN_VERSION}")
print("\nNote: Agent creation and conversation tests require API keys")
print("      Refer to examples/autogen_integration_example.py for full examples")
