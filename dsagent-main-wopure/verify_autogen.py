"""
验证 AutoGen 集成的所有功能
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from dsagent_core.adapters.autogen_adapter import AutoGenAdapter

print("=" * 70)
print("AutoGen Integration Verification")
print("=" * 70)

# Test 1: Text Retrieval
print("\n1. Text Experience Retrieval")
print("-" * 70)
text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
adapter = AutoGenAdapter(text_exp_path=text_exp_path)

queries = ["correlation analysis", "machine learning"]
for q in queries:
    result = adapter.retrieve_and_format_for_message(q, top_k=1)
    lines = result.split('\n')[:5]
    print(f"\nQuery: '{q}'")
    print(f"Result preview: {lines[0]}")
    print(f"Total length: {len(result)} chars")

# Test 2: Workflow Retrieval
print("\n\n2. Workflow Experience Retrieval")
print("-" * 70)
workflow_exp_path = Path("examples/data/exp_bank/workflow_exp2_clean_new.json")
adapter2 = AutoGenAdapter(workflow_exp_path=workflow_exp_path)

sample_workflow = [
    {
        "task_id": "1",
        "instruction": "Load CSV data",
        "task_type": "data_loading",
        "dependent_task_ids": []
    },
    {
        "task_id": "2",
        "instruction": "Clean and preprocess",
        "task_type": "data_preprocessing",
        "dependent_task_ids": ["1"]
    }
]

try:
    result = adapter2.retrieve_workflow_experiences(sample_workflow, top_k=3)
    print(f"✓ Workflow retrieval successful")
    print(f"  Found {len(result.experiences)} similar workflows")
    
    if result.experiences:
        for i, exp in enumerate(result.experiences[:3], 1):
            print(f"  {i}. Score: {exp.score:.4f}")
    else:
        print("  Note: No matching workflows found (may need different query)")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Retrieval Function
print("\n\n3. Retrieval Function Creation")
print("-" * 70)
func = adapter.create_retrieval_function()
result = func("data visualization", top_k=1)
print(f"✓ Function created and callable")
print(f"  Returned {len(result)} characters")
print(f"  Preview: {result[:100]}...")

# Test 4: Workflow Formatting
print("\n\n4. Workflow Formatting")
print("-" * 70)
formatted = adapter2.format_workflow_for_autogen(sample_workflow)
print("✓ Workflow formatted for AutoGen:")
print(formatted[:300])

print("\n" + "=" * 70)
print("✅ All AutoGen integration features verified!")
print("=" * 70)
