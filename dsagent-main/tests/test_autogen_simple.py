"""
Simple test runner for AutoGen integration (without pytest)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsagent_core.adapters.autogen_adapter import (
    AutoGenAdapter,
    create_dsagent_autogen_adapter,
    AUTOGEN_VERSION,
    AUTOGEN_AVAILABLE
)

def test_autogen_availability():
    """Test if AutoGen is available."""
    print("Test 1: AutoGen Availability")
    if not AUTOGEN_AVAILABLE:
        print("  ✗ FAILED: AutoGen not available")
        return False
    
    print(f"  ✓ PASSED: AutoGen available (version {AUTOGEN_VERSION})")
    return True


def test_adapter_creation():
    """Test adapter creation."""
    print("\nTest 2: Adapter Creation")
    try:
        adapter = AutoGenAdapter()
        print("  ✓ PASSED: Adapter instance created")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_convenience_function():
    """Test convenience function."""
    print("\nTest 3: Convenience Function")
    try:
        adapter = create_dsagent_autogen_adapter()
        if not isinstance(adapter, AutoGenAdapter):
            print("  ✗ FAILED: Wrong type returned")
            return False
        print("  ✓ PASSED: Convenience function works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_text_retrieval():
    """Test text experience retrieval."""
    print("\nTest 4: Text Experience Retrieval")
    text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
    
    if not text_exp_path.exists():
        print(f"  ⊘ SKIPPED: Experience file not found: {text_exp_path}")
        return None
    
    try:
        adapter = AutoGenAdapter(text_exp_path=text_exp_path)
        
        # Test retrieval
        result = adapter.retrieve_and_format_for_message("correlation analysis", top_k=2)
        
        if not result or len(result) == 0:
            print("  ✗ FAILED: Empty result")
            return False
        
        print(f"  ✓ PASSED: Retrieved {len(result)} characters")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_retrieval_function_creation():
    """Test creating retrieval function."""
    print("\nTest 5: Retrieval Function Creation")
    text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
    
    if not text_exp_path.exists():
        print(f"  ⊘ SKIPPED: Experience file not found: {text_exp_path}")
        return None
    
    try:
        adapter = AutoGenAdapter(text_exp_path=text_exp_path)
        func = adapter.create_retrieval_function()
        
        if not callable(func):
            print("  ✗ FAILED: Function not callable")
            return False
        
        # Test function execution
        result = func("data preprocessing", top_k=1)
        if not isinstance(result, str):
            print("  ✗ FAILED: Function returned wrong type")
            return False
        
        print("  ✓ PASSED: Retrieval function created and callable")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_workflow_formatting():
    """Test workflow formatting for AutoGen."""
    print("\nTest 6: Workflow Formatting")
    try:
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
        
        if not formatted or "Task 1" not in formatted or "Task 2" not in formatted:
            print("  ✗ FAILED: Formatting incorrect")
            return False
        
        print("  ✓ PASSED: Workflow formatted correctly")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_workflow_retrieval():
    """Test workflow retrieval."""
    print("\nTest 7: Workflow Retrieval")
    workflow_exp_path = Path("examples/data/exp_bank/workflow_exp2_clean_new.json")
    
    if not workflow_exp_path.exists():
        print(f"  ⊘ SKIPPED: Workflow file not found: {workflow_exp_path}")
        return None
    
    try:
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
        
        if not hasattr(result, 'experiences'):
            print("  ✗ FAILED: Result has no experiences")
            return False
        
        print(f"  ✓ PASSED: Workflow retrieval successful")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("AutoGen Integration Test Suite")
    print("=" * 70)
    
    tests = [
        test_autogen_availability,
        test_adapter_creation,
        test_convenience_function,
        test_text_retrieval,
        test_retrieval_function_creation,
        test_workflow_formatting,
        test_workflow_retrieval,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  ⊘ Skipped: {skipped}")
    
    if failed > 0:
        print("\n❌ Some tests failed!")
        return 1
    elif passed == 0:
        print("\n⚠️  No tests passed (all skipped)")
        return 0
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
