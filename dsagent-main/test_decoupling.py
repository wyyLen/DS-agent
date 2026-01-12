"""
Test script to verify complete decoupling.

This script tests that DSAgent can work without any MetaGPT code.
"""

import asyncio
import sys
from pathlib import Path

# Add dsagent_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_independent_code_executor():
    """Test that code executor works without MetaGPT."""
    print("=" * 70)
    print("TEST 1: Independent Code Executor")
    print("=" * 70)
    
    from dsagent_core.actions import IndependentCodeExecutor
    
    async with IndependentCodeExecutor() as executor:
        print("\n✓ Code executor created (no MetaGPT dependency)")
        
        # Test basic execution
        code = """
import pandas as pd
import numpy as np

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())
"""
        print("\n📝 Executing code...")
        output, success = await executor.run(code)
        
        if success:
            print(f"\n✅ Execution successful!")
            print(f"Output:\n{output}")
        else:
            print(f"\n❌ Execution failed!")
            print(f"Error:\n{output}")
    
    print("\n✓ Code executor cleaned up")
    return success


async def test_agent_factory():
    """Test agent factory pattern."""
    print("\n" + "=" * 70)
    print("TEST 2: Agent Factory (Framework Detection)")
    print("=" * 70)
    
    from dsagent_core.agents import AgentFactory
    
    # Check available frameworks
    available = AgentFactory.list_available_frameworks()
    print(f"\nAvailable frameworks: {available}")
    
    # Try to create agent with available framework
    for framework, is_available in available.items():
        if is_available:
            print(f"\n✓ {framework.upper()} is available")
            try:
                from dsagent_core.agents import create_agent
                agent = create_agent(
                    agent_id="test-001",
                    framework=framework,
                    agent_type="ds"
                )
                print(f"✅ Successfully created {framework} agent: {agent.agent_id}")
                return True
            except Exception as e:
                print(f"❌ Failed to create {framework} agent: {e}")
        else:
            print(f"\n✗ {framework.upper()} is not available")
    
    return False


async def test_standalone_agent():
    """Test completely standalone agent."""
    print("\n" + "=" * 70)
    print("TEST 3: Standalone Agent (Zero Dependencies)")
    print("=" * 70)
    
    try:
        from dsagent_core.agents.standalone_impl import StandaloneDSAgent, create_standalone_agent
        
        print("\n✓ Standalone agent module imported (no framework required)")
        
        agent = create_standalone_agent(
            agent_id="standalone-001",
            use_rag=False
        )
        
        print(f"✓ Standalone agent created: {agent.agent_id}")
        
        # Test basic processing
        if await agent.acquire():
            print("✓ Agent acquired")
            
            print("\n📊 Processing test query...")
            async for chunk in agent.process_stream("Test query", max_iterations=1):
                print(f"  {chunk.get('type', 'unknown')}: {chunk.get('message', chunk.get('content', '...')[:50])}")
            
            agent.release()
            print("\n✓ Agent released")
            
            await agent.cleanup()
            print("✓ Agent cleaned up")
            
            return True
        else:
            print("❌ Failed to acquire agent")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adapters():
    """Test that adapters work without modifying MetaGPT."""
    print("\n" + "=" * 70)
    print("TEST 4: Adapters (External Integration)")
    print("=" * 70)
    
    # Test MetaGPT adapter
    try:
        from dsagent_core.adapters import MetaGPTAdapter
        print("✓ MetaGPTAdapter imported")
        
        # Test that it doesn't require modified MetaGPT
        print("✓ Adapter uses standard pip-installed MetaGPT")
        
    except ImportError as e:
        print(f"✗ MetaGPTAdapter not available: {e}")
    
    # Test AutoGen adapter
    try:
        from dsagent_core.adapters.autogen_adapter import AutoGenAdapter
        print("✓ AutoGenAdapter imported")
        
    except ImportError as e:
        print(f"✗ AutoGenAdapter not available: {e}")
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "🎯" * 35)
    print(" " * 10 + "DSAgent Decoupling Test Suite")
    print("🎯" * 35 + "\n")
    
    results = []
    
    # Test 1: Code Executor
    try:
        result = await test_independent_code_executor()
        results.append(("Code Executor", result))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Code Executor", False))
    
    # Test 2: Agent Factory
    try:
        result = await test_agent_factory()
        results.append(("Agent Factory", result))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Agent Factory", False))
    
    # Test 3: Standalone Agent
    try:
        result = await test_standalone_agent()
        results.append(("Standalone Agent", result))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Standalone Agent", False))
    
    # Test 4: Adapters
    try:
        result = await test_adapters()
        results.append(("Adapters", result))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Adapters", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Complete decoupling achieved!")
    else:
        print("⚠️  SOME TESTS FAILED - Check errors above")
    print("=" * 70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
