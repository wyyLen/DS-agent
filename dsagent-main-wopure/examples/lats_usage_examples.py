"""
DSAgent Core - LATS Usage Examples

This file demonstrates how to use the framework-agnostic LATS implementation
with different agent frameworks.
"""

import asyncio
import os
from pathlib import Path


# =============================================================================
# Example 1: Using LATS with MetaGPT
# =============================================================================

async def example_metagpt_lats():
    """Example: Use LATS with MetaGPT framework."""
    print("=" * 60)
    print("Example 1: LATS with MetaGPT")
    print("=" * 60)
    
    from dsagent_core.adapters import MetaGPTLATSAdapter, METAGPT_LATS_AVAILABLE
    
    if not METAGPT_LATS_AVAILABLE:
        print("❌ MetaGPT LATS adapter not available")
        return
    
    # Create MetaGPT LATS adapter
    lats = MetaGPTLATSAdapter(
        use_exp_driven_search=True,
        max_depth=10,
        high_reward_threshold=7.0
    )
    
    # Set goal
    lats.goal = "Analyze the iris dataset and build a classification model"
    
    # Run LATS search
    print("\n🔍 Running LATS search...")
    best_node, all_nodes = await lats.run(iterations=5, n_generate_sample=2)
    
    print(f"\n✅ Search completed!")
    print(f"  - Nodes explored: {len(all_nodes)}")
    print(f"  - Best reward: {best_node.reward:.2f}")
    print(f"  - Solution depth: {best_node.depth}")
    
    # Get solution path
    solution_path = lats.lats_core.get_solution_path(best_node)
    print(f"\n📋 Solution steps: {len(solution_path)}")
    for i, step in enumerate(solution_path, 1):
        thought = step.get('thought', {})
        if isinstance(thought, dict):
            print(f"  {i}. {thought.get('thought', '')}")
    
    # Calculate cost
    prompt_tokens, completion_tokens = lats.calculate_total_cost()
    print(f"\n💰 Cost: {prompt_tokens} prompt + {completion_tokens} completion tokens")


# =============================================================================
# Example 2: Using LATS with AutoGen
# =============================================================================

async def example_autogen_lats():
    """Example: Use LATS with AutoGen framework."""
    print("\n" + "=" * 60)
    print("Example 2: LATS with AutoGen")
    print("=" * 60)
    
    from dsagent_core.adapters import create_autogen_lats, AUTOGEN_LATS_AVAILABLE
    
    if not AUTOGEN_LATS_AVAILABLE:
        print("❌ AutoGen LATS adapter not available")
        return
    
    # Get API key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ DASHSCOPE_API_KEY not set")
        return
    
    # Create AutoGen LATS adapter
    lats = create_autogen_lats(
        api_key=api_key,
        model="qwen-plus",
        max_depth=8,
        high_reward_threshold=7.0
    )
    
    # Run LATS and get formatted result
    print("\n🔍 Running LATS search...")
    result = await lats.run_and_format(
        goal="Load titanic data, analyze it, and build a survival prediction model",
        iterations=5,
        n_generate_sample=2
    )
    
    print(f"\n✅ Search completed!")
    print(f"  - Goal: {result['goal']}")
    print(f"  - Best reward: {result['best_reward']:.2f}")
    print(f"  - Nodes explored: {result['nodes_explored']}")
    print(f"  - Solution depth: {result['depth']}")
    
    print(f"\n📋 Solution steps: {len(result['solution_steps'])}")
    for i, step in enumerate(result['solution_steps'], 1):
        print(f"  {i}. {step['thought'][:80]}...")
    
    print(f"\n🎯 Final output:")
    print(result['final_output'][:200] + "..." if len(result['final_output']) > 200 else result['final_output'])
    
    # Clean up
    await lats.cleanup()


# =============================================================================
# Example 3: Direct use of LATS Core (custom implementation)
# =============================================================================

async def example_custom_lats():
    """Example: Use LATS core directly with custom implementations."""
    print("\n" + "=" * 60)
    print("Example 3: LATS Core with Custom Implementations")
    print("=" * 60)
    
    from dsagent_core.search.lats_core import (
        LATSCore,
        LATSNode,
        CodeExecutor,
        ThoughtGenerator,
        ActionGenerator,
        StateEvaluator
    )
    from typing import Tuple, List, Dict, Any
    
    # Define custom implementations
    class SimpleCodeExecutor(CodeExecutor):
        async def execute(self, code: str, context: Dict[str, Any]) -> Tuple[bool, str]:
            # Simulate code execution
            return True, f"Executed: {code[:50]}..."
        
        async def terminate(self):
            pass
    
    class SimpleThoughtGenerator(ThoughtGenerator):
        async def generate(
            self,
            node: LATSNode,
            context: Dict[str, Any],
            n_samples: int = 1
        ) -> List[Dict[str, Any]]:
            # Generate simple thoughts
            return [
                {
                    'thought': f'Step {node.depth + 1}: Continue analysis',
                    'task_type': 'finish' if node.depth >= 2 else 'other'
                }
                for _ in range(n_samples)
            ]
    
    class SimpleActionGenerator(ActionGenerator):
        async def generate(
            self,
            thought: Dict[str, Any],
            context: Dict[str, Any]
        ) -> str:
            return f"# Code for: {thought.get('thought', '')}\nprint('Done')"
    
    class SimpleStateEvaluator(StateEvaluator):
        async def evaluate(
            self,
            trajectory: str,
            is_terminal: bool,
            context: Dict[str, Any]
        ) -> float:
            # Simple heuristic: deeper = better
            depth = trajectory.count('Step')
            return min(depth * 3, 10.0)
    
    # Create LATS core with custom implementations
    lats_core = LATSCore(
        thought_generator=SimpleThoughtGenerator(),
        action_generator=SimpleActionGenerator(),
        code_executor=SimpleCodeExecutor(),
        state_evaluator=SimpleStateEvaluator(),
        max_depth=5
    )
    
    # Run search
    print("\n🔍 Running custom LATS search...")
    best_node, all_nodes = await lats_core.search(
        goal="Example problem",
        iterations=3,
        n_generate_sample=2
    )
    
    print(f"\n✅ Search completed!")
    print(f"  - Nodes explored: {len(all_nodes)}")
    print(f"  - Best reward: {best_node.reward:.2f}")
    print(f"  - Solution depth: {best_node.depth}")
    
    # Get solution
    solution = lats_core.get_solution_path(best_node)
    print(f"\n📋 Solution has {len(solution)} steps")
    
    await lats_core.cleanup()


# =============================================================================
# Example 4: Comparison of frameworks
# =============================================================================

async def example_framework_comparison():
    """Example: Compare LATS across different frameworks."""
    print("\n" + "=" * 60)
    print("Example 4: Framework Comparison")
    print("=" * 60)
    
    from dsagent_core.adapters import (
        METAGPT_LATS_AVAILABLE,
        AUTOGEN_LATS_AVAILABLE
    )
    
    print("\n📊 Framework Availability:")
    print(f"  - MetaGPT LATS: {'✅ Available' if METAGPT_LATS_AVAILABLE else '❌ Not available'}")
    print(f"  - AutoGen LATS: {'✅ Available' if AUTOGEN_LATS_AVAILABLE else '❌ Not available'}")
    
    print("\n📝 Feature Comparison:")
    print("┌────────────────────┬──────────┬──────────┐")
    print("│ Feature            │ MetaGPT  │ AutoGen  │")
    print("├────────────────────┼──────────┼──────────┤")
    print("│ Tree Search        │    ✅    │    ✅    │")
    print("│ Code Execution     │    ✅    │    ✅    │")
    print("│ Experience Driven  │    ✅    │    ✅    │")
    print("│ Cost Tracking      │    ✅    │    ⚠️    │")
    print("│ Streaming          │    ✅    │    ⚠️    │")
    print("└────────────────────┴──────────┴──────────┘")
    
    print("\n💡 Both frameworks now support LATS through dsagent_core!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("🚀 DSAgent Core - LATS Examples")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("\n⚠️  Warning: DASHSCOPE_API_KEY not set")
        print("   Some examples may not work properly")
    
    try:
        # Run examples
        await example_framework_comparison()
        
        # Run MetaGPT example if available
        try:
            await example_metagpt_lats()
        except Exception as e:
            print(f"\n⚠️  MetaGPT example failed: {e}")
        
        # Run AutoGen example if available
        try:
            await example_autogen_lats()
        except Exception as e:
            print(f"\n⚠️  AutoGen example failed: {e}")
        
        # Run custom example
        await example_custom_lats()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
