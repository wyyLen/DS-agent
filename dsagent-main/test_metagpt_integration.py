"""
测试DSAgent Core与MetaGPT的集成效果

此脚本验证：
1. 新适配器能否正确加载现有的经验库
2. 文本经验检索是否与原有系统兼容
3. 工作流经验检索是否与原有系统兼容
4. 检索结果格式是否符合预期
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dsagent_core.adapters import MetaGPTAdapter
from metagpt.schema import Plan, Task
from metagpt.const import EXAMPLE_DATA_PATH, EXP_PLAN, WORKFLOW_EXP


def test_text_retrieval():
    """测试文本经验检索"""
    print("=" * 80)
    print("测试 1: 文本经验检索")
    print("=" * 80)
    
    try:
        # 初始化适配器
        adapter = MetaGPTAdapter(text_exp_path=EXP_PLAN)
        print(f"✓ 成功初始化文本检索器")
        print(f"  经验库路径: {EXP_PLAN}")
        print(f"  经验数量: {len(adapter.text_retriever.experiences)}")
        
        # 测试查询
        test_queries = [
            "How to analyze correlation between variables?",
            "How to handle missing values in dataset?",
            "How to build a machine learning model for prediction?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n查询 {i}: {query}")
            start_time = time.time()
            result = adapter.retrieve_text_experiences(query, top_k=3)
            elapsed = time.time() - start_time
            
            print(f"  检索时间: {elapsed:.3f}秒")
            print(f"  找到 {len(result.experiences)} 条相关经验")
            
            for j, exp in enumerate(result.experiences, 1):
                print(f"\n  经验 {j} (得分: {exp.score:.2f}):")
                content_preview = exp.content[:150].replace('\n', ' ')
                print(f"    内容预览: {content_preview}...")
                if exp.metadata:
                    print(f"    元数据: {exp.metadata}")
        
        # 测试格式化输出（用于LLM提示）
        print(f"\n\n--- 格式化输出测试 ---")
        result = adapter.retrieve_text_experiences(test_queries[0], top_k=2)
        formatted = adapter.format_experiences_for_prompt(result)
        print(f"格式化后的经验（前500字符）:")
        print(formatted[:500])
        
        print(f"\n✓ 文本经验检索测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 文本经验检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_retrieval():
    """测试工作流经验检索"""
    print("\n\n" + "=" * 80)
    print("测试 2: 工作流经验检索")
    print("=" * 80)
    
    try:
        # 初始化适配器
        adapter = MetaGPTAdapter(workflow_exp_path=WORKFLOW_EXP)
        print(f"✓ 成功初始化工作流检索器")
        print(f"  经验库路径: {WORKFLOW_EXP}")
        print(f"  工作流数量: {len(adapter.workflow_retriever.experiences)}")
        
        # 创建测试用的Plan
        test_plans = [
            # 测试案例1: 简单的数据分析流程
            {
                "name": "简单数据分析",
                "tasks": [
                    Task(
                        task_id="1",
                        instruction="Load and inspect the dataset",
                        task_type="pda",
                        dependent_task_ids=[]
                    ),
                    Task(
                        task_id="2",
                        instruction="Analyze correlation between features",
                        task_type="correlation analysis",
                        dependent_task_ids=["1"]
                    ),
                    Task(
                        task_id="3",
                        instruction="Visualize the results",
                        task_type="visualization",
                        dependent_task_ids=["2"]
                    )
                ]
            },
            # 测试案例2: 机器学习流程
            {
                "name": "机器学习预测",
                "tasks": [
                    Task(
                        task_id="1",
                        instruction="Load and preprocess data",
                        task_type="pda",
                        dependent_task_ids=[]
                    ),
                    Task(
                        task_id="2",
                        instruction="Feature engineering",
                        task_type="feature engineering",
                        dependent_task_ids=["1"]
                    ),
                    Task(
                        task_id="3",
                        instruction="Train machine learning model",
                        task_type="machine learning",
                        dependent_task_ids=["2"]
                    ),
                    Task(
                        task_id="4",
                        instruction="Evaluate model performance",
                        task_type="model evaluation",
                        dependent_task_ids=["3"]
                    )
                ]
            }
        ]
        
        for test_case in test_plans:
            print(f"\n测试案例: {test_case['name']}")
            plan = Plan(goal=test_case['name'])
            plan.add_tasks(test_case['tasks'])
            
            print(f"  任务数量: {len(plan.tasks)}")
            print(f"  任务类型: {[task.task_type for task in plan.tasks]}")
            
            start_time = time.time()
            result = adapter.retrieve_workflow_experiences(plan, top_k=3)
            elapsed = time.time() - start_time
            
            print(f"  检索时间: {elapsed:.3f}秒")
            print(f"  找到 {len(result.experiences)} 个相似工作流")
            
            for i, exp in enumerate(result.experiences, 1):
                print(f"\n  工作流 {i} (相似度: {exp.score:.3f}):")
                print(f"    任务数量: {exp.metadata.get('num_tasks', 'N/A')}")
                task_types = exp.metadata.get('task_types', [])
                print(f"    任务类型: {', '.join(task_types[:5])}{'...' if len(task_types) > 5 else ''}")
                if 'exp' in exp.metadata:
                    exp_preview = exp.metadata['exp'][:100].replace('\n', ' ')
                    print(f"    经验摘要: {exp_preview}...")
        
        print(f"\n✓ 工作流经验检索测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 工作流经验检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_usage():
    """测试组合使用（模拟实际DSAgent流程）"""
    print("\n\n" + "=" * 80)
    print("测试 3: 组合使用（模拟实际DSAgent流程）")
    print("=" * 80)
    
    try:
        # 初始化完整的适配器
        adapter = MetaGPTAdapter(
            text_exp_path=EXP_PLAN,
            workflow_exp_path=WORKFLOW_EXP
        )
        print(f"✓ 成功初始化完整适配器")
        print(f"  文本经验: {len(adapter.text_retriever.experiences)} 条")
        print(f"  工作流经验: {len(adapter.workflow_retriever.experiences)} 个")
        
        # 模拟用户查询
        user_goal = "Analyze the housing price dataset and build a prediction model"
        print(f"\n用户目标: {user_goal}")
        
        # 步骤1: 检索相关文本经验
        print(f"\n步骤 1: 检索相关文本经验...")
        text_result = adapter.retrieve_text_experiences(user_goal, top_k=2)
        print(f"  找到 {len(text_result.experiences)} 条相关经验")
        for i, exp in enumerate(text_result.experiences, 1):
            print(f"  - 经验 {i} (得分: {exp.score:.2f}): {exp.content[:80]}...")
        
        # 步骤2: 创建初步计划
        print(f"\n步骤 2: 创建初步计划...")
        plan = Plan(goal=user_goal)
        plan.add_tasks([
            Task(task_id="1", instruction="Load housing data", task_type="pda", dependent_task_ids=[]),
            Task(task_id="2", instruction="Exploratory data analysis", task_type="statistical analysis", dependent_task_ids=["1"]),
            Task(task_id="3", instruction="Feature engineering", task_type="feature engineering", dependent_task_ids=["2"]),
            Task(task_id="4", instruction="Build prediction model", task_type="machine learning", dependent_task_ids=["3"]),
            Task(task_id="5", instruction="Evaluate model", task_type="model evaluation", dependent_task_ids=["4"])
        ])
        print(f"  计划包含 {len(plan.tasks)} 个任务")
        
        # 步骤3: 检索相似工作流
        print(f"\n步骤 3: 检索相似工作流...")
        workflow_result = adapter.retrieve_workflow_experiences(plan, top_k=2)
        print(f"  找到 {len(workflow_result.experiences)} 个相似工作流")
        for i, exp in enumerate(workflow_result.experiences, 1):
            print(f"  - 工作流 {i} (相似度: {exp.score:.3f}): {exp.metadata.get('num_tasks', 0)} 个任务")
        
        # 步骤4: 格式化用于LLM
        print(f"\n步骤 4: 格式化经验用于LLM提示...")
        text_formatted = adapter.format_experiences_for_prompt(text_result)
        workflow_formatted = adapter.format_experiences_for_prompt(workflow_result)
        
        combined_prompt = f"""
用户目标: {user_goal}

相关文本经验:
{text_formatted}

相似工作流:
{workflow_formatted}

请基于以上经验制定详细的执行计划...
"""
        print(f"  组合提示长度: {len(combined_prompt)} 字符")
        print(f"  提示预览（前300字符）:")
        print(combined_prompt[:300].replace('\n', '\n  '))
        
        print(f"\n✓ 组合使用测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 组合使用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_with_existing_code():
    """测试与现有代码的兼容性"""
    print("\n\n" + "=" * 80)
    print("测试 4: 与现有代码的兼容性")
    print("=" * 80)
    
    try:
        # 检查能否正确读取现有格式的经验文件
        adapter = MetaGPTAdapter(
            text_exp_path=EXP_PLAN,
            workflow_exp_path=WORKFLOW_EXP
        )
        
        # 测试1: 检查文本经验格式
        print("\n检查文本经验格式...")
        if adapter.text_retriever.experiences:
            sample_exp = adapter.text_retriever.experiences[0]
            print(f"  ✓ 经验条目格式: {type(sample_exp)}")
            print(f"  ✓ 包含字段: content={bool(sample_exp.content)}, metadata={bool(sample_exp.metadata)}")
        
        # 测试2: 检查工作流经验格式
        print("\n检查工作流经验格式...")
        if adapter.workflow_retriever.experiences:
            sample_exp = adapter.workflow_retriever.experiences[0]
            print(f"  ✓ 经验条目格式: {type(sample_exp)}")
            print(f"  ✓ 包含字段: content={bool(sample_exp.content)}, metadata={bool(sample_exp.metadata)}")
            if 'workflow' in sample_exp.metadata:
                print(f"  ✓ 工作流结构正确")
        
        # 测试3: 验证与MetaGPT Plan的转换
        print("\n检查与MetaGPT Plan的转换...")
        plan = Plan(goal="Test goal")
        plan.add_tasks([
            Task(task_id="1", instruction="Test task", task_type="test", dependent_task_ids=[])
        ])
        workflow_dict = adapter._plan_to_workflow(plan)
        print(f"  ✓ Plan → workflow dict 转换成功")
        print(f"  ✓ 转换后包含 {len(workflow_dict)} 个任务")
        
        print(f"\n✓ 兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DSAgent Core 与 MetaGPT 集成测试" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = []
    
    # 运行测试
    results.append(("文本经验检索", test_text_retrieval()))
    results.append(("工作流经验检索", test_workflow_retrieval()))
    results.append(("组合使用", test_combined_usage()))
    results.append(("兼容性检查", test_compatibility_with_existing_code()))
    
    # 输出总结
    print("\n\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！DSAgent Core 与 MetaGPT 集成正常！")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个测试失败，请检查问题。")
        return 1


if __name__ == "__main__":
    exit(main())
