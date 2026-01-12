"""
DSAgent启动脚本 - 使用新的DSAgent Core适配器

这个脚本展示如何启动DSAgent并使用重构后的经验检索机制。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream
from metagpt.const import EXAMPLE_DATA_PATH
from metagpt.logs import logger


async def run_dsagent(
    goal: str,
    data_path: str = None,
    use_rag: bool = True,
    use_workflow_rag: bool = True,
    auto_run: bool = True
):
    """
    运行DSAgent处理数据科学任务
    
    Args:
        goal: 用户目标/任务描述
        data_path: 数据文件路径（可选）
        use_rag: 是否使用文本经验检索
        use_workflow_rag: 是否使用工作流经验检索
        auto_run: 是否自动执行代码
    """
    logger.info("=" * 80)
    logger.info("启动 DSAgent (使用 DSAgent Core 适配器)")
    logger.info("=" * 80)
    logger.info(f"任务目标: {goal}")
    logger.info(f"使用文本RAG: {use_rag}")
    logger.info(f"使用工作流RAG: {use_workflow_rag}")
    logger.info("=" * 80)
    
    # 创建DSAgent实例
    # DSAgentStream内部已经配置了经验检索
    agent = DSAgentStream(
        use_rag=use_rag,
        auto_run=auto_run,
    )
    
    # 构建完整的任务描述
    if data_path:
        full_goal = f"File: {data_path}\nTask: {goal}"
    else:
        full_goal = goal
    
    logger.info("\n开始处理任务...")
    logger.info("-" * 80)
    
    # 执行任务
    result = await agent.run(full_goal)
    
    logger.info("-" * 80)
    logger.info("任务完成！")
    logger.info(f"最终结果: {result}")
    
    return result


async def example_correlation_analysis():
    """示例1: 相关性分析任务"""
    goal = """
    Analyze the correlation between different features in the dataset.
    Please:
    1. Load and inspect the data
    2. Calculate correlation coefficients between numerical features
    3. Visualize the correlation matrix
    4. Identify the most strongly correlated feature pairs
    """
    
    # 使用示例数据集
    data_path = str(EXAMPLE_DATA_PATH / "di_dataset/da_bench/da-dev-tables/549_da.csv")
    
    result = await run_dsagent(
        goal=goal,
        data_path=data_path,
        use_rag=True,
        use_workflow_rag=True,
        auto_run=True
    )
    
    return result


async def example_prediction_model():
    """示例2: 预测模型构建"""
    goal = """
    Build a machine learning model to predict the target variable.
    Steps:
    1. Load and explore the dataset
    2. Perform feature engineering
    3. Split data into train/test sets
    4. Train multiple models and compare performance
    5. Select the best model and report metrics
    """
    
    data_path = str(EXAMPLE_DATA_PATH / "di_dataset/da_bench/da-dev-tables/549_da.csv")
    
    result = await run_dsagent(
        goal=goal,
        data_path=data_path,
        use_rag=True,
        use_workflow_rag=True,
        auto_run=True
    )
    
    return result


async def example_custom_task():
    """示例3: 自定义任务"""
    goal = input("请输入您的数据科学任务描述: ")
    
    use_data_file = input("是否指定数据文件? (y/n): ").lower() == 'y'
    data_path = None
    if use_data_file:
        data_path = input("请输入数据文件路径: ")
    
    result = await run_dsagent(
        goal=goal,
        data_path=data_path,
        use_rag=True,
        use_workflow_rag=True,
        auto_run=True
    )
    
    return result


async def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 80)
    print("DSAgent 交互模式 (使用 DSAgent Core)")
    print("=" * 80)
    print("\n选择示例任务:")
    print("1. 相关性分析")
    print("2. 预测模型构建")
    print("3. 自定义任务")
    print("0. 退出")
    print()
    
    choice = input("请选择 (0-3): ").strip()
    
    if choice == "1":
        await example_correlation_analysis()
    elif choice == "2":
        await example_prediction_model()
    elif choice == "3":
        await example_custom_task()
    elif choice == "0":
        print("退出程序。")
        return
    else:
        print("无效选择，请重试。")
        await interactive_mode()


async def quick_test():
    """快速测试 - 使用简单任务验证系统"""
    logger.info("\n" + "=" * 80)
    logger.info("快速测试模式")
    logger.info("=" * 80)
    
    goal = """
    Load the dataset and display basic information including:
    - Number of rows and columns
    - Column names and data types
    - First few rows of data
    - Summary statistics
    """
    
    # 使用示例数据
    data_path = str(EXAMPLE_DATA_PATH / "di_dataset/da_bench/da-dev-tables/549_da.csv")
    
    logger.info(f"\n测试任务: 数据集基本信息分析")
    logger.info(f"数据文件: {data_path}")
    
    result = await run_dsagent(
        goal=goal,
        data_path=data_path,
        use_rag=True,
        use_workflow_rag=True,
        auto_run=True
    )
    
    logger.info("\n✓ 快速测试完成！")
    return result


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            # 快速测试模式
            asyncio.run(quick_test())
        elif mode == "interactive":
            # 交互模式
            asyncio.run(interactive_mode())
        elif mode == "example1":
            # 示例1: 相关性分析
            asyncio.run(example_correlation_analysis())
        elif mode == "example2":
            # 示例2: 预测模型
            asyncio.run(example_prediction_model())
        else:
            print(f"未知模式: {mode}")
            print_usage()
    else:
        # 默认运行交互模式
        asyncio.run(interactive_mode())


def print_usage():
    """打印使用说明"""
    print("""
使用方法:
    python run_dsagent.py [mode]

模式选项:
    test         - 快速测试模式（简单任务验证）
    interactive  - 交互模式（默认）
    example1     - 运行示例1：相关性分析
    example2     - 运行示例2：预测模型构建

示例:
    python run_dsagent.py test
    python run_dsagent.py interactive
    python run_dsagent.py example1
""")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                   DSAgent - 数据科学智能助手                                ║
║                   (基于 DSAgent Core 框架无关架构)                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出。")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
