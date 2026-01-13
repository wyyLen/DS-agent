# DSAgent 框架升级说明 - LATS 核心化

## 🎯 升级概述

本次升级将 DSAgent 的核心机制（文本检索、工作流检索、LATS 树搜索）抽取到 `dsagent_core` 模块，实现框架无关的设计，现在 **AutoGen 框架也完全支持 LATS** 功能！

## ✨ 主要改进

### 1. 框架无关的核心实现

所有核心功能现在都独立于特定框架：

```
dsagent_core/
├── retrieval/              # 文本和工作流检索
├── search/
│   ├── tree_search.py      # 通用树搜索
│   └── lats_core.py        # ✨ LATS 核心实现（NEW）
└── adapters/
    ├── metagpt_lats_adapter.py   # ✨ MetaGPT LATS 适配器（NEW）
    └── autogen_lats_adapter.py   # ✨ AutoGen LATS 适配器（NEW）
```

### 2. AutoGen 现在支持 LATS！

之前 AutoGen 不支持树搜索，现在通过适配器完全支持：

```python
from dsagent_core.adapters import create_autogen_lats

# 创建 AutoGen LATS
lats = create_autogen_lats(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen-plus"
)

# 运行树搜索
result = await lats.run_and_format(
    goal="分析泰坦尼克数据并构建预测模型",
    iterations=10
)

print(f"探索了 {result['nodes_explored']} 个节点")
print(f"最佳方案奖励: {result['best_reward']}")
```

### 3. MetaGPT 使用新的核心模块

MetaGPT 的 LATS 实现也迁移到核心模块：

```python
from dsagent_core.adapters import MetaGPTLATSAdapter

lats = MetaGPTLATSAdapter(use_exp_driven_search=True)
lats.goal = "你的任务"
best_node, all_nodes = await lats.run(iterations=10)
```

### 4. 完全兼容现有代码

现有代码无需修改，可以继续使用：

```python
# 旧代码仍然可以工作
from metagpt.strategy.lats_react import LanguageAgentTreeSearch

lats = LanguageAgentTreeSearch(goal=task)
best, nodes = await lats.run(iterations=10)
```

但建议迁移到新 API 以获得更好的框架兼容性。

## 📊 功能对比

| 功能 | MetaGPT (旧) | MetaGPT (新) | AutoGen (旧) | AutoGen (新) |
|------|--------------|--------------|--------------|--------------|
| 基础 Agent | ✅ | ✅ | ✅ | ✅ |
| 文本检索 | ✅ | ✅ | ✅ | ✅ |
| 工作流检索 | ✅ | ✅ | ✅ | ✅ |
| **LATS 树搜索** | ✅ | ✅ | ❌ | **✅ 新增** |
| 流式输出 | ✅ | ✅ | ✅ | ✅ |
| 成本追踪 | ✅ | ✅ | ⚠️ | ⚠️ |

## 🚀 快速开始

### 使用 AutoGen LATS

```python
import asyncio
from dsagent_core.adapters import create_autogen_lats

async def main():
    lats = create_autogen_lats(
        api_key="your-dashscope-key",
        model="qwen-plus"
    )
    
    result = await lats.run_and_format(
        goal="分析数据集并构建分类模型",
        iterations=10,
        n_generate_sample=2
    )
    
    print(f"解决方案包含 {len(result['solution_steps'])} 个步骤")
    for i, step in enumerate(result['solution_steps'], 1):
        print(f"{i}. {step['thought']}")

asyncio.run(main())
```

### 使用 MetaGPT LATS

```python
from dsagent_core.adapters import MetaGPTLATSAdapter

async def main():
    lats = MetaGPTLATSAdapter(use_exp_driven_search=True)
    lats.goal = "分析数据集并构建分类模型"
    
    best_node, all_nodes = await lats.run(iterations=10)
    print(f"探索了 {len(all_nodes)} 个节点")
    print(f"最佳奖励: {best_node.reward}")

asyncio.run(main())
```

### 自定义实现

```python
from dsagent_core.search import LATSCore
from dsagent_core.search.lats_core import (
    CodeExecutor, ThoughtGenerator, ActionGenerator, StateEvaluator
)

# 实现自己的组件
class MyExecutor(CodeExecutor):
    async def execute(self, code, context):
        # 自定义执行逻辑
        return True, "execution result"
    
    async def terminate(self):
        pass

# 创建核心引擎
lats = LATSCore(
    thought_generator=MyThoughtGenerator(),
    action_generator=MyActionGenerator(),
    code_executor=MyExecutor(),
    state_evaluator=MyEvaluator()
)

best, all_nodes = await lats.search(goal="任务", iterations=10)
```

## 📚 文档

- **完整文档**: `dsagent_core/LATS_README.md`
- **使用示例**: `examples/lats_usage_examples.py`
- **API 参考**: 查看各适配器的 docstring

## 🔧 技术架构

### LATS 核心组件

```python
# 核心接口
class LATSCore:
    """框架无关的 LATS 核心引擎"""
    
    def __init__(
        self,
        thought_generator: ThoughtGenerator,  # 生成思维/计划
        action_generator: ActionGenerator,    # 生成代码/动作
        code_executor: CodeExecutor,          # 执行代码
        state_evaluator: StateEvaluator      # 评估状态
    )
    
    async def search(self, goal, iterations, n_generate_sample):
        """运行树搜索算法"""
        # 1. 选择节点 (UCT)
        # 2. 扩展节点 (生成子节点)
        # 3. 评估节点 (打分)
        # 4. 反向传播 (更新价值)
```

### 适配器模式

每个框架实现四个接口：

1. **ThoughtGenerator**: 生成下一步的思维/计划
2. **ActionGenerator**: 将思维转换为可执行代码
3. **CodeExecutor**: 执行代码并返回结果
4. **StateEvaluator**: 评估当前状态的质量

## 🎁 优势

1. **统一接口**: 所有框架使用相同的 LATS 核心算法
2. **易于扩展**: 添加新框架只需实现 4 个接口
3. **独立测试**: 核心逻辑可以独立测试
4. **维护简单**: 核心代码集中在一处
5. **向后兼容**: 现有代码无需修改

## 📖 示例对比

### 之前（AutoGen 不支持 LATS）

```python
# AutoGen 只能使用简单的 ReAct
from autogen import AssistantAgent

agent = AssistantAgent(name="DataScientist")
# 无法使用树搜索 ❌
```

### 现在（AutoGen 完全支持 LATS）

```python
# AutoGen 现在支持完整的 LATS 树搜索 ✅
from dsagent_core.adapters import create_autogen_lats

lats = create_autogen_lats(api_key="key", model="qwen-plus")
result = await lats.run_and_format(goal="complex task", iterations=10)
```

## 🔄 迁移建议

### 对于 MetaGPT 用户

建议迁移到新 API：

```python
# 旧代码
from metagpt.strategy.lats_react import LanguageAgentTreeSearch
lats = LanguageAgentTreeSearch(goal=task)

# 新代码（推荐）
from dsagent_core.adapters import MetaGPTLATSAdapter
lats = MetaGPTLATSAdapter()
lats.goal = task
```

### 对于 AutoGen 用户

直接使用新功能：

```python
from dsagent_core.adapters import create_autogen_lats

lats = create_autogen_lats(api_key="key")
result = await lats.run_and_format(goal="task", iterations=10)
```

## 🧪 测试

运行示例：

```bash
# 运行所有 LATS 示例
python examples/lats_usage_examples.py

# 或单独运行
cd examples
python -m ds_agent.lats  # MetaGPT 示例
```

## 📝 更新日志

### v0.2.0 (2026-01-13)

- ✨ 新增：框架无关的 LATS 核心实现
- ✨ 新增：AutoGen LATS 适配器（AutoGen 现在支持树搜索！）
- ✨ 新增：MetaGPT LATS 适配器
- 📚 新增：完整文档和使用示例
- 🔧 改进：更好的模块化和可扩展性

## 🤝 贡献

欢迎为其他框架添加适配器！

需要实现的接口：
- `ThoughtGenerator`
- `ActionGenerator`
- `CodeExecutor`
- `StateEvaluator`

参考 `autogen_lats_adapter.py` 或 `metagpt_lats_adapter.py` 的实现。

## 📄 许可

MIT License
