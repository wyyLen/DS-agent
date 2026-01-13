# DSAgent Core - 框架无关的核心机制

## 概述

`dsagent_core` 是 DSAgent 的核心组件库，提供了框架无关的实现，可以被 MetaGPT、AutoGen 或其他 Agent 框架复用。

## 核心功能

### 1. 文本经验检索 (Text Experience Retrieval)

基于 BM25 算法的文本经验匹配，用于检索相似的问题解决方案。

```python
from dsagent_core import TextExperienceRetriever
from pathlib import Path

# 初始化检索器
retriever = TextExperienceRetriever(
    experience_path=Path("data/exp_bank/plan_exp.json"),
    top_k=5
)

# 检索相关经验
result = retriever.retrieve(query="如何处理缺失值")
for exp in result.experiences:
    print(f"相关度: {exp.score:.2f}")
    print(f"内容: {exp.content}")
```

### 2. 工作流经验检索 (Workflow Experience Retrieval)

基于图结构的工作流匹配，用于检索相似的任务执行流程。

```python
from dsagent_core import WorkflowExperienceRetriever

# 初始化检索器
retriever = WorkflowExperienceRetriever(
    experience_path=Path("data/exp_bank/workflow_exp.json")
)

# 定义当前工作流
current_workflow = [
    {"task_id": "1", "instruction": "加载数据", "task_type": "data_preprocess"},
    {"task_id": "2", "instruction": "探索性分析", "task_type": "eda"}
]

# 检索相似工作流
result = retriever.retrieve(query=current_workflow, top_k=3)
```

### 3. LATS 树搜索 (Language Agent Tree Search)

框架无关的树搜索算法实现，支持自主探索解决方案空间。

#### 核心组件

- **LATSCore**: 核心搜索引擎，框架无关
- **LATSNode**: 搜索树节点
- **CodeExecutor**: 代码执行接口（需实现）
- **ThoughtGenerator**: 思维生成接口（需实现）
- **ActionGenerator**: 动作生成接口（需实现）
- **StateEvaluator**: 状态评估接口（需实现）

## 适配器 (Adapters)

### MetaGPT 适配器

使用 MetaGPT 的能力实现 LATS：

```python
from dsagent_core.adapters import MetaGPTLATSAdapter

# 创建适配器
lats = MetaGPTLATSAdapter(
    use_exp_driven_search=True,
    max_depth=10,
    high_reward_threshold=7.0
)

# 设置目标并运行
lats.goal = "分析数据并构建预测模型"
best_node, all_nodes = await lats.run(iterations=10, n_generate_sample=2)

# 或者使用增强版本（包含结论生成）
conclusion = await lats.enhance_run(iterations=10)
```

### AutoGen 适配器

使用 AutoGen 的能力实现 LATS：

```python
from dsagent_core.adapters import create_autogen_lats
import os

# 创建适配器
lats = create_autogen_lats(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen-plus",
    max_depth=10
)

# 运行并获取格式化结果
result = await lats.run_and_format(
    goal="加载泰坦尼克数据，分析并构建生存预测模型",
    iterations=10,
    n_generate_sample=2
)

print(f"最佳奖励: {result['best_reward']}")
print(f"解决方案步骤: {len(result['solution_steps'])}")
```

### AutoGen 基础适配器

提供 RAG 和检索功能：

```python
from dsagent_core.adapters import AutoGenAdapter
from pathlib import Path

# 创建适配器
adapter = AutoGenAdapter(
    text_exp_path=Path("data/exp_bank/plan_exp.json"),
    workflow_exp_path=Path("data/exp_bank/workflow_exp.json")
)

# 检索经验
result = adapter.retrieve_text_experience(
    query="如何处理数据预处理",
    top_k=3
)

# 在 AutoGen agent 中注册检索功能
agent = AssistantAgent(name="DataScientist", llm_config=llm_config)
adapter.register_with_agent(agent)
```

## 架构设计

```
dsagent_core/
├── __init__.py              # 主入口
├── interfaces.py            # 核心接口定义
├── retrieval/              # 检索模块
│   ├── base.py             # 基类和接口
│   ├── text_retriever.py   # 文本检索
│   └── workflow_retriever.py # 工作流检索
├── search/                 # 搜索模块
│   ├── tree_search.py      # 通用树搜索
│   └── lats_core.py        # LATS 核心实现
└── adapters/               # 框架适配器
    ├── metagpt_adapter.py      # MetaGPT 基础适配器
    ├── metagpt_lats_adapter.py # MetaGPT LATS 适配器
    ├── autogen_adapter.py      # AutoGen 基础适配器
    └── autogen_lats_adapter.py # AutoGen LATS 适配器
```

## 框架对比

| 特性 | MetaGPT | AutoGen | 核心模块 |
|------|---------|---------|----------|
| 文本检索 | ✅ | ✅ | ✅ |
| 工作流检索 | ✅ | ✅ | ✅ |
| LATS 树搜索 | ✅ | ✅ | ✅ |
| 代码执行 | ✅ | ✅ | 接口 |
| LLM 调用 | ✅ | ✅ | 接口 |
| 成本追踪 | ✅ | ⚠️ | - |
| 流式输出 | ✅ | ⚠️ | - |

## 使用场景

### 场景 1: 使用 MetaGPT 框架

```python
from dsagent_core.adapters import MetaGPTLATSAdapter

lats = MetaGPTLATSAdapter(use_exp_driven_search=True)
lats.goal = "你的任务"
result = await lats.enhance_run(iterations=10)
```

### 场景 2: 使用 AutoGen 框架

```python
from dsagent_core.adapters import create_autogen_lats

lats = create_autogen_lats(api_key="your-key", model="qwen-plus")
result = await lats.run_and_format(goal="你的任务", iterations=10)
```

### 场景 3: 自定义实现

```python
from dsagent_core.search import LATSCore
from dsagent_core.search.lats_core import (
    CodeExecutor, ThoughtGenerator, ActionGenerator, StateEvaluator
)

# 实现你自己的组件
class MyCodeExecutor(CodeExecutor):
    async def execute(self, code, context):
        # 自定义执行逻辑
        pass

# 创建核心引擎
lats = LATSCore(
    thought_generator=MyThoughtGenerator(),
    action_generator=MyActionGenerator(),
    code_executor=MyCodeExecutor(),
    state_evaluator=MyStateEvaluator()
)

# 运行搜索
best, all_nodes = await lats.search(goal="任务", iterations=10)
```

## 优势

1. **框架无关**: 核心逻辑独立于特定框架
2. **易于扩展**: 通过实现接口即可支持新框架
3. **复用性强**: 检索和搜索机制可在不同项目中复用
4. **维护简单**: 核心逻辑集中，便于维护和升级
5. **测试友好**: 可以单独测试核心组件

## 示例代码

查看完整示例：
- `examples/lats_usage_examples.py` - LATS 使用示例
- `examples/ds_agent/lats.py` - MetaGPT LATS 示例
- `dsagent_core/tests/` - 单元测试

## 迁移指南

### 从 MetaGPT LATS 迁移

原代码：
```python
from metagpt.strategy.lats_react import LanguageAgentTreeSearch

lats = LanguageAgentTreeSearch(goal=task)
best, nodes = await lats.run(iterations=10)
```

新代码：
```python
from dsagent_core.adapters import MetaGPTLATSAdapter

lats = MetaGPTLATSAdapter()
lats.goal = task
best, nodes = await lats.run(iterations=10)
```

### 添加 AutoGen LATS 支持

新增功能，原来不支持：
```python
from dsagent_core.adapters import create_autogen_lats

lats = create_autogen_lats(api_key="key", model="qwen-plus")
result = await lats.run_and_format(goal=task, iterations=10)
```

## API 参考

### LATSCore

```python
class LATSCore:
    def __init__(
        self,
        thought_generator: ThoughtGenerator,
        action_generator: ActionGenerator,
        code_executor: CodeExecutor,
        state_evaluator: StateEvaluator,
        max_depth: int = 10,
        exploration_weight: float = 1.4,
        high_reward_threshold: float = 7.0
    )
    
    async def search(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[LATSNode, List[LATSNode]]
```

### MetaGPTLATSAdapter

```python
class MetaGPTLATSAdapter:
    def __init__(
        self,
        use_exp_driven_search: bool = True,
        max_depth: int = 10,
        high_reward_threshold: float = 7.0
    )
    
    async def run(
        self,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Tuple[LATSNode, List[LATSNode]]
    
    async def enhance_run(
        self,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> str
```

### AutoGenLATSAdapter

```python
class AutoGenLATSAdapter:
    def __init__(
        self,
        model_client,
        max_depth: int = 10,
        high_reward_threshold: float = 7.0
    )
    
    async def run(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Tuple[LATSNode, List[LATSNode]]
    
    async def run_and_format(
        self,
        goal: str,
        iterations: int = 10,
        n_generate_sample: int = 2
    ) -> Dict[str, Any]
```

## 贡献

欢迎贡献新的适配器或改进现有实现！

## 许可

MIT License
